import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
from dataclasses import asdict
from tokenizer import get_tokenizer
import math
import wandb
import argparse

from model import BigramLanguageModel, Config
from dataloader import tokenizing_data_loader_with_state_bos_bestfit
from engine import Engine
from flash_attention import USE_FA3

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, help="name of the run")
args = parser.parse_args()

# [arguments]
# model arch
depth = 16 # depth of transformer model
head_dim = 32 # target head dimension for attention (fa3 requires head_dim divisible by 8)
max_seq_len = 512 # max context length
window_pattern = "SSSL"
num_q_heads = 12
num_kv_heads = 2

# optimization
save_every = 2000
sample_every = 500
warmup_steps = 40
warmdown_ratio = 0.65
final_lr_frac = 0.05
weight_decay = 0.28
device_batch_size = 16 # per device batch size
total_batch_size = -1 # total batch size in tokens

# training horizon
target_param_data_ratio = 8 # calculate num_iterations to maintain data:param ratio
num_iterations = 30000 # num optimization steps
# [arguments end]

device = "cuda" if torch.cuda.is_available() else "cpu"

# wandb logging
wandb_run = wandb.init(project="gpt", name=args.run_name)

# flash attention status
using_fa3 = USE_FA3
if using_fa3:
  print("Using Flash Attention 3")
else:
  print("Using PyTorch SDPA")

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print("Vocab size:", vocab_size)

# initialize the model
def build_model_meta(depth):
  model_dim = num_q_heads * head_dim

  config = Config(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=depth,
    n_head=num_q_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    window_pattern=window_pattern,
  )

  model_meta = BigramLanguageModel(config)
  return model_meta

model = build_model_meta(depth)
model_config = model.config
model_config_kwargs = asdict(model_config)
print(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device)
model.init_weights()

orig_model = model
model = torch.compile(model, dynamic=False)

param_counts = model.num_scaling_params()
num_params = param_counts["total"]
print("num_params", num_params)

def get_scaling_params(m):
  params_counts = m.num_scaling_params()
  scaling_params = params_counts["transformer_matrices"] + params_counts["lm_head"]
  return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(target_param_data_ratio * num_scaling_params)

d12_ref = build_model_meta(12)
D_REF = target_param_data_ratio * get_scaling_params(d12_ref)
B_REF = 2**19 # optim batch size at d12 = 524,288 tokens

if total_batch_size == -1:
  batch_size_ratio = target_tokens / D_REF
  predicted_batch_size = B_REF * batch_size_ratio ** 0.383
  total_batch_size = 2 ** round(math.log2(predicted_batch_size))
  print(f"Auto-computed optimal batch size: {total_batch_size} tokens")

weight_decay_scaled = weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)

optimizer = model.setup_optimizer()

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
  os.makedirs(checkpoint_dir, exist_ok=True)
  model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pth")
  torch.save(model_data, model_path)
  print(f"Saved model parameters to {model_path}")

  meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
  with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta_data, f, indent=2)
  print(f"Saved metadata to: {meta_path}")

# init dataloaders
train_loader = tokenizing_data_loader_with_state_bos_bestfit(tokenizer, device_batch_size, max_seq_len, "train", device=device)
x, y, dataloader_state_dict = next(train_loader)

# lr schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
  warmup_iters = warmup_steps
  warmdown_iters = round(warmdown_ratio * num_iterations)
  if it < warmup_iters:
    return (it + 1) / warmup_iters
  elif it <= num_iterations - warmdown_iters:
    return 1.0
  else:
    progress = (num_iterations - it) / warmdown_iters
    return progress * 1.0 + (1 - progress) * final_lr_frac

# momentum scheduler for muon optimizer (warms up to 0.97, warms down to 0.9 during lr warmdown)
def get_muon_momentum(it):
  warmdown_iters = round(warmdown_ratio * num_iterations)
  warmdown_start = num_iterations - warmdown_iters
  if it < 400:
    frac = it / 400
    return (1 - frac) * 0.85 + frac * 0.97
  elif it >= warmdown_start:
    progress = (it - warmdown_start) / warmdown_iters
    return 0.97 * (1 - progress) + 0.90 * progress
  else:
    return 0.97

# weight decay scheduler for muon optimizer (cos decay to zero over the course of trainiing)
def get_weight_decay(it):
  return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * 1
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

# training loop
step = 0
smooth_train_loss = 0

while True:
  last_step = step == num_iterations

  if sample_every > 0 and (last_step or (step > 0 and step % sample_every == 0)):
    model.eval()
    prompts = [
      "The capital of Sweden is"
    ]
    engine = Engine(orig_model, tokenizer)
    for prompt in prompts:
      tokens = tokenizer(prompt)
      sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
      print(tokenizer.decode(sample[0]))
    model.train()

  if last_step or (step > 0 and save_every > 0 and step % save_every == 0):
    save_checkpoint(
      "checkpoints",
      step,
      orig_model.state_dict(),
      optimizer.state_dict(),
      {
        "step": step,
        "model_config": model_config_kwargs
      }
    )

  if last_step:
    break

  # single training step
  torch.cuda.synchronize()

  for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    train_loss = loss.detach()
    loss.backward()
  
    x, y, dataloader_state_dict = next(train_loader)

  # step the optimizer
  lrm = get_lr_multiplier(step)
  muon_momentum = get_muon_momentum(step)
  muon_weight_decay = get_weight_decay(step)

  for group in optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lrm
    if group["kind"] == "muon":
      group["momentum"] = muon_momentum
      group["weight_decay"] = muon_weight_decay

  optimizer.step()

  model.zero_grad(set_to_none=True)
  train_loss_f = train_loss.item()
  torch.cuda.synchronize()

  # logging
  ema_beta = 0.9
  smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
  debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

  if step % 100 == 0:
    print(f"step {step:05d}/{num_iterations:05d} | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f}")
    log_data = {
      "step": step,
      "train/loss": debiased_smooth_loss,
      "train/lrm": lrm,
    }
    wandb_run.log(log_data)

  step += 1
