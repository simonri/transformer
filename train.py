import torch
import os
import json
from dataclasses import asdict
from tokenizer import Tokenizer

from model import BigramLanguageModel, Config
from dataloader import tokenizing_data_loader_with_state_bos_bestfit

# [arguments]
# optimization
device_batch_size = 32
num_iterations = 3000
save_every = 500
max_seq_len = 32 # this is very low

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_pretrained("o200k_harmony")
vocab_size = tokenizer.get_vocab_size()
print("Vocab size:", vocab_size)

# initialize the model
def build_model_meta():
  config = Config(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
  )

  model_meta = BigramLanguageModel(config)
  return model_meta

model = build_model_meta()
model_config = model.config
model_config_kwargs = asdict(model_config)
model.to_empty(device=device)
model.init_weights()

orig_model = model
model = torch.compile(model, dynamic=False)

param_counts = model.num_scaling_params()
num_params = param_counts['total']
print("num_params", num_params)

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

# training loop
step = 0

while True:
  last_step = step == num_iterations

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

  loss = model(x, y)
  loss.backward()
  
  x, y, dataloader_state_dict = next(train_loader)

  optimizer.step()

  model.zero_grad(set_to_none=True)
  torch.cuda.synchronize()

  step += 1
