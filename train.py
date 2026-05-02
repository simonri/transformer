from model import BigramLanguageModel, Config
from tokenizer import Tokenizer
import torch
import os
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
eval_iters = 200
max_iters = 5000
eval_interval = 500

cfg = Config()
tokenizer = Tokenizer.from_pretrained("o200k_harmony")
vocab_size = tokenizer.get_vocab_size()
print("Vocab size:", vocab_size)

cfg.vocab_size = vocab_size


with open("input.txt", "r", encoding="utf-8") as f:
  text = f.read()

chars = sorted(list(set(text)))

# create encoding and decoding functions
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
  return [stoi[c] for c in s]

def decode(indices):
  return "".join([itos[i] for i in indices])

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - cfg.sequence_len, (batch_size,))
  x = torch.stack([data[i:i+cfg.sequence_len] for i in ix])
  y = torch.stack([data[i+1:i+cfg.sequence_len+1] for i in ix])
  x = x.to(device)
  y = y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ["train", "val"]:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


# initialize the model
model = BigramLanguageModel(cfg)
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


# training loop
step = 0
save_every = 500

while True:
  last_step = step == max_iters

  if step % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  if last_step or (step > 0 and save_every > 0 and step % save_every == 0):
    save_checkpoint(
      "checkpoints",
      step,
      orig_model.state_dict(),
      optimizer.state_dict(),
      {
        "step": step,
      }
    )

  if last_step:
    break

  xb, yb = get_batch("train")

  loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  step += 1

print(decode(list(model.generate([0], max_tokens=500))))
