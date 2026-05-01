from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from flash_attention import flash_attn


batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
dropout = 0.2


@dataclass
class Config:
  n_layer: int = 12
  n_head: int = 6
  n_kv_head: int = 6
  n_embd: int = 30


def norm(x):
  return F.rms_norm(x, (x.size(-1)))

with open("input.txt", "r", encoding="utf-8") as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

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
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class CausalSelfAttention(nn.Module):
  def __init__(self, config, layer_idx):
    super().__init__()
    self.layer_idx = layer_idx
    self.n_head = config.n_head
    self.n_kv_head = config.n_kv_head
    self.n_embd = config.n_embd
    self.head_dim = self.n_embd // self.n_head
    assert self.n_embd % self.n_head == 0
    assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
    self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
    self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
    self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
    self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
    self.ve_gate_channels = 12
    self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

  def forward(self, x, ve, cos_sin, window_size, kv_cache):
    B, T, C = x.size()

    # Project the input to get queries, keys, and values
    # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

    # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
    if ve is not None:
      ve = ve.view(B, T, self.n_kv_head, self.head_dim)
      gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
      v = v + gate.unsqueeze(-1) * ve

    # Apply Rotary Embeddings to queries and keys to get relative positional encoding
    cos, sin = cos_sin
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k) # QK norm
    q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
    k = k * 1.2

    # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
    # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
    if kv_cache is None:
      # Training: causal attention with optional sliding window
      y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    else:
      # Inference: use flash_attn_with_kvcache which handles cache management
      k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
      y = flash_attn.flash_attn_with_kvcache(
        q, k_cache, v_cache,
        k=k, v=v,
        cache_seqlens=kv_cache.cache_seqlens,
        causal=True,
        window_size=window_size,
      )
      # Advance position after last layer processes
      if self.layer_idx == kv_cache.n_layers - 1:
        kv_cache.advance(T)

    # Re-assemble the heads and project back to residual stream
    y = y.contiguous().view(B, T, -1)
    y = self.c_proj(y)
    return y

class Linear(nn.Linear):
  def forward(self, x):
    return F.linear(x, self.weight.to(dtype=x.dtype))

def has_ve(layer_idx, n_layer):
  return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
  assert x.ndim == 4 # multihead attn
  d = x.shape[3] // 2
  x1, x2 = x[..., :d], x[..., d:]
  y1 = x1 * cos + x2 * sin
  y2 = x1 * (-sin) + x2 * cos
  return torch.cat([y1, y2], 3)

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
    self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

  def forward(self, x):
    x = self.c_fc(x)
    x = F.relu(x).square()
    x = self.c_proj(x)
    return x

class Block(nn.Module):
  """ transformer block """
  def __init__(self, config, layer_idx):
    super().__init__()
    self.attn = CausalSelfAttention(config, layer_idx)
    self.mlp = MLP(config)

  def forward(self, x, ve, cos_sin, window_size, kv_cache):
    x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
    x = x + self.mlp(norm(x))
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size, config):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
    self.position_embedding_table = nn.Embedding(block_size, config.n_embd)
    self.blocks = nn.Sequential(
      *[Block(config, layer_idx) for layer_idx in range(config.n_layer)],
      nn.LayerNorm(config.n_embd),
    )
    self.lm_head = nn.Linear(config.n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx) # (B, T, C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
    x = tok_emb + pos_emb # (B, T, C)
    x = self.blocks(x) # (B, T, C)
    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      # crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits, loss = self(idx_cond)
      # get the next token
      logits = logits[:, -1, :]
      # apply softmax to get the probabilities
      probs = F.softmax(logits, dim=-1)
      # get the next token
      idx_next = torch.multinomial(probs, num_samples=1)
      # append the next token to the input
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

cfg = Config()

model = BigramLanguageModel(vocab_size, cfg).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb, yb = get_batch("train")

  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
