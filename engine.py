import torch
from collections import deque
from torch.nn import functional as F

from kv_cache import KVCache

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
  assert temperature >= 0
  if temperature == 0.0:
    return torch.argmax(logits, dim=-1, keepdim=True)
  if top_k is not None and top_k > 0:
    k = min(top_k, logits.size(-1))
    vals, idx = torch.topk(logits, k, dim=-1)
    vals = vals / temperature
    probs = F.softmax(vals, dim=-1)
    choice = torch.multinomial(probs, num_samples=1, generator=rng)
    return idx.gather(1, choice)
  else:
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)

class RowState:
  def __init__(self, current_tokens=None):
    self.current_tokens = current_tokens or []
    self.forced_tokens = deque()
    self.completed = False

class Engine:
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  @torch.inference_mode()
  def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42, stop_tokens=None):
    device = self.model.get_device()
    dtype = torch.float32

    # set seed
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    stop_tokens = set(stop_tokens or [])

    # 1) run a batch 1 prefill of prompt tokens
    m = self.model.config
    kv_model_kwargs = {
      "num_heads": m.n_kv_head,
      "head_dim": m.n_embd // m.n_kv_head,
      "num_layers": m.n_layer,
    }
    kv_cache_prefill = KVCache(
      batch_size=1,
      seq_len=len(tokens),
      device=device,
      dtype=dtype,
      **kv_model_kwargs,
    )
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
    logits = logits[:, -1, :].expand(num_samples, -1)

    # 2) replace the kv cache for each sample/row
    kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else m.sequence_len
    kv_cache_decode = KVCache(
      batch_size=num_samples,
      seq_len=kv_length_hint,
      device=device,
      dtype=dtype,
      **kv_model_kwargs,
    )
    kv_cache_decode.prefill(kv_cache_prefill)
    del kv_cache_prefill

    # 3) init states for samples
    row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

    # 4) gen loop
    num_generated = 0
    while True:
      if max_tokens is not None and num_generated >= max_tokens:
        break

      if all(state.completed for state in row_states):
        break

      next_ids = sample_next_token(logits, rng, temperature, top_k)
      sampled_tokens = next_ids[:, 0].tolist()

      token_column = []
      token_masks = []

      for i, state in enumerate(row_states):
        is_forced = len(state.forced_tokens) > 0
        next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
        is_stop = next_token in stop_tokens

        token_masks.append(0 if is_forced or is_stop else 1)
        token_column.append(next_token)

        if is_stop:
          state.completed = True
        else:
          state.current_tokens.append(next_token)

      yield token_column, token_masks
      num_generated += 1

      if all(state.completed for state in row_states):
        break

      ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
      logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
