import torch

from kv_cache import KVCache

class Engine:
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  @torch.inference_mode()
  def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
    device = self.model.get_device()
    dtype = torch.float32

    # set seed
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

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

    num_generated = 0

    while True:
      if max_tokens is not None and num_generated >= max_tokens:
        break

      next_ids = torch.argmax(logits, dim=-1, keepdim=True)
      sampled_tokens = next_ids[:, 0].tolist()

      token_column = []
      token_masks = []

      for i in range(num_samples):
        token_masks.append(1)
        next_token = sampled_tokens[i]
        token_column.append(next_token)

      yield token_column, token_masks
      num_generated += 1

      ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
      logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
