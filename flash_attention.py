import torch
import torch.nn.functional as F

def _sdpa_attention(q, k, v, window_size, enable_gqa):
  """
  SDPA attention with sliding window support.
  q, k, v are (B, H, T, D) format.
  """
  Tq = q.size(2)
  Tk = k.size(2)
  window = window_size[0]

  # Full context, same length
  if (window < 0 or window >= Tq) and Tq == Tk:
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

  # Single token generation
  if Tq == 1:
    if window >= 0 and window < Tk:
      # window is "left" tokens we need to include (window + 1) keys total
      start = max(0, Tk - (window + 1))
      k = k[:, :, start:, :]
      v = v[:, :, start:, :]
    return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

  # Need explicit mask for sliding window/chunk inference
  device = q.device
  # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
  row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
  col_idx = torch.arange(Tk, device=device).unsqueeze(0)
  mask = col_idx <= row_idx

  # sliding window (left)
  if window >= 0 and window < Tk:
    mask = mask & ((row_idx - col_idx) <= window)

  return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

def flash_attn_func(q, k, v, casual=False, window_size=(-1, -1)):
  """
  fa for training (no kv cache)
  """
  # sdpa
  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)
  enable_gqa = q.size(1) != k.size(1)
  y = _sdpa_attention(q, k, v, window_size, enable_gqa)
  return y.transpose(1, 2)

def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None, causal=False, window_size=(-1, -1)):
  """
  fa for inference (with kv cache)
  """
  # sdpa
  B, T_new, H, D = q.shape
  pos = cache_seqlens[0].item()

  # insert new k, v into cache
  if k is not None and v is not None:
    k_cache[:, pos:pos+T_new, :, :] = k
    v_cache[:, pos:pos+T_new, :, :] = v

  # get full cache up to curr pos + new tokens
  end_pos = pos + T_new
  k_full = k_cache[:, :end_pos, :, :]
  v_full = v_cache[:, :end_pos, :, :]

  # transpose to sdpa layout
  q_sdpa = q.transpose(1, 2)
  k_sdpa = k_full.transpose(1, 2)
  v_sdpa = v_full.transpose(1, 2)

  enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
  y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

  return y_sdpa.transpose(1, 2)

from types import SimpleNamespace
flash_attn = SimpleNamespace(
  flash_attn_func=flash_attn_func,
  flash_attn_with_kvcache=flash_attn_with_kvcache,
)
