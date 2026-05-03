import torch
import torch.nn.functional as F
from types import SimpleNamespace

def _load_flash_attention_3():
  """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
  if not torch.cuda.is_available():
    return None
  try:
    major, _ = torch.cuda.get_device_capability()
    # FA3 kernels are compiled for Hopper (sm90) only
    # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
    if major != 9:
        return None
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from kernels import get_kernel
    return get_kernel('varunneal/flash-attention-3').flash_attn_interface
  except Exception:
    return None

_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

print("HAS_FA3", HAS_FA3)

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

USE_FA3 = True

def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
  """
  fa for training (no kv cache)
  """
  if USE_FA3:
    return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

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
  if USE_FA3:
    return _fa3.flash_attn_with_kvcache(
      q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
      causal=causal, window_size=window_size
    )

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

flash_attn = SimpleNamespace(
  flash_attn_func=flash_attn_func,
  flash_attn_with_kvcache=flash_attn_with_kvcache,
)
