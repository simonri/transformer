import tiktoken

from functools import lru_cache

class Tokenizer:
  def __init__(self, enc, bos_token):
    self.enc = enc
    self.bos_token_id = self.encode_special(bos_token)

  @classmethod
  def from_pretrained(cls, tiktoken_name):
    enc = tiktoken.get_encoding(tiktoken_name)
    return cls(enc, "<|endoftext|>")

  @lru_cache(maxsize=32)
  def encode_special(self, text):
    return self.enc.encode_single_token(text)

  def encode(self, text, prepend=None, append=None, num_threads=None):
    if prepend is not None:
      prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
    if append is not None:
      append_id = append if isinstance(append, int) else self.encode_special(append)

    if isinstance(text, str):
      ids = self.enc.encode_ordinary(text)
      if prepend is not None:
        ids.insert(0, prepend_id)
      if append is not None:
        ids.append(append_id)
    elif isinstance(text, list):
      ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
      if prepend is not None:
        for ids_row in ids:
          ids_row.insert(0, prepend_id)
      if append is not None:
        for ids_row in ids:
          ids_row.append(append_id)
    else:
      raise ValueError("text must be a string or list")
    
    return ids

  def decode(self, ids):
    return self.enc.decode(ids)

  def get_special_tokens(self):
    return self.enc.special_tokens_set

  def get_vocab_size(self):
    return self.enc.n_vocab

  def get_bos_token_id(self):
    return self.bos_token_id

  def __call__(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

def get_tokenizer():
  return Tokenizer.from_pretrained("o200k_harmony")
