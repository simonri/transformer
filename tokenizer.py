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

  def encode(self, text):
    ids = self.enc.encode_ordinary(text)
    return ids

  def decode(self, ids):
    return self.enc.decode(ids)

  def get_vocab_size(self):
    return self.enc.n_vocab
