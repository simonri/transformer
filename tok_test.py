from tokenizer import Tokenizer


tokenizer = Tokenizer.from_pretrained("o200k_harmony")

vocab_size = tokenizer.enc.n_vocab
print("Vocab size:", vocab_size)

print(tokenizer.encode("hello there!"))
print(tokenizer.decode(tokenizer.encode("hello there!")))

print("BOS token:", tokenizer.decode([tokenizer.get_bos_token_id()]))

for special in tokenizer.get_special_tokens():
  if "reserved" not in special:
    print(special)
