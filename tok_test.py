from tokenizer import Tokenizer


tokenizer = Tokenizer.from_pretrained("o200k_harmony")

vocab_size = tokenizer.enc.n_vocab
print("Vocab size:", vocab_size)

print(tokenizer.encode("hello there!"))
print(tokenizer.decode(tokenizer.encode("hello there!")))