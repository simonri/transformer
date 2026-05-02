import tiktoken


tokenizer = tiktoken.get_encoding("o200k_harmony")

vocab_size = tokenizer.max_token_value + 1
print("Vocab size:", vocab_size)

print(tokenizer.encode("hello there!"))
print(tokenizer.decode(tokenizer.encode("hello there!")))