import os
import torch
import json

from model import BigramLanguageModel, Config
from tokenizer import get_tokenizer
from engine import Engine

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
  model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pth")
  model_data = torch.load(model_path, map_location=device)

  optimizer_data = None

  meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
  with open(meta_path, "r", encoding="utf-8") as f:
    meta_data = json.load(f)

  return model_data, optimizer_data, meta_data

def build_model(checkpoint_dir, step, device, phase):
  assert phase in ["train", "eval"]
  model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)

  model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
  model_config_kwargs = meta_data["model_config"]

  model_config = Config(**model_config_kwargs)

  model = BigramLanguageModel(model_config)

  # load the model state
  model.to_empty(device=device)
  model.init_weights()
  model.load_state_dict(model_data, strict=True, assign=True)

  if phase == "eval":
    model.eval()
  else:
    model.train()

  tokenizer = get_tokenizer()

  assert tokenizer.get_vocab_size() == model_config.vocab_size, f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model vocab size {model_config.vocab_size}"
  return model, tokenizer

def load_model_from_dir(checkpoints_dir, device, phase, step=None):
  model, tokenizer = build_model(checkpoints_dir, step, device, phase)
  return model, tokenizer

def load_model(*args, **kwargs):
  checkpoints_dir = os.path.join("checkpoints")
  return load_model_from_dir(checkpoints_dir, *args, **kwargs)

device = "cuda" if torch.cuda.is_available() else "cpu"
step = 3000

model, tokenizer = load_model(device, phase="eval", step=step)

bos = tokenizer.get_bos_token_id()

start_token, end_token = tokenizer.encode_special("<|start|>"), tokenizer.encode_special("<|end|>")
message_token = tokenizer.encode_special("<|message|>")
user_role_token = tokenizer.encode_special("user")
assistant_role_token = tokenizer.encode_special("assistant")

engine = Engine(model, tokenizer)

conversation_tokens = [bos]

while True:
  try:
    user_input = input("\nUser: ").strip()
  except (EOFError, KeyboardInterrupt):
    print("\nGoodbye!")
    break

  if not user_input:
    continue

  # example input
  # <|start|>user<|message|>What is 2 + 2?<|end|>
  # <|start|>assistant

  conversation_tokens.append(start_token)
  conversation_tokens.append(user_role_token)
  conversation_tokens.append(message_token)
  conversation_tokens.extend(tokenizer.encode(user_input))
  conversation_tokens.append(end_token)
  conversation_tokens.append(start_token)
  conversation_tokens.append(assistant_role_token)

  generate_kwargs = {
    "num_samples": 1,
    "max_tokens": 256,
    "temperature": 0.7,
    "top_k": 40
  }
  response_tokens = []
  print("\nAssistant: ", end="", flush=True)
  for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
    token = token_column[0]
    response_tokens.append(token)
    token_text = tokenizer.decode([token])
    print(token_text, end="", flush=True)
  print()

  print(conversation_tokens)
