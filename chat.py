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
step = 30000

model, tokenizer = load_model(device, phase="eval", step=step)

bos = tokenizer.get_bos_token_id() # <|endoftext|>

engine = Engine(model, tokenizer)

conversation_text = ""

while True:
  try:
    user_input = input("\nUser: ").strip()
  except (EOFError, KeyboardInterrupt):
    print("\nGoodbye!")
    break

  if not user_input:
    continue

  prompt_text = f"{conversation_text}User: {user_input}\nAssistant:"
  prompt_tokens = [bos]
  prompt_tokens.extend(tokenizer.encode(prompt_text))

  generate_kwargs = {
    "num_samples": 1,
    "max_tokens": 256,
    "temperature": 0.6,
    "top_k": 50,
    "stop_tokens": [bos],
  }
  response_tokens = []
  print("\nAssistant: ", end="", flush=True)
  for token_column, token_masks in engine.generate(prompt_tokens, **generate_kwargs):
    if token_masks[0] == 0:
      continue

    token = token_column[0]
    response_tokens.append(token)
    token_text = tokenizer.decode([token])
    print(token_text, end="", flush=True)

  print()

  response_text = tokenizer.decode(response_tokens)
  conversation_text = f"{prompt_text}{response_text}\n\n"
