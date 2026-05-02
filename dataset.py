import os
from multiprocessing import Pool
import requests

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
DATA_DIR = os.path.join("base_data_climbmix")

def list_parquet_files(data_dir=None):
  """ Looks into a data dir and returns full paths to all parquet files. """
  data_dir = DATA_DIR if data_dir is None else data_dir

  parquet_files = sorted([
    f for f in os.listdir(data_dir)
    if f.endswith('.parquet') and not f.endswith('.tmp')
  ])
  parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
  return parquet_paths

def index_to_filename(index):
  return f"shard_{index:05d}.parquet"

def download_single_file(index):
  filename = index_to_filename(index)
  filepath = os.path.join(DATA_DIR, filename)
  if os.path.exists(filepath):
    print(f"Skipping {filepath} (already exists)")
    return True

  url = f"{BASE_URL}/{filename}"
  print(f"Downloading {filename}")

  try:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    temp_path = filepath + f".tmp"
    with open(temp_path, "wb") as f:
      for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
          f.write(chunk)

    os.rename(temp_path, filepath)
    print(f"Successfully downloaded {filename}")
    return True
  
  except (requests.RequestException, IOError) as e:
    print(f"Failed for {filename}: {e}")
    # Clean up any partial files
    for path in [filepath + f".tmp", filepath]:
      if os.path.exists(path):
        try:
          os.remove(path)
        except:
          pass

  return False

if __name__ == "__main__":
  num_workers = 4
  num_files = 4

  os.makedirs(DATA_DIR, exist_ok=True)

  num_train_shards = MAX_SHARD if num_files == -1 else min(num_files, MAX_SHARD)
  ids_to_download = list(range(num_train_shards))
  ids_to_download.append(MAX_SHARD)

  with Pool(processes=num_workers) as pool:
    results = pool.map(download_single_file, ids_to_download)

  # Report results
  successful = sum(1 for success in results if success)
  print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
