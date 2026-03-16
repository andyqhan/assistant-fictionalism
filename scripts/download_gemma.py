"""Pre-download Gemma models to HF cache for offline use on compute nodes."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not set in environment or .env"
print(f"HF_TOKEN loaded (length={len(token)})")

models = [
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]

cache_dir = "/scratch/ah7660/hf_cache"

for model in models:
    print(f"Downloading {model}...")
    snapshot_download(model, cache_dir=cache_dir, token=token)
    print(f"  Done: {model}")

print("All models downloaded.")
