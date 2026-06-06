from huggingface_hub import snapshot_download
import os

# 'CV' (Continuous Video) instead of 'CI'
model_name = "Cosmos-0.1-Tokenizer-CV8x8x8" 
hf_repo = "nvidia/" + model_name
local_dir = "pretrained_ckpts/" + model_name

os.makedirs(local_dir, exist_ok=True)
print(f"Downloading {model_name}...")

snapshot_download(repo_id=hf_repo, local_dir=local_dir)
print("Download complete!")