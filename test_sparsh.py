import torch
from huggingface_hub import snapshot_download
import os
import glob
from tactile_ssl.build_encoder import build_encoder

print("Downloading Sparsh-X (ALL modalities, Base size) weights from Hugging Face...")
# 1. Download the full multimodal model
local_dir = snapshot_download(repo_id="facebook/sparsh-x-all")

# 2. Find the exact checkpoint file
ckpt_files = glob.glob(os.path.join(local_dir, "*.pth")) + glob.glob(os.path.join(local_dir, "*.safetensors")) + glob.glob(os.path.join(local_dir, "*.bin"))
ckpt_path = ckpt_files[0] if ckpt_files else None
print(f"Found checkpoint at: {ckpt_path}")

config_path = "config/encoder/digit360_sparshx.yaml"

print("Building Sparsh-X encoder...")
device = "cpu"

# 3. Override the config to tell it to build the 'base' model size
overrides = ["model_size=base"]

sparsh_encoder = build_encoder(config_path, ckpt_path=ckpt_path, device=device, mode="eval", overrides=overrides)

print("Generating dummy multimodal tactile data...")
# Because we are using the multimodal model, we must provide dummy data for ALL sensors
tactile_img = torch.randn(1, 6, 224, 224).to(device)
tactile_audio = torch.randn(1, 224, 256).to(device)
tactile_imu = torch.randn(1, 224, 3).to(device)
tactile_pressure = torch.randn(1, 224, 1).to(device)

input_dict = {
    "img": tactile_img,
    "mic": tactile_audio,
    "imu": tactile_imu,
    "pressure": tactile_pressure,
}

print("Encoding multimodal tactile data into tokens...")
with torch.no_grad():
    tactile_rep = sparsh_encoder(input_dict)

print("\n--- SUCCESS! ---")
for key, value in tactile_rep.items():
    print(f"Output Token '{key}' Shape: {value.shape} (Dense features ready for the World Model!)")