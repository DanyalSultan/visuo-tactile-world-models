# aloha_main.py

import torch
import numpy as np
import h5py
import os
from tqdm import tqdm # Progress bar for the loop
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from tactile_ssl.build_encoder import build_encoder

# torch and numpy are the maths engines, Torch handles neural network tensors;
# Numpy handles raw data arrays from the file
# h5py allows the reading of high-speed data recorded by the ALOHA kit
# os for saving tokens to a directory
# tqdm for a progress bar
# cosmos_tokenizer (vision) and tactle_ssl are the two perception models

# --- 1. SETTINGS & HARDWARE ---
# On the ALOHA Workstation, this will automatically use the NVIDIA GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# DEVICE automatically detects whether on ALOHA (CUDA) or Laptop (CPU)
# DTYPE sets the mathematical precision depending on the device

# Paths to ALOHA recording and model weights
HDF5_PATH = "data/demo_0.hdf5"  # Update to own recorded episode
VIDEO_CKPT = "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/encoder.jit"
SPARSH_CONFIG = "config/encoder/digit360_sparshx.yaml"
SPARSH_CKPT = "pretrained_ckpts/sparsh_base.pth"

output_dir = "data/latents" # Directory where tokens will be stored

# --- 2. THE MULTIMODAL PERCEPTION ENGINE ---
class AlohaPerception:
    def __init__(self):
        print(f"--- Initializing AlohaPerception on {DEVICE} ---")
        
        # Initialise Cosmos Video
        self.v_encoder = CausalVideoTokenizer(checkpoint_enc=VIDEO_CKPT, device=DEVICE)
        self.v_encoder._enc_model = self.v_encoder._enc_model.to(DTYPE)
        
        # Initialise Sparsh-X (Tactile)
        self.s_encoder = build_encoder(SPARSH_CONFIG, ckpt_path=SPARSH_CKPT, device=DEVICE, overrides=["model_size=base"])

    def process_step(self, video_frames, tactile_img):
        """
        Transforms raw ALOHA sensor data into World Model Tokens.
        video_frames: [9, H, W, C] numpy array
        tactile_img: [H, W, C] numpy array from Taxim/GelSight
        """
        with torch.no_grad():
            # Prep Vision: Normalise and permute to [Batch, Chan, Time, H, W] as Cosmos expects
            v_tensor = torch.from_numpy(video_frames).permute(3, 0, 1, 2).float() / 255.0
            v_tensor = v_tensor.unsqueeze(0).to(DEVICE).to(DTYPE)

            # Prep Tactile: Normalise and permute to [Batch, Chan, H, W] as Sparsh expects
            t_tensor = torch.from_numpy(tactile_img).permute(2, 0, 1).float() / 255.0
            t_tensor = t_tensor.unsqueeze(0).to(DEVICE)
            
            # Run Encoders
            v_output = self.v_encoder._enc_model(v_tensor)
            vision_tokens = v_output[0].flatten(2).transpose(1, 2) # [1, Seq, 16]
            
            # Sparsh expects a dict
            tactile_dict = {"img": t_tensor}
            tactile_tokens = self.s_encoder(tactile_dict)["img"] # [1, 196, 768]

            return vision_tokens, tactile_tokens

# --- 3. DATASET LOADING, EXECUTION & SAVING ---
def run_pipeline():
    if not os.path.exists(HDF5_PATH):
        print(f"Error: Demo file {HDF5_PATH} not found. Ensure you have recorded data first.")
        return

    # Initialise the engine
    engine = AlohaPerception()
    os.makedirs(output_dir, exist_ok=True) # Create folder if it doesn't exist

    with h5py.File(HDF5_PATH, 'r') as root:
        # ALOHA 2.0 typically stores images at 30Hz or 50Hz
        # Determine total frames to process the entire demo
        total_frames = root['observations/qpos'].shape[0]
        print(f"Processing {total_frames} frames from {HDF5_PATH}...")
        
        # Loop through the demo using a 9-frame sliding window
        for i in tqdm(range(0, total_frames - 9)):
            
            # Extract Visuals (Top camera is standard for global view)
            video_window = root['observations/images/top'][i : i+9]
            
            # Extract Tactile (Assuming Taxim data was saved under this key)
            tactile_frame = root['observations/images/tactile'][i+8]
            
            # Extract Proprioception (Joint positions)
            qpos = root['observations/qpos'][i+8]
            
            # Extract Actions (To train the World Model on consequences)
            action = root['action'][i+8]

            # Tokenise
            v_tokens, t_tokens = engine.process_step(video_window, tactile_frame)

            # Save as synchronised PyTorch files (.pt)
            # Bundle all modalities into a single step file for easy training
            step_data = {
                "vision": v_tokens.cpu(),
                "tactile": t_tokens.cpu(),
                "qpos": torch.from_numpy(qpos),
                "action": torch.from_numpy(action)
            }
            
            torch.save(step_data, f"{output_dir}/latent_step_{i:04d}.pt")

    print(f"\n--- SUCCESS: ALL TOKENS SAVED TO {output_dir} ---")
    print("Ready for World Model Fusion.")

if __name__ == "__main__":
    run_pipeline()