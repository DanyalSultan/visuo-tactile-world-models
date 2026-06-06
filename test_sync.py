import os
import json
import numpy as np

task_folder = "data/RH20T_cfg7/task_0001_user_0014_scene_0001_cfg_0007"
tactile_path = f"{task_folder}/transformed/tactile.npy"
hf_path = f"{task_folder}/transformed/high_freq_data.npy"
meta_path = f"{task_folder}/metadata.json"

def investigate_clocks():
    print("--- Loading Data ---\n")
    tactile = np.load(tactile_path)
    hf = np.load(hf_path, allow_pickle=True).item()['base']
    
    # 1. Look at the Tactile Metadata columns (first 3 numbers)
    print("TACTILE METADATA (First 5 frames):")
    for i in range(5):
        print(f"Frame {i}: {tactile[i][:3]}")
        
    # 2. Look at the TCP Timestamps
    print("\nTCP TIMESTAMPS (First 5 frames):")
    for i in range(5):
        print(f"Frame {i}: {hf[i]['timestamp']}")
        
    # 3. Check the scene metadata for clues
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print("\n--- Scene Metadata.json ---")
        for key, value in meta.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    investigate_clocks()