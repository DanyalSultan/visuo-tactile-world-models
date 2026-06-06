import numpy as np

tactile_file = "data/RH20T_cfg7/task_0001_user_0014_scene_0001_cfg_0007/transformed/tactile.npy"

def format_for_sparsh():
    # 1. Load the raw array
    data = np.load(tactile_file)
    print(f"Raw shape: {data.shape}")
    
    # 2. Grab the very first frame to test
    frame_0 = data[0]
    
    # 3. Slice the array 
    # (Assuming the first 3 elements are timestamps/IDs and the last 96 are tactile)
    metadata = frame_0[:3]
    tactile_flat = frame_0[3:]
    
    # 4. Reshape into the (Fingers, Taxels, Axes) format the paper specified
    try:
        tactile_3d = tactile_flat.reshape((2, 16, 3))
        
        print("\n--- Success! Data Ready for Sparsh-X ---")
        print(f"Extracted Metadata (Timestamps): {metadata}")
        print(f"Reshaped Tactile Tensor: {tactile_3d.shape}")
        
    except ValueError as e:
        print(f"Reshape failed. The metadata might be at the end of the array instead of the beginning.")

if __name__ == "__main__":
    format_for_sparsh()