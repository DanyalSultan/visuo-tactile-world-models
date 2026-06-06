import numpy as np

# Exact path to the joint file
joint_path = "data/RH20T_cfg7/task_0001_user_0014_scene_0001_cfg_0007/transformed/joint.npy"

def inspect_joints():
    print(f"Loading {joint_path}...\n")
    data = np.load(joint_path, allow_pickle=True).item()
    
    # 1. What are the actual keys?
    keys = list(data.keys())
    print(f"Actual Keys found: {keys}")
    
    if not keys:
        print("Dictionary is empty!")
        return
        
    # 2. Look inside the first key 
    first_key = keys[0]
    joint_data = data[first_key]
    
    print(f"\n--- Inside Key: '{first_key}' ---")
    print(f"Data Type: {type(joint_data)}")
    
    # If it's a dictionary of timestamps to joint angles
    if isinstance(joint_data, dict):
        timestamps = list(joint_data.keys())
        print(f"Total timestamps recorded: {len(timestamps)}")
        
        first_time = timestamps[0]
        sample_angles = joint_data[first_time]
        print(f"Sample Timestamp: {first_time}")
        print(f"Sample Joint Angles Shape: {np.array(sample_angles).shape}")
        print(f"Sample Joint Angles: {sample_angles}")
        
    # If it's a list/array like high_freq_data was in RH20main.py
    elif isinstance(joint_data, list) or isinstance(joint_data, np.ndarray):
        print(f"Length/Shape: {len(joint_data)}")
        print(f"First element: {joint_data[0]}")

if __name__ == "__main__":
    inspect_joints()