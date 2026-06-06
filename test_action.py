import numpy as np
import os

task_folder = "data/RH20T_cfg7/task_0001_user_0014_scene_0001_cfg_0007"
action_path = f"{task_folder}/robot_command/tcpcommand_timestamp.npy"

def inspect_actions():
    print(f"Loading {action_path}...\n")
    
    # Load with pickle allowed, just in case it's an object array
    data = np.load(action_path, allow_pickle=True)
    
    # Unwrap it if it's a 0-dimensional dictionary/object
    if data.ndim == 0:
        data = data.item()
        
    print(f"Base Data Type: {type(data)}")
    
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Dictionary Keys Found: {keys}")
        
        # Look inside the first key
        first_key = keys[0]
        action_array = data[first_key]
        print(f"\n--- Inside Key: '{first_key}' ---")
        print(f"Total Commands Recorded: {len(action_array)}")
        print(f"First Command Data: {action_array[0]}")
        
    elif isinstance(data, np.ndarray):
        print(f"Array Shape: {data.shape}")
        print(f"First Row Data: {data[0]}")

if __name__ == "__main__":
    if os.path.exists(action_path):
        inspect_actions()
    else:
        print("Path not found! Double check the directory.")