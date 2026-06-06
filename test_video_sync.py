import cv2
import numpy as np

# --- Paths ---
task_folder = "data/RH20T_cfg7/task_0001_user_0014_scene_0001_cfg_0007"
cam_folder = f"{task_folder}/cam_037522061512"

video_path = f"{cam_folder}/color.mp4"
vision_time_path = f"{cam_folder}/timestamps.npy"
tactile_path = f"{task_folder}/transformed/tactile.npy"
hf_path = f"{task_folder}/transformed/high_freq_data.npy"

def test_sync():
    print("--- 1. Loading Timestamps ---")
    # Load 118 Video Timestamps
    vid_timestamps = np.load(vision_time_path, allow_pickle=True).item()['color']
    
    # Load and align the 1980 High-Frequency Timestamps (just like in RH20main.py)
    tactile_raw = np.load(tactile_path)
    hf_data = np.load(hf_path, allow_pickle=True).item()['base']
    num_frames = min(len(tactile_raw), len(hf_data))
    hf_aligned = hf_data[-num_frames:] 
    
    # Extract just the timestamps from the dictionary
    hf_timestamps = np.array([frame['timestamp'] for frame in hf_aligned])
    
    print(f"High-Freq Frames: {len(hf_timestamps)}")
    print(f"Video Frames: {len(vid_timestamps)}")

    print("\n--- 2. Performing Nearest Neighbor Sync ---")
    vid_idx_mapping = []
    
    for hf_t in hf_timestamps:
        # Calculate the absolute time difference between this HF frame and ALL video frames
        time_diffs = np.abs(vid_timestamps - hf_t)
        # Find the index of the video frame with the smallest time difference
        closest_idx = np.argmin(time_diffs)
        vid_idx_mapping.append(closest_idx)
        
    print(f"Successfully mapped {len(vid_idx_mapping)} HF frames to visual frames.")
    print(f"Unique video frames utilized: {len(set(vid_idx_mapping))} / {len(vid_timestamps)}")

    print("\n--- 3. Extracting MP4 into RAM ---")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        # Convert OpenCV BGR to standard RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    print(f"Extracted {len(frames)} physical image arrays.")
    
    # Sanity Check
    print("\n--- Alignment Sanity Check ---")
    print(f"HF Frame 0    --> maps to Video Frame {vid_idx_mapping[0]}")
    print(f"HF Frame 1000 --> maps to Video Frame {vid_idx_mapping[1000]}")
    print(f"HF Frame 1979 --> maps to Video Frame {vid_idx_mapping[-1]}")
    
    # Test grabbing an actual synchronized image
    test_image = frames[vid_idx_mapping[1000]]
    print(f"\nShape of synchronized image at timestep 1000: {test_image.shape}")

if __name__ == "__main__":
    test_sync()