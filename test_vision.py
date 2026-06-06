import cv2
import numpy as np
import os

cam_folder = "data/RH20T_cfg7/task_0001_user_0014_scene_0001_cfg_0007/cam_037522061512"
video_path = f"{cam_folder}/color.mp4"
timestamp_path = f"{cam_folder}/timestamps.npy"

def inspect_vision():
    print(f"Loading Vision Data from: {cam_folder}...\n")
    
    # 1. Load the dictionary and extract the color array
    cam_timestamps_dict = np.load(timestamp_path, allow_pickle=True).item()
    color_timestamps = cam_timestamps_dict['color']
    
    print(f"Color Timestamps Found: {len(color_timestamps)}")
    print(f"First 5 Timestamps: {color_timestamps[:5]}")
    
    # 2. Open the video file natively
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"\n--- MP4 File Stats ---")
    print(f"Total MP4 Frames: {frame_count}")
    print(f"Recorded FPS: {fps}")
    
    # 3. Check for dropped frames
    if len(color_timestamps) == frame_count:
        print("\nSuccess! The timestamps perfectly match the MP4 frames.")
    else:
        print(f"\nWarning: Frame count mismatch!")

if __name__ == "__main__":
    inspect_vision()