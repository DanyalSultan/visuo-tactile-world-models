import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RH20T_Dataset(Dataset):
    def __init__(self, data_dir, task_name, cam_serial="037522061512"):
        self.task_path = os.path.join(data_dir, task_name)
        
        # 1. Paths
        self.tactile_path = os.path.join(self.task_path, "transformed/tactile.npy")
        self.hf_path = os.path.join(self.task_path, "transformed/high_freq_data.npy")
        self.cam_folder = os.path.join(self.task_path, f"cam_{cam_serial}")
        self.action_path = os.path.join(self.task_path, "robot_command/tcpcommand_timestamp.npy")

        # 2. Extract MP4 to RAM FIRST
        video_path = os.path.join(self.cam_folder, "color.mp4")
        cap = cv2.VideoCapture(video_path)
        self.frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # 3. Load Timestamps & Fix Units
        vision_time_path = os.path.join(self.cam_folder, "timestamps.npy")
        self.vid_timestamps = np.load(vision_time_path, allow_pickle=True).item()['color']
        
        self.hf_data = np.load(self.hf_path, allow_pickle=True).item()['base']
        hf_raw_timestamps = np.array([f['timestamp'] for f in self.hf_data])

        # Timestamp Unit Normalization
        if np.mean(self.vid_timestamps) > np.mean(hf_raw_timestamps) * 100:
            self.vid_timestamps = self.vid_timestamps / 1000.0  
        elif np.mean(hf_raw_timestamps) > np.mean(self.vid_timestamps) * 100:
            hf_raw_timestamps = hf_raw_timestamps / 1000.0      

        # Video Overlap & Dropped Packet Clamp
        vid_start = self.vid_timestamps[0]
        vid_end = self.vid_timestamps[-1]
        
        tactile_raw = np.load(self.tactile_path)
        
        # Find the absolute minimum safe length between the high-freq sensors
        safe_max_len = min(len(self.hf_data), len(tactile_raw))
        
        # Find the indices where the HF data actually overlaps with the Video
        valid_indices = np.where((hf_raw_timestamps[:safe_max_len] >= vid_start) & 
                                 (hf_raw_timestamps[:safe_max_len] <= vid_end))[0]
        
        if len(valid_indices) == 0:
            # Fallback if timestamps are completely corrupted
            start_idx, end_idx = 0, min(safe_max_len - 1, len(self.frames) * 20)
        else:
            start_idx, end_idx = valid_indices[0], valid_indices[-1]

        # Slice arrays safely
        self.hf_aligned = self.hf_data[start_idx : end_idx + 1]
        self.tactile_aligned = tactile_raw[start_idx : end_idx + 1]
        
        # Set absolute number of frames
        self.num_frames = min(len(self.hf_aligned), len(self.tactile_aligned))
        
        # Map video frames to HF frames safely
        hf_timestamps = hf_raw_timestamps[start_idx : end_idx + 1]
        self.vid_idx_mapping = [np.argmin(np.abs(self.vid_timestamps - t)) for t in hf_timestamps[:self.num_frames]]

        # 4. Action Sync: Nearest Neighbor
        self.action_raw = np.load(self.action_path)
        action_timestamps = self.action_raw[:, -1]
        
        if np.mean(action_timestamps) > np.mean(hf_raw_timestamps) * 100:
            action_timestamps = action_timestamps / 1000.0
            
        self.action_idx_mapping = [np.argmin(np.abs(action_timestamps - t)) for t in hf_timestamps[:self.num_frames]]

    def __len__(self):
        return self.num_frames - 1 

    def __getitem__(self, idx):
        # --- TACTILE ---
        tactile_tensor = torch.tensor(self.tactile_aligned[idx][3:].reshape((2, 16, 3)), dtype=torch.float32)
        # --- PROPRIOCEPTION ---
        proprio_tensor = torch.tensor(self.hf_aligned[idx]['tcp'], dtype=torch.float32)
        # --- VISION ---
        vision_tensor = torch.tensor(self.frames[self.vid_idx_mapping[idx]], dtype=torch.float32).permute(2, 0, 1)
        
        # --- ACTION COMMAND ---
        action_idx = self.action_idx_mapping[idx]
        action_tensor = torch.tensor(self.action_raw[action_idx][:6], dtype=torch.float32)

        # --- THE TARGET (t+1) ---
        target_tcp = self.hf_aligned[idx + 1]['tcp']
        target_tensor = torch.tensor(target_tcp, dtype=torch.float32)

        return {
            "vision": vision_tensor,
            "tactile": tactile_tensor,
            "proprio": proprio_tensor,
            "action": action_tensor,
            "target": target_tensor
        }