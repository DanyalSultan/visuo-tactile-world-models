import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
from tqdm import tqdm

# Import custom modules
from RH20main import RH20T_Dataset
from cosmos_tokenizer.image_lib import ImageTokenizer
from tactile_encoder import SparshTactileEncoder
from fusion_layer import MultimodalFusion
from world_transformer import PredictiveWorldModel

def train():
    # --- AUTOMATIC DEVICE DETECTION ---
    # Automatically use NVIDIA GPUs if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- INITIALIZING WORLD MODEL ON {device.type.upper()}  ---")

    # 1. Dynamic Multitask Dataset
    data_dir = "data"
    cfg_folder = "RH20T_cfg7"
    full_cfg_path = os.path.join(data_dir, cfg_folder)
    
    print(f"Scanning {full_cfg_path} for tasks...")
    all_task_folders = [f for f in os.listdir(full_cfg_path) if f.startswith("task_")]
    
    datasets = []
    for task_folder in all_task_folders:
        try:
            # Create a dataset for each individual task
            task_path = f"{cfg_folder}/{task_folder}"
            datasets.append(RH20T_Dataset(data_dir, task_path))
        except Exception as e:
            print(f"Skipping {task_folder} due to missing data/error: {e}")
            
    # Glue them all together into one massive dataset
    master_dataset = ConcatDataset(datasets)
    print(f"Successfully loaded {len(datasets)} tasks.")
    print(f"Total globally aligned frames available: {len(master_dataset)}")
    
    # Increase num_workers for faster data loading on multi-core servers
    dataloader = DataLoader(master_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. Vision Pipeline (Cosmos)
    print("Loading Cosmos Vision Tokenizer...")
    vision_transform = T.Resize((256, 256))
    vision_encoder = ImageTokenizer(checkpoint_enc="pretrained_ckpts/Cosmos-0.1-Tokenizer-CI8x8/encoder.jit", device=device)
    
    # 3. Custom Deep Learning Modules
    print("Loading Sparsh-X, Fusion Layer, and Transformer...")
    tactile_encoder = SparshTactileEncoder(embed_dim=256).to(device)
    fusion_layer = MultimodalFusion(embed_dim=256).to(device)
    world_model = PredictiveWorldModel(embed_dim=256, num_heads=8, num_layers=4).to(device)

    # 4. Optimiser & Loss Function
    trainable_params = list(tactile_encoder.parameters()) + \
                       list(fusion_layer.parameters()) + \
                       list(world_model.parameters())
                       
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    criterion = nn.MSELoss()

    # --- THE TRAINING LOOP ---
    epochs = 50 # Bumped up for the cluster
    print("\n--- STARTING TRAINING ---")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Move all tensors to the target device (GPU)
            b_vision = batch['vision'].to(device, non_blocking=True)
            b_tactile = batch['tactile'].to(device, non_blocking=True)
            b_proprio = batch['proprio'].to(device, non_blocking=True)
            b_action = batch['action'].to(device, non_blocking=True)
            b_target = batch['target'].to(device, non_blocking=True)

            optimizer.zero_grad()

            # Encode Vision
            v_input = vision_transform(b_vision).to(torch.bfloat16) / 255.0
            with torch.no_grad():
                (vision_latents,) = vision_encoder.encode(v_input)
            vision_latents = vision_latents.to(torch.float32)

            # Encode Tactile & Fuse
            tactile_latents = tactile_encoder(b_tactile)
            world_sequence = fusion_layer(vision_latents, tactile_latents, b_proprio, b_action)

            # Predict & Backprop
            predicted_pose = world_model(world_sequence)
            loss = criterion(predicted_pose, b_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"MSE Loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss/len(dataloader)
        print(f" Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f}")
        
        # Save checkpoints after every epoch!
        torch.save({
            'epoch': epoch,
            'tactile_state': tactile_encoder.state_dict(),
            'fusion_state': fusion_layer.state_dict(),
            'world_state': world_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"world_model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()