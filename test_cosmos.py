import torch
import torchvision.transforms as T
from torchvision.io import read_image
import os
from cosmos_tokenizer.image_lib import ImageTokenizer

# 1. Define model and paths
model_name = "Cosmos-0.1-Tokenizer-CI8x8"
checkpoint_path = f"pretrained_ckpts/{model_name}/encoder.jit"

# Stick to CPU for this initial test to avoid any AMD/DirectML JIT conflicts
device = "cpu" 

# 2. filename for use when lab data is collected
lab_image_path = "lab_robot_view.jpg"

print(f"Loading Cosmos Vision Tokenizer ({model_name})...")
encoder = ImageTokenizer(checkpoint_enc=checkpoint_path, device=device)

print("\nChecking for real lab data...")
if os.path.exists(lab_image_path):
    print(f"-> Found real image: '{lab_image_path}'! Loading it now...")
    
    # Read the raw RGB image (Output is [Channels, Height, Width] in 0-255)
    raw_image = read_image(lab_image_path)
    
    # Cosmos minimum resolution is 256px, and it needs to be forced into a standard shape
    transform = T.Compose([
        T.Resize((256, 256)), 
    ])
    
    # Add the Batch dimension at the front: [Batch, Channels, Height, Width]
    # And convert to bfloat16 (the mathematical format Cosmos was trained on)
    input_tensor = transform(raw_image).unsqueeze(0).to(device).to(torch.bfloat16)
    
    # Normalize pixel values from [0, 255] to standard float ranges [0, 1]
    input_tensor = input_tensor / 255.0

else:
    print(f"-> No real image found named '{lab_image_path}'.")
    print("-> Generating dummy lab camera data (Static Noise)...")
    
    # Generate fake RGB image [Batch, 3-Channels, 256-Height, 256-Width]
    input_tensor = torch.randn(1, 3, 256, 256).to(device).to(torch.bfloat16)

print(f"\nInput Visual Data Shape (Robot's Eyes): {input_tensor.shape}")

print("Compressing visual data into Cosmos Tokens...")
with torch.no_grad():
    # Run the encoder!
    (vision_tokens,) = encoder.encode(input_tensor)

print("\n--- SUCCESS! ---")
print(f"Output Vision Token Shape: {vision_tokens.shape}")
print("(These dense visual features are now ready to be glued to your Sparsh tactile tokens!)")