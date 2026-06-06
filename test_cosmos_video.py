import torch
import os
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

# 1. Point to the Video weights
model_name = "Cosmos-0.1-Tokenizer-CV8x8x8"
checkpoint_path = f"pretrained_ckpts/{model_name}/encoder.jit"

# Force CPU
device = "cpu" 

print(f"Loading Cosmos Video Tokenizer ({model_name})...")
encoder = CausalVideoTokenizer(checkpoint_enc=checkpoint_path, device=device)

# Force the core PyTorch model into float32
encoder._enc_model = encoder._enc_model.to(torch.float32)

# 2. Generating Dummy Video Data
print("Generating dummy ALOHA video clip (9 frames)...")
input_tensor = torch.randn(1, 3, 9, 256, 256).to(device).to(torch.float32)

print(f"\nInput Video Shape (Robot's Eyes): {input_tensor.shape}")

print("Compressing video data into Cosmos Tokens...")
with torch.no_grad():
    # Feed the data directly into the raw PyTorch model
    raw_output = encoder._enc_model(input_tensor)
    
    # The raw model returns a tuple: (tokens, distribution_data)
    # We just want the tokens, which is the first item [0]
    vision_tokens = raw_output[0]

print("\n--- SUCCESS! ---")
print(f"Output Video Token Shape: {vision_tokens.shape}")