import torch
import torch.nn as nn

class SparshTactileEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Input shape per frame: 2 fingers * 16 taxels * 3 axes = 96 raw features
        self.input_dim = 2 * 16 * 3
        
        # A classic 3-layer MLP architecture for continuous sensor data
        # We use GELU activation as it's the standard for modern Transformer architectures
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, tactile_tensor):
        """
        Expects tactile_tensor of shape: [Batch, 2, 16, 3]
        """
        # 1. Flatten the physical dimensions, but KEEP the Batch dimension
        # Shape goes from [Batch, 2, 16, 3] -> [Batch, 96]
        flat_tactile = tactile_tensor.view(tactile_tensor.size(0), -1)
        
        # 2. Pass the raw features through the projection network
        # Shape goes from [Batch, 96] -> [Batch, embed_dim]
        tactile_latent = self.network(flat_tactile)
        
        return tactile_latent

# --- Quick Test ---
if __name__ == "__main__":
    # Simulate the exact batch from RH20main.py
    dummy_tactile_batch = torch.randn(4, 2, 16, 3)
    
    print("Initializing Sparsh-X Encoder...")
    encoder = SparshTactileEncoder(embed_dim=256)
    
    print("Compressing tactile data...")
    latent_tokens = encoder(dummy_tactile_batch)
    
    print("\nSUCCESS!")
    print(f"Raw Input Shape:   {dummy_tactile_batch.shape}")
    print(f"Latent Embeddings: {latent_tokens.shape}")