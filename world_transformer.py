import torch
import torch.nn as nn

class PredictiveWorldModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        
        # 1. The Transformer Backbone
        # We use batch_first=True because our sequence is [Batch, Seq, Embed]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True 
        )
        # Stack multiple layers to build deeper reasoning
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 2. The Prediction Head (The Output)
        # A world model's job is to predict the *next* state. 
        # For simplicity, predict the next 7D Cartesian pose of the robot.
        self.pose_predictor = nn.Linear(embed_dim, 7)

    def forward(self, world_sequence):
        """
        Expects world_sequence of shape: [Batch, 1026, 256]
        """
        # 1. Pass the fused sequence through the Self-Attention layers
        # Shape remains [Batch, 1026, 256], but the features are now highly contextualised
        contextual_sequence = self.transformer(world_sequence)
        
        # 2. Extract the relevant token to make prediction
        # In the fusion layer, the proprioception token was the very last one added (index -1)
        # Grab its newly updated context to predict where the arm should move next
        proprio_context = contextual_sequence[:, -1, :] # Shape: [Batch, 256]
        
        # 3. Project back to physical 3D space
        next_pose_pred = self.pose_predictor(proprio_context) # Shape: [Batch, 7]
        
        return next_pose_pred

# Quick Test 
if __name__ == "__main__":
    # Simulating the output from fusion_layer.py
    dummy_world_sequence = torch.randn(4, 1026, 256)
    
    print("Initializing Predictive World Model...")
    model = PredictiveWorldModel(embed_dim=256, num_heads=8, num_layers=4)
    
    print("Running forward pass through Transformer...")
    predicted_pose = model(dummy_world_sequence)
    
    print("\n--- SUCCESS! ---")
    print(f"Input Sequence Shape: {dummy_world_sequence.shape}")
    print(f"Predicted Next Pose:  {predicted_pose.shape} (Ready for Loss Calculation!)")