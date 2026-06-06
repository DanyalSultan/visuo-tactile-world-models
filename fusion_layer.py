import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Projectors
        self.vision_proj = nn.Linear(16, embed_dim)
        self.proprio_proj = nn.Sequential(nn.Linear(7, 128), nn.GELU(), nn.Linear(128, embed_dim))
        
        # Action Projector (Projects 6-DoF command to 256)
        self.action_proj = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, embed_dim))
        
        # Modality Tags
        self.vision_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.tactile_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.proprio_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.action_embed = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, vision_latents, tactile_latents, proprio_raw, action_raw):
        B = vision_latents.size(0)
        
        # Vision [B, 1024, 256]
        v_tokens = self.vision_proj(vision_latents.view(B, 16, -1).permute(0, 2, 1)) + self.vision_embed
        
        # Tactile [B, 1, 256]
        t_tokens = tactile_latents.unsqueeze(1) + self.tactile_embed
        
        # Proprioception [B, 1, 256]
        p_tokens = self.proprio_proj(proprio_raw).unsqueeze(1) + self.proprio_embed
        
        # Action [B, 1, 256]
        a_tokens = self.action_proj(action_raw).unsqueeze(1) + self.action_embed
        
        # Concatenate everything (1024 + 1 + 1 + 1 = 1027)
        world_sequence = torch.cat([v_tokens, t_tokens, p_tokens, a_tokens], dim=1)
        
        return world_sequence

if __name__ == "__main__":
    fusion = MultimodalFusion(embed_dim=256)
    dummy_vision = torch.randn(4, 16, 32, 32)
    dummy_tactile = torch.randn(4, 256)
    dummy_proprio = torch.randn(4, 7)
    dummy_action = torch.randn(4, 6)
    
    world_state = fusion(dummy_vision, dummy_tactile, dummy_proprio, dummy_action)
    print(f"Final Action-Conditioned Transformer Sequence: {world_state.shape}")