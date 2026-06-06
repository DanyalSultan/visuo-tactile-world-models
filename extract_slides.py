import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Import custom synchronised dataloader
from RH20main import RH20T_Dataset

# Custom Styling Colours
BG_COLOR = '#100a2b'       # Deep purple/navy background
TEXT_COLOR = '#ffffff'     # White text
ACCENT_PINK = '#ff57a0'    # Neon magenta/pink
ACCENT_CYAN = '#4df0ff'    # Neon cyan
GRID_COLOR = '#ffffff'     # White (will be set to low opacity)

def apply_custom_style():
    """Applies the global dark theme to matplotlib"""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "font.family": "sans-serif",
        "font.weight": "bold"
    })

def generate_presentation_visuals():
    apply_custom_style()
    print("LOADING SYNCHRONIZED DATASET")
    dataset = RH20T_Dataset("data", "RH20T_cfg7/task_0001_user_0014_scene_0001_cfg_0007")

    # 1. Map the tactile timeline
    raw_magnitudes = []
    print(f"Scanning {len(dataset)} aligned frames for contact events...")
    
    for idx in range(len(dataset)):
        tactile_tensor = dataset[idx]['tactile']
        force_magnitude = torch.sum(torch.abs(tactile_tensor)).item()
        raw_magnitudes.append(force_magnitude)

    # NORMALISE THE DATA (ZERO THE SENSOR)
    min_force = min(raw_magnitudes)
    normalized_forces = [f - min_force for f in raw_magnitudes]
    max_force = max(normalized_forces)

    # 2. Plot and save the master timeline
    plt.figure(figsize=(12, 4))
    plt.plot(normalized_forces, color=ACCENT_CYAN, linewidth=2.5, label='Calibrated Force')
    
    plt.title("Calibrated Tactile Force over Time", fontsize=14, color=ACCENT_PINK, pad=15)
    plt.xlabel("Synchronized Frame Index", fontsize=12)
    plt.ylabel("Relative Force Magnitude", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.15)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_alpha(0.3)
    plt.gca().spines['left'].set_alpha(0.3)

    plt.tight_layout()
    plt.savefig("tactile_timeline.png", facecolor=BG_COLOR, edgecolor='none', dpi=300)
    print("\n Saved calibrated master timeline: 'tactile_timeline.png'")

# 3. Helper function with MANUAL VIDEO OVERRIDE
    def save_slide_image(tactile_idx, manual_video_idx, filename):
        # Bypass the broken dataloader sync and grab the raw video frame directly
        # Ensure we don't go out of bounds of the actual video length
        safe_vid_idx = min(manual_video_idx, len(dataset.frames) - 1)
        
        # Grab the raw RGB frame from memory
        image = dataset.frames[safe_vid_idx]
        
        # Grab the correctly aligned tactile force
        force = normalized_forces[tactile_idx] 
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})
        fig.patch.set_facecolor(BG_COLOR)
        
        # Left Side: Vision
        ax1.imshow(image)
        ax1.set_title(f"Visual Modality", fontsize=16, color=TEXT_COLOR, pad=15)
        ax1.axis('off')
        
        # Right Side: Tactile
        bar_color = ACCENT_PINK if force > (max_force * 0.1) else ACCENT_CYAN
        display_force = force if force > 0 else (max_force * 0.02)
        
        bars = ax2.bar(['Gripper Force'], [display_force], color=bar_color, width=0.4)
        ax2.set_ylim(0, max_force * 1.1) 
        
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + (max_force*0.02), 
                     f'{force:,.0f}', ha='center', va='bottom', color=bar_color, fontsize=14, fontweight='bold')

        ax2.set_title("Tactile Modality", fontsize=16, color=TEXT_COLOR, pad=15)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.15)
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_alpha(0.3)
        ax2.spines['bottom'].set_alpha(0.3)
        
        plt.tight_layout()
        plt.savefig(filename, facecolor=BG_COLOR, edgecolor='none', dpi=300)
        plt.close()
        print(f"Generated Slide Visual: {filename}")

    # --- GENERATE THE PRESENTATION VISUALS (MANUAL OVERRIDE) ---
    print("\n--- GENERATING SLIDES ---")
    # format: save_slide_image(tactile_index, manual_video_frame, filename)
    
# --- FINAL REFINED SLIDE GENERATION ---
    print("\n--- GENERATING SLIDES ---")
    
    # 1. THE APPROACH: Clear visibility, zero force.
    save_slide_image(50, 20, "slide_1_approach.png")         
    
    # 2. THE OCCLUSION (The Blind Spot): 
    # Use Frame 60 for Video. Use Tactile Index 140 (Before the spike).
    # RESULT: Gripper blocks the view, but the bar stays BLUE.
    save_slide_image(140, 89, "slide_2_hover_occlusion.png") 
    
    # 3. THE CONTACT: 
    # Use Frame 90 for Video. Use Tactile Index 155 (After the spike).
    # RESULT: Gripper is at the same spot, but the bar turns PINK.
    save_slide_image(155, 90, "slide_3_grasp_contact.png")

if __name__ == "__main__":
    generate_presentation_visuals()