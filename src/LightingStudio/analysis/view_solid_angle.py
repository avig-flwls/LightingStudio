import torch
import numpy as np
from pathlib import Path
from coolname import generate_slug
from .utils import pixel_solid_angles
from .io import write_exr
from einops import repeat


OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"

def main():
    # Set up dimensions (common HDRI dimensions)
    H, W = 1024, 2048
    
    # Generate random experiment name
    experiment_name = generate_slug(2)
    output_dir = Path(OUTPUT_DIR) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Call pixel_solid_angles function
    print(f"Computing pixel solid angles for {H}x{W} image...")
    solid_angles = pixel_solid_angles(H, W, device)  # Returns (H, 1)
    print(f"Solid angles shape: {solid_angles.shape}")
    print(f"Solid angles range: [{solid_angles.min():.6f}, {solid_angles.max():.6f}]")
    
    # Convert to 3-channel format for EXR (RGB format expected by write_exr)
    # Broadcast the solid angle values to all 3 channels
    solid_angles_3ch = repeat(solid_angles, "h 1 -> h w 3", w=W)  # (H, W, 3)
    
    # Save as EXR
    output_path = output_dir / "pixel_solid_angles.exr"
    print(f"Saving solid angles to: {output_path}")
    write_exr(solid_angles_3ch, str(output_path))
    
    print("Complete!")
    print(f"Solid angle statistics:")
    print(f"  Total solid angle: {solid_angles.sum().item() * W:.6f} steradians")
    print(f"  Expected total (4π): {4 * np.pi:.6f} steradians")
    print(f"  Ratio: {solid_angles.sum().item() * W / (4 * np.pi):.6f}")

if __name__ == "__main__":
    main()
