import torch
import numpy as np
from pathlib import Path
from coolname import generate_slug
from src.LightingStudio.analysis.utils.io import write_exr
from src.LightingStudio.analysis.core.sph import get_cos_lobe_as_env_map
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
    env_map_of_cos_lobe = get_cos_lobe_as_env_map(H, W, device)  # Returns (H, W, 3)
   
    # Save as EXR
    output_path = output_dir / "env_map_of_cos_lobe.exr"
    print(f"Saving env_map_of_cos_lobe to: {output_path}")
    write_exr(env_map_of_cos_lobe, str(output_path))

    print("Complete!")

if __name__ == "__main__":
    main()
