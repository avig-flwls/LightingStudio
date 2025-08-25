import argparse
from ..io import read_exrs, write_exr
from .sph import project_env_to_coefficients, reconstruct_sph_coeffs_to_env
import torch
from coolname import generate_slug
from pathlib import Path
import json
import time

OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdri", type=str, nargs='+', required=True, help="List of HDRI file paths")
    parser.add_argument("--l_max", type=int, required=True, help="Maximum band index")
    args = parser.parse_args()

    experiment_name = generate_slug(2)
    output_dir = Path(OUTPUT_DIR) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hdris = read_exrs(args.hdri).to(device)

    for hdri_path, hdri in zip(args.hdri, hdris):
        hdri_path = Path(hdri_path)
        print(f"Processing {hdri_path}...")

        print("Projecting environment map to spherical harmonics...")
        H, W, _ = hdri.shape
        sph_coeffs, sph_basis = project_env_to_coefficients(hdri, l_max=args.l_max)

        print(f"Sph coeffs shape: {sph_coeffs.shape}")
        print(f"Sph basis shape: {sph_basis.shape}")

        print("Reconstructing environment map from spherical harmonics...")
        reconstructed_hdris = reconstruct_sph_coeffs_to_env(H, W, sph_coeffs, sph_basis)

        for l in range(args.l_max + 1):
            print(f"Reconstructed environment map for band {l}...")
            write_exr(reconstructed_hdris[l, ...], output_dir / f"{hdri_path.stem}_reconstructed_{l}.exr")
