import argparse
from .median_cut import median_cut_sampling, median_cut_sampling_to_cpu, visualize_samples
from ..io import read_exrs, write_exr
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



