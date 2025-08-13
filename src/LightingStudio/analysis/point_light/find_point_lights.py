import argparse
from .median_cut import median_cut_sampling, median_cut_sampling_to_cpu, visualize_samples
from ..utils import read_exrs, write_exr
import torch
from coolname import generate_slug
from pathlib import Path
import json

OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"

# python -m src.LightingStudio.analysis.point_light.find_point_lights --hdri "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Bakery.exr" --n_samples 32

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdri", type=str, nargs='+', required=True, help="List of HDRI file paths")
    parser.add_argument("--n_samples", type=int, required=True)
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

        samples = median_cut_sampling_to_cpu(hdri, args.n_samples)

        vis_hdri = visualize_samples(hdri, samples)
        write_exr(vis_hdri, output_dir / f"{hdri_path.stem}_median_cut.exr")

        # Save samples
        with open(output_dir / f"{hdri_path.stem}.json", "w") as f:
            json.dump(samples, f, indent=2)
 
