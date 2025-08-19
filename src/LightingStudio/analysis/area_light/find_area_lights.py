import argparse
from ..point_light.median_cut import median_cut_sampling, median_cut_sampling_to_cpu, visualize_samples
from .density_estimation import expand_map_exact, expand_map_fast
from ..utils import read_exrs, write_exr
import torch
from coolname import generate_slug
from pathlib import Path
import json

OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"

# KEEP IN MIND: this works only if number of n_samples > 1024

# python -m src.LightingStudio.analysis.area_light.find_area_lights --hdri "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Bakery.exr" --n_samples 1024

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

        samples_cpu = median_cut_sampling_to_cpu(hdri, args.n_samples)
        density_map_exact = expand_map_exact(hdri, samples_cpu, min_count=4, normalize=True)
        density_map_fast = expand_map_fast(hdri, samples_cpu, min_count=4, normalize=True)

        # Save density map
        write_exr(density_map_exact, output_dir / f"{hdri_path.stem}_density_map_exact.exr")
        write_exr(density_map_fast, output_dir / f"{hdri_path.stem}_density_map_fast.exr")

        # Save samples visualization
        vis_hdri = visualize_samples(hdri, samples_cpu)
        write_exr(vis_hdri, output_dir / f"{hdri_path.stem}_median_cut.exr")

        # Save samples - convert to dict for JSON serialization
        samples_dict = [sample.to_dict() for sample in samples_cpu]
        with open(output_dir / f"{hdri_path.stem}.json", "w") as f:
            json.dump(samples_dict, f, indent=2)


 
