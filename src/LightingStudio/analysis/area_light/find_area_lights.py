from ..io import read_exrs, write_exr
import argparse
from ..point_light.median_cut import median_cut_sampling, median_cut_sampling_to_cpu, visualize_samples
from .density_estimation import expand_map_exact, expand_map_fast
import torch
from coolname import generate_slug
from pathlib import Path
import json
import time
import logging

OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"

# python -m src.LightingStudio.analysis.area_light.find_area_lights --hdri "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Bakery.exr" --n_samples 1024
# python -m src.LightingStudio.analysis.area_light.find_area_lights --hdri "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Bakery.exr" "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Games Room 02.exr" "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Factory Canteen 02.exr" --n_samples 1024


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to console
        ]
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdri", type=str, nargs='+', required=True, help="List of HDRI file paths")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples (must be > 1024)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging")
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("Debug logging enabled")

    if args.n_samples < 1024:
        parser.error("The value of --n_samples must be greater than 1024.")

    experiment_name = generate_slug(2)
    output_dir = Path(OUTPUT_DIR) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set up logger for this script
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    hdris = read_exrs(args.hdri).to(device)
    logger.info(f"Loaded {len(hdris)} HDRI files")

    for hdri_path, hdri in zip(args.hdri, hdris):
        hdri_path = Path(hdri_path)
        print(f"Processing {hdri_path}...")
        logger.info(f"Processing {hdri_path.name} with shape {hdri.shape}")

        # Start analysis
        print("Starting analysis...")
        logger.info("Starting median cut analysis...")
        
        # Time median cut sampling
        start_time = time.time()
        samples_cpu = median_cut_sampling_to_cpu(hdri, args.n_samples)
        sampling_time = time.time() - start_time
        logger.info(f"Median cut sampling complete in {sampling_time:.2f} seconds.")
        
        # Time exact density map
        start_time = time.time()
        density_map_exact = expand_map_exact(hdri, samples_cpu, min_count=4, normalize=True)
        exact_time = time.time() - start_time
        logger.info(f"Exact density map complete in {exact_time:.2f} seconds.")
        
        # Time fast density map
        start_time = time.time()
        density_map_fast = expand_map_fast(hdri, samples_cpu, min_count=4, normalize=True)
        fast_time = time.time() - start_time
        logger.info(f"Fast density map complete in {fast_time:.2f} seconds.")

        # Save density map
        write_exr(density_map_exact, output_dir / f"{hdri_path.stem}_density_map_exact.exr")
        write_exr(density_map_fast, output_dir / f"{hdri_path.stem}_density_map_fast.exr")

        # Time samples visualization
        start_time = time.time()
        vis_hdri = visualize_samples(hdri, samples_cpu)
        vis_time = time.time() - start_time
        logger.info(f"Samples visualization complete in {vis_time:.2f} seconds.")
        write_exr(vis_hdri, output_dir / f"{hdri_path.stem}_median_cut.exr")

        # Save samples - convert to dict for JSON serialization
        samples_dict = [sample.to_dict() for sample in samples_cpu]
        with open(output_dir / f"{hdri_path.stem}.json", "w") as f:
            json.dump(samples_dict, f, indent=2)
        
        # Log timing summary
        total_time = sampling_time + exact_time + fast_time + vis_time
        logger.info(f"Timing Summary for {hdri_path.stem}:")
        logger.info(f"  Median cut sampling: {sampling_time:.2f}s")
        logger.info(f"  Exact density map:   {exact_time:.2f}s")
        logger.info(f"  Fast density map:    {fast_time:.2f}s")
        logger.info(f"  Visualization:       {vis_time:.2f}s")
        logger.info(f"  Total analysis time: {total_time:.2f}s")
        logger.info("-" * 50)


 
