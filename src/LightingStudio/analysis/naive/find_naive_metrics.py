import argparse
from ..io import read_exrs, write_exr
from .intensity_calculation import naive_metrics, naive_metrics_cpu
import torch
from coolname import generate_slug
from pathlib import Path
import json
import time
import logging

OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"

# python -m src.LightingStudio.analysis.naive.find_naive_metrics --hdri "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Bakery.exr"
# python -m src.LightingStudio.analysis.naive.find_naive_metrics --hdri "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Bakery.exr" "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Games Room 02.exr" "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Greenhouse.exr"

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
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging")
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("Debug logging enabled")

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
        logger.info("Starting median cut analysis...")
        
        # Time median cut sampling
        start_time = time.time()
        metrics_cpu = naive_metrics_cpu(hdri)
        sampling_time = time.time() - start_time
        logger.info(f"Naive metrics complete in {sampling_time:.2f} seconds.")

        # Save metrics - convert to dict for JSON serialization
        metrics_dict = metrics_cpu.to_dict()
        with open(output_dir / f"{hdri_path.stem}.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Log timing summary
        total_time = sampling_time
        logger.info(f"Timing Summary for {hdri_path.stem}:")
        logger.info(f"  Naive metrics: {sampling_time:.2f}s")
        logger.info(f"  Total analysis time: {total_time:.2f}s")
        logger.info("-" * 50)
 
