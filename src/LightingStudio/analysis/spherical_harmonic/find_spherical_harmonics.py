import argparse
from ..io import read_exrs, write_exr
from .sph import project_env_to_coefficients, reconstruct_sph_coeffs_to_env
from .sph_metrics import get_sph_metrics, get_sph_metrics_cpu, visualize_sph_metrics
import torch
from coolname import generate_slug
from pathlib import Path
import json
import time
import logging

OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"

# python -m src.LightingStudio.analysis.spherical_harmonic.find_spherical_harmonics --hdri "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Games Room 02.exr" --l_max 4   

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
    parser.add_argument("--l_max", type=int, required=True, help="Maximum band index")
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
        logger.info("Starting spherical harmonics analysis...")
        
        # Time spherical harmonics projection
        start_time = time.time()
        logger.info("Projecting environment map to spherical harmonics...")
        H, W, _ = hdri.shape
        sph_coeffs, sph_basis = project_env_to_coefficients(hdri, l_max=args.l_max)
        projection_time = time.time() - start_time
        logger.info(f"Spherical harmonics projection complete in {projection_time:.2f} seconds.")

        logger.info(f"Sph coeffs shape: {sph_coeffs.shape}")
        logger.info(f"Sph basis shape: {sph_basis.shape}")

        # Time SPH metrics computation
        start_time = time.time()
        logger.info("Computing SPH metrics...")
        sph_metrics_cpu = get_sph_metrics_cpu(hdri, l_max=args.l_max)
        metrics_time = time.time() - start_time
        logger.info(f"SPH metrics computation complete in {metrics_time:.2f} seconds.")
        
        # Log SPH metrics information
        logger.info("SPH Metrics Summary:")
        logger.info(f"  Dominant direction: {sph_metrics_cpu.dominant_direction}")
        logger.info(f"  Dominant pixel: {sph_metrics_cpu.dominant_pixel}")
        logger.info(f"  Dominant color: {sph_metrics_cpu.dominant_color}")
        logger.info(f"  Area intensity: {sph_metrics_cpu.area_intensity}")

        # Time visualization creation
        start_time = time.time()
        logger.info("Creating SPH metrics visualization...")
        vis_hdri = visualize_sph_metrics(hdri, sph_metrics_cpu)
        visualization_time = time.time() - start_time
        logger.info(f"Visualization creation complete in {visualization_time:.2f} seconds.")

        # Time reconstruction process
        start_time = time.time()
        logger.info("Reconstructing environment map from spherical harmonics...")
        reconstructed_hdris = reconstruct_sph_coeffs_to_env(H, W, sph_coeffs, sph_basis)
        reconstruction_time = time.time() - start_time
        logger.info(f"Reconstruction complete in {reconstruction_time:.2f} seconds.")

        # Time output writing
        start_time = time.time()
        
        # Write SPH metrics visualization
        logger.info("Writing SPH metrics visualization...")
        write_exr(vis_hdri, output_dir / f"{hdri_path.stem}_sph_metrics_visualization.exr")
        
        # Write reconstructed environment maps
        for l in range(args.l_max + 1):
            logger.info(f"Writing reconstructed environment map for band {l}...")
            write_exr(reconstructed_hdris[l, ...], output_dir / f"{hdri_path.stem}_reconstructed_{l}.exr")
        output_time = time.time() - start_time
        logger.info(f"Output writing complete in {output_time:.2f} seconds.")

        # Save SPH metrics - convert to dict for JSON serialization
        with open(output_dir / f"{hdri_path.stem}.json", "w") as f:
            json.dump(sph_metrics_cpu.to_dict(), f, indent=2)
        
        # Log timing summary
        total_time = projection_time + metrics_time + visualization_time + reconstruction_time + output_time
        logger.info(f"Timing Summary for {hdri_path.stem}:")
        logger.info(f"  Spherical harmonics projection: {projection_time:.2f}s")
        logger.info(f"  SPH metrics computation:        {metrics_time:.2f}s")
        logger.info(f"  Visualization creation:         {visualization_time:.2f}s")
        logger.info(f"  Reconstruction:                 {reconstruction_time:.2f}s")
        logger.info(f"  Output writing:                 {output_time:.2f}s")
        logger.info(f"  Total analysis time:            {total_time:.2f}s")
        logger.info("-" * 50)
