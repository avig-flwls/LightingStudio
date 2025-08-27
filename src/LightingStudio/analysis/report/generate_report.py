import argparse
import json
import logging
import time
from pathlib import Path

import torch
from coolname import generate_slug

from src.LightingStudio.analysis.utils.io import read_exr, write_exr, find_hdri_files, exr_to_png_tensor
from src.LightingStudio.analysis.core.median_cut import (
    median_cut_sampling,
    median_cut_sampling_to_cpu,
    visualize_samples,
)
from src.LightingStudio.analysis.core.density_estimation import (
    expand_map_exact,
    expand_map_fast,
)
from src.LightingStudio.analysis.core.intensity_calculation import (
    naive_metrics,
    naive_metrics_cpu,
)
from src.LightingStudio.analysis.core.sph import (
    get_sph_metrics,
    get_sph_metrics_cpu,
    project_env_to_coefficients,
    reconstruct_sph_coeffs_to_env,
    visualize_sph_metrics,
)
from src.LightingStudio.analysis.report.html_report import generate_html_report
from src.LightingStudio.analysis.report.aggregate_statistics import generate_aggregate_statistics_html

OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"

# Usage examples:
# 
# Process a single HDRI file:
# python -m src.LightingStudio.analysis.report.generate_report --hdri "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Bakery.exr" --n_samples 1024 --l_max 3
#
# Process multiple HDRI files (also generates aggregate statistics):
# python -m src.LightingStudio.analysis.report.generate_report --hdri "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Bakery.exr" "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Games Room 02.exr" "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k\Abandoned Factory Canteen 02.exr" --n_samples 1024 --l_max 3
#
# Process all HDRI files in a folder (also generates aggregate statistics):
# python -m src.LightingStudio.analysis.report.generate_report --folder "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k" --n_samples 1024 --l_max 3
#
# Generate aggregate statistics for existing experiment:
# python -m src.LightingStudio.analysis.report.aggregate_statistics "C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments\experiment-name"


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
    
    # Create mutually exclusive group for input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--hdri", type=str, nargs='+', help="List of HDRI file paths")
    input_group.add_argument("--folder", type=str, help="Folder containing HDRI files")
    
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples (must be > 1024)")
    parser.add_argument("--l_max", type=int, required=True, help="Maximum spherical harmonic band")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging")
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("Debug logging enabled")

    # if args.n_samples < 1024:
    #     parser.error("The value of --n_samples must be greater than 1024.")

    # Determine HDRI file paths based on input type
    if args.hdri:
        hdri_paths = args.hdri
        print(f"Processing {len(hdri_paths)} HDRI files from command line arguments")
    elif args.folder:
        try:
            hdri_paths = find_hdri_files(args.folder)
            print(f"Found {len(hdri_paths)} HDRI files in folder: {args.folder}")
        except ValueError as e:
            parser.error(str(e))
    else:
        parser.error("Either --hdri or --folder must be specified")

    # Validate that all files exist
    for hdri_path in hdri_paths:
        if not Path(hdri_path).exists():
            parser.error(f"HDRI file does not exist: {hdri_path}")

    experiment_name = generate_slug(2)
    output_dir = Path(OUTPUT_DIR) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set up logger for this script
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # ------------------------------------------------------------
    # Validate files first to build navigation list
    # ------------------------------------------------------------
    print("Validating HDRI files...")
    valid_hdri_paths = []
    hdri_names = []
    
    for hdri_path in hdri_paths:
        hdri_path = Path(hdri_path)
        try:
            # Quick validation - just try to read without loading to GPU
            test_hdri = read_exr(str(hdri_path))
            valid_hdri_paths.append(hdri_path)
            hdri_names.append(hdri_path.stem)
            logger.debug(f"Validated {hdri_path.name}")
        except ValueError as e:
            logger.warning(f"Skipping {hdri_path.name}: {e}")
            print(f"Warning: Skipping {hdri_path.name} - {e}")
            continue
    
    logger.info(f"Validation complete: {len(valid_hdri_paths)}/{len(hdri_paths)} files are valid")
    print(f"Validation complete: {len(valid_hdri_paths)}/{len(hdri_paths)} files are valid")
    
    if not valid_hdri_paths:
        logger.error("No valid HDRI files found!")
        print("Error: No valid HDRI files found!")
        exit(1)
    
    # ------------------------------------------------------------
    # Process valid files
    # ------------------------------------------------------------

    for hdri_path in valid_hdri_paths:
        print(f"Processing {hdri_path}...")
        
        # Read the HDRI file (we know it's valid from validation)
        hdri = read_exr(str(hdri_path)).to(device)
        logger.info(f"Processing {hdri_path.name} with shape {hdri.shape}")

        # Create subfolder for this HDRI
        hdri_output_dir = output_dir / hdri_path.stem
        hdri_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"HDRI output directory: {hdri_output_dir}")

        # ------------------------------------------------------------
        # Start Analysis
        # ------------------------------------------------------------

        # Start analysis
        print("Starting analysis...")
        
        # Time median cut sampling
        start_time = time.time()
        samples_cpu = median_cut_sampling_to_cpu(hdri, args.n_samples)
        sampling_time = time.time() - start_time
        logger.info(f"Median cut sampling complete in {sampling_time:.2f} seconds.")
        
        # Time exact density map
        start_time = time.time()
        # density_map_exact = expand_map_exact(hdri, samples_cpu, min_count=4, normalize=True)
        exact_time = time.time() - start_time
        logger.info(f"Exact density map complete in {exact_time:.2f} seconds.")
        
        # Time fast density map
        start_time = time.time()
        density_map_fast = expand_map_fast(hdri, samples_cpu, min_count=4, normalize=True)
        fast_time = time.time() - start_time
        logger.info(f"Fast density map complete in {fast_time:.2f} seconds.")

        # Time naive metrics
        start_time = time.time()
        naive_metrics_cpu(hdri)
        naive_time = time.time() - start_time
        logger.info(f"Naive metrics complete in {naive_time:.2f} seconds.")

        # Time spherical harmonic projection 
        start_time = time.time()
        env_map_sph_coeffs, sph_basis = project_env_to_coefficients(hdri, args.l_max)
        project_time = time.time() - start_time
        logger.info(f"Spherical harmonic projection complete in {project_time:.2f} seconds.")

        # Time spherical harmonic reconstruction
        start_time = time.time()
        env_map_reconstructed = reconstruct_sph_coeffs_to_env(hdri.shape[0], hdri.shape[1], env_map_sph_coeffs, sph_basis)
        reconstruct_time = time.time() - start_time
        logger.info(f"Spherical harmonic reconstruction complete in {reconstruct_time:.2f} seconds.")

        # Time spherical harmonic metrics
        start_time = time.time()
        sph_metrics_cpu = get_sph_metrics_cpu(hdri, args.l_max)
        sph_metrics_time = time.time() - start_time
        logger.info(f"Spherical harmonic metrics complete in {sph_metrics_time:.2f} seconds.")

        # ------------------------------------------------------------
        # Save images as PNG for web display
        # ------------------------------------------------------------
        
        # Create web directory
        web_dir = hdri_output_dir / "web"
        web_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original HDRI as PNG
        exr_to_png_tensor(hdri, web_dir / f"{hdri_path.stem}_original.png", gamma=2.2, exposure=0.0)

        # Save density map as PNG
        exr_to_png_tensor(density_map_fast, web_dir / f"{hdri_path.stem}_density_map_fast.png", gamma=2.2, exposure=0.0)

        # Save samples visualization as PNG
        vis_hdri = visualize_samples(hdri, samples_cpu)
        exr_to_png_tensor(vis_hdri, web_dir / f"{hdri_path.stem}_median_cut.png", gamma=2.2, exposure=0.0)

        # Save reconstructed environment maps as PNG
        for l in range(args.l_max + 1):
            exr_to_png_tensor(env_map_reconstructed[l, ...], web_dir / f"{hdri_path.stem}_reconstructed_{l}.png", gamma=2.2, exposure=0.0)

        # Save spherical harmonic metrics visualization as PNG
        vis_sph_metrics = visualize_sph_metrics(hdri, sph_metrics_cpu)
        exr_to_png_tensor(vis_sph_metrics, web_dir / f"{hdri_path.stem}_sph_metrics.png", gamma=2.2, exposure=0.0)
        
        # # Still save original HDRI as EXR for reference (optional)
        # write_exr(hdri, hdri_output_dir / f"{hdri_path.stem}_original.exr")

        # ------------------------------------------------------------
        # Save Metrics as JSON
        # ------------------------------------------------------------

        # Save samples 
        samples_dict = [sample.to_dict() for sample in samples_cpu]
        with open(hdri_output_dir / f"{hdri_path.stem}_samples.json", "w") as f:
            json.dump(samples_dict, f, indent=2)

        # Save naive metrics
        naive_metrics_dict = naive_metrics_cpu(hdri).to_dict()
        with open(hdri_output_dir / f"{hdri_path.stem}_naive_metrics.json", "w") as f:
            json.dump(naive_metrics_dict, f, indent=2)

        # Save spherical harmonic metrics
        sph_metrics_dict = sph_metrics_cpu.to_dict()
        with open(hdri_output_dir / f"{hdri_path.stem}_sph_metrics.json", "w") as f:
            json.dump(sph_metrics_dict, f, indent=2)

        # Log timing summary
        total_time = sampling_time + exact_time + fast_time + naive_time + project_time + reconstruct_time + sph_metrics_time
        logger.info(f"Timing Summary for {hdri_path.stem}:")
        logger.info(f"  Median cut sampling: {sampling_time:.2f}s")
        logger.info(f"  Exact density map:   {exact_time:.2f}s")
        logger.info(f"  Fast density map:    {fast_time:.2f}s")
        logger.info(f"  Naive metrics:       {naive_time:.2f}s")
        logger.info(f"  Spherical harmonic projection: {project_time:.2f}s")
        logger.info(f"  Spherical harmonic reconstruction: {reconstruct_time:.2f}s")
        logger.info(f"  Spherical harmonic metrics: {sph_metrics_time:.2f}s")
        logger.info(f"  Total analysis time: {total_time:.2f}s")
        logger.info("-" * 50)

        # ------------------------------------------------------------
        # Generate HTML Report
        # ------------------------------------------------------------
        
        html_path = generate_html_report(hdri_output_dir, hdri_path.stem, hdri_names)
        logger.info(f"HTML report generated: {html_path}")

    # ------------------------------------------------------------
    # Processing Summary
    # ------------------------------------------------------------
    logger.info(f"Processing complete: {len(valid_hdri_paths)}/{len(hdri_paths)} files processed successfully")
    print(f"Processing complete: {len(valid_hdri_paths)}/{len(hdri_paths)} files processed successfully")

    # ------------------------------------------------------------
    # Generate Aggregate Statistics (after all individual reports)
    # ------------------------------------------------------------
    
    if len(hdri_names) > 1:  # Only generate if we have multiple HDRIs
        logger.info("Generating aggregate statistics...")
        print("Generating aggregate statistics...")
        
        # Quick verification that JSON files exist
        json_files_found = 0
        for hdri_name in hdri_names:
            hdri_subdir = output_dir / hdri_name
            naive_path = hdri_subdir / f"{hdri_name}_naive_metrics.json"
            sph_path = hdri_subdir / f"{hdri_name}_sph_metrics.json"
            if naive_path.exists() and sph_path.exists():
                json_files_found += 1
                
        logger.info(f"Found JSON files for {json_files_found}/{len(hdri_names)} HDRIs")
        print(f"Found JSON files for {json_files_found}/{len(hdri_names)} HDRIs")
        
        aggregate_html_path = generate_aggregate_statistics_html(output_dir)
        logger.info(f"Aggregate statistics generated: {aggregate_html_path}")
        print(f"Aggregate statistics webpage created: {aggregate_html_path}")

 
