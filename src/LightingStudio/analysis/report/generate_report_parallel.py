import argparse
import json
import logging
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import torch
from coolname import generate_slug

import shutil
import subprocess

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
from src.LightingStudio.analysis.blender_interface.blender_renderer import process_hdri_with_blender

OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"

# Usage examples:
# 
# Process a single HDRI file (saves both EXR and PNG):
# python -m src.LightingStudio.analysis.report.generate_report_parallel --hdri "path/to/file.exr" --n_samples 1024 --l_max 3
#
# Process a single HDRI file (PNG only for faster processing):
# python -m src.LightingStudio.analysis.report.generate_report_parallel --hdri "path/to/file.exr" --n_samples 1024 --l_max 3 --png-only
#
# Process all HDRI files in a folder with parallel processing (auto-detects CPU cores):
# python -m src.LightingStudio.analysis.report.generate_report_parallel --folder "path/to/hdris" --n_samples 1024 --l_max 3 --png-only
# python -m src.LightingStudio.analysis.report.generate_report_parallel --folder "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k_small" --n_samples 1024 --l_max 3 --png-only

# python -m src.LightingStudio.analysis.report.generate_report_parallel --folder "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k_small_med" --n_samples 2 --l_max 3 --png-only --process 4

# Process with specific number of parallel processes:
# python -m src.LightingStudio.analysis.report.generate_report_parallel --folder "path/to/hdris" --n_samples 1024 --l_max 3 --png-only --processes 4
#
# Process sequentially (single-threaded):
# python -m src.LightingStudio.analysis.report.generate_report_parallel --folder "path/to/hdris" --n_samples 1024 --l_max 3 --png-only --processes 1

# Only run analysis on the following HDRIs:
# python -m src.LightingStudio.analysis.report.aggregate_statistics  "C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments\dainty-flounder"

def process_single_hdri(hdri_path, output_dir, hdri_names, n_samples, l_max, png_only, device_str="cuda"):
    """
    Process a single HDRI file. This function is designed to be called in parallel.
    
    Args:
        hdri_path: Path to the HDRI file
        output_dir: Base experiment output directory  
        hdri_names: List of all HDRI names for navigation
        n_samples: Number of samples for median cut
        l_max: Maximum spherical harmonic band
        png_only: Whether to save PNG only or both EXR and PNG
        device_str: Device string ("cuda" or "cpu")
    
    Returns:
        Dict with processing results and timing info
    """
    try:
        hdri_path = Path(hdri_path)
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        
        print(f"Processing {hdri_path.name} on {device}")
        
        # Read the HDRI file
        hdri = read_exr(str(hdri_path)).to(device)
        
        # Create subfolder for this HDRI
        hdri_output_dir = output_dir / hdri_path.stem
        hdri_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track timing
        timing = {}
        
        # Start analysis
        start_time = time.time()
        samples_cpu = median_cut_sampling_to_cpu(hdri, n_samples)
        timing['sampling'] = time.time() - start_time
        
        # # Time exact density map
        # start_time = time.time()
        # density_map_exact = expand_map_exact(hdri, samples_cpu, min_count=4, normalize=True)
        # timing['exact_density'] = time.time() - start_time
        
        # Time fast density map
        start_time = time.time()
        density_map_fast = expand_map_fast(hdri, samples_cpu, min_count=4, normalize=True)
        timing['fast_density'] = time.time() - start_time

        # Time naive metrics
        start_time = time.time()
        naive_metrics_cpu(hdri)
        timing['naive'] = time.time() - start_time

        # Time spherical harmonic projection 
        start_time = time.time()
        env_map_sph_coeffs, sph_basis = project_env_to_coefficients(hdri, l_max)
        timing['sph_project'] = time.time() - start_time

        # Time spherical harmonic reconstruction
        start_time = time.time()
        env_map_reconstructed = reconstruct_sph_coeffs_to_env(hdri.shape[0], hdri.shape[1], env_map_sph_coeffs, sph_basis)
        timing['sph_reconstruct'] = time.time() - start_time

        # Time spherical harmonic metrics
        start_time = time.time()
        sph_metrics_cpu = get_sph_metrics_cpu(hdri, l_max)
        timing['sph_metrics'] = time.time() - start_time

        # Save images based on png_only flag
        web_dir = hdri_output_dir / "web"
        web_dir.mkdir(parents=True, exist_ok=True)
        
        if png_only:
            # Save original HDRI as PNG only
            exr_to_png_tensor(hdri, web_dir / f"{hdri_path.stem}_original.png", gamma=2.2, exposure=0.0)
            exr_to_png_tensor(density_map_fast, web_dir / f"{hdri_path.stem}_density_map_fast.png", gamma=2.2, exposure=0.0)
            vis_hdri = visualize_samples(hdri, samples_cpu)
            exr_to_png_tensor(vis_hdri, web_dir / f"{hdri_path.stem}_median_cut.png", gamma=2.2, exposure=0.0)
            for l in range(l_max + 1):
                exr_to_png_tensor(env_map_reconstructed[l, ...], web_dir / f"{hdri_path.stem}_reconstructed_{l}.png", gamma=2.2, exposure=0.0)
            vis_sph_metrics = visualize_sph_metrics(hdri, sph_metrics_cpu)
            exr_to_png_tensor(vis_sph_metrics, web_dir / f"{hdri_path.stem}_sph_metrics.png", gamma=2.2, exposure=0.0)
        else:
            # Save both EXR and PNG
            write_exr(hdri, hdri_output_dir / f"{hdri_path.stem}_original.exr")
            exr_to_png_tensor(hdri, web_dir / f"{hdri_path.stem}_original.png", gamma=2.2, exposure=0.0)
            write_exr(density_map_fast, hdri_output_dir / f"{hdri_path.stem}_density_map_fast.exr")
            exr_to_png_tensor(density_map_fast, web_dir / f"{hdri_path.stem}_density_map_fast.png", gamma=2.2, exposure=0.0)
            vis_hdri = visualize_samples(hdri, samples_cpu)
            write_exr(vis_hdri, hdri_output_dir / f"{hdri_path.stem}_median_cut.exr")
            exr_to_png_tensor(vis_hdri, web_dir / f"{hdri_path.stem}_median_cut.png", gamma=2.2, exposure=0.0)
            for l in range(l_max + 1):
                write_exr(env_map_reconstructed[l, ...], hdri_output_dir / f"{hdri_path.stem}_reconstructed_{l}.exr")
                exr_to_png_tensor(env_map_reconstructed[l, ...], web_dir / f"{hdri_path.stem}_reconstructed_{l}.png", gamma=2.2, exposure=0.0)
            vis_sph_metrics = visualize_sph_metrics(hdri, sph_metrics_cpu)
            write_exr(vis_sph_metrics, hdri_output_dir / f"{hdri_path.stem}_sph_metrics.exr")
            exr_to_png_tensor(vis_sph_metrics, web_dir / f"{hdri_path.stem}_sph_metrics.png", gamma=2.2, exposure=0.0)

        # Save Metrics as JSON
        samples_dict = [sample.to_dict() for sample in samples_cpu]
        with open(hdri_output_dir / f"{hdri_path.stem}_samples.json", "w") as f:
            json.dump(samples_dict, f, indent=2)

        naive_metrics_dict = naive_metrics_cpu(hdri).to_dict()
        with open(hdri_output_dir / f"{hdri_path.stem}_naive_metrics.json", "w") as f:
            json.dump(naive_metrics_dict, f, indent=2)

        sph_metrics_dict = sph_metrics_cpu.to_dict()
        with open(hdri_output_dir / f"{hdri_path.stem}_sph_metrics.json", "w") as f:
            json.dump(sph_metrics_dict, f, indent=2)

        # Blender renderings - abstracted into dedicated module
        start_time = time.time()
        blender_success = process_hdri_with_blender(hdri_path, hdri_output_dir)
        timing['blender_processing'] = time.time() - start_time
        
        if blender_success:
            print("Blender processing completed successfully")
        else:
            print("Blender processing encountered errors")

        # Generate HTML Report
        html_path = generate_html_report(hdri_output_dir, hdri_path.stem, hdri_names)
        
        # Calculate total time
        timing['total'] = sum(timing.values())
        
        print(f"Completed {hdri_path.name} in {timing['total']:.2f}s")
        
        return {
            'success': True,
            'hdri_name': hdri_path.stem,
            'html_path': html_path,
            'timing': timing
        }
        
    except Exception as e:
        print(f"Error processing {hdri_path}: {e}")
        return {
            'success': False,
            'hdri_name': hdri_path.stem if 'hdri_path' in locals() else str(hdri_path),
            'error': str(e)
        }


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
    parser.add_argument("--png-only", action="store_true", help="Save images as PNG only (default: save both EXR and PNG)")
    parser.add_argument("--processes", "-p", type=int, default=None, help=f"Number of parallel processes (default: {cpu_count()}, use 1 for sequential)")
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("Debug logging enabled")

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
    # Determine processing mode and setup
    # ------------------------------------------------------------
    
    # Set number of processes
    if args.processes is None:
        num_processes = cpu_count()
    else:
        num_processes = args.processes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    
    print(f"Processing {len(valid_hdri_paths)} HDRIs using {num_processes} process(es) on {device}")
    logger.info(f"Using {num_processes} process(es) on {device}")
    
    # ------------------------------------------------------------
    # Process files (parallel or sequential)
    # ------------------------------------------------------------
    
    start_total_time = time.time()
    
    if num_processes == 1:
        # Sequential processing
        print("Running sequential processing...")
        results = []
        for hdri_path in valid_hdri_paths:
            result = process_single_hdri(
                hdri_path, output_dir, hdri_names, 
                args.n_samples, args.l_max, args.png_only, device_str
            )
            results.append(result)
    else:
        # Parallel processing
        print(f"Running parallel processing with {num_processes} processes...")
        
        # Create partial function with fixed arguments
        process_func = partial(
            process_single_hdri,
            output_dir=output_dir,
            hdri_names=hdri_names,
            n_samples=args.n_samples,
            l_max=args.l_max,
            png_only=args.png_only,
            device_str="cpu"  # Force CPU for multiprocessing to avoid CUDA issues
        )
        
        # Use multiprocessing
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_func, valid_hdri_paths)
    
    total_time = time.time() - start_total_time
    
    # ------------------------------------------------------------
    # Process results and generate summary
    # ------------------------------------------------------------
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\nProcessing Summary:")
    print(f"  Total files: {len(valid_hdri_paths)}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Failed: {len(failed_results)}")
    print(f"  Total time: {total_time:.2f}s")
    
    if successful_results:
        avg_time = sum(r['timing']['total'] for r in successful_results) / len(successful_results)
        print(f"  Average time per HDRI: {avg_time:.2f}s")
        
        if num_processes > 1:
            sequential_estimate = avg_time * len(successful_results)
            speedup = sequential_estimate / total_time
            print(f"  Estimated speedup: {speedup:.1f}x")
    
    if failed_results:
        print(f"\nFailed files:")
        for result in failed_results:
            print(f"  - {result['hdri_name']}: {result['error']}")
    
    logger.info(f"Processing complete: {len(successful_results)}/{len(valid_hdri_paths)} files processed successfully")
    
    # Update hdri_names to only include successful ones for aggregate stats
    successful_hdri_names = [r['hdri_name'] for r in successful_results]

    # ------------------------------------------------------------
    # Generate Aggregate Statistics (after all individual reports)
    # ------------------------------------------------------------
    
    if len(successful_hdri_names) > 1:  # Only generate if we have multiple HDRIs
        logger.info("Generating aggregate statistics...")
        print("Generating aggregate statistics...")
        
        aggregate_html_path = generate_aggregate_statistics_html(output_dir)
        logger.info(f"Aggregate statistics generated: {aggregate_html_path}")
        print(f"Aggregate statistics webpage created: {aggregate_html_path}")
    elif len(successful_hdri_names) == 1:
        print("Only one HDRI processed successfully - skipping aggregate statistics")
    else:
        print("No HDRIs processed successfully - cannot generate aggregate statistics")
