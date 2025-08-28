#!/usr/bin/env python3
"""
Sky Environment Generation Script

This script generates synthetic sky environments using the Hosek-Wilkie analytic sky model.
It can generate both CIE XYZ and linear sRGB sky environments and save them as EXR files.

Usage examples:

# Generate a basic sky environment (auto-named folder):
python -m src.LightingStudio.ingest.run_skybox_light --width 2048 --height 1024 --sun-elevation 15 --sun-azimuth 0 --turbidity 3.0 --albedo 0.1

# Generate a brighter sky (default brightness is 0.1):
python -m src.LightingStudio.ingest.run_skybox_light --width 2048 --height 1024 --sun-elevation 15 --sun-azimuth 0 --turbidity 3.0 --brightness 0.3

# Generate a darker sky:
python -m src.LightingStudio.ingest.run_skybox_light --width 2048 --height 1024 --sun-elevation 15 --sun-azimuth 0 --turbidity 3.0 --brightness 0.05

# Generate a single sky with custom output path:
python -m src.LightingStudio.ingest.run_skybox_light --width 2048 --height 1024 --sun-elevation 45 --sun-azimuth 180 --turbidity 5.0 --albedo 0.3 --brightness 0.2 --add-sun --output sky_with_sun.exr --png-preview sky_with_sun.png

# Generate in linear sRGB space with custom folder name:
python -m src.LightingStudio.ingest.run_skybox_light --width 1024 --height 512 --sun-elevation 30 --sun-azimuth 90 --space RGB --brightness 0.15 --name my_sky_collection

# Generate multiple skies with different parameters (automatically creates unique output folder):
python -m src.LightingStudio.ingest.run_skybox_light --width 2048 --height 1024 --sun-elevation 15 30 45 --sun-azimuth 0 90 180 --turbidity 2.0 3.0 5.0 --brightness 0.08

"""

import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from coolname import generate_slug

from src.LightingStudio.ingest.skybox_light_utils import (
    generate_and_save_sky,
    generate_sky_latlong,
    xyz_to_linear_srgb,
    save_sky_environment,
)

OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\skybox_lights"



def validate_sun_angles(elevation: float, azimuth: float) -> tuple[float, float]:
    """Validate and convert sun angles from degrees to radians."""
    if not -90 <= elevation <= 90:
        raise ValueError(f"Sun elevation must be between -90° and 90°, got {elevation}°")

    # Normalize azimuth to 0-360 range
    azimuth = azimuth % 360

    return math.radians(elevation), math.radians(azimuth)


def generate_multiple_skies(
    elevations: list[float],
    azimuths: list[float],
    turbidities: list[float],
    width: int,
    height: int,
    output_dir: Path,
    albedo: float = 0.1,
    space: str = "XYZ",
    add_sun: bool = False,
    sun_intensity: float = 50.0,
    sun_radius_deg: float = 0.27,
    device: Optional[str] = None,
    brightness_scale: float = 0.1,
) -> list[Path]:
    """
    Generate multiple sky environments with different parameters.

    Args:
        elevations: List of sun elevations in degrees
        azimuths: List of sun azimuths in degrees
        turbidities: List of turbidity values
        width: Output width
        height: Output height
        output_dir: Output directory
        albedo: Ground albedo (0-1)
        space: Color space ("XYZ" or "RGB")
        add_sun: Whether to add sun disc
        sun_intensity: Sun disc intensity
        sun_radius_deg: Sun disc radius in degrees
        device: Torch device
        brightness_scale: Brightness scale factor (0.01-1.0)

    Returns:
        List of generated file paths
    """
    generated_files = []

    # Generate combinations of parameters
    combinations = []
    for elev in elevations:
        for azim in azimuths:
            for turb in turbidities:
                combinations.append((elev, azim, turb))

    print(f"Generating {len(combinations)} sky environments...")

    for i, (elevation, azimuth, turbidity) in enumerate(combinations):
        # Validate angles
        sun_elevation_rad, sun_azimuth_rad = validate_sun_angles(elevation, azimuth)

        # Create descriptive filename
        filename = f"sky_e{elevation:03.0f}_a{azimuth:03.0f}_t{turbidity:.1f}_{width}x{height}_{space.lower()}"
        exr_path = output_dir / f"{filename}.exr"
        png_path = output_dir / f"{filename}.png"

        print(f"[{i+1}/{len(combinations)}] Generating: {filename}")

        try:
            # Generate sky
            sky = generate_sky_latlong(
                width=width,
                height=height,
                sun_azimuth=sun_azimuth_rad,
                sun_elevation=sun_elevation_rad,
                turbidity=turbidity,
                albedo=albedo,
                space=space,
                device=device,
                brightness_scale=brightness_scale,
            )

            # Add sun disc if requested
            if add_sun:
                from src.LightingStudio.ingest.skybox_light_utils import add_sun_disc
                sky = add_sun_disc(
                    sky,
                    sun_azimuth=sun_azimuth_rad,
                    sun_elevation=sun_elevation_rad,
                    intensity=sun_intensity,
                    radius_deg=sun_radius_deg,
                )

            # Save files
            save_sky_environment(sky, str(exr_path), str(png_path))
            generated_files.extend([exr_path, png_path])

        except Exception as e:
            logging.error(f"Failed to generate {filename}: {e}")
            continue

    return generated_files


def main():
    """Main function for command line interface."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to console
        ]
    )

    parser = argparse.ArgumentParser(
        description="Generate synthetic sky environments using Hosek-Wilkie analytic sky model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input parameters
    parser.add_argument("--width", type=int, default=2048, help="Output image width (default: 2048)")
    parser.add_argument("--height", type=int, default=1024, help="Output image height (default: 1024)")

    # Sun parameters
    parser.add_argument("--sun-elevation", type=float, nargs='+', default=[15.0],
                       help="Sun elevation angle(s) in degrees (default: 15.0)")
    parser.add_argument("--sun-azimuth", type=float, nargs='+', default=[0.0],
                       help="Sun azimuth angle(s) in degrees (default: 0.0)")

    # Sky parameters
    parser.add_argument("--turbidity", type=float, nargs='+', default=[3.0],
                       help="Sky turbidity value(s), 1-10 (default: 3.0)")
    parser.add_argument("--albedo", type=float, default=0.1,
                       help="Ground albedo, 0-1 (default: 0.1)")
    parser.add_argument("--space", choices=["XYZ", "RGB"], default="XYZ",
                       help="Color space for output (default: XYZ)")
    parser.add_argument("--brightness", type=float, default=0.1,
                       help="Brightness scale factor, 0.01-1.0 (default: 0.1, lower = darker)")

    # Sun disc options
    parser.add_argument("--add-sun", action="store_true",
                       help="Add a sun disc to the sky")
    parser.add_argument("--sun-intensity", type=float, default=50.0,
                       help="Sun disc intensity (default: 50.0)")
    parser.add_argument("--sun-radius-deg", type=float, default=0.27,
                       help="Sun disc radius in degrees (default: 0.27)")

    # Output options
    parser.add_argument("--output", type=str,
                       help="Output EXR file path (single sky, optional - will use auto-generated folder if not specified)")
    parser.add_argument("--name", type=str,
                       help="Custom name for the output folder (default: auto-generated)")

    parser.add_argument("--png-preview", type=str,
                       help="Optional PNG preview file path")
    parser.add_argument("--device", type=str,
                       help="Torch device (default: auto-detect)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose (DEBUG) logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("Debug logging enabled")

    # Validate parameters
    if any(t < 1.0 or t > 10.0 for t in args.turbidity):
        parser.error("Turbidity must be between 1.0 and 10.0")

    if not 0.0 <= args.albedo <= 1.0:
        parser.error("Albedo must be between 0.0 and 1.0")

    if not 0.01 <= args.brightness <= 1.0:
        parser.error("Brightness must be between 0.01 and 1.0")

    if args.width <= 0 or args.height <= 0:
        parser.error("Width and height must be positive integers")

    # Set up device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("Sky Environment Generator")
    print("========================")
    print(f"Device: {device}")
    print(f"Output size: {args.width}x{args.height}")
    print(f"Color space: {args.space}")
    print(f"Turbidity range: {min(args.turbidity)} - {max(args.turbidity)}")
    print(f"Sun elevation range: {min(args.sun_elevation)}° - {max(args.sun_elevation)}°")
    print(f"Sun azimuth range: {min(args.sun_azimuth)}° - {max(args.sun_azimuth)}°")
    print(f"Ground albedo: {args.albedo}")
    print(f"Brightness scale: {args.brightness}")
    if args.add_sun:
        print(f"Sun disc: enabled (intensity: {args.sun_intensity}, radius: {args.sun_radius_deg}°)")
    print()

    # Determine if we're generating single or multiple skies
    num_combinations = (len(args.sun_elevation) * len(args.sun_azimuth) *
                       len(args.turbidity))

    try:
        # Ensure base output directory exists
        base_output_dir = Path(OUTPUT_DIR)
        base_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique folder name
        if args.name:
            folder_name = args.name
        else:
            folder_name = generate_slug(2)

        output_dir = base_output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {output_dir}")

        if num_combinations == 1 and args.output:
            # Single sky generation with custom output path
            sun_elevation_rad, sun_azimuth_rad = validate_sun_angles(
                args.sun_elevation[0], args.sun_azimuth[0]
            )

            print("Generating single sky environment...")

            # Generate sky
            sky = generate_sky_latlong(
                width=args.width,
                height=args.height,
                sun_azimuth=sun_azimuth_rad,
                sun_elevation=sun_elevation_rad,
                turbidity=args.turbidity[0],
                albedo=args.albedo,
                space=args.space,
                device=device,
                brightness_scale=args.brightness,
            )

            # Add sun disc if requested
            if args.add_sun:
                from src.LightingStudio.ingest.skybox_light_utils import add_sun_disc
                sky = add_sun_disc(
                    sky,
                    sun_azimuth=sun_azimuth_rad,
                    sun_elevation=sun_elevation_rad,
                    intensity=args.sun_intensity,
                    radius_deg=args.sun_radius_deg,
                )

            # Save files
            png_path = args.png_preview if args.png_preview else None
            save_sky_environment(sky, args.output, png_path)

            print("✓ Sky environment generated successfully!")
            print(f"  EXR: {args.output}")
            if png_path:
                print(f"  PNG: {png_path}")

        else:
            # Multiple sky generation or single with auto-naming
            generated_files = generate_multiple_skies(
                elevations=args.sun_elevation,
                azimuths=args.sun_azimuth,
                turbidities=args.turbidity,
                width=args.width,
                height=args.height,
                output_dir=output_dir,
                albedo=args.albedo,
                space=args.space,
                add_sun=args.add_sun,
                sun_intensity=args.sun_intensity,
                sun_radius_deg=args.sun_radius_deg,
                device=device,
                brightness_scale=args.brightness,
            )

            print("\n✓ Generated sky environments!")
            print(f"  Total files: {len(generated_files)}")
            print(f"  Output directory: {output_dir}")
            print(f"  EXR files: {len([f for f in generated_files if f.suffix == '.exr'])}")
            print(f"  PNG files: {len([f for f in generated_files if f.suffix == '.png'])}")

    except RuntimeError as e:
        # Handle coefficient loading errors with more helpful message
        if "Hosek-Wilkie" in str(e) and "coefficients" in str(e):
            print("\n" + "="*60)
            print("COEFFICIENT LOADING ERROR")
            print("="*60)
            print(str(e))
            print("\nThe coefficient files should be in the archive folder.")
            print("Please ensure these files exist:")
            print("- archive/ArHosekSkyModelData_CIEXYZ.h")
            print("- archive/ArHosekSkyModelData_RGB.h")
            print("="*60)
        else:
            logging.error(f"Failed to generate sky environment: {e}")
            print(f"Error: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"Failed to generate sky environment: {e}")
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()