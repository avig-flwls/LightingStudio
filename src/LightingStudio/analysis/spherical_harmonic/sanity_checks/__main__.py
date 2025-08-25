"""
Main entry point for spherical harmonic sanity checks package.

This script provides a unified interface for running both unit tests and visualizations.
"""

import argparse
import sys
from pathlib import Path
from coolname import generate_slug

from .unit_tests import run_all_sanity_checks
from .visualization import visualize_multiple_harmonics, visualize_spherical_harmonic_basis

OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"


def main():
    """
    Main function providing unified interface for unit tests and visualizations.
    """
    parser = argparse.ArgumentParser(description='Run spherical harmonic sanity checks and visualizations')
    
    # Test parameters
    parser.add_argument('--height', '-H', type=int, default=64, help='Height of test coordinates (default: 64)')
    parser.add_argument('--width', '-W', type=int, default=128, help='Width of test coordinates (default: 128)')
    parser.add_argument('--l-max', type=int, default=4, help='Maximum spherical harmonic band (default: 4)')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', help='Generate basis function visualizations')
    parser.add_argument('--vis-resolution', type=int, default=64, help='Resolution for visualizations (default: 64)')
    parser.add_argument('--single-vis', type=str, nargs=2, metavar=('L', 'M'), 
                        help='Visualize single spherical harmonic Y_l^m (provide l and m as integers)')
    parser.add_argument('--multiple-vis', type=str, nargs='+', metavar='L,M', 
                        help='Visualize multiple spherical harmonics (e.g., --multiple-vis 1,0 2,1 2,-1)')
    
    # Mode selection
    parser.add_argument('--tests-only', action='store_true', 
                        help='Run only unit tests (skip visualizations)')
    parser.add_argument('--vis-only', action='store_true', 
                        help='Run only visualizations (skip unit tests)')
    
    args = parser.parse_args()
    
    # Create experiment-specific output directory if visualizations are requested
    if args.visualize or args.single_vis or args.multiple_vis or not args.tests_only:
        experiment_name = generate_slug(2)
        output_dir = Path(OUTPUT_DIR) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = None
    
    print("Spherical Harmonic Sanity Checks")
    print("=" * 40)
    print(f"Parameters: H={args.height}, W={args.width}, l_max={args.l_max}")
    if output_dir:
        print(f"Visualization resolution: {args.vis_resolution}")
    print()
    
    all_passed = True
    
    # Run unit tests unless disabled
    if not args.vis_only:
        print("RUNNING UNIT TESTS")
        print("=" * 20)
        results = run_all_sanity_checks(args.height, args.width, args.l_max)
        
        # Check if all tests passed
        tests_passed = (
            results['coordinate_conversion_match'] and
            results['cartesian_vectorized_match'] and 
            results['spherical_vectorized_match'] and 
            results['basis_accuracy_match']
        )
        
        if tests_passed:
            print("🎉 ALL UNIT TESTS PASSED!")
        else:
            print("❌ SOME UNIT TESTS FAILED!")
            all_passed = False
        
        print()
    
    # Run visualizations unless disabled
    if not args.tests_only and output_dir:
        print("RUNNING VISUALIZATIONS")
        print("=" * 20)
        
        # Generate comparison visualizations if requested
        if args.visualize:
            print("Generating basis function comparison visualizations...")
            from .visualization import visualize_basis_comparison
            visualize_basis_comparison(
                l_max=min(args.l_max, 2),  # Limit for reasonable visualization
                resolution=args.vis_resolution,
                save_dir=str(output_dir)
            )
            print("Comparison visualization complete!")
            print()
        
        # Generate single visualization if requested
        if args.single_vis:
            l, m = int(args.single_vis[0]), int(args.single_vis[1])
            print(f"Generating visualization for Y_{l}^{m}...")
            
            save_path = str(output_dir / f"Y_{l}_{m}")
            
            plotter = visualize_spherical_harmonic_basis(
                l=l, m=m, 
                resolution=args.vis_resolution,
                save_path=save_path
            )
            print()
        
        # Generate multiple visualizations if requested
        if args.multiple_vis:
            harmonics_list = []
            for lm_str in args.multiple_vis:
                try:
                    l, m = map(int, lm_str.split(','))
                    harmonics_list.append((l, m))
                except ValueError:
                    print(f"Invalid format for harmonic: {lm_str}. Use format 'l,m' (e.g., '2,1')")
                    continue
            
            if harmonics_list:
                visualize_multiple_harmonics(
                    harmonics_list=harmonics_list,
                    resolution=args.vis_resolution,
                    save_dir=str(output_dir)
                )
                print("Multiple visualizations complete!")
                print()
    
    # Final summary
    print("=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS COMPLETED SUCCESSFULLY!")
    else:
        print("❌ SOME CHECKS FAILED!")
    
    if output_dir and (args.visualize or args.single_vis or args.multiple_vis or not args.tests_only):
        print(f"\nFiles saved to: {output_dir}")
        print("\nTo view HTML files:")
        print("  - Double-click .html files in file explorer")
        print("  - Drag .html files to your web browser")
        print("  - Open file://path/to/file.html in browser")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())


"""
Usage Examples:

# Run all unit tests (no visualizations)
python -m src.LightingStudio.analysis.spherical_harmonic.sanity_checks --tests-only
python -m src.LightingStudio.analysis.spherical_harmonic.sanity_checks --tests-only -H 1024 -W 2048 --l-max 4  

# Visualize a single spherical harmonic with lobes (HTML output)
python -m src.LightingStudio.analysis.spherical_harmonic.sanity_checks --single-vis 2 1

# Visualize multiple spherical harmonics
python -m src.LightingStudio.analysis.spherical_harmonic.sanity_checks --multiple-vis 0,0 1,0 1,1 2,0

# Run tests with high-resolution visualization
python -m src.LightingStudio.analysis.spherical_harmonic.sanity_checks -H 256 -W 512 --l-max 6 --single-vis 3 2 --vis-resolution 128

# Just visualizations (skip tests)
python -m src.LightingStudio.analysis.spherical_harmonic.sanity_checks --vis-only --multiple-vis 1,0 2,1 2,-1
"""
