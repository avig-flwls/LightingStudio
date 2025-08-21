"""
Sanity checks and verification tests for spherical harmonic functions.

This module contains tests to verify:
1. Cartesian vs spherical basis function equivalence (accuracy and speed)
2. Vectorized vs non-vectorized implementations (accuracy and speed)
3. Visualization of basis functions using PyVista
"""

import torch
import numpy as np
import time
import pyvista as pv
from typing import Tuple, Optional

from .sph import (
    cartesian_to_sph_basis, cartesian_to_sph_basis_vectorized,
    spherical_to_sph_basis, spherical_to_sph_basis_vectorized,
    project_env_to_coefficients, sph_indices_total
)
from ..utils import generate_spherical_coordinates_map, spherical_to_cartesian


def test_vectorized_vs_original(cartesian_coordinates: torch.Tensor, l_max: int = 4) -> bool:
    """
    Test function to verify the vectorized implementation produces identical results to the original.
    
    :params cartesian_coordinates: Test coordinates on unit sphere
    :params l_max: Maximum spherical harmonic band to test
    :returns: True if results match within tolerance
    """
    # Compute with both methods
    original_result = cartesian_to_sph_basis(cartesian_coordinates, l_max)
    vectorized_result = cartesian_to_sph_basis_vectorized(cartesian_coordinates, l_max)
    
    # Check if results are close
    is_close = torch.allclose(original_result, vectorized_result, rtol=1e-6, atol=1e-8)
    
    if not is_close:
        max_diff = torch.max(torch.abs(original_result - vectorized_result))
        print(f"Results differ! Max difference: {max_diff.item():.2e}")
        return False
    
    print(f"✓ Vectorized implementation matches original for l_max={l_max}")
    return True


def test_spherical_vectorized_vs_original(spherical_coordinates: torch.Tensor, l_max: int = 6) -> bool:
    """
    Test function to verify the spherical vectorized implementation produces identical results to the original.
    
    :param spherical_coordinates: Test coordinates (theta, phi)
    :param l_max: Maximum spherical harmonic band to test
    :return: True if results match within tolerance
    """
    # Compute with both methods
    original_result = spherical_to_sph_basis(spherical_coordinates, l_max)
    vectorized_result = spherical_to_sph_basis_vectorized(spherical_coordinates, l_max)
    
    # Check if results are close
    is_close = torch.allclose(original_result, vectorized_result, rtol=1e-5, atol=1e-7)
    
    if not is_close:
        max_diff = torch.max(torch.abs(original_result - vectorized_result))
        print(f"Results differ! Max difference: {max_diff.item():.2e}")
        
        # Find which terms differ most
        diff = torch.abs(original_result - vectorized_result)
        max_idx = torch.argmax(diff.flatten())
        flat_shape = diff.numel() // diff.shape[-1]
        spatial_idx = max_idx // diff.shape[-1]
        term_idx = max_idx % diff.shape[-1]
        
        print(f"Max difference at spatial index {spatial_idx}, term {term_idx}")
        print(f"Original: {original_result.flatten()[max_idx]:.6f}")
        print(f"Vectorized: {vectorized_result.flatten()[max_idx]:.6f}")
        return False
    
    print(f"✓ Spherical vectorized implementation matches original for l_max={l_max}")
    return True


def test_cartesian_vs_spherical_basis(H: int = 64, W: int = 128, l_max: int = 4, device: torch.device = None) -> Tuple[bool, dict]:
    """
    Test to compare cartesian basis vs spherical basis functions for accuracy and speed.
    
    :param H: Height of test image
    :param W: Width of test image  
    :param l_max: Maximum spherical harmonic band to test
    :param device: Device to run tests on
    :return: Tuple of (accuracy_match, timing_results)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing cartesian vs spherical basis on {device} with H={H}, W={W}, l_max={l_max}")
    
    # Generate test coordinates
    spherical_coordinates = generate_spherical_coordinates_map(H, W, device)
    cartesian_coordinates = spherical_to_cartesian(spherical_coordinates)
    
    timing_results = {}
    
    # Test cartesian basis (original and vectorized)
    start_time = time.time()
    cartesian_original = cartesian_to_sph_basis(cartesian_coordinates, l_max)
    timing_results['cartesian_original'] = time.time() - start_time
    
    start_time = time.time()
    cartesian_vectorized = cartesian_to_sph_basis_vectorized(cartesian_coordinates, l_max)
    timing_results['cartesian_vectorized'] = time.time() - start_time
    
    # Test spherical basis (original and vectorized) 
    start_time = time.time()
    spherical_original = spherical_to_sph_basis(spherical_coordinates, l_max)
    timing_results['spherical_original'] = time.time() - start_time
    
    start_time = time.time()
    spherical_vectorized = spherical_to_sph_basis_vectorized(spherical_coordinates, l_max)
    timing_results['spherical_vectorized'] = time.time() - start_time
    
    # Check accuracy between cartesian and spherical (using vectorized versions for speed)
    accuracy_match = torch.allclose(cartesian_vectorized, spherical_vectorized, rtol=1e-4, atol=1e-6)
    
    if not accuracy_match:
        max_diff = torch.max(torch.abs(cartesian_vectorized - spherical_vectorized))
        print(f"❌ Cartesian vs Spherical basis differ! Max difference: {max_diff.item():.2e}")
    else:
        print(f"✓ Cartesian and Spherical basis match within tolerance")
    
    # Print timing results
    print(f"Timing results:")
    for method, time_taken in timing_results.items():
        print(f"  {method}: {time_taken:.4f}s")
    
    # Calculate speedups
    if timing_results['cartesian_original'] > 0:
        cartesian_speedup = timing_results['cartesian_original'] / timing_results['cartesian_vectorized']
        print(f"  Cartesian vectorized speedup: {cartesian_speedup:.2f}x")
    
    if timing_results['spherical_original'] > 0:
        spherical_speedup = timing_results['spherical_original'] / timing_results['spherical_vectorized']
        print(f"  Spherical vectorized speedup: {spherical_speedup:.2f}x")
    
    return accuracy_match, timing_results


def visualize_spherical_harmonic_basis(l: int, m: int, resolution: int = 64, save_path: Optional[str] = None) -> pv.Plotter:
    """
    Visualize a specific spherical harmonic basis function using PyVista.
    
    :param l: Spherical harmonic degree
    :param m: Spherical harmonic order
    :param resolution: Resolution of the sphere mesh
    :param save_path: Optional path to save the visualization
    :return: PyVista plotter object
    """
    # Create sphere mesh
    sphere = pv.Sphere(radius=1.0, theta_resolution=resolution, phi_resolution=resolution)
    
    # Get points on sphere surface
    points = sphere.points
    n_points = points.shape[0]
    
    # Convert to spherical coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Convert to our spherical coordinate convention
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arcsin(z / r)  # elevation angle [-π/2, π/2]
    phi = np.arctan2(y, x)    # azimuthal angle [-π, π]
    
    spherical_coords = torch.tensor(np.stack([theta, phi], axis=1), dtype=torch.float32)
    
    # Compute spherical harmonic basis function
    from .sph import spherical_to_sph_eval
    sph_values = spherical_to_sph_eval(spherical_coords, l, m).numpy()
    
    # Add scalar data to mesh
    sphere['Y_lm'] = sph_values
    
    # Create plotter (use off_screen for saving screenshots)
    plotter = pv.Plotter(off_screen=bool(save_path))
    plotter.add_mesh(
        sphere, 
        scalars='Y_lm',
        cmap='RdBu',
        show_scalar_bar=True,
        scalar_bar_args={'title': f'Y_{l}^{m}'}
    )
    
    plotter.add_title(f'Spherical Harmonic Y_{l}^{m}', font_size=16)
    plotter.show_grid()
    
    if save_path:
        plotter.screenshot(save_path)
        print(f"Visualization saved to {save_path}")
    
    return plotter


def visualize_basis_comparison(l_max: int = 2, resolution: int = 32, save_dir: Optional[str] = None) -> None:
    """
    Create a comparison visualization of cartesian vs spherical basis functions.
    
    :param l_max: Maximum spherical harmonic band to visualize
    :param resolution: Resolution of sphere meshes
    :param save_dir: Optional directory to save visualizations
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_terms = sph_indices_total(l_max)
    
    # Create visualization for each basis function
    for idx in range(min(n_terms, 9)):  # Limit to first 9 for reasonable visualization
        # Determine l and m from index
        l = int(np.sqrt(idx))
        m = idx - l*l - l
        
        print(f"Visualizing basis function {idx}: Y_{l}^{m}")
        
        # Create side-by-side comparison (use off_screen for saving screenshots)
        plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 500), off_screen=bool(save_dir))
        
        # Create sphere mesh and get its points
        sphere = pv.Sphere(radius=1.0, theta_resolution=resolution//2, phi_resolution=resolution)
        points = sphere.points  # Get actual mesh points
        n_points = points.shape[0]
        
        # Convert PyVista points to our coordinate systems
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Convert to spherical coordinates (theta, phi)
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arcsin(z / r)  # elevation angle [-π/2, π/2]
        phi = np.arctan2(y, x)    # azimuthal angle [-π, π]
        
        # Convert to tensors
        cartesian_coords = torch.tensor(points, dtype=torch.float32, device=device)
        spherical_coords = torch.tensor(np.stack([theta, phi], axis=1), dtype=torch.float32, device=device)
        
        # Compute basis functions at these specific points
        cartesian_basis = cartesian_to_sph_basis_vectorized(cartesian_coords.unsqueeze(0), l_max).squeeze(0)
        spherical_basis = spherical_to_sph_basis_vectorized(spherical_coords.unsqueeze(0), l_max).squeeze(0)
        
        # Extract the specific basis function
        cartesian_values = cartesian_basis[:, idx].cpu().numpy()
        spherical_values = spherical_basis[:, idx].cpu().numpy()
        diff_values = cartesian_values - spherical_values
        
        # Cartesian basis
        plotter.subplot(0, 0)
        sphere1 = sphere.copy()
        sphere1['Cartesian'] = cartesian_values
        plotter.add_mesh(sphere1, scalars='Cartesian', cmap='RdBu', show_scalar_bar=True)
        plotter.add_title(f'Cartesian Y_{l}^{m}')
        
        # Spherical basis
        plotter.subplot(0, 1)
        sphere2 = sphere.copy()
        sphere2['Spherical'] = spherical_values
        plotter.add_mesh(sphere2, scalars='Spherical', cmap='RdBu', show_scalar_bar=True)
        plotter.add_title(f'Spherical Y_{l}^{m}')
        
        # Difference
        plotter.subplot(0, 2)
        sphere3 = sphere.copy()
        sphere3['Difference'] = diff_values
        plotter.add_mesh(sphere3, scalars='Difference', cmap='viridis', show_scalar_bar=True)
        plotter.add_title(f'Difference (max: {np.max(np.abs(diff_values)):.2e})')
        
        if save_dir:
            save_path = f"{save_dir}/basis_comparison_Y_{l}_{m}.png"
            plotter.screenshot(save_path)
            print(f"Saved comparison to {save_path}")
        else:
            plotter.show()


def run_all_sanity_checks(H: int = 64, W: int = 128, l_max: int = 4) -> dict:
    """
    Run all sanity checks and return comprehensive results.
    
    :param H: Height for test coordinates
    :param W: Width for test coordinates  
    :param l_max: Maximum spherical harmonic band to test
    :return: Dictionary with all test results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running all sanity checks on {device}")
    print("=" * 60)
    
    results = {}
    
    # Generate test coordinates
    spherical_coordinates = generate_spherical_coordinates_map(H, W, device)
    cartesian_coordinates = spherical_to_cartesian(spherical_coordinates)
    
    # Test 1: Vectorized vs Original (Cartesian)
    print("1. Testing Cartesian Vectorized vs Original:")
    results['cartesian_vectorized_match'] = test_vectorized_vs_original(cartesian_coordinates, l_max)
    print()
    
    # Test 2: Vectorized vs Original (Spherical)  
    print("2. Testing Spherical Vectorized vs Original:")
    results['spherical_vectorized_match'] = test_spherical_vectorized_vs_original(spherical_coordinates, l_max)
    print()
    
    # Test 3: Cartesian vs Spherical Basis
    print("3. Testing Cartesian vs Spherical Basis:")
    accuracy_match, timing_results = test_cartesian_vs_spherical_basis(H, W, l_max, device)
    results['basis_accuracy_match'] = accuracy_match
    results['timing_results'] = timing_results
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY:")
    print(f"✓ All vectorized implementations match: {all([results['cartesian_vectorized_match'], results['spherical_vectorized_match']])}")
    print(f"✓ Cartesian and Spherical basis match: {results['basis_accuracy_match']}")
    print(f"✓ Fastest method: {min(timing_results.items(), key=lambda x: x[1])[0]}")
    
    return results


def main():
    """
    Main function to run spherical harmonic sanity checks.
    
    This function provides a command-line interface to run various tests
    and optionally generate visualizations.
    """
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Run spherical harmonic sanity checks')
    parser.add_argument('--height', '-H', type=int, default=64, help='Height of test coordinates (default: 64)')
    parser.add_argument('--width', '-W', type=int, default=128, help='Width of test coordinates (default: 128)')
    parser.add_argument('--l-max', type=int, default=4, help='Maximum spherical harmonic band (default: 4)')
    parser.add_argument('--visualize', action='store_true', help='Generate basis function visualizations')
    parser.add_argument('--save-dir', type=str, help='Directory to save visualizations (default: no saving)')
    parser.add_argument('--vis-resolution', type=int, default=32, help='Resolution for visualizations (default: 32)')
    parser.add_argument('--single-vis', type=str, nargs=2, metavar=('L', 'M'), 
                        help='Visualize single spherical harmonic Y_l^m (provide l and m as integers)')
    
    args = parser.parse_args()
    
    print("Spherical Harmonic Sanity Checks")
    print("=" * 40)
    print(f"Parameters: H={args.height}, W={args.width}, l_max={args.l_max}")
    print()
    
    # Run comprehensive sanity checks
    results = run_all_sanity_checks(args.height, args.width, args.l_max)
    
    # Check if all tests passed
    all_passed = (
        results['cartesian_vectorized_match'] and 
        results['spherical_vectorized_match'] and 
        results['basis_accuracy_match']
    )
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Spherical harmonic implementation is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    
    # Generate visualizations if requested
    if args.visualize:
        print()
        print("Generating basis function comparison visualizations...")
        if args.save_dir and not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        visualize_basis_comparison(
            l_max=min(args.l_max, 2),  # Limit for reasonable visualization
            resolution=args.vis_resolution,
            save_dir=args.save_dir
        )
        print("Visualization complete!")
    
    # Generate single visualization if requested
    if args.single_vis:
        l, m = int(args.single_vis[0]), int(args.single_vis[1])
        print(f"Generating visualization for Y_{l}^{m}...")
        
        save_path = None
        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, f"Y_{l}_{m}.png")
        
        plotter = visualize_spherical_harmonic_basis(
            l=l, m=m, 
            resolution=args.vis_resolution,
            save_path=save_path
        )
        
        if not save_path:  # Only show if not saving
            plotter.show()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

    """
    # Basic run with default parameters
    python -m src.LightingStudio.analysis.spherical_harmonic.sanity_checks 

    # Run with custom parameters
    python src/LightingStudio/analysis/spherical_harmonic/sanity_checks.py --height 128 --width 256 --l-max 6

    # Run with visualizations
    python src/LightingStudio/analysis/spherical_harmonic/sanity_checks.py --visualize --save-dir ./visualizations

    # Visualize a specific spherical harmonic (e.g., Y_2^1)
    python src/LightingStudio/analysis/spherical_harmonic/sanity_checks.py --single-vis 2 1 --save-dir ./visualizations
    """
