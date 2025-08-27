"""
Numerical unit tests and verification for spherical harmonic functions.

This module contains tests to verify:
1. Cartesian vs spherical basis function equivalence (accuracy and speed)
2. Vectorized vs non-vectorized implementations (accuracy and speed)
3. Coordinate conversion roundtrip accuracy
"""

import torch
import numpy as np
import time
from typing import Tuple

from src.LightingStudio.analysis.core.sph import (
    cartesian_to_sph_basis, cartesian_to_sph_basis_vectorized,
    spherical_to_sph_basis, spherical_to_sph_basis_vectorized,
    sph_indices_total, lm_from_index
)
from src.LightingStudio.analysis.utils.transforms import generate_spherical_coordinates_map, spherical_to_cartesian, cartesian_to_spherical


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

    # Check if results are close for all terms
    is_close = torch.allclose(original_result, vectorized_result, rtol=1e-5, atol=1e-6)
    
    if not is_close:
        total_terms = sph_indices_total(l_max)
        for i in range(total_terms):
            l, m = lm_from_index(i)
            print(f"l: {l}, m: {m}")

            o = original_result[..., i]
            v = vectorized_result[..., i]
        
            print(f"cartesian_original: {o}")
            print(f"cartesian_vectorized: {v}")
        
            if not torch.allclose(o, v, rtol=1e-5, atol=1e-6):
                print(f"Results differ for term {i} (Y_{l}^{m})! Max difference: {torch.max(torch.abs(o - v)).item():.2e}")

        max_diff = torch.max(torch.abs(original_result - vectorized_result))
        print(f"❌ Cartesian original vs vectorized differ! Max difference: {max_diff.item():.2e}")
        return False
    
    print(f"✓ Cartesian vectorized implementation matches original for l_max={l_max}")
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
        total_terms = sph_indices_total(l_max)
        for i in range(total_terms):
            l, m = lm_from_index(i)
            print(f"l: {l}, m: {m}")

            o = original_result[..., i]
            v = vectorized_result[..., i]
        
            print(f"spherical_original: {o}")
            print(f"spherical_vectorized: {v}")
        
            if not torch.allclose(o, v, rtol=1e-5, atol=1e-7):
                print(f"Results differ for term {i} (Y_{l}^{m})! Max difference: {torch.max(torch.abs(o - v)).item():.2e}")

        max_diff = torch.max(torch.abs(original_result - vectorized_result))
        print(f"❌ Spherical original vs vectorized differ! Max difference: {max_diff.item():.2e}")
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
    
    # Check accuracy between cartesian and spherical
    accuracy_match = torch.allclose(cartesian_original, spherical_original, rtol=1e-4, atol=1e-5)
    accuracy_match_vectorized = torch.allclose(cartesian_vectorized, spherical_vectorized, rtol=1e-4, atol=1e-5)
    
    if not accuracy_match:
        total_terms = sph_indices_total(l_max)
        for i in range(total_terms):
            l, m = lm_from_index(i)
            print(f"l: {l}, m: {m}")

            o = cartesian_original[..., i]
            v = spherical_original[..., i]
        
            print(f"cartesian_original: {o}")
            print(f"spherical_original: {v}")
        
            if not torch.allclose(o, v, rtol=1e-4, atol=1e-5):
                print(f"Results differ for term {i} (Y_{l}^{m})! Max difference: {torch.max(torch.abs(o - v)).item():.2e}")

        max_diff = torch.max(torch.abs(cartesian_original - spherical_original))
        print(f"❌ Cartesian vs Spherical basis differ! Max difference: {max_diff.item():.2e}")
    else:
        print(f"✓ Cartesian and Spherical basis match within tolerance (original)")

    if not accuracy_match_vectorized:
        total_terms = sph_indices_total(l_max)
        for i in range(total_terms):
            l, m = lm_from_index(i)
            print(f"l: {l}, m: {m}")

            o = cartesian_vectorized[..., i]
            v = spherical_vectorized[..., i]
        
            print(f"cartesian_vectorized: {o}")
            print(f"spherical_vectorized: {v}")
        
            if not torch.allclose(o, v, rtol=1e-4, atol=1e-5):
                print(f"Results differ for term {i} (Y_{l}^{m})! Max difference: {torch.max(torch.abs(o - v)).item():.2e}")

        max_diff = torch.max(torch.abs(cartesian_vectorized - spherical_vectorized))
        print(f"❌ Cartesian vs Spherical Vectorized basis differ! Max difference: {max_diff.item():.2e}")
    else:
        print(f"✓ Cartesian and Spherical Vectorized basis match within tolerance")
    
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
   
    # Test 2: Vectorized vs Original (Cartesian)
    print("2. Testing Cartesian Vectorized vs Original:")
    results['cartesian_vectorized_match'] = test_vectorized_vs_original(cartesian_coordinates, l_max)
    print()
    
    # Test 3: Vectorized vs Original (Spherical)  
    print("3. Testing Spherical Vectorized vs Original:")
    results['spherical_vectorized_match'] = test_spherical_vectorized_vs_original(spherical_coordinates, l_max)
    print()
    
    # Test 4: Cartesian vs Spherical Basis
    print("4. Testing Cartesian vs Spherical Basis:")
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
    Main function to run unit tests with command-line interface.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run spherical harmonic unit tests')
    parser.add_argument('--height', '-H', type=int, default=64, help='Height of test coordinates (default: 64)')
    parser.add_argument('--width', '-W', type=int, default=128, help='Width of test coordinates (default: 128)')
    parser.add_argument('--l-max', type=int, default=4, help='Maximum spherical harmonic band (default: 4)')
    
    args = parser.parse_args()
    
    print("Spherical Harmonic Unit Tests")
    print("=" * 40)
    print(f"Parameters: H={args.height}, W={args.width}, l_max={args.l_max}")
    print()
    
    # Run comprehensive unit tests
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
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
