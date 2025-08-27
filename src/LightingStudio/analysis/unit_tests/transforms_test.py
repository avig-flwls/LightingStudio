import torch
import numpy as np
from src.LightingStudio.analysis.utils.transforms import spherical_to_cartesian, cartesian_to_spherical, generate_spherical_coordinates_map

def spherical_allclose(spherical_coords1: torch.Tensor, spherical_coords2: torch.Tensor) -> bool:
    """
    Check if two spherical coordinates are close.
    
    Coordinate system:
    - theta (elevation): [-π/2, π/2] where -π/2 is south pole, +π/2 is north pole
    - phi (azimuthal): [-π, π] wrapping around the unit circle
    
    :param spherical_coords1: First spherical coordinates (..., 2)
    :param spherical_coords2: Second spherical coordinates (..., 2)
    :return: True if coordinates are close, False otherwise
    """
    theta_a, phi_a = spherical_coords1[..., 0], spherical_coords1[..., 1]
    theta_b, phi_b = spherical_coords2[..., 0], spherical_coords2[..., 1]

    # Check for finite values
    finite_a = torch.isfinite(theta_a) & torch.isfinite(phi_a)
    finite_b = torch.isfinite(theta_b) & torch.isfinite(phi_b)
    both_finite = finite_a & finite_b
    
    # If not both finite, they're not close
    if not torch.all(both_finite):
        return False

    # Theta comparison: direct comparison since theta ∈ [-π/2, π/2]
    theta_close = torch.abs(theta_a - theta_b) <= 1e-6

    # Pole detection: at poles (theta ≈ ±π/2), phi is undefined
    # cos(theta) ≈ 0 when theta ≈ ±π/2
    cos_theta_a = torch.cos(theta_a)
    cos_theta_b = torch.cos(theta_b)
    at_pole_a = torch.abs(cos_theta_a) <= 1e-6
    at_pole_b = torch.abs(cos_theta_b) <= 1e-6
    either_at_pole = at_pole_a | at_pole_b

    # Phi comparison: handle wrapping on unit circle
    # For points not at poles, compare phi with proper wrapping
    dphi = phi_a - phi_b
    # Normalize to [-π, π] using atan2 to handle wrapping
    dphi_wrapped = torch.atan2(torch.sin(dphi), torch.cos(dphi))
    phi_close = torch.abs(dphi_wrapped) <= 1e-6
    
    # At poles, phi is undefined, so we ignore phi comparison
    phi_close_or_pole = phi_close | either_at_pole

    return torch.all(theta_close & phi_close_or_pole)


def test_coordinate_conversion_roundtrip(spherical_coordinates: torch.Tensor) -> bool:
    """
    Test that coordinate conversion functions are inverse operations.
    
    Verifies that spherical -> cartesian -> spherical returns the original coordinates.
    
    :param spherical_coordinates: Test spherical coordinates (theta, phi)
    :return: True if conversion roundtrip is successful
    """
    cartesian_coordinates = spherical_to_cartesian(spherical_coordinates)
    converted_back = cartesian_to_spherical(cartesian_coordinates)
    
    if not spherical_allclose(spherical_coordinates, converted_back):
        print("❌ Coordinate conversion roundtrip failed!")
        print(f"Original spherical coordinates: {spherical_coordinates.shape}")
        print(f"Converted back spherical coordinates: {converted_back.shape}")
        print(f"Original: {spherical_coordinates}")
        print(f"Converted back: {converted_back}")
        return False
    
    print("✓ Coordinate conversion roundtrip successful")
    return True


def run_all_transform_tests(H: int = 64, W: int = 128) -> dict:
    """
    Run all coordinate transformation tests and return comprehensive results.
    
    :param H: Height for test coordinates
    :param W: Width for test coordinates
    :return: Dictionary with all test results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running all coordinate transformation tests on {device}")
    print("=" * 60)
    
    results = {}
    
    # Generate test coordinates
    print("1. Generating test coordinates...")
    spherical_coordinates = generate_spherical_coordinates_map(H, W, device)
    print(f"✓ Generated {H}x{W} spherical coordinate map")
    print()
    
    # Test 1: Coordinate Conversion Roundtrip
    print("2. Testing Coordinate Conversion Roundtrip:")
    results['coordinate_conversion_match'] = test_coordinate_conversion_roundtrip(spherical_coordinates)
    print()
    
    # Test 2: Edge cases (poles and wrap-around)
    print("3. Testing Edge Cases (Poles and Wrap-around):")
    edge_cases = torch.tensor([
        [np.pi/2, 0],      # North pole
        [-np.pi/2, 0],     # South pole  
        [0, -np.pi],       # Wrap around at -π
        [0, np.pi],        # Wrap around at π
        [0, 0],            # Equator, prime meridian
        [np.pi/4, np.pi/2], # 45° elevation, 90° azimuth
        [-np.pi/4, -np.pi/2], # -45° elevation, -90° azimuth
    ], device=device, dtype=torch.float32)
    
    results['edge_case_conversion_match'] = test_coordinate_conversion_roundtrip(edge_cases)
    print()
    
    # Test 3: Unit sphere constraint
    print("4. Testing Unit Sphere Constraint:")
    cartesian_coords = spherical_to_cartesian(spherical_coordinates)
    magnitudes = torch.norm(cartesian_coords, dim=-1)
    unit_sphere_check = torch.allclose(magnitudes, torch.ones_like(magnitudes), rtol=1e-6, atol=1e-6)
    
    if unit_sphere_check:
        print("✓ All points lie on unit sphere")
        results['unit_sphere_constraint'] = True
    else:
        max_deviation = torch.max(torch.abs(magnitudes - 1.0))
        print(f"❌ Points deviate from unit sphere! Max deviation: {max_deviation.item():.2e}")
        results['unit_sphere_constraint'] = False
    print()
    
    # Test 4: Spherical coordinate ranges
    print("5. Testing Spherical Coordinate Ranges:")
    theta = spherical_coordinates[..., 0]
    phi = spherical_coordinates[..., 1]
    
    theta_range_check = torch.all(theta >= -np.pi/2) and torch.all(theta <= np.pi/2)
    phi_range_check = torch.all(phi >= -np.pi) and torch.all(phi <= np.pi)
    
    if theta_range_check:
        print(f"✓ Theta in valid range [-π/2, π/2]: [{theta.min():.3f}, {theta.max():.3f}]")
    else:
        print(f"❌ Theta out of range: [{theta.min():.3f}, {theta.max():.3f}]")
    
    if phi_range_check:
        print(f"✓ Phi in valid range [-π, π]: [{phi.min():.3f}, {phi.max():.3f}]")
    else:
        print(f"❌ Phi out of range: [{phi.min():.3f}, {phi.max():.3f}]")
    
    results['theta_range_check'] = theta_range_check
    results['phi_range_check'] = phi_range_check
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY:")
    print(f"✓ Coordinate conversion roundtrip works: {results['coordinate_conversion_match']}")
    print(f"✓ Edge case conversion works: {results['edge_case_conversion_match']}")
    print(f"✓ Unit sphere constraint satisfied: {results['unit_sphere_constraint']}")
    print(f"✓ Theta range valid: {results['theta_range_check']}")
    print(f"✓ Phi range valid: {results['phi_range_check']}")
    
    all_passed = all([
        results['coordinate_conversion_match'],
        results['edge_case_conversion_match'], 
        results['unit_sphere_constraint'],
        results['theta_range_check'],
        results['phi_range_check']
    ])
    
    print(f"✓ All transform tests passed: {all_passed}")
    
    return results


def main():
    """
    Main function to run transform tests with command-line interface.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run coordinate transformation unit tests')
    parser.add_argument('--height', '-H', type=int, default=64, help='Height of test coordinates (default: 64)')
    parser.add_argument('--width', '-W', type=int, default=128, help='Width of test coordinates (default: 128)')
    
    args = parser.parse_args()
    
    print("Coordinate Transformation Unit Tests")
    print("=" * 40)
    print(f"Parameters: H={args.height}, W={args.width}")
    print()
    
    # Run comprehensive transform tests
    results = run_all_transform_tests(args.height, args.width)
    
    # Check if all tests passed
    all_passed = all([
        results['coordinate_conversion_match'],
        results['edge_case_conversion_match'],
        results['unit_sphere_constraint'], 
        results['theta_range_check'],
        results['phi_range_check']
    ])
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Coordinate transformation implementation is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
