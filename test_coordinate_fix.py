#!/usr/bin/env python3
"""
Test script to verify the coordinate conversion fixes.
"""

import torch
import numpy as np
import sys
import os

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from LightingStudio.analysis.utils import (
    generate_spherical_coordinates_map, 
    spherical_to_cartesian, 
    cartesian_to_spherical,
    spherical_to_pixel,
    cartesian_to_pixel
)

def test_coordinate_round_trip():
    """Test that coordinate conversions are consistent."""
    print("Testing coordinate conversion round-trip consistency...")
    
    H, W = 256, 512  # Typical HDRI dimensions
    device = torch.device('cpu')
    
    # Test a few specific pixels
    test_pixels = [
        [0, 0],        # Top-left
        [H//2, W//2],  # Center
        [H-1, W-1],    # Bottom-right
        [H//4, W//4],  # Quarter point
        [3*H//4, 3*W//4]  # Three-quarter point
    ]
    
    print(f"Testing with image dimensions: {H}x{W}")
    
    for pixel_y, pixel_x in test_pixels:
        print(f"\nTesting pixel ({pixel_y}, {pixel_x}):")
        
        # Get spherical coordinates for this pixel
        spherical_map = generate_spherical_coordinates_map(H, W, device=device)
        spherical_coord = spherical_map[pixel_y, pixel_x, :]  # [theta, phi]
        print(f"  Spherical coord: theta={spherical_coord[0]:.4f}, phi={spherical_coord[1]:.4f}")
        
        # Convert to cartesian
        cartesian_coord = spherical_to_cartesian(spherical_coord.unsqueeze(0))  # Add batch dim
        cartesian_coord = cartesian_coord.squeeze(0)  # Remove batch dim
        print(f"  Cartesian coord: {cartesian_coord}")
        
        # Convert back to pixel via spherical path
        pixel_from_spherical = spherical_to_pixel(spherical_coord.unsqueeze(0), H, W)
        pixel_from_spherical = pixel_from_spherical.squeeze(0)
        print(f"  Pixel from spherical: {pixel_from_spherical}")
        
        # Convert back to pixel via cartesian path
        pixel_from_cartesian = cartesian_to_pixel(cartesian_coord.unsqueeze(0), H, W)
        pixel_from_cartesian = pixel_from_cartesian.squeeze(0)
        print(f"  Pixel from cartesian: {pixel_from_cartesian}")
        
        # Check consistency
        original_pixel = torch.tensor([pixel_y, pixel_x], dtype=torch.int32)
        spherical_error = torch.abs(pixel_from_spherical - original_pixel).max().item()
        cartesian_error = torch.abs(pixel_from_cartesian - original_pixel).max().item()
        
        print(f"  Spherical round-trip error: {spherical_error}")
        print(f"  Cartesian round-trip error: {cartesian_error}")
        
        if spherical_error <= 1 and cartesian_error <= 1:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")
            return False
    
    print("\n✓ All coordinate conversion tests passed!")
    return True

def test_comparison_with_old_implementation():
    """Compare with the old implementation's dir_to_pixel function."""
    print("\nTesting comparison with old implementation...")
    
    H, W = 256, 512
    device = torch.device('cpu')
    
    # Old implementation's dir_to_pixel function (adapted)
    def old_dir_to_pixel(direction, H, W):
        x, y, z = direction
        theta = np.arccos(np.clip(y, -1.0, 1.0))  # [0, π] 
        phi = np.arctan2(z, x)                    # [-π, π)
        u = int(((phi + np.pi) / (2 * np.pi)) * W) % W
        v = int((theta / np.pi) * H)
        return u, v  # (x, y) format
    
    # Test with a few known directions
    test_directions = [
        [1.0, 0.0, 0.0],   # +X direction
        [0.0, 1.0, 0.0],   # +Y direction (up)
        [0.0, 0.0, 1.0],   # +Z direction
        [-1.0, 0.0, 0.0],  # -X direction
        [0.0, -1.0, 0.0],  # -Y direction (down)
        [0.0, 0.0, -1.0],  # -Z direction
    ]
    
    for direction in test_directions:
        direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)
        print(f"\nTesting direction: {direction}")
        
        # Old implementation result
        old_u, old_v = old_dir_to_pixel(direction, H, W)
        print(f"  Old implementation: u={old_u}, v={old_v}")
        
        # New implementation result
        new_pixel = cartesian_to_pixel(direction_tensor.unsqueeze(0), H, W).squeeze(0)
        new_y, new_x = new_pixel[0].item(), new_pixel[1].item()
        print(f"  New implementation: y={new_y}, x={new_x}")
        
        # Note: old returns (u=x, v=y), new returns [y, x]
        print(f"  Comparison: old(u,v)=({old_u},{old_v}) vs new(y,x)=({new_y},{new_x})")


if __name__ == "__main__":
    print("Testing coordinate conversion fixes...")
    
    success = test_coordinate_round_trip()
    test_comparison_with_old_implementation()
    
    if success:
        print("\n🎉 Coordinate conversion fixes appear to be working correctly!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
