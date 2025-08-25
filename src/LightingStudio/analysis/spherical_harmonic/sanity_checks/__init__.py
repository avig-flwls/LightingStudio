"""
Sanity checks and verification tests for spherical harmonic functions.

This package contains:
- unit_tests: Numerical equivalence tests and accuracy verification
- visualization: 3D visualization functions for spherical harmonics
"""

from .unit_tests import (
    test_vectorized_vs_original,
    test_spherical_vectorized_vs_original,
    test_coordinate_conversion_roundtrip,
    test_cartesian_vs_spherical_basis,
    run_all_sanity_checks
)

from .visualization import (
    visualize_spherical_harmonic_basis,
    visualize_multiple_harmonics,
    visualize_basis_comparison,
    load_and_view_3d_file
)

__all__ = [
    # Unit tests
    'test_vectorized_vs_original',
    'test_spherical_vectorized_vs_original', 
    'test_coordinate_conversion_roundtrip',
    'test_cartesian_vs_spherical_basis',
    'run_all_sanity_checks',
    
    # Visualization
    'visualize_spherical_harmonic_basis',
    'visualize_multiple_harmonics',
    'visualize_basis_comparison',
    'load_and_view_3d_file'
]
