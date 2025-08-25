"""
3D visualization functions for spherical harmonic basis functions.

This module provides interactive visualization capabilities including:
1. Lobe visualization with 3D radius variation
2. HTML export for web browser viewing
3. Traditional 3D mesh formats (VTP, PLY)
4. Comparison visualizations between different implementations
"""

import torch
import numpy as np
import os
import pyvista as pv
from typing import Optional
from pathlib import Path

from ..sph import (
    cartesian_to_sph_basis_vectorized,
    spherical_to_sph_basis_vectorized,
    sph_indices_total, lm_from_index, spherical_to_sph_eval
)
from ...utils import generate_spherical_coordinates_map, spherical_to_cartesian


def visualize_spherical_harmonic_basis(l: int, m: int, resolution: int = 64, save_path: Optional[str] = None) -> pv.Plotter:
    """
    Visualize a specific spherical harmonic basis function with lobes as HTML.
    
    :param l: Spherical harmonic degree
    :param m: Spherical harmonic order
    :param resolution: Resolution of the sphere mesh
    :param save_path: Optional path to save the HTML file (without extension)
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
    sph_values = spherical_to_sph_eval(spherical_coords, l, m).numpy()
    
    # Create lobe visualization by varying radius
    # Offset and scale the values to create nice looking lobes
    min_val, max_val = np.min(sph_values), np.max(sph_values)
    val_range = max_val - min_val
    
    if val_range > 0:
        # Normalize to [0, 1] and scale to create visible lobes
        normalized_values = (sph_values - min_val) / val_range
        # Map to radius range [0.3, 1.5] to create clear lobe structure
        radius_scale = 0.3 + 1.2 * normalized_values
    else:
        # Handle constant function case
        radius_scale = np.ones_like(sph_values)
    
    # Create new points by scaling original directions by the radius
    new_points = points * radius_scale[:, np.newaxis]
    
    # Create new mesh with modified points
    lobe_mesh = sphere.copy()
    lobe_mesh.points = new_points
    
    # Add both the original values and radius scaling as data
    lobe_mesh['Y_lm'] = sph_values
    lobe_mesh['Radius_Scale'] = radius_scale
    
    # Create plotter
    plotter = pv.Plotter(off_screen=bool(save_path))
    plotter.add_mesh(
        lobe_mesh, 
        scalars='Y_lm',
        cmap='RdBu',
        show_scalar_bar=True,
        scalar_bar_args={'title': f'Y_{l}^{m}'}
    )
    
    # Add a reference unit sphere for comparison
    reference_sphere = pv.Sphere(radius=1.0, theta_resolution=resolution//4, phi_resolution=resolution//2)
    plotter.add_mesh(reference_sphere, style='wireframe', color='gray', opacity=0.3, line_width=1)
    
    plotter.add_title(f'Spherical Harmonic Lobes Y_{l}^{m}', font_size=16)
    
    plotter.show_grid()
    
    if save_path:
        # Save interactive HTML file
        html_path = f"{save_path}.html"
        plotter.export_html(html_path)
        print(f"Interactive HTML saved to {html_path}")
        print(f"  Open in browser: file://{os.path.abspath(html_path)}")
    
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


def visualize_multiple_harmonics(harmonics_list: list, resolution: int = 64, save_dir: Optional[str] = None) -> None:
    """
    Create visualizations for multiple spherical harmonic basis functions.
    
    :param harmonics_list: List of (l, m) tuples to visualize
    :param resolution: Resolution of sphere meshes
    :param save_dir: Optional directory to save visualizations
    """
    print(f"Creating {len(harmonics_list)} spherical harmonic visualizations")
    
    for i, (l, m) in enumerate(harmonics_list):
        print(f"Generating visualization {i+1}/{len(harmonics_list)}: Y_{l}^{m}")
        
        if save_dir:
            save_path = f"{save_dir}/Y_{l}_{m}"
        else:
            save_path = None
        
        # Create visualization
        plotter = visualize_spherical_harmonic_basis(
            l=l, m=m,
            resolution=resolution,
            save_path=save_path
        )
        
        if not save_dir:
            # Show interactively if not saving
            plotter.show()


def load_and_view_3d_file(file_path: str, title: Optional[str] = None) -> None:
    """
    Load and interactively view a saved 3D spherical harmonic file.
    
    :param file_path: Path to the .vtp or .ply file
    :param title: Optional title for the visualization
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Load the mesh
    mesh = pv.read(file_path)
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add the mesh with appropriate coloring
    if 'Y_lm' in mesh.array_names:
        # Has spherical harmonic values - use those for coloring
        plotter.add_mesh(
            mesh, 
            scalars='Y_lm',
            cmap='RdBu',
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Y_lm values'}
        )
    else:
        # No special coloring, just show the mesh
        plotter.add_mesh(mesh, show_edges=True)
    
    # Set title
    if title:
        plotter.add_title(title, font_size=16)
    else:
        filename = Path(file_path).name
        plotter.add_title(f'3D Visualization: {filename}', font_size=16)
    
    plotter.show_grid()
    plotter.show()


def main():
    """
    Main function to run visualization demos with command-line interface.
    """
    import argparse
    from coolname import generate_slug
    
    OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"
    
    parser = argparse.ArgumentParser(description='Run spherical harmonic visualizations')
    parser.add_argument('--vis-resolution', type=int, default=64, help='Resolution for visualizations (default: 64)')
    parser.add_argument('--single-vis', type=str, nargs=2, metavar=('L', 'M'), 
                        help='Visualize single spherical harmonic Y_l^m (provide l and m as integers)')
    parser.add_argument('--multiple-vis', type=str, nargs='+', metavar='L,M', 
                        help='Visualize multiple spherical harmonics (e.g., --multiple-vis 1,0 2,1 2,-1)')
    
    args = parser.parse_args()
    
    # Create experiment-specific output directory
    experiment_name = generate_slug(2)
    output_dir = Path(OUTPUT_DIR) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print("Spherical Harmonic Visualizations")
    print("=" * 40)
    print(f"Parameters: resolution={args.vis_resolution}")
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
    
    if not args.single_vis and not args.multiple_vis:
        print("No visualization options specified. Use --single-vis or --multiple-vis")
        print("For help: python -m src.LightingStudio.analysis.spherical_harmonic.sanity_checks.visualization --help")
    
    return 0


if __name__ == "__main__":
    exit(main())


"""
Usage Examples:

# Visualize a single spherical harmonic with lobes (saves HTML)
python src/LightingStudio/analysis/spherical_harmonic/sanity_checks/visualization.py --single-vis 2 1

# Visualize multiple spherical harmonics
python src/LightingStudio/analysis/spherical_harmonic/sanity_checks/visualization.py --multiple-vis 0,0 1,0 1,1 2,0

# High resolution visualization
python src/LightingStudio/analysis/spherical_harmonic/sanity_checks/visualization.py --single-vis 3 2 --vis-resolution 128

# HTML files can be opened directly in any web browser:
# - Double-click the .html file in file explorer, or
# - Drag the .html file to your browser, or  
# - Use file://path/to/file.html in browser address bar
"""
