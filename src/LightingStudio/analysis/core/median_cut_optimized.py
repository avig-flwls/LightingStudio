"""
Optimized Median Cut Algorithm using pre-allocated tensors instead of heapq.
This version avoids Python lists and heap operations for better GPU performance.
"""

import torch
from typing import List, Optional, Dict, Any
from src.LightingStudio.analysis.utils.transforms import luminance, pixel_solid_angles, generate_spherical_coordinates_map, spherical_to_cartesian, cartesian_to_pixel
from src.LightingStudio.analysis.report.datatypes import SampleGPU, SampleCPU


def median_cut_sampling_optimized(hdri: torch.Tensor, n_samples: int, device: Optional[torch.device] = None) -> List[SampleGPU]:
    """
    Optimized median cut sampling using pre-allocated tensors.
    
    Key optimizations:
    1. Pre-allocate all region storage as tensors
    2. Use tensor operations for sorting instead of heapq
    3. Batch energy calculations
    4. Minimize CPU-GPU transfers
    
    Args:
        hdri: HDRI tensor with shape (H, W, 3) in linear RGB
        n_samples: Number of light samples to generate
        device: Optional device to run computation on (defaults to hdri.device)
    
    Returns:
        List of SampleGPU objects
    """
    if device is None:
        device = hdri.device
    
    hdri = hdri.to(device)
    H, W, _ = hdri.shape
    
    # Compute energy map once
    lum = luminance(hdri)
    pixel_area, sin_theta = pixel_solid_angles(H, W, device=device)
    sa_map = pixel_area * sin_theta
    sa_full = sa_map.expand(H, W)
    energy_map = lum * sa_full
    
    # Pre-allocate tensors for region management
    # We need at most 2*n_samples regions during processing
    max_regions = 2 * n_samples
    
    # Region bounds: [y0, y1, x0, x1] for each region
    region_bounds = torch.zeros((max_regions, 4), dtype=torch.int32, device=device)
    # Energy values for each region
    region_energies = torch.zeros(max_regions, dtype=torch.float32, device=device)
    # Valid mask for active regions
    region_valid = torch.zeros(max_regions, dtype=torch.bool, device=device)
    
    # Initialize with full image
    region_bounds[0] = torch.tensor([0, H, 0, W], device=device)
    region_energies[0] = torch.sum(energy_map)
    region_valid[0] = True
    
    # Main splitting loop
    region_idx = 1  # Track total regions created (start at 1 since we already have the full image)
    while torch.sum(region_valid) < n_samples and region_idx < max_regions - 1:
        # Find region with maximum energy among valid regions
        valid_energies = torch.where(region_valid, region_energies, torch.tensor(-float('inf'), device=device))
        max_idx = torch.argmax(valid_energies)
        
        if valid_energies[max_idx] <= 0:
            break  # No more splittable regions
        
        # Get the region to split
        y0, y1, x0, x1 = region_bounds[max_idx].tolist()
        
        # Perform split using tensor operations
        h, w = y1 - y0, x1 - x0
        
        if h < 2 and w < 2:
            # Can't split further
            region_valid[max_idx] = False
            continue
        
        # Vectorized split decision
        if h < 2:
            axis = 1
        elif w < 2:
            axis = 0
        elif w > h:
            axis = 1
        else:
            axis = 0
        
        # Compute split position using tensor operations
        if axis == 0:  # Split rows
            if h >= 2:
                row_energies = torch.sum(energy_map[y0:y1, x0:x1], dim=1)
                cumsum = torch.cumsum(row_energies, dim=0)
                total = cumsum[-1]
                if total > 0:
                    k = torch.searchsorted(cumsum, total * 0.5).item()
                    k = max(1, min(k, h - 1))
                else:
                    k = h // 2
                
                # Create children
                child_a = (y0, y0 + k, x0, x1)
                child_b = (y0 + k, y1, x0, x1)
            else:
                region_valid[max_idx] = False
                continue
        else:  # Split columns
            if w >= 2:
                col_energies = torch.sum(energy_map[y0:y1, x0:x1], dim=0)
                cumsum = torch.cumsum(col_energies, dim=0)
                total = cumsum[-1]
                if total > 0:
                    k = torch.searchsorted(cumsum, total * 0.5).item()
                    k = max(1, min(k, w - 1))
                else:
                    k = w // 2
                
                # Create children
                child_a = (y0, y1, x0, x0 + k)
                child_b = (y0, y1, x0 + k, x1)
            else:
                region_valid[max_idx] = False
                continue
        
        # Add children if we have space (before marking parent invalid)
        if region_idx + 1 < max_regions:
            # Mark parent as invalid only after successfully creating children
            region_valid[max_idx] = False
            # Child A
            region_bounds[region_idx] = torch.tensor(child_a, device=device)
            region_energies[region_idx] = torch.sum(energy_map[child_a[0]:child_a[1], child_a[2]:child_a[3]])
            region_valid[region_idx] = True
            
            # Child B
            region_bounds[region_idx + 1] = torch.tensor(child_b, device=device)
            region_energies[region_idx + 1] = torch.sum(energy_map[child_b[0]:child_b[1], child_b[2]:child_b[3]])
            region_valid[region_idx + 1] = True
            
            region_idx += 2
    
    # Collect final regions (top n_samples by energy)
    # Include all regions up to region_idx (or max_regions)
    total_regions = min(region_idx, max_regions)
    valid_mask = region_valid[:total_regions]
    valid_energies = region_energies[:total_regions]
    valid_bounds = region_bounds[:total_regions]
    
    # Get top n_samples regions by energy
    num_valid_regions = valid_mask.sum().item()
    num_regions = min(n_samples, num_valid_regions)
    samples = []
    
    # If we have no valid regions, use the full image as a single sample
    if num_valid_regions == 0:
        y0, y1, x0, x1 = 0, H, 0, W
        stats = compute_region_stats_optimized(hdri, y0, y1, x0, x1, energy_map, sa_full)
        sample = SampleGPU(
            direction=stats['direction'],
            power_rgb=stats['power_rgb'],
            solid_angle=stats['solid_angle'],
            avg_radiance_rgb=stats['avg_radiance_rgb'],
            rect=stats['rect'],
            pixel_coords=stats['pixel_coords'],
            energy=stats['energy']
        )
        samples.append(sample)
        return samples
    
    if num_regions > 0:
        # Filter to only valid regions
        valid_indices = torch.where(valid_mask)[0]
        valid_energies_filtered = valid_energies[valid_indices]
        valid_bounds_filtered = valid_bounds[valid_indices]
        
        # Sort by energy (descending)
        sorted_indices = torch.argsort(valid_energies_filtered, descending=True)
        top_indices = sorted_indices[:num_regions]
        
        # Batch compute statistics for all regions
        for i, idx in enumerate(top_indices):
            bounds = valid_bounds_filtered[idx]
            y0, y1, x0, x1 = bounds.tolist()
            
            
            # Compute stats (reusing precomputed maps)
            stats = compute_region_stats_optimized(hdri, y0, y1, x0, x1, energy_map, sa_full)
            
            sample = SampleGPU(
                direction=stats['direction'],
                power_rgb=stats['power_rgb'],
                solid_angle=stats['solid_angle'],
                avg_radiance_rgb=stats['avg_radiance_rgb'],
                rect=stats['rect'],
                pixel_coords=stats['pixel_coords'],
                energy=stats['energy']
            )
            samples.append(sample)
    
    return samples


def compute_region_stats_optimized(hdri: torch.Tensor, y0: int, y1: int, x0: int, x1: int, 
                                  energy_map: torch.Tensor, sa_map: torch.Tensor) -> Dict[str, Any]:
    """
    Optimized version that reuses precomputed maps.
    
    Args:
        hdri: HDRI tensor
        y0, y1, x0, x1: Region bounds
        energy_map: Precomputed energy map (luminance * solid angle)
        sa_map: Precomputed solid angle map
    
    Returns:
        Dictionary with region statistics
    """
    H, W, _ = hdri.shape
    device = hdri.device
    
    # Use pre-computed maps
    sa_region = sa_map[y0:y1, x0:x1]
    rgb_region = hdri[y0:y1, x0:x1, :]
    
    # Batch compute all values
    energy = energy_map[y0:y1, x0:x1].sum()
    power_rgb = (rgb_region * sa_region.unsqueeze(-1)).sum(dim=(0, 1))
    total_solid_angle = sa_region.sum()
    
    # Compute direction using tensor operations
    spherical_coords = generate_spherical_coordinates_map(H, W, device=device)
    spherical_region = spherical_coords[y0:y1, x0:x1, :]
    cartesian_dirs = spherical_to_cartesian(spherical_region)
    
    # Energy-weighted direction
    lum_region = luminance(rgb_region)
    weights = (lum_region * sa_region).unsqueeze(-1)
    weighted_dir = (cartesian_dirs * weights).sum(dim=(0, 1))
    
    dir_norm = torch.linalg.norm(weighted_dir)
    if dir_norm < 1e-8:
        # Fallback
        direction = cartesian_dirs.mean(dim=(0, 1))
        direction = direction / torch.linalg.norm(direction)
    else:
        direction = weighted_dir / dir_norm
    
    pixel_coords = cartesian_to_pixel(direction, H, W)
    avg_radiance = power_rgb / torch.clamp(total_solid_angle, min=1e-12)
    
    return {
        'power_rgb': power_rgb,
        'direction': direction,
        'pixel_coords': pixel_coords,
        'energy': energy,
        'solid_angle': total_solid_angle,
        'avg_radiance_rgb': avg_radiance,
        'rect': (int(y0), int(y1), int(x0), int(x1))
    }


def median_cut_sampling_to_cpu_optimized(hdri: torch.Tensor, n_samples: int, device: Optional[torch.device] = None) -> List[SampleCPU]:
    """
    Convenience function that performs optimized median cut sampling and converts results to CPU/numpy.
    
    Args:
        hdri: HDRI tensor with shape (H, W, 3) in linear RGB
        n_samples: Number of light samples to generate
        device: Optional device to run computation on (defaults to hdri.device)
    
    Returns:
        List of SampleCPU objects with CPU tensors converted to lists
    """
    samples = median_cut_sampling_optimized(hdri, n_samples, device)
    
    # Convert tensors to CPU and then to lists/floats
    cpu_samples = []
    for sample in samples:
        cpu_sample = SampleCPU(
            direction=sample.direction.cpu().numpy().tolist(),
            power_rgb=sample.power_rgb.cpu().numpy().tolist(),
            solid_angle=sample.solid_angle.cpu().item(),
            avg_radiance_rgb=sample.avg_radiance_rgb.cpu().numpy().tolist(),
            rect=sample.rect,
            pixel_coords=sample.pixel_coords.cpu().numpy().tolist(),
            energy=sample.energy.cpu().item()
        )
        cpu_samples.append(cpu_sample)
    
    return cpu_samples


# Additional optimization: Batch process multiple HDRIs on GPU
def median_cut_sampling_batch(hdris: torch.Tensor, n_samples: int, device: Optional[torch.device] = None) -> List[List[SampleGPU]]:
    """
    Process multiple HDRIs in a batch on GPU.
    
    Args:
        hdris: Batch of HDRIs with shape (B, H, W, 3)
        n_samples: Number of samples per HDRI
    
    Returns:
        List of sample lists, one per HDRI
    """
    if device is None:
        device = hdris.device
    
    B, H, W, _ = hdris.shape
    
    # Compute energy maps for all HDRIs at once
    lum_batch = luminance(hdris)  # (B, H, W)
    pixel_area, sin_theta = pixel_solid_angles(H, W, device=device)
    sa_map = pixel_area * sin_theta
    sa_full = sa_map.unsqueeze(0).expand(B, H, W)
    energy_maps = lum_batch * sa_full  # (B, H, W)
    
    # Process each HDRI (could be further optimized)
    all_samples = []
    for i in range(B):
        samples = median_cut_sampling_optimized(hdris[i], n_samples, device)
        all_samples.append(samples)
    
    return all_samples

