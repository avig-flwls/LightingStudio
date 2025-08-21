"""
Median Cut Algorithm for Light Probe Sampling
https://vgl.ict.usc.edu/Research/MedianCut/MedianCutSampling.pdf

This module implements the median cut algorithm for sampling HDRI environment maps.
It uses PyTorch throughout for GPU acceleration and consistency with the analysis utilities.
"""

import heapq
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import torch

from ..utils import luminance, pixel_solid_angles, generate_spherical_coordinates_map, spherical_to_cartesian, cartesian_to_pixel
from ..datatypes import Region, SampleGPU, SampleCPU


def cumulative_axis_sums(energy: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Sum energy along the orthogonal axis to get a 1D marginal distribution.
    
    Args:
        energy: 2D energy tensor
        axis: 0 to sum over columns (per-row totals), 1 to sum over rows (per-column totals)
    
    Returns:
        1D tensor of marginal sums
    """
    if axis == 0:
        return energy.sum(dim=1)  # sum over columns -> shape [H]
    else:
        return energy.sum(dim=0)  # sum over rows -> shape [W]


def find_median_cut_index(marginal: torch.Tensor) -> int:
    """
    Find split index k such that sum[0:k] ≈ sum[k:].
    
    Args:
        marginal: 1D tensor of non-negative energy values
    
    Returns:
        Split index k in [1, len-1] to ensure non-empty halves
    """
    cumsum = torch.cumsum(marginal, dim=0)
    total = cumsum[-1]
    
    if total <= 0.0:
        # Fallback to middle if no energy
        return max(1, len(marginal) // 2)
    
    half = total * 0.5
    k = torch.searchsorted(cumsum, half).item()
    k = int(torch.clamp(torch.tensor(k), 1, len(marginal) - 1).item())
    return k


def split_region_by_median(energy: torch.Tensor, y0: int, y1: int, x0: int, x1: int) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Split a rectangular region by median energy along the longer axis.
    
    Args:
        energy: 2D energy tensor
        y0, y1, x0, x1: Rectangle bounds (y1 and x1 are exclusive)
    
    Returns:
        Tuple of two child rectangles: ((y0, y1, x0, x1), (y0, y1, x0, x1))
        Returns the original region twice if it cannot be split without creating zero-area regions
    """
    h = y1 - y0
    w = x1 - x0
    
    # Check if region is too small to split (need at least 2 pixels in each dimension to split)
    if h < 2 and w < 2:
        return (y0, y1, x0, x1), (y0, y1, x0, x1)
    
    # Choose split axis: prefer the dimension that has at least 2 pixels
    if h < 2:
        axis = 1  # Can only split columns
    elif w < 2:
        axis = 0  # Can only split rows
    elif w > h:
        axis = 1  # split columns
    elif h > w:
        axis = 0  # split rows
    else:
        # Tie-breaker: axis with larger energy variance
        region_energy = energy[y0:y1, x0:x1]
        row_marg = cumulative_axis_sums(region_energy, axis=0)
        col_marg = cumulative_axis_sums(region_energy, axis=1)
        axis = 0 if torch.var(row_marg) >= torch.var(col_marg) else 1

    if axis == 0:
        # Split rows
        if h < 2:
            return (y0, y1, x0, x1), (y0, y1, x0, x1)
        marg = cumulative_axis_sums(energy[y0:y1, x0:x1], axis=0)
        k = find_median_cut_index(marg)
        child_a = (y0, y0 + k, x0, x1)
        child_b = (y0 + k, y1, x0, x1)
    else:
        # Split columns
        if w < 2:
            return (y0, y1, x0, x1), (y0, y1, x0, x1)
        marg = cumulative_axis_sums(energy[y0:y1, x0:x1], axis=1)
        k = find_median_cut_index(marg)
        child_a = (y0, y1, x0, x0 + k)
        child_b = (y0, y1, x0 + k, x1)
    
    return child_a, child_b


def compute_region_stats(hdri: torch.Tensor, y0: int, y1: int, x0: int, x1: int) -> Dict[str, Any]:
    """
    Compute statistics for a rectangular region of the HDRI.
    
    Args:
        hdri: HDRI tensor with shape (H, W, 3) in linear RGB
        y0, y1, x0, x1: Rectangle bounds (y1 and x1 are exclusive)
        
    Returns:
        Dictionary containing:
        - power_rgb: Total RGB power (∑ L_rgb * Δω) () treat this as radiant flux per steradian integrated over region (i.e., "power" the light should emit).
        - direction: Energy-weighted average direction (unit vector)
        - energy: Total luminance energy (∑ luminance * Δω)
        - solid_angle: Total solid angle (∑ Δω)
        - avg_radiance_rgb: Average radiance (power / solid_angle)
        - rect: Rectangle bounds
    """
    logger = logging.getLogger(__name__)
    
    H, W, _ = hdri.shape
    device = hdri.device
    
    logger.debug(f"Computing region stats for region ({y0}:{y1}, {x0}:{x1}) on HDRI shape {(H, W)} device={device}")
    
    # Get solid angles for the region
    sa_map = pixel_solid_angles(H, W, device=device)  # (H, 1)
    sa_region = sa_map[y0:y1, :]  # (region_h, 1)
    sa_region = sa_region.expand(y1 - y0, x1 - x0)  # (region_h, region_w)
    
    # Get RGB and luminance for the region
    rgb_region = hdri[y0:y1, x0:x1, :]  # (region_h, region_w, 3)
    lum_region = luminance(rgb_region) # (region_h, region_w)
    
    logger.debug(f"Region shape: {rgb_region.shape}, solid angle shape: {sa_region.shape}")
    
    # Compute energy and power
    energy = torch.sum(lum_region * sa_region)
    power_rgb = torch.sum(rgb_region * sa_region.unsqueeze(-1), dim=(0, 1))  # (3,)
    total_solid_angle = torch.sum(sa_region)
    
    logger.debug(f"Energy: {energy:.6f}, Power RGB: {power_rgb}, Total solid angle: {total_solid_angle:.6f}")
    
    # Compute energy-weighted average direction
    # Generate spherical coordinates for the region
    spherical_coords = generate_spherical_coordinates_map(H, W, device=device)  # (H, W, 2)
    spherical_region = spherical_coords[y0:y1, x0:x1, :]  # (region_h, region_w, 2)
    
    # Convert to cartesian directions (device is automatically inherited)
    cartesian_dirs = spherical_to_cartesian(spherical_region)  # (region_h, region_w, 3)
    
    # Weight by luminance * solid_angle
    weights = (lum_region * sa_region).unsqueeze(-1)  # (region_h, region_w, 1)
    weighted_dir = torch.sum(cartesian_dirs * weights, dim=(0, 1))  # (3,)

    # Log weight statistics to debug small weight issues
    weight_sum = torch.sum(weights)
    weight_max = torch.max(weights)
    weight_min = torch.min(weights)
    logger.debug(f"Weight statistics - sum: {weight_sum:.2e}, max: {weight_max:.2e}, min: {weight_min:.2e}")
    logger.debug(f"Weighted direction (unnormalized): {weighted_dir}, magnitude: {torch.linalg.norm(weighted_dir):.2e}")

    # Normalize to unit vector with fallback for near-zero vectors
    dir_norm = torch.linalg.norm(weighted_dir)
    
    # Check if the weighted direction is too small (indicating uniform/dark region)
    if dir_norm < 1e-8:
        logger.debug(f"Weighted direction norm ({dir_norm:.2e}) is very small - using geometric mean fallback")
        # Fallback: use simple geometric mean of all directions in the region
        avg_dir = torch.mean(cartesian_dirs, dim=(0, 1))  # (3,)
        avg_dir_norm = torch.linalg.norm(avg_dir)
        direction = avg_dir / avg_dir_norm
        logger.debug(f"Fallback direction (unnormalized): {avg_dir}, norm: {avg_dir_norm:.6f}")
    else:
        direction = weighted_dir / dir_norm

    logger.debug(f"Final direction: {direction}, norm: {torch.linalg.norm(direction):.6f}")

    # Find pixel location of direction
    pixel_coords = cartesian_to_pixel(direction, H, W)

    # Compute average radiance
    avg_radiance = power_rgb / torch.clamp(total_solid_angle, min=1e-12)
    
    logger.debug(f"Pixel coordinates: {pixel_coords}, Average radiance: {avg_radiance}")
    
    result = {
        'power_rgb': power_rgb,
        'direction': direction,
        'pixel_coords': pixel_coords,
        'energy': energy,
        'solid_angle': total_solid_angle,
        'avg_radiance_rgb': avg_radiance,
        'rect': (int(y0), int(y1), int(x0), int(x1))
    }
    
    logger.debug(f"Computed stats for region ({y0}:{y1}, {x0}:{x1}): "
                f"energy={energy:.6f}, solid_angle={total_solid_angle:.6f}, "
                f"power_sum={torch.sum(power_rgb):.6f}")
    
    return result


def median_cut_sampling(hdri: torch.Tensor, n_samples: int, device: Optional[torch.device] = None) -> List[SampleGPU]:
    """
    Perform median cut sampling on a lat-long HDRI environment map.
    
    Args:
        hdri: HDRI tensor with shape (H, W, 3) in linear RGB
        n_samples: Number of light samples to generate
        device: Optional device to run computation on (defaults to hdri.device)
    
    Returns:
        List of SampleGPU objects, each containing:
        - direction: Unit direction vector as torch.Tensor [x, y, z]
        - power_rgb: Total RGB power as torch.Tensor [r, g, b] over the region
        - solid_angle: Total solid angle as torch.Tensor
        - avg_radiance_rgb: Average radiance as torch.Tensor [r, g, b] (power / solid_angle)
        - rect: Pixel bounds (y0, y1, x0, x1) of the region
        - pixel_coords: Pixel coordinates as torch.Tensor [u, v]
        - energy: Total luminance energy as torch.Tensor
    """
    if device is None:
        device = hdri.device
    
    # Ensure hdri is on the correct device
    hdri = hdri.to(device)
    H, W, _ = hdri.shape
    
    # Compute luminance and solid angles
    lum = luminance(hdri)  # (H, W)
    sa_map = pixel_solid_angles(H, W, device=device)  # (H, 1)
    sa_full = sa_map.expand(H, W)  # (H, W)
    
    # Create energy map for splitting decisions
    energy_map = lum * sa_full  # (H, W)
    
    # Initialize heap with the full image region
    initial_energy = torch.sum(energy_map).item()
    heap: List[Region] = []
    heapq.heappush(heap, Region(
        sort_key=-initial_energy,
        y0=0, y1=H, x0=0, x1=W,
        energy=initial_energy
    ))
    
    regions: List[Tuple[int, int, int, int]] = []
    
    # Greedily split the most energetic region until we have n_samples regions
    while len(regions) + len(heap) < n_samples:
        if not heap:
            break  # Nothing left to split
        
        region = heapq.heappop(heap)
        child_a, child_b = split_region_by_median(energy_map, region.y0, region.y1, region.x0, region.x1)
        
        # Add children to heap with their energies (avoid duplicates when split fails)
        for (yy0, yy1, xx0, xx1) in (child_a, child_b):
            # Validate child region
            if yy1 > yy0 and xx1 > xx0:  # Must have positive width and height
                # Check if this is the same as the original region (happens when split fails)
                if (yy0, yy1, xx0, xx1) == (region.y0, region.y1, region.x0, region.x1):
                    continue
                
                child_energy = torch.sum(energy_map[yy0:yy1, xx0:xx1]).item()
                heapq.heappush(heap, Region(
                    sort_key=-child_energy,
                    y0=yy0, y1=yy1, x0=xx0, x1=xx1,
                    energy=child_energy
                ))
            else:
                pass  # Skip invalid child regions silently
    
    # Collect remaining regions from heap
    while heap and len(regions) < n_samples:
        region = heapq.heappop(heap)
        regions.append((region.y0, region.y1, region.x0, region.x1))
    
    # Generate light samples from regions
    samples = []
    for (y0, y1, x0, x1) in regions:
        stats = compute_region_stats(hdri, y0, y1, x0, x1)
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


def median_cut_sampling_to_cpu(hdri: torch.Tensor, n_samples: int, device: Optional[torch.device] = None) -> List[SampleCPU]:
    """
    Convenience function that performs median cut sampling and converts results to CPU/numpy.
    
    Args:
        hdri: HDRI tensor with shape (H, W, 3) in linear RGB
        n_samples: Number of light samples to generate
        device: Optional device to run computation on (defaults to hdri.device)
    
    Returns:
        List of SampleCPU objects with CPU tensors converted to lists:
        - direction: Unit direction vector [x, y, z] as list
        - power_rgb: Total RGB power [r, g, b] as list
        - solid_angle: Total solid angle as float
        - avg_radiance_rgb: Average radiance [r, g, b] as list
        - rect: Pixel bounds (y0, y1, x0, x1) of the region
        - pixel_coords: Pixel coordinates [u, v] as list
        - energy: Total luminance energy as float
    """
    samples = median_cut_sampling(hdri, n_samples, device)
    
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

def visualize_samples(hdri: torch.Tensor, samples: List[Union[SampleGPU, SampleCPU]]) -> torch.Tensor:
    """
    Visualize the samples on the HDRI with blue boxes around regions and red circles around pixel coordinates.
    
    Args:
        hdri: Input HDRI tensor (H, W, 3)
        samples: List of SampleGPU or SampleCPU objects
    
    Returns:
        Visualization tensor with blue region boundaries and red pixel markers
    """
    H, W, _ = hdri.shape
    vis_hdri = hdri.clone()
    
    # Define colors
    blue_color = torch.tensor([0.0, 0.0, 1.0], device=hdri.device, dtype=hdri.dtype)
    red_color = torch.tensor([1.0, 0.0, 0.0], device=hdri.device, dtype=hdri.dtype)
    
    for sample in samples:
        # Get sample data from SampleGPU or SampleCPU objects
        pixel_coords = sample.pixel_coords
        rect = sample.rect
        y0, y1, x0, x1 = rect
        
        # Ensure coordinates are within bounds
        y0, y1 = max(0, y0), min(H, y1)
        x0, x1 = max(0, x0), min(W, x1)
        
        # Handle different pixel_coords types (tensor vs list)
        if isinstance(pixel_coords, torch.Tensor):
            pixel_x = max(0, min(W-1, int(pixel_coords[0].item())))
            pixel_y = max(0, min(H-1, int(pixel_coords[1].item())))
        else:
            # pixel_coords is a list (SampleCPU)
            pixel_x = max(0, min(W-1, int(pixel_coords[0])))
            pixel_y = max(0, min(H-1, int(pixel_coords[1])))
        
        # Draw blue box around the region
        # Top and bottom borders
        if y0 < H:
            vis_hdri[y0, x0:x1, :] = blue_color
        if y1-1 >= 0 and y1-1 < H:
            vis_hdri[y1-1, x0:x1, :] = blue_color
        
        # Left and right borders
        if x0 < W:
            vis_hdri[y0:y1, x0, :] = blue_color
        if x1-1 >= 0 and x1-1 < W:
            vis_hdri[y0:y1, x1-1, :] = blue_color
        
        # Draw red circle (border) around the pixel coordinate
        # We'll draw a small cross pattern and circle outline
        center_y, center_x = pixel_y, pixel_x
        
        # Store the original center pixel color
        original_center_color = vis_hdri[center_y, center_x, :].clone()
        
        # Draw circle with radius 2 (just the border)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y_coord = center_y + dy
                x_coord = center_x + dx
                
                # Check bounds
                if 0 <= y_coord < H and 0 <= x_coord < W:
                    # Calculate distance from center
                    dist_sq = dy*dy + dx*dx
                    
                    # Draw circle border (distance 1.5 to 2.5)
                    if 2 <= dist_sq <= 6:  # Approximate circle border
                        vis_hdri[y_coord, x_coord, :] = red_color
        
        # Restore the center pixel with original color
        vis_hdri[center_y, center_x, :] = original_center_color
    
    return vis_hdri

