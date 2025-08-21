"""
Density Estimation based on Expand Map (Sec. 2.3)
https://www.banterle.com/francesco/publications/download/graphite2006.pdf


This module implements the density estimation based on Expand Map.
It uses PyTorch throughout for GPU acceleration and consistency with the analysis utilities.
"""
import torch
import numpy as np
import cv2
from scipy.spatial import cKDTree
from typing import List, Union
from ..datatypes import SampleGPU, SampleCPU

def expand_map_exact(hdri: torch.Tensor, samples: List[Union[SampleGPU, SampleCPU]],
                     rt: int = 16, gamma: float = 0.918, beta: float = 1.953,
                     min_count: int = 4, normalize: bool = True, tile: int = 128) -> torch.Tensor:
    """
    Faithful implementation of Sec. 2.3 with per-pixel r_max (Eq. 16-17).
    This can be slow on large images; use tile processing + SciPy KD-tree.

    Args:
        hdri: torch.Tensor, HDRI image
        samples: List of SampleGPU or SampleCPU objects
        rt: radius of influence in pixels (paper used 16)
        gamma, beta: Gaussian kernel parameters (paper: 0.918, 1.953)
        min_count:  minimum #seeds in neighborhood A_x to accept; else weight->0 (paper suggests 4-6)
        normalize:  scale result to [0,1] by dividing by max
        tile:       tile size for chunked processing

    Returns:
        density_map: HxW float32 Expand Map in [0,1] if normalize=True
    """

    # Initialize density map
    H, W, _ = hdri.shape
    density_map = torch.zeros((H, W), dtype=torch.float32, device=hdri.device)

    # Extract sample data - handle both SampleGPU and SampleCPU
    samples_xy = []
    psi = []
    
    for sample in samples:
        # Handle pixel_coords
        if isinstance(sample.pixel_coords, torch.Tensor):
            # SampleGPU case - convert tensor to numpy
            coords = sample.pixel_coords.cpu().numpy()
        else:
            # SampleCPU case - already a list
            coords = np.array(sample.pixel_coords)
        samples_xy.append(coords)
        
        # Handle energy
        if isinstance(sample.energy, torch.Tensor):
            # SampleGPU case - extract scalar value
            energy = sample.energy.cpu().item()
        else:
            # SampleCPU case - already a float
            energy = sample.energy
        psi.append(energy)
    
    samples_xy = np.array(samples_xy)
    psi = np.array(psi) # TODO try luminance, or energy, or something else

    tree = cKDTree(samples_xy)
    inv_one_minus_exp_beta = 1.0 / (1.0 - np.exp(-beta) + 1e-12)

    # Process in tiles to limit memory
    for y0 in range(0, H, tile):
        y1 = min(y0 + tile, H)
        for x0 in range(0, W, tile):
            x1 = min(x0 + tile, W)
            # Query all pixels in the tile at once (KD-tree supports vectorized query_ball_point)
            xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            qpts = np.stack([xs.ravel().astype(np.float32), ys.ravel().astype(np.float32)], axis=1)
            neigh = tree.query_ball_point(qpts, r=rt)  # list of arrays (variable length)

            tile_vals = np.zeros(qpts.shape[0], dtype=np.float32)
            for idx, nn in enumerate(neigh):
                if len(nn) < min_count: # TODO: try different min_count
                    continue
                qx, qy = qpts[idx]
                # distances to neighbors
                xy = samples_xy[nn]  # Kx2
                dx = xy[:, 0] - qx
                dy = xy[:, 1] - qy
                d2 = dx*dx + dy*dy
                rmax2 = np.max(d2)
                if rmax2 <= 1e-12:  # degenerate: all neighbors on the pixel
                    # All weights collapse to gamma * [1 - (1 - e^0)/(1 - e^-beta)] = gamma
                    w = gamma
                    tile_vals[idx] = w * np.sum(psi[nn])
                    continue

                # Eq. 16: w_g^p = gamma * [1 - (1 - exp(-beta * d^2 / (2*rmax^2))) / (1 - exp(-beta))]
                a = -beta * d2 / (2.0 * rmax2 + 1e-12)
                wgp = gamma * (1.0 - (1.0 - np.exp(a)) * inv_one_minus_exp_beta)

                # Eq. 17: Lambda = sum Psi_p * w_g^p
                tile_vals[idx] = float(np.dot(psi[nn], wgp))

            density_map[y0:y1, x0:x1] = torch.from_numpy(tile_vals.reshape((y1 - y0), (x1 - x0))).to(hdri.device)

    if normalize and torch.max(density_map) > 0:
        density_map = density_map / torch.max(density_map)
    return density_map


def expand_map_fast(hdri: torch.Tensor, samples: List[Union[SampleGPU, SampleCPU]],
                    rt: int = 16, gamma: float = 0.918, beta: float = 1.953,
                    min_count: int = 4, normalize: bool = True, 
                    border: int = cv2.BORDER_REPLICATE) -> torch.Tensor:
    """
    Much faster approximation: assume r_max == rt for all pixels.
    Then Eq. 16 weights depend only on distance d, so the map is
    convolution of an impulse image (Psi at seeds) with a fixed kernel.

    Also computes a neighbor-count map with a binary disc kernel and
    zeros out pixels with count < min_count, as suggested in the paper.

    Args:
        hdri: torch.Tensor, HDRI image
        samples: List of SampleGPU or SampleCPU objects
        rt: radius of influence in pixels (paper used 16)
        gamma, beta: Gaussian kernel parameters (paper: 0.918, 1.953)
        min_count: minimum #seeds in neighborhood A_x to accept; else weight->0
        normalize: scale result to [0,1] by dividing by max
        border: OpenCV border mode for filter2D

    Returns:
        density_map: HxW float32 Expand Map in [0,1] if normalize=True
    """
    H, W, _ = hdri.shape
    
    if len(samples) == 0:
        return torch.zeros((H, W), dtype=torch.float32, device=hdri.device)

    # Extract sample data - handle both SampleGPU and SampleCPU
    samples_xy = []
    psi = []
    
    for sample in samples:
        # Handle pixel_coords
        if isinstance(sample.pixel_coords, torch.Tensor):
            # SampleGPU case - convert tensor to numpy
            coords = sample.pixel_coords.cpu().numpy()
        else:
            # SampleCPU case - already a list
            coords = np.array(sample.pixel_coords)
        samples_xy.append(coords)
        
        # Handle energy
        if isinstance(sample.energy, torch.Tensor):
            # SampleGPU case - extract scalar value
            energy = sample.energy.cpu().item()
        else:
            # SampleCPU case - already a float
            energy = sample.energy
        psi.append(energy)
    
    samples_xy = np.array(samples_xy)
    psi = np.array(psi)

    # Impulse image with Psi at seeds
    imp = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    
    xs = np.clip(np.round(samples_xy[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(samples_xy[:, 1]).astype(int), 0, H - 1)
    
    for x, y, val in zip(xs, ys, psi):
        imp[y, x] += float(val)
        cnt[y, x] += 1.0

    # Build radial kernel from Eq. 16 with r_max = rt
    diam = 2 * rt + 1
    yy, xx = np.mgrid[-rt:rt+1, -rt:rt+1]
    d2 = (xx * xx + yy * yy).astype(np.float32)
    mask = (d2 <= (rt * rt)).astype(np.float32)

    a = -beta * d2 / (2.0 * (rt * rt) + 1e-12)
    kernel = gamma * (1.0 - (1.0 - np.exp(a)) / (1.0 - np.exp(-beta) + 1e-12))
    kernel *= mask
    # No need to renormalize: Eq. 17 uses the normalized kernel form directly

    # Binary disc kernel for neighbor counts
    disc = mask

    # Convolve
    dens = cv2.filter2D(imp, -1, kernel, borderType=border)
    nmap = cv2.filter2D(cnt, -1, disc, borderType=border)

    dens[nmap < float(min_count)] = 0.0

    if normalize and np.max(dens) > 0:
        dens = dens / np.max(dens)
    
    # Convert back to PyTorch tensor and move to the same device as input
    density_map = torch.from_numpy(dens).to(hdri.device)
    return density_map


