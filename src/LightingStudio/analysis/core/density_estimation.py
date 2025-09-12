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
from typing import List, Union, Tuple
from src.LightingStudio.analysis.report.datatypes import SampleGPU, SampleCPU


def merge_wraparound_components(labels: np.ndarray, num_labels: int, binary_img: np.ndarray) -> Tuple[np.ndarray, int]:
    """Merge components that wrap around the equirectangular image edges."""
    height, width = labels.shape
    
    # Find components that touch the left and right edges
    left_edge_labels = set(labels[:, 0]) - {0}  # Exclude background
    right_edge_labels = set(labels[:, -1]) - {0}
    
    # Create a mapping for label merging
    label_map = {i: i for i in range(num_labels)}
    
    # For each component on the left edge, check if it connects to the right edge
    for left_label in left_edge_labels:
        # Get pixels of this component on the left edge
        left_pixels = np.where((labels[:, 0] == left_label) & (binary_img[:, 0] > 0))[0]
        
        if len(left_pixels) > 0:
            # Check corresponding pixels on the right edge
            for y in left_pixels:
                # Check if there's a component on the right edge at the same height
                if binary_img[y, -1] > 0 and labels[y, -1] != 0:
                    right_label = labels[y, -1]
                    if right_label != left_label and right_label in right_edge_labels:
                        # Merge these components - map the higher label to the lower one
                        min_label = min(left_label, right_label)
                        max_label = max(left_label, right_label)
                        label_map[max_label] = min_label
    
    # Apply transitive closure to handle chains of merges
    for i in range(num_labels):
        root = i
        while label_map[root] != root:
            root = label_map[root]
        label_map[i] = root
    
    # Create new labels with merged components
    merged_labels = np.zeros_like(labels)
    new_label_count = 1
    new_label_map = {0: 0}  # Background stays 0
    
    for old_label in range(1, num_labels):
        merged_label = label_map[old_label]
        if merged_label not in new_label_map:
            new_label_map[merged_label] = new_label_count
            new_label_count += 1
        merged_labels[labels == old_label] = new_label_map[merged_label]
    
    return merged_labels, new_label_count - 1


def filter_components_by_size(labels: np.ndarray, num_labels: int, min_size: int, binary_img: np.ndarray) -> Tuple[np.ndarray, int]:
    """Filter out components smaller than min_size pixels."""
    # Count actual pixels for each label in the merged image
    label_sizes = {}
    for label in range(1, num_labels + 1):
        label_sizes[label] = np.sum(labels == label)
    
    # Create filtered labels image
    filtered_labels = np.zeros_like(labels)
    new_label = 1
    
    # Skip background (label 0)
    for label in range(1, num_labels + 1):
        # Check if component is large enough
        if label in label_sizes and label_sizes[label] >= min_size:
            filtered_labels[labels == label] = new_label
            new_label += 1
    
    return filtered_labels, new_label - 1


def create_labeled_image(labels: np.ndarray) -> np.ndarray:
    """Create a color-coded visualization of labeled components."""
    if np.max(labels) > 0:
        label_hue = np.uint8(179*labels/np.max(labels))
    else:
        label_hue = np.zeros_like(labels, dtype=np.uint8)
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    return labeled_img


def analyze_connected_components(density_map: np.ndarray, threshold: int = 127, min_component_size: int = 10) -> Tuple[np.ndarray, int]:
    """
    Analyze connected components in the density map, handling equirectangular wraparound.
    
    Args:
        density_map: 2D numpy array of the density map
        threshold: Threshold for binary conversion (default: 127)
        min_component_size: Minimum size for components in pixels (default: 10)
    
    Returns:
        component_visualization: Color-coded visualization of filtered components
        num_components: Number of components after filtering
    """
    # Convert to uint8 if needed
    if density_map.dtype != np.uint8:
        # Apply gamma correction to match the saved PNG (which uses gamma=2.2)
        density_map_gamma = np.power(density_map, 1.0/2.2)
        density_map_uint8 = (density_map_gamma * 255).astype(np.uint8)
    else:
        density_map_uint8 = density_map
    
    # Convert to binary
    _, binary_img = cv2.threshold(density_map_uint8, threshold, 255, cv2.THRESH_BINARY)
    
    # Get connected components
    num_labels_original, labels_original = cv2.connectedComponents(binary_img)
    
    # Merge components that wrap around the image edges
    labels_merged, num_labels_merged = merge_wraparound_components(
        labels_original, num_labels_original, binary_img
    )
    
    # Filter components by size
    labels_filtered, num_components = filter_components_by_size(
        labels_merged, num_labels_merged, min_component_size, binary_img
    )
    
    # Create color visualization
    component_visualization = create_labeled_image(labels_filtered)
    
    return component_visualization, num_components

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
                    border: int = cv2.BORDER_REPLICATE,
                    component_threshold: int = 127,
                    min_component_size: int = 100) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, int]]:
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
        analyze_components: whether to analyze connected components (default: True)
        component_threshold: threshold for binary conversion (default: 127)
        min_component_size: minimum size for components in pixels (default: 10)

    Returns:
        If analyze_components is False:
            density_map: HxW float32 Expand Map in [0,1] if normalize=True
        If analyze_components is True:
            tuple of (density_map, component_visualization, num_components):
                - density_map: HxW float32 Expand Map in [0,1] if normalize=True
                - component_visualization: HxW BGR uint8 color-coded components
                - num_components: int, number of components after filtering
    """
    H, W, _ = hdri.shape
    
    if len(samples) == 0:
        empty_density = torch.zeros((H, W), dtype=torch.float32, device=hdri.device)
        empty_components = torch.zeros((H, W, 3), dtype=torch.uint8, device=hdri.device)
        return empty_density, empty_components, 0

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

    # Analyze connected components
    component_vis, num_components = analyze_connected_components(
        dens, threshold=component_threshold, min_component_size=min_component_size
    )
    # Convert component visualization to torch tensor
    component_vis_tensor = torch.from_numpy(component_vis).to(hdri.device)
    return density_map, component_vis_tensor, num_components


