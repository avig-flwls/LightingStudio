import torch
from typing import Optional
from src.LightingStudio.analysis.utils.transforms import pixel_solid_angles, luminance
from src.LightingStudio.analysis.report.datatypes import NaiveMetricsGPU, NaiveMetricsCPU


def naive_metrics(env_map: torch.Tensor, device: Optional[torch.device] = None) -> NaiveMetricsGPU:
    """
    Calculate the intensity of the env map using the naive method.

    Args:
        env_map: HDRI tensor with shape (H, W, 3) in linear RGB
        device: Optional device to run computation on (defaults to env_map.device)

    Returns:
        NaiveMetricsGPU object containing:
        - global_color: Global Color (rgb) as torch.Tensor
        - global_intensity: Global Intensity as torch.Tensor
    """
    if device is None:
        device = env_map.device
    
    # Ensure env_map is on the correct device
    env_map = env_map.to(device)
    H, W, _ = env_map.shape

    # Get solid angles for the entire env map
    pixel_area, sin_theta = pixel_solid_angles(H, W, device=device)
    sa_map = sin_theta 
    # sa_map = pixel_area * sin_theta # (H, 1)

    # Get RGB and luminance for the region
    lum = luminance(env_map) # (H, W)

    global_color = torch.mean(env_map * sa_map[..., None], dim=(0, 1)) # (3)
    global_color = 255 * (global_color / torch.linalg.norm(global_color))

    global_intensity = torch.mean(lum * sa_map) # (1)

    return NaiveMetricsGPU(
        global_color=global_color,
        global_intensity=global_intensity
    )


def naive_metrics_cpu(env_map: torch.Tensor, device: Optional[torch.device] = None) -> NaiveMetricsCPU:
    """
    Convenience function that performs naive metrics calculation and converts results to CPU/numpy.
    
    Args:
        env_map: HDRI tensor with shape (H, W, 3) in linear RGB
        device: Optional device to run computation on (defaults to env_map.device)
    
    Returns:
        NaiveMetricsCPU object with CPU tensors converted to lists:
        - global_color: Global Color (rgb) as list
        - global_intensity: Global Intensity as float
    """
    metrics = naive_metrics(env_map, device)
    
    # Convert tensors to CPU and then to lists/floats
    cpu_metrics = NaiveMetricsCPU(
        global_color=metrics.global_color.cpu().numpy().tolist(),
        global_intensity=metrics.global_intensity.cpu().item()
    )
    
    return cpu_metrics

