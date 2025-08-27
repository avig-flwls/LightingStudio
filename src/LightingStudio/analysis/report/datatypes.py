"""
Common data types for lighting analysis.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Union, List
import torch


@dataclass(order=True)
class Region:
    """A rectangular region in the HDRI for median cut subdivision."""
    # The heap will maximize energy by using negative sort key
    sort_key: float
    y0: int
    y1: int
    x0: int
    x1: int
    energy: float


@dataclass
class SampleGPU:
    """A light sample generated from HDRI analysis with GPU tensors."""
    direction: torch.Tensor  # Unit direction vector [x, y, z]
    power_rgb: torch.Tensor  # Total RGB power [r, g, b] over the region
    solid_angle: torch.Tensor  # Total solid angle
    avg_radiance_rgb: torch.Tensor  # Average radiance [r, g, b] (power / solid_angle)
    rect: Tuple[int, int, int, int]  # Pixel bounds (y0, y1, x0, x1) of the region
    pixel_coords: torch.Tensor  # Pixel coordinates [u, v]
    energy: torch.Tensor  # Total luminance energy (∑ luminance * Δω)

@dataclass
class NaiveMetricsGPU:
    """A report for the lighting analysis."""
    global_color: torch.Tensor # Global Color (rgb)
    global_intensity: torch.Tensor # Global Intensity (1)

@dataclass
class NaiveMetricsCPU:
    """A report for the lighting analysis with CPU/serializable data."""
    global_color: List[float] # Global Color (rgb) as list
    global_intensity: float # Global Intensity as float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'global_color': self.global_color,
            'global_intensity': self.global_intensity
        }

@dataclass
class SPHMetrics:
    """A report for the lighting analysis."""
    dc_color: torch.Tensor # DC Term (3)
    sph_coeffs: torch.Tensor # Spherical Harmonic Coefficients (n_terms, 3) 

    # Dominant Direction (all normalized)
    dominant_direction: torch.Tensor # Dominant Direction (xyz)
    dominant_direction_rgb_color_difference: torch.Tensor # Dominant Direction (rgb) based on color difference
    dominant_direction_rgb_luminance: torch.Tensor # Dominant Direction (rgb) based on luminance

    # Dominant Direction in Pixel Space
    dominant_pixel: torch.Tensor # Dominant Pixel (uv)
    dominant_pixel_rgb_color_difference: torch.Tensor # Dominant Pixel (rgb) based on color difference 
    dominant_pixel_rgb_luminance: torch.Tensor # Dominant Pixel (rgb) based on luminance

    # Dominant Color
    dominant_color: torch.Tensor # Dominant Color (rgb) based on sph coeffs
    dominant_color_rgb_color_difference: torch.Tensor # Dominant Color (rgb) based on color difference 
    dominant_color_rgb_luminance: torch.Tensor # Dominant Color (rgb) based on luminance

    # Intensity
    area_intensity: torch.Tensor # Area Intensity (rgb)
    area_intensity_rgb_color_difference: torch.Tensor # Area Intensity (rgb) based on color difference
    area_intensity_rgb_luminance: torch.Tensor # Area Intensity (rgb) based on luminance

@dataclass
class SPHMetricsCPU:
    """A report for the lighting analysis with CPU/serializable data."""

    dc_color: List[float] # DC Term (3) as list
    sph_coeffs: List[List[float]] # Spherical Harmonic Coefficients (n_terms, 3) as nested list

    # Dominant Direction (all normalized)
    dominant_direction: List[float] # Dominant Direction (xyz) as list
    dominant_direction_rgb_color_difference: List[float] # Dominant Direction (rgb) based on color difference as list
    dominant_direction_rgb_luminance: List[float] # Dominant Direction (rgb) based on luminance as list

    # Dominant Direction in Pixel Space
    dominant_pixel: List[float] # Dominant Pixel (uv) as list
    dominant_pixel_rgb_color_difference: List[float] # Dominant Pixel (rgb) based on color difference as list
    dominant_pixel_rgb_luminance: List[float] # Dominant Pixel (rgb) based on luminance as list

    # Dominant Color
    dominant_color: List[float] # Dominant Color (rgb) based on sph coeffs as list
    dominant_color_rgb_color_difference: List[float] # Dominant Color (rgb) based on color difference as list
    dominant_color_rgb_luminance: List[float] # Dominant Color (rgb) based on luminance as list

    # Intensity
    area_intensity: List[float] # Area Intensity (rgb) as list
    area_intensity_rgb_color_difference: List[float] # Area Intensity (rgb) based on color difference as list
    area_intensity_rgb_luminance: List[float] # Area Intensity (rgb) based on luminance as list
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'dc_color': self.dc_color,
            'sph_coeffs': self.sph_coeffs,
            'dominant_direction': self.dominant_direction,
            'dominant_direction_rgb_color_difference': self.dominant_direction_rgb_color_difference,
            'dominant_direction_rgb_luminance': self.dominant_direction_rgb_luminance,
            'dominant_pixel': self.dominant_pixel,
            'dominant_pixel_rgb_color_difference': self.dominant_pixel_rgb_color_difference,
            'dominant_pixel_rgb_luminance': self.dominant_pixel_rgb_luminance,
            'dominant_color': self.dominant_color,
            'dominant_color_rgb_color_difference': self.dominant_color_rgb_color_difference,
            'dominant_color_rgb_luminance': self.dominant_color_rgb_luminance,
            'area_intensity': self.area_intensity,
            'area_intensity_rgb_color_difference': self.area_intensity_rgb_color_difference,
            'area_intensity_rgb_luminance': self.area_intensity_rgb_luminance
        }

@dataclass
class SampleCPU:
    """A light sample generated from HDRI analysis with CPU/serializable data."""
    direction: List[float]  # Unit direction vector [x, y, z] as list
    power_rgb: List[float]  # Total RGB power [r, g, b] as list
    solid_angle: float  # Total solid angle as float
    avg_radiance_rgb: List[float]  # Average radiance [r, g, b] as list
    rect: Tuple[int, int, int, int]  # Pixel bounds (y0, y1, x0, x1) of the region
    pixel_coords: List[float]  # Pixel coordinates [u, v] as list
    energy: float  # Total luminance energy as float (∑ luminance * Δω)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'direction': self.direction,
            'power_rgb': self.power_rgb,
            'solid_angle': self.solid_angle,
            'avg_radiance_rgb': self.avg_radiance_rgb,
            'rect': self.rect,
            'pixel_coords': self.pixel_coords,
            'energy': self.energy
        }
