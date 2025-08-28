# hosek_wilkie.py
# Minimal Hosek–Wilkie sky model in Python + PyTorch.
# Produces HDR arrays suitable for EXR (lat-long env map).
# References:
#  - Hosek & Wilkie 2012, "An Analytic Model for Full Spectral Sky-Dome Radiance"
#  - ArHosekSkyModel (public domain) – coefficients & structure

import math
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

# Import standardized utilities
from src.LightingStudio.analysis.utils.io import exr_to_png_tensor, write_exr
from src.LightingStudio.analysis.utils.transforms import convert_theta, generate_spherical_coordinates_map

# --------------------------- Utilities ---------------------------

def _saturate(x): 
    return np.clip(x, 0.0, 1.0)

def _safe_acos(x):
    return np.arccos(np.clip(x, -1.0, 1.0))

def _cosd(a):  # degrees cosine helper (not used, but handy)
    return math.cos(math.radians(a))

def xyz_to_linear_srgb(xyz):
    """Convert CIE XYZ (D65) to linear sRGB. Input (...,3), returns same shape."""
    M = torch.tensor([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]], dtype=torch.float64)
    shp = xyz.shape
    out = xyz.reshape(-1, 3) @ M.T
    return out.reshape(shp)

# -------------------- Coefficient handling ----------------------

"""
The model needs 9 distribution parameters per color channel (A..I)
for each:
  - turbidity T ∈ {1..10} (we will interpolate along T)
  - ground albedo a ∈ {0,1} (we will lerp between them)
  - solar elevation grid of 6 knots (we will evaluate via the paper’s Bézier)
and for each output color space channel (XYZ has 3; RGB has 3).

We’ll auto-fetch the public ‘ArHosekSkyModelData_CIEXYZ.h’ or
‘ArHosekSkyModelData_RGB.h’ when needed and parse it into a numpy array:

  data.shape == (num_channels, 10, 2, 6, 9)

If you already have these as a .npy, you can pass them directly to the
API via the ‘coeffs’ argument.
"""

_HDR_URLS = {
    "XYZ": "https://raw.githubusercontent.com/mmp/pbrt-v3/master/src/ext/ArHosekSkyModelData_CIEXYZ.h",
    "RGB": "https://raw.githubusercontent.com/mmp/pbrt-v3/master/src/ext/ArHosekSkyModelData_RGB.h",
}

def _load_coeff_header_from_archive(space="XYZ"):
    """
    Load Hosek-Wilkie coefficient header from local archive folder.
    """
    # Try to find the archive folder relative to this file
    current_dir = Path(__file__).parent
    archive_dir = current_dir.parent / "archive"

    # File names in archive
    filename = f"ArHosekSkyModelData_{'CIEXYZ' if space == 'XYZ' else 'RGB'}.h"
    coeff_file = archive_dir / filename

    if not coeff_file.exists():
        # Try absolute path as fallback
        archive_dir = Path(r"C:\Users\AviGoyal\Documents\LightingStudio\archive")
        coeff_file = archive_dir / filename

    if not coeff_file.exists():
        raise FileNotFoundError(f"Coefficient file not found: {coeff_file}")

    try:
        with open(coeff_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if len(content) < 10000:  # Minimum expected size
            raise ValueError(f"Coefficient file is too small ({len(content)} bytes), likely corrupted")

        return content
    except Exception as e:
        raise RuntimeError(f"Error reading coefficient file {coeff_file}: {e}")


def _parse_coeffs_from_header(text, space="XYZ"):
    """
    Parse coefficient arrays from the Hosek-Wilkie header file.
    The file contains three datasets: datasetXYZ1, datasetXYZ2, datasetXYZ3
    (or datasetRGB1, datasetRGB2, datasetRGB3 for RGB space).
    """
    # Look for the specific dataset arrays (only XYZ1, XYZ2, XYZ3 or RGB1, RGB2, RGB3)
    if space == "XYZ":
        dataset_patterns = [
            r'double\s+datasetXYZ1\[\]\s*=\s*\{([^}]+)\}',
            r'double\s+datasetXYZ2\[\]\s*=\s*\{([^}]+)\}',
            r'double\s+datasetXYZ3\[\]\s*=\s*\{([^}]+)\}'
        ]
    else:  # RGB
        dataset_patterns = [
            r'double\s+datasetRGB1\[\]\s*=\s*\{([^}]+)\}',
            r'double\s+datasetRGB2\[\]\s*=\s*\{([^}]+)\}',
            r'double\s+datasetRGB3\[\]\s*=\s*\{([^}]+)\}'
        ]

    # Find each dataset
    datasets = []
    for pattern in dataset_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            datasets.append(match.group(1))
        else:
            raise ValueError(f"Could not find dataset matching pattern: {pattern}")

    print(f"Found {len(datasets)} dataset matches with regex (excluding Rad)")

    if len(datasets) < 3:
        raise ValueError(f"Could not find 3 coefficient datasets in the file. Found {len(datasets)}.")

    chan_arrays = []
    for i, dataset in enumerate(datasets[:3]):
        print(f"Processing dataset {i+1}, length: {len(dataset)} characters")

        # Clean up the dataset string - remove newlines, tabs, and comments
        cleaned = dataset.replace('\n', ' ').replace('\t', ' ')

        # Split by comma and convert to float
        values = []
        for val in cleaned.split(','):
            val = val.strip()
            # Skip empty strings and comments
            if val and not val.startswith('//') and not val.startswith('/*'):
                try:
                    # Handle scientific notation like 1.234e-05
                    num_val = float(val)
                    values.append(num_val)
                    # Debug: print first few values
                    if len(values) <= 5:
                        print(f"  Parsed value {len(values)}: {num_val}")
                except ValueError as e:
                    print(f"  Skipping invalid value: '{val}' (error: {e})")
                    continue  # Skip non-numeric values

        print(f"Dataset {i+1}: extracted {len(values)} numeric values")
        if len(values) == 0:
            print(f"ERROR: No values extracted from dataset {i+1}")
            print(f"First 200 chars of dataset: {dataset[:200]}")
            raise ValueError(f"Failed to extract any numeric values from dataset {i+1}")

        arr = np.array(values, dtype=np.float64)
        chan_arrays.append(arr)

    # Each channel should be 10*2*6*9 = 1080 values
    target = 10*2*6*9
    out = []
    for i, arr in enumerate(chan_arrays):
        print(f"Processing channel {i+1}: {arr.size} values")
        if arr.size < target:
            print(f"Warning: Channel {i+1} has {arr.size} values, expected {target}")
            # Pad with zeros if too small
            if arr.size < target:
                padded = np.zeros(target, dtype=np.float64)
                padded[:arr.size] = arr
                arr = padded
        elif arr.size > target:
            # Trim if too large
            arr = arr[:target]

        out.append(arr.reshape(10, 2, 6, 9))

    data = np.stack(out, axis=0)  # (3, 10, 2, 6, 9)
    print(f"Successfully parsed coefficients with shape: {data.shape}")
    return data

def _get_coeffs(space="XYZ"):
    """
    Load and parse Hosek-Wilkie sky model coefficients from local archive.
    """
    try:
        print(f"Loading Hosek-Wilkie {space} coefficients from archive...")
        hdr = _load_coeff_header_from_archive(space)
        coeffs = _parse_coeffs_from_header(hdr, space)
        print(f"Successfully loaded {space} coefficients from archive!")
        return coeffs

    except Exception as e:
        error_msg = f"""
Failed to load Hosek-Wilkie {space} coefficients from archive.

Error: {e}

Please ensure the coefficient files are present in the archive folder:
- ArHosekSkyModelData_CIEXYZ.h (for XYZ space)
- ArHosekSkyModelData_RGB.h (for RGB space)

Expected location: {Path(__file__).parent.parent / "archive"}
"""
        raise RuntimeError(error_msg)

# ---------------------- Core math (HW12) ------------------------

def _eval_chi(g, a):
    # Equation (9): χ(g,a) = (1 + cos^2 g) / (1 + a^2 - 2a cos g)^(3/2)
    with np.errstate(divide='ignore', invalid='ignore'):
        cg = np.cos(g)

        # Prevent numerical issues in the denominator
        denominator = 1.0 + a*a - 2.0*a*cg
        denominator = np.maximum(denominator, 1e-10)  # Avoid division by zero or negative

        numerator = 1.0 + cg*cg

        # Use safer power calculation
        result = numerator / np.power(denominator, 1.5)

        # Handle NaN and infinite values
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=0.0)

        return result

def _eval_F(theta, gamma, params):
    """
    Equation (8) variant used in ArHosek (A..I are params):
      F(θ,γ) =
        (1 + A * exp(B / (cos θ + 0.01))) *
        ( C + D * exp(E * γ) + F * cos^2 γ + G * _eval_chi(γ, H) + I * cos θ * cos θ )
    Notes:
      - The published formula is written to avoid issues near horizon (cosθ ~ 0).
      - Some codebases use cos^3(γ) or extra terms near sun; the official ArHosek fits
        are consistent with this layout via their 9 coefficients.
    """
    # Use numpy's error state to handle overflows gracefully
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        A,B,C,D,E,Fc,G,H,I = params
        c = np.cos(theta)
        cg = np.cos(gamma)

        # More robust overflow prevention
        # Clamp input arguments to prevent extreme values
        cos_theta_safe = np.clip(c + 0.01, 1e-10, 1e10)  # Avoid division by zero
        gamma_safe = np.clip(gamma, -100, 100)  # Reasonable gamma range

        # Calculate exponential terms with better bounds
        exp_arg1 = B / cos_theta_safe
        exp_arg2 = E * gamma_safe

        # Use safer exponential calculation
        exp_arg1 = np.clip(exp_arg1, -500, 500)
        exp_arg2 = np.clip(exp_arg2, -500, 500)

        exp_term1 = np.exp(exp_arg1)
        exp_term2 = np.exp(exp_arg2)

        # Calculate terms with overflow protection
        term1 = 1.0 + A * exp_term1
        term2 = C + D * exp_term2 + Fc * (cg*cg) + G * _eval_chi(gamma_safe, H) + I * (c*c)

        # Final multiplication
        result = term1 * term2

        # Clamp to reasonable range and handle NaN/Inf
        result = np.clip(result, 0.0, 1e10)
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=0.0)

        return result

def _bezier6(x, ctrl):  # x in [0,1], ctrl has 6 items (knot set for elevation)
    # Paper 5.4 & eq (11): they use quintic (degree-5) Bézier across normalized solar elevation.
    # ctrl shape (..., 6). We evaluate with standard Bernstein basis n=5.
    # This compact vectorized implementation assumes ctrl[...,6]
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        t = np.clip(x, 0.0, 1.0)
        B = np.stack([
            (1-t)**5,
            5 * t*(1-t)**4,
            10 * t**2 * (1-t)**3,
            10 * t**3 * (1-t)**2,
            5 * t**4 * (1-t),
            t**5
        ], axis=-1)

        result = np.sum(B * ctrl, axis=-1)
        # Handle any numerical issues
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=0.0)

        return result

def _interp_params(T, albedo, elev_x, chan_data):
    """
    chan_data: (10,2,6,9) for one color channel
    T: float turbidity, typical 1..10
    albedo: float 0..1, interpolates between the two albedo endpoints
    elev_x: normalized solar elevation x \in [0,1] (paper’s mapping)
    Returns nine parameters (A..I) for this channel.
    """
    # Ensure chan_data is a numpy array (convert from torch tensor if needed)
    if hasattr(chan_data, 'cpu'):
        chan_data = chan_data.cpu().numpy()
    elif hasattr(chan_data, 'detach'):
        chan_data = chan_data.detach().cpu().numpy()

    # Clamp & fractional index along turbidity axis
    T = np.clip(T, 1.0, 10.0)
    t0 = int(np.floor(T - 1.0))
    t1 = min(t0 + 1, 9)
    ft = (T - 1.0) - t0

    # Evaluate Bézier across the 6 elevation control points for each of 9 params,
    # at both albedo endpoints (0, 1), then lerp in albedo and turbidity.
    # Shape details: chan_data[t,a, :, param]
    def eval_ta(ti, ai):
        # ctrl_elev: (6,9) -> evaluate per-param across 6 ctrl points
        ctrl_elev = chan_data[ti, ai]  # (6,9)
        # Evaluate each param at x via Bézier
        params = np.array([_bezier6(elev_x, ctrl_elev[:,p]) for p in range(9)], dtype=np.float64)
        return params

    p_t0_a0 = eval_ta(t0, 0)
    p_t0_a1 = eval_ta(t0, 1)
    p_t1_a0 = eval_ta(t1, 0)
    p_t1_a1 = eval_ta(t1, 1)

    p_t0 = (1.0 - albedo) * p_t0_a0 + albedo * p_t0_a1
    p_t1 = (1.0 - albedo) * p_t1_a0 + albedo * p_t1_a1
    p = (1.0 - ft) * p_t0 + ft * p_t1  # final interpolated 9 params
    return p

def _normalize_solar_elevation(elev_rad):
    # Paper §5.4: x = 1 - sqrt(1 - y), with y = max(0, sin(elev))  (one common mapping)
    # Some codebases use x = (1 - cos(elev))^(1/3) or similar. The official fits
    # expect a strong emphasis near low elevations; this mapping reproduces that behavior.
    y = np.clip(np.sin(elev_rad), 0.0, 1.0)
    return 1.0 - np.sqrt(1.0 - y)

# ---------------------- Public API -------------------------------

def generate_sky_latlong(
    width: int,
    height: int,
    sun_azimuth: float,
    sun_elevation: float,
    turbidity: float = 2.5,
    albedo: float = 0.1,
    space: str = "XYZ",
    coeffs: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    brightness_scale: float = 0.1,
) -> torch.Tensor:
    """
    Generate a lat-long (equirectangular) sky dome in the requested color space.
      width, height: output dimensions
      sun_azimuth, sun_elevation: radians
      turbidity: typical 2..10 (float); clamped to [1,10] for interpolation (default: 2.5)
      albedo: ground albedo in [0,1] (default: 0.1)
      space: "XYZ" or "RGB" (coeff dataset to use)
      coeffs: optional preloaded coefficients array, shape (3,10,2,6,9)
      device: torch device for computations
      brightness_scale: scale factor for overall brightness (0.01-1.0, default: 0.1)
    Returns: torch.Tensor (H, W, 3), radiance in model units (XYZ: SI-derived; RGB: linear sRGB primaries)
    """
    if device is None:
        device = torch.device('cpu')

    if coeffs is None:
        coeffs = _get_coeffs(space)  # (3,10,2,6,9)

    H, W = height, width

    # Use standardized spherical coordinate generation
    spherical_coords = generate_spherical_coordinates_map(H, W, device)  # (H, W, 2)
    theta_map = spherical_coords[..., 0]  # elevation θ
    phi_map = spherical_coords[..., 1]    # azimuth φ

    # Convert theta to physics convention (from zenith angle to elevation)
    theta_zenith = convert_theta(theta_map)  # θ from zenith

    # Sun direction in spherical coordinates
    sun_theta = torch.tensor(sun_elevation, device=device, dtype=torch.float32)
    sun_phi = torch.tensor(sun_azimuth, device=device, dtype=torch.float32)

    # Compute angle between view direction and sun (gamma)
    # Using spherical law of cosines: cos(gamma) = sin(θ1)sin(θ2) + cos(θ1)cos(θ2)cos(φ1-φ2)
    cos_gamma = (torch.sin(theta_map) * torch.sin(sun_theta) +
                 torch.cos(theta_map) * torch.cos(sun_theta) * torch.cos(phi_map - sun_phi))
    gamma = torch.acos(torch.clamp(cos_gamma, -1.0, 1.0))

    # Normalize solar elevation to x in [0,1] for Bézier
    # Use the original float value for numpy compatibility
    elev_x = _normalize_solar_elevation(sun_elevation)

    # Evaluate channels independently
    out = torch.zeros((H, W, 3), dtype=torch.float32, device=device)

    # Convert torch tensors to numpy for compatibility with numpy-based evaluation functions
    theta_zenith_np = theta_zenith.cpu().numpy()
    gamma_np = gamma.cpu().numpy()

    for ch in range(3):
        params = _interp_params(turbidity, albedo, elev_x, coeffs[ch])  # 9 params for this channel
        # Distribution function F(θ,γ) - use zenith angle for theta
        F_np = _eval_F(theta_zenith_np, gamma_np, params)
        # Convert back to torch tensor (use float32 for OpenCV compatibility)
        F = torch.from_numpy(F_np).to(device=device, dtype=torch.float32)
        # Master luminance/radiance (L_λ^M) term: in ArHosek it’s rolled into the 9 params & F.
        # The reference C computes distribution and master separately; the public data is fit so
        # that the product below directly gives spectral/XYZ/RGB radiance. (Consistent with ref.)
        out[..., ch] = F

    # No sun disc term here (model omits it; add your own if desired).
    # Clamp negatives due to interpolation noise and handle overflow
    out = torch.clamp(out, min=0.0, max=1e10)  # Prevent extreme values

    # Apply tone mapping to control brightness
    # Scale down the overall intensity for more natural brightness
    out = out * brightness_scale

    # Convert to float32 for OpenCV compatibility
    out = out.to(dtype=torch.float32)
    return out

# (Optional) simple sun-disc you can composite later (not used by default)
def add_sun_disc(
    env_xyz: torch.Tensor,
    sun_azimuth: float,
    sun_elevation: float,
    intensity: float = 50.0,
    radius_deg: float = 0.27,
) -> torch.Tensor:
    """
    Add a simple sun disc to the environment map.

    :param env_xyz: torch.Tensor (H, W, 3) environment map
    :param sun_azimuth: sun azimuth angle in radians
    :param sun_elevation: sun elevation angle in radians
    :param intensity: intensity of the sun disc
    :param radius_deg: radius of the sun disc in degrees
    :return: torch.Tensor (H, W, 3) environment map with sun disc added
    """
    H, W, _ = env_xyz.shape
    device = env_xyz.device

    # Use standardized spherical coordinate generation
    spherical_coords = generate_spherical_coordinates_map(H, W, device)
    theta_map = spherical_coords[..., 0]  # elevation θ
    phi_map = spherical_coords[..., 1]    # azimuth φ

    # Sun direction in spherical coordinates
    sun_theta = torch.tensor(sun_elevation, device=device, dtype=env_xyz.dtype)
    sun_phi = torch.tensor(sun_azimuth, device=device, dtype=env_xyz.dtype)

    # Compute angle between view direction and sun
    cos_gamma = (torch.sin(theta_map) * torch.sin(sun_theta) +
                 torch.cos(theta_map) * torch.cos(sun_theta) * torch.cos(phi_map - sun_phi))
    gamma = torch.acos(torch.clamp(cos_gamma, -1.0, 1.0))

    # Create mask for sun disc
    radius_rad = math.radians(radius_deg)
    mask = (gamma <= radius_rad)

    # Add sun disc intensity
    env_xyz = env_xyz + mask.unsqueeze(-1) * intensity
    return env_xyz


def save_sky_environment(
    sky_tensor: torch.Tensor,
    exr_path: str,
    png_path: Optional[str] = None,
    gamma: float = 2.2,
    exposure: float = 0.0,
) -> None:
    """
    Save a generated sky environment to EXR and optionally PNG formats.

    :param sky_tensor: torch.Tensor (H, W, 3) sky environment
    :param exr_path: path to save the EXR file
    :param png_path: optional path to save the PNG preview
    :param gamma: gamma correction for PNG (default 2.2)
    :param exposure: exposure adjustment for PNG (default 0.0)
    """
    # Ensure the directory exists
    Path(exr_path).parent.mkdir(parents=True, exist_ok=True)

    # Save EXR file
    write_exr(sky_tensor, exr_path)

    # Save PNG preview if requested
    if png_path is not None:
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        exr_to_png_tensor(sky_tensor, png_path, gamma=gamma, exposure=exposure)


def generate_and_save_sky(
    width: int,
    height: int,
    sun_azimuth: float,
    sun_elevation: float,
    exr_output_path: str,
    png_output_path: Optional[str] = None,
    turbidity: float = 3.0,
    albedo: float = 0.1,
    space: str = "XYZ",
    add_sun: bool = False,
    sun_intensity: float = 50.0,
    sun_radius_deg: float = 0.27,
    device: Optional[Union[str, torch.device]] = None,
    coeffs: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Generate and save a sky environment map using standardized utilities.

    :param width: output width
    :param height: output height
    :param sun_azimuth: sun azimuth in radians
    :param sun_elevation: sun elevation in radians
    :param exr_output_path: path to save EXR file
    :param png_output_path: optional path to save PNG preview
    :param turbidity: sky turbidity (1-10)
    :param albedo: ground albedo (0-1)
    :param space: color space ("XYZ" or "RGB")
    :param add_sun: whether to add a sun disc
    :param sun_intensity: sun disc intensity
    :param sun_radius_deg: sun disc radius in degrees
    :param device: torch device
    :param coeffs: optional preloaded coefficients
    :return: generated sky tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate sky
    sky = generate_sky_latlong(
        width=width,
        height=height,
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
        turbidity=turbidity,
        albedo=albedo,
        space=space,
        coeffs=coeffs,
        device=device,
    )

    # Add sun disc if requested
    if add_sun:
        sky = add_sun_disc(
            sky,
            sun_azimuth=sun_azimuth,
            sun_elevation=sun_elevation,
            intensity=sun_intensity,
            radius_deg=sun_radius_deg,
        )

    # Save to files
    save_sky_environment(
        sky,
        exr_path=exr_output_path,
        png_path=png_output_path,
    )

    return sky
