import torch
from pathlib import Path
from coolname import generate_slug
from tqdm import tqdm
import argparse
import random
import numpy as np
import math
import re
import unicodedata
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


from src.LightingStudio.analysis.utils.io import write_exr
from src.LightingStudio.analysis.utils.transforms import cartesian_to_spherical, convert_theta, generate_spherical_coordinates_map, spherical_to_cartesian, spherical_to_pixel


def xyY_to_XYZ(xyY: torch.Tensor) -> torch.Tensor:
    """
    Convert CIE xyY to XYZ

    : params xyY: (H, W, 3) tensor of xyY values

    : returns XYZ: (H, W, 3) tensor of XYZ values
    
    Source:
    [19] xyY_to_XYZ function
    """
    x, y, Y = xyY[..., 0], xyY[..., 1], xyY[..., 2]
    # Avoid division by zero
    y_safe = torch.clamp(y, min=1e-6)

    return  torch.stack([x * Y / y_safe, Y, (1 - x - y) * Y / y_safe], dim=-1)

def XYZ_to_RGB(XYZ: torch.Tensor) -> torch.Tensor:
    """
    Convert CIE XYZ to linear RGB

    : params XYZ: (H, W, 3) tensor of XYZ values
    
    : returns RGB: (H, W, 3) tensor of RGB values
    
    Source:
    [19] XYZ_to_RGB function
    """
    XYZ_to_linear = torch.tensor([
        [3.24096994, -1.53738318, -0.49861076],
        [-0.96924364,  1.8759675,  0.04155506],
        [0.55630080, -0.20397696,  1.05697151]
    ], device=XYZ.device, dtype=XYZ.dtype).T

    # TODO: Check if XYZ_to_linear should be transposed??
    return  torch.einsum("hwc,cd->hwd", XYZ, XYZ_to_linear)

def tonemap(img: torch.Tensor, exposure: float = 0.1) -> torch.Tensor:
    """
    Tonemap

    : params img: torch.Tensor, image to tonemap
    : params exposure: float, exposure of the image
    """
    return 2.0 / (1.0 + torch.exp(-exposure * img)) - 1.0

def perez_radiance_distribution_parameters(turbidity: float) -> torch.Tensor:
    """
    Perez radiance distribution parameters

    : params turbidity: float, turbidity of the sky

    : returns perez_params: torch.Tensor (3, 5) 

    Source:
    [17] Appendix A.2
    """
    mx = torch.tensor([
        [-0.0193, -0.0665, -0.0004, -0.0641, -0.0033],
        [-0.2592, 0.0008, 0.2125, -0.8989, 0.0452],
    ]).T

    my = torch.tensor([
        [-0.0167, -0.0950, -0.0079, -0.0441, -0.0109],
        [-0.2608, 0.0092, 0.2102, -1.6537, 0.0529],
    ]).T

    mY = torch.tensor([
        [0.1787, -0.3554, -0.0227, 0.1206, -0.0670],
        [-1.4630, 0.4275, 5.3251, -2.5771, 0.3703],
    ]).T

    m = torch.stack([mx, my, mY], dim=-1) # shape (5, 2, 3)
    m = torch.permute(m, (2, 0, 1)) # shape (3, 5, 2)

    turbidity = torch.tensor([turbidity, 1.0]).unsqueeze(-1) # shape (2, 1)

    perez_params = torch.einsum("hwc,cd->hwd", m, turbidity).squeeze(-1) # shape (3, 5)
    assert perez_params.shape == (3, 5)

    return perez_params
    
def calculate_zenith_chromaticity_and_luminance(turbidity: float, theta_s: torch.Tensor) -> torch.Tensor:
    """
    Zenith chromaticity and luminance

    : params turbidity: float, turbidity of the sky
    : params theta_s: torch.Tensor, sun zenith angle

    : return zenith_chromaticity_and_luminance: torch.Tensor (3, 1)

    Source:
    [17] Appendix A.2
    """

    T_vec = torch.tensor([turbidity * turbidity, turbidity, 1.0]).unsqueeze(0) # shape (1, 3) 
    Z_vec = torch.tensor([theta_s * theta_s * theta_s, theta_s * theta_s, theta_s, 1.0]).unsqueeze(-1) # shape (4, 1)

    mx = torch.tensor([
        [0.0017, -0.0037, 0.0021, 0.0000],
        [-0.02902, 0.06377, -0.03202, 0.00394],
        [0.11693, -0.21196, 0.06052, 0.25886],
    ])
    my = torch.tensor([
        [0.00275, -0.00610, 0.00316, 0.0000],
        [-0.04214, 0.08970, -0.04153, 0.00516],
        [0.15346, -0.26756, 0.06670, 0.26688],
    ])


    Lx = torch.matmul(T_vec, torch.matmul(mx, Z_vec)).squeeze(0) # shape(1)
    Ly = torch.matmul(T_vec, torch.matmul(my, Z_vec)).squeeze(0) # shape(1)

    chi = (4.0 / 9.0 - turbidity / 120.0) * (torch.pi - 2.0 * theta_s)
    LY = (4.0453 * turbidity - 4.9710) * torch.tan(chi) - 0.2155 * turbidity + 2.4192
    LY = torch.tensor([LY])

    zenith_chromaticity_and_luminance = torch.stack([Lx, Ly, LY], dim=-1).squeeze(0)
    assert zenith_chromaticity_and_luminance.shape == (3,)

    return zenith_chromaticity_and_luminance 

def calculate_gamma(theta_s: torch.Tensor, phi_s: torch.Tensor, theta_v: torch.Tensor, phi_v: torch.Tensor) -> torch.Tensor:
    """
    Calculate gamma

    : params theta_s: torch.Tensor, sun zenith angle (1)
    : params phi_s: torch.Tensor, sun azimuth angle (1)
    : params theta_v: torch.Tensor, viewing zenith angle (H, W)
    : params phi_v: torch.Tensor, viewing azimuth angle (H, W)

    : returns gamma: torch.Tensor, gamma angle between sun and viewing direction
    
    Source:
    [19] angle funciton 
    """
    return torch.acos(torch.sin(theta_s) * torch.cos(phi_s) * torch.sin(theta_v) * torch.cos(phi_v) + \
                      torch.sin(theta_s) * torch.sin(phi_s) * torch.sin(theta_v) * torch.sin(phi_v) + \
                      torch.cos(theta_s) * torch.cos(theta_v))

def perez_F(theta: torch.Tensor, gamma: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Perez F

    : params theta: torch.Tensor, zenith angle (H, W)
    : params gamma: torch.Tensor, gamma angle (H, W)
    : params coeffs: torch.Tensor, Perez coefficients (5)

    : returns F: torch.Tensor, Perez F (H, W)

    Source: 
    [17] Equation 3
    [19] F function
    """
    cos_theta = torch.cos(theta)
    cos_gamma_squared = torch.cos(gamma) * torch.cos(gamma)

    assert coeffs.shape == (5,), f"coeffs.shape: {coeffs.shape}"

    A = coeffs[0]
    B = coeffs[1]
    C = coeffs[2]
    D = coeffs[3]
    E = coeffs[4]

    return (1.0 + A * torch.exp(B / cos_theta)) * (1.0 + C * torch.exp(D * gamma) + E * cos_gamma_squared)

def generate_preetham_sky(H: int, W: int, turbidity: float, sun_dir: torch.Tensor) -> torch.Tensor:
    """
    Generate Preetham based hdri map

    : params H: int, height of the image
    : params W: int, width of the image
    : params turbidity: float, turbidity of the sky
    : params sun_dir: torch.Tensor, normalized direction of the sun

    : returns hdri: torch.Tensor, hdri map (H, W, 3)
    """

    # Sun angles (relative to zenith)
    spherical_coordinates = cartesian_to_spherical(sun_dir)

    phi_s = spherical_coordinates[..., 1] # shape (1)
    theta_s = convert_theta(spherical_coordinates[..., 0]) # shape (1)

    assert theta_s.shape == (1,), f"theta_s.shape: {theta_s.shape}"
    assert phi_s.shape == (1,), f"phi_s.shape: {phi_s.shape}"

    # Perez params
    perez_params = perez_radiance_distribution_parameters(turbidity) # shape (3, 5)

    # Zenith luminance
    zenith_chromaticity_and_luminance = calculate_zenith_chromaticity_and_luminance(turbidity, theta_s) # shape (3)

    # Generate hdri map
    # TODO: fix device stuff

    # Viewing angles
    spherical_coordinates = generate_spherical_coordinates_map(H, W) # shape (H, W, 2)
    phi_v = spherical_coordinates[..., 1] # shape (H, W)
    theta_v = convert_theta(spherical_coordinates[..., 0]) # shape (H, W)

    gamma = calculate_gamma(theta_s, phi_s, theta_v, phi_v) # shape (H, W)

    # Calculate x, y, Y from F
    x = zenith_chromaticity_and_luminance[0] * perez_F(theta_v, gamma, perez_params[0]) / perez_F(torch.tensor([0.0]), theta_s, perez_params[0])
    y = zenith_chromaticity_and_luminance[1] * perez_F(theta_v, gamma, perez_params[1]) / perez_F(torch.tensor([0.0]), theta_s, perez_params[1])
    Y = zenith_chromaticity_and_luminance[2] * perez_F(theta_v, gamma, perez_params[2]) / perez_F(torch.tensor([0.0]), theta_s, perez_params[2])

    # Convert from xyY to XYZ to RGB
    XYZ = xyY_to_XYZ(torch.stack([x, y, Y], dim=-1))
    RGB = XYZ_to_RGB(XYZ)

    # Mask out lower hemisphere
    lower_hemi = (torch.abs(theta_v) > 0.48 * torch.pi)[..., None] # (H, W, 1) bool
    RGB = torch.where(lower_hemi, torch.zeros_like(RGB), RGB)

    # Clamp negative values
    RGB = torch.clamp(RGB, 0.0, None)

    # Tonemap
    tonemap_RGB = tonemap(RGB)

    return tonemap_RGB

def get_sm_rad_from_long_deg(lon_deg: torch.Tensor) -> torch.Tensor:
    """
    Get standard meridian radians from longitude degrees
    """
    # ensure floating dtype for trig/deg2rad
    lon = lon_deg.to(dtype=torch.get_default_dtype())
    sm_deg = 15.0 * torch.round(lon / 15.0)
    return torch.deg2rad(sm_deg)


def spherical_coordinates_from_sun_position(
    J: torch.Tensor,          # Julian day of year, 1..365 (or 366), radians-safe torch scalar/tensor
    ts: torch.Tensor,         # local standard time, decimal hours [0..24)
    lat: torch.Tensor,        # site latitude, radians (+N)
    lon: torch.Tensor,        # site longitude, radians (+E)
    SM: torch.Tensor,         # standard meridian for time zone, radians (+E). e.g., UTC-8 -> SM = deg2rad(-120)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Spherical coordinates from sun position

    : params J: torch.Tensor, Julian day of year, 1..365 (or 366), radians-safe torch scalar/tensor
    : params ts: torch.Tensor, local standard time, decimal hours [0..24)
    : params lat: torch.Tensor, site latitude, radians (+N)
    : params lon: torch.Tensor, site longitude, radians (+E)
    : params SM: torch.Tensor, standard meridian for time zone, radians (+E). e.g., UTC-8 -> SM = deg2rad(-120)
    
    : returns t: torch.Tensor, solar time (decimal hours)
    : returns theta_s: torch.Tensor, solar zenith angle (radians)  [0 .. pi], <= pi/2 when sun is above horizon
    : returns phi_s: torch.Tensor, solar azimuth (radians), positive to the WEST of South (per paper)

    Source:
    [17] Appendix A.6
    """

    # --- Solar time t (decimal hours) -----------------------------------------
    # t = ts + 0.170 * sin(4π(J - 80)/373) - 0.129 * sin(2π(J - 8)/355) + 12*(SM - lon)/π
    term1 = 0.170 * torch.sin((4.0 * torch.tensor(torch.pi)) * (J - 80.0) / 373.0)
    term2 = 0.129 * torch.sin((2.0 * torch.tensor(torch.pi)) * (J - 8.0)  / 355.0)
    t = ts + term1 - term2 + 12.0 * (SM - lon) / torch.pi  # solar time (hours) 

    # --- Solar declination δ (radians) -----------------------------------------
    # δ = 0.4093 * sin(2π(J - 81)/368)
    delta = 0.4093 * torch.sin((2.0 * torch.tensor(torch.pi)) * (J - 81.0) / 368.0)  

    # --- Hour angle H (radians) using solar time t -----------------------------
    # Standard: H = π/12 * (t - 12). (Solar noon -> H=0)
    # H = (pi / 12.0) * (t - 12.0)
    H = (torch.pi * t / 12.0) 

    # --- Solar zenith θ_s and azimuth φ_s -------------------------------------
    # PAPER’s forms (A.6):
    # θ_s = pi/2 - arcsin( sin(lat) sin(δ) + cos(lat) cos(δ) cos(H) )
    # φ_s = atan2( -cos(δ) sin(H),  cos(lat) sin(δ) - sin(lat) cos(δ) cos(H) )
    # (This atan2 yields azimuth measured from SOUTH, positive toward WEST, matching the paper’s convention.)

    theta_s = torch.pi/2.0 - torch.asin(torch.sin(lat) * torch.sin(delta) - torch.cos(lat) * torch.cos(delta) * torch.cos(H))
    phi_s = torch.atan2(-torch.cos(delta) * torch.sin(H), torch.cos(lat) * torch.sin(delta) - torch.sin(lat) * torch.cos(delta) * torch.cos(H))

    return t, theta_s, phi_s

def make_red_circle(img: torch.Tensor, sun_pixel: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Make a red circle around the sun pixel
    """
    H, W = img.shape[0], img.shape[1]
    # sun_pixel has shape (2,) where it contains [x, y]
    sun_x = sun_pixel[0].item()
    sun_y = sun_pixel[1].item()
    
    for i in range(H):
        for j in range(W):
            # Note: i corresponds to y (row) and j corresponds to x (column)
            if (i - sun_y)**2 + (j - sun_x)**2 <= radius**2:
                img[i, j, :] = torch.tensor([1.0, 0.0, 0.0])
    return img

def month_to_julian_day_range(month: int) -> tuple[int, int]:
    """
    Map month (1-12) to Julian day range
    
    : params month: int, month number (1-12)
    : returns tuple: (start_day, end_day) Julian day range for the month
    """
    # Approximate Julian day ranges for each month (non-leap year)
    month_ranges = {
        1: (1, 31),      # January
        2: (32, 59),     # February  
        3: (60, 90),     # March
        4: (91, 120),    # April
        5: (121, 151),   # May
        6: (152, 181),   # June
        7: (182, 212),   # July
        8: (213, 243),   # August
        9: (244, 273),   # September
        10: (274, 304),  # October
        11: (305, 334),  # November
        12: (335, 365)   # December
    }
    
    if month not in month_ranges:
        raise ValueError(f"Month must be between 1-12, got {month}")
    
    return month_ranges[month]

def generate_random_turbidity() -> float:
    """
    Generate a random turbidity value between 2.0 and 8.0 with exponential weighting.
    
    Uses exponential distribution that heavily favors lower turbidity values (clearer skies)
    and becomes less likely at higher turbidity values (hazier conditions).
    
    Turbidity values represent atmospheric conditions:
    - 2.0: Very clear sky (excellent visibility) - most likely
    - 4.0: Average clear sky conditions  
    - 6.0: Slightly hazy conditions
    - 8.0: Hazy/smoggy conditions - least likely
    
    Returns:
        float: Exponentially weighted random turbidity value between 2.0 and 8.0
    """
    # Generate exponential random variable (lambda=1.0 gives good distribution)
    exp_value = random.expovariate(1.0)
    
    # Clamp to reasonable range (0 to 3) before scaling
    exp_value = min(exp_value, 3.0)
    
    # Scale from [0, 3] to [2.0, 8.0] range
    turbidity = 2.0 + (exp_value / 3.0) * 6.0
    
    return turbidity


def unicode_to_ascii_safe(text: str) -> str:
    """
    Convert Unicode text to ASCII-safe characters for Windows file paths.
    
    This function handles diacritics and special characters that can cause issues
    with OpenCV file I/O on Windows systems.
    
    Args:
        text: Unicode string (e.g., "Khánh Hòa Province")
        
    Returns:
        ASCII-safe string (e.g., "Khanh_Hoa_Province")
    """
    # Normalize Unicode characters to NFD (decomposed form)
    # This separates base characters from combining diacritics
    normalized = unicodedata.normalize('NFD', text)
    
    # Remove combining characters (diacritics)
    ascii_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    
    # Replace any remaining non-ASCII characters with underscores
    ascii_text = re.sub(r'[^\x00-\x7F]', '_', ascii_text)
    
    # Clean up spaces and punctuation for filename use
    ascii_text = re.sub(r'[^\w\-]', '_', ascii_text)
    ascii_text = re.sub(r'_+', '_', ascii_text)  # Remove multiple underscores
    ascii_text = ascii_text.strip('_')
    
    return ascii_text


def rand_lat_lon_deg() -> tuple[float, float]:
    """
    Generate random latitude and longitude coordinates with uniform distribution on sphere
    
    Uses proper spherical sampling to avoid clustering at poles:
    - Longitude: uniform distribution [-180, 180] degrees
    - Latitude: arcsin(2u-1) distribution to account for sphere geometry
    
    : returns tuple: (latitude_deg, longitude_deg) in degrees
    
    Source:
    Uniform sampling on sphere surface
    """
    u = random.random()
    v = random.random()
    
    lon_deg = 360.0 * v - 180.0
    lat_rad = math.asin(2.0 * u - 1.0)   # φ ~ arcsin(2u-1) for uniform sphere sampling
    lat_deg = math.degrees(lat_rad)
    
    return lat_deg, lon_deg

def get_location_name(lat_deg: float, lon_deg: float) -> str:
    """
    Get location name from latitude and longitude coordinates using reverse geocoding
    
    : params lat_deg: float, latitude in degrees
    : params lon_deg: float, longitude in degrees
    : returns str: location name (city, country) or coordinates if geocoding fails
    """
    try:
        geolocator = Nominatim(user_agent="lighting_studio_hdri_generator", timeout=3)
        location = geolocator.reverse(f"{lat_deg}, {lon_deg}", exactly_one=True, language='en')
        
        if location and location.address:
            # Extract meaningful parts from the address
            address_parts = location.address.split(', ')
            
            # Try to get city and country, fallback to available parts
            if len(address_parts) >= 2:
                # Usually the last part is country, second to last might be state/region
                country = address_parts[-1]
                city_or_region = None
                
                # Look for a city-like component (avoid postal codes, coordinates)
                for part in reversed(address_parts[:-1]):
                    if not re.match(r'^\d+', part) and len(part) > 2:  # Skip numbers and very short strings
                        city_or_region = part
                        break
                
                if city_or_region:
                    # Clean up the location name for filename use
                    location_name = f"{city_or_region}_{country}"
                else:
                    location_name = country
                
                # Convert Unicode characters to ASCII-safe equivalents for Windows compatibility
                location_name = unicode_to_ascii_safe(location_name)
                
                return location_name
        
    except (GeocoderTimedOut, GeocoderServiceError, Exception):
        pass  # Fall back to coordinates
    
    # Fallback to coordinates if geocoding fails
    lat_str = f"{abs(lat_deg):.1f}{'N' if lat_deg >= 0 else 'S'}"
    lon_str = f"{abs(lon_deg):.1f}{'E' if lon_deg >= 0 else 'W'}"
    return f"{lat_str}_{lon_str}"

OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"


def generate_single_hdri(H: int, W: int, elevation: float, azimuth: float, output_dir: Path) -> None:
    """
    Generate a single HDRI from direct parameters
    
    : params H: int, height of the image
    : params W: int, width of the image  
    : params elevation: float, elevation angle in radians
    : params azimuth: float, azimuth angle in radians
    : params output_dir: Path, output directory
    """
    # Generate random turbidity for this HDRI
    turbidity = generate_random_turbidity()
    
    print(f"Generating single HDRI with parameters:")
    print(f"  Resolution: {H}x{W}")
    print(f"  Turbidity: {turbidity:.2f} (randomly generated)")
    print(f"  Elevation: {elevation:.4f} rad ({np.degrees(elevation):.2f}°)")
    print(f"  Azimuth: {azimuth:.4f} rad ({np.degrees(azimuth):.2f}°)")
    
    # Create spherical coordinates and convert to cartesian direction
    spherical_coordinates = torch.stack([torch.tensor([elevation]), torch.tensor([azimuth])], dim=-1)
    sun_dir = spherical_to_cartesian(spherical_coordinates)
    
    # Generate HDRI
    img = generate_preetham_sky(H, W, turbidity, sun_dir)
    
    # Save the image
    output_path = output_dir / f"single_hdri_T{turbidity:.2f}_elev{np.degrees(elevation):.1f}_azim{np.degrees(azimuth):.1f}.exr"
    write_exr(img, str(output_path))
    print(f"Saved HDRI to: {output_path}")

def generate_batch_hdri(H: int, W: int, n_locations: int, months: list[int], output_dir: Path, horizon_only: bool = False, horizon_threshold: float = 10.0) -> None:
    """
    Generate batch HDRIs from random locations and specified months
    
    : params H: int, height of the image
    : params W: int, width of the image
    : params n_locations: int, number of random locations to sample
    : params months: list[int], list of months (1-12) to generate for
    : params output_dir: Path, output directory
    : params horizon_only: bool, only generate images with sun near horizon
    : params horizon_threshold: float, maximum elevation angle in degrees for horizon sampling
    """
    print(f"Generating batch HDRIs with parameters:")
    print(f"  Resolution: {H}x{W}")
    print(f"  Turbidity: Random (2.0-8.0) for each HDRI")
    print(f"  Number of locations: {n_locations}")
    print(f"  Months: {months}")
    if horizon_only:
        print(f"  Horizon only mode: True (max elevation: {horizon_threshold}°)")
    else:
        print(f"  Horizon only mode: False")
    
    count = 0
    
    # Generate random locations
    for location_idx in range(n_locations):
        # Sample random latitude and longitude
        lat_deg, lon_deg = rand_lat_lon_deg()
        
        # Get location name for this coordinate
        location_name = get_location_name(lat_deg, lon_deg)

        lat_deg_tensor = torch.tensor([lat_deg])
        lon_deg_tensor = torch.tensor([lon_deg])
        SM = get_sm_rad_from_long_deg(lon_deg_tensor)
        
        print(f"\nLocation {location_idx + 1}: {location_name} (Lat={lat_deg:.2f}°, Lon={lon_deg:.2f}°)")
        
        # Process each month
        for month in months:
            start_day, end_day = month_to_julian_day_range(month)
            
            # Sample a few days from the month (e.g., beginning, middle, end)
            sample_days = [start_day, (start_day + end_day) // 2, end_day]
            
            for J in sample_days:
                # Generate random turbidity for this specific HDRI
                turbidity = generate_random_turbidity()
                print(f"Day {J}, Turbidity: {turbidity:.2f}")

                # Sample all 24 hours of the day
                for ts in range(24):
                    lat_radians = torch.deg2rad(lat_deg_tensor)
                    lon_radians = torch.deg2rad(lon_deg_tensor)
                    
                    t, theta_s, phi_s = spherical_coordinates_from_sun_position(
                        torch.tensor(J), torch.tensor(ts), lat_radians, lon_radians, SM
                    )
                   
                    elev = torch.pi/2.0 - torch.abs(theta_s)
                    azim = phi_s
                    
                    # Convert elevation to degrees for horizon filtering
                    elev_degrees = np.degrees(elev.item())
                    
                    # If horizon_only mode is enabled, skip images where sun is too high
                    within_horizon = (elev_degrees > 0.0 and elev_degrees < horizon_threshold)
                    if (horizon_only and not within_horizon):
                        continue
                   
                    spherical_coordinates = torch.stack([elev, azim], dim=-1)
                    sun_dir = spherical_to_cartesian(spherical_coordinates)
                    
                    img = generate_preetham_sky(H, W, turbidity, sun_dir)
                    
                    # Create descriptive filename
                    horizon_suffix = "_horizon" if horizon_only else ""
                    # output_path = output_dir / f"batch_hdri_{location_name}_month{month}_day{J}_time{ts:02d}_elev{elev_degrees:.1f}{horizon_suffix}_f_{count:04d}.exr"
                    output_path = output_dir / f"batch_hdri_{location_name}_f_{count:04d}.exr"
                    write_exr(img, str(output_path))
                    count += 1
    
    print(f"\nGenerated {count} HDRIs total")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate Preetham sky HDRIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single HDRI with specific parameters (turbidity randomly assigned)
  python skylight_model.py single --height 512 --width 1024 --elevation 0.5 --azimuth 1.2
  
  # Generate batch HDRIs from random locations (turbidity randomly assigned per HDRI)
  python skylight_model.py batch --height 256 --width 512 --n_locations 5 --months 6 7 8
  
  # Generate sunset/sunrise HDRIs only (sun near horizon)
  python skylight_model.py batch --height 256 --width 512 --n_locations 3 --months 6 7 8 --horizon_only --horizon_threshold 15.0
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Generation mode')
    
    # Single HDRI mode
    single_parser = subparsers.add_parser('single', help='Generate single HDRI from direct parameters')
    single_parser.add_argument('--height', '-H', type=int, required=True, help='Height of the HDRI')
    single_parser.add_argument('--width', '-W', type=int, required=True, help='Width of the HDRI')
    single_parser.add_argument('--elevation', '-e', type=float, required=True, help='Elevation angle in radians')
    single_parser.add_argument('--azimuth', '-a', type=float, required=True, help='Azimuth angle in radians')
    
    # Batch HDRI mode
    batch_parser = subparsers.add_parser('batch', help='Generate batch HDRIs from random locations and months')
    batch_parser.add_argument('--height', '-H', type=int, required=True, help='Height of the HDRI')
    batch_parser.add_argument('--width', '-W', type=int, required=True, help='Width of the HDRI')
    batch_parser.add_argument('--n_locations', '-n', type=int, required=True, help='Number of random locations to sample')
    batch_parser.add_argument('--months', '-m', type=int, nargs='+', required=True, 
                            help='Months to generate for (1-12)', choices=range(1, 13))
    batch_parser.add_argument('--horizon_only', action='store_true', 
                            help='Only generate images with sun near horizon (sunset/sunrise conditions)')
    batch_parser.add_argument('--horizon_threshold', type=float, default=10.0,
                            help='Maximum elevation angle in degrees for horizon sampling (default: 10.0)')
    
    # Common arguments
    for subparser in [single_parser, batch_parser]:
        subparser.add_argument('--output_dir', '-o', type=str, default=None, 
                             help='Output directory (default: auto-generated)')
        subparser.add_argument('--seed', '-s', type=int, default=None,
                             help='Random seed for reproducibility')
    
    return parser.parse_args()

# python -m src.LightingStudio.ingest.skylight_model batch -H 512 -W 1024 -n 1 -m 1 --horizon_only
# python -m src.LightingStudio.ingest.skylight_model single -H 512 -W 1024 -e 0.7854 -a 1.570

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_arguments()
    
    if args.mode is None:
        print("Error: Please specify a mode (single or batch)")
        print("Use --help for usage information")
        exit(1)
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Set random seed to: {args.seed}")
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        experiment_name = generate_slug(2)
        output_dir = Path(OUTPUT_DIR) / experiment_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Validate image dimensions
    if args.height <= 0 or args.width <= 0:
        print("Error: Height and width must be positive integers")
        exit(1)
    
    # Execute based on mode
    if args.mode == 'single':
        generate_single_hdri(args.height, args.width, 
                            args.elevation, args.azimuth, output_dir)
    
    elif args.mode == 'batch':
        if args.n_locations <= 0:
            print("Error: Number of locations must be positive")
            exit(1)
        
        generate_batch_hdri(args.height, args.width, 
                          args.n_locations, args.months, output_dir,
                          args.horizon_only, args.horizon_threshold)
