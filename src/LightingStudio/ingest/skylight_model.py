import torch
from pathlib import Path
from coolname import generate_slug
from tqdm import tqdm

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
    lower_hemi = (torch.abs(theta_v) > 0.5 * torch.pi)[..., None] # (H, W, 1) bool
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


OUTPUT_DIR = r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments"


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    H, W = 256, 512
    T = 2.3

    # Original Direct    
    # elev = torch.tensor([(1/8)*torch.pi/2.0])
    # azim = torch.tensor([-(7/8)*torch.pi])

    # elev = torch.tensor([0.3670])
    # azim = torch.tensor([-1.8645])

    # print("elev: ", elev)
    # print("azim: ", azim)

    # spherical_coordinates = torch.stack([elev, azim], dim=-1)
    # print(f"spherical_coordinates: {spherical_coordinates}, with shape: {spherical_coordinates.shape}")

    # sun_dir = spherical_to_cartesian(spherical_coordinates)
    # print("sun_dir: ", sun_dir)

    # img = generate_preetham_sky(H, W, T, sun_dir)
    # write_exr(img, "preetham_sky_original_direction.exr")

    # Create random directory for output
    experiment_name = generate_slug(2)
    output_dir = Path(OUTPUT_DIR) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    lat_deg = torch.tensor([19.74])
    lon_deg = torch.tensor([-155.98])
    SM = get_sm_rad_from_long_deg(lon_deg)

    count = 0
    for J in tqdm(range(1), desc="Generating Preetham sky"):
        for ts in range(24):
            lat_radians = torch.deg2rad(lat_deg)
            lon_radians = torch.deg2rad(lon_deg)

            t, theta_s, phi_s = spherical_coordinates_from_sun_position(J, ts, lat_radians, lon_radians, SM)

            elev = torch.pi/2.0 - torch.abs(theta_s)
            azim = phi_s
        
            spherical_coordinates = torch.stack([elev, azim], dim=-1)
            sun_dir = spherical_to_cartesian(spherical_coordinates)
            sun_pixel = spherical_to_pixel(spherical_coordinates.unsqueeze(0), H, W).squeeze(0).squeeze(0)

            print(
                f"count: {count}, "
                # f"lat_deg: {lat_deg}, lon_deg: {lon_deg}, "
                # f"lat_radians: {lat_radians}, lon_radians: {lon_radians}, SM: {SM}, t: {t}, "
                f"theta_s: {theta_s}, phi_s: {phi_s}, elev: {elev}, azim: {azim}, "
                f"spherical_coordinates: {spherical_coordinates}, sun_dir: {sun_dir}, sun_pixel: {sun_pixel}"
                # f"J: {J}, ts: {ts}"
            )
            print("-"*100)

            img = generate_preetham_sky(H, W, T, sun_dir)  # (H,W,3) linear sRGB

            # make a pink circle around sun_pixel
            img = make_red_circle(img, sun_pixel, 10)

            # Write EXR to the random directory
            output_path = output_dir / f"preetham_hawaii_{count}.exr"
            write_exr(img, str(output_path))
            count += 1
