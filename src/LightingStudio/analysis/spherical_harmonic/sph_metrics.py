import torch
from einops import rearrange, repeat
from typing import Union

from src.LightingStudio.analysis.spherical_harmonic.sph import project_env_to_coefficients, project_direction_into_coefficients, sph_l_max_from_indices_total, l_from_index, sph_indices_total
from src.LightingStudio.analysis.utils import cartesian_to_spherical, convert_theta, generate_spherical_coordinates_map
from src.LightingStudio.analysis.datatypes import SPHMetrics, SPHMetricsCPU
from src.LightingStudio.analysis.utils import cartesian_to_pixel

def get_dominant_direction(sph_coeffs: torch.Tensor) ->  tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TODO: think about what about returning dominant direction in r, g, b directly and not adding them together???

    Get the dominant direction from the spherical harmonic coefficients.
    
    :params sph_coeffs: (n_terms, 3) where each column is r, g, b

    :return dominant_direction_normalized: (3) the dominant direction (with vector norm)
    :return dominant_direction_rgb_normalized: (3) the dominant direction rgb (with vector norm)
    :return dominant_direction_rgb_luminance_normalized: (3) the dominant direction rgb (with vector norm)
    
    Source: 
    [7] Section 3.3 NOT [2] page 4
    [8] Extracting dominant light direction section
    [9] GetLightingEnvironment function, page 95
    [13] Image. 
    [14] Weighted Color Difference
    """

    red_band_1 = sph_coeffs[1:4, 0]
    green_band_1 = sph_coeffs[1:4, 1]
    blue_band_1 = sph_coeffs[1:4, 2]

    # Use [7] Section 3.3 NOT [2] page 4
    # The reason there is a negative sign in the 1st index is the difference in artist and science definition of env_map direction.
    red_band_aligned_xyz = torch.tensor([-red_band_1[2], -red_band_1[0], red_band_1[1]], device=sph_coeffs.device, dtype=sph_coeffs.dtype)
    red_band_aligned_xyz = red_band_aligned_xyz / torch.linalg.norm(red_band_aligned_xyz)

    green_band_aligned_xyz = torch.tensor([-green_band_1[2], -green_band_1[0], green_band_1[1]], device=sph_coeffs.device, dtype=sph_coeffs.dtype)
    green_band_aligned_xyz = green_band_aligned_xyz / torch.linalg.norm(green_band_aligned_xyz)

    blue_band_aligned_xyz = torch.tensor([-blue_band_1[2], -blue_band_1[0], blue_band_1[1]], device=sph_coeffs.device, dtype=sph_coeffs.dtype)
    blue_band_aligned_xyz = blue_band_aligned_xyz / torch.linalg.norm(blue_band_aligned_xyz)

    color_xyz = torch.stack([red_band_aligned_xyz, green_band_aligned_xyz, blue_band_aligned_xyz], dim=1)

    # Dominant Direction in xyz coordinate
    dominant_direction = torch.sum(color_xyz, dim=1)
    dominant_direction_normalized = dominant_direction/torch.linalg.norm(dominant_direction)

    # Dominant Direction in rgb coordinate color_difference 
    rgb_color_difference_constants = torch.tensor([0.3, 0.59, 0.11], device=sph_coeffs.device, dtype=sph_coeffs.dtype)
    dominant_direction_rgb_color_difference = torch.sum(rgb_color_difference_constants * color_xyz, dim=1)
    dominant_direction_rgb_color_difference_normalized = dominant_direction_rgb_color_difference/torch.linalg.norm(dominant_direction_rgb_color_difference)

    # Dominant Direction in rgb coordinate luminance
    rgb_luminance_constants = torch.tensor([0.2126, 0.7152, 0.0722], device=sph_coeffs.device, dtype=sph_coeffs.dtype)
    dominant_direction_rgb_luminance = torch.sum(rgb_luminance_constants * color_xyz, dim=1)
    dominant_direction_rgb_luminance_normalized = dominant_direction_rgb_luminance/torch.linalg.norm(dominant_direction_rgb_luminance)

    return (dominant_direction_normalized, 
            dominant_direction_rgb_color_difference_normalized,
            dominant_direction_rgb_luminance_normalized)

def get_dominant_color(dominant_direction: torch.Tensor, env_map_sph_coeffs: torch.Tensor) -> torch.Tensor:
    """
    
    :params dominant_direction: (3)
    :params env_map_sph_coeffs: (n_terms, 3)

    :returns dominant_color: (3)

    Source:
    [8] code specLighting function
    [8] Extracting dominant light intensity section.
    """

    # Get sph_coeffs of each color.
    sph_coeffs_r = env_map_sph_coeffs[:,0]
    sph_coeffs_g = env_map_sph_coeffs[:,1]
    sph_coeffs_b = env_map_sph_coeffs[:,2]

    n_terms = env_map_sph_coeffs.shape[0]
    l_max = sph_l_max_from_indices_total(n_terms)

    # Project dominant direction into sph_coeffs.
    dominant_direction = rearrange(dominant_direction, "c -> 1 c")
    direction_sph_coeffs = project_direction_into_coefficients(dominant_direction, l_max) # (1, n_terms)
    direction_sph_coeffs = rearrange(direction_sph_coeffs, "1 n_terms -> n_terms") # (n_terms)

    # TODO: maybe we need to normalize the light??
    # direction_sph_coeffs *= (16*np.pi)/17
    denominator = torch.dot(direction_sph_coeffs, direction_sph_coeffs)

    color = 255.0 * torch.tensor([torch.dot(sph_coeffs_r, direction_sph_coeffs) / denominator,
                                  torch.dot(sph_coeffs_g, direction_sph_coeffs) / denominator,
                                  torch.dot(sph_coeffs_b, direction_sph_coeffs) / denominator], device=env_map_sph_coeffs.device, dtype=env_map_sph_coeffs.dtype)
    
    color = torch.clamp(color, 0, 255)
    return color

def get_cos_lobe_as_env_map(H:int, W:int, device: torch.device = None) -> torch.Tensor:
    """
    The value of N dot L placed in an environment map.
    
    : returns env_map_as_cos_lobe: (H, W, 3)

    Source:
    [7] cosine_lobe definition.
    """

    spherical_coordinates = generate_spherical_coordinates_map(H,W, device=device) # (H, W, 2)
    theta = spherical_coordinates[..., 0]                           # (H, W)
    theta = convert_theta(theta)                                    # (H, W)

    cos_lobe = torch.maximum(torch.zeros_like(theta), torch.cos(theta))  # (H, W)
    env_map_of_cos_lobe = repeat(cos_lobe, "h w -> h w c", c=3) # (H, W, 3)

    return env_map_of_cos_lobe

def get_area_normalization_term(env_map_sph_coeffs : torch.Tensor, cos_lobe_sph_coeffs: torch.Tensor, cartesian_direction: torch.Tensor, l_max: int) ->  torch.Tensor:
    """
    Compute Incoming Radiance (intensity) over hemisphere defined by the normal aligned to the dominant direction.
    Compute integral over omega(dominant_direction) L(w) T(w). 

    
    1. Rotate the cos_lobe_sph_coeffs into the direction of cartesian_direction
    2. Dot Product of rotated_cos_lobe_sph_coeffs and env_map_sph_coeffs

    : params env_map_sph_coeffs: (n_terms, 3)
    : params cos_lobe_sph_coeffs: (n_terms, 3)
    : params cartesian_direction: (3)
    : params l_max: 

    : returns area_intensity: (3) one for each color (r, g, b) = (c_light_r, c_light_g, c_light_b)

    Source:
    [8] Extracting dominant light intensity section
    [6] RenderDiffuseIrradiance function. cosine_lobe variable.
    [7] Equation 6.
    """

    # Project dominant direction into sph_coeffs.
    dominant_direction = rearrange(cartesian_direction, "c -> 1 c")
    direction_sph_coeffs = project_direction_into_coefficients(dominant_direction, l_max) # (1, n_terms)
    direction_sph_coeffs = rearrange(direction_sph_coeffs, "1 n_terms -> n_terms") # (n_terms)

    # TODO: test if cos lobe coeffs is correct.

    scaled_direction_sph_coeffs = torch.zeros_like(direction_sph_coeffs)
    for i in range(sph_indices_total(l_max)):
        l = l_from_index(i)  # noqa: E741
        scale_factor = torch.sqrt(torch.tensor(4*torch.pi/ (2 * l + 1), device=env_map_sph_coeffs.device, dtype=env_map_sph_coeffs.dtype))
        scaled_direction_sph_coeffs[i] = scale_factor * direction_sph_coeffs[i]

    scaled_repeated_direction_sph_coeffs = repeat(scaled_direction_sph_coeffs, "n_terms -> n_terms c", c=3)
    rotated_cos_lobe_sph_coeffs = torch.multiply(scaled_repeated_direction_sph_coeffs, cos_lobe_sph_coeffs) # (n_terms, 3)

    # The integral of the product of two spherical harmonic functions is equivalent to the dot product of their coefficients
    area_intensity = torch.sum(torch.multiply(env_map_sph_coeffs, rotated_cos_lobe_sph_coeffs), axis=0)  # (3)
    return area_intensity

def get_sph_metrics(env_map: torch.Tensor, l_max: int) -> SPHMetrics:
    """
    Get the SPH metrics for the environment map.

    Extract Single Light Direction, Single Light Intensity, Single Light Color (SLNA).

    We are constructing a new lighting source.
    When it is created, the light source has direction: dominant_direction.
    But it has no color. So we need to give it a color.

    Steps:
    1. Get Dominant Light Direction
    (SKIP) 2. Get Normalization Factor
    3. Get Dominant Light Color

    
    : params env_map: (H, W, 3)
    : params l_max: int
    : returns sph_metrics: SPHMetrics

    Source:
    [8] Full Page

    WARNING: 
    I think that this is not the right algorithm for our use case.
    
    They use it for a gaming like setup with where you have vertex  and fragment shaders and decompose the way
    a material is rendered into the diffuse and specular components.

    final_color  = diffuse(env_sph_coeff, N) + specular(dominant_light_color, dominant_light_direction, specular_exponent)

    diffuse(...) = use traditional SPH based lighting as done in [4]
    specular(...) = pi * lambertBrdf * dominant_light_color *pow(NdotH, glossiness)*NdotL (here L is dominant_light_direction)
    
    But this setup doesn't make sense for our case.
    For a customer looking at the Digital Human platform they will only see the rendered image, that means the lighting entangled with the material.
    So if we only let them select an hdri via the lighting direction, the light_color isn't accurately capturing the overall low frequency tint.
    
    So we should instead also try to capture the ambient color or DC term.
    """

    H, W, _ = env_map.shape

    # Get Spherical Harmonic Coefficients
    env_map_sph_coeffs, _ = project_env_to_coefficients(env_map, l_max)
    
    # Get Dominant Direction
    dd, dd_rgb_color_difference, dd_rgb_luminance = get_dominant_direction(env_map_sph_coeffs)

    # Get Dominant Pixel
    dpixel, dpixel_rgb_color_difference, dpixel_rgb_luminance =  cartesian_to_pixel(torch.stack([dd, dd_rgb_color_difference, dd_rgb_luminance], dim=0), H, W)

    # Get Dominant Color
    dcolor = get_dominant_color(dd, env_map_sph_coeffs)
    dcolor_rgb_color_difference = get_dominant_color(dd_rgb_color_difference, env_map_sph_coeffs)
    dcolor_rgb_luminance = get_dominant_color(dd_rgb_luminance, env_map_sph_coeffs)

    # Get Area Intensity
    cos_lobe_env_map = get_cos_lobe_as_env_map(H, W, device=env_map.device)
    cos_lobe_sph_coeffs, _ = project_env_to_coefficients(cos_lobe_env_map, l_max)

    area_intensity = get_area_normalization_term(env_map_sph_coeffs, cos_lobe_sph_coeffs, cartesian_direction=dd, l_max=l_max)
    area_intensity_rgb_color_difference = get_area_normalization_term(env_map_sph_coeffs, cos_lobe_sph_coeffs, cartesian_direction=dd_rgb_color_difference, l_max=l_max)
    area_intensity_rgb_luminance = get_area_normalization_term(env_map_sph_coeffs, cos_lobe_sph_coeffs, cartesian_direction=dd_rgb_luminance, l_max=l_max)

    return SPHMetrics(
        sph_coeffs=env_map_sph_coeffs,
        dominant_direction=dd,
        dominant_direction_rgb_color_difference=dd_rgb_color_difference,
        dominant_direction_rgb_luminance=dd_rgb_luminance,
        dominant_pixel=dpixel,
        dominant_pixel_rgb_color_difference=dpixel_rgb_color_difference,
        dominant_pixel_rgb_luminance=dpixel_rgb_luminance,
        dominant_color=dcolor,
        dominant_color_rgb_color_difference=dcolor_rgb_color_difference,
        dominant_color_rgb_luminance=dcolor_rgb_luminance,
        area_intensity=area_intensity,
        area_intensity_rgb_color_difference=area_intensity_rgb_color_difference,
        area_intensity_rgb_luminance=area_intensity_rgb_luminance)


def get_sph_metrics_cpu(env_map: torch.Tensor, l_max: int) -> SPHMetricsCPU:
    # Get GPU metrics first
    metrics = get_sph_metrics(env_map, l_max)
    
    # Convert all tensors to CPU and then to lists/floats
    cpu_metrics = SPHMetricsCPU(
        sph_coeffs=metrics.sph_coeffs.cpu().numpy().tolist(),
        dominant_direction=metrics.dominant_direction.cpu().numpy().tolist(),
        dominant_direction_rgb_color_difference=metrics.dominant_direction_rgb_color_difference.cpu().numpy().tolist(),
        dominant_direction_rgb_luminance=metrics.dominant_direction_rgb_luminance.cpu().numpy().tolist(),
        dominant_pixel=metrics.dominant_pixel.cpu().numpy().tolist(),
        dominant_pixel_rgb_color_difference=metrics.dominant_pixel_rgb_color_difference.cpu().numpy().tolist(),
        dominant_pixel_rgb_luminance=metrics.dominant_pixel_rgb_luminance.cpu().numpy().tolist(),
        dominant_color=metrics.dominant_color.cpu().numpy().tolist(),
        dominant_color_rgb_color_difference=metrics.dominant_color_rgb_color_difference.cpu().numpy().tolist(),
        dominant_color_rgb_luminance=metrics.dominant_color_rgb_luminance.cpu().numpy().tolist(),
        area_intensity=metrics.area_intensity.cpu().numpy().tolist(),
        area_intensity_rgb_color_difference=metrics.area_intensity_rgb_color_difference.cpu().numpy().tolist(),
        area_intensity_rgb_luminance=metrics.area_intensity_rgb_luminance.cpu().numpy().tolist()
    )
    
    return cpu_metrics


def visualize_sph_metrics(hdri: torch.Tensor, sph_metrics: Union[SPHMetrics, SPHMetricsCPU]) -> torch.Tensor:
    """
    Visualize the SPH metrics on the HDRI with colored circles around dominant pixels and a legend.
    
    Args:
        hdri: Input HDRI tensor (H, W, 3)
        sph_metrics: SPHMetrics or SPHMetricsCPU object containing dominant pixel information
    
    Returns:
        Visualization tensor with colored circles around dominant pixels and a legend
    """
    H, W, _ = hdri.shape
    vis_hdri = hdri.clone()
    
    # Define colors for the three dominant pixels
    red_color = torch.tensor([1.0, 0.0, 0.0], device=hdri.device, dtype=hdri.dtype)
    green_color = torch.tensor([0.0, 1.0, 0.0], device=hdri.device, dtype=hdri.dtype)  
    blue_color = torch.tensor([0.0, 0.0, 1.0], device=hdri.device, dtype=hdri.dtype)
    white_color = torch.tensor([1.0, 1.0, 1.0], device=hdri.device, dtype=hdri.dtype)
    
    # Get dominant pixel coordinates
    dominant_pixels = [
        (sph_metrics.dominant_pixel, red_color, "Dominant Pixel"),
        (sph_metrics.dominant_pixel_rgb_color_difference, green_color, "RGB Color Difference"),
        (sph_metrics.dominant_pixel_rgb_luminance, blue_color, "RGB Luminance")
    ]
    
    # Draw circles around each dominant pixel
    for pixel_coords, color, label in dominant_pixels:
        # Handle different pixel_coords types (tensor vs list)
        if isinstance(pixel_coords, torch.Tensor):
            pixel_x = max(0, min(W-1, int(pixel_coords[0].item())))
            pixel_y = max(0, min(H-1, int(pixel_coords[1].item())))
        else:
            # pixel_coords is a list (SPHMetricsCPU)
            pixel_x = max(0, min(W-1, int(pixel_coords[0])))
            pixel_y = max(0, min(H-1, int(pixel_coords[1])))
        
        # Draw colored circle (border) around the dominant pixel
        center_y, center_x = pixel_y, pixel_x
        
        # Store the original center pixel color
        original_center_color = vis_hdri[center_y, center_x, :].clone()
        
        # Draw circle with radius 3 (just the border)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                y_coord = center_y + dy
                x_coord = center_x + dx
                
                # Check bounds
                if 0 <= y_coord < H and 0 <= x_coord < W:
                    # Calculate distance from center
                    dist_sq = dy*dy + dx*dx
                    
                    # Draw circle border (distance 2.5 to 3.5)
                    if 6 <= dist_sq <= 12:  # Approximate circle border
                        vis_hdri[y_coord, x_coord, :] = color
        
        # Restore the center pixel with original color
        vis_hdri[center_y, center_x, :] = original_center_color
    
    # Create legend in the top-left corner
    legend_height = 60
    legend_width = 200
    legend_x_start = 10
    legend_y_start = 10
    
    # Ensure legend doesn't go out of bounds
    legend_x_end = min(W, legend_x_start + legend_width)
    legend_y_end = min(H, legend_y_start + legend_height)
    
    # Create semi-transparent white background for legend
    legend_alpha = 0.7
    for y in range(legend_y_start, legend_y_end):
        for x in range(legend_x_start, legend_x_end):
            if 0 <= y < H and 0 <= x < W:
                vis_hdri[y, x, :] = legend_alpha * white_color + (1 - legend_alpha) * vis_hdri[y, x, :]
    
    # Draw legend items (colored circles and text-like patterns)
    legend_items = [
        (red_color, "Dominant Pixel", 0),
        (green_color, "RGB Color Diff", 15),
        (blue_color, "RGB Luminance", 30)
    ]
    
    for color, label, y_offset in legend_items:
        legend_y = legend_y_start + 10 + y_offset
        legend_x = legend_x_start + 10
        
        # Draw small colored circle as legend marker
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y_coord = legend_y + dy
                x_coord = legend_x + dx
                
                if 0 <= y_coord < H and 0 <= x_coord < W:
                    dist_sq = dy*dy + dx*dx
                    if dist_sq <= 4:  # Small filled circle
                        vis_hdri[y_coord, x_coord, :] = color
        
        # Create simple text representation using colored pixels
        # This is a simplified version - in practice you might want to use actual text rendering
        text_start_x = legend_x + 15
        text_y = legend_y
        
        # Draw a simple line pattern to represent text
        for i in range(min(50, legend_x_end - text_start_x)):
            x_coord = text_start_x + i
            if 0 <= text_y < H and 0 <= x_coord < W:
                # Create a simple pattern to represent text
                if i % 8 < 6:  # Simple text-like pattern
                    vis_hdri[text_y, x_coord, :] = color * 0.8
    
    return vis_hdri

