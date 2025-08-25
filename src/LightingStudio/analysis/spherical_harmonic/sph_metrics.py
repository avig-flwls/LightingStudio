import torch
from einops import rearrange, repeat

from src.LightingStudio.analysis.spherical_harmonic.sph import project_env_to_coefficients, project_direction_into_coefficients, sph_l_max_from_indices_total, l_from_index, sph_indices_total
from src.LightingStudio.analysis.utils import cartesian_to_spherical, convert_theta, generate_spherical_coordinates_map


def get_dominant_direction(sph_coeffs: torch.Tensor) ->  tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TODO: think about what about returning dominant direction in r, g, b directly and not adding them together???

    Get the dominant direction from the spherical harmonic coefficients.
    
    :params sph_coeffs: (n_terms, 3) where each column is r, g, b

    :return dominant_direction: (3) the dominant direction in xyz coordinate 
    :return dominant_direction_normalized: (3) the dominant direction (with vector norm)
    :return dominant_direction_rgb: (3) the dominant direction with (rgb scaling constants source unknown...)
    :return dominant_direction_rgb_normalized: (3) the dominant direction rgb (with vector norm)
    
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
    red_band_aligned_xyz = torch.tensor([-red_band_1[2],-red_band_1[0],red_band_1[1]])
    green_band_aligned_xyz = torch.tensor([-green_band_1[2],-green_band_1[0],green_band_1[1]])
    blue_band_aligned_xyz = torch.tensor([-blue_band_1[2],-blue_band_1[0],blue_band_1[1]])


    dominant_direction = red_band_aligned_xyz + blue_band_aligned_xyz + green_band_aligned_xyz
    
    red_constant = 0.3
    green_constant = 0.59
    blue_constant = 0.11

    dominant_direction_rgb =    red_constant * dominant_direction + \
                                green_constant * green_band_aligned_xyz + \
                                blue_constant * blue_band_aligned_xyz

    return (dominant_direction, 
            dominant_direction/torch.linalg.norm(dominant_direction), 
            dominant_direction_rgb, 
            dominant_direction_rgb/torch.linalg.norm(dominant_direction_rgb))

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
    denominator = np.dot(direction_sph_coeffs, direction_sph_coeffs)

    color = torch.tensor([torch.dot(sph_coeffs_r, direction_sph_coeffs) / denominator,
                      torch.dot(sph_coeffs_g, direction_sph_coeffs) / denominator,
                      torch.dot(sph_coeffs_b, direction_sph_coeffs) / denominator])
    
    return color


def get_approximate_env_map(H:int, W:int, cartesian_direction: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """
    Approximate directional light as if it were in an environment map.
    
    This means it is black (0's) for all pixels except for one which has all the color. It does
    this by find the pixel index that the cartesian_direction maps to.
    
    : params H:
    : params W:
    : params cartesian_direction: (3)
    
    : returns env_map: (H, W, 3)
    """

    cartesian_direction = rearrange(cartesian_direction, "c -> 1 c")    # (1, 3)
    spherical_direction = cartesian_to_spherical(cartesian_direction)   # (1, 2)

    theta = spherical_direction[0,0]
    phi = spherical_direction[0,1]

    theta_divisor = torch.pi / H
    phi_divisor = 2.0*torch.pi / W

    # We just assume that that the cell that the remainder falls in, is the one that has the color.
    if (theta > torch.pi/2) or (theta < -torch.pi/2):
        raise ValueError(f'theta: {theta} should be between [-pi/2, pi/2]')

    theta_cell = int(torch.clip(int(H/2) - torch.sign(theta)* torch.abs(theta) / theta_divisor, 0, H - 1)) # NOTE: there is negative sign here because, positive theta angle means we want to move "up" in elevation. 
    phi_cell = int(torch.clip(int(W/2) + torch.sign(phi)* torch.abs(phi) / phi_divisor, 0, W - 1))

    env_map = torch.zeros((H, W, len(color)))
    env_map[theta_cell, phi_cell] = color

    return env_map


def get_cos_lobe_as_env_map(H:int, W:int) -> torch.Tensor:
    """
    The value of N dot L placed in an environment map.
    
    : returns env_map_as_cos_lobe: (H, W, 3)

    Source:
    [7] cosine_lobe definition.
    """

    spherical_coordinates = generate_spherical_coordinates_map(H,W) # (H, W, 2)
    theta = spherical_coordinates[..., 0]                           # (H, W)
    theta = convert_theta(theta)                                    # (H, W)

    cos_lobe = torch.maximum(0, torch.cos(theta))  # (H, W)
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
    [6] RenderDiffuseIrradiance function.
    [7] Equation 6.
    """

    # Project dominant direction into sph_coeffs.
    dominant_direction = rearrange(cartesian_direction, "c -> 1 c")
    direction_sph_coeffs = project_direction_into_coefficients(dominant_direction, l_max) # (1, n_terms)
    direction_sph_coeffs = rearrange(direction_sph_coeffs, "1 n_terms -> n_terms") # (n_terms)

    # TODO check that here cos_lobe_sph_coeffs should be the same for each l band. pass!

    # TODO something needs to be repeated here!!!! after discussion with Jiahao

    scaled_direction_sph_coeffs = torch.zeros_like(direction_sph_coeffs)
    for i in range(sph_indices_total(l_max)):
        l = l_from_index(i)  # noqa: E741
        scale_factor = torch.sqrt(4*torch.pi/ (2 * l + 1))
        scaled_direction_sph_coeffs[i] = scale_factor * direction_sph_coeffs[i]

    scaled_repeated_direction_sph_coeffs = repeat(scaled_direction_sph_coeffs, "n_terms -> n_terms c", c=3)
    rotated_cos_lobe_sph_coeffs = torch.multiply(scaled_repeated_direction_sph_coeffs, cos_lobe_sph_coeffs) # (n_terms, 3)

    # The integral of the product of two spherical harmonic functions is equivalent to the dot product of their coefficients
    area_intensity = torch.sum(torch.multiply(env_map_sph_coeffs, rotated_cos_lobe_sph_coeffs), axis=0)  # (3)
    return area_intensity

def get_point_normalization_term(dominant_direction: torch.Tensor) -> torch.Tensor:
    """
    WARNING: This was not implemented because [0,1] lighting was not being used...


    Find the "normalization term"
    
    
    Definition:
        Mathematically:

        
        Physically:


    
    Steps:
        1. 


    : params dominant_direction: (3,)
    : return normalization_term: (3,) one for each color (r, g, b) = (c_light_r, c_light_g, c_light_b)

    

    Source:
    [8] Extracting dominant light intensity section
    [10] Punctual Light Sources section and Equation 9
    """

    optional_lambertian_scalar = 1/torch.pi
    c = 1

    return c

def extract_single_light_no_ambient(env_map: torch.Tensor, l_max: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Single Light Direction, Single Light Intensity, Single Light Color (SLNA).

    We are constructing a new lighting source.
    When it is created, the light source has direction: dominant_direction.
    But it has no color. So we need to give it a color.

    Steps:
    1. Get Dominant Light Direction
    (SKIP) 2. Get Normalization Factor
    3. Get Dominant Light Color

    
    : params

    : returns chosen_direction: (3) this is the dominant direction (which might be rgb and normalized)
    : returns color: (3) This color is NOT between [0, 1]
    : returns approx_env_map: (H, W, 3) approximate directional light as if it were in an environment map 

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

    # if l_max > 2:
    #     raise ValueError(f"l_max for Lighting Analysis is only supported for l_max: {l_max} <= 2")

    # Get Direction
    env_map_sph_coeffs, _ = project_env_to_coefficients(env_map, l_max)
    dd, dd_norm, dd_rgb, dd_rgb_norm = get_dominant_direction(env_map_sph_coeffs)
    chosen_direction = dd_rgb_norm

    # Get Color
    color = get_dominant_color(dominant_direction=chosen_direction, env_map_sph_coeffs=env_map_sph_coeffs)
    
    # Get Intensity
    H, W, _ = env_map.shape
    cos_lobe_env_map = get_cos_lobe_as_env_map(H, W)

    cos_lob_sph_coeffs, _ = project_env_to_coefficients(cos_lobe_env_map, l_max)
    area_intensity = get_area_normalization_term(env_map_sph_coeffs=env_map_sph_coeffs, cos_lobe_sph_coeffs=cos_lob_sph_coeffs, cartesian_direction=chosen_direction, l_max=l_max)
    # single_intensity  = 

    # Approximate Env Map
    H, W, _ = env_map.shape
    approx_env_map = get_approximate_env_map(H, W, chosen_direction, color)

    return chosen_direction, color, area_intensity, approx_env_map
