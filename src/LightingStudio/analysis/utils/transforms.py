import sys
import numpy as np
import torch
from einops import repeat


def luminance(batch_rgb: torch.Tensor) -> torch.Tensor:
    """
    Compute the luminance of an rgb image.

    :param rgb: (..., 3)
    :return: (...)
    """
    # ITU-R BT.709 luminance
    return 0.2126 * batch_rgb[...,0] + 0.7152 * batch_rgb[...,1] + 0.0722 * batch_rgb[...,2]


def pixel_solid_angles(H:int, W:int, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the solid angle of a pixel.
    In spherical coordinates there is a formula for the differential, d Ω = sin ⁡ θ d θ d φ , where θ is the colatitude (angle from the North Pole).
    
    Definition:
    1. https://community.zemax.com/people-pointers-9/how-to-calculate-the-solid-angle-for-a-pixel-in-angle-space-2376
    2. https://en.wikipedia.org/wiki/Solid_angle 

    :param H: height
    :param W: width
    :param device: torch device to place the result tensor on
    :return: pixel_area (1), sin_theta: (H, 1)

    Source:
    [5] getSolidAngle function
    [6] ProjectEnvironment function
    [3] Page 16   
    [11] definition
    [6] ImageYToTheta function   
    """
    theta_range = np.pi
    phi_range = 2.0 * np.pi
    pixel_area = (theta_range / H) * (phi_range / W)

    theta_centers = torch.pi * (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    sin_theta = torch.maximum(torch.sin(theta_centers), torch.tensor(0.0, device=device)) # (H,) # Avoid tiny negatives at poles

    return pixel_area, sin_theta.unsqueeze(-1)  # (H, 1)

def convert_theta(theta:torch.Tensor) -> torch.Tensor:
    """
    Convert elevation angle to zenith angle (colatitude).
    
    Elevation angle: measured from horizontal plane, positive upward
        - π/2 at zenith (straight up)
        - 0 at horizon
        - -π/2 at nadir (straight down)
    
    Zenith angle (colatitude): measured from z-axis downward
        - 0 at zenith (straight up)
        - π/2 at horizon
        - π at nadir (straight down)

    : params theta: elevation angle in radians
    : returns new_theta: zenith angle (colatitude) in radians

    Source:
    [11] definition
    [6] ImageYToTheta function
    """
    
    # Convert from elevation to zenith angle
    new_theta = torch.pi/2.0 - theta
    return new_theta

def generate_spherical_coordinates_map(H:int, W:int, device: torch.device = None) -> torch.Tensor:
    """
    Create map of size (H, W, 2), where at each pixel value we know the theta and phi value.
    
    PolyHaven Environment Map (artist):

                            +pi/2   +-------------------------------------------+
                                    |+Z        +Z        +Z       +Z          +Z|               
                                    |                                           |
                                    |                                           |
    Elevation (theta)          0    |-X        +Y        +X       -Y          -X|
                                    |                                           |
                                    |                                           |
                                    |-Z        -Z        -Z       -Z          -Z|
                            -pi/2   +-------------------------------------------+
                                -pi        -pi/2        0     +pi/2           pi

                                                    Azimuthal (phi) 

        
    Mathematical (science) (this is what is returned):
                            +pi/2   +-------------------------------------------+
                                    |+Z        +Z        +Z       +Z          +Z|               
                                    |                                           |
                                    |                                           |
    Elevation (theta)          0    |-X        -Y        +X       +Y          -X|
                                    |                                           |
                                    |                                           |
                                    |-Z        -Z        -Z       -Z          -Z|
                            -pi/2   +-------------------------------------------+
                                -pi        -pi/2        0     +pi/2           pi

                                                    Azimuthal (phi) 



                Numpy/OpenCV:
                                (0,0)       (0,W)
                                    +-------+
                                    |       |
                                    +-------+
                                (H,0)       (H,W)

                                
                Nuke:
                                (H,0)       (W,H)
                                    +-------+
                                    |       |
                                    +-------+
                                (0,0)       (W,0)

    :params H: height
    :params W: width

    :return spherical_coordinates: (H, W, 2)

    Source: [5] xy2ll function
    """

    # Setup 
    theta_range = np.pi
    phi_range   = 2.0 * np.pi

    theta_offset = -np.pi/2.0
    phi_offset = -np.pi

    # Create Map 
    theta_s = torch.tensor(np.arange(H)/H, device=device, dtype=torch.float32) * theta_range + theta_offset # (H)
    theta_s = -1.0*theta_s                                              # make sure that the top left corner is +pi/2 for theta
    phi_s = torch.tensor(np.arange(W)/W, device=device, dtype=torch.float32) * phi_range + phi_offset       # (W)

    # Create coordinate map directly using broadcasting and stack
    theta_map = repeat(theta_s, "h -> h w", w=W)     # (H, W)
    phi_map = repeat(phi_s, "w -> h w", h=H)         # (H, W)
    
    spherical_coordinates = torch.stack([theta_map, phi_map], dim=-1)  # (H, W, 2)

    return spherical_coordinates

def spherical_to_cartesian(spherical_coordinates: torch.Tensor) -> torch.Tensor:
    """
    Convert from spherical coordinates to cartesian coordinates.
    
    :params spherical_coordinates (..., 2) where at each position on the image, we have the spherical coordinate on the sphere.
    :returns cartesian_coordinates (..., 3) 

    Sources: [1] sph2cart function
    """
    theta, phi = spherical_coordinates[..., 0], spherical_coordinates[..., 1]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    # Construct cartesian coordinates directly using stack
    x = cos_theta * cos_phi   # x
    y = cos_theta * sin_phi   # y
    z = sin_theta             # z
    
    cartesian_coordinates = torch.stack([x, y, z], dim=-1)

    # Sanity Check
    radial_length = torch.sqrt(torch.sum(cartesian_coordinates**2, dim=-1))
    assert torch.allclose(radial_length, torch.ones_like(radial_length)), f'Radial length not close to 1.0'

    return cartesian_coordinates

def cartesian_to_spherical(cartesian_coordinates: torch.Tensor) -> torch.Tensor:
    """
    Convert from cartesian coordinates to spherical coordinates.
    
    :params cartesian_coordinates (..., 3) where at each position on the image, we have the cartesian coordinate on the sphere.
    :returns spherical_coordinates (..., 2) 

    Sources: [1] cart2sph function
    """
    x, y, z = cartesian_coordinates[..., 0], cartesian_coordinates[..., 1], cartesian_coordinates[..., 2]
    radial_length = torch.sqrt(torch.sum(cartesian_coordinates**2, dim=-1))
    sqrt_x_2_y_2 = torch.sqrt(x**2 + y**2)

    assert torch.allclose(radial_length, torch.ones_like(radial_length)), f'Radial length not close to 1.0'

    # Construct spherical coordinates directly using stack
    theta = torch.arctan2(z, sqrt_x_2_y_2)  # theta (elevation)
    phi = torch.arctan2(y, x)               # phi (azimuthal)
    
    spherical_coordinates = torch.stack([theta, phi], dim=-1)

    return spherical_coordinates


def spherical_to_pixel(spherical_coordinates: torch.Tensor, H:int, W:int) -> torch.Tensor:
    """
    Convert from spherical coordinates to pixel coordinates.
    Undo the mapping from generate_spherical_coordinates_map

    :params spherical_coordinates (..., 2) where at each position on the image, we have the spherical coordinate on the sphere.
    :params H: height
    :params W: width
    :returns pixel_coordinates (..., 2)
    """
    theta, phi = spherical_coordinates[..., 0], spherical_coordinates[..., 1]

    # TODO: Document this somewhere, that we don't need to actually account for the artist's definition of the enviornment map??
    # # Flip x to account for the artist's definition of the enviornment map
    # phi = -1.0 * phi

    # Flip y to account for the fact that (0,0) is the top left corner
    theta = -1.0 * theta

    x = torch.round(torch.remainder(torch.div(phi + np.pi, 2 * np.pi) * W, W)).to(torch.int32) # (..., 1)
    y = torch.round(torch.remainder(torch.div(theta + np.pi/2.0, np.pi) * H, H)).to(torch.int32) # (..., 1)

    pixel_coordinates = torch.stack([x, y], dim=-1)

    return pixel_coordinates


def cartesian_to_pixel(cartesian_coordinates: torch.Tensor, H:int, W:int) -> torch.Tensor:
    """
    Convert from cartesian coordinates to pixel coordinates.
    Undo the mapping from generate_spherical_coordinates_map

    :params cartesian_coordinates (..., 3) where at each position on the image, we have the cartesian coordinate on the sphere.
    :params H: height
    :params W: width
    :returns pixel_coordinates (..., 2)
    """
    spherical_coordinates = cartesian_to_spherical(cartesian_coordinates)
    pixel_coordinates = spherical_to_pixel(spherical_coordinates, H, W)

    return pixel_coordinates    





