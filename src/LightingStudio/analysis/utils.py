import sys
import os

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2
import numpy as np
import torch
from einops import repeat

def read_exrs(exr_paths: list[str]) -> torch.Tensor:
    """
    Read in list of exr files as rgb.

    : return image: (B, H, W, 3)
    """
    images = []
 
    for i, exr_path in enumerate(exr_paths):
        image = cv2.imread(str(exr_path),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W, C = image_rgb.shape
        assert(C == 3), f'The number of channels C:{C} >3 which is not possible...'
        images.append(torch.from_numpy(image_rgb))

    return torch.stack(images, dim=0)

def write_exr(image: torch.Tensor, exr_path: str):
    """
    Write an image to an exr file.

    :param image: (H, W, 3)
    :param exr_path: path to write the exr file to
    """
    cv2.imwrite(exr_path, cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2BGR))


def luminance(batch_rgb: torch.Tensor) -> torch.Tensor:
    """
    Compute the luminance of an rgb image.

    :param rgb: (..., 3)
    :return: (...)
    """
    # ITU-R BT.709 luminance
    return 0.2126 * batch_rgb[...,0] + 0.7152 * batch_rgb[...,1] + 0.0722 * batch_rgb[...,2]


def pixel_solid_angles(H:int, W:int, device: torch.device = None) -> torch.Tensor:
    """
    Compute the solid angle of a pixel.
    In spherical coordinates there is a formula for the differential, d Ω = sin ⁡ θ d θ d φ , where θ is the colatitude (angle from the North Pole).
    
    Definition:
    1. https://community.zemax.com/people-pointers-9/how-to-calculate-the-solid-angle-for-a-pixel-in-angle-space-2376
    2. https://en.wikipedia.org/wiki/Solid_angle 

    :param H: height
    :param W: width
    :param device: torch device to place the result tensor on
    :return: (H, 1)

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

    theta_centers = torch.tensor(np.arange(H, dtype=np.float32) + 0.5, device=device) # (H)
    solid_angle = torch.maximum(theta_centers * pixel_area, torch.tensor(0.0, device=device)) # (H,) # Avoid tiny negatives at poles
    
    return solid_angle.unsqueeze(-1)  # (H, 1)

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
    theta_s = torch.tensor(np.arange(H)/H, device=device) * theta_range + theta_offset # (H)
    theta_s = -1.0*theta_s                                              # make sure that the top left corner is +pi/2 for theta
    phi_s = torch.tensor(np.arange(W)/W, device=device) * phi_range + phi_offset       # (W)

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





