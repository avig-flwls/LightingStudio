import os

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2
import torch
import numpy as np
from pathlib import Path


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


def exr_to_png(exr_path: str, png_path: str, gamma: float = 2.2, exposure: float = 0.0):
    """
    Convert an EXR file to PNG for web display.
    
    :param exr_path: path to the input EXR file
    :param png_path: path to write the PNG file to
    :param gamma: gamma correction value (default 2.2)
    :param exposure: exposure adjustment in stops (default 0.0)
    """
    # Read the EXR file
    image = cv2.imread(str(exr_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise ValueError(f"Could not read EXR file: {exr_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply exposure adjustment
    if exposure != 0.0:
        image_rgb = image_rgb * (2.0 ** exposure)
    
    # Apply tone mapping for HDR content
    # Simple tone mapping: clamp and gamma correct
    image_rgb = np.clip(image_rgb, 0.0, 1.0)
    
    # Apply gamma correction
    image_rgb = np.power(image_rgb, 1.0 / gamma)
    
    # Convert to 8-bit
    image_8bit = (image_rgb * 255).astype(np.uint8)
    
    # Convert back to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2BGR)
    
    # Write PNG
    cv2.imwrite(str(png_path), image_bgr)


def exr_to_png_tensor(image: torch.Tensor, png_path: str, gamma: float = 2.2, exposure: float = 0.0):
    """
    Convert a tensor image to PNG for web display.
    
    :param image: (H, W, 3) tensor
    :param png_path: path to write the PNG file to
    :param gamma: gamma correction value (default 2.2)
    :param exposure: exposure adjustment in stops (default 0.0)
    """
    # Convert to numpy
    image_np = image.cpu().numpy()
    
    # Apply exposure adjustment
    if exposure != 0.0:
        image_np = image_np * (2.0 ** exposure)
    
    # Apply tone mapping for HDR content
    # Simple tone mapping: clamp and gamma correct
    image_np = np.clip(image_np, 0.0, 1.0)
    
    # Apply gamma correction
    image_np = np.power(image_np, 1.0 / gamma)
    
    # Convert to 8-bit
    image_8bit = (image_np * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2BGR)
    
    # Write PNG
    cv2.imwrite(str(png_path), image_bgr)
