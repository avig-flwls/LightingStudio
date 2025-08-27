import os

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2
import torch


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
