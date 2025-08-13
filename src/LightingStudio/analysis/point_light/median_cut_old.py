# median_cut_hdri.py
# A Median Cut Algorithm for Light Probe Sampling (lat-long HDRI)
# Dependencies: numpy, imageio (pip install numpy imageio imageio[ffmpeg])


import sys
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# from __future__ import annotations
import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

import cv2

# --------- Utilities

def read_hdr_latlong(exr_path: str) -> np.ndarray:
    # """
    # Reads an HDR environment map in lat-long (equirectangular) layout.
    # Returns float32 array [H, W, 3] in linear RGB (assumed).
    # """
    # img = iio.imread(path).astype(np.float32)
    # if img.ndim == 2:
    #     img = np.stack([img, img, img], axis=-1)
    # if img.shape[-1] == 4:
    #     img = img[..., :3]
    # return img

    """
    Read in exr file as rgb.

    : return image: (H, W, 3)
    """
    image = cv2.imread(str(exr_path),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    H, W, C = image_rgb.shape
    assert(C == 3), f'The number of channels C:{C} >3 which is not possible...'

    return image_rgb
def luminance(rgb: np.ndarray) -> np.ndarray:
    # ITU-R BT.709 luminance
    return 0.2126 * rgb[...,0] + 0.7152 * rgb[...,1] + 0.0722 * rgb[...,2]

def pixel_solid_angles(h: int, w: int) -> np.ndarray:
    """
    Solid angle per pixel for a lat-long projection.
    Δθ = π/H, Δφ = 2π/W, pixel center at θ_i = (i+0.5)/H * π
    Δω(i) = Δθ * Δφ * sin(θ_i)
    Returned shape [H, 1] broadcastable to [H, W] (same for all columns in a row).
    """
    dtheta = math.pi / h
    dphi   = 2.0 * math.pi / w
    theta_centers = (np.arange(h, dtype=np.float32) + 0.5) * dtheta
    row_sa = (dtheta * dphi) * np.sin(theta_centers)
    row_sa = np.maximum(row_sa, 0.0).reshape(h, 1)  # avoid tiny negatives at poles
    return row_sa  # broadcast with [H, W]

def index_to_angles(i: np.ndarray, j: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel indices to spherical angles (θ, φ) at pixel centers.
    θ in [0, π], φ in [-π, π).
    """
    theta = ( (i + 0.5) / h ) * math.pi
    phi   = ( (j + 0.5) / w ) * (2.0 * math.pi) - math.pi
    return theta, phi

def angles_to_dirs(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Spherical to Cartesian directions, returns [..., 3].
    θ: polar (0 at +Y/up), φ: azimuth [-π, π) around +X axis with +Z forward if desired.
    Here we use Y-up convention:
      x =  sinθ cosφ
      y =  cosθ
      z =  sinθ sinφ
    """
    sin_t = np.sin(theta)
    dirs = np.stack([
        sin_t * np.cos(phi),
        np.cos(theta),
        sin_t * np.sin(phi)
    ], axis=-1)
    return dirs

# --------- Median Cut Core

@dataclass(order=True)
class Region:
    # The heap will maximize energy by using negative sort key
    sort_key: float
    y0: int
    y1: int
    x0: int
    x1: int
    energy: float

def cumulative_axis_sums(energy: np.ndarray, axis: int) -> np.ndarray:
    """
    Sum energy along the orthogonal axis to get a 1D marginal distribution.
    - If axis=0 (split rows), we want per-row totals: sum over columns -> shape [H]
    - If axis=1 (split cols), we want per-column totals: sum over rows   -> shape [W]
    """
    if axis == 0:
        return energy.sum(axis=1)
    else:
        return energy.sum(axis=0)

def find_median_cut_index(marginal: np.ndarray) -> int:
    """
    Given non-negative 1D marginal energy, find split index k such that
    sum[0:k] ≈ sum[k:].
    Returns k in [1, len-1] to ensure non-empty halves.
    """
    cumsum = np.cumsum(marginal)
    total = cumsum[-1]
    if total <= 0.0:
        # fallback to middle
        return max(1, len(marginal)//2)
    half = total * 0.5
    k = int(np.searchsorted(cumsum, half))
    k = int(np.clip(k, 1, len(marginal)-1))
    return k

def split_region_by_median(energy: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    """
    Split the rectangle [y0:y1, x0:x1] by median energy along the longer axis (in pixels).
    Returns two child rectangles (inclusive-exclusive bounds).
    """
    h = y1 - y0
    w = x1 - x0
    # Choose split axis: longer side (in pixels). If equal, choose the axis with larger energy variance marginally.
    if w > h:
        axis = 1  # split columns
    elif h > w:
        axis = 0  # split rows
    else:
        # tie-breaker by marginal variance
        row_marg = cumulative_axis_sums(energy[y0:y1, x0:x1], axis=0)  # per-row
        col_marg = cumulative_axis_sums(energy[y0:y1, x0:x1], axis=1)  # per-col
        axis = 0 if np.var(row_marg) >= np.var(col_marg) else 1

    if axis == 0:
        # Split rows
        marg = cumulative_axis_sums(energy[y0:y1, x0:x1], axis=0)
        k = find_median_cut_index(marg)
        a = (y0, y0 + k, x0, x1)
        b = (y0 + k, y1, x0, x1)
    else:
        # Split cols
        marg = cumulative_axis_sums(energy[y0:y1, x0:x1], axis=1)
        k = find_median_cut_index(marg)
        a = (y0, y1, x0, x0 + k)
        b = (y0, y1, x0 + k, x1)
    return a, b

def region_stats(rgb: np.ndarray, lum: np.ndarray, sa: np.ndarray,
                 y0: int, y1: int, x0: int, x1: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute region aggregates:
      - total RGB power: sum(L_rgb * Δω)
      - energy for splitting: sum(luminance * Δω)
      - energy-weighted average direction (unit)
      - region solid angle: sum(Δω)
    """
    # Slices
    rgb_r = rgb[y0:y1, x0:x1, :]
    lum_r = lum[y0:y1, x0:x1]
    sa_r  = sa[y0:y1, x0:x1]

    # Energy and power
    energy = float(np.sum(lum_r * sa_r))          # scalar
    power_rgb = np.sum(rgb_r * sa_r[..., None], axis=(0,1))  # [3]
    region_sa = float(np.sum(sa_r))

    # Direction weighted by luminance * Δω
    H = rgb.shape[0]
    W = rgb.shape[1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    theta, phi = index_to_angles(ys.astype(np.float32), xs.astype(np.float32), H, W)
    dirs = angles_to_dirs(theta, phi)  # [h,w,3]
    wts = (lum_r * sa_r)[..., None]    # [h,w,1]
    dir_vec = np.sum(dirs * wts, axis=(0,1))
    norm = np.linalg.norm(dir_vec) + 1e-12
    dir_unit = dir_vec / norm

    return power_rgb.astype(np.float32), dir_unit.astype(np.float32), energy, region_sa

def median_cut_sampling(hdri: np.ndarray, n_samples: int) -> List[dict]:
    """
    Perform median cut sampling on a lat-long HDRI.
    Returns a list of light samples:
      {
        'direction': [x,y,z],        # unit vector
        'power_rgb': [r,g,b],        # ∑ L_rgb(ω) dω over region
        'region_solid_angle': float, # ∑ dω over region
        'avg_radiance_rgb': [r,g,b], # power / solid angle (useful for debugging)
        'rect': (y0,y1,x0,x1)        # pixel bounds of region
      }
    """
    H, W, _ = hdri.shape
    sa_rows = pixel_solid_angles(H, W)     # [H,1]
    sa = np.broadcast_to(sa_rows, (H, W))  # [H,W]
    lum = luminance(hdri)                  # [H,W]

    # region energy map used for splitting
    energy_map = lum * sa                  # [H,W]

    # Initialize heap with full image region
    power_rgb, dir_unit, energy, region_sa = region_stats(hdri, lum, sa, 0, H, 0, W)
    heap: List[Region] = []
    heapq.heappush(heap, Region(sort_key=-energy, y0=0, y1=H, x0=0, x1=W, energy=energy))

    regions: List[Tuple[int,int,int,int]] = []

    # Greedily split the most energetic region until we have n_samples regions
    while len(regions) + len(heap) < n_samples:
        if not heap:
            break  # nothing left to split
        r = heapq.heappop(heap)
        a, b = split_region_by_median(energy_map, r.y0, r.y1, r.x0, r.x1)

        # push children with their energies
        for (yy0, yy1, xx0, xx1) in (a, b):
            e_child = float(np.sum(energy_map[yy0:yy1, xx0:xx1]))
            heapq.heappush(heap, Region(sort_key=-e_child, y0=yy0, y1=yy1, x0=xx0, x1=xx1, energy=e_child))

    # Collect all remaining regions in heap; if we already reached n via exact splits, also collect.
    while heap and len(regions) < n_samples:
        r = heapq.heappop(heap)
        regions.append((r.y0, r.y1, r.x0, r.x1))

    # Produce light samples
    samples = []
    for (y0, y1, x0, x1) in regions:
        p_rgb, d_unit, E, reg_sa = region_stats(hdri, lum, sa, y0, y1, x0, x1)
        avg_L = p_rgb / max(reg_sa, 1e-12)
        samples.append({
            'direction': d_unit.tolist(),
            'power_rgb': p_rgb.tolist(),
            'region_solid_angle': float(reg_sa),
            'avg_radiance_rgb': avg_L.tolist(),
            'rect': (int(y0), int(y1), int(x0), int(x1)),
        })
    return samples

# --------- Simple CLI for testing

# if __name__ == "__main__":
#     import argparse, json, os

#     ap = argparse.ArgumentParser(description="Median Cut Light Probe Sampling")
#     ap.add_argument("input", help="Path to HDRI (EXR/HDR) in lat-long format")
#     ap.add_argument("-n", "--n_samples", type=int, default=16, help="Number of lights to generate")
#     ap.add_argument("-o", "--out_json", type=str, default="", help="Where to write samples JSON")
#     args = ap.parse_args()

#     env = read_hdr_latlong(args.input)
#     samples = median_cut_sampling(env, args.n_samples)

#     out = {
#         "source": os.path.basename(args.input),
#         "n_samples": len(samples),
#         "samples": samples
#     }
#     if args.out_json:
#         with open(args.out_json, "w", encoding="utf-8") as f:
#             json.dump(out, f, indent=2)
#         print(f"Wrote {args.out_json}")
#     else:
#         print(json.dumps(out, indent=2))

def dir_to_pixel(direction, H, W):
    # direction is (x,y,z) with y-up
    x, y, z = direction
    theta = np.arccos(np.clip(y, -1.0, 1.0))  # [0, π]
    phi = np.arctan2(z, x)                    # [-π, π)
    u = int(((phi + np.pi) / (2 * np.pi)) * W) % W
    v = int((theta / np.pi) * H)
    return u, v

if __name__ == "__main__":
    import argparse, json, os

    ap = argparse.ArgumentParser(description="Median Cut Light Probe Sampling")
    ap.add_argument("input", help="Path to HDRI (EXR/HDR) in lat-long format")
    ap.add_argument("-n", "--n_samples", type=int, default=16, help="Number of lights to generate")
    ap.add_argument("-o", "--out_json", type=str, default="", help="Where to write samples JSON")
    ap.add_argument("-v", "--vis_path", type=str, default="", help="Optional output EXR/PNG for visualization")
    args = ap.parse_args()

    env = read_hdr_latlong(args.input)
    samples = median_cut_sampling(env, args.n_samples)

    out = {
        "source": os.path.basename(args.input),
        "n_samples": len(samples),
        "samples": samples
    }
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.out_json}")
    else:
        print(json.dumps(out, indent=2))

    # Visualization
    if args.vis_path:
        vis_img = env.copy()
        H, W, _ = vis_img.shape
        for s in samples:
            u, v = dir_to_pixel(s['direction'], H, W)
            color = (1.0, 0.0, 0.0)  # red in HDR
            px_color = tuple(int(c*255) for c in color)  # for PNG 0-255
            if args.vis_path.lower().endswith(".exr"):
                # For HDR: draw directly with float RGB
                radius = 5
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        if dx*dx + dy*dy <= radius*radius:
                            vv = np.clip(v+dy, 0, H-1)
                            uu = (u+dx) % W
                            vis_img[vv, uu] = color
            else:
                # For LDR: convert to 8-bit for OpenCV drawing
                ldr = np.clip(vis_img / np.max(vis_img), 0, 1)
                ldr = (ldr * 255).astype(np.uint8)
                cv2.circle(ldr, (u, v), 5, (0,0,255), -1)  # BGR red
                vis_img = ldr

        if args.vis_path.lower().endswith(".exr"):
            cv2.imwrite(args.vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(args.vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"Wrote visualization to {args.vis_path}")