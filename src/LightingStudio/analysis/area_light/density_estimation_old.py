"""
Median Cut Algorithm for Light Probe Sampling
https://vgl.ict.usc.edu/Research/MedianCut/MedianCutSampling.pdf

This module implements the median cut algorithm for sampling HDRI environment maps.
It uses PyTorch throughout for GPU acceleration and consistency with the analysis utilities.
"""


# expand_map.py
# Implements the "Expand Map" (density estimation) from Banterle et al., Graphite 2006.
# Sec. 2.3, Eqs. (15)-(17). Defaults match the paper: rt=16, gamma=0.918, beta=1.953.
# Requires: numpy; optional: scipy (for exact KD-tree mode); OpenCV for seed helpers.

import numpy as np
import cv2

# ---------- Helpers to get seeds P = {(x_p, y_p), Psi_p} ----------

def seeds_from_median_cut_regions(lum, regions, psi_mode="sum"):
    """
    Build seed points from rectangular regions (e.g., median-cut output).
    Each region -> one seed at its luminance-weighted centroid with Psi set from region.
      lum:        float32 luminance image HxW
      regions:    list of (y0,y1,x0,x1)
      psi_mode:   'sum' (paper's "light color sum" spirit) or 'mean' or 'peak'

    Returns:
      seeds_xy: float32 Nx2 array in (x, y)
      psi:      float32 N array with luminance values (Psi_p)
    """
    H, W = lum.shape
    pts, vals = [], []
    for (y0, y1, x0, x1) in regions:
        sub = lum[y0:y1, x0:x1]
        if sub.size == 0 or np.all(sub <= 0):
            continue
        # luminance-weighted centroid in pixel coords
        ys, xs = np.mgrid[y0:y1, x0:x1]
        w = sub
        wsum = np.sum(w)
        cy = np.sum(ys * w) / (wsum + 1e-12)
        cx = np.sum(xs * w) / (wsum + 1e-12)

        if psi_mode == "sum":
            psi_p = float(np.sum(sub))
        elif psi_mode == "mean":
            psi_p = float(np.mean(sub))
        elif psi_mode == "peak":
            psi_p = float(np.max(sub))
        else:
            raise ValueError("psi_mode must be 'sum', 'mean', or 'peak'.")

        pts.append((cx, cy))
        vals.append(psi_p)
    if not pts:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32), np.asarray(vals, dtype=np.float32)


def seeds_from_luminance_threshold(lum, thresh_percentile=98, nms_radius=3):
    """
    Simpler seed extractor: pick bright LOCAL MAXIMA above a percentile.
    Returns (seeds_xy, psi) with Psi_p = luminance at the peak.
    """
    thr = np.percentile(lum, thresh_percentile)
    mask = lum >= thr
    # non-maximum suppression via grayscale dilation
    k = (2 * nms_radius + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(lum, kernel)
    peaks = (lum == dil) & mask

    ys, xs = np.where(peaks)
    psi = lum[ys, xs].astype(np.float32)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    return pts, psi


# ---------- Exact Expand Map (per-pixel r_max, Eq. 16-17) ----------

def expand_map_exact(lum, seeds_xy, psi, rt=16, gamma=0.918, beta=1.953,
                     min_count=4, normalize=True, tile=128):
    """
    Faithful implementation of Sec. 2.3 with per-pixel r_max (Eq. 16-17).
    This can be slow on large images; use tile processing + SciPy KD-tree.

    Args:
      lum:        HxW float32 luminance (not directly used in Eqs. 15-17, but kept for symmetry)
      seeds_xy:   Nx2 float32 seed positions (x, y)
      psi:        N   float32 luminance values Psi_p for each seed
      rt:         radius of influence in pixels (paper used 16)
      gamma, beta:Gaussian kernel parameters (paper: 0.918, 1.953)
      min_count:  minimum #seeds in neighborhood A_x to accept; else weight->0 (paper suggests 4-6)
      normalize:  scale result to [0,1] by dividing by max
      tile:       tile size for chunked processing

    Returns: HxW float32 Expand Map in [0,1] if normalize=True
    """
    try:
        from scipy.spatial import cKDTree
    except Exception as e:
        raise RuntimeError("expand_map_exact requires SciPy (scipy.spatial.cKDTree). "
                           "Install SciPy or use expand_map_fast().") from e

    H, W = lum.shape
    out = np.zeros((H, W), dtype=np.float32)
    if len(seeds_xy) == 0:
        return out

    tree = cKDTree(seeds_xy)  # seeds are (x,y)
    inv_one_minus_exp_beta = 1.0 / (1.0 - np.exp(-beta) + 1e-12)

    # Process in tiles to limit memory
    for y0 in range(0, H, tile):
        y1 = min(y0 + tile, H)
        for x0 in range(0, W, tile):
            x1 = min(x0 + tile, W)
            # Query all pixels in the tile at once (KD-tree supports vectorized query_ball_point)
            xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            qpts = np.stack([xs.ravel().astype(np.float32), ys.ravel().astype(np.float32)], axis=1)
            neigh = tree.query_ball_point(qpts, r=rt)  # list of arrays (variable length)

            tile_vals = np.zeros(qpts.shape[0], dtype=np.float32)
            for idx, nn in enumerate(neigh):
                if len(nn) < min_count:
                    continue
                qx, qy = qpts[idx]
                # distances to neighbors
                xy = seeds_xy[nn]  # Kx2
                dx = xy[:, 0] - qx
                dy = xy[:, 1] - qy
                d2 = dx*dx + dy*dy
                rmax2 = np.max(d2)
                if rmax2 <= 1e-12:  # degenerate: all neighbors on the pixel
                    # All weights collapse to gamma * [1 - (1 - e^0)/(1 - e^-beta)] = gamma
                    w = gamma
                    tile_vals[idx] = w * np.sum(psi[nn])
                    continue

                # Eq. 16: w_g^p = gamma * [1 - (1 - exp(-beta * d^2 / (2*rmax^2))) / (1 - exp(-beta))]
                a = -beta * d2 / (2.0 * rmax2 + 1e-12)
                wgp = gamma * (1.0 - (1.0 - np.exp(a)) * inv_one_minus_exp_beta)

                # Eq. 17: Lambda = sum Psi_p * w_g^p
                tile_vals[idx] = float(np.dot(psi[nn], wgp))

            out[y0:y1, x0:x1] = tile_vals.reshape((y1 - y0), (x1 - x0))

    if normalize and np.max(out) > 0:
        out = out / np.max(out)
    return out


# ---------- Fast approximation (convolution with fixed r_max=rt) ----------

def expand_map_fast(lum, seeds_xy, psi, rt=16, gamma=0.918, beta=1.953,
                    min_count=4, normalize=True, border=cv2.BORDER_REPLICATE):
    """
    Much faster approximation: assume r_max == rt for all pixels.
    Then Eq. 16 weights depend only on distance d, so the map is
    convolution of an impulse image (Psi at seeds) with a fixed kernel.

    Also computes a neighbor-count map with a binary disc kernel and
    zeros out pixels with count < min_count, as suggested in the paper.

    Args:
      lum:        HxW float32 luminance
      seeds_xy:   Nx2 float32 seed positions (x,y)
      psi:        N float32 Psi_p values
      rt:         radius of influence
      gamma,beta: kernel parameters
      min_count:  neighbor threshold in A_x
      normalize:  scale to [0,1]
      border:     OpenCV border mode for filter2D

    Returns: HxW float32 Expand Map
    """
    H, W = lum.shape
    if len(seeds_xy) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # Impulse image with Psi at seeds
    imp = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    xs = np.clip(np.round(seeds_xy[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(seeds_xy[:, 1]).astype(int), 0, H - 1)
    for x, y, val in zip(xs, ys, psi):
        imp[y, x] += float(val)
        cnt[y, x] += 1.0

    # Build radial kernel from Eq. 16 with r_max = rt
    diam = 2 * rt + 1
    yy, xx = np.mgrid[-rt:rt+1, -rt:rt+1]
    d2 = (xx * xx + yy * yy).astype(np.float32)
    mask = (d2 <= (rt * rt)).astype(np.float32)

    a = -beta * d2 / (2.0 * (rt * rt) + 1e-12)
    kernel = gamma * (1.0 - (1.0 - np.exp(a)) / (1.0 - np.exp(-beta) + 1e-12))
    kernel *= mask
    # No need to renormalize: Eq. 17 uses the normalized kernel form directly

    # Binary disc kernel for neighbor counts
    disc = mask

    # Convolve
    dens = cv2.filter2D(imp, -1, kernel, borderType=border)
    nmap = cv2.filter2D(cnt, -1, disc,   borderType=border)

    dens[nmap < float(min_count)] = 0.0

    if normalize and np.max(dens) > 0:
        dens = dens / np.max(dens)
    return dens


# ---------- Convenience wrapper ----------

def compute_expand_map(lum, regions=None, seeds=None, psi=None, rt=16,
                       gamma=0.918, beta=1.953, min_count=4, mode="exact",
                       thresh_percentile=98, nms_radius=3, psi_mode="sum"):
    """
    High-level entry point.

    Provide either:
      - regions: list of (y0,y1,x0,x1) from median-cut; OR
      - seeds=(N,2) and psi=(N,)

    Returns:
      expand_map: HxW float32 in [0,1]
    """
    if seeds is None or psi is None:
        if regions is None:
            # Fall back to bright local maxima as seeds
            seeds, psi = seeds_from_luminance_threshold(lum, thresh_percentile, nms_radius)
        else:
            seeds, psi = seeds_from_median_cut_regions(lum, regions, psi_mode=psi_mode)

    if mode == "exact":
        return expand_map_exact(lum, seeds, psi, rt=rt, gamma=gamma, beta=beta, min_count=min_count)
    elif mode == "fast":
        return expand_map_fast(lum, seeds, psi, rt=rt, gamma=gamma, beta=beta, min_count=min_count)
    else:
        raise ValueError("mode must be 'exact' or 'fast'")
