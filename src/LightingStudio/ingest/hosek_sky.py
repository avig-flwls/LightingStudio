import numpy as np
import torch
from src.LightingStudio.analysis.utils.io import write_exr
import re
from pathlib import Path

# ---------------------------------------------
# Hosek–Wilkie 2012 Sky (NumPy, equirectangular)
# ---------------------------------------------
# Data layout expected (same as your C++ reference):
#   coeffsX, coeffsY, coeffsZ: shape (2, 10, 6, 9)   # [albedo in {0,1}][turbidity 1..10][quintic 0..5][9 coeffs]
#   radX,    radY,    radZ:    shape (2, 10, 6)      # same indices, 6 quintic control points
#
# Tip: if your header provides C arrays, you can np.array(...) them with the shapes above.

# --- color transforms (sRGB D65 primaries) ---
_M_RGB_to_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float64)

_M_XYZ_to_RGB = np.array([
    [ 3.24096994, -1.53738318, -0.49861076],
    [-0.96924364,  1.87596750,  0.04155506],
    [ 0.55630080, -0.20397696,  1.05697151]
], dtype=np.float64)

def _rgb_to_xyz(rgb):
    return _M_RGB_to_XYZ @ rgb

# --- quintic (Bernstein / Bezier) weights for s in [0,1] ---
def _quintic_weights(s):
    is1 = 1.0 - s
    is2 = is1*is1; is3 = is2*is1; is4 = is2*is2; is5 = is2*is3
    s2 = s*s; s3 = s2*s; s4 = s2*s2; s5 = s2*s3
    # [ (1-s)^5, 5(1-s)^4 s, 10(1-s)^3 s^2, 10(1-s)^2 s^3, 5(1-s) s^4, s^5 ]
    return np.array([is5, 5*is4*s, 10*is3*s2, 10*is2*s3, 5*is1*s4, s5], dtype=np.float64)

def _solar_elevation_param(theta_s):
    # theta_s = sun zenith angle. Elevation = pi/2 - theta_s
    elev = max(0.0, np.pi*0.5 - float(theta_s))
    # Hosek uses cubic-root re-parameterization of normalized elevation
    return (elev / (np.pi*0.5))**(1.0/3.0)

def _find_hosek_coeffs(dataset9, datasetR, turbidity, albedo_scalar, theta_s):
    """
    dataset9: (2,10,6,9), datasetR: (2,10,6)
    turbidity: float ~ [1,10] (Hosek tables cover 1..10 internally; spec often 2..10)
    albedo_scalar: in [0,1] for this XYZ channel
    theta_s: sun zenith angle
    returns: (coeffs[9], radiance_scale)  with radiance_scale in XYZ units pre-683
    """
    # Clamp turbidity to [1,10], then fractional blend between tbi and tbi+1
    t = float(turbidity)
    t = max(1.0, min(10.0, t))
    tbi = int(np.floor(t))
    if tbi == 10:  # edge: stick to the last bin
        tbi = 9
        tbf = 1.0
    else:
        tbf = t - tbi

    # Quintic param from solar elevation
    s = _solar_elevation_param(theta_s)
    w = _quintic_weights(s)  # (6,)

    # Evaluate quintic along the 6 control points for both albedo slices and both turbidity bins
    # ic[4,9]: [a=0,t=tbi-1], [a=1,t=tbi-1], [a=0,t=tbi], [a=1,t=tbi]
    ic = np.empty((4, 9), dtype=np.float64)
    ir = np.empty((4,), dtype=np.float64)

    for a in (0, 1):
        for k, tb in enumerate((tbi-1, tbi)):
            # coeffs: (6,9) -> weighted sum by w -> (9,)
            c69 = dataset9[a, tb-1]   # note: t bins 1..10 => index tb-1
            ic[a + 2*k] = w @ c69     # (9,)
            # radiance controls: (6,) -> scalar
            r6 = datasetR[a, tb-1]
            ir[a + 2*k] = float(w @ r6)

    # Bilinear blend in (albedo, turbidity) square
    # coefficients weights:
    cw = np.array([
        (1.0 - albedo_scalar) * (1.0 - tbf),
        (      albedo_scalar) * (1.0 - tbf),
        (1.0 - albedo_scalar) * (      tbf),
        (      albedo_scalar) * (      tbf),
    ], dtype=np.float64)

    coeffs = cw @ ic   # (9,)
    rad    = cw @ ir   # scalar
    return coeffs, float(rad)

def _eval_hosek_F(cos_theta, gamma, cos_gamma, coeffs):
    # coeff layout matches your GLSL reference exactly
    A = coeffs[0]
    B = coeffs[1]
    C = coeffs[2]
    D = coeffs[3]
    E = coeffs[4]
    F = coeffs[5]
    G = coeffs[6]
    I = coeffs[7]
    H = coeffs[8]  # chi uses H

    chi = (1.0 + cos_gamma*cos_gamma) / np.power(1.0 + G*G - 2.0*G*cos_gamma, 1.5)
    cos_theta = np.maximum(cos_theta, 0.01)  # horizon guard
    return (1.0 + A*np.exp(B / cos_theta)) * (C + D*np.exp(E*gamma) + F*(cos_gamma**2) + H*chi + I*np.sqrt(cos_theta))

def _dir_from_equirect(u, v):
    lon = (u - 0.5) * 2.0*np.pi
    lat = (0.5 - v) * np.pi
    cl = np.cos(lat)
    return np.stack([cl*np.sin(lon), np.sin(lat), cl*np.cos(lon)], axis=-1)  # x,y,z (y is up)

def _normalize(v):
    n = np.linalg.norm(v)
    return v/n if n > 0 else v

def hosek_wilkie_sky_equirect(
    H, W,
    turbidity,
    sun_dir,
    albedo_rgb,
    hosek_data,
    return_rgb=True
):
    """
    Render Hosek–Wilkie sky to an equirectangular (H, W, 3) image.

    Args:
        H, W         : ints
        turbidity    : float (recommended range ~2..10; tables internally 1..10)
        sun_dir      : 3-vector (toward sun), world up = +Y
        albedo_rgb   : (3,) ground diffuse RGB albedo in [0,1] (used per-channel, converted to XYZ)
        hosek_data   : dict with:
            'coeffsX','coeffsY','coeffsZ' -> (2,10,6,9) float64 arrays
            'radX','radY','radZ'          -> (2,10,6)   float64 arrays
        return_rgb   : if True returns linear sRGB; else CIE XYZ

    Returns:
        (H, W, 3) float32
    """
    sun = _normalize(np.asarray(sun_dir, dtype=np.float64))
    albedo_rgb = np.clip(np.asarray(albedo_rgb, dtype=np.float64), 0.0, 1.0)
    albedo_xyz = _rgb_to_xyz(albedo_rgb)
    # Clamp albedo scalars into [0,1] for the dataset blend
    ax, ay, az = [float(np.clip(a, 0.0, 1.0)) for a in albedo_xyz]

    # Sun zenith
    cos_theta_s = np.clip(sun[1], -1.0, 1.0)
    theta_s = float(np.arccos(cos_theta_s))
    gamma_s = theta_s
    cos_gamma_s = float(np.cos(gamma_s))

    # Per-XYZ coefficients + radiance scales (pre-683)
    cX, rX = _find_hosek_coeffs(hosek_data['coeffsX'], hosek_data['radX'], turbidity, ax, theta_s)
    cY, rY = _find_hosek_coeffs(hosek_data['coeffsY'], hosek_data['radY'], turbidity, ay, theta_s)
    cZ, rZ = _find_hosek_coeffs(hosek_data['coeffsZ'], hosek_data['radZ'], turbidity, az, theta_s)

    # Convert to photometric units (luminous) as in reference
    radXYZ = np.array([rX, rY, rZ], dtype=np.float64)

    # Build pixel directions (equirect)
    u = (np.arange(W, dtype=np.float64) + 0.5) / W
    v = (np.arange(H, dtype=np.float64) + 0.5) / H
    uu, vv = np.meshgrid(u, v)
    dirs = _dir_from_equirect(uu, vv)              # (H,W,3)
    cos_theta = np.clip(dirs[..., 1], 0.0, 1.0)    # upper hemisphere only
    cos_gamma = np.clip(np.sum(dirs * sun, axis=-1), -1.0, 1.0)
    gamma = np.arccos(cos_gamma)

    # Evaluate F for each XYZ channel
    FX = _eval_hosek_F(cos_theta, gamma, cos_gamma, cX)
    FY = _eval_hosek_F(cos_theta, gamma, cos_gamma, cY)
    FZ = _eval_hosek_F(cos_theta, gamma, cos_gamma, cZ)

    # XYZ radiance
    X = FX * radXYZ[0]
    Y = FY * radXYZ[1]
    Z = FZ * radXYZ[2]
    XYZ = np.stack([X, Y, Z], axis=-1)

    # Zero lower hemisphere explicitly
    mask0 = (cos_theta <= 0.0)[..., None]
    XYZ = np.where(mask0, 0.0, XYZ)

    if not return_rgb:
        return XYZ.astype(np.float32)

    # XYZ -> linear sRGB
    RGB = XYZ @ _M_XYZ_to_RGB.T
    RGB = np.clip(RGB, 0.0, None)
    return RGB.astype(np.float32)

_FLOAT_RE = re.compile(
    r"""(?x)
    [+-]?                           # sign
    (?:
        (?:\d+\.\d*|\.\d+|\d+)      # 123. , .123 , 123
        (?:[eE][+-]?\d+)?           # optional exponent
    )
    """
)

def _grab_floats(text, start_idx, count):
    """Return a list of `count` floats starting from text[start_idx:]."""
    vals = []
    pos = start_idx
    while len(vals) < count:
        m = _FLOAT_RE.search(text, pos)
        if not m:
            raise ValueError("Ran out of numbers before reaching expected count.")
        vals.append(float(m.group(0)))
        pos = m.end()
    return vals, pos

def _parse_array(text, symbol, shape):
    """
    Parse a C float array with name `symbol` and reshape to `shape`.
    Only takes the exact number of floats required by `shape`.
    """
    # Find the array declaration
    # e.g., float kHosekCoeffsX[2][10][6][9] =
    decl_re = re.compile(
        rf"float\s+{re.escape(symbol)}\s*\[[^\]]+\](?:\s*\[[^\]]+\])*?\s*=",
        re.MULTILINE
    )
    m = decl_re.search(text)
    if not m:
        raise ValueError(f"Couldn't find declaration for {symbol}")

    # Find the first "{" after the declaration; begin scanning numbers from there
    brace_idx = text.find("{", m.end())
    if brace_idx == -1:
        raise ValueError(f"Couldn't find opening brace for {symbol}")

    total = int(np.prod(shape))
    arr, _ = _grab_floats(text, brace_idx, total)
    return np.array(arr, dtype=np.float64).reshape(shape)

# --- public API --------------------------------------------------------------

def load_hosek_xyz_header(path):
    """
    Load Hosek-Wilkie XYZ coefficient tables from the official header file.

    Returns a dict with:
      - 'coeffs': {'X','Y','Z'} each (2,10,6,9)
      - 'radiance': {'X','Y','Z'} each (2,10,6)
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    coeffsX = _parse_array(text, "kHosekCoeffsX", (2, 10, 6, 9))
    coeffsY = _parse_array(text, "kHosekCoeffsY", (2, 10, 6, 9))
    coeffsZ = _parse_array(text, "kHosekCoeffsZ", (2, 10, 6, 9))

    radX = _parse_array(text, "kHosekRadX", (2, 10, 6))
    radY = _parse_array(text, "kHosekRadY", (2, 10, 6))
    radZ = _parse_array(text, "kHosekRadZ", (2, 10, 6))

    return {
        "coeffs": {"X": coeffsX, "Y": coeffsY, "Z": coeffsZ},
        "radiance": {"X": radX, "Y": radY, "Z": radZ},
        # flat aliases for legacy call-sites
        "coeffsX": coeffsX, "coeffsY": coeffsY, "coeffsZ": coeffsZ,
        "radX": radX, "radY": radY, "radZ": radZ,
    }

# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":

    H, W = 256, 512
    T = 3.0

    elev = np.deg2rad(30.0)
    azim = np.deg2rad(80.0)  # 0 = +Z, 90 = +X, 180 = -Z
    sun_dir = np.array([np.cos(elev)*np.sin(azim), np.sin(elev), np.cos(elev)*np.cos(azim)])
    albedo_rgb = np.array([0.2, 0.2, 0.2])  # dark ground

    hosek_data = load_hosek_xyz_header(
        r"C:\Users\AviGoyal\Documents\LightingStudio\archive\HosekDataXYZ.h"
    )

    # Quick sanity checks (will raise if shapes are wrong)
    assert hosek_data["coeffsX"].shape == (2, 10, 6, 9)
    assert hosek_data["radY"].shape == (2, 10, 6)

    img = hosek_wilkie_sky_equirect(
        H=H, W=W,
        turbidity=T,
        sun_dir=sun_dir,   
        albedo_rgb=albedo_rgb,
        hosek_data=hosek_data
    )

    write_exr(torch.from_numpy(img), "hosek_sky_albedo.exr")