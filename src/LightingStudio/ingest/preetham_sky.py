import numpy as np
import torch
from src.LightingStudio.analysis.utils.io import write_exr

# -----------------------------
# Preetham 1999 sky (NumPy)
# -----------------------------

# Perez A..E for each channel given turbidity T
def _perez_params(T):
    # Order: [A, B, C, D, E]
    Y = np.array([
         0.1787*T - 1.4630,
        -0.3554*T + 0.4275,
        -0.0227*T + 5.3251,
         0.1206*T - 2.5771,
        -0.0670*T + 0.3703
    ], dtype=np.float64)

    x = np.array([
        -0.0193*T - 0.2592,
        -0.0665*T + 0.0008,
        -0.0004*T + 0.2125,
        -0.0641*T - 0.8989,
        -0.0033*T + 0.0452
    ], dtype=np.float64)

    y = np.array([
        -0.0167*T - 0.2608,
        -0.0950*T + 0.0092,
        -0.0079*T + 0.2102,
        -0.0441*T - 1.6537,
        -0.0109*T + 0.0529
    ], dtype=np.float64)

    return x, y, Y  # each is [A,B,C,D,E]

# Zenith chromaticity (Preetham cubic in sun-zenith angle, quadratic in T)
def _zenith_chroma(T, theta_s):
    # theta_s is sun zenith angle (0 at zenith, pi/2 at horizon)
    t = theta_s
    t2, t3 = t*t, t*t*t
    T2 = T*T

    x = ( 0.00166*t3 - 0.00375*t2 + 0.00209*t + 0.0  )*T2 \
      + (-0.02903*t3 + 0.06377*t2 - 0.03202*t + 0.00394)*T  \
      + ( 0.11693*t3 - 0.21196*t2 + 0.06052*t + 0.25886)

    y = ( 0.00275*t3 - 0.00610*t2 + 0.00316*t + 0.0  )*T2 \
      + (-0.04214*t3 + 0.08970*t2 - 0.04153*t + 0.00516)*T  \
      + ( 0.15346*t3 - 0.26756*t2 + 0.06670*t + 0.26688)
    return x, y

# Zenith luminance (Preetham analytic, in kcd/m^2)
def _zenith_luminance(T, theta_s):
    chi = (4.0/9.0 - T/120.0)*(np.pi - 2.0*theta_s)
    Lz = (4.0453*T - 4.9710)*np.tan(chi) - 0.2155*T + 2.4192
    # Clamp to positive to avoid edge tan artifacts
    return float(max(Lz, 0.0))

# -----------------------------
# Bicubic (Catmull-Rom) helper
# -----------------------------

def _catmull_rom(p0, p1, p2, p3, t):
    # t in [0,1]
    a = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3
    b =  1.0*p0 - 2.5*p1 + 2.0*p2 - 0.5*p3
    c = -0.5*p0 + 0.5*p2
    d =  p1
    return ((a*t + b)*t + c)*t + d

def _bicubic_grid(xs, ys, table, x, y):
    """
    Separable Catmull-Rom over regular grid.
    xs: (Nx,), ys: (My,), table: (My, Nx)
    x in [xs[0], xs[-1]], y in [ys[0], ys[-1]]
    """
    xs = np.asarray(xs); ys = np.asarray(ys); tbl = np.asarray(table)
    # locate x cell
    x = np.clip(x, xs[0], xs[-1]); y = np.clip(y, ys[0], ys[-1])

    ix = np.searchsorted(xs, x) - 1
    iy = np.searchsorted(ys, y) - 1
    ix = np.clip(ix, 1, len(xs)-3)
    iy = np.clip(iy, 1, len(ys)-3)

    tx = (x - xs[ix]) / (xs[ix+1] - xs[ix])
    ty = (y - ys[iy]) / (ys[iy+1] - ys[iy])

    # four rows around y
    col = np.empty(4, dtype=np.float64)
    for j in range(-1, 3):
        row = tbl[iy+j, ix-1:ix+3]
        col[j+1] = _catmull_rom(row[0], row[1], row[2], row[3], tx)
    return _catmull_rom(col[0], col[1], col[2], col[3], ty)

# Build bicubic tables once per call (cheap)
def _build_zenith_tables():
    T_grid = np.linspace(2.0, 12.0, 11)              # 2..12
    th_grid = np.linspace(0.0, 0.5*np.pi, 19)        # 0..pi/2
    L_tbl = np.zeros((len(th_grid), len(T_grid)), np.float64)
    x_tbl = np.zeros_like(L_tbl)
    y_tbl = np.zeros_like(L_tbl)
    for i, th in enumerate(th_grid):
        for j, T in enumerate(T_grid):
            L_tbl[i, j] = _zenith_luminance(T, th)
            x_tbl[i, j], y_tbl[i, j] = _zenith_chroma(T, th)
    return T_grid, th_grid, L_tbl, x_tbl, y_tbl

_Tg, _thg, _Ltbl, _xtbl, _ytbl = _build_zenith_tables()

def _zenith_bicubic(T, theta_s):
    # theta_s = sun zenith angle
    Lz = _bicubic_grid(_Tg, _thg, _Ltbl, T, theta_s)
    xz = _bicubic_grid(_Tg, _thg, _xtbl, T, theta_s)
    yz = _bicubic_grid(_Tg, _thg, _ytbl, T, theta_s)
    return float(max(Lz, 0.0)), float(xz), float(yz)

# -----------------------------
# Color conversions
# -----------------------------

# xyY -> XYZ
def _xyY_to_XYZ(x, y, Y):
    y = max(y, 1e-6)
    X = x * (Y / y)
    Z = (1.0 - x - y) * (Y / y)
    return np.array([X, Y, Z], dtype=np.float64)

# XYZ -> linear sRGB (D65)
_M_XYZ_to_RGB = np.array([
    [ 3.24096994, -1.53738318, -0.49861076],
    [-0.96924364,  1.87596750,  0.04155506],
    [ 0.55630080, -0.20397696,  1.05697151]
], dtype=np.float64)

def _XYZ_to_RGB(XYZ):
    return _M_XYZ_to_RGB @ XYZ

# -----------------------------
# Perez function
# -----------------------------

def _perez_F(cos_theta, gamma, cos_gamma, A, B, C, D, E):
    # guard near-horizon to avoid overflow
    cos_theta = np.maximum(cos_theta, 1e-4)
    return (1.0 + A*np.exp(B / cos_theta)) * (1.0 + C*np.exp(D*gamma) + E*(cos_gamma**2))

# -----------------------------
# Utilities
# -----------------------------

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _dir_from_equirect(u, v):
    """
    u in [0,1), v in [0,1]  (lon, lat); up = +Y
    returns unit vector (x,y,z)
    """
    lon = (u - 0.5) * 2.0*np.pi       # -pi..pi
    lat = (0.5 - v) * np.pi           # +pi/2 (up) .. -pi/2 (down)
    clat = np.cos(lat)
    x = clat * np.sin(lon)
    y = np.sin(lat)
    z = clat * np.cos(lon)
    return np.stack([x, y, z], axis=-1)

# -----------------------------
# MAIN: equirectangular renderer
# -----------------------------

def preetham_sky_equirect(H, W, turbidity, sun_dir, return_rgb=True):
    """
    Render Preetham sky to an equirectangular (H, W, 3) image.

    Args:
        H, W         : output height/width (ints)
        turbidity    : float, typical 2..12
        sun_dir      : 3-vector, direction TO the sun in world coords (up = +Y)
        return_rgb   : if True -> linear sRGB; else -> CIE XYZ

    Returns:
        img : (H, W, 3) float32, lower hemisphere set to 0
    """
    T = float(turbidity)
    sun = _normalize(np.asarray(sun_dir, dtype=np.float64))

    # Sun angles (relative to zenith)
    cos_theta_s = np.clip(sun[1], -1.0, 1.0)   # y is "up"
    theta_s = np.arccos(np.clip(cos_theta_s, -1.0, 1.0))  # sun zenith angle

    # Perez params
    A_x, A_y, A_Y = _perez_params(T)

    # Zenith via bicubic (meets your "bi-cubic function" requirement)
    Lz, xz, yz = _zenith_bicubic(T, theta_s)

    # Normalization factors f0 = 1 / F(theta=0, gamma=theta_s)
    cos_theta0 = 1.0
    gamma_s = theta_s
    cos_gamma_s = np.cos(gamma_s)
    f0_x = 1.0 / _perez_F(cos_theta0, gamma_s, cos_gamma_s, *A_x)
    f0_y = 1.0 / _perez_F(cos_theta0, gamma_s, cos_gamma_s, *A_y)
    f0_Y = 1.0 / _perez_F(cos_theta0, gamma_s, cos_gamma_s, *A_Y)

    # Build per-pixel directions
    u = (np.arange(W, dtype=np.float64) + 0.5) / W
    v = (np.arange(H, dtype=np.float64) + 0.5) / H
    uu, vv = np.meshgrid(u, v)
    dirs = _dir_from_equirect(uu, vv)              # (H,W,3)
    cos_theta = np.clip(dirs[..., 1], 0.0, 1.0)    # only upper hemisphere contributes
    # angle to sun
    cos_gamma = np.clip(np.sum(dirs * sun, axis=-1), -1.0, 1.0)
    gamma = np.arccos(cos_gamma)

    # Perez F for x, y, Y
    Fx = _perez_F(cos_theta, gamma, cos_gamma, *A_x) * f0_x
    Fy = _perez_F(cos_theta, gamma, cos_gamma, *A_y) * f0_y
    FY = _perez_F(cos_theta, gamma, cos_gamma, *A_Y) * f0_Y

    # xyY at each pixel
    x = xz * Fx
    y = yz * Fy
    Y = Lz * FY

    # xyY -> XYZ (vectorized)
    y_safe = np.clip(y, 1e-6, None)
    X = x * (Y / y_safe)
    Z = (1.0 - x - y) * (Y / y_safe)
    XYZ = np.stack([X, Y, Z], axis=-1)

    # Zero out lower hemisphere (cos_theta==0 already suppresses Perez to finite, but force to 0)
    mask = (cos_theta <= 0.0)[..., None]
    XYZ = np.where(mask, 0.0, XYZ)

    if not return_rgb:
        return XYZ.astype(np.float32)

    # XYZ -> linear sRGB
    RGB = XYZ @ _M_XYZ_to_RGB.T
    # Clamp negative numerical wiggles
    RGB = np.clip(RGB, 0.0, None)
    return RGB.astype(np.float32)


# -----------------------------
# Optional: expose the pieces you asked for (A..E & bicubic handles)
# -----------------------------

def preetham_params_and_zenith_functions(turbidity, sun_dir):
    """
    Convenience helper returning:
      - Perez A..E tuples for x, y, Y
      - bicubic evaluators for zenith Y, x, y as callables of (T, theta_s)
      - sun zenith angle (theta_s) detected from sun_dir
    """
    T = float(turbidity)
    sun = _normalize(np.asarray(sun_dir, dtype=np.float64))
    theta_s = float(np.arccos(np.clip(sun[1], -1.0, 1.0)))

    Ax, Ay, AY = _perez_params(T)

    def zenith_Y(Tq, thetq): return _zenith_bicubic(float(Tq), float(thetq))[0]
    def zenith_x(Tq, thetq): return _zenith_bicubic(float(Tq), float(thetq))[1]
    def zenith_y(Tq, thetq): return _zenith_bicubic(float(Tq), float(thetq))[2]

    return {
        "Perez_params": {
            "x": {"A": Ax[0], "B": Ax[1], "C": Ax[2], "D": Ax[3], "E": Ax[4]},
            "y": {"A": Ay[0], "B": Ay[1], "C": Ay[2], "D": Ay[3], "E": Ay[4]},
            "Y": {"A": AY[0], "B": AY[1], "C": AY[2], "D": AY[3], "E": AY[4]},
        },
        "zenith_bicubic": {"Y": zenith_Y, "x": zenith_x, "y": zenith_y},
        "sun_zenith_angle": theta_s
    }

# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    H, W = 256, 512
    T = 2.3
    # Sun toward south, 30° elevation (up = +Y)
    elev = np.deg2rad(30.0)
    azim = np.deg2rad(80.0)  # 0 = +Z, 90 = +X, 180 = -Z
    sun_dir = np.array([np.cos(elev)*np.sin(azim), np.sin(elev), np.cos(elev)*np.cos(azim)])
    img = preetham_sky_equirect(H, W, T, sun_dir)  # (H,W,3) linear sRGB

    write_exr(torch.from_numpy(img), "preetham_sky_2.exr")

