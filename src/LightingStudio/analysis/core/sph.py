import cv2  
import torch
import math
from src.LightingStudio.analysis.utils.transforms import generate_spherical_coordinates_map, spherical_to_cartesian, pixel_solid_angles, convert_theta, cartesian_to_spherical, cartesian_to_pixel
from einops import einsum, rearrange, repeat
from typing import Union

from src.LightingStudio.analysis.report.datatypes import SPHMetrics, SPHMetricsCPU

# -----------------------------
# Cartesian to Spherical Harmonic Basis
# -----------------------------

def cartesian_to_sph_eval(cartesian_coordinates: torch.Tensor, l:int, m:int) -> torch.Tensor:  # noqa: E741
    """
    
    :params cartesian_coordinates (..., 3): where at each position on the image, we have the cartesian coordinate on the sphere.
    :params l
    :params m

    :returns Y (...): the basis function evaluated at a specific l and m for each position in cartesian_coordinates
    
    Source:
    [2] Appendix A2 Polynomial Forms of SH Basis
    [6] EvalSH function.
    """

    # Prep
    x, y, z = cartesian_coordinates[...,0], cartesian_coordinates[...,1], cartesian_coordinates[...,2]
    radial_length = torch.sqrt(torch.pow(x, 2) + torch.pow(y,2) + torch.pow(z, 2))
    assert (torch.allclose(radial_length, torch.ones_like(radial_length))), f'{radial_length} is not close to 1.0'

    # Compute Y
    if(l == 0 and m ==0):
        # 0.5 * np.sqrt(1.0/np.pi)
        return 0.282095*torch.ones_like(x)
    
    elif(l == 1 and m ==-1):
        # -np.sqrt(3/(4.0*np.pi)) * y
        return -0.48860251 * y
    
    elif(l == 1 and m == 0):   
        # np.sqrt(3/(4.0*np.pi)) * z
        return 0.48860251 * z

    elif(l ==1 and m == 1):
        # -np.sqrt(3/(4.0*np.pi)) * x
        return -0.48860251 * x

    elif(l == 2 and m == -2):
        # 0.5 * np.sqrt(15/np.pi)
        return 1.092548 * x * y

    elif(l == 2 and m == -1):
        # -0.5 * np.sqrt(15/np.pi) * y * z
        return -1.092548 * y * z

    elif(l == 2 and m == 0):
        # 0.25 * np.sqrt(5/np.pi) (3.0 * z*z - 1)
        return 0.315392 * (3*torch.pow(z,2) - 1.0)

    elif(l == 2 and m ==1):
        # -0.5 * np.sqrt(15/np.pi) * x * z
        return -1.092548 * x * z

    elif(l == 2 and m == 2):
        # 0.25 *  np.sqrt(15/np.pi) * (x*x - y*y)
        return 0.546274 * (torch.pow(x,2) - torch.pow(y, 2))
    
    elif(l == 3 and m == -3):
        # -0.25 * sqrt(35/(2pi)) * y * (3x^2 - y^2)
        return -0.590044 * y * (3.0 * torch.pow(x,2) - torch.pow(y,2))

    elif(l == 3 and m == -2):
        # 0.5 * sqrt(105/pi) * x * y * z
        return 2.890611 * x * y * z

    elif(l == 3 and m == -1):
        # -0.25 * sqrt(21/(2pi)) * y * (4z^2-x^2-y^2)
        return -0.457046 * y * (4.0 * torch.pow(z,2) - torch.pow(x,2) - torch.pow(y,2))

    elif(l == 3 and m == 0):
        # 0.25 * sqrt(7/pi) * z * (2z^2 - 3x^2 - 3y^2)
        return 0.373176 * z * (5.0 * torch.pow(z,2) - 3.0)

    elif(l == 3 and m == 1):
        # -0.25 * sqrt(21/(2pi)) * x * (4z^2-x^2-y^2)
        return -0.457046 * x *(4.0 * torch.pow(z,2) - torch.pow(x,2) - torch.pow(y,2))

    elif(l == 3 and m == 2):
        # 0.25 * sqrt(105/pi) * z * (x^2 - y^2)
        return 1.445306 * z * (torch.pow(x,2) - torch.pow(y,2))

    elif(l == 3 and m == 3):
        # -0.25 * sqrt(35/(2pi)) * x * (x^2-3y^2)
        return -0.590044 * x * (torch.pow(x,2) - 3.0 * torch.pow(y,2))

    elif(l == 4 and m == -4):
        # 0.75 * sqrt(35/pi) * x * y * (x^2-y^2)
        return 2.503343 * x * y * (torch.pow(x,2) - torch.pow(y,2))

    elif(l == 4 and m == -3):
        # -0.75 * sqrt(35/(2pi)) * y * z * (3x^2-y^2)
        return -1.770131 * y * z * (3.0 *torch.pow(x,2) - torch.pow(y,2))

    elif(l == 4 and m == -2):
        # 0.75 * sqrt(5/pi) * x * y * (7z^2-1)
        return 0.946175 * x * y * (7.0 * torch.pow(z,2) - 1.0)

    elif(l == 4 and m == -1):
        # -0.75 * sqrt(5/(2pi)) * y * z * (7z^2-3)
        return -0.669047 * y * z * (7.0 * torch.pow(z,2) - 3.0)

    elif(l == 4 and m == 0):
        # 3/16 * sqrt(1/pi) * (35z^4-30z^2+3)
        return 0.105786 * (35.0 * torch.pow(z,4) - 30.0 * torch.pow(z,2) + 3.0)

    elif(l == 4 and m == 1):
        # -0.75 * sqrt(5/(2pi)) * x * z * (7z^2-3)
        return -0.669047 * x * z * (7.0 * torch.pow(z,2) - 3.0)

    elif(l == 4 and m == 2):
        # 3/8 * sqrt(5/pi) * (x^2 - y^2) * (7z^2 - 1)
        return 0.473087 * (torch.pow(x,2) - torch.pow(y,2)) * (7.0 * torch.pow(z,2) - 1.0)

    elif(l == 4 and m == 3):
        # -0.75 * sqrt(35/(2pi)) * x * z * (x^2 - 3y^2)
        return -1.770131 * x * z * (torch.pow(x,2) - 3.0 * torch.pow(y,2))

    elif(l == 4 and m == 4):
        # 3/16*sqrt(35/pi) * (x^2 * (x^2 - 3y^2) - y^2 * (3x^2 - y^2))
        return 0.625836 * (torch.pow(x,4) - 6.0 * torch.pow(y, 2)*torch.pow(x,2) + torch.pow(y,4))

    else:
        raise ValueError(f'l:{l} and m:{m}, but those are not supported at this time')


def cartesian_to_sph_basis(cartesian_coordinates: torch.Tensor, l_max: int) -> torch.Tensor:
    """
    Convert from cartesian_coordinates to the spherical harmonic basis (Ylm).

    :params cartesian_coordinates (..., 3): where at each position on the image, we have the cartesian coordinate on the sphere.
    :params l_max:  The maximum number of bands. The order of the basis to compute.
    :returns Ylm (..., n_terms): The spherical harmonic basis function evaluated at each coordinate.

    Sources: 
    [2] Appendix A2 Polynomial Forms of SH Basis
    [4] Equation 10
    [5] shEvaluate function.
    """

    # Construct Ylm
    cartesian_shape = cartesian_coordinates.shape       # (..., 3)
    total_indices = sph_indices_total(l_max)
    Ylm_shape = (*cartesian_shape[:-1], total_indices)  # (..., n_terms)
    Ylm = torch.zeros(Ylm_shape, device=cartesian_coordinates.device, dtype=cartesian_coordinates.dtype)

    for l in range(l_max +1):  # noqa: E741
        for m in range(-l, l+1):
            index = sph_index_from_lm(l, m)
            Ylm[..., index] = cartesian_to_sph_eval(cartesian_coordinates, l, m)

    return Ylm


def cartesian_to_sph_basis_vectorized(cartesian_coordinates: torch.Tensor, l_max: int) -> torch.Tensor:
    """
    GPU-optimized vectorized version of cartesian_to_sph_basis that computes all basis functions simultaneously.
    
    :params cartesian_coordinates (..., 3): where at each position on the image, we have the cartesian coordinate on the sphere.
    :params l_max:  The maximum number of bands. The order of the basis to compute.
    :returns Ylm (..., n_terms): The spherical harmonic basis function evaluated at each coordinate.
    """
    # Extract coordinates and precompute powers
    x, y, z = cartesian_coordinates[..., 0], cartesian_coordinates[..., 1], cartesian_coordinates[..., 2]
    
    # Precompute commonly used powers to avoid recomputation
    x2, y2, z2 = x * x, y * y, z * z
    x3, y3, z3 = x2 * x, y2 * y, z2 * z
    x4, y4, z4 = x3 * x, y3 * y, z3 * z
    
    # Verify unit sphere constraint
    radial_length = torch.sqrt(x2 + y2 + z2)
    assert torch.allclose(radial_length, torch.ones_like(radial_length)), f'{radial_length.max().item():.6f} is not close to 1.0'
    
    # Initialize output tensor
    cartesian_shape = cartesian_coordinates.shape
    total_indices = sph_indices_total(l_max)
    Ylm_shape = (*cartesian_shape[:-1], total_indices)
    Ylm = torch.zeros(Ylm_shape, device=cartesian_coordinates.device, dtype=cartesian_coordinates.dtype)
    
    # Precomputed coefficients for better readability
    c00 = 0.282095  # 0.5 * sqrt(1/pi)
    c1 = 0.48860251  # sqrt(3/(4*pi))
    c2_2 = 1.092548  # 0.5 * sqrt(15/pi)
    c2_0 = 0.315392  # 0.25 * sqrt(5/pi)
    c2_2_half = 0.546274  # 0.25 * sqrt(15/pi)
    
    # l=0 terms
    if l_max >= 0:
        Ylm[..., sph_index_from_lm(0, 0)] = c00
    
    # l=1 terms  
    if l_max >= 1:
        Ylm[..., sph_index_from_lm(1, -1)] = -c1 * y
        Ylm[..., sph_index_from_lm(1, 0)] = c1 * z
        Ylm[..., sph_index_from_lm(1, 1)] = -c1 * x
    
    # l=2 terms
    if l_max >= 2:
        Ylm[..., sph_index_from_lm(2, -2)] = c2_2 * x * y
        Ylm[..., sph_index_from_lm(2, -1)] = -c2_2 * y * z
        Ylm[..., sph_index_from_lm(2, 0)] = c2_0 * (3 * z2 - 1.0)
        Ylm[..., sph_index_from_lm(2, 1)] = -c2_2 * x * z
        Ylm[..., sph_index_from_lm(2, 2)] = c2_2_half * (x2 - y2)
    
    # l=3 terms
    if l_max >= 3:
        c3_3 = 0.590044  # 0.25 * sqrt(35/(2*pi))
        c3_2 = 2.890611  # 0.5 * sqrt(105/pi)
        c3_1 = 0.457046  # 0.25 * sqrt(21/(2*pi))
        c3_0 = 0.373176  # 0.25 * sqrt(7/pi)
        
        Ylm[..., sph_index_from_lm(3, -3)] = -c3_3 * y * (3 * x2 - y2)
        Ylm[..., sph_index_from_lm(3, -2)] = c3_2 * x * y * z
        Ylm[..., sph_index_from_lm(3, -1)] = -c3_1 * y * (4 * z2 - x2 - y2)
        Ylm[..., sph_index_from_lm(3, 0)] = c3_0 * z * (5 * z2 - 3.0)
        Ylm[..., sph_index_from_lm(3, 1)] = -c3_1 * x * (4 * z2 - x2 - y2)
        Ylm[..., sph_index_from_lm(3, 2)] = c3_2 * 0.5 * z * (x2 - y2)  # 1.445306 = c3_2 * 0.5
        Ylm[..., sph_index_from_lm(3, 3)] = -c3_3 * x * (x2 - 3 * y2)
    
    # l=4 terms  
    if l_max >= 4:
        c4_4 = 2.503343   # 0.75 * sqrt(35/pi)
        c4_3 = 1.770131   # 0.75 * sqrt(35/(2*pi))
        c4_2 = 0.946175   # 0.75 * sqrt(5/pi)
        c4_1 = 0.669047   # 0.75 * sqrt(5/(2*pi))
        c4_0 = 0.105786   # 3/16 * sqrt(1/pi)
        
        Ylm[..., sph_index_from_lm(4, -4)] = c4_4 * x * y * (x2 - y2)
        Ylm[..., sph_index_from_lm(4, -3)] = -c4_3 * y * z * (3 * x2 - y2)
        Ylm[..., sph_index_from_lm(4, -2)] = c4_2 * x * y * (7 * z2 - 1.0)
        Ylm[..., sph_index_from_lm(4, -1)] = -c4_1 * y * z * (7 * z2 - 3.0)
        Ylm[..., sph_index_from_lm(4, 0)] = c4_0 * (35 * z4 - 30 * z2 + 3.0)
        Ylm[..., sph_index_from_lm(4, 1)] = -c4_1 * x * z * (7 * z2 - 3.0)
        Ylm[..., sph_index_from_lm(4, 2)] = c4_2 * 0.5 * (x2 - y2) * (7 * z2 - 1.0)  # 0.473087 = c4_2 * 0.5
        Ylm[..., sph_index_from_lm(4, 3)] = -c4_3 * x * z * (x2 - 3 * y2)
        Ylm[..., sph_index_from_lm(4, 4)] = c4_4 * 0.25 * (x4 - 6 * y2 * x2 + y4)  # 0.625836 = c4_4 * 0.25
    
    return Ylm


# -----------------------------
# Spherical to Spherical Harmonic Basis
# -----------------------------

def spherical_to_sph_eval(spherical_coordinates: torch.Tensor, l:int, m:int) -> torch.Tensor:  # noqa: E741
    """
    Real, fully-normalized spherical harmonics Y_l^m(θ, φ) with Condon–Shortley phase.
    Stable for l,m up to ~1e3 (double), easily fine for l<=50.

    Input convention (matches your code):
      spherical_coordinates[..., 0] = elevation θ in [-π/2, π/2]
      spherical_coordinates[..., 1] = azimuth   φ in (−π, π]
      cosΘ = cos(convert_theta(θ))

    Real basis:
      Y_l^0     =  K(l,0) P_l^0(x)
      Y_l^m     =  √2 K(l,m) cos(mφ) P_l^m(x),          m > 0
      Y_l^{−m}  =  √2 K(l,|m|) sin(|m|φ) P_l^{|m|}(x),  m < 0
      
      Source:
      [3] Equation 6.
      [5] P function.
    """
    
    # ---- precision & prep ----
    theta = spherical_coordinates[..., 0].to(torch.float64)
    phi   = spherical_coordinates[..., 1].to(torch.float64)
    x     = torch.cos(convert_theta(theta).to(torch.float64))  # x = cosΘ
    m_abs = abs(m)

    # ---- Associated Legendre P_l^m(x): stable diagonal + upward l-recurrence ----
    # Seed P_m^m by iterative product: pmm = (-1)^m * prod_{k=1..m} [(2k-1) * sqrt(1-x^2)]
    # This avoids explicit double factorial and works well for m<=~100 in float64.
    s = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))  # robust sinΘ
    pmm = torch.ones_like(x, dtype=torch.float64)
    if m_abs > 0:
        for k in range(1, m_abs + 1):
            pmm = -pmm * (2.0 * k - 1.0) * s

    if l == m_abs:
        P_lm = pmm
    else:
        pmmp1 = (2.0 * m_abs + 1.0) * x * pmm  # P_{m+1}^m
        if l == m_abs + 1:
            P_lm = pmmp1
        else:
            p_lm_2 = pmm
            p_lm_1 = pmmp1
            for ll in range(m_abs + 2, l + 1):
                pll = ((2.0 * ll - 1.0) * x * p_lm_1 - (ll + m_abs - 1.0) * p_lm_2) / (ll - m_abs)
                p_lm_2, p_lm_1 = p_lm_1, pll
            P_lm = p_lm_1

    # ---- Normalization K(l,m) via log-gamma (GPU-safe) ----
    # log K = 0.5 * [ log(2l+1) - log(4π) + lgamma(l-m+1) - lgamma(l+m+1) ]
    l_t = torch.tensor(float(l), dtype=torch.float64, device=x.device)
    m_t = torch.tensor(float(m_abs), dtype=torch.float64, device=x.device)
    logK = 0.5 * (
        torch.log(2.0 * l_t + 1.0)
        - torch.log(torch.tensor(4.0 * math.pi, dtype=torch.float64, device=x.device))
        + torch.lgamma(l_t - m_t + 1.0)
        - torch.lgamma(l_t + m_t + 1.0)
    )
    K = torch.exp(logK)  # scalar tensor on device

    # ---- Real Y ----
    sqrt2 = math.sqrt(2.0)
    if m == 0:
        Y = K * P_lm
    elif m > 0:
        Y = sqrt2 * K * torch.cos(m * phi) * P_lm
    else:
        Y = sqrt2 * K * torch.sin(-m * phi) * P_lm

    # cast back to input dtype
    return Y.to(spherical_coordinates.dtype)


def spherical_to_sph_basis(spherical_coordinates: torch.Tensor, l_max: int) -> torch.Tensor:
    """
    Looping version (clear & reliable). Uses the stable evaluator above.
    Y[..., idx(l,m)] matches fully-normalized real SH with Condon–Shortley phase.

    Convert from spherical_coordinates to the spherical harmonic basis (Ylm).

    :params spherical_coordinates (..., 2): where at each position on the image, we have the spherical coordinate on the sphere.
    :params l_max:  The maximum number of bands. The order of the basis to compute.
    :returns Ylm (..., n_terms): The spherical harmonic basis function evaluated at each coordinate. (H, W, n_terms)

    Sources: 
    [4] Equation 10.
    [5] SH function.
    """
    
    sph_shape = spherical_coordinates.shape
    n_terms   = sph_indices_total(l_max)
    Ylm = torch.empty((*sph_shape[:-1], n_terms),
                      device=spherical_coordinates.device,
                      dtype=spherical_coordinates.dtype)
    for l in range(l_max + 1):  # noqa: E741
        for m in range(-l, l + 1):
            Ylm[..., sph_index_from_lm(l, m)] = spherical_to_sph_eval(spherical_coordinates, l, m)
    return Ylm


def spherical_to_sph_basis_vectorized(spherical_coordinates: torch.Tensor, l_max: int) -> torch.Tensor:
    """
    Vectorized real spherical harmonics up to l_max using a stable diagonal seed
    and upward (in l) recurrence. All math in float64, then cast back.
    """
    # ---- precision & angles ----
    theta = spherical_coordinates[..., 0].to(torch.float64)
    phi   = spherical_coordinates[..., 1].to(torch.float64)
    x     = torch.cos(convert_theta(theta).to(torch.float64))  # x = cosΘ
    s     = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))      # robust sinΘ

    Hshape = spherical_coordinates.shape[:-1]
    n_terms = sph_indices_total(l_max)

    # ---- build all P_l^m(x) for m>=0 ----
    P = torch.zeros((*Hshape, n_terms), dtype=torch.float64, device=x.device)
    # P_0^0 = 1
    P[..., sph_index_from_lm(0, 0)] = 1.0

    for m in range(0, l_max + 1):
        # Diagonal P_m^m
        if m > 0:
            pmm = torch.ones_like(x, dtype=torch.float64)
            for k in range(1, m + 1):
                pmm = -pmm * (2.0 * k - 1.0) * s
            P[..., sph_index_from_lm(m, m)] = pmm
        # Next band P_{m+1}^m
        if m < l_max:
            pmmp1 = (2.0 * m + 1.0) * x * P[..., sph_index_from_lm(m, m)]
            P[..., sph_index_from_lm(m + 1, m)] = pmmp1
        # Upward recurrence for l >= m+2
        p_lm_2 = P[..., sph_index_from_lm(m, m)]
        p_lm_1 = P[..., sph_index_from_lm(m + 1, m)] if m < l_max else None
        for l in range(m + 2, l_max + 1):
            idx_lm  = sph_index_from_lm(l, m)
            idx_l1m = sph_index_from_lm(l - 1, m)
            idx_l2m = sph_index_from_lm(l - 2, m)
            P[..., idx_lm] = ((2.0 * l - 1.0) * x * P[..., idx_l1m] - (l + m - 1.0) * P[..., idx_l2m]) / (l - m)

    # ---- normalization K(l,m) for all (l,m) using lgamma (vectorized over indices) ----
    # Prepare index arrays of shape (n_terms,)
    ls, ms = [], []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            ls.append(l); ms.append(m)
    l_arr = torch.tensor(ls, dtype=torch.float64, device=x.device)
    m_arr = torch.tensor(ms, dtype=torch.float64, device=x.device)
    a_m   = torch.abs(m_arr)

    logK = 0.5 * (
        torch.log(2.0 * l_arr + 1.0)
        - torch.log(torch.tensor(4.0 * math.pi, dtype=torch.float64, device=x.device))
        + torch.lgamma(l_arr - a_m + 1.0)
        - torch.lgamma(l_arr + a_m + 1.0)
    )
    K = torch.exp(logK)  # (n_terms,)

    # ---- assemble real Y ----
    Y = torch.zeros((*Hshape, n_terms), dtype=torch.float64, device=x.device)
    sqrt2 = math.sqrt(2.0)

    # m = 0 terms: indices where ms==0
    mask_m0 = (m_arr == 0.0)
    idxs_m0 = torch.nonzero(mask_m0, as_tuple=False).squeeze(-1)
    if idxs_m0.numel() > 0:
        Y[..., idxs_m0] = P[..., idxs_m0] * K[idxs_m0]

    # m > 0 and m < 0 use the same |m| P and K; we apply cos/sin
    # We’ll fill per l by looping m (cheap: O(l_max^2) small)
    for l in range(l_max + 1):
        for m in range(1, l + 1):
            idx_pos = sph_index_from_lm(l,  m)
            idx_neg = sph_index_from_lm(l, -m)
            common  = P[..., idx_pos] * K[idx_pos]  # uses m>=0 slot
            Y[..., idx_pos] = sqrt2 * common * torch.cos(m * phi)
            Y[..., idx_neg] = sqrt2 * common * torch.sin(m * phi)

    return Y.to(spherical_coordinates.dtype)


# -----------------------------
# Project Environment Map to Coefficients
# -----------------------------
def project_env_to_coefficients(hdri: torch.Tensor, l_max:int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project the Environment Map to get the SPH coefficients.

    :params hdri: (H, W, 3) the image where each pixel holds the intensity value in rgb
    :params l_max: The number of bands to use.

    :returns sph_coeffs: (n_terms, 3)
    :return sph_basis: (..., n_terms), but in our case (H, W, n_terms)

    Source:
    [4] Equation 10
    [5] getCoefficientsFromImage, getCoefficientsMatrix function
    [6] ProjectEnvironment function
    """

    hdri_r, hdri_g, hdri_b = hdri[..., 0], hdri[..., 1], hdri[..., 2] # (H, W)

    # Get Basis
    H, W, _ = hdri.shape
    spherical_coordinates = generate_spherical_coordinates_map(H, W, hdri.device)
    
    if l_max > 4:
        # Spherical basis (vectorized)
        print(f"l_max > 4: Using spherical basis (vectorized)")
        sph_basis = spherical_to_sph_basis_vectorized(spherical_coordinates, l_max = l_max) # (..., n_terms) 
    else:
        # Cartesian basis (vectorized)
        print(f"l_max <= 4: Using cartesian basis (vectorized)")
        cartesian_coordinates = spherical_to_cartesian(spherical_coordinates)
        sph_basis = cartesian_to_sph_basis_vectorized(cartesian_coordinates, l_max = l_max) # (..., n_terms)

    # Get Solid Angle
    pixel_area, sin_theta = pixel_solid_angles(H, W, hdri.device) # (H, 1)
    # solid_angle = sin_theta * pixel_area
    solid_angle = torch.ones_like(sin_theta)


    # # Integrate to get coefficients
    # sph_coeffs = torch.stack([
    #     torch.sum(hdri_r[:, :, None] * sph_basis * solid_angle[..., None], dim=(0, 1)), # sum((H, W, None) * (..., n_terms) * (H, 1, None), axis(0,1)) = (n_terms)
    #     torch.sum(hdri_g[:, :, None] * sph_basis * solid_angle[..., None], dim=(0, 1)),
    #     torch.sum(hdri_b[:, :, None] * sph_basis * solid_angle[..., None], dim=(0, 1))
    # ], dim=1) # (n_terms, 3)

    # Mean to get coefficients
    sph_coeffs = torch.stack([
        torch.mean(hdri_r[:, :, None] * sph_basis * solid_angle[..., None], dim=(0, 1)), # sum((H, W, None) * (..., n_terms) * (H, 1, None), axis(0,1)) = (n_terms)
        torch.mean(hdri_g[:, :, None] * sph_basis * solid_angle[..., None], dim=(0, 1)),
        torch.mean(hdri_b[:, :, None] * sph_basis * solid_angle[..., None], dim=(0, 1))
    ], dim=1) # (n_terms, 3)
    
    return sph_coeffs, sph_basis

def project_direction_into_coefficients(direction: torch.Tensor, l_max:int) -> torch.Tensor:
    """
    Project a single direction into the SPH coefficients.

    We assume that the function value of L(theta, phi) = 1 at direction.
    Thus the sph_coefficients value just the value of Y(theta, phi) evaluated at that direction.
    
    Mathematically:
        Y_lm = integral integral delta L() Y() d_theta d_phi

    : params direction: direction of light source (..., 3)
    : params l_max: the number of bands

    :returns sph_coeffs: (..., n_terms)

    Source:
    [4] Equation 10.
    """

    assert(direction.shape[-1] == 3), f'direction can only be in cartesian coordinates so it must be of length 3, but we are getting {direction.shape}'


    if l_max <= 4:
        return cartesian_to_sph_basis_vectorized(direction, l_max)
    else:
        direction_as_spherical = cartesian_to_spherical(direction)
        return spherical_to_sph_basis_vectorized(direction_as_spherical, l_max)


def reconstruct_sph_coeffs_to_env(H:int, W:int, sph_coeffs: torch.Tensor, sph_basis:torch.Tensor) -> torch.Tensor:
    """
    Reconstruct (env) irradiance map from the spherical harmonic coefficients.
    
    : params H: height
    : params W: width
    : params sph_coeffs: (n_terms, 3)
    : params sph_basis: (OPTIONAL) (..., n_terms) which in our case is (H, W, n_terms)
    : return irradiance_maps: (l_max+1, H, W, 3) cumulative reconstructions for each l_max.

    Source:
    [12] equation 2.
    [5] shReconstructSignal function.
    """

    # Prep
    n_terms, _ = sph_coeffs.shape
    l_max = sph_l_max_from_indices_total(n_terms)

    # Get Solid Angle
    pixel_area, sin_theta = pixel_solid_angles(H, W, sph_coeffs.device) # (H, 1)
    # solid_angle = sin_theta * pixel_area
    solid_angle = torch.ones_like(sin_theta)

    # Compute cumulative irradiance maps (reconstruction up to each l_max)
    irradiance_maps = torch.zeros((l_max + 1, H, W, 3), device=sph_coeffs.device, dtype=sph_coeffs.dtype)
    for l in range(l_max + 1):  # noqa: E741
        curr_n_terms = sph_indices_total(l)
        # This gives cumulative reconstruction from l=0 to l=l
        irradiance_map = einsum(sph_basis[..., 0:curr_n_terms], sph_coeffs[0:curr_n_terms, ...], "h w n_terms, n_terms c -> h w c") / solid_angle[..., None] # TODO: Hack divide by solid angle to normalize the irradiance map??
        irradiance_maps[l, ...] = irradiance_map
        
    return irradiance_maps


# -----------------------------
# Spherical Harmonic Indexing
# -----------------------------
def sph_term_within_band(l: int) -> int:  # (2l + 1)
    return 2 * l + 1

def sph_indices_total(l_max: int) -> int:  # (l_max + 1)^2
    return (l_max + 1) * (l_max + 1)

def sph_index_from_lm(l: int, m: int) -> int:
    # band l occupies indices [l*l, (l+1)^2 - 1], with m mapped as (l + m)
    # => index = l*l + (l + m) = l*l + l + m
    return l * l + l + m

def l_from_index(idx: int) -> int:
    # exact integer sqrt: largest l with l*l <= idx
    return math.isqrt(idx)

def sph_l_max_from_indices_total(n_terms: int) -> int:
    # inverse of (l_max + 1)^2; validate if you like
    root = math.isqrt(n_terms)
    # optional guard:
    # assert root * root == n_terms, f"n_terms={n_terms} is not a perfect square"
    return root - 1

def lm_from_index(idx: int) -> tuple[int, int]:
    l = l_from_index(idx)
    m = idx - (l * l + l)   # <-- the key fix
    return l, m

# ------------------------------------------------------------
# Dominant Light Direction and Color
# ------------------------------------------------------------
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
    eps = 1e-8  # Small epsilon to avoid division by zero

    red_band_aligned_xyz = torch.tensor(
        [-red_band_1[2], -red_band_1[0], red_band_1[1]],
        device=sph_coeffs.device,
        dtype=sph_coeffs.dtype,
    )
    red_norm = torch.linalg.norm(red_band_aligned_xyz)
    red_band_aligned_xyz = red_band_aligned_xyz / (red_norm + eps)

    green_band_aligned_xyz = torch.tensor(
        [-green_band_1[2], -green_band_1[0], green_band_1[1]],
        device=sph_coeffs.device,
        dtype=sph_coeffs.dtype,
    )
    green_norm = torch.linalg.norm(green_band_aligned_xyz)
    green_band_aligned_xyz = green_band_aligned_xyz / (green_norm + eps)

    blue_band_aligned_xyz = torch.tensor(
        [-blue_band_1[2], -blue_band_1[0], blue_band_1[1]],
        device=sph_coeffs.device,
        dtype=sph_coeffs.dtype,
    )
    blue_norm = torch.linalg.norm(blue_band_aligned_xyz)
    blue_band_aligned_xyz = blue_band_aligned_xyz / (blue_norm + eps)

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

    color = torch.tensor([torch.dot(sph_coeffs_r, direction_sph_coeffs) / denominator,
                                  torch.dot(sph_coeffs_g, direction_sph_coeffs) / denominator,
                                  torch.dot(sph_coeffs_b, direction_sph_coeffs) / denominator], device=env_map_sph_coeffs.device, dtype=env_map_sph_coeffs.dtype)
    
    # Normalize color to 0-255
    # TODO: this doesn't match with the blog post, kinda a hack.
    color = 255.0 * (color / torch.linalg.norm(color))
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
    env_map_sph_coeffs, sph_basis = project_env_to_coefficients(env_map, l_max)
    env_map_reconstructed = reconstruct_sph_coeffs_to_env(H, W, env_map_sph_coeffs, sph_basis)

    # Get DC Term
    # TODO: test which dc_color is correct.
    # dc_color = env_map_reconstructed[0, 0, 0, :] # any pixel is fine since it is the same for all pixels
    # dc_color = 255 * (dc_color / torch.linalg.norm(dc_color))
    dc_color = 255 * (env_map_sph_coeffs[0, :] / torch.linalg.norm(env_map_sph_coeffs[0, :]))

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
        dc_color=dc_color,
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
        dc_color=metrics.dc_color.cpu().numpy().tolist(),
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
    Visualize the SPH metrics with colored circles at dominant pixel locations and a legend on a dark gray background.
    
    Args:
        hdri: Input HDRI tensor (H, W, 3) - used for dimensions and device info
        sph_metrics: SPHMetrics or SPHMetricsCPU object containing dominant pixel information
    
    Returns:
        Visualization tensor with colored circles at dominant pixels and legend on a dark gray background
    """
    H, W, _ = hdri.shape
    
    # Create light gray background (easier to see colored elements)
    background_color = torch.tensor([0.2, 0.2, 0.2], device=hdri.device, dtype=hdri.dtype)
    vis_hdri = background_color.expand(H, W, 3).clone()
    
    # Define colors for the three dominant pixels
    red_color = torch.tensor([1.0, 0.0, 0.0], device=hdri.device, dtype=hdri.dtype)
    green_color = torch.tensor([0.0, 1.0, 0.0], device=hdri.device, dtype=hdri.dtype)  
    blue_color = torch.tensor([0.0, 0.0, 1.0], device=hdri.device, dtype=hdri.dtype)
    
    # Get dominant pixel coordinates and draw circles around them
    dominant_pixels = [
        (sph_metrics.dominant_pixel, red_color, "Dominant Pixel"),
        (sph_metrics.dominant_pixel_rgb_color_difference, green_color, "Dominant Pixel RGB Color Difference"),
        (sph_metrics.dominant_pixel_rgb_luminance, blue_color, "Dominant Pixel RGB Luminance")
    ]
    
    # Draw filled circles at each dominant pixel location
    for pixel_coords, color, label in dominant_pixels:
        # Handle different pixel_coords types (tensor vs list)
        if isinstance(pixel_coords, torch.Tensor):
            pixel_x = max(0, min(W-1, int(pixel_coords[0].item())))
            pixel_y = max(0, min(H-1, int(pixel_coords[1].item())))
        else:
            # pixel_coords is a list (SPHMetricsCPU)
            pixel_x = max(0, min(W-1, int(pixel_coords[0])))
            pixel_y = max(0, min(H-1, int(pixel_coords[1])))
        
        # Draw filled colored circle at the dominant pixel location
        center_y, center_x = pixel_y, pixel_x
        circle_radius = 8  # Larger circle for visibility
        
        for dy in range(-circle_radius, circle_radius + 1):
            for dx in range(-circle_radius, circle_radius + 1):
                y_coord = center_y + dy
                x_coord = center_x + dx
                
                # Check bounds
                if 0 <= y_coord < H and 0 <= x_coord < W:
                    # Calculate distance from center
                    dist_sq = dy*dy + dx*dx
                    
                    # Draw filled circle
                    if dist_sq <= circle_radius * circle_radius:
                        vis_hdri[y_coord, x_coord, :] = color
    
    # Convert to numpy for OpenCV text rendering
    vis_image = (vis_hdri.cpu().numpy() * 255).astype('uint8')
    
    # Create legend with proper text labels using OpenCV
    legend_items = [
        (red_color, "Dominant Pixel", 0),
        (green_color, "Dominant Pixel RGB Color Difference", 20),
        (blue_color, "Dominant Pixel RGB Luminance", 40)
    ]
    
    # Legend styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color_white = (255, 255, 255)  # White text for readability on dark background
    legend_x_start = 10
    legend_y_start = 30
    
    for color, label, y_offset in legend_items:
        legend_y = legend_y_start + y_offset
        legend_x = legend_x_start + 10
        
        # Convert color to BGR for OpenCV (0-255 range)
        color_bgr = tuple(int(c * 255) for c in color.cpu().numpy()[::-1])  # RGB to BGR
        
        # Draw filled colored circle as legend marker
        cv2.circle(vis_image, (legend_x, legend_y - 5), 8, color_bgr, -1)
        
        # Add text label next to the circle
        text_x = legend_x + 25
        text_y = legend_y
        cv2.putText(vis_image, label, (text_x, text_y), font, font_scale, text_color_white, font_thickness)
    
    # Convert back to torch tensor with same device and dtype as input
    vis_hdri = torch.from_numpy(vis_image.astype('float32') / 255.0).to(device=hdri.device, dtype=hdri.dtype)
    
    return vis_hdri


