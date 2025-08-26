from enum import Enum
import torch
import numpy as np
import math
from einops import einsum
from ..utils import generate_spherical_coordinates_map, spherical_to_cartesian, pixel_solid_angles, convert_theta, cartesian_to_spherical


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


# def spherical_to_sph_eval(spherical_coordinates: torch.Tensor, l:int, m:int) -> torch.Tensor:  # noqa: E741
#     """
    
#     :params spherical_coordinates (..., 2): where at each position on the image, we have the spherical coordinate on the sphere.
#     :params l
#     :params m

#     :returns Y (...): the basis function evaluated at a specific l and m for each position in spherical_coordinates
    
#     Source:
#     [3] Equation 6.
#     [5] P function.
#     """

#     def P(l:int, m:int, theta: torch.Tensor) -> float:
#         """
#         Evaluate Associated Legendre Polynomial P(l,m,x) at x 

#         :TODO  debug against https://github.com/mmp/pbrt-v2/blob/e6f6334f3c26ca29eba2b27af4e60fec9fdc7a8d/src/core/sh.cpp#L43

#         : params l:
#         : params m:
#         : params theta: (H, W) The polar angle θ is measured between the z-axis and the radial line r.

#         : return pmm, pmmp1, pll: (H, W)
#         """

#         pmm = torch.ones_like(theta)
        
#         if(m > 0):
#             somx2 = torch.sqrt((1-theta)*(1+theta))
#             # Use numerical stable computation for high m values
#             # pmm *= (-1)^m * (2m-1)!! * (somx2)^m
#             # Factor computation: (2m-1)!! = 1*3*5*...*(2m-1)
            
#             # Clamp somx2 to avoid numerical issues at poles
#             somx2_clamped = torch.clamp(somx2, min=1e-10)
            
#             if m <= 30:  # Use direct computation for small m
#                 factor = 1.0
#                 for i in range(1, m + 1):
#                     pmm *= (-factor) * somx2
#                     factor += 2.0
#             else:  # Use log-space for large m
#                 # log((2m-1)!!) = sum(log(2i-1)) for i=1 to m
#                 log_double_factorial = sum(math.log(2*i - 1) for i in range(1, m + 1))
#                 log_somx2_pow_m = m * torch.log(somx2_clamped)
#                 sign = (-1.0) ** m
                
#                 pmm = sign * torch.exp(log_double_factorial + log_somx2_pow_m)
#                 # Handle poles where somx2 was originally 0
#                 pmm = torch.where(somx2 < 1e-10, torch.zeros_like(pmm), pmm)
        
#         if (l == m):
#             return pmm
        
#         pmmp1 = theta * (2*m + 1)* pmm
#         if (l == m + 1):
#             return pmmp1

#         pll = torch.zeros_like(theta)
#         for ll in range(m+2, l+1):
#             pll = ((2*ll - 1.0)*theta*pmmp1 - (ll + m - 1.0)*pmm)/ (ll-m)
#             pmm = pmmp1
#             pmmp1 = pll

#         return pll
    
#     def K(l:int, m:int) -> torch.Tensor:
#         """
#         return K with numerical stability for high l values.
#         """
#         a = abs(m)
#         # Use log-space computation to avoid factorial overflow
#         if a == 0:
#             log_fr = 0.0
#         else:
#             log_fr = sum(math.log(i) for i in range(l - a + 1, l + a + 1))
#             log_fr = -log_fr  # Because we want (l-a)!/(l+a)!
        
#         # K = sqrt((2l+1) * fr / (4π))
#         log_K = 0.5 * (math.log(2 * l + 1) + log_fr - math.log(4.0 * math.pi))
#         K_value = math.exp(log_K)
#         K = torch.tensor(K_value, device=spherical_coordinates.device, dtype=spherical_coordinates.dtype)
#         return K 

#     # Prep
#     theta, phi = spherical_coordinates[..., 0], spherical_coordinates[..., 1] # (H, W)
#     cos_theta = torch.cos(convert_theta(theta))

#     # Compute Y
#     if (m==0):
#         return K(l,m) * P (l, m, cos_theta)
#     elif(m > 0):
#         return math.sqrt(2)*K(l,m)*torch.cos(m*phi)*P(l,m,cos_theta)
#     else:
#         return math.sqrt(2)*K(l,m)*torch.sin(-m*phi)*P(l,-m,cos_theta)

# def spherical_to_sph_basis(spherical_coordinates: torch.Tensor, l_max: int) -> torch.Tensor:
#     """
#     Convert from spherical_coordinates to the spherical harmonic basis (Ylm).

#     :params spherical_coordinates (..., 2): where at each position on the image, we have the spherical coordinate on the sphere.
#     :params l_max:  The maximum number of bands. The order of the basis to compute.
#     :returns Ylm (..., n_terms): The spherical harmonic basis function evaluated at each coordinate. (H, W, n_terms)

#     Sources: 
#     [4] Equation 10.
#     [5] SH function.
#     """

#     # Construct Ylm
#     spherical_shape = spherical_coordinates.shape       # (..., 3)
#     total_indices = sph_indices_total(l_max)
#     Ylm_shape = (*spherical_shape[:-1], total_indices)  # (..., n_terms)
#     Ylm = torch.zeros(Ylm_shape, device=spherical_coordinates.device, dtype=spherical_coordinates.dtype)

#     for l in range(l_max +1):  # noqa: E741
#         for m in range(-l, l+1):
#             index = sph_index_from_lm(l, m)
#             Ylm[..., index] = spherical_to_sph_eval(spherical_coordinates, l, m)

#     return Ylm


# def spherical_to_sph_basis_vectorized(spherical_coordinates: torch.Tensor, l_max: int) -> torch.Tensor:
#     """
#     Vectorized real spherical harmonics Y_l^m for all (l,m) up to l_max.

#     Input angles follow your convention:
#       θ = elevation in [-π/2, π/2], φ = azimuth in (−π, π].
#     Internally: cosΘ = cos(convert_theta(θ))  (so cosΘ = sin(elevation) with your converter).

#     Real basis:
#       Y_l^0    =  K(l,0) P_l^0(cosΘ)
#       Y_l^m    =  √2 K(l,m) cos(mφ) P_l^m(cosΘ),              m > 0
#       Y_l^{−m} =  √2 K(l,m) sin(mφ) P_l^m(cosΘ),              m > 0

#     P_l^m includes Condon–Shortley phase.
#     """
#     import math

#     def _norm_constants(l_max: int, device, dtype) -> torch.Tensor:
#         n_terms = sph_indices_total(l_max)
#         K = torch.zeros(n_terms, device=device, dtype=dtype)
#         sqrt4pi = math.sqrt(4.0 * math.pi)
#         for l in range(l_max + 1):
#             for m in range(-l, l + 1):
#                 idx = sph_index_from_lm(l, m)
#                 a = abs(m)
#                 # Use log-space computation to avoid factorial overflow for high l values
#                 # log(fr) = log((l-a)!) - log((l+a)!) = sum(log(i)) for i in range(l-a+1, l+a+1)
#                 if a == 0:
#                     log_fr = 0.0
#                 else:
#                     log_fr = sum(math.log(i) for i in range(l - a + 1, l + a + 1))
#                     log_fr = -log_fr  # Because we want (l-a)!/(l+a)!
                
#                 # K[idx] = sqrt((2l+1) * fr) / sqrt(4π) 
#                 # log(K[idx]) = 0.5 * (log(2l+1) + log(fr)) - 0.5 * log(4π)
#                 log_K = 0.5 * (math.log(2 * l + 1) + log_fr - math.log(4.0 * math.pi))
#                 K[idx] = math.exp(log_K)
#         return K

#     def _associated_legendre_all(l_max: int, x: torch.Tensor) -> torch.Tensor:
#         """
#         Compute P_l^m(x) for all 0<=m<=l<=l_max (vectorized).
#         Includes CS phase:
#           P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}
#           P_{m+1}^m = (2m+1) x P_m^m
#           P_l^m = ((2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m) / (l-m)  for l>=m+2
#         We only fill m>=0 entries; negative-m aren’t needed for P in the real basis.
#         """
#         shape = x.shape
#         device, dtype = x.device, x.dtype
#         n_terms = sph_indices_total(l_max)
#         P = torch.zeros((*shape, n_terms), device=device, dtype=dtype)

#         # robust sinΘ from x=cosΘ
#         sin_th = torch.sqrt((1 - x).clamp_min(0) * (1 + x).clamp_min(0))

#         # l=0, m=0
#         P[..., sph_index_from_lm(0, 0)] = 1.0

#         # *** Seed P_1^0 as well (m=0 case): P_{0+1}^0 = (2*0+1) * x * P_0^0 = x
#         if l_max >= 1:
#             P[..., sph_index_from_lm(1, 0)] = x

#         # Diagonals P_m^m and next band P_{m+1}^m for m>=1
#         for m in range(1, l_max + 1):
#             # Use log-space computation for numerical stability
#             # (2m-1)!! = (2m-1) * (2m-3) * ... * 3 * 1
#             log_df = sum(math.log(2 * i - 1) for i in range(1, m + 1))
            
#             # P_mm = (-1)^m * (2m-1)!! * (sin_theta)^m
#             # Use log space: log(P_mm) = m*log(-1) + log_df + m*log(sin_theta)
#             # Note: (-1)^m is just a sign, handle separately
#             sign = (-1.0) ** m
            
#             # Clamp sin_th to avoid log(0)
#             sin_th_clamped = torch.clamp(sin_th, min=1e-10)
#             log_sin_th_pow_m = m * torch.log(sin_th_clamped)
            
#             P_mm = sign * torch.exp(log_df + log_sin_th_pow_m)
            
#             # Handle the case where sin_th was originally 0 (poles)
#             P_mm = torch.where(sin_th < 1e-10, torch.zeros_like(P_mm), P_mm)
            
#             P[..., sph_index_from_lm(m, m)] = P_mm
#             if m < l_max:
#                 P[..., sph_index_from_lm(m + 1, m)] = (2 * m + 1) * x * P_mm

#         # Three-term recurrence for l >= m+2, all m>=0
#         for m in range(0, l_max + 1):
#             for l in range(m + 2, l_max + 1):
#                 idx_lm  = sph_index_from_lm(l, m)
#                 idx_l1m = sph_index_from_lm(l - 1, m)
#                 idx_l2m = sph_index_from_lm(l - 2, m)
#                 P[..., idx_lm] = ((2 * l - 1) * x * P[..., idx_l1m] - (l + m - 1) * P[..., idx_l2m]) / (l - m)
#         return P

#     # ---- compute ----
#     theta = spherical_coordinates[..., 0]  # elevation
#     phi   = spherical_coordinates[..., 1]  # azimuth
#     device, dtype = theta.device, theta.dtype

#     # cosΘ via your converter (Θ = θ - π/2; cos(Θ) = sin(θ))
#     x = torch.cos(convert_theta(theta))

#     # P_l^m(cosΘ) for m>=0, and normalization constants
#     P = _associated_legendre_all(l_max, x)
#     K = _norm_constants(l_max, device=device, dtype=dtype)

#     # Build Y
#     n_terms = sph_indices_total(l_max)
#     Y = torch.zeros((*spherical_coordinates.shape[:-1], n_terms), device=device, dtype=dtype)
#     sqrt2 = math.sqrt(2.0)

#     for l in range(l_max + 1):
#         # m = 0
#         idx0 = sph_index_from_lm(l, 0)
#         Y[..., idx0] = K[idx0] * P[..., idx0]
#         # m > 0 (reuse |m| for P and K)
#         for m in range(1, l + 1):
#             idx_pos = sph_index_from_lm(l,  m)
#             idx_neg = sph_index_from_lm(l, -m)
#             common  = K[idx_pos] * P[..., idx_pos]  # P_l^{m} with m>=0
#             Y[..., idx_pos] = sqrt2 * common * torch.cos(m * phi)
#             Y[..., idx_neg] = sqrt2 * common * torch.sin(m * phi)

#     return Y


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
    solid_angle = pixel_area * sin_theta

    # Integrate to get coefficients
    sph_coeffs = torch.stack([
        torch.sum(hdri_r[:, :, None] * sph_basis * solid_angle[..., None], dim=(0, 1)), # sum((H, W, None) * (..., n_terms) * (H, 1, None), axis(0,1)) = (n_terms)
        torch.sum(hdri_g[:, :, None] * sph_basis * solid_angle[..., None], dim=(0, 1)),
        torch.sum(hdri_b[:, :, None] * sph_basis * solid_angle[..., None], dim=(0, 1))
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

    # Compute cumulative irradiance maps (reconstruction up to each l_max)
    irradiance_maps = torch.zeros((l_max + 1, H, W, 3), device=sph_coeffs.device, dtype=sph_coeffs.dtype)
    for l in range(l_max + 1):  # noqa: E741
        curr_n_terms = sph_indices_total(l)
        # This gives cumulative reconstruction from l=0 to l=l
        irradiance_map = einsum(sph_basis[..., 0:curr_n_terms], sph_coeffs[0:curr_n_terms, ...], "h w n_terms, n_terms c -> h w c") 
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