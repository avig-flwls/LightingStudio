from enum import Enum
import torch
import numpy as np
import math

from ..utils import generate_spherical_coordinates_map, spherical_to_cartesian, pixel_solid_angles, convert_theta


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
    
    # For l_max > 4, fall back to the original implementation
    if l_max > 4:
        for l in range(5, l_max + 1):  # noqa: E741
            for m in range(-l, l + 1):
                index = sph_index_from_lm(l, m)
                Ylm[..., index] = cartesian_to_sph_eval(cartesian_coordinates, l, m)
    
    return Ylm




def spherical_to_sph_eval(spherical_coordinates: torch.Tensor, l:int, m:int) -> torch.Tensor:  # noqa: E741
    """
    
    :params spherical_coordinates (..., 2): where at each position on the image, we have the spherical coordinate on the sphere.
    :params l
    :params m

    :returns Y (...): the basis function evaluated at a specific l and m for each position in spherical_coordinates
    
    Source:
    [3] Equation 6.
    [5] P function.
    """

    def P(l:int, m:int, theta: torch.Tensor) -> float:
        """
        Evaluate Associated Legendre Polynomial P(l,m,x) at x 

        :TODO  debug against https://github.com/mmp/pbrt-v2/blob/e6f6334f3c26ca29eba2b27af4e60fec9fdc7a8d/src/core/sh.cpp#L43

        : params l:
        : params m:
        : params theta: (H, W) The polar angle θ is measured between the z-axis and the radial line r.

        : return pmm, pmmp1, pll: (H, W)
        """

        pmm = torch.ones_like(theta)
        
        if(m > 0):
            somx2 = torch.sqrt((1-theta)*(1+theta))
            factor = 1.0

            for i in range(1, m + 1):
                pmm *= (-factor) * somx2
                factor += 2.0
        
        if (l == m):
            return pmm
        
        pmmp1 = theta * (2*m + 1)* pmm
        if (l == m + 1):
            return pmmp1

        pll = torch.zeros_like(theta)
        for ll in range(m+2, l+1):
            pll = ((2*ll - 1.0)*theta*pmmp1 - (ll + m - 1.0)*pmm)/ (ll-m)
            pmm = pmmp1
            pmmp1 = pll

        return pll
    
    def K(l:int, m:int) -> torch.Tensor:
        """
        return K.
        """
        K_value = ((2 * l + 1) * math.factorial(l-abs(m))) / (4*torch.pi*math.factorial(l+abs(m)))
        K = torch.sqrt(torch.tensor(K_value, device=spherical_coordinates.device, dtype=spherical_coordinates.dtype))
        return K 

    # Prep
    theta, phi = spherical_coordinates[..., 0], spherical_coordinates[..., 1] # (H, W)

    # Compute Y
    if (m==0):
        return K(l,m) * P (l, m, torch.cos(convert_theta(theta)))
    elif(m > 0):
        return math.sqrt(2)*K(l,m)*torch.cos(m*phi)*P(l,m,torch.cos(convert_theta(theta)))
    else:
        return math.sqrt(2)*K(l,m)*torch.sin(-m*phi)*P(l,-m,torch.cos(convert_theta(theta)))

def spherical_to_sph_basis(spherical_coordinates: torch.Tensor, l_max: int) -> torch.Tensor:
    """
    Convert from spherical_coordinates to the spherical harmonic basis (Ylm).

    :params spherical_coordinates (..., 2): where at each position on the image, we have the spherical coordinate on the sphere.
    :params l_max:  The maximum number of bands. The order of the basis to compute.
    :returns Ylm (..., n_terms): The spherical harmonic basis function evaluated at each coordinate. (H, W, n_terms)

    Sources: 
    [4] Equation 10.
    [5] SH function.
    """

    # Construct Ylm
    spherical_shape = spherical_coordinates.shape       # (..., 3)
    total_indices = sph_indices_total(l_max)
    Ylm_shape = (*spherical_shape[:-1], total_indices)  # (..., n_terms)
    Ylm = torch.zeros(Ylm_shape, device=spherical_coordinates.device, dtype=spherical_coordinates.dtype)

    for l in range(l_max +1):  # noqa: E741
        for m in range(-l, l+1):
            index = sph_index_from_lm(l, m)
            Ylm[..., index] = spherical_to_sph_eval(spherical_coordinates, l, m)

    return Ylm


def associated_legendre_polynomial_vectorized(l_max: int, cos_theta: torch.Tensor) -> torch.Tensor:
    """
    Vectorized computation of Associated Legendre polynomials for all (l,m) pairs up to l_max.
    
    :param l_max: Maximum spherical harmonic band
    :param cos_theta: Cosine of polar angle, shape (...,)
    :return: P_lm values, shape (..., n_terms) where n_terms = (l_max+1)^2
    """
    shape = cos_theta.shape
    device = cos_theta.device
    dtype = cos_theta.dtype
    
    # Initialize output tensor
    n_terms = sph_indices_total(l_max)
    P = torch.zeros((*shape, n_terms), device=device, dtype=dtype)
    
    # Precompute sin_theta for efficiency
    sin_theta = torch.sqrt(1 - cos_theta * cos_theta)
    
    # l=0, m=0: P_0^0 = 1
    if l_max >= 0:
        P[..., sph_index_from_lm(0, 0)] = 1.0
    
    # l=1 terms
    if l_max >= 1:
        P[..., sph_index_from_lm(1, -1)] = sin_theta      # P_1^1
        P[..., sph_index_from_lm(1, 0)] = cos_theta       # P_1^0  
        P[..., sph_index_from_lm(1, 1)] = sin_theta       # P_1^1 (same as P_1^{-1})
    
    # Higher order terms using recurrence relations
    for l in range(2, l_max + 1):  # noqa: E741
        # For m = l (diagonal terms): P_l^l = (-1)^l * (2l-1)!! * sin^l(theta)
        if l <= l_max:
            # Compute (2l-1)!! = 1*3*5*...*(2l-1)
            double_factorial = 1.0
            for i in range(1, l + 1):
                double_factorial *= (2 * i - 1)
            
            sign = (-1) ** l
            P[..., sph_index_from_lm(l, l)] = sign * double_factorial * (sin_theta ** l)
            P[..., sph_index_from_lm(l, -l)] = P[..., sph_index_from_lm(l, l)]  # P_l^{-l} = P_l^l
        
        # For m = l-1: P_l^{l-1} = cos_theta * (2l-1) * P_{l-1}^{l-1}
        if l >= 1:
            P[..., sph_index_from_lm(l, l-1)] = cos_theta * (2*l - 1) * P[..., sph_index_from_lm(l-1, l-1)]
            P[..., sph_index_from_lm(l, -(l-1))] = P[..., sph_index_from_lm(l, l-1)]  # P_l^{-(l-1)} = P_l^{l-1}
        
        # For m < l-1: Use three-term recurrence relation
        # P_l^m = ((2l-1)*cos_theta*P_{l-1}^m - (l+m-1)*P_{l-2}^m) / (l-m)
        for m in range(l-2, -1, -1):
            if l >= 2:
                numerator = (2*l - 1) * cos_theta * P[..., sph_index_from_lm(l-1, m)] - (l + m - 1) * P[..., sph_index_from_lm(l-2, m)]
                P[..., sph_index_from_lm(l, m)] = numerator / (l - m)
                if m > 0:
                    P[..., sph_index_from_lm(l, -m)] = P[..., sph_index_from_lm(l, m)]
    
    return P


def spherical_harmonic_normalization_vectorized(l_max: int, device: torch.device = None) -> torch.Tensor:
    """
    Precompute normalization constants K(l,m) for all spherical harmonic terms.
    
    :param l_max: Maximum spherical harmonic band
    :param device: Device to place tensor on
    :return: K values, shape (n_terms,) where n_terms = (l_max+1)^2
    """
    n_terms = sph_indices_total(l_max)
    K = torch.zeros(n_terms, device=device)
    
    for l in range(l_max + 1):  # noqa: E741
        for m in range(-l, l + 1):
            idx = sph_index_from_lm(l, m)
            abs_m = abs(m)
            
            # K(l,m) = sqrt((2*l+1) * (l-|m|)! / (4*pi * (l+|m|)!))
            factorial_ratio = math.factorial(l - abs_m) / math.factorial(l + abs_m)
            K[idx] = math.sqrt((2 * l + 1) * factorial_ratio / (4 * torch.pi))
    
    return K


def spherical_to_sph_basis_vectorized(spherical_coordinates: torch.Tensor, l_max: int) -> torch.Tensor:
    """
    GPU-optimized vectorized version of spherical_to_sph_basis that computes all basis functions simultaneously.
    
    :param spherical_coordinates: (..., 2) spherical coordinates (theta, phi)
    :param l_max: Maximum number of bands
    :return: Ylm (..., n_terms) spherical harmonic basis functions
    """
    # Extract coordinates
    theta, phi = spherical_coordinates[..., 0], spherical_coordinates[..., 1]
    
    # Convert to physics convention and get cosine
    theta_physics = theta  # Assuming convert_theta is already applied if needed
    cos_theta = torch.cos(theta_physics)
    
    # Compute Associated Legendre polynomials for all (l,m) pairs
    P = associated_legendre_polynomial_vectorized(l_max, cos_theta)
    
    # Get normalization constants
    K = spherical_harmonic_normalization_vectorized(l_max, device=spherical_coordinates.device)
    
    # Initialize output
    shape = spherical_coordinates.shape
    n_terms = sph_indices_total(l_max)
    Ylm = torch.zeros((*shape[:-1], n_terms), device=spherical_coordinates.device, dtype=spherical_coordinates.dtype)
    
    # Compute spherical harmonics
    for l in range(l_max + 1):  # noqa: E741
        for m in range(-l, l + 1):
            idx = sph_index_from_lm(l, m)
            
            if m == 0:
                # Y_l^0 = K(l,0) * P_l^0(cos_theta)
                Ylm[..., idx] = K[idx] * P[..., idx]
            elif m > 0:
                # Y_l^m = sqrt(2) * K(l,m) * cos(m*phi) * P_l^m(cos_theta)
                Ylm[..., idx] = torch.sqrt(torch.tensor(2.0)) * K[idx] * torch.cos(m * phi) * P[..., idx]
            else:  # m < 0
                # Y_l^m = sqrt(2) * K(l,|m|) * sin(|m|*phi) * P_l^{|m|}(cos_theta)
                abs_m = abs(m)
                abs_m_idx = sph_index_from_lm(l, abs_m)
                Ylm[..., idx] = torch.sqrt(torch.tensor(2.0)) * K[abs_m_idx] * torch.sin(abs_m * phi) * P[..., abs_m_idx]
    
    return Ylm





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

def sph_term_within_band(l: int) -> int:  # noqa: E741
	return (l*2)+1

def sph_indices_total(l_max: int) -> int:
	return (l_max + 1) * (l_max + 1)

def sph_index_from_lm(l:int, m:int) -> int:
	return l*l+l+m

def l_from_index(idx:int) -> int:
	return int(np.sqrt(idx))

def sph_l_max_from_indices_total(n_terms: int) -> int:
	return int(np.sqrt(n_terms) - 1)

