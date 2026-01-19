# !pip install illustris_python
import sys

import numpy as np
import matplotlib.pyplot as plt
import h5py

import scipy
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.cosmology import FlatLambdaCDM, Planck18
from astropy import units as u
import astropy.constants as const

# from abacusnbody.analysis.tsc import tsc_parallel
import time
import yaml
import glob
# import json
# import pprint 

# sys.path.append('../../illustrisPython/')
import illustris_python as il 
# from stacker import SimulationStacker

# Filter Functions:

def fft_smoothed_map(D, fwhm_arcmin, pixel_size_arcmin):
    """
    Smooth a 2D map D with a Gaussian beam of FWHM (arcmin) using FFTs.
        Periodic boundary conditions are assumed (same as gaussian_filter(mode='wrap')).


    Args:
        D (np.ndarray): 2D map to smooth.
        fwhm_arcmin (float): Full width at half maximum of the Gaussian beam (in arcminutes).
        pixel_size_arcmin (float): Size of each pixel (in arcminutes).

    Returns:
        np.ndarray: Smoothed 2D map.

    completed tests: 
        Test to make sure that this works as expected when compared to gaussian_smoothed_map
    """
    
    D = np.asarray(D, dtype=np.float32)
    n0, n1 = D.shape
    assert n0 == n1, "Map must be square for this implementation."
    n = n0

    # Build flat-sky multipole grid l_x, l_y (radian^-1)
    #TODO: this should be the physical rather than angular variant, incorrect
    theta_pix_rad = np.deg2rad(pixel_size_arcmin / 60.0) 
    lx = np.fft.fftfreq(n, d=theta_pix_rad) * 2.0 * np.pi
    ly = lx
    LX, LY = np.meshgrid(lx, ly, indexing='xy')
    ellsq = LX**2 + LY**2

    # Beam transfer function B_ell
    B_ell = gauss_beam(ellsq, fwhm_arcmin)  # uses radians internally

    # FFT, multiply, inverse FFT
    dfour = scipy.fft.fftn(D, workers=-1)
    dksmo = dfour * B_ell
    drsmo = np.real(scipy.fft.ifftn(dksmo, workers=-1))
    return drsmo
    
def gauss_beam(ellsq, fwhm):
    """
    Gaussian beam of size fwhm
    
    Args:
        ellsq: squared angular multipole term used in the Gaussian beam function's exponent. 
            In flat-sky this is defined as (l_x^2 + l_y^2), where l_x and l_y are the Fourier space frequencies in rad^-1.
        fwhm: Gaussian FWHM of the beam (in arcminutes)
        
    Returns:
        np.ndarray: The Gaussian beam transfer function.
    """
    tht_fwhm = np.deg2rad(fwhm/60.)
    return np.exp(-0.5*(tht_fwhm**2.)*(ellsq)/(8.*np.log(2.)))

def gaussian_smoothed_map(D, fwhm_arcmin, pixel_size_arcmin):
    """
    DEPRECIATED: Check below for new Convolution code
    Convolve the map with a Gaussian beam.

    Args:
        D (np.ndarray): 2D numpy array of the field for the given particle type.
        fwhm_arcmin (float): Full width at half maximum of the Gaussian beam in arcminutes.
        pixel_size_arcmin (float): Size of the pixel in arcminutes.

    Returns:
        np.ndarray: Convolved 2D numpy array.
    """
    
    sigma_pixels = fwhm_arcmin / (2.355 * pixel_size_arcmin)

    # Apply Gaussian filter
    convolved_map = gaussian_filter(D, sigma=sigma_pixels, mode='wrap')
    
    return convolved_map

def format_string_sci(num):
    """
    Converts a number to scientific notation with up to 2 decimal places,
    removing trailing zeroes after the decimal point.

    Args:
        num (float or int): The number to convert.

    Returns:
        str: Scientific notation string (e.g., '5e12', '5.2e13', '5.12e12').
    """
    base, exponent = f"{num:.2e}".split('e')
    base = base.rstrip('0').rstrip('.')  # Remove trailing zeros and decimal if needed
    return f"{base}e{int(exponent)}"

    
# Unit Conversions and Cosmology
def ksz_from_delta_sigma(
    delta_sigma,
    z_l,
    v_los = 300 * u.km / u.s,
    cosmology = Planck18,
    mu_e = 1.14,
    delta_sigma_is_comoving = False,
    cov_delta_sigma = None,   # optional covariance on ΔΣ
    return_tau = False,
):
    """
    Convert weak-lensing ΔΣ(R) to kSZ ΔT(R) in μK assuming Σ_gas = f_b * ΔΣ_phys
    with f_b = Ω_b / Ω_m and fully ionized gas.

    Parameters
    ----------
    delta_sigma : astropy.units.Quantity
        ΔΣ (mass surface density), any mass/area unit (e.g., 200 * u.Msun/u.pc**2).
    z_l : float
        Lens redshift (only used if `delta_sigma_is_comoving=True`).
    v_los : Quantity, default 300 km/s
        Electron-weighted LOS peculiar velocity (sign convention: +away gives -ΔT).
    cosmology : astropy.cosmology instance
        Cosmology to use for T_CMB, Ω_b and Ω_m.
d    mu_e : float, default 1.14
        Mean molecular weight per free electron.
    delta_sigma_is_comoving : bool
        If True, convert ΔΣ_com → ΔΣ_phys by multiplying by (1+z_l)^2.
    cov_delta_sigma : None or array-like or astropy Quantity
        Optional covariance matrix for ΔΣ. If a Quantity, its unit should be (mass/area)^2.
        If comoving, the code multiplies it by (1+z_l)^4 to convert to physical units.
    return_tau : bool
        If True, also return τ (per element) implied by f_b * ΔΣ.

    Returns
    -------
    dT_muK : np.ndarray or float
        kSZ temperature in μK (same shape as `delta_sigma`).
    tau : np.ndarray or float, optional
        Optical depth (returned if `return_tau=True`).
    cov_dT_muK : np.ndarray, optional
        Covariance in μK^2 (returned if `cov_delta_sigma` is not None).
        
    TODO:
    Check the correctness of the unit handling in the covariance propagation.
    """
    T_CMB = cosmology.Tcmb0
    # T_CMB = 2.7255 * u.K  # FIRAS/Planck normalization
    Omega_b = cosmology.Ob0
    Omega_m = cosmology.Om0
    
    if not hasattr(delta_sigma, "unit"):
        raise TypeError("`delta_sigma` must be an astropy Quantity with mass/area units.")

    # Convert comoving ΔΣ → physical ΔΣ
    ds_phys = delta_sigma * ((1.0 + z_l) ** 2 if delta_sigma_is_comoving else 1.0)
    ds_phys = ds_phys.to(u.kg / u.m**2)

    # Constant gas fraction
    f_b = Omega_b / Omega_m

    # Electron column and optical depth
    Sigma_gas = f_b * ds_phys                               # kg/m^2
    N_e = (Sigma_gas / (mu_e * const.m_p)).to(1 / u.m**2)         # 1/m^2 # type: ignore
    tau = (const.sigma_T * N_e).decompose().value                 # dimensionless # type: ignore

    # Linear coefficient α mapping Σ_tot (phys) → ΔT (μK)
    alpha = (-T_CMB * (v_los / const.c) * const.sigma_T / (mu_e * const.m_p)) # K * m^2/kg # type: ignore
    alpha *= f_b                                            # include f_b
    # As μK per (kg/m^2):
    alpha_muK_per_Sigma = alpha.to(u.uK / (u.kg / u.m**2)).value

    # Mean ΔT in μK
    dT_muK = (alpha * ds_phys).to(u.uK).value

    # If covariance is provided, propagate it: Cov_T = α^2 * Cov_Σphys
    outputs = [dT_muK]
    if return_tau:
        outputs.append(tau)

    if cov_delta_sigma is not None:
        cov = cov_delta_sigma
        # If quantity, convert to (kg/m^2)^2; if plain array, assume same units as delta_sigma
        if hasattr(cov, "unit"):
            cov = cov.to((u.kg / u.m**2)**2).value
        else:
            cov = np.asarray(cov, dtype=float)
            # If ΔΣ was comoving, rescale covariance by (1+z)^4 to get physical
            if delta_sigma_is_comoving:
                cov = cov * (1.0 + z_l)**4

            # If ΔΣ was not a Quantity (but here it always is), we would need a unit factor.

        cov_shape = np.shape(cov)
        ds = np.atleast_1d(ds_phys.value)
        if cov_shape != (ds.size, ds.size):
            raise ValueError(f"`cov_delta_sigma` must be an (N,N) matrix with N={ds.size}.")

        cov_dT = (alpha_muK_per_Sigma**2) * cov   # μK^2
        outputs.append(cov_dT)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)

    

def comoving_to_arcmin(L_com_kpch, z, cosmo=Planck18):
    """Convert a comoving length at redshift z into angular size [arcmin].

    Args:
        L_com_kpch (float or array-like): Comoving length in kpc/h.
        z (float): Redshift.
        cosmo (astropy.cosmology, optional): Cosmology object. Defaults to Planck18.

    Returns:
        float or array-like: Angular size in arcminutes.
    """
    # Convert kpc/h -> Mpc
    L_com = (np.asarray(L_com_kpch) * u.kpc) / cosmo.h   # comoving kpc
    L_com = L_com.to(u.Mpc)                              # comoving Mpc

    # Comoving line-of-sight distance
    DM = cosmo.comoving_transverse_distance(z)  # Mpc

    # Angular size
    theta = (L_com / DM) * u.rad

    return theta.to(u.arcmin).value


def arcmin_to_comoving(theta_arcmin, z, cosmo=Planck18):
    """Convert an angular size [arcmin] at redshift z into comoving length [kpc/h].

    Args:
        theta_arcmin (float or array-like): Angular size in arcminutes.
        z (float): Redshift.
        cosmo (astropy.cosmology, optional): Cosmology object. Defaults to Planck18.

    Returns:
        float or array-like: Comoving length in kpc/h.
    """
    # Angular size in radians
    theta = np.asarray(theta_arcmin) * u.arcmin

    # Comoving line-of-sight distance
    DM = cosmo.comoving_transverse_distance(z)  # Mpc

    # Comoving length
    L_com = (theta.to(u.rad).value * DM).to(u.Mpc)

    # Convert Mpc -> kpc/h
    return (L_com.to(u.kpc) * cosmo.h).value
    

def bins_from_geomean_monotonic(r_desired, bins0=None, pick='mid'):
    """
    Recover strictly increasing bin edges from desired geometric-mean “centers.”

    Given a positive, strictly increasing array `r_desired` of length N where each element is
    intended to be the geometric mean of consecutive edges,
      r_desired[i] = sqrt(bins[i] * bins[i+1]),
    this function returns a strictly increasing array of edges `bins` (length N+1). The
    construction depends on the choice of the leftmost edge `bins0 = bins[0]`. Monotonicity
    imposes an open interval (lower, upper) for valid `bins0`; this function computes that
    interval and either validates a provided `bins0` or selects one according to `pick`.

    Args:
      r_desired (array_like): Length-N positive array of target geometric means (centers).
      bins0 (Optional[float]): Optional initial left edge b0. If provided, it must satisfy
        lower < b0 < upper, where (lower, upper) is the allowed interval computed by this function.
      pick (str): Strategy to choose `bins0` when it is not provided. Options:
        - 'mid' or 'mean': arithmetic midpoint of (lower, upper) [default].
        - 'geom' or 'geometric': geometric mean sqrt(lower * upper).
        - 'lower': just above the lower bound (lower + 1e-12).
        - 'upper': just below the upper bound (upper − 1e-12).

    Returns:
      Tuple[np.ndarray, Tuple[float, float]]:
        - bins: (N+1,) strictly increasing array of edges such that
          sqrt(bins[i] * bins[i+1]) == r_desired[i] for all i.
        - (lower, upper): Open interval for valid `bins0`: lower < bins0 < upper.

    Raises:
      ValueError: If any element of `r_desired` is non-positive; if no monotonic solution exists
        for the given `r_desired`; if a provided `bins0` lies outside (lower, upper); or if
        `pick` is not recognized.
      RuntimeError: If the constructed `bins` fail the strict monotonicity check
        (should not occur for valid inputs).

    Notes:
      - The recurrence uses bins[i+1] = r_desired[i]**2 / bins[i]. Strictly increasing bins imply
        bins[i] < r_desired[i] at each step, which yields alternating constraints on `bins0` that
        define the open interval (lower, upper).
      - `r_desired` need only be positive and increasing; it does not have to be logarithmically
        spaced. If you want log-even **centers**, supply `r_desired` from `np.geomspace`.
    """
    import numpy as np
    r_desired = np.asarray(r_desired, dtype=float)
    if np.any(r_desired <= 0):
        raise ValueError("r_desired must be positive")
    n = len(r_desired)

    # compute allowed interval for bins0 by propagating k_i and alternating exponent sign
    lower = 0.0
    upper = np.inf
    k = 1.0
    s = 1  # sign: +1 for even index b_i = k*b0, -1 for odd b_i = k/b0
    for i in range(n):
        if s == 1:
            # constraint: k * b0 < r_desired[i]  => b0 < r_desired[i] / k
            upper = min(upper, r_desired[i] / k)
        else:
            # constraint: k / b0 < r_desired[i]  => b0 > k / r_desired[i]
            lower = max(lower, k / r_desired[i])
        # update k and sign for next index
        k = r_desired[i]**2 / k
        s = -s

    if not (lower < upper and upper > 0):
        raise ValueError("No monotonic increasing solution for given r_desired")

    # choose bins0 if not provided
    if bins0 is None:
        if pick in ('mid', 'mean'):
            bins0 = 0.5 * (lower + upper)
        elif pick in ('geom', 'geometric'):
            bins0 = np.sqrt(lower * upper)
        elif pick == 'lower':
            bins0 = lower + 1e-12
        elif pick == 'upper':
            bins0 = upper - 1e-12
        else:
            raise ValueError("unknown pick option")
    else:
        bins0 = float(bins0)
        if not (lower < bins0 < upper):
            raise ValueError(f"bins0={bins0} outside allowed interval ({lower}, {upper})")

    # build bins via recurrence
    bins = np.empty(n + 1, dtype=float)
    bins[0] = bins0
    for i in range(n):
        bins[i + 1] = r_desired[i]**2 / bins[i]

    # final monotonicity check (safety)
    if not np.all(bins[1:] > bins[:-1]):
        raise RuntimeError("constructed bins are not strictly increasing (unexpected)")

    return bins, (lower, upper)