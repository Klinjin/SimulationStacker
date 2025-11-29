import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from utils import arcmin_to_comoving, comoving_to_arcmin, bins_from_geomean_monotonic
import astropy.units as u
from astropy import cosmology
from typing import Optional, Tuple, Union
ArrayLike = Union[float, np.ndarray]

def total_mass(mass_grid, r_grid, r, pixel_size=1.0):
    """Calculate the total mass within a given radius.

    Args:
        mass_grid (np.ndarray): The mass distribution array.
        r_grid (np.ndarray): The radial grid corresponding to the mass distribution.
        r (float): The radius at which to calculate the total mass.
        pixel_size (float, optional): placeholder

    Returns:
        float: The total mass within the given radius. M_sun
    """
    pixArea = (pixel_size)**2
    return float(np.sum(mass_grid[r_grid < r]))

def CAP(mass_grid, r_grid, r, pixel_size=1.0):
    """Calculate the Circular Averages Profile (CAP) for a given radius.

    Args:
        mass_grid (np.ndarray): The mass distribution array.
        r_grid (np.ndarray): The radial grid corresponding to the mass distribution.
        r (float): The radius at which to calculate the CAP.
        pixel_size (float, optional): placeholder -- area unit conversion in stacker.stackMap()

    Returns:
        float: The value of the CAP at the given radius. M_sun or muK
    """
    
    r1 = r * np.sqrt(2.0)
    inDisk = 1.0 * (r_grid <= r)
    inRing = 1.0 * (r_grid > r) * (r_grid <= r1)
    sum_ring = np.sum(inRing)
    if sum_ring > 0:
        inRing *= np.sum(inDisk) / sum_ring
    
    return float(np.sum((inDisk - inRing) * mass_grid))

def delta_sigma(mass_grid, r_grid, r, dr=0.6, pixel_size=1.0):
    """Calculate the excess surface mass density (ΔΣ) for a given radius.

    Args:
        mass_grid (np.ndarray)s: The mass distribution array.
        r_grid (np.ndarray): The radial grid corresponding to the mass distribution.
        r (float): The radius at which to calculate ΔΣ.
        dr (float, optional): The width of the annulus. Defaults to 0.6. (arcminutes)
        pixel_size (float, optional): Physical pixel size (pc or arcmin). Currently unused but kept for API consistency.

    Returns:
        float: The value of ΔΣ at the given radius. M_sun / kpc^2 
    """
    
    if dr is None:
        dr = np.sqrt(2) / 2 * r
    
        
    mean_sigma = np.sum(mass_grid[r_grid < r]) / (np.pi * r**2)
    r_mask = np.logical_and((r_grid >= r), (r_grid < r + dr))
    # sigma_value = np.sum(mass_grid[r_mask]) / (2 * np.pi * r * dr)
    sigma_value = np.sum(mass_grid[r_mask]) / (np.pi * ((r + dr)**2 - r**2))
    return float(mean_sigma - sigma_value)



def delta_sigma_kernel_map(
    mass_grid: np.ndarray,
    r_grid: np.ndarray,
    r: float,
    dr: float = 0.5,
    pixel_size: float = 1,
) -> float:
    """Compute ΔΣ using an analytical compensated kernel.

    This builds a compensated (aperture minus annulus) kernel from the provided
    radial grid and computes the dot product with the provided mass map.

    Args:
        mass_grid (np.ndarray): 2-D surface mass density map (same shape as
            ``r_grid``), in mass units per pixel.
        r_grid (np.ndarray): 2-D array of radial distances from the center for
            each pixel (same shape as ``mass_grid``). Units must match the units
            used for ``r`` and ``dr``.
        r (float): Aperture radius at which to evaluate ΔΣ.
        dr (float, optional): Thickness of the outer annulus (R < r < R+dr).
            Must be positive. Defaults to 0.5.
        pixel_size (float, optional): Linear size of one pixel in physical
            units (pc or arcmin). Used to scale the summed kernel value to physical area.
            Defaults to 1.

    Returns:
        float: The computed ΔΣ value (mass per unit area) as a Python float.

    Raises:
        ValueError: If ``dr`` is not positive.

    Notes:
        The kernel is constructed so that the integral (sum) over the kernel is
        zero (compensated). The function returns the sum of element-wise
        multiplication of ``mass_grid`` and the kernel, scaled by
        ``pixel_size_pc**2``.
    """

    if dr <= 0:
        raise ValueError("dr must be positive.")

    R_out = r + dr

    # Build compensated kernel analytically from r_grid
    kernel              = np.zeros_like(r_grid, dtype=float)
    kernel[r_grid < r]  = +1.0 / (np.pi * r**2)
    annulus             = (r_grid >= r) & (r_grid < R_out)
    kernel[annulus]     = -1.0 / (np.pi * (R_out**2 - r**2))
    # print(kernel.mean())
    kernel             -= kernel.mean()  # ensure ∫K dA = 0 numerically
    
    pixArea = (pixel_size)**2

    return float(np.sum(mass_grid * kernel) * pixArea)

def delta_sigma_kernel(
    mass_grid: np.ndarray,
    r_grid: np.ndarray,
    r: float,
    dr: float = 0.5,
    pixel_size: float = 1,
) -> float:
    """Compute ΔΣ using an analytical compensated kernel.

    This builds a compensated (aperture minus annulus) kernel from the provided
    radial grid and computes the dot product with the provided mass map.

    Args:
        mass_grid (np.ndarray): 2-D surface mass density map (same shape as
            ``r_grid``), in mass units per pixel.
        r_grid (np.ndarray): 2-D array of radial distances from the center for
            each pixel (same shape as ``mass_grid``). Units must match the units
            used for ``r`` and ``dr``.
        r (float): Aperture radius at which to evaluate ΔΣ.
        dr (float, optional): Thickness of the outer annulus (R < r < R+dr).
            Must be positive. Defaults to 0.5.
        pixel_size (float, optional): Linear size of one pixel in physical
            units (pc or arcmin). Used to scale the summed kernel value to physical area.
            Defaults to 1.

    Returns:
        float: The computed ΔΣ value (mass per unit area) as a Python float.

    Raises:
        ValueError: If ``dr`` is not positive.

    Notes:
        The kernel is constructed so that the integral (sum) over the kernel is
        zero (compensated). The function returns the sum of element-wise
        multiplication of ``mass_grid`` and the kernel, scaled by
        ``pixel_size_pc**2``.
    """

    if dr <= 0:
        raise ValueError("dr must be positive.")

    R_out = r + dr
    pixArea = (pixel_size)**2

    # Build compensated kernel analytically from r_grid
    kernel              = np.zeros_like(r_grid, dtype=float)
    disk                = (r_grid < r)
    kernel[disk]        = +1.0 / (pixArea * np.sum(disk))
    annulus             = (r_grid >= r) & (r_grid < R_out)
    kernel[annulus]     = -1.0 / (pixArea * np.sum(annulus))
    # print(kernel.mean())
    # kernel             -= kernel.mean()  # ensure ∫K dA = 0 numerically
    
    return float(np.sum(mass_grid * kernel))

def delta_sigma_ring(
                    mass_grid: np.ndarray,
                    r_grid: np.ndarray,
                    r: ArrayLike,
                    *,
                    pixel_area_pc2: Optional[float] = None,
                    pixel_size_pc: Optional[float] = None,
                    connectivity: int = 8,
                    return_parts: bool = False,
) -> Union[float, np.ndarray, Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]]:
    """
    Excess surface density ΔΣ(R) = Σ̄(<R) − Σ(R) using a *border-of-core* definition for Σ(R).

    Parameters
    ----------
    mass_grid : (Ny, Nx) array
        Mass per pixel (e.g., Msun*h per pixel). NaNs are ignored.
    r_grid : (Ny, Nx) array
        Radial map from the chosen center (same units as R, but only the inequality r<R matters).
    r : float or array-like
        Radii at which to evaluate ΔΣ. (The bordering set is defined by pixel adjacency, not by a width.)
    pixel_area_pc2 : float, optional
        Physical area of a pixel in pc^2. If given, used as-is.
    pixel_size_pc : float, optional
        Physical pixel size (pc). Used only if `pixel_area_pc2` is None. Then pixel_area_pc2 = pixel_size_pc**2.
    connectivity : {4, 8}, default=8
        Neighbor definition for the “border”: 4-connected (N,S,E,W) or 8-connected (also diagonals).
    return_parts : bool, default=False
        If True, return (ΔΣ, Σ̄(<R), Σ(R)).

    Returns
    -------
    ΔΣ : float or ndarray of shape like r
        In Msun*h / pc^2. If return_parts=True, also returns (mean_sigma, ring_sigma).

    Notes
    -----
    • Core set:    C(R)  = { (i,j) : r_grid[i,j] < R }
    • Border set:  B(R)  = { neighbors of C(R) by chosen connectivity }  C(R)
      Implemented via logical shifts of the core mask (no scipy required).
    • Σ̄(<R) and Σ(R) are computed as pixel-wise means divided by pixel area, which equals
      (sum mass) / (sum area) when all pixels have the same area.
    """
    if pixel_area_pc2 is None:
        if pixel_size_pc is None:
            raise ValueError("Provide `pixel_area_pc2` (pc^2) or `pixel_size_pc` (pc).")
        pixel_area_pc2 = float(pixel_size_pc) ** 2

    mass = np.asarray(mass_grid, dtype=float)
    rad  = np.asarray(r_grid, dtype=float)
    good = np.isfinite(mass) & np.isfinite(rad)

    r_arr = np.atleast_1d(r).astype(float)
    out_delta = np.empty_like(r_arr, dtype=float)
    out_mean  = np.empty_like(r_arr, dtype=float)
    out_ring  = np.empty_like(r_arr, dtype=float)

    # neighbor offsets for the chosen connectivity
    if connectivity == 4:
        offsets = [(-1,0),(1,0),(0,-1),(0,1)]
    elif connectivity == 8:
        offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    def _shift_mask(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """Shift boolean mask by (dy,dx) with zero/False padding (no wrap)."""
        H, W = m.shape
        out = np.zeros_like(m, dtype=bool)
        y0 = max(0,  dy) ; y1 = min(H, H+dy)
        x0 = max(0,  dx) ; x1 = min(W, W+dx)
        out[y0:y1, x0:x1] = m[y0-dy:y1-dy, x0-dx:x1-dx]
        return out

    for i, R in enumerate(r_arr):
        core = good & (rad < R)
        n_core = core.sum()
        if n_core == 0:
            raise ValueError(f"No pixels with r < {R:.6g}. Increase R or check r_grid.")

        # build border: neighbors of core minus core
        nb_any = np.zeros_like(core, dtype=bool)
        for dy, dx in offsets:
            nb_any |= _shift_mask(core, dy, dx)
        border = nb_any & (~core) & good

        n_border = border.sum()
        if n_border == 0:
            raise ValueError(
                f"No bordering pixels for R={R:.6g} with connectivity={connectivity}. "
                "Increase R or check grid size/center."
            )

        core_mean_mass_per_pix   = np.nanmean(mass[core])
        border_mean_mass_per_pix = np.nanmean(mass[border])

        mean_sigma  = core_mean_mass_per_pix   / pixel_area_pc2
        ring_sigma  = border_mean_mass_per_pix / pixel_area_pc2
        delta_sigma = mean_sigma - ring_sigma

        out_mean[i]  = mean_sigma
        out_ring[i]  = ring_sigma
        out_delta[i] = delta_sigma

    if np.ndim(r) == 0:
        if return_parts:
            return float(out_delta[0]), float(out_mean[0]), float(out_ring[0])
        return float(out_delta[0])
    else:
        if return_parts:
            return out_delta, out_mean, out_ring
        return out_delta


def delta_sigma_mccarthy(
    mass_map: np.ndarray,
    theta_grid_arcmin: np.ndarray,
    pixel_scale_arcmin: float,
    *,
    # choose ONE of the following two pathways:
    z: Optional[float] = None,
    cosmo: Optional[cosmology.Cosmology] = None,   # e.g. astropy.cosmology.FlatLambdaCDM(...)
    # OR
    chi_mpc_over_h: Optional[float] = None, # if you already know χ in Mpc/h, supply it here

    # binning and selection
    # choose ONE of the following two pathways:
    rbins_mpc_over_h: Optional[np.ndarray] = None,
    # OR
    rmin_theta: Optional[float] = None,  # if rbins_mpc_over_h is None, set rmin = chi * rmin_theta (arcmin)
    rmax_theta: Optional[float] = None,  # if rbins_mpc_over_h is None, set rmax = chi * rmax_theta (arcmin)
    n_rbins: int = 9,                    # if rbins_mpc_over_h is None, use this many log-spaced bins
    
    theta_max_arcmin: Optional[float] = None,
    # bookkeeping
    nan_policy: str = "ignore",
    return_per_pixel: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Compute ΔΣ(R) from a total-mass angular map using the interior-mean minus local Σ construction.

    This implements the procedure described in McCarthy et al. (arXiv:2410.19905):
    (1) convert angular pixels to comoving area via A_pix = (χ · θ_pix)^2 (small-angle),
    (2) convert each pixel's angular separation θ to projected comoving radius R = χ · θ,
    (3) sort pixels by R and define ΔΣ_k = ⟨Σ(<R_k)⟩ − Σ(R_k),
    (4) bin ΔΣ by R and report ΔΣ-weighted bin centers.

    Args:
      mass_map (np.ndarray): Total mass per pixel (e.g., Msun·h) on the sky with shape (Ny, Nx).
      theta_grid_arcmin (np.ndarray): Angular separation of each pixel from the galaxy center,
        in arcminutes, shape (Ny, Nx).
      pixel_scale_arcmin (float): Angular size of one pixel on a side (arcminutes). Assumes
        square pixels and the small-angle approximation.

      z (Optional[float]): Redshift of the lens plane/shell. Must be provided together with
        `cosmo` if `chi_mpc_over_h` is not provided.
      cosmo (Optional[cosmology.Cosmology]): Astropy cosmology. Used with `z` to compute the
        comoving transverse distance D_M(z) and convert it to χ in Mpc/h.
      chi_mpc_over_h (Optional[float]): Comoving radial distance χ in Mpc/h. If provided,
        `z` and `cosmo` are not required.

      rbins_mpc_over_h (Optional[np.ndarray]): Radial bin edges in Mpc/h. Must be a strictly
        increasing 1D array if provided. If None, bins are constructed from `rmin_theta`,
        `rmax_theta`, and `n_rbins`.
      rmin_theta (Optional[float]): Minimum angular radius (arcminutes) used to set the inner
        radial scale when `rbins_mpc_over_h` is None. Converted to comoving using χ.
      rmax_theta (Optional[float]): Maximum angular radius (arcminutes) used to set the outer
        radial scale when `rbins_mpc_over_h` is None. Converted to comoving using χ.
      n_rbins (int): Number of desired radial bins when `rbins_mpc_over_h` is None.

      theta_max_arcmin (Optional[float]): If provided, only pixels with θ ≤ `theta_max_arcmin`
        (arcminutes) are used.
      nan_policy (str): How to handle NaNs. "ignore" (default) drops NaN pixels; "raise"
        raises a ValueError if NaNs are present.
      return_per_pixel (bool): If True, also return the per-pixel (R_i, ΔΣ_i) prior to binning.

    Returns:
      Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        - R_centers: (Nb,) array of ΔΣ-weighted bin centers (Mpc/h).
        - dSigma_binned: (Nb,) array of mean ΔΣ per bin with units of
          (mass units)/(Mpc/h)^2 (e.g., Msun·h/(Mpc/h)^2).
        - counts: (Nb,) array with the number of pixels in each bin.
        - (R_all, dSigma_all): Optional tuple of per-pixel projected radii and ΔΣ values.

    Raises:
      ValueError: If neither (`chi_mpc_over_h`) nor (`z` and `cosmo`) are provided;
        if no valid pixels remain after masking/cuts; if `rbins_mpc_over_h` is invalid;
        or if `rmin_theta`/`rmax_theta` are missing when needed.
      ImportError: If `z` and `cosmo` are provided but Astropy units are not available.

    Notes:
      - Pixel comoving area: A_pix = (χ · θ_pix)^2 with θ_pix in radians.
      - Projected comoving radius: R = χ · θ with θ in radians.
      - Binning uses `np.digitize`; values exactly equal to the leftmost edge may be excluded
        due to the left-open, right-closed convention.
      - When `rbins_mpc_over_h` is None, bin edges are constructed so that the geometric mean
        of consecutive edges matches a desired monotonically increasing sequence derived from
        `rmin_theta`, `rmax_theta`, and `n_rbins`. See `bins_from_geomean_monotonic`.
    """
    # ---- validate angular inputs
    mass = np.asarray(mass_map, dtype=float)
    theta_arcmin = np.asarray(theta_grid_arcmin, dtype=float)

    # ---- get χ [Mpc/h]
    if chi_mpc_over_h is None:
        if (z is None) or (cosmo is None):
            raise ValueError("Provide either chi_mpc_over_h OR (z AND cosmo).")

        if u is None:
            raise ImportError("astropy is required to compute χ from (z, cosmo). Install astropy.")

        # comoving transverse distance D_M(z) in Mpc
        chi_Mpc = cosmo.comoving_transverse_distance(z).to_value(u.Mpc) # type: ignore
        h = cosmo.H0.value / 100.0 # type: ignore
        chi = chi_Mpc * h  # → Mpc/h
    else:
        chi = float(chi_mpc_over_h)

    # ---- masks and optional theta cut
    good = np.isfinite(mass) & np.isfinite(theta_arcmin)
    if nan_policy == "raise" and not np.all(good):
        raise ValueError("NaNs or infs found in inputs; set nan_policy='ignore' to drop them.")
    if theta_max_arcmin is not None:
        good &= (theta_arcmin <= float(theta_max_arcmin))

    if np.count_nonzero(good) == 0:
        raise ValueError("No valid pixels after masking/theta cut.")

    # ---- angular → comoving conversions
    arcmin_to_rad = np.pi / (180.0 * 60.0)
    theta_rad = theta_arcmin[good] * arcmin_to_rad
    pixel_scale_rad = float(pixel_scale_arcmin) * arcmin_to_rad

    # Pixel comoving linear size and area (small-angle)
    L_pix = chi * pixel_scale_rad                  # Mpc/h
    # Convert L_pix to pc for area calculation
    L_pix *= 1e6                                   # pc/h 
    L_pix /= (cosmo.H0.value / 100.0) # pc  (h cancels out) # type: ignore

    A_pix = L_pix ** 2                              # (pc)^2   (same for all pixels here)


    # Projected comoving radius per pixel
    R = (theta_rad * chi).ravel()                   # Mpc/h

    # ---- Σ map (comoving surface density): mass per pixel divided by comoving pixel area
    Sigma = (mass[good].ravel()) / A_pix            # (mass units) / (Mpc/h)^2

    # ---- sort by radius and build per-pixel ΔΣ = <Σ(<R)> - Σ(R)
    # print(R)
    order = np.argsort(R)
    R_sorted = R[order]
    # R_sorted = arcmin_to_comoving(R_sorted, z) if z is not None else R_sorted
    # print(R_sorted)
    S_sorted = Sigma[order]

    cumsum = np.cumsum(S_sorted, dtype=float)
    idx = np.arange(1, S_sorted.size + 1, dtype=float)
    S_cummean = cumsum / idx

    dSigma_pixels = S_cummean - S_sorted

    # ---- choose radial bins if not provided
    if rbins_mpc_over_h is None:
        # TODO: Check over this part to make sure it makes sense.
        if rmin_theta is None or rmax_theta is None:
            raise ValueError("If `rbins_mpc_over_h` is None, provide both `rmin_theta` and `rmax_theta`.")

        if n_rbins < 1:
            # Single "bin" fallback
            R_center = np.array([R_sorted.mean()])
            dSigma_bin = np.array([dSigma_pixels.mean()])
            counts = np.array([dSigma_pixels.size])
            if return_per_pixel:
                return R_center, dSigma_bin, counts, (R_sorted, dSigma_pixels)
            return R_center, dSigma_bin, counts # type: ignore
        
        # r_intended = np.linspace(rmin_theta, rmax_theta, n_rbins)  # arcmin
        rp_min = arcmin_to_comoving(rmin_theta, z) / 1000  # Mpc/h
        rp_max = arcmin_to_comoving(rmax_theta, z) / 1000  # Mpc/h
        r_desired = np.linspace(rp_min, rp_max, n_rbins)  # Mpc/h
        rbins = bins_from_geomean_monotonic(r_desired, pick='geom')[0]
        # print(rbins)
        convert_r_theta = True
        # rbins = np.geomspace(rmin, rmax, n_rbins+1)  # default: 12 bins
    else:
        rbins = np.asarray(rbins_mpc_over_h, dtype=float)
        if rbins.ndim != 1 or rbins.size < 2 or not np.all(np.diff(rbins) > 0):
            raise ValueError("`rbins_mpc_over_h` must be a 1D strictly increasing array of edges.")
        convert_r_theta = False

    # ---- bin ΔΣ by R with ΔΣ-weighted centers
    which = np.digitize(R_sorted, rbins) - 1
    Nb = rbins.size - 1
    dSigma_binned = np.empty(Nb, dtype=float)
    R_center = np.empty(Nb, dtype=float)
    counts = np.zeros(Nb, dtype=int)

    for b in range(Nb):
        sel = (which == b)
        counts[b] = np.count_nonzero(sel)
        if counts[b] == 0:
            # print("Warning: empty radial bin", b, "in [", rbins[b], rbins[b+1], "] Mpc/h")
            dSigma_binned[b] = np.nan
            R_center[b] = 0.5 * (rbins[b] + rbins[b+1])
            continue
        vals = dSigma_pixels[sel]
        radii = R_sorted[sel]
        dSigma_binned[b] = np.mean(vals)
        denom = np.sum(vals)
        # TODO: this line is wrong. Fix it.
        R_center[b] = (np.sum(vals * radii) / denom) if denom != 0 else np.mean(radii)

    if convert_r_theta:
        # R_center = comoving_to_arcmin(R_center * 1000, z)  # already in Mpc/h
        R_center = comoving_to_arcmin(r_desired * 1000, z) # convert to arcmin

    if return_per_pixel:
        return R_center, dSigma_binned, counts, (R_sorted, dSigma_pixels)
    return R_center, dSigma_binned, counts # type: ignore


def CAP_from_mass(r, radii_2D, M_2D, k=3):
    """Calculate the Circular Averages Profile (CAP) from 1D mass distribution.

    Args:
        r (float): The radius at which to calculate the CAP.
        radii_2D (np.ndarray): The radial grid corresponding to the 2D mass distribution.
        M_2D (np.ndarray): The 2D mass distribution array.
        k (int, optional): The degree of the spline interpolation. Defaults to 3.

    Raises:
        ValueError: If the dimensions of the input arrays are not compatible.

    Returns:
        float: The value of the CAP at the given radius.
    """
    
    r = np.atleast_1d(r)
    if M_2D.ndim == 1:
        M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D, k=k)
        return 2 * M_interp(r) - M_interp(np.sqrt(2) * r) # type: ignore
    elif M_2D.ndim == 2:
        result = []
        for i in range(M_2D.shape[1]):
            M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D[:, i], k=k)
            result.append(2 * M_interp(r) - M_interp(np.sqrt(2) * r)) # type: ignore
        return np.stack(result, axis=-1)
    else:
        raise ValueError("M_2D must be either a 1D or 2D array.")

def DSigma_from_mass(r, radii_2D, M_2D, k=3):
    """Calculate the excess surface mass density (ΔΣ) from 1D mass distribution.

    Args:
        r (float): The radius at which to calculate ΔΣ.
        radii_2D (np.ndarray): The radial grid corresponding to the 2D mass distribution.
        M_2D (np.ndarray): The 2D mass distribution array.
        k (int, optional): The degree of the spline interpolation. Defaults to 3.

    Raises:
        ValueError: If the dimensions of the input arrays are not compatible.

    Returns:
        float: The value of ΔΣ at the given radius.
    """
    
    r = np.atleast_1d(r)
    if M_2D.ndim == 1:
        M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D, k=k)
        dM_dr_interp = M_interp.derivative()
        return M_interp(r) / (np.pi * r**2) - dM_dr_interp(r) / (2 * np.pi * r)
    elif M_2D.ndim == 2:
        result = []
        for i in range(M_2D.shape[1]):
            M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D[:, i], k=k)
            dM_dr_interp = M_interp.derivative()
            result.append(M_interp(r) / (np.pi * r**2) - dM_dr_interp(r) / (2 * np.pi * r))
        return np.stack(result, axis=-1)
    else:
        raise ValueError("M_2D must be either a 1D or 2D array.")