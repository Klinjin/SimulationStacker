"""make_fgas_profiles.py
========================
Generate a two-panel figure of the normalised ionized-gas fraction profile

    f_gas^{norm}(R) = [CAP^{ionized_gas}(R) / CAP^{total}(R)]
                    / (OmegaBaryon / Omega0)

for two halo selection methods:

  Left  panel : mass-cut FoF halo selection (select_massive_halos)
  Right panel : SHAM / abundance-matching subhalo selection

Both 2D (beam-convolved, arcmin radii) and 3D (spherical, comoving kpc/h
radii) stacking are supported via the ``stack.dim`` YAML flag.

2D path
-------
Four ``stackMap`` calls per simulation (two particle types × two selection
methods).  Each particle map is loaded once and cached in ``stacker.maps``,
then reused for both halo selection methods.

3D path
-------
Two ``makeField`` calls build the 3D density fields for ``ionized_gas`` and
``total``.  The mass-cut path mirrors ``compute_3d_profile_ratio`` from
``make_ratios3x2.py``.  The SHAM path mirrors the ``use_subhalos=True``
branch of ``SimulationStacker.stack_on_array``.

Note: the YAML ``filter_type`` / ``filter_type_2`` fields are unused in 3D
stacking; the 3D path always accumulates all pixels within a sphere
(equivalent to a 3D cumulative filter).

Usage
-----
    python make_fgas_profiles.py -p configs/fgas_profiles_z05.yaml
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import yaml
import argparse

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.append('../src/')
from utils import comoving_to_arcmin
from stacker import SimulationStacker
from halos import select_halos
from mask_utils import get_cutout_indices_3d, sum_over_cutouts

# ---------------------------------------------------------------------------
# Global matplotlib style  (matches make_ratios3x2.py exactly)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex":      True,
    "mathtext.fontset": "cm",
    "font.size":        20,
    "axes.titlesize":   20,
    "axes.labelsize":   20,
    "xtick.labelsize":  20,
    "ytick.labelsize":  20,
    "legend.fontsize":  13,
})

# Per-suite colourmaps: IllustrisTNG sims → twilight, SIMBA sims → hsv.
_COLOURMAPS = {'IllustrisTNG': 'twilight', 'SIMBA': 'hsv'}

# Ordered reference TNG sim list used to fix colour positions so that omitting
# TNG100-1 from the config does not shift the colours of TNG300-1/Illustris-1.
# Extend this list here if new TNG sims are added.
_TNG_REFERENCE_ORDER = ['TNG100-1', 'TNG300-1', 'Illustris-1']

# Panel labels: (a) = left (mass-cut), (b) = right (SHAM).
_PANEL_LABELS = ['(a)', '(b)']

# Default OmegaBaryon fallbacks (not stored in all simulation headers).
_OMEGA_BARYON_ILLUSTRIS_DEFAULT = 0.0456
_OMEGA_BARYON_SIMBA_DEFAULT     = 0.048


# ===========================================================================
# Helper functions
# ===========================================================================

def setup_stacker(sim: dict, sim_type_name: str, redshift: float):
    """Instantiate a SimulationStacker and derive cosmological quantities.

    Mirrors the same function in make_ratios3x2.py.

    Parameters
    ----------
    sim : dict
        Single simulation entry from the YAML ``simulations`` block.
        Must contain ``name`` and ``snapshot``; SIMBA entries also need
        ``feedback``.
    sim_type_name : str
        ``'IllustrisTNG'`` or ``'SIMBA'``.
    redshift : float
        Target redshift.

    Returns
    -------
    stacker     : SimulationStacker
    OmegaBaryon : float
    cosmo       : FlatLambdaCDM
    sim_label   : str   — human-readable legend label
    """
    sim_name = sim['name']
    snapshot = sim['snapshot']

    if sim_type_name == 'IllustrisTNG':
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type_name)
        try:
            OmegaBaryon = stacker.header['OmegaBaryon']
        except KeyError:
            # Illustris-1 and older TNG runs do not store OmegaBaryon in header.
            OmegaBaryon = _OMEGA_BARYON_ILLUSTRIS_DEFAULT
        sim_label = sim_name

    elif sim_type_name == 'SIMBA':
        feedback = sim['feedback']
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type_name,
                                    feedback=feedback)
        OmegaBaryon = _OMEGA_BARYON_SIMBA_DEFAULT
        sim_label = f"{sim_name}_{feedback}"

    else:
        raise ValueError(f"Unknown simulation type: {sim_type_name!r}")

    cosmo = FlatLambdaCDM(
        H0=100 * stacker.header['HubbleParam'],
        Om0=stacker.header['Omega0'],
        Tcmb0=2.7255 * u.K,
        Ob0=OmegaBaryon,
    )

    return stacker, OmegaBaryon, cosmo, sim_label


def _fgas_profile_and_err(profiles_ig: np.ndarray,
                           profiles_tot: np.ndarray,
                           OmegaBaryon: float,
                           Omega0: float):
    """Compute the baryon-fraction-normalised f_gas profile and its
    propagated standard error from per-halo stacked profiles.

    f_gas^{norm}(R) = [mean(ionized_gas)(R) / mean(total)(R)]
                    / (OmegaBaryon / Omega0)

    Error propagation (two means assumed independent):

        d(f_gas)/f_gas = sqrt( (d(A)/A)^2 + (d(B)/B)^2 )

    where A = mean(ionized_gas), B = mean(total).

    Parameters
    ----------
    profiles_ig, profiles_tot : ndarray, shape (n_radii, n_halos)
        Per-halo stacked ionized_gas and total profiles.
    OmegaBaryon : float
    Omega0 : float
        Total matter density parameter.

    Returns
    -------
    fgas_norm : ndarray, shape (n_radii,)
    err       : ndarray, shape (n_radii,)
        One-sigma uncertainty from quadrature propagation.
    """
    mean_ig  = np.mean(profiles_ig,  axis=1)
    mean_tot = np.mean(profiles_tot, axis=1)

    se_ig  = np.std(profiles_ig,  axis=1) / np.sqrt(profiles_ig.shape[1])
    se_tot = np.std(profiles_tot, axis=1) / np.sqrt(profiles_tot.shape[1])

    # Guard against near-zero means to avoid silent NaN / inf in both the
    # central value and the relative-error terms.
    with np.errstate(invalid='ignore', divide='ignore'):
        fgas_norm = np.where(
            np.abs(mean_tot) > 0,
            mean_ig / mean_tot / (OmegaBaryon / Omega0),
            np.nan,
        )
        rel_ig  = np.where(np.abs(mean_ig)  > 0, se_ig  / mean_ig,  np.nan)
        rel_tot = np.where(np.abs(mean_tot) > 0, se_tot / mean_tot, np.nan)

    err = np.abs(fgas_norm) * np.sqrt(rel_ig ** 2 + rel_tot ** 2)

    return fgas_norm, err


def _print_selection_stats(stacker: SimulationStacker, params: dict) -> None:
    """Print diagnostic statistics for both halo selection methods.

    For the mass-cut selection: number of selected FoF groups and their
    mean GroupMass.

    For the SHAM selection: number of selected subhalos, mean host halo mass,
    number of unique parent FoF halos, and the satellite fraction.

    The SHAM selection mirrors the ``use_subhalos=True`` branch of
    ``SimulationStacker.stack_on_array``: subhalos are ranked and selected
    by stellar mass (``SubhaloMStar``), consistent with the observational
    motivation for stellar-mass-based abundance matching.

    The satellite fraction is computed separately using total bound mass
    (``SubhaloMass``) to identify the central of each FoF group — the
    subhalo with the highest ``SubhaloMass`` — independent of the stellar
    mass used for selection.  All other selected subhalos in the same group
    are counted as satellites.
    """
    halo_mass_avg         = params['halo_mass_avg']
    halo_mass_upper       = params['halo_mass_upper']
    halo_abundance_target = params['halo_abundance_target']

    haloes    = stacker.loadHalos()
    subhalos  = stacker.loadSubHalos()

    # ------------------------------------------------------------------
    # Mass-cut selection
    # ------------------------------------------------------------------
    halo_mask = select_halos(haloes['GroupMass'], 'massive',
                             target_average_mass=halo_mass_avg,
                             upper_mass_bound=halo_mass_upper)
    selected_group_masses = haloes['GroupMass'][halo_mask]
    n_mass     = selected_group_masses.shape[0]
    mean_mass  = np.mean(selected_group_masses)
    print(f"  [Mass-cut]  N_halos = {n_mass:d},  "
          f"<M_group> = {mean_mass:.3e}  (log10 = {np.log10(mean_mass):.2f})")

    # ------------------------------------------------------------------
    # SHAM selection  (mirrors stack_on_array use_subhalos=True branch)
    # Selection uses stellar mass (SubhaloMStar), matching stack_on_array.
    # SubhaloMass (total bound mass) is kept separately for is_central below.
    # ------------------------------------------------------------------
    haloMStar_sub = subhalos['SubhaloMStar']  # used for SHAM ranking
    haloMass_sub  = subhalos['SubhaloMass']   # used for is_central detection

    if halo_mass_upper is not None:
        parent_mass = haloes['GroupMass'][subhalos['SubhaloGrNr']]
        valid       = np.where(parent_mass <= halo_mass_upper)[0]
        local_mask  = select_halos(haloMStar_sub[valid], 'abundance',
                                   target_number=halo_abundance_target,
                                   Lbox=stacker.header['BoxSize'])
        sham_mask = valid[local_mask]
    else:
        sham_mask = select_halos(haloMStar_sub, 'abundance',
                                 target_number=halo_abundance_target,
                                 Lbox=stacker.header['BoxSize'])

    n_sham         = len(sham_mask)
    selected_grps  = subhalos['SubhaloGrNr'][sham_mask]
    unique_grps    = np.unique(selected_grps)
    n_unique_halos = len(unique_grps)

    # Mean host halo mass: use GroupMass of each unique parent halo, counting
    # each halo once regardless of how many selected subhalos it contributes.
    mean_host_mass = np.mean(haloes['GroupMass'][unique_grps])

    # Identify central subhalos: for each FoF group, the central is the
    # subhalo with the highest SubhaloMass.  np.maximum.at accumulates the
    # per-group maximum over ALL subhalos (not just the selected ones).
    n_grps       = haloes['GroupMass'].shape[0]
    max_sub_mass = np.zeros(n_grps)
    np.maximum.at(max_sub_mass, subhalos['SubhaloGrNr'].astype(int), haloMass_sub)

    selected_masses = haloMass_sub[sham_mask]
    is_central      = np.isclose(selected_masses, max_sub_mass[selected_grps])
    n_centrals      = int(np.sum(is_central))
    n_satellites    = int(np.sum(~is_central))
    sat_frac        = n_satellites / n_sham

    print(f"  [SHAM]      N_subhalos = {n_sham:d},  "
          f"<M_host> = {mean_host_mass:.3e}  (log10 = {np.log10(mean_host_mass):.2f})")
    print(f"              N_unique_halos = {n_unique_halos:d},  "
          f"N_centrals = {n_centrals:d},  N_satellites = {n_satellites:d},  "
          f"satellite fraction = {sat_frac:.3f}")


def compute_fgas_2d(stacker: SimulationStacker, params: dict,
                    OmegaBaryon: float, cosmo: FlatLambdaCDM):
    """Stack ionized_gas and total 2D maps under both halo selections and
    compute the normalised f_gas profile for each.

    Four stackMap calls are made:

      Call 1: ionized_gas, mass-cut  → loads & caches ionized_gas map
      Call 2: total,       mass-cut  → loads & caches total map
      Call 3: ionized_gas, SHAM      → reuses cached ionized_gas map
      Call 4: total,       SHAM      → reuses cached total map

    Note on SHAM abundance target
    ------------------------------
    stackMap does not expose halo_abundance_target, so the SHAM calls always
    use the stack_on_array default of 5e-4 (cMpc/h)^-3.

    Parameters
    ----------
    stacker     : SimulationStacker
    params      : dict
        Stacking parameters from the YAML config.
    OmegaBaryon : float
    cosmo       : FlatLambdaCDM
        Cosmology object for the R200c → arcmin conversion.

    Returns
    -------
    radii        : ndarray — arcmin (scaled by rad_distance)
    fgas_mass    : ndarray — normalised f_gas for mass-cut selection
    err_mass     : ndarray
    fgas_sham    : ndarray — normalised f_gas for SHAM selection
    err_sham     : ndarray
    R200c_arcmin : float   — mean R200c of mass-cut halos in arcmin
    """
    pType  = params['particle_type']
    fType  = params['filter_type']
    pType2 = params['particle_type_2']
    fType2 = params['filter_type_2']
    pixelSize       = params['pixel_size']
    minR            = params['min_radius_2d']
    maxR            = params['max_radius_2d']
    nRadii          = params['num_radii_2d']
    radDist         = params['rad_distance']
    projection      = params['projection']
    load            = params['load_field']
    save            = params['save_field']
    sub_mean        = params['subtract_mean']
    z               = params['redshift']
    halo_mass_avg   = params['halo_mass_avg']
    halo_mass_upper = params['halo_mass_upper']

    # subtract_mean=True is unsafe: stackMap subtracts/restores in-place on
    # cached maps.  With 4 calls sharing 2 cached maps, FP rounding accumulates
    # between the mass-cut and SHAM calls for each map.
    if sub_mean:
        raise NotImplementedError(
            "subtract_mean=True is not supported in compute_fgas_2d. "
            "The four stackMap calls share two cached particle maps; the "
            "in-place subtract/restore cycle accumulates floating-point "
            "error between the mass-cut and SHAM calls for each map."
        )

    # Shared keyword arguments for all four stackMap calls.
    _common = dict(
        minRadius=minR, maxRadius=maxR, numRadii=nRadii,
        z=z, projection=projection, save=save, load=load,
        radDistance=radDist, pixelSize=pixelSize, subtract_mean=sub_mean,
    )

    # ------------------------------------------------------------------
    # Call 1: ionized_gas, mass-cut
    # ------------------------------------------------------------------
    radii, profiles_ig_mass = stacker.stackMap(
        pType, filterType=fType,
        use_subhalos=False,
        halo_mass_avg=halo_mass_avg, halo_mass_upper=halo_mass_upper,
        **_common,
    )

    # ------------------------------------------------------------------
    # Call 2: total, mass-cut
    # ------------------------------------------------------------------
    radii_2, profiles_tot_mass = stacker.stackMap(
        pType2, filterType=fType2,
        use_subhalos=False,
        halo_mass_avg=halo_mass_avg, halo_mass_upper=halo_mass_upper,
        **_common,
    )

    # ------------------------------------------------------------------
    # Call 3: ionized_gas, SHAM
    # Reuses the cached ionized_gas map; stacks with abundance-matched subhalos.
    # ------------------------------------------------------------------
    radii_3, profiles_ig_sham = stacker.stackMap(
        pType, filterType=fType,
        use_subhalos=True,
        **_common,
    )

    # ------------------------------------------------------------------
    # Call 4: total, SHAM
    # Reuses the cached total map; stacks with abundance-matched subhalos.
    # ------------------------------------------------------------------
    radii_4, profiles_tot_sham = stacker.stackMap(
        pType2, filterType=fType2,
        use_subhalos=True,
        **_common,
    )

    # All four calls use identical radius parameters; assert defensively.
    assert (np.allclose(radii, radii_2) and
            np.allclose(radii, radii_3) and
            np.allclose(radii, radii_4)), (
        "Stacking radii must match across all four stackMap calls. "
        "Check that minRadius/maxRadius/numRadii are identical for all calls."
    )

    # Compute normalised f_gas profiles for each selection.
    Omega0 = stacker.header['Omega0']
    fgas_mass, err_mass = _fgas_profile_and_err(
        profiles_ig_mass, profiles_tot_mass, OmegaBaryon, Omega0)
    fgas_sham, err_sham = _fgas_profile_and_err(
        profiles_ig_sham, profiles_tot_sham, OmegaBaryon, Omega0)

    # R200c from mass-cut halos, converted to arcmin.
    haloes    = stacker.loadHalos()
    halo_mask = select_halos(haloes['GroupMass'], 'massive',
                             target_average_mass=halo_mass_avg,
                             upper_mass_bound=halo_mass_upper)
    R200c_kpch   = np.mean(haloes['GroupRad'][halo_mask])
    R200c_arcmin = comoving_to_arcmin(R200c_kpch, z, cosmo=cosmo)

    return radii * radDist, fgas_mass, err_mass, fgas_sham, err_sham, R200c_arcmin


def compute_fgas_3d(stacker: SimulationStacker, params: dict,
                    OmegaBaryon: float):
    """Compute 3D spherical f_gas profiles for both mass-cut and SHAM
    halo selections.

    Builds 3D density fields for ``ionized_gas`` and ``total`` particle
    types, then stacks them in spherical apertures for each selection method.

    The mass-cut path mirrors ``compute_3d_profile_ratio`` from
    ``make_ratios3x2.py``.  The SHAM path mirrors the ``use_subhalos=True``
    branch of ``SimulationStacker.stack_on_array``, including the optional
    pre-filter of subhalos by parent FoF group mass.

    Parameters
    ----------
    stacker     : SimulationStacker
    params      : dict
        Stacking parameters from the YAML config.
    OmegaBaryon : float

    Returns
    -------
    radii      : ndarray — comoving kpc/h
    fgas_mass  : ndarray — normalised f_gas for mass-cut selection
    err_mass   : ndarray
    fgas_sham  : ndarray — normalised f_gas for SHAM selection
    err_sham   : ndarray
    R200c_kpch : float   — mean R200c of mass-cut halos (comoving kpc/h)
    """
    nPixels               = params['n_pixels']
    minR                  = params['min_radius_3d']
    maxR                  = params['max_radius_3d']
    nRadii                = params['num_radii_3d']
    projection            = params['projection']
    save                  = params['save_field']
    load                  = params['load_field']
    sub_mean              = params['subtract_mean']
    halo_mass_avg         = params['halo_mass_avg']
    halo_mass_upper       = params['halo_mass_upper']
    halo_abundance_target = params['halo_abundance_target']
    pType  = params['particle_type']
    pType2 = params['particle_type_2']
    Omega0 = stacker.header['Omega0']

    # Build 3D fields for both particle types.
    field_ig  = stacker.makeField(pType,  nPixels=nPixels, dim='3D',
                                   projection=projection, save=save, load=load)
    field_ig  = field_ig  - np.mean(field_ig)  if sub_mean else field_ig

    field_tot = stacker.makeField(pType2, nPixels=nPixels, dim='3D',
                                   projection=projection, save=save, load=load)
    field_tot = field_tot - np.mean(field_tot) if sub_mean else field_tot

    # Physical scale: comoving kpc/h per pixel.
    kpc_per_pixel = stacker.header['BoxSize'] / field_ig.shape[0]

    radii = np.linspace(minR, maxR, nRadii)

    # ---------------------------------------------------------------
    # Mass-cut selection
    # ---------------------------------------------------------------
    haloes    = stacker.loadHalos()
    halo_mask = select_halos(haloes['GroupMass'], 'massive',
                             target_average_mass=halo_mass_avg,
                             upper_mass_bound=halo_mass_upper)
    GroupPos_masked = (
        np.round(haloes['GroupPos'][halo_mask] / kpc_per_pixel).astype(int)
        % nPixels
    )
    R200c_kpch = np.mean(haloes['GroupRad'][halo_mask])

    profiles_ig_mass  = []
    profiles_tot_mass = []
    for r in radii:
        rr  = np.ones(GroupPos_masked.shape[0]) * r / kpc_per_pixel
        idx = get_cutout_indices_3d(field_ig, GroupPos_masked, rr)
        profiles_ig_mass.append(sum_over_cutouts(field_ig,  idx.copy()))
        profiles_tot_mass.append(sum_over_cutouts(field_tot, idx.copy()))

    profiles_ig_mass  = np.array(profiles_ig_mass)   # (n_radii, n_halos)
    profiles_tot_mass = np.array(profiles_tot_mass)

    # ---------------------------------------------------------------
    # SHAM selection
    # Mirrors the use_subhalos=True branch of stack_on_array, including
    # the optional pre-filter by parent FoF group mass.
    # ---------------------------------------------------------------
    subhalos      = stacker.loadSubHalos()
    haloMStar_sub = subhalos['SubhaloMStar']  # stellar mass used for SHAM ranking
    haloPos_sub   = subhalos['SubhaloPos']

    if halo_mass_upper is not None:
        # Pre-filter subhalos by parent FoF group mass (reuse loaded haloes).
        parent_mass = haloes['GroupMass'][subhalos['SubhaloGrNr']]
        valid       = np.where(parent_mass <= halo_mass_upper)[0]
        local_mask  = select_halos(haloMStar_sub[valid], 'abundance',
                                   target_number=halo_abundance_target,
                                   Lbox=stacker.header['BoxSize'])
        sham_mask = valid[local_mask]
    else:
        sham_mask = select_halos(haloMStar_sub, 'abundance',
                                 target_number=halo_abundance_target,
                                 Lbox=stacker.header['BoxSize'])

    SubhaloPos_masked = (
        np.round(haloPos_sub[sham_mask] / kpc_per_pixel).astype(int)
        % nPixels
    )

    profiles_ig_sham  = []
    profiles_tot_sham = []
    for r in radii:
        rr  = np.ones(SubhaloPos_masked.shape[0]) * r / kpc_per_pixel
        idx = get_cutout_indices_3d(field_ig, SubhaloPos_masked, rr)
        profiles_ig_sham.append(sum_over_cutouts(field_ig,  idx.copy()))
        profiles_tot_sham.append(sum_over_cutouts(field_tot, idx.copy()))

    profiles_ig_sham  = np.array(profiles_ig_sham)   # (n_radii, n_halos)
    profiles_tot_sham = np.array(profiles_tot_sham)

    # ---------------------------------------------------------------
    # Compute normalised f_gas profiles for each selection.
    # ---------------------------------------------------------------
    fgas_mass, err_mass = _fgas_profile_and_err(
        profiles_ig_mass, profiles_tot_mass, OmegaBaryon, Omega0)
    fgas_sham, err_sham = _fgas_profile_and_err(
        profiles_ig_sham, profiles_tot_sham, OmegaBaryon, Omega0)

    return radii, fgas_mass, err_mass, fgas_sham, err_sham, R200c_kpch


# ===========================================================================
# Main
# ===========================================================================

def main(path2config: str, verbose: bool = True):
    """Generate the two-panel f_gas profile figure.

    Parameters
    ----------
    path2config : str
        Path to the YAML configuration file.
    verbose : bool
        If True, print progress messages to stdout.
    """
    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_cfg = config.get('stack', {})
    plot_cfg  = config.get('plot',  {})

    dim = stack_cfg.get('dim', '2D')
    if dim not in ('2D', '3D'):
        raise ValueError(f"stack.dim must be '2D' or '3D', got {dim!r}")

    params = {
        'dim':             dim,
        'redshift':        stack_cfg.get('redshift',        0.5),
        'projection':      stack_cfg.get('projection',      'yz'),
        'load_field':      stack_cfg.get('load_field',      True),
        'save_field':      stack_cfg.get('save_field',      True),
        'subtract_mean':   stack_cfg.get('subtract_mean',   False),
        'particle_type':   stack_cfg.get('particle_type',   'ionized_gas'),
        'filter_type':     stack_cfg.get('filter_type',     'CAP'),
        'particle_type_2': stack_cfg.get('particle_type_2', 'total'),
        'filter_type_2':   stack_cfg.get('filter_type_2',   'CAP'),
        # 2D parameters
        'pixel_size':      stack_cfg.get('pixel_size',      0.5),
        'min_radius_2d':   stack_cfg.get('min_radius_2d',   1.0),
        'max_radius_2d':   stack_cfg.get('max_radius_2d',   10.0),
        'num_radii_2d':    stack_cfg.get('num_radii_2d',    11),
        'rad_distance':    stack_cfg.get('rad_distance',    1.0),
        # 3D parameters
        'n_pixels':        stack_cfg.get('n_pixels',        1000),
        'min_radius_3d':   stack_cfg.get('min_radius_3d',   200.0),
        'max_radius_3d':   stack_cfg.get('max_radius_3d',   4000.0),
        'num_radii_3d':    stack_cfg.get('num_radii_3d',    11),
        # Halo selection
        'halo_mass_avg':         stack_cfg.get('halo_mass_avg',         10 ** 13.22),
        'halo_mass_upper':       stack_cfg.get('halo_mass_upper',       5e14),
        # Used by the 3D SHAM path; the 2D path uses the stack_on_array default.
        'halo_abundance_target': stack_cfg.get('halo_abundance_target', 5e-4),
    }

    redshift = params['redshift']

    # ------------------------------------------------------------------
    # Plotting configuration
    # ------------------------------------------------------------------
    now       = datetime.now()
    yr_string = now.strftime("%Y-%m")
    dt_string = now.strftime("%m-%d")
    figPath   = Path(plot_cfg.get('fig_path', '../figures/')) / yr_string / dt_string
    figPath.mkdir(parents=True, exist_ok=True)

    figName         = plot_cfg.get('fig_name',       'fgas_profiles_z05')
    figType         = plot_cfg.get('fig_type',        'pdf')
    plot_error_bars = plot_cfg.get('plot_error_bars', True)

    # ------------------------------------------------------------------
    # Pre-assign colours per simulation suite:
    #   IllustrisTNG → 'twilight', positions fixed by _TNG_REFERENCE_ORDER
    #                  so omitting TNG100-1 does not shift other colours.
    #   SIMBA (1 sim) → 'hsv' at position 0.85 (last of 6, matching the
    #                   tSZ reference scripts); multiple sims spread over
    #                   [0.2, 0.85] as usual.
    # ------------------------------------------------------------------
    all_sims_flat = []
    for suite in config['simulations']:
        for sim in suite['sims']:
            all_sims_flat.append((suite['sim_type'], sim))

    # TNG: build a colour lookup keyed by sim name from the reference order.
    _tng_cmap       = matplotlib.colormaps[_COLOURMAPS['IllustrisTNG']]  # type: ignore
    _tng_ref_clrs   = _tng_cmap(np.linspace(0.2, 0.85, len(_TNG_REFERENCE_ORDER)))
    _tng_colour_map = {name: _tng_ref_clrs[i]
                       for i, name in enumerate(_TNG_REFERENCE_ORDER)}

    # SIMBA: single sim → position 0.85 (matches tSZ reference); else spread.
    _n_simba    = sum(1 for stype, _ in all_sims_flat if stype == 'SIMBA')
    _simba_cmap = matplotlib.colormaps[_COLOURMAPS['SIMBA']]  # type: ignore
    _simba_clrs = (_simba_cmap(np.linspace(0.2, 0.85, 6))[-1:]
                   if _n_simba == 1
                   else _simba_cmap(np.linspace(0.2, 0.85, _n_simba)))

    _simba_idx = 0
    sim_colours = {}
    for stype, sim in all_sims_flat:
        if stype == 'IllustrisTNG':
            name = sim['name']
            if name not in _tng_colour_map:
                raise ValueError(
                    f"TNG sim {name!r} is not in _TNG_REFERENCE_ORDER. "
                    f"Add it there to maintain consistent colours."
                )
            sim_colours[name] = _tng_colour_map[name]
        elif stype == 'SIMBA':
            label = f"{sim['name']}_{sim['feedback']}"
            sim_colours[label] = _simba_clrs[_simba_idx]
            _simba_idx += 1
        else:
            raise ValueError(
                f"No colormap defined for sim_type {stype!r}. "
                f"Known types: {list(_COLOURMAPS)}"
            )

    # ------------------------------------------------------------------
    # Create figure — 1×2 panels, 9 × 7 inches, shared y-axis
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 7), sharey=True)
    ax_mass = axes[0]   # left  panel: mass-cut selection
    ax_sham = axes[1]   # right panel: SHAM selection

    # R200c reference line: cached from the first IllustrisTNG sim processed.
    R200c_ref   = None
    R200c_label = None

    t0 = time.time()

    # ------------------------------------------------------------------
    # Loop over all simulations in the order they appear in the config.
    # ------------------------------------------------------------------
    for suite in config['simulations']:
        sim_type_name = suite['sim_type']

        for sim in suite['sims']:
            sim_name = sim['name']
            if verbose:
                fb_str = (f"  feedback={sim.get('feedback')}"
                          if sim_type_name == 'SIMBA' else '')
                print(f"\n{'='*55}")
                print(f"  {sim_name}{fb_str}  ({sim_type_name})")
                print(f"{'='*55}")

            stacker, OmegaBaryon, cosmo, sim_label = setup_stacker(
                sim, sim_type_name, redshift)
            colour = sim_colours[sim_label]

            if verbose:
                _print_selection_stats(stacker, params)

            if dim == '2D':
                if verbose:
                    pT  = params['particle_type']
                    pT2 = params['particle_type_2']
                    print(f"  Stacking {pT}/{pT2} 2D profiles "
                          f"(mass-cut ×2, SHAM ×2)...")
                radii, fgas_mass, err_mass, fgas_sham, err_sham, R200c_val = \
                    compute_fgas_2d(stacker, params, OmegaBaryon, cosmo)
            else:
                if verbose:
                    pT  = params['particle_type']
                    pT2 = params['particle_type_2']
                    print(f"  Stacking {pT}/{pT2} 3D profiles "
                          f"(mass-cut + SHAM)...")
                radii, fgas_mass, err_mass, fgas_sham, err_sham, R200c_val = \
                    compute_fgas_3d(stacker, params, OmegaBaryon)

            if verbose:
                units = 'arcmin' if dim == '2D' else 'kpc/h'
                print(f"  R200c(mass-cut) = {R200c_val:.3f} {units}")

            # Cache R200c from the first IllustrisTNG sim for the vline.
            if sim_type_name == 'IllustrisTNG' and R200c_ref is None:
                R200c_ref   = R200c_val
                R200c_label = sim_label

            # ---- Plot mass-cut panel ----
            ax_mass.plot(radii, fgas_mass, label=sim_label, color=colour,
                         lw=2, marker='o')
            if plot_error_bars:
                ax_mass.fill_between(radii,
                                     fgas_mass - err_mass,
                                     fgas_mass + err_mass,
                                     color=colour, alpha=0.2)

            # ---- Plot SHAM panel ----
            ax_sham.plot(radii, fgas_sham, label=sim_label, color=colour,
                         lw=2, marker='o')
            if plot_error_bars:
                ax_sham.fill_between(radii,
                                     fgas_sham - err_sham,
                                     fgas_sham + err_sham,
                                     color=colour, alpha=0.2)

    # ------------------------------------------------------------------
    # Axis decorations
    # ------------------------------------------------------------------
    x_label  = 'R [arcmin]' if dim == '2D' else 'R [comoving kpc/h]'
    xlim_max = (params['max_radius_2d'] * params['rad_distance'] + 0.5
                if dim == '2D' else None)

    for ax, title, panel_label in zip(
        [ax_mass, ax_sham], ['Mass-cut', 'SHAM'], _PANEL_LABELS
    ):
        # Unity reference line — the cosmic baryon fraction in normalised units.
        ax.axhline(1.0, color='k', ls='--', lw=2, zorder=0)

        # Shaded ±5 % band for visual reference.
        ax.axhspan(0.95, 1.05, color='grey', alpha=0.15, zorder=0,
                   label=r'$\pm 5\%$')

        # Vertical dotted line at mean R200c from TNG300-1 mass-cut halos.
        if R200c_ref is not None:
            ax.axvline(R200c_ref, color='gray', ls=':', lw=2,
                       label=rf'$\langle R_{{200c}} \rangle$ ({R200c_label})')

        ax.set_xlabel(x_label, fontsize=20)
        ax.set_xlim(0.0, xlim_max)
        ax.set_title(title, fontsize=20)
        ax.grid(True)

        # Panel label in the top-left corner.
        ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
                fontsize=18, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none',
                          alpha=0.7))

    # Y-axis label on left panel only (sharey=True shares ticks but not label).
    ax_mass.set_ylabel(
        r'$f_{\rm gas}(R) \;/\; (\Omega_b / \Omega_m)$',
        fontsize=20,
    )

    # Legend on right panel only.
    ax_sham.legend(loc='best', fontsize=13)

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    fig.tight_layout()
    out_path = figPath / f'{figName}_{dim}.{figType}'
    fig.savefig(out_path, dpi=300)  # type: ignore
    plt.close(fig)

    elapsed = (time.time() - t0) / 60
    print(f"\nFigure saved to: {out_path}")
    print(f"Total time: {elapsed:.2f} minutes")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate f_gas profile figure (mass-cut vs SHAM).")
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/fgas_profiles_z05.yaml',
        help='Path to the YAML configuration file.',
    )
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")
    main(**args)
