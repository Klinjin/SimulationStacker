"""make_ratios_mass_bins.py
=========================
Generate a 4×3 figure of particle-type fraction profiles normalised by the
cosmic baryon fraction (OmegaBaryon / OmegaMatter), broken down by halo mass
bin.  Intended for Appendix C.

Layout
------
Rows:    mass bins 0–3 as defined by ``halo_ind()`` in ``halos.py``:
           bin 0 → 5e11–1e12  Msun
           bin 1 → 1e12–1e13  Msun
           bin 2 → 1e13–1e14  Msun
           bin 3 → 1e14–1e19  Msun
Columns: col 0 = 3D spherical profiles   (radius in comoving kpc/h)
         col 1 = 2D projected, cumulative (radius in arcmin)
         col 2 = 2D projected, CAP filter (radius in arcmin)

Simulations plotted on every panel as separate coloured lines:
  TNG300-1, Illustris-1, m100n1024_s50

Usage
-----
    python make_ratios_mass_bins.py -p configs/ratios_mass_bins_z05.yaml
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
from utils import arcmin_to_comoving, comoving_to_arcmin
from stacker import SimulationStacker
from halos import halo_ind, select_binned_halos
from mask_utils import get_cutout_indices_3d, sum_over_cutouts

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore

# ---------------------------------------------------------------------------
# Global matplotlib style
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
# This matches the assignment in the tSZ reference scripts.
_COLOURMAPS = {'IllustrisTNG': 'twilight', 'SIMBA': 'hsv'}

# Ordered reference TNG sim list used to fix colour positions so that omitting
# TNG100-1 from the config does not shift the colours of TNG300-1/Illustris-1.
# Extend this list here if new TNG sims are added.
_TNG_REFERENCE_ORDER = ['TNG100-1', 'TNG300-1', 'Illustris-1']

# Subplot panel labels in reading order (4 rows × 3 cols = 12 panels).
_PANEL_LABELS = [
    '(a)', '(b)', '(c)',
    '(d)', '(e)', '(f)',
    '(g)', '(h)', '(i)',
    '(j)', '(k)', '(l)',
]

# Default OmegaBaryon for Illustris-1 (not stored in header).
_OMEGA_BARYON_ILLUSTRIS_DEFAULT = 0.0456
# Default OmegaBaryon for SIMBA (not stored in header).
_OMEGA_BARYON_SIMBA_DEFAULT = 0.048


# ===========================================================================
# Helper functions
# ===========================================================================

def setup_stacker(sim: dict, sim_type_name: str, redshift: float):
    """Instantiate a SimulationStacker and derive cosmological quantities.

    Parameters
    ----------
    sim : dict
        Single simulation entry from the YAML ``simulations`` block.
        Must contain ``name`` and ``snapshot``; SIMBA entries also need
        ``feedback``.
    sim_type_name : str
        ``'IllustrisTNG'`` or ``'SIMBA'``.
    redshift : float
        Target simulation redshift.

    Returns
    -------
    stacker : SimulationStacker
    OmegaBaryon : float
    cosmo : FlatLambdaCDM
    sim_label : str
        Human-readable label for legends (includes feedback suffix for SIMBA).
    """
    sim_name = sim['name']
    snapshot = sim['snapshot']

    if sim_type_name == 'IllustrisTNG':
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type_name)
        try:
            OmegaBaryon = stacker.header['OmegaBaryon']
        except KeyError:
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


def _profile_ratio_and_err(profiles0: np.ndarray, profiles1: np.ndarray,
                            OmegaBaryon: float, Omega0: float):
    """Compute the baryon-fraction-normalised mean ratio and its propagated
    standard error from per-halo stacked profiles.

    Parameters
    ----------
    profiles0, profiles1 : ndarray of shape (n_radii, n_halos)
        Stacked profiles for the numerator and denominator particle types.
    OmegaBaryon : float
    Omega0 : float
        Total matter density parameter.

    Returns
    -------
    ratio : ndarray of shape (n_radii,)
    err   : ndarray of shape (n_radii,)
    """
    mean0 = np.mean(profiles0, axis=1)
    mean1 = np.mean(profiles1, axis=1)
    ratio = mean0 / mean1 / (OmegaBaryon / Omega0)

    # Propagate standard errors in quadrature.
    se0 = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
    se1 = np.std(profiles1, axis=1) / np.sqrt(profiles1.shape[1])
    err = np.abs(ratio) * np.sqrt((se0 / mean0) ** 2 + (se1 / mean1) ** 2)

    return ratio, err


def compute_3d_profile_ratio(stacker: SimulationStacker,
                              pType: str, pType2: str,
                              params: dict,
                              OmegaBaryon: float,
                              halo_mask: np.ndarray):
    """Compute 3D spherical-shell fraction profiles.

    Builds 3D density fields with ``stacker.makeField``, uses the supplied
    ``halo_mask`` to select halos, and accumulates the enclosed mass in
    spherical apertures using ``get_cutout_indices_3d`` / ``sum_over_cutouts``.

    Parameters
    ----------
    stacker : SimulationStacker
    pType, pType2 : str
        Numerator and denominator particle types.
    params : dict
        Sub-dict of stack parameters (``n_pixels``, ``min_radius_3d``,
        ``max_radius_3d``, ``num_radii_3d``, ``projection``,
        ``save_field``, ``load_field``, ``subtract_mean``).
    OmegaBaryon : float
    halo_mask : ndarray of int
        Integer indices into the halo catalogue selecting halos for stacking.
        Caller is responsible for computing this via ``select_binned_halos``.

    Returns
    -------
    radii : ndarray   — comoving kpc/h
    ratio : ndarray
    err   : ndarray
    R200c : float     — mean R200c for the selected halos (comoving kpc/h)
    """
    nPixels    = params['n_pixels']
    minR       = params['min_radius_3d']
    maxR       = params['max_radius_3d']
    nRadii     = params['num_radii_3d']
    projection = params['projection']
    save       = params['save_field']
    load       = params['load_field']
    sub_mean   = params['subtract_mean']

    # Build 3D fields for both particle types.
    field0 = stacker.makeField(pType, nPixels=nPixels, dim='3D',
                               projection=projection, save=save, load=load)
    field0 = field0 - np.mean(field0) if sub_mean else field0

    field1 = stacker.makeField(pType2, nPixels=nPixels, dim='3D',
                               projection=projection, save=save, load=load)
    field1 = field1 - np.mean(field1) if sub_mean else field1

    # Physical scale: comoving kpc/h per pixel.
    kpc_per_pixel = stacker.header['BoxSize'] / field0.shape[0]

    # Use the externally-provided halo_mask; no internal selection here.
    haloes = stacker.loadHalos()
    GroupPos_masked = (
        np.round(haloes['GroupPos'][halo_mask] / kpc_per_pixel).astype(int) % nPixels
    )
    R200c = np.mean(haloes['GroupRad'][halo_mask])  # comoving kpc/h

    # Stack in spherical apertures at each radius.
    radii     = np.linspace(minR, maxR, nRadii)
    profiles0 = []
    profiles1 = []
    for r in radii:
        rr = np.ones(GroupPos_masked.shape[0]) * r / kpc_per_pixel
        idx = get_cutout_indices_3d(field0, GroupPos_masked, rr)
        profiles0.append(sum_over_cutouts(field0, idx.copy()))
        profiles1.append(sum_over_cutouts(field1, idx.copy()))

    profiles0 = np.array(profiles0)   # shape (n_radii, n_halos)
    profiles1 = np.array(profiles1)

    ratio, err = _profile_ratio_and_err(profiles0, profiles1,
                                        OmegaBaryon, stacker.header['Omega0'])
    return radii, ratio, err, R200c


def compute_2d_profile_ratio(stacker: SimulationStacker,
                              pType: str, pType2: str,
                              filterType: str, filterType2: str,
                              params: dict,
                              OmegaBaryon: float,
                              halo_mask: np.ndarray):
    """Compute 2D projected fraction profiles via ``stackMap``.

    Parameters
    ----------
    stacker : SimulationStacker
    pType, pType2 : str
        Numerator and denominator particle types.
    filterType, filterType2 : str
        Filter applied when stacking (e.g. ``'CAP'``, ``'cumulative'``).
    params : dict
        Sub-dict of stack parameters (``pixel_size``, ``min_radius_2d``,
        ``max_radius_2d``, ``num_radii_2d``, ``rad_distance``, ``projection``,
        ``save_field``, ``load_field``, ``subtract_mean``).
    OmegaBaryon : float
    halo_mask : ndarray of int
        Integer indices into the halo catalogue selecting halos for stacking.
        Passed directly to ``stacker.stackMap``; internal selection is skipped.

    Returns
    -------
    radii : ndarray   — arcmin (scaled by ``rad_distance``)
    ratio : ndarray
    err   : ndarray
    """
    pixelSize   = params['pixel_size']
    minR        = params['min_radius_2d']
    maxR        = params['max_radius_2d']
    nRadii      = params['num_radii_2d']
    radDistance = params['rad_distance']
    projection  = params['projection']
    save        = params['save_field']
    load        = params['load_field']
    sub_mean    = params['subtract_mean']

    radii0, profiles0 = stacker.stackMap(
        pType, filterType=filterType,
        minRadius=minR, maxRadius=maxR, numRadii=nRadii,
        save=save, load=load, radDistance=radDistance,
        pixelSize=pixelSize, projection=projection,
        subtract_mean=sub_mean, halo_mask=halo_mask,
    )
    radii1, profiles1 = stacker.stackMap(
        pType2, filterType=filterType2,
        minRadius=minR, maxRadius=maxR, numRadii=nRadii,
        save=save, load=load, radDistance=radDistance,
        pixelSize=pixelSize, projection=projection,
        subtract_mean=sub_mean, halo_mask=halo_mask,
    )

    ratio, err = _profile_ratio_and_err(profiles0, profiles1,
                                        OmegaBaryon, stacker.header['Omega0'])
    # radii0 and radii1 share the same x-axis (same stacking parameters).
    return radii0 * radDistance, ratio, err


def plot_panel(ax, radii: np.ndarray, ratio: np.ndarray, err: np.ndarray,
               label: str, colour, plot_error_bars: bool):
    """Draw a single profile line with an optional shaded error band.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    radii : ndarray
    ratio, err : ndarray
    label : str
    colour : colour spec accepted by matplotlib
    plot_error_bars : bool
    """
    ax.plot(radii, ratio, label=label, color=colour, lw=2, marker='o')
    if plot_error_bars:
        ax.fill_between(radii, ratio - err, ratio + err,
                        color=colour, alpha=0.2)


def configure_subplot(ax, row_idx: int, col_idx: int,
                      pType: str, pType2: str,
                      R200c_kpch: float | None,
                      R200c_arcmin: float | None,
                      forward_arcmin, inverse_arcmin,
                      max_radius_2d: float, rad_distance: float,
                      suite_name: str, panel_label: str,
                      n_rows: int = 2):
    """Apply axis decorations to a single subplot panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    row_idx, col_idx : int
        Position in the grid.
    pType, pType2 : str
        Particle type names used for the y-axis label.
    R200c_kpch : float or None
        R200c in comoving kpc/h for the vertical reference line (col 0 only).
    R200c_arcmin : float or None
        R200c in arcmin for the vertical reference line (cols 1, 2 only).
    forward_arcmin, inverse_arcmin : callable
        Conversion functions between arcmin and comoving kpc/h, used to add
        a secondary x-axis.
    max_radius_2d : float
        Upper limit for 2D x-axis (arcmin).
    rad_distance : float
        Scaling factor applied to 2D arcmin radii.
    suite_name : str
        Unused in this script; retained for API compatibility with
        ``make_ratios3x2.py``.
    panel_label : str
        Subplot letter, e.g. ``'(a)'``.
    n_rows : int, optional
        Total number of rows in the figure. Used to identify the top row
        (for column titles and secondary-axis labels) and the bottom row
        (for primary x-axis labels). Defaults to 2 for backward compatibility.
    """
    is_top_row    = (row_idx == 0)
    is_bottom_row = (row_idx == n_rows - 1)

    # --- Horizontal reference line at unity ---
    ax.axhline(1.0, color='k', ls='--', lw=2)

    # --- R200c vertical reference line ---
    if col_idx == 0 and R200c_kpch is not None:
        ax.axvline(R200c_kpch, color='gray', ls=':', lw=2, label=r'$R_{200c}$')
    elif col_idx > 0 and R200c_arcmin is not None:
        ax.axvline(R200c_arcmin, color='gray', ls=':', lw=2, label=r'$R_{200c}$')

    # --- Axis limits ---
    ax.set_xlim(0.0, None if col_idx == 0
                else (max_radius_2d * rad_distance + 0.5))
    ax.grid(True)
    if is_bottom_row:
        ax.set_ylim(0.0, 1.2)

    # --- Y axis label (left column only) ---
    if col_idx == 0:
        ax.set_ylabel(
            rf'$\frac{{\mathrm{{{pType}}}}}{{\mathrm{{{pType2}}}}} \;/\; (\Omega_b / \Omega_m)$',
            fontsize=18,
        )

    # --- X axis labels and secondary axis ---
    col_titles = ['3D cumulative', '2D cumulative', '2D CAP']
    # This line ensures x-axis tick labels are visible on the bottom row, even when sharing x-axes across rows.
    # ax.tick_params(axis='x', labelbottom=True)
    if col_idx == 0:
        if is_bottom_row:
            ax.set_xlabel('R [comoving kpc/h]', fontsize=18)
            # Secondary x-axis ticks visible on all rows.
            # secax = ax.secondary_xaxis('top')
        else:
            # Secondary x-axis ticks visible on all rows.
            # secax = ax.secondary_xaxis('top')
            if is_top_row:
                secax = ax.secondary_xaxis('top')
                # Label and column title only on the top row.
                secax.set_xlabel('R [comoving kpc/h]', fontsize=18)
                ax.set_title(col_titles[col_idx], fontsize=18)
    else:
        if is_bottom_row:
            ax.set_xlabel('R [arcmin]', fontsize=18)
            # Secondary x-axis ticks visible on all rows.
            # secax = ax.secondary_xaxis('top',
            #                            functions=(forward_arcmin, inverse_arcmin))
        else:
            # Secondary x-axis ticks visible on all rows.
            # secax = ax.secondary_xaxis('top',
            #                            functions=(forward_arcmin, inverse_arcmin))
            if is_top_row:
                secax = ax.secondary_xaxis('top',
                                           functions=(forward_arcmin, inverse_arcmin))
                # Label and column title only on the top row.
                secax.set_xlabel('R [comoving kpc/h]', fontsize=18)
                ax.set_title(col_titles[col_idx], fontsize=18)

    # --- Subplot panel label in top-left corner ---
    ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
            fontsize=18, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))


# ===========================================================================
# Main
# ===========================================================================

def main(path2config: str, ptype: str, verbose: bool = True):
    """Generate the 4×3 particle-fraction ratio figure, broken down by mass bin.

    Parameters
    ----------
    path2config : str
        Path to the YAML configuration file.
    ptype : str
        Particle type to plot (overrides config if not None).
    verbose : bool
        If True, print progress messages to stdout.
    """
    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_cfg = config.get('stack', {})
    plot_cfg  = config.get('plot', {})

    # --- Shared stacking parameters ---
    redshift    = stack_cfg.get('redshift', 0.5)
    projection  = stack_cfg.get('projection', 'xy')
    save_field  = stack_cfg.get('save_field', True)
    load_field  = stack_cfg.get('load_field', True)
    subtract_mn = stack_cfg.get('subtract_mean', False)
    pType       = ptype if ptype is not None else stack_cfg.get('particle_type', 'ionized_gas')
    pType2      = stack_cfg.get('particle_type_2', 'total')

    # Mass bin indices to loop over (can be edited in config to drop bins).
    mass_bin_indices = stack_cfg.get('mass_bin_indices', [0, 1, 2, 3])
    n_bins = len(mass_bin_indices)

    # --- 3D column parameters ---
    params_3d = {
        'n_pixels':      stack_cfg.get('n_pixels', 1000),
        'min_radius_3d': stack_cfg.get('min_radius_3d', 200.0),
        'max_radius_3d': stack_cfg.get('max_radius_3d', 4000.0),
        'num_radii_3d':  stack_cfg.get('num_radii_3d', 11),
        'projection':    projection,
        'save_field':    save_field,
        'load_field':    load_field,
        'subtract_mean': subtract_mn,
    }

    # --- 2D column parameters ---
    rad_distance  = stack_cfg.get('rad_distance', 1.0)
    max_radius_2d = stack_cfg.get('max_radius_2d', 10.0)
    params_2d = {
        'pixel_size':    stack_cfg.get('pixel_size', 0.5),
        'min_radius_2d': stack_cfg.get('min_radius_2d', 1.0),
        'max_radius_2d': max_radius_2d,
        'num_radii_2d':  stack_cfg.get('num_radii_2d', 11),
        'rad_distance':  rad_distance,
        'projection':    projection,
        'save_field':    save_field,
        'load_field':    load_field,
        'subtract_mean': subtract_mn,
    }

    # Filter types for columns 1 (cumulative) and 2 (CAP).
    ft_col1  = stack_cfg.get('filter_type_col1',   'cumulative')
    ft2_col1 = stack_cfg.get('filter_type_2_col1', 'cumulative')
    ft_col2  = stack_cfg.get('filter_type_col2',   'CAP')
    ft2_col2 = stack_cfg.get('filter_type_2_col2', 'CAP')

    # --- Plotting parameters ---
    now       = datetime.now()
    yr_string = now.strftime("%Y-%m")
    dt_string = now.strftime("%m-%d")
    figPath   = Path(plot_cfg.get('fig_path', '../figures/')) / yr_string / dt_string
    figPath.mkdir(parents=True, exist_ok=True)

    figName         = plot_cfg.get('fig_name', 'ratios_mass_bins')
    figType         = plot_cfg.get('fig_type', 'pdf')
    plot_error_bars = plot_cfg.get('plot_error_bars', True)

    # ------------------------------------------------------------------
    # Build flat list of (sim_dict, sim_type_name) from config.
    # Order: TNG suite first, then SIMBA; within each suite, preserve
    # the order given in the YAML.
    # ------------------------------------------------------------------
    all_sim_entries = []  # list of (sim_dict, sim_type_name)
    for suite in config['simulations']:
        sim_type_name = suite['sim_type']
        for sim in suite['sims']:
            all_sim_entries.append((sim, sim_type_name))

    # One colour per simulation, fixed across all rows.
    # TNG: positions fixed by _TNG_REFERENCE_ORDER so omitting TNG100-1 does
    #      not shift the colours of TNG300-1/Illustris-1.
    # SIMBA (1 sim): position 0.85 (last of 6, matching tSZ reference scripts).
    _tng_cmap       = matplotlib.colormaps[_COLOURMAPS['IllustrisTNG']]  # type: ignore
    _tng_ref_clrs   = _tng_cmap(np.linspace(0.2, 0.85, len(_TNG_REFERENCE_ORDER)))
    _tng_colour_map = {name: _tng_ref_clrs[i]
                       for i, name in enumerate(_TNG_REFERENCE_ORDER)}

    _n_simba    = sum(1 for _, stype in all_sim_entries if stype == 'SIMBA')
    _simba_cmap = matplotlib.colormaps[_COLOURMAPS['SIMBA']]  # type: ignore
    _simba_clrs = (_simba_cmap(np.linspace(0.2, 0.85, 6))[-1:]
                   if _n_simba == 1
                   else _simba_cmap(np.linspace(0.2, 0.85, _n_simba)))

    _simba_idx = 0
    colours = []
    for sim, sim_type_name in all_sim_entries:
        if sim_type_name == 'IllustrisTNG':
            name = sim['name']
            if name not in _tng_colour_map:
                raise ValueError(
                    f"TNG sim {name!r} is not in _TNG_REFERENCE_ORDER. "
                    f"Add it there to maintain consistent colours."
                )
            colours.append(_tng_colour_map[name])
        elif sim_type_name == 'SIMBA':
            colours.append(_simba_clrs[_simba_idx])
            _simba_idx += 1
        else:
            raise ValueError(
                f"No colormap defined for sim_type {sim_type_name!r}. "
                f"Known types: {list(_COLOURMAPS)}"
            )

    # ------------------------------------------------------------------
    # Instantiate all stackers once, outside the mass-bin loop.
    # Reusing the same stacker across bins avoids re-loading maps that
    # are identical for all bins (only halo selection differs).
    # ------------------------------------------------------------------
    if verbose:
        print("Initialising stackers...")

    sim_data = []  # list of (stacker, OmegaBaryon, cosmo, sim_label, colour)
    for (sim, sim_type_name), colour in zip(all_sim_entries, colours):
        stacker, OmegaBaryon, cosmo, sim_label = setup_stacker(
            sim, sim_type_name, redshift)
        sim_data.append((stacker, OmegaBaryon, cosmo, sim_label, colour))
        if verbose:
            print(f"  Loaded: {sim_label}")

    # Arcmin ↔ comoving kpc/h conversion functions, built from TNG300-1
    # (first entry, assumed to be TNG300-1 per the config ordering).
    tng300_stacker, _, cosmo_tng300, _, _ = sim_data[0]

    def _fwd_arcmin(arcmin):
        return arcmin_to_comoving(arcmin, redshift, cosmo_tng300)

    def _inv_arcmin(comov):
        return comoving_to_arcmin(comov, redshift, cosmo_tng300)

    t0 = time.time()

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(n_bins, 3, figsize=(18, 9),
                             sharex='col', sharey=True)

    # ------------------------------------------------------------------
    # Outer loop: mass bins (rows)
    # ------------------------------------------------------------------
    for row_idx, bin_ind in enumerate(mass_bin_indices):
        _, _, bin_label_raw = halo_ind(bin_ind)
        bin_label = bin_label_raw.rstrip(', ')

        if verbose:
            print(f"\n{'='*60}")
            print(f"Row {row_idx}: mass bin {bin_ind}  ({bin_label})")
            print(f"{'='*60}")

        # R200c reference line for this row comes from TNG300-1.
        tng300_halos    = tng300_stacker.loadHalos()
        tng300_bin_mask = select_binned_halos(tng300_halos['GroupMass'], bin_ind)
        R200c_kpch   = np.mean(tng300_halos['GroupRad'][tng300_bin_mask])
        R200c_arcmin = comoving_to_arcmin(R200c_kpch, redshift, cosmo_tng300)

        # ------------------------------------------------------------------
        # Inner loop: simulations
        # ------------------------------------------------------------------
        for j, (stacker, OmegaBaryon, cosmo, sim_label, colour) in enumerate(sim_data):
            # Select halos for this (sim, bin) pair.
            haloes    = stacker.loadHalos()
            halo_mask = select_binned_halos(haloes['GroupMass'], bin_ind)
            n_halos   = len(halo_mask)

            print(f"  {sim_label}  bin {bin_ind}: {n_halos} halos selected")
            if n_halos < 20:
                print(f"  WARNING: only {n_halos} halos in bin {bin_ind} "
                      f"for {sim_label} — consider dropping this bin.")

            # Legend label includes halo count so the reader can see N
            # directly without cross-referencing stdout.
            label = rf"{sim_label}  ($N={n_halos}$)"

            # ==============================================================
            # Column 0 — 3D spherical profiles
            # ==============================================================
            if verbose:
                print(f"    Computing 3D profiles...")
            radii_3d, ratio_3d, err_3d, _ = compute_3d_profile_ratio(
                stacker, pType, pType2, params_3d, OmegaBaryon, halo_mask)

            plot_panel(axes[row_idx, 0], radii_3d, ratio_3d, err_3d,
                       label, colour, plot_error_bars)

            # ==============================================================
            # Column 1 — 2D cumulative profiles
            # ==============================================================
            if verbose:
                print(f"    Computing 2D cumulative profiles "
                      f"(filter={ft_col1}/{ft2_col1})...")
            radii_2d_cum, ratio_2d_cum, err_2d_cum = compute_2d_profile_ratio(
                stacker, pType, pType2, ft_col1, ft2_col1,
                params_2d, OmegaBaryon, halo_mask)

            plot_panel(axes[row_idx, 1], radii_2d_cum, ratio_2d_cum, err_2d_cum,
                       label, colour, plot_error_bars)

            # ==============================================================
            # Column 2 — 2D CAP profiles
            # ==============================================================
            if verbose:
                print(f"    Computing 2D CAP profiles "
                      f"(filter={ft_col2}/{ft2_col2})...")
            radii_2d_cap, ratio_2d_cap, err_2d_cap = compute_2d_profile_ratio(
                stacker, pType, pType2, ft_col2, ft2_col2,
                params_2d, OmegaBaryon, halo_mask)

            plot_panel(axes[row_idx, 2], radii_2d_cap, ratio_2d_cap, err_2d_cap,
                       label, colour, plot_error_bars)

        # ------------------------------------------------------------------
        # Axis decorations for this row
        # ------------------------------------------------------------------
        for col_idx in range(3):
            configure_subplot(
                ax=axes[row_idx, col_idx],
                row_idx=row_idx,
                col_idx=col_idx,
                pType=pType,
                pType2=pType2,
                R200c_kpch=R200c_kpch,
                R200c_arcmin=R200c_arcmin,
                forward_arcmin=_fwd_arcmin,
                inverse_arcmin=_inv_arcmin,
                max_radius_2d=max_radius_2d,
                rad_distance=rad_distance,
                suite_name='',
                panel_label=_PANEL_LABELS[row_idx * 3 + col_idx],
                n_rows=n_bins,
            )

        # Row label on the left of col 0.
        axes[row_idx, 0].annotate(
            bin_label,
            xy=(-0.30, 0.5), xycoords='axes fraction',
            ha='right', va='center', rotation=90,
            fontsize=13, fontweight='bold',
        )

        # Per-row legend to the right of col 2.  Each row has its own
        # N= labels so the reader can immediately see halo counts.
        handles, labels_leg = axes[row_idx, 2].get_legend_handles_labels()
        axes[row_idx, 2].legend(
            handles, labels_leg,
            bbox_to_anchor=(1.01, 1.0), loc='upper left',
            frameon=True, fontsize=11,
        )

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    fig.tight_layout(rect=[0, 0, 1.01, 1])  # type: ignore
    out_path = figPath / f'{figName}_{pType}.{figType}'
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
        description="Generate 4×3 particle-fraction ratio figure by mass bin.")
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/ratios_mass_bins_z05.yaml',
        help='Path to the YAML configuration file.',
    )
    parser.add_argument(
        '--ptype',
        type=str,
        default=None,
        help='Override particle type from config (e.g. "ionized_gas").',
    )
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")
    main(**args)
