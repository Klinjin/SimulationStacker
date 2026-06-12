"""
make_baryonFraction.py
======================
Produces two stacked-area figures showing baryon-type fractions normalised by
the total baryonic mass (so all components always sum to 1):

    Figure 1 — 3D stacking: 3 rows × 1 col, figsize (9, 9).
               X-axis in comoving kpc/h.
    Figure 2 — 2D stacking: 3 rows × 1 col, figsize (9, 9).
               X-axis in arcmin; secondary top axis in comoving kpc/h.

Rows correspond to the simulations listed in the config (top → bottom).

Unlike make_stackArea.py, the denominator is the sum of all baryon-type
fields rather than the total matter field, so the stacked areas always
integrate to 1 and no Omega_b / Omega_m normalisation is applied.

Usage
-----
    python make_baryonFraction.py -p ./configs/stackArea_z05.yaml

Config file format
------------------
Same format as make_stackArea.py / configs/stackArea_z05.yaml.
``particle_type_2`` is ignored; the denominator is derived internally.

Dependencies
------------
* SimulationStacker src/ package (stacker, halos, mask_utils, utils)
* illustris_python (on sys.path one level up)
* astropy, matplotlib, numpy, yaml
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import cast

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import yaml
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# ---------------------------------------------------------------------------
# Internal package imports
# ---------------------------------------------------------------------------
sys.path.append('../src/')
from utils import arcmin_to_comoving, comoving_to_arcmin
from stacker import SimulationStacker
from halos import select_massive_halos
from mask_utils import get_cutout_indices_3d, sum_over_cutouts

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# ---------------------------------------------------------------------------
# Helper: create a SimulationStacker and associated cosmology
# ---------------------------------------------------------------------------

def make_stacker(sim: dict, redshift: float):
    """Initialise a SimulationStacker for the given simulation entry.

    Parameters
    ----------
    sim : dict
        A single entry from the ``simulations`` list in the config file.
        Required keys: ``sim_type``, ``name``, ``snapshot``.
        SIMBA entries also require ``feedback``.
    redshift : float
        Target redshift.

    Returns
    -------
    stacker : SimulationStacker
    cosmo : FlatLambdaCDM
    sim_label : str
        Human-readable label for plot titles.
    """
    sim_type = sim['sim_type']
    sim_name = sim['name']
    snapshot = sim['snapshot']

    if sim_type == 'IllustrisTNG':
        stacker = SimulationStacker(sim_name, snapshot, z=redshift, simType=sim_type)
        sim_label = sim_name
        try:
            OmegaBaryon = stacker.header['OmegaBaryon']
        except KeyError:
            OmegaBaryon = 0.0456

    elif sim_type == 'SIMBA':
        feedback = sim['feedback']
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type, feedback=feedback)
        sim_label = f"{sim_name}_{feedback}"
        OmegaBaryon = 0.048

    else:
        raise ValueError(f"Unknown sim_type '{sim_type}'. Supported: 'IllustrisTNG', 'SIMBA'.")

    cosmo = FlatLambdaCDM(
        H0=100 * stacker.header['HubbleParam'],
        Om0=stacker.header['Omega0'],
        Tcmb0=2.7255 * u.K,
        Ob0=OmegaBaryon,
    )
    return stacker, cosmo, sim_label


# ---------------------------------------------------------------------------
# 3-D stacking
# ---------------------------------------------------------------------------

def run_3d_stacking(stacker, baryon_types, nPixels, minRadius, maxRadius, nRadii,
                    projection, saveField, loadField, ax, colours,
                    radDistance, sphere=True, dr=0.0, verbose=True):
    """Build 3-D density fields and plot the baryon-fraction stacked-area profile.

    The denominator is the sum of all baryon fields so that the stacked areas
    always sum to 1 at every radius.

    Parameters
    ----------
    stacker : SimulationStacker
    baryon_types : list[str]
        Component particle types, e.g. ['ionized_gas', 'neutral_gas', 'Stars', 'BH'].
    nPixels : int
        Grid resolution for 3-D fields.
    minRadius : float
        Minimum stacking radius [comoving kpc/h].
    maxRadius : float
        Maximum stacking radius [comoving kpc/h].
    nRadii : int
        Number of radial bins.
    projection : str
        Projection axis for field ('xy', 'xz', or 'yz').
    saveField : bool
        Cache fields to disk.
    loadField : bool
        Try to load cached fields from disk.
    ax : matplotlib.axes.Axes
    colours : array-like
        One colour per baryon type.
    radDistance : float
        Multiplicative scaling applied to radii for the x-axis.
    sphere : bool
        If True (default), each bin accumulates all mass within a sphere of
        radius R (cumulative aperture).  If False, each bin considers only the
        shell between R and R+dr, computed as sphere(R+dr) − sphere(R).
    dr : float
        Shell width [comoving kpc/h].  Used only when ``sphere=False``.  Should
        be set by the caller (``main`` computes a sensible default from the
        radii spacing).
    verbose : bool
    """
    baryon_fields = {}
    for bt in baryon_types:
        if verbose:
            print(f"  Building 3D field: {bt}")
        baryon_fields[bt] = stacker.makeField(bt, nPixels=nPixels, dim='3D',
                                              projection=projection,
                                              save=saveField, load=loadField)

    # Use the first field to derive voxel size (all fields share the same grid)
    first_field = next(iter(baryon_fields.values()))
    kpcPerPixel = stacker.header['BoxSize'] / first_field.shape[0]
    if verbose:
        print(f"  kpcPerPixel = {kpcPerPixel:.3f}")

    haloes = stacker.loadHalos()
    haloMass = haloes['GroupMass']
    halo_mask = select_massive_halos(haloMass, 10**13.22, 5e14)

    haloes['GroupMass'] = haloes['GroupMass'][halo_mask]
    haloes['GroupRad'] = haloes['GroupRad'][halo_mask]
    GroupPos_px = np.round(haloes['GroupPos'][halo_mask] / kpcPerPixel).astype(int) % nPixels

    if verbose:
        print(f"  Number of selected haloes: {halo_mask.sum()}")

    radii = np.linspace(minRadius, maxRadius, nRadii)

    profiles_baryon = {bt: [] for bt in baryon_types}

    t0 = time.time()
    for r in radii:
        if sphere:
            # Cumulative sphere of radius R
            rr = np.ones(len(haloes['GroupMass'])) * r / kpcPerPixel
            mask_indices = get_cutout_indices_3d(first_field, GroupPos_px, rr)
            for bt in baryon_types:
                profiles_baryon[bt].append(
                    sum_over_cutouts(baryon_fields[bt], mask_indices.copy())
                )
        else:
            # Shell from R to R+dr: sphere(R+dr) - sphere(R)
            rr_inner = np.ones(len(haloes['GroupMass'])) * r / kpcPerPixel
            rr_outer = np.ones(len(haloes['GroupMass'])) * (r + dr) / kpcPerPixel
            mask_inner = get_cutout_indices_3d(first_field, GroupPos_px, rr_inner)
            mask_outer = get_cutout_indices_3d(first_field, GroupPos_px, rr_outer)
            for bt in baryon_types:
                shell_sum = (
                    sum_over_cutouts(baryon_fields[bt], mask_outer.copy())
                    - sum_over_cutouts(baryon_fields[bt], mask_inner.copy())
                )
                profiles_baryon[bt].append(shell_sum)
        if verbose:
            print(f"    r={r:.0f} kpc/h  elapsed={time.time()-t0:.1f}s")

    for bt in baryon_types:
        profiles_baryon[bt] = np.array(profiles_baryon[bt])  # type: ignore

    # Mean over haloes at each radius for each component
    means = {bt: np.mean(profiles_baryon[bt], axis=1) for bt in baryon_types}

    # Denominator: sum of all baryon components (so fractions sum to 1)
    total_baryon = sum(means[bt] for bt in baryon_types)

    fractions = [means[bt] / total_baryon for bt in baryon_types]

    ax.stackplot(radii * radDistance, fractions, labels=baryon_types,
                 alpha=0.8, colors=colours)


# ---------------------------------------------------------------------------
# 2-D stacking
# ---------------------------------------------------------------------------

def run_2d_stacking(stacker, baryon_types, filterType, minRadius, maxRadius, nRadii,
                    projection, saveField, loadField, radDistance,
                    ax, colours, inverse_arcmin, forward_arcmin, verbose=True):
    """Stack 2-D projected maps and plot the baryon-fraction stacked-area profile.

    The denominator is the sum of all stacked baryon maps so fractions sum to 1.

    Parameters
    ----------
    stacker : SimulationStacker
    baryon_types : list[str]
    filterType : str
        Filter applied to all maps (e.g. 'CAP').
    minRadius : float
        Minimum radius [comoving kpc/h], converted internally to arcmin.
    maxRadius : float
        Maximum radius [comoving kpc/h], converted internally to arcmin.
    nRadii : int
    projection : str
    saveField : bool
    loadField : bool
    radDistance : float
    ax : matplotlib.axes.Axes
    colours : array-like
    inverse_arcmin : callable
        comoving kpc/h → arcmin.
    forward_arcmin : callable
        arcmin → comoving kpc/h (for secondary axis).
    verbose : bool

    Returns
    -------
    maxRadius_arcmin : float
        Maximum stacking radius in arcmin (used to set xlim on the caller).
    """
    minRadius_arcmin = inverse_arcmin(minRadius)
    maxRadius_arcmin = inverse_arcmin(maxRadius)

    if verbose:
        print(f"  2D stacking: {minRadius_arcmin:.2f} – {maxRadius_arcmin:.2f} arcmin")

    profiles_baryon = {}
    radii_out = None
    for bt in baryon_types:
        if verbose:
            print(f"  Stacking 2D map: {bt}")
        t1 = time.time()
        radii0, profiles_baryon[bt] = stacker.stackMap(
            bt, filterType=filterType,
            minRadius=minRadius_arcmin, maxRadius=maxRadius_arcmin, numRadii=nRadii,
            save=saveField, load=loadField, radDistance=radDistance,
            projection=projection,
        )
        if radii_out is None:
            radii_out = radii0
        if verbose:
            print(f"    done in {time.time()-t1:.1f}s")

    means = {bt: np.mean(profiles_baryon[bt], axis=1) for bt in baryon_types}

    # Denominator: sum of all baryon-component means
    total_baryon = sum(means[bt] for bt in baryon_types)

    fractions = [means[bt] / total_baryon for bt in baryon_types]

    ax.stackplot(radii_out * radDistance, fractions, labels=baryon_types,
                 alpha=0.8, colors=colours)

    return maxRadius_arcmin


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(path2config: str, verbose: bool = True):
    """Load config, run 3-D and 2-D stacking for each simulation, and save two
    separate 3×1 figures (one for 3D stacking, one for 2D stacking).

    Parameters
    ----------
    path2config : str
        Path to the YAML configuration file.
    verbose : bool
        If True, print progress messages during stacking.

    Config keys added under ``stack:``
    -----------------------------------
    sphere : bool, default True
        If True, each 3D radial bin accumulates all mass within a sphere of
        radius R.  If False, each bin considers only the shell from R to R+dr
        (computed as sphere(R+dr) − sphere(R)).
    dr : float, optional
        Shell width in comoving kpc/h.  Only used when ``sphere: false``.
        Defaults to the spacing of the ``np.linspace`` radii grid,
        i.e. ``(max_radius - min_radius) / (num_radii - 1)``.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_config = config.get('stack', {})
    plot_config  = config.get('plot', {})

    # -----------------------------------------------------------------------
    # Read stacking parameters
    # -----------------------------------------------------------------------
    redshift     = stack_config.get('redshift', 0.5)
    loadField    = stack_config.get('load_field', True)
    saveField    = stack_config.get('save_field', True)
    radDistance  = stack_config.get('rad_distance', 1.0)
    baryon_types = stack_config.get('baryon_types', ['ionized_gas', 'neutral_gas', 'Stars', 'BH'])
    projection   = stack_config.get('projection', 'yz')
    filterType   = stack_config.get('filter_type', 'CAP')
    minRadius    = stack_config.get('min_radius', 200.0)   # comoving kpc/h
    maxRadius    = stack_config.get('max_radius', 6000.0)
    nRadii       = stack_config.get('num_radii', 15)
    nPixels      = stack_config.get('n_pixels', 1000)
    sphere       = stack_config.get('sphere', True)
    dr           = stack_config.get('dr', None)            # comoving kpc/h; shell width

    # Resolve default shell width: spacing of the np.linspace radii grid
    if not sphere and dr is None:
        if nRadii > 1:
            dr = (maxRadius - minRadius) / (nRadii - 1)
        else:
            dr = maxRadius * 0.1
        if verbose:
            print(f"Shell mode: dr not specified, using radii spacing dr={dr:.1f} kpc/h")

    # -----------------------------------------------------------------------
    # Read plotting parameters
    # -----------------------------------------------------------------------
    now       = datetime.now()
    yr_string = now.strftime("%Y-%m")
    dt_string = now.strftime("%m-%d")

    figPath = Path(plot_config.get('fig_path', '../figures/')) / yr_string / dt_string
    figPath.mkdir(parents=True, exist_ok=True)
    figName = plot_config.get('fig_name', 'combined')
    figType = plot_config.get('fig_type', 'pdf')

    colourmap = matplotlib.colormaps['plasma']  # type: ignore
    colours   = colourmap(np.linspace(0.0, 0.8, len(baryon_types)))

    sims   = config['simulations']
    n_sims = len(sims)

    # -----------------------------------------------------------------------
    # Create two figures: 3D (n_sims rows × 1 col) and 2D (n_sims rows × 1 col)
    # sharex=True on the 3D figure so a single x-axis at the top suffices
    # -----------------------------------------------------------------------
    fig_3d, _axes_3d = plt.subplots(n_sims, 1, figsize=(9, 9), sharey=True, sharex=True)
    fig_2d, _axes_2d = plt.subplots(n_sims, 1, figsize=(9, 9), sharey=True)
    axes_3d = cast(list[Axes], [_axes_3d] if n_sims == 1 else list(_axes_3d))
    axes_2d = cast(list[Axes], [_axes_2d] if n_sims == 1 else list(_axes_2d))

    t_total = time.time()

    for row, sim in enumerate(sims):
        sim_type = sim['sim_type']
        sim_name = sim['name']
        if verbose:
            print(f"\n=== Processing simulation [{row+1}/{n_sims}]: {sim_name} ({sim_type}) ===")

        stacker, cosmo, sim_label = make_stacker(sim, redshift)

        def forward_arcmin(arcmin, _redshift=redshift, _cosmo=cosmo):
            return arcmin_to_comoving(arcmin, _redshift, _cosmo)

        def inverse_arcmin(comoving, _redshift=redshift, _cosmo=cosmo):
            return comoving_to_arcmin(comoving, _redshift, _cosmo)

        # -------------------------------------------------------------------
        # 3-D stacking
        # -------------------------------------------------------------------
        ax_3d = axes_3d[row]
        if verbose:
            print("  [3D] starting ...")
        run_3d_stacking(
            stacker=stacker,
            baryon_types=baryon_types,
            nPixels=nPixels,
            minRadius=minRadius,
            maxRadius=maxRadius,
            nRadii=nRadii,
            projection=projection,
            saveField=saveField,
            loadField=loadField,
            ax=ax_3d,
            colours=colours,
            radDistance=radDistance,
            sphere=sphere,
            dr=dr if dr is not None else 0.0,
            verbose=verbose,
        )
        ax_3d.set_ylabel('Baryon fraction')
        ax_3d.set_xlim(0.0, maxRadius * radDistance)
        ax_3d.set_ylim(0.0, 1.0)
        ax_3d.grid(True)
        # Show bottom ticks on all panels but suppress labels; labels live on
        # the top of the top panel and bottom of the bottom panel (set after the loop).
        ax_3d.tick_params(axis='x', bottom=True, labelbottom=False, top=True, labeltop=False)
        # Sim label in a tight box at the lower-left corner
        ax_3d.text(0.03, 0.05, sim_label, transform=ax_3d.transAxes, fontsize=18,
                   va='bottom', ha='left',
                   bbox=dict(boxstyle='square,pad=0.1', facecolor='white',
                             edgecolor='gray', alpha=0.85))
        if row == n_sims - 1:
            ax_3d.legend(loc='lower right')

        # -------------------------------------------------------------------
        # 2-D stacking
        # -------------------------------------------------------------------
        ax_2d = axes_2d[row]
        if verbose:
            print("  [2D] starting ...")
        maxRadius_arcmin = run_2d_stacking(
            stacker=stacker,
            baryon_types=baryon_types,
            filterType=filterType,
            minRadius=minRadius,
            maxRadius=maxRadius,
            nRadii=nRadii,
            projection=projection,
            saveField=saveField,
            loadField=loadField,
            radDistance=radDistance,
            ax=ax_2d,
            colours=colours,
            inverse_arcmin=inverse_arcmin,
            forward_arcmin=forward_arcmin,
            verbose=verbose,
        )
        ax_2d.set_ylabel('Baryon fraction')
        ax_2d.set_xlim(0.0, maxRadius_arcmin * radDistance)
        ax_2d.set_ylim(0.0, 1.0)
        ax_2d.grid(True)
        # Bottom x label and ticks only on the bottom panel
        if row == n_sims - 1:
            ax_2d.set_xlabel('R [arcmin]')
            ax_2d.legend(loc='lower right')
            ax_2d.tick_params(axis='x', bottom=True, labelbottom=True, top=True, labeltop=False)
        else:
            ax_2d.tick_params(axis='x', bottom=True, labelbottom=False, top=True, labeltop=False)
        # Sim label in a tight box at the lower-left corner
        ax_2d.text(0.03, 0.05, sim_label, transform=ax_2d.transAxes, fontsize=18,
                   va='bottom', ha='left',
                   bbox=dict(boxstyle='square,pad=0.1', facecolor='white',
                             edgecolor='gray', alpha=0.85))
        
        # Secondary x-axis (comoving kpc/h) on top panel
        if row == 0:
            secax = ax_2d.secondary_xaxis('top', functions=(forward_arcmin, inverse_arcmin))
            secax.set_xlabel('R [comoving kpc/h]')

    # -----------------------------------------------------------------------
    # Finalise and save
    # -----------------------------------------------------------------------
    # 3D: move x-axis ticks and label to the top of the top panel
    axes_3d[0].xaxis.tick_top()
    axes_3d[0].xaxis.set_label_position('top')
    axes_3d[0].set_xlabel('R [comoving kpc/h]')
    axes_3d[0].tick_params(axis='x',  bottom=True, labelbottom=False, top=True, labeltop=True)
    axes_3d[-1].tick_params(axis='x', bottom=True, labelbottom=True, top=True, labeltop=False)
    axes_3d[-1].set_xlabel('R [comoving kpc/h]')

    fig_3d.suptitle(f'Baryon Fractions (3D) at $z={redshift}$', fontsize=18)
    fig_3d.tight_layout()
    out_3d = figPath / f'{figName}_3D_baryonFraction.{figType}'
    print(f'Saving 3D figure to {out_3d}')
    fig_3d.savefig(out_3d, dpi=300)  # type: ignore
    plt.close(fig_3d)

    fig_2d.suptitle(f'Baryon Fractions (2D) at $z={redshift}$', fontsize=18)
    fig_2d.tight_layout()
    out_2d = figPath / f'{figName}_2D_baryonFraction.{figType}'
    print(f'Saving 2D figure to {out_2d}')
    fig_2d.savefig(out_2d, dpi=300)  # type: ignore
    plt.close(fig_2d)

    print(f'Done!  Total elapsed time: {time.time()-t_total:.1f}s')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Produce 3D and 2D baryon-fraction stacked-area figures '
                    '(normalised by total baryons, not total matter) for multiple simulations.'
    )
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/stackArea_z05.yaml',
        help='Path to the YAML configuration file.',
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose progress output.',
    )
    args = parser.parse_args()
    print(f"Arguments: {vars(args)}")
    main(path2config=args.path2config, verbose=not args.quiet)
