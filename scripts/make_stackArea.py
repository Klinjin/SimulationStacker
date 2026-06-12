"""
make_stackArea.py
=================
Produces a combined baryon-fraction stacked-area figure with 6 subplots arranged
in two rows of 3 columns:

    Row 0 (top)    — 3D stacking: radial profiles summed directly over 3-D density
                     fields.  X-axis is in comoving kpc/h.
    Row 1 (bottom) — 2D stacking: radial profiles measured from projected 2-D maps.
                     X-axis is in arcmin; a secondary top axis shows comoving kpc/h.

Columns correspond to (left → right): TNG300-1, Illustris-1, SIMBA m100n1024.

Each panel shows a stacked-area plot of the mean baryon-type fractions (normalised
by the cosmic baryon fraction Omega_b / Omega_m) as a function of projected radius.
A horizontal dashed line at y=1 indicates a perfectly baryon-traced total field.

Usage
-----
    python make_stackArea.py -p ./configs/stackArea_z05.yaml

Config file format
------------------
See configs/stackArea_z05.yaml for a fully annotated example.  The top-level keys
are ``stack``, ``plot``, and ``simulations``.  The ``simulations`` list must contain
exactly three entries in the order: TNG300-1, Illustris-1, SIMBA.

Dependencies
------------
* SimulationStacker src/ package (stacker, halos, mask_utils, utils, SZstacker)
* illustris_python (on sys.path one level up)
* astropy, matplotlib, numpy, yaml
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
# Global matplotlib style — Computer Modern / LaTeX-compatible fonts
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
    """Initialise a SimulationStacker for the given simulation entry and return
    the stacker together with its FlatLambdaCDM cosmology and OmegaBaryon.

    Parameters
    ----------
    sim : dict
        A single entry from the ``simulations`` list in the config file.
        Required keys: ``sim_type``, ``name``, ``snapshot``.
        SIMBA entries also require ``feedback``.
    redshift : float
        Target redshift (used for cosmology and stacker initialisation).

    Returns
    -------
    stacker : SimulationStacker
    cosmo   : FlatLambdaCDM
    OmegaBaryon : float
    sim_label   : str  — human-readable label for plot titles
    """
    sim_type = sim['sim_type']
    sim_name = sim['name']
    snapshot = sim['snapshot']

    if sim_type == 'IllustrisTNG':
        stacker = SimulationStacker(sim_name, snapshot, z=redshift, simType=sim_type)
        try:
            OmegaBaryon = stacker.header['OmegaBaryon']
        except KeyError:
            # Illustris-1 does not store OmegaBaryon; use the standard value
            OmegaBaryon = 0.0456
        sim_label = sim_name

    elif sim_type == 'SIMBA':
        feedback = sim['feedback']
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type, feedback=feedback)
        OmegaBaryon = 0.048  # Standard value for SIMBA runs
        sim_label = f"{sim_name}_{feedback}"

    else:
        raise ValueError(f"Unknown sim_type '{sim_type}'.  Supported: 'IllustrisTNG', 'SIMBA'.")

    cosmo = FlatLambdaCDM(
        H0=100 * stacker.header['HubbleParam'],
        Om0=stacker.header['Omega0'],
        Tcmb0=2.7255 * u.K,
        Ob0=OmegaBaryon,
    )
    return stacker, cosmo, OmegaBaryon, sim_label


# ---------------------------------------------------------------------------
# 3-D stacking: radial profiles from 3-D density fields
# ---------------------------------------------------------------------------

def run_3d_stacking(stacker, OmegaBaryon, baryon_types, pType2,
                    nPixels, minRadius, maxRadius, nRadii,
                    projection, saveField, loadField,
                    ax, colours, radDistance, verbose=True):
    """Build 3-D density fields, cut out spheres around massive haloes, and plot
    the resulting baryon-fraction stacked-area profile.

    The radii are in comoving kpc/h (the native 3-D field unit).  A sphere of
    radius r is grown from each halo centre and the enclosed mass summed via
    ``get_cutout_indices_3d`` / ``sum_over_cutouts``.

    Parameters
    ----------
    stacker    : SimulationStacker
    OmegaBaryon : float
    baryon_types : list[str]  — e.g. ['ionized_gas', 'neutral_gas', 'Stars', 'BH']
    pType2     : str          — particle type used as the total-mass denominator
    nPixels    : int          — grid resolution for 3-D fields
    minRadius  : float        — minimum stacking radius [comoving kpc/h]
    maxRadius  : float        — maximum stacking radius [comoving kpc/h]
    nRadii     : int          — number of radial bins
    projection : str          — projection axis for field ('xy', 'xz', 'yz') — passed
                                through to makeField (the 3-D field is not projected)
    saveField  : bool         — cache fields to disk
    loadField  : bool         — try to load cached fields from disk
    ax         : matplotlib Axes
    colours    : array-like   — one colour per baryon type
    radDistance : float       — multiplicative scaling applied to radii for x-axis
    verbose    : bool
    """
    # Build the 3-D field for each baryon type (numerators of the fractions)
    baryon_fields = {}
    for bt in baryon_types:
        if verbose:
            print(f"  Building 3D field: {bt}")
        baryon_fields[bt] = stacker.makeField(bt, nPixels=nPixels, dim='3D',
                                              projection=projection,
                                              save=saveField, load=loadField)

    # Build the total-mass 3-D field (common denominator for all baryon fractions)
    if verbose:
        print(f"  Building 3D field: {pType2} (total)")
    field_total = stacker.makeField(pType2, nPixels=nPixels, dim='3D',
                                    projection=projection,
                                    save=saveField, load=loadField)

    # Physical size of a single voxel [comoving kpc/h per pixel]
    kpcPerPixel = stacker.header['BoxSize'] / field_total.shape[0]
    if verbose:
        print(f"  kpcPerPixel = {kpcPerPixel:.3f}")

    # Load haloes and select massive ones (log10 M > 13.22, M < 5e14 M_sun/h)
    haloes = stacker.loadHalos()
    haloMass = haloes['GroupMass']
    halo_mask = select_massive_halos(haloMass, 10**13.22, 5e14)

    haloes['GroupMass'] = haloes['GroupMass'][halo_mask]
    haloes['GroupRad'] = haloes['GroupRad'][halo_mask]  # R200c [comoving kpc/h]
    GroupPos_px = np.round(haloes['GroupPos'][halo_mask] / kpcPerPixel).astype(int) % nPixels

    if verbose:
        print(f"  Number of selected haloes: {halo_mask.shape}")

    # Linearly-spaced radii at which to evaluate the stacked profiles
    radii = np.linspace(minRadius, maxRadius, nRadii)  # [comoving kpc/h]

    # Accumulate profile arrays: shape will be (nRadii, nHalos) after stacking
    profiles_baryon = {bt: [] for bt in baryon_types}
    profiles_total = []

    t0 = time.time()
    for r in radii:
        # Cutout radius converted to pixels (same for all haloes at a given r)
        rr = np.ones(len(haloes['GroupMass'])) * r / kpcPerPixel
        mask_indices = get_cutout_indices_3d(field_total, GroupPos_px, rr)
        for bt in baryon_types:
            profiles_baryon[bt].append(sum_over_cutouts(baryon_fields[bt], mask_indices.copy()))
        profiles_total.append(sum_over_cutouts(field_total, mask_indices.copy()))
        if verbose:
            print(f"    r={r:.0f} kpc/h  elapsed={time.time()-t0:.1f}s")

    # Convert profile lists to arrays: (nRadii, nHalos)
    for bt in baryon_types:
        profiles_baryon[bt] = np.array(profiles_baryon[bt]) # type: ignore
    profiles_total = np.array(profiles_total)

    # Mean over haloes at each radius
    mean_total = np.mean(profiles_total, axis=1)  # (nRadii,)

    # Baryon fractions normalised by the cosmic baryon fraction so that a
    # uniformly baryon-traced field returns 1 everywhere
    fractions = []
    bt_labels = []
    for bt in baryon_types:
        mean_bt = np.mean(profiles_baryon[bt], axis=1)
        fractions.append(mean_bt / mean_total / (OmegaBaryon / stacker.header['Omega0']))
        bt_labels.append(bt)

    # Draw the stacked-area plot on the provided axis
    ax.stackplot(radii * radDistance, fractions, labels=bt_labels, alpha=0.8, colors=colours)


# ---------------------------------------------------------------------------
# 2-D stacking: radial profiles from projected 2-D maps
# ---------------------------------------------------------------------------

def run_2d_stacking(stacker, cosmo, OmegaBaryon, baryon_types, pType2,
                    filterType, filterType2, minRadius, maxRadius, nRadii,
                    projection, saveField, loadField, radDistance,
                    ax, colours, forward_arcmin, inverse_arcmin, verbose=True):
    """Stack 2-D projected maps and plot the resulting baryon-fraction profile.

    The stacking radii are expressed in arcmin (converted from the comoving kpc/h
    values in the config using the simulation cosmology).  The x-axis of the
    returned plot is in arcmin; the caller is responsible for adding a secondary
    comoving-kpc/h axis via ``forward_arcmin`` / ``inverse_arcmin``.

    Parameters
    ----------
    stacker     : SimulationStacker
    cosmo       : FlatLambdaCDM  — simulation cosmology (used implicitly via
                                   forward_arcmin / inverse_arcmin passed in)
    OmegaBaryon : float
    baryon_types : list[str]
    pType2      : str           — particle type for the total-mass denominator map
    filterType  : str           — filter applied to baryon maps (e.g. 'CAP')
    filterType2 : str           — filter applied to total-mass map
    minRadius   : float         — minimum radius [comoving kpc/h], converted to arcmin
    maxRadius   : float         — maximum radius [comoving kpc/h], converted to arcmin
    nRadii      : int
    projection  : str
    saveField   : bool
    loadField   : bool
    radDistance : float         — multiplicative scaling for the x-axis display
    ax          : matplotlib Axes
    colours     : array-like
    forward_arcmin  : callable  — arcmin → comoving kpc/h (for secondary axis)
    inverse_arcmin  : callable  — comoving kpc/h → arcmin (for secondary axis)
    verbose     : bool
    """
    # Convert stacking radii from comoving kpc/h → arcmin using the sim cosmology
    minRadius_arcmin = inverse_arcmin(minRadius)
    maxRadius_arcmin = inverse_arcmin(maxRadius)

    if verbose:
        print(f"  2D stacking: {minRadius_arcmin:.2f} – {maxRadius_arcmin:.2f} arcmin")

    # Stack the total-mass map (denominator)
    radii1, profiles_total = stacker.stackMap(
        pType2, filterType=filterType2,
        minRadius=minRadius_arcmin, maxRadius=maxRadius_arcmin, numRadii=nRadii,
        save=saveField, load=loadField, radDistance=radDistance,
        projection=projection,
    )

    # Stack each baryon-type map (numerators)
    profiles_baryon = {}
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
        if verbose:
            print(f"    done in {time.time()-t1:.1f}s")

    # Mean over haloes
    mean_total = np.mean(profiles_total, axis=1)  # (nRadii,)

    fractions = []
    bt_labels = []
    for bt in baryon_types:
        mean_bt = np.mean(profiles_baryon[bt], axis=1)
        fractions.append(mean_bt / mean_total / (OmegaBaryon / stacker.header['Omega0']))
        bt_labels.append(bt)

    # x-axis: radii in arcmin scaled by radDistance
    ax.stackplot(radii0 * radDistance, fractions, labels=bt_labels, alpha=0.8, colors=colours)

    # Store the final arcmin radius so the caller can set xlim
    return maxRadius_arcmin


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(path2config: str, verbose: bool = True):
    """Load config, run 3-D and 2-D stacking for each simulation, and save the
    combined 2×3 figure.

    Parameters
    ----------
    path2config : str
        Path to the YAML configuration file.
    verbose : bool
        If True, print progress messages during stacking.
    """

    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_config = config.get('stack', {})
    plot_config  = config.get('plot', {})

    # -----------------------------------------------------------------------
    # Read stacking parameters (shared between 3-D and 2-D)
    # -----------------------------------------------------------------------
    redshift    = stack_config.get('redshift', 0.5)
    loadField   = stack_config.get('load_field', True)
    saveField   = stack_config.get('save_field', True)
    radDistance = stack_config.get('rad_distance', 1.0)
    baryon_types = stack_config.get('baryon_types', ['ionized_gas', 'neutral_gas', 'Stars', 'BH'])
    projection  = stack_config.get('projection', 'yz')
    pixelSize   = stack_config.get('pixel_size', 0.5)   # arcmin, for 2-D maps
    beamSize    = stack_config.get('beam_size', 1.6)    # arcmin, Gaussian smoothing

    filterType  = stack_config.get('filter_type', 'CAP')    # applied to baryon maps
    filterType2 = stack_config.get('filter_type_2', 'CAP')  # applied to total map
    pType2      = stack_config.get('particle_type_2', 'total')

    minRadius   = stack_config.get('min_radius', 200.0)  # comoving kpc/h
    maxRadius   = stack_config.get('max_radius', 6000.0)
    nRadii      = stack_config.get('num_radii', 15)
    nPixels     = stack_config.get('n_pixels', 1000)     # pixels per side for 3-D fields

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

    # Colour palette: one colour per baryon type, drawn from the 'plasma' colourmap
    colourmap = matplotlib.colormaps['plasma']  # type: ignore
    colours   = colourmap(np.linspace(0.0, 0.8, len(baryon_types)))

    sims = config['simulations']
    n_sims = len(sims)  # expected to be 3

    # -----------------------------------------------------------------------
    # Create the figure: 2 rows × n_sims columns
    #   Row 0 — 3-D stacking profiles (x: comoving kpc/h)
    #   Row 1 — 2-D map stacking profiles (x: arcmin)
    # sharey='row' keeps the y-scale the same within each row for easy comparison
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, n_sims, figsize=(18, 9), sharey='row')

    t_total = time.time()

    for col, sim in enumerate(sims):
        sim_type = sim['sim_type']
        sim_name = sim['name']
        if verbose:
            print(f"\n=== Processing simulation [{col+1}/{n_sims}]: {sim_name} ({sim_type}) ===")

        # Build the shared stacker and cosmology for this simulation
        stacker, cosmo, OmegaBaryon, sim_label = make_stacker(sim, redshift)

        # Arcmin ↔ comoving kpc/h conversion functions (sim-specific cosmology)
        def forward_arcmin(arcmin, _redshift=redshift, _cosmo=cosmo):
            return arcmin_to_comoving(arcmin, _redshift, _cosmo)

        def inverse_arcmin(comoving, _redshift=redshift, _cosmo=cosmo):
            return comoving_to_arcmin(comoving, _redshift, _cosmo)

        # -------------------------------------------------------------------
        # Top row: 3-D stacking
        # -------------------------------------------------------------------
        ax_3d = axes[0, col]
        if verbose:
            print(f"  [3D] starting ...")
        run_3d_stacking(
            stacker=stacker,
            OmegaBaryon=OmegaBaryon,
            baryon_types=baryon_types,
            pType2=pType2,
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
            verbose=verbose,
        )
        # Style: 3-D subplot
        ax_3d.set_xlabel('R [comoving kpc/h]')
        ax_3d.set_ylabel(r'Baryon fraction $/ \, (\Omega_b / \Omega_m)$')
        ax_3d.axhline(1.0, color='k', ls='--', lw=2)
        ax_3d.set_xlim(0.0, maxRadius * radDistance)
        # ax_3d.legend(loc='lower right')
        ax_3d.grid(True)
        ax_3d.set_title(f'{sim_label} (3D)')

        # -------------------------------------------------------------------
        # Bottom row: 2-D map stacking
        # -------------------------------------------------------------------
        ax_2d = axes[1, col]
        if verbose:
            print(f"  [2D] starting ...")
        maxRadius_arcmin = run_2d_stacking(
            stacker=stacker,
            cosmo=cosmo,
            OmegaBaryon=OmegaBaryon,
            baryon_types=baryon_types,
            pType2=pType2,
            filterType=filterType,
            filterType2=filterType2,
            minRadius=minRadius,
            maxRadius=maxRadius,
            nRadii=nRadii,
            projection=projection,
            saveField=saveField,
            loadField=loadField,
            radDistance=radDistance,
            ax=ax_2d,
            colours=colours,
            forward_arcmin=forward_arcmin,
            inverse_arcmin=inverse_arcmin,
            verbose=verbose,
        )
        # Style: 2-D subplot
        ax_2d.set_xlabel('R [arcmin]')
        ax_2d.set_ylabel(r'Baryon fraction $/ \, (\Omega_b / \Omega_m)$')
        ax_2d.axhline(1.0, color='k', ls='--', lw=2)
        ax_2d.set_xlim(0.0, maxRadius_arcmin * radDistance)
        ax_2d.grid(True)
        ax_2d.set_title(f'{sim_label} (2D)')
        if col == n_sims - 1:  # Only add legend to the rightmost subplot to avoid duplicates
            ax_2d.legend(loc='lower right')

        # Secondary x-axis on the 2-D subplot showing comoving kpc/h
        secax_x = ax_2d.secondary_xaxis('top', functions=(forward_arcmin, inverse_arcmin))
        secax_x.set_xlabel('R [comoving kpc/h]')

    # -----------------------------------------------------------------------
    # Figure-level labels and layout
    # -----------------------------------------------------------------------
    # Row labels placed as text on the leftmost axes so that shared-y axes do
    # not duplicate the y-label on every panel
    axes[0, 0].annotate('3D stacking', xy=(-0.25, 0.5), xycoords='axes fraction',
                        ha='right', va='center', rotation=90, fontsize=14,
                        fontweight='bold')
    axes[1, 0].annotate('2D stacking', xy=(-0.25, 0.5), xycoords='axes fraction',
                        ha='right', va='center', rotation=90, fontsize=14,
                        fontweight='bold')

    # fig.suptitle(f'Baryon Fractions at $z={redshift}$', fontsize=20)
    fig.tight_layout()

    out_path = figPath / f'{figName}_z{redshift}_{filterType}_stackArea.{figType}'
    print(f'Saving figure to {out_path}')
    fig.savefig(out_path, dpi=300) # type: ignore
    plt.close(fig)

    print(f'Done!  Total elapsed time: {time.time()-t_total:.1f}s')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Produce a combined 3D/2D baryon-fraction stacked-area figure '
                    'for multiple simulations.'
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
