"""make_hod_ratio.py
==================
Generate a figure comparing two halo selection methods for the stacked tau
(optical depth) 2D CAP profile.

For each simulation the script computes:

    ratio(R) = tau_CAP^{SHAM}(R)  /  tau_CAP^{mass-cut}(R)

where the two selection methods are:

  mass-cut : FoF haloes selected by select_massive_halos
             (halo_mass_avg ≈ 10^13.22 Msun, upper bound 5×10^14 Msun)

  SHAM     : subhaloes selected by abundance matching via
             select_abundance_subhalos (target number density
             5×10^-4 (cMpc/h)^-3, the stack_on_array default)

Both stacking runs share the same cached tau particle map — the underlying
2D projected field is built from raw particle data and is independent of
halo selection.  Only the stacking centres (halo/subhalo positions) differ.

The unit conversion applied inside stackMap for tau (optical depth →
micro-Kelvin) is a single scalar factor that cancels in the ratio, so the
ratio is dimensionless regardless of units.

Layout
------
Single panel with one line per simulation, shaded uncertainty bands, a
horizontal dashed reference line at 1.0 with a ±5 % shaded band, and a
vertical dotted line at the mean R200c of TNG300-1 (from the mass-cut
halo selection).

Usage
-----
    python make_hod_ratio.py -p configs/hod_ratio_z05.yaml
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
# This matches the assignment in the tSZ reference scripts.
_COLOURMAPS = {'IllustrisTNG': 'twilight', 'SIMBA': 'hsv'}

# Ordered reference TNG sim list used to fix colour positions so that omitting
# TNG100-1 from the config does not shift the colours of TNG300-1/Illustris-1.
# Extend this list here if new TNG sims are added.
_TNG_REFERENCE_ORDER = ['TNG100-1', 'TNG300-1', 'Illustris-1']

# Panel label (single panel; kept as list for easy extension).
_PANEL_LABELS = ['(a)']

# Default OmegaBaryon fallbacks (not stored in all simulation headers).
_OMEGA_BARYON_ILLUSTRIS_DEFAULT = 0.0456
_OMEGA_BARYON_SIMBA_DEFAULT     = 0.048


# ===========================================================================
# Helper functions  (mirror style of make_ratios3x2.py)
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
        Target redshift for angular distance calculations.

    Returns
    -------
    stacker   : SimulationStacker
    cosmo     : FlatLambdaCDM
    sim_label : str   — human-readable legend label
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

    return stacker, cosmo, sim_label


def _ratio_and_err(profiles_sham: np.ndarray,
                   profiles_mass: np.ndarray):
    """Compute the SHAM / mass-cut ratio with propagated standard error.

    Parameters
    ----------
    profiles_sham, profiles_mass : ndarray, shape (n_radii, n_halos)
        Per-halo stacked tau CAP profiles for the SHAM and mass-cut
        selections respectively.

    Returns
    -------
    ratio : ndarray, shape (n_radii,)
        Mean ratio tau_SHAM / tau_mass-cut at each radius.
    err   : ndarray, shape (n_radii,)
        One-sigma uncertainty from standard error propagation in quadrature.
        For a ratio f = A / B:  df/f = sqrt( (dA/A)^2 + (dB/B)^2 ).
    """
    mean_sham = np.mean(profiles_sham, axis=1)   # shape (n_radii,)
    mean_mass = np.mean(profiles_mass, axis=1)
    ratio = mean_sham / mean_mass

    # Standard error on each mean: SE = std / sqrt(N_halos)
    se_sham = np.std(profiles_sham, axis=1) / np.sqrt(profiles_sham.shape[1])
    se_mass = np.std(profiles_mass, axis=1) / np.sqrt(profiles_mass.shape[1])

    # Quadrature propagation for ratio f = A / B: df/f = sqrt((dA/A)^2 + (dB/B)^2).
    # Guard against near-zero means (tau can approach zero at large radii) to
    # avoid silent NaN / inf in the relative standard errors.
    with np.errstate(invalid='ignore', divide='ignore'):
        rel_se_sham = np.where(np.abs(mean_sham) > 0, se_sham / mean_sham, np.nan)
        rel_se_mass = np.where(np.abs(mean_mass) > 0, se_mass / mean_mass, np.nan)
    err = np.abs(ratio) * np.sqrt(rel_se_sham ** 2 + rel_se_mass ** 2)

    return ratio, err


def compute_hod_ratio(stacker: SimulationStacker, params: dict,
                      cosmo: FlatLambdaCDM):
    """Stack the tau map under both halo selections and compute their ratio.

    The tau particle map is created / loaded once by the first stackMap call
    and reused by the second, because stacker.maps caches by
    (pType, z, projection, pixelSize, beamSize).  The halo selection logic
    runs entirely inside stack_on_array and does not affect the stored map.

    Note on SHAM abundance target
    ------------------------------
    stackMap does not expose halo_abundance_target, so the SHAM run always
    uses the stack_on_array default of 5e-4 (cMpc/h)^-3.  Since we want the
    default selections this is intentional and documented in the YAML config.

    Parameters
    ----------
    stacker : SimulationStacker
    params  : dict
        Stacking parameters extracted from the YAML config.
    cosmo   : FlatLambdaCDM
        Cosmology object for the R200c → arcmin conversion.

    Returns
    -------
    radii        : ndarray  — stacking radii in arcmin (scaled by rad_distance)
    ratio        : ndarray  — tau_CAP(SHAM) / tau_CAP(mass-cut)
    err          : ndarray  — one-sigma uncertainty on ratio
    R200c_arcmin : float    — mean R200c of mass-cut halos, in arcmin
    """
    pType      = params['particle_type']
    filterType = params['filter_type']
    pixelSize  = params['pixel_size']
    minR       = params['min_radius']
    maxR       = params['max_radius']
    nRadii     = params['num_radii']
    radDist    = params['rad_distance']
    projection = params['projection']
    load       = params['load_field']
    save       = params['save_field']
    sub_mean   = params['subtract_mean']
    z          = params['redshift']

    # subtract_mean=True is unsafe here: stackMap subtracts and restores the
    # mean in-place on the cached map.  Floating-point rounding from the
    # subtract/restore cycle would corrupt the map seen by the second
    # stackMap call.  The YAML default is False; raise early if misconfigured.
    if sub_mean:
        raise NotImplementedError(
            "subtract_mean=True is not supported in make_hod_ratio.py. "
            "Both stackMap calls share the same cached tau map; the "
            "in-place subtract/restore cycle in stackMap accumulates "
            "floating-point error between the mass-cut and SHAM calls."
        )

    # Mass-cut selection parameters (passed directly into stackMap).
    halo_mass_avg   = params['halo_mass_avg']
    halo_mass_upper = params['halo_mass_upper']

    # ------------------------------------------------------------------
    # Stack 1: mass-cut halo selection  (use_subhalos=False)
    #
    # This call also loads (or creates) the tau map and caches it in
    # stacker.maps.  The second stackMap call below will find the cached
    # map and skip the expensive field computation entirely.
    # ------------------------------------------------------------------
    radii, profiles_mass = stacker.stackMap(
        pType,
        filterType=filterType,
        minRadius=minR,
        maxRadius=maxR,
        numRadii=nRadii,
        z=z,
        projection=projection,
        save=save,
        load=load,
        radDistance=radDist,
        pixelSize=pixelSize,
        subtract_mean=sub_mean,
        use_subhalos=False,
        halo_mass_avg=halo_mass_avg,
        halo_mass_upper=halo_mass_upper,
    )

    # ------------------------------------------------------------------
    # Stack 2: SHAM / abundance-matching selection  (use_subhalos=True)
    #
    # The tau map is already cached in stacker.maps from the call above.
    # Only the halo catalogue and selection change: SubhaloMass is loaded
    # and the top-N subhalos by mass are selected to match the target number
    # density (5e-4 (cMpc/h)^-3, the stack_on_array default).
    # ------------------------------------------------------------------
    radii_sham, profiles_sham = stacker.stackMap(
        pType,
        filterType=filterType,
        minRadius=minR,
        maxRadius=maxR,
        numRadii=nRadii,
        z=z,
        projection=projection,
        save=save,
        load=load,
        radDistance=radDist,
        pixelSize=pixelSize,
        subtract_mean=sub_mean,
        use_subhalos=True,
    )
    # Both calls use identical radius parameters so the returned radii arrays
    # must match.  Assert defensively in case of future parameter divergence.
    assert np.allclose(radii, radii_sham), (
        "SHAM and mass-cut stacking radii must match; got different arrays. "
        "Check that minRadius/maxRadius/numRadii are identical for both calls."
    )

    # ------------------------------------------------------------------
    # Compute ratio and propagate standard errors in quadrature.
    #
    # The tau unit conversion applied inside stackMap (optical depth →
    # micro-Kelvin) is a scalar factor identical for both selections and
    # therefore cancels out in the ratio.
    # ------------------------------------------------------------------
    ratio, err = _ratio_and_err(profiles_sham, profiles_mass)

    # ------------------------------------------------------------------
    # Mean R200c from mass-cut halo selection, converted to arcmin.
    # Used for the vertical reference line in the figure.
    # ------------------------------------------------------------------
    haloes    = stacker.loadHalos()
    halo_mask = select_halos(haloes['GroupMass'], 'massive',
                             target_average_mass=halo_mass_avg,
                             upper_mass_bound=halo_mass_upper)
    R200c_kpch   = np.mean(haloes['GroupRad'][halo_mask])   # comoving kpc/h
    R200c_arcmin = comoving_to_arcmin(R200c_kpch, z, cosmo=cosmo)

    # Return radii scaled by rad_distance, matching the convention in
    # compute_2d_profile_ratio from make_ratios3x2.py.  radii from stackMap
    # is in arcmin; radDist=1.0 (the YAML default) is a pure display scale
    # factor and leaves the values unchanged.
    return radii * radDist, ratio, err, R200c_arcmin


# ===========================================================================
# Main
# ===========================================================================

def main(path2config: str, verbose: bool = True):
    """Generate the SHAM vs mass-cut tau CAP ratio figure.

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

    # Collect all stacking parameters into a single dict for easy passing.
    params = {
        'redshift':        stack_cfg.get('redshift',        0.5),
        'projection':      stack_cfg.get('projection',      'yz'),
        'load_field':      stack_cfg.get('load_field',      True),
        'save_field':      stack_cfg.get('save_field',      True),
        'subtract_mean':   stack_cfg.get('subtract_mean',   False),
        'particle_type':   stack_cfg.get('particle_type',   'tau'),
        'filter_type':     stack_cfg.get('filter_type',     'CAP'),
        'pixel_size':      stack_cfg.get('pixel_size',      0.5),
        'min_radius':      stack_cfg.get('min_radius',      1.0),
        'max_radius':      stack_cfg.get('max_radius',      10.0),
        'num_radii':       stack_cfg.get('num_radii',       11),
        'rad_distance':    stack_cfg.get('rad_distance',    1.0),
        # Mass-cut parameters: passed through to stackMap → stack_on_array.
        'halo_mass_avg':   stack_cfg.get('halo_mass_avg',   10 ** 13.22),
        'halo_mass_upper': stack_cfg.get('halo_mass_upper', 5e14),
        # SHAM target: stackMap does not expose halo_abundance_target so the
        # stack_on_array default of 5e-4 (cMpc/h)^-3 is always used.
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

    figName         = plot_cfg.get('fig_name',       'hod_ratio_z05')
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
    # Build a flat list of (sim_type, sim_dict) preserving config order.
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

    _simba_idx  = 0
    sim_colours = {}   # key: sim_label → colour
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
    # Create figure — single panel, 9 × 7 inches
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    # R200c reference line: stored from the first IllustrisTNG sim processed
    # (TNG300-1 when the config lists IllustrisTNG first, as in the default
    # hod_ratio_z05.yaml).  Uses the mass-cut halo selection.
    R200c_ref_arcmin = None
    R200c_ref_label  = None

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

            # ---- Instantiate stacker and cosmology ----
            stacker, cosmo, sim_label = setup_stacker(sim, sim_type_name,
                                                       redshift)
            colour = sim_colours[sim_label]

            # ---- Compute SHAM / mass-cut ratio at each radius ----
            if verbose:
                print(f"  Stacking tau CAP profiles (mass-cut then SHAM)...")

            radii, ratio, err, R200c_arcmin = compute_hod_ratio(
                stacker, params, cosmo)

            if verbose:
                print(f"  R200c(mass-cut) = {R200c_arcmin:.3f} arcmin")
                print(f"  ratio range:     [{ratio.min():.3f}, {ratio.max():.3f}]")

            # Cache R200c from the first IllustrisTNG sim for the vline.
            if sim_type_name == 'IllustrisTNG' and R200c_ref_arcmin is None:
                R200c_ref_arcmin = R200c_arcmin
                R200c_ref_label  = sim_label

            # ---- Plot line and shaded ±1σ uncertainty band ----
            ax.plot(radii, ratio, label=sim_label, color=colour, lw=2,
                    marker='o')
            if plot_error_bars:
                ax.fill_between(radii, ratio - err, ratio + err,
                                color=colour, alpha=0.2)

    # ------------------------------------------------------------------
    # Axis decorations
    # ------------------------------------------------------------------

    # Horizontal dashed black line at unity — the no-difference reference.
    ax.axhline(1.0, color='k', ls='--', lw=2, zorder=0)

    # Shaded ±5 % band for visual reference (grey, behind data lines).
    ax.axhspan(0.95, 1.05, color='grey', alpha=0.15, zorder=0,
               label=r'$\pm 5\%$')

    # Vertical dotted line at mean R200c from TNG300-1 mass-cut halos.
    if R200c_ref_arcmin is not None:
        ax.axvline(R200c_ref_arcmin, color='gray', ls=':', lw=2,
                   label=rf'$\langle R_{{200c}} \rangle$ ({R200c_ref_label})')

    ax.set_xlabel('R [arcmin]', fontsize=20)
    ax.set_ylabel(
        r'$\tau_{\rm CAP}^{\textrm{SHAM}}(R) \;/\; '
        r'\tau_{\rm CAP}^{\textrm{mass-cut}}(R)$',
        fontsize=18,
    )
    ax.set_xlim(0.0, params['max_radius'] * params['rad_distance'] + 0.5)
    ax.grid(True)
    ax.legend(loc='best', fontsize=13)

    # Panel label in the top-left corner (matches make_ratios3x2.py convention).
    ax.text(0.03, 0.97, _PANEL_LABELS[0], transform=ax.transAxes,
            fontsize=18, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    fig.tight_layout()
    out_path = figPath / f'{figName}.{figType}'
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
        description="Generate tau CAP SHAM vs mass-cut ratio figure.")
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/hod_ratio_z05.yaml',
        help='Path to the YAML configuration file.',
    )
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")
    main(**args)
