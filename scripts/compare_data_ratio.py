"""compare_data_ratio.py

Plot the simulated particle-type mass ratio (pType / pType2), normalised by the
cosmic baryon fraction, for all configured simulations on a single set of axes.
Observational data (optional) are overlaid as error-bar markers.

The plotted quantity is::

    R(r) = <Sigma_pType(r)> / <Sigma_pType2(r)>  /  (Omega_b / Omega_m)

where the angle brackets denote the mean over all stacked haloes, and r is the
projected aperture radius.  R = 1 indicates that the chosen component tracks
baryons in exact proportion to the cosmic mean.

Usage
-----
    python compare_data_ratio.py -p configs/mass_ratio_data_z05.yaml
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import argparse
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

sys.path.append('../src/')
from utils import arcmin_to_comoving, comoving_to_arcmin  # type: ignore
from stacker import SimulationStacker  # type: ignore

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore  # noqa: F401 (needed by stacker internals)

# ---------------------------------------------------------------------------
# Matplotlib style: Computer Modern / LaTeX-compatible serif fonts
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

# ---------------------------------------------------------------------------
# Fallback Omega_b values for simulations whose snapshot headers omit the key.
# IllustrisTNG value: from Illustris-1 documentation.
# SIMBA value: Planck 2015 cosmology used by the SIMBA suite.
# ---------------------------------------------------------------------------
_OMEGA_B_TNG_FALLBACK   = 0.0456
_OMEGA_B_SIMBA_FALLBACK = 0.048


def load_measurements_npz(path: str) -> dict:
    """Load a nested dict saved by the companion save_measurements_npz helper.

    The ``.npz`` file is expected to use keys of the form
    ``"outer_key/inner_key"``, which are reconstructed into a two-level dict.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        Nested ``{outer_key: {inner_key: np.ndarray}}`` mapping.
    """
    archive = np.load(path)
    out: dict = {}
    for k in archive.files:
        outer_key, inner_key = k.split("/", 1)
        out.setdefault(outer_key, {})[inner_key] = archive[k]
    return out


def main(path2config: str, verbose: bool = True) -> None:
    """Run the mass-ratio comparison and save the output figure.

    For each simulation listed in the config the script:

    1. Stacks the numerator particle type (``particle_type``) and denominator
       particle type (``particle_type_2``) using the respective filter types.
    2. Computes the ratio-of-means normalised by the cosmic baryon fraction::

           R(r) = mean(Sigma_num, axis=halos) / mean(Sigma_den, axis=halos)
                  / (Omega_b / Omega_m)

    3. Propagates the standard error of the mean through the ratio in
       quadrature::

           sigma_R / R = sqrt( (sigma_num/mean_num)^2 + (sigma_den/mean_den)^2 )

    Observational data (``plot_data: true`` in the config) are overlaid as
    square error-bar markers with a small horizontal jitter to separate
    overlapping series.

    All simulations are plotted on a single axes, with IllustrisTNG and SIMBA
    families drawn from different colourmaps for visual distinction.

    Args:
        path2config: Path to the YAML configuration file.
        verbose: If True, print progress messages to stdout.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_config = config.get('stack', {})
    plot_config  = config.get('plot',  {})

    # ---- Stacking parameters (read from YAML with sensible defaults) ----
    redshift     = stack_config.get('redshift', 0.5)
    load_field   = stack_config.get('load_field', True)
    save_field   = stack_config.get('save_field', True)
    rad_distance = stack_config.get('rad_distance', 1.0)   # arcmin per unit radius
    pType        = stack_config.get('particle_type',   'ionized_gas')
    filter_type  = stack_config.get('filter_type',     'CAP')
    pType2       = stack_config.get('particle_type_2', 'total')
    filter_type2 = stack_config.get('filter_type_2',   'DSigma')
    projection   = stack_config.get('projection', 'yz')
    pixel_size   = stack_config.get('pixel_size', 0.5)    # arcmin
    beam_size    = stack_config.get('beam_size', None)     # arcmin; None → no smoothing
    min_radius   = stack_config.get('min_radius', 1.0)
    max_radius   = stack_config.get('max_radius', 6.0)
    n_radii      = stack_config.get('num_radii', 11)
    use_subhalos = stack_config.get('use_subhalos', False)
    mask_haloes  = stack_config.get('mask_haloes', False)
    mask_radii   = stack_config.get('mask_radii', 3.0)

    # ---- Output path: figures/<year-month>/<month-day>/ ----
    now      = datetime.now()
    fig_path = (
        Path(plot_config.get('fig_path', '../figures/'))
        / now.strftime("%Y-%m")
        / now.strftime("%m-%d")
    )
    fig_path.mkdir(parents=True, exist_ok=True)

    fig_name        = plot_config.get('fig_name', 'compare_data_ratio')
    fig_type        = plot_config.get('fig_type', 'png')
    plot_error_bars = plot_config.get('plot_error_bars', True)
    do_plot_data    = plot_config.get('plot_data', False)

    # ---- Single axes for all simulations ----
    fig, ax = plt.subplots(figsize=(10, 8))

    # Reference cosmology for the secondary x-axis (populated from the first
    # simulation processed; both TNG and SIMBA typically share similar
    # cosmological parameters so using the first is adequate for axis labelling).
    cosmo_ref: Optional[FlatLambdaCDM] = None

    colourmaps = ['plasma', 'twilight']

    t0 = time.time()

    for i, sim_group in enumerate(config['simulations']):
        sim_type_name = sim_group['sim_type']

        cmap    = matplotlib.colormaps[colourmaps[i]]  # type: ignore[attr-defined]
        n_sims  = len(sim_group['sims'])
        colours = cmap(np.linspace(0.2, 0.85, n_sims))

        for j, sim in enumerate(sim_group['sims']):
            sim_name = sim['name']
            snapshot = sim['snapshot']

            # ---- Instantiate the stacker and resolve Omega_b ----
            if sim_type_name == 'IllustrisTNG':
                stacker   = SimulationStacker(sim_name, snapshot, z=redshift, simType=sim_type_name)
                sim_label = sim_name
                try:
                    omega_b = stacker.header['OmegaBaryon']
                except KeyError:
                    # Illustris-1 omits OmegaBaryon from the header
                    omega_b = _OMEGA_B_TNG_FALLBACK
                    if verbose:
                        print(f"  [warn] OmegaBaryon missing in {sim_label} header; "
                              f"using fallback {_OMEGA_B_TNG_FALLBACK}")

            elif sim_type_name == 'SIMBA':
                feedback  = sim['feedback']
                sim_label = f"{sim_name}_{feedback}"
                stacker   = SimulationStacker(sim_name, snapshot, z=redshift,
                                             simType=sim_type_name, feedback=feedback)
                try:
                    omega_b = stacker.header['OmegaBaryon']
                except KeyError:
                    omega_b = _OMEGA_B_SIMBA_FALLBACK
                    if verbose:
                        print(f"  [warn] OmegaBaryon missing in {sim_label} header; "
                              f"using fallback {_OMEGA_B_SIMBA_FALLBACK}")

            else:
                raise ValueError(f"Unknown simulation type: {sim_type_name!r}. "
                                 "Expected 'IllustrisTNG' or 'SIMBA'.")

            # Build the cosmology object for this simulation (used to populate
            # the reference cosmology for the secondary x-axis on first call).
            cosmo = FlatLambdaCDM(
                H0=100 * stacker.header['HubbleParam'],
                Om0=stacker.header['Omega0'],
                Tcmb0=2.7255 * u.K,
                Ob0=omega_b,
            )
            if cosmo_ref is None:
                cosmo_ref = cosmo   # retain first cosmology for secondary axis

            if verbose:
                print(f"Processing {sim_label} (snapshot {snapshot})")

            # ---- Stack numerator and denominator ----
            # Note: pixelSize/beamSize are intentionally omitted for the numerator
            # (pType) so it uses stackMap's defaults, matching the original behaviour.
            # The denominator (pType2) passes pixelSize/beamSize explicitly.
            base_kwargs = dict(
                minRadius=min_radius,
                maxRadius=max_radius,
                numRadii=n_radii,
                projection=projection,
                save=save_field,
                load=load_field,
                radDistance=rad_distance,
                mask=mask_haloes,
                maskRad=mask_radii,
                use_subhalos=use_subhalos,
            )

            # profiles shape: (n_radii, n_halos)
            radii0, profiles0 = stacker.stackMap(pType,  filterType=filter_type,  **base_kwargs)
            radii1, profiles1 = stacker.stackMap(pType2, filterType=filter_type2,
                                                 pixelSize=pixel_size, beamSize=beam_size,
                                                 **base_kwargs)

            # When plotting DM vs total, rescale profiles0 so it is expressed in
            # baryon-equivalent units rather than raw DM mass.
            if pType == 'DM' and pType2 == 'total':
                profiles0 = profiles0 / ((stacker.header['Omega0'] - omega_b) / omega_b)

            # ---- Ratio-of-means normalised by the cosmic baryon fraction ----
            # Using ratio-of-means (not mean-of-ratio) for consistency with the
            # error propagation below.
            f_baryon      = omega_b / stacker.header['Omega0']
            mean0         = np.mean(profiles0, axis=1)   # (n_radii,)
            mean1         = np.mean(profiles1, axis=1)
            profiles_plot = mean0 / mean1 / f_baryon

            ax.plot(
                radii0 * rad_distance,
                profiles_plot,
                label=sim_label,
                color=colours[j],
                lw=2,
                marker='o',
            )

            if plot_error_bars:
                # Standard error of the mean for each component, propagated
                # through the ratio in quadrature.
                err0         = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
                err1         = np.std(profiles1, axis=1) / np.sqrt(profiles1.shape[1])
                profiles_err = np.abs(profiles_plot) * np.sqrt((err0 / mean0)**2 + (err1 / mean1)**2)

                ax.fill_between(
                    radii0 * rad_distance,
                    profiles_plot - profiles_err,
                    profiles_plot + profiles_err,
                    color=colours[j],
                    alpha=0.2,
                )

    # ---- Overlay observational data points (optional) ----
    if do_plot_data and 'data_path' in plot_config:
        data         = load_measurements_npz(plot_config['data_path'])
        n_data       = len(data)
        cmap         = mpl.colormaps['gist_rainbow']  # type: ignore[attr-defined]
        data_colours = cmap(np.linspace(0, 1.0, 4)) # type: ignore[attr-defined]

        for k, key in enumerate(data.keys()):
            # Symmetric horizontal jitter to separate overlapping error bars.
            # Example for n_data=4: offsets [-0.075, -0.025, 0.025, 0.075] arcmin.
            print(key)
            if key == 'source_bin_0':
                colour = 'black'
                fmt = 's'
                label = f'DESI x ACT x HSC combined'
            else:
                break
                colour = data_colours[k]
                fmt = 'o'
                label = f'HSC Bin {k+1}'
            
            jitter = (k - (n_data - 1) / 2) * 0.05
            jitter = 0.0  # disable jitter for now since we only plot the one combined data point

            ax.errorbar(
                data[key]['ksz_theta_arcmin'] + jitter,
                data[key]['ratio'],
                yerr=data[key]['ratio_err'],
                fmt=fmt,
                color=colour,
                label=label,
                markersize=6,
                capsize=2,
            )

    # ---- Secondary x-axis: comoving kpc/h ----
    # Only added if at least one simulation was successfully processed.
    if cosmo_ref is not None:
        secax_x = ax.secondary_xaxis(
            'top',
            functions=(
                lambda arcmin: arcmin_to_comoving(arcmin, redshift, cosmo_ref),
                lambda kpc_h:  comoving_to_arcmin(kpc_h,  redshift, cosmo_ref),
            ),
        )
        secax_x.set_xlabel('R [comoving kpc/h]')

    # ---- Axes labels and cosmetics ----
    # Dashed horizontal line at R=1: the baryon fraction equals the cosmic mean.
    if pType == 'ionized_gas':
        label_pType = 'kSZ'
    if pType2 == 'total':
        label_pType2 = 'lens'
    
    ax.axhline(1.0, color='k', ls='--', lw=1.5, label='_nolegend_')
    ax.set_xlabel('R [arcmin]')
    ax.set_ylabel(
        rf'$\frac{{\langle \Delta \Sigma_{{\rm {label_pType}}} \rangle}}{{\langle \Delta \Sigma_{{\rm {label_pType2}}} \rangle}} \times \frac{{\Omega_m}}{{\Omega_b}}$'
    )
    ax.set_xlim(0.0, max_radius * rad_distance + 0.5)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    # ax.set_title(rf'Ratio at $z={redshift}$')

    fig.tight_layout()

    # ---- Save figure ----
    out_stem = f'combined_fig_{pType}_{pType2}_{fig_name}_z{redshift}_{filter_type}_{filter_type2}'
    out_path = fig_path / f'{out_stem}.{fig_type}'
    print(f'Saving figure to {out_path}')
    fig.savefig(out_path, dpi=150)  # type: ignore
    plt.close(fig)

    print(f'Done. Elapsed: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot simulated mass ratio against observational data on a single axes.',
    )
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/mass_ratio_data_z05.yaml',
        help='Path to the YAML configuration file.',
    )
    args = vars(parser.parse_args())
    print(f"Config: {args['path2config']}")
    main(**args)
