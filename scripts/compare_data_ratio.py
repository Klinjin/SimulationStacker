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

import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from mpi4py import MPI

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import argparse
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

sys.path.append('/pscratch/sd/l/lindajin/SimulationStacker/src/')
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
    "text.usetex": False,   # latex not available on compute nodes; use matplotlib mathtext
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
    sp_ratio_path   = plot_config.get('sp_ratio_path', None)  # path to Ptot_Pdm_ratio_k_le15.npz

    # ---- MPI communicator ----
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    colourmaps = ['plasma', 'twilight']
    t0 = time.time()

    # ---- Per-sim-group: split sim_indices across MPI ranks, gather to rank 0 ----
    # gathered_groups is a list of dicts (one per sim_group) built only on rank 0;
    # all other ranks exit main() after the gather loop and do no plotting.
    gathered_groups: list = []

    for i, sim_group in enumerate(config['simulations']):
        sim_type_name = sim_group['sim_type']
        sim_entry     = sim_group['sims'][0]   # CAMELS: one entry per group
        sim_name      = sim_entry['name']
        snapshot      = sim_entry['snapshot']
        n_sims        = sim_entry.get('sample', 20)
        seed_val      = sim_entry.get('seed', 42)

        rng         = random.Random(seed_val)
        sim_indices = rng.sample(range(1024), n_sims)

        # ---- Distribute sim_indices evenly across MPI ranks ----
        chunks          = np.array_split(sim_indices, size)
        local_sim_chunk = chunks[rank]   # this rank's subset of CAMELS indices

        local_results: list = []

        for sim_i in local_sim_chunk:
            # Global position in sim_indices — used to assign a consistent colour.
            global_j = sim_indices.index(sim_i)

            # ---- Instantiate the stacker and resolve Omega_b ----
            if sim_type_name == 'IllustrisTNG':
                stacker = SimulationStacker(sim_index=sim_i,
                                           snapshot=snapshot,
                                           simType=sim_type_name)
                sim_label = f"{sim_name}[{sim_i}]"
                try:
                    omega_b = stacker.header['OmegaBaryon']
                except KeyError:
                    omega_b = _OMEGA_B_TNG_FALLBACK
                    if verbose:
                        print(f"  [rank {rank}] OmegaBaryon missing in {sim_label} header; "
                              f"using fallback {_OMEGA_B_TNG_FALLBACK}")

            elif sim_type_name == 'SIMBA':
                feedback  = sim_entry['feedback']
                sim_label = f"{sim_name}_{feedback}[{sim_i}]"
                stacker   = SimulationStacker(sim_name, snapshot, z=redshift,
                                             simType=sim_type_name, feedback=feedback)
                try:
                    omega_b = stacker.header['OmegaBaryon']
                except KeyError:
                    omega_b = _OMEGA_B_SIMBA_FALLBACK
                    if verbose:
                        print(f"  [rank {rank}] OmegaBaryon missing in {sim_label} header; "
                              f"using fallback {_OMEGA_B_SIMBA_FALLBACK}")

            else:
                raise ValueError(f"Unknown simulation type: {sim_type_name!r}. "
                                 "Expected 'IllustrisTNG' or 'SIMBA'.")

            if verbose:
                print(f"  [rank {rank}] Processing {sim_label} (snapshot {snapshot})")

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
                pixelSize=pixel_size,
                beamSize=beam_size,
                # use_subhalos=use_subhalos,
            )

            # profiles shape: (n_radii, n_halos)
            radii0, profiles0 = stacker.stackMap(pType,  filterType=filter_type,  **base_kwargs)
            radii1, profiles1 = stacker.stackMap(pType2, filterType=filter_type2, **base_kwargs)

            if pType == 'DM' and pType2 == 'total':
                profiles0 = profiles0 / ((stacker.header['Omega0'] - omega_b) / omega_b)

            local_results.append({
                'sim_i':       sim_i,
                'global_j':    global_j,
                'sim_label':   sim_label,
                'radii':       radii0,
                'profiles0':   profiles0,   # (n_radii, n_halos)
                'profiles1':   profiles1,
                'omega_b':     omega_b,
                'Omega0':      stacker.header['Omega0'],
                'HubbleParam': stacker.header['HubbleParam'],
            })

        # ---- Gather all local result lists to rank 0 ----
        all_rank_results = comm.gather(local_results, root=0)

        if rank == 0:
            # Flatten list-of-lists and restore original sim_indices order.
            flat = [item for sublist in all_rank_results for item in sublist]
            flat.sort(key=lambda d: d['global_j'])
            gathered_groups.append({
                'sim_type_name': sim_type_name,
                'n_sims':        n_sims,
                'seed_val':      seed_val,
                'sim_indices':   sim_indices,
                'cmap_name':     colourmaps[i % len(colourmaps)],
                'results':       flat,
            })

    # ---- Non-root ranks are done — only rank 0 plots ----
    if rank != 0:
        if verbose:
            print(f"  [rank {rank}] Done. Elapsed: {time.time() - t0:.1f} s")
        return

    # ---- Rank 0: load SP(k) suppression for colormap ranking ----
    # Load Ptot/PDM ratio array (shape: 1024 x n_k) if path is provided.
    sp_colors_by_group: list = []   # per-group RGBA colour arrays keyed by sim_i
    sp_norm_by_group:  list = []    # per-group Normalize for colorbar
    if sp_ratio_path is not None:
        _f          = np.load(sp_ratio_path)
        _Ptot       = _f['Ptot_Pdm_ratio']   # shape (1024, n_k)
        _k          = _f['k']
        _k10_idx    = np.argmin(np.abs(_k - 10.0))
        _cmap_sp    = matplotlib.colormaps['plasma']   # type: ignore[attr-defined]
        for group in gathered_groups:
            _indices = group['sim_indices']            # list of CAMELS indices
            _sp      = _Ptot[_indices, _k10_idx]       # SP(k=10) for the selected sims
            _ranks   = np.argsort(np.argsort(_sp))     # rank within subset
            _colours = _cmap_sp(_ranks / max(len(_sp) - 1, 1))
            # keyed by sim_i for O(1) lookup
            _colour_map = {sim_i: _colours[local_j]
                           for local_j, sim_i in enumerate(_indices)}
            sp_colors_by_group.append(_colour_map)
            sp_norm_by_group.append(
                plt.Normalize(vmin=_sp.min(), vmax=_sp.max())
            )
    else:
        # Fall back to original sequential colourmaps per group
        for group in gathered_groups:
            _cmap    = matplotlib.colormaps[group['cmap_name']]  # type: ignore[attr-defined]
            _colours = _cmap(np.linspace(0.2, 0.85, group['n_sims']))
            _colour_map = {rec['sim_i']: _colours[rec['global_j']]
                           for rec in group['results']}
            sp_colors_by_group.append(_colour_map)
            sp_norm_by_group.append(None)

    # ---- Rank 0: build the figure from gathered results ----
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(right=0.82)   # leave room for colorbar on right
    cosmo_ref: Optional[FlatLambdaCDM] = None

    for grp_idx, group in enumerate(gathered_groups):
        colour_map = sp_colors_by_group[grp_idx]

        for i, rec in enumerate(group['results']):
            radii0    = rec['radii']
            profiles0 = rec['profiles0']
            profiles1 = rec['profiles1']
            omega_b   = rec['omega_b']
            Omega0    = rec['Omega0']
            colour    = colour_map[rec['sim_i']]

            if cosmo_ref is None:
                cosmo_ref = FlatLambdaCDM(
                    H0=100 * rec['HubbleParam'],
                    Om0=Omega0,
                    Tcmb0=2.7255 * u.K,
                    Ob0=omega_b,
                )

            # ---- Ratio-of-means normalised by the cosmic baryon fraction ----
            f_baryon      = omega_b / Omega0
            mean0         = np.mean(profiles0, axis=1)   # (n_radii,)
            mean1         = np.mean(profiles1, axis=1)
            profiles_plot = mean0 / mean1 / f_baryon

            print(f"Plotting {rec['sim_label']}")
            ax.plot(
                radii0 * rad_distance,
                profiles_plot,
                color=colour,
                lw=2,
                marker='o',
                label=f'CAMELS_{sim_name}_n{n_sims}s{seed_val}' if i == 0 else None,
            )

            if plot_error_bars:
                err0         = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
                err1         = np.std(profiles1, axis=1) / np.sqrt(profiles1.shape[1])
                profiles_err = np.abs(profiles_plot) * np.sqrt((err0 / mean0)**2 + (err1 / mean1)**2)

                ax.fill_between(
                    radii0 * rad_distance,
                    profiles_plot - profiles_err,
                    profiles_plot + profiles_err,
                    color=colour,
                    alpha=0.2,
                )

    # ---- Colorbar: SP(k=10) ranking ----
    # Use the last group's norm (all groups share the same k so ranges are comparable).
    _last_norm = sp_norm_by_group[-1]
    if _last_norm is not None:
        _cmap_sp_cb = matplotlib.colormaps['plasma']   # type: ignore[attr-defined]
        _sm = plt.cm.ScalarMappable(cmap=_cmap_sp_cb, norm=_last_norm)
        _sm.set_array([])
        cbar_ax = fig.add_axes([0.84, 0.10, 0.03, 0.80])
        cbar    = fig.colorbar(_sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(
            r'$SP(k=10\,h/\mathrm{Mpc}) = P_{\rm tot}(k)/P_{\rm DM}(k)$',
            fontsize=13,
        )

    # Expose last-group metadata for the output filename.
    last_group  = gathered_groups[-1]
    n_sims      = last_group['n_sims']
    seed_val    = last_group['seed_val']
    sim_indices = last_group['sim_indices']

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
    ax.set_ylim(0.0, 1.6)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title(rf'Ratio at $z={redshift}$')

    fig.tight_layout(rect=[0, 0, 0.82, 1])  # keep right margin reserved for colorbar

    # ---- Save figure ----
    out_stem = f'CAMELS_{sim_name}_n{n_sims}s{seed_val}_{pType}-{filter_type}_{pType2}-{filter_type2}_{fig_name}_SPk10_z{redshift}'
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
