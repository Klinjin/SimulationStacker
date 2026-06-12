import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

# from scipy.stats import binned_statistic_2d
# from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import matplotlib
import matplotlib.cm as cm

# from abacusnbody.analysis.tsc import tsc_parallel
import time

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# Import packages

sys.path.append('../src/')
# from filter_utils import *
from utils import ksz_from_delta_sigma, arcmin_to_comoving, comoving_to_arcmin
from SZstacker import SZMapStacker # type: ignore
from stacker import SimulationStacker

sys.path.append('../../illustrisPython/')
import illustris_python as il # type: ignore

import yaml
import argparse
from pathlib import Path
from datetime import datetime

# --- NEW: set default font to Computer Modern (with fallbacks) and increase tick fontsize ---
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "mathtext.fontset": "cm",
    # Base font sizes (adjust as desired)
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})
# --- END NEW ---

def save_measurements_npz(path, data):
    """
    Save a nested dict-of-dicts to an NPZ file without pickling.

    The input must have the structure:
        {
            outer_key: {
                inner_key: array_like,
                ...
            },
            ...
        }

    All inner values are converted to NumPy arrays and stored under
    keys of the form "outer_key/inner_key".

    Args:
        path (str): Output .npz file path.
        data (dict): Nested dict of array-like values.

    Raises:
        TypeError: If data is not a dict-of-dicts.
        ValueError: If an inner value cannot be converted to a NumPy array.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    flat = {}

    for outer_key, inner in data.items():
        if not isinstance(inner, dict):
            raise TypeError(
                f"Value for {outer_key!r} must be a dict, got {type(inner)}"
            )

        for inner_key, value in inner.items():
            try:
                arr = np.asarray(value)
            except Exception as e:
                raise ValueError(
                    f"Could not convert {outer_key}/{inner_key} to array"
                ) from e

            flat[f"{outer_key}/{inner_key}"] = arr

    np.savez_compressed(path, **flat)

def load_measurements_npz(path):
    """
    Load a nested dict-of-dicts saved by save_measurements_npz.
    """
    z = np.load(path)
    out = {}

    for k in z.files:
        outer_key, inner_key = k.split("/", 1)
        out.setdefault(outer_key, {})[inner_key] = z[k]

    return out

def main(path2config, verbose=True):
    """Main function to process the simulation maps.

    Args:
        path2config (str): Path to the configuration file.
        verbose (bool, optional): If True, prints detailed information. Defaults to True.

    Raises:
        ValueError: If the configuration file is invalid or missing required fields.
    """

    with open(path2config) as f:
        config = yaml.safe_load(f)
    
    stack_config = config.get('stack', {})
    plot_config = config.get('plot', {})
    
    # Stacking parameters
    redshift = stack_config.get('redshift', 0.5)
    filterType = stack_config.get('filter_type', 'CAP')
    loadField = stack_config.get('load_field', True)
    saveField = stack_config.get('save_field', True)
    radDistance = stack_config.get('rad_distance', 1.0)
    # baryon_types: list of particle types to show as stacked fractional contributions
    baryon_types = stack_config.get('baryon_types', ['Stars', 'BH', 'ionized_gas', 'neutral_gas'])
    projection = stack_config.get('projection', 'xy')
    pixelSize = stack_config.get('pixel_size', 0.5)
    beamSize = stack_config.get('beam_size', 1.6) # in arcmin, for Gaussian smoothing of the maps before stacking. Set to 0 for no smoothing.
    
    filterType2 = stack_config.get('filter_type_2', 'DSigma')
    pType2 = stack_config.get('particle_type_2', 'total')

    minRadius = stack_config.get('min_radius', 1.0)
    maxRadius = stack_config.get('max_radius', 10.0)
    nRadii = stack_config.get('num_radii', 11)


    # fractionType = config['fraction_type']

    # Plotting parameters
    # get the datetime for file naming
    now = datetime.now()
    yr_string = now.strftime("%Y-%m")
    dt_string = now.strftime("%m-%d")

    figPath = Path(plot_config.get('fig_path', '../figures/')) / yr_string / dt_string
    figPath.mkdir(parents=True, exist_ok=True)
    plotErrorBars = plot_config.get('plot_error_bars', True)
    figName = plot_config.get('fig_name', 'default_figure')
    figType = plot_config.get('fig_type', 'pdf')

    colourmaps = ['hot', 'cool']
    colourmaps = ['plasma', 'twilight']
    colourmap = matplotlib.colormaps['plasma'] # type: ignore
    colours = colourmap(np.linspace(0.0, 0.8, len(baryon_types)))

    # fig, ax = plt.subplots(figsize=(10,8))
    fig, (ax_tng, ax_simba) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    t0 = time.time()
    for i, sim_type in enumerate(config['simulations']):
        sim_type_name = sim_type['sim_type']
        
        colourmap = matplotlib.colormaps[colourmaps[i]] # type: ignore
        
        if sim_type_name == 'IllustrisTNG':
            TNG_sims = sim_type['sims']
            # colours = colourmap(np.linspace(0.2, 0.85, len(TNG_sims)))
            ax = ax_tng
            tng_name = sim_type['sims'][0]['name']
        if sim_type_name == 'SIMBA':
            SIMBA_sims = sim_type['sims']
            # colours = colourmap(np.linspace(0.2, 0.85, len(SIMBA_sims)))
            ax = ax_simba

        if verbose:
            print(f"Processing simulations of type: {sim_type_name}")
        
        for j, sim in enumerate(sim_type['sims']):
            sim_name = sim['name']
            snapshot = sim['snapshot']
            
            if verbose:
                print(f"Processing simulation: {sim_name}")
            
            if sim_type_name == 'IllustrisTNG':
                
                stacker = SimulationStacker(sim_name, snapshot, z=redshift, 
                                       simType=sim_type_name)
                try:
                    OmegaBaryon = stacker.header['OmegaBaryon']
                except KeyError:
                    OmegaBaryon = 0.0456  # Default value for Illustris-1
                cosmo_tng = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'], Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K, Ob0=OmegaBaryon)                    
                def forward_arcmin(arcmin):
                    return arcmin_to_comoving(arcmin, redshift, cosmo_tng)
                def inverse_arcmin(comoving):
                    return comoving_to_arcmin(comoving, redshift, cosmo_tng)
                

            elif sim_type_name == 'SIMBA':
                # SIMBA simulations have different feedback models               
                feedback = sim['feedback']

                sim_name_show = sim_name + '_' + feedback
                if verbose:
                    print(f"Processing feedback model: {feedback}")
                
                stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                       simType=sim_type_name, 
                                       feedback=feedback)
                OmegaBaryon = 0.048  # Default value for SIMBA
                cosmo_simba = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'], Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K, Ob0=OmegaBaryon)
                def forward_arcmin(arcmin):
                    return arcmin_to_comoving(arcmin, redshift, cosmo_simba)
                def inverse_arcmin(comoving):
                    return comoving_to_arcmin(comoving, redshift, cosmo_simba)

            else:
                raise ValueError(f"Unknown simulation type: {sim_type_name}")
            minRadius_arcmin = inverse_arcmin(minRadius * radDistance)
            maxRadius_arcmin = inverse_arcmin(maxRadius * radDistance)

            # Stack the total field (denominator) and each baryon type (numerators)
            radii1, profiles1 = stacker.stackMap(pType2, filterType=filterType2, minRadius=minRadius_arcmin,
                                                 maxRadius=maxRadius_arcmin, numRadii=nRadii,
                                                #  pixelSize=pixelSize, beamSize=beamSize,
                                                 save=saveField, load=loadField, radDistance=radDistance,
                                                 projection=projection)
            # radii1 = forward_arcmin(radii1) / radDistance # convert back to physical distance units for plotting on x-axis
            # profiles_baryon[bt]: shape (nRadii, nHalos); CAP sum per halo per radius for each baryon type
            profiles_baryon = {}
            for bt in baryon_types:
                t1 = time.time()
                print(f"Stacking baryon type: {bt}")
                radii0, profiles_baryon[bt] = stacker.stackMap(bt, filterType=filterType, minRadius=minRadius_arcmin,
                                                               maxRadius=maxRadius_arcmin, numRadii=nRadii,
                                                               save=saveField, load=loadField, radDistance=radDistance,
                                                               projection=projection)
                # radii0 = forward_arcmin(radii0) / radDistance # convert back to physical distance units for plotting on x-axis
                print(f"Done stacking {bt} in {time.time() - t1:.1f} seconds")

            if sim_type_name == 'SIMBA':
                # SIMBA simulations have different feedback models               
                sim_name = sim_name + '_' + sim['feedback']

            # Mean total profile across haloes at each radius (denominator)
            mean_total = np.mean(profiles1, axis=1)  # shape: (nRadii,)

            # For each baryon type compute its fraction of the total field,
            # normalised by the cosmic baryon fraction (Omega_b / Omega_m) so that
            # a perfectly baryon-traced field would sum to 1.
            fractions = []
            bt_labels = []
            for bt in baryon_types:
                mean_bt = np.mean(profiles_baryon[bt], axis=1)  # shape: (nRadii,)
                fractions.append(mean_bt / mean_total / (OmegaBaryon / stacker.header['Omega0']))
                bt_labels.append(bt)

            # Stacked-area plot: each band is one baryon type's normalised contribution
            ax.stackplot(radii0 * radDistance, fractions, labels=bt_labels, alpha=0.8, colors=colours)

    T_CMB = 2.7255
    v_c = 300000 / 299792458 # velocity over speed of light.

    def forward_arcmin(arcmin):
        return arcmin_to_comoving(arcmin, redshift, cosmo_tng)
    def inverse_arcmin(comoving):
        return comoving_to_arcmin(comoving, redshift, cosmo_tng)


        
        # data_path = plot_config['data_path']
        # data = np.load(data_path)
        # r_data = data['theta_arcmins']
        # profile_data = data['prof']
        # profile_err = np.sqrt(np.diag(data['cov']))
        # plt.errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k', label=plot_config['data_label'], markersize=8)

    # ax.set_xlabel('Radius (arcmin)')
    # ax.set_ylabel('f')
    # ax_simba.set_xlabel('R [arcmin]', fontsize=18)
    # ax_tng.set_xlabel('R [arcmin]', fontsize=18)
    # ax.set_ylabel(r'$\frac{T_{kSZ} / \pi R^2}{\Delta \Sigma}$', fontsize=18)
    # ax_tng.set_ylabel(rf'$\frac{{{pType}}}{{{pType2}}} \; / \; (\Omega_b / \Omega_m)$', fontsize=18)
    # ax_simba.axhline(1.0, color='k', ls='--', lw=2)
    # ax_tng.axhline(1.0, color='k', ls='--', lw=2)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # --- Secondary Y axis examples ---

    # 1) Multiplicative (recommended for log scale): y2 = k * y1
    # k = 1/ (T_CMB * v_c * 1e6)  # constant factor between axes
    # secax = ax.secondary_yaxis('right',
    #                            functions=(lambda y: y * k,      # forward
    #                                       lambda y: y / k))     # inverse
    # secax.set_ylabel(r'$\tau_{\rm CAP} = T_{kSZ}/T_{CMB}\;\; c/v_{rms}$', fontsize=18)

    # 2) If you really need an additive offset (ensure y+C > 0 on log scale):
    # C = 5.0  # additive offset in the same units
    # secax = ax.secondary_yaxis('right',
    #                          functions=(lambda y: y + C,      # forward
    #                                     lambda y: y - C))     # inverse
    # secax.set_ylabel(r'$T_{kSZ}$ + C [$\mu K \rm{arcmin}^2$]')

    # ax.set_xlim(0.0, 6.5)
    # ax.set_ylim(0, 1.2)
    # ax.axhline(1.0, color='k', ls='--', lw=2)
    # ax_tng.legend(loc='lower right', fontsize=12)
    # ax_simba.legend(loc='lower right', fontsize=12)
    # ax_tng.grid(True)
    # ax_simba.grid(True)
    # ax_tng.set_title('IllustrisTNG', fontsize=16)
    # ax_simba.set_title('SIMBA', fontsize=16)
    
    # Configure both subplots
    for ax, title in zip([ax_tng, ax_simba], [tng_name, 'SIMBA']):
        ax.set_xlabel('R [arcmin]', fontsize=18)
        # ax.set_yscale('log')
        
        # Set tick label font size
        # ax.tick_params(axis='both', which='major', labelsize=14)
        # ax.tick_params(axis='both', which='minor', labelsize=12)

        # Y-axis: sum of baryon-type fractions, each normalised by (Omega_b / Omega_m)
        ax.set_ylabel(r'Baryon fraction $/ \, (\Omega_b / \Omega_m)$', fontsize=18)
        
        secax_x = ax.secondary_xaxis('top', functions=(forward_arcmin, inverse_arcmin))
        secax_x.set_xlabel('R [comoving kpc/h]', fontsize=18)
        # secax_x.tick_params(axis='x', which='major', labelsize=14)

        ax.axhline(1.0, color='k', ls='--', lw=2)
        # ax.set_xlim(0.0, maxRadius * radDistance + 0.5)
        ax.set_xlim(0.0, maxRadius_arcmin * radDistance + 0.5) # type: ignore
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True)
        ax.set_title(f'{title}')#, fontsize=20)
    
    fig.suptitle(f'Baryon Fractions at z={redshift}', fontsize=20)
    
    fig.tight_layout()
    # fig.savefig(figPath / f'{figName}_{pType}_z{redshift}_ratio.{figType}', dpi=300) # type: ignore
    print(f'Saving figure to {figPath / f"stackArea_2D_{pType2}_{figName}_z{redshift}_{filterType}_2D_stackArea.{figType}"}')
    fig.savefig(figPath / f'stackArea_2D_{pType2}_{figName}_z{redshift}_{filterType}_2D_stackArea.{figType}', dpi=300) # type: ignore
    plt.close(fig)
    
    print('Done!!!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str, default='./configs/2D_stackArea_z05.yaml', help='Path to the configuration file.')
    # parser.add_argument("--set", nargs=2, action="append",
    #                     metavar=("KEY", "VALUE"),
    #                     help="Override with dotted.key  value")
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)