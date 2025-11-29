import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# from scipy.stats import binned_statistic_2d
# from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import matplotlib
import matplotlib.cm as cm

# from abacusnbody.analysis.tsc import tsc_parallel
import time

from astropy.cosmology import FlatLambdaCDM, Planck18
import astropy.units as u

# Import packages

sys.path.append('../src/')
# from filter_utils import *
# from SZstacker import SZMapStacker # type: ignore
from stacker import SimulationStacker
from utils import arcmin_to_comoving, comoving_to_arcmin
from halos import select_massive_halos

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

# # Set matplotlib to use Computer Modern font
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Computer Modern Roman']
# plt.rcParams['text.usetex'] = True
# plt.rcParams['mathtext.fontset'] = 'cm'

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
    pType = stack_config.get('particle_type', 'tau')
    projection = stack_config.get('projection', 'xy')

    # maskHaloes and maskRadii will be set in the loop
    pixelSize = stack_config.get('pixel_size', 0.5) # in arcmin

    # Plotting parameters
    # get the datetime for file naming
    now = datetime.now()
    dt_string = now.strftime("%m-%d")

    figPath = Path(plot_config.get('fig_path')) / dt_string
    figPath.mkdir(parents=False, exist_ok=True)
    plotErrorBars = plot_config.get('plot_error_bars', True)
    figName = plot_config.get('fig_name', 'default_figure')
    figType = plot_config.get('fig_type', 'pdf')

    colourmaps = ['hot', 'cool']
    colourmaps = ['hsv', 'twilight']

    # Create (2, 4) subplots: top row for TNG, bottom row for SIMBA
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    
    # Define mask configurations: [maskRadii=1, 2, 3, False]
    mask_configs = [
        {'maskHaloes': True, 'maskRadii': 1.0},
        {'maskHaloes': True, 'maskRadii': 2.0},
        {'maskHaloes': True, 'maskRadii': 3.0},
        {'maskHaloes': False, 'maskRadii': None}
    ]
    
    t0 = time.time()
    
    # Loop over mask configurations (columns)
    for col_idx, mask_config in enumerate(mask_configs):
        maskHaloes = mask_config['maskHaloes']
        maskRadii = mask_config['maskRadii']
        
        if verbose:
            if maskHaloes:
                print(f"\n=== Processing column {col_idx + 1}: Masked with radius {maskRadii} ===")
            else:
                print(f"\n=== Processing column {col_idx + 1}: No masking ===")
        
        # Loop over simulation types (rows)
        for row_idx, sim_type in enumerate(config['simulations']):
            sim_type_name = sim_type['sim_type']
            ax = axes[row_idx, col_idx]
            
            colourmap = matplotlib.colormaps[colourmaps[row_idx]] # type: ignore
            
            if sim_type_name == 'IllustrisTNG':
                TNG_sims = sim_type['sims']
                colours = colourmap(np.linspace(0.2, 0.85, len(TNG_sims)))
            elif sim_type_name == 'SIMBA':
                SIMBA_sims = sim_type['sims']
                colours = colourmap(np.linspace(0.2, 0.85, len(SIMBA_sims)))
            else:
                raise ValueError(f"Unknown simulation type: {sim_type_name}")

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

                    radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, minRadius=1.0, maxRadius=6.0, pixelSize=pixelSize, # type: ignore
                                                         save=saveField, load=loadField, radDistance=radDistance,
                                                         projection=projection, mask=maskHaloes, maskRad=maskRadii)

                    try:
                        OmegaBaryon = stacker.header['OmegaBaryon']
                        # Use IllustrisTNG cosmology for plot later.
                        cosmo = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'], Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K, Ob0=OmegaBaryon)
                        
                        haloes = stacker.loadHalos(stacker.simType)
                        haloMass = haloes['GroupMass']
                        
                        halo_mask = select_massive_halos(haloMass, 10**(13.22), 5e14) # TODO: make this configurable from user input
                        haloes['GroupMass'] = haloes['GroupMass'][halo_mask]
                        haloes['GroupRad'] = haloes['GroupRad'][halo_mask] # in kpc/h
                        R200C = np.mean(haloes['GroupRad']) # in kpc / h
                    except KeyError:
                        OmegaBaryon = 0.0456  # Default value for Illustris-1

                elif sim_type_name == 'SIMBA':
                    feedback = sim['feedback']
                    sim_name_show = sim_name + '_' + feedback
                    if verbose:
                        print(f"Processing feedback model: {feedback}")
                    
                    stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                                simType=sim_type_name, 
                                                feedback=feedback)
                    
                    radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, minRadius=1.0, maxRadius=6.0, pixelSize=pixelSize, # type: ignore
                                                         save=saveField, load=loadField, radDistance=radDistance,
                                                         projection=projection, mask=maskHaloes, maskRad=maskRadii)
                    
                    OmegaBaryon = 0.048  # Default value for SIMBA
                    sim_name = sim_name_show
                else:
                    raise ValueError(f"Unknown simulation type: {sim_type_name}")

                # Plotting
                T_CMB = 2.7255
                v_c = 300000 / 299792458 # velocity over speed of light.
                
                profiles_plot = np.mean(profiles0, axis=1)
                ax.plot(radii0 * radDistance, profiles_plot, label=sim_name, color=colours[j], lw=2, marker='o')
                if plotErrorBars:
                    profiles_err = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
                    upper = profiles_plot + profiles_err
                    lower = profiles_plot - profiles_err
                    ax.fill_between(radii0 * radDistance, 
                                    lower, 
                                    upper, 
                                    color=colours[j], alpha=0.2)

        # Plot data only on the last column (col_idx == 3)
        if col_idx == 3 and plot_config['plot_data']:
            data_path = plot_config['data_path']
            data = np.load(data_path)
            r_data = data['theta_arcmins']
            profile_data = data['signal']
            profile_err = data['noise']
                    
            # Plot data on both rows of the last column
            for row_idx in range(2):
                axes[row_idx, col_idx].errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k', 
                                                 label=plot_config['data_label'], markersize=5, zorder=10)

    # Configure all subplots
    T_CMB = 2.7255
    v_c = 300000 / 299792458
    k = 1 / (T_CMB * v_c * 1e6)
    
    def forward_arcmin(arcmin):
        return arcmin_to_comoving(arcmin, redshift, cosmo)
    def inverse_arcmin(comoving):
        return comoving_to_arcmin(comoving, redshift, cosmo)
    
    
    for row_idx in range(2):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            
            if col_idx != 3:
                R200C_arcmin = comoving_to_arcmin(R200C * u.kpc / u.h, redshift, cosmo)
                ax.axvline(R200C_arcmin * (col_idx + 1), color='k', linestyle='--', lw=1)

            # Set x-label only on bottom row
            if row_idx == 1:
                ax.set_xlabel('R [arcmin]')
            
            # Set y-label only on leftmost column
            if col_idx == 0:
                ax.set_ylabel(r'$T_{kSZ}$ [$\mu K \rm{arcmin}^2$]')
            
            # Set secondary x-axis on top row
            if row_idx == 0:
                secax_x = ax.secondary_xaxis('top',
                                             functions=(forward_arcmin,
                                                       inverse_arcmin))
                secax_x.set_xlabel('R [kpc/h]')
            
            # Set secondary y-axis only on rightmost column
            if col_idx == 3:
                ax.legend(loc='lower right', fontsize=12)
                secax = ax.secondary_yaxis('right',
                                           functions=(lambda y: y * k,
                                                     lambda y: y / k))
                if row_idx == 0:
                    secax.set_ylabel(r'$\tau_{\rm CAP} = T_{kSZ}/T_{CMB}\;\; c/v_{rms}$')
                else:
                    secax.set_ylabel(r'$\tau_{\rm CAP} = T_{kSZ}/T_{CMB}\;\; c/v_{rms}$')
            
            ax.set_yscale('log')
            ax.set_xlim(0.0, 6.5)
            ax.grid(True)
            
            # Set column titles on top row
            if row_idx == 0:
                if col_idx < 3:
                    ax.set_title(f'Masked ($R_{{mask}} = {mask_configs[col_idx]["maskRadii"]:.0f} R_{{200c}}$)')
                else:
                    ax.set_title('No Masking')
    
    # Set row labels
    fig.text(0.02, 0.25, 'IllustrisTNG', fontsize=20, va='center', rotation=90, ha='center')
    fig.text(0.02, 0.75, 'SIMBA', fontsize=20, va='center', rotation=90, ha='center')

    fig.suptitle(f'Stacked kSZ profiles, {filterType} filter, z={redshift}', fontsize=22)
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))  # Leave space on left for row labels and top for title
    fig.savefig(figPath / f'{figName}_{pType}_z{redshift}_masking_comparison.{figType}', dpi=300) # type: ignore
    plt.close(fig)
    
    print('Done!!!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str, default='./configs/tau_z05_CAP_masked.yaml', help='Path to the configuration file.')
    # parser.add_argument("--set", nargs=2, action="append",
    #                     metavar=("KEY", "VALUE"),
    #                     help="Override with dotted.key  value")
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)