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

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# Import packages

sys.path.append('../src/')
# from filter_utils import *
# from SZstacker import SZMapStacker # type: ignore
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

    minRadius = stack_config.get('min_radius', 1.0)
    maxRadius = stack_config.get('max_radius', 10.0)
    nRadii = stack_config.get('num_radii', 11)

    maskHaloes = stack_config.get('mask_haloes', False)
    maskRadii = stack_config.get('mask_radii', 2.0) # in virial radii

    pixelSize = stack_config.get('pixel_size', 0.5) # in arcmin

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
    colourmaps = ['hsv', 'twilight']

    beamSizes = [1.6, 2.1]
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), sharey='row')
    
    t0 = time.time()
    for beam_idx, beamSize in enumerate(beamSizes):
        for i, sim_type in enumerate(config['simulations']):
            sim_type_name = sim_type['sim_type']
            
            colourmap = matplotlib.colormaps[colourmaps[i]] # type: ignore
            
            if sim_type_name == 'IllustrisTNG':
                TNG_sims = sim_type['sims']
                colours = colourmap(np.linspace(0.2, 0.85, len(TNG_sims)))
                ax = axes[beam_idx, 0]  # Plot TNG on left subplot
            if sim_type_name == 'SIMBA':
                SIMBA_sims = sim_type['sims']
                colours = colourmap(np.linspace(0.2, 0.85, len(SIMBA_sims)))
                ax = axes[beam_idx, 1]  # Plot SIMBA on right subplot

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

                    # radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, minRadius=minRadius, maxRadius=maxRadius, pixelSize=pixelSize, # type: ignore
                    #                                      save=saveField, load=loadField, radDistance=radDistance,
                    #                                      projection=projection, mask=maskHaloes, maskRad=maskRadii)


                    try:
                        OmegaBaryon = stacker.header['OmegaBaryon']
                    except KeyError:
                        OmegaBaryon = 0.0456  # Default value for Illustris-1

                    

                elif sim_type_name == 'SIMBA':
                    # SIMBA simulations have different feedback models               
                    feedback = sim['feedback']

                    sim_name_show = sim_name + '_' + feedback
                    if verbose:
                        print(f"Processing feedback model: {feedback}")
                    
                    stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                                simType=sim_type_name, 
                                                feedback=feedback)

                    # radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, minRadius=minRadius, maxRadius=maxRadius, pixelSize=pixelSize, # type: ignore
                    #                                      save=saveField, load=loadField, radDistance=radDistance,
                    #                                      projection=projection, mask=maskHaloes, maskRad=maskRadii)
                    
                    OmegaBaryon = 0.048  # Default value for SIMBA

                    # if fractionType == 'gas':
                    #     fraction = profiles0 / (profiles0 + profiles1 + profiles4 + profiles5) / (OmegaBaryon / stacker.header['Omega0']) # OmegaBaryon = 0.048 from Planck 2015
                    # elif fractionType == 'baryon':
                    #     fraction = (profiles0 + profiles4 + profiles5) / (profiles0 + profiles1 + profiles4 + profiles5) / (OmegaBaryon / stacker.header['Omega0']) # OmegaBaryon = 0.048 from Planck 2015

                    # fraction_plot = np.median(fraction, axis=1)
                    # ax.plot(radii0 * radDistance, fraction_plot, label=sim_name_show, color=colours[j], lw=2)
                    # # ax.plot(radii0 * radDistance, profiles0.mean(axis=1), label=sim_name_show, color=colours[j], lw=2)
                    # # ax.plot(radii0 * radDistance, profiles0, label=sim_name_show, color=colours[j], lw=2)
                    # if plotErrorBars:
                    #     fraction_err = np.std(fraction, axis=1) / np.sqrt(fraction.shape[1])
                    #     upper = np.percentile(fraction, 75, axis=1)
                    #     lower = np.percentile(fraction, 25, axis=1)
                    #     ax.fill_between(radii0 * radDistance, 
                    #                     lower, 
                    #                     upper, 
                    #                     color=colours[j], alpha=0.2)
                else:
                    raise ValueError(f"Unknown simulation type: {sim_type_name}")

                radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, minRadius=minRadius, maxRadius=maxRadius, 
                                                     numRadii=nRadii, pixelSize=pixelSize, beamSize=beamSize,
                                                     save=False, load=False, radDistance=radDistance,
                                                     projection=projection, mask=maskHaloes, maskRad=maskRadii)
                
                # Now for Plotting
                
                # if fractionType == 'gas':
                #     fraction = profiles0 / (profiles0 + profiles1 + profiles4) / (OmegaBaryon / stacker.header['Omega0']) # OmegaBaryon = 0.048 from Planck 2015
                # elif fractionType == 'baryon':
                #     fraction = (profiles0 + profiles4) / (profiles0 + profiles1 + profiles4) / (OmegaBaryon / stacker.header['Omega0']) # OmegaBaryon = 0.048 from Planck 2015
                
                
                T_CMB = 2.7255
                # speed of light:
                # v_c = 0.0007
                v_c = 300000 / 299792458 # velocity over speed of light.
                # v_c = 1.06e-3
                # The conversion from tau to micro-Kelvin for kSZ is T_CMB * (v/c) * 1e6
                # This is already done in the SZstacker.py file when loading the tau field.
                # So here we do not need to do it again.
                # profiles0 = profiles0 * T_CMB * 1e6 * v_c # Convert to micro-Kelvin
                
                if sim_type_name == 'SIMBA':
                    # SIMBA simulations have different feedback models               
                    sim_name = sim_name + '_' + sim['feedback'] 
                
                profiles_plot = np.mean(profiles0, axis=1)
                ax.plot(radii0 * radDistance, profiles_plot, label=sim_name, color=colours[j], lw=2, marker='o')
                if plotErrorBars:
                    profiles_err = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
                    # upper = np.percentile(profiles0, 75, axis=1)
                    # lower = np.percentile(profiles0, 25, axis=1)

                    upper = profiles_plot + profiles_err
                    lower = profiles_plot - profiles_err
                    ax.fill_between(radii0 * radDistance, 
                                    lower, 
                                    upper, 
                                    color=colours[j], alpha=0.2)

        T_CMB = 2.7255
        v_c = 300000 / 299792458 # velocity over speed of light.

        if plot_config['plot_data']:
            data_path = plot_config['data_path']
            rad_key = plot_config.get('rad_key', 'RApArcmin')
            data_key = plot_config.get('data_key', 'pz1_act_dr6_fiducial')
            data_err_key = data_key + '_err'
            # data = np.load(data_path)
            data = pd.read_csv(data_path)
            r_data = data[rad_key]
            profile_data = data[data_key]
            profile_err = data[data_err_key]
            # Plot data on both subplots
            axes[beam_idx, 0].errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k', 
                           label=plot_config['data_label'], markersize=8, zorder=10)
            axes[beam_idx, 1].errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k', 
                             label=plot_config['data_label'], markersize=8, zorder=10)

        # Configure both subplots
        for ax, title in zip([axes[beam_idx, 0], axes[beam_idx, 1]], ['IllustrisTNG', 'SIMBA']):
            ax.set_xlabel('R [arcmin]', fontsize=18)
            ax.set_yscale(plot_config.get('yscale', 'log'))
            
            # Set tick label font size
            # ax.tick_params(axis='both', which='major', labelsize=14)
            # ax.tick_params(axis='both', which='minor', labelsize=12)

            if title == 'IllustrisTNG':
                ax.set_ylabel(r'Compton-$y$ [$\rm{arcmin}^2$]')#, fontsize=18)
            # elif title == 'SIMBA':
            #     secax = ax.secondary_yaxis('right',
            #                            functions=(lambda y: y * k,
            #                                      lambda y: y / k))
            #     secax.set_ylabel(r'$\tau_{\rm CAP} = T_{kSZ}/T_{CMB}\;\; c/v_{rms}$')#, fontsize=18)
                # secax.tick_params(axis='y', which='major', labelsize=14)
            
            # Secondary Y axis
            k = 1 / (T_CMB * v_c * 1e6)

            xmin = np.max((minRadius * radDistance - 1.0, 0))
            ax.set_xlim(xmin, maxRadius * radDistance + 0.5)
            ax.legend(loc='best')#, fontsize=20)
            ax.grid(True)
            ax.set_title(f'{title} (beam={beamSize}")')#, fontsize=20)

    fig.suptitle(f'Stacked {pType} profiles, {filterType} filter, z={redshift}', fontsize=20)
    fig.tight_layout()
    if maskHaloes:
        fig.savefig(figPath / f'beamTest_{figName}_{pType}_z{redshift}_masked_{maskRadii:.1f}.{figType}', dpi=300) # type: ignore
    else:
        fig.savefig(figPath / f'beamTest_{figName}_{pType}_z{redshift}_{filterType}.{figType}', dpi=300) # type: ignore
    plt.close(fig)
    print('Done!!!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str, default='./configs/tSZ_z05_CAP.yaml', help='Path to the configuration file.')
    # parser.add_argument("--set", nargs=2, action="append",
    #                     metavar=("KEY", "VALUE"),
    #                     help="Override with dotted.key  value")
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)