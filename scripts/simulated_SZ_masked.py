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
    redshift = stack_config['redshift']
    filterType = stack_config['filter_type']
    loadField = stack_config['load_field']
    saveField = stack_config['save_field']
    radDistance = stack_config['rad_distance']
    pType = stack_config['particle_type']
    projection = stack_config.get('projection', 'xy')
    maskRad = stack_config.get('mask_radii', 1.0) # in units of R200c

    # fractionType = stack_config['fraction_type']

    # Plotting parameters
    figPath = Path(plot_config['fig_path'])
    figPath.mkdir(parents=False, exist_ok=True)
    plotErrorBars = plot_config['plot_error_bars']
    figName = plot_config['fig_name']
    figType = plot_config['fig_type']

    colourmaps = ['hot', 'cool']

    fig, ax = plt.subplots(figsize=(10,8))
    t0 = time.time()
    for i, sim_type in enumerate(config['simulations']):
        sim_type_name = sim_type['sim_type']
        
        colourmap = matplotlib.colormaps[colourmaps[i]] # type: ignore
        
        if sim_type_name == 'IllustrisTNG':
            TNG_sims = sim_type['sims']
            colours = colourmap(np.linspace(0.2, 0.85, len(TNG_sims)))
        if sim_type_name == 'SIMBA':
            SIMBA_sims = sim_type['sims']
            colours = colourmap(np.linspace(0.2, 0.85, len(SIMBA_sims)))

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

                radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, maxRadius=6.0, # type: ignore
                                                     save=saveField, load=loadField, radDistance=radDistance,
                                                     projection=projection, mask=True, maskRad=maskRad)


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
                
                radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, maxRadius=6.0,  # type: ignore
                                                     save=saveField, load=loadField, radDistance=radDistance,
                                                     projection=projection, mask=True, maskRad=maskRad)
                
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
                upper = np.percentile(profiles0, 75, axis=1)
                lower = np.percentile(profiles0, 25, axis=1)
                ax.fill_between(radii0 * radDistance, 
                                lower, 
                                upper, 
                                color=colours[j], alpha=0.2)

    T_CMB = 2.7255
    v_c = 300000 / 299792458 # velocity over speed of light.

    if plot_config['plot_data']:
        data_path = plot_config['data_path']
        data = np.load(data_path)
        r_data = data['theta_arcmins']
        profile_data = data['prof']
        profile_err = np.sqrt(np.diag(data['cov']))
        plt.errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k', label=plot_config['data_label'], markersize=8)

    # ax.set_xlabel('Radius (arcmin)')
    # ax.set_ylabel('f')
    ax.set_xlabel('R [arcmin]', fontsize=18)
    ax.set_ylabel(r'$T_{kSZ}$ [$\mu K \rm{arcmin}^2$]', fontsize=18)
    # ax.set_xscale('log')
    ax.set_yscale('log')
    # --- Secondary Y axis examples ---

    # 1) Multiplicative (recommended for log scale): y2 = k * y1
    k = 1/ (T_CMB * v_c * 1e6)  # constant factor between axes
    secax = ax.secondary_yaxis('right',
                               functions=(lambda y: y * k,      # forward
                                          lambda y: y / k))     # inverse
    secax.set_ylabel(r'$\tau_{\rm CAP} = T_{kSZ}/T_{CMB}\;\; c/v_{rms}$', fontsize=18)

    # 2) If you really need an additive offset (ensure y+C > 0 on log scale):
    # C = 5.0  # additive offset in the same units
    # secax = ax.secondary_yaxis('right',
    #                          functions=(lambda y: y + C,      # forward
    #                                     lambda y: y - C))     # inverse
    # secax.set_ylabel(r'$T_{kSZ}$ + C [$\mu K \rm{arcmin}^2$]')

    ax.set_xlim(0.0, 6.5)
    # ax.set_ylim(0, 1.2)
    # ax.axhline(1.0, color='k', ls='--', lw=2)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True)
    ax.set_title(f'{filterType} filter (Masked {maskRad} R200c) at z={redshift}', fontsize=18)
    
    fig.tight_layout()
    fig.savefig(figPath / f'{figName}_{pType}_z{redshift}_masked_{maskRad:.1f}.{figType}', dpi=300) # type: ignore
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