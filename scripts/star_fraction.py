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

# --- NEW: set default font to Computer Modern (with fallbacks) and increase tick fontsize ---
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    # Base font sizes (adjust as desired)
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})
# --- END NEW ---

# from abacusnbody.analysis.tsc import tsc_parallel
import time

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# Import packages

sys.path.append('../src/')
# from filter_utils import *
from SZstacker import SZMapStacker # type: ignore
from stacker import SimulationStacker

sys.path.append('../../illustrisPython/')
import illustris_python as il # type: ignore

import yaml
import argparse
from pathlib import Path
from datetime import datetime

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
    
    redshift = config['redshift']
    filterType = config['filter_type']
    plotErrorBars = config['plot_error_bars']
    loadField = config['load_field']
    saveField = config['save_field']
    radDistance = config['rad_distance']
    pType = config['particle_type']
    projection = config.get('projection', 'xy')
    maskRad = config.get('mask_radii', 1.0) # in units of R200c

    nPixels = config.get('n_pixels', 4)
    
    # fractionType = config['fraction_type']
    now = datetime.now()
    dt_string = now.strftime("%m-%d")

    figPath = Path(config.get('fig_path')) / dt_string
    figPath.mkdir(parents=False, exist_ok=True)
    
    figName = config['fig_name']
    figType = config['fig_type']
    
    # --- NEW: check if we should load plot_dict from file ---
    load_from_file = config.get('load', False)
    load_path = config.get('load_path', None)
    # --- END NEW ---
    
    colourmaps = ['hot', 'cool']

    fig, ax = plt.subplots(figsize=(12,8))
    
    # --- NEW: ensure axis tick labels use the intended fontsize (per-axis enforcement) ---
    ax.tick_params(axis='both', which='both', labelsize=14)
    # --- END NEW ---
    
    plot_dict = {}
    t0 = time.time()
    
    # --- NEW: if load is true, load plot_dict from file and skip processing ---
    if load_from_file:
        if load_path is None:
            raise ValueError("Config parameter 'load' is True but 'load_path' is not specified.")
        load_path_obj = Path(load_path)
        if not load_path_obj.exists():
            raise FileNotFoundError(f"Load path does not exist: {load_path}")
        
        with open(load_path_obj, 'r') as f:
            plot_dict = yaml.safe_load(f)
        
        print(f"Loaded star fraction data from {load_path}")
        print("Loaded star fraction values (simulation: fraction):")
        for name, val in plot_dict.items():
            print(f"  {name}: {val:.6f}")
    else:
        # --- END NEW ---
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

                    # if sim_name == 'TNG300-1':
                    particles0 = stacker.makeField('gas', nPixels=nPixels)
                    # particles1 = stacker.makeField('DM')
                    particles4 = stacker.makeField('Stars', nPixels=nPixels)
                    particles5 = stacker.makeField('BH', nPixels=nPixels)
                    # else:
                    #     particles0 = stacker.loadSubsets('gas')
                    #     # particles1 = stacker.loadSubsets('DM')
                    #     particles4 = stacker.loadSubsets('Stars')
                    #     particles5 = stacker.loadSubsets('BH')
                    
                    # radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, maxRadius=6.0, # type: ignore
                    #                                      save=saveField, load=loadField, radDistance=radDistance,
                    #                                      projection=projection, mask=True, maskRad=maskRad)


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
                    
                    particles0 = stacker.makeField('gas', nPixels=nPixels)
                    # particles1 = stacker.makeField('DM')
                    particles4 = stacker.makeField('Stars', nPixels=nPixels)
                    particles5 = stacker.makeField('BH', nPixels=nPixels)
                    
                    # particles0 = stacker.loadSubsets('gas')
                    # # particles1 = stacker.loadSubsets('DM')
                    # particles4 = stacker.loadSubsets('Stars')
                    # particles5 = stacker.loadSubsets('BH')

                    # radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, maxRadius=6.0,  # type: ignore
                    #                                      save=saveField, load=loadField, radDistance=radDistance,
                    #                                      projection=projection, mask=True, maskRad=maskRad)
                    
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
                    # sim_name = sim['feedback']

                # if sim_name == 'TNG300-1':
                plot_dict[sim_name] = np.sum(particles4) / (np.sum(particles0) + np.sum(particles4) + np.sum(particles5)) 
                # else:
                    # plot_dict[sim_name] = np.sum(particles4['Masses']) / (np.sum(particles0['Masses']) + np.sum(particles4['Masses']) + np.sum(particles5['Masses'])) # type: ignore
                print(f"{sim_name}: Star Fraction = {plot_dict[sim_name]:.4f}")
                
                # profiles_plot = np.mean(profiles0, axis=1)
                # ax.plot(radii0 * radDistance, profiles_plot, label=sim_name, color=colours[j], lw=2, marker='o')
                # if plotErrorBars:
                #     profiles_err = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
                #     upper = np.percentile(profiles0, 75, axis=1)
                #     lower = np.percentile(profiles0, 25, axis=1)
                #     ax.fill_between(radii0 * radDistance, 
                #                     lower, 
                #                     upper, 
                #                     color=colours[j], alpha=0.2)
        # --- NEW: close the else block for load_from_file ---
        # (the loop above is now inside this else block)
    # --- END NEW ---

    T_CMB = 2.7255
    v_c = 300000 / 299792458 # velocity over speed of light.

    # if config['plot_data']:
    #     data_path = config['data_path']
    #     data = np.load(data_path)
    #     r_data = data['theta_arcmins']
    #     profile_data = data['prof']
    #     profile_err = np.sqrt(np.diag(data['cov']))
    #     plt.errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k', label=config['data_label'], markersize=8)


    labels = list(plot_dict.keys())
    values = list(plot_dict.values())

    plt.bar(labels, values, width=0.6, color='skyblue', edgecolor='k', alpha=0.7)


    # ax.set_xlabel('Radius (arcmin)')
    # ax.set_ylabel('f')
    ax.set_xlabel('Simulation Suites', fontsize=18)
    ax.set_ylabel(r'Stellar Fraction (compared to total baryons.)', fontsize=18)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # --- Secondary Y axis examples ---

    # # 1) Multiplicative (recommended for log scale): y2 = k * y1
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

    # ax.set_xlim(0.0, 6.5)  # Remove this line - it's cutting off bars
    ax.set_xlim(-0.5, len(labels) - 0.5)  # Dynamic x-limits based on number of bars
    # ax.set_ylim(0, 1.2)
    # ax.axhline(1.0, color='k', ls='--', lw=2)
    # ax.legend(loc='lower right', fontsize=12)
    ax.grid(True)
    ax.set_title(f'Baryon Star Fraction, z = {redshift}', fontsize=18)
    ax.set_xticklabels(labels, rotation=60, ha='right')
    
    # --- NEW: print and save plot_dict ---
    # Print all values in plot_dict
    print("Star fraction values (simulation: fraction):")
    for name, val in plot_dict.items():
        try:
            print(f"{name}: {float(val):.6f}")
        except Exception:
            print(f"{name}: {val}")

    # Save plot_dict to a YAML file next to the figure
    if not load_from_file:
        out_dict_path = figPath / f"{figName}_z{redshift}_star_fraction.yaml"
        try:
            # convert numpy types to native Python types for safe serialization
            serializable_dict = {k: float(v) for k, v in plot_dict.items()}
            with open(out_dict_path, "w") as f:
                yaml.safe_dump(serializable_dict, f)
            print(f"Saved star fraction dictionary to {out_dict_path}")
        except Exception as e:
            print(f"Failed to save star fraction dictionary: {e}")
        # --- END NEW ---
    
    fig.tight_layout()
    fig.savefig(figPath / f'{figName}_z{redshift}.{figType}', dpi=300) # type: ignore
    plt.close(fig)
    
    print('Done!!!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str, default='./configs/star_fraction.yaml', help='Path to the configuration file.')
    # parser.add_argument("--set", nargs=2, action="append",
    #                     metavar=("KEY", "VALUE"),
    #                     help="Override with dotted.key  value")
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)