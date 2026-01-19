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
from utils import ksz_from_delta_sigma, arcmin_to_comoving
from SZstacker import SZMapStacker # type: ignore
from stacker import SimulationStacker
from halos import select_massive_halos, halo_ind
from mask_utils import get_cutout_mask_3d, get_cutout_indices_3d

sys.path.append('../../illustrisPython/')
import illustris_python as il # type: ignore

import yaml
import argparse
from pathlib import Path
from datetime import datetime


# def get_remaining_fraction_wrapper(field, Pos_array, Rad_array):
#     """
#     Most memory-efficient version - uses a set to track unique masked indices.
#     """
#     print('get cutout indices')
#     all_indices = get_cutout_indices_3d(field, Pos_array, Rad_array)
    
#     print('collect unique masked indices')
#     masked_indices = set()
#     for indices in all_indices:
#         z, y, x = indices[:, 0], indices[:, 1], indices[:, 2]
#         # Convert to tuples for set uniqueness
#         masked_indices.update(zip(z, y, x))
    
#     print('calculate masked sum')
#     total_sum = np.sum(field)
    
#     # Convert back to arrays and sum
#     if masked_indices:
#         masked_indices_array = np.array(list(masked_indices))
#         z, y, x = masked_indices_array[:, 0], masked_indices_array[:, 1], masked_indices_array[:, 2]
#         masked_sum = np.sum(field[z, y, x])
#     else:
#         masked_sum = 0.
    
#     fraction = (total_sum - masked_sum) / total_sum
#     del all_indices, masked_indices
#     return fraction

def get_remaining_fraction_wrapper(field, Pos_array, Rad_array):

    cutout_mask = get_cutout_mask_3d(field, Pos_array, Rad_array)
    
    total_sum = np.sum(field)
    masked_sum = np.sum(field[cutout_mask])
    fraction = (total_sum - masked_sum) / total_sum
    # Alternative simpler calculation:
    del cutout_mask
    return fraction

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
    
    filterType2 = stack_config.get('filter_type_2', 'DSigma')
    pType2 = stack_config.get('particle_type_2', 'total')

    minRadius = stack_config.get('min_radius', 1.0)
    maxRadius = stack_config.get('max_radius', 4.0)
    nRadii = stack_config.get('num_radii', 9)
    
    # New parameters for the 3D stacked field
    nPixels = stack_config.get('n_pixels', 1000)
    halo_mass_avg = stack_config.get('halo_mass_avg', 10**(13.22))
    halo_mass_upper = stack_config.get('halo_mass_upper', 5*10**(14))
    

    # fractionType = config['fraction_type']

    # Plotting parameters
    # get the datetime for file naming
    now = datetime.now()
    dt_string = now.strftime("%m-%d")

    figPath = Path(plot_config.get('fig_path')) / dt_string
    figPath.mkdir(parents=False, exist_ok=True)
    plotErrorBars = plot_config.get('plot_error_bars', True)
    figName = plot_config.get('fig_name', 'default_figure')
    figType = plot_config.get('fig_type', 'pdf')


    radii = np.linspace(minRadius, maxRadius, nRadii)  # in virial radii units.
    colourmaps = ['hot', 'cool']
    colourmaps = ['hsv', 'twilight']

    fig, (ax_tng, ax_simba) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)
    
    t0 = time.time()
    for i, sim_type in enumerate(config['simulations']):
        sim_type_name = sim_type['sim_type']
        
        colourmap = matplotlib.colormaps[colourmaps[i]] # type: ignore
        
        if sim_type_name == 'IllustrisTNG':
            TNG_sims = sim_type['sims']
            colours = colourmap(np.linspace(0.2, 0.85, len(TNG_sims)))
            ax = ax_tng
        if sim_type_name == 'SIMBA':
            SIMBA_sims = sim_type['sims']
            colours = colourmap(np.linspace(0.2, 0.85, len(SIMBA_sims)))
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
                # stacker_tot = SimulationStacker(sim_name, snapshot, z=redshift, 
                #                                simType=sim_type_name)
                try:
                    OmegaBaryon = stacker.header['OmegaBaryon']
                except KeyError:
                    OmegaBaryon = 0.0456  # Default value for Illustris-1
                cosmo = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'], Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K, Ob0=OmegaBaryon)                    

            elif sim_type_name == 'SIMBA':
                # SIMBA simulations have different feedback models               
                feedback = sim['feedback']

                sim_name_show = sim_name + '_' + feedback
                if verbose:
                    print(f"Processing feedback model: {feedback}")
                
                stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                       simType=sim_type_name, 
                                       feedback=feedback)
                # stacker_tot = SimulationStacker(sim_name, snapshot, z=redshift, 
                #                                simType=sim_type_name, 
                #                                feedback=feedback)
                OmegaBaryon = 0.048  # Default value for SIMBA
                cosmo = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'], Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K, Ob0=OmegaBaryon)

            else:
                raise ValueError(f"Unknown simulation type: {sim_type_name}")

            field_0 = stacker.makeField(pType, nPixels=nPixels, save=True, load=True, dim='3D')

            haloes = stacker.loadHalos(stacker.simType)
            haloMass = haloes['GroupMass']
            haloPos = haloes['GroupPos']
            halo_mask = select_massive_halos(haloMass, halo_mass_avg, halo_mass_upper)
            
            kpcPerPixel = stacker.header['BoxSize'] / nPixels # kpc/h per pixel
            
            GroupMass_masked = haloes['GroupMass'][halo_mask]
            GroupRad_masked = haloes['GroupRad'][halo_mask] / kpcPerPixel # in pixels
            GroupPos_masked = np.round(haloes['GroupPos'][halo_mask] / kpcPerPixel).astype(int)
            
            fraction_0 = np.zeros_like(radii)
            for i, r in enumerate(radii):
            
                fraction_0[i] = get_remaining_fraction_wrapper(field_0, GroupPos_masked, GroupRad_masked * r)
            
            del field_0

            field_1 = stacker.makeField(pType2, nPixels=nPixels, save=True, load=True, dim='3D')
            fraction_1 = np.zeros_like(radii)
            for i, r in enumerate(radii):
            
                fraction_1[i] = get_remaining_fraction_wrapper(field_1, GroupPos_masked, GroupRad_masked * r)
                print(i, r, fraction_0[i], fraction_1[i])


            # Now for Plotting
            
            
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


            # Plot the ratio:
            ax.plot(radii, fraction_0 / fraction_1, label=sim_name, color=colours[j], lw=2, marker='o')
            
            # If we want area-averaged CAP profile:
            # profiles0 = profiles0 / (np.pi*radii0**2)[:, np.newaxis]
            # profiles1 = profiles1 * (np.pi*radii1**2)[:, np.newaxis]
            
            # Plot profiles0 on left subplot
            # profiles_plot0 = np.mean(profiles0, axis=1)
            # ax1.plot(radii, fraction_0, label=sim_name, color=colours[j], lw=2, marker='o')
            # if plotErrorBars:
            #     err0 = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
            #     upper0 = profiles_plot0 + err0
            #     lower0 = profiles_plot0 - err0
            #     ax1.fill_between(radii0 * radDistance, lower0, upper0, color=colours[j], alpha=0.2)
            
            # Plot profiles1 on right subplot
            # profiles_plot1 = np.mean(profiles1, axis=1)
            # ax2.plot(radii, fraction_1, label=sim_name, color=colours[j], lw=2, marker='o')
            # if plotErrorBars:
            #     err1 = np.std(profiles1, axis=1) / np.sqrt(profiles1.shape[1])
            #     upper1 = profiles_plot1 + err1
            #     lower1 = profiles_plot1 - err1
            #     ax2.fill_between(radii1 * radDistance, lower1, upper1, color=colours[j], alpha=0.2)

    T_CMB = 2.7255
    v_c = 300000 / 299792458 # velocity over speed of light.

    # if plot_config['plot_data']:
    #     data_path = plot_config['data_path']
    #     data = np.load(data_path)
    #     r_data = data['theta_arcmins']
    #     profile_data = data['prof']
    #     profile_err = np.sqrt(np.diag(data['cov']))
    #     ax1.errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k', label=plot_config['data_label'], markersize=8)
    #     # Optionally also plot on ax2 if relevant

    # Configure left subplot (profiles0)
    ax_tng.set_xlabel('R [virial radii]', fontsize=18)
    # ax_tng.set_ylabel(r'$T_{kSZ}$ [$\mu K \rm{arcmin}^2$]', fontsize=18)
    # ax_tng.set_xlim(0.0, maxRadius * radDistance + 0.5)
    ax_tng.legend(loc='best', fontsize=12)
    ax_tng.grid(True)
    # ax_tng.set_title(f'{pType} {filterType} profiles at z={redshift}', fontsize=18)
    
    # Configure right subplot (profiles1)
    ax_simba.set_xlabel('R [virial radii]', fontsize=18)
    # ax_simba.set_xlim(0.0, maxRadius * radDistance + 0.5)
    ax_simba.legend(loc='best', fontsize=12)
    ax_simba.grid(True)
    # ax_simba.set_title(f'{pType2} {filterType2} profiles at z={redshift}', fontsize=18)
    
    fig.tight_layout()
    fig.savefig(figPath / f'{pType}_{pType2}_{figName}_z{redshift}_3Dremainder.{figType}', dpi=300) # type: ignore
    plt.close(fig)
    
    print('Done!!!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str, default='./configs/3D_masked_profiles_z05.yaml', help='Path to the configuration file.')
    # parser.add_argument("--set", nargs=2, action="append",
    #                     metavar=("KEY", "VALUE"),
    #                     help="Override with dotted.key  value")
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)