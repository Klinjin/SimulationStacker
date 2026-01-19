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
from utils import ksz_from_delta_sigma, arcmin_to_comoving, comoving_to_arcmin
from stacker import SimulationStacker
from halos import select_massive_halos
from mask_utils import get_cutout_indices_3d, sum_over_cutouts

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
    maxRadius = stack_config.get('max_radius', 10.0)
    
    nPixels = stack_config.get('n_pixels', 1000)
    nRadii = stack_config.get('num_radii', 11)

    subtract_mean = stack_config.get('subtract_mean', False)


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

    colourmaps = ['hot', 'cool']
    colourmaps = ['hsv', 'twilight']

    # Create 2x2 subplot grid with shared axes
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), sharex='col', sharey='row')
    
    # Define particle type configurations for each row
    ptype_configs = [
        {'pType': 'gas', 'pType2': 'total'},
        {'pType': 'baryon', 'pType2': 'total'}
    ]
    
    t0 = time.time()
    
    # Loop over rows (particle type configurations)
    for row_idx, ptype_config in enumerate(ptype_configs):
        pType_current = ptype_config['pType']
        pType2_current = ptype_config['pType2']
        
        print(f"\nProcessing configuration: {pType_current}/{pType2_current}")
        
        for i, sim_type in enumerate(config['simulations']):
            sim_type_name = sim_type['sim_type']
            
            colourmap = matplotlib.colormaps[colourmaps[i]] # type: ignore
            
            if sim_type_name == 'IllustrisTNG':
                TNG_sims = sim_type['sims']
                colours = colourmap(np.linspace(0.2, 0.85, len(TNG_sims)))
                ax = axes[row_idx, 0]  # Left column
            if sim_type_name == 'SIMBA':
                SIMBA_sims = sim_type['sims']
                colours = colourmap(np.linspace(0.2, 0.85, len(SIMBA_sims)))
                ax = axes[row_idx, 1]  # Right column

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
                    cosmo_tng = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'], Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K, Ob0=OmegaBaryon)

                    # radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, minRadius=minRadius, maxRadius=maxRadius, # type: ignore
                    #                                      save=saveField, load=loadField, radDistance=radDistance,
                    #                                      projection=projection)
                    # radii1, profiles1 = stacker.stackMap(pType2, filterType=filterType2, minRadius=minRadius, maxRadius=maxRadius, # type: ignore
                    #                                         save=saveField, load=loadField, radDistance=radDistance,
                    #                                         projection=projection)


                

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
                    cosmo_simba = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'], Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K, Ob0=OmegaBaryon)


                    # Now stack the maps:

                    # radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, minRadius=minRadius, maxRadius=maxRadius,  # type: ignore
                    #                                      save=saveField, load=loadField, radDistance=radDistance,
                    #                                      projection=projection)
                    # radii1, profiles1 = stacker.stackMap(pType2, filterType=filterType2, minRadius=minRadius, maxRadius=maxRadius, # type: ignore
                    #                                         save=saveField, load=loadField, radDistance=radDistance,
                    #                                         projection=projection)

                else:
                    raise ValueError(f"Unknown simulation type: {sim_type_name}")
                
                # Make Fields:
                
                field1 = stacker.makeField(pType_current, nPixels=nPixels, dim='3D', projection=projection,
                                              save=saveField, load=loadField)
                field1 = field1 - np.mean(field1) if subtract_mean else field1
                field2 = stacker.makeField(pType2_current, nPixels=nPixels, dim='3D', projection=projection,
                                              save=saveField, load=loadField)
                field2 = field2 - np.mean(field2) if subtract_mean else field2
                kpcPerPixel = stacker.header['BoxSize'] / field1.shape[0] # in comoving kpc/h

                print(f'{sim_name} kpcPerPixel: {kpcPerPixel}')
                                              
                # Halo selection:
                haloes = stacker.loadHalos(stacker.simType)
                haloMass = haloes['GroupMass']
                
                halo_mask = select_massive_halos(haloMass, 10**(13.22), 5e14) # TODO: make this configurable from user input
                # halo_mask = select_massive_halos(haloMass, 10**(13.22), None) # TODO: make this configurable from user input
                haloes['GroupMass'] = haloes['GroupMass'][halo_mask]
                haloes['GroupRad'] = haloes['GroupRad'][halo_mask] # in comoving kpc/h
                GroupPos_masked = np.round(haloes['GroupPos'][halo_mask] / kpcPerPixel).astype(int) % nPixels
                if sim_name == 'TNG100-1':
                    R200C = np.mean(haloes['GroupRad']) # in comoving kpc/h
                
                # GroupPos_masked = np.random.randint(0, nPixels, size=GroupPos_masked.shape) # For testing purposes only
                # Now stack the 3D fields:
                radii = np.linspace(minRadius, maxRadius, nRadii) # in comoving kpc/h
                profiles0 = []
                profiles1 = []
                t0 = time.time()
                for r in radii:
                    rr = np.ones(len(haloes['GroupMass'])) * r / kpcPerPixel # in comoving kpc/h
                    mask_indices = get_cutout_indices_3d(field1, GroupPos_masked, rr)
                    profiles0.append(sum_over_cutouts(field1, mask_indices.copy()))
                    profiles1.append(sum_over_cutouts(field2, mask_indices.copy()))
                    print(rr[0], time.time() - t0)
                profiles0 = np.array(profiles0)
                profiles1 = np.array(profiles1)
                print(time.time() - t0)
                            
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
                
                # If we want area-averaged CAP profile:
                # profiles0 = profiles0 / (np.pi*radii0**2)[:, np.newaxis]
                # profiles1 = profiles1 * (np.pi*radii1**2)[:, np.newaxis]
                plot_term = profiles0 / profiles1 # TODO
                # plot_term = profiles1

                # profiles_plot = np.mean(plot_term, axis=1)
                profiles_plot = np.mean(profiles0, axis=1) / np.mean(profiles1, axis=1) / (OmegaBaryon / stacker.header['Omega0'])
                # profiles_plot = np.mean(profiles0, axis=1) / np.mean(profiles1, axis=1)
                # profiles_plot = np.median(plot_term, axis=1)
                ax.plot(radii * radDistance, profiles_plot, label=sim_name, color=colours[j], lw=2, marker='o')
                if plotErrorBars:
                    err0 = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
                    err1 = np.std(profiles1, axis=1) / np.sqrt(profiles1.shape[1]) # type: ignore
                    # profiles_err = np.std(plot_term, axis=1) / np.sqrt(plot_term.shape[1])
                    profiles_err = np.abs(profiles_plot) * np.sqrt( (err0 / np.mean(profiles0, axis=1))**2 + (err1 / np.mean(profiles1, axis=1))**2 )


                    # profiles_err = np.std(plot_term, axis=1) / np.sqrt(plot_term.shape[1])
                    # upper = np.percentile(plot_term, 75, axis=1)
                    # lower = np.percentile(plot_term, 25, axis=1)
                    upper = profiles_plot + profiles_err
                    lower = profiles_plot - profiles_err
                    ax.fill_between(radii * radDistance, 
                                    lower, 
                                    upper, 
                                    color=colours[j], alpha=0.2)

    T_CMB = 2.7255
    v_c = 300000 / 299792458 # velocity over speed of light.

    def forward_arcmin(arcmin):
        return arcmin_to_comoving(arcmin, redshift, cosmo_tng)
    def inverse_arcmin(comoving):
        return comoving_to_arcmin(comoving, redshift, cosmo_tng)

    if plot_config['plot_data']:
        data_path = plot_config['data_path']
        data = np.load(data_path)
        r_data = data['theta_arcmins']
        profile_data = data['prof']
        profile_err = np.sqrt(np.diag(data['cov']))
        plt.errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k', label=plot_config['data_label'], markersize=8)

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
    
    # Configure all subplots
    for row_idx in range(2):
        pType_current = ptype_configs[row_idx]['pType']
        pType2_current = ptype_configs[row_idx]['pType2']
        
        for col_idx, title in enumerate(['IllustrisTNG', 'SIMBA']):
            ax = axes[row_idx, col_idx]
            
            # Only set x-label on bottom row
            if row_idx == 1:
                ax.set_xlabel('R [comoving kpc/h]', fontsize=18)
            
            # Only set y-label on left column
            if col_idx == 0:
                ax.set_ylabel(rf'$\frac{{{pType_current}}}{{{pType2_current}}} \; / \; (\Omega_b / \Omega_m)$', fontsize=18)
            
            # # Add secondary y-axis only for right column (SIMBA)
            # if col_idx == 1:
            #     k = 1 / (T_CMB * v_c * 1e6)
            #     secax = ax.secondary_yaxis('right',
            #                            functions=(lambda y: y * k,
            #                                      lambda y: y / k))
            #     # Only label the top-right secondary axis to avoid clutter
            #     if row_idx == 0:
            #         secax.set_ylabel(r'$\tau_{\rm CAP} = T_{kSZ}/T_{CMB}\;\; c/v_{rms}$')
            
            ax.axvline(R200C, color='gray', ls=':', lw=2, label=r'$R_{200c}$')
            ax.axhline(1.0, color='k', ls='--', lw=2)
            ax.set_xlim(0.0, None)
            ax.legend(loc='lower right', fontsize=20)
            ax.grid(True)
            
            # Add title only to top row
            if row_idx == 0:
                ax.set_title(f'{title}', fontsize=20)
    
    # fig.suptitle(f'Profile Ratios at z={redshift}', fontsize=22, y=0.995)
    
    fig.tight_layout()
    fig.savefig(figPath / f'2x2_{figName}_z{redshift}_3D_ratio.{figType}', dpi=300) # type: ignore
    plt.close(fig)
    
    print('Done!!!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str, default='./configs/3D_mass_ratio_z05.yaml', help='Path to the configuration file.')
    # parser.add_argument("--set", nargs=2, action="append",
    #                     metavar=("KEY", "VALUE"),
    #                     help="Override with dotted.key  value")
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)