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
from SZstacker import SZMapStacker # type: ignore
from stacker import SimulationStacker
from utils import fft_smoothed_map, comoving_to_arcmin

sys.path.append('../../illustrisPython/')
import illustris_python as il # type: ignore

import yaml
import argparse
from pathlib import Path

def test_filter_on_array(stacker, array, filterFunc, minRadius=0.1, maxRadius=4.5, numRadii=25,
                        projection='xy', radDistance=1000.0, radDistanceUnits='kpc/h', 
                        halo_mass_avg=10**(13.22), halo_mass_upper=5*10**(14), z=None, 
                        pixelSize=0.5, filter_kwargs=None):
    """
    Standalone function to test different filters on a 2D array without modifying stacker classes.
    
    Args:
        stacker: SimulationStacker or SZMapStacker instance (for accessing halos and metadata)
        array (np.ndarray): 2D array to stack on. Shape must be (nPixels, nPixels).
        filterFunc: Filter function to apply. Should have signature filterFunc(cutout, rr, radius, **kwargs).
        minRadius (float): Minimum radius for stacking.
        maxRadius (float): Maximum radius for stacking.
        numRadii (int): Number of radial bins.
        projection (str): Direction projection ('xy', 'xz', or 'yz').
        radDistance (float): Radial distance units for stacking.
        radDistanceUnits (str): Units for radDistance ('kpc/h' or 'arcmin').
        halo_mass_avg (float): Average halo mass for selection.
        halo_mass_upper (float): Upper mass bound for selection.
        z (float, optional): Redshift for angular distance calculation.
        pixelSize (float): Pixel size in arcminutes.
        filter_kwargs (dict, optional): Additional keyword arguments to pass to filterFunc.
    
    Returns:
        tuple: (radii, profiles) - 1D radii array and 2D profiles array.
    """
    
    if filter_kwargs is None:
        filter_kwargs = {}
    
    nPixels = array.shape[0]
    assert array.shape == (nPixels, nPixels), f"Array must be square, got shape: {array.shape}"

    # Load the halo catalog and select halos
    haloes = stacker.loadHalos(stacker.simType)
    haloMass = haloes['GroupMass']
    haloPos = haloes['GroupPos']
    
    from halos import select_massive_halos
    halo_mask = select_massive_halos(haloMass, halo_mass_avg, halo_mass_upper)
    
    print(f'Number of halos selected: {halo_mask.shape[0]}')
    
    # Convert radDistance to pixels based on units
    if radDistanceUnits == 'kpc/h':
        kpcPerPixel = stacker.header['BoxSize'] / nPixels
        RadPixel = radDistance / kpcPerPixel
    elif radDistanceUnits == 'arcmin':
        if z is None:
            z = stacker.z
        cosmo = stacker.cosmo
        theta_arcmin = comoving_to_arcmin(stacker.header['BoxSize'], z, cosmo=cosmo)
        arcminPerPixel = theta_arcmin / nPixels
        RadPixel = radDistance / arcminPerPixel
    else:
        raise ValueError(f"radDistanceUnits must be 'kpc/h' or 'arcmin', got: {radDistanceUnits}")
    
    # Set up radial bins and cutout size
    radii = np.linspace(minRadius, maxRadius, numRadii)
    n_vir = int(radii.max() + 1)

    # Do stacking
    profiles = []
    for j, haloID in enumerate(halo_mask):
        # Get halo position for the specified projection
        if projection == 'xy':
            haloPos_2D = haloPos[haloID, :2]
        elif projection == 'xz':
            haloPos_2D = haloPos[haloID, [0, 2]]
        elif projection == 'yz':
            haloPos_2D = haloPos[haloID, 1:]
        else:
            raise NotImplementedError('Projection type not implemented: ' + projection)
        
        # Convert halo position to pixel coordinates
        haloLoc = np.round(haloPos_2D / (stacker.header['BoxSize'] / nPixels)).astype(int)
        
        # Create cutout and radial distance grid
        cutout = SimulationStacker.cutout_2d_periodic(array, haloLoc, n_vir*RadPixel)
        rr = SimulationStacker.radial_distance_grid(cutout, (-n_vir, n_vir))
        
        # Apply filter at each radius
        profile = []
        for rad in radii:
            filt_result = filterFunc(cutout, rr, rad, **filter_kwargs)
            profile.append(filt_result)
        
        profile = np.array(profile)
        profiles.append(profile)
        
    profiles = np.array(profiles).T
    
    return radii, profiles

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

    maskHaloes = stack_config.get('mask_haloes', False)
    maskRadii = stack_config.get('mask_radii', 2.0) # in virial radii
    
    pixelSize = stack_config.get('pixel_size', 0.5) # in arcmin

    # fractionType = config['fraction_type']

    # Plotting parameters
    figPath = Path(plot_config.get('fig_path'))
    figPath.mkdir(parents=False, exist_ok=True)
    plotErrorBars = plot_config.get('plot_error_bars', True)
    figName = plot_config.get('fig_name', 'default_figure')
    figType = plot_config.get('fig_type', 'pdf')
    
    maskHaloes = config.get('mask_haloes', False)
    maskRadii = config.get('mask_radii', 2.0) # in
    
    colourmaps = ['hot', 'cool']

    # Create a figure with subplots for each filter type
    from filters import delta_sigma, delta_sigma_ring, delta_sigma_kernel_map, CAP, delta_sigma_kernel
    
    filter_dict = {
        # 'delta_sigma': delta_sigma,
        # 'delta_sigma_ring': delta_sigma_ring,
        'delta_sigma_compensated': delta_sigma_kernel_map,
        'CAP': CAP,
        'delta_sigma_kernel': delta_sigma_kernel,
    }

    fig, axes = plt.subplots(1, len(filter_dict), figsize=(len(filter_dict.keys())*6, 6), sharey=True)
    if len(filter_dict) == 1:
        axes = [axes]
    
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
                
                stacker = SZMapStacker(sim_name, snapshot, z=redshift, 
                                       simType=sim_type_name)

                # Load or create the map once
                fieldKey = (pType, redshift, projection, 0.5)
                if not hasattr(stacker, 'maps'):
                    stacker.maps = {}
                if fieldKey not in stacker.maps:
                    stacker.maps[fieldKey] = stacker.makeMap(pType, z=redshift, projection=projection,
                                                             save=saveField, load=loadField, pixelSize=0.5, 
                                                             mask=maskHaloes, maskRad=maskRadii)
                
                map_data = stacker.maps[fieldKey]
                
                # Test each filter
                for ax_idx, (filter_name, filter_func) in enumerate(filter_dict.items()):
                    print(f"Testing filter: {filter_name}")
                    
                    # Set filter-specific kwargs
                    filter_kwargs = {}
                    if filter_name == 'delta_sigma':
                        filter_kwargs = {'dr': 0.6, 'pixel_size': 1.0}
                    elif filter_name == 'delta_sigma_ring':
                        filter_kwargs = {'pixel_size': 0.5, 'connectivity': 8}
                    elif filter_name == 'delta_sigma_compensated':
                        filter_kwargs = {'dr': 0.6, 'pixel_size': 1.0}
                    elif filter_name == 'delta_sigma_kernel' or filter_name == 'delta_sigma_kernel_map':
                        filter_kwargs = {'dr': 0.6, 'pixel_size': 0.5}
                    elif filter_name == 'CAP':
                        filter_kwargs = {}
                    
                    radii0, profiles0 = test_filter_on_array(
                        stacker=stacker,
                        array=map_data,
                        filterFunc=filter_func,
                        minRadius=0.5,
                        maxRadius=6.0,
                        numRadii=11,
                        projection=projection,
                        radDistance=radDistance,
                        radDistanceUnits='arcmin',
                        z=redshift,
                        pixelSize=0.5,
                        filter_kwargs=filter_kwargs
                    )
                    
                    # Plot on the corresponding subplot
                    profiles_plot = np.mean(profiles0, axis=1)

                    if filter_name == 'CAP':
                        profiles_plot /= (radii0**2 * np.pi)  # Invert CAP results for plotting
                    
                    axes[ax_idx].plot(radii0 * radDistance, profiles_plot, label=sim_name, 
                                     color=colours[j], lw=2, marker='o')
                    if plotErrorBars:
                        upper = np.percentile(profiles0, 75, axis=1) / (radii0**2 * np.pi) if filter_name == 'CAP' else np.percentile(profiles0, 75, axis=1)
                        lower = np.percentile(profiles0, 25, axis=1) / (radii0**2 * np.pi) if filter_name == 'CAP' else np.percentile(profiles0, 25, axis=1)
                        axes[ax_idx].fill_between(radii0 * radDistance, lower, upper,
                                                  color=colours[j], alpha=0.2)

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
                
                stacker = SZMapStacker(sim_name, snapshot, z=redshift,
                                       simType=sim_type_name, 
                                       feedback=feedback)
                
                # Load or create the map once
                fieldKey = (pType, redshift, projection, 0.5)
                if not hasattr(stacker, 'maps'):
                    stacker.maps = {}
                if fieldKey not in stacker.maps:
                    stacker.maps[fieldKey] = stacker.makeMap(pType, z=redshift, projection=projection,
                                                             save=saveField, load=loadField, pixelSize=0.5,
                                                             mask=maskHaloes, maskRad=maskRadii)
                                                             
                
                map_data = stacker.maps[fieldKey]
                
                # Test each filter
                for ax_idx, (filter_name, filter_func) in enumerate(filter_dict.items()):
                    print(f"Testing filter: {filter_name} on {sim_name_show}")
                    
                    # Set filter-specific kwargs
                    filter_kwargs = {}
                    if filter_name == 'delta_sigma':
                        filter_kwargs = {'dr': 0.6, 'pixel_size': 1.0}
                    elif filter_name == 'delta_sigma_ring':
                        filter_kwargs = {'pixel_size': 0.5, 'connectivity': 8}
                    elif filter_name == 'delta_sigma_compensated':
                        filter_kwargs = {'dr': 0.6, 'pixel_size': 1.0}
                    elif filter_name == 'delta_sigma_kernel' or filter_name == 'delta_sigma_kernel_map':
                        filter_kwargs = {'dr': 0.6, 'pixel_size': 0.5}
                    elif filter_name == 'CAP':
                        filter_kwargs = {}
                                        
                    radii0, profiles0 = test_filter_on_array(
                        stacker=stacker,
                        array=map_data,
                        filterFunc=filter_func,
                        minRadius=0.5,
                        maxRadius=6.0,
                        numRadii=11,
                        projection=projection,
                        radDistance=radDistance,
                        radDistanceUnits='arcmin',
                        z=redshift,
                        pixelSize=0.5,
                        filter_kwargs=filter_kwargs
                    )
                    
                    profiles_plot = np.mean(profiles0, axis=1)

                    if filter_name == 'CAP':
                        profiles_plot /= (radii0**2 * np.pi)
                    
                    axes[ax_idx].plot(radii0 * radDistance, profiles_plot, label=sim_name_show, 
                                     color=colours[j], lw=2, marker='o')
                    if plotErrorBars:
                        upper = np.percentile(profiles0, 75, axis=1) / (radii0**2 * np.pi) if filter_name == 'CAP' else np.percentile(profiles0, 75, axis=1)
                        lower = np.percentile(profiles0, 25, axis=1) / (radii0**2 * np.pi) if filter_name == 'CAP' else np.percentile(profiles0, 25, axis=1)
                        axes[ax_idx].fill_between(radii0 * radDistance, lower, upper,
                                                  color=colours[j], alpha=0.2)
                
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

    T_CMB = 2.7255
    v_c = 300000 / 299792458 # velocity over speed of light.

    # if plot_config['plot_data']:
    if False:
        data_path = plot_config['data_path']
        data = np.load(data_path)
        r_data = data['theta_arcmins']
        # profile_data = data['prof']
        # profile_err = np.sqrt(np.diag(data['cov']))

        profile_data = data['signal']
        profile_err = data['noise']
        
        for ax in axes:
            ax.errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k', 
                       label=plot_config['data_label'], markersize=8)

    # Configure each subplot
    for ax_idx, (filter_name, _) in enumerate(filter_dict.items()):
        axes[ax_idx].set_xlabel('R [arcmin]', fontsize=14)
        axes[ax_idx].set_ylabel(r'$\Delta\Sigma$ [arbitrary units]', fontsize=14)
        axes[ax_idx].set_yscale('log')
        axes[ax_idx].set_xlim(0.0, 6.5)
        axes[ax_idx].legend(loc='lower right', fontsize=10)
        axes[ax_idx].grid(True)
        axes[ax_idx].set_title(f'{filter_name} at z={redshift}', fontsize=14)
    
    fig.tight_layout()
    fig.savefig(figPath / f'{figName}_{pType}_z{redshift}_filter_comparison.{figType}', dpi=300)
    plt.close(fig)
    
    print('Done!!!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str, default='./configs/tau_z05_CAP.yaml', help='Path to the configuration file.')
    # parser.add_argument("--set", nargs=2, action="append",
    #                     metavar=("KEY", "VALUE"),
    #                     help="Override with dotted.key  value")
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)