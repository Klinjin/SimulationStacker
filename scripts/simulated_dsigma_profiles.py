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
import astropy.constants as const
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
from astropy.table import Table
from copy import deepcopy


def mass_to_temp(mass_density, z, cosmology=Planck18):
    """Convert mass profile units to kSZ temperature fluctuation units

    Args:
        mass_density (ndarray): Mass density in Msun/area. Area does not change, so can be arcmin^2 or kpc^2.
        z (float): Redshift.
        delta_sigma_is_comoving (bool, optional): If True, delta_sigma is in comoving units. Defaults to True.
        cosmology (FlatLambdaCDM, optional): Cosmology object. Required if delta_sigma_is_comoving is True.

    Returns:
        ndarray: kSZ temperature fluctuation in micro-Kelvin.
    """
    mu_e = 1.14  # Mean molecular weight per free electron, assuming primordial composition
    T_CMB = 2.7255 * u.K
    c = 299792458 * u.m / u.s
    v_rms = 300000 * u.m / u.s  # Example velocity, adjust as needed
    Omega_b = cosmology.Ob0
    Omega_m = cosmology.Om0
    
    # Constant gas fraction
    f_b = Omega_b / Omega_m

    # Electron column and optical depth
    Sigma_gas = f_b * mass_density                               # kg/m^2
    N_e = (Sigma_gas / (mu_e * const.m_p)).value #.to(1 / u.m**2)         # 1/m^2 # type: ignore
    tau = (const.sigma_T * N_e).decompose().value                 # dimensionless # type: ignore

    factor = const.sigma_T.value / (mu_e * const.m_p)  # 1/kg # type: ignore
    
    # if delta_sigma_is_comoving:
    #     if cosmology is None:
    #         raise ValueError("Cosmology must be provided if delta_sigma_is_comoving is True.")
    #     E_z = cosmology.efunc(z)
    #     factor = (1 + z)**2 * E_z
    # else:
    #     factor = 1.0
    print(factor)
    print(mass_density)
    # print(T_CMB * (v_rms / c) * mass_density * factor)
    kSZ_temp = (T_CMB * (v_rms / c) * mass_density.to(u.kg) * factor).to(u.uK).value
    #.to(u.microkelvin, equivalencies=u.temperature_energy())
    return kSZ_temp

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
    # radDistance = 1000.0 # convert from Mpc to kpc
    pType = stack_config.get('particle_type', 'tau')
    projection = stack_config.get('projection', 'xy')
    pixelSize = stack_config.get('pixel_size', 0.5)
    beamSize = stack_config.get('beam_size', None)

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
    colourmaps = ['hsv', 'twilight']

    star_fraction_dict_path = '../figures/2026-01/01-26/star_fraction_z0.5_star_fraction.yaml'
    load_path_obj = Path(star_fraction_dict_path)
    with open(load_path_obj, 'r') as f:
        star_fraction_dict = yaml.safe_load(f)

    # fig, ax = plt.subplots(figsize=(10,8))
    fig, (ax_tng, ax_simba) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
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
                sim_name_show = sim_name
                
                stacker = SimulationStacker(sim_name, snapshot, z=redshift, 
                                            simType=sim_type_name)
                
                try:
                    OmegaBaryon = stacker.header['OmegaBaryon']
                except KeyError:
                    OmegaBaryon = 0.0456  # Default value for Illustris-1
                
                cosmo = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'], Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K, Ob0=OmegaBaryon)                    
                

            elif sim_type_name == 'SIMBA':
                # SIMBA simulations have different feedback models               
                feedback = sim['feedback']
                OmegaBaryon = 0.048  # Default value for SIMBA

                sim_name_show = sim_name + '_' + feedback
                if verbose:
                    print(f"Processing feedback model: {feedback}")
                
                stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                            simType=sim_type_name, 
                                            feedback=feedback)
                cosmo = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'], Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K, Ob0=OmegaBaryon)
                
                
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

            # Now we do the stacking after configuring the stacker
            # TEST!!! making a map without beam smoothing.
            # map_ = stacker.makeMap(pType, projection=projection, save=False, load=False, beamsize=None) # type: ignore
            # map_ = stacker.makeMap(pType, projection=projection, save=saveField, load=loadField) # type: ignore
            # map_ = map_ / (1 - star_fraction_dict[sim_name_show])
            # stacker.setMap(pType, map_, z=redshift)
            
            # radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, minRadius=minRadius, 
            #                                      maxRadius=maxRadius, numRadii=nRadii, # type: ignore
            #                                      save=saveField, load=loadField, radDistance=radDistance,
            #                                      projection=projection)

            radii1, profiles1 = stacker.stackMap(pType, filterType=filterType, minRadius=minRadius, 
                                                 maxRadius=maxRadius, numRadii=nRadii, 
                                                 pixelSize=pixelSize, beamSize=beamSize,
                                                 save=saveField, load=loadField, radDistance=radDistance,
                                                 projection=projection)
            # profiles1 = mass_to_temp(profiles1 * u.Msun, z=redshift, cosmology=cosmo) # convert to kSZ
                                                    
            # minRad_mpch = arcmin_to_comoving(minRadius, redshift, cosmo) / 1000.0
            # maxRad_mpch = arcmin_to_comoving(maxRadius, redshift, cosmo) / 1000.0
            
            # theta_arcmin = comoving_to_arcmin(stacker.header['BoxSize'], redshift, cosmo=cosmo)
            # # pixelSize = 0.2
            # nPixels = np.ceil(theta_arcmin / pixelSize).astype(int)
            # print(f"theta_arcmin: {theta_arcmin}, nPixels: {nPixels}")

            # # print(f"minRad_mpch: {minRad_mpch}, maxRad_mpch: {maxRad_mpch}")
            # radii1, profiles1 = stacker.stackField(pType, filterType=filterType, minRadius=minRad_mpch, 
            #                                        maxRadius=maxRad_mpch, numRadii=nRadii, # type: ignore
            #                                        save=saveField, load=loadField, radDistance=radDistance, nPixels=nPixels,
            #                                        projection=projection)

            h = stacker.header['HubbleParam']
            # profiles1 = ksz_from_delta_sigma(profiles1 * u.Msun / u.kpc**2 * h**2, redshift, delta_sigma_is_comoving=True, cosmology=cosmo) # convert to kSZ
            # profiles1 = ksz_from_delta_sigma(profiles1 * u.Msun / u.kpc**2 * h, redshift, delta_sigma_is_comoving=True, cosmology=cosmo) # convert to kSZ
            # profiles1 = ksz_from_delta_sigma(profiles1 * u.Msun / u.kpc**2 * h, redshift, delta_sigma_is_comoving=False, cosmology=cosmo) # convert to kSZ
            # profiles1 = -1.0 * profiles1 # negative sign since kSZ from delta_sigma has a negative sign. # type: ignore
            # profiles1 = np.abs(profiles1) # take absolute value, since some profiles are negative.

            
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
            # plot_term = profiles0 / profiles1 # TODO
            # plot_term = profiles1

            # profiles_plot = np.mean(plot_term, axis=1)
            # profiles_plot = np.mean(profiles0, axis=1) / np.mean(profiles1, axis=1) / (OmegaBaryon / stacker.header['Omega0'])
            # profiles_plot = np.mean(profiles0, axis=1) / np.mean(profiles1, axis=1)
            profiles_plot = np.mean(profiles1, axis=1)
            # profiles_plot = np.median(plot_term, axis=1)
            ax.plot(radii1 * radDistance, profiles_plot, label=sim_name, color=colours[j], lw=2, marker='o')
            if plotErrorBars:
                # err0 = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
                err1 = np.std(profiles1, axis=1) / np.sqrt(profiles1.shape[1])
                # profiles_err = np.std(plot_term, axis=1) / np.sqrt(plot_term.shape[1])
                # profiles_err = np.abs(profiles_plot) * np.sqrt( (err0 / np.mean(profiles0, axis=1))**2 + (err1 / np.mean(profiles1, axis=1))**2 )
                profiles_err = err1
                
                
                # profiles_err = np.std(plot_term, axis=1) / np.sqrt(plot_term.shape[1])
                # upper = np.percentile(plot_term, 75, axis=1)
                # lower = np.percentile(plot_term, 25, axis=1)
                upper = profiles_plot + profiles_err
                lower = profiles_plot - profiles_err
                ax.fill_between(radii1 * radDistance, 
                                lower, 
                                upper, 
                                color=colours[j], alpha=0.2)

    T_CMB = 2.7255
    v_c = 300000 / 299792458 # velocity over speed of light.

    if plot_config['plot_data']:
        # TODO: Check that the distance conversion here (without h) is valid!
        data_path = plot_config['data_path']

        data = Table.read(data_path) # type: ignore
        # r_data = data['theta_arcmins']
        # profile_data = data['prof']
        # profile_err = data['prof_err']
        r_data = data['rp'] * radDistance# / 0.6774 # in kpc
        profile_data = deepcopy(data['ds']) * u.Msun / u.pc**2 * (radDistance)**2 # in Msun/arcmin^2
        ds_measurement_cov = data['cov'] * (u.Msun / u.pc**2)**2 # in uK
        profile_err = np.sqrt(np.diag(ds_measurement_cov)) * (radDistance)**2
        print(r_data)
        print(profile_data)
        # data = np.load(data_path)
        # r_data = data['theta_arcmins']
        # profile_data = data['prof']
        # profile_err = np.sqrt(np.diag(data['cov']))
        ax_tng.errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k', label=plot_config['data_label'], markersize=8)
        ax_simba.errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k', label=plot_config['data_label'], markersize=8)

    # ax.set_xlabel('Radius (arcmin)')
    # ax.set_ylabel('f')
    ax_tng.set_xlabel('R [kpc/h]', fontsize=18)
    ax_simba.set_xlabel('R [kpc/h]', fontsize=18)
    # ax_tng.set_ylabel(r'$\frac{T_{kSZ} / \pi R^2}{\Delta \Sigma}$', fontsize=18)
    ax_tng.set_ylabel(rf'$\Delta \Sigma$({pType})', fontsize=18)
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

    # ax_tng.set_xlim(0.0, maxRadius * radDistance + 0.5)
    # ax_simba.set_xlim(0.0, maxRadius * radDistance + 0.5)
    # ax.set_ylim(0, 1.2)
    # ax.axhline(1.0, color='k', ls='--', lw=2)
    ax_tng.legend(loc='upper right', fontsize=12)
    ax_simba.legend(loc='upper right', fontsize=12)
    ax_tng.grid(True)
    ax_simba.grid(True)
    fig.suptitle(f'Ratio at z={redshift}', fontsize=18)
    
    fig.tight_layout()
    # fig.savefig(figPath / f'{figName}_{pType}_z{redshift}_ratio.{figType}', dpi=300) # type: ignore
    fig.savefig(figPath / f'{pType}_{pType2}_{figName}_z{redshift}_{filterType}_{filterType2}_ratio2.{figType}', dpi=300) # type: ignore
    plt.close(fig)
    
    print('Done!!!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str, default='./configs/dsigma_profile_z05.yaml', help='Path to the configuration file.')
    # parser.add_argument("--set", nargs=2, action="append",
    #                     metavar=("KEY", "VALUE"),
    #                     help="Override with dotted.key  value")
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)