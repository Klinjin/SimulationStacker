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
from stacker import SimulationStacker # type: ignore

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
    
    redshift = config['redshift']
    filterType = config['filter_type']
    plotErrorBars = config['plot_error_bars']
    loadField = config['load_field']
    saveField = config['save_field']
    radDistance = config['rad_distance']
    
    fractionType = config['fraction_type']

    haloMassUpper = config['halo_mass_upper']  # in log10(Msun/h)
    print(haloMassUpper, type(haloMassUpper))

    figPath = Path(config['fig_path'])
    figPath.mkdir(parents=False, exist_ok=True)
    
    figName = config['fig_name']
    figType = config['fig_type']
    
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
                
                radii0, profiles0 = stacker.stackMap('gas', filterType=filterType, maxRadius=6.0, # type: ignore
                                                     save=saveField, load=loadField, radDistance=radDistance, halo_mass_upper=haloMassUpper)
                radii1, profiles1 = stacker.stackMap('DM', filterType=filterType, maxRadius=6.0, 
                                                     save=saveField, load=loadField, radDistance=radDistance, halo_mass_upper=haloMassUpper)
                radii4, profiles4 = stacker.stackMap('Stars', filterType=filterType, maxRadius=6.0, 
                                                     save=saveField, load=loadField, radDistance=radDistance, halo_mass_upper=haloMassUpper)
                radii5, profiles5 = stacker.stackMap('BH', filterType=filterType, maxRadius=6.0, 
                                                     save=saveField, load=loadField, radDistance=radDistance, halo_mass_upper=haloMassUpper)


                try:
                    OmegaBaryon = stacker.header['OmegaBaryon']
                except KeyError:
                    OmegaBaryon = 0.0456  # Default value for Illustris-1

                if fractionType == 'gas':
                    fraction = profiles0 / (profiles0 + profiles1 + profiles4) / (OmegaBaryon / stacker.header['Omega0']) # OmegaBaryon = 0.048 from Planck 2015
                elif fractionType == 'baryon':
                    fraction = (profiles0 + profiles4) / (profiles0 + profiles1 + profiles4) / (OmegaBaryon / stacker.header['Omega0']) # OmegaBaryon = 0.048 from Planck 2015
                elif fractionType == 'plotall':
                    fraction = profiles0 + profiles1 + profiles4 + profiles5

                fraction_plot = np.median(fraction, axis=1)
                ax.plot(radii0 * radDistance, fraction_plot, label=sim_name, color=colours[j], lw=2)
                # ax.plot(radii0 * radDistance, profiles0.mean(axis=1), label=sim_name, color=colours[j], lw=2)
                # ax.plot(radii0 * radDistance, profiles0, label=sim_name, color=colours[j], lw=2)
                if plotErrorBars:
                    fraction_err = np.std(fraction, axis=1) / np.sqrt(fraction.shape[1])
                    upper = np.percentile(fraction, 75, axis=1)
                    lower = np.percentile(fraction, 25, axis=1)
                    ax.fill_between(radii0 * radDistance, 
                                    lower, 
                                    upper, 
                                    color=colours[j], alpha=0.2)


            elif sim_type_name == 'SIMBA':
                # SIMBA simulations have different feedback models               
                feedback = sim['feedback']

                sim_name_show = sim_name + '_' + feedback
                if verbose:
                    print(f"Processing feedback model: {feedback}")
                
                stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                            simType=sim_type_name, 
                                            feedback=feedback)
                
                radii0, profiles0 = stacker.stackMap('gas', filterType=filterType, maxRadius=6.0,  # type: ignore
                                                     save=saveField, load=loadField, radDistance=radDistance, halo_mass_upper=haloMassUpper)
                radii1, profiles1 = stacker.stackMap('DM', filterType=filterType, maxRadius=6.0, 
                                                     save=saveField, load=loadField, radDistance=radDistance, halo_mass_upper=haloMassUpper)
                radii4, profiles4 = stacker.stackMap('Stars', filterType=filterType, maxRadius=6.0, 
                                                     save=saveField, load=loadField, radDistance=radDistance, halo_mass_upper=haloMassUpper)
                radii5, profiles5 = stacker.stackMap('BH', filterType=filterType, maxRadius=6.0, 
                                                     save=saveField, load=loadField, radDistance=radDistance, halo_mass_upper=haloMassUpper)

                OmegaBaryon = 0.048  # Default value for SIMBA

                if fractionType == 'gas':
                    fraction = profiles0 / (profiles0 + profiles1 + profiles4 + profiles5) / (OmegaBaryon / stacker.header['Omega0']) # OmegaBaryon = 0.048 from Planck 2015
                elif fractionType == 'baryon':
                    fraction = (profiles0 + profiles4 + profiles5) / (profiles0 + profiles1 + profiles4 + profiles5) / (OmegaBaryon / stacker.header['Omega0']) # OmegaBaryon = 0.048 from Planck 2015
                elif fractionType == 'plotall':
                    fraction = profiles0 + profiles1 + profiles4 + profiles5

                fraction_plot = np.median(fraction, axis=1)
                ax.plot(radii0 * radDistance, fraction_plot, label=sim_name_show, color=colours[j], lw=2)
                # ax.plot(radii0 * radDistance, profiles0.mean(axis=1), label=sim_name_show, color=colours[j], lw=2)
                # ax.plot(radii0 * radDistance, profiles0, label=sim_name_show, color=colours[j], lw=2)
                if plotErrorBars:
                    fraction_err = np.std(fraction, axis=1) / np.sqrt(fraction.shape[1])
                    upper = np.percentile(fraction, 75, axis=1)
                    lower = np.percentile(fraction, 25, axis=1)
                    ax.fill_between(radii0 * radDistance, 
                                    lower, 
                                    upper, 
                                    color=colours[j], alpha=0.2)
            else:
                raise ValueError(f"Unknown simulation type: {sim_type_name}")


    # ax.set_xlabel('Radius (arcmin)')
    # ax.set_ylabel('f')
    ax.set_xlabel('R [arcmin]', fontsize=18)
    # ax.set_ylabel(r'$f_{\rm gas}(< R) / (\Omega_b/\Omega_m)$', fontsize=18)
    ax.set_ylabel(r'Cumulative Total 2D Mass', fontsize=18)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlim(0.0, 6.5)
    # ax.set_ylim(0, 1.2)
    ax.axhline(1.0, color='k', ls='--', lw=2)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True)
    ax.set_title(f'{filterType} filter at z={redshift}', fontsize=18)
    
    fig.tight_layout()
    fig.savefig(figPath / f'{figName}_{filterType}.{figType}', dpi=300) # type: ignore
    plt.close(fig)
    
    print('Done!!!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str, default='./configs/config_z05.yaml', help='Path to the configuration file.')
    # parser.add_argument("--set", nargs=2, action="append",
    #                     metavar=("KEY", "VALUE"),
    #                     help="Override with dotted.key  value")
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)