import argparse
import gc
from pathlib import Path
import warnings
import time
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from matplotlib.offsetbox import AnchoredText
import matplotlib.pylab as pylab

import scipy as sp
from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq
from scipy.optimize import minimize
from scipy import interpolate
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter

# from classy import Class
# import abacusnbody
import abacusnbody.analysis # type: ignore
from abacusnbody.analysis.tsc import tsc_parallel #put it on a grid using tsc interpolation # type: ignore
from abacusnbody.analysis.power_spectrum import calc_pk_from_deltak #computes power spectrum from density contrast, not specific to abacus # type: ignore

# from nbodykit.lab import * # type: ignore
import h5py
import hdf5plugin

sys.path.append('../src/')
# from field_utils import *
from mask_utils import *

# sys.path.append('../illustrisPython/')
import illustris_python as il
# import json
import yaml
import pprint
from datetime import datetime

# print('Import Parameters from JSON')
# # JSON Parameters:
# param_Dict = json.loads(sys.argv[1])
# locals().update(param_Dict)

# print('Parameters:')
# pprint.pprint(param_Dict)

'''
python -u mask_sims_3D_JSON.py '{"sim": "TNG100-1", "snapshot": 99, "nPixels":2000}'

JSON Parameters
sim = 'TNG300-2' # 'TNG300', 'TNG100', or 'Illustris' for simulation suite, '-1', '-2' or '-3' for resolution
pType = 'DM' # particle type; 'gas' or 'DM' or 'Stars'
snapshot = 99 # Redshift snapshot; currently only 99 (z=0) or 67 (z=0.5) for TNG or 135 (z=0) for Illustris
nPixels = 1080 # size of the output box
n_vir = 1.0 # numner of radii to do the stacking
'''

parser = argparse.ArgumentParser(description='Process config.')
parser.add_argument('-p', '--path2config', type=str, default='./configs/mask_fraction.yaml', help='Path to the configuration file.')
args = vars(parser.parse_args())
print("Using config file:", args['path2config'])

with open(args['path2config'], 'r') as f:
    config = yaml.safe_load(f)

# sim = 'Illustris-1'
# snapshot = 135 # Redshift snapshot; currently only 99 (z=0) or 67 (z=0.5)

sim = config.get('sim', 'TNG100-1')
snapshot = config.get('snapshot', 99)
nPixels = config.get('nPixels', 1000)

save_field = config.get('save_field', False)
save_profile = config.get('save_profile', False)

# Other Parameters
# nThreads = 64
z_mock = 0.0
z_str = '0.0'


now = datetime.now()
dt_string = now.strftime("%m-%d")

figPath = '../figures/2025-11/'
figPath = Path(figPath) / dt_string
figPath.mkdir(parents=False, exist_ok=True)

save_path =  '/pscratch/sd/r/rhliu/projects/IllustrisTNG/products/3D/'
products_save_path = '/pscratch/sd/r/rhliu/projects/Unbound_Gas/'


# Get Header
basePath = '/pscratch/sd/r/rhliu/simulations/IllustrisTNG/' + sim + '/output/'
# Get snapshot header:
with h5py.File(il.snapshot.snapPath(basePath, snapshot), 'r') as f:
    header = dict(f['Header'].attrs.items())

Lbox = header['BoxSize'] # kpc/h

# Simulation Parameters
h = 0.6774
kpcPerPixel = Lbox / nPixels # technically kpc/h per pixel
kNyquist = 1 / (kpcPerPixel*2) * 1000 # h/Mpc units

# Load product grids
gridPath = '/pscratch/sd/r/rhliu/simulations/IllustrisTNG/products/3D/'

field_gas = np.load(gridPath + sim + '_0'+ str(snapshot) +'_' + 'gas' +'_' + str(nPixels) + '.npy')
field_DM = np.load(gridPath + sim + '_0'+ str(snapshot) +'_' + 'DM' +'_' + str(nPixels) + '.npy')

# if pType == 'gas':
#     # field = field_gas.copy()
#     field = np.load(gridPath + sim + '_0'+ str(snapshot) +'_' + 'gas' +'_' + str(nPixels) + '.npy')
# elif pType == 'DM':
#     # field = field_DM.copy()
#     field = np.load(gridPath + sim + '_0'+ str(snapshot) +'_' + 'DM' +'_' + str(nPixels) + '.npy')
# else:
#     raise NotImplementedError('Particle Type not Implemented')


# Load haloes
haloes = il.groupcat.loadHalos(basePath, snapshot)

GroupPos = haloes['GroupPos']
GroupMass = haloes['GroupMass'] * 1e10 / 0.6774
GroupRad = haloes['Group_R_TopHat200']

# Quick and dirty function for masking halo thresholds
'''
def make_mass_mask(ind):
    if ind == -1:
        mass_max = 1e12
        mass_min = -1
        title_str = r'$ M_{\rm halo} < 10^{11} M_\odot$, ' # type: ignore
    elif ind == 0:
        mass_min = 1e11 # solar masses
        mass_max = 1e12 # solar masses
        title_str = r'$1\times 10^{11} M_\odot < M_{\rm halo} < 10^{12} M_\odot$, ' # type: ignore
    elif ind == 1:
        mass_min = 1e12 # solar masses
        mass_max = 1e13 # solar masses
        title_str = r'$1\times 10^{12} M_\odot < M_{\rm halo} < 10^{13} M_\odot$, ' # type: ignore
    elif ind == 2:
        mass_min = 1e13 # solar masses
        mass_max = 1e14 # solar masses
        title_str = r'$1\times 10^{13} M_\odot < M_{\rm halo} < 10^{14} M_\odot$, ' # type: ignore
    elif ind == 3:
        mass_min = 1e14 # solar masses
        mass_max = 1e19 # solar masses
        title_str = r'$M_{\rm halo} > 10^{14} M_\odot$, ' # type: ignore
    else:
        print('Wrong ind')
    
    Mass_mask = np.where(np.logical_and((GroupMass >= mass_min), (GroupMass < mass_max)))[0]
    return Mass_mask, title_str
'''

# Now we make the masks:
t0 = time.time()
mass_thresholds = [1e13, 1e12, 1e11]
mass_strs = ['1e13', '1e12', '1e11']
n_virs = np.linspace(0, 4, 9)

def get_remaining_fraction_wrapper(field, Pos_array, Rad_array):
    field2 = field.copy()
    cutout_mask = get_cutout_mask_3d(field, Pos_array, Rad_array)
    field2[cutout_mask] = 0.
    fraction = np.sum(field2) / np.sum(field)
    del field2, cutout_mask
    return fraction

fig, ax = plt.subplots(1, 2, figsize=(10,8), sharey=True)

t0 = time.time()
for j, mass_threshold in enumerate(mass_thresholds):

    mass_str = mass_strs[j]
    
    Mass_mask = GroupMass >= mass_threshold
    GroupPos_masked = np.round(GroupPos[Mass_mask] / kpcPerPixel).astype(int)
    GroupRad_masked = GroupRad[Mass_mask] / kpcPerPixel
    print(GroupRad_masked.shape)
    
    # Now we make the masks:
    
    # t0 = time.time()
    # cutout_masks = []
    # for i in range(3):
    #     cutout_mask = get_cutout_mask_3d(field, GroupPos_masked, GroupRad_masked * (i+1))
    #     # save_str = save_path + 'MTNG_mask_' + mass_str + '_z_' + z_str + '_' + str(i+1)
    #     # np.save(save_str, cutout_mask)
    #     cutout_masks.append(cutout_mask)
    #     print(i, time.time()-t0)

    # Now we make the plots:
    fractions_gas = []
    fractions_DM = []
    for n in n_virs:
        if n==0:
            fractions_gas.append(1.)
            fractions_DM.append(1.)
        else:
            fraction_gas = get_remaining_fraction_wrapper(field_gas, GroupPos_masked, GroupRad_masked * n)
            fraction_DM = get_remaining_fraction_wrapper(field_DM, GroupPos_masked, GroupRad_masked * n)
            fractions_gas.append(fraction_gas)
            fractions_DM.append(fraction_DM)
            print(n, time.time()-t0)

            
    fractions_gas = np.array(fractions_gas)
    fractions_DM = np.array(fractions_DM)

    ax[0].plot(n_virs, fractions_gas, label=r'$M_{\rm Halo}\geq$ ' + mass_str + r' $M_{\odot}$')
    ax[1].plot(n_virs, fractions_DM, label=r'$M_{\rm Halo}\geq$ ' + mass_str + r' $M_{\odot}$')
    # plt.scatter(n_virs, fractions_gas/fractions_DM, label=r'$M_{\rm Halo}\geq$ ' + mass_str + r' $M_{\odot}$')

ax[0].set_title('Gas', fontsize=15)
ax[1].set_title('DM', fontsize=15)
# title_axi0 = AnchoredText('Gas', prop=dict(fontsize=13), loc=1, borderpad=0, frameon=True)
# ax[0].add_artist(title_axi0)  
ax[0].grid()

# title_axi1 = AnchoredText('Gas', prop=dict(fontsize=13), loc=1, borderpad=0, frameon=True)
# ax[1].add_artist(title_axi1)  
ax[1].grid()

# plt.yscale('log')
ax[0].set_xlabel('$N$ virial radii masked', fontsize=15)
ax[1].set_xlabel('$N$ virial radii masked', fontsize=15)
ax[0].set_ylabel('Remaining mass fraction', fontsize=15)
plt.legend(fontsize=15)
fig.suptitle('{} Simulation Box, snapshot {}'.format(sim, snapshot) 
            #  + r'  $\frac{ \mathrm{Gas\,\, remaining}}{\rm DM\,\, remaining}$'
             , fontsize=20)
plt.tight_layout()
# plt.show()
plt.savefig(figPath + sim + '_Gas_DM_fractions' + '.png', dpi=80)

if save_profile:
    save_dict = {}
    save_dict['gas_fraction'] = fractions_gas
    save_dict['DM_fraction'] = fractions_DM
    save_dict['n_virs'] = n_virs

    np.savez(products_save_path + sim + 'fractions.npz', **save_dict)


'''

cutout_masks = []
title_strs = []

# Make the masks:
t0 = time.time()
cutout_masks = []
title_strs = []
for i in range(4):
    Mass_mask, title_str = make_mass_mask(i)
    GroupPos_masked = np.round(GroupPos[Mass_mask] / kpcPerPixel).astype(int) % nPixels
    GroupRad_masked = GroupRad[Mass_mask] / kpcPerPixel
    cutout_mask = get_cutout_mask_3d(field, GroupPos_masked, GroupRad_masked * n_vir)
    cutout_masks.append(np.logical_not(cutout_mask))
    title_strs.append(title_str)

    print(i, time.time()-t0)

mass_min = 1e11
Mass_mask = np.where(GroupMass >= mass_min)[0]
GroupPos_masked = np.round(GroupPos[Mass_mask] / kpcPerPixel).astype(int) % nPixels
GroupRad_masked = GroupRad[Mass_mask] / kpcPerPixel
cutout_mask = get_cutout_mask_3d(field, GroupPos_masked, GroupRad_masked * n_vir)
cutout_masks.append(cutout_mask)
title_strs.append(r'$M_{\rm halo} < 10^{11} M_\odot$')

# Now plot the power spectrum
kmin = 1e-2
kmax = 2 * kNyquist
kmax = 50
kbins = 30
k_bin_edges = np.logspace(np.log10(kmin), np.log10(kmax), kbins)

colourmap = matplotlib.colormaps['plasma']
colours = colourmap(np.linspace(0, 0.85, len(cutout_masks)+2))

fig = plt.figure(figsize=(8,8))

for i in range(len(cutout_masks)+1):
    field_copy = field.copy()
    if i==0:
        field_copy = field_copy / np.mean(field_copy) - 1
        k, power = calc_power2(field_copy, k_bin_edges, Lbox=Lbox/1000)
        plt.loglog(k, (4*np.pi*k**3) * power, label='Original', c='k')
    else:
        # field_copy[cutout_masks[i-1]] = 0.
        field_copy[cutout_masks[i-1]] = field_copy.mean()
        field_copy = field_copy / np.mean(field_copy) - 1
        
        k, power = calc_power2(field_copy, k_bin_edges, Lbox=Lbox/1000)
        plt.loglog(k, (4*np.pi*k**3) * power, label=title_strs[i-1], c=colours[i])
        if i==1:
            power2 = power.copy()
        else:
            power2 = power2 + power

    print(i, time.time()-t0)

# for i in range(5):
#     field_copy = field.copy()
#     if i==0:
#         continue
#     elif i==1:
#         field_copy[cutout_masks[i-1]] = 0.
#         k, power = calc_power2(field_copy, k_bin_edges, Lbox=Lbox/1000)
#     else:
#         field_copy[cutout_masks[i-1]] = 0.
#         k2, power2 = calc_power2(field_copy, k_bin_edges, Lbox=Lbox/1000)
#         power = power + power2
        
plt.loglog(k, (4*np.pi*k**3) * power2, label='Sum', c=colours[-1])

plt.axvline(kNyquist, c='k', alpha=0.5, linestyle='-.')
plt.xlabel(r'$k$')
plt.ylabel(r'$4\pi k^3 P(k)$')

# fig.suptitle('{} Simulation Box, pType {}, snapshot {} \n (replace masked regions with field mean)'.format(sim, pType, snapshot), fontsize=15)
fig.suptitle('{} Simulation Box, pType {}, snapshot {}'.format(sim, pType, snapshot), fontsize=15)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(figPath + 'PowerSpectra_mean_' + sim + '_' + pType + '_' + str(nPixels) +'.png', dpi=100)
'''
print('Done!!', time.time() - t0)
