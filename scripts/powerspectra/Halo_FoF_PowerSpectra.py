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
import abacusnbody.analysis
from abacusnbody.analysis.tsc import tsc_parallel #put it on a grid using tsc interpolation
from abacusnbody.analysis.power_spectrum import calc_pk_from_deltak #computes power spectrum from density contrast, not specific to abacus 

from nbodykit.lab import *
import h5py
import hdf5plugin

sys.path.append('../src/')
from field_utils import *
from data_utils import *
from mask_utils import *

import sys
sys.path.append('../illustrisPython/')
import illustris_python as il
# JSON Parameters
sim = 'TNG300-2' # 'TNG300', 'TNG100', or 'Illustris' for simulation suite, '-1', '-2' for resolution
sim = 'TNG100-1' # 'TNG300', 'TNG100', or 'Illustris' for simulation suite, '-1', '-2' for resolution
# pType = 'gas' # particle type; 'gas' or 'DM' or 'Stars'
pType = 'DM' # particle type; 'gas' or 'DM' or 'Stars'
snapshot = 99 # Redshift snapshot; currently only 99 (z=0) or 67 (z=0.5)
nPixels = 2000 # size of the output box
nPixels = 1080 # size of the output box

n_vir = 1.0

sim = 'Illustris-1'
snapshot = 135 # Redshift snapshot; currently only 99 (z=0) or 67 (z=0.5)


# Other Parameters
# nThreads = 64
z_mock = 0.0
z_str = '0.0'
figPath = '../figures/2025-07/07-10/'
save_path =  '/pscratch/sd/r/rhliu/projects/IllustrisTNG/products/3D/'


# Get Header
basePath = '/pscratch/sd/r/rhliu/projects/IllustrisTNG/' + sim + '/output/'
# Get snapshot header:
with h5py.File(il.snapshot.snapPath(basePath, snapshot), 'r') as f:
    header = dict(f['Header'].attrs.items())

Lbox = header['BoxSize'] # kpc/h

# Simulation Parameters
h = 0.6774
kpcPerPixel = Lbox / nPixels # technically kpc/h per pixel
kNyquist = 1 / (kpcPerPixel*2) * 1000 # h/Mpc units

# Load product grids
gridPath = '/pscratch/sd/r/rhliu/projects/IllustrisTNG/products/3D/'

# field_gas = np.load(gridPath + sim + '_0'+ str(snapshot) +'_' + 'gas' +'_' + str(nPixels) + '.npy')
# field_DM = np.load(gridPath + sim + '_0'+ str(snapshot) +'_' + 'DM' +'_' + str(nPixels) + '.npy')

if pType == 'gas':
    # field = field_gas.copy()
    field = np.load(gridPath + sim + '_0'+ str(snapshot) +'_' + 'gas' +'_' + str(nPixels) + '.npy')
elif pType == 'DM':
    # field = field_DM.copy()
    field = np.load(gridPath + sim + '_0'+ str(snapshot) +'_' + 'DM' +'_' + str(nPixels) + '.npy')
else:
    raise NotImplementedError('Particle Type not Implemented')


# Load haloes
haloes = il.groupcat.loadHalos(basePath, snapshot)

GroupPos = haloes['GroupPos']
GroupMass = haloes['GroupMass'] * 1e10 / 0.6774
GroupRad = haloes['Group_R_TopHat200']

# Quick and dirty function for masking halo thresholds

def make_mass_mask(ind):
    if ind == -1:
        mass_max = 1e12
        mass_min = -1
        title_str = r'$ M_{\rm halo} < 10^{11} M_\odot$, '
    elif ind == 0:
        mass_min = 1e11 # solar masses
        mass_max = 1e12 # solar masses
        title_str = r'$1\times 10^{11} M_\odot < M_{\rm halo} < 10^{12} M_\odot$, '
    elif ind == 1:
        mass_min = 1e12 # solar masses
        mass_max = 1e13 # solar masses
        title_str = r'$1\times 10^{12} M_\odot < M_{\rm halo} < 10^{13} M_\odot$, '
    elif ind == 2:
        mass_min = 1e13 # solar masses
        mass_max = 1e14 # solar masses
        title_str = r'$1\times 10^{13} M_\odot < M_{\rm halo} < 10^{14} M_\odot$, '
    elif ind == 3:
        mass_min = 1e14 # solar masses
        mass_max = 1e19 # solar masses
        title_str = r'$M_{\rm halo} > 10^{14} M_\odot$, '
    else:
        print('Wrong ind')
    
    Mass_mask = np.where(np.logical_and((GroupMass >= mass_min), (GroupMass < mass_max)))[0]
    return Mass_mask, title_str


# Make the masks:
t0 = time.time()
halo_fields = []
title_strs = []
for i in range(4):
    Mass_mask, title_str = make_mass_mask(i)
    grid = np.zeros((nPixels, nPixels, nPixels), dtype=np.float32)
    result = tsc_paste_halos(grid, Mass_mask, basePath, snapshot, pType, Lbox)
    halo_fields.append(result)
    title_strs.append(title_str)

    print(i, time.time()-t0)

# mass_min = 1e11
# Mass_mask = np.where(GroupMass >= mass_min)[0]
# GroupPos_masked = np.round(GroupPos[Mass_mask] / kpcPerPixel).astype(int) % nPixels
# GroupRad_masked = GroupRad[Mass_mask] / kpcPerPixel
# cutout_mask = get_cutout_mask_3d(field, GroupPos_masked, GroupRad_masked * n_vir)
# cutout_masks.append(cutout_mask)
# title_strs.append(r'$M_{\rm halo} < 10^{11} M_\odot$')

# Now plot the power spectrum
kmin = 1e-2
kmax = 2 * kNyquist
kmax = 50
kbins = 30
k_bin_edges = np.logspace(np.log10(kmin), np.log10(kmax), kbins)

colourmap = matplotlib.colormaps['plasma']
colours = colourmap(np.linspace(0, 0.85, len(halo_fields)+2))

fig = plt.figure(figsize=(8,8))

for i in range(len(halo_fields)+1):
    if i==0:
        field_copy = field.copy()
        field_copy = field_copy / np.mean(field_copy) - 1
        k, power = calc_power2(field_copy, k_bin_edges, Lbox=Lbox/1000)
        plt.loglog(k, (4*np.pi*k**3) * power, label='Original', c='k')
    else:
        field_copy = halo_fields[i-1].copy()
        # field_copy[cutout_masks[i-1]] = 0.
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
plt.savefig(figPath + 'PowerSpectra_FoF_' + sim + '_' + pType + '_' + str(nPixels) +'.png', dpi=100)

print('Done!!', time.time() - t0)
