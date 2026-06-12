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
from loadIO import load_data    
from utils import fft_smoothed_map, gaussian_smoothed_map

# sys.path.append('../../illustrisPython/')
import illustris_python as il # type: ignore

import yaml
import argparse
from pathlib import Path
import glob
from pathlib import Path
from datetime import datetime
from copy import deepcopy

now = datetime.now()
yr_string = now.strftime("%Y-%m")
dt_string = now.strftime("%m-%d")
savePath = Path('../figures/') / yr_string / dt_string
savePath.mkdir(parents=True, exist_ok=True)

# sim = 'TNG300-2'
# sim = 'TNG100-1'
# stacker = SimulationStacker(sim, 67, z=0.5,)

stacker = SimulationStacker('m100n1024', 125, z=0.5, simType='SIMBA', feedback='s50')

# nPixels = 261
nPixels = 1000
# particleType = 'tSZ'
particleType = 'tau'
# tau_map = stacker.loadData('tau', nPixels=nPixels, projection='yz', type='map')
tau_map = stacker.makeField(particleType, nPixels=nPixels, 
                            projection='yz', save=True)

# haloes = stacker.loadHalos()
# haloMass = haloes['GroupMass']
# haloRad = haloes['GroupRad']
# haloPos = haloes['GroupPos']

from halos import select_massive_halos
# halo_mask = select_massive_halos(haloMass, 10**(13.22), 5e14)

from mapMaker import create_masked_field




fig, ax = plt.subplots(2,2, figsize=(11, 10))
ax = ax.flatten()
norm = SymLogNorm(linthresh=1e-9, linscale=0.2, 
                  vmin=np.min(tau_map), vmax=np.max(tau_map), base=10) # type: ignore
# norm = LogNorm(vmin=np.min(tau_map), vmax=np.max(tau_map)) # type: ignore


ax[0].imshow(tau_map, norm=norm)
ax[0].set_title(f'Original {particleType} Map')

for i in range(3):
    i = i + 1
    # halo_cat = deepcopy(haloes[halo_mask])
    # haloes['GroupRad'] = haloRad[halo_mask] * i
    # masked_tau_maps = create_masked_field(stacker, halo_cat=halo_cat, pType='tau',
    #                                       nPixels=nPixels, projection='yz')
    masked_tau_map = stacker.makeField(particleType, nPixels=nPixels, projection='yz', 
                                       mask=True, maskRad=i, save=True)

    ax[i].imshow(masked_tau_map, norm=norm)
    ax[i].set_title(rf'$R_{{\rm mask}} = {i} R_{{200c}}$')

plt.tight_layout()
# plt.show()
plt.savefig(savePath / f'masked_{particleType}_{stacker.sim}.png', dpi=300)
print('Done!!')