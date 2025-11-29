# !pip install illustris_python
import sys

import numpy as np
import matplotlib.pyplot as plt
import h5py

import scipy
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# from abacusnbody.analysis.tsc import tsc_parallel
import time
import yaml
import glob
# import json
# import pprint 

# sys.path.append('../../illustrisPython/')
import illustris_python as il 

sys.path.append('/pscratch/sd/l/lindajin/SimulationStacker/src/')
# from tools import numba_tsc_3D, hist2d_numba_seq
from utils import fft_smoothed_map, comoving_to_arcmin
from halos import select_massive_halos, halo_ind, filter_edge_halo
from filters import total_mass, delta_sigma, CAP, CAP_from_mass, DSigma_from_mass, delta_sigma_mccarthy, delta_sigma_kernel, delta_sigma_ring
from loadIO import snap_path, load_halos, load_subsets, load_subset, load_data, save_data
from mapMaker import create_field, create_masked_field

try:
    import Pk_library as PKL
    import MAS_library as MASL
    HAS_PK_LIBRARY = True
except ImportError:
    HAS_PK_LIBRARY = False
    print("Warning: Pk_library and MAS_library not available. 3D power spectrum functions will not work.")


class SimulationStacker(object):

    def __init__(self, 
                 sim_index: int, 
                 snapshot: int = 74, 
                 nPixels=2000, 
                 simType='IllustrisTNG'):
        """DEPRECIATED henry's args for record only

        Args:
            sim (str): Simulation Instance, One of ['TNG300-1', 'TNG300-2', 'TNG100-1', 'TNG100-2', 'm50n512', 'm100n1024']
            snapshot (int): _description_
            nPixels (int, optional): Pixel size of the output 2D field, i.e. the number of pixels in each direction.
            simType (str, optional): Simulation type, one of ['IllustrisTNG', 'SIMBA']. Defaults to 'IllustrisTNG'.
            feedback (str, optional): feedback types for SIMBA. Defaults to None. One of 
                ['s50', 's50nox', 's50noagn', 's50nofb', 's50nojet'].
            z (float, optional): Redshift of the snapshot. Defaults to 0.0.

        UPDATED Args:
            sim_index (int): Simulation index, used to identify the specific simulation instance.
            snapshot (str, optional): Snapshot number as a string. Defaults to '074'.
        """
        self.sim_index = sim_index
        self.simType = simType
        self.snapshot = snapshot
        self.chunkNum = 0  # loading in Header from chunk 0 only for now.

        if self.simType == 'IllustrisTNG':
            # these paths are parent folders to data, not full paths to files
            self.sim_path = f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{self.sim_index}'
            self.snap_path = self.sim_path + '/snapdir_' + str(snapshot).zfill(3) + '/'
            self.catalog_path = self.sim_path + '/groups_' + str(snapshot).zfill(3) + '/'

        else:
            raise NotImplementedError('Simulation type not implemented')

        self.nPixels = nPixels
        self.nPixels_map = None  # to be set when making map

        with h5py.File(self.snap_path + 'snap_'+ str(snapshot).zfill(3) + f'.{self.chunkNum}.hdf5', 'r') as f:
            self.header = dict(f['Header'].attrs.items())
        
        # self.Lbox = self.header['BoxSize'] # kpc/h
        self.h = self.header['HubbleParam'] # Hubble parameter
        self.z = self.header['Redshift'] if self.simType == 'IllustrisTNG' else 0.47

        # Define cosmology
        self.cosmo = FlatLambdaCDM(H0=100 * self.h, Om0=self.header['Omega0'], Tcmb0=2.7255 * u.K, Ob0=self.header['OmegaBaryon'])
        
        # self.kpcPerPixel = self.Lbox / self.nPixels # technically kpc/h per pixel
        self.fields = {}
        self.maps = {}

    def makeField(self, pType, nPixels=None, projection='xy', save=False, load=True, 
                  mask=False, maskRad=3.0, base_path=None, dim='2D'):
        """Uses a histogram binning to make projected fields (either 2D or 3D) of a given particle type from the simulation.

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', 'Stars', 'BH' for mass maps, 'tSZ', 'kSZ', or 'tau' for SZ maps,
                and 'total' for all masses combined.
            nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to self.nPixels.
            projection (str, optional): Direction of the field projection. Currently only 'xy' is implemented. Defaults to 'xy'.
            save (bool, optional): If True, saves the field to a file. Defaults to False.
            load (bool, optional): If True, loads the field from a file if it exists and returns the field. Defaults to True.
            mask (bool, optional): If True, masks out areas outside of haloes in the field. Defaults to False.
            maskRad (float, optional): Number of virial radii around each halo to keep unmasked. Only used if mask=True. 
                Defaults to 3x virial radii.
            base_path (str, optional): Base path for loading/saving data. Defaults to None, which uses the default path.
            dim (str, optional): Dimension of the field to create. Either '2D' or '3D'. Defaults to '2D'.

        Raises:
            NotImplementedError: If field is not one of the ones listed above.

        Returns:
            np.ndarray: 2D or 3D numpy array of the field for the given particle type.

        TODO: Handle saving and loading of the fields for the masked case.
        """
    
        if nPixels is None:
            nPixels = self.nPixels

        if load:
            try:
                return self.loadData(pType, nPixels=nPixels, projection=projection, type='field', 
                                     mask=mask, maskRad=maskRad, base_path=base_path, dim=dim)
            except ValueError as e:
                print(e)
                print("Computing the field instead...")
                
        if mask:
            haloes = self.loadHalos(self.simType)
            haloMass = haloes['GroupMass']
            
            halo_mask, _ = select_massive_halos(haloMass, 10**(13.22), 5e14) # TODO: make this configurable from user input
            haloes['GroupMass'] = haloes['GroupMass'][halo_mask]
            haloes['GroupRad'] = haloes['GroupRad'][halo_mask] * maskRad # in kpc/h
            haloes['GroupPos'] = haloes['GroupPos'][halo_mask]

            field = create_masked_field(self, halo_cat=haloes, pType=pType, nPixels=nPixels, projection=projection,
                                        save3D=True, load3D=load, base_path=base_path, dim=dim) # TODO: make save3D and load3D configurable
        else:
            field = create_field(self, pType, nPixels, projection, dim=dim)
        
        if save:
            # TODO: Handle saving and loading of the fields for the masked case.
            save_data(field, self.simType, self.sim_index, pType, nPixels, 
                      projection, 'field', mask=mask, maskRad=maskRad, base_path=base_path, dim=dim)

        return field

    def makeMap(self, pType, projection='xy', beamsize=1.6, save=False, load=True, 
                pixelSize=0.5, mask=False, maskRad=3.0, base_path=None):
        """Make a 2D map convolved with a beam for a given particle type.
        This is more realistic than makeField

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', 'Stars', 'BH' for mass maps, 'tSZ', 'kSZ', or 'tau' for SZ maps,
                and 'total' for all masses combined.
            z (float, optional): Redshift of the snapshot. Defaults to None, in which case self.z is used.
            # nPixels (int, optional): Number of pixels in each direction of the 2D map. Defaults to self.nPixels.
            projection (str, optional): Direction of the map projection. Currently only 'xy' is implemented. Defaults to 'xy'.
            beamsize (float, optional): Size of the beam in arcminutes. Defaults to 1.6.
            save (bool, optional): If True, saves the map to a file. Defaults to False.
            load (bool, optional): If True, loads the map from a file if it exists and returns the map. Defaults to True.
            pixelSize (float, optional): The theoretical expected size of each pixel in arcminutes. Defaults to 0.5. arcminPerPixel overrides this to the exact size.
            mask (bool, optional): If True, masks out areas outside of haloes in the map. Defaults to False.
            maskRad (float, optional): Number of virial radii around each halo to keep unmasked. Only used if mask=True.
                Defaults to 3x virial radii.
            base_path (str, optional): Base path for loading/saving data. Defaults to None, which uses the default path.

        Returns:
            np.ndarray: 2D numpy array of the map for the given particle type.
        """        
            
        # Use stored cosmology

        # Get distance to the snapshot redshift
        # dA = cosmo.angular_diameter_distance(z).to(u.kpc).value
        # dA *= self.header['HubbleParam']  # Convert to kpc/h
        
        # Get the box size in angular units.
        # theta_arcmin = np.degrees(self.header['BoxSize'] / dA) * 60  # Convert to arcminutes # TODO: this is wrong for sure!! Change ASAP
        theta_arcmin = comoving_to_arcmin(self.header['BoxSize'], self.z, cosmo=self.cosmo)
        print(f"Box size: {self.header['BoxSize']} kpc/h , Map size at z={self.z}: {theta_arcmin:.2f} arcmin")

        # Round up to the nearest integer, pixel size is 0.5 arcmin as in ACT
        nPixels = np.ceil(theta_arcmin / pixelSize).astype(int)
        arcminPerPixel = theta_arcmin / nPixels  # Arcminutes per pixel, this is the true pixelSize after rounding.
        self.nPixels_map  = nPixels  # Store the nPixels used for the map in the instance
        # beamsize_pixel = beamsize / arcminPerPixel  # Convert arcminutes to pixels

        
        # Now that we know the expected pixel size, we try to load the map first before computing it:
        if load:
            try:
                return self.loadData(pType, nPixels=nPixels, projection=projection, type='map', 
                                     mask=mask, maskRad=maskRad, base_path=base_path)
            except ValueError as e:
                print(e)
                print("Computing the map instead...")    
        
        # If we don't have the map pre-saved, we then make the map. 
        # Since this is before doing beam convolution, this step is fine to do using makeField.
        map_ = self.makeField(pType, nPixels=nPixels, projection=projection, save=False, load=load, 
                              mask=mask, maskRad=maskRad, base_path=base_path)

        # Convolve the map with a Gaussian beam (only if beamsize is not None)
        if beamsize is not None:
            map_ = fft_smoothed_map(map_, beamsize, pixel_size_arcmin=arcminPerPixel)

        if save:
            save_data(map_, self.simType, self.sim_index, pType, nPixels, 
                      projection, 'map', mask=mask, maskRad=maskRad, base_path=base_path)


        return map_

    def convolveMap(self, map_, fwhm_arcmin, pixel_size_arcmin):
        """
        DEPRECIATED: Check fft_smoothed_map function in utils.py for new convolution code.
        Convolve the map with a Gaussian beam.

        Args:
            map_ (np.ndarray): 2D numpy array of the field for the given particle type.
            fwhm_arcmin (float): Full width at half maximum of the Gaussian beam in arcminutes.
            pixel_size_arcmin (float): Size of the pixel in arcminutes.

        Returns:
            np.ndarray: Convolved 2D numpy array.
        """
        
        sigma_pixels = fwhm_arcmin / (2.355 * pixel_size_arcmin)

        # Apply Gaussian filter
        convolved_map = gaussian_filter(map_, sigma=sigma_pixels, mode='wrap')
        
        return convolved_map
    
    def setMap(self, pType, map_, z=None, projection='xy', pixelSize=0.5):
        """Set a precomputed map for a given particle type.

        Args:
            pType (str): Particle type to set the map for.
            map_ (np.ndarray): 2D numpy array of the map.
            z (float, optional): Redshift of the snapshot. Defaults to None, in which case self.z is used.
            projection (str, optional): Direction of the field projection. Defaults to 'xy'. Options are ['xy', 'yz', 'xz']
        """

        if z is None:
            z = self.z

        self.maps[(pType, z, projection, pixelSize)] = map_

    def setField(self, pType, field_, nPixels=None, projection='xy'):
        """Set a precomputed field for a given particle type.

        Args:
            pType (str): Particle type to set the field for.
            field_ (np.ndarray): 2D numpy array of the field.
            nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to self.nPixels.
            projection (str, optional): Direction of the field projection. Currently only 'xy' is implemented. Defaults to 'xy'.
        """

        if nPixels is None:
            nPixels = self.nPixels

        self.fields[(pType, nPixels, projection)] = field_
    

    def stackMap(self, pType, filterType='cumulative', minRadius=0.5, maxRadius=6.0, numRadii=11,
                 z=None, projection='xy', save=False, load=True, radDistance=1.0, pixelSize=0.5, 
                 halo_number_density=2.4e-3,halo_mass_avg=10**(13.22), halo_mass_upper=5*10**(14), mask=False, maskRad=3.0,
                 subtract_mean=False):
        """Stack the map of a given particle type.

        Args:
            pType (str): Particle type to stack.
            filterType (str, optional): Type of filter to apply. Defaults to 'cumulative'.
            minRadius (float, optional): Minimum radius for stacking. Defaults to 0.2.
            maxRadius (float, optional): Maximum radius for stacking. Defaults to 6.0.
            numRadii (int, optional): Number of radial bins for stacking. Defaults to 11.
            z (float, optional): Redshift of the snapshot. Defaults to None, in which case self.z is used.
            projection (str, optional): Direction of the field projection. Defaults to 'xy'. Options are ['xy', 'yz', 'xz']
            save (bool, optional): If True, saves the stacked map to a file. Defaults to True.
            load (bool, optional): If True, loads the stacked map from a file if it exists. Defaults to True.
            radDistance (float, optional): Radial distance units for stacking. Defaults to 1 arcmin.
                Note there is no None option here as in stackField.
            pixelSize (float, optional): Size of each pixel in arcminutes. Defaults to 0.5.
            halo_mass_avg (float, optional): Average halo mass for selecting halos. Defaults to 10**(13.22).
            halo_mass_upper (float, optional): Upper mass bound for selecting halos. Defaults to None.
            mask (bool, optional): If True, masks out areas outside of haloes in the map. Defaults to False.
            maskRad (float, optional): Number of virial radii around each halo to keep unmasked. Only used if mask=True.
                Defaults to 3x virial radii.
            subtract_mean (bool, optional): If True, subtracts the mean of the map before stacking. Defaults to False.

        Returns:
            radii, profiles: Stacked radial profiles (2D) and their corresponding radii (1D).
            
        TODO:
            Add a wrapper for automatic stacking along all 3 projections.
            Implement the DSigma filter for stacking.
        """

        if z is None:
            z = self.z
        
        # Load or create the map
        fieldKey = (pType, z, projection, pixelSize)
        if not (fieldKey in self.maps and self.maps[fieldKey] is not None):
            self.maps[fieldKey] = self.makeMap(pType, projection=projection,
                                               save=save, load=load, pixelSize=pixelSize, mask=mask, maskRad=maskRad)

        # If subtract_mean is True, subtract the mean of the map before stacking.
        if subtract_mean:
            map_mean = np.mean(self.maps[fieldKey])
            self.maps[fieldKey] -= map_mean

        # Use the abstracted stacking function
        radii, profiles = self.stack_on_array(
            array=self.maps[fieldKey],
            filterType=filterType,
            minRadius=minRadius,
            maxRadius=maxRadius,
            numRadii=numRadii,
            projection=projection,
            radDistance=radDistance,
            radDistanceUnits='arcmin',
            halo_number_density=halo_number_density,
            pixelSize=pixelSize
        )
        
        # restore the mean if subtracted
        if subtract_mean:
            self.maps[fieldKey] += map_mean

       # Unit Conversion specific to SZ maps:
        T_CMB = 2.7255
        if pType == 'tau':
            # In the case of the tau field, we want to do unit conversion from optical depth units to micro-Kelvin.
            # This is done by multiplying the tau field by T_CMB * (v/c)
            v_c = 300000 / 299792458 # velocity over speed of light.
            pixArea = (pixelSize**2) # Convert to arcmin^2 units
            # factor = 1
            profiles = profiles * T_CMB * 1e6 * v_c * pixArea # Convert to micro-Kelvin, the units for kSZ in data.
        elif pType == 'kSZ':
            # TODO: kSZ unit conversion
            pixArea = (pixelSize**2) # Convert to arcmin^2 units
            profiles = profiles * pixArea # Convert to arcmin^2 units
            # pass
        elif pType == 'tSZ':
            # TODO: tSZ unit conversion
            # factor = (180.*60./np.pi)**2
            pixArea = (pixelSize**2) # Convert to arcmin^2 units
            profiles = profiles * pixArea # Convert to arcmin^2 units
            # pass
        else:
            # No unit conversion for other fields.
            pass
        
        return radii, profiles 

    def stackField(self, pType, filterType='cumulative', minRadius=0.1, maxRadius=4.5, numRadii=25,
                   projection='xy', nPixels=None, save=False, load=True, radDistance=1000, pixelSize=0.5,
                   halo_number_density=2.4e-3, mask=False, maskRad=3.0, subtract_mean=False):
        """Do stacking on the computed field.

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', or 'Stars'
            filterType (str, optional): Stacked Filter Types. One of ['cumulative', 'CAP', 'DSigma']. Defaults to 'cumulative'.
            minRadius (float, optional): Minimum radius in kpc/h for the stacking. Defaults to 0.1.
            maxRadius (float, optional): Maximum radius in kpc/h for the stacking. Defaults to 4.5.
            numRadii (int, optional): Number of radial bins for the stacking. Defaults to 25.
            projection (str, optional): Direction of the field projection. Currently only 'xy' is implemented. Defaults to 'xy'.
            nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to self.nPixels.
            save (bool, optional): If True, saves the stacked field to a file. Defaults to True.
            load (bool, optional): If True, loads the stacked field from a file if it exists. Defaults to True.
            radDistance (float, optional): Radial distance units for stacking. Defaults to 1000 kpc/h (so converts to 1 Mpc/h).
                If None, uses the mean halo radius from the halo catalog.
            mask (bool, optional): If True, masks out areas outside of haloes in the field. Defaults to False.
            maskRad (float, optional): Number of virial radii around each halo to keep unmasked. Only used if mask=True. 
                Defaults to 3x virial radii.
            subtract_mean (bool, optional): If True, subtracts the mean of the field before stacking. Defaults to False.

        Raises:
            NotImplementedError: If pType is not one of the ones listed above.

        Returns:
            radii, profiles : 1D radii and 2D profiles for the stacked field.
        """

        if nPixels is None:
            nPixels = self.nPixels

        # Load or create the field
        fieldKey = (pType, nPixels, projection)
        if not (fieldKey in self.fields and self.fields[fieldKey] is not None):
            self.fields[fieldKey] = self.makeField(pType, nPixels=nPixels, projection=projection,
                                                   save=save, load=load, mask=mask, maskRad=maskRad)
        else:
            assert self.fields[fieldKey].shape == (nPixels, nPixels), \
                f"Field shape mismatch: {self.fields[fieldKey].shape} != {(nPixels, nPixels)}"

        # Handle radDistance = None case
        if radDistance is None:
            haloes = self.loadHalos(self.simType)
            mass_min, mass_max, _ = halo_ind(2)
            halo_mask = np.where(np.logical_and((haloes['GroupMass'] > mass_min), (haloes['GroupMass'] < mass_max)))[0]
            radDistance = haloes['GroupRad'][halo_mask].mean()

        # If subtract_mean is True, subtract the mean of the map before stacking.
        if subtract_mean:
            field_mean = np.mean(self.fields[fieldKey])
            self.fields[fieldKey] -= field_mean

        # Use the abstracted stacking function
        radii, profiles = self.stack_on_array(
            array=self.fields[fieldKey],
            filterType=filterType,
            minRadius=minRadius,
            maxRadius=maxRadius,
            numRadii=numRadii,
            projection=projection,
            radDistance=radDistance,
            radDistanceUnits='kpc/h',
            halo_number_density=halo_number_density
        )
        
        # restore the mean if subtracted
        # TODO: this may introduce weird numerics behaviour, check later
        if subtract_mean:
            self.fields[fieldKey] += field_mean

        # Apply post-processing for CAP filter
        # This is taken care of in the `stack_on_array` function now
        # if filterType == 'CAP':
        #     radii_CAP = np.linspace(minRadius, maxRadius, 25)
        #     cap_profiles = CAP_from_mass(radii_CAP, radii, profiles.mean(axis=1))
        #     return radii_CAP, cap_profiles
        
        return radii, profiles

    def stack_on_array(self, array, filterType='cumulative', minRadius=0.1, maxRadius=4.5, numRadii=25,
                       projection='xy', radDistance=1000.0, radDistanceUnits='kpc/h', 
                       halo_number_density=2.4e-3, pixelSize=0.5):
        """Abstract stacking function that works on any 2D array.

        Args:
            array (np.ndarray): 2D array to stack on. Requires shape (nPixels, nPixels) such that the array is square.
            filterType (str, optional): Stacked Filter Types. One of ['cumulative', 'CAP', 'DSigma']. Defaults to 'cumulative'.
            minRadius (float, optional): Minimum radius for stacking. Defaults to 0.1.
            maxRadius (float, optional): Maximum radius for stacking. Defaults to 4.5.
            numRadii (int, optional): Number of radial bins for stacking. Defaults to 25.
            projection (str, optional): Direction projection used. Defaults to 'xy'.
            radDistance (float, optional): Radial distance units for stacking. Defaults to 1000.
            radDistanceUnits (str, optional): Units for radDistance. Either 'kpc/h' or 'arcmin'. Defaults to 'kpc/h'.
            halo_mass_avg (float, optional): Average halo mass for selecting halos. Defaults to 10**(13.22).
            halo_mass_upper (float, optional): Upper mass bound for selecting halos. Defaults to 5*10**(14).
            z (float, optional): Redshift for angular distance calculation (required if radDistanceUnits='arcmin'). Defaults to None.
            pixelSize (float, optional): Pixel size in arcminutes (required if radDistanceUnits='arcmin'). Defaults to 0.5.

        Returns:
            tuple: (radii, profiles) - 1D radii array and 2D profiles array.
        """
        
        nPixels = array.shape[0]
        assert array.shape == (nPixels, nPixels), f"Array must be square, got shape: {array.shape}"

        # Load the halo catalog and select halos
        haloes = self.loadHalos()
        haloMass = haloes['SubhaloMass']  # in 1e10 Msun/h
        haloPos = haloes['SubhaloPos']

        halo_mask, mass_list = select_massive_halos(haloMass, self.header['BoxSize'], halo_number_density)
        self.halo_mass_selected = mass_list  # Store for reference

        print(f'Number of halos selected: {halo_mask.shape[0]} at Mass threshold: {mass_list[0]: .2e} ~ {mass_list[-1]: .2e} Msun/h')

        # Convert radDistance to pixels based on units
        if radDistanceUnits == 'kpc/h':
            kpcPerPixel = self.header['BoxSize'] / nPixels
            RadPixel = radDistance / kpcPerPixel
            pixelSize_true = kpcPerPixel
        elif radDistanceUnits == 'arcmin':
            z = self.z
            # Calculate arcmin per pixel using stored cosmology
            # dA = self.cosmo.angular_diameter_distance(z).to(u.kpc).value
            # dA *= self.header['HubbleParam']  # Convert to kpc/h
            # theta_arcmin = np.degrees(self.header['BoxSize'] / dA) * 60
            theta_arcmin = comoving_to_arcmin(self.header['BoxSize'], z, cosmo=self.cosmo)
            arcminPerPixel = theta_arcmin / nPixels
            RadPixel = radDistance / arcminPerPixel
            pixelSize_true = arcminPerPixel
        else:
            raise ValueError(f"radDistanceUnits must be 'kpc/h' or 'arcmin', got: {radDistanceUnits}")
        
        # Set up filter function
        if filterType == 'cumulative':
            filterFunc = total_mass
        elif filterType == 'CAP':
            filterFunc = CAP
        elif filterType == 'DSigma':
            filterFunc = delta_sigma_kernel
            # filterFunc = delta_sigma_ring
        elif filterType == 'DSigma_mccarthy':
            filterFunc = delta_sigma_mccarthy
            z = self.z
            if radDistanceUnits != 'arcmin':
                raise ValueError('DSigma_mccarthy filter currently requires radDistanceUnits to be arcmin')
        else:
            raise NotImplementedError('Filter Type not implemented: ' + filterType)

        # Set up radial bins and cutout size
        if radDistanceUnits == 'kpc/h':
            # Linear bins from r_min to r_max (kSZ interest)
            radii_linear = np.linspace(minRadius, maxRadius, numRadii)
            # Log bins from r_max to 15 1000*kpc/h (lensing interest) Motivated by MacCarthy 2025
            radii_log = np.logspace(np.log10(maxRadius), np.log10(15), numRadii)
            # Concatenate, removing duplicate at r_split
            radii = np.concatenate([radii_linear, radii_log[1:]])
        else:  # arcmin units
            radii = np.linspace(minRadius, maxRadius, numRadii) # in radDistance units 


        if filterType == 'CAP':
            n_vir = int(np.ceil(np.sqrt(2) * maxRadius)) + 1
        else:
            n_vir = int(radii.max() + 1)  # number of virial radii to cutout

        # Do stacking
        drop_halo_count = 0
        profiles = []
        kept_masses = []
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
            if radDistanceUnits == 'kpc/h':
                haloLoc = np.round(haloPos_2D / (self.header['BoxSize'] / nPixels)).astype(int)
            else:  # arcmin units
                haloLoc = np.round(haloPos_2D / (self.header['BoxSize'] / nPixels)).astype(int)
        
            if filter_edge_halo(haloLoc, nPixels, int(np.ceil(np.sqrt(2) * maxRadius))*RadPixel):
                drop_halo_count += 1
                continue  # Skip this halo
            
            # Track the mass of kept halos
            kept_masses.append(mass_list[j])

            # Create cutout and radial distance grid
            cutout = SimulationStacker.cutout_2d_periodic(array, haloLoc, n_vir*RadPixel)
            rr = SimulationStacker.radial_distance_grid(cutout, (-n_vir, n_vir)) #same unit as n_vir, i.e. radDistance 
            
            
            if filterType == 'DSigma_mccarthy':
                # Use the Delta Sigma filter from Ian McCarthy et al. 2024
                # pass
                if radDistanceUnits != 'arcmin':
                    raise ValueError('DSigma_mccarthy filter currently requires radDistanceUnits to be arcmin')
                radii, profile, _ = delta_sigma_mccarthy(cutout, rr, pixel_scale_arcmin=pixelSize_true, z=z, # type: ignore
                                                         cosmo=self.cosmo, rmin_theta=minRadius, rmax_theta=maxRadius, n_rbins=numRadii)
            elif filterType == 'DSigma':
                # # TODO: This does not work with stackField for some reason.
                # if radDistanceUnits == 'arcmin':
                #     dr = pixelSize_true # 0.5 arcmin in pixels
                # else:
                #     # dr = 0.2 # 0.2 kpc/h in pixels
                #     dr = 3 / RadPixel # number of radDistance units for 3 pixel

                dr = (radii[1]-radii[0])/2
                profile = []
                for rad in radii:
                    # TODO: pixel_size unit conversions!! Important
                    filt_result = filterFunc(cutout, rr, rad, dr=dr, pixel_size=pixelSize_true)  # dr: #same unit as n_vir, i.e. radDistance arcmin or 1000kpc/h
                    profile.append(filt_result)
                
                profile = np.array(profile)
            else:
                
                # Apply filters at each radius
                profile = []
                for rad in radii:
                    # TODO: pixel_size here is placeholder. Carefully check units!
                    filt_result = filterFunc(cutout, rr, rad, pixel_size=1.) # type: ignore
                    profile.append(filt_result)
                
                profile = np.array(profile)
            
            profiles.append(profile)
            
        profiles = np.array(profiles).T
        self.halo_mass_selected = np.array(kept_masses)

        print(f'Dropped {drop_halo_count} halos because too close to the box edge for CAP (sqrt(2)*maxR)')

        return radii, profiles

    # Other util functions:
    
    @staticmethod
    def cutout_2d_periodic(array, center, length):
        """
        Returns a square cutout from a 2D array with periodic boundary conditions.
    
        Parameters:
        - array: 2D numpy array
        - center: tuple (x, y) center index
        - length: float or int, half-width of the cutout (will be rounded)
    
        Returns:
        - 2D numpy array cutout of shape (2*length+1, 2*length+1)
        """
        length = int(round(length))
        x, y = center
        size = 2 * length + 1
    
        # Generate index ranges with wrapping
        row_indices = [(x + i) % array.shape[0] for i in range(-length, length + 1)]
        col_indices = [(y + j) % array.shape[1] for j in range(-length, length + 1)]
    
        # Use np.ix_ to create a 2D index grid
        cutout = array[np.ix_(row_indices, col_indices)]
    
        return cutout
    
    @staticmethod
    def radial_distance_grid(array, bounds):
        """
        array: 2D numpy array (only shape is used)
        bounds: tuple ((x_min, x_max), (y_min, y_max)) representing physical bounds
        """
        rows, cols = array.shape
        xy_min, xy_max = bounds
    
        # Generate coordinate values for each axis
        x_coords = np.linspace(xy_min, xy_max, cols)
        y_coords = np.linspace(xy_min, xy_max, rows)
    
        # Create meshgrid of coordinates
        X, Y = np.meshgrid(x_coords, y_coords)
    
        # Calculate distances from the center (0,0)
        radial_distances = np.sqrt(X**2 + Y**2)
        
        return radial_distances
    

    
    # Some tools for file handling and loading:

    def loadHalos(self):
        """Load halo data for the specified simulation type."""
        return load_halos(self.sim_path, self.snapshot, self.simType, 
                     header=self.header)

    def loadSubsets(self, pType, keys=None):
        """Load particle subsets for the specified particle type."""
        return load_subsets(self.simPath, self.snapshot, self.simType, pType,
                          header=self.header, keys=keys)

    def loadSubset(self, pType, keys=None):
        """Load a subset of particles from the snapshot."""
        return load_subset(self.snap_Path, pType, 
                          header=self.header, keys=keys)

    def loadData(self, pType, nPixels=None, projection='xy', type='field', 
                 mask=False, maskRad=3.0, base_path=None, dim='2D'):
        """Load a precomputed field or map from file."""
        if nPixels is None:
            nPixels = self.nPixels
        return load_data(self.simType, self.sim_index, pType, nPixels, projection, type, 
                         mask=mask, maskRad=maskRad, base_path=base_path, dim=dim)


### Baryon Suppression Functions ###

    def get_field_baryon_suppression(self, grid=512, save=False, 
                                    threads=1):
        """Compute baryon suppression in 3D using power spectrum and correlation functions.
        
        Computes the cross-correlation between DM-only and total matter (DM+gas+stars+BH)
        density fields to quantify baryon feedback effects on matter clustering.

        Args:
            grid (int, optional): Grid resolution for 3D density field (grid^3 voxels). Defaults to 512.
            plot (bool, optional): If True, plots the power spectrum and correlation function. Defaults to False.
            save (bool, optional): If True, saves results to npz file. Defaults to False.
            threads (int, optional): Number of threads for power spectrum calculation. Defaults to None (uses all available).
            base_path (str, optional): Base path for saving results. Defaults to simulation data path.

        Returns:
            dict: Dictionary containing:
                - 'k': wavenumber array (h/Mpc)
                - 'PX_dm_tot': cross power spectrum P_dm×tot(k)
                - 'P_dm': DM-only power spectrum P_dm(k)
                - 'P_tot': total matter power spectrum P_tot(k)
                - 'r': correlation distance array (Mpc/h)
                - 'XX_dm_tot': cross correlation function ξ_dm×tot(r)
                - 'X_dm': DM-only correlation function ξ_dm(r)
                - 'X_tot': total matter correlation function ξ_tot(r)

        Raises:
            ImportError: If Pk_library and MAS_library are not installed.
            
        Notes:
            - Uses the existing makeField infrastructure with dim='3D'
            - Baryon suppression ratio: R(k) = P_dm×tot(k) / sqrt(P_dm(k) * P_tot(k))
            - Correlation suppression: ξ_dm×tot(r) / ξ_dm(r)
        """
        if not HAS_PK_LIBRARY:
            raise ImportError(
                "Pk_library and MAS_library are required for baryon suppression calculation. "
                "Install via: pip install Pk_library MAS_library"
            )
        
        
        if threads is None:
            threads = 256  # Default to many threads for HPC environments
        
        print(f"Computing 3D density fields at grid resolution {grid}^3...")
        
        # Generate 3D fields using existing makeField infrastructure
        # DM-only field
        print("Computing DM field...")
        filed_dm = self.makeField('DM', nPixels=grid,  
                                  save=False, load=False, dim='3D')
        
        # Total matter field (gas + DM + stars + BH)
        # We'll compute each separately and combine them
        print("Computing total matter field...")
        field_total = self.makeField('total', nPixels=grid,
                                  save=False, load=False, dim='3D')
                
        # Convert to overdensity fields
        print("Converting to overdensity fields...")
        delta_dm = filed_dm/ np.mean(filed_dm, dtype=np.float64);  delta_dm -= 1.0
        delta_tot = field_total / np.mean(field_total, dtype=np.float64);  delta_tot -= 1.0

        # Compute power spectra
        print("Computing power spectra...")
        BoxSize = self.header['BoxSize']
        
        # Cross power spectrum
        kX, PX_dm_tot = self._compute_pk_3D(delta_dm, delta_tot, BoxSize, grid, threads)
        
        # Auto power spectra
        k, P_dm = self._compute_pk_3D(delta_dm, None, BoxSize, grid, threads)
        _, P_tot = self._compute_pk_3D(delta_tot, None, BoxSize, grid, threads)
        
        # Compute correlation functions
        print("Computing correlation functions...")
        rX, XX_dm_tot = self._compute_corr_3D(delta_dm, delta_tot, BoxSize, grid, threads)
        r, X_dm = self._compute_corr_3D(delta_dm, None, BoxSize, grid, threads)
        _, X_tot = self._compute_corr_3D(delta_tot, None, BoxSize, grid, threads)
        
        # Store results
        results = {
            'k': k,
            'PX_dm_tot': PX_dm_tot,
            'P_dm': P_dm,
            'P_tot': P_tot,
            'r': r,
            'XX_dm_tot': XX_dm_tot,
            'X_dm': X_dm,
            'X_tot': X_tot
        }
        
        if save:
            print('Saving Baryon Suppression Field...')
            # Save as NPZ using common utility; ensure dim key uses '3D'
            save_data(results, self.simType, self.sim_index, 'Pk', grid,
                       data_type='field', dim='3D', file_type='npz')
        else:
            return results
        

    def _compute_pk_3D(self, field, field_b=None, BoxSize=50000, grid=512, threads=1):
        """Compute the 3D power spectrum for 3D fields.
        
        Args:
            field (np.ndarray): Primary field, shape (grid, grid, grid). DM field if field_b is provided.
            field_b (np.ndarray, optional): Secondary baryonic field for cross power spectrum. Defaults to None.
            BoxSize (float, optional): Physical size of the box in Mpc/h. Defaults to 50.
            grid (int, optional): Grid resolution. Defaults to 512.
            threads (int, optional): Number of threads for computation. Defaults to 1.

        Returns:
            tuple: (k, Pk) - wavenumber array and power spectrum
        """
        assert field.ndim == 3, "Input field must have shape (N, N, N)"
        
        MAS = 'CIC'
        verbose = False
        axis = 0
        
        k_nyq = np.pi * grid / BoxSize
        
        # Convert to float32 for Pk_library compatibility
        field = field.astype(np.float32)
        
        Pk3D = PKL.Pk(field, BoxSize, axis=axis, MAS=MAS, threads=threads, verbose=verbose)
        k = Pk3D.k3D
        Pk_val = Pk3D.Pk[:, 0]  # monopole
        
        if field_b is not None:
            field_b = field_b.astype(np.float32)
            
            # Cross power spectrum
            Pk3D = PKL.XPk([field, field_b], BoxSize, axis=axis, MAS=[MAS, MAS], threads=threads)
            k = Pk3D.k3D
            Pk_val = Pk3D.XPk[:, 0, 0]  # monopole of 1-2 cross P(k)
        
        return k[k <= k_nyq], Pk_val[k <= k_nyq]

    def _compute_corr_3D(self, field, field_b=None, BoxSize=50000, grid=512, threads=1):
        """Compute the 3D correlation function for 3D fields.
        
        Args:
            field (np.ndarray): Primary field, shape (grid, grid, grid).
            field_b (np.ndarray, optional): Secondary field for cross correlation. Defaults to None.
            BoxSize (float, optional): Physical size of the box in Mpc/h. Defaults to 50.
            grid (int, optional): Grid resolution. Defaults to 512.
            threads (int, optional): Number of threads for computation. Defaults to 1.

        Returns:
            tuple: (r, Xi) - distance array and correlation function
        """
        assert field.ndim == 3, "Input field must have shape (N, N, N)"
        
        MAS = 'CIC'
        verbose = False
        axis = 0
        
        # Convert to float32 for Pk_library compatibility
        field = field.astype(np.float32)
        
        CF  = PKL.Xi(field, BoxSize, MAS, axis, threads)
        
        # get the auto-correlation
        r      = CF.r3D      #radii in Mpc/h
        xi0    = CF.xi[:,0]  #correlation function (monopole)

        if field_b is not None:
            field_b = field_b.astype(np.float32)

            # Cross-correlation
            CCF = PKL.XXi(field, field_b, BoxSize, axis=axis, MAS=[MAS, MAS], threads=threads)
            r      = CCF.r3D      #radii in Mpc/h
            xi0   = CCF.xi[:,0] 
        
        return r[r<=BoxSize], xi0[r<=BoxSize]
