# sys.path.append('../../illustrisPython/')
import illustris_python as il 

# from tools import numba_tsc_3D, hist2d_numba_seq
# from stacker import SimulationStacker
from utils import fft_smoothed_map
from halos import select_massive_halos, halo_ind
from filters import total_mass, delta_sigma, CAP, CAP_from_mass, DSigma_from_mass
from loadIO import snap_path, load_halos, load_subsets, load_subset, load_data, save_data
from astropy.cosmology import FlatLambdaCDM, Planck18
import astropy.units as u
import numpy as np
import glob
from scipy.stats import binned_statistic_2d
from abacusnbody.analysis.tsc import tsc_parallel #put it on a grid using tsc interpolation # type: ignore
import MAS_library as MASL

from mask_utils import get_cutout_mask_3d

import time

# def compute_cosmological_parameters(header, z, cosmology=Planck18):
#     """Compute cosmological parameters for angular distance calculations.

#     Args:
#         header (dict): Header information from the simulation snapshot.
#         z (float): Redshift of the snapshot.

#     Returns:
#         cosmo (FlatLambdaCDM): Cosmology object for angular distance calculations.
#         dA (float): Angular diameter distance at redshift z. Units: kpc/h
#         theta_arcmin (float): Angular size of the simulation box in arcminutes.
#     """
    
#     # cosmo = FlatLambdaCDM(H0=100 * header['HubbleParam'], Om0=header['Omega0'], Tcmb0=2.7255 * u.K)
#     dA = cosmology.angular_diameter_distance(z).to(u.kpc).value
#     dA *= header['HubbleParam']  # Convert to kpc/h
#     theta_arcmin = np.degrees(header['BoxSize'] / dA) * 60
#     return cosmology, dA, theta_arcmin

# def calculate_pixel_parameters(theta_arcmin, pixelSize):
#     """Calculate pixel grid parameters. This is used for creating maps with pixels of a given size.

#     Args:
#         theta_arcmin (float): Angular size of the simulation box in arcminutes.
#         pixelSize (float): Size of each pixel in arcminutes.

#     Returns:
#         nPixels (int): Number of pixels in each direction.
#         arcminPerPixel (float): Angular size of each pixel in arcminutes.
#     """
    
#     nPixels = np.ceil(theta_arcmin / pixelSize).astype(int)
#     arcminPerPixel = theta_arcmin / nPixels
#     return nPixels, arcminPerPixel

# def create_basic_field(stacker, pType, nPixels, projection,  
#                       weight_calculator=None, required_keys=None):
#     """Create a field for any particle type with customizable weighting.
    
#     Args:
#         stacker: The stacker object
#         pType: Particle type
#         nPixels: Number of pixels
#         projection: Projection direction
#         weight_calculator: Function to calculate weights from particles. 
#                           If None, uses masses.
#         required_keys: List of keys to load from particles. 
#                       If None, uses ['Coordinates', 'Masses'].
#     """
#     # if load:
#     #     try:
#     #         return stacker.loadData(pType, nPixels=nPixels, projection=projection, type='field')
#     #     except ValueError as e:
#     #         print(e)
#     #         print("Computing the field instead...")
    
#     # Set default keys if not provided
#     if required_keys is None:
#         required_keys = ['Coordinates', 'Masses']
    
#     # Common field creation logic
#     folderPath = stacker.snapPath(stacker.simType, pathOnly=True)
#     if stacker.simType == 'IllustrisTNG':
#         snaps = glob.glob(folderPath + 'snap_*.hdf5')
#     elif stacker.simType == 'SIMBA':
#         snaps = glob.glob(folderPath)
    
#     gridSize = [nPixels, nPixels]
#     minMax = [0, stacker.header['BoxSize']]
#     field_total = np.zeros(gridSize)
    
#     t0 = time.time()
#     for i, snap in enumerate(snaps):
#         particles = stacker.loadSubset(pType, snapPath=snap, keys=required_keys)
#         coordinates = particles['Coordinates']
        
#         # Calculate weights using provided function or default to masses
#         if weight_calculator is not None:
#             weights = weight_calculator(particles, stacker)
#         else:
#             weights = particles['Masses'] * 1e10 / stacker.header['HubbleParam']
        
#         # Handle projection
#         coordinates = get_projected_coordinates(coordinates, projection)
        
#         # Bin the data
#         result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], weights, 
#                                     'sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
#         field_total += result.statistic
        
#         if i % 10 == 0:
#             print(f'Processed {i} snapshots, time elapsed: {time.time() - t0:.2f} seconds')
    
#     # if save:
#     #     save_field_data(stacker, pType, nPixels, projection, field_total)
    
#     return field_total

# def create_sz_field(stacker, pType, nPixels, projection):
#     """Create SZ-specific fields (tSZ, kSZ, tau)

#     Args:
#         stacker (SZStacker): SZStacker object for loading and saving data.
#         pType (str): Particle type to create the field for. Either 'tSZ', 'kSZ', or 'tau'.
#         nPixels (int): Number of pixels of the map in each direction. The created field will be square.
#         projection (str): Projection direction (One of 'xy', 'xz', 'yz').

#     Returns:
#         2D np.ndarray: The created SZ field. 2D array of shape (nPixels, nPixels).
#     """
    
#     # Define the required keys for SZ calculations
#     sz_keys = ['Coordinates', 'Masses', 'ElectronAbundance', 
#                'InternalEnergy', 'Density', 'Velocities']
    
#     # Create the weight calculator function for SZ fields
#     def sz_weight_calculator(particles, stacker_obj):
#         # SZ field constants
#         sz_constants = get_sz_constants()
        
#         # Get simulation parameters
#         h = stacker_obj.header['HubbleParam']
#         unit_mass = 1.e10 * (sz_constants['solar_mass'] / h)
#         unit_dens = 1.e10 * (sz_constants['solar_mass'] / h) / (sz_constants['kpc_to_cm'] / h)**3
#         unit_vol = (sz_constants['kpc_to_cm'] / h)**3
        
#         z = stacker_obj.z
#         a = 1./(1 + z)
        
#         # Calculate SZ quantities
#         sz_quantities = calculate_sz_quantities(particles, sz_constants, unit_dens, unit_vol, a)
        
#         # Return appropriate weights based on pType
#         return get_sz_weights(pType, sz_quantities)
    
#     # Use the generalized create_basic_field function
#     return create_basic_field(stacker, pType, nPixels, projection, 
#                              weight_calculator=sz_weight_calculator,
#                              required_keys=sz_keys)

# def get_projected_coordinates(coordinates, projection):
#     """Helper function for getting 2D coordinates for specified projection.

#     Args:
#         coordinates (np.ndarray): 3D coordinates of particles.
#         projection (str): Projection direction ('xy', 'xz', or 'yz').

#     Raises:
#         ValueError: If projection type is not one of 'xy', 'xz' or 'yz'.

#     Returns:
#         np.ndarray: 2D projected coordinates.
#     """
    
#     if projection == 'xy':
#         return coordinates[:, :2]
#     elif projection == 'xz':
#         return coordinates[:, [0, 2]]
#     elif projection == 'yz':
#         return coordinates[:, 1:]
#     else:
#         raise ValueError("Projection type not one of 'xy', 'xz' or 'yz': " + projection)

# def get_sz_constants():
#     """Return dictionary of SZ-related physical constants.

#     Returns:
#         dict: Dictionary of SZ-related physical constants.
#     """
#     # Physical constants in cgs units
#     gamma = 5/3. # unitless. Adiabatic Index
#     k_B = 1.3807e-16 # cgs (erg/K), Boltzmann constant
#     m_p = 1.6726e-24 # g, mass of proton
#     unit_c = 1.e10 # TNG faq is wrong (see README.md)
#     # unit_c = 1.023**2*1.e10
#     X_H = 0.76 # unitless, primordial hydrogen fraction
#     sigma_T = 6.6524587158e-29*1.e2**2 # cm^2, thomson cross section
#     m_e = 9.10938356e-28 # g, electron mass
#     c = 29979245800. # cm/s, speed of light
#     const = k_B*sigma_T/(m_e*c**2) # cgs (cm^2/K), constant for compton y computation
#     kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
#     solar_mass = 1.989e33 # g
    
#     return {
#         'gamma': gamma, # Adiabatic index
#         'k_B': k_B, # Boltzmann constant in cgs
#         'm_p': m_p, # Proton mass in grams
#         'unit_c': unit_c, # Velocity unit in cm/s
#         'X_H': X_H, # Hydrogen mass fraction
#         'sigma_T': sigma_T, # Thomson cross-section in cm^2
#         'm_e': m_e, # Electron mass in grams
#         'c': c, # Speed of light in cm/s
#         'kpc_to_cm': kpc_to_cm, # kpc to cm conversion
#         'solar_mass': solar_mass # Solar mass in grams
#     }

# def calculate_sz_quantities(particles, constants, unit_dens, unit_vol, a):
#     """Calculate SZ-related quantities from particle properties.

#     Args:
#         particles (dict): Dictionary containing particle properties.
#         constants (dict): Dictionary containing physical constants.
#         unit_dens (float): Density unit conversion factor.
#         unit_vol (float): Volume unit conversion factor.
#         a (float): Scale factor.

#     Returns:
#         dict: Dictionary containing calculated SZ-related quantities.
#     """

#     # Extract particle properties
#     Co = particles['Coordinates']
#     EA = particles['ElectronAbundance']
#     IE = particles['InternalEnergy']
#     D = particles['Density']
#     M = particles['Masses']
#     V = particles['Velocities']
    
#     # Calculate derived quantities
#     dV = M/D
#     D *= unit_dens
    
#     # Obtain electron temperature, electron density and velocity.
#     Te = ((constants['gamma'] - 1.) * IE / constants['k_B'] * 
#           4 * constants['m_p'] / (1 + 3 * constants['X_H'] + 4 * constants['X_H'] * EA) * 
#           constants['unit_c']) # K
#     ne = EA * constants['X_H'] * D / constants['m_p'] # ccm^-3 # True for TNG and mixed for MTNG because of unit difference
#     Ve = V * np.sqrt(a) # km/s
    
#     # Calculate SZ signals
#     const = constants['k_B'] * constants['sigma_T'] / (constants['m_e'] * constants['c']**2) # const for y compton computation.
    
#     return {
#         'dY': const * (ne * Te * dV) * unit_vol,
#         'b': constants['sigma_T'] * (ne[:, None] * (Ve/constants['c']) * dV[:, None]) * unit_vol,
#         'tau': constants['sigma_T'] * (ne * dV) * unit_vol
#     }

# def get_sz_weights(pType, sz_quantities):
#     """Get appropriate weights for different SZ field types.

#     Args:
#         pType (str): The type of SZ field ('tSZ', 'kSZ', or 'tau').
#         sz_quantities (dict): Dictionary containing SZ-related quantities.

#     Raises:
#         ValueError: If the particle type is not recognized.

#     Returns:
#         np.ndarray: The weights for the specified SZ field type.
#     """
#     if pType == 'tSZ':
#         return sz_quantities['dY']
#     elif pType == 'kSZ':
#         # For kSZ, we need to handle the velocity component properly
#         # Assuming we want the line-of-sight component (z-direction for 'xy' projection)
#         return sz_quantities['b'][:, 2]  # Take z-component for xy projection
#     elif pType == 'tau':
#         return sz_quantities['tau']
#     else:
#         raise ValueError('Particle type not recognized: ' + pType)

def create_field(stacker, pType, nPixels, projection, dim='2D'):
    """Wrapper function to create the appropriate field type.

    Args:
        stacker (SimulationStacker): The stacker instance.
        pType (str): The type of particle ('tSZ', 'kSZ', or 'tau').
        nPixels (int): The number of pixels in the map.
        projection (str): The projection direction ('xy', 'yz', or 'xz').
        save (bool): Whether to save the field data.
        load (bool): Whether to load the field data.

    Returns:
        np.ndarray: The created field data.
    """
    
    sz_types = ['tSZ', 'kSZ', 'tau']
    combined_types = ['total', 'baryon']
    
    if pType in sz_types:
        return make_sz_field(stacker, pType, nPixels, projection, dim=dim)
    elif pType in combined_types:
        return make_combined_field(stacker, pType, nPixels, projection, dim=dim)
    else:
        return make_mass_field(stacker, pType, nPixels, projection, dim=dim)


def make_sz_field(stacker, pType, nPixels=None, projection='xy', dim='2D'):
    """Create a map from the simulation data. 
    Much of the algo for this method comes from the SZ_TNG repo on Github authored by Boryana Hadzhiyska.

    Args:
        stacker (SimulationStacker): The stacker instance.
        pType (str): The type of particle to use for the map. Either 'tSZ', 'kSZ', or 'tau'. Added 'tau_DM'
            Note that in the case of 'kSZ', an optical depth (tau) map will be created instead of a velocity map.
        z (float, optional): The redshift to use for the map. Defaults to None.
        nPixels: Size of the output map in pixels. Defaults to stacker.nPixels.
        projection (str, optional): The projection direction ('xy', 'yz', or 'xz'). Defaults to 'xy'.
        dim (str, optional): Dimension of the map ('2D' or '3D'). Defaults to '2D'.
    
    TODO: Implement the same functionality for DM. 
    """
    if nPixels is None:
        nPixels = stacker.nPixels
        
    if pType == 'tau_DM':
        pType = 'tau'
        print("Warning: 'tau_DM' is experimental.")
        use_tau_DM = True
    else:
        use_tau_DM = False

    # If loading fails, we then compute the necessary fields. Much of the code below is the same as the SZ_TNG repo.            
    # Define some necessary parameters: (Physical units in cgs)
    gamma = 5/3. # unitless. Adiabatic Index
    k_B = 1.3807e-16 # cgs (erg/K), Boltzmann constant
    m_p = 1.6726e-24 # g, mass of proton
    unit_c = 1.e10 # TNG faq is wrong (see README.md)
    # unit_c = 1.023**2*1.e10
    X_H = 0.76 # unitless, primordial hydrogen fraction
    sigma_T = 6.6524587158e-29*1.e2**2 # cm^2, thomson cross section
    m_e = 9.10938356e-28 # g, electron mass
    c = 29979245800. # cm/s, speed of light
    const = k_B*sigma_T/(m_e*c**2) # cgs (cm^2/K), constant for compton y computation
    kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
    solar_mass = 1.989e33 # g


    # sim params (same for MTNG and TNG)
    h = stacker.header['HubbleParam'] # Hubble Parameter
    unit_mass = 1.e10*(solar_mass/h)
    # unit_mass = 1.0 # msun/h, units already converted.
    unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
    # unit_dens = 1./(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
    unit_vol = (kpc_to_cm/h)**3 # cancels unit dens division from previous, so h doesn't matter neither does kpc


    # z = zs[snaps == snapshot] # TODO: Implement the automatic selection of snapshot from redshift (or vice versa)
    z = stacker.z
    a = 1./(1+z) # scale factor
    Lbox_hkpc = stacker.header['BoxSize'] # kpc/h


    # Now we compute the SZ fields iteratively, iterating over every snapshot chunk: (Note SIMBA only has one chunk)
    
    # Get chunks:        
    folderPath = stacker.snap_path
    if stacker.simType == 'IllustrisTNG':
        snaps = glob.glob(folderPath + 'snap_*.hdf5') # TODO: fix this for SIMBA (done I think)
    elif stacker.simType == 'SIMBA':
        # print('Folder Path:', folderPath + f'*_{stacker.snapshot}.hdf5')
        snaps = glob.glob(folderPath)
        # print('Snaps:', snaps)

    # Convert coordinates to pixel coordinates        
    if dim == '2D':
        gridSize = [nPixels, nPixels]
    elif dim == '3D':
        gridSize = [nPixels, nPixels, nPixels]
    else:
        raise ValueError("dim must be either '2D' or '3D': " + dim)
    
    Lbox = stacker.header['BoxSize'] # kpc/h
    minMax = [0, Lbox]
    field_total = np.zeros(gridSize)
    
    t0 = time.time()
    for i, snap in enumerate(snaps):

        # particles = stacker.loadSubset(pType, snapPath=snap, keys=['Coordinates', 'Masses', 'ElectronAbundance', 'InternalEnergy', 'Density', 'Velocities'])
        particles = load_subset(snap, pType, header=stacker.header, sim_name=stacker.simType,
                                keys=['Coordinates', 'Masses', 'ElectronAbundance', 'InternalEnergy', 'Density', 'Velocities'])

        Co = particles['Coordinates']
        EA = particles['ElectronAbundance']
        IE = particles['InternalEnergy']
        D = particles['Density']
        M = particles['Masses']
        V = particles['Velocities']

        
        # for each cell, compute its total volume (gas mass by gas density) and convert density units
        dV = M/D # ckpc/h^3 
        D *= unit_dens # g/ccm^3 # True for TNG and mixed for MTNG because of unit difference
        # unit_c = 1.e10 # TNG faq is wrong (see README.md)

        # obtain electron temperature, electron number density and velocity
        Te = (gamma - 1.)*IE/k_B * 4*m_p/(1 + 3*X_H + 4*X_H*EA) * unit_c # K
        ne = EA*X_H*D/m_p # ccm^-3 # True for TNG and mixed for MTNG because of unit difference
        Ve = V*np.sqrt(a) # km/s

        # compute the contribution to the y and b signals of each cell
        # ne*dV cancel unit length of simulation and unit_vol converts ckpc/h^3 to cm^3
        # both should be unitless (const*Te/d_A**2 is cm^2/cm^2; sigma_T/d_A^2 is unitless)
        dY = const*(ne*Te*dV)*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h)/nPixels)**2.#d_A**2 # Compton Y parameter
        b = sigma_T*(ne[:, None]*(Ve/c)*dV[:, None])*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h)/nPixels)**2.#d_A**2 # kSZ signal
        tau = sigma_T*(ne*dV)*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h)/nPixels)**2.#d_A**2 # Optical depth. This is what we use for tau

        # Now we make the fields:
        
        if dim == '2D':
        
            if projection == 'xy':
                coordinates = Co[:, :2]  # Take x and y coordinates
            elif projection == 'xz':
                coordinates = Co[:, [0, 2]] # Take x and z coordinates
            elif projection == 'yz':
                coordinates = Co[:, 1:] # Take y and z coordinates
            else:
                raise NotImplementedError('Projection type not implemented: ' + projection)

            if pType == 'tSZ':
                # field = hist2d_numba_seq(np.array([coordinates[:, 0], coordinates[:, 1]]), bins=gridSize, ranges=(minMax, minMax), weights=dY)
                result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], values=dY, 
                                            statistic='sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
            elif pType == 'kSZ':
                # field = hist2d_numba_seq(np.array([coordinates[:, 0], coordinates[:, 1]]), bins=gridSize, ranges=(minMax, minMax), weights=b)
                result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], values=b, 
                                            statistic='sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
            elif pType == 'tau':
                # field = hist2d_numba_seq(np.array([coordinates[:, 0], coordinates[:, 1]]), bins=gridSize, ranges=(minMax, minMax), weights=tau)
                result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], values=tau, 
                                            statistic='sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
            else:
                raise ValueError('Particle type not recognized: ' + pType)
            
            field = result.statistic
            field_total += field
        
        elif dim == '3D':
            if pType == 'tSZ':
                field_total = tsc_parallel(Co, field_total, Lbox, weights=dY)
            elif pType == 'kSZ':
                field_total = tsc_parallel(Co, field_total, Lbox, weights=b)
            elif pType == 'tau':
                field_total = tsc_parallel(Co, field_total, Lbox, weights=tau)
            else:
                raise ValueError('Particle type not recognized: ' + pType)

        print(f'Processed {i} snapshots, time elapsed: {time.time() - t0:.2f} seconds')
    
    
    print('hist2d time:', time.time() - t0)
    
    return field_total

def make_mass_field(stacker, pType, nPixels=None, projection='xy', dim='2D'):
    """Used a histogram binning to make projected 2D fields of a given particle type from the simulation.

    Args:
        stacker (SimulationStacker): The stacker instance.
        pType (str): Particle Type. One of 'gas', 'DM', 'Stars', or 'BH'
        nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to stacker.nPixels.
        projection (str, optional): Direction of the field projection. Currently only 'xy' is implemented. Defaults to 'xy'.
        dim (str, optional): Dimension of the map ('2D' or '3D'). Defaults to '2D'.

    Raises:
        NotImplementedError: If field is not one of the ones listed above.

    Returns:
        np.ndarry: 2D numpy array of the field for the given particle type.
        
    TODO:
        Add directionality to the fields (i.e. x, y, and z projected 2D fields.)
    """
    if nPixels is None:
        nPixels = stacker.nPixels
            
    Lbox = stacker.header['BoxSize'] # kpc/h
    
    # Get all particle snap chunks:

    folderPath = stacker.snap_path
    if stacker.simType == 'IllustrisTNG':
        snaps = glob.glob(folderPath + 'snap_*.hdf5') # TODO: fix this for SIMBA (done I think)
    elif stacker.simType == 'SIMBA':
        snaps = glob.glob(folderPath)
        print('Snaps:', snaps)
    # The code below does the statistic by chunk rather than by the whole dataset
    
    # Initialize empty maps
    if dim == '2D':
        gridSize = [nPixels, nPixels]
    elif dim == '3D':
        gridSize = [nPixels, nPixels, nPixels]
    else:
        raise ValueError("dim must be either '2D' or '3D': " + dim)
    
    Lbox = stacker.header['BoxSize'] # kpc/h
    minMax = [0, Lbox]
    field_total = np.zeros(gridSize)
    
    t0 = time.time()
    for i, snap in enumerate(snaps):
        # particles = stacker.loadSubset(pType, snapPath=snap)
        particles = load_subset(snap, pType, header=stacker.header, sim_name=stacker.simType)
        coordinates = particles['Coordinates'] # kpc/h
        masses = particles['Masses']  * 1e10 / stacker.header['HubbleParam'] # Msun/h
        
        
        if dim == '2D':
            
            if projection == 'xy':
                coordinates = coordinates[:, :2]  # Take x and y coordinates
            elif projection == 'xz':
                coordinates = coordinates[:, [0, 2]] # Take x and z coordinates
            elif projection == 'yz':
                coordinates = coordinates[:, 1:] # Take y and z coordinates
            else:
                raise NotImplementedError('Projection type not implemented: ' + projection)
            
            # Convert coordinates to pixel coordinates        

            result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], masses, 
                                        'sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
            field = result.statistic
            
            field_total += field
            
        elif dim == '3D':
            # field_total = tsc_parallel(coordinates, field_total, Lbox, weights=masses)
            MAS     = 'CIC'  #mass-assigment scheme
            verbose = False   #print information on progress

            # define 3D density field
            field = np.zeros((nPixels,nPixels,nPixels), dtype=np.float32)  # use float64 for better precision

            # construct 3D density field
            MASL.MA(coordinates.astype(np.float32), field, Lbox, MAS, W=masses.astype(np.float32), verbose=verbose)

            # at this point, field contains the effective gas mass in each voxel
            field_total += field

        if i % 10 == 0:
            print(f'Processed {i} snapshots, time elapsed: {time.time() - t0:.2f} seconds')

    print('Binned statistic time:', time.time() - t0)

    return field_total

def make_combined_field(stacker, pType, nPixels=None, projection='xy', dim='2D', load=True, base_path=None):
    """Create a combined mass field by summing over all particle types.

    Args:
        stacker (SimulationStacker): The stacker instance.
        pType (str): The type of particle to use for the map. Either 'total' (for all particle types) or 'baryon' (for gas and stars only).
        nPixels (int, optional): The number of pixels in the map. Defaults to stacker.nPixels.
        projection (str, optional): The projection direction ('xy', 'yz', or 'xz'). Defaults to 'xy'.
        dim (str, optional): Dimension of the map ('2D' or '3D'). Defaults to '2D'.

    Raises:
        ValueError: If pType is not 'total'.

    Returns:
        np.ndarray: The created total mass field data.
    """
    if nPixels is None:
        nPixels = stacker.nPixels

    if pType == 'total':
        particle_types = ['gas', 'DM', 'Stars', 'BH']
    elif pType == 'baryon':
        particle_types = ['gas', 'Stars', 'BH']
    else:
        raise ValueError("pType must be 'total' or 'baryon' for make_combined_field.")

    if dim == '2D':
        gridSize = [nPixels, nPixels]
    elif dim == '3D':
        gridSize = [nPixels, nPixels, nPixels]
    else:
        raise ValueError("dim must be either '2D' or '3D': " + dim)
    total_field = np.zeros(gridSize)

    for pt in particle_types:
        print("Processing particle type:", pt)
        if load:
            try:
                field = load_data(stacker.simType, stacker.sim_index, pt, nPixels, projection, 'field', 
                                  mask=False, maskRad=3.0, base_path=base_path, dim=dim)
            except ValueError as e:
                print(e)
                print("Computing the field instead...")
                field = make_mass_field(stacker, pt, nPixels, projection, dim=dim)
        total_field += field

    return total_field

def create_masked_field(stacker, pType, nPixels, halo_cat, projection='xy', 
                        save3D=False, load3D=False, base_path=None, dim='2D'):
    """Create a masked field, where objects outside of n radii of the halo catalogue
    is masked out.

    Args:
        stacker (SimulationStacker): The stacker instance.
        pType (str): The type of particle to use for the map.
        nPixels (int): The number of pixels in the map.
        halo_cat (dict): The halo catalog containing halo positions and radii.
        projection (str, optional): The projection direction ('xy', 'yz', or 'xz'). Defaults to 'xy'.
        save3D (bool, optional): Whether to save the 3D field. Defaults to False.
        load3D (bool, optional): Whether to load the 3D field. Defaults to False.
        base_path (str, optional): The base path for saving/loading data. Defaults to None.
        dim (str, optional): Dimension of the map ('2D' or '3D'). Defaults to '2D'.

    Raises:
        NotImplementedError: If the projection type is not implemented.

    Returns:
        np.ndarray: The created masked field data.
    """
    
    if dim not in ['2D', '3D']:
        raise ValueError("dim must be either '2D' or '3D': " + dim)

    # First make the field:
    
    if load3D:
        try:
            field_3D = load_data(stacker.simType, stacker.sim_index, pType, nPixels, 
                                 projection, type='field', mask=False, maskRad=3.0, 
                                 base_path=base_path, dim='3D')
            save3D = False # No need to save if we loaded
            print('Loaded 3D field successfully.')
        except ValueError as e:
            print(e)
            print("Computing the 3D field instead...")
            field_3D = create_field(stacker, pType, nPixels, projection, dim='3D')
    else:
        field_3D = create_field(stacker, pType, nPixels, projection, dim='3D')


    if save3D:
        save_data(field_3D, stacker.simType, stacker.sim_index, pType, nPixels, 
                  projection, 'field', mask=False, maskRad=3.0, base_path=base_path, dim='3D')

    # sz_types = ['tSZ', 'kSZ', 'tau']
    
    # if pType in sz_types:
    #     field_3D = make_sz_field(stacker, pType, nPixels, projection=projection, dim='3D')
    # elif pType == 'total':
    #     field_3D = make_total_field(stacker, pType, nPixels, projection=projection, dim='3D')
    # else:
    #     field_3D = make_mass_field(stacker, pType, nPixels, projection=projection, dim='3D')
   
    # Now we do masking
    kpcPerPixel = stacker.header['BoxSize'] / nPixels # kpc/h per pixel
    GroupPos = halo_cat['GroupPos'] # kpc/h
    GroupRad = halo_cat['GroupRad'] # kpc/h
    
    GroupPos_masked = np.round(GroupPos / kpcPerPixel).astype(int)
    GroupRad_masked = GroupRad / kpcPerPixel
    cutout_mask = get_cutout_mask_3d(field_3D, GroupPos_masked, GroupRad_masked)
    field_3D_masked = field_3D * cutout_mask
    
    if dim == '3D':
        return field_3D_masked
    # Project to 2D
    
    if projection == 'xy':
        field_2D_masked = np.sum(field_3D_masked, axis=2)
    elif projection == 'xz':
        field_2D_masked = np.sum(field_3D_masked, axis=1)
    elif projection == 'yz':
        field_2D_masked = np.sum(field_3D_masked, axis=0)
    else:
        raise NotImplementedError('Projection type not implemented: ' + projection)
    # Finalize the 2D masked field
    return field_2D_masked