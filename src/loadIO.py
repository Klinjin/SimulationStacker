import h5py
import halos
import numpy as np
import glob
import illustris_python as il
from pathlib import Path


def snap_path(sim_path, snapshot, sim_type, sim_name=None, feedback=None, chunk_num=0, path_only=False):
    """Get the snapshot path for the given simulation type.

    Args:
        sim_path (str): Base path to the simulation.
        snapshot (int): Snapshot number.
        sim_type (str): The type of simulation (e.g., 'IllustrisTNG', 'SIMBA').
        sim_name (str, optional): Name of the simulation (for SIMBA).
        feedback (str, optional): Feedback type (for SIMBA).
        chunk_num (int): The chunk number for the simulation (Only used for IllustrisTNG).
        path_only (bool): If True, returns only the folder path.

    Returns:
        str: The path to the snapshot file or folder.
    """
    if sim_type == 'IllustrisTNG':
        folder_path = sim_path + '/snapdir_' + str(snapshot).zfill(3) + '/'
        snap_path = il.snapshot.snapPath(sim_path, snapshot, chunkNum=chunk_num)
    elif sim_type == 'SIMBA':
        folder_path = sim_path + 'snapshots/'
        snap_path = folder_path + 'snap_' + sim_name + '_' + str(snapshot) + '.hdf5'
        folder_path = snap_path  # SIMBA has different file structure
    
    if path_only:
        return folder_path
    else:
        return snap_path


def load_halos(sim_path, snapshot, sim_type, sim_name=None, header=None):
    """Load halo data for the specified simulation type.

    Args:
        sim_path (str): Base path to the simulation.
        snapshot (int): Snapshot number.
        sim_type (str): The type of simulation (e.g., 'IllustrisTNG', 'SIMBA').
        sim_name (str, optional): Name of the simulation (for SIMBA).
        header (dict, optional): Simulation header (required for SIMBA).

    Returns:
        dict: A dictionary containing halo properties (e.g., mass, position, radius).
    """
    if sim_type == 'IllustrisTNG':
        haloes = {}
        haloes_cat = il.groupcat.loadHalos(sim_path, snapshot)
        # haloes['GroupPos'] = haloes_cat['GroupPos']
        haloes['GroupRad'] = haloes_cat['Group_R_TopHat200']
        sub_h = haloes_cat['GroupFirstSub'][:]
        central_subhalo_ids = sub_h[np.where(sub_h >= 0)]

        fields = ['SubhaloMass','SubhaloMassInMaxRadType','SubhaloVmaxRad', 'SubhaloMassInRadType','SubhaloPos','SubhaloGrNr','SubhaloLen']
        subhalos = il.groupcat.loadSubhalos(sim_path, snapshot, fields=fields)
        haloes['SubhaloMass']   = subhalos['SubhaloMass'][central_subhalo_ids] * 1e10   
        haloes['SubhaloPos'] = subhalos['SubhaloPos'][central_subhalo_ids]
        haloes['SubhaloGrNr'] = subhalos['SubhaloGrNr'][central_subhalo_ids]
        haloes['SubhaloStellarMassInRadType'] = subhalos['SubhaloMassInRadType'][central_subhalo_ids,4] * 1e10
        
        grp_ids = subhalos['SubhaloGrNr'][central_subhalo_ids]
        haloes['GroupMass'] = haloes_cat['Group_M_Crit500'][grp_ids] * 1e10  # Convert to Msun/h
    del haloes_cat 
    return haloes

def load_subhalos(sim_path, snapshot, sim_type, sim_name=None, header=None):
    """Load subhalo data for the specified simulation type.

    Args:
        sim_path (str): Base path to the simulation.
        snapshot (int): Snapshot number.
        sim_type (str): The type of simulation (e.g., 'IllustrisTNG', 'SIMBA').
        sim_name (str, optional): Name of the simulation (for SIMBA).
        header (dict, optional): Simulation header (required for SIMBA).

    Returns:
        dict: A dictionary containing subhalo properties (e.g., mass, position).
    """
    if sim_type == 'IllustrisTNG':
        subhaloes = {}
        subhaloes_cat = il.groupcat.loadSubhalos(sim_path, snapshot)
        
        subhaloes['SubhaloPos'] = subhaloes_cat['SubhaloPos']
        subhaloes['SubhaloMass'] = subhaloes_cat['SubhaloMass'] * 1e10  # Convert to Msun/h
        subhaloes['SubhaloGrNr'] = subhaloes_cat['SubhaloGrNr']
        # subhaloes['SubhaloID'] = subhaloes_cat['SubhaloID']
        subhaloes['SubhaloMStar'] = subhaloes_cat['SubhaloMassType'][:, 4] * 1e10  # Stellar mass in Msun/h
        
    elif sim_type == 'SIMBA':
        if header is None:
            raise ValueError("Header is required for SIMBA simulations")
        
        subhalo_path = sim_path + 'catalogs/' + sim_name + '_' + str(snapshot) + '.hdf5'
        subhaloes = {}
        # CAESAR identifies galaxies with a 6D Friends-of-Friends (6DFOF)
        # algorithm that runs only on baryonic particles (gas + stars + BH).
        # DM is excluded from the galaxy finder by design, so
        # galaxy_data['dicts']['masses.total'] is a purely baryonic mass
        # (~10^10–10^11 Msun) — NOT analogous to TNG's SubhaloMass, which
        # includes all gravitationally bound particles (DM + baryons).
        subhaloes_cat = load_as_dict(subhalo_path, 'galaxy_data')

        # parent_halo_index: confirmed in CAESAR docs and source as the index
        # into halo_data for each galaxy's parent FoF halo.
        parent_idx = subhaloes_cat['parent_halo_index']

        subhaloes['SubhaloPos']   = subhaloes_cat['pos'] * header['HubbleParam']  # kpc/h

        # SubhaloMass proxy for SHAM: baryonic mass + DM within a 30 kpc aperture.
        #
        # There is no CAESAR equivalent to TNG's SubhaloMass (SUBFIND total
        # bound mass including DM).  Three options were considered:
        #
        #   Option 1 — baryonic mass only (masses.total, ~10^11 Msun):
        #     Unique per galaxy; allows satellite selection; but 100× lower
        #     than the mass-cut selection scale, so SHAM selects a different
        #     population entirely.
        #
        #   Option 2 — baryonic + 30 kpc aperture DM (masses.total + masses.dm_30kpc):
        #     Still unique per galaxy; includes a local DM contribution that
        #     partially captures the galaxy's DM environment; physically
        #     closer to SUBFIND's satellite-subhalo mass than baryons alone.
        #     Currently in use (chosen as the best available approximation), while
        #     SHAM selection in stack_on_array uses SubhaloMStar (stellar mass)
        #
        #   Option 3 — parent FoF halo mass:
        #     Right scale for centrals (~10^13 Msun) but degenerate — all
        #     galaxies in the same halo share the same mass, so SHAM selects
        #     all galaxies from the few most massive halos rather than a
        #     representative mix of centrals and satellites.
        #
        # NOTE: dm_30kpc is a spherical aperture sum, not a bound-mass
        # calculation.  For deeply embedded satellites the aperture may
        # include DM belonging to the host rather than the satellite itself.
        # This is a known limitation with no better alternative in CAESAR.
        subhaloes['SubhaloMass']  = (
            subhaloes_cat['dicts']['masses.dm_30kpc'] +
            subhaloes_cat['dicts']['masses.total']
        ) * header['HubbleParam']                                                   # Msun/h

        subhaloes['SubhaloGrNr']  = parent_idx                                     # confirmed key
        subhaloes['SubhaloID']    = subhaloes_cat['GroupID']
        subhaloes['SubhaloMStar'] = subhaloes_cat['dicts']['masses.stellar'] * header['HubbleParam']  # Msun/h

    return subhaloes

def load_subsets(sim_path, snapshot, sim_type, p_type, sim_name=None, feedback=None, header=None, keys=None):
    """Load particle subsets for the specified particle type.

    Args:
        sim_path (str): Base path to the simulation.
        snapshot (int): Snapshot number.
        sim_type (str): The type of simulation.
        p_type (str): The type of particles to load (e.g., 'gas', 'DM', 'Stars', 'tSZ', 'kSZ', 'tau').
        sim_name (str, optional): Name of the simulation (for SIMBA).
        feedback (str, optional): Feedback type (for SIMBA).
        header (dict, optional): Simulation header (required for mass conversion).
        keys (list, optional): Specific keys to load. If None, defaults are used based on particle type.

    Returns:
        dict: A dictionary containing the particle properties.
    """
    if header is None:
        raise ValueError("Header is required for mass conversion")
    
    # Handle SZ particle types by mapping to gas with appropriate fields
    if p_type in ['tSZ', 'kSZ', 'tau']:
        actual_p_type = 'gas'
        if keys is None:
            keys = ['Coordinates', 'Masses', 'ElectronAbundance', 'InternalEnergy', 'Density', 'Velocities']
    else:
        actual_p_type = p_type
        if keys is None:
            keys = ['Coordinates', 'Masses']
        
    if sim_type == 'IllustrisTNG':
        if actual_p_type == 'gas':
            if keys is None:
                particles = il.snapshot.loadSubset(sim_path, snapshot, actual_p_type, fields=['Masses', 'Coordinates'])
            else:
                particles = il.snapshot.loadSubset(sim_path, snapshot, actual_p_type, fields=keys)
        elif actual_p_type == 'DM':
            # Handle DM case where we need ParticleIDs for mass
            dm_keys = list(keys) if keys else ['Coordinates', 'Masses']
            if 'Masses' in dm_keys:
                dm_keys[dm_keys.index('Masses')] = 'ParticleIDs'
                particles = il.snapshot.loadSubset(sim_path, snapshot, actual_p_type, fields=dm_keys)
                particles['Masses'] = header['MassTable'][1] * np.ones_like(particles['ParticleIDs'])
                del particles['ParticleIDs']
            else:
                particles = il.snapshot.loadSubset(sim_path, snapshot, actual_p_type, fields=dm_keys)
        elif actual_p_type in ['Stars', 'BH']:
            particles = il.snapshot.loadSubset(sim_path, snapshot, actual_p_type, fields=keys)
        else:
            raise NotImplementedError('Particle Type not implemented')
                                    
    elif sim_type == 'SIMBA':
        if actual_p_type == 'gas':
            p_type_val = 'PartType0'
        elif actual_p_type == 'DM':
            p_type_val = 'PartType1'
        elif actual_p_type == 'Stars':
            p_type_val = 'PartType4'
        elif actual_p_type == 'BH':
            p_type_val = 'PartType5'
        else:
            raise NotImplementedError('Particle Type not implemented')
        
        snap_path = sim_path + 'snapshots/snap_' + sim_name + '_' + str(snapshot) + '.hdf5'
        particles = {}
        with h5py.File(snap_path, 'r') as f:
            for key in keys:
                particles[key] = f[p_type_val][key][:] # type: ignore
        
    # particles['Masses'] = particles['Masses'] * 1e10 / header['HubbleParam']  # Convert masses to Msun/h
    return particles


def load_subset(snap_file, p_type, sim_type='IllustrisTNG', header=None, keys=None, sim_name=None):
    """Load a subset of particles from a specific snapshot file.

    Args:
        sim_path (str): Base path to the simulation.
        snapshot (int): Snapshot number.
        sim_type (str): The type of simulation.
        p_type (str): The type of particles to load (e.g., 'gas', 'DM', 'Stars', 'tSZ', 'kSZ', 'tau').
        snap_path (str): The path to the snapshot file.
        header (dict, optional): Simulation header (required for mass conversion).
        keys (list, optional): The keys to load from the snapshot.
        sim_name (str, optional): Name of the simulation (for SIMBA).

    Returns:
        dict: A dictionary containing the particle properties.
    """
    if keys is None:
        keys = ['Coordinates', 'Masses']
    read_keys = list(keys)  # copy so we can mutate safely
    
    if header is None:
        raise ValueError("Header is required for mass conversion")
    
    # Handle SZ particle types by mapping to gas
    if p_type in ['tSZ', 'kSZ', 'tau']:
        actual_p_type = 'gas'
    else:
        actual_p_type = p_type
    
    add_mass = False  # Handle IllustrisTNG DM case without Masses field
    if actual_p_type == 'gas':
        p_type_val = 'PartType0'
    elif actual_p_type == 'DM':
        p_type_val = 'PartType1'
        if 'Masses' in read_keys and sim_type == 'IllustrisTNG':
            read_keys[read_keys.index('Masses')] = 'ParticleIDs'
            add_mass = True
    elif actual_p_type == 'Stars':
        p_type_val = 'PartType4'
    elif actual_p_type == 'BH':
        p_type_val = 'PartType5'
    else:
        raise NotImplementedError(f'Particle Type not implemented: {actual_p_type}')

    particles = {}
    with h5py.File(snap_file, 'r') as f:
        file_header = dict(f['Header'].attrs.items())
        for key in read_keys:
            particles[key] = f[p_type_val][key][:] # type: ignore

    if add_mass:
        particles['Masses'] = header['MassTable'][1] * np.ones_like(particles['ParticleIDs'])  # DM mass

    if (not 'ParticleIDs' in keys) and ('ParticleIDs' in particles):
        del particles['ParticleIDs']  # Remove ParticleIDs if we added Masses
            
    # particles['Masses'] = particles['Masses'] * 1e10 / header['HubbleParam']  # Convert masses to Msun/h
    return particles


def _get_data_filepath(sim_type, sim_index, p_type, n_pixels,
                       projection='xy', data_type='field', dim='2D',
                       mask=False, maskRad=2.0, base_path=None, file_type='npy'):
    """Generate the file path for saving/loading data.

    Args:
        sim_type (str): The type of simulation.
        sim_name (str): Name of the simulation.
        snapshot (int): Snapshot number.
        feedback (str): Feedback type (for SIMBA).
        p_type (str): Particle type.
        n_pixels (int): Number of pixels.
        projection (str): Projection direction.
        data_type (str): Type of data ('field' or 'map').
        dim (str): Dimension of the data ('2D' or '3D').
        mask (bool): Whether to apply halo masking.
        maskRad (float): Radius for masking in units of R200c.
        base_path (str): Base path to the directory.

    Returns:
        Path: Full file path for the data file.
    """
    if base_path is None:
        base_path = '/pscratch/sd/l/lindajin/CAMELS/'
    
    if dim == '3D' and data_type == 'map':
        raise ValueError("3D maps are not supported. Please use 'field' for 3D data.")
    
    # Build suffix
    suffix = f'_{dim}map' if data_type == 'map' else f'_{dim}field'
    if mask:
        suffix += f'_masked{maskRad}R200c'
    
    # Build filename
    if sim_type == 'IllustrisTNG':
        if dim == '2D':
            filename = f'{p_type}_{n_pixels}_{projection}{suffix}.{file_type}'
        else:  # 3D
            filename = f'{p_type}_{n_pixels}{suffix}.{file_type}'
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")
    
    # Build directory path
    dir_path = Path(f'{base_path}/{sim_type}/L50n512_SB35/SB35_{sim_index}/data/')

    
    return dir_path / filename


def load_data(sim_type, sim_index, p_type, n_pixels, 
              projection='xy', data_type='field', dim='2D', 
              mask=False, maskRad=2.0,
              base_path=None):
    """Load a precomputed field or map from file.

    Args:
        sim_type (str): The type of simulation.
        sim_index (int): Simulation index.
        p_type (str): Particle type.
        n_pixels (int): Number of pixels.
        projection (str): Projection direction.
        data_type (str): Type of data to load ('field' or 'map').
        dim (str): Dimension of the data ('2D' or '3D'). For 3D, only 'field' is supported.
        mask (bool): Whether to apply halo masking.
        maskRad (float): Radius for masking in units of R200c.
        base_path (str): Base path to the directory containing data.

    Returns:
        np.ndarray: 2D numpy array of the field or map.
    """
    filepath = _get_data_filepath(sim_type, sim_index, p_type, n_pixels,
                                   projection, data_type, dim, mask, maskRad, base_path)
    
    try:
        data = np.load(filepath)
    except FileNotFoundError:
        raise ValueError(f"Data file '{filepath}' not found. Please compute it first.")

    return data


def save_data(data, sim_type, sim_index, p_type, n_pixels, 
              projection='xy', data_type='field', dim='2D',
              mask=False, maskRad=2.0,
              base_path=None, mkdir=True, file_type='npy'):
    """Save a field or map to file.

    Args:
        data (np.ndarray): 2D array to save OR a tuple of arrays for npz.
        sim_type (str): The type of simulation.
        sim_index (int): Simulation index.
        p_type (str): Particle type.
        n_pixels (int): Number of pixels.
        projection (str): Projection direction.
        data_type (str): Type of data ('field' or 'map').
        dim (str): Dimension of the data ('2D' or '3D'). For 3D, only 'field' is supported.
        mask (bool): Whether to apply halo masking.
        maskRad (float): Radius for masking in units of R200c.
        base_path (str): Base path to the directory for saving data.
        mkdir (bool): Whether to create the directory if it doesn't exist.
    Returns:
        None
    """
    filepath = _get_data_filepath(sim_type, sim_index, p_type, n_pixels,
                                   projection, data_type, dim, mask, maskRad, base_path, file_type)
    
    if mkdir:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    print('Saving data to:', filepath)
    if file_type == 'npz':
        # Support dicts and sequences for multi-array saving
        if isinstance(data, dict):
            np.savez(filepath, **data)
        elif isinstance(data, (list, tuple)):
            np.savez(filepath, *data)
        else:
            # Save single array under key 'arr_0' by default
            np.savez(filepath, data=data)
    else:
        np.save(filepath, data)


def load_group_to_dict(group):
    """Recursively load an HDF5 group into nested dicts of NumPy arrays.

    Args:
        group (h5py.Group): The HDF5 group to load, e.g. f['galaxy_data'].

    Returns:
        dict: Nested dict with datasets as NumPy arrays and subgroups as dicts.
    """
    out = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            out[key] = item[:]  # fully load dataset into memory
        elif isinstance(item, h5py.Group):
            out[key] = load_group_to_dict(item)
        # you can add an `else` branch if you care about other HDF5 object types
    return out

def load_as_dict(file_path, group_name):
    """Load an HDF5 file group into nested dicts of NumPy arrays.

    Args:
        file_path (str): Path to the HDF5 file.
        group_name (str): Name of the group to load.

    Returns:
        dict: Nested dict with datasets as NumPy arrays and subgroups as dicts.
    """
    with h5py.File(file_path, 'r') as f:
        group = f[group_name]
        return load_group_to_dict(group)