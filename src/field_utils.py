import argparse
import gc
import os
from pathlib import Path
import warnings

import asdf
import numpy as np
import yaml
import numba
#from np.fft import fftfreq, fftn, ifftn
from scipy.fft import rfftn, irfftn

from abacusnbody.metadata import get_meta
from abacusnbody.analysis.power_spectrum import calc_pk_from_deltak #computes power spectrum from density contrast, not specific to abacus 

from asdf.exceptions import AsdfWarning
warnings.filterwarnings('ignore', category=AsdfWarning)
from nbodykit.lab import *
# DEFAULTS = {'path2config': 'config/abacus_heft.yaml'}

def compress_asdf(asdf_fn, table, header):
    r"""Compress data dictionaries into an ASDF file using BLOSC compression.

    Args:
        asdf_fn (str): Output filename for the ASDF file.
        table (dict): Dictionary of field data arrays to compress.
        header (dict): Metadata header dictionary.
    """
    # cram into a dictionary
    data_dict = {}
    for field in table.keys():
        data_dict[field] = table[field]

    # create data tree structure
    data_tree = {
        "data": data_dict,
        "header": header,
    }

    # set compression options here
    compression_kwargs=dict(typesize="auto", shuffle="shuffle", compression_block_size=12*1024**2, blosc_block_size=3*1024**2, nthreads=4)
    with asdf.AsdfFile(data_tree) as af, open(asdf_fn, 'wb') as fp: # where data_tree is the ASDF dict tree structure
        af.write_to(fp)#, all_array_compression='blsc', compression_kwargs=compression_kwargs)

def load_lagrangians(path):
    """Load initial (Lagrangian) positions for the MillenniumTNG simulation.

    Reads and concatenates 10 serialized NumPy position files from snapshot 000.

    Args:
        path (str): Directory path containing the Lagrangian position files.

    Returns:
        np.ndarray: Concatenated initial positions, shape (N,).
    """
    lagrangians = []
    for i in range(10):
        i = i+1
        # lagrangians.append(np.load(path + 'lagrangian_position_sorted_264_MTNG-L500-1080-A_part{}_of_10.npy'.format(i)))
        lagrangians.append(np.load(path + 'position_sorted_000_MTNG-L500-1080-A_part{}_of_10.npy'.format(i)))
    return np.concatenate(lagrangians)

def load_positions(z, path):
    """Load comoving positions for the MillenniumTNG simulation at redshift z.

    Reads and concatenates 10 serialized NumPy position files.

    Args:
        z (float): Redshift; must be one of {0.0, 0.5, 1.0}.
        path (str): Directory path containing the position files.

    Returns:
        np.ndarray: Concatenated comoving positions, shape (N,).

    Raises:
        Exception: If z is not one of the supported redshifts.
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    pos = []
    for i in range(10):
        i = i+1
        pos.append(np.load(path + 'position_sorted_' + string + '_MTNG-L500-1080-A_part{}_of_10.npy'.format(i)))
    return np.concatenate(pos)

def load_velocities(z, path):
    """Load velocities for the MillenniumTNG simulation at redshift z.

    Reads and concatenates 10 serialized NumPy velocity files.

    Args:
        z (float): Redshift; must be one of {0.0, 0.5, 1.0}.
        path (str): Directory path containing the velocity files.

    Returns:
        np.ndarray: Concatenated velocities, shape (N,).

    Raises:
        Exception: If z is not one of the supported redshifts.
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    vel = []
    for i in range(10):
        i = i+1
        vel.append(np.load(path + 'velocity_sorted_' + string + '_MTNG-L500-1080-A_part{}_of_10.npy'.format(i)))
    return np.concatenate(vel)

def load_tau(z, path):
    """Load the optical depth (tau) 3D field for the MillenniumTNG simulation.

    Args:
        z (float): Redshift; must be one of {0.0, 0.5, 1.0}.
        path (str): Directory path containing the tau field file.

    Returns:
        np.ndarray: 3D optical depth field.

    Raises:
        Exception: If z is not one of the supported redshifts.
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    result = np.load(path + 'tau_3d_snap_' + string + '.npy')
    return result

def load_Y_compton(z, path):
    """Load the Compton-Y 3D field for the MillenniumTNG simulation.

    Args:
        z (float): Redshift; must be one of {0.0, 0.5, 1.0}.
        path (str): Directory path containing the Compton-Y field file.

    Returns:
        np.ndarray: 3D Compton-Y (tSZ) field.

    Raises:
        Exception: If z is not one of the supported redshifts.
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    result = np.load(path + 'Y_compton_3d_snap_' + string + '.npy')
    return result

def load_GroupPos(z, path):
    """Load halo group center-of-mass positions for the MillenniumTNG simulation.

    Args:
        z (float): Redshift; must be one of {0.0, 0.5, 1.0}.
        path (str): Directory path containing the group position file.

    Returns:
        np.ndarray: Halo positions array.

    Raises:
        Exception: If z is not one of the supported redshifts.
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    result = np.load(path + 'GroupPos_fp_' + string + '.npy')
    return result


def load_GroupMass(z, path):
    """Load halo group masses (M_TopHat200) for the MillenniumTNG simulation.

    Args:
        z (float): Redshift; must be one of {0.0, 0.5, 1.0}.
        path (str): Directory path containing the group mass file.

    Returns:
        np.ndarray: Halo masses array.

    Raises:
        Exception: If z is not one of the supported redshifts.
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    result = np.load(path + 'Group_M_TopHat200_fp_' + string + '.npy')
    return result

def load_GroupRad(z, path):
    """Load halo group radii (R_TopHat200) for the MillenniumTNG simulation.

    Args:
        z (float): Redshift; must be one of {0.0, 0.5, 1.0}.
        path (str): Directory path containing the group radius file.

    Returns:
        np.ndarray: Halo radii array.

    Raises:
        Exception: If z is not one of the supported redshifts.
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    result = np.load(path + 'Group_R_TopHat200_fp_' + string + '.npy')
    return result

def load_SubhaloPos(z, path):
    """Load subhalo positions for the MillenniumTNG simulation.

    Args:
        z (float): Redshift; must be one of {0.0, 0.5, 1.0}.
        path (str): Directory path containing the subhalo position file.

    Returns:
        np.ndarray: Subhalo positions array.

    Raises:
        Exception: If z is not one of the supported redshifts.
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    result = np.load(path + 'SubhaloPos_fp_' + string + '.npy')
    return result

def load_SubhaloMassType(z, path):
    """Load subhalo masses by particle type for the MillenniumTNG simulation.

    Args:
        z (float): Redshift; must be one of {0.0, 0.5, 1.0}.
        path (str): Directory path containing the subhalo mass-type file.

    Returns:
        np.ndarray: Subhalo mass-by-type array, shape (N_subhalos, N_types).

    Raises:
        Exception: If z is not one of the supported redshifts.
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    result = np.load(path + 'SubhaloMassType_fp_' + string + '.npy')
    return result

def load_SubhaloGroupNr(z, path):
    """Load parent group index for each subhalo in the MillenniumTNG simulation.

    Args:
        z (float): Redshift; must be one of {0.0, 0.5, 1.0}.
        path (str): Directory path containing the subhalo group number file.

    Returns:
        np.ndarray: Parent group indices, shape (N_subhalos,), dtype int.

    Raises:
        Exception: If z is not one of the supported redshifts.
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    result = np.load(path + 'SubhaloGroupNr_fp_' + string + '.npy')
    return result


def make_cross_corr(field1, field2, Lbox=500, kmax=10):
    """Compute the cross-correlation coefficient r_cc(k) using nbodykit.

    Args:
        field1 (np.ndarray): First 3D density field, shape (N, N, N).
        field2 (np.ndarray): Second 3D density field, shape (N, N, N).
        Lbox (float, optional): Simulation box size in Mpc. Defaults to 500.
        kmax (float, optional): Maximum wavenumber in h/Mpc. Defaults to 10.

    Returns:
        tuple: (kk, r_cc) where kk is wavenumber bin centres, shape (K,),
            and r_cc = P_cross / sqrt(P1 * P2) is the cross-correlation
            coefficient, shape (K,).
    """
    mesh1 = ArrayMesh(field1, BoxSize=[Lbox]*3)
    mesh2 = ArrayMesh(field2, BoxSize=[Lbox]*3)

    r1 = FFTPower(mesh1, mode='1d', kmax=kmax)
    r2 = FFTPower(mesh2, mode='1d', kmax=kmax)
    cross = FFTPower(mesh1, second=mesh2, mode='1d', kmax=kmax)

    kk = cross.power['k'][1:]
    Pk1 = r1.power['power'].real[1:]
    Pk2 = r2.power['power'].real[1:]
    Pk_cross = cross.power['power'].real[1:]
    r_cc = Pk_cross / np.sqrt(Pk1 * Pk2)
    del mesh1, mesh2
    gc.collect()
    return kk, r_cc

def make_cross_corr2(field1, field2, Lbox=500, kmax=10, kbins=101):
    """Compute the cross-correlation coefficient r_cc(k) using the Abacus toolkit.

    Args:
        field1 (np.ndarray): First 3D density field, shape (N, N, N).
        field2 (np.ndarray): Second 3D density field, shape (N, N, N).
        Lbox (float, optional): Simulation box size in Mpc. Defaults to 500.
        kmax (float, optional): Maximum wavenumber in h/Mpc. Defaults to 10.
        kbins (int, optional): Number of wavenumber bins. Defaults to 101.

    Returns:
        tuple: (k_avg, r_cc) where k_avg is wavenumber bin centres, shape (K,),
            and r_cc = P_cross / sqrt(P1 * P2), shape (K,).
    """
    k_bin_edges = np.linspace(1e-2, kmax, kbins)
    mu_bin_edges = np.array([0., 1.])
    field1_fft = (rfftn(field1, workers=-1)/ np.complex64(field1.size))
    field2_fft = (rfftn(field2, workers=-1)/ np.complex64(field2.size))
    result1 = calc_pk_from_deltak(field1_fft, Lbox, k_bin_edges, mu_bin_edges)
    pk1 = result1['power']
    k_avg1 = result1['k_avg']
    
    result2 = calc_pk_from_deltak(field2_fft, Lbox, k_bin_edges, mu_bin_edges)
    pk2 = result2['power']
    k_avg2 = result2['k_avg']

    resultx = calc_pk_from_deltak(field1_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=field2_fft)
    pkx = resultx['power']
    k_avgx = resultx['k_avg']
    
    r_cc = pkx / np.sqrt(pk1 * pk2)
    return k_avg1, r_cc

def make_cross_corr3(field1, field2, k_bin_edges, Lbox=500):
    """Compute the cross-correlation coefficient r_cc(k) with custom bin edges.

    Uses the Abacus toolkit.

    Args:
        field1 (np.ndarray): First 3D density field, shape (N, N, N).
        field2 (np.ndarray): Second 3D density field, shape (N, N, N).
        k_bin_edges (np.ndarray): Wavenumber bin edges in h/Mpc, shape (K+1,).
        Lbox (float, optional): Simulation box size in Mpc. Defaults to 500.

    Returns:
        tuple: (k_avg, r_cc) where k_avg is wavenumber bin centres, shape (K,),
            and r_cc = P_cross / sqrt(P1 * P2), shape (K,).
    """
    mu_bin_edges = np.array([0., 1.])
    field1_fft = (rfftn(field1, workers=-1)/ np.complex64(field1.size))
    field2_fft = (rfftn(field2, workers=-1)/ np.complex64(field2.size))
    result1 = calc_pk_from_deltak(field1_fft, Lbox, k_bin_edges, mu_bin_edges)
    pk1 = result1['power']
    k_avg1 = result1['k_avg']
    
    result2 = calc_pk_from_deltak(field2_fft, Lbox, k_bin_edges, mu_bin_edges)
    pk2 = result2['power']
    k_avg2 = result2['k_avg']

    resultx = calc_pk_from_deltak(field1_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=field2_fft)
    pkx = resultx['power']
    k_avgx = resultx['k_avg']
    
    r_cc = pkx / np.sqrt(pk1 * pk2)
    return k_avg1, r_cc


def calc_power(field1, Lbox=500, kmax=10, kmin=1e-2, kbins=101, field2=None, Nmodes=False):
    """Calculate the power spectrum (or cross-power) using the Abacus toolkit.

    Args:
        field1 (np.ndarray): 3D density field, shape (N, N, N).
        Lbox (float, optional): Simulation box size in Mpc. Defaults to 500.
        kmax (float, optional): Maximum wavenumber in h/Mpc. Defaults to 10.
        kmin (float, optional): Minimum wavenumber in h/Mpc. Defaults to 1e-2.
        kbins (int, optional): Number of wavenumber bins. Defaults to 101.
        field2 (np.ndarray, optional): Second field for cross-power. If None,
            computes auto-power of field1. Defaults to None.
        Nmodes (bool, optional): If True, also return the number of modes per
            bin. Defaults to False.

    Returns:
        tuple: (k_avg, pk) or (k_avg, pk, nmodes) if Nmodes=True.
            k_avg is wavenumber bin centres in h/Mpc, shape (K,).
            pk is power spectrum in (Mpc/h)^3, shape (K,).
            nmodes is number of modes per bin, shape (K,).
    """
    k_bin_edges = np.linspace(kmin, kmax, kbins)
    mu_bin_edges = np.array([0., 1.])
    
    field1_fft = (rfftn(field1, workers=-1)/ np.complex64(field1.size))
    if field2 is None:
        result = calc_pk_from_deltak(field1_fft, Lbox, k_bin_edges, mu_bin_edges)
    else:
        field2_fft = (rfftn(field2, workers=-1)/ np.complex64(field2.size))
        result = calc_pk_from_deltak(field1_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=field2_fft)

    pk1 = result['power']
    k_avg1 = result['k_avg']
    
    if not Nmodes:
        return k_avg1, pk1
    else:
        Nmodes = result['N_mode']
        return k_avg1, pk1, Nmodes


def calc_power2(field1, k_bin_edges, Lbox=500, field2=None, Nmodes=False):
    """Calculate the power spectrum with custom bin edges using the Abacus toolkit.

    Args:
        field1 (np.ndarray): 3D density field, shape (N, N, N).
        k_bin_edges (np.ndarray): Wavenumber bin edges in h/Mpc, shape (K+1,).
        Lbox (float, optional): Simulation box size in Mpc. Defaults to 500.
        field2 (np.ndarray, optional): Second field for cross-power. If None,
            computes auto-power of field1. Defaults to None.
        Nmodes (bool, optional): If True, also return the number of modes per
            bin. Defaults to False.

    Returns:
        tuple: (k_avg, pk) or (k_avg, pk, nmodes) if Nmodes=True.
            k_avg is wavenumber bin centres in h/Mpc, shape (K,).
            pk is power spectrum in (Mpc/h)^3, shape (K,).
            nmodes is number of modes per bin, shape (K,).
    """
    mu_bin_edges = np.array([0., 1.])
    
    field1_fft = (rfftn(field1, workers=-1)/ np.complex64(field1.size))
    if field2 is None:
        result = calc_pk_from_deltak(field1_fft, Lbox, k_bin_edges, mu_bin_edges)
    else:
        field2_fft = (rfftn(field2, workers=-1)/ np.complex64(field2.size))
        result = calc_pk_from_deltak(field1_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=field2_fft)

    pk1 = result['power']
    k_avg1 = result['k_avg']
    
    if not Nmodes:
        return k_avg1, pk1
    else:
        Nmodes = result['N_mode']
        return k_avg1, pk1, Nmodes

def gaussian_filter(field, nmesh, lbox, kcut):
    """Apply a Gaussian smoothing filter to a 3D field in Fourier space.

    Multiplies the FFT of the field by exp(-k^2 / (2 * kcut^2)) and
    returns the inverse FFT as a smoothed real-space field.

    Args:
        field (np.ndarray): 3D field to filter, shape (nmesh, nmesh, nmesh).
        nmesh (int): Size of the mesh along each dimension.
        lbox (float): Simulation box size in Mpc/h.
        kcut (float): Gaussian cutoff wavenumber in h/Mpc.

    Returns:
        np.ndarray: Gaussian-filtered field, shape (nmesh, nmesh, nmesh),
            dtype float32.
    """

    # fourier transform field
    field_fft = rfftn(field, workers=-1).astype(np.complex64)

    # inverse fourier transform
    f_filt = irfftn(filter_field(field_fft, nmesh, lbox, kcut), workers=-1).astype(np.float32)
    return f_filt

@numba.njit(parallel=True, fastmath=True)
def filter_field(delta_k, n1d, L, kcut, dtype=np.float32):
    r"""Apply a Gaussian smoothing filter to a Fourier-space field in-place.

    Multiplies each mode by exp(-k^2 / (2 * kcut^2)). JIT-compiled with
    Numba for performance.

    Args:
        delta_k (np.ndarray): Fourier-space field, shape (n1d, n1d, n1d//2+1).
            Modified in place.
        n1d (int): Size of the mesh along the first two dimensions.
        L (float): Simulation box size in Mpc/h.
        kcut (float): Gaussian cutoff wavenumber in h/Mpc.
        dtype (np.dtype, optional): Float type for intermediate calculations.
            Defaults to np.float32.

    Returns:
        np.ndarray: Filtered Fourier-space field (same object as delta_k).
    """
    # define number of modes along last dimension
    kzlen = n1d//2 + 1
    numba.get_num_threads()
    dk = dtype(2. * np.pi / L)
    norm = dtype(2. * kcut**2)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        kx = dtype(i)*dk if i < n1d//2 else dtype(i - n1d)*dk
        for j in range(n1d):
            ky = dtype(j)*dk if j < n1d//2 else dtype(j - n1d)*dk
            for k in range(kzlen):
                kz = dtype(k)*dk
                kmag2 = (kx**2 + ky**2 + kz**2)
                delta_k[i, j, k] = np.exp(-kmag2 / norm) * delta_k[i, j, k]
    return delta_k


@numba.njit(parallel=True, fastmath=True)
def get_n2_fft(delta_k, n1d, L, dtype=np.float32):
    r"""Compute nabla^2 delta in Fourier space as -k^2 * delta_k.

    JIT-compiled with Numba for performance.

    Args:
        delta_k (np.ndarray): Fourier-space field, shape (n1d, n1d, n1d//2+1).
        n1d (int): Size of the mesh along the first two dimensions.
        L (float): Simulation box size in Mpc/h.
        dtype (np.dtype, optional): Float type for intermediate calculations.
            Defaults to np.float32.

    Returns:
        np.ndarray: Fourier-space Laplacian field -k^2 * delta_k,
            shape (n1d, n1d, n1d//2+1), same dtype as delta_k.
    """
    # define number of modes along last dimension
    kzlen = n1d//2 + 1
    numba.get_num_threads()
    dk = dtype(2. * np.pi / L)

    # initialize field
    n2_fft = np.zeros((n1d, n1d, kzlen), dtype=delta_k.dtype)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        kx = dtype(i)*dk if i < n1d//2 else dtype(i - n1d)*dk
        for j in range(n1d):
            ky = dtype(j)*dk if j < n1d//2 else dtype(j - n1d)*dk
            for k in range(kzlen):
                kz = dtype(k)*dk
                kmag2 = (kx**2 + ky**2 + kz**2)
                n2_fft[i, j, k] = -kmag2 * delta_k[i, j, k]
    return n2_fft

@numba.njit(parallel=True, fastmath=True)
def get_sij_fft(i_comp, j_comp, delta_k, n1d, L, dtype=np.float32):
    r"""Compute the (i, j) component of the tidal tensor in Fourier space.

    Computes s_ij = (k_i k_j / k^2 - delta_ij / 3) * delta_k.
    JIT-compiled with Numba for performance.

    Args:
        i_comp (int): First tensor index (0=x, 1=y, 2=z).
        j_comp (int): Second tensor index (0=x, 1=y, 2=z).
        delta_k (np.ndarray): Fourier-space field, shape (n1d, n1d, n1d//2+1).
        n1d (int): Size of the mesh along the first two dimensions.
        L (float): Simulation box size in Mpc/h.
        dtype (np.dtype, optional): Float type for intermediate calculations.
            Defaults to np.float32.

    Returns:
        np.ndarray: Fourier-space tidal tensor component s_ij,
            shape (n1d, n1d, n1d//2+1), same dtype as delta_k.
    """
    # define number of modes along last dimension
    kzlen = n1d//2 + 1
    numba.get_num_threads()
    dk = dtype(2. * np.pi / L)
    if i_comp == j_comp:
        delta_ij_over_3 = dtype(1./3.)
    else:
        delta_ij_over_3 = dtype(0.)

    # initialize field
    s_ij_fft = np.zeros((n1d, n1d, kzlen), dtype=delta_k.dtype)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        kx = dtype(i)*dk if i < n1d//2 else dtype(i - n1d)*dk
        if i_comp == 0:
            ki = kx
        if j_comp == 0:
            kj = kx
        for j in range(n1d):
            ky = dtype(j)*dk if j < n1d//2 else dtype(j - n1d)*dk
            if i_comp == 1:
                ki = ky
            if j_comp == 1:
                kj = ky
            for k in range(kzlen):
                kz = dtype(k)*dk
                if i + j + k > 0:
                    kmag2_inv = dtype(1.)/(kx**2 + ky**2 + kz**2)
                else:
                    kmag2_inv = dtype(0.)
                if i_comp == 2:
                    ki = kz
                if j_comp == 2:
                    kj = kz
                s_ij_fft[i, j, k] = delta_k[i, j, k] * (ki *kj * kmag2_inv - delta_ij_over_3)
    return s_ij_fft

@numba.njit(parallel=True, fastmath=True)
def add_ij(final_field, field_to_add, n1d, factor=1., dtype=np.float32):
    r"""Accumulate factor * field_to_add^2 into final_field in-place.

    JIT-compiled with Numba for performance.

    Args:
        final_field (np.ndarray): Accumulator field, shape (n1d, n1d, n1d).
            Modified in place.
        field_to_add (np.ndarray): Field to square and accumulate,
            shape (n1d, n1d, n1d).
        n1d (int): Size of the mesh along each dimension.
        factor (float, optional): Multiplicative scale factor. Defaults to 1.
        dtype (np.dtype, optional): Float type. Defaults to np.float32.
    """
    factor = dtype(factor)
    for i in numba.prange(n1d):
        for j in range(n1d):
            for k in range(n1d):
                final_field[i, j, k] += factor * field_to_add[i, j, k]**2
    return

def get_dk_to_s2(delta_k, nmesh, lbox):
    r"""Compute the squared tidal field s^2 = s_ij s_ij from the density FFT.

    Computes all 6 independent components of the symmetric tidal tensor
    s_ij = (k_i k_j / k^2 - delta_ij / 3) * delta_k, inverse-transforms
    each, and sums their squares (with factor 2 for off-diagonal terms).

    Args:
        delta_k (np.ndarray): Fourier-transformed density field,
            shape (nmesh, nmesh, nmesh//2+1).
        nmesh (int): Size of the mesh along each dimension.
        lbox (float): Simulation box size in Mpc/h.

    Returns:
        np.ndarray: Real-space squared tidal field s^2,
            shape (nmesh, nmesh, nmesh), dtype float32.
    """
    # Compute the symmetric tide at every Fourier mode which we'll reshape later
    # Order is xx, xy, xz, yy, yz, zz
    jvec = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    # compute s_ij and do the summation
    tidesq = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
    for i in range(len(jvec)):
        if jvec[i][0] != jvec[i][1]:
            factor = 2.
        else:
            factor = 1.
        add_ij(tidesq, irfftn(get_sij_fft(jvec[i][0], jvec[i][1], delta_k, nmesh, lbox), workers=-1), nmesh, factor)
    return tidesq

def get_dk_to_n2(delta_k, nmesh, lbox):
    """Compute the Laplacian of the density field: nabla^2 delta = IFFT(-k^2 delta_k).

    Args:
        delta_k (np.ndarray): Fourier-transformed density field,
            shape (nmesh, nmesh, nmesh//2+1).
        nmesh (int): Size of the mesh along each dimension.
        lbox (float): Simulation box size in Mpc/h.

    Returns:
        np.ndarray: Real-space Laplacian field nabla^2 delta,
            shape (nmesh, nmesh, nmesh), dtype float32.
    """
    # Compute -k^2 delta which is the gradient
    nabla2delta = irfftn(get_n2_fft(delta_k, nmesh, lbox), workers=-1).astype(np.float32)
    return nabla2delta

def get_fields(delta_lin, Lbox, nmesh):
    """Compute delta, delta^2, s^2, and nabla^2 delta from the linear density field.

    All output fields are mean-subtracted. Memory is freed aggressively
    during computation to minimize peak usage.

    Args:
        delta_lin (np.ndarray): Linear density field, shape (nmesh, nmesh, nmesh).
        Lbox (float): Simulation box size in Mpc/h.
        nmesh (int): Size of the mesh along each dimension.

    Returns:
        tuple: (d, d2, s2, n2) where each is an np.ndarray of shape
            (nmesh, nmesh, nmesh), dtype float32:
            - d: Mean-subtracted density contrast delta.
            - d2: Mean-subtracted delta^2.
            - s2: Mean-subtracted squared tidal field s_ij s^ij.
            - n2: Laplacian nabla^2 delta (not mean-subtracted).
    """

    # get delta
    delta_fft = rfftn(delta_lin, workers=-1).astype(np.complex64)
    fmean = np.mean(delta_lin, dtype=np.float64)
    d = delta_lin-fmean
    gc.collect()
    print("Generated delta")

    # get delta^2
    d2 = delta_lin * delta_lin
    fmean = np.mean(d2, dtype=np.float64)
    d2 -= fmean
    del delta_lin
    gc.collect()
    print("Generated delta^2")

    # get s^2
    s2 = get_dk_to_s2(delta_fft, nmesh, Lbox)
    fmean = np.mean(s2, dtype=np.float64)
    s2 -= fmean
    print("Generated s_ij s^ij")

    # get n^2
    n2 = get_dk_to_n2(delta_fft, nmesh, Lbox)
    print("Generated nabla^2")

    return d, d2, s2, n2