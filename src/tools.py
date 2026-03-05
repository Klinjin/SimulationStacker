import numpy as np
import numba
from numba import njit
from nbodykit.lab import ArrayCatalog, FieldMesh
#from nbodykit.base.mesh import MeshFilter

#from pixell import enmap, enplot, utils
#import rotfuncs


@njit(nogil=True, parallel=False)
def hist2d_numba_seq(tracks, bins, ranges, weights=np.empty(0), dtype=np.float32):
    """Numba-JIT 2D histogram, 8-9x faster than np.histogram2d.

    Precompiling the signature enables faster repeated calls. The nogil flag
    means this could in principle be threaded.

    Args:
        tracks (np.ndarray): Particle coordinates, shape (2, N). Row 0 is the
            first axis, row 1 is the second axis.
        bins (array-like): Number of bins along each axis, length 2.
        ranges (np.ndarray): Bin ranges, shape (2, 2). Row i is [min_i, max_i].
        weights (np.ndarray, optional): Per-particle weights, shape (N,).
            If empty (default), particles are counted with unit weight.
        dtype (np.dtype, optional): Accumulator dtype. Defaults to np.float32.

    Returns:
        np.ndarray: 2D histogram array, shape (bins[0], bins[1]), dtype float64.
    """
    #H = np.zeros((bins[0], bins[1]), dtype=np.uint64)
    H = np.zeros((bins[0], bins[1]), dtype=np.float64)
    delta = 1/((ranges[:,1] - ranges[:,0]) / bins)
    Nw = len(weights)

    for t in range(tracks.shape[1]):
        i = (tracks[0,t] - ranges[0,0]) * delta[0]
        j = (tracks[1,t] - ranges[1,0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            if Nw == 1:
                H[int(i),int(j)] += 1.
            else:
                H[int(i),int(j)] += weights[t]

    return H


@numba.vectorize
def rightwrap(x, L):
    """Wrap a coordinate to [0, L) by subtracting L if x >= L.

    Vectorized Numba ufunc for periodic right-side wrapping.

    Args:
        x (float): Coordinate value to wrap.
        L (float): Box/grid size defining the periodic boundary.

    Returns:
        float: Wrapped coordinate, equal to x if x < L, or x - L otherwise.
    """
    if x >= L:
        return x - L
    return x

@njit
def dist(pos1, pos2, L=None):
    """Calculate L2 norm distances between a set of points and a reference.

    Computes pairwise distances between pos1 and pos2, with optional periodic
    wrapping. JIT-compiled with Numba for performance.

    Args:
        pos1 (np.ndarray): Set of points, shape (N, m).
        pos2 (np.ndarray): Reference point(s), shape (N, m), (m,), or (1, m).
            If shape is (1, m) or (m,), the single point is broadcast against
            all rows of pos1.
        L (float, optional): Box size for periodic wrapping. If given, distances
            are computed with minimum image convention. Defaults to None.

    Returns:
        np.ndarray: Distances, shape (N,).
    """
    
    # read dimension of data
    N, nd = pos1.shape
    
    # allow pos2 to be a single point
    pos2 = np.atleast_2d(pos2)
    assert pos2.shape[-1] == nd
    broadcast = len(pos2) == 1
    
    dist = np.empty(N, dtype=pos1.dtype)
    
    i2 = 0
    for i in range(N):
        delta = 0.
        for j in range(nd):
            dx = pos1[i][j] - pos2[i2][j]
            if L is not None:
                if dx >= L/2:
                    dx -= L
                elif dx < -L/2:
                    dx += L
            delta += dx*dx
        dist[i] = np.sqrt(delta)
        if not broadcast:
            i2 += 1
    return dist

@numba.jit(nopython=True, nogil=True)
def numba_tsc_3D(positions, density, boxsize, weights=np.empty(0)):
    """Deposit particle masses/weights onto a 3D grid using Triangular Shape Cloud (TSC) interpolation.

    Accumulates weighted particle contributions into a 3D density grid
    using TSC (quadratic spline) kernel. Supports both 2D and 3D grids
    (detected automatically from grid shape). Uses periodic boundary
    conditions via the rightwrap helper.

    Args:
        positions (np.ndarray): Particle positions in simulation units,
            shape (N, 3). Must be in [0, boxsize).
        density (np.ndarray): Output density grid to accumulate into,
            shape (gx, gy, gz). Modified in place. Set gz=1 for 2D.
        boxsize (float): Simulation box size in the same units as positions.
        weights (np.ndarray, optional): Per-particle weights, shape (N,).
            If empty (default), each particle contributes weight 1.
            If length 1, that scalar weight is broadcast to all particles.

    Returns:
        np.ndarray: The updated density grid (same object as input density).
    """
    gx = np.uint32(density.shape[0])
    gy = np.uint32(density.shape[1])
    gz = np.uint32(density.shape[2])
    threeD = gz != 1
    W = 1.
    Nw = len(weights)
    for n in range(len(positions)):
        # broadcast scalar weights
        if Nw == 1:
            W = weights[0]
        elif Nw > 1:
            W = weights[n]
        
        # convert to a position in the grid
        px = (positions[n,0]/boxsize)*gx # used to say boxsize+0.5
        py = (positions[n,1]/boxsize)*gy # used to say boxsize+0.5
        if threeD:
            pz = (positions[n,2]/boxsize)*gz # used to say boxsize+0.5
        
        # round to nearest cell center
        ix = np.int32(round(px))
        iy = np.int32(round(py))
        if threeD:
            iz = np.int32(round(pz))
        
        # calculate distance to cell center
        dx = ix - px
        dy = iy - py
        if threeD:
            dz = iz - pz
        
        # find the tsc weights for each dimension
        wx = .75 - dx**2
        wxm1 = .5*(.5 + dx)**2
        wxp1 = .5*(.5 - dx)**2
        wy = .75 - dy**2
        wym1 = .5*(.5 + dy)**2
        wyp1 = .5*(.5 - dy)**2
        if threeD:
            wz = .75 - dz**2
            wzm1 = .5*(.5 + dz)**2
            wzp1 = .5*(.5 - dz)**2
        else:
            wz = 1.
        
        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixm1 = (ix - 1)
        ixw  = rightwrap(ix    , gx)
        ixp1 = rightwrap(ix + 1, gx)
        iym1 = (iy - 1)
        iyw  = rightwrap(iy    , gy)
        iyp1 = rightwrap(iy + 1, gy)
        if threeD:
            izm1 = (iz - 1)
            izw  = rightwrap(iz    , gz)
            izp1 = rightwrap(iz + 1, gz)
        else:
            izw = np.uint32(0)
        
        # change the 9 or 27 cells that the cloud touches
        density[ixm1, iym1, izw ] += wxm1*wym1*wz  *W
        density[ixm1, iyw , izw ] += wxm1*wy  *wz  *W
        density[ixm1, iyp1, izw ] += wxm1*wyp1*wz  *W
        density[ixw , iym1, izw ] += wx  *wym1*wz  *W
        density[ixw , iyw , izw ] += wx  *wy  *wz  *W
        density[ixw , iyp1, izw ] += wx  *wyp1*wz  *W
        density[ixp1, iym1, izw ] += wxp1*wym1*wz  *W
        density[ixp1, iyw , izw ] += wxp1*wy  *wz  *W
        density[ixp1, iyp1, izw ] += wxp1*wyp1*wz  *W
        
        if threeD:
            density[ixm1, iym1, izm1] += wxm1*wym1*wzm1*W
            density[ixm1, iym1, izp1] += wxm1*wym1*wzp1*W

            density[ixm1, iyw , izm1] += wxm1*wy  *wzm1*W
            density[ixm1, iyw , izp1] += wxm1*wy  *wzp1*W

            density[ixm1, iyp1, izm1] += wxm1*wyp1*wzm1*W
            density[ixm1, iyp1, izp1] += wxm1*wyp1*wzp1*W

            density[ixw , iym1, izm1] += wx  *wym1*wzm1*W
            density[ixw , iym1, izp1] += wx  *wym1*wzp1*W

            density[ixw , iyw , izm1] += wx  *wy  *wzm1*W
            density[ixw , iyw , izp1] += wx  *wy  *wzp1*W

            density[ixw , iyp1, izm1] += wx  *wyp1*wzm1*W
            density[ixw , iyp1, izp1] += wx  *wyp1*wzp1*W

            density[ixp1, iym1, izm1] += wxp1*wym1*wzm1*W
            density[ixp1, iym1, izp1] += wxp1*wym1*wzp1*W

            density[ixp1, iyw , izm1] += wxp1*wy  *wzm1*W
            density[ixp1, iyw , izp1] += wxp1*wy  *wzp1*W

            density[ixp1, iyp1, izm1] += wxp1*wyp1*wzm1*W
            density[ixp1, iyp1, izp1] += wxp1*wyp1*wzp1*W