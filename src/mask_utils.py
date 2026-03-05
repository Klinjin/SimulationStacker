import numpy as np
import time

def precompute_offsets_3d(radius):
    """Precompute relative (dz, dy, dx) offsets within a sphere of the given radius.

    Args:
        radius (float): Sphere radius in voxel units.

    Returns:
        np.ndarray: Offset array of shape (N, 3). Each row is (dz, dy, dx).
            All offsets satisfy dz^2 + dy^2 + dx^2 <= radius^2.
    """
    r = int(np.ceil(radius))
    dz, dy, dx = np.meshgrid(
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        indexing='ij'
    )
    dist_sq = dz**2 + dy**2 + dx**2
    mask = dist_sq <= radius**2
    return np.stack((dz[mask], dy[mask], dx[mask]), axis=1)

def get_cutout_mask_3d(array, centers, radii):
    """Create a 3D boolean mask indicating voxels within specified radii of centers.

    Uses periodic boundary conditions (modulo wrap) on all three axes.

    Args:
        array (np.ndarray): 3D array whose shape defines the grid bounds,
            shape (nz, ny, nx).
        centers (list of tuple): Center coordinates (z, y, x) for each aperture.
        radii (list of float): Radii in voxel units, one per center.

    Returns:
        np.ndarray: Boolean mask with the same shape as array. True where
            at least one center contributes a voxel within its radius.
    """
    _offset_cache = {}
    shape = np.array(array.shape)
    mask = np.zeros_like(array, dtype=bool)

    for center, radius in zip(centers, radii):
        center = np.array(center)

        if radius not in _offset_cache:
            _offset_cache[radius] = precompute_offsets_3d(radius)
        offsets = _offset_cache[radius]

        # Apply offsets and wrap with modulo
        candidate_indices = (center + offsets) % shape
        z, y, x = candidate_indices[:, 0], candidate_indices[:, 1], candidate_indices[:, 2]
        mask[z, y, x] = True

    return mask

def get_cutout_indices_3d(array, centers, radii):
    """Return the 3D voxel indices within specified radii of each center.

    Uses periodic boundary conditions (modulo wrap) on all three axes.

    Args:
        array (np.ndarray): 3D array whose shape defines the grid bounds,
            shape (nz, ny, nx).
        centers (list of tuple): Center coordinates (z, y, x) for each aperture.
        radii (list of float): Radii in voxel units, one per center.

    Returns:
        list of np.ndarray: One array per center-radius pair. Each array has
            shape (M_i, 3) where M_i is the voxel count within that radius.
            Each row is a (z, y, x) voxel coordinate.
    """
    _offset_cache = {}
    shape = np.array(array.shape)
    all_indices = []

    for center, radius in zip(centers, radii):
        center = np.array(center)

        if radius not in _offset_cache:
            _offset_cache[radius] = precompute_offsets_3d(radius)
        offsets = _offset_cache[radius]

        # Apply offsets and wrap with modulo
        candidate_indices = (center + offsets) % shape
        all_indices.append(candidate_indices)

    return all_indices

def sum_over_cutouts(array, all_indices):
    """Compute the sum of array values within each set of provided 3D indices.

    Args:
        array (np.ndarray): 3D input array, shape (nz, ny, nx).
        all_indices (list of np.ndarray): Index arrays, typically from
            get_cutout_indices_3d. Each element has shape (M_i, 3).

    Returns:
        np.ndarray: Summed values, shape (len(all_indices),). Element i
            contains the sum of array values at the voxels in all_indices[i].
    """
    result = np.empty(len(all_indices), dtype=array.dtype)
    for i, inds in enumerate(all_indices):
        z, y, x = inds[:, 0], inds[:, 1], inds[:, 2]
        result[i] = array[z, y, x].sum()
    return result


def precompute_offsets_2d(radius):
    """Precompute relative (dy, dx) offsets within a circle of the given radius.

    Args:
        radius (float): Circle radius in pixel units.

    Returns:
        np.ndarray: Offset array of shape (N, 2). Each row is (dy, dx).
            All offsets satisfy dy^2 + dx^2 <= radius^2.
    """
    r = int(np.floor(radius))
    dy, dx = np.meshgrid(
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        indexing='ij'
    )
    dist_sq = dy**2 + dx**2
    mask = dist_sq <= radius**2
    return np.stack((dy[mask], dx[mask]), axis=1)

def get_cutout_mask_2d(array, centers, radii):
    """Create a 2D boolean mask indicating pixels within specified radii of centers.

    Uses hard boundary conditions; pixels outside array bounds are excluded
    (no periodic wrapping).

    Args:
        array (np.ndarray): 2D array whose shape defines the pixel grid bounds,
            shape (ny, nx).
        centers (list of tuple): Center coordinates (row, col) for each aperture.
        radii (list of float): Radii in pixel units, one per center.

    Returns:
        np.ndarray: Boolean mask with the same shape as array. True where
            at least one center contributes a pixel within its radius.

    Note:
        Deprecated return value was a flat list of unique (row, col) indices.
        Current return value is a boolean mask.
    """
    _offset_cache_2d = {}
    shape = np.array(array.shape)
    unique_indices = set()

    for center, radius in zip(centers, radii):
        center = np.array(center)

        # Cache and reuse offset grid
        if radius not in _offset_cache_2d:
            _offset_cache_2d[radius] = precompute_offsets_2d(radius)
        offsets = _offset_cache_2d[radius]

        # Add offsets to center
        candidate_indices = center + offsets

        # Check bounds
        in_bounds = np.all((candidate_indices >= 0) & (candidate_indices < shape), axis=1)
        valid = candidate_indices[in_bounds]

        unique_indices.update(map(tuple, valid))

    list_indices = list(unique_indices)
    arr_ind = np.array(list_indices)
    indices = (arr_ind[:, 0], arr_ind[:, 1])
    mask = np.zeros_like(array, dtype=bool)
    mask[indices] = True
    return mask

    