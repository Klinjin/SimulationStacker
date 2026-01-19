import numpy as np

def halo_ind(ind):
    if ind == 0:
        return 5e11, 1e12, r'$5\times 10^{11} M_\odot < M_{\rm halo} < 10^{12} M_\odot$, '
    elif ind == 1:
        return 1e12, 1e13, r'$1\times 10^{12} M_\odot < M_{\rm halo} < 10^{13} M_\odot$, '
    elif ind == 2:
        return 1e13, 1e19, r'$1\times 10^{13} M_\odot < M_{\rm halo} < 10^{19} M_\odot$, '
    else:
        raise ValueError("Wrong ind")


def select_massive_halos(halo_masses, Boxsize, number_density, upper_mass_bound=None):
    halo_masses = np.asarray(halo_masses)
    
    # Apply upper mass bound filter
    if upper_mass_bound is not None:
        valid_mask = halo_masses <= upper_mass_bound
        filtered = halo_masses[valid_mask]
        valid_indices = np.where(valid_mask)[0]
    else:
        filtered = halo_masses
        valid_indices = np.arange(len(halo_masses))

    # Calculate mass threshold based on number density
    target_count = int(number_density * (Boxsize/1e3)**3)
    target_count = min(target_count, len(filtered))
    
    if target_count == 0:
        # Return empty indices and empty masses for consistency
        return np.array([], dtype=int), np.array([], dtype=halo_masses.dtype)
    
    # Find mass threshold
    mass_threshold = np.partition(filtered, -target_count)[-target_count]
    
    # Select halos above mass threshold
    mass_condition = filtered >= mass_threshold

    
    return valid_indices[mass_condition], filtered[mass_condition]


def filter_edge_halo(haloPos, Boxsize, maxRadius):
    """Check if a single halo is too close to box edges for CAP filter.

    Returns True if the halo should be filtered out (too close to edge),
    False if the halo is valid (far enough from edges).

    Parameters:
    - haloPos: array-like of shape (2,) or (3,); single halo position in pixels.
               Only the first two components (x, y) are used.
    - Boxsize: float or int; box size in pixels (assumed square box).
    - maxRadius: float; maximum CAP radius in pixels.

    Returns:
    - bool: True if halo should be filtered out, False if halo is valid.
    """
    pos = np.asarray(haloPos)
    if pos.ndim != 1:
        raise ValueError("haloPos must be a 1D array for a single halo")
    
    if pos.shape[0] < 2:
        raise ValueError("haloPos must have at least two coordinates (x,y)")
    
    x, y = pos[0], pos[1]
    margin = float(maxRadius)

    # Filter out if within margin of any edge
    too_close_x = (x < margin) or (x > float(Boxsize) - margin)
    too_close_y = (y < margin) or (y > float(Boxsize) - margin)
    
    return too_close_x or too_close_y


