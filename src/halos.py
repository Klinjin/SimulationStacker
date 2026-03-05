import numpy as np

def halo_ind(ind):
    """Return mass bin boundaries and label string for a given bin index.

    Args:
        ind (int): Mass bin index. Must be 0, 1, or 2.

    Returns:
        tuple: A 3-tuple (mass_min, mass_max, label) where mass_min and
            mass_max are the lower and upper halo mass bounds in M☉, and
            label is a LaTeX-formatted string describing the bin.

    Raises:
        ValueError: If ind is not 0, 1, or 2.
    """
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

    
    return valid_indices[mass_condition]


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


def select_abundance_subhalos(halo_masses, target_number, Lbox):
    """Select the most massive halos to match a target number density.

    Sorts halos by mass in descending order and selects the top N halos
    such that N / box_volume matches the target number density.

    Args:
        halo_masses (array-like): Array of halo masses.
        target_number (float): Target number density for the selected halos.
            Units: (cMpc/h)^-3.
        Lbox (float): Length of the simulation box in ckpc/h.

    Returns:
        np.ndarray: Integer indices into halo_masses for the selected halos,
            sorted by decreasing mass, shape (N_selected,).
    """
    box_volume = (Lbox / 1e3) ** 3  # Convert ckpc/h to cMpc/h
    Ngal = int(target_number * box_volume) # Total number of galaxies desired
    
    
    idx_selected = np.argsort(halo_masses)[::-1][:Ngal]
    
    # sorted_m = halo_masses[order]
    # cum_counts = np.arange(1, len(sorted_m) + 1)
    # cum_number_density = cum_counts / box_volume

    # idx = np.searchsorted(cum_number_density, target_number, side='right')
    # if idx == 0:
    #     raise ValueError("No subset of halos meets the target number density.")
    # cutoff = idx
    # selected = order[:cutoff]
    return idx_selected
