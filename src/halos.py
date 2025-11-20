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
        return np.array([], dtype=int)
    
    # Find mass threshold
    mass_threshold = np.partition(filtered, -target_count)[-target_count]
    
    # Select halos above mass threshold
    mass_condition = filtered >= mass_threshold

    
    return valid_indices[mass_condition], [mass_threshold, filtered[mass_condition].max()]