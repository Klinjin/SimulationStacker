# Cell for Delta Sigma profiles by number density
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# Volume of the simulation box
box_volume = 50**3  # (Mpc/h)^3

# Number densities to plot
number_densities_to_plot = [1.0e-4, 2.8e-4, 5e-4, 1.0e-3, 2.4e-3]

# Calculate number of halos for each density
n_halos_per_density = [int(nd * box_volume) for nd in number_densities_to_plot]
print("Number of halos for each density:", n_halos_per_density)

# Extract colors from prop_cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
cycle_colors = prop_cycle.by_key()['color']

# Get radial bins
radii = r1_mpch  # in Mpc/h

# Number density labels
nd_labels = [
    r'$n = 1.0 \times 10^{-4}$ (Mpc/h)$^{-3}$',
    r'$n = 2.8 \times 10^{-4}$ (Mpc/h)$^{-3}$',
    r'$n = 5.0 \times 10^{-4}$ (Mpc/h)$^{-3}$',
    r'$n = 1.0 \times 10^{-3}$ (Mpc/h)$^{-3}$',
    r'$n = 2.4 \times 10^{-3}$ (Mpc/h)$^{-3}$'
]
nd_colors = [cycle_colors[i] for i in range(5)]

fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH_DOUBLE, 12))
axes = axes.flatten()

# For each number density
for nd_idx, (n_halos, nd_label) in enumerate(zip(n_halos_per_density, nd_labels)):
    ax = axes[nd_idx]
    
    # Collect profiles from all simulations for this number density
    all_profiles_this_nd = []
    
    for sim_id in range(len(delta_sigma_xy_profiles_all_sims)):
        # Select first n_halos (halos are already sorted by descending mass)
        # Collect profiles from all three projections
        profiles_xy = delta_sigma_xy_profiles_all_sims[sim_id][:, :n_halos]
        profiles_xz = delta_sigma_xz_profiles_all_sims[sim_id][:, :n_halos]
        profiles_yz = delta_sigma_yz_profiles_all_sims[sim_id][:, :n_halos]
        
        # Concatenate along the halo dimension
        profiles_this_nd = np.concatenate([profiles_xy, profiles_xz, profiles_yz], axis=1)
        all_profiles_this_nd.append(profiles_this_nd)
    
    if len(all_profiles_this_nd) > 0:
        # Concatenate all profiles across all simulations
        all_profiles_concatenated = np.concatenate(all_profiles_this_nd, axis=1)
        all_profiles_concatenated *= 1e-6  # Convert from kpc^-2 to pc^-2
        
        # Calculate median and percentiles
        median_profile = np.nanmedian(all_profiles_concatenated, axis=1)
        p16_profile = np.nanpercentile(all_profiles_concatenated, 16, axis=1)
        p84_profile = np.nanpercentile(all_profiles_concatenated, 84, axis=1)
        
        # Plot median with shaded region
        ax.plot(radii[:len(median_profile)], median_profile, 
               color=nd_colors[nd_idx], linewidth=2, 
               label=f'Median ({all_profiles_concatenated.shape[1]} halos)')
        ax.fill_between(radii[:len(median_profile)], 
                       p16_profile, 
                       p84_profile, 
                       color=nd_colors[nd_idx], alpha=0.3)
        
        print(f"Number density {nd_idx}: {all_profiles_concatenated.shape[1]} total halo profiles (from {len(all_profiles_this_nd)} sims × 3 projections × {n_halos} halos)")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('R [Mpc/h]')
    ax.set_ylabel(r'$\Delta \Sigma (M_{total}) [M_\odot h / \mathrm{pc}^2]$')
    ax.legend(loc='upper right')
    ax.grid(True, which='both')
    ax.set_title(nd_label)

# Hide the last subplot if we have fewer than 6 densities
if len(number_densities_to_plot) < 6:
    axes[-1].axis('off')

fig.tight_layout()
plt.show()
fig.savefig(FIG_SAVE_DIR + 'DeltaSigma_profiles_by_number_density.pdf', bbox_inches='tight')


# Cell for T_kSZ profiles by number density
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# Get radial bins
radii = r0_arcmin  # in arcmin

# Extract colors from prop_cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
cycle_colors = prop_cycle.by_key()['color']

# Number density labels
nd_labels = [
    r'$n = 1.0 \times 10^{-4}$ (Mpc/h)$^{-3}$',
    r'$n = 2.8 \times 10^{-4}$ (Mpc/h)$^{-3}$',
    r'$n = 5.0 \times 10^{-4}$ (Mpc/h)$^{-3}$',
    r'$n = 1.0 \times 10^{-3}$ (Mpc/h)$^{-3}$',
    r'$n = 2.4 \times 10^{-3}$ (Mpc/h)$^{-3}$'
]
nd_colors = [cycle_colors[i] for i in range(5)]

fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH_DOUBLE, 12))
axes = axes.flatten()

# For each number density
for nd_idx, (n_halos, nd_label) in enumerate(zip(n_halos_per_density, nd_labels)):
    ax = axes[nd_idx]
    
    # Collect profiles from all simulations for this number density
    all_profiles_this_nd = []
    
    for sim_id in range(len(ksz_profiles_xy_all_sims)):
        # Select first n_halos (halos are already sorted by descending mass)
        # Collect profiles from all three projections
        profiles_xy = ksz_profiles_xy_all_sims[sim_id][:, :n_halos]
        profiles_xz = ksz_profiles_xz_all_sims[sim_id][:, :n_halos]
        profiles_yz = ksz_profiles_yz_all_sims[sim_id][:, :n_halos]
        
        # Concatenate along the halo dimension
        profiles_this_nd = np.concatenate([profiles_xy, profiles_xz, profiles_yz], axis=1)
        all_profiles_this_nd.append(profiles_this_nd)
    
    if len(all_profiles_this_nd) > 0:
        # Concatenate all profiles across all simulations
        all_profiles_concatenated = np.concatenate(all_profiles_this_nd, axis=1)
        
        # Calculate median and percentiles
        median_profile = np.nanmedian(all_profiles_concatenated, axis=1)
        p16_profile = np.nanpercentile(all_profiles_concatenated, 16, axis=1)
        p84_profile = np.nanpercentile(all_profiles_concatenated, 84, axis=1)
        
        # Plot median with shaded region
        ax.plot(radii[:len(median_profile)], median_profile, 
               color=nd_colors[nd_idx], linewidth=2, 
               label=f'Median ({all_profiles_concatenated.shape[1]} halos)')
        ax.fill_between(radii[:len(median_profile)], 
                       p16_profile, 
                       p84_profile, 
                       color=nd_colors[nd_idx], alpha=0.3)
        
        print(f"Number density {nd_idx}: {all_profiles_concatenated.shape[1]} total halo profiles (from {len(all_profiles_this_nd)} sims × 3 projections × {n_halos} halos)")
    
    ax.set_xlabel('R [arcmin]')
    ax.set_ylabel(r'$T_{kSZ}$ [$\mu K \rm{arcmin}^2$]')
    ax.set_yscale('log')
    ax.legend(loc='lower right')
    ax.grid(True, which='both')
    ax.set_title(nd_label)

# Hide the last subplot if we have fewer than 6 densities
if len(number_densities_to_plot) < 6:
    axes[-1].axis('off')

fig.tight_layout()
plt.show()
fig.savefig(FIG_SAVE_DIR + 'kSZ_profiles_by_number_density.pdf', bbox_inches='tight')
