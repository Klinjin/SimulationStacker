import os 
import sys
import argparse
import yaml
sys.path.append('/pscratch/sd/l/lindajin/SimulationStacker/src/')
from stacker import *
from utils import ksz_from_delta_sigma, arcmin_to_comoving, comoving_to_arcmin
from mpi4py import MPI 
import numpy as np
from tqdm import tqdm

# Parse CLI arguments for config path
parser = argparse.ArgumentParser(description='Process config.')
parser.add_argument('-p', '--path2config', type=str, default='./second_data_stacker_configs.yaml', help='Path to the configuration file.')
args = vars(parser.parse_args())

# Load YAML config
path2config = args['path2config']
with open(path2config, 'r') as f:
    config = yaml.safe_load(f)

# Stacking parameters
sim_type = config.get('sim_type', 'IllustrisTNG')
snapshot = config.get('snapshot', 74)
sim_name = config.get('sim_name', 'L50n512_SB35')

loadField = config.get('load_field', False)
saveField = config.get('save_field', False)
radDistance = config.get('rad_distance', 1.0)
projection_list = config.get('projection', 'xy')

filterType0 = config.get('filter_type', 'CAP')
pType0 = config.get('particle_type', 'tau')

filterType1 = config.get('filter_type_2', 'DSigma')
pType1 = config.get('particle_type_2', 'total')

# Radius and density parameters - ensure proper numeric types
minRadius_arcmin = config.get('min_radius', 0.14)
maxRadius_arcmin = config.get('max_radius', 6.0)
minRadius_arcmin = float(eval(minRadius_arcmin)) if isinstance(minRadius_arcmin, str) else float(minRadius_arcmin)
maxRadius_arcmin = float(eval(maxRadius_arcmin)) if isinstance(maxRadius_arcmin, str) else float(maxRadius_arcmin)
numRadii = int(config.get('num_radii', 12))

number_density = config.get('number_density',  2.4e-3)
number_density = float(eval(number_density)) if isinstance(number_density, str) else float(number_density)

nPixels = int(config.get('n_pixels', 1000))
pixelSize = float(config.get('pixel_size', 0.5))
radDistance = float(radDistance)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create simulation IDs from 0 to 1023
train_sim_ids = np.arange(1024)

# Only rank 0 prints initial info
if rank == 0:
    print(f"Starting MPI analysis with {size} processes")
    print(f"Processing {len(train_sim_ids)} simulations for z =0.47 (snapshot {snapshot})")

    # Print actual parameters being used (after defaults applied)
    print("\n" + "="*60)
    print("Configuration loaded from:", path2config)
    print("="*60)
    print(f"  load_field: {loadField}")
    print(f"  save_field: {saveField}")
    print(f"  rad_distance: {radDistance}")
    print(f"  projection: {projection_list}")
    print(f"  filter_type: {filterType0}")
    print(f"  particle_type: {pType0}")
    print(f"  filter_type_2: {filterType1}")
    print(f"  particle_type_2: {pType1}")
    print(f"  min_radius: {minRadius_arcmin}")
    print(f"  max_radius: {maxRadius_arcmin}")
    print(f"  num_radii: {numRadii}")
    print(f"  number_density: {number_density}")
    print(f"  n_pixels: {nPixels}")
    print(f"  pixel_size: {pixelSize}")
    print("="*60 + "\n")

# Distribute simulations across MPI processes
sim_per_rank = len(train_sim_ids) // size
remainder = len(train_sim_ids) % size

# Calculate start and end indices for this rank
if rank < remainder:
    start_id = rank * (sim_per_rank + 1)
    end_id = start_id + sim_per_rank + 1
else:
    start_id = rank * sim_per_rank + remainder
    end_id = start_id + sim_per_rank

# Only rank 0 prints distribution info
if rank == 0:
    print(f"Simulations distribution: {sim_per_rank} base + {remainder} extra")
    for r in range(size):
        if r < remainder:
            s = r * (sim_per_rank + 1)
            e = s + sim_per_rank + 1
        else:
            s = r * sim_per_rank + remainder
            e = s + sim_per_rank
        print(f"  Rank {r}: Sims {s} to {e-1} ({e-s} simulations)")

# Wait for rank 0 to finish printing
comm.Barrier()

# Get the simulation indices for this rank
local_sim_indices = train_sim_ids[start_id:end_id]
num_local_sims = len(local_sim_indices)

# Process simulations assigned to this rank
for i, sim_i in enumerate(tqdm(local_sim_indices, desc=f"Rank {rank}")):
    try:
        print(f'Rank {rank}: Reading profiles for simulation {sim_type}_{sim_i} ({i+1}/{num_local_sims})')
        profile_data = {}
        # Initialize stacker for this simulation
        stacker = SimulationStacker(sim_index=sim_i, 
                 snapshot=snapshot, 
                 simType=sim_type)
        
        # stacker.get_field_baryon_suppression(save=True) # DONE: @Sat 29 Nov 2025 04:35:36 AM PST (tested) 

        minRad_mpch = arcmin_to_comoving(minRadius_arcmin, stacker.z, stacker.cosmo) / 1000.0
        maxRad_mpch = arcmin_to_comoving(maxRadius_arcmin, stacker.z, stacker.cosmo) / 1000.0
        arcmin_per_kpc = comoving_to_arcmin(1.0, stacker.z, cosmo=stacker.cosmo)
        area_conv = arcmin_per_kpc ** 2  # [arcmin^2] per [kpc/h]^2

        for projection in (projection_list if isinstance(projection_list, list) else [projection_list]):

            radii0, profiles0 = stacker.stackMap(pType0, filterType=filterType0, minRadius=minRadius_arcmin, maxRadius=maxRadius_arcmin, numRadii=numRadii,
                                                save=saveField, load=loadField, radDistance=radDistance, halo_number_density=number_density,
                                                projection=projection, pixelSize=pixelSize)
            radii1, profiles1 = stacker.stackField(pType1, filterType=filterType1, minRadius=minRad_mpch, maxRadius=maxRad_mpch, numRadii=numRadii, # type: ignore
                                                    save=saveField, load=loadField, radDistance=1000, nPixels=nPixels,
                                                    projection=projection, halo_number_density=number_density)
            profiles1_arcmin2 = np.array(profiles1, dtype=np.float64) / area_conv

            # Save each number density as a separate key
            profile_data[f'prof0_{pType0}_{filterType0}_{projection}'] = profiles0
            profile_data[f'prof1_{pType1}_{filterType1}_arcmin2_{projection}'] = profiles1_arcmin2
            profile_data[f'prof1_{pType1}_{filterType1}_kpch2_{projection}'] = profiles1

        radii0_mpch = arcmin_to_comoving(radii0, stacker.z, stacker.cosmo) / 1000.0
        radii1_arcmin = comoving_to_arcmin(radii1 * 1000.0, stacker.z, cosmo=stacker.cosmo)
        
        # Add radii and metadata to profile_data
        profile_data.update({
            f'r0_arcmin': radii0,
            f'r0_to_mpch': radii0_mpch,
            f'r1_mpch': radii1,
            f'r1_to_arcmin': radii1_arcmin,
            'halo_masses': stacker.halo_mass_selected,
            'number_density': number_density,
            'nPixels01': [stacker.nPixels_map, nPixels],
            'pixelSize01': [pixelSize, stacker.pixelSize_true],
            'methods01': ['stackMap', 'stackField'],
            'units01': ['arcmin', 'kpc/h'],
            'fb': stacker.fb,
            'HubbleParam':stacker.h
        })

        # Ensure save directory exists
        save_dir = os.path.join(stacker.sim_path, 'data')
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, 
                    f'Profiles_{pType0}-{filterType0}_{pType1}-{filterType1}.npz')
        np.savez(save_path, **profile_data)
        print(f'Rank {rank}: Saved profiles to {save_path}')
                 
    except Exception as e:
        print(f"ERROR: Rank {rank} failed on sim_i={sim_i}", flush=True)
        print(f"Exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Continue to next simulation instead of crashing
        continue

# Synchronize all processes before finishing
comm.Barrier()

if rank == 0:
    print("All processes completed successfully!")