# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

SimulationStacker is a Python toolkit for halo stacking analysis on cosmological simulations (IllustrisTNG and SIMBA). It creates 2D/3D projected fields from particle data, applies stacking filters, and generates comparison figures between simulations and observations.

## Running Scripts

Scripts are run from the `scripts/` directory with a YAML config file:
```bash
cd scripts/
python make_ratios3x2.py -p configs/ratios_3x2_z05.yaml
```

SLURM batch submission:
```bash
sbatch scripts/runCPU.sh
```

## Testing

Tests are Jupyter notebooks in `tests/`. There is no pytest suite. Validate changes interactively:
```bash
jupyter notebook tests/test_3D_fields.ipynb
```

## Architecture

**`src/` is not an installed package** — scripts import by adding `../src/` to `sys.path`. The core library has these layers:

1. **`stacker.py`** — `SimulationStacker` class, the main entry point. Holds simulation metadata (sim name, snapshot, redshift, nPixels), orchestrates field creation and stacking, and manages caching via HDF5 files.

2. **`mapMaker.py`** — Low-level field creation from particle data. Called by `stacker.makeField()`. Supports particle types: `gas`, `dm`, `stars`, `bh`, `tSZ`, `kSZ`, `tau`, `ionized_gas`, `baryon`, `total`.

3. **`filters.py`** — Stacking filter functions: `CAP` (Circular Aperture Profile, the primary method), `DSigma`/`delta_sigma` (excess surface mass density), `total_mass`, `upsilon`. These operate on 2D projected fields and return radial profiles.

4. **`loadIO.py`** — Data I/O for TNG and SIMBA group catalogs and particle data. Differences between the two sims are handled here (e.g., halo radius field names differ). Uses `illustris_python` for TNG; reads HDF5 directly for SIMBA.

5. **`tools.py`** — Numba-JIT-compiled histogram binning (`hist2d_numba_seq`, `numba_tsc_3D`). Performance-critical; avoid touching unless necessary.

6. **`field_utils.py`**, **`utils.py`**, **`mask_utils.py`**, **`halos.py`** — Supporting utilities for field statistics, cosmology, masking, and halo selection.

7. **`SZstacker.py`** — Deprecated SZ-specific stacker subclass; prefer the parent `SimulationStacker` for new work.

## Key Public API: SimulationStacker

```python
s = SimulationStacker(sim='TNG300-1', snapshot=67, nPixels=1000,
                      simType='IllustrisTNG', feedback=None, z=0.5)
```

The two main workflows are **fields** (simulation-native units, kpc/h radii) and **maps** (beam-convolved, arcmin radii):

| Method | Purpose |
|--------|---------|
| `makeField(pType, nPixels, projection, save, load, dim)` | Create raw 2D/3D projected field from particles |
| `makeMap(pType, z, projection, beamSize, pixelSize, save, load)` | Create beam-convolved 2D map |
| `stackField(pType, filterType, minRadius, maxRadius, numRadii, ...)` | Stack field and return radial profile |
| `stackMap(pType, filterType, minRadius, maxRadius, numRadii, ...)` | Stack beam-convolved map |
| `stack_on_array(array, filterType, radDistance, radDistanceUnits, ...)` | Stack any 2D array directly |
| `setField(pType, field_, nPixels, projection)` | Inject a precomputed field |

`filterType` options: `'CAP'` (default), `'cumulative'`, `'DSigma'`, `'ringring'`, `'upsilon'`, `'DSigma_mccarthy'`.

## Data Caching

Precomputed fields and maps are cached as `.npy` files under:
```
/pscratch/sd/r/rhliu/simulations/{IllustrisTNG|SIMBA}/products/{2D|3D}/
```
Filename convention: `{sim}_{snapshot}_{pType}_{nPixels}_{projection}.npy`

Use `load=True` / `save=True` in `makeField()`/`makeMap()` to control read/write. Results are also cached in-memory in `stacker.fields` and `stacker.maps`.

## Unit Conventions

| Quantity | IllustrisTNG | SIMBA |
|----------|-------------|-------|
| Mass | 10^10 M☉/h | M☉/h |
| Positions | ckpc/h | ckpc/h |
| Halo radius field | `Group_R_Mean200` | `virial_quantities.r200c` |
| DM particle mass | from `MassTable[1]` (no per-particle field) | per-particle field |

SZ fields (tSZ, kSZ, τ) are computed from gas `ElectronAbundance`, `InternalEnergy`, `Density`, and `Velocities`, and use CGS constants defined in `mapMaker.py`.

## Configuration System

All analysis scripts read YAML configs. Key config sections:

```yaml
stack:
  redshift: 0.5
  projection: 'yz'          # 'xy', 'xz', or 'yz'
  load_field: true           # Load cached HDF5 field
  save_field: true           # Save computed field to HDF5
  particle_type: 'gas'
  n_pixels: 1000
  halo_mass_min: 1.665e13    # Msun
  filter_type: 'CAP'         # 'CAP', 'cumulative', 'DSigma'
plot:
  fig_path: '../figures/'
  fig_name: 'output_name'
simulations:
  - sim_type: 'IllustrisTNG'
    sims:
      - name: 'TNG300-1'
        snapshot: 67
  - sim_type: 'SIMBA'
    sims:
      - name: 'm100n1024'
        snapshot: 136
        feedback: 's50'
```

## Supported Simulations

- **IllustrisTNG**: `TNG300-1`, `TNG300-2`, `TNG100-1`, `TNG100-2`, `TNG50-1`, `Illustris-1`, `Illustris-2`
- **SIMBA**: `m50n512`, `m100n1024` with feedback variants `s50`, `s50nox`, `s50noagn`, `s50nofb`, `s50nojet`

Simulation data lives at hardcoded paths in `stacker.py` under `/pscratch/sd/r/rhliu/simulations/` — this is known technical debt, do not refactor without being asked.

## Pre-push Code Review Hook

`.claude/hooks/pre-push` runs an autonomous Claude code review before pushes. It is opt-in:
```bash
REVIEW=1 git push
```
Without `REVIEW=1`, the hook exits immediately without reviewing.
