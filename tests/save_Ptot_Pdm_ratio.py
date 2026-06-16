"""Save matter power suppression P_tot/P_dm and k arrays for snap74 and snap82.

Loads per-simulation Pk_512_3Dfield_snap{74,82}.npz files from CAMELS-TNG SB35
(1024 sims) and stores the suppression ratio P_tot(k)/P_dm(k) together with the
wavenumber array (converted h/kpc -> h/Mpc) into npz files under DH_profile_kSZ_WL/data.
"""

from pathlib import Path
import numpy as np

base_path_template = '/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/'
n_sims   = 1024
snap_ids = [74, 82]
out_dir  = Path('/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data')
out_dir.mkdir(parents=True, exist_ok=True)

for snap in snap_ids:
    k_ref      = None          # common k grid (h/Mpc)
    ratio_rows = []            # one row per sim (NaN row if missing)
    n_ok       = 0

    for sim_id in range(n_sims):
        fpath = Path(base_path_template.format(sim_id)) / f'Pk_512_3Dfield_snap{snap}.npz'
        if not fpath.exists():
            ratio_rows.append(None)
            continue
        d     = np.load(fpath)
        k_arr = d['k'] * 1e3                       # h/kpc -> h/Mpc
        ratio = d['P_tot'] / d['P_dm']
        if k_ref is None:
            k_ref = k_arr
        ratio_rows.append(ratio)
        n_ok += 1

    if k_ref is None:
        print(f'snap{snap}: no Pk files found, skipping')
        continue

    # Assemble aligned (n_sims, n_k) array; missing sims -> NaN rows.
    Ptot_Pdm_ratio = np.full((n_sims, k_ref.size), np.nan)
    for sim_id, ratio in enumerate(ratio_rows):
        if ratio is not None:
            Ptot_Pdm_ratio[sim_id] = ratio

    out_path = out_dir / f'Ptot_Pdm_ratio_snap{snap}.npz'
    np.savez(out_path, k=k_ref, Ptot_Pdm_ratio=Ptot_Pdm_ratio)
    print(f'snap{snap}: {n_ok}/{n_sims} sims loaded -> saved {out_path} '
          f'(k: {k_ref.size}, ratio: {Ptot_Pdm_ratio.shape})')
