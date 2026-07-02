#!/usr/bin/env python3
"""
Compute PTE (probability-to-exceed) and PTE-equivalent significance for
CAMELS simulation profiles fitted to DESI kSZ and HSC GGL observations.

Profile files used (pre-computed from the CAMELS SB35 seventh-gen Latin-
Hypercube run, 1024 independent cosmology+astrophysics realizations):
  mean_kSZ_profiles_nd_{nd_idx}_n_{n_halos}_snap{74,82}.npz
      prof        (20,)      : mean T_kSZ [μK·arcmin²] across all sims
      prof_std    (20,)      : std across sims [μK·arcmin²]
      cov_of_mean (20,20)    : Cov(prof) across the LH ensemble, i.e.
                               Cov(per-sim profile)/N_sims; REQUIRED for
                               the full-covariance (σ_total) fit.
      profiles    (N_sims,20): per-sim mean T_kSZ (over halos+projections),
                               kept for provenance/recomputation, not read
                               directly if cov_of_mean is present.
      n_sims      ()         : number of sims contributing to prof/cov_of_mean
      radii       (20,)      : aperture radii [arcmin]

  mean_DeltaSigma_profiles_nd_{nd_idx}_n_{n_halos}_snap{74,82}.npz
      prof        (20,)      : mean ΔΣ [M_sun/pc²] across sims
      prof_std    (20,)      : std across sims
      cov_of_mean (20,20)    : Cov(prof) across the LH ensemble; REQUIRED
                               for σ_total
      profiles    (N_sims,20): per-sim mean ΔΣ, kept for provenance
      n_sims      ()         : number of sims contributing to prof/cov_of_mean
      radii       (20,)      : projected radii [comoving Mpc/h]

  'cov_of_mean'/'profiles'/'n_sims' are written by
  test_CAMELS_seventh_gen.ipynb (cells that build mean_DeltaSigma_profiles_*
  and mean_kSZ_profiles_*). Files generated before those fields were added
  only have 'prof'/'prof_std'; for those, σ_total cannot be computed and is
  reported as NaN.

Comparisons performed:
  DESI LRG kSZ  (z~0.47, 9 arcmin bins)  ← snap74 kSZ
  DESI BGS kSZ  (z~0.21, 13 arcmin bins) ← snap82 kSZ
  HSC S16A GGL  (3 mass bins)            ← snap74 ΔΣ  (LRG-like, z~0.47)
                                           snap82 ΔΣ  (BGS-like, z~0.21)

Amplitude-only fit:  d = A · m
  A_best  = (m^T C^{-1} d) / (m^T C^{-1} m)
  σ_A     = 1 / sqrt(m^T C^{-1} m)
  χ²_min  = (d − A_best m)^T C^{-1} (d − A_best m),  DOF = N_bins − 1
  PTE     = P(χ² ≥ χ²_min | DOF)
  σ_PTE   = norm.isf(PTE)   [one-sided Gaussian equivalent]

Two goodness-of-fit significances are reported for every comparison:

  σ_obs   : C = C_obs only. C_obs is the full (generally non-diagonal)
            covariance of the *observational* measurement where the public
            data release provides one (DESI LRG/BGS kSZ), or the diagonal
            jackknife-error covariance where it does not (HSC S16A GGL — no
            public off-diagonal covariance for that release). This is the
            naive treatment: the CAMELS ensemble-mean template m is treated
            as a fixed, perfectly known curve.

  σ_total : C = C_obs + C_template. C_template = W C_sim/N_sims W^T is the
            covariance of the *template*, propagated through the linear
            radial interpolation matrix W from C_sim/N_sims, the covariance
            of the CAMELS ensemble mean profile across the N_sims Latin-
            Hypercube realizations. C_sim captures how coherently a shared
            astrophysical driver (feedback strength, star-formation
            efficiency, etc.) shifts the profile across many radial bins at
            once — an effect that is strongest at large radii, where
            baryonic feedback dominates the shape of the outer profile. This
            mirrors the same qualitative feature (strong correlations across
            bins at large radii) already visible in the DESI kSZ observational
            covariance matrix, but for the theory side of the comparison,
            which the σ_obs-only treatment ignores entirely.

  σ_total is the scientifically appropriate significance whenever the
  simulation ensemble's own scatter is a non-negligible fraction of the
  total error budget; σ_obs is retained for reference/comparison so the
  size of that effect is visible directly in the output tables and figures.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_SIM = '/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data/'
DATA_OBS = '/pscratch/sd/l/lindajin/SimulationStacker/tests/DESI_kSZ_data/'

# ── Cosmology ──────────────────────────────────────────────────────────────────
H0_h = 0.677          # h fixed for all CAMELS IllustrisTNG SB35 sims
Z74  = 0.47;  A74 = 1.0 / (1.0 + Z74)   # scale factor at snap74
Z82  = 0.21;  A82 = 1.0 / (1.0 + Z82)   # scale factor at snap82

# ── snap74 number density grid (LRG, z~0.47) ──────────────────────────────────
ND_IDX_74  = [0, 1, 2, 3, 4]
N_HALOS_74 = [12, 35, 62, 125, 300]              # n_halos = nd × (50 Mpc/h)³
ND_VALS_74 = [1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3]   # (Mpc/h)⁻³
ND_LABELS_74 = [
    r'$n=10^{-4}$ (LRG)',
    r'$n=2.8\times10^{-4}$ (LRG)',
    r'$n=5\times10^{-4}$ (LRG)',
    r'$n=10^{-3}$ (LRG)',
    r'$n=2.4\times10^{-3}$ (LRG)',
]

# ── snap82 number density grid (BGS, z~0.21) ──────────────────────────────────
# n_halos/box_volume = true physical nd; the saved 'number_density' field
# inherited snap74 values and must not be used.
ND_IDX_82  = [0, 1, 2, 3, 4]
N_HALOS_82 = [125, 500, 1000, 2500, 3750]
ND_VALS_82 = [1.0e-3, 4.0e-3, 8.0e-3, 2.0e-2, 3.0e-2]   # (Mpc/h)⁻³
ND_LABELS_82 = [
    r'$n=10^{-3}$ (BGS)',
    r'$n=4\times10^{-3}$ (BGS)',
    r'$n=8\times10^{-3}$ (BGS)',
    r'$n=2\times10^{-2}$ (BGS)',
    r'$n=3\times10^{-2}$ (BGS)',
]

ND_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# ══════════════════════════════════════════════════════════════════════════════
#  Core statistics
# ══════════════════════════════════════════════════════════════════════════════

def fit_amplitude_analytic(data, cov_inv, template):
    """
    Analytic maximum-likelihood amplitude for  d = A·m.

    A_best = (m^T C^{-1} d) / (m^T C^{-1} m)
    σ_A    = 1 / sqrt(m^T C^{-1} m)
    χ²_min = (d − A·m)^T C^{-1} (d − A·m)  evaluated at A_best
    """
    mCid   = template @ cov_inv @ data
    mCim   = template @ cov_inv @ template
    A_best = mCid / mCim
    sig_A  = 1.0 / np.sqrt(mCim)
    resid  = data - A_best * template
    chi2   = float(resid @ cov_inv @ resid)
    return A_best, chi2, sig_A


def pte_and_sigma(chi2_val, dof):
    """
    PTE  = P(χ² ≥ χ²_val | dof)
    σ_eq = norm.isf(PTE)   [one-sided Gaussian equivalent]
    """
    pte   = float(stats.chi2.sf(chi2_val, dof))
    sigma = float(stats.norm.isf(np.clip(pte, 1e-16, 1.0)))
    return pte, sigma


def linear_interp_matrix(r_src, r_dst):
    """
    Build the matrix W (len(r_dst) x len(r_src)) such that  y_dst = W @ y_src
    reproduces piecewise-linear interpolation (with linear extrapolation
    outside [r_src.min(), r_src.max()]) of a function sampled at r_src,
    evaluated at r_dst. r_src must be sorted ascending.

    Because interpolation is linear in y_src, this matrix also propagates
    covariance:  Cov(y_dst) = W @ Cov(y_src) @ W.T
    """
    r_src = np.asarray(r_src, dtype=float)
    r_dst = np.asarray(r_dst, dtype=float)
    n_src = len(r_src)
    W = np.zeros((len(r_dst), n_src))
    for i, x in enumerate(r_dst):
        j = np.searchsorted(r_src, x) - 1
        j = int(np.clip(j, 0, n_src - 2))
        x0, x1 = r_src[j], r_src[j + 1]
        t = (x - x0) / (x1 - x0)
        W[i, j]     = 1.0 - t
        W[i, j + 1] = t
    return W


def sim_mean_covariance(profiles):
    """
    Covariance of the CAMELS ensemble-mean profile, C_sim/N_sims, from the
    per-sim profile array (N_sims, N_bins). Sims with any NaN bin (failed
    load / shape mismatch, see notebook try/except) are dropped first.

    Returns (cov_of_mean, n_sims_used), or (None, 0) if fewer than 2 valid
    sims are available.
    """
    profiles = np.asarray(profiles, dtype=float)
    valid = ~np.any(np.isnan(profiles), axis=1)
    profiles = profiles[valid]
    n_sims = profiles.shape[0]
    if n_sims < 2:
        return None, 0
    cov_sim = np.cov(profiles, rowvar=False)
    return cov_sim / n_sims, n_sims


def _load_cov_of_mean(f):
    """
    Get Cov(prof) across the CAMELS LH ensemble from an open npz archive.
    Prefers the precomputed 'cov_of_mean' field (written directly by the
    generation notebook); falls back to recomputing it from the raw per-sim
    'profiles' array for older files that only saved that. Returns None if
    neither is present.
    """
    if 'cov_of_mean' in f:
        cov = f['cov_of_mean'].copy()
        return None if np.any(np.isnan(cov)) else cov
    if 'profiles' in f:
        cov, _ = sim_mean_covariance(f['profiles'])
        return cov
    return None


def _profile_mean_std(f):
    """
    Ensemble mean/std profile from the per-sim 'profiles' array (N_sims,
    N_bins), dropping sims with any NaN bin — the same criterion the
    generation notebook uses before computing 'cov_of_mean', so the two stay
    consistent. Falls back to precomputed 'prof'/'prof_std' fields for files
    generated before 'profiles' replaced them. Returns (mean, std).
    """
    if 'profiles' in f:
        profiles = np.asarray(f['profiles'], dtype=float)
        valid = ~np.any(np.isnan(profiles), axis=1)
        profiles = profiles[valid]
        return profiles.mean(axis=0), profiles.std(axis=0)
    return f['prof'].copy(), f['prof_std'].copy()


# ══════════════════════════════════════════════════════════════════════════════
#  Simulation profile loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_ksz_profiles(snap):
    """
    Load pre-computed mean T_kSZ profiles.

    Returns dict  nd_idx → {radii_arcmin, prof_mean, prof_std, cov_of_mean}
    cov_of_mean is None if the npz predates the 'profiles' field.
    """
    idx_list  = ND_IDX_74  if snap == 74 else ND_IDX_82
    halo_list = N_HALOS_74 if snap == 74 else N_HALOS_82

    models = {}
    for nd_idx, n_halos in zip(idx_list, halo_list):
        fname = DATA_SIM + f'mean_kSZ_profiles_nd_{nd_idx}_n_{n_halos}_snap{snap}.npz'
        try:
            f = np.load(fname)
        except FileNotFoundError:
            print(f'  [warn] kSZ file not found: {fname}')
            models[nd_idx] = None
            continue
        # The snap82 profiles in these files were divided by 0.64 in the notebook
        # (a placeholder for r_v_sim).  Undo that factor so the profile is in raw
        # T_kSZ [μK·arcmin²] and A_best is directly interpretable.
        # For snap74 no such division was applied (it used per-sim r_v which
        # averaged out to ~1 over the ensemble, so the effect is small).
        norm = 0.64 if snap == 82 else 1.0
        prof_mean, prof_std = _profile_mean_std(f)
        cov_of_mean = _load_cov_of_mean(f)
        if cov_of_mean is not None:
            cov_of_mean = cov_of_mean * norm**2   # Cov(a*x) = a^2 Cov(x)
        else:
            print(f'  [warn] {fname} has no "cov_of_mean"/"profiles" field; '
                  f'σ_total cannot be computed for nd_idx={nd_idx} (regenerate the file).')
        models[nd_idx] = {
            'radii_arcmin': f['radii'].copy(),
            'prof_mean'   : prof_mean * norm,
            'prof_std'    : prof_std * norm,
            'cov_of_mean' : cov_of_mean,
        }
    return models


def load_dsigma_profiles(snap):
    """
    Load pre-computed mean ΔΣ profiles.

    Converts comoving Mpc/h → physical Mpc:  r_phys = r_com × a / h

    Returns dict  nd_idx → {radii_mpch, radii_phys_mpc, prof_mean, prof_std, cov_of_mean}
    """
    a_snap    = A74 if snap == 74 else A82
    idx_list  = ND_IDX_74  if snap == 74 else ND_IDX_82
    halo_list = N_HALOS_74 if snap == 74 else N_HALOS_82

    models = {}
    for nd_idx, n_halos in zip(idx_list, halo_list):
        fname = DATA_SIM + f'mean_DeltaSigma_profiles_nd_{nd_idx}_n_{n_halos}_snap{snap}.npz'
        try:
            f = np.load(fname)
        except FileNotFoundError:
            print(f'  [warn] ΔΣ file not found: {fname}')
            models[nd_idx] = None
            continue
        # 'R' (comoving Mpc/h) replaced the earlier 'radii' key when the
        # notebook was extended to also save 'th' (arcmin); fall back to
        # 'radii' for files generated before that rename.
        r_com = (f['R'] if 'R' in f else f['radii']).copy()
        prof_mean, prof_std = _profile_mean_std(f)
        cov_of_mean = _load_cov_of_mean(f)
        if cov_of_mean is None:
            print(f'  [warn] {fname} has no "cov_of_mean"/"profiles" field; '
                  f'σ_total cannot be computed for nd_idx={nd_idx} (regenerate the file).')
        models[nd_idx] = {
            'radii_mpch'    : r_com,
            'radii_phys_mpc': r_com * a_snap / H0_h,   # physical Mpc
            'prof_mean'     : prof_mean,                # M_sun/pc²
            'prof_std'      : prof_std,
            'cov_of_mean'   : cov_of_mean,
        }
    return models


# ══════════════════════════════════════════════════════════════════════════════
#  Observational data loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_desi_lrg_ksz():
    """DESI LRG kSZ, z~0.47.  9 bins × [1.0, 1.625, …, 6.0] arcmin.  μK·arcmin²."""
    d = np.load(DATA_OBS + 'Fig1_Fig8_dr10_allfoot_perbin_sigmaz0.0500_dr6_corr_pzbin1.npz')
    return {
        'label' : 'DESI LRG kSZ (z~0.47)',
        'theta' : d['theta_arcmins'].astype(float),   # (9,) arcmin
        'prof'  : d['prof'].astype(float),             # (9,) μK·arcmin²
        'cov'   : d['cov'].astype(float),              # (9,9)
        'n_bins': 9,
        'snap'  : 74,
    }


def load_desi_bgs_ksz():
    """DESI BGS kSZ, z~0.21.  13 bins × [1.0, 1.8125, …, 10.75] arcmin.  μK·arcmin²."""
    d = np.load(DATA_OBS + 'Fig5_BGS_BRIGHT-20.2.npz')
    return {
        'label' : 'DESI BGS kSZ (z~0.21)',
        'theta' : d['th'].astype(float),              # (13,) arcmin
        'prof'  : d['prof'].astype(float),            # (13,) μK·arcmin²
        'cov'   : d['cov'].astype(float),             # (13,13)
        'n_bins': 13,
        'snap'  : 82,
    }


def load_hsc_ggl():
    """
    HSC S16A GGL ΔΣ, 3 log-Mvir bins.
    r_mpc in physical Mpc; dsigma_lr in M_sun/pc² (jackknife errors only —
    no public off-diagonal covariance for this data release).
    """
    bins = [
        ('log Mvir = 13.00–13.42', 'hsc_s16a_dsigma_logmvir_13.00_13.42.npy'),
        ('log Mvir = 13.42–13.83', 'hsc_s16a_dsigma_logmvir_13.42_13.83.npy'),
        ('log Mvir = 13.83–14.25', 'hsc_s16a_dsigma_logmvir_13.83_14.25.npy'),
    ]
    out = []
    for label, fname in bins:
        d = np.load(DATA_OBS + fname)
        out.append({
            'label'      : f'HSC S16A GGL ({label})',
            'r_mpc'      : d['r_mpc'].astype(float),        # physical Mpc
            'dsigma'     : d['dsigma_lr'].astype(float),    # M_sun/pc²
            'dsigma_err' : d['dsigma_err_jk'].astype(float),
            'n_bins'     : len(d['r_mpc']),
        })
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  PTE computation
# ══════════════════════════════════════════════════════════════════════════════

_NAN_KEYS = ['nd_idx', 'nd', 'A_best_obs', 'sig_A_obs', 'chi2_obs', 'dof',
             'chi2_dof_obs', 'pte_obs', 'sigma_obs',
             'A_best_total', 'sig_A_total', 'chi2_total',
             'chi2_dof_total', 'pte_total', 'sigma_total']


def _nan_result(nd_idx, nd_val):
    return {k: (nd_idx if k == 'nd_idx' else nd_val if k == 'nd' else np.nan)
            for k in _NAN_KEYS}


def _fit_both_sigmas(data, cov_obs, template, cov_template, dof):
    """
    Fit d = A*m twice: once with C = C_obs (naive, σ_obs), once with
    C = C_obs + C_template (σ_total). Returns a dict of both results.
    cov_template may be None, in which case the σ_total block is NaN.
    """
    cov_inv_obs = np.linalg.inv(cov_obs)
    A_obs, chi2_obs, sigA_obs = fit_amplitude_analytic(data, cov_inv_obs, template)
    pte_obs, sig_obs = pte_and_sigma(chi2_obs, dof)

    if cov_template is not None:
        cov_total     = cov_obs + cov_template
        cov_inv_total = np.linalg.inv(cov_total)
        A_tot, chi2_tot, sigA_tot = fit_amplitude_analytic(data, cov_inv_total, template)
        pte_tot, sig_tot = pte_and_sigma(chi2_tot, dof)
    else:
        A_tot = sigA_tot = chi2_tot = pte_tot = sig_tot = np.nan

    return {
        'A_best_obs': A_obs, 'sig_A_obs': sigA_obs, 'chi2_obs': chi2_obs,
        'dof': dof, 'chi2_dof_obs': chi2_obs / dof,
        'pte_obs': pte_obs, 'sigma_obs': sig_obs,
        'A_best_total': A_tot, 'sig_A_total': sigA_tot, 'chi2_total': chi2_tot,
        'chi2_dof_total': chi2_tot / dof if np.isfinite(chi2_tot) else np.nan,
        'pte_total': pte_tot, 'sigma_total': sig_tot,
    }


def compute_ksz_pte(obs, sim_models, nd_idx_list, nd_vals, nd_labels):
    """
    For each nd model: build the linear-interpolation matrix from the sim's
    radial grid onto the observed theta grid, fit amplitude A twice (σ_obs
    using only C_obs, σ_total using C_obs + C_template propagated from the
    CAMELS ensemble covariance), and report both PTEs / σ.

    Interpolation is linear within the sim's radial support [1, 6] arcmin,
    with linear extrapolation outside it (never restrict the support before
    interpolating — the DESI radii are within the sim's sampled range for
    both LRG and BGS grids, so extrapolation is not actually invoked, but
    keeping it available avoids silently dropping edge bins).
    """
    theta_obs = obs['theta']
    data      = obs['prof']
    cov_obs   = obs['cov']
    dof       = obs['n_bins'] - 1

    results = []
    for nd_idx, nd_val, lbl in zip(nd_idx_list, nd_vals, nd_labels):
        m = sim_models.get(nd_idx)
        if m is None:
            results.append(_nan_result(nd_idx, nd_val))
            continue

        r_sim = m['radii_arcmin']
        if r_sim.max() < theta_obs.min() or r_sim.min() > theta_obs.max():
            results.append(_nan_result(nd_idx, nd_val))
            continue

        W        = linear_interp_matrix(r_sim, theta_obs)
        template = W @ m['prof_mean']
        cov_tmpl = W @ m['cov_of_mean'] @ W.T if m['cov_of_mean'] is not None else None

        fit = _fit_both_sigmas(data, cov_obs, template, cov_tmpl, dof)
        results.append({'nd_idx': nd_idx, 'nd': nd_val, 'label': lbl,
                         'template': template, **fit})
    return results


def compute_ggl_pte(hsc_obs, sim_models, nd_idx_list, nd_vals, nd_labels):
    """
    For each nd model: interpolate sim ΔΣ onto the HSC r_mpc grid within the
    radial overlap, fit amplitude A twice (σ_obs, σ_total), and report both.

    C_obs is diagonal (jackknife errors; no off-diagonal HSC covariance is
    publicly available for S16A) — this limitation is independent of, and
    not fixed by, the σ_total treatment, which only adds the *simulation*
    ensemble's covariance. DOF = N_overlap_bins − 1.
    """
    r_obs  = hsc_obs['r_mpc']
    data   = hsc_obs['dsigma']
    err    = hsc_obs['dsigma_err']

    results = []
    for nd_idx, nd_val, lbl in zip(nd_idx_list, nd_vals, nd_labels):
        m = sim_models.get(nd_idx)
        if m is None:
            results.append(_nan_result(nd_idx, nd_val))
            continue

        r_sim = m['radii_phys_mpc']   # physical Mpc
        mask = (r_obs >= r_sim.min()) & (r_obs <= r_sim.max())
        if mask.sum() < 2:
            results.append(_nan_result(nd_idx, nd_val))
            continue

        r_obs_sub = r_obs[mask]
        W         = linear_interp_matrix(r_sim, r_obs_sub)
        template  = np.full(len(r_obs), np.nan)
        template[mask] = W @ m['prof_mean']

        d_sub    = data[mask]
        t_sub    = template[mask]
        cov_obs  = np.diag(err[mask]**2)   # diagonal, no public off-diagonal HSC covariance
        cov_tmpl = W @ m['cov_of_mean'] @ W.T if m['cov_of_mean'] is not None else None
        dof_sub  = mask.sum() - 1

        fit = _fit_both_sigmas(d_sub, cov_obs, t_sub, cov_tmpl, dof_sub)
        results.append({'nd_idx': nd_idx, 'nd': nd_val, 'label': lbl,
                         'template': template, 'mask': mask, **fit})
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Output formatting
# ══════════════════════════════════════════════════════════════════════════════

def print_pte_table(title, results):
    print(f'\n{"="*118}')
    print(f'  {title}')
    print(f'{"="*118}')
    hdr = (f"{'nd_idx':>6}  {'nd [(Mpc/h)^-3]':>16}  {'A_obs':>7}  {'χ²/ν_obs':>9}  "
           f"{'PTE_obs':>10}  {'σ_obs':>6}  |  {'A_tot':>7}  {'χ²/ν_tot':>9}  "
           f"{'PTE_tot':>10}  {'σ_tot':>6}")
    print(hdr)
    print('-' * 118)
    for r in results:
        if np.isnan(r.get('chi2_obs', np.nan)):
            print(f"  nd={r['nd_idx']}: no file or insufficient radial overlap")
            continue
        print(f"{r['nd_idx']:>6}  {r['nd']:>16.3e}  {r['A_best_obs']:>7.3f}  "
              f"{r['chi2_dof_obs']:>9.3f}  {r['pte_obs']:>10.3e}  {r['sigma_obs']:>6.2f}  |  "
              f"{r['A_best_total']:>7.3f}  {r['chi2_dof_total']:>9.3f}  "
              f"{r['pte_total']:>10.3e}  {r['sigma_total']:>6.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _sigma_panel(ax, results, nd_colors):
    """σ_obs (open markers) vs σ_total (filled markers) against nd, per model."""
    valid = [r for r in results if not np.isnan(r.get('sigma_obs', np.nan))]
    nd_xs      = [r['nd']         for r in valid]
    sig_obs    = [r['sigma_obs']  for r in valid]
    sig_total  = [r['sigma_total'] for r in valid]
    cols       = nd_colors[:len(valid)]

    ax.scatter(nd_xs, sig_obs, facecolors='none', edgecolors=cols, s=70, lw=1.5,
               zorder=4, label=r'$\sigma_{\rm obs}$ (C$_{\rm obs}$ only)')
    ax.scatter(nd_xs, sig_total, c=cols, s=70, edgecolors='k', lw=0.6,
               zorder=5, label=r'$\sigma_{\rm total}$ (C$_{\rm obs}$+C$_{\rm sim}$)')
    for x, y0, y1, c in zip(nd_xs, sig_obs, sig_total, cols):
        if np.isfinite(y1):
            ax.plot([x, x], [y0, y1], color=c, lw=1.0, alpha=0.6, zorder=3)
    ax.axhline(0, ls='--', color='gray', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$n_d$ [(Mpc/$h$)$^{-3}$]', fontsize=10)
    ax.set_ylabel(r'$\sigma_\mathrm{PTE}$', fontsize=10)
    ax.invert_yaxis()
    ax.legend(fontsize=6.5, loc='best')


def plot_ksz_fits(obs_lrg, obs_bgs, res_lrg, res_bgs, figpath=None):
    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1], hspace=0.35, wspace=0.3)

    rows = [
        (obs_lrg, res_lrg, r'$\theta$ [arcmin]'),
        (obs_bgs, res_bgs, r'$\theta$ [arcmin]'),
    ]

    for row, (obs, res, xlabel) in enumerate(rows):
        ax_p = fig.add_subplot(gs[row, 0])
        ax_s = fig.add_subplot(gs[row, 1])

        theta = obs['theta']
        data  = obs['prof']
        yerr  = np.sqrt(np.diag(obs['cov']))

        ax_p.errorbar(theta, data, yerr=yerr, fmt='ko', ms=5, capsize=3,
                      lw=1.5, zorder=10, label=obs['label'])

        for r, col in zip(res, ND_COLORS):
            if r.get('template') is None or np.isnan(r.get('A_best_total', np.nan)):
                continue
            ax_p.plot(theta, r['A_best_total'] * r['template'], color=col, lw=1.8,
                      label=r['label'] + rf' ($\chi^2/\nu={r["chi2_dof_total"]:.2f}$)')

        ax_p.set_xlabel(xlabel, fontsize=11)
        ax_p.set_ylabel(r'$T_\mathrm{kSZ}$ [$\mu$K arcmin²]', fontsize=11)
        ax_p.set_title(obs['label'], fontsize=11)
        ax_p.legend(fontsize=7, loc='upper left')

        _sigma_panel(ax_s, res, ND_COLORS)
        ax_s.set_title('Goodness of fit', fontsize=10)

    if figpath:
        fig.savefig(figpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {figpath}')
    return fig


def plot_ggl_fits(hsc_datasets, res_lrg_list, res_bgs_list, figpath=None):
    """
    Two rows × n_mass_bins columns.
    Top row: snap74 (LRG-like) ΔΣ vs HSC.
    Bottom row: snap82 (BGS-like) ΔΣ vs HSC.
    Right column: σ_obs vs σ_total against nd, per HSC mass bin.
    """
    n_mass = len(hsc_datasets)
    fig = plt.figure(figsize=(5 * n_mass + 3, 10))
    gs  = gridspec.GridSpec(2, n_mass + 1, width_ratios=[3]*n_mass + [2],
                             hspace=0.4, wspace=0.35)

    snap_rows = [
        (res_lrg_list, 'snap74 (z~0.47, LRG-like)'),
        (res_bgs_list, 'snap82 (z~0.21, BGS-like)'),
    ]

    for row, (res_list, snap_title) in enumerate(snap_rows):
        ax_s = fig.add_subplot(gs[row, n_mass])

        for col, (hsc_obs, res) in enumerate(zip(hsc_datasets, res_list)):
            ax_p = fig.add_subplot(gs[row, col])
            r_obs = hsc_obs['r_mpc']

            ax_p.errorbar(r_obs, hsc_obs['dsigma'], yerr=hsc_obs['dsigma_err'],
                          fmt='ko', ms=4, capsize=3, lw=1.5, zorder=10, label='HSC S16A')

            for r, col_c in zip(res, ND_COLORS):
                if np.isnan(r.get('A_best_total', np.nan)) or r.get('template') is None:
                    continue
                mask   = r.get('mask', np.ones(len(r_obs), bool))
                scaled = r['A_best_total'] * r['template']
                ax_p.plot(r_obs[mask], scaled[mask], color=col_c, lw=1.8,
                          label=r['label'] + rf' ($\chi^2/\nu={r["chi2_dof_total"]:.2f}$)')

            ax_p.set_xscale('log')
            yvals = np.concatenate([hsc_obs['dsigma']] +
                                   [r['A_best_total'] * r['template'][r['mask']]
                                    for r in res
                                    if not np.isnan(r.get('A_best_total', np.nan))
                                    and r.get('template') is not None])
            if np.all(yvals > 0):
                ax_p.set_yscale('log')
            ax_p.set_xlabel(r'$R$ [physical Mpc]', fontsize=10)
            ax_p.set_ylabel(r'$\Delta\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]', fontsize=10)
            ax_p.set_title(f'{snap_title}\n{hsc_obs["label"]}', fontsize=9)
            ax_p.legend(fontsize=6.5)

        # σ panel: one obs/total pair of curves per HSC mass bin
        mass_labels = ['13.0–13.4', '13.4–13.8', '13.8–14.3']
        markers = ['o', 's', '^']
        res_by_mass = res_lrg_list if row == 0 else res_bgs_list
        for mass_i, (res_m, mlbl) in enumerate(zip(res_by_mass, mass_labels)):
            nd_xs     = [r['nd']          for r in res_m if not np.isnan(r.get('sigma_obs', np.nan))]
            sig_obs   = [r['sigma_obs']   for r in res_m if not np.isnan(r.get('sigma_obs', np.nan))]
            sig_total = [r['sigma_total'] for r in res_m if not np.isnan(r.get('sigma_obs', np.nan))]
            ax_s.plot(nd_xs, sig_obs, marker=markers[mass_i], ls='--', lw=1.2, alpha=0.7,
                      label=f'log M={mlbl} (obs)')
            ax_s.plot(nd_xs, sig_total, marker=markers[mass_i], ls='-', lw=1.6,
                      label=f'log M={mlbl} (total)')

        ax_s.axhline(0, ls='--', color='gray', lw=1)
        ax_s.set_xscale('log')
        ax_s.set_xlabel(r'$n_d$ [(Mpc/$h$)$^{-3}$]', fontsize=10)
        ax_s.set_ylabel(r'$\sigma_\mathrm{PTE}$', fontsize=10)
        ax_s.set_title(f'{snap_title}\nGoodness of fit', fontsize=9)
        ax_s.legend(fontsize=6.5)
        ax_s.invert_yaxis()

    if figpath:
        fig.savefig(figpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {figpath}')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Loading simulation profiles ...')
    sim_ksz_74 = load_ksz_profiles(snap=74)
    sim_ksz_82 = load_ksz_profiles(snap=82)
    sim_ds_74  = load_dsigma_profiles(snap=74)
    sim_ds_82  = load_dsigma_profiles(snap=82)

    print('Loading observational data ...')
    obs_lrg = load_desi_lrg_ksz()
    obs_bgs = load_desi_bgs_ksz()
    hsc_ggl = load_hsc_ggl()

    # ── DESI LRG kSZ (snap74) ─────────────────────────────────────────────────
    print('\nDESI LRG kSZ  (snap74, z~0.47) ...')
    res_lrg_ksz = compute_ksz_pte(obs_lrg, sim_ksz_74,
                                   ND_IDX_74, ND_VALS_74, ND_LABELS_74)
    print_pte_table('DESI LRG kSZ vs snap74  [fit: A × sim mean T_kSZ]',
                    res_lrg_ksz)

    # ── DESI BGS kSZ (snap82) ─────────────────────────────────────────────────
    print('\nDESI BGS kSZ  (snap82, z~0.21) ...')
    res_bgs_ksz = compute_ksz_pte(obs_bgs, sim_ksz_82,
                                   ND_IDX_82, ND_VALS_82, ND_LABELS_82)
    print_pte_table('DESI BGS kSZ vs snap82  [fit: A × sim mean T_kSZ]',
                    res_bgs_ksz)

    # ── HSC GGL vs snap74 ΔΣ (LRG-like selection) ────────────────────────────
    print('\nHSC GGL vs snap74 ΔΣ (LRG-like, z~0.47) ...')
    res_ggl_lrg = []
    for hsc_obs in hsc_ggl:
        res = compute_ggl_pte(hsc_obs, sim_ds_74, ND_IDX_74, ND_VALS_74, ND_LABELS_74)
        print_pte_table(f'{hsc_obs["label"]} vs snap74 ΔΣ', res)
        res_ggl_lrg.append(res)

    # ── HSC GGL vs snap82 ΔΣ (BGS-like selection) ────────────────────────────
    print('\nHSC GGL vs snap82 ΔΣ (BGS-like, z~0.21) ...')
    res_ggl_bgs = []
    for hsc_obs in hsc_ggl:
        res = compute_ggl_pte(hsc_obs, sim_ds_82, ND_IDX_82, ND_VALS_82, ND_LABELS_82)
        print_pte_table(f'{hsc_obs["label"]} vs snap82 ΔΣ', res)
        res_ggl_bgs.append(res)

    # ── Figures ───────────────────────────────────────────────────────────────
    print('\nGenerating figures ...')
    plot_ksz_fits(
        obs_lrg, obs_bgs, res_lrg_ksz, res_bgs_ksz,
        figpath='/pscratch/sd/l/lindajin/SimulationStacker/tests/profile_PTE_kSZ.png',
    )
    plot_ggl_fits(
        hsc_ggl, res_ggl_lrg, res_ggl_bgs,
        figpath='/pscratch/sd/l/lindajin/SimulationStacker/tests/profile_PTE_GGL.png',
    )
    plt.show()

    return res_lrg_ksz, res_bgs_ksz, res_ggl_lrg, res_ggl_bgs


if __name__ == '__main__':
    main()
