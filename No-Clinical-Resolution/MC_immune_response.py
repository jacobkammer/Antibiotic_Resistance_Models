"""
Monte Carlo simulation varying k_immune with a log-normal distribution.
k_immune ~ LogNormal(mean=0.14719, CV=0.25)
4-panel figure: S_b, R_b, S_res, R_res — each with individual MC trajectories,
5th–95th percentile band, and median.

Mean raised from the model's own default (0.12) to 0.142 so that ~70% of
samples land above the R_b/R_res escape threshold (k_immune = 0.1255 h^-1,
see ImmuneResponse_threshold_sweep.py) and resolve the infection -- at the
model default, only ~31-37% of samples resolved. CV was then widened from
0.20 to 0.25 (dropping resolution to ~64%), so the mean was re-solved to
0.14719 -- the 30th percentile of LogNormal(mean, CV=0.25) sits exactly at
the 0.1255 threshold -- restoring resolution to ~70%.
"""

import importlib.util
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# ── Load model module ─────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("model_mod", "model.ClinicalResponse.py")
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)

# ── Simulation settings ───────────────────────────────────────────────────────
NUM_ITER    = 1000
TOTAL_H     = 1944  # 21d pre-tx + 4d vancomycin + 42d linezolid + 14d post-tx follow-up
VANCO_START = 504
t_eval      = np.linspace(0, TOTAL_H, 1450)
t_days      = t_eval / 24.0

pk       = model_mod.PharmacokineticModel()
van_func = pk.concentration_function("vancomycin", TOTAL_H, VANCO_START)
lzd_func = pk.concentration_function("linezolid",  TOTAL_H, VANCO_START + pk.van_duration)

rho_S = 0.60    # lowered from 0.80 for slower post-treatment R_b regrowth; Emax_l auto-follows
rho_R = 0.55    # directly tuned (~8.3% fitness cost relative to rho_S, down from 20%)

params = {
    "rho_S":           rho_S,
    "rho_R":           rho_R,
    "rho_res_S":       0.175,  # scaled 5x (from 0.035) alongside rho_S
    "rho_res_R":       0.1765,  # narrow window just above the reservoir persistence threshold (0.17594055) so S_b can also establish in blood -- see model.ClinicalResponse.py
    "Emax_v":          0.40,
    "EC50_V":          0.245,
    "Emax_l":          0.8,  # fixed, decoupled from rho_S (was tied for "perfect bacteriostasis")
    "EC50_L":          1.0,
    "B_max_blood":     6000,
    "B_max_reservoir": 1e4,    # lowered from 4.5e6 so escaped R_res plateaus well above the LOD but far below its old level
    "van_res_fraction":0.15,
    "lzd_res_fraction":0.45,
    "f_r_b":           5e-5,  # restored to original
    "f_b_r":           1e-5,
}

# Fixed initial conditions (blood starts sterile; reservoir seeded at 100 CFU/mL each)
Y0 = [0.0, 0.0, 100.0, 100.0]

# ── Log-normal sampling of k_immune ──────────────────────────────────────────
# CV = sqrt(exp(σ²) - 1)  →  σ = sqrt(ln(1 + CV²))
# μ  = ln(mean) - σ²/2   ensures E[k_immune] = 0.12
K_IMMUNE_MEAN = 0.14719   # re-tuned from 0.142 after CV widened to 0.25, to hold resolution at ~70%
CV            = 0.25      # widened from 0.20

sigma_ln = np.sqrt(np.log(1.0 + CV**2))
mu_ln    = np.log(K_IMMUNE_MEAN) - sigma_ln**2 / 2.0

np.random.seed(42)
k_immune_samples = np.random.lognormal(mu_ln, sigma_ln, NUM_ITER)

print(f"k_immune  mean={k_immune_samples.mean():.4f}  "
      f"std={k_immune_samples.std():.4f}  "
      f"CV={k_immune_samples.std()/k_immune_samples.mean():.3f}  "
      f"[5th–95th]: [{np.percentile(k_immune_samples,5):.4f}, "
      f"{np.percentile(k_immune_samples,95):.4f}]")

# ── Storage ───────────────────────────────────────────────────────────────────
LOD = 10.0   # CFU/mL — values below this are treated as below detection

S_b_hist   = np.full((NUM_ITER, len(t_eval)), np.nan)
R_b_hist   = np.full((NUM_ITER, len(t_eval)), np.nan)
S_res_hist = np.full((NUM_ITER, len(t_eval)), np.nan)
R_res_hist = np.full((NUM_ITER, len(t_eval)), np.nan)

# ── Monte Carlo loop ──────────────────────────────────────────────────────────
for i, k_im in enumerate(k_immune_samples):
    immune_i = model_mod.ImmuneResponse(k_immune=k_im)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = odeint(
            model_mod.dual_reservoir_model, Y0, t_eval,
            args=(params, van_func, lzd_func, immune_i),
            rtol=1e-7, atol=1e-9, mxstep=5000,
        )

    # Clip negatives, then mask below LOD with NaN (log-scale friendly)
    s_b, r_b, s_r, r_r = [np.clip(sol[:, j], 0, None) for j in range(4)]
    S_b_hist[i, :]   = np.where(s_b < LOD, np.nan, s_b)
    R_b_hist[i, :]   = np.where(r_b < LOD, np.nan, r_b)
    S_res_hist[i, :] = np.where(s_r < LOD, np.nan, s_r)
    R_res_hist[i, :] = np.where(r_r < LOD, np.nan, r_r)

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{NUM_ITER} done")

# ── Treatment-window boundaries ───────────────────────────────────────────────
vanco_start_d = VANCO_START / 24.0
lzd_start_d   = (VANCO_START + pk.van_duration) / 24.0
lzd_end_d     = (VANCO_START + pk.van_duration + pk.lzd_duration) / 24.0

# ── Panel helper ──────────────────────────────────────────────────────────────
N_TRACES = 40   # individual trajectories to draw per panel


def _draw_panel(ax, history, S_or_R, compartment, color_main, y_max):
    """
    Render one MC panel: shaded treatment windows, individual traces,
    5th–95th band, and median line.
    """
    # Treatment shading
    ax.axvspan(vanco_start_d, lzd_start_d, color="steelblue", alpha=0.10,
               label="Vancomycin")
    ax.axvspan(lzd_start_d,  lzd_end_d,   color="gold",      alpha=0.15,
               label="Linezolid")
    ax.axvline(lzd_end_d, color="darkgoldenrod", ls="--", lw=1.0, alpha=0.7)

    # LOD reference
    ax.axhline(LOD, color="black", ls=":", lw=0.9, alpha=0.6,
               label=f"LOD ({int(LOD)} CFU/mL)")

    # Individual trajectories (random subsample)
    rng_idx = np.random.default_rng(0).choice(NUM_ITER, N_TRACES, replace=False)
    for i in rng_idx:
        ax.plot(t_days, history[i, :], color=color_main, alpha=0.12, lw=0.7)

    # Percentile band + median (suppress all-NaN warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p50  = np.nanpercentile(history, 50, axis=0)
        p05  = np.nanpercentile(history,  5, axis=0)
        p95  = np.nanpercentile(history, 95, axis=0)

    ax.fill_between(t_days, p05, p95, color=color_main, alpha=0.22,
                    label="5th–95th pctile")
    ax.plot(t_days, p50, color=color_main, lw=2.2, label="Median")

    # Axis formatting
    loc_tag  = "Blood"     if compartment == "blood" else "Reservoir"
    strain   = "Sensitive" if S_or_R == "S"           else "Resistant"
    ylabel   = "Blood Burden (CFU/mL)" if compartment == "blood" else "Reservoir Burden (CFU/mL)"
    var_sym  = f"$S_b$"     if (S_or_R == "S" and compartment == "blood")    else \
               f"$R_b$"     if (S_or_R == "R" and compartment == "blood")    else \
               f"$S_{{res}}$" if (S_or_R == "S" and compartment == "reservoir") else \
               f"$R_{{res}}$"

    ax.set_yscale("log")
    ax.set_ylim(LOD * 0.8, y_max)
    ax.set_xlim(0, t_days[-1])
    ax.set_xlabel("Time (Days)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{strain} — {loc_tag}  ({var_sym})", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.7)


# ── 4-panel figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

fig.suptitle(
    f"Monte Carlo — Immune Response Variability\n"
    rf"$k_{{immune}}$ ~ LogNormal  (mean = {K_IMMUNE_MEAN}, CV = {CV}),  "
    f"n = {NUM_ITER} simulations",
    fontsize=13, fontweight="bold",
)

_draw_panel(axes[0, 0], S_b_hist,   S_or_R="S", compartment="blood",      color_main="royalblue", y_max=1e6)
_draw_panel(axes[0, 1], R_b_hist,   S_or_R="R", compartment="blood",      color_main="crimson",   y_max=1e6)
_draw_panel(axes[1, 0], S_res_hist, S_or_R="S", compartment="reservoir",  color_main="royalblue", y_max=2e7)
_draw_panel(axes[1, 1], R_res_hist, S_or_R="R", compartment="reservoir",  color_main="crimson",   y_max=2e7)

# Panel labels
for label, ax in zip(["A", "B", "C", "D"], axes.flat):
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

out_file = "mc_immune_response.png"
fig.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out_file}")
plt.show()
