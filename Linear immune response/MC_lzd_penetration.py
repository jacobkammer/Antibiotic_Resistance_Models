"""
Monte Carlo simulation varying lzd_res_fraction (linezolid tissue penetration).
Log-normal distribution fitted so:
  5th  percentile ≈ 0.45  (current baseline)
  95th percentile ≈ 0.80  (upper plausible penetration)
4-panel figure: S_b, R_b, S_res, R_res — individual MC trajectories,
5th–95th percentile band, and median.
"""

import copy
import importlib.util
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# ── Load model module ─────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("model_mod", "model.LinearIM.py")
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)

# ── Simulation settings ───────────────────────────────────────────────────────
NUM_ITER    = 300
TOTAL_H     = 1944  # 21d pre-tx + 4d vancomycin + 42d linezolid + 14d post-tx follow-up
VANCO_START = 504
t_eval      = np.linspace(0, TOTAL_H, 1450)
t_days      = t_eval / 24.0

pk           = model_mod.PharmacokineticModel()
immune_model = model_mod.ImmuneResponse(k_immune=0.12)

van_func = pk.concentration_function("vancomycin", TOTAL_H, VANCO_START)
lzd_func = pk.concentration_function("linezolid",  TOTAL_H, VANCO_START + pk.van_duration)

rho_S = 0.16
rho_R = 0.128   # directly tuned (20% fitness cost relative to rho_S)

BASE_PARAMS = {
    "rho_S":           rho_S,
    "rho_R":           rho_R,
    "rho_res_S":       0.035,
    "rho_res_R":       0.024,   # lowered below the 20%-fitness-cost value (0.028) so R_res clears before linezolid ends
    "Emax_v":          0.40,
    "EC50_V":          1.5,
    "Emax_l":          rho_S,
    "EC50_L":          1.0,
    "B_max_blood":     5e5,
    "B_max_reservoir": 4.5e6,
    "van_res_fraction":0.15,
    "lzd_res_fraction":0.45,   # ← this is the parameter being varied
    "f_r_b":           5e-5,
    "f_b_r":           1e-5,
}

# Fixed initial conditions
Y0 = [0.0, 0.0, 100.0, 100.0]

# ── Log-normal sampling of lzd_res_fraction ───────────────────────────────────
# Fit so that 5th pctile = 0.45 (baseline) and 95th pctile = 0.80
# P5:  ln(0.45) = μ - 1.6449·σ
# P95: ln(0.80) = μ + 1.6449·σ
LZD_RES_P05 = 0.45
LZD_RES_P95 = 0.80
Z95 = 1.6449   # norm.ppf(0.95)

mu_ln    = (np.log(LZD_RES_P05) + np.log(LZD_RES_P95)) / 2.0
sigma_ln = (np.log(LZD_RES_P95) - np.log(LZD_RES_P05)) / (2.0 * Z95)
lzd_mean = np.exp(mu_ln + sigma_ln**2 / 2.0)

np.random.seed(42)
lzd_res_samples = np.random.lognormal(mu_ln, sigma_ln, NUM_ITER)

print(f"lzd_res_fraction  mean={lzd_res_samples.mean():.4f}  "
      f"std={lzd_res_samples.std():.4f}  "
      f"CV={lzd_res_samples.std()/lzd_res_samples.mean():.3f}  "
      f"[5th–95th]: [{np.percentile(lzd_res_samples, 5):.4f}, "
      f"{np.percentile(lzd_res_samples, 95):.4f}]")

# ── Storage ───────────────────────────────────────────────────────────────────
LOD = 10.0   # CFU/mL

S_b_hist   = np.full((NUM_ITER, len(t_eval)), np.nan)
R_b_hist   = np.full((NUM_ITER, len(t_eval)), np.nan)
S_res_hist = np.full((NUM_ITER, len(t_eval)), np.nan)
R_res_hist = np.full((NUM_ITER, len(t_eval)), np.nan)

# ── Monte Carlo loop ──────────────────────────────────────────────────────────
for i, lzd_frac in enumerate(lzd_res_samples):
    p_i = copy.copy(BASE_PARAMS)
    p_i["lzd_res_fraction"] = lzd_frac

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = odeint(
            model_mod.dual_reservoir_model, Y0, t_eval,
            args=(p_i, van_func, lzd_func, immune_model),
            rtol=1e-8, atol=1e-10, mxstep=5000,
        )

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
N_TRACES = 40


def _draw_panel(ax, history, S_or_R, compartment, color_main, y_max):
    """Render one MC panel: treatment windows, individual traces, band, median."""
    ax.axvspan(vanco_start_d, lzd_start_d, color="steelblue", alpha=0.10,
               label="Vancomycin")
    ax.axvspan(lzd_start_d,  lzd_end_d,   color="gold",      alpha=0.15,
               label="Linezolid")
    ax.axvline(lzd_end_d, color="darkgoldenrod", ls="--", lw=1.0, alpha=0.7)

    ax.axhline(LOD, color="black", ls=":", lw=0.9, alpha=0.6,
               label=f"LOD ({int(LOD)} CFU/mL)")

    rng_idx = np.random.default_rng(0).choice(NUM_ITER, N_TRACES, replace=False)
    for k in rng_idx:
        ax.plot(t_days, history[k, :], color=color_main, alpha=0.12, lw=0.7)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p50 = np.nanpercentile(history, 50, axis=0)
        p05 = np.nanpercentile(history,  5, axis=0)
        p95 = np.nanpercentile(history, 95, axis=0)

    ax.fill_between(t_days, p05, p95, color=color_main, alpha=0.22,
                    label="5th–95th pctile")
    ax.plot(t_days, p50, color=color_main, lw=2.2, label="Median")

    loc_tag = "Blood"     if compartment == "blood" else "Reservoir"
    strain  = "Sensitive" if S_or_R == "S"           else "Resistant"
    ylabel  = "Blood Burden (CFU/mL)" if compartment == "blood" else "Reservoir Burden (CFU/mL)"
    var_sym = (f"$S_b$"       if (S_or_R == "S" and compartment == "blood")      else
               f"$R_b$"       if (S_or_R == "R" and compartment == "blood")      else
               f"$S_{{res}}$" if (S_or_R == "S" and compartment == "reservoir")  else
               f"$R_{{res}}$")

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
    f"Monte Carlo — Linezolid Reservoir Penetration\n"
    rf"$f_{{lzd,res}}$ ~ LogNormal  "
    rf"[5th–95th pctile: {LZD_RES_P05}–{LZD_RES_P95}],  "
    rf"mean $\approx$ {lzd_mean:.3f},  n = {NUM_ITER} simulations",
    fontsize=13, fontweight="bold",
)

_draw_panel(axes[0, 0], S_b_hist,   S_or_R="S", compartment="blood",     color_main="royalblue", y_max=1e6)
_draw_panel(axes[0, 1], R_b_hist,   S_or_R="R", compartment="blood",     color_main="crimson",   y_max=1e6)
_draw_panel(axes[1, 0], S_res_hist, S_or_R="S", compartment="reservoir", color_main="royalblue", y_max=2e7)
_draw_panel(axes[1, 1], R_res_hist, S_or_R="R", compartment="reservoir", color_main="crimson",   y_max=2e7)

for label, ax in zip(["A", "B", "C", "D"], axes.flat):
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

out_file = "mc_lzd_penetration.png"
fig.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out_file}")
plt.show()
