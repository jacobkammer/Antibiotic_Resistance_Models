# =============================================================================
# Deterministic k_immune threshold sweep (0.02-0.65 h^-1).
#
# The fixed immune baseline and the MC sampling mean come from parameters.py
# and are both 0.18 h^-1. Every other parameter remains at BASE_PARAMS.
# At the accepted calibration, final R_res and final R_b both cross the LOD at
# 0.093 h^-1. The companion MC analysis gives 27.8% bacteremia relapse
# (278/1000), classified by the resistant blood trajectory rather than R_res.
#
# This script runs a fine, deterministic (non-Monte-Carlo) sweep of k_immune,
# records the FINAL R_b / R_res at the end of each run, selects the k_immune
# values that finish above the LOD, and localizes the final-count
# classification switch for each compartment with Brent's method.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from parameters import (
    BASE_PARAMS, K_IMMUNE_DEFAULT, LOD, TOTAL_H, VANCO_START, Y0,
    canonical_time_grid, save_figure,
)
from solver import integrate_model

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

# all figures are written into figures/ alongside the scripts
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# FIGURES PRODUCED BY THIS SCRIPT -- ALL DETERMINISTIC (no sampling anywhere)
#   immune_response_threshold_sweep.png -- final R_b/R_res across the swept grid.
#   The Monte Carlo treatment of k_immune lives in MC_immune_response.py.

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODULE_NAME = "model_Bacteremia.py"
if not os.path.exists(MODULE_NAME):
    print(f"ERROR: Cannot find '{MODULE_NAME}' in the current directory.", file=sys.stderr)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("model_mod", MODULE_NAME)
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)

# ---------------------------------------------------------------------------
# Simulation settings (protocol and baseline from parameters.py)
# ---------------------------------------------------------------------------

total_h     = TOTAL_H
vanco_start = VANCO_START
t_eval      = canonical_time_grid()   # matches MC_immune_response.py
t_days      = t_eval / 24.0

pk = model_mod.PharmacokineticModel()

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid",  total_h, vanco_start + pk.van_duration)

K_IMMUNE_BASE = K_IMMUNE_DEFAULT

# Model and immune baseline parameters come from parameters.py. Only k_immune
# is varied by this deterministic sweep.



def final_counts(k_immune):
    """Run one deterministic simulation at a given k_immune; return (final R_b, final R_res)."""
    immune_i = model_mod.ImmuneResponse(k_immune=k_immune)
    sol = integrate_model(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(BASE_PARAMS, van_func, lzd_func, immune_i),
    )
    return sol[-1, 1], sol[-1, 3]


# ---------------------------------------------------------------------------
# Fine sweep across k_immune in [0.02, 0.65]
# ---------------------------------------------------------------------------
KIM_MIN, KIM_MAX = 0.02, 0.65
N_POINTS = 127   # step = 0.005; includes the 0.18 h^-1 baseline

kim_sweep   = np.linspace(KIM_MIN, KIM_MAX, N_POINTS)
final_R_b   = np.zeros(N_POINTS)
final_R_res = np.zeros(N_POINTS)

print(f"Sweeping k_immune across [{KIM_MIN}, {KIM_MAX}]  ({N_POINTS} points)...", flush=True)
for i, kim in enumerate(kim_sweep):
    final_R_b[i], final_R_res[i] = final_counts(kim)
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{N_POINTS} complete", flush=True)

# k_immune values that finish ABOVE the limit of detection (i.e. resistant escape)
escape_R_b   = kim_sweep[final_R_b   > LOD]
escape_R_res = kim_sweep[final_R_res > LOD]

print(f"\nk_immune values in [{KIM_MIN}, {KIM_MAX}] with final R_b   > LOD: "
      f"{escape_R_b.min():.4f}-{escape_R_b.max():.4f} h^-1 ({len(escape_R_b)}/{N_POINTS} points)"
      if len(escape_R_b) else "\nNo swept k_immune values left final R_b above the LOD.")
print(f"k_immune values in [{KIM_MIN}, {KIM_MAX}] with final R_res > LOD: "
      f"{escape_R_res.min():.4f}-{escape_R_res.max():.4f} h^-1 ({len(escape_R_res)}/{N_POINTS} points)"
      if len(escape_R_res) else "No swept k_immune values left final R_res above the LOD.")


# ---------------------------------------------------------------------------
# Classification-switch localization with Brent's bracket-preserving method
# ---------------------------------------------------------------------------
def find_threshold(compartment_idx):
    """Localize the final-count classification switch with Brent's method."""
    def f(k_immune):
        immune_i = model_mod.ImmuneResponse(k_immune=k_immune)
        sol = integrate_model(
            model_mod.dual_reservoir_model,
            Y0, t_eval,
            args=(BASE_PARAMS, van_func, lzd_func, immune_i),
        )
        return sol[-1, compartment_idx] - LOD

    lo_val, hi_val = f(KIM_MIN), f(KIM_MAX)
    if lo_val < 0 or hi_val > 0:
        return None   # no sign change in range -> no clean threshold to bisect
    return brentq(f, KIM_MIN, KIM_MAX, xtol=1e-6)   # same tolerance as MC_immune_response.py


threshold_R_b   = find_threshold(1)
threshold_R_res = find_threshold(3)


def _describe(name, threshold, escapes):
    """Report a compartment's escape threshold, disambiguating a None result
    as 'always escapes' vs 'always clears' using the swept data (a missing
    brentq root is ambiguous on its own)."""
    if threshold is not None:
        print(f"\nPrecise escape threshold (final {name} crosses LOD):   "
              f"k_immune = {threshold:.3f} h^-1")
    elif len(escapes) == N_POINTS:
        print(f"\nNo clean {name} threshold found -- {name} escapes (stays above LOD) "
              f"across the entire swept range [{KIM_MIN}, {KIM_MAX}].")
    else:
        print(f"\nNo clean {name} threshold found -- {name} clears across the entire swept range.")


_describe("R_b", threshold_R_b, escape_R_b)
_describe("R_res", threshold_R_res, escape_R_res)

if threshold_R_res is not None:
    below_threshold = K_IMMUNE_BASE < threshold_R_res
    print(f"\nmodel_Bacteremia.py's own default k_immune = {K_IMMUNE_BASE} is "
          f"{'BELOW' if below_threshold else 'AT/ABOVE'} the R_res escape threshold "
          f"({threshold_R_res:.3f}) -> R_res {'escapes' if below_threshold else 'clears'}")
elif len(escape_R_res) == N_POINTS:
    print(f"\nmodel_Bacteremia.py's own default k_immune = {K_IMMUNE_BASE} -> "
          f"R_res escapes (stays above LOD) regardless of k_immune in [{KIM_MIN}, {KIM_MAX}].")
else:
    print(f"\nmodel_Bacteremia.py's own default k_immune = {K_IMMUNE_BASE} -> "
          f"R_res clears regardless of k_immune in [{KIM_MIN}, {KIM_MAX}].")

# ---------------------------------------------------------------------------
# FIGURE: final R_b / R_res vs k_immune, with escape zone(s) highlighted
# Like Emax_l, LOW k_immune is the escape zone here (weaker immune system).
# ---------------------------------------------------------------------------
FLOOR = LOD * 0.5   # display floor for values that fell below the LOD (i.e. 0)

disp_R_b   = np.where(final_R_b   <= LOD, FLOOR, final_R_b)
disp_R_res = np.where(final_R_res <= LOD, FLOOR, final_R_res)

fig, ax = plt.subplots(figsize=(11, 6.5))

if threshold_R_res is not None:
    ax.axvspan(KIM_MIN, threshold_R_res, color="lightcoral", alpha=0.15,
               label=f"$R_{{res}}$ escapes ($k_{{immune}} \\leq$ {threshold_R_res:.2g})")
if threshold_R_b is not None:
    ax.axvspan(KIM_MIN, threshold_R_b, color="firebrick", alpha=0.18,
               label=f"$R_b$ escapes ($k_{{immune}} \\leq$ {threshold_R_b:.2g})")

ax.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")
ax.axvline(K_IMMUNE_BASE, color="gray", ls="-.", lw=1.2, alpha=0.7,
           label=f"Baseline ($k_{{immune}}$ = {K_IMMUNE_BASE:.2f})")

ax.plot(kim_sweep, disp_R_res, color="indianred", lw=2.0, marker="o", ms=3,
        label="Final $R_{res}$ (reservoir)")
ax.plot(kim_sweep, disp_R_b, color="darkred", lw=2.0, marker="o", ms=3,
        label="Final $R_b$ (blood)")

if threshold_R_res is not None:
    ax.axvline(threshold_R_res, color="indianred", ls="--", lw=1.2, alpha=0.8)
if threshold_R_b is not None:
    ax.axvline(threshold_R_b, color="darkred", ls="--", lw=1.2, alpha=0.8)

ax.set_yscale("log")
# Keep the full numerical sweep above, but use a focused display window derived
# from the located switch so the transition stays centred after any
# recalibration, widened if needed so the baseline k_immune is also visible.
PLOT_KIM_MIN = 0.0
_centre = threshold_R_res if threshold_R_res is not None else KIM_MAX / 2
PLOT_KIM_MAX = min(KIM_MAX, np.ceil(max(2 * _centre, 1.1 * K_IMMUNE_BASE) / 0.01) * 0.01)
ax.set_xlim(PLOT_KIM_MIN, PLOT_KIM_MAX)
ax.set_ylim(FLOOR * 0.8, max(FLOOR * 20, final_R_b.max(), final_R_res.max()) * 2)
ax.set_xlabel(r"$k_{immune}$ (h$^{-1}$)")
ax.set_ylabel("Final resistant bacterial count (CFU/mL)")
# ax.set_title(r"Resistant Escape Threshold vs $k_{immune}$ (host immune clearance) — end-of-simulation counts",
#              fontsize=19, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(loc="upper right", fontsize=15, framealpha=0.85)

fig.tight_layout()
out_file = os.path.join(FIGURES_DIR, "immune_response_threshold_sweep.png")
save_figure(fig, out_file, __file__)
print(f"\nSaved: {out_file}")
