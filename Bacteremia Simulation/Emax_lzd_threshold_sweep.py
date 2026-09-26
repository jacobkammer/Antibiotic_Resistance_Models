# =============================================================================
# Deterministic Emax_L threshold sweep (0.70--2.2 h^-1)
#
# Every parameter except Emax_l comes from the shared BASE_PARAMS. The fixed
# Emax_l baseline and the MC sampling mean are both 1.65 h^-1. Lower Emax_l is
# the reservoir-persistence side of the switch. The accepted final-R_res
# crossing is 1.55 h^-1; the companion MC analysis gives 22.0% bacteremia
# relapse (220/1000).
#
# This script runs a fine, deterministic (non-Monte-Carlo) sweep of Emax_l
# across the centralized literature bounds, records final R_b / R_res,
# selects the Emax_l values that finish above the LOD, and locates the exact
# crossing point for each compartment via root-finding.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from parameters import (
    BASE_PARAMS, EMAX_L_BOUNDS, LOD, TOTAL_H, VANCO_START, Y0,
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
#   emax_lzd_threshold_sweep.png      -- final R_b/R_res across the swept grid.
#   emax_terminal_burden_vs_emax.png  -- same runs as a bar chart (12 bars).
#   The Monte Carlo treatment of Emax_l lives in MC_emax_lzd.py.

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
# Simulation settings (matches MC_ec50_lzd.py / MC_emax_lzd.py)
# ---------------------------------------------------------------------------

total_h     = TOTAL_H
vanco_start = VANCO_START
t_eval      = canonical_time_grid()
t_days      = t_eval / 24.0

pk           = model_mod.PharmacokineticModel()
immune_model = model_mod.ImmuneResponse()

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid",  total_h, vanco_start + pk.van_duration)

vanco_start_days = vanco_start / 24.0
lzd_start_days   = (vanco_start + pk.van_duration) / 24.0
lzd_end_days     = (vanco_start + pk.van_duration + pk.lzd_duration) / 24.0

EMAX_L_BASE = BASE_PARAMS["Emax_l"]

# Model parameters come from parameters.py. Only Emax_l is varied.



def final_counts(emax_l):
    """Run one deterministic simulation at a given Emax_l; return (final R_b, final R_res)."""
    params_i = {**BASE_PARAMS, "Emax_l": emax_l}
    sol = integrate_model(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(params_i, van_func, lzd_func, immune_model),
    )
    r_b   = sol[-1, 1]
    r_res = sol[-1, 3]
    return r_b, r_res


# ---------------------------------------------------------------------------
# Fine sweep across the centralized Emax_l bounds.
# ---------------------------------------------------------------------------
EMAX_MIN, EMAX_MAX = EMAX_L_BOUNDS
N_POINTS = 111   # step = 0.02

emax_sweep  = np.linspace(EMAX_MIN, EMAX_MAX, N_POINTS)
final_R_b   = np.zeros(N_POINTS)
final_R_res = np.zeros(N_POINTS)

print(f"Sweeping Emax_l across [{EMAX_MIN}, {EMAX_MAX}]  ({N_POINTS} points)...", flush=True)
for i, emax in enumerate(emax_sweep):
    final_R_b[i], final_R_res[i] = final_counts(emax)
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{N_POINTS} complete", flush=True)

# Emax_l values that finish ABOVE the limit of detection (i.e. resistant escape)
escape_R_b   = emax_sweep[final_R_b   > LOD]
escape_R_res = emax_sweep[final_R_res > LOD]

print(f"\nEmax_l values in [{EMAX_MIN}, {EMAX_MAX}] with final R_b   > LOD: "
      f"{escape_R_b.min():.3f}-{escape_R_b.max():.3f} h^-1 ({len(escape_R_b)}/{N_POINTS} points)"
      if len(escape_R_b) else "\nNo swept Emax_l values left final R_b above the LOD.")
print(f"Emax_l values in [{EMAX_MIN}, {EMAX_MAX}] with final R_res > LOD: "
      f"{escape_R_res.min():.3f}-{escape_R_res.max():.3f} h^-1 ({len(escape_R_res)}/{N_POINTS} points)"
      if len(escape_R_res) else "No swept Emax_l values left final R_res above the LOD.")

# ---------------------------------------------------------------------------
# Classification-switch localization with Brent's bracket-preserving method
# ---------------------------------------------------------------------------
def find_reservoir_threshold():
    """Localize the monotone final-R_res switch with Brent's method."""
    def f(emax_l):
        params_i = {**BASE_PARAMS, "Emax_l": emax_l}
        sol = integrate_model(
            model_mod.dual_reservoir_model,
            Y0, t_eval,
            args=(params_i, van_func, lzd_func, immune_model),
        )
        return sol[-1, 3] - LOD

    if f(EMAX_MIN) < 0 or f(EMAX_MAX) > 0:
        return None   # no sign change in range -> no clean threshold to bisect
    return brentq(f, EMAX_MIN, EMAX_MAX, xtol=1e-3)


threshold_R_res = find_reservoir_threshold()

print(f"\nPrecise reservoir-persistence threshold (final R_res crosses LOD): "
      f"{threshold_R_res:.3f} h^-1" if threshold_R_res else "No clean R_res threshold found in range.")

# ---------------------------------------------------------------------------
# FIGURE: final R_b / R_res vs Emax_l, with escape zone(s) highlighted
# Note the direction flip vs the EC50_L sweep: LOW Emax_l is the escape zone.
# ---------------------------------------------------------------------------
FLOOR = LOD * 0.5   # display floor for values that fell below the LOD (i.e. 0)

disp_R_b   = np.where(final_R_b   <= LOD, FLOOR, final_R_b)
disp_R_res = np.where(final_R_res <= LOD, FLOOR, final_R_res)

fig, ax = plt.subplots(figsize=(11, 6.5))

if threshold_R_res is not None:
    ax.axvspan(EMAX_MIN, threshold_R_res, color="lightcoral", alpha=0.15,
               label=f"$R_{{res}}$ escapes ($Emax_L \\leq$ {threshold_R_res:.2g})")

ax.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")
ax.axvline(EMAX_L_BASE, color="gray", ls="-.", lw=1.2, alpha=0.7,
           label=f"Baseline ($Emax_L$ = {EMAX_L_BASE:.2f})")

ax.plot(emax_sweep, disp_R_res, color="indianred", lw=2.0, marker="o", ms=3,
        label="Final $R_{res}$ (reservoir)")
ax.plot(emax_sweep, disp_R_b, color="darkred", lw=2.0, marker="o", ms=3,
        label="Final $R_b$ (blood)")

if threshold_R_res is not None:
    ax.axvline(threshold_R_res, color="indianred", ls="--", lw=1.2, alpha=0.8)

ax.set_yscale("log")
ax.set_xlim(EMAX_MIN, EMAX_MAX)
ax.set_ylim(FLOOR * 0.8, max(FLOOR * 20, final_R_b.max(), final_R_res.max()) * 2)
ax.set_xlabel(r"$Emax_L$ (h$^{-1}$)")
ax.set_ylabel("Final resistant bacterial count (CFU/mL)")
#ax.set_title(r"Resistant Escape Threshold vs $Emax_L$ (linezolid) — end-of-simulation counts",
             #fontsize=19, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(loc="upper right", fontsize=15, framealpha=0.85)

fig.tight_layout()
out_file = os.path.join(FIGURES_DIR, "emax_lzd_threshold_sweep.png")
save_figure(fig, out_file, __file__)
print(f"\nSaved: {out_file}")

# ---------------------------------------------------------------------------
# FIGURE: "Terminal burden vs Emax_L" -- same bar-chart design as
# ec50_terminal_burden_vs_ec50.png (EC50_lzd_threshold_sweep.py), but mirrored
# left-to-right: for Emax_l, LOW values persist and HIGH values are suppressed.
# Runs that clear are drawn as short bars reaching the LOD; runs that persist
# are drawn to their actual final count.
# ---------------------------------------------------------------------------
BAR_STEP = 10   # every 10th sweep point -> 12 evenly spaced bars
idx = np.arange(0, N_POINTS, BAR_STEP)
bar_emax  = emax_sweep[idx]
bar_R_res = final_R_res[idx]

escaped   = bar_R_res > LOD
bar_width = 0.7 * (bar_emax[1] - bar_emax[0])
bar_floor = LOD * 0.5   # log-axis baseline for the bars

fig2, ax2 = plt.subplots(figsize=(10, 5.5))

ax2.bar(bar_emax[~escaped], LOD - bar_floor, bottom=bar_floor, width=bar_width,
        color="#4c72b0", edgecolor="black", linewidth=0.6,
        label="At or below LOD (suppressed)")
ax2.bar(bar_emax[escaped], bar_R_res[escaped] - bar_floor, bottom=bar_floor, width=bar_width,
        color="#c44e52", edgecolor="black", linewidth=0.6,
        label="Persists (final $R_{res}$)")
ax2.axhline(LOD, color="gray", ls="--", lw=1.0, label=f"LOD ({int(LOD)} CFU/mL)")

ax2.set_yscale("log")
ax2.set_ylim(bar_floor, final_R_res.max() * 3)
ax2.set_xlim(0.8, 1.6)   # zoomed on the persistence/suppression transition
ax2.set_xlabel(r"$Emax_L$ (h$^{-1}$)")
ax2.set_ylabel("Terminal $R_{res}$ burden (CFU/mL)")
ax2.set_title(r"Terminal Burden vs $Emax_L$", fontsize=18, fontweight="bold", pad=14)
ax2.grid(True, axis="y", which="both", ls=":", alpha=0.3)
ax2.legend(loc="center right", fontsize=12, framealpha=0.9)

fig2.tight_layout()
out_file = os.path.join(FIGURES_DIR, "emax_terminal_burden_vs_emax.png")
save_figure(fig2, out_file, __file__)
print(f"Saved: {out_file}")
