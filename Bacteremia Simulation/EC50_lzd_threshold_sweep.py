# =============================================================================
# Deterministic EC50_L threshold sweep (0.05-3.3 mg/L).
#
# Every parameter except EC50_L comes from the shared BASE_PARAMS. The fixed
# EC50_L baseline and the MC sampling mean are both 0.56 mg/L. Higher EC50_L
# weakens linezolid potency and is the reservoir-persistence side of the
# switch. The accepted final-R_res transition is 0.76 mg/L; the companion MC
# analysis gives 22.0% bacteremia relapse (220/1000).
#
# What it does, step by step:
#   1. Runs one fixed-parameter ODE simulation per EC50_L value on a 131-point
#      grid across [0.05, 3.3] mg/L, keeping every other parameter
#      at BASE_PARAMS, and records only the final R_b/R_res at end-of-sim.
#   2. Flags which EC50_L values finish above the LOD (10 CFU/mL) -- i.e.
#      where the resistant strain persists/relapses rather than clearing.
#   3. Uses scipy.optimize.brentq to localize the reservoir classification
#      switch rather than relying on the coarse grid resolution.
#   4. Plots final R_b/R_res vs. EC50_L (log y-axis), shades the escape
#      zone(s) above each threshold, and saves ec50_lzd_threshold_sweep.png.
#
# This is the deterministic companion to MC_ec50_lzd.py and independently
# derives the threshold used to interpret its reservoir-persistence fraction.
# this file creates two figures:
#   1. ec50_lzd_threshold_sweep.png - final R_b/R_res vs. EC50_L (log y-axis), shades the escape
#      zone(s) above each threshold
#   2. ec50_terminal_burden_vs_ec50.png - bar chart of terminal R_res burden vs. EC50_L at
#      12 values subsampled from the same sweep: bars reach the LOD for runs that clear and
#      the true final count for runs that persist. NOT a zoom of figure 1.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from parameters import (
    BASE_PARAMS, LOD, TOTAL_H, VANCO_START, Y0, canonical_time_grid, save_figure,
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
#   ec50_lzd_threshold_sweep.png     -- final R_b and R_res across the 131-point
#       grid, persistence zone shaded, reservoir threshold marked.
#   ec50_terminal_burden_vs_ec50.png -- the same runs as a bar chart:
#       12 bars subsampled from the same grid, R_res only.
#   Each marker/point is ONE fixed-parameter run, not one virtual patient. The
#   Monte Carlo treatment of EC50_L lives in MC_ec50_lzd.py.

# ---------------------------------------------------------------------------
# Load model
# import model_Bacteremia.py will also work, however the code below is more explicit
# and allows for better error handling.
# ---------------------------------------------------------------------------
MODULE_NAME = "model_Bacteremia.py"#file to load
if not os.path.exists(MODULE_NAME):
    print(f"ERROR: Cannot find '{MODULE_NAME}' in the current directory.", file=sys.stderr)# check if file exists
    sys.exit(1)

spec = importlib.util.spec_from_file_location("model_mod", MODULE_NAME)#create spec from file location
model_mod = importlib.util.module_from_spec(spec)# create module from spec
sys.modules["model_mod"] = model_mod#add module to sys.modules
spec.loader.exec_module(model_mod)#execute module

# ---------------------------------------------------------------------------
# Simulation settings use the same shared baseline as MC_ec50_lzd.py.
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
# Model parameters come from parameters.py (single source of truth).


#-------------------------------------------------------------------------------------
# Function to run deterministic simulation. Used by sweep and by find_threshold to evaluate
# the effectiveness of linezolid at different EC50_L values.
#-------------------------------------------------------------------------------------
def final_counts(ec50_l):
    """Run one deterministic simulation at a given EC50_L; return (final R_b, final R_res)."""
    params_i = {**BASE_PARAMS, "EC50_L": ec50_l} #creates a parameter dictionary with the given EC50_L value
    sol = integrate_model(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(params_i, van_func, lzd_func, immune_model),
    )
    r_b   = sol[-1, 1] #final R_b value
    r_res = sol[-1, 3] #final R_res value
    return r_b, r_res #returns the final values of R_b and R_res as a tuple


# ---------------------------------------------------------------------------
# Fine sweep across EC50_L in [0.05, 3.3], covering the configured Monte Carlo
# distribution while leaving room for a threshold to move after recalibration.
# ---------------------------------------------------------------------------
EC50_MIN, EC50_MAX = 0.05, 3.3
N_POINTS = 131   # matches the Monte Carlo companion sweep

ec50_sweep = np.linspace(EC50_MIN, EC50_MAX, N_POINTS)
final_R_b   = np.zeros(N_POINTS)
final_R_res = np.zeros(N_POINTS)

print(f"Sweeping EC50_L across [{EC50_MIN}, {EC50_MAX}]  ({N_POINTS} points)...", flush=True)
for i, ec in enumerate(ec50_sweep):
    final_R_b[i], final_R_res[i] = final_counts(ec)
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{N_POINTS} complete", flush=True)

# EC50_L values that finish ABOVE the limit of detection (i.e. resistant escape)
escape_R_b   = ec50_sweep[final_R_b   > LOD]
escape_R_res = ec50_sweep[final_R_res > LOD]

print(f"\nEC50_L values in [{EC50_MIN}, {EC50_MAX}] with final R_b   > LOD: "
      f"{escape_R_b.min():.2f}-{escape_R_b.max():.2f} mg/L ({len(escape_R_b)}/{N_POINTS} points)"
      if len(escape_R_b) else "\nNo swept EC50_L values pushed final R_b above the LOD.")
print(f"EC50_L values in [{EC50_MIN}, {EC50_MAX}] with final R_res > LOD: "
      f"{escape_R_res.min():.2f}-{escape_R_res.max():.2f} mg/L ({len(escape_R_res)}/{N_POINTS} points)"
      if len(escape_R_res) else "No swept EC50_L values pushed final R_res above the LOD.")

# ---------------------------------------------------------------------------
# Classification-switch localization with Brent's bracket-preserving method
#---------------------------------------------------------------------------
#How it works: f(ec50_l) runs one full ODE simulation at a candidate EC50_L 
#and returns final_count - LOD — positive if the compartment ends up above the detection limit, 
#negative if below. scipy.optimize.brentq(f, EC50_MIN, EC50_MAX, xtol=1e-3) 
# (imported at the top) contracts the sign-changing bracket to 0.001 mg/L.
# Because hard extinction makes f discontinuous, the result localizes the
# classification switch; final burden need not equal the LOD there.
# Only the reservoir (R_res) switch is localized; relapse in blood is
# classified from the R_b trajectory in MC_ec50_lzd.py.
# The f(EC50_MIN) > 0 or f(EC50_MAX) < 0 guard checks brentq's precondition --
# opposite signs at the two endpoints (a genuine crossing exists in range) --
# and returns None instead of letting brentq raise.
# ---------------------------------------------------------------------------
def find_reservoir_threshold():
    """Localize the monotone final-R_res switch with Brent's method."""
    def f(ec50_l):
        params_i = {**BASE_PARAMS, "EC50_L": ec50_l}
        sol = integrate_model(
            model_mod.dual_reservoir_model,
            Y0, t_eval,
            args=(params_i, van_func, lzd_func, immune_model),
        )
        return sol[-1, 3] - LOD

    if f(EC50_MIN) > 0 or f(EC50_MAX) < 0:
        return None   # no sign change in range -> no clean threshold to bisect
    return brentq(f, EC50_MIN, EC50_MAX, xtol=1e-3)


threshold_R_res = find_reservoir_threshold()

print(f"\nPrecise reservoir-persistence threshold (final R_res crosses LOD): "
      f"{threshold_R_res:.3f} mg/L" if threshold_R_res else "No clean R_res threshold found in range.")

# ---------------------------------------------------------------------------
# FIGURE: final R_b / R_res vs EC50_L, with escape zone(s) highlighted
# ---------------------------------------------------------------------------
FLOOR = LOD * 0.5   # display floor for values that fell below the LOD (i.e. 0)

disp_R_b   = np.where(final_R_b   <= LOD, FLOOR, final_R_b)
disp_R_res = np.where(final_R_res <= LOD, FLOOR, final_R_res)

fig, ax = plt.subplots(figsize=(11, 6.5))

if threshold_R_res is not None:
    ax.axvspan(threshold_R_res, EC50_MAX, color="lightcoral", alpha=0.15,
               label=f"$R_{{res}}$ escapes ($EC_{{50,L}} \\geq$ {threshold_R_res:.2g})")

ax.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")

ax.plot(ec50_sweep, disp_R_res, color="indianred", lw=2.0, marker="o", ms=3,
        label="Final $R_{res}$ (reservoir)")
ax.plot(ec50_sweep, disp_R_b, color="darkred", lw=2.0, marker="o", ms=3,
        label="Final $R_b$ (blood)")

if threshold_R_res is not None:
    ax.axvline(threshold_R_res, color="indianred", ls="--", lw=1.2, alpha=0.8)

ax.set_yscale("log")
ax.set_xlim(EC50_MIN, EC50_MAX)
ax.set_ylim(FLOOR * 0.8, max(FLOOR * 20, final_R_b.max(), final_R_res.max()) * 2)
ax.set_xlabel(r"$EC_{50,L}$ (mg/L)")
ax.set_ylabel("Final resistant bacterial count (CFU/mL)")
#ax.set_title(r"Resistant Escape Threshold vs $EC_{50,L}$ (linezolid) — end-of-simulation counts",
             #fontsize=19, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(loc="center right", fontsize=15, framealpha=0.85)   # moved off the transition

fig.tight_layout()
out_file = os.path.join(FIGURES_DIR, "ec50_lzd_threshold_sweep.png")
save_figure(fig, out_file, __file__)
print(f"\nSaved: {out_file}")

# ---------------------------------------------------------------------------
# FIGURE: "Terminal burden vs EC50_L" -- bar chart of terminal R_res at a
# subsample of the swept EC50_L values. Runs that clear are drawn as short bars
# reaching the LOD; runs that persist are drawn to their actual final count.
# ---------------------------------------------------------------------------
BAR_STEP = 11   # every 11th sweep point -> 12 evenly spaced bars
idx = np.arange(0, N_POINTS, BAR_STEP)
bar_ec50  = ec50_sweep[idx]
bar_R_res = final_R_res[idx]

escaped   = bar_R_res > LOD
bar_width = 0.7 * (bar_ec50[1] - bar_ec50[0])
bar_floor = LOD * 0.5   # log-axis baseline for the bars

fig2, ax2 = plt.subplots(figsize=(10, 5.5))

ax2.bar(bar_ec50[~escaped], LOD - bar_floor, bottom=bar_floor, width=bar_width,
        color="#4c72b0", edgecolor="black", linewidth=0.6,
        label="At or below LOD (suppressed)")
ax2.bar(bar_ec50[escaped], bar_R_res[escaped] - bar_floor, bottom=bar_floor, width=bar_width,
        color="#c44e52", edgecolor="black", linewidth=0.6,
        label="Persists (final $R_{res}$)")
ax2.axhline(LOD, color="gray", ls="--", lw=1.0, label=f"LOD ({int(LOD)} CFU/mL)")

ax2.set_yscale("log")
ax2.set_ylim(bar_floor, final_R_res.max() * 3)
ax2.set_xlim(EC50_MIN, 2.0)
ax2.set_xlabel(r"$EC_{50,L}$ (mg/L)")
ax2.set_ylabel("Terminal $R_{res}$ burden (CFU/mL)")
ax2.set_title(r"Terminal Burden vs $EC_{50,L}$", fontsize=18, fontweight="bold", pad=14)
ax2.grid(True, axis="y", which="both", ls=":", alpha=0.3)
ax2.legend(loc="center left", fontsize=12, framealpha=0.9)

fig2.tight_layout()
out_file = os.path.join(FIGURES_DIR, "ec50_terminal_burden_vs_ec50.png")
save_figure(fig2, out_file, __file__)
print(f"Saved: {out_file}")
