# =============================================================================
# Deterministic EC50_L threshold sweep (0.3 - 2.0 mg/L)
#
#
#
# rho_res_R was finally narrowed to 0.1765 -- just above the reservoir's own
# 0.17594055 persistence threshold, but below the ~0.1766-0.1770 point where
# R_b starts winning the blood race -- so S_b CAN also establish a visible
# infection (peak ~4.8e3 CFU/mL). This is a much thinner margin: the EC50_L
# escape threshold moved to 0.988 mg/L at rho_S=0.61 -- right next to baseline
# (1.0), so roughly half of EC50_L samples would show the reservoir fully
# clearing rather than persisting. This is accepted -- the point is to show
# S_b establishing, not to guarantee persistence in every sample.
#
# rho_S was then raised 0.60 -> 0.61 (see MC_ec50_lzd.py's own rho_S
# sensitivity analysis: a vancomycin-driven competitive-release effect, where
# S_b grows larger pre-vancomycin then gets cleared, leaving more of blood's
# shared carrying capacity open for R_b to claim), which pulled the escape
# threshold down further to 0.932 mg/L -- confirmed by this script's own
# bisection below.
#
# rho_res_S was then given a 10% growth-rate advantage over rho_res_R
# (0.19415 vs 0.1765, a fitness cost for R mirroring the blood compartment),
# and lzd_res_fraction was lowered 0.45 -> 0.30 to compensate (Emax_l_res
# scales with rho_res_S, so the higher rho_res_S alone would otherwise wipe
# out the reservoir entirely). Net effect: the EC50_L escape threshold moved
# DOWN to 0.868 mg/L (confirmed by this script's own bisection below) --
# further from baseline (1.0) than before, so a larger majority (~67%,
# 114/171 grid points, spanning [0.87, 2.0]) of the swept range now shows
# the reservoir persisting rather than clearing.
#
# This script runs a fine, deterministic (non-Monte-Carlo) sweep of EC50_L
# across [0.1, 1.3], records the FINAL R_b / R_res at the end of each run,
# selects the EC50_L values that finish above the LOD, and locates the exact
# crossing point for each compartment via root-finding.
#
# What it does, step by step:
#   1. Runs one fixed-parameter ODE simulation per EC50_L value on a 121-point
#      grid (step 0.01) across [0.1, 1.3] mg/L, keeping every other parameter
#      at BASE_PARAMS, and records only the final R_b/R_res at end-of-sim.
#   2. Flags which EC50_L values finish above the LOD (10 CFU/mL) -- i.e.
#      where the resistant strain persists/relapses rather than clearing.
#   3. Uses scipy.optimize.brentq (bisection) to pinpoint the *precise*
#      EC50_L where final R_b (and, separately, final R_res) crosses the LOD,
#      rather than relying on the coarse grid resolution.
#   4. Plots final R_b/R_res vs. EC50_L (log y-axis), shades the escape
#      zone(s) above each threshold, and saves ec50_lzd_threshold_sweep.png.
#
# This is the tool used to derive/verify the exact ESCAPE_THRESH value
# consumed by MC_ec50_lzd.py: every time rho_res_S/rho_res_R was retuned in
# that file's history (see narrative above), this script was rerun to
# re-locate where the threshold moved to.
# this file creates two figures:
#   1. ec50_lzd_threshold_sweep.png - final R_b/R_res vs. EC50_L (log y-axis), shades the escape
#      zone(s) above each threshold
#   2. ec50_terminal_burden_vs_ec50.png - terminal R_res burden vs. EC50_L as a categorical
#      outcome plot: 24 markers subsampled from the same sweep, open circles pinned at the LOD
#      for runs that clear and filled circles at the true final count for runs that escape,
#      with the bisected switch dividing the suppression and escape zones. NOT a zoom of
#      figure 1 -- an earlier version of this header described it as one, which was wrong.
#
# EC50_L was then consolidated onto a SINGLE project-wide value. Previously
# the project carried two: a fixed baseline EC50_L = 1.0 used by every script
# that holds EC50_L constant, and a sampling mean EC50_L_BASE = 0.6 used by
# MC_ec50_lzd.py. That also contradicted MC_ec50_lzd.py's own docstring claim
# that its sampling is "mean-corrected so E[EC50_L] lands exactly on
# baseline". Both are now 0.6 mg/L.
#
# Lowering the fixed EC50_L 1.0 -> 0.6 makes linezolid more potent, which on
# its own put the baseline BELOW the EC50_L escape threshold: the reservoir
# cleared at baseline and the k_immune sweep lost its threshold entirely (no
# k_immune in [0, 3] produced relapse -- even zero immune clearance cleared
# the infection). rho_S was therefore raised 0.61 -> 0.72 h^-1, with rho_R
# rescaled 0.55 -> 0.649 to preserve the ~10% fitness cost, restoring the
# reservoir to the escape side. The mechanism is indirect: the reservoir kill
# rate is Emax_l_res = Emax_l * (rho_res_S / rho_S), so a HIGHER rho_S
# weakens reservoir killing. rho_res_R/rho_res_S were tried first and
# rejected -- every value of either that reached baseline escape collapsed
# peak S_b from ~4.8e3 to ~2 CFU/mL (R_b crowding S_b out of blood).
#
# Net effect here: the bisected threshold moved 0.868 -> 0.488 mg/L (both
# compartments, agreeing to within the 1e-3 mg/L tolerance). Unaffected by
# the EC50_L consolidation itself, since this sweep overrides EC50_L; it
# responds only to the rho_S change.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq

from parameters import BASE_PARAMS, LOD, TOTAL_H, VANCO_START, Y0

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
#   ec50_lzd_threshold_sweep.png     -- final R_b and R_res across the 121-point
#       grid, escape zones shaded, both bisected thresholds marked.
#   ec50_terminal_burden_vs_ec50.png -- the same runs as a categorical outcome
#       plot: 24 markers subsampled from the same grid, R_res only.
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
# Simulation settings (matches MC_ec50_lzd.py: rho_S=0.72, EC50_L escape
# threshold at 0.488 mg/L with the current rho_res_S/rho_res_R/lzd_res_fraction
# baseline -- see header narrative above)
# ---------------------------------------------------------------------------

total_h     = TOTAL_H
vanco_start = VANCO_START
t_eval      = np.linspace(0, total_h, 1150)
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
    sol = odeint(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(params_i, van_func, lzd_func, immune_model),
        rtol=1e-7, atol=1e-9, mxstep=5000,
    )
    r_b   = sol[-1, 1] #final R_b value
    r_res = sol[-1, 3] #final R_res value
    return r_b, r_res #returns the final values of R_b and R_res as a tuple


# ---------------------------------------------------------------------------
# Fine sweep across EC50_L in [0.1, 1.3]
# ---------------------------------------------------------------------------
# Re-centred from [0.3, 2.0]. That bracket was chosen when the threshold sat at
# 0.868 mg/L, roughly a third of the way along; at the current 0.488 it left the
# suppressed zone squeezed into the leftmost ~11% of the axis and both figures
# read as 'escape everywhere'. [0.1, 1.3] restores the threshold to ~32% of the
# span while still showing a clear escape plateau to its right.
EC50_MIN, EC50_MAX = 0.1, 1.3
N_POINTS = 121   # step = 0.01

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
# Precise crossing point (root of final_count(EC50_L) - LOD) via bisection
#---------------------------------------------------------------------------
#How it works: f(ec50_l) runs one full ODE simulation at a candidate EC50_L 
#and returns final_count - LOD — positive if the compartment ends up above the detection limit, 
#negative if below. scipy.optimize.brentq(f, EC50_MIN, EC50_MAX, xtol=1e-3) 
#(imported at the top, from scipy.optimize import brentq) then does bisection-based root-finding on that function, 
#homing in on the exact EC50_L where f crosses zero — i.e., where the final count exactly equals the LOD — to a tolerance of 0.001 mg/L.
# It's called once for compartment_idx=1 (R_b) and once for compartment_idx=3 (R_res), which is why the earlier rerun printed two threshold values
#(both landing at 0.488 mg/L under the current baseline, agreeing to within the 1e-3 mg/L tolerance,
#since R_b and R_res share the same eff_blood and cross together).
##The f(EC50_MIN) > 0 or f(EC50_MAX) < 0 guard is brentq's precondition — it requires the function to have opposite signs at the two endpoints
#(i.e., a genuine crossing exists in range) before bisecting, otherwise it returns None rather than raising.
# ---------------------------------------------------------------------------
def find_threshold(compartment_idx):
    """Bisect for the EC50_L where the final count first crosses the LOD."""
    def f(ec50_l):
        params_i = {**BASE_PARAMS, "EC50_L": ec50_l}
        sol = odeint(
            model_mod.dual_reservoir_model,
            Y0, t_eval,
            args=(params_i, van_func, lzd_func, immune_model),
            rtol=1e-7, atol=1e-9, mxstep=5000,
        )
        return sol[-1, compartment_idx] - LOD

    if f(EC50_MIN) > 0 or f(EC50_MAX) < 0:
        return None   # no sign change in range -> no clean threshold to bisect
    return brentq(f, EC50_MIN, EC50_MAX, xtol=1e-3)


threshold_R_b   = find_threshold(1)
threshold_R_res = find_threshold(3)

print(f"\nPrecise escape threshold (final R_b   crosses LOD):   "
      f"{threshold_R_b:.3f} mg/L" if threshold_R_b else "\nNo clean R_b   threshold found in range.")
print(f"Precise escape threshold (final R_res crosses LOD):   "
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
if threshold_R_b is not None:
    ax.axvspan(threshold_R_b, EC50_MAX, color="firebrick", alpha=0.18,
               label=f"$R_b$ escapes ($EC_{{50,L}} \\geq$ {threshold_R_b:.2g})")

ax.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")

ax.plot(ec50_sweep, disp_R_res, color="indianred", lw=2.0, marker="o", ms=3,
        label="Final $R_{res}$ (reservoir)")
ax.plot(ec50_sweep, disp_R_b, color="darkred", lw=2.0, marker="o", ms=3,
        label="Final $R_b$ (blood)")

if threshold_R_res is not None:
    ax.axvline(threshold_R_res, color="indianred", ls="--", lw=1.2, alpha=0.8)
if threshold_R_b is not None:
    ax.axvline(threshold_R_b, color="darkred", ls="--", lw=1.2, alpha=0.8)

ax.set_yscale("log")
ax.set_xlim(EC50_MIN, EC50_MAX)
ax.set_ylim(FLOOR * 0.8, max(FLOOR * 20, final_R_b.max(), final_R_res.max()) * 2)
ax.set_xlabel(r"$EC_{50,L}$ (mg/L)")
ax.set_ylabel("Final resistant bacterial count (CFU/mL)")
ax.set_title(r"Resistant Escape Threshold vs $EC_{50,L}$ (linezolid) — end-of-simulation counts",
             fontsize=19, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(loc="center right", fontsize=15, framealpha=0.85)   # moved off the transition

fig.tight_layout()
out_file = os.path.join(FIGURES_DIR, "ec50_lzd_threshold_sweep.png")
fig.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out_file}")

# ---------------------------------------------------------------------------
# FIGURE: "Terminal burden vs EC50_L" -- recreates, with real simulated data,
# the schematic Panel B design sketched earlier (invented data, illustrative
# only): one marker per swept EC50_L value (subsampled for legibility), open
# circles pinned to the LOD for runs that clear, filled circles at their
# actual final count for runs that escape, split by the bisected threshold.
# ---------------------------------------------------------------------------
SUBSAMPLE = 24
idx = np.linspace(0, N_POINTS - 1, SUBSAMPLE).astype(int)
scatter_ec50  = ec50_sweep[idx]
scatter_R_res = final_R_res[idx]

escaped = scatter_R_res > LOD
thr = threshold_R_res if threshold_R_res is not None else EC50_MIN

fig2, ax2 = plt.subplots(figsize=(10, 5.5))

ax2.axvspan(EC50_MIN, thr, color="#4c72b0", alpha=0.12)
ax2.axvspan(thr, EC50_MAX, color="#c44e52", alpha=0.12)
ax2.text(0.02, 0.94, "suppression", transform=ax2.transAxes, fontsize=15,
         color="#2f4f7a", fontweight="bold", va="top")
ax2.text(0.98, 0.94, "escape", transform=ax2.transAxes, fontsize=15,
         color="#8c2f34", fontweight="bold", va="top", ha="right")

ax2.axhline(LOD, color="gray", ls="--", lw=1.0, label=f"LOD ({int(LOD)} CFU/mL)")
ax2.axvline(thr, color="black", ls="--", lw=1.2, label=fr"Switch ($EC_{{50,L}}$ = {thr:.2g})")

ax2.scatter(scatter_ec50[~escaped], np.full((~escaped).sum(), LOD),
            facecolors="none", edgecolors="#4c72b0", s=70, linewidths=1.6,
            label="At or below LOD (suppressed)")
ax2.scatter(scatter_ec50[escaped], scatter_R_res[escaped],
            facecolors="#c44e52", edgecolors="black", s=70, linewidths=0.6,
            label="Escape (final $R_{res}$)")

ax2.set_yscale("log")
ax2.set_ylim(LOD * 0.5, final_R_res.max() * 3)
ax2.set_xlim(EC50_MIN, EC50_MAX)
ax2.set_xlabel(r"$EC_{50,L}$ (mg/L)")
ax2.set_ylabel("Terminal $R_{res}$ burden (CFU/mL)")
ax2.set_title(fr"Terminal Burden vs $EC_{{50,L}}$ — switch at {thr:.2g} mg/L",
              fontsize=18, fontweight="bold", pad=14)
ax2.grid(True, which="both", ls=":", alpha=0.3)
ax2.legend(loc="center left", fontsize=12, framealpha=0.9)

fig2.tight_layout()
out_file = os.path.join(FIGURES_DIR, "ec50_terminal_burden_vs_ec50.png")
fig2.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"Saved: {out_file}")
