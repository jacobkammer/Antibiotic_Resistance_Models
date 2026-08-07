# =============================================================================
# Deterministic Emax_L threshold sweep (0.0 - 1.0 h^-1)
#
# Mirrors EC50_lzd_threshold_sweep.py, but for linezolid's maximum effect
# (Emax_l) instead of its potency (EC50_L). Unlike EC50_L, LOWER Emax_l is
# worse here: below some threshold, linezolid's ceiling effect is too weak to
# hold resistant growth in check and R_b/R_res escape.
#
# B_max_reservoir was lowered 4.5e6 -> 1e4 (see baseline_reservoir_
# clearance.py) so an escaped R_res plateaus far below its old level.
# rho_S/rho_R were lowered 0.80/0.64 -> 0.60/0.55, and Emax_l was UNCOUPLED
# from rho_S and fixed at 0.8 (previously it auto-followed rho_S to 0.6,
# which sat below this sweep's threshold of ~0.6517 h^-1). At that point
# Emax_l = 0.8 sat ABOVE the threshold, and ALL FOUR compartments cleared at
# baseline -- the only point in this project where that happened.
#
# rho_res_R was then raised 0.145 -> 0.20 (precise clearance threshold
# 0.17594055, found by root-finding) -- now a fitness ADVANTAGE over
# rho_res_S (0.175) rather than a cost. This pushed this sweep's own joint
# R_b/R_res escape threshold up from ~0.6517 h^-1 to ~0.9156 h^-1: baseline
# Emax_l = 0.8 sat back BELOW it, so R_b and R_res persisted again, with R_b
# visibly CLEARING during the linezolid course before relapsing. BUT at
# rho_res_R = 0.20, R_res's reservoir growth advantage over S_res (0.175)
# let R_b win the pretreatment race for blood's shared carrying capacity
# outright, crowding S_b out entirely (never a visible infection in this
# sweep or the MC scripts).
#
# rho_res_R was finally narrowed to 0.1765 -- just above the reservoir's own
# 0.17594055 persistence threshold, but below the point where R_b starts
# winning the blood race -- so S_b CAN also establish a visible infection
# (peak ~4.8e3 CFU/mL). This is a much thinner margin: this sweep's escape
# threshold moved to 0.8027 h^-1, right next to baseline (0.8), so roughly
# half of Emax_l samples in MC_emax_lzd.py will show the reservoir fully
# clearing rather than persisting. This is accepted -- the point is to show
# S_b establishing, not to guarantee persistence in every sample.
#
# rho_res_S was then given a 10% growth-rate advantage over rho_res_R
# (0.19415 vs 0.1765, a fitness cost for R mirroring the blood compartment),
# and lzd_res_fraction was lowered 0.45 -> 0.30 to compensate (Emax_l_res
# scales with rho_res_S, so the higher rho_res_S alone would otherwise wipe
# out the reservoir entirely). Net effect: this sweep's escape threshold
# moved UP to 0.8414 h^-1 (confirmed by this script's own bisection below),
# now comfortably above baseline (0.8) rather than straddling it -- so
# baseline Emax_l sits inside the escape/persist zone with margin, and a
# larger majority (~84%, 85/101 grid points, spanning [0.0, 0.84]) of the
# swept range now shows the reservoir persisting rather than clearing.
#
# This script runs a fine, deterministic (non-Monte-Carlo) sweep of Emax_l
# across [0.0, 1.0], records the FINAL R_b / R_res at the end of each run,
# selects the Emax_l values that finish above the LOD, and locates the exact
# crossing point for each compartment via root-finding.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

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
LOD = 10.0   # limit of detection (CFU/mL)

total_h     = 1944  # 21d pre-tx + 4d vancomycin + 42d linezolid + 14d post-tx follow-up
vanco_start = 504
t_eval      = np.linspace(0, total_h, 1150)
t_days      = t_eval / 24.0

pk           = model_mod.PharmacokineticModel()
immune_model = model_mod.ImmuneResponse()

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid",  total_h, vanco_start + pk.van_duration)

vanco_start_days = vanco_start / 24.0
lzd_start_days   = (vanco_start + pk.van_duration) / 24.0
lzd_end_days     = (vanco_start + pk.van_duration + pk.lzd_duration) / 24.0

rho_S        = 0.61    # raised from 0.60 to match MC_emax_lzd.py / the EC50_L and immune-response
                        # threshold sweeps. At the current rho_res_S/rho_res_R/lzd_res_fraction
                        # baseline the Emax_l escape threshold is 0.8414 h^-1 (bisected below)
rho_R        = 0.55    # directly tuned (~8.3% fitness cost relative to rho_S, down from 20%)

EMAX_L_BASE = 0.8   # fixed, decoupled from rho_S = 0.6 (was tied for "perfect bacteriostasis")

BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_R":        0.1765,  # narrow window just above the reservoir persistence threshold (0.17594055) so S_b can also establish in blood -- see model_Bacteremia.py
    "rho_res_S":        0.19415,  # 1.1x rho_res_R -- sensitive strain grows 10% faster in the reservoir
    "Emax_v":           0.40,
    "EC50_V":           0.245,
    "EC50_L":           1.0,
    "B_max_blood":      6000,
    "B_max_reservoir":  1e4,    # lowered from 4.5e6 so escaped R_res plateaus well above the LOD but far below its old level
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.30,    # lowered from 0.45 -- Emax_l_res scales with rho_res_S; keeps R_res persistent despite S's reservoir growth advantage
    "f_r_b":            5e-5,  # restored to original
    "f_b_r":            1e-5,
}

Y0 = [0.0, 0.0, 100.0, 100.0]


def final_counts(emax_l):
    """Run one deterministic simulation at a given Emax_l; return (final R_b, final R_res)."""
    params_i = {**BASE_PARAMS, "Emax_l": emax_l}
    sol = odeint(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(params_i, van_func, lzd_func, immune_model),
        rtol=1e-7, atol=1e-9, mxstep=5000,
    )
    r_b   = sol[-1, 1]
    r_res = sol[-1, 3]
    return r_b, r_res


# ---------------------------------------------------------------------------
# Fine sweep across Emax_l in [0.0, 1.0]
# ---------------------------------------------------------------------------
EMAX_MIN, EMAX_MAX = 0.0, 1.0
N_POINTS = 101   # step = 0.01

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
# Precise crossing point (root of final_count(Emax_l) - LOD) via bisection
# ---------------------------------------------------------------------------
def find_threshold(compartment_idx):
    """Bisect for the Emax_l where the final count first crosses the LOD."""
    def f(emax_l):
        params_i = {**BASE_PARAMS, "Emax_l": emax_l}
        sol = odeint(
            model_mod.dual_reservoir_model,
            Y0, t_eval,
            args=(params_i, van_func, lzd_func, immune_model),
            rtol=1e-7, atol=1e-9, mxstep=5000,
        )
        return sol[-1, compartment_idx] - LOD

    if f(EMAX_MIN) < 0 or f(EMAX_MAX) > 0:
        return None   # no sign change in range -> no clean threshold to bisect
    return brentq(f, EMAX_MIN, EMAX_MAX, xtol=1e-4)


threshold_R_b   = find_threshold(1)
threshold_R_res = find_threshold(3)

print(f"\nPrecise escape threshold (final R_b   crosses LOD):   "
      f"{threshold_R_b:.4f} h^-1" if threshold_R_b else "\nNo clean R_b   threshold found in range.")
print(f"Precise escape threshold (final R_res crosses LOD):   "
      f"{threshold_R_res:.4f} h^-1" if threshold_R_res else "No clean R_res threshold found in range.")

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
if threshold_R_b is not None:
    ax.axvspan(EMAX_MIN, threshold_R_b, color="firebrick", alpha=0.18,
               label=f"$R_b$ escapes ($Emax_L \\leq$ {threshold_R_b:.2g})")

ax.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")
ax.axvline(EMAX_L_BASE, color="gray", ls="-.", lw=1.2, alpha=0.7,
           label=f"Baseline ($Emax_L$ = {EMAX_L_BASE:.2f})")

ax.plot(emax_sweep, disp_R_res, color="indianred", lw=2.0, marker="o", ms=3,
        label="Final $R_{res}$ (reservoir)")
ax.plot(emax_sweep, disp_R_b, color="darkred", lw=2.0, marker="o", ms=3,
        label="Final $R_b$ (blood)")

if threshold_R_res is not None:
    ax.axvline(threshold_R_res, color="indianred", ls="--", lw=1.2, alpha=0.8)
if threshold_R_b is not None:
    ax.axvline(threshold_R_b, color="darkred", ls="--", lw=1.2, alpha=0.8)

ax.set_yscale("log")
ax.set_xlim(EMAX_MIN, EMAX_MAX)
ax.set_ylim(FLOOR * 0.8, max(FLOOR * 20, final_R_b.max(), final_R_res.max()) * 2)
ax.set_xlabel(r"$Emax_L$ (h$^{-1}$)")
ax.set_ylabel("Final resistant bacterial count (CFU/mL)")
ax.set_title(r"Resistant Escape Threshold vs $Emax_L$ (linezolid) — end-of-simulation counts",
             fontsize=19, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(loc="upper right", fontsize=15, framealpha=0.85)

fig.tight_layout()
fig.savefig("emax_lzd_threshold_sweep.png", dpi=300, bbox_inches="tight")
print("\nSaved: emax_lzd_threshold_sweep.png")

# ---------------------------------------------------------------------------
# FIGURE: "Terminal burden vs Emax_L" -- same design as
# ec50_terminal_burden_vs_ec50.png (EC50_lzd_threshold_sweep.py), but mirrored
# left-to-right: for Emax_l, LOW values are the escape zone and HIGH values
# are suppression (the opposite of EC50_L). One marker per swept Emax_l value
# (subsampled for legibility), open circles pinned to the LOD for runs that
# clear, filled circles at their actual final count for runs that escape,
# split by the bisected threshold.
# ---------------------------------------------------------------------------
SUBSAMPLE = 24
idx = np.linspace(0, N_POINTS - 1, SUBSAMPLE).astype(int)
scatter_emax  = emax_sweep[idx]
scatter_R_res = final_R_res[idx]

escaped = scatter_R_res > LOD
thr = threshold_R_res if threshold_R_res is not None else EMAX_MAX

fig2, ax2 = plt.subplots(figsize=(10, 5.5))

ax2.axvspan(EMAX_MIN, thr, color="#c44e52", alpha=0.12)
ax2.axvspan(thr, EMAX_MAX, color="#4c72b0", alpha=0.12)
ax2.text(0.02, 0.94, "escape", transform=ax2.transAxes, fontsize=15,
         color="#8c2f34", fontweight="bold", va="top")
ax2.text(0.98, 0.94, "suppression", transform=ax2.transAxes, fontsize=15,
         color="#2f4f7a", fontweight="bold", va="top", ha="right")

ax2.axhline(LOD, color="gray", ls="--", lw=1.0, label=f"LOD ({int(LOD)} CFU/mL)")
ax2.axvline(thr, color="black", ls="--", lw=1.2, label=fr"Switch ($Emax_L$ = {thr:.2g})")

ax2.scatter(scatter_emax[~escaped], np.full((~escaped).sum(), LOD),
            facecolors="none", edgecolors="#4c72b0", s=70, linewidths=1.6,
            label="At or below LOD (suppressed)")
ax2.scatter(scatter_emax[escaped], scatter_R_res[escaped],
            facecolors="#c44e52", edgecolors="black", s=70, linewidths=0.6,
            label="Escape (final $R_{res}$)")

ax2.set_yscale("log")
ax2.set_ylim(LOD * 0.5, final_R_res.max() * 3)
ax2.set_xlim(EMAX_MIN, EMAX_MAX)
ax2.set_xlabel(r"$Emax_L$ (h$^{-1}$)")
ax2.set_ylabel("Terminal $R_{res}$ burden (CFU/mL)")
ax2.set_title(fr"Terminal Burden vs $Emax_L$ — switch at {thr:.2g} h$^{{-1}}$",
              fontsize=18, fontweight="bold", pad=14)
ax2.grid(True, which="both", ls=":", alpha=0.3)
ax2.legend(loc="center right", fontsize=12, framealpha=0.9)

fig2.tight_layout()
fig2.savefig("emax_terminal_burden_vs_emax.png", dpi=300, bbox_inches="tight")
print("Saved: emax_terminal_burden_vs_emax.png")
