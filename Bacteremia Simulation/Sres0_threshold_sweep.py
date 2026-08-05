# =============================================================================
# Deterministic S_res_0 sensitivity sweep (1 - 2000 CFU/mL)
#
# baseline_reservoir_clearance.py showed that with model_Bacteremia.py's own
# defaults (S_res_0 = 100, rho_S = 0.61, rho_R = 0.55, rho_res_S = 0.175,
# EC50_L = 1.0, eff_blood = 1.0, B_max_reservoir = 1e4, Emax_l = 0.8 --
# UNCOUPLED from rho_S, which previously made it auto-follow rho_S down to
# 0.6), ALL FOUR compartments briefly cleared at one point in this project's
# history ("Reservoir fully cleared: True"), since fixing Emax_l at 0.8 made
# it exceed every species' growth rate.
#
# rho_res_R was then deliberately raised 0.145 -> 0.20 (precise clearance
# threshold 0.17594055, found by root-finding) -- now a fitness ADVANTAGE
# over rho_res_S (0.175) rather than a cost -- specifically to restore
# "no clinical resolution": R_res persisted indefinitely, while R_b visibly
# CLEARED during the linezolid course before relapsing. BUT at
# rho_res_R = 0.20, R_res's reservoir growth advantage over S_res (0.175)
# was large enough that R_b won the pretreatment race for blood's shared
# carrying capacity outright, crowding S_b out entirely -- no visible
# sensitive-strain infection anywhere in this project's Monte Carlo scripts.
#
# rho_res_R was finally narrowed to 0.1765 -- just above the reservoir's own
# 0.17594055 persistence threshold, but below the point (~0.1766-0.1770)
# where R_b starts winning the blood race outright -- so S_b CAN also
# establish a visible infection (peak ~4.8e3 CFU/mL, confirmed via
# model_Bacteremia.py's own __main__). This is a much thinner
# margin for R_res's own persistence, so some Monte Carlo samples in other
# scripts (which also vary EC50_L, Emax_l, etc.) may show the reservoir
# fully clearing rather than persisting -- accepted, since the point is to
# show S_b establishing.
#
# This script asks: how sensitive is that outcome to the initial reservoir
# sensitive-strain load (S_res_0) alone? All other parameters (including
# R_res_0 = 100) are held at baseline. A fine, deterministic sweep of S_res_0
# is run, the final R_b / R_res are recorded, and the exact S_res_0 at which
# each compartment's outcome flips from "escapes" to "clears" is located via
# root-finding.
#
# Role in the project / relationship to other scripts
# -----------------------------------------------------------------------
# This is a standalone diagnostic: nothing else in the project imports or
# calls it. That's unlike EC50_lzd_threshold_sweep.py, which exists
# specifically to re-derive MC_ec50_lzd.py's ESCAPE_THRESH and so feeds back
# into another script. This one is purely a one-parameter sensitivity
# analysis -- and the odd one out among the project's threshold-sweep
# scripts, since every other sweep (EC50_L, Emax_l, immune response) varies
# a pharmacodynamic parameter, while this one varies an INITIAL CONDITION,
# S_res_0, to see how the outcome depends on the initial reservoir
# sensitive-strain load rather than on drug effect or host response.
#
# Key finding: it empirically demonstrates an asymmetry already asserted in
# MC_initial_conditions.py's comments (around R_RES_0_BASE) -- S_res_0 and
# R_res_0 don't compete for reservoir carrying capacity, so S_res_0 has NO
# effect on whether R_res itself persists there; that's governed purely by
# rho_res_R vs. the reservoir's own persistence threshold (~0.17594055,
# independent of S_res_0). But S_res DOES translocate into blood and compete
# with R_b there for blood's shared carrying capacity (B_max_blood), so
# raising S_res_0 can crowd out R_b even though it never touches R_res's
# fate. That's exactly what the sweep below shows: final R_res is flat
# across S_res_0 (no threshold, or one driven by something other than
# reservoir competition), while final R_b has a real S_res_0-dependent
# escape/clear threshold.
#
# In other words, this script is the evidence that justifies treating
# S_res_0 and R_res_0 as having decoupled effects in
# MC_initial_conditions.py: S_res_0 matters for blood-compartment
# competitive dynamics (R_b), but is irrelevant to reservoir-compartment
# persistence (R_res). That distinction isn't obvious from the ODEs alone --
# it hinges on which compartments actually compete for capacity in
# model_Bacteremia.py -- hence running this sweep to check it empirically.
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
# Simulation settings (matches model_Bacteremia.py's own __main__ block)
# ---------------------------------------------------------------------------
LOD = 10.0   # limit of detection (CFU/mL)

total_h     = 1944  # 21d pre-tx + 4d vancomycin + 42d linezolid + 14d post-tx follow-up
vanco_start = 504
t_eval      = np.linspace(0, total_h, 1150)

pk           = model_mod.PharmacokineticModel()
immune_model = model_mod.ImmuneResponse(k_immune=0.12)

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid",  total_h, vanco_start + pk.van_duration)

rho_S        = 0.61    # lowered from 0.80 for slower post-treatment R_b regrowth; Emax_l auto-follows
rho_R        = 0.55    # directly tuned (~8.3% fitness cost relative to rho_S, down from 20%)

SRES0_BASE = 100   # model_Bacteremia.py's own default

BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_S":        0.175,  # scaled 5x (from 0.035) alongside rho_S
    "rho_res_R":        0.1765,  # narrow window just above the reservoir persistence threshold (0.17594055) so S_b can also establish in blood -- see model_Bacteremia.py
    "Emax_v":           0.40,
    "EC50_V":           0.245,
    "Emax_l":           0.8,  # fixed, decoupled from rho_S (was tied for "perfect bacteriostasis")
    "EC50_L":           1.0,
    "B_max_blood":      6000,
    "B_max_reservoir":  1e4,    # lowered from 4.5e6 so escaped R_res plateaus well above the LOD but far below its old level
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.45,
    "f_r_b":            5e-5,  # restored to original
    "f_b_r":            1e-5,
}

R_RES_0 = 100.0   # fixed


def final_counts(s_res_0):
    """Run one deterministic simulation at a given S_res_0; return (final R_b, final R_res)."""
    y0 = [0.0, 0.0, s_res_0, R_RES_0]
    sol = odeint(
        model_mod.dual_reservoir_model,
        y0, t_eval,
        args=(BASE_PARAMS, van_func, lzd_func, immune_model),
        rtol=1e-8, atol=1e-10, mxstep=5000,
    )
    return sol[-1, 1], sol[-1, 3]


# ---------------------------------------------------------------------------
# Fine sweep across S_res_0 in [1, 2000]
# ---------------------------------------------------------------------------
SRES0_MIN, SRES0_MAX = 1, 2000
N_POINTS = 101

sres0_sweep = np.linspace(SRES0_MIN, SRES0_MAX, N_POINTS)
final_R_b   = np.zeros(N_POINTS)
final_R_res = np.zeros(N_POINTS)

print(f"Sweeping S_res_0 across [{SRES0_MIN}, {SRES0_MAX}]  ({N_POINTS} points)...", flush=True)
for i, s0 in enumerate(sres0_sweep):
    final_R_b[i], final_R_res[i] = final_counts(s0)
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{N_POINTS} complete", flush=True)

escape_R_b   = sres0_sweep[final_R_b   > LOD]
escape_R_res = sres0_sweep[final_R_res > LOD]

print(f"\nS_res_0 values in [{SRES0_MIN}, {SRES0_MAX}] with final R_b   > LOD: "
      f"{escape_R_b.min():.0f}-{escape_R_b.max():.0f} CFU/mL ({len(escape_R_b)}/{N_POINTS} points)"
      if len(escape_R_b) else "\nNo swept S_res_0 values left final R_b above the LOD.")
print(f"S_res_0 values in [{SRES0_MIN}, {SRES0_MAX}] with final R_res > LOD: "
      f"{escape_R_res.min():.0f}-{escape_R_res.max():.0f} CFU/mL ({len(escape_R_res)}/{N_POINTS} points)"
      if len(escape_R_res) else "No swept S_res_0 values left final R_res above the LOD.")


# ---------------------------------------------------------------------------
# Precise crossing point (root of final_count(S_res_0) - LOD) via bisection
# ---------------------------------------------------------------------------
def find_threshold(compartment_idx, lo, hi):
    def f(s_res_0):
        y0 = [0.0, 0.0, s_res_0, R_RES_0]
        sol = odeint(
            model_mod.dual_reservoir_model,
            y0, t_eval,
            args=(BASE_PARAMS, van_func, lzd_func, immune_model),
            rtol=1e-8, atol=1e-10, mxstep=5000,
        )
        return sol[-1, compartment_idx] - LOD

    if f(lo) < 0 or f(hi) > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-2)


# Search the whole swept domain for the crossing point -- with rho_res_R now
# high enough that R_res may never clear at all, a narrow hardcoded bracket
# can miss the real crossing (or find none when there truly is none).
threshold_R_b   = find_threshold(1, SRES0_MIN, SRES0_MAX)
threshold_R_res = find_threshold(3, SRES0_MIN, SRES0_MAX)


def _describe(name, threshold, escapes):
    """Report a compartment's clearance threshold, disambiguating a None
    result as 'always escapes' vs 'always clears' using the swept data
    (a missing brentq root is ambiguous on its own)."""
    if threshold is not None:
        print(f"\nPrecise clearance threshold (final {name} crosses LOD):   "
              f"S_res_0 = {threshold:.1f} CFU/mL")
    elif len(escapes) == N_POINTS:
        print(f"\nNo clean {name} threshold found -- {name} escapes (stays above LOD) "
              f"across the entire swept range [{SRES0_MIN}, {SRES0_MAX}].")
    else:
        print(f"\nNo clean {name} threshold found -- {name} clears across the entire swept range.")


_describe("R_b", threshold_R_b, escape_R_b)
_describe("R_res", threshold_R_res, escape_R_res)

if threshold_R_res is not None:
    below_threshold = SRES0_BASE < threshold_R_res
    print(f"\nmodel_Bacteremia.py's own default S_res_0 = {SRES0_BASE} is "
          f"{'BELOW' if below_threshold else 'AT/ABOVE'} the R_res clearance threshold "
          f"({threshold_R_res:.1f}) -> R_res {'escapes' if below_threshold else 'clears'}")
elif len(escape_R_res) == N_POINTS:
    print(f"\nmodel_Bacteremia.py's own default S_res_0 = {SRES0_BASE} -> "
          f"R_res escapes (stays above LOD) regardless of S_res_0 in "
          f"[{SRES0_MIN}, {SRES0_MAX}] (rho_res_R = 0.1765 is high enough on its own).")
else:
    print(f"\nmodel_Bacteremia.py's own default S_res_0 = {SRES0_BASE} -> "
          f"R_res clears regardless of S_res_0 in [{SRES0_MIN}, {SRES0_MAX}].")

# ---------------------------------------------------------------------------
# FIGURE: final R_b / R_res vs S_res_0, with escape zone(s) highlighted
# ---------------------------------------------------------------------------
FLOOR = LOD * 0.5

disp_R_b   = np.where(final_R_b   <= LOD, FLOOR, final_R_b)
disp_R_res = np.where(final_R_res <= LOD, FLOOR, final_R_res)

fig, ax = plt.subplots(figsize=(11, 6.5))

if threshold_R_res is not None:
    ax.axvspan(SRES0_MIN, threshold_R_res, color="lightcoral", alpha=0.15,
               label=f"$R_{{res}}$ escapes ($S_{{res,0}} \\leq$ {threshold_R_res:.0f})")
if threshold_R_b is not None:
    ax.axvspan(SRES0_MIN, threshold_R_b, color="firebrick", alpha=0.18,
               label=f"$R_b$ escapes ($S_{{res,0}} \\leq$ {threshold_R_b:.0f})")

ax.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")
ax.axvline(SRES0_BASE, color="gray", ls="-.", lw=1.5, alpha=0.8,
           label=f"model_Bacteremia.py default ($S_{{res,0}}$ = {SRES0_BASE})")

ax.plot(sres0_sweep, disp_R_res, color="indianred", lw=2.0, marker="o", ms=3,
        label="Final $R_{res}$ (reservoir)")
ax.plot(sres0_sweep, disp_R_b, color="darkred", lw=2.0, marker="o", ms=3,
        label="Final $R_b$ (blood)")

if threshold_R_res is not None:
    ax.axvline(threshold_R_res, color="indianred", ls="--", lw=1.2, alpha=0.8)
if threshold_R_b is not None:
    ax.axvline(threshold_R_b, color="darkred", ls="--", lw=1.2, alpha=0.8)

ax.set_yscale("log")
ax.set_xlim(SRES0_MIN, SRES0_MAX)
ax.set_ylim(FLOOR * 0.8, max(FLOOR * 20, final_R_b.max(), final_R_res.max()) * 2)
ax.set_xlabel(r"Initial reservoir sensitive load $S_{res,0}$ (CFU/mL)")
ax.set_ylabel("Final resistant bacterial count (CFU/mL)")
ax.set_title(r"Resistant Reservoir Escape vs Initial $S_{res,0}$ — end-of-simulation counts",
             fontsize=19, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(loc="upper right", fontsize=15, framealpha=0.85)

fig.tight_layout()
fig.savefig("sres0_threshold_sweep.png", dpi=300, bbox_inches="tight")
print("\nSaved: sres0_threshold_sweep.png")
