# =============================================================================
# Deterministic S_res_0 sensitivity sweep (1 - 2000 CFU/mL)
#
# baseline_reservoir_clearance.py showed that with model.LinearIM.py's own
# defaults (S_res_0 = 100), the reservoir resistant strain (R_res) does NOT
# clear -- it persists at ~9,175 CFU/mL indefinitely, even though the
# reservoir sensitive strain (S_res) and both blood strains clear normally.
#
# This script asks: how sensitive is that outcome to the initial reservoir
# sensitive-strain load (S_res_0) alone? All other parameters (including
# R_res_0 = 100) are held at baseline. A fine, deterministic sweep of S_res_0
# is run, the final R_b / R_res are recorded, and the exact S_res_0 at which
# each compartment's outcome flips from "escapes" to "clears" is located via
# root-finding.
#
# Mechanism: a larger initial S_res population depletes the reservoir's shared
# logistic carrying capacity faster (and interacts with the exchange terms
# and linezolid's timing), leaving R_res less room to establish itself before
# linezolid engages. A bigger S_res_0 is therefore *protective* against
# resistant escape in the reservoir -- the opposite of what intuition about
# a competing strain might suggest.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODULE_NAME = "model.LinearIM.py"
if not os.path.exists(MODULE_NAME):
    print(f"ERROR: Cannot find '{MODULE_NAME}' in the current directory.", file=sys.stderr)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("model_mod", MODULE_NAME)
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)

# ---------------------------------------------------------------------------
# Simulation settings (matches model.LinearIM.py's own __main__ block)
# ---------------------------------------------------------------------------
LOD = 10.0   # limit of detection (CFU/mL)

total_h     = 1944  # 21d pre-tx + 4d vancomycin + 42d linezolid + 14d post-tx follow-up
vanco_start = 504
t_eval      = np.linspace(0, total_h, 1150)

pk           = model_mod.PharmacokineticModel()
immune_model = model_mod.ImmuneResponse(k_immune=0.12)

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid",  total_h, vanco_start + pk.van_duration)

rho_S        = 0.16
rho_R        = 0.128   # directly tuned (20% fitness cost relative to rho_S)

SRES0_BASE = 100   # model.LinearIM.py's own default

BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_S":        0.035,
    "rho_res_R":        0.024,   # lowered below the 20%-fitness-cost value (0.028) so R_res clears before linezolid ends
    "Emax_v":           0.40,
    "EC50_V":           1.5,
    "Emax_l":           rho_S,
    "EC50_L":           1.0,
    "B_max_blood":      5e5,
    "B_max_reservoir":  4.5e6,
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.45,
    "f_r_b":            5e-5,
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


threshold_R_b   = find_threshold(1, 1, 50)
threshold_R_res = find_threshold(3, 400, 800)

print(f"\nPrecise clearance threshold (final R_b   crosses LOD):   "
      f"S_res_0 = {threshold_R_b:.1f} CFU/mL" if threshold_R_b else
      "\nNo clean R_b threshold found -- R_b clears across the entire swept range.")
print(f"Precise clearance threshold (final R_res crosses LOD):   "
      f"S_res_0 = {threshold_R_res:.1f} CFU/mL" if threshold_R_res else
      "No clean R_res threshold found -- R_res clears across the entire swept range.")

if threshold_R_res is not None:
    below_threshold = SRES0_BASE < threshold_R_res
    print(f"\nmodel.LinearIM.py's own default S_res_0 = {SRES0_BASE} is "
          f"{'BELOW' if below_threshold else 'AT/ABOVE'} the R_res clearance threshold "
          f"({threshold_R_res:.1f}) -> R_res {'escapes' if below_threshold else 'clears'}")
else:
    print(f"\nmodel.LinearIM.py's own default S_res_0 = {SRES0_BASE} -> "
          f"R_res clears regardless of S_res_0 in [{SRES0_MIN}, {SRES0_MAX}] "
          f"(rho_res_R = 0.024 is now low enough on its own).")

# ---------------------------------------------------------------------------
# FIGURE: final R_b / R_res vs S_res_0, with escape zone(s) highlighted
# Note the direction: here HIGH S_res_0 clears the reservoir; LOW S_res_0
# is the escape zone (opposite intuition from a "bigger initial infection").
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
           label=f"model.LinearIM.py default ($S_{{res,0}}$ = {SRES0_BASE})")

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
             fontsize=12, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

fig.tight_layout()
fig.savefig("sres0_threshold_sweep.png", dpi=300, bbox_inches="tight")
print("\nSaved: sres0_threshold_sweep.png")
