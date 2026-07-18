# =============================================================================
# Deterministic EC50_L threshold sweep (0.3 - 2.0 mg/L)
#
# Growth rates (rho_S/rho_R/rho_res_S/rho_res_R) were scaled 5x and Emax_l
# raised to match; EC50_L was independently raised 0.7 -> 1.0.
# B_max_reservoir was also lowered 4.5e6 -> 1e4 (see baseline_reservoir_
# clearance.py) so an escaped R_res plateaus far below its old level.
# rho_S/rho_R were lowered 0.80/0.64 -> 0.60/0.55, and Emax_l was UNCOUPLED
# from rho_S and fixed at 0.8 -- since 0.8 then exceeded every species'
# growth rate, linezolid became bactericidal against everything, pushing this
# threshold up to 1.796 mg/L and making ALL FOUR compartments clear at
# baseline (EC50_L = 1.0), the only point in this project where that
# happened.
#
# rho_res_R was then raised 0.145 -> 0.20 (precise clearance threshold
# 0.17594055, found by root-finding) -- now a fitness ADVANTAGE over
# rho_res_S (0.175) rather than a cost. This pulled the EC50_L escape
# threshold back down to 0.573 mg/L: baseline EC50_L = 1.0 sat back ABOVE
# it, so R_b and R_res persisted again, with R_b visibly CLEARING during the
# linezolid course itself before relapsing after treatment ends. BUT at
# rho_res_R = 0.20, R_res's reservoir growth advantage over S_res (0.175)
# was large enough that R_b won the pretreatment race for blood's shared
# carrying capacity outright, crowding S_b out entirely (S_b never
# established a visible infection in any of these sweeps or the MC scripts).
#
# rho_res_R was finally narrowed to 0.1765 -- just above the reservoir's own
# 0.17594055 persistence threshold, but below the ~0.1766-0.1770 point where
# R_b starts winning the blood race -- so S_b CAN also establish a visible
# infection (peak ~4.8e3 CFU/mL). This is a much thinner margin: the EC50_L
# escape threshold moved to 0.988 mg/L, right next to baseline (1.0), so
# roughly half of EC50_L samples in MC_ec50_lzd.py will show the reservoir
# fully clearing rather than persisting. This is accepted -- the point is to
# show S_b establishing, not to guarantee persistence in every sample.
#
# This script runs a fine, deterministic (non-Monte-Carlo) sweep of EC50_L
# across [0.3, 2.0], records the FINAL R_b / R_res at the end of each run,
# selects the EC50_L values that finish above the LOD, and locates the exact
# crossing point for each compartment via root-finding.
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
MODULE_NAME = "model.ClinicalResponse.py"
if not os.path.exists(MODULE_NAME):
    print(f"ERROR: Cannot find '{MODULE_NAME}' in the current directory.", file=sys.stderr)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("model_mod", MODULE_NAME)
model_mod = importlib.util.module_from_spec(spec)
sys.modules["model_mod"] = model_mod
spec.loader.exec_module(model_mod)

# ---------------------------------------------------------------------------
# Simulation settings (matches MC_ec50_lzd.py)
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

rho_S        = 0.60    # lowered from 0.80 for slower post-treatment R_b regrowth; Emax_l auto-follows
rho_R        = 0.55    # directly tuned (~8.3% fitness cost relative to rho_S, down from 20%)

BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_S":        0.175,  # scaled 5x (from 0.035) alongside rho_S
    "rho_res_R":        0.1765,  # narrow window just above the reservoir persistence threshold (0.17594055) so S_b can also establish in blood -- see model.ClinicalResponse.py
    "Emax_v":           0.40,
    "EC50_V":           0.245,
    "Emax_l":           0.8,  # fixed, decoupled from rho_S (was tied for "perfect bacteriostasis")
    "B_max_blood":      6000,
    "B_max_reservoir":  1e4,    # lowered from 4.5e6 so escaped R_res plateaus well above the LOD but far below its old level
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.45,
    "f_r_b":            5e-5,  # restored to original
    "f_b_r":            1e-5,
}

Y0 = [0.0, 0.0, 100.0, 100.0]


def final_counts(ec50_l):
    """Run one deterministic simulation at a given EC50_L; return (final R_b, final R_res)."""
    params_i = {**BASE_PARAMS, "EC50_L": ec50_l}
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
# Fine sweep across EC50_L in [0.3, 2.0]
# ---------------------------------------------------------------------------
EC50_MIN, EC50_MAX = 0.3, 2.0
N_POINTS = 171   # step = 0.01

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
               label=f"$R_{{res}}$ escapes ($EC_{{50,L}} \\geq$ {threshold_R_res:.2f})")
if threshold_R_b is not None:
    ax.axvspan(threshold_R_b, EC50_MAX, color="firebrick", alpha=0.18,
               label=f"$R_b$ escapes ($EC_{{50,L}} \\geq$ {threshold_R_b:.2f})")

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
             fontsize=12, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.85)

fig.tight_layout()
fig.savefig("ec50_lzd_threshold_sweep.png", dpi=300, bbox_inches="tight")
print("\nSaved: ec50_lzd_threshold_sweep.png")

plt.show()
