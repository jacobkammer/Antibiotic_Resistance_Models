# =============================================================================
# Deterministic S_res_0 threshold sweep (1 - 2000 CFU/mL)
#
# NOTE ON PROVENANCE: the original Sres0_threshold_sweep.py source was lost;
# this file was reconstructed from the surviving bytecode
# (__pycache__/Sres0_threshold_sweep.cpython-312.pyc), which preserved the
# full set of module constants, names, string literals and the signatures and
# call structure of final_counts / find_threshold / _describe. The numerical
# behaviour therefore matches the original exactly; comment prose and minor
# plot styling are reconstructions.
#
# Mirrors EC50_lzd_threshold_sweep.py / Emax_lzd_threshold_sweep.py /
# ImmuneResponse_threshold_sweep.py, but sweeps the INITIAL reservoir
# sensitive load S_res_0 instead of a drug or host parameter, asking whether
# resistant escape is an artifact of the initial reservoir burden assumed in
# the main text (model_Bacteremia.py's own default, S_res_0 = 100 CFU/mL).
#
# Like Emax_l and k_immune -- and unlike EC50_L -- LOW values are the escape
# side here: a smaller initial sensitive reservoir means less competition for
# the resistant strain inside the reservoir, so R_res is more likely to
# persist. The bisection therefore looks for a crossing with f(lo) > 0 and
# f(hi) < 0.
#
# The parameter baseline below is the project's current one, not the one the
# original .pyc was compiled against. The recovered bytecode carried
# rho_S = 0.61, rho_R = 0.55 and EC50_L = 1.0; those were superseded when
# EC50_L was consolidated onto a single project-wide 0.6 mg/L and rho_S was
# raised to 0.72 h^-1 (rho_R rescaled to 0.649) to compensate -- see the
# header narrative in MC_ec50_lzd.py. Everything else matches the .pyc.
#
# this file creates one figure:
#   1. sres0_threshold_sweep.png - final R_b/R_res vs. S_res_0 (log y-axis),
#      shading the escape zone below each threshold
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq

from parameters import BASE_PARAMS, LOD, TOTAL_H, VANCO_START, Y0, save_figure

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
#   sres0_threshold_sweep.png -- final R_b/R_res across the 101-point S_res_0 grid.

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
# Simulation settings
# ---------------------------------------------------------------------------

total_h     = TOTAL_H
vanco_start = VANCO_START
t_eval      = np.linspace(0, total_h, 1150)

pk           = model_mod.PharmacokineticModel()
immune_model = model_mod.ImmuneResponse(k_immune=0.12)

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid",  total_h, vanco_start + pk.van_duration)

rho_S = 0.72    # raised from 0.61 -- see header note above
rho_R = 0.649   # ~10% fitness cost relative to rho_S

SRES0_BASE = 100   # model_Bacteremia.py's own default initial reservoir sensitive load

# Model parameters come from parameters.py (single source of truth).

R_RES_0 = 100.0   # initial resistant reservoir load, held fixed while S_res_0 is swept


def final_counts(s_res_0):
    """Run one deterministic simulation at a given S_res_0; return (final R_b, final R_res)."""
    sol = odeint(
        model_mod.dual_reservoir_model,
        [0.0, 0.0, s_res_0, R_RES_0], t_eval,
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

# S_res_0 values that finish ABOVE the limit of detection (i.e. resistant escape)
escape_R_b   = sres0_sweep[final_R_b   > LOD]
escape_R_res = sres0_sweep[final_R_res > LOD]

print(f"\nS_res_0 values in [{SRES0_MIN}, {SRES0_MAX}] with final R_b   > LOD: "
      f"{escape_R_b.min():.0f}-{escape_R_b.max():.0f} CFU/mL ({len(escape_R_b)}/{N_POINTS} points)"
      if len(escape_R_b) else "\nNo swept S_res_0 values left final R_b above the LOD.")
print(f"S_res_0 values in [{SRES0_MIN}, {SRES0_MAX}] with final R_res > LOD: "
      f"{escape_R_res.min():.0f}-{escape_R_res.max():.0f} CFU/mL ({len(escape_R_res)}/{N_POINTS} points)"
      if len(escape_R_res) else "No swept S_res_0 values left final R_res above the LOD.")


# ---------------------------------------------------------------------------
# Precise crossing point via bisection. Escape is the LOW side here, so a
# genuine crossing has f(lo) > 0 and f(hi) < 0.
# ---------------------------------------------------------------------------
def find_threshold(compartment_idx, lo, hi):
    """Bisect for the S_res_0 where the final count crosses the LOD."""
    def f(s_res_0):
        sol = odeint(
            model_mod.dual_reservoir_model,
            [0.0, 0.0, s_res_0, R_RES_0], t_eval,
            args=(BASE_PARAMS, van_func, lzd_func, immune_model),
            rtol=1e-8, atol=1e-10, mxstep=5000,
        )
        return sol[-1, compartment_idx] - LOD

    if f(lo) * f(hi) > 0:
        return None   # no sign change in range -> no clean threshold to bisect
    return brentq(f, lo, hi, xtol=0.01)


threshold_R_b   = find_threshold(1, SRES0_MIN, SRES0_MAX)
threshold_R_res = find_threshold(3, SRES0_MIN, SRES0_MAX)


def _describe(name, threshold, escapes):
    """Report a compartment's clearance threshold, disambiguating a None
    result as 'always escapes' vs 'always clears' using the swept data."""
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
    print(f"\nmodel_Bacteremia.py's own default S_res_0 = {SRES0_BASE}"
          f" -> R_res escapes (stays above LOD) regardless of S_res_0 in "
          f"[{SRES0_MIN}, {SRES0_MAX}] (rho_res_R = 0.1765 is high enough on its own).")
else:
    print(f"\nmodel_Bacteremia.py's own default S_res_0 = {SRES0_BASE}"
          f" -> R_res clears regardless of S_res_0 in [{SRES0_MIN}, {SRES0_MAX}].")

# ---------------------------------------------------------------------------
# FIGURE: final R_b / R_res vs S_res_0, with escape zone(s) highlighted
# ---------------------------------------------------------------------------
FLOOR = LOD * 0.5   # display floor for values that fell below the LOD

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
ax.axvline(SRES0_BASE, color="gray", ls="-.", lw=1.5, alpha=0.7,
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
ax.set_xlabel("Initial reservoir sensitive load $S_{res,0}$ (CFU/mL)")
ax.set_ylabel("Final resistant bacterial count (CFU/mL)")
ax.set_title("Resistant Reservoir Escape vs Initial $S_{res,0}$ — end-of-simulation counts",
             fontsize=19, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(loc="upper right", fontsize=15, framealpha=0.85)

fig.tight_layout()
out_file = os.path.join(FIGURES_DIR, "sres0_threshold_sweep.png")
save_figure(fig, out_file, __file__)
print(f"\nSaved: {out_file}")
