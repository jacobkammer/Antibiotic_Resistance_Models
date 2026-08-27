# =============================================================================
# Deterministic k_immune threshold sweep (0.02 - 0.155 h^-1)
#
# Mirrors EC50_lzd_threshold_sweep.py / Emax_lzd_threshold_sweep.py, but for
# the host immune clearance rate (k_immune) instead of a drug parameter.
# Baseline k_immune = 0.12 h^-1. model_Bacteremia.py's own __main__
# asserts net_growth_blood = rho_S - eff_blood*k_immune > 0 (the host must
# not clear the infection on its own before any antibiotic is given), i.e.
# k_immune < rho_S/eff_blood. With rho_S = 0.6 and eff_blood = 1.0 (a
# strain-specific eff_blood_S/eff_blood_R split was tried and reverted -- see
# model_Bacteremia.py's ImmuneResponse class), that ceiling is 0.6
# h^-1, well above the existing [0.02, 0.155] range.
#
# Like Emax_l, LOWER k_immune is worse here: a weaker immune system lets
# resistant bacteria escape in blood and/or reservoir. Emax_l was later
# UNCOUPLED from rho_S and fixed at 0.8 (previously it auto-followed rho_S to
# 0.6) -- since 0.8 then exceeded every species' growth rate, linezolid alone
# was strong enough to fully clear both compartments regardless of host
# immunity across the entire swept range. rho_res_R was then raised
# 0.145 -> 0.20 (precise clearance threshold 0.17594055) -- now a fitness
# ADVANTAGE over rho_res_S (0.175) -- which reversed this again: R_b and
# R_res escaped across the ENTIRE swept range, with R_b still visibly
# clearing during the linezolid course before relapsing. BUT at
# rho_res_R = 0.20, R_res's reservoir growth advantage over S_res (0.175)
# let R_b win the pretreatment race for blood's shared carrying capacity
# outright, crowding S_b out entirely.
#
# rho_res_R was finally narrowed to 0.1765 -- just above the reservoir's own
# 0.17594055 persistence threshold, but below the point where R_b starts
# winning the blood race -- so S_b CAN also establish a visible infection.
# This is a much thinner margin: this sweep now finds a real crossing point
# at k_immune = 0.1255 h^-1 (previously there was none in range). Baseline
# k_immune = 0.12 sits just BELOW it, so R_b/R_res still escape at baseline,
# but only barely -- a modestly weaker immune system (k_immune > 0.1255)
# would suppress them instead.
#
# rho_S was then raised 0.60 -> 0.61 (see MC_ec50_lzd.py's rho_S sensitivity
# analysis) to match every other threshold-sweep/MC script in this project.
# This pushed the crossing point from 0.1255 up to 0.1521 h^-1. Baseline
# k_immune = 0.12 is still comfortably below it, so R_b/R_res still escape
# at baseline -- now with more margin than before, not less.
#
# This script runs a fine, deterministic (non-Monte-Carlo) sweep of k_immune,
# records the FINAL R_b / R_res at the end of each run, selects the k_immune
# values that finish above the LOD, and locates the exact crossing point for
# each compartment via root-finding.
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
# Net effect here: the R_res threshold moved ~0.2000 -> 0.203 h^-1 (R_b to
# 0.197). Without the rho_S compensation this sweep would have had no
# threshold at all under the consolidated EC50_L.
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
# Simulation settings (matches MC_immune_response.py / model.py's own defaults)
# ---------------------------------------------------------------------------

total_h     = TOTAL_H
vanco_start = VANCO_START
t_eval      = np.linspace(0, total_h, 1150)
t_days      = t_eval / 24.0

pk = model_mod.PharmacokineticModel()

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid",  total_h, vanco_start + pk.van_duration)

# rho_S raised 0.61 -> 0.72 (rho_R 0.55 -> 0.649, ~10% fitness cost preserved) to offset
# EC50_L 1.0 -> 0.6 while keeping lzd_res_fraction at 0.30: reservoir kill scales as
# Emax_l_res = Emax_l * (rho_res_S / rho_S), so a higher rho_S weakens it.
rho_S = 0.72    # raised from 0.60 to match MC_immune_response.py / the EC50_L and Emax_l
                # threshold sweeps -- this shifts the k_immune escape threshold up from
                # 0.1255 to 0.1521 h^-1 (bisected and confirmed directly against this
                # script's own BASE_PARAMS)
rho_R = 0.649    # directly tuned (~8.3% fitness cost relative to rho_S, down from 20%)

K_IMMUNE_BASE = 0.12   # model_Bacteremia.py's own default

# Model parameters come from parameters.py (single source of truth).



def final_counts(k_immune):
    """Run one deterministic simulation at a given k_immune; return (final R_b, final R_res)."""
    immune_i = model_mod.ImmuneResponse(k_immune=k_immune)
    sol = odeint(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(BASE_PARAMS, van_func, lzd_func, immune_i),
        rtol=1e-8, atol=1e-10, mxstep=5000,
    )
    return sol[-1, 1], sol[-1, 3]


# ---------------------------------------------------------------------------
# Fine sweep across k_immune in [0.02, 0.155]
# ---------------------------------------------------------------------------
KIM_MIN, KIM_MAX = 0.02, 0.22   # widened from 0.155 -- at rho_S=0.61 the threshold (0.1521)
                                 # sits too close to the old upper bound to show the
                                 # "clears" side of the switch
N_POINTS = 101

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
# Precise crossing point (root of final_count(k_immune) - LOD) via bisection
# ---------------------------------------------------------------------------
def find_threshold(compartment_idx):
    """Bisect for the k_immune where the final count first crosses the LOD."""
    def f(k_immune):
        immune_i = model_mod.ImmuneResponse(k_immune=k_immune)
        sol = odeint(
            model_mod.dual_reservoir_model,
            Y0, t_eval,
            args=(BASE_PARAMS, van_func, lzd_func, immune_i),
            rtol=1e-8, atol=1e-10, mxstep=5000,
        )
        return sol[-1, compartment_idx] - LOD

    lo_val, hi_val = f(KIM_MIN), f(KIM_MAX)
    if lo_val < 0 or hi_val > 0:
        return None   # no sign change in range -> no clean threshold to bisect
    return brentq(f, KIM_MIN, KIM_MAX, xtol=1e-3)


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
ax.set_xlim(KIM_MIN, KIM_MAX)
ax.set_ylim(FLOOR * 0.8, max(FLOOR * 20, final_R_b.max(), final_R_res.max()) * 2)
ax.set_xlabel(r"$k_{immune}$ (h$^{-1}$)")
ax.set_ylabel("Final resistant bacterial count (CFU/mL)")
ax.set_title(r"Resistant Escape Threshold vs $k_{immune}$ (host immune clearance) — end-of-simulation counts",
             fontsize=19, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(loc="upper right", fontsize=15, framealpha=0.85)

fig.tight_layout()
out_file = os.path.join(FIGURES_DIR, "immune_response_threshold_sweep.png")
fig.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out_file}")
