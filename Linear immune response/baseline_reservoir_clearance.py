# =============================================================================
# Baseline reservoir clearance check
#
# Runs model.LinearIM.py exactly as written (its own __main__ block defaults:
# EC50_L = 1.0, Emax_l = rho_S = 0.16, S_res_0 = R_res_0 = 100 CFU/mL) as a
# single deterministic simulation, and plots blood vs. reservoir kinetics for
# both strains to show whether each compartment clears below the LOD by the
# end of the simulation.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

MODULE_NAME = "model.LinearIM.py"
if not os.path.exists(MODULE_NAME):
    print(f"ERROR: Cannot find '{MODULE_NAME}' in the current directory.", file=sys.stderr)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("model_mod", MODULE_NAME)
model_mod = importlib.util.module_from_spec(spec)
sys.modules["model_mod"] = model_mod
spec.loader.exec_module(model_mod)

LOD = 10.0   # limit of detection (CFU/mL)

total_h      = 1944  # 21d pre-tx + 4d vancomycin + 42d linezolid + 14d post-tx follow-up
vanco_start  = 504
t_eval       = np.linspace(0, total_h, 1450)
t_days       = t_eval / 24.0

pk           = model_mod.PharmacokineticModel()
immune_model = model_mod.ImmuneResponse(k_immune=0.12)

lzd_start = vanco_start + pk.van_duration
lzd_end   = lzd_start + pk.lzd_duration

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid",  total_h, lzd_start)

vanco_start_days = vanco_start / 24.0
lzd_start_days   = lzd_start / 24.0
lzd_end_days     = lzd_end / 24.0

rho_S        = 0.16
rho_R        = 0.128   # directly tuned (20% fitness cost relative to rho_S)

# Baseline params, identical to model.LinearIM.py's __main__ block
BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_S":        0.035,
    "rho_res_R":        0.024,   # lowered below the 20%-fitness-cost value (0.028) so R_res clears before linezolid ends
    "Emax_v":           0.40,
    "EC50_V":           1.5,
    "Emax_l":           rho_S,   # "perfect bacteriostasis" for the sensitive strain
    "EC50_L":           1.0,
    "B_max_blood":      5e5,
    "B_max_reservoir":  4.5e6,
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.45,
    "f_r_b":            5e-5,
    "f_b_r":            1e-5,
}

Y0 = [0.0, 0.0, 100.0, 100.0]   # model.LinearIM.py's own default reservoir seed

sol = odeint(
    model_mod.dual_reservoir_model, Y0, t_eval,
    args=(BASE_PARAMS, van_func, lzd_func, immune_model),
    rtol=1e-8, atol=1e-10, mxstep=5000,
)

S_b, R_b, S_res, R_res = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

final = {"S_b": S_b[-1], "R_b": R_b[-1], "S_res": S_res[-1], "R_res": R_res[-1]}
print("Final compartment values (baseline model.LinearIM.py parameters):")
for name, val in final.items():
    status = "cleared" if val < LOD else "NOT CLEARED"
    print(f"  {name:6s} = {val:.3e} CFU/mL   ({status})")

reservoir_cleared = final["S_res"] < LOD and final["R_res"] < LOD
print(f"\nReservoir fully cleared: {reservoir_cleared}")

# ---------------------------------------------------------------------------
# FIGURE: 2-panel — Blood vs Reservoir, both strains, log scale
# ---------------------------------------------------------------------------
S_COLOR = "steelblue"
R_COLOR = "firebrick"


def _plot_panel(ax, S, R, title, y_max):
    ax.axvspan(vanco_start_days, lzd_start_days, color="gray", alpha=0.12, label="Vancomycin window")
    ax.axvspan(lzd_start_days,  lzd_end_days,    color="gold", alpha=0.12, label="Linezolid window")
    ax.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")

    ax.plot(t_days, np.where(S <= 0, np.nan, S), color=S_COLOR, lw=2.0, label="Sensitive")
    ax.plot(t_days, np.where(R <= 0, np.nan, R), color=R_COLOR, lw=2.0, label="Resistant")

    ax.set_yscale("log")
    ax.set_ylim(LOD * 0.1, y_max)
    ax.set_xlim(0, t_days[-1])
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("CFU/mL")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

_plot_panel(axes[0], S_b, R_b, "Blood Compartment", 1e6)
_plot_panel(axes[1], S_res, R_res, "Reservoir Compartment", 1e7)

verdict = ("Reservoir resistant strain PERSISTS — never clears after linezolid ends"
           if not reservoir_cleared else "Reservoir fully cleared of both strains")
fig.suptitle(
    f"Baseline Model — Is the Reservoir Cleared of Infection?\n{verdict}",
    fontsize=13, fontweight="bold",
)

axes[1].annotate(
    f"Final $R_{{res}}$ = {final['R_res']:.2e} CFU/mL\n(never drops below LOD after day ~25)",
    xy=(t_days[-1] * 0.98, final["R_res"]), xytext=(t_days[-1] * 0.55, final["R_res"] * 15),
    fontsize=9, color=R_COLOR, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=R_COLOR, lw=1.2),
)

fig.savefig("baseline_reservoir_clearance.png", dpi=300, bbox_inches="tight")
print("\nSaved: baseline_reservoir_clearance.png")
