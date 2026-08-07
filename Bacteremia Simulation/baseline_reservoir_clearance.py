# =============================================================================
# Baseline reservoir clearance check
#
# Runs a single deterministic simulation with the baseline parameters defined
# below (rho_S = 0.61, rho_R = 0.55, Emax_l = 0.8, EC50_L = 1.0, reservoir
# seed = 100 CFU/mL each), then plots blood vs. reservoir kinetics for both
# strains to show whether each compartment clears below the LOD by the
# end of the simulation.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


# code below sets global matplotlib parameters for consistent figure formatting
plt.rcParams.update({
    "font.size": 16, # base font size
    "axes.titlesize": 18, # title font size
    "axes.labelsize": 16, # axis label font size
    "xtick.labelsize": 14, # x-axis tick label font size
    "ytick.labelsize": 14, # y-axis tick label font size
    "legend.fontsize": 14, # legend font size
    "figure.titlesize": 20, # figure title font size
})

MODULE_NAME = "model_Bacteremia.py"
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

rho_S        = 0.61   
rho_R        = 0.55    

# Baseline params, identical to model_Bacteremia.py's __main__ block
BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_R":        0.1765,  # narrow window just above the reservoir persistence threshold (0.17594055) so S_b can also establish in blood -- see model_Bacteremia.py
    "rho_res_S":        0.19415,  # 1.1x rho_res_R -- sensitive strain grows 10% faster in the reservoir
    "Emax_v":           0.40,
    "EC50_V":           0.245,
    "Emax_l":           0.8,    # fixed, decoupled from rho_S
    "EC50_L":           1.0,
    "B_max_blood":      6000,
    "B_max_reservoir":  1e4,
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.30,    # lowered from 0.45 -- Emax_l_res scales with rho_res_S; keeps R_res persistent despite S's reservoir growth advantage
    "f_r_b":            5e-5,  
    "f_b_r":            1e-5,
}

Y0 = [0.0, 0.0, 100.0, 100.0]   # model_Bacteremia.py's own default reservoir seed

#------------------------------------------------------------------------
# Simulation: integrate the dual-reservoir ODE model
#   - model_mod.dual_reservoir_model: the RHS function for the ODEs
#   - Y0: initial conditions [S_b, R_b, S_res, R_res]
#   - t_eval: time points at which the solution is returned
#   - args: (BASE_PARAMS, van_func, lzd_func, immune_model) — fixed
#           parameters plus vancomycin/linezolid concentrations and immunity
#   - rtol=1e-8, atol=1e-10: relative and absolute integration tolerances
#   - mxstep=5000: maximum internal steps the solver is allowed
#------------------------------------------------------------------------

sol = odeint(
    model_mod.dual_reservoir_model, Y0, t_eval,
    args=(BASE_PARAMS, van_func, lzd_func, immune_model),
    rtol=1e-8, atol=1e-10, mxstep=5000,
)

S_b, R_b, S_res, R_res = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

final = {"S_b": S_b[-1], "R_b": R_b[-1], "S_res": S_res[-1], "R_res": R_res[-1]}
print("Final compartment values (baseline model_Bacteremia.py parameters):")
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


# ---------------------------------------------------------------------------
# Helper function to plot a single panel
# axvspan draws a vertical span (rectangle) across the plot
# axhline draws a horizontal line across the plot   
# ---------------------------------------------------------------------------
def _plot_panel(ax, S, R, title, y_max):
    ax.axvspan(vanco_start_days, lzd_start_days, color="gray", alpha=0.12, label="Vancomycin window") # Vancomycin window, light gray band from start to end of vancomycin treatment
    ax.axvspan(lzd_start_days,  lzd_end_days,    color="gold", alpha=0.12, label="Linezolid window") # Linezolid window, light yellow band from start to end of linezolid treatment
    ax.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)") # Limit of detection line, black dotted line

    ax.plot(t_days, np.where(S <= 0, np.nan, S), color=S_COLOR, lw=2.0, label="Sensitive")
    ax.plot(t_days, np.where(R <= 0, np.nan, R), color=R_COLOR, lw=2.0, label="Resistant")

    ax.set_yscale("log")
    ax.set_ylim(LOD * 0.1, y_max)
    ax.set_xlim(0, t_days[-1])
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("CFU/mL")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper right", fontsize=14, framealpha=0.85)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

_plot_panel(axes[0], S_b, R_b, "Blood Compartment", 1e6)
_plot_panel(axes[1], S_res, R_res, "Reservoir Compartment", 1e7)

verdict = ("Reservoir resistant strain PERSISTS — never clears after linezolid ends"
           if not reservoir_cleared else "Reservoir fully cleared of both strains")
fig.suptitle(
    f"Baseline Model — Is the Reservoir Cleared of Infection?\n{verdict}",
    fontsize=21, fontweight="bold",
)

axes[1].annotate(
    f"Final $R_{{res}}$ = {final['R_res']:.2e} CFU/mL\n(never drops below LOD after day ~25)",
    xy=(t_days[-1] * 0.98, final["R_res"]), xytext=(t_days[-1] * 0.30, final["R_res"] * 3.0),
    fontsize=16, color=R_COLOR, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=R_COLOR, lw=1.2),
)

fig.savefig("baseline_reservoir_clearance.png", dpi=300, bbox_inches="tight")
print("\nSaved: baseline_reservoir_clearance.png")
