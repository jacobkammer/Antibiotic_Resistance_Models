# =============================================================================
# Baseline reservoir clearance check
#
# Runs a single deterministic simulation with BASE_PARAMS and Y0 from
# parameters.py, then plots blood vs. reservoir kinetics for both strains to
# show whether each compartment is below the LOD at the simulation endpoint.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from parameters import BASE_PARAMS, LOD, TOTAL_H, VANCO_START, Y0, save_figure
from solver import integrate_model


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

# all figures are written into figures/ alongside the scripts
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# FIGURES PRODUCED BY THIS SCRIPT -- DETERMINISTIC (no sampling anywhere)
#   baseline_reservoir_clearance.png

MODULE_NAME = "model_Bacteremia.py"
if not os.path.exists(MODULE_NAME):
    print(f"ERROR: Cannot find '{MODULE_NAME}' in the current directory.", file=sys.stderr)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("model_mod", MODULE_NAME)
model_mod = importlib.util.module_from_spec(spec)
sys.modules["model_mod"] = model_mod
spec.loader.exec_module(model_mod)


total_h     = TOTAL_H
vanco_start = VANCO_START
t_eval       = np.linspace(0, total_h, 1450)
t_days       = t_eval / 24.0

pk           = model_mod.PharmacokineticModel()
immune_model = model_mod.ImmuneResponse()

lzd_start = vanco_start + pk.van_duration
lzd_end   = lzd_start + pk.lzd_duration

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid",  total_h, lzd_start)

vanco_start_days = vanco_start / 24.0
lzd_start_days   = lzd_start / 24.0
lzd_end_days     = lzd_end / 24.0
# Model parameters come from parameters.py (single source of truth).


#------------------------------------------------------------------------
# Simulation: integrate the dual-reservoir ODE model
#   - model_mod.dual_reservoir_model: the RHS function for the ODEs
#   - Y0: initial conditions [S_b, R_b, S_res, R_res]
#   - t_eval: time points at which the solution is returned
#   - args: (BASE_PARAMS, van_func, lzd_func, immune_model) — fixed
#           parameters plus vancomycin/linezolid concentrations and immunity
#   - integration settings and positive-state validation are centralized in
#     solver.py / parameters.SOLVER_CONFIG
#------------------------------------------------------------------------

sol = integrate_model(
    model_mod.dual_reservoir_model, Y0, t_eval,
    (BASE_PARAMS, van_func, lzd_func, immune_model),
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

if reservoir_cleared:
    verdict = "Reservoir fully cleared of both strains"
elif final["R_res"] >= LOD:
    verdict = "Reservoir resistant strain persists at the endpoint"
else:
    verdict = "Resistant reservoir cleared; sensitive reservoir remains above LOD"
fig.suptitle(
    f"Baseline Model — Is the Reservoir Cleared of Infection?\n{verdict}",
    fontsize=21, fontweight="bold",
)

r_res_annotation = ("above LOD at endpoint" if final["R_res"] >= LOD
                    else "below LOD at endpoint")
r_res_plot_y = max(final["R_res"], LOD * 0.2)
axes[1].annotate(
    f"Final $R_{{res}}$ = {final['R_res']:.2e} CFU/mL\n({r_res_annotation})",
    xy=(t_days[-1] * 0.98, r_res_plot_y),
    xytext=(t_days[-1] * 0.30, max(r_res_plot_y * 3.0, LOD * 2.0)),
    fontsize=16, color=R_COLOR, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=R_COLOR, lw=1.2),
)

out_file = os.path.join(FIGURES_DIR, "baseline_reservoir_clearance.png")
save_figure(fig, out_file, __file__)
print(f"\nSaved: {out_file}")
