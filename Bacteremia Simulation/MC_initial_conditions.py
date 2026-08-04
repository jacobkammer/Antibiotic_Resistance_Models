import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import importlib.util
import os

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

# Load model
MODULE_NAME = "model_Bacteremia.py"
spec = importlib.util.spec_from_file_location("model_mod", MODULE_NAME)
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)

num_iterations = 1000
total_h = 1944  # 21d pre-tx + 4d vancomycin + 42d linezolid + 14d post-tx follow-up
vanco_start = 504
t_eval = np.linspace(0, total_h, 1450)
t_days = t_eval / 24.0

pk = model_mod.PharmacokineticModel()
immune_model = model_mod.ImmuneResponse()

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid", total_h, vanco_start + pk.van_duration)

rho_S = 0.61    # raised from 0.60 -- see rho_S sensitivity analysis. Unlike the EC50_L/
                # Emax_l/k_immune analyses, this script's outcome (governed by the
                # reservoir's own S_res_0/R_res_0 persistence threshold) showed no
                # detectable change at rho_S=0.61 in a single-point check, since rho_S
                # does not appear in the dS_res/dR_res equations at all
rho_R = 0.55    # directly tuned (~8.3% fitness cost relative to rho_S, down from 20%)

params = {
    "rho_S":           rho_S,
    "rho_R":           rho_R,
    "rho_res_S":       0.175,  # scaled 5x (from 0.035) alongside rho_S
    "rho_res_R":       0.1765,  # narrow window just above the reservoir persistence threshold (0.17594055) so S_b can also establish in blood -- see model_Bacteremia.py
    "Emax_v":          0.40,
    "EC50_V":          0.245,
    "Emax_l":          0.8,  # fixed, decoupled from rho_S (was tied for "perfect bacteriostasis")
    "EC50_L":          1.0,
    "B_max_blood":     6000,
    "B_max_reservoir": 1e4,    # lowered from 4.5e6 so escaped R_res plateaus well above the LOD but far below its old level
    "van_res_fraction":0.15,
    "lzd_res_fraction":0.45,
    "f_r_b":           5e-5,  # restored to original
    "f_b_r":           1e-5,
}

S_b_history   = np.zeros((num_iterations, len(t_eval)))
R_b_history   = np.zeros((num_iterations, len(t_eval)))
S_res_history = np.zeros((num_iterations, len(t_eval)))
R_res_history = np.zeros((num_iterations, len(t_eval)))

np.random.seed(42)
# CV = sqrt(exp(σ²) - 1)  →  σ = sqrt(ln(1 + CV²))
# μ  = ln(mean) - σ²/2   ensures E[S_res_0] = S_RES_0_BASE, E[R_res_0] = R_RES_0_BASE
# (same mean-corrected, CV-based convention as MC_ec50_lzd.py / MC_emax_lzd.py / MC_immune_response.py)
CV = 60.0    # deliberately extreme: with both means fixed at 100 CFU/mL (matching the
             # other MC scripts), reaching ~70% resolution against the ~7.57 CFU/mL
             # reservoir persistence threshold requires this much right-skew -- illustrates
             # just how uncertain/heterogeneous the true founding reservoir population is

S_RES_0_BASE = 100    # matches the fixed reservoir initial condition used in the other
                       # MC scripts (MC_ec50_lzd.py, MC_emax_lzd.py, MC_immune_response.py)
R_RES_0_BASE = 100    # reservoir persistence threshold (R_res_0 ~= 7.57 CFU/mL, found by
                       # bisection at rho_res_R=0.1765) determines clinical resolution --
                       # S_res_0 has no effect on this threshold (S_res and R_res don't
                       # compete for the shared reservoir carrying capacity the way
                       # S_b/R_b do in blood)

sigma_ln = np.sqrt(np.log(1.0 + CV**2))
s_res_mu = np.log(S_RES_0_BASE) - sigma_ln**2 / 2.0
r_res_mu = np.log(R_RES_0_BASE) - sigma_ln**2 / 2.0

S_res_0_samples = np.random.lognormal(s_res_mu, sigma_ln, num_iterations)
R_res_0_samples = np.random.lognormal(r_res_mu, sigma_ln, num_iterations)

for i in range(num_iterations):
    y0 = [0.0, 0.0, S_res_0_samples[i], R_res_0_samples[i]]
    solution = odeint(model_mod.dual_reservoir_model, y0, t_eval,
                      args=(params, van_func, lzd_func, immune_model))

    S_b_history[i, :]   = np.where(solution[:, 0] < 10.0, 0.0, solution[:, 0])
    R_b_history[i, :]   = np.where(solution[:, 1] < 10.0, 0.0, solution[:, 1])
    S_res_history[i, :] = np.where(solution[:, 2] < 10.0, 0.0, solution[:, 2])
    R_res_history[i, :] = np.where(solution[:, 3] < 10.0, 0.0, solution[:, 3])

vanco_start_days = vanco_start / 24.0
lzd_start_days   = (vanco_start + pk.van_duration) / 24.0
lzd_end_days     = (vanco_start + pk.van_duration + pk.lzd_duration) / 24.0

# -----------------------------------------------------------------------------
# FIGURE 1: 4-panel kinetics
# -----------------------------------------------------------------------------
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))


def format_kinetic_panel(ax, title, y_max, label, legend_loc="lower right"):
    ax.axvspan(vanco_start_days, lzd_start_days, color="gray",   alpha=0.15, label="Vancomycin Window")
    ax.axvspan(lzd_start_days,  lzd_end_days,   color="yellow", alpha=0.15, label="Linezolid Window")
    ax.axhline(y=10.0, color="black", linestyle=":", alpha=0.7,  label="Limit of Detection (10 CFU/mL)")
    ax.set_yscale("log")
    ax.set_ylim(10, y_max)
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel(f"{label} (CFU/mL)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc=legend_loc, fontsize="small")


# Panel A: Sensitive Blood
ax1 = axes[0, 0]
med = np.percentile(S_b_history, 50, axis=0)
low = np.percentile(S_b_history,  5, axis=0)
high = np.percentile(S_b_history, 95, axis=0)
for i in range(min(15, num_iterations)):
    ax1.plot(t_days, np.where(S_b_history[i, :] == 0, np.nan, S_b_history[i, :]),
             color="royalblue", alpha=0.15, lw=1)
ax1.fill_between(t_days, np.where(low == 0, np.nan, low), np.where(high == 0, np.nan, high),
                 color="blue", alpha=0.15)
ax1.plot(t_days, np.where(med == 0, np.nan, med), "b-", lw=2, label="Sensitive Blood Median")
format_kinetic_panel(ax1, "Panel A: Sensitive Strain in Blood ($S_b$)", 1e6, "Blood Load")

# Panel B: Resistant Blood
ax2 = axes[0, 1]
med = np.percentile(R_b_history, 50, axis=0)
low = np.percentile(R_b_history,  5, axis=0)
high = np.percentile(R_b_history, 95, axis=0)
for i in range(min(15, num_iterations)):
    ax2.plot(t_days, np.where(R_b_history[i, :] == 0, np.nan, R_b_history[i, :]),
             color="lightcoral", alpha=0.15, lw=1)
ax2.fill_between(t_days, np.where(low == 0, np.nan, low), np.where(high == 0, np.nan, high),
                 color="red", alpha=0.15)
ax2.plot(t_days, np.where(med == 0, np.nan, med), "r-", lw=2, label="Resistant Blood Median")
format_kinetic_panel(ax2, "Panel B: Resistant Strain in Blood ($R_b$)", 1e6, "Blood Load")

# Panel C: Sensitive Reservoir
ax3 = axes[1, 0]
med = np.percentile(S_res_history, 50, axis=0)
low = np.percentile(S_res_history,  5, axis=0)
high = np.percentile(S_res_history, 95, axis=0)
for i in range(min(15, num_iterations)):
    ax3.plot(t_days, np.where(S_res_history[i, :] == 0, np.nan, S_res_history[i, :]),
             color="royalblue", alpha=0.15, lw=1)
ax3.fill_between(t_days, np.where(low == 0, np.nan, low), np.where(high == 0, np.nan, high),
                 color="blue", alpha=0.15)
ax3.plot(t_days, np.where(med == 0, np.nan, med), "b-", lw=2, label="Sensitive Reservoir Median")
format_kinetic_panel(ax3, "Panel C: Sensitive Strain in Reservoir ($S_{res}$)", 2e7, "Reservoir Burden")

# Panel D: Resistant Reservoir
ax4 = axes[1, 1]
med = np.percentile(R_res_history, 50, axis=0)
low = np.percentile(R_res_history,  5, axis=0)
high = np.percentile(R_res_history, 95, axis=0)
for i in range(min(15, num_iterations)):
    ax4.plot(t_days, np.where(R_res_history[i, :] == 0, np.nan, R_res_history[i, :]),
             color="lightcoral", alpha=0.15, lw=1)
ax4.fill_between(t_days, np.where(low == 0, np.nan, low), np.where(high == 0, np.nan, high),
                 color="red", alpha=0.15)
ax4.plot(t_days, np.where(med == 0, np.nan, med), "r-", lw=2, label="Resistant Reservoir Median")
format_kinetic_panel(ax4, "Panel D: Resistant Strain in Reservoir ($R_{res}$)", 2e7, "Reservoir Burden")

plt.tight_layout()
plt.savefig("mc_initial_conditions_kinetics.png", dpi=300, bbox_inches="tight")
print("Saved: mc_initial_conditions_kinetics.png")

# -----------------------------------------------------------------------------
# FIGURE 2: Sampling distributions
# -----------------------------------------------------------------------------
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(S_res_0_samples, bins=25, color="royalblue", edgecolor="black", alpha=0.7)
plt.axvline(np.median(S_res_0_samples), color="blue", linestyle="--", lw=2,
            label=f"Median: {np.median(S_res_0_samples):.1f}")
plt.xlabel("Initial $S_{res}$ (CFU/mL)")
plt.ylabel("Frequency")
plt.title("Initial Sensitive Strain Sampling Distribution")
plt.legend(loc="upper right")
plt.grid(True, ls=":", alpha=0.4)

plt.subplot(1, 2, 2)
plt.hist(R_res_0_samples, bins=25, color="lightcoral", edgecolor="black", alpha=0.7)
plt.axvline(np.median(R_res_0_samples), color="red", linestyle="--", lw=2,
            label=f"Median: {np.median(R_res_0_samples):.1f}")
plt.xlabel("Initial $R_{res}$ (CFU/mL)")
plt.ylabel("Frequency")
plt.title("Initial Resistant Strain Sampling Distribution")
plt.legend(loc="upper right")
plt.grid(True, ls=":", alpha=0.4)

plt.tight_layout()
plt.savefig("mc_initial_conditions_distributions.png", dpi=300, bbox_inches="tight")
print("Saved: mc_initial_conditions_distributions.png")
