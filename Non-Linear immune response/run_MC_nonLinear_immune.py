# =============================================================================
# Monte Carlo Simulation: Initial Conditions Exploration (Nonlinear Immune)
# Imports from model.NonLinearIM.py
# =============================================================================
import importlib.util
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# 1. Target the corrected model file name
MODULE_NAME = "model.NonLinearIM.py"
if not os.path.exists(MODULE_NAME):
    print(f"ERROR: Cannot find '{MODULE_NAME}' in the current directory.", file=sys.stderr)
    print("Please ensure this script is in the same folder as your model file.", file=sys.stderr)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("model_mod", MODULE_NAME)
model_mod = importlib.util.module_from_spec(spec)
sys.modules["model_mod"] = model_mod
spec.loader.exec_module(model_mod)

print(f"Successfully linked to {MODULE_NAME} Structure.", flush=True)


def run_nonlinear_monte_carlo(num_iterations=200):
    print(f"Starting Nonlinear Monte Carlo Simulation ({num_iterations} iterations)...", flush=True)

    # Timeline Parameters
    total_h = 1344
    vanco_start = 504
    t_eval = np.linspace(0, total_h, 1000)
    t_days = t_eval / 24.0

    pk = model_mod.PharmacokineticModel()
    immune_model = model_mod.ImmuneResponse()  # Instantiates the saturable class

    van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
    lzd_func = pk.concentration_function("linezolid",  total_h, vanco_start + pk.van_duration)

    # Growth Parameters
    fitness_cost = 0.20
    rho_S = 0.16
    rho_R = (1 - fitness_cost) * rho_S

    params = {
        "rho_S":           rho_S,   
        "rho_R":           rho_R,   
        "rho_res_S":       0.035,    
        "rho_res_R":       0.035 * (1 - fitness_cost),  
        "Emax_v":          0.40,   
        "EC50_V":          1.5,    
        "Emax_l":          rho_S,  
        "EC50_L":          1.0,    
        "B_max_blood":     5e5,   
        "B_max_reservoir": 1e7,   
        "van_res_fraction":0.15,  
        "lzd_res_fraction":0.45,  
        "f_r_b":           5e-5,   
        "f_b_r":           1e-5,    
    }

    # Storage matrices for patient histories
    S_b_history   = np.zeros((num_iterations, len(t_eval)))
    R_b_history   = np.zeros((num_iterations, len(t_eval)))
    S_res_history = np.zeros((num_iterations, len(t_eval)))
    R_res_history = np.zeros((num_iterations, len(t_eval)))

    # Log-Normal Parameter Setup
    np.random.seed(42)
    sigma = 0.4
    s_res_mu = np.log(1000)
    r_res_mu = np.log(100)

    S_res_0_samples = np.random.lognormal(s_res_mu, sigma, num_iterations)
    R_res_0_samples = np.random.lognormal(r_res_mu, sigma, num_iterations)

    # Simulation Execution Loop
    for i in range(num_iterations):
        y0 = [0.0, 0.0, S_res_0_samples[i], R_res_0_samples[i]]
        solution = odeint(model_mod.dual_reservoir_model, y0, t_eval, args=(params, van_func, lzd_func, immune_model))
        
        # Apply strict Limit of Detection (LOD) threshold (<10 CFU/mL -> 0)
        S_b_history[i, :]   = np.where(solution[:, 0] < 10.0, 0.0, solution[:, 0])
        R_b_history[i, :]   = np.where(solution[:, 1] < 10.0, 0.0, solution[:, 1])
        S_res_history[i, :] = np.where(solution[:, 2] < 10.0, 0.0, solution[:, 2])
        R_res_history[i, :] = np.where(solution[:, 3] < 10.0, 0.0, solution[:, 3])

    # Plot Window Configurations
    vanco_start_days = vanco_start / 24.0
    lzd_start_days   = (vanco_start + pk.van_duration) / 24.0
    lzd_end_days     = (vanco_start + pk.van_duration + pk.lzd_duration) / 24.0

    # -------------------------------------------------------------------------
    # FIGURE 1: 4-PANEL ISOLATED LOG-KINETICS 
    # -------------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))

    def format_panel(ax, title, y_max, label, legend_loc="lower left"):
        ax.axvspan(vanco_start_days, lzd_start_days, color='gray', alpha=0.15, label="Vancomycin Window")
        ax.axvspan(lzd_start_days, lzd_end_days, color='yellow', alpha=0.15, label="Linezolid Window")
        ax.axhline(y=10.0, color="black", linestyle=":", alpha=0.7, label="Limit of Detection (10 CFU/mL)")
        ax.set_yscale('log')
        ax.set_ylim(10, y_max)
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel(f"{label} (CFU/mL)")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(loc=legend_loc, fontsize="x-small")

    # Panel A: Sensitive Blood (Legend set to clear upper-right space)
    ax1 = axes[0, 0]
    med, low, high = np.percentile(S_b_history, [50, 5, 95], axis=0)
    for i in range(min(15, num_iterations)):
        ax1.plot(t_days, np.where(S_b_history[i, :] == 0, np.nan, S_b_history[i, :]), color="royalblue", alpha=0.12, lw=1)
    ax1.fill_between(t_days, np.where(low == 0, np.nan, low), np.where(high == 0, np.nan, high), color="blue", alpha=0.15)
    ax1.plot(t_days, np.where(med == 0, np.nan, med), "b-", lw=2, label="Sensitive Blood Median")
    format_panel(ax1, "Panel A: Sensitive Strain in Blood ($S_b$)", 1e6, "Blood Load", legend_loc="upper right")

    # Panel B: Resistant Blood (Legend set to clear upper-right space)
    ax2 = axes[0, 1]
    med, low, high = np.percentile(R_b_history, [50, 5, 95], axis=0)
    for i in range(min(15, num_iterations)):
        ax2.plot(t_days, np.where(R_b_history[i, :] == 0, np.nan, R_b_history[i, :]), color="lightcoral", alpha=0.12, lw=1)
    ax2.fill_between(t_days, np.where(low == 0, np.nan, low), np.where(high == 0, np.nan, high), color="red", alpha=0.15)
    ax2.plot(t_days, np.where(med == 0, np.nan, med), "r-", lw=2, label="Resistant Blood Median")
    format_panel(ax2, "Panel B: Resistant Strain in Blood ($R_b$)", 1e6, "Blood Load", legend_loc="upper right")

    # Panel C: Sensitive Reservoir
    ax3 = axes[1, 0]
    med, low, high = np.percentile(S_res_history, [50, 5, 95], axis=0)
    for i in range(min(15, num_iterations)):
        ax3.plot(t_days, np.where(S_res_history[i, :] == 0, np.nan, S_res_history[i, :]), color="royalblue", alpha=0.12, lw=1)
    ax3.fill_between(t_days, np.where(low == 0, np.nan, low), np.where(high == 0, np.nan, high), color="blue", alpha=0.15)
    ax3.plot(t_days, np.where(med == 0, np.nan, med), "b-", lw=2, label="Sensitive Reservoir Median")
    format_panel(ax3, "Panel C: Sensitive Strain in Reservoir ($S_{res}$)", 2e7, "Reservoir Burden")

    # Panel D: Resistant Reservoir
    ax4 = axes[1, 1]
    med, low, high = np.percentile(R_res_history, [50, 5, 95], axis=0)
    for i in range(min(15, num_iterations)):
        ax4.plot(t_days, np.where(R_res_history[i, :] == 0, np.nan, R_res_history[i, :]), color="lightcoral", alpha=0.12, lw=1)
    ax4.fill_between(t_days, np.where(low == 0, np.nan, low), np.where(high == 0, np.nan, high), color="red", alpha=0.15)
    ax4.plot(t_days, np.where(med == 0, np.nan, med), "r-", lw=2, label="Resistant Reservoir Median")
    format_panel(ax4, "Panel D: Resistant Strain in Reservoir ($R_{res}$)", 2e7, "Reservoir Burden")

    plt.tight_layout()
    plt.savefig("mc_saturable_kinetics.png", dpi=300, bbox_inches="tight")
    print("Saved kinetics panel as 'mc_saturable_kinetics.png'.")

    # -------------------------------------------------------------------------
    # FIGURE 2: INITIAL LOG-NORMAL SAMPLING HISTOGRAMS
    # -------------------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    # Sensitive Parameter Initial Space
    plt.subplot(1, 2, 1)
    plt.hist(S_res_0_samples, bins=25, color="royalblue", edgecolor="black", alpha=0.7)
    plt.axvline(np.median(S_res_0_samples), color="blue", linestyle="--", lw=2, label=f"Median: {np.median(S_res_0_samples):.1f}")
    plt.xlabel("Initial $S_{res}$ (CFU/mL)")
    plt.ylabel("Frequency")
    plt.title("Initial Sensitive Strain Sampling Distribution")
    plt.legend(loc="upper right")
    plt.grid(True, ls=":", alpha=0.4)

    # Resistant Parameter Initial Space
    plt.subplot(1, 2, 2)
    plt.hist(R_res_0_samples, bins=25, color="lightcoral", edgecolor="black", alpha=0.7)
    plt.axvline(np.median(R_res_0_samples), color="red", linestyle="--", lw=2, label=f"Median: {np.median(R_res_0_samples):.1f}")
    plt.xlabel("Initial $R_{res}$ (CFU/mL)")
    plt.ylabel("Frequency")
    plt.title("Initial Resistant Strain Sampling Distribution")
    plt.legend(loc="upper right")
    plt.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig("mc_saturable_distributions.png", dpi=300, bbox_inches="tight")
    print("Saved distribution histograms as 'mc_saturable_distributions.png'.")


if __name__ == "__main__":
    run_nonlinear_monte_carlo(200)
    print("Monte Carlo workflow loop completed successfully.")