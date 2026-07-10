# =============================================================================
# Monte Carlo Simulation: Transfer Rates Explored Separately
# Two independent MC sweeps are run:
#   Sweep 1 — vary f_r_b (reservoir→blood), hold f_b_r at baseline
#   Sweep 2 — vary f_b_r (blood→reservoir), hold f_r_b at baseline
# Each sweep samples its transfer rate from a log-normal distribution while
# every other parameter (including the other transfer rate) stays fixed at
# baseline, isolating the effect of each rate on bacterial kinetics.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODULE_NAME = "model.LinearIM.py"
if not os.path.exists(MODULE_NAME):
    print(f"ERROR: Cannot find '{MODULE_NAME}' in the current directory.", file=sys.stderr)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("model_mod", MODULE_NAME)
model_mod = importlib.util.module_from_spec(spec)
sys.modules["model_mod"] = model_mod
spec.loader.exec_module(model_mod)

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
NUM_ITERATIONS = 400
LOD            = 10.0       # limit of detection (CFU/mL)
SIGMA          = 0.9        # log-normal spread applied to whichever rate is varied

total_h      = 1944  # 21d pre-tx + 4d vancomycin + 42d linezolid + 14d post-tx follow-up
vanco_start  = 504
t_eval       = np.linspace(0, total_h, 1150)
t_days       = t_eval / 24.0

pk           = model_mod.PharmacokineticModel()
immune_model = model_mod.ImmuneResponse()

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid",  total_h, vanco_start + pk.van_duration)

vanco_start_days = vanco_start / 24.0
lzd_start_days   = (vanco_start + pk.van_duration) / 24.0
lzd_end_days     = (vanco_start + pk.van_duration + pk.lzd_duration) / 24.0

# ---------------------------------------------------------------------------
# Fixed parameters (all except the swept transfer rate held at baseline)
# ---------------------------------------------------------------------------
rho_S        = 0.16
rho_R        = 0.128   # directly tuned (20% fitness cost relative to rho_S)

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
}

F_R_B_BASE = 5e-5   # reservoir -> blood baseline
F_B_R_BASE = 1e-5   # blood -> reservoir baseline

# Fixed initial conditions — only the swept rate is varied across iterations
Y0 = [0.0, 0.0, 100.0, 100.0]


# ---------------------------------------------------------------------------
# Run one Monte Carlo sweep: vary a single transfer rate, hold the other fixed
# ---------------------------------------------------------------------------
def run_transfer_rate_sweep(vary_name, vary_baseline, fixed_name, fixed_value, seed):
    rng = np.random.default_rng(seed)
    samples = rng.lognormal(np.log(vary_baseline), SIGMA, NUM_ITERATIONS)

    S_b_hist   = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
    R_b_hist   = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
    S_res_hist = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
    R_res_hist = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)

    peak_S_b   = np.zeros(NUM_ITERATIONS)
    peak_R_b   = np.zeros(NUM_ITERATIONS)
    peak_S_res = np.zeros(NUM_ITERATIONS)
    final_R_b  = np.zeros(NUM_ITERATIONS)

    print(f"Running {NUM_ITERATIONS} iterations (sweep: {vary_name}, "
          f"{fixed_name} fixed at {fixed_value:.1e})...", flush=True)

    for i in range(NUM_ITERATIONS):
        params_i = {**BASE_PARAMS, vary_name: samples[i], fixed_name: fixed_value}

        sol = odeint(
            model_mod.dual_reservoir_model,
            Y0, t_eval,
            args=(params_i, van_func, lzd_func, immune_model),
            rtol=1e-7, atol=1e-9, mxstep=5000,
        )

        sb   = np.where(sol[:, 0] < LOD, 0.0, sol[:, 0])
        rb   = np.where(sol[:, 1] < LOD, 0.0, sol[:, 1])
        sres = np.where(sol[:, 2] < LOD, 0.0, sol[:, 2])
        rres = np.where(sol[:, 3] < LOD, 0.0, sol[:, 3])

        S_b_hist[i]   = sb
        R_b_hist[i]   = rb
        S_res_hist[i] = sres
        R_res_hist[i] = rres

        peak_S_b[i]   = np.max(sb)
        peak_R_b[i]   = np.max(rb)
        peak_S_res[i] = np.max(sres)
        final_R_b[i]  = rb[-1]

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{NUM_ITERATIONS} complete", flush=True)

    return {
        "vary_name": vary_name,
        "samples":   samples,
        "S_b":       S_b_hist,
        "R_b":       R_b_hist,
        "S_res":     S_res_hist,
        "R_res":     R_res_hist,
        "peak_S_b":   peak_S_b,
        "peak_R_b":   peak_R_b,
        "peak_S_res": peak_S_res,
        "final_R_b":  final_R_b,
    }


sweep_f_r_b = run_transfer_rate_sweep("f_r_b", F_R_B_BASE, "f_b_r", F_B_R_BASE, seed=42)
sweep_f_b_r = run_transfer_rate_sweep("f_b_r", F_B_R_BASE, "f_r_b", F_R_B_BASE, seed=43)

print("Simulations complete. Generating figures...", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def annotate_windows(ax):
    ax.axvspan(vanco_start_days, lzd_start_days, color="gray", alpha=0.12, label="Vancomycin")
    ax.axvspan(lzd_start_days,  lzd_end_days,    color="gold", alpha=0.12, label="Linezolid")
    ax.axhline(LOD, color="black", ls=":", lw=0.9, alpha=0.6, label="LOD (10 CFU/mL)")


def style_kinetic(ax, title, ylabel, ylim_top):
    ax.set_yscale("log")
    ax.set_ylim(LOD * 0.5, ylim_top)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize=7, ncol=2)


RATE_LABELS = {
    "f_r_b": r"$f_{r \to b}$ (reservoir $\to$ blood)",
    "f_b_r": r"$f_{b \to r}$ (blood $\to$ reservoir)",
}


# ---------------------------------------------------------------------------
# FIGURE: 4-panel trajectory kinetics for one sweep
# Panels: S_b, R_b, S_res, R_res
# ---------------------------------------------------------------------------
def plot_sweep_kinetics(sweep, filename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Bacterial Kinetics — Monte Carlo sweep of {RATE_LABELS[sweep['vary_name']]}",
                 fontsize=12, fontweight="bold")

    panels = [
        (axes[0, 0], sweep["S_b"],   "royalblue", "blue", "Sensitive Blood ($S_b$)",     1e6),
        (axes[0, 1], sweep["R_b"],   "lightcoral", "red",  "Resistant Blood ($R_b$)",     1e6),
        (axes[1, 0], sweep["S_res"], "royalblue", "blue", "Sensitive Reservoir ($S_{res}$)", 2e7),
        (axes[1, 1], sweep["R_res"], "lightcoral", "red",  "Resistant Reservoir ($R_{res}$)", 2e7),
    ]

    for ax, hist, lc, mc, title, ylim in panels:
        annotate_windows(ax)

        for j in range(min(20, NUM_ITERATIONS)):
            tr = np.where(hist[j] == 0, np.nan, hist[j])
            ax.plot(t_days, tr, color=lc, alpha=0.12, lw=0.8)

        med  = np.nanpercentile(hist, 50, axis=0)
        lo5  = np.nanpercentile(hist,  5, axis=0)
        hi95 = np.nanpercentile(hist, 95, axis=0)

        med  = np.where(med  == 0, np.nan, med)
        lo5  = np.where(lo5  == 0, np.nan, lo5)
        hi95 = np.where(hi95 == 0, np.nan, hi95)

        ax.fill_between(t_days, lo5, hi95, color=lc, alpha=0.18)
        ax.plot(t_days, med, color=mc, lw=2, label=f"Median (n={NUM_ITERATIONS})")

        style_kinetic(ax, title, "CFU/mL", ylim)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig)


plot_sweep_kinetics(sweep_f_r_b, "mc_frb_sweep_kinetics.png")
plot_sweep_kinetics(sweep_f_b_r, "mc_fbr_sweep_kinetics.png")


# ---------------------------------------------------------------------------
# FIGURE: outcome scatter plots vs the swept rate (log scale)
# 4 panels: peak S_b, peak R_b, peak S_res, final R_b
# ---------------------------------------------------------------------------
def plot_sweep_outcomes(sweep, filename):
    log10_rate = np.log10(sweep["samples"])

    outcomes = [
        (sweep["peak_S_b"],   "Peak $S_b$ (CFU/mL)"),
        (sweep["peak_R_b"],   "Peak $R_b$ (CFU/mL)"),
        (sweep["peak_S_res"], "Peak $S_{res}$ (CFU/mL)"),
        (sweep["final_R_b"],  "Final $R_b$ (CFU/mL)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Outcomes vs {RATE_LABELS[sweep['vary_name']]}",
                 fontsize=12, fontweight="bold")

    for ax, (y_vals, ylabel) in zip(axes.flat, outcomes):
        nonzero = y_vals > LOD
        ax.scatter(log10_rate[nonzero], y_vals[nonzero],
                   s=18, alpha=0.5, color="steelblue", edgecolors="none")
        ax.set_yscale("log")
        ax.set_xlabel(f"$\\log_{{10}}$({RATE_LABELS[sweep['vary_name']]})")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", ls=":", alpha=0.35)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig)


plot_sweep_outcomes(sweep_f_r_b, "mc_frb_sweep_outcomes.png")
plot_sweep_outcomes(sweep_f_b_r, "mc_fbr_sweep_outcomes.png")


# ---------------------------------------------------------------------------
# FIGURE: sampled rate distributions for both sweeps
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Sampled Transfer Rate Distributions (independent sweeps)",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.hist(sweep_f_r_b["samples"], bins=35, color="tomato", edgecolor="white", alpha=0.75)
ax.axvline(F_R_B_BASE, color="darkred", ls="--", lw=1.5, label=f"Baseline ({F_R_B_BASE:.0e})")
ax.set_xscale("log")
ax.set_xlabel(RATE_LABELS["f_r_b"])
ax.set_ylabel("Frequency")
ax.set_title(f"Sweep 1: {RATE_LABELS['f_r_b']} varied\n(f_b_r fixed at {F_B_R_BASE:.0e})")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.35)

ax = axes[1]
ax.hist(sweep_f_b_r["samples"], bins=35, color="steelblue", edgecolor="white", alpha=0.75)
ax.axvline(F_B_R_BASE, color="navy", ls="--", lw=1.5, label=f"Baseline ({F_B_R_BASE:.0e})")
ax.set_xscale("log")
ax.set_xlabel(RATE_LABELS["f_b_r"])
ax.set_ylabel("Frequency")
ax.set_title(f"Sweep 2: {RATE_LABELS['f_b_r']} varied\n(f_r_b fixed at {F_R_B_BASE:.0e})")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.35)

plt.tight_layout()
plt.savefig("mc_transfer_rate_sweep_distributions.png", dpi=300, bbox_inches="tight")
print("Saved: mc_transfer_rate_sweep_distributions.png")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(sweep, fixed_name, fixed_value):
    pk_Sb = sweep["peak_S_b"]
    pk_Rb = sweep["peak_R_b"]
    print(f"\n--- Sweep: {sweep['vary_name']} varied ({fixed_name} fixed at {fixed_value:.1e}) ---")
    print(f"  Peak S_b  median={np.median(pk_Sb[pk_Sb > LOD]):.1e}  "
          f"[5th={np.percentile(pk_Sb[pk_Sb > LOD], 5):.1e}, "
          f"95th={np.percentile(pk_Sb[pk_Sb > LOD], 95):.1e}] CFU/mL")
    if np.any(pk_Rb > LOD):
        print(f"  Peak R_b  median={np.median(pk_Rb[pk_Rb > LOD]):.1e}  "
              f"[5th={np.percentile(pk_Rb[pk_Rb > LOD], 5):.1e}, "
              f"95th={np.percentile(pk_Rb[pk_Rb > LOD], 95):.1e}] CFU/mL")
    else:
        print("  Peak R_b: no samples above LOD")


print_summary(sweep_f_r_b, "f_b_r", F_B_R_BASE)
print_summary(sweep_f_b_r, "f_r_b", F_R_B_BASE)

print("\nDone.")
