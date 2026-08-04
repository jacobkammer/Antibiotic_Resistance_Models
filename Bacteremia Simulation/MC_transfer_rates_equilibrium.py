# =============================================================================
# Deterministic Log-Spaced Grid Sweep: Transfer Rates Explored Separately
# Two independent grid sweeps are run:
#   Sweep 1 — vary f_r_b (reservoir→blood), hold f_b_r at baseline
#   Sweep 2 — vary f_b_r (blood→reservoir), hold f_r_b at baseline
# Each sweep steps its transfer rate across a fixed log-spaced grid (same
# absolute range for both rates) while every other parameter (including the
# other transfer rate) stays fixed at baseline, isolating the effect of each
# rate on bacterial kinetics across a wide dynamic range.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODULE_NAME = "model_Bacteremia.py"
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
LOD            = 10.0       # limit of detection (CFU/mL)
RATE_MIN       = 1e-8       # lower bound of the log-spaced grid (shared by both rates)
RATE_MAX       = 1e-2       # upper bound of the log-spaced grid (shared by both rates)
COARSE_POINTS  = 40         # coarse coverage of the full range, for context
ZOOM_MIN       = 1e-3       # transition band identified in the first grid sweep
ZOOM_MAX       = 1e-2
ZOOM_POINTS    = 160        # dense coverage within the zoom band


def build_grid():
    coarse = np.logspace(np.log10(RATE_MIN), np.log10(RATE_MAX), COARSE_POINTS)
    coarse = coarse[(coarse < ZOOM_MIN) | (coarse > ZOOM_MAX)]
    zoom   = np.logspace(np.log10(ZOOM_MIN), np.log10(ZOOM_MAX), ZOOM_POINTS)
    return np.unique(np.concatenate([coarse, zoom]))


RATE_GRID  = build_grid()
NUM_POINTS = len(RATE_GRID)

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
rho_S        = 0.61    # raised from 0.60 -- see rho_S sensitivity analysis. At the actual
                       # baseline transfer rates used below (f_r_b=5e-5, f_b_r=1e-5), this
                       # produces no change (both sit far below the 1e-3-1e-2 transition
                       # band), but it does shift behavior inside that band itself
rho_R        = 0.55    # directly tuned (~8.3% fitness cost relative to rho_S, down from 20%)

BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_S":        0.175,  # scaled 5x (from 0.035) alongside rho_S
    "rho_res_R":        0.1765,  # narrow window just above the reservoir persistence threshold (0.17594055) so S_b can also establish in blood -- see model_Bacteremia.py
    "Emax_v":           0.40,
    "EC50_V":           0.245,
    "Emax_l":           0.8,  # fixed, decoupled from rho_S (was tied for "perfect bacteriostasis")
    "EC50_L":           1.0,
    "B_max_blood":      6000,
    "B_max_reservoir":  1e4,    # lowered from 4.5e6 so escaped R_res plateaus well above the LOD but far below its old level
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.45,
}

F_R_B_BASE = 5e-5   # reservoir -> blood baseline (restored to original)
F_B_R_BASE = 1e-5   # blood -> reservoir baseline

# Fixed initial conditions — only the swept rate is varied across iterations
Y0 = [0.0, 0.0, 100.0, 100.0]


# ---------------------------------------------------------------------------
# Run one grid sweep: step a single transfer rate across a log-spaced grid,
# hold the other transfer rate fixed at baseline
# ---------------------------------------------------------------------------
def run_transfer_rate_sweep(vary_name, vary_baseline, fixed_name, fixed_value):
    rates = RATE_GRID

    S_b_hist   = np.full((NUM_POINTS, len(t_eval)), np.nan)
    R_b_hist   = np.full((NUM_POINTS, len(t_eval)), np.nan)
    S_res_hist = np.full((NUM_POINTS, len(t_eval)), np.nan)
    R_res_hist = np.full((NUM_POINTS, len(t_eval)), np.nan)

    peak_S_b   = np.zeros(NUM_POINTS)
    peak_R_b   = np.zeros(NUM_POINTS)
    peak_S_res = np.zeros(NUM_POINTS)
    final_R_b  = np.zeros(NUM_POINTS)

    print(f"Running {NUM_POINTS} grid points (sweep: {vary_name}, "
          f"{fixed_name} fixed at {fixed_value:.1e})...", flush=True)

    for i in range(NUM_POINTS):
        params_i = {**BASE_PARAMS, vary_name: rates[i], fixed_name: fixed_value}

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

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{NUM_POINTS} complete", flush=True)

    return {
        "vary_name": vary_name,
        "rates":     rates,
        "S_b":       S_b_hist,
        "R_b":       R_b_hist,
        "S_res":     S_res_hist,
        "R_res":     R_res_hist,
        "peak_S_b":   peak_S_b,
        "peak_R_b":   peak_R_b,
        "peak_S_res": peak_S_res,
        "final_R_b":  final_R_b,
    }


sweep_f_r_b = run_transfer_rate_sweep("f_r_b", F_R_B_BASE, "f_b_r", F_B_R_BASE)
sweep_f_b_r = run_transfer_rate_sweep("f_b_r", F_B_R_BASE, "f_r_b", F_R_B_BASE)

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
    ax.set_title(title, fontsize=17)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize=13, ncol=2)


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
    fig.suptitle(f"Bacterial Kinetics — Log-Spaced Grid Sweep of {RATE_LABELS[sweep['vary_name']]}",
                 fontsize=19, fontweight="bold")

    panels = [
        (axes[0, 0], sweep["S_b"],   "royalblue", "blue", "Sensitive Blood ($S_b$)",     1e6),
        (axes[0, 1], sweep["R_b"],   "lightcoral", "red",  "Resistant Blood ($R_b$)",     1e6),
        (axes[1, 0], sweep["S_res"], "royalblue", "blue", "Sensitive Reservoir ($S_{res}$)", 2e7),
        (axes[1, 1], sweep["R_res"], "lightcoral", "red",  "Resistant Reservoir ($R_{res}$)", 2e7),
    ]

    trace_idx = np.linspace(0, NUM_POINTS - 1, min(20, NUM_POINTS)).astype(int)

    for ax, hist, lc, mc, title, ylim in panels:
        annotate_windows(ax)

        for j in trace_idx:
            tr = np.where(hist[j] == 0, np.nan, hist[j])
            ax.plot(t_days, tr, color=lc, alpha=0.12, lw=0.8)

        med  = np.nanpercentile(hist, 50, axis=0)
        lo5  = np.nanpercentile(hist,  5, axis=0)
        hi95 = np.nanpercentile(hist, 95, axis=0)

        med  = np.where(med  == 0, np.nan, med)
        lo5  = np.where(lo5  == 0, np.nan, lo5)
        hi95 = np.where(hi95 == 0, np.nan, hi95)

        ax.fill_between(t_days, lo5, hi95, color=lc, alpha=0.18)
        ax.plot(t_days, med, color=mc, lw=2, label=f"Median across grid (n={NUM_POINTS})")

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
def plot_sweep_outcomes(sweep, filename, rate_range=None, title_suffix=""):
    rates = sweep["rates"]
    in_range = np.ones_like(rates, dtype=bool)
    if rate_range is not None:
        lo, hi = rate_range
        in_range = (rates >= lo) & (rates <= hi)
    log10_rate = np.log10(rates)

    outcomes = [
        (sweep["peak_S_b"],   "Peak $S_b$ (CFU/mL)"),
        (sweep["peak_R_b"],   "Peak $R_b$ (CFU/mL)"),
        (sweep["peak_S_res"], "Peak $S_{res}$ (CFU/mL)"),
        (sweep["final_R_b"],  "Final $R_b$ (CFU/mL)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Outcomes vs {RATE_LABELS[sweep['vary_name']]}{title_suffix} "
                 f"(log-spaced grid, n={in_range.sum()})",
                 fontsize=19, fontweight="bold")

    for ax, (y_vals, ylabel) in zip(axes.flat, outcomes):
        nonzero = (y_vals > LOD) & in_range
        ax.plot(log10_rate[nonzero], y_vals[nonzero],
                marker="o", ms=3, lw=1, alpha=0.8, color="steelblue")
        ax.set_yscale("log")
        if not nonzero.any():
            # e.g. Peak/Final R_b now clears throughout -- give the empty
            # log-scale axis an explicit range so tight_layout doesn't crash
            ax.set_ylim(LOD * 0.1, LOD * 10)
            ax.set_xlim(log10_rate[in_range].min(), log10_rate[in_range].max())
            ax.text(0.5, 0.5, "All values below LOD", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, color="gray")
        ax.set_xlabel(f"$\\log_{{10}}$({RATE_LABELS[sweep['vary_name']]})")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", ls=":", alpha=0.35)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig)


plot_sweep_outcomes(sweep_f_r_b, "mc_frb_sweep_outcomes.png")
plot_sweep_outcomes(sweep_f_b_r, "mc_fbr_sweep_outcomes.png")

plot_sweep_outcomes(sweep_f_r_b, "mc_frb_sweep_outcomes_zoom.png",
                     rate_range=(ZOOM_MIN, ZOOM_MAX), title_suffix=" — zoom [1e-3, 1e-2]")
plot_sweep_outcomes(sweep_f_b_r, "mc_fbr_sweep_outcomes_zoom.png",
                     rate_range=(ZOOM_MIN, ZOOM_MAX), title_suffix=" — zoom [1e-3, 1e-2]")


# ---------------------------------------------------------------------------
# FIGURE: tested rate grid coverage for both sweeps (shared log-spaced grid)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
fig.suptitle(f"Log-Spaced Grid Coverage ({NUM_POINTS} points, {RATE_MIN:.1e}-{RATE_MAX:.1e})",
             fontsize=19, fontweight="bold")

ax = axes[0]
ax.eventplot(sweep_f_r_b["rates"], colors="tomato", lineoffsets=0, linelengths=0.8)
ax.axvline(F_R_B_BASE, color="darkred", ls="--", lw=1.5, label=f"Baseline ({F_R_B_BASE:.1e})")
ax.set_xscale("log")
ax.set_yticks([])
ax.set_title(f"Sweep 1: {RATE_LABELS['f_r_b']} varied (f_b_r fixed at {F_B_R_BASE:.1e})", fontsize=17)
ax.legend(fontsize=14)
ax.grid(True, which="both", ls=":", alpha=0.35)

ax = axes[1]
ax.eventplot(sweep_f_b_r["rates"], colors="steelblue", lineoffsets=0, linelengths=0.8)
ax.axvline(F_B_R_BASE, color="navy", ls="--", lw=1.5, label=f"Baseline ({F_B_R_BASE:.1e})")
ax.set_xscale("log")
ax.set_yticks([])
ax.set_xlabel("Transfer rate (shared log-spaced grid)")
ax.set_title(f"Sweep 2: {RATE_LABELS['f_b_r']} varied (f_r_b fixed at {F_R_B_BASE:.1e})", fontsize=17)
ax.legend(fontsize=14)
ax.grid(True, which="both", ls=":", alpha=0.35)

plt.tight_layout()
plt.savefig("mc_transfer_rate_sweep_grid.png", dpi=300, bbox_inches="tight")
print("Saved: mc_transfer_rate_sweep_grid.png")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(sweep, fixed_name, fixed_value):
    pk_Sb = sweep["peak_S_b"]
    pk_Rb = sweep["peak_R_b"]
    print(f"\n--- Sweep: {sweep['vary_name']} varied ({fixed_name} fixed at {fixed_value:.1e}) ---")
    if np.any(pk_Sb > LOD):
        print(f"  Peak S_b  median={np.median(pk_Sb[pk_Sb > LOD]):.1e}  "
              f"[5th={np.percentile(pk_Sb[pk_Sb > LOD], 5):.1e}, "
              f"95th={np.percentile(pk_Sb[pk_Sb > LOD], 95):.1e}] CFU/mL")
    else:
        print("  Peak S_b: no samples above LOD")
    if np.any(pk_Rb > LOD):
        print(f"  Peak R_b  median={np.median(pk_Rb[pk_Rb > LOD]):.1e}  "
              f"[5th={np.percentile(pk_Rb[pk_Rb > LOD], 5):.1e}, "
              f"95th={np.percentile(pk_Rb[pk_Rb > LOD], 95):.1e}] CFU/mL")
    else:
        print("  Peak R_b: no samples above LOD")


print_summary(sweep_f_r_b, "f_b_r", F_B_R_BASE)
print_summary(sweep_f_b_r, "f_r_b", F_R_B_BASE)

print("\nDone.")
