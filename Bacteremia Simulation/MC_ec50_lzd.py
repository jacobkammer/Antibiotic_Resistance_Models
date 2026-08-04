# =============================================================================
# Monte Carlo Simulation: Linezolid EC50 (EC50_L)
# Samples EC50_L ~ LogNormal(mean=1.0 mg/L, CV=1.117), mean-corrected so
# E[EC50_L] lands exactly on baseline (same convention as MC_immune_response.py
# and MC_emax_lzd.py), while every other parameter stays fixed, isolating the
# effect of linezolid potency/susceptibility on bacterial kinetics.
#
# Growth rates were scaled 5x (rho_S/rho_R) alongside the original Emax_l
# increase, but the EC50_L change independently weakens linezolid's effect at
# realistic (sub-saturating) concentrations, which growth-rate scaling cannot
# offset.
#
# 
# rho_S/rho_R were lowered 0.80/0.64 -> 0.61/0.55, and Emax_l was UNCOUPLED
# from rho_S and fixed at 0.8 -- at that point Emax_l exceeded every species'
# growth rate and ALL FOUR compartments cleared at baseline, the only point
# in this project where that happened.
#
# 
#
# rho_res_R was finally narrowed to 0.1765 -- just above the reservoir's own
# 0.17594055 persistence threshold, but below the point where R_b starts
# winning the blood race -- so S_b CAN also establish a visible infection
# (peak ~4.8e3 CFU/mL). This pulled the EC50_L escape threshold up to
# 0.988 mg/L, right next to baseline (1.0): only ~47% of sampled EC50_L
# values now suppress the reservoir, vs. ~53% that let it persist and
# relapse. Both outcomes are accepted here -- the point of this rho_res_R is
# to let S_b establish, not to guarantee reservoir persistence in every
# sample.
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq

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
NUM_ITERATIONS = 1000
LOD            = 10.0       # limit of detection (CFU/mL)
CV             = 1.117      # log-normal spread applied to EC50_L 
SEED           = 44

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
# Fixed parameters (all except EC50_L held at baseline)
# ---------------------------------------------------------------------------
rho_S        = 0.61    # 
rho_R        = 0.55    # (10%fitness cost)

EC50_L_BASE = 1.0

BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_S":        0.175,  # 
    "rho_res_R":        0.1765,  # narrow window just above the reservoir persistence threshold (0.17594055) so S_b can also establish in blood -- see model_Bacteremia.py
    "Emax_v":           0.40,
    "EC50_V":           0.245,
    "Emax_l":           0.8,  # fixed, decoupled from rho_S 
    "B_max_blood":      6000,
    "B_max_reservoir":  1e4,    
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.45,
    "f_r_b":            5e-5,  
    "f_b_r":            1e-5,
}

# Fixed initial conditions — only EC50_L is varied across iterations
Y0 = [0.0, 0.0, 100.0, 100.0]

# ---------------------------------------------------------------------------
# Run Monte Carlo sweep: vary EC50_L, hold everything else fixed
# ---------------------------------------------------------------------------
#np.random.lognormal() generates samples from a lognormal distribution
sigma_ln = np.sqrt(np.log(1.0 + CV**2)) #converts CV to sigma for lognormal distribution
mu_ln    = np.log(EC50_L_BASE) - sigma_ln**2 / 2.0 # ensures E[EC50_L] = EC50_L_BASE

rng = np.random.default_rng(SEED)
ec50_l_samples = rng.lognormal(mu_ln, sigma_ln, NUM_ITERATIONS)

# ---------------------------------------------------------------------------
# Continuous deterministic switch curve + precise escape threshold.
#
# Replaces the old fixed ESCAPE_THRESH=0.988 (which no longer matched this
# file's own rho_S=0.61 -- see EC50_lzd_threshold_sweep.py, which bisects the
# same BASE_PARAMS to ~0.93) with a value bisected directly here, and
# replaces the 5-bin categorical grouping (which only ever produced two flat
# levels either side of one straddling bin) with a fine, continuous sweep --
# the actual switch curve -- used by the combined figure below.
# ---------------------------------------------------------------------------
def _final_counts(ec50_l):
    """One deterministic run at a fixed EC50_L; returns (final R_b, final R_res)."""
    params_i = {**BASE_PARAMS, "EC50_L": ec50_l}
    sol = odeint(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(params_i, van_func, lzd_func, immune_model),
        rtol=1e-7, atol=1e-9, mxstep=5000,
    )
    return sol[-1, 1], sol[-1, 3]


SWEEP_MIN, SWEEP_MAX = 0.3, 2.0
N_SWEEP = 121   # step ~0.014

print(f"\nRunning deterministic switch-curve sweep across [{SWEEP_MIN}, {SWEEP_MAX}] "
      f"({N_SWEEP} points)...", flush=True)
sweep_ec50  = np.linspace(SWEEP_MIN, SWEEP_MAX, N_SWEEP)
sweep_R_b   = np.zeros(N_SWEEP)
sweep_R_res = np.zeros(N_SWEEP)
for i, ec in enumerate(sweep_ec50):
    sweep_R_b[i], sweep_R_res[i] = _final_counts(ec)


def _f_res(ec50_l):
    return _final_counts(ec50_l)[1] - LOD


if _f_res(SWEEP_MIN) > 0 or _f_res(SWEEP_MAX) < 0:
    ESCAPE_THRESH = float(sweep_ec50[np.argmax(sweep_R_res > LOD)])  # fallback: coarse crossing
else:
    ESCAPE_THRESH = brentq(_f_res, SWEEP_MIN, SWEEP_MAX, xtol=1e-3)

print(f"Precise escape threshold (final R_res crosses LOD): {ESCAPE_THRESH:.3f} mg/L")

REGIME_COLORS = {"suppressed": "#4c72b0", "escape": "#c44e52"}  # blue / red

S_b_hist   = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
R_b_hist   = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
S_res_hist = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
R_res_hist = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)

print(f"Running {NUM_ITERATIONS} iterations (sweep: EC50_L, baseline={EC50_L_BASE:.2f})...",
      flush=True)

for i in range(NUM_ITERATIONS):
    params_i = {**BASE_PARAMS, "EC50_L": ec50_l_samples[i]}

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

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{NUM_ITERATIONS} complete", flush=True)

print("Simulations complete. Generating figure...", flush=True)


# ---------------------------------------------------------------------------
# Helper functions for formatting the time-course plots
# ---------------------------------------------------------------------------
def annotate_windows(ax):# adds shaded regions for antibiotic windows
    ax.axvspan(vanco_start_days, lzd_start_days, color="gray", alpha=0.12, label="Vancomycin")
    ax.axvspan(lzd_start_days,  lzd_end_days,    color="gold", alpha=0.12, label="Linezolid")
    ax.axhline(LOD, color="black", ls=":", lw=0.9, alpha=0.6, label="LOD (10 CFU/mL)")


def style_kinetic(ax, title, ylabel, ylim_top): # formats a kinetic (bacterial count vs. time) plot
    ax.set_yscale("log")
    ax.set_ylim(LOD * 0.5, ylim_top)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=17)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize=13, ncol=2)


# ---------------------------------------------------------------------------
# FIGURE: 4-panel trajectory kinetics — S_b, R_b, S_res, R_res
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(r"Bacterial Kinetics — Monte Carlo sweep of $EC_{50,L}$ (linezolid)",
             fontsize=19, fontweight="bold")

panels = [
    (axes[0, 0], S_b_hist,   "royalblue",  "blue", "Sensitive Blood ($S_b$)",         1e6),
    (axes[0, 1], R_b_hist,   "lightcoral", "red",  "Resistant Blood ($R_b$)",         1e6),
    (axes[1, 0], S_res_hist, "royalblue",  "blue", "Sensitive Reservoir ($S_{res}$)", 2e7),
    (axes[1, 1], R_res_hist, "lightcoral", "red",  "Resistant Reservoir ($R_{res}$)", 2e7),
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
plt.savefig("mc_ec50_lzd_sweep_kinetics.png", dpi=300, bbox_inches="tight")
print("Saved: mc_ec50_lzd_sweep_kinetics.png")


# ---------------------------------------------------------------------------
# Outcome metrics (one summary value per iteration)
# ---------------------------------------------------------------------------
peak_S_b    = np.nanmax(S_b_hist, axis=1)
peak_R_b    = np.nanmax(R_b_hist, axis=1)
peak_S_res  = np.nanmax(S_res_hist, axis=1)
final_R_b   = np.nan_to_num(R_b_hist[:, -1],   nan=0.0)
final_R_res = np.nan_to_num(R_res_hist[:, -1], nan=0.0)


# ---------------------------------------------------------------------------
# FIGURE: outcome scatter plots vs EC50_L (log scale)
# 4 panels: peak S_b, peak R_b, peak S_res, final R_b
# ---------------------------------------------------------------------------
log10_ec50 = np.log10(ec50_l_samples)

outcomes = [
    (peak_S_b,   "Peak $S_b$ (CFU/mL)"),
    (peak_R_b,   "Peak $R_b$ (CFU/mL)"),
    (peak_S_res, "Peak $S_{res}$ (CFU/mL)"),
    (final_R_b,  "Final $R_b$ (CFU/mL)"),
]

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
fig2.suptitle(r"Outcomes vs $EC_{50,L}$ (linezolid)", fontsize=19, fontweight="bold")

for ax, (y_vals, ylabel) in zip(axes2.flat, outcomes):
    nonzero = y_vals > LOD
    ax.scatter(log10_ec50[nonzero], y_vals[nonzero],
               s=18, alpha=0.5, color="darkorange", edgecolors="none")
    ax.axvline(np.log10(EC50_L_BASE), color="black", ls="--", lw=1.0, alpha=0.7,
               label=f"Baseline ({EC50_L_BASE:.1f})")
    ax.set_yscale("log")
    if not nonzero.any():
        # e.g. Peak/Final R_b now clears for every sampled EC50_L -- give the
        # empty log-scale axis an explicit range so tight_layout doesn't crash
        ax.set_ylim(LOD * 0.1, LOD * 10)
        ax.text(0.5, 0.5, "All values below LOD", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color="gray")
    ax.set_xlabel(r"$\log_{10}(EC_{50,L})$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(fontsize=13)

plt.tight_layout()
plt.savefig("mc_ec50_lzd_sweep_outcomes.png", dpi=300, bbox_inches="tight")
print("Saved: mc_ec50_lzd_sweep_outcomes.png")


# ---------------------------------------------------------------------------
# FIGURE: switch curve (continuous, deterministic) + sampled patient
# population relative to the escape threshold.
#
# Top panel is the actual result: final resistant burden vs EC50_L,
# continuous and bisected rather than binned, so the transition at
# ESCAPE_THRESH reads as a genuine switch rather than a coarse step between a
# handful of categories.
# Bottom panel answers the question the switch alone doesn't: given the same
# realistic inter-patient EC50_L variability used for the kinetics/outcomes
# figures above, what fraction of virtual patients actually land on each
# side of it. Each sample is colored by its own simulated outcome (final
# R_res vs LOD), not by which side of the threshold its EC50_L happens to
# fall on, so the figure stays consistent with the simulated data.
# ---------------------------------------------------------------------------
FLOOR = LOD * 0.5   # display floor for values that fell below the LOD (i.e. 0)

disp_sweep_R_b   = np.where(sweep_R_b   <= LOD, FLOOR, sweep_R_b)
disp_sweep_R_res = np.where(sweep_R_res <= LOD, FLOOR, sweep_R_res)

escaped_mask    = final_R_res > LOD
frac_escape     = escaped_mask.mean()
escaped_ec50    = ec50_l_samples[escaped_mask]
suppressed_ec50 = ec50_l_samples[~escaped_mask]

fig3, (ax_top, ax_hist) = plt.subplots(
    2, 1, figsize=(11, 8.5), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08},
)
fig3.suptitle(r"Resistant Escape Switch vs $EC_{50,L}$ (linezolid)",
              fontsize=19, fontweight="bold")

# --- Top: continuous switch curve ---
ax_top.axvspan(SWEEP_MIN, ESCAPE_THRESH, color=REGIME_COLORS["suppressed"], alpha=0.10)
ax_top.axvspan(ESCAPE_THRESH, SWEEP_MAX, color=REGIME_COLORS["escape"],     alpha=0.10)
ax_top.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")
ax_top.axvline(ESCAPE_THRESH, color="black", ls="--", lw=1.4,
               label=fr"Switch ($EC_{{50,L}}$ = {ESCAPE_THRESH:.2g})")

ax_top.plot(sweep_ec50, disp_sweep_R_res, color="indianred", lw=2.2,
            label="Final $R_{res}$ (reservoir)")
ax_top.plot(sweep_ec50, disp_sweep_R_b, color="darkred", lw=2.2,
            label="Final $R_b$ (blood)")

ax_top.set_yscale("log")
ax_top.set_ylim(FLOOR * 0.8, max(sweep_R_b.max(), sweep_R_res.max()) * 3)
ax_top.set_ylabel("Final resistant count (CFU/mL)")
ax_top.grid(True, which="both", ls=":", alpha=0.35)
ax_top.legend(loc="upper left", fontsize=13, framealpha=0.9)

# --- Bottom: where the sampled patient population falls, by actual outcome ---
bin_edges = np.linspace(SWEEP_MIN, SWEEP_MAX, 31)
ax_hist.hist(suppressed_ec50, bins=bin_edges, color=REGIME_COLORS["suppressed"],
             alpha=0.75, edgecolor="white", linewidth=0.4,
             label=f"Suppressed ({(1 - frac_escape) * 100:.0f}% of patients)")
ax_hist.hist(escaped_ec50, bins=bin_edges, color=REGIME_COLORS["escape"],
             alpha=0.75, edgecolor="white", linewidth=0.4,
             label=f"Escape ({frac_escape * 100:.0f}% of patients)")
ax_hist.axvline(ESCAPE_THRESH, color="black", ls="--", lw=1.4)

ax_hist.set_xlim(SWEEP_MIN, SWEEP_MAX)
ax_hist.set_xlabel(r"$EC_{50,L}$ (mg/L)")
ax_hist.set_ylabel("Patients\n(n)")
ax_hist.grid(True, axis="y", ls=":", alpha=0.35)
ax_hist.legend(loc="upper right", fontsize=12, framealpha=0.9)

fig3.savefig("mc_ec50_lzd_switch.png", dpi=300, bbox_inches="tight")
print("Saved: mc_ec50_lzd_switch.png")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print(f"\n--- Sweep: EC50_L varied (baseline={EC50_L_BASE:.2f}) ---")
print(f"  EC50_L sampled  median={np.median(ec50_l_samples):.2f}  "
      f"[5th={np.percentile(ec50_l_samples, 5):.2f}, "
      f"95th={np.percentile(ec50_l_samples, 95):.2f}]")
if np.any(peak_S_b > LOD):
    print(f"  Peak S_b  median={np.median(peak_S_b[peak_S_b > LOD]):.1e}  "
          f"[5th={np.percentile(peak_S_b[peak_S_b > LOD], 5):.1e}, "
          f"95th={np.percentile(peak_S_b[peak_S_b > LOD], 95):.1e}] CFU/mL")
else:
    print("  Peak S_b: no samples above LOD")
if np.any(peak_R_b > LOD):
    print(f"  Peak R_b  median={np.median(peak_R_b[peak_R_b > LOD]):.1e}  "
          f"[5th={np.percentile(peak_R_b[peak_R_b > LOD], 5):.1e}, "
          f"95th={np.percentile(peak_R_b[peak_R_b > LOD], 95):.1e}] CFU/mL")
else:
    print("  Peak R_b: no samples above LOD")

print("\nDone.")
