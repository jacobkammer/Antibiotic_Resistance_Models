# =============================================================================
# Monte Carlo Simulation: Linezolid EC50 (EC50_L)
# Samples EC50_L from a log-normal distribution centred on baseline (1.0 mg/L)
# while every other parameter stays fixed, isolating the effect of linezolid
# potency/susceptibility on bacterial kinetics.
#
# Growth rates were scaled 5x (rho_S/rho_R) alongside the original Emax_l
# increase, but the EC50_L change independently weakens linezolid's effect at
# realistic (sub-saturating) concentrations, which growth-rate scaling cannot
# offset.
#
# B_max_reservoir was lowered 4.5e6 -> 1e4 (see baseline_reservoir_
# clearance.py) so an escaped R_res plateaus far below its old level.
# rho_S/rho_R were lowered 0.80/0.64 -> 0.60/0.55, and Emax_l was UNCOUPLED
# from rho_S and fixed at 0.8 -- at that point Emax_l exceeded every species'
# growth rate and ALL FOUR compartments cleared at baseline, the only point
# in this project where that happened.
#
# rho_res_R was then raised 0.145 -> 0.20 (precise clearance threshold
# 0.17594055) -- now a fitness ADVANTAGE over rho_res_S (0.175) -- which
# pulled the joint R_b/R_res escape threshold back down to 0.573 mg/L and
# restored "no clinical resolution": R_b visibly CLEARS during the linezolid
# course before relapsing. BUT at rho_res_R = 0.20, R_res's reservoir growth
# advantage over S_res (0.175) was large enough that R_b won the
# pretreatment race for blood's shared carrying capacity outright, crowding
# S_b out entirely -- no visible sensitive-strain infection anywhere in this
# project's Monte Carlo scripts.
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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODULE_NAME = "model.ClinicalResponse.py"
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
SIGMA          = 0.9        # log-normal spread applied to EC50_L
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
rho_S        = 0.60    # lowered from 0.80 for slower post-treatment R_b regrowth; Emax_l auto-follows
rho_R        = 0.55    # directly tuned (~8.3% fitness cost relative to rho_S, down from 20%)

EC50_L_BASE = 1.0

BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_S":        0.175,  # scaled 5x (from 0.035) alongside rho_S
    "rho_res_R":        0.1765,  # narrow window just above the reservoir persistence threshold (0.17594055) so S_b can also establish in blood -- see model.ClinicalResponse.py
    "Emax_v":           0.40,
    "EC50_V":           0.245,
    "Emax_l":           0.8,  # fixed, decoupled from rho_S (was tied for "perfect bacteriostasis")
    "B_max_blood":      6000,
    "B_max_reservoir":  1e4,    # lowered from 4.5e6 so escaped R_res plateaus well above the LOD but far below its old level
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.45,
    "f_r_b":            5e-5,  # restored to original
    "f_b_r":            1e-5,
}

# Fixed initial conditions — only EC50_L is varied across iterations
Y0 = [0.0, 0.0, 100.0, 100.0]

# ---------------------------------------------------------------------------
# Run Monte Carlo sweep: vary EC50_L, hold everything else fixed
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
ec50_l_samples = rng.lognormal(np.log(EC50_L_BASE), SIGMA, NUM_ITERATIONS)

# ---------------------------------------------------------------------------
# Resistant-strain escape bins & threshold (Final Rb / Rres only)
# Precise threshold located by direct root-finding against the model's
# current defaults (rho_S = 0.60, rho_R = 0.55, rho_res_R = 0.1765 -- a
# narrow window just above the reservoir's own 0.17594055 persistence
# threshold, chosen so S_b can also establish in blood -- Emax_l = 0.8
# FIXED, decoupled from rho_S, see model.ClinicalResponse.py's
# ImmuneResponse class -- EC50_V = 0.245, eff_blood = 1.0), see
# EC50_lzd_threshold_sweep.py:
#   Joint R_b / R_res escape threshold = 0.988 mg/L
# (up from 0.573 mg/L at rho_res_R = 0.20 -- the thinner rho_res_R margin
# pulls this threshold right next to baseline). R_b and R_res cross the LOD
# together (a single shared eff_blood couples their fates). Baseline
# EC50_L = 1.0 sits just above the threshold, so R_b/R_res persist at
# baseline (R_b visibly clears during the linezolid course, then relapses
# ~3 days after treatment ends) -- but only ~53% of sampled EC50_L values
# do so; the rest suppress the reservoir entirely. Both outcomes are
# accepted; S_b establishing is the priority for this parameterization.
#
# Bin edges tightened around the new threshold (0.988, right at baseline).
# ---------------------------------------------------------------------------
ESCAPE_THRESH = 0.988   # both R_b and R_res cross the LOD here (joint threshold)

res_bin_edges    = np.array([0.40, 0.60, 0.80, ESCAPE_THRESH, 1.10, 1.30, 1.60, 2.00])
RES_EC50_CENTERS = np.sqrt(res_bin_edges[:-1] * res_bin_edges[1:])
RES_EC50_LABELS  = [f"{c:.2f}" for c in RES_EC50_CENTERS]
N_RES_EC50_BINS  = len(RES_EC50_CENTERS)

REGIME_COLORS = {
    "suppressed":  "#4c72b0",  # blue — both R_b and R_res stay under the LOD
    "both_escape": "#c44e52",  # red  — both R_b and R_res cross the LOD
}


def _regime(lo, hi):
    if hi <= ESCAPE_THRESH:
        return "suppressed"
    return "both_escape"


res_bin_regimes = [_regime(res_bin_edges[b], res_bin_edges[b + 1]) for b in range(N_RES_EC50_BINS)]
res_bin_colors  = [REGIME_COLORS[r] for r in res_bin_regimes]

res_in_range = (ec50_l_samples >= res_bin_edges[0]) & (ec50_l_samples <= res_bin_edges[-1])
res_bin_idx  = np.digitize(ec50_l_samples, res_bin_edges[1:-1])

regime_handles = [
    mpatches.Patch(facecolor=REGIME_COLORS["suppressed"],  alpha=0.65, label=f"Suppressed (< {ESCAPE_THRESH:.3f})"),
    mpatches.Patch(facecolor=REGIME_COLORS["both_escape"], alpha=0.65, label=f"Both escape (≥ {ESCAPE_THRESH:.3f})"),
]

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


# ---------------------------------------------------------------------------
# FIGURE: 4-panel trajectory kinetics — S_b, R_b, S_res, R_res
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(r"Bacterial Kinetics — Monte Carlo sweep of $EC_{50,L}$ (linezolid)",
             fontsize=12, fontweight="bold")

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
peak_S_b   = np.nanmax(S_b_hist, axis=1)
peak_R_b   = np.nanmax(R_b_hist, axis=1)
peak_S_res = np.nanmax(S_res_hist, axis=1)
final_R_b  = np.nan_to_num(R_b_hist[:, -1], nan=0.0)


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
fig2.suptitle(r"Outcomes vs $EC_{50,L}$ (linezolid)", fontsize=12, fontweight="bold")

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
                ha="center", va="center", fontsize=9, color="gray")
    ax.set_xlabel(r"$\log_{10}(EC_{50,L})$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("mc_ec50_lzd_sweep_outcomes.png", dpi=300, bbox_inches="tight")
print("Saved: mc_ec50_lzd_sweep_outcomes.png")


# ---------------------------------------------------------------------------
# FIGURE: bar chart of FINAL resistant-strain levels at representative
# EC50_L values (R_b, R_res only). Uses the same escape bins/thresholds and
# regime coloring as the boxplot figure below.
# ---------------------------------------------------------------------------
bar_final_R_b, bar_final_R_res = [], []

print("\nRunning single simulations at representative EC50_L levels...", flush=True)
for ec in RES_EC50_CENTERS:
    params_i = {**BASE_PARAMS, "EC50_L": ec}
    sol = odeint(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(params_i, van_func, lzd_func, immune_model),
        rtol=1e-7, atol=1e-9, mxstep=5000,
    )

    rb   = sol[-1, 1] if sol[-1, 1] >= LOD else 0.0
    rres = sol[-1, 3] if sol[-1, 3] >= LOD else 0.0

    bar_final_R_b.append(rb)
    bar_final_R_res.append(rres)

bar_panels = [
    (bar_final_R_b,   "Final $R_b$ (CFU/mL)",     1e6),
    (bar_final_R_res, "Final $R_{res}$ (CFU/mL)", 8e7),
]

FLOOR = LOD * 0.5   # display floor for values that fell below the LOD (i.e. 0)

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle(r"Final Resistant-Strain Levels at Representative $EC_{50,L}$ Values (linezolid)",
              fontsize=12, fontweight="bold")

for ax, (values, ylabel, ylim) in zip(axes3.flat, bar_panels):
    disp_values = [FLOOR if v <= 0 else v for v in values]
    ax.bar(RES_EC50_LABELS, disp_values, color=res_bin_colors, edgecolor="black", linewidth=0.6)

    ax.axhline(LOD, color="black", ls=":", lw=0.9, alpha=0.6, label="LOD (10 CFU/mL)")
    ax.set_yscale("log")
    ax.set_ylim(LOD * 0.3, ylim)
    ax.set_xlabel(r"$EC_{50,L}$ (mg/L)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontsize=10, pad=12)
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.35)
    ax.legend(handles=regime_handles + [ax.get_legend_handles_labels()[0][0]],
              labels=[h.get_label() for h in regime_handles] + ["LOD (10 CFU/mL)"],
              fontsize=6.5, loc="upper left")

plt.tight_layout()
plt.savefig("mc_ec50_lzd_bar_levels.png", dpi=300, bbox_inches="tight")
print("Saved: mc_ec50_lzd_bar_levels.png")


# ---------------------------------------------------------------------------
# FIGURE: boxplots of FINAL (end-of-simulation) resistant bacterial counts
# (R_b, R_res only), built from the full 400-run Monte Carlo sweep above and
# grouped into the escape bins/thresholds defined earlier.
# ---------------------------------------------------------------------------
final_R_b   = np.nan_to_num(R_b_hist[:, -1],   nan=0.0)
final_R_res = np.nan_to_num(R_res_hist[:, -1], nan=0.0)

RES_XLABEL = r"$EC_{50,L}$ (mg/L)"

final_box_panels = [
    (final_R_b,   "Final $R_b$ (CFU/mL)",     1e6),
    (final_R_res, "Final $R_{res}$ (CFU/mL)", 8e7),
]

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
fig4.suptitle(r"Final Resistant Bacterial Counts vs $EC_{50,L}$ — Monte Carlo Sweep (linezolid)",
              fontsize=12, fontweight="bold")

for ax, (final_vals, ylabel, ylim) in zip(axes4.flat, final_box_panels):
    box_data = []
    for b in range(N_RES_EC50_BINS):
        mask = res_in_range & (res_bin_idx == b)
        vals = final_vals[mask]
        box_data.append(np.where(vals <= 0, FLOOR, vals))

    bp = ax.boxplot(box_data, positions=range(N_RES_EC50_BINS), widths=0.6,
                     patch_artist=True, showfliers=True,
                     medianprops=dict(color="black", lw=1.5))
    for patch, color in zip(bp["boxes"], res_bin_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.axhline(LOD, color="black", ls=":", lw=0.9, alpha=0.6, label="LOD (10 CFU/mL)")
    ax.set_xticks(range(N_RES_EC50_BINS))
    ax.set_xticklabels(RES_EC50_LABELS)
    ax.set_yscale("log")
    ax.set_ylim(LOD * 0.3, ylim)
    ax.set_xlabel(RES_XLABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontsize=10, pad=12)
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.35)
    ax.legend(handles=regime_handles + [ax.get_legend_handles_labels()[0][0]],
              labels=[h.get_label() for h in regime_handles] + ["LOD (10 CFU/mL)"],
              fontsize=6.5, loc="upper left")

plt.tight_layout()
plt.savefig("mc_ec50_lzd_final_boxplot.png", dpi=300, bbox_inches="tight")
print("Saved: mc_ec50_lzd_final_boxplot.png")


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
