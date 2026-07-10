# =============================================================================
# Monte Carlo Simulation: Linezolid EC50 (EC50_L)
# Samples EC50_L from a log-normal distribution centred on baseline (1.0 mg/L)
# while every other parameter stays fixed, isolating the effect of linezolid
# potency/susceptibility on bacterial kinetics.
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
rho_S        = 0.16
rho_R        = 0.128   # directly tuned (20% fitness cost relative to rho_S)

EC50_L_BASE = 1.0

BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_S":        0.035,
    "rho_res_R":        0.024,   # lowered below the 20%-fitness-cost value (0.028) so R_res clears before linezolid ends
    "Emax_v":           0.40,
    "EC50_V":           1.5,
    "Emax_l":           rho_S,
    "B_max_blood":      5e5,
    "B_max_reservoir":  4.5e6,
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
rng = np.random.default_rng(SEED)
ec50_l_samples = rng.lognormal(np.log(EC50_L_BASE), SIGMA, NUM_ITERATIONS)

# ---------------------------------------------------------------------------
# Resistant-strain escape bins & thresholds (Final Rb / Rres only)
# Bin edges chosen from EC50_lzd_threshold_sweep.py, which located the exact
# EC50_L values where resistant counts cross the LOD by end-of-simulation
# (with S_res_0 = 100 and rho_res_R = 0.024, matching model.LinearIM.py's
# own defaults -- lowering rho_res_R pushed both thresholds well above
# baseline EC50_L = 1.0, so the reservoir clears comfortably at baseline):
#   R_res escape threshold ~= 1.86 mg/L
#   R_b   escape threshold ~= 2.62 mg/L
# Bins are built around those two thresholds so every resistant-strain figure
# shows all three regimes: fully suppressed / reservoir-only escape / both escape.
# ---------------------------------------------------------------------------
RES_ESCAPE_THRESH   = 1.863   # R_res crosses LOD here
BLOOD_ESCAPE_THRESH = 2.616   # R_b   crosses LOD here

res_bin_edges    = np.array([0.70, 1.00, 1.30, 1.60, RES_ESCAPE_THRESH, 2.20, BLOOD_ESCAPE_THRESH, 3.10, 3.50])
RES_EC50_CENTERS = np.sqrt(res_bin_edges[:-1] * res_bin_edges[1:])
RES_EC50_LABELS  = [f"{c:.2f}" for c in RES_EC50_CENTERS]
N_RES_EC50_BINS  = len(RES_EC50_CENTERS)

REGIME_COLORS = {
    "suppressed":        "#4c72b0",  # blue   — both compartments stay under LOD
    "reservoir_escapes": "#dd8452",  # orange — only R_res crosses the LOD
    "both_escape":       "#c44e52",  # red    — both R_b and R_res cross the LOD
}


def _regime(lo, hi):
    if hi <= RES_ESCAPE_THRESH:
        return "suppressed"
    if hi <= BLOOD_ESCAPE_THRESH:
        return "reservoir_escapes"
    return "both_escape"


res_bin_regimes = [_regime(res_bin_edges[b], res_bin_edges[b + 1]) for b in range(N_RES_EC50_BINS)]
res_bin_colors  = [REGIME_COLORS[r] for r in res_bin_regimes]

res_in_range = (ec50_l_samples >= res_bin_edges[0]) & (ec50_l_samples <= res_bin_edges[-1])
res_bin_idx  = np.digitize(ec50_l_samples, res_bin_edges[1:-1])

regime_handles = [
    mpatches.Patch(facecolor=REGIME_COLORS["suppressed"],        alpha=0.65, label=f"Suppressed (< {RES_ESCAPE_THRESH:.2f})"),
    mpatches.Patch(facecolor=REGIME_COLORS["reservoir_escapes"], alpha=0.65, label=f"$R_{{res}}$ escapes ({RES_ESCAPE_THRESH:.2f}–{BLOOD_ESCAPE_THRESH:.2f})"),
    mpatches.Patch(facecolor=REGIME_COLORS["both_escape"],       alpha=0.65, label=f"Both escape (≥ {BLOOD_ESCAPE_THRESH:.2f})"),
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
print(f"  Peak S_b  median={np.median(peak_S_b[peak_S_b > LOD]):.1e}  "
      f"[5th={np.percentile(peak_S_b[peak_S_b > LOD], 5):.1e}, "
      f"95th={np.percentile(peak_S_b[peak_S_b > LOD], 95):.1e}] CFU/mL")
if np.any(peak_R_b > LOD):
    print(f"  Peak R_b  median={np.median(peak_R_b[peak_R_b > LOD]):.1e}  "
          f"[5th={np.percentile(peak_R_b[peak_R_b > LOD], 5):.1e}, "
          f"95th={np.percentile(peak_R_b[peak_R_b > LOD], 95):.1e}] CFU/mL")
else:
    print("  Peak R_b: no samples above LOD")

print("\nDone.")
