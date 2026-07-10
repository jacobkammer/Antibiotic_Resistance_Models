# =============================================================================
# Monte Carlo Simulation: Linezolid Emax (Emax_l)
# Samples Emax_l from a log-normal distribution centred on baseline (0.16 h^-1,
# = rho_S, "perfect bacteriostasis") while every other parameter stays fixed,
# isolating the effect of linezolid's maximum killing capacity on bacterial
# kinetics. Mirrors MC_ec50_lzd.py, but for Emax_l instead of EC50_L.
#
# Unlike EC50_L (where HIGH values are bad), for Emax_l LOW values are bad:
# below a threshold, linezolid's ceiling effect can no longer hold resistant
# growth in check and R_b / R_res escape. Thresholds located by
# Emax_lzd_threshold_sweep.py:
#   R_b   escape threshold ~= 0.147 h^-1
#   R_res escape threshold ~= 0.158 h^-1
# (baseline sits at 0.16 h^-1, just above both)
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
SIGMA          = 0.3        # log-normal spread applied to Emax_l
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
# Fixed parameters (all except Emax_l held at baseline)
# ---------------------------------------------------------------------------
rho_S        = 0.16
rho_R        = 0.128   # directly tuned (20% fitness cost relative to rho_S)

EMAX_L_BASE = rho_S   # 0.16 h^-1

BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_S":        0.035,
    "rho_res_R":        0.024,   # lowered below the 20%-fitness-cost value (0.028) so R_res clears before linezolid ends
    "Emax_v":           0.40,
    "EC50_V":           1.5,
    "EC50_L":           1.0,
    "B_max_blood":      5e5,
    "B_max_reservoir":  4.5e6,
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.45,
    "f_r_b":            5e-5,
    "f_b_r":            1e-5,
}

# Fixed initial conditions — only Emax_l is varied across iterations
Y0 = [0.0, 0.0, 100.0, 100.0]

# ---------------------------------------------------------------------------
# Run Monte Carlo sweep: vary Emax_l, hold everything else fixed
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
emax_l_samples = rng.lognormal(np.log(EMAX_L_BASE), SIGMA, NUM_ITERATIONS)

# ---------------------------------------------------------------------------
# Resistant-strain escape bins & thresholds (Final Rb / Rres only)
# Bin edges chosen from Emax_lzd_threshold_sweep.py, which located the exact
# Emax_l values where resistant counts cross the LOD by end-of-simulation
# (with S_res_0 = 100 and rho_res_R = 0.024, matching model.LinearIM.py's
# own defaults -- lowering rho_res_R pushed both thresholds well below
# baseline Emax_l = 0.16, so the reservoir clears comfortably at baseline):
#   R_b   escape threshold ~= 0.110 h^-1  (higher Emax_l -> R_b suppressed)
#   R_res escape threshold ~= 0.128 h^-1  (higher Emax_l -> R_res suppressed)
# Bins are built around those two thresholds so every resistant-strain figure
# shows all three regimes: both escape / reservoir-only escape / suppressed.
# Direction is flipped vs EC50_L: here LOW Emax_l is the escape zone.
# ---------------------------------------------------------------------------
BLOOD_ESCAPE_THRESH = 0.1096   # R_b   crosses LOD here (increasing Emax_l)
RES_ESCAPE_THRESH   = 0.1283   # R_res crosses LOD here (increasing Emax_l)

emax_bin_edges    = np.array([0.06, 0.08, 0.10, BLOOD_ESCAPE_THRESH, RES_ESCAPE_THRESH, 0.15, 0.18, 0.22])
EMAX_L_CENTERS    = np.sqrt(emax_bin_edges[:-1] * emax_bin_edges[1:])
EMAX_L_LABELS     = [f"{c:.2f}" for c in EMAX_L_CENTERS]
N_EMAX_L_BINS     = len(EMAX_L_CENTERS)

REGIME_COLORS = {
    "both_escape":       "#c44e52",  # red    — both R_b and R_res cross the LOD (escaped)
    "reservoir_escapes": "#dd8452",  # orange — only R_res still crosses the LOD
    "suppressed":        "#4c72b0",  # blue   — both compartments stay under LOD
}


def _regime(lo, hi):
    if hi <= BLOOD_ESCAPE_THRESH:
        return "both_escape"
    if hi <= RES_ESCAPE_THRESH:
        return "reservoir_escapes"
    return "suppressed"


emax_bin_regimes = [_regime(emax_bin_edges[b], emax_bin_edges[b + 1]) for b in range(N_EMAX_L_BINS)]
emax_bin_colors  = [REGIME_COLORS[r] for r in emax_bin_regimes]

emax_in_range = (emax_l_samples >= emax_bin_edges[0]) & (emax_l_samples <= emax_bin_edges[-1])
emax_bin_idx  = np.digitize(emax_l_samples, emax_bin_edges[1:-1])

regime_handles = [
    mpatches.Patch(facecolor=REGIME_COLORS["both_escape"],       alpha=0.65, label=f"Both escape (≤ {BLOOD_ESCAPE_THRESH:.3f})"),
    mpatches.Patch(facecolor=REGIME_COLORS["reservoir_escapes"], alpha=0.65, label=f"$R_{{res}}$ escapes ({BLOOD_ESCAPE_THRESH:.3f}–{RES_ESCAPE_THRESH:.3f})"),
    mpatches.Patch(facecolor=REGIME_COLORS["suppressed"],        alpha=0.65, label=f"Suppressed (> {RES_ESCAPE_THRESH:.3f})"),
]

S_b_hist   = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
R_b_hist   = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
S_res_hist = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
R_res_hist = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)

print(f"Running {NUM_ITERATIONS} iterations (sweep: Emax_l, baseline={EMAX_L_BASE:.3f})...",
      flush=True)

for i in range(NUM_ITERATIONS):
    params_i = {**BASE_PARAMS, "Emax_l": emax_l_samples[i]}

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

print("Simulations complete. Generating figures...", flush=True)


# ---------------------------------------------------------------------------
# FIGURE: bar chart of FINAL resistant-strain levels at representative
# Emax_l values (R_b, R_res only). Uses the same escape bins/thresholds and
# regime coloring as the boxplot figure below.
# ---------------------------------------------------------------------------
bar_final_R_b, bar_final_R_res = [], []

print("\nRunning single simulations at representative Emax_l levels...", flush=True)
for emax in EMAX_L_CENTERS:
    params_i = {**BASE_PARAMS, "Emax_l": emax}
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

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle(r"Final Resistant-Strain Levels at Representative $Emax_L$ Values (linezolid)",
              fontsize=12, fontweight="bold")

for ax, (values, ylabel, ylim) in zip(axes1.flat, bar_panels):
    disp_values = [FLOOR if v <= 0 else v for v in values]
    ax.bar(EMAX_L_LABELS, disp_values, color=emax_bin_colors, edgecolor="black", linewidth=0.6)

    ax.axhline(LOD, color="black", ls=":", lw=0.9, alpha=0.6, label="LOD (10 CFU/mL)")
    ax.set_yscale("log")
    ax.set_ylim(LOD * 0.3, ylim)
    ax.set_xlabel(r"$Emax_L$ (h$^{-1}$)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontsize=10, pad=12)
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.35)
    ax.legend(handles=regime_handles + [ax.get_legend_handles_labels()[0][0]],
              labels=[h.get_label() for h in regime_handles] + ["LOD (10 CFU/mL)"],
              fontsize=6.5, loc="upper right")

plt.tight_layout()
plt.savefig("mc_emax_lzd_bar_levels.png", dpi=300, bbox_inches="tight")
print("Saved: mc_emax_lzd_bar_levels.png")


# ---------------------------------------------------------------------------
# FIGURE: boxplots of FINAL (end-of-simulation) resistant bacterial counts
# (R_b, R_res only), built from the full 400-run Monte Carlo sweep above and
# grouped into the escape bins/thresholds defined earlier.
# ---------------------------------------------------------------------------
final_R_b   = np.nan_to_num(R_b_hist[:, -1],   nan=0.0)
final_R_res = np.nan_to_num(R_res_hist[:, -1], nan=0.0)

final_box_panels = [
    (final_R_b,   "Final $R_b$ (CFU/mL)",     1e6),
    (final_R_res, "Final $R_{res}$ (CFU/mL)", 8e7),
]

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle(r"Final Resistant Bacterial Counts vs $Emax_L$ — Monte Carlo Sweep (linezolid)",
              fontsize=12, fontweight="bold")

for ax, (final_vals, ylabel, ylim) in zip(axes2.flat, final_box_panels):
    box_data = []
    for b in range(N_EMAX_L_BINS):
        mask = emax_in_range & (emax_bin_idx == b)
        vals = final_vals[mask]
        box_data.append(np.where(vals <= 0, FLOOR, vals))

    bp = ax.boxplot(box_data, positions=range(N_EMAX_L_BINS), widths=0.6,
                     patch_artist=True, showfliers=True,
                     medianprops=dict(color="black", lw=1.5))
    for patch, color in zip(bp["boxes"], emax_bin_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.axhline(LOD, color="black", ls=":", lw=0.9, alpha=0.6, label="LOD (10 CFU/mL)")
    ax.set_xticks(range(N_EMAX_L_BINS))
    ax.set_xticklabels(EMAX_L_LABELS)
    ax.set_yscale("log")
    ax.set_ylim(LOD * 0.3, ylim)
    ax.set_xlabel(r"$Emax_L$ (h$^{-1}$)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontsize=10, pad=12)
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.35)
    ax.legend(handles=regime_handles + [ax.get_legend_handles_labels()[0][0]],
              labels=[h.get_label() for h in regime_handles] + ["LOD (10 CFU/mL)"],
              fontsize=6.5, loc="upper right")

plt.tight_layout()
plt.savefig("mc_emax_lzd_final_boxplot.png", dpi=300, bbox_inches="tight")
print("Saved: mc_emax_lzd_final_boxplot.png")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\n--- Sweep: Emax_l varied (baseline={EMAX_L_BASE:.3f}) ---")
print(f"  Emax_l sampled  median={np.median(emax_l_samples):.3f}  "
      f"[5th={np.percentile(emax_l_samples, 5):.3f}, "
      f"95th={np.percentile(emax_l_samples, 95):.3f}]")
if np.any(final_R_b > LOD):
    print(f"  Final R_b  median={np.median(final_R_b[final_R_b > LOD]):.1e}  "
          f"[5th={np.percentile(final_R_b[final_R_b > LOD], 5):.1e}, "
          f"95th={np.percentile(final_R_b[final_R_b > LOD], 95):.1e}] CFU/mL "
          f"(n={np.sum(final_R_b > LOD)}/{NUM_ITERATIONS} escaped)")
else:
    print("  Final R_b: no samples above LOD")

print("\nDone.")
