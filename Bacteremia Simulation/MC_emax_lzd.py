# =============================================================================
# Monte Carlo Simulation: Linezolid Emax (Emax_l)
#
# Samples Emax_l from a mean-corrected log-normal distribution. Its arithmetic
# mean equals BASE_PARAMS["Emax_l"]. Relapse means resistant blood
# clears by linezolid completion and becomes detectable during follow-up.
# =============================================================================
import importlib.util
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from solver import integrate_model

from parameters import (
    BASE_PARAMS, EMAX_L_BOUNDS, LOD, NUM_ITERATIONS, RNG_SEEDS, SAMPLING,
    TOTAL_H, VANCO_START, Y0, bacteremia_outcome_masks, canonical_time_grid,
    save_figure,
)

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

# all figures are written into figures/ alongside the scripts
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# FIGURES PRODUCED BY THIS SCRIPT
#   mc_emax_lzd_sweep_kinetics.png -- MONTE CARLO. 4 panels (S_b, R_b, S_res,
#       R_res); 1000 sampled patients, median + 5th-95th percentile band.
#       Mirrors mc_ec50_lzd_sweep_kinetics.png for direct comparison.
#   mc_emax_lzd_switch.png -- MIXED, one source per panel:
#       top    = DETERMINISTIC  (111-point fixed-parameter sweep + brentq root)
#       bottom = MONTE CARLO    (1000 sampled patients, coloured by outcome)

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
EMAX_L_MEAN, CV = SAMPLING["Emax_l"]
SEED            = RNG_SEEDS["MC_emax_lzd"]

total_h     = TOTAL_H
vanco_start = VANCO_START
t_eval       = canonical_time_grid()
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
# Model parameters and the Emax_l sampling distribution come from
# parameters.py. Only Emax_l is varied across patients.

# Fixed initial conditions — only Emax_l is varied across iterations

# ---------------------------------------------------------------------------
# Run Monte Carlo sweep: vary Emax_l, hold everything else fixed
# ---------------------------------------------------------------------------
# Mean-corrected log-normal parameters: E[Emax_l] = EMAX_L_MEAN.
sigma_ln = np.sqrt(np.log(1.0 + CV**2))
mu_ln = np.log(EMAX_L_MEAN) - 0.5 * sigma_ln**2
rng = np.random.default_rng(SEED)
emax_l_samples = rng.lognormal(mu_ln, sigma_ln, NUM_ITERATIONS)

# ---------------------------------------------------------------------------
# Continuous deterministic switch curve + precise escape threshold.
#
# Mirrors MC_ec50_lzd.py's fine sweep + Brent localization, but
# direction-flipped: for Emax_l, LOW values are the escape zone (a weaker drug
# ceiling can't hold resistant growth in check), the opposite of EC50_L. The
# sweep feeds the combined switch/population figure below.
# ---------------------------------------------------------------------------
def _final_counts(emax_l):
    """One deterministic run at a fixed Emax_l; returns (final R_b, final R_res)."""
    params_i = {**BASE_PARAMS, "Emax_l": emax_l}
    sol = integrate_model(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(params_i, van_func, lzd_func, immune_model),
    )
    return sol[-1, 1], sol[-1, 3]


SWEEP_MIN, SWEEP_MAX = EMAX_L_BOUNDS
N_SWEEP = 111   # step ~0.0136 across EMAX_L_BOUNDS

print(f"\nRunning deterministic switch-curve sweep across [{SWEEP_MIN}, {SWEEP_MAX}] "
      f"({N_SWEEP} points)...", flush=True)
sweep_emax  = np.linspace(SWEEP_MIN, SWEEP_MAX, N_SWEEP)
sweep_R_b   = np.zeros(N_SWEEP)
sweep_R_res = np.zeros(N_SWEEP)
for i, emax in enumerate(sweep_emax):
    sweep_R_b[i], sweep_R_res[i] = _final_counts(emax)


escapes_on_grid = sweep_R_res >= LOD
crossings = np.flatnonzero(escapes_on_grid[:-1] & ~escapes_on_grid[1:])
if len(crossings) == 1:
    bracket_index = crossings[0]

    def _reservoir_objective(emax_l):
        return _final_counts(emax_l)[1] - LOD

    ESCAPE_THRESH = brentq(
        _reservoir_objective,
        sweep_emax[bracket_index], sweep_emax[bracket_index + 1],
        xtol=1e-3,
    )
else:
    ESCAPE_THRESH = float("nan")

if np.isnan(ESCAPE_THRESH):
    print(f"No reservoir-persistence threshold exists in [{SWEEP_MIN}, {SWEEP_MAX}]: R_res "
          f"{'persists' if (sweep_R_res > LOD).all() else 'clears'} across the entire swept range.")
else:
    print(f"Precise reservoir-persistence threshold: {ESCAPE_THRESH:.3f} h^-1")

REGIME_COLORS = {"suppressed": "#4c72b0", "escape": "#c44e52"}  # blue / red
OUTCOME_COLORS = {
    "resolved": "#4c72b0",    # blue
    "relapse": "#dd8452",     # orange
    "persistent": "#c44e52",  # red
}

S_b_hist   = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
R_b_hist   = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
S_res_hist = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
R_res_hist = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)

print(f"Running {NUM_ITERATIONS} iterations (sweep: Emax_l, mean={EMAX_L_MEAN:.3f})...",
      flush=True)

for i in range(NUM_ITERATIONS):
    params_i = {**BASE_PARAMS, "Emax_l": emax_l_samples[i]}

    sol = integrate_model(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(params_i, van_func, lzd_func, immune_model),
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
# Outcome classification, computed here because the kinetics figure below needs
# it. Relapse (escaped_mask) is resistant blood below the LOD at linezolid
# completion followed by detectable R_b in follow-up; reservoir persistence
# (final R_res >= LOD) is tracked separately.
# ---------------------------------------------------------------------------
final_R_b    = np.nan_to_num(R_b_hist[:, -1], nan=0.0)
final_R_res  = np.nan_to_num(R_res_hist[:, -1], nan=0.0)
escaped_mask, persistent_bacteremia, resolved_mask = bacteremia_outcome_masks(
    R_b_hist, t_eval
)
reservoir_persistent = final_R_res >= LOD



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
    ax.set_xlim(0, t_days[-1])   # shared time axis, so all four panels align
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=17)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize=13, ncol=2)


# ---------------------------------------------------------------------------
# FIGURE: 4-panel trajectory kinetics -- S_b, R_b, S_res, R_res.
# Mirrors mc_ec50_lzd_sweep_kinetics.png exactly (same layout, colours, band
# definition and sub-LOD handling), but over the Emax_l sample rather than
# the EC50_L one, so the two are directly comparable side by side.
# ---------------------------------------------------------------------------
fig0, axes0 = plt.subplots(2, 2, figsize=(14, 10))
#fig0.suptitle(r"Bacterial Kinetics — Monte Carlo sweep of $E_{max,L}$ (linezolid)",
              #fontsize=19, fontweight="bold")

panels = [
    (axes0[0, 0], S_b_hist,   "royalblue",  "blue", "Sensitive Blood ($S_b$)",         1e6),
    (axes0[0, 1], R_b_hist,   "lightcoral", "red",  "Resistant Blood ($R_b$)",         1e6),
    (axes0[1, 0], S_res_hist, "royalblue",  "blue", "Sensitive Reservoir ($S_{res}$)", 2e7),
    (axes0[1, 1], R_res_hist, "lightcoral", "red",  "Resistant Reservoir ($R_{res}$)", 2e7),
]

for ax, hist, lc, mc, title, ylim in panels:
    annotate_windows(ax)

    # sub-LOD values are stored as 0.0 by the MC loop above; map them to NaN so
    # the log axis shows gaps rather than a floor line
    for j in range(min(20, NUM_ITERATIONS)):
        tr = np.where(hist[j] == 0, np.nan, hist[j])
        ax.plot(t_days, tr, color=lc, alpha=0.12, lw=0.8)

    # Summarise over the RELAPSED patients (escaped_mask) only. Taking the median over all 1000
    # rows makes the post-treatment relapse vanish from the plot: sub-LOD values
    # are stored as 0, relapse is a minority outcome, so the all-patient median is
    # 0 (-> NaN) exactly where the relapse happens. Restricting to relapsed rows
    # and using nanmedian keeps the relapse visible and states its n in the
    # legend. Sub-LOD points within that subset become NaN, so a gap here means
    # "not detectable in any relapsed patient", not "no data".
    esc = np.where(hist[escaped_mask] == 0, np.nan, hist[escaped_mask])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # all-NaN slices are expected on drug
        med  = np.nanmedian(esc, axis=0)
        lo5  = np.nanpercentile(esc,  5, axis=0)
        hi95 = np.nanpercentile(esc, 95, axis=0)

    ax.fill_between(t_days, lo5, hi95, color=lc, alpha=0.18)
    ax.plot(t_days, med, color=mc, lw=2,
            label=f"Median of escaped (n={int(escaped_mask.sum())})")

    style_kinetic(ax, title, "CFU/mL", ylim)

fig0.tight_layout()
out_file = os.path.join(FIGURES_DIR, "mc_emax_lzd_sweep_kinetics.png")
save_figure(fig0, out_file, __file__)
print(f"Saved: {out_file}")


# ---------------------------------------------------------------------------
# FIGURE: switch curve (continuous, deterministic) + sampled patient
# population relative to the escape threshold. Mirrors MC_ec50_lzd.py's
# combined switch/population figure, direction-flipped for Emax_l: escape is
# the LOW side, suppression the HIGH side.
# ---------------------------------------------------------------------------
# final_R_res / escaped_mask are defined earlier (the kinetics figure needs them)

FLOOR = LOD * 0.5   # display floor for values that fell below the LOD (i.e. 0)

disp_sweep_R_b   = np.where(sweep_R_b   <= LOD, FLOOR, sweep_R_b)
disp_sweep_R_res = np.where(sweep_R_res <= LOD, FLOOR, sweep_R_res)

frac_relapse    = escaped_mask.mean()
resolved_emax   = emax_l_samples[resolved_mask]
relapse_emax    = emax_l_samples[escaped_mask]
persistent_emax = emax_l_samples[persistent_bacteremia]

fig1, (ax_top, ax_hist) = plt.subplots(
    2, 1, figsize=(11, 8.5), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08},
)
#fig1.suptitle(r"Resistant Bacteremia Outcomes vs $Emax_L$ (linezolid)",
              #fontsize=19, fontweight="bold")

# --- Top: continuous switch curve ---
ax_top.axvspan(SWEEP_MIN, ESCAPE_THRESH, color=REGIME_COLORS["escape"],     alpha=0.10)
ax_top.axvspan(ESCAPE_THRESH, SWEEP_MAX, color=REGIME_COLORS["suppressed"], alpha=0.10)
ax_top.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")
ax_top.axvline(ESCAPE_THRESH, color="black", ls="--", lw=1.4,
               label=fr"Switch ($Emax_L$ = {ESCAPE_THRESH:.3g})")

ax_top.plot(sweep_emax, disp_sweep_R_res, color="indianred", lw=2.2,
            label="Final $R_{res}$ (reservoir)")
ax_top.plot(sweep_emax, disp_sweep_R_b, color="darkred", lw=2.2,
            label="Final $R_b$ (blood)")

ax_top.set_yscale("log")
ax_top.set_ylim(FLOOR * 0.8, max(sweep_R_b.max(), sweep_R_res.max()) * 3)
ax_top.set_ylabel("Final resistant count (CFU/mL)")
ax_top.grid(True, which="both", ls=":", alpha=0.35)
ax_top.legend(loc="upper right", fontsize=13, framealpha=0.9)

# --- Bottom: where the sampled patient population falls, by actual outcome ---
bin_edges = np.linspace(SWEEP_MIN, SWEEP_MAX, 31)
ax_hist.hist(
    [resolved_emax, relapse_emax, persistent_emax],
    bins=bin_edges,
    stacked=True,
    color=[OUTCOME_COLORS["resolved"], OUTCOME_COLORS["relapse"],
           OUTCOME_COLORS["persistent"]],
    alpha=0.80,
    edgecolor="white",
    linewidth=0.4,
    label=[
        f"Clinical resolution ({resolved_mask.mean() * 100:.1f}%)",
        f"Bacteremia relapse ({frac_relapse * 100:.1f}%)",
        f"Persistent bacteremia ({persistent_bacteremia.mean() * 100:.1f}%)",
    ],
)
ax_hist.axvline(ESCAPE_THRESH, color="black", ls="--", lw=1.4)

ax_hist.set_xlim(SWEEP_MIN, SWEEP_MAX)
ax_hist.set_xlabel(r"$Emax_L$ (h$^{-1}$)")
ax_hist.set_ylabel("Cases\n(n)")
ax_hist.grid(True, axis="y", ls=":", alpha=0.35)
ax_hist.legend(loc="upper right", fontsize=12, framealpha=0.9)

out_file = os.path.join(FIGURES_DIR, "mc_emax_lzd_switch.png")
save_figure(fig1, out_file, __file__)
print(f"Saved: {out_file}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\n--- Sweep: Emax_l varied (sampling mean={EMAX_L_MEAN:.3f}) ---")
print(f"  Emax_l sampled  median={np.median(emax_l_samples):.3f}  "
      f"[5th={np.percentile(emax_l_samples, 5):.3f}, "
      f"95th={np.percentile(emax_l_samples, 95):.3f}]")
# Outcome split -- quoted in manuscript10.01.tex and in
# threshold_changes_summary.txt, so print it rather than leaving it readable
# only off the histogram legend in mc_emax_lzd_switch.png. Note the sense is
# inverted relative to EC50_L: here relapse is Emax_l BELOW the threshold.
n_relapse = int(escaped_mask.sum())
n_total  = int(escaped_mask.size)
print(f"  Reservoir threshold Emax_l = {ESCAPE_THRESH:.3f} h^-1 (persistence below)")
print(f"  Bacteremia relapse {frac_relapse * 100:5.1f}%  ({n_relapse}/{n_total})")
print(f"  Persistent failure {persistent_bacteremia.mean() * 100:5.1f}%  "
      f"({persistent_bacteremia.sum()}/{n_total})")
print(f"  Resolution         {resolved_mask.mean() * 100:5.1f}%  "
      f"({resolved_mask.sum()}/{n_total})")
if np.any(final_R_b > LOD):
    print(f"  Final R_b  median={np.median(final_R_b[final_R_b > LOD]):.1e}  "
          f"[5th={np.percentile(final_R_b[final_R_b > LOD], 5):.1e}, "
          f"95th={np.percentile(final_R_b[final_R_b > LOD], 95):.1e}] CFU/mL "
          f"(n={np.sum(final_R_b > LOD)}/{NUM_ITERATIONS} escaped)")
else:
    print("  Final R_b: no samples above LOD")

print("\nDone.")
