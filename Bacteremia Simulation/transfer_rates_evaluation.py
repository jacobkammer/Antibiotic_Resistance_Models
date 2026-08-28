# =============================================================================
# Deterministic Log-Spaced Grid Sweep: Transfer Rates Explored Separately
# Two independent grid sweeps are run:
#   Sweep 1 — vary f_r_b (reservoir→blood), hold f_b_r at baseline
#   Sweep 2 — vary f_b_r (blood→reservoir), hold f_r_b at baseline
# Each sweep steps its transfer rate across a fixed log-spaced grid (same
# absolute range for both rates) while every other parameter (including the
# other transfer rate) stays fixed at baseline, isolating the effect of each
# rate on bacterial kinetics across a wide dynamic range.
#
# A third section then re-integrates a handful of rates inside f_r_b's relapse
# plateau on a finer time grid, to resolve the within-course dynamics the grid
# sweeps are too coarse to show: how far R_res is suppressed during linezolid,
# and how long after withdrawal R_b takes to cross back above the LOD.
#
# Nothing here is Monte Carlo -- every rate is a fixed, deterministic grid
# point (renamed from MC_transfer_rates_equilibrium.py for that reason).
# =============================================================================
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq

from parameters import BASE_PARAMS, LOD, TOTAL_H, VANCO_START, Y0, save_figure

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

# FIGURES PRODUCED BY THIS SCRIPT -- ALL DETERMINISTIC (no sampling anywhere)
#   frb_sweep_kinetics.png / fbr_sweep_kinetics.png -- trajectories, one per grid point.
#   frb_sweep_regimes.png  -- regime summary, RESISTANT COMPARTMENTS ONLY.
#   fbr_sweep_regimes.png  -- regime summary, retains the sensitive-strain curves.
#   transfer_rate_sweep_grid.png -- the log-spaced grid itself.
#   Every point is ONE fixed-parameter run over a chosen transfer rate.

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
RATE_MIN       = 1e-7       # lower bound of the log-spaced grid (shared by both rates)
RATE_MAX       = 1e0        # upper bound of the log-spaced grid (shared by both rates)
COARSE_POINTS  = 40         # coarse coverage of the full range, for context
ZOOM_MIN       = 1e-3       # transition band identified in the first grid sweep
ZOOM_MAX       = 1e-2
ZOOM_POINTS    = 160        # dense coverage within the zoom band

# Second dense band, covering the reservoir-collapse regime. Above roughly
# 3e-2 h^-1 the reservoir is drained into blood faster than it refills and
# peak S_res falls from ~6.7e3 to its 1e2 CFU/mL initial value -- the same
# one-directional-transfer effect that produces the clearance threshold. The
# coarse pass alone puts only ~6 points across those last 1.5 decades, too
# few to resolve the collapse, so it gets its own band rather than diluting
# the clearance band above (whose 160 points/decade resolve the sharp
# relapse-to-clearance switch).
COLLAPSE_MIN    = 1e-2
COLLAPSE_MAX    = 1e0
COLLAPSE_POINTS = 120       # 60 points/decade across the collapse


def build_grid():
    """Build a log-spaced RATE_GRID used by run_transfer_rate_sweep.

    1. Generate a coarse log-spaced grid across the full range.
    2. Remove coarse points inside either dense band.
    3. Add dense log-spaced sets inside the clearance band (ZOOM_MIN-ZOOM_MAX)
       and the reservoir-collapse band (COLLAPSE_MIN-COLLAPSE_MAX).
    4. Merge, sort, and deduplicate the points before returning.
    """
    coarse = np.logspace(np.log10(RATE_MIN), np.log10(RATE_MAX), COARSE_POINTS)
    in_zoom     = (coarse >= ZOOM_MIN)     & (coarse <= ZOOM_MAX)
    in_collapse = (coarse >= COLLAPSE_MIN) & (coarse <= COLLAPSE_MAX)
    coarse = coarse[~(in_zoom | in_collapse)]
    zoom     = np.logspace(np.log10(ZOOM_MIN), np.log10(ZOOM_MAX), ZOOM_POINTS)
    collapse = np.logspace(np.log10(COLLAPSE_MIN), np.log10(COLLAPSE_MAX), COLLAPSE_POINTS)
    return np.unique(np.concatenate([coarse, zoom, collapse])) #remove duplicates and sorts the combined array


RATE_GRID  = build_grid()
NUM_POINTS = len(RATE_GRID)

total_h     = TOTAL_H
vanco_start = VANCO_START
t_eval       = np.linspace(0, total_h, 1150)
t_days       = t_eval / 24.0

pk           = model_mod.PharmacokineticModel()
immune_model = model_mod.ImmuneResponse()

van_func = pk.concentration_function("vancomycin", total_h, vanco_start)
lzd_func = pk.concentration_function("linezolid",  total_h, vanco_start + pk.van_duration)

vanco_start_days = vanco_start / 24.0
lzd_start_days   = (vanco_start + pk.van_duration) / 24.0
lzd_end_days     = (vanco_start + pk.van_duration + pk.lzd_duration) / 24.0
# Model parameters come from parameters.py (single source of truth).

F_R_B_BASE = 5e-5   # reservoir -> blood baseline (restored to original)
F_B_R_BASE = 1e-5   # blood -> reservoir baseline

# Fixed initial conditions — only the swept rate is varied across iterations


# ---------------------------------------------------------------------------
# Run one grid sweep: step a single transfer rate across a log-spaced grid,
# hold the other transfer rate fixed at baseline. It returns a dictionary
# with the results for each parameter.  steps through the grid and runs the model
# for each parameter value.
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

# Run the transfer rate sweeps twice: once with f_r_b as the swept parameter, once with f_b_r
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
    out_file = os.path.join(FIGURES_DIR, filename)
    save_figure(plt, out_file, __file__)
    print(f"Saved: {out_file}")
    plt.close(fig)


plot_sweep_kinetics(sweep_f_r_b, "frb_sweep_kinetics.png")
plot_sweep_kinetics(sweep_f_b_r, "fbr_sweep_kinetics.png")


# ---------------------------------------------------------------------------
# Precise regime thresholds for the f_r_b sweep (bisection, same convention
# as the other threshold-sweep scripts in this project): a low threshold
# where bacteremia first establishes, and a high threshold where established
# resistant bacteremia stops relapsing and clears instead. f_b_r has no such
# thresholds anywhere in the swept range (confirmed by Sweep 2 above).
# ---------------------------------------------------------------------------
def _final_R_b_at(f_r_b_value):
    params_i = {**BASE_PARAMS, "f_r_b": f_r_b_value, "f_b_r": F_B_R_BASE}
    sol = odeint(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(params_i, van_func, lzd_func, immune_model),
        rtol=1e-8, atol=1e-10, mxstep=5000,
    )
    return sol[-1, 1]


threshold_establish = brentq(lambda r: _final_R_b_at(r) - LOD, 1e-6, 1e-4)
threshold_clear      = brentq(lambda r: _final_R_b_at(r) - LOD, 5e-3, 1e-2)
print(f"f_r_b establishment threshold (final R_b crosses LOD, low side): "
      f"{threshold_establish:.3e} h^-1", flush=True)
print(f"f_r_b clearance threshold (final R_b crosses LOD, high side):    "
      f"{threshold_clear:.3e} h^-1", flush=True)


# ---------------------------------------------------------------------------
# FIGURE: single-panel regime summary per sweep -- peak S_b, peak R_b, and
# final R_b overlaid on one log-scale axis (plus peak S_res as a light
# reference line), with the swept rate's regime shaded when the sweep has
# genuine thresholds to show.
# ---------------------------------------------------------------------------
def plot_regime_switch(sweep, filename, thresholds=None, title_suffix="",
                       show_sensitive=True):
    """Regime summary for one sweep.

    show_sensitive=False drops the sensitive-strain series (peak S_b and the
    peak S_res reference line), leaving only the resistant compartments. Used
    for the f_r_b figure, whose argument is about R_b's establishment and
    clearance thresholds; the f_b_r figure keeps them, since its own section
    discusses peak S_res directly.
    """
    rates      = sweep["rates"]
    log10_rate = np.log10(rates)
    FLOOR = LOD * 0.5   # display floor for sub-LOD values, so cleared/
                         # unestablished regions show as a flat line at the
                         # floor rather than a gap

    peak_S_b   = np.where(sweep["peak_S_b"]   <= LOD, FLOOR, sweep["peak_S_b"])
    peak_R_b   = np.where(sweep["peak_R_b"]   <= LOD, FLOOR, sweep["peak_R_b"])
    final_R_b  = np.where(sweep["final_R_b"]  <= LOD, FLOOR, sweep["final_R_b"])
    peak_S_res = sweep["peak_S_res"]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    if thresholds is not None:
        lo, hi = thresholds
        log_lo, log_hi = np.log10(lo), np.log10(hi)
        ax.axvspan(log10_rate.min(), log_lo, color="gray",           alpha=0.12, label="Unestablished")
        ax.axvspan(log_lo, log_hi,           color="salmon",         alpha=0.15, label="Relapses (plateau)")
        ax.axvspan(log_hi, log10_rate.max(), color="mediumseagreen", alpha=0.12, label="Cleared")
        ax.axvline(log_lo, color="black",      ls="--", lw=1.6, alpha=0.9,
                   label=f"Establishment threshold ({lo:.2e})")
        ax.axvline(log_hi, color="darkorange", ls="--", lw=1.6, alpha=0.9,
                   label=f"Clearance threshold ({hi:.2e})")

    ax.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.6, label=f"LOD ({int(LOD)} CFU/mL)")

    if show_sensitive:
        ax.plot(log10_rate, peak_S_res, color="mediumpurple", lw=1.3, ls=":",
                alpha=0.8, label="Peak $S_{res}$ (reservoir, for reference)")
        ax.plot(log10_rate, peak_S_b,  color="royalblue", lw=2.2, label="Peak $S_b$")
    ax.plot(log10_rate, peak_R_b,  color="firebrick",  lw=2.2, label="Peak $R_b$")
    ax.plot(log10_rate, final_R_b, color="firebrick",  lw=2.2, ls="--", label="Final $R_b$")

    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.8, 1.2e4)
    ax.set_xlim(log10_rate.min(), log10_rate.max())
    ax.set_xlabel(f"$\\log_{{10}}$({RATE_LABELS[sweep['vary_name']]})")
    ax.set_ylabel("CFU/mL")
    ax.set_title(f"Regime summary vs {RATE_LABELS[sweep['vary_name']]}{title_suffix} "
                 f"(log-spaced grid, n={NUM_POINTS})",
                 fontsize=18, fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="best", fontsize=11, framealpha=0.9, ncol=1)

    plt.tight_layout()
    out_file = os.path.join(FIGURES_DIR, filename)
    save_figure(plt, out_file, __file__)
    print(f"Saved: {out_file}")
    plt.close(fig)


plot_regime_switch(sweep_f_r_b, "frb_sweep_regimes.png",
                    thresholds=(threshold_establish, threshold_clear),
                    show_sensitive=False)
plot_regime_switch(sweep_f_b_r, "fbr_sweep_regimes.png",
                    title_suffix=" (no threshold in swept range)")


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
out_file = os.path.join(FIGURES_DIR, "transfer_rate_sweep_grid.png")
save_figure(plt, out_file, __file__)
print(f"Saved: {out_file}")


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


# ---------------------------------------------------------------------------
# Within-course relapse analysis across the f_r_b relapse plateau
#
# The grid sweeps above resolve outcomes (peak and final burden per rate) but
# not the timing of the post-treatment rebound: at 1150 points over 1944 h the
# step is ~1.7 h, too coarse to place the LOD crossing to better than a
# fraction of a day, and the sweeps clip sub-LOD values to zero, which discards
# exactly the suppressed R_b levels this section reports. Both sweeps are
# therefore left untouched and a small number of rates inside the plateau are
# re-integrated here on a finer grid, unclipped.
#
# Supplies the numbers cited in supplement.tex: the suppressed R_res level
# during linezolid, and the relapse timing once it is withdrawn.
# ---------------------------------------------------------------------------
FINE_POINTS   = 4000     # vs 1150 for the sweeps -- resolves the LOD crossing
FINE_RTOL     = 1e-9     # tighter tolerances for fine grid
FINE_ATOL     = 1e-11    # tighter tolerances for fine grid
PROBE_POINTS  = 6        # illustrative rates spanning the plateau (+ baseline)
RANGE_POINTS  = 40       # dense pass, for the reported R_res range

t_fine      = np.linspace(0, total_h, FINE_POINTS)
t_fine_days = t_fine / 24.0

lzd_start_h = vanco_start + pk.van_duration
lzd_end_h   = lzd_start_h + pk.lzd_duration
on_lzd      = (t_fine >= lzd_start_h) & (t_fine <= lzd_end_h)
post_lzd    = t_fine > lzd_end_h
i_lzd_end   = np.argmin(np.abs(t_fine - lzd_end_h))


def run_fine(f_r_b_value):
    """One course at a single f_r_b on the fine grid. Deliberately unclipped:
    sub-LOD R_b is the quantity of interest during the linezolid course."""
    params_i = {**BASE_PARAMS, "f_r_b": f_r_b_value, "f_b_r": F_B_R_BASE}
    return odeint(
        model_mod.dual_reservoir_model,
        Y0, t_fine,
        args=(params_i, van_func, lzd_func, immune_model),
        rtol=FINE_RTOL, atol=FINE_ATOL, mxstep=10000, # increased max steps for fine grid
    )


def relapse_metrics(sol):
    R_b, R_res = sol[:, 1], sol[:, 3]
    # first post-withdrawal crossing back above the LOD
    above = np.where(post_lzd & (R_b > LOD))[0]
    lag = (t_fine_days[above[0]] - lzd_end_h / 24.0) if above.size else np.nan
    return {
        "peak_R_b":      np.max(R_b),
        "min_R_b_lzd":   np.min(R_b[on_lzd]),
        "R_res_lzd_end": R_res[i_lzd_end],
        "R_res_min_lzd": np.min(R_res[on_lzd]),
        "final_R_b":     R_b[-1],
        "rebound_lag_d": lag,
    }


print(f"\n\n=== Within-course relapse across the f_r_b plateau "
      f"({threshold_establish:.3e} .. {threshold_clear:.3e} h^-1) ===")
print(f"Linezolid course: day {lzd_start_h / 24.0:.1f} -> {lzd_end_h / 24.0:.1f}; "
      f"follow-up ends day {total_h / 24.0:.1f}")
print(f"Fine grid: {FINE_POINTS} points over {total_h} h, "
      f"rtol={FINE_RTOL:.0e}, atol={FINE_ATOL:.0e}, sub-LOD values unclipped\n")

# baseline plus points spanning the plateau, held clear of both thresholds
probe_rates = np.sort(np.concatenate((
    [F_R_B_BASE],
    np.logspace(np.log10(threshold_establish * 1.5),
                np.log10(threshold_clear * 0.7), PROBE_POINTS),
)))

header = (f"{'f_r_b':>10} {'peak R_b':>10} {'min R_b LZD':>12} "
          f"{'R_res LZD end':>14} {'final R_b':>10} {'rebound lag (d)':>16}")
print(header)
print("-" * len(header))

for r in probe_rates:
    m = relapse_metrics(run_fine(r))
    tag = " *" if np.isclose(r, F_R_B_BASE) else ""
    lag = f"{m['rebound_lag_d']:.1f}" if np.isfinite(m["rebound_lag_d"]) else "none"
    print(f"{r:10.3e} {m['peak_R_b']:10.2e} {m['min_R_b_lzd']:12.2e} "
          f"{m['R_res_lzd_end']:14.2e} {m['final_R_b']:10.2e} {lag:>16}{tag}")

print("  * model baseline rate")

# Dense pass: the R_res suppressed-plateau range quoted in the supplement.
# Sampled strictly inside the thresholds -- at the boundaries themselves R_res
# is still falling, so the lower figure would be pulled toward the LOD.
range_rates = np.logspace(np.log10(threshold_establish * 1.02),
                          np.log10(threshold_clear * 0.98), RANGE_POINTS)

res_end, res_min, peak_R_b_fine, final_R_b_fine = [], [], [], []
for r in range_rates:
    m = relapse_metrics(run_fine(r))
    res_end.append(m["R_res_lzd_end"])
    res_min.append(m["R_res_min_lzd"])
    peak_R_b_fine.append(m["peak_R_b"])
    final_R_b_fine.append(m["final_R_b"])

res_end = np.array(res_end)
res_min = np.array(res_min)
peak_R_b_fine  = np.array(peak_R_b_fine)
final_R_b_fine = np.array(final_R_b_fine)

print(f"\n--- R_res during linezolid, {RANGE_POINTS} rates strictly inside the plateau "
      f"({range_rates[0]:.3e} .. {range_rates[-1]:.3e} h^-1) ---")
print(f"  R_res at end of course : {res_end.min():.3e} .. {res_end.max():.3e} CFU/mL")
print(f"  R_res min during course: {res_min.min():.3e} .. {res_min.max():.3e} CFU/mL")
print(f"  monotonically decreasing in f_r_b: {bool(np.all(np.diff(res_end) < 0))}")
print(f"    at plateau bottom ({range_rates[0]:.2e}): {res_end[0]:.1f} CFU/mL")
print(f"    at plateau top    ({range_rates[-1]:.2e}): {res_end[-1]:.1f} CFU/mL")
print(f"    fold change across plateau: {res_end[0] / res_end[-1]:.1f}x")
print(f"  R_res stays above the LOD throughout: {bool(np.all(res_min > LOD))} "
      f"(closest approach {res_min.min() / LOD:.1f}x LOD)")
print(f"  final R_b / peak R_b: {(final_R_b_fine / peak_R_b_fine).min():.3f} .. "
      f"{(final_R_b_fine / peak_R_b_fine).max():.3f}  (1.0 => relapse is complete)")

print("\nDone.")
