"""Monte Carlo simulation of host immune-clearance variability.

``k_immune`` is sampled from the mean-corrected log-normal distribution in
``parameters.SAMPLING``. Its sampling mean equals the fixed model default;
all other model and protocol parameters use the shared baseline configuration.

Relapse means resistant blood clears by linezolid completion and becomes
detectable during follow-up. Reservoir persistence is reported separately.
"""

import importlib.util
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from solver import integrate_model

from parameters import (
    BASE_PARAMS, LOD, NUM_ITERATIONS, RNG_SEEDS, SAMPLING, TOTAL_H,
    VANCO_START, Y0, bacteremia_outcome_masks, canonical_time_grid, save_figure,
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
#   mc_immune_response.png        -- MONTE CARLO. 2 panels: Blood (S_b + R_b)
#       and Reservoir (S_res + R_res) over 1000 sampled patients, median +
#       5th-95th percentile band.
#   mc_immune_response_switch.png -- MIXED, one source per panel:
#       top    = DETERMINISTIC  (101-point fixed-parameter sweep + Brent switch)
#       bottom = MONTE CARLO    (1000 sampled patients, coloured by outcome)

# ── Load model module ─────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("model_mod", "model_Bacteremia.py")
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)

# ── Simulation settings ───────────────────────────────────────────────────────
NUM_ITER    = NUM_ITERATIONS
t_eval      = canonical_time_grid()
t_days      = t_eval / 24.0

pk       = model_mod.PharmacokineticModel()
van_func = pk.concentration_function("vancomycin", TOTAL_H, VANCO_START)
lzd_func = pk.concentration_function("linezolid",  TOTAL_H, VANCO_START + pk.van_duration)
# Model parameters come from parameters.py (single source of truth).
params = dict(BASE_PARAMS)

# Fixed initial conditions (blood starts sterile; reservoir seeded at 100 CFU/mL each)

# ── Log-normal sampling of k_immune ──────────────────────────────────────────
# CV = sqrt(exp(σ²) - 1)  →  σ = sqrt(ln(1 + CV²))
# μ  = ln(mean) - σ²/2   ensures E[k_immune] = K_IMMUNE_MEAN
K_IMMUNE_MEAN, CV = SAMPLING["k_immune"]

sigma_ln = np.sqrt(np.log(1.0 + CV**2))
mu_ln    = np.log(K_IMMUNE_MEAN) - sigma_ln**2 / 2.0

rng = np.random.default_rng(RNG_SEEDS["MC_immune_response"])
k_immune_samples = rng.lognormal(mu_ln, sigma_ln, NUM_ITER)

print(f"k_immune  mean={k_immune_samples.mean():.4f}  "
      f"std={k_immune_samples.std():.4f}  "
      f"CV={k_immune_samples.std()/k_immune_samples.mean():.3f}  "
      f"[5th–95th]: [{np.percentile(k_immune_samples,5):.4f}, "
      f"{np.percentile(k_immune_samples,95):.4f}]")

# ---------------------------------------------------------------------------
# Continuous deterministic switch curve + precise escape threshold.
# Mirrors the companion analyses' fine sweep and switch localization.
# If a crossing exists, lower k_immune is the persistence side.
# ---------------------------------------------------------------------------
def _final_counts(k_immune):
    """One deterministic run at a fixed k_immune; returns (final R_b, final R_res)."""
    immune_i = model_mod.ImmuneResponse(k_immune=k_immune)
    sol = integrate_model(
        model_mod.dual_reservoir_model, Y0, t_eval,
        args=(params, van_func, lzd_func, immune_i),
    )
    return sol[-1, 1], sol[-1, 3]


SWEEP_MIN, SWEEP_MAX = 0.02, 0.65   # covers the configured sampling distribution
N_SWEEP = 101

print(f"\nRunning deterministic switch-curve sweep across [{SWEEP_MIN}, {SWEEP_MAX}] "
      f"({N_SWEEP} points)...", flush=True)
sweep_k_immune = np.linspace(SWEEP_MIN, SWEEP_MAX, N_SWEEP)
sweep_R_b      = np.zeros(N_SWEEP)
sweep_R_res    = np.zeros(N_SWEEP)
for i, kim in enumerate(sweep_k_immune):
    sweep_R_b[i], sweep_R_res[i] = _final_counts(kim)


escapes_on_grid = sweep_R_res >= LOD
crossings = np.flatnonzero(escapes_on_grid[:-1] & ~escapes_on_grid[1:])
if len(crossings) == 1:
    bracket_index = crossings[0]

    def _reservoir_objective(k_immune):
        return _final_counts(k_immune)[1] - LOD

    ESCAPE_THRESH = brentq(
        _reservoir_objective,
        sweep_k_immune[bracket_index], sweep_k_immune[bracket_index + 1],
        xtol=1e-6,
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

# ── Storage ───────────────────────────────────────────────────────────────────
S_b_hist   = np.full((NUM_ITER, len(t_eval)), np.nan)
R_b_hist   = np.full((NUM_ITER, len(t_eval)), np.nan)
S_res_hist = np.full((NUM_ITER, len(t_eval)), np.nan)
R_res_hist = np.full((NUM_ITER, len(t_eval)), np.nan)

# ── Monte Carlo loop ──────────────────────────────────────────────────────────
for i, k_im in enumerate(k_immune_samples):
    immune_i = model_mod.ImmuneResponse(k_immune=k_im)
    sol = integrate_model(
        model_mod.dual_reservoir_model, Y0, t_eval,
        args=(params, van_func, lzd_func, immune_i),
    )

    # Clip negatives, then mask below LOD with NaN (log-scale friendly)
    s_b, r_b, s_r, r_r = [np.clip(sol[:, j], 0, None) for j in range(4)]
    S_b_hist[i, :]   = np.where(s_b < LOD, np.nan, s_b)
    R_b_hist[i, :]   = np.where(r_b < LOD, np.nan, r_b)
    S_res_hist[i, :] = np.where(s_r < LOD, np.nan, s_r)
    R_res_hist[i, :] = np.where(r_r < LOD, np.nan, r_r)

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{NUM_ITER} done")

# ── Treatment-window boundaries ───────────────────────────────────────────────
vanco_start_d = VANCO_START / 24.0
lzd_start_d   = (VANCO_START + pk.van_duration) / 24.0
lzd_end_d     = (VANCO_START + pk.van_duration + pk.lzd_duration) / 24.0

# ── Panel helper ──────────────────────────────────────────────────────────────
N_TRACES = 40   # individual trajectories to draw per panel


def _draw_series(ax, history, light_color, median_color, name):
    """
    Draw one population (S or R) on a panel: individual traces, 5th-95th band
    and median. Statistics are over ALL sampled patients; sub-LOD points are
    NaN, so each time point summarises the patients detectable at that time.
    """
    # Individual trajectories (random subsample)
    rng_idx = np.random.default_rng(RNG_SEEDS["MC_immune_response_traces"]).choice(
        NUM_ITER, N_TRACES, replace=False
    )
    for i in rng_idx:
        ax.plot(t_days, history[i, :], color=light_color, alpha=0.12, lw=0.7)

    # Percentile band + median (suppress all-NaN warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p50  = np.nanpercentile(history, 50, axis=0)
        p05  = np.nanpercentile(history,  5, axis=0)
        p95  = np.nanpercentile(history, 95, axis=0)

    ax.fill_between(t_days, p05, p95, color=light_color, alpha=0.18)
    ax.plot(t_days, p50, color=median_color, lw=2, label=f"{name} median")


# ── 2-panel figure: Blood (S_b + R_b), Reservoir (S_res + R_res) ─────────────
# Same layout, colours and window shading as mc_ec50_lzd_sweep_kinetics.png and
# mc_emax_lzd_sweep_kinetics.png.
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# fig.suptitle(
#     f"Monte Carlo — Immune Response Variability\n"
#     rf"$k_{{immune}}$ ~ LogNormal  (mean = {K_IMMUNE_MEAN}, CV = {CV}),  "
#     f"n = {NUM_ITER} simulations",
#     fontsize=21, fontweight="bold",
# )

panels = [
    (axes[0], "Blood", 1e6, [
        (S_b_hist, "royalblue",  "blue", "$S_b$"),
        (R_b_hist, "lightcoral", "red",  "$R_b$"),
    ]),
    (axes[1], "Reservoir", 2e7, [
        (S_res_hist, "royalblue",  "blue", "$S_{res}$"),
        (R_res_hist, "lightcoral", "red",  "$R_{res}$"),
    ]),
]

for ax, title, y_max, series in panels:
    ax.axvspan(vanco_start_d, lzd_start_d, color="gray", alpha=0.12, label="Vancomycin")
    ax.axvspan(lzd_start_d,  lzd_end_d,    color="gold", alpha=0.12, label="Linezolid")
    ax.axhline(LOD, color="black", ls=":", lw=0.9, alpha=0.6,
               label=f"LOD ({int(LOD)} CFU/mL)")

    for history, lc, mc, name in series:
        _draw_series(ax, history, lc, mc, name)

    ax.set_yscale("log")
    ax.set_ylim(LOD * 0.5, y_max)
    ax.set_xlim(0, t_days[-1])
    ax.set_xlabel("Time (days)", fontsize=20)
    ax.set_ylabel("CFU/mL", fontsize=20)
    ax.set_title(title, fontsize=21)
    ax.tick_params(labelsize=18)
    ax.grid(True, which="both", ls=":", alpha=0.35)

# Blood legend in the empty upper left; reservoir legend lower right, matching
# the EC50 and Emax kinetics figures.
axes[0].legend(loc="upper left", fontsize=13, ncol=2, framealpha=0.9)
axes[1].legend(loc="lower right", fontsize=13, ncol=1, framealpha=0.9)
fig.tight_layout()

out_file = os.path.join(FIGURES_DIR, "mc_immune_response.png")
save_figure(fig, out_file, __file__)
print(f"\nSaved: {out_file}")

# ── Outcome classification from the resistant blood trajectory: relapse is
#    R_b below the LOD at linezolid completion and detectable in follow-up;
#    persistent failure is R_b still detectable at completion ───────────────
final_R_b   = np.nan_to_num(R_b_hist[:, -1],   nan=0.0)
final_R_res = np.nan_to_num(R_res_hist[:, -1], nan=0.0)
relapsed, persistent_bacteremia, resolved = bacteremia_outcome_masks(
    R_b_hist, t_eval
)

print(f"\nPost-linezolid bacteremia relapse: "
      f"{relapsed.sum()}/{NUM_ITER} = {relapsed.mean()*100:.1f}%")
print(f"Persistent treatment failure: {persistent_bacteremia.sum()}/{NUM_ITER} = "
      f"{persistent_bacteremia.mean()*100:.1f}%")
print(f"Resolution: {resolved.sum()}/{NUM_ITER} = {resolved.mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# FIGURE: switch curve (continuous, deterministic) + sampled patient
# population relative to the escape threshold. Mirrors MC_ec50_lzd.py's /
# MC_emax_lzd.py's combined switch/population figure, direction-flipped for
# k_immune (same direction as Emax_l): escape is the LOW side, suppression
# the HIGH side.
# ---------------------------------------------------------------------------
FLOOR = LOD * 0.5   # display floor for values that fell below the LOD (i.e. 0)

disp_sweep_R_b   = np.where(sweep_R_b   <= LOD, FLOOR, sweep_R_b)
disp_sweep_R_res = np.where(sweep_R_res <= LOD, FLOOR, sweep_R_res)

escaped_mask       = relapsed
reservoir_persistent = final_R_res >= LOD
frac_escape        = escaped_mask.mean()
escaped_k_immune   = k_immune_samples[escaped_mask]
suppressed_k_immune = k_immune_samples[~escaped_mask]

# Outcome split -- quoted in mrsa_endocarditis_ieee.tex and in
# threshold_changes_summary.txt, so print it rather than leaving it readable
# only off the histogram legend below. The endpoint is bacteremia relapse,
# matching the definition used in the other Monte Carlo analyses.
n_escape = int(escaped_mask.sum())
n_total  = int(escaped_mask.size)
if np.isnan(ESCAPE_THRESH):
    print(f"  Reservoir threshold none in [{SWEEP_MIN}, {SWEEP_MAX}] h^-1")
else:
    print(f"  Reservoir threshold k_immune = {ESCAPE_THRESH:.3f} h^-1 (persistence below)")
print(f"  Bacteremia relapse {frac_escape * 100:5.1f}%  ({n_escape}/{n_total})")
print(f"  Persistent failure {persistent_bacteremia.mean() * 100:5.1f}%  "
      f"({persistent_bacteremia.sum()}/{n_total})")
print(f"  Resolution         {resolved.mean() * 100:5.1f}%  "
      f"({resolved.sum()}/{n_total})")

# Displayed switch window.  The deterministic sweep spans [0.02, 0.65] h^-1,
# but plotting all of it pins the transition against the left spine.  The
# window is derived from the switch (0 to twice the switch, rounded up to
# 0.01 h^-1) so the transition stays centred after any recalibration; the
# sampled patients beyond it are the flat, resolved right tail.
PLOT_KIM_MIN = 0.0
PLOT_KIM_MAX = (np.ceil(2 * ESCAPE_THRESH / 0.01) * 0.01
                if not np.isnan(ESCAPE_THRESH) else SWEEP_MAX)

fig2, (ax_top, ax_hist) = plt.subplots(
    2, 1, figsize=(11, 9.5), sharex=True,
    gridspec_kw={"height_ratios": [2.6, 1.4], "hspace": 0.08},
)

# --- Top: continuous response curve ---
if np.isnan(ESCAPE_THRESH):
    ax_top.axvspan(PLOT_KIM_MIN, SWEEP_MAX, color=REGIME_COLORS["suppressed"],
                   alpha=0.10, label="R_res below LOD across swept range")
else:
    # Shade from the window edge, not SWEEP_MIN: below the threshold the
    # reservoir persists regardless, so leaving 0--SWEEP_MIN blank would only
    # add dead space on the left.
    ax_top.axvspan(PLOT_KIM_MIN, ESCAPE_THRESH, color=REGIME_COLORS["escape"], alpha=0.10)
    ax_top.axvspan(ESCAPE_THRESH, SWEEP_MAX, color=REGIME_COLORS["suppressed"], alpha=0.10)
ax_top.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")
if not np.isnan(ESCAPE_THRESH):
    ax_top.axvline(ESCAPE_THRESH, color="black", ls="--", lw=1.4,
                   label=fr"Switch ($k_{{immune}}$ = {ESCAPE_THRESH:.2g})")

ax_top.plot(sweep_k_immune, disp_sweep_R_res, color="indianred", lw=2.2,
            label="Final $R_{res}$ (reservoir)")
ax_top.plot(sweep_k_immune, disp_sweep_R_b, color="darkred", lw=2.2,
            label="Final $R_b$ (blood)")

ax_top.set_yscale("log")
ax_top.set_ylim(FLOOR * 0.8, max(sweep_R_b.max(), sweep_R_res.max()) * 3)
ax_top.set_ylabel("Final resistant count (CFU/mL)", fontsize=20)
ax_top.grid(True, which="both", ls=":", alpha=0.35)
ax_top.legend(loc="upper right", fontsize=16, framealpha=0.9)

# --- Bottom: sampled patient population by actual bacteremia outcome ---
bin_edges = np.linspace(PLOT_KIM_MIN, PLOT_KIM_MAX, 26)
ax_hist.hist(
    [k_immune_samples[resolved], k_immune_samples[relapsed],
     k_immune_samples[persistent_bacteremia]],
    bins=bin_edges,
    stacked=True,
    color=[OUTCOME_COLORS["resolved"], OUTCOME_COLORS["relapse"],
           OUTCOME_COLORS["persistent"]],
    alpha=0.80,
    edgecolor="white",
    linewidth=0.4,
    label=[
        f"Clinical resolution ({resolved.mean() * 100:.1f}%)",
        f"Bacteremia relapse ({relapsed.mean() * 100:.1f}%)",
        f"Persistent bacteremia ({persistent_bacteremia.mean() * 100:.1f}%)",
    ],
)
if not np.isnan(ESCAPE_THRESH):
    ax_hist.axvline(ESCAPE_THRESH, color="black", ls="--", lw=1.4)

ax_hist.set_xlim(PLOT_KIM_MIN, PLOT_KIM_MAX)
ax_hist.set_xlabel(r"$k_{immune}$ (h$^{-1}$)", fontsize=20)
ax_hist.set_ylabel("Cases\n(n)", fontsize=20)
ax_hist.grid(True, axis="y", ls=":", alpha=0.35)
# Headroom above the tallest bar so the legend sits over empty space
# instead of covering the resolved-patient bars.
ax_hist.set_ylim(0, ax_hist.get_ylim()[1] * 1.9)
ax_hist.legend(loc="upper right", fontsize=15, framealpha=0.9)
for _ax in (ax_top, ax_hist):
    _ax.tick_params(labelsize=18)

out_file = os.path.join(FIGURES_DIR, "mc_immune_response_switch.png")
save_figure(fig2, out_file, __file__)
print(f"Saved: {out_file}")
