"""
Monte Carlo simulation varying k_immune with a log-normal distribution.
k_immune ~ LogNormal(mean=0.243, CV=0.25)
4-panel figure: S_b, R_b, S_res, R_res — each with individual MC trajectories,
5th–95th percentile band, and median.

Mean raised from the model's own default (0.12) to 0.142 so that ~70% of
samples land above the R_b/R_res escape threshold (k_immune = 0.1255 h^-1,
see ImmuneResponse_threshold_sweep.py) and resolve the infection -- at the
model default, only ~31-37% of samples resolved. CV was then widened from
0.20 to 0.25 (dropping resolution to ~64%), so the mean was re-solved to
0.14719 -- the 30th percentile of LogNormal(mean, CV=0.25) sits exactly at
the 0.1255 threshold -- restoring resolution to ~70%. Later raised to 0.172
(see K_IMMUNE_MEAN below). Then rho_res_S was given a 10% growth advantage
over rho_res_R (fitness cost for R in the reservoir, mirroring blood), which
pushed the escape threshold up to ~0.2000 h^-1 -- the mean was re-solved
again to 0.243 to restore a 20-30% relapse rate (see K_IMMUNE_MEAN below).

EC50_L was then consolidated onto a SINGLE project-wide value. Previously
the project carried two: a fixed baseline EC50_L = 1.0 used by every script
that holds EC50_L constant, and a sampling mean EC50_L_BASE = 0.6 used by
MC_ec50_lzd.py. That also contradicted MC_ec50_lzd.py's own docstring claim
that its sampling is "mean-corrected so E[EC50_L] lands exactly on
baseline". Both are now 0.6 mg/L.

Lowering the fixed EC50_L 1.0 -> 0.6 makes linezolid more potent, which on
its own put the baseline BELOW the EC50_L escape threshold: the reservoir
cleared at baseline and the k_immune sweep lost its threshold entirely (no
k_immune in [0, 3] produced relapse -- even zero immune clearance cleared
the infection). rho_S was therefore raised 0.61 -> 0.72 h^-1, with rho_R
rescaled 0.55 -> 0.649 to preserve the ~10% fitness cost, restoring the
reservoir to the escape side. The mechanism is indirect: the reservoir kill
rate is Emax_l_res = Emax_l * (rho_res_S / rho_S), so a HIGHER rho_S
weakens reservoir killing. rho_res_R/rho_res_S were tried first and
rejected -- every value of either that reached baseline escape collapsed
peak S_b from ~4.8e3 to ~2 CFU/mL (R_b crowding S_b out of blood).

Net effect here: the threshold moved ~0.2000 -> 0.203 h^-1 and relapse
24.3% -> 26.0% (260/1000). Without the rho_S compensation this sweep would
have had no threshold at all under the consolidated EC50_L.
"""

import importlib.util
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq

from parameters import BASE_PARAMS, LOD, TOTAL_H, VANCO_START, Y0

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
#   mc_immune_response.png        -- MONTE CARLO. 4-panel kinetics over 1000
#       sampled patients, median + 5th-95th percentile band.
#   mc_immune_response_switch.png -- MIXED, one source per panel:
#       top    = DETERMINISTIC  (101-point fixed-parameter sweep + brentq root)
#       bottom = MONTE CARLO    (1000 sampled patients, coloured by outcome)

# ── Load model module ─────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("model_mod", "model_Bacteremia.py")
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)

# ── Simulation settings ───────────────────────────────────────────────────────
NUM_ITER    = 1000
t_eval      = np.linspace(0, TOTAL_H, 1450)
t_days      = t_eval / 24.0

pk       = model_mod.PharmacokineticModel()
van_func = pk.concentration_function("vancomycin", TOTAL_H, VANCO_START)
lzd_func = pk.concentration_function("linezolid",  TOTAL_H, VANCO_START + pk.van_duration)
# Model parameters come from parameters.py (single source of truth).
params = dict(BASE_PARAMS)

# Fixed initial conditions (blood starts sterile; reservoir seeded at 100 CFU/mL each)

# ── Log-normal sampling of k_immune ──────────────────────────────────────────
# CV = sqrt(exp(σ²) - 1)  →  σ = sqrt(ln(1 + CV²))
# μ  = ln(mean) - σ²/2   ensures E[k_immune] = 0.12
K_IMMUNE_MEAN = 0.243     # re-tuned from 0.172 -- giving rho_res_S a 10% growth advantage
                          # over rho_res_R (see rho_res_S above) makes the reservoir far more
                          # robust to host immunity: the R_b/R_res escape threshold shifted
                          # from 0.1521 h^-1 to ~0.2000 h^-1 (bisected directly). 0.172 now
                          # puts only ~22% of samples above threshold (78% relapse, verified
                          # via 1000-iteration MC). 0.243 puts ~76% of samples above
                          # threshold, targeting ~24% relapse (20-30% target band).
CV            = 0.25

sigma_ln = np.sqrt(np.log(1.0 + CV**2))
mu_ln    = np.log(K_IMMUNE_MEAN) - sigma_ln**2 / 2.0

np.random.seed(42)
k_immune_samples = np.random.lognormal(mu_ln, sigma_ln, NUM_ITER)

print(f"k_immune  mean={k_immune_samples.mean():.4f}  "
      f"std={k_immune_samples.std():.4f}  "
      f"CV={k_immune_samples.std()/k_immune_samples.mean():.3f}  "
      f"[5th–95th]: [{np.percentile(k_immune_samples,5):.4f}, "
      f"{np.percentile(k_immune_samples,95):.4f}]")

# ---------------------------------------------------------------------------
# Continuous deterministic switch curve + precise escape threshold.
# Mirrors MC_ec50_lzd.py / MC_emax_lzd.py's own fine sweep + bisection.
# Direction matches Emax_l: LOW k_immune (weak host immunity) is the escape
# zone, the opposite of EC50_L.
# ---------------------------------------------------------------------------
def _final_counts(k_immune):
    """One deterministic run at a fixed k_immune; returns (final R_b, final R_res)."""
    immune_i = model_mod.ImmuneResponse(k_immune=k_immune)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = odeint(
            model_mod.dual_reservoir_model, Y0, t_eval,
            args=(params, van_func, lzd_func, immune_i),
            rtol=1e-7, atol=1e-9, mxstep=5000,
        )
    return sol[-1, 1], sol[-1, 3]


SWEEP_MIN, SWEEP_MAX = 0.02, 0.65   # widened from 0.22 -- at the new rho_res_S/rho_res_R ratio the
                                     # escape threshold moved to ~0.20 and K_IMMUNE_MEAN=0.243 puts
                                     # the 99th percentile of sampled k_immune near 0.42
N_SWEEP = 101

print(f"\nRunning deterministic switch-curve sweep across [{SWEEP_MIN}, {SWEEP_MAX}] "
      f"({N_SWEEP} points)...", flush=True)
sweep_k_immune = np.linspace(SWEEP_MIN, SWEEP_MAX, N_SWEEP)
sweep_R_b      = np.zeros(N_SWEEP)
sweep_R_res    = np.zeros(N_SWEEP)
for i, kim in enumerate(sweep_k_immune):
    sweep_R_b[i], sweep_R_res[i] = _final_counts(kim)


def _f_res(k_immune):
    return _final_counts(k_immune)[1] - LOD


if _f_res(SWEEP_MIN) < 0 or _f_res(SWEEP_MAX) > 0:
    # No sign change across the bracket. Only fall back to the coarse grid crossing if the
    # grid actually contains one -- otherwise np.argmax on an all-False array silently
    # returns index 0, i.e. SWEEP_MIN, which would be plotted as a real switch.
    escapes_on_grid = sweep_R_res > LOD
    if escapes_on_grid.any() and not escapes_on_grid.all():
        ESCAPE_THRESH = float(sweep_k_immune[np.argmax(sweep_R_res <= LOD)])
    else:
        ESCAPE_THRESH = float("nan")
else:
    ESCAPE_THRESH = brentq(_f_res, SWEEP_MIN, SWEEP_MAX, xtol=1e-3)

if np.isnan(ESCAPE_THRESH):
    print(f"No escape threshold exists in [{SWEEP_MIN}, {SWEEP_MAX}]: R_res "
          f"{'escapes' if (sweep_R_res > LOD).all() else 'clears'} across the entire swept range.")
else:
    print(f"Precise escape threshold (final R_res crosses LOD): {ESCAPE_THRESH:.3f} h^-1")

REGIME_COLORS = {"suppressed": "#4c72b0", "escape": "#c44e52"}  # blue / red

# ── Storage ───────────────────────────────────────────────────────────────────
S_b_hist   = np.full((NUM_ITER, len(t_eval)), np.nan)
R_b_hist   = np.full((NUM_ITER, len(t_eval)), np.nan)
S_res_hist = np.full((NUM_ITER, len(t_eval)), np.nan)
R_res_hist = np.full((NUM_ITER, len(t_eval)), np.nan)

# ── Monte Carlo loop ──────────────────────────────────────────────────────────
for i, k_im in enumerate(k_immune_samples):
    immune_i = model_mod.ImmuneResponse(k_immune=k_im)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = odeint(
            model_mod.dual_reservoir_model, Y0, t_eval,
            args=(params, van_func, lzd_func, immune_i),
            rtol=1e-7, atol=1e-9, mxstep=5000,
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


def _draw_panel(ax, history, S_or_R, compartment, color_main, y_max):
    """
    Render one MC panel: shaded treatment windows, individual traces,
    5th–95th band, and median line.
    """
    # Treatment shading
    ax.axvspan(vanco_start_d, lzd_start_d, color="steelblue", alpha=0.10,
               label="Vancomycin")
    ax.axvspan(lzd_start_d,  lzd_end_d,   color="gold",      alpha=0.15,
               label="Linezolid")
    ax.axvline(lzd_end_d, color="darkgoldenrod", ls="--", lw=1.0, alpha=0.7)

    # LOD reference
    ax.axhline(LOD, color="black", ls=":", lw=0.9, alpha=0.6,
               label=f"LOD ({int(LOD)} CFU/mL)")

    # Individual trajectories (random subsample)
    rng_idx = np.random.default_rng(0).choice(NUM_ITER, N_TRACES, replace=False)
    for i in rng_idx:
        ax.plot(t_days, history[i, :], color=color_main, alpha=0.12, lw=0.7)

    # Percentile band + median (suppress all-NaN warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p50  = np.nanpercentile(history, 50, axis=0)
        p05  = np.nanpercentile(history,  5, axis=0)
        p95  = np.nanpercentile(history, 95, axis=0)

    ax.fill_between(t_days, p05, p95, color=color_main, alpha=0.22,
                    label="5th–95th pctile")
    ax.plot(t_days, p50, color=color_main, lw=2.2, label="Median")

    # Axis formatting
    loc_tag  = "Blood"     if compartment == "blood" else "Reservoir"
    strain   = "Sensitive" if S_or_R == "S"           else "Resistant"
    ylabel   = "Blood Burden (CFU/mL)" if compartment == "blood" else "Reservoir Burden (CFU/mL)"
    var_sym  = f"$S_b$"     if (S_or_R == "S" and compartment == "blood")    else \
               f"$R_b$"     if (S_or_R == "R" and compartment == "blood")    else \
               f"$S_{{res}}$" if (S_or_R == "S" and compartment == "reservoir") else \
               f"$R_{{res}}$"

    ax.set_yscale("log")
    ax.set_ylim(LOD * 0.8, y_max)
    ax.set_xlim(0, t_days[-1])
    ax.set_xlabel("Time (Days)", fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.set_title(f"{strain} — {loc_tag}  ({var_sym})", fontsize=18, fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(fontsize=13, loc="upper left", framealpha=0.7)


# ── 4-panel figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

fig.suptitle(
    f"Monte Carlo — Immune Response Variability\n"
    rf"$k_{{immune}}$ ~ LogNormal  (mean = {K_IMMUNE_MEAN}, CV = {CV}),  "
    f"n = {NUM_ITER} simulations",
    fontsize=21, fontweight="bold",
)

_draw_panel(axes[0, 0], S_b_hist,   S_or_R="S", compartment="blood",      color_main="royalblue", y_max=1e6)
_draw_panel(axes[0, 1], R_b_hist,   S_or_R="R", compartment="blood",      color_main="crimson",   y_max=1e6)
_draw_panel(axes[1, 0], S_res_hist, S_or_R="S", compartment="reservoir",  color_main="royalblue", y_max=2e7)
_draw_panel(axes[1, 1], R_res_hist, S_or_R="R", compartment="reservoir",  color_main="crimson",   y_max=2e7)

# Panel labels
for label, ax in zip(["A", "B", "C", "D"], axes.flat):
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="top")

out_file = os.path.join(FIGURES_DIR, "mc_immune_response.png")
fig.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out_file}")

# ── Relapse/escape rate: fraction of runs where R_b or R_res is still above
#    the LOD at end-of-simulation (i.e. resistant infection persists/relapses
#    rather than resolving) ────────────────────────────────────────────────
final_R_b   = np.nan_to_num(R_b_hist[:, -1],   nan=0.0)
final_R_res = np.nan_to_num(R_res_hist[:, -1], nan=0.0)
relapsed    = (final_R_b > LOD) | (final_R_res > LOD)

print(f"\nRelapse (final R_b and/or R_res > LOD): "
      f"{relapsed.sum()}/{NUM_ITER} = {relapsed.mean()*100:.1f}%")
print(f"Resolution (both compartments cleared):  "
      f"{(~relapsed).sum()}/{NUM_ITER} = {(~relapsed).mean()*100:.1f}%")

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

escaped_mask       = final_R_res > LOD
frac_escape        = escaped_mask.mean()
escaped_k_immune   = k_immune_samples[escaped_mask]
suppressed_k_immune = k_immune_samples[~escaped_mask]

# Escape/suppressed split -- quoted in supplement.tex (tab:zscore_summary), so
# print it rather than leaving it readable only off the histogram legend below,
# which rounds to whole percent. NOTE this is NOT the relapse rate printed
# above: escape is the reservoir alone (final_R_res > LOD), relapse is either
# compartment, so escaped is a subset of relapsed and the two can differ. The
# other two rows of tab:zscore_summary are escape fractions, so this is the
# figure that belongs alongside them.
n_escape = int(escaped_mask.sum())
n_total  = int(escaped_mask.size)
print(f"  Escape threshold  k_immune = {ESCAPE_THRESH:.3f} h^-1 (escape below)")
print(f"  Escape      {frac_escape * 100:5.1f}%  ({n_escape}/{n_total})")
print(f"  Suppressed  {(1 - frac_escape) * 100:5.1f}%  ({n_total - n_escape}/{n_total})")
print(f"  (relapse rate above counts either compartment: "
      f"{relapsed.sum()}/{n_total}; escape is a subset)")

fig2, (ax_top, ax_hist) = plt.subplots(
    2, 1, figsize=(11, 8.5), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08},
)
fig2.suptitle(r"Resistant Escape Switch vs $k_{immune}$ (host immune clearance)",
              fontsize=19, fontweight="bold")

# --- Top: continuous switch curve ---
ax_top.axvspan(SWEEP_MIN, ESCAPE_THRESH, color=REGIME_COLORS["escape"],     alpha=0.10)
ax_top.axvspan(ESCAPE_THRESH, SWEEP_MAX, color=REGIME_COLORS["suppressed"], alpha=0.10)
ax_top.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")
ax_top.axvline(ESCAPE_THRESH, color="black", ls="--", lw=1.4,
               label=fr"Switch ($k_{{immune}}$ = {ESCAPE_THRESH:.2g})")

ax_top.plot(sweep_k_immune, disp_sweep_R_res, color="indianred", lw=2.2,
            label="Final $R_{res}$ (reservoir)")
ax_top.plot(sweep_k_immune, disp_sweep_R_b, color="darkred", lw=2.2,
            label="Final $R_b$ (blood)")

ax_top.set_yscale("log")
ax_top.set_ylim(FLOOR * 0.8, max(sweep_R_b.max(), sweep_R_res.max()) * 3)
ax_top.set_ylabel("Final resistant count (CFU/mL)")
ax_top.grid(True, which="both", ls=":", alpha=0.35)
ax_top.legend(loc="upper right", fontsize=13, framealpha=0.9)

# --- Bottom: where the sampled patient population falls, by actual outcome ---
bin_edges = np.linspace(SWEEP_MIN, SWEEP_MAX, 31)
ax_hist.hist(suppressed_k_immune, bins=bin_edges, color=REGIME_COLORS["suppressed"],
             alpha=0.75, edgecolor="white", linewidth=0.4,
             label=f"Suppressed ({(1 - frac_escape) * 100:.0f}% of patients)")
ax_hist.hist(escaped_k_immune, bins=bin_edges, color=REGIME_COLORS["escape"],
             alpha=0.75, edgecolor="white", linewidth=0.4,
             label=f"Escape ({frac_escape * 100:.0f}% of patients)")
ax_hist.axvline(ESCAPE_THRESH, color="black", ls="--", lw=1.4)

ax_hist.set_xlim(SWEEP_MIN, SWEEP_MAX)
ax_hist.set_xlabel(r"$k_{immune}$ (h$^{-1}$)")
ax_hist.set_ylabel("Patients\n(n)")
ax_hist.grid(True, axis="y", ls=":", alpha=0.35)
ax_hist.legend(loc="upper right", fontsize=12, framealpha=0.9)

out_file = os.path.join(FIGURES_DIR, "mc_immune_response_switch.png")
fig2.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"Saved: {out_file}")
