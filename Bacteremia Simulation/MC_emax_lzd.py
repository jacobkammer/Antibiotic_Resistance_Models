# =============================================================================
# Monte Carlo Simulation: Linezolid Emax (Emax_l)
# Samples Emax_l ~ LogNormal(mean=1.085 h^-1, CV=0.307), mean-corrected so
# E[Emax_l] lands exactly on baseline (same convention as MC_immune_response.py
# and MC_ec50_lzd.py), while every other parameter stays fixed, isolating the
# effect of linezolid's maximum killing capacity on bacterial kinetics. Mirrors
# MC_ec50_lzd.py, but for Emax_l instead of EC50_L.
#
# Growth rates (rho_S/rho_R/rho_res_S/rho_res_R) were scaled 5x alongside the
# original Emax_l increase (from 0.16 to 0.8), and EC50_L was independently
# raised to 1.0. rho_S/rho_R were later lowered 0.80/0.64 -> 0.60/0.55
# (~8.3% fitness cost, down from 20%) for slower post-treatment R_b regrowth.
# Emax_l originally auto-followed rho_S (so it would have dropped to 0.6),
# but was then UNCOUPLED from rho_S and fixed at 0.8 -- see
# model_Bacteremia.py's ImmuneResponse class for why.
#
# Unlike EC50_L (where HIGH values are bad), for Emax_l LOW values are bad:
# below a threshold, linezolid's ceiling effect can no longer hold resistant
# growth in check and R_b/R_res escape.
#
# B_max_reservoir was lowered 4.5e6 -> 1e4 (see baseline_reservoir_
# clearance.py) so an escaped R_res plateaus far below its old level. At
# rho_res_R = 0.145, this sweep's joint R_b/R_res escape threshold was
# ~0.6517 h^-1, and fixed Emax_l = 0.8 sat ABOVE it, so ALL FOUR compartments
# cleared at baseline -- the only point in this project where that happened.
#
# rho_res_R was then raised 0.145 -> 0.20 (precise clearance threshold
# 0.17594055) -- now a fitness ADVANTAGE over rho_res_S (0.175) rather than
# a cost -- which pushed the threshold up to ~0.9156 h^-1, and restored
# "no clinical resolution": R_b visibly CLEARED during the linezolid course
# before relapsing. BUT at rho_res_R = 0.20, R_res's reservoir growth
# advantage over S_res (0.175) was large enough that R_b won the
# pretreatment race for blood's shared carrying capacity outright, crowding
# S_b out entirely -- no visible sensitive-strain infection anywhere in this
# project's Monte Carlo scripts.
#
# rho_res_R was finally narrowed to 0.1765 -- just above the reservoir's own
# 0.17594055 persistence threshold, but below the point where R_b starts
# winning the blood race -- so S_b CAN also establish a visible infection
# (peak ~4.8e3 CFU/mL). This pulled this sweep's escape threshold down to
# 0.8027 h^-1, right next to baseline (0.8): only ~48% of sampled Emax_l
# values now let the reservoir persist and relapse, vs. ~52% that suppress
# it entirely. Both outcomes are accepted here -- the point of this
# rho_res_R is to let S_b establish, not to guarantee reservoir persistence
# in every sample.
#
# rho_res_S was then given a 10% growth-rate advantage over rho_res_R
# (0.19415 vs 0.1765, a fitness cost for R mirroring the blood compartment),
# and lzd_res_fraction was lowered 0.45 -> 0.30 to compensate (Emax_l_res
# scales with rho_res_S, so the higher rho_res_S alone would otherwise wipe
# out the reservoir entirely). Net effect: this sweep's escape threshold
# moved UP to 0.8414 h^-1 (confirmed by this script's own bisection,
# matching Emax_lzd_threshold_sweep.py) -- now above baseline (0.8) rather
# than straddling it, so baseline itself sat in the escape/persist zone.
# Against the sampled Emax_l distribution (median=0.778, [5th=0.481,
# 95th=1.235]), that put 608/1000 = ~61% of samples below threshold
# (escape/relapse) and ~39% above it (suppressed).
#
# EMAX_L_BASE was then raised 0.8 -> 1.085 (CV=0.307 unchanged) to bring
# this sweep's escape rate in line with MC_immune_response.py's relapse
# rate (24.3%, 243/1000) rather than leaving Emax_l as the one sweep whose
# baseline sat almost on top of its own escape threshold. Solved
# analytically (mean such that the lognormal's CDF at the 0.8414 h^-1
# threshold equals 0.243) and confirmed by a direct 1000-sample MC run:
# 23.6% escape, sampled median=1.056 [5th=0.652, 95th=1.676] -- close to
# the immune-response escape rate as intended.
#
# EC50_L was then consolidated onto a SINGLE project-wide value. Previously
# the project carried two: a fixed baseline EC50_L = 1.0 used by every script
# that holds EC50_L constant, and a sampling mean EC50_L_BASE = 0.6 used by
# MC_ec50_lzd.py. That also contradicted MC_ec50_lzd.py's own docstring claim
# that its sampling is "mean-corrected so E[EC50_L] lands exactly on
# baseline". Both are now 0.6 mg/L.
#
# Lowering the fixed EC50_L 1.0 -> 0.6 makes linezolid more potent, which on
# its own put the baseline BELOW the EC50_L escape threshold: the reservoir
# cleared at baseline and the k_immune sweep lost its threshold entirely (no
# k_immune in [0, 3] produced relapse -- even zero immune clearance cleared
# the infection). rho_S was therefore raised 0.61 -> 0.72 h^-1, with rho_R
# rescaled 0.55 -> 0.649 to preserve the ~10% fitness cost, restoring the
# reservoir to the escape side. The mechanism is indirect: the reservoir kill
# rate is Emax_l_res = Emax_l * (rho_res_S / rho_S), so a HIGHER rho_S
# weakens reservoir killing. rho_res_R/rho_res_S were tried first and
# rejected -- every value of either that reached baseline escape collapsed
# peak S_b from ~4.8e3 to ~2 CFU/mL (R_b crowding S_b out of blood).
#
# Net effect here: essentially none. The threshold moved 0.842 -> 0.843 h^-1
# and escape 23.6% -> 23.8% (238/1000, same SEED=44) -- the lower EC50_L and
# the higher rho_S very nearly cancel for this sweep.
# =============================================================================
import importlib.util
import os
import sys
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
NUM_ITERATIONS = 1000
CV             = 0.307      # log-normal spread applied to Emax_l (equiv. to prior sigma=0.3)
SEED           = 44

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

# ---------------------------------------------------------------------------
# Fixed parameters (all except Emax_l held at baseline)
# ---------------------------------------------------------------------------
# rho_S raised 0.61 -> 0.72 (rho_R 0.55 -> 0.649, ~10% fitness cost preserved) to offset
# EC50_L 1.0 -> 0.6 while keeping lzd_res_fraction at 0.30: reservoir kill scales as
# Emax_l_res = Emax_l * (rho_res_S / rho_S), so a higher rho_S weakens it.
rho_S        = 0.72    # raised from 0.60 -- see rho_S sensitivity analysis. At the current
                       # rho_res_S/rho_res_R/lzd_res_fraction baseline the Emax_l escape
                       # threshold is 0.842 h^-1 (bisected above)
rho_R        = 0.649    # directly tuned (~8.3% fitness cost relative to rho_S, down from 20%)

EMAX_L_BASE = 1.085 # raised from 0.8 -- see header narrative above (targets ~24% escape,
                     # matching MC_immune_response.py's relapse rate)

# Model parameters come from parameters.py (single source of truth).

# Fixed initial conditions — only Emax_l is varied across iterations

# ---------------------------------------------------------------------------
# Run Monte Carlo sweep: vary Emax_l, hold everything else fixed
# ---------------------------------------------------------------------------
# CV = sqrt(exp(σ²) - 1)  →  σ = sqrt(ln(1 + CV²))
# μ  = ln(mean) - σ²/2   ensures E[Emax_l] = EMAX_L_BASE
sigma_ln = np.sqrt(np.log(1.0 + CV**2))
mu_ln    = np.log(EMAX_L_BASE) - sigma_ln**2 / 2.0

rng = np.random.default_rng(SEED)
emax_l_samples = rng.lognormal(mu_ln, sigma_ln, NUM_ITERATIONS)

# ---------------------------------------------------------------------------
# Continuous deterministic switch curve + precise escape threshold.
#
# Mirrors MC_ec50_lzd.py's own fine sweep + bisection, but direction-flipped:
# for Emax_l, LOW values are the escape zone (a weaker drug ceiling can't
# hold resistant growth in check), the opposite of EC50_L. Replaces the old
# hardcoded ESCAPE_THRESH=0.8161 with a value bisected directly here, and
# replaces the 4-bin categorical grouping with a fine, continuous sweep used
# by the combined switch/population figure below.
# ---------------------------------------------------------------------------
def _final_counts(emax_l):
    """One deterministic run at a fixed Emax_l; returns (final R_b, final R_res)."""
    params_i = {**BASE_PARAMS, "Emax_l": emax_l}
    sol = odeint(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(params_i, van_func, lzd_func, immune_model),
        rtol=1e-7, atol=1e-9, mxstep=5000,
    )
    return sol[-1, 1], sol[-1, 3]


SWEEP_MIN, SWEEP_MAX = 0.0, 2.2   # widened from 1.0 -- EMAX_L_BASE=1.085 with CV=0.307 puts the
                                   # 99th percentile of sampled Emax_l near 2.08
N_SWEEP = 111   # step = 0.02

print(f"\nRunning deterministic switch-curve sweep across [{SWEEP_MIN}, {SWEEP_MAX}] "
      f"({N_SWEEP} points)...", flush=True)
sweep_emax  = np.linspace(SWEEP_MIN, SWEEP_MAX, N_SWEEP)
sweep_R_b   = np.zeros(N_SWEEP)
sweep_R_res = np.zeros(N_SWEEP)
for i, emax in enumerate(sweep_emax):
    sweep_R_b[i], sweep_R_res[i] = _final_counts(emax)


def _f_res(emax_l):
    return _final_counts(emax_l)[1] - LOD


if _f_res(SWEEP_MIN) < 0 or _f_res(SWEEP_MAX) > 0:
    # No sign change across the bracket. Only fall back to the coarse grid crossing if the
    # grid actually contains one -- otherwise np.argmax on an all-False array silently
    # returns index 0, i.e. SWEEP_MIN, which would be plotted as a real switch.
    escapes_on_grid = sweep_R_res > LOD
    if escapes_on_grid.any() and not escapes_on_grid.all():
        ESCAPE_THRESH = float(sweep_emax[np.argmax(sweep_R_res <= LOD)])
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
# Escape classification, computed here because the kinetics figure below needs
# it: each patient is escaped iff their own final R_res is above the LOD.
# ---------------------------------------------------------------------------
final_R_res  = np.nan_to_num(R_res_hist[:, -1], nan=0.0)
escaped_mask = final_R_res > LOD



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
fig0.suptitle(r"Bacterial Kinetics — Monte Carlo sweep of $E_{max,L}$ (linezolid)",
              fontsize=19, fontweight="bold")

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

    # Summarise over the ESCAPED patients only. Taking the median over all 1000
    # rows makes the post-treatment relapse vanish from the plot: sub-LOD values
    # are stored as 0, escape is a minority outcome, so the all-patient median is
    # 0 (-> NaN) exactly where the relapse happens. Restricting to escaped rows
    # and using nanmedian keeps the relapse visible and states its n in the
    # legend. Sub-LOD points within that subset become NaN, so a gap here means
    # "not detectable in any escaped patient", not "no data".
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
fig0.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"Saved: {out_file}")


# ---------------------------------------------------------------------------
# FIGURE: switch curve (continuous, deterministic) + sampled patient
# population relative to the escape threshold. Mirrors MC_ec50_lzd.py's
# combined switch/population figure, direction-flipped for Emax_l: escape is
# the LOW side, suppression the HIGH side.
# ---------------------------------------------------------------------------
final_R_b   = np.nan_to_num(R_b_hist[:, -1],   nan=0.0)
# final_R_res / escaped_mask are defined earlier (the kinetics figure needs them)

FLOOR = LOD * 0.5   # display floor for values that fell below the LOD (i.e. 0)

disp_sweep_R_b   = np.where(sweep_R_b   <= LOD, FLOOR, sweep_R_b)
disp_sweep_R_res = np.where(sweep_R_res <= LOD, FLOOR, sweep_R_res)

frac_escape     = escaped_mask.mean()
escaped_emax    = emax_l_samples[escaped_mask]
suppressed_emax = emax_l_samples[~escaped_mask]

fig1, (ax_top, ax_hist) = plt.subplots(
    2, 1, figsize=(11, 8.5), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08},
)
fig1.suptitle(r"Resistant Escape Switch vs $Emax_L$ (linezolid)",
              fontsize=19, fontweight="bold")

# --- Top: continuous switch curve ---
ax_top.axvspan(SWEEP_MIN, ESCAPE_THRESH, color=REGIME_COLORS["escape"],     alpha=0.10)
ax_top.axvspan(ESCAPE_THRESH, SWEEP_MAX, color=REGIME_COLORS["suppressed"], alpha=0.10)
ax_top.axhline(LOD, color="black", ls=":", lw=1.0, alpha=0.7, label=f"LOD ({int(LOD)} CFU/mL)")
ax_top.axvline(ESCAPE_THRESH, color="black", ls="--", lw=1.4,
               label=fr"Switch ($Emax_L$ = {ESCAPE_THRESH:.2g})")

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
ax_hist.hist(suppressed_emax, bins=bin_edges, color=REGIME_COLORS["suppressed"],
             alpha=0.75, edgecolor="white", linewidth=0.4,
             label=f"Suppressed ({(1 - frac_escape) * 100:.0f}% of patients)")
ax_hist.hist(escaped_emax, bins=bin_edges, color=REGIME_COLORS["escape"],
             alpha=0.75, edgecolor="white", linewidth=0.4,
             label=f"Escape ({frac_escape * 100:.0f}% of patients)")
ax_hist.axvline(ESCAPE_THRESH, color="black", ls="--", lw=1.4)

ax_hist.set_xlim(SWEEP_MIN, SWEEP_MAX)
ax_hist.set_xlabel(r"$Emax_L$ (h$^{-1}$)")
ax_hist.set_ylabel("Patients\n(n)")
ax_hist.grid(True, axis="y", ls=":", alpha=0.35)
ax_hist.legend(loc="upper right", fontsize=12, framealpha=0.9)

out_file = os.path.join(FIGURES_DIR, "mc_emax_lzd_switch.png")
fig1.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"Saved: {out_file}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\n--- Sweep: Emax_l varied (baseline={EMAX_L_BASE:.3f}) ---")
print(f"  Emax_l sampled  median={np.median(emax_l_samples):.3f}  "
      f"[5th={np.percentile(emax_l_samples, 5):.3f}, "
      f"95th={np.percentile(emax_l_samples, 95):.3f}]")
# Escape/suppressed split -- quoted in supplement.tex (tab:zscore_summary) and
# in threshold_changes_summary.txt, so print it rather than leaving it readable
# only off the histogram legend in mc_emax_lzd_switch.png. Note the sense is
# inverted relative to EC50_L: here escape is Emax_l BELOW the threshold.
n_escape = int(escaped_mask.sum())
n_total  = int(escaped_mask.size)
print(f"  Escape threshold  Emax_l = {ESCAPE_THRESH:.3f} h^-1 (escape below)")
print(f"  Escape      {frac_escape * 100:5.1f}%  ({n_escape}/{n_total})")
print(f"  Suppressed  {(1 - frac_escape) * 100:5.1f}%  ({n_total - n_escape}/{n_total})")
if np.any(final_R_b > LOD):
    print(f"  Final R_b  median={np.median(final_R_b[final_R_b > LOD]):.1e}  "
          f"[5th={np.percentile(final_R_b[final_R_b > LOD], 5):.1e}, "
          f"95th={np.percentile(final_R_b[final_R_b > LOD], 95):.1e}] CFU/mL "
          f"(n={np.sum(final_R_b > LOD)}/{NUM_ITERATIONS} escaped)")
else:
    print("  Final R_b: no samples above LOD")

print("\nDone.")
