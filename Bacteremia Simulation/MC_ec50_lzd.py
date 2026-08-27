# =============================================================================
# Monte Carlo Simulation: Linezolid EC50 (EC50_L)
# Samples EC50_L ~ LogNormal(mean=0.6 mg/L, CV=1.1), mean-corrected so
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
# Current baseline: rho_S = 0.61, rho_R = 0.55 (~10% fitness cost), and
# Emax_l = 0.8 (fixed and decoupled from rho_S). The EC50_L escape threshold
# is now bisected directly in this script, rather than hard-coded.

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
#
# rho_res_S was then given a 10% growth-rate advantage over rho_res_R
# (0.19415 vs 0.1765, a fitness cost for R mirroring the blood compartment),
# and lzd_res_fraction was lowered 0.45 -> 0.30 to compensate (Emax_l_res
# scales with rho_res_S, so the higher rho_res_S alone would otherwise wipe
# out the reservoir entirely). Net effect: the EC50_L escape threshold moved
# DOWN to 0.868 mg/L (confirmed by this script's own bisection, matching
# EC50_lzd_threshold_sweep.py). At baseline EC50_L_BASE=1.0, that put ~60%
# of samples at or below threshold (suppressed) and ~40% above it
# (escape/relapse).
#
# EC50_L_BASE was then lowered 1.0 -> 0.695 (CV=1.117 unchanged) to bring
# this sweep's escape rate in line with MC_immune_response.py's relapse
# rate (24.3%, 243/1000) and MC_emax_lzd.py's escape rate (23.6%, after its
# own EMAX_L_BASE retune), rather than leaving EC50_L as an outlier at
# ~40%. Solved analytically (mean such that the lognormal's CDF at the
# 0.868 mg/L threshold equals 1 - 0.243) and confirmed by a direct
# 1000-sample MC run: 27.1% escape, sampled median=0.489 [5th=0.115,
# 95th=1.954] -- within the +/-5-point tolerance of both other sweeps.
#
# Emax_v/EC50_V (vancomycin) were later raised 0.40/0.245 -> 1.7/0.25 to
# match literature values; confirmed (via direct bisection/MC checks) to
# leave the 0.868 mg/L EC50_L escape threshold and all downstream
# calibrations in this file completely unchanged -- vancomycin never acts
# on R_b/R_res directly, and its effect on S_b/S_res fully reconverges
# before the vancomycin course even ends.
#
# EC50_L_BASE was then lowered again 0.695 -> 0.6 (CV=1.117 unchanged) --
# a deliberate departure from the ~24-27% cross-script calibration target,
# not a re-tune toward it: this makes linezolid more potent at realistic
# (sub-saturating) concentrations, shifting more of the sampled
# distribution below the (unchanged) 0.868 mg/L escape threshold. Confirmed
# by a direct 1000-sample MC run (same SEED=44): 21.8% escape (218/1000),
# sampled median=0.422 [5th=0.099, 95th=1.686] -- below the other two
# sweeps' 24-27% band by design.
#
# CV was then narrowed 1.117 -> 1.1 (EC50_L_BASE=0.6 unchanged) -- a small
# tightening of inter-patient variability that leaves the escape rate
# essentially untouched: analytically, the lognormal's CDF at the 0.868
# mg/L threshold barely moves (19.46% -> 19.47%), and a direct 1000-sample
# MC run (same SEED=44) landed on the identical 21.8% escape (218/1000),
# sampled median=0.425 [5th=0.102, 95th=1.675] -- indistinguishable from
# the CV=1.117 run above at this sample size.
#
# this file generates 2 figures:
# 1. mc_ec50_lzd_sweep_kinetics.png
# 2. mc_ec50_lzd_switch.png
# (a third, mc_ec50_lzd_sweep_outcomes.png -- per-patient scatter of peak/final
# burdens vs sampled EC50_L -- was removed along with its plotting block.)
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
# Net effect here: the escape threshold moved 0.868 -> 0.488 mg/L and escape
# rose 21.8% -> 43.6% (436/1000, same SEED=44). This sweep is unaffected by
# the EC50_L consolidation itself (it overrides EC50_L with its own sampled
# draws) and responds only to the rho_S change. At mean 0.6 vs threshold
# 0.488 the calibration now sits ~0.21 SD from its own switch -- the closest
# of the three sweeps, which is why its escape fraction approaches one half.
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
#   mc_ec50_lzd_sweep_kinetics.png  -- MONTE CARLO. 4 panels (S_b, R_b, S_res,
#       R_res); 1000 sampled patients, median + 5th-95th percentile band.
#   mc_ec50_lzd_switch.png          -- MIXED, one source per panel:
#       top    = DETERMINISTIC  (131-point fixed-parameter sweep + brentq root)
#       bottom = MONTE CARLO    (the 1000 samples, coloured by each patient's
#                                own simulated final R_res vs LOD -- NOT by
#                                which side of the plotted threshold they fall)

# ---------------------------------------------------------------------------
# Load model
# import model_Bacteremia would also load the model, however the code below
# is more explicit and allows for better error handling.
# ---------------------------------------------------------------------------
MODULE_NAME = "model_Bacteremia.py"#file to load
if not os.path.exists(MODULE_NAME):
    print(f"ERROR: Cannot find '{MODULE_NAME}' in the current directory.", file=sys.stderr)#check if file exists
    sys.exit(1)

spec = importlib.util.spec_from_file_location("model_mod", MODULE_NAME)#create spec from file location
model_mod = importlib.util.module_from_spec(spec)#create module from spec
sys.modules["model_mod"] = model_mod#add module to sys.modules
spec.loader.exec_module(model_mod)#execute module

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
NUM_ITERATIONS = 1000#number of iterations
CV             = 1.1        # lowered from 1.117 -- log-normal spread applied to EC50_L
                             # (coefficient of variation); negligible effect on escape rate,
                             # see header narrative above
SEED           = 44#seed for random number generator

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
# Fixed parameters (all except EC50_L held at baseline)
# ---------------------------------------------------------------------------
# rho_S raised 0.61 -> 0.72 (rho_R 0.55 -> 0.649, ~10% fitness cost preserved) to offset
# EC50_L 1.0 -> 0.6 while keeping lzd_res_fraction at 0.30: reservoir kill scales as
# Emax_l_res = Emax_l * (rho_res_S / rho_S), so a higher rho_S weakens it.
rho_S        = 0.72    
rho_R        = 0.649    #10% fitness cost

EC50_L_BASE = 0.6     # lowered again from 0.695 -- see header narrative above. This is a
                      # deliberate departure from the ~24-27% cross-script calibration target
                      # (MC_immune_response.py / MC_emax_lzd.py); it now targets ~19-23%
                      # escape instead, making linezolid more potent at realistic
                      # (sub-saturating) concentrations.

# Model parameters come from parameters.py (single source of truth).

# Fixed initial conditions — only EC50_L is varied across iterations

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


SWEEP_MIN, SWEEP_MAX = 0.05, 3.3   # widened from [0.3, 2.0] -- at EC50_L_BASE=0.6, CV=1.1
                                    # (lowered from 1.117), the 1st-99th percentile of sampled
                                    # EC50_L is roughly [0.05, 3.20], still comfortably covered
                                    # by this range
N_SWEEP = 131   # step ~0.025

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
    # No sign change across the bracket. Only fall back to the coarse grid crossing if the
    # grid actually contains one -- otherwise np.argmax on an all-False array silently
    # returns index 0, i.e. SWEEP_MIN, which would be plotted as a real switch.
    escapes_on_grid = sweep_R_res > LOD
    if escapes_on_grid.any() and not escapes_on_grid.all():
        ESCAPE_THRESH = float(sweep_ec50[np.argmax(escapes_on_grid)])
    else:
        ESCAPE_THRESH = float("nan")
else:
    ESCAPE_THRESH = brentq(_f_res, SWEEP_MIN, SWEEP_MAX, xtol=1e-3)

if np.isnan(ESCAPE_THRESH):
    print(f"No escape threshold exists in [{SWEEP_MIN}, {SWEEP_MAX}]: R_res "
          f"{'escapes' if (sweep_R_res > LOD).all() else 'clears'} across the entire swept range.")
else:
    print(f"Precise escape threshold (final R_res crosses LOD): {ESCAPE_THRESH:.3f} mg/L")

REGIME_COLORS = {"suppressed": "#4c72b0", "escape": "#c44e52"}  # blue / red

S_b_hist   = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
R_b_hist   = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
S_res_hist = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)
R_res_hist = np.full((NUM_ITERATIONS, len(t_eval)), np.nan)

print(f"Running {NUM_ITERATIONS} iterations (sweep: EC50_L, baseline={EC50_L_BASE:.2f})...",
      flush=True)
# ---------------------------------------------------------------------------
# Monte Carlo Simulation loop for each iteration
# ---------------------------------------------------------------------------

for i in range(NUM_ITERATIONS):
    params_i = {**BASE_PARAMS, "EC50_L": ec50_l_samples[i]} # copies the baseline parameters and overrides EC50_L

    sol = odeint(
        model_mod.dual_reservoir_model,
        Y0, t_eval,
        args=(params_i, van_func, lzd_func, immune_model),
        rtol=1e-7, atol=1e-9, mxstep=5000,
    )

    sb   = np.where(sol[:, 0] < LOD, 0.0, sol[:, 0]) #replaces any values below LOD with 0.0
    rb   = np.where(sol[:, 1] < LOD, 0.0, sol[:, 1]) #replaces any values below LOD with 0.0
    sres = np.where(sol[:, 2] < LOD, 0.0, sol[:, 2]) #replaces any values below LOD with 0.0
    rres = np.where(sol[:, 3] < LOD, 0.0, sol[:, 3]) #replaces any values below LOD with 0.0    

    S_b_hist[i]   = sb #stores the full time course for S_b
    R_b_hist[i]   = rb #stores the full time course for R_b
    S_res_hist[i] = sres #stores the full time course for S_res
    R_res_hist[i] = rres #stores the full time course for R_res

    if (i + 1) % 50 == 0: #prints progress every 50 iterations
        print(f"  {i + 1}/{NUM_ITERATIONS} complete", flush=True)

print("Simulations complete. Generating figure...", flush=True)# prints completion message

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

plt.tight_layout()
out_file = os.path.join(FIGURES_DIR, "mc_ec50_lzd_sweep_kinetics.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"Saved: {out_file}")


# ---------------------------------------------------------------------------
# Outcome metrics (one summary value per iteration).
# peak_S_b/peak_R_b feed the console summary. final_R_res/escaped_mask are
# defined earlier, above the kinetics figure, which also needs them.
# ---------------------------------------------------------------------------
peak_S_b    = np.nanmax(S_b_hist, axis=1)
peak_R_b    = np.nanmax(R_b_hist, axis=1)




# ---------------------------------------------------------------------------
# FIGURE: switch curve (continuous, deterministic) + sampled patient
# population relative to the escape threshold.
#
# Top panel is the actual result: final resistant burden vs EC50_L,
# continuous and bisected rather than binned, so the transition at
# ESCAPE_THRESH reads as a genuine switch rather than a coarse step between a
# handful of categories.
# Bottom panel answers the question the switch alone doesn't: given the same
# realistic inter-patient EC50_L variability used for the kinetics figure
# above, what fraction of virtual patients actually land on each
# side of it. Each sample is colored by its own simulated outcome (final
# R_res vs LOD), not by which side of the threshold its EC50_L happens to
# fall on, so the figure stays consistent with the simulated data.
# ---------------------------------------------------------------------------
FLOOR = LOD * 0.5   # display floor for values that fell below the LOD (i.e. 0)

disp_sweep_R_b   = np.where(sweep_R_b   <= LOD, FLOOR, sweep_R_b)
disp_sweep_R_res = np.where(sweep_R_res <= LOD, FLOOR, sweep_R_res)

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

out_file = os.path.join(FIGURES_DIR, "mc_ec50_lzd_switch.png")
fig3.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"Saved: {out_file}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print(f"\n--- Sweep: EC50_L varied (baseline={EC50_L_BASE:.2f}) ---")
print(f"  EC50_L sampled  median={np.median(ec50_l_samples):.2f}  "
      f"[5th={np.percentile(ec50_l_samples, 5):.2f}, "
      f"95th={np.percentile(ec50_l_samples, 95):.2f}]")
# Escape/suppressed split -- quoted in supplement.tex (tab:zscore_summary) and
# in threshold_changes_summary.txt, so print it rather than leaving it readable
# only off the histogram legend in mc_ec50_lzd_switch.png.
n_escape = int(escaped_mask.sum())
n_total  = int(escaped_mask.size)
print(f"  Escape threshold  EC50_L = {ESCAPE_THRESH:.3f} mg/L")
print(f"  Escape      {frac_escape * 100:5.1f}%  ({n_escape}/{n_total})")
print(f"  Suppressed  {(1 - frac_escape) * 100:5.1f}%  ({n_total - n_escape}/{n_total})")
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
