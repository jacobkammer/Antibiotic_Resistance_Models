# =============================================================================
# Monte Carlo Simulation: Linezolid Emax (Emax_l)
# Samples Emax_l ~ LogNormal(mean=0.8 h^-1, CV=0.307), mean-corrected so
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
CV             = 0.307      # log-normal spread applied to Emax_l (equiv. to prior sigma=0.3)
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
rho_S        = 0.61    # raised from 0.60 -- see rho_S sensitivity analysis: this shifts the
                       # Emax_l escape threshold from 0.8027 up to 0.8161 h^-1 via a
                       # vancomycin-driven competitive-release effect (S_b grows larger
                       # pre-vancomycin, then gets cleared, leaving more of blood's shared
                       # carrying capacity open for R_b to claim)
rho_R        = 0.55    # directly tuned (~8.3% fitness cost relative to rho_S, down from 20%)

EMAX_L_BASE = 0.8   # fixed, decoupled from rho_S (was tied for "perfect bacteriostasis")

BASE_PARAMS = {
    "rho_S":            rho_S,
    "rho_R":            rho_R,
    "rho_res_S":        0.175,  # scaled 5x (from 0.035) alongside rho_S
    "rho_res_R":        0.1765,  # narrow window just above the reservoir persistence threshold (0.17594055) so S_b can also establish in blood -- see model_Bacteremia.py
    "Emax_v":           0.40,
    "EC50_V":           0.245,
    "EC50_L":           1.0,
    "B_max_blood":      6000,
    "B_max_reservoir":  1e4,    # lowered from 4.5e6 so escaped R_res plateaus well above the LOD but far below its old level
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.45,
    "f_r_b":            5e-5,  # restored to original
    "f_b_r":            1e-5,
}

# Fixed initial conditions — only Emax_l is varied across iterations
Y0 = [0.0, 0.0, 100.0, 100.0]

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


SWEEP_MIN, SWEEP_MAX = 0.0, 1.0
N_SWEEP = 101   # step = 0.01

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
    ESCAPE_THRESH = float(sweep_emax[np.argmax(sweep_R_res <= LOD)])  # fallback: coarse crossing
else:
    ESCAPE_THRESH = brentq(_f_res, SWEEP_MIN, SWEEP_MAX, xtol=1e-4)

print(f"Precise escape threshold (final R_res crosses LOD): {ESCAPE_THRESH:.4f} h^-1")

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
# FIGURE: switch curve (continuous, deterministic) + sampled patient
# population relative to the escape threshold. Mirrors MC_ec50_lzd.py's
# combined switch/population figure, direction-flipped for Emax_l: escape is
# the LOW side, suppression the HIGH side.
# ---------------------------------------------------------------------------
final_R_b   = np.nan_to_num(R_b_hist[:, -1],   nan=0.0)
final_R_res = np.nan_to_num(R_res_hist[:, -1], nan=0.0)

FLOOR = LOD * 0.5   # display floor for values that fell below the LOD (i.e. 0)

disp_sweep_R_b   = np.where(sweep_R_b   <= LOD, FLOOR, sweep_R_b)
disp_sweep_R_res = np.where(sweep_R_res <= LOD, FLOOR, sweep_R_res)

escaped_mask    = final_R_res > LOD
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

fig1.savefig("mc_emax_lzd_switch.png", dpi=300, bbox_inches="tight")
print("Saved: mc_emax_lzd_switch.png")


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
