"""
run_MC_other_params.py
========================
Monte Carlo exploration of non-transfer-rate parameters.

Design differences from run_MC_transfer_rates.py:
  - Transfer rates (f_r_b, f_b_r) are FIXED at nominal values to isolate
    the effect of other parameters from transfer-rate variability.
  - PD parameters (Emax_v, EC50_V, Emax_l, EC50_L), reservoir growth rate
    (rho_res_S), and carrying capacities (B_max_blood, B_max_reservoir) are
    varied.  Immune clearance (k_immune), growth-rate multiplier, fitness
    cost, and vancomycin bone penetration (van_res_fraction) are FIXED.
  - Parameters are sampled from log-normal distributions whose 95%
    intervals span biologically plausible ranges.
  - Output figures are saved to Figures/OtherParams_*.png.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import importlib.util
import os
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


def lognormal_from_range(lower, upper):
    """Compute (mu, sigma) for log-normal so that ~95% of draws fall in [lower, upper].

    The median of the distribution equals the geometric mean of the bounds.
    """
    mu    = (np.log(lower) + np.log(upper)) / 2.0
    sigma = (np.log(upper) - np.log(lower)) / (2 * 1.96)
    return mu, sigma


# --- Load model ---
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, "model_other_params_exploration.py")

spec = importlib.util.spec_from_file_location("model.other_params_exploration", MODEL_PATH)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

fig_dir = os.path.join(THIS_DIR, "Figures")
os.makedirs(fig_dir, exist_ok=True)

PharmacokineticModel = model_module.PharmacokineticModel
ImmuneResponse       = model_module.ImmuneResponse
dual_reservoir_model = model_module.dual_reservoir_model

print('Other-Parameters Monte Carlo Exploration')
print('Using: model_other_params_exploration.py')
print('=' * 70)

# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
np.random.seed(42)

# -------------------------------------------------------------------------
# Fixed simulation settings
# -------------------------------------------------------------------------
n_simulations = 100
total_h       = 1344   # 56 days
vanco_start   = 504    # h (day 21)

# -------------------------------------------------------------------------
# Fixed transfer rates (nominal values)
# -------------------------------------------------------------------------
F_R_B_FIXED = 5e-5   # h⁻¹  reservoir → blood (Kostakioti et al. 2013)
F_B_R_FIXED = 1e-5   # h⁻¹  blood → reservoir

print(f"Fixed transfer rates: f_r_b = {F_R_B_FIXED:.0e}, f_b_r = {F_B_R_FIXED:.0e} h^-1")
print(f"Simulations: {n_simulations}  |  Total time: {total_h} h ({total_h/24:.0f} days)")
print(f"Vancomycin start: {vanco_start} h (day {vanco_start/24:.0f})")
print()

# -------------------------------------------------------------------------
# Fixed growth / immune / penetration parameters
# -------------------------------------------------------------------------
K_IMMUNE_FIXED     = 0.12    # h^-1 (Regoes et al. 2004; Drusano 2004)
RHO_S_MULT_FIXED   = 1.50   # multiplier on eff_blood * k_immune
FITNESS_COST_FIXED = 0.20   # Foucault et al. 2009
VAN_RES_FRAC_FIXED = 0.15   # Graziani et al. 1988

EFF_BLOOD  = 1.0
RHO_S_FIXED = RHO_S_MULT_FIXED * EFF_BLOOD * K_IMMUNE_FIXED
RHO_R_FIXED = (1 - FITNESS_COST_FIXED) * RHO_S_FIXED

print("Fixed parameters:")
print(f"  k_immune:         {K_IMMUNE_FIXED} h^-1")
print(f"  rho_S multiplier: {RHO_S_MULT_FIXED}  ->  rho_S = {RHO_S_FIXED:.4f} h^-1")
print(f"  fitness_cost:     {FITNESS_COST_FIXED}  ->  rho_R = {RHO_R_FIXED:.4f} h^-1")
print(f"  van_res_fraction: {VAN_RES_FRAC_FIXED}")
print()

# -------------------------------------------------------------------------
# Parameter sampling ranges (biologically plausible)
# -------------------------------------------------------------------------
# Reservoir (biofilm/bone) sensitive growth: 5-10x slower than planktonic
RHO_RES_S_RANGE = (0.05, 0.15)   # h^-1

# Vancomycin PD -- max kill rate: literature 0.10-0.40 h^-1
# (Zhi et al. 2007; Campion et al. 2005)
EMAX_V_RANGE = (0.10, 0.40)   # h^-1

# Vancomycin EC50 -- near MRSA MIC: 1-3 mg/L (EUCAST)
EC50_V_RANGE = (1.0, 3.0)   # mg/L

# Linezolid bacteriostatic max effect -- capped at rho_S per simulation
# to enforce true bacteriostasis
EMAX_L_RANGE = (0.05, 0.20)   # h^-1

# Linezolid EC50 -- 2-5 mg/L
# (Sandberg et al. 2010)
EC50_L_RANGE = (2.0, 5.0)   # mg/L

# Blood carrying capacity -- peak bacteremia 1e4-1e6 CFU/mL
# (Nolan & Beaty 1976; Wisplinghoff et al.)
B_MAX_BLOOD_RANGE = (1e5, 1e6)   # CFU/mL

# Reservoir carrying capacity -- tissue burden 1e6-1e8 CFU/g
B_MAX_RES_RANGE = (1e6, 1e8)   # CFU/mL equiv

# -------------------------------------------------------------------------
# Precompute log-normal parameters (mu, sigma) from biological ranges
# Median = geometric mean of bounds; 95% CI spans the range
# -------------------------------------------------------------------------
LN_RHO_RES_S   = lognormal_from_range(*RHO_RES_S_RANGE)
LN_EMAX_V      = lognormal_from_range(*EMAX_V_RANGE)
LN_EC50_V      = lognormal_from_range(*EC50_V_RANGE)
LN_EMAX_L      = lognormal_from_range(*EMAX_L_RANGE)
LN_EC50_L      = lognormal_from_range(*EC50_L_RANGE)
LN_B_MAX_BLOOD = lognormal_from_range(*B_MAX_BLOOD_RANGE)
LN_B_MAX_RES   = lognormal_from_range(*B_MAX_RES_RANGE)

print("Parameter sampling (log-normal, 95% CI = biological range):")
print(f"  rho_res_S:        median {np.exp(LN_RHO_RES_S[0]):.3f},  95% CI [{RHO_RES_S_RANGE[0]:.2f}, {RHO_RES_S_RANGE[1]:.2f}] h^-1")
print(f"  Emax_v:           median {np.exp(LN_EMAX_V[0]):.3f},  95% CI [{EMAX_V_RANGE[0]:.2f}, {EMAX_V_RANGE[1]:.2f}] h^-1")
print(f"  EC50_V:           median {np.exp(LN_EC50_V[0]):.3f},  95% CI [{EC50_V_RANGE[0]:.1f}, {EC50_V_RANGE[1]:.1f}] mg/L")
print(f"  Emax_l:           median {np.exp(LN_EMAX_L[0]):.3f},  95% CI [{EMAX_L_RANGE[0]:.2f}, {EMAX_L_RANGE[1]:.2f}] h^-1 (capped at rho_S)")
print(f"  EC50_L:           median {np.exp(LN_EC50_L[0]):.3f},  95% CI [{EC50_L_RANGE[0]:.1f}, {EC50_L_RANGE[1]:.1f}] mg/L")
print(f"  B_max_blood:      median {np.exp(LN_B_MAX_BLOOD[0]):.2e},  95% CI [{B_MAX_BLOOD_RANGE[0]:.0e}, {B_MAX_BLOOD_RANGE[1]:.0e}]")
print(f"  B_max_reservoir:  median {np.exp(LN_B_MAX_RES[0]):.2e},  95% CI [{B_MAX_RES_RANGE[0]:.0e}, {B_MAX_RES_RANGE[1]:.0e}]")
print()

# -------------------------------------------------------------------------
# FIXED initial conditions — bone is the infection origin
# -------------------------------------------------------------------------
y0_fixed = [0.0, 0.0, 1e4, 1e4]   # [S_b, R_b, S_res, R_res] CFU/mL
print(f"Fixed ICs: S_b={y0_fixed[0]:.0f}, R_b={y0_fixed[1]:.0f}, "
      f"S_res={y0_fixed[2]:.0e}, R_res={y0_fixed[3]:.0e} CFU/mL")
print()

# -------------------------------------------------------------------------
# Time grid
# -------------------------------------------------------------------------
t_eval = np.linspace(0, total_h, 400)
t_days = t_eval / 24.0
vanco_start_days = vanco_start / 24.0

# -------------------------------------------------------------------------
# Boxplot timepoints covering full simulation
# -------------------------------------------------------------------------
boxplot_times      = [72, 336, 504, 600, 840, 984, 1344]   # hours
boxplot_times_days = [t / 24.0 for t in boxplot_times]
boxplot_indices    = [np.argmin(np.abs(t_eval - t)) for t in boxplot_times]

# -------------------------------------------------------------------------
# Storage
# -------------------------------------------------------------------------
S_b_results   = []
R_b_results   = []
S_res_results = []
R_res_results = []

param_log = {
    'rho_res_S': [], 'rho_res_R': [],
    'Emax_v': [], 'EC50_V': [], 'Emax_l': [], 'EC50_L': [],
    'B_max_blood': [], 'B_max_reservoir': [],
}

S_b_box   = {t: [] for t in boxplot_times}
R_b_box   = {t: [] for t in boxplot_times}
S_res_box = {t: [] for t in boxplot_times}
R_res_box = {t: [] for t in boxplot_times}

# -------------------------------------------------------------------------
# Monte Carlo loop
# -------------------------------------------------------------------------
print(f"Running {n_simulations} simulations...")
print('-' * 40)

successful_runs = 0
for sim in range(n_simulations):
    if sim % 20 == 0:
        print(f"  Progress: {sim}/{n_simulations}...")

    # --- Sample parameters (log-normal) ---
    rho_res_S   = np.random.lognormal(*LN_RHO_RES_S)
    emax_v      = np.random.lognormal(*LN_EMAX_V)
    ec50_v      = np.random.lognormal(*LN_EC50_V)
    emax_l      = np.random.lognormal(*LN_EMAX_L)
    ec50_l      = np.random.lognormal(*LN_EC50_L)
    b_max_blood = np.random.lognormal(*LN_B_MAX_BLOOD)
    b_max_res   = np.random.lognormal(*LN_B_MAX_RES)

    # Derived
    rho_res_R = rho_res_S * (1 - FITNESS_COST_FIXED)

    # Enforce bacteriostatic constraint: Emax_l cannot exceed rho_S
    emax_l = min(emax_l, RHO_S_FIXED)

    # --- Log parameters ---
    param_log['rho_res_S'].append(rho_res_S)
    param_log['rho_res_R'].append(rho_res_R)
    param_log['Emax_v'].append(emax_v)
    param_log['EC50_V'].append(ec50_v)
    param_log['Emax_l'].append(emax_l)
    param_log['EC50_L'].append(ec50_l)
    param_log['B_max_blood'].append(b_max_blood)
    param_log['B_max_reservoir'].append(b_max_res)

    # --- Build parameter dict ---
    params = {
        'rho_S':            RHO_S_FIXED,
        'rho_R':            RHO_R_FIXED,
        'rho_res_S':        rho_res_S,
        'rho_res_R':        rho_res_R,
        'Emax_v':           emax_v,
        'EC50_V':           ec50_v,
        'Emax_l':           emax_l,
        'EC50_L':           ec50_l,
        'B_max_blood':      b_max_blood,
        'B_max_reservoir':  b_max_res,
        'van_res_fraction': VAN_RES_FRAC_FIXED,
        'lzd_res_fraction': 0.45,
        'f_r_b':            F_R_B_FIXED,
        'f_b_r':            F_B_R_FIXED,
    }

    # --- PK and immune (both fixed) ---
    pk = PharmacokineticModel()
    immune_model = ImmuneResponse(k_immune=K_IMMUNE_FIXED)

    lzd_start = vanco_start + pk.van_duration
    van_func  = pk.concentration_function('vancomycin', total_h, vanco_start)
    lzd_func  = pk.concentration_function('linezolid',  total_h, lzd_start)

    try:
        solution = odeint(
            dual_reservoir_model,
            y0_fixed,
            t_eval,
            args=(params, van_func, lzd_func, immune_model),
            rtol=1e-6, atol=1e-8, mxstep=5000,
        )

        solution = np.clip(solution, 0, None)

        S_b_results.append(solution[:, 0])
        R_b_results.append(solution[:, 1])
        S_res_results.append(solution[:, 2])
        R_res_results.append(solution[:, 3])

        for i, t in enumerate(boxplot_times):
            idx = boxplot_indices[i]
            S_b_box[t].append(np.log10(max(solution[idx, 0], 1.0)))
            R_b_box[t].append(np.log10(max(solution[idx, 1], 1.0)))
            S_res_box[t].append(np.log10(max(solution[idx, 2], 1.0)))
            R_res_box[t].append(np.log10(max(solution[idx, 3], 1.0)))

        successful_runs += 1
    except Exception:
        for key in param_log:
            param_log[key].pop()
        continue

print(f"\nComplete: {successful_runs}/{n_simulations} runs successful.")

# -------------------------------------------------------------------------
# Save parameter log
# -------------------------------------------------------------------------
log_df = pd.DataFrame({
    'Simulation':      range(1, successful_runs + 1),
    'rho_res_S':       param_log['rho_res_S'],
    'rho_res_R':       param_log['rho_res_R'],
    'Emax_v':          param_log['Emax_v'],
    'EC50_V':          param_log['EC50_V'],
    'Emax_l':          param_log['Emax_l'],
    'EC50_L':          param_log['EC50_L'],
    'B_max_blood':     param_log['B_max_blood'],
    'B_max_reservoir': param_log['B_max_reservoir'],
})
log_df.to_csv('other_params_MC_log.csv', index=False)
print(f"Parameter log saved to: other_params_MC_log.csv")
print(log_df.describe())

# -------------------------------------------------------------------------
# Convert to arrays
# -------------------------------------------------------------------------
S_b_results   = np.array(S_b_results)
R_b_results   = np.array(R_b_results)
S_res_results = np.array(S_res_results)
R_res_results = np.array(R_res_results)
emax_v_arr    = np.array(param_log['Emax_v'])

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def percentile_bands(arr):
    return (
        np.percentile(arr, 50, axis=0),
        np.percentile(arr, 25, axis=0),
        np.percentile(arr, 75, axis=0),
        np.percentile(arr,  5, axis=0),
        np.percentile(arr, 95, axis=0),
    )

def log_plot_values(values):
    return np.where(np.asarray(values) < 1.0, 1.0, values)

def add_treatment_lines(ax, vd, ld, le):
    ax.axvline(vd, color='red',      ls='--', lw=1.5, label=f'Vanco start (d{vd:.0f})')
    ax.axvline(ld, color='darkblue', ls='--', lw=1.5, label=f'LZD start (d{ld:.0f})')
    ax.axvline(le, color='black',    ls=':',  lw=1.5, label=f'LZD end (d{le:.0f})')

pk_ref    = PharmacokineticModel()
lzd_start = vanco_start + pk_ref.van_duration
vd = vanco_start_days
ld = lzd_start / 24.0
le = (lzd_start + pk_ref.lzd_duration) / 24.0

plt.style.use('seaborn-v0_8-darkgrid')

# =========================================================================
# Figure 1: 4-panel trajectories coloured by Emax_v
# =========================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axS, axR, axSr, axRr = axes1[0, 0], axes1[0, 1], axes1[1, 0], axes1[1, 1]

for ax in axes1.flat:
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim([0, t_days.max()])

norm = plt.Normalize(EMAX_V_RANGE[0], EMAX_V_RANGE[1])
cmap = cm.viridis

for i in range(successful_runs):
    colour = cmap(norm(emax_v_arr[i]))
    alpha  = 0.25
    for ax, arr in zip([axS, axR, axSr, axRr],
                       [S_b_results, R_b_results, S_res_results, R_res_results]):
        ax.semilogy(t_days, log_plot_values(arr[i]), color=colour, lw=0.6, alpha=alpha)

for ax, results, label, color in zip(
    [axS, axR, axSr, axRr],
    [S_b_results, R_b_results, S_res_results, R_res_results],
    ['Sensitive Blood (S_b)', 'Resistant Blood (R_b)',
     'Sensitive Reservoir (S_res)', 'Resistant Reservoir (R_res)'],
    ['blue', 'orange', 'purple', 'green']
):
    med, p25, p75, p5, p95 = percentile_bands(results)
    ax.semilogy(t_days, log_plot_values(med), color=color, lw=2.5, label='Median', zorder=5)
    ax.fill_between(t_days, log_plot_values(p25), log_plot_values(p75), color=color, alpha=0.35, label='IQR 25–75%')
    ax.fill_between(t_days, log_plot_values(p5),  log_plot_values(p95), color=color, alpha=0.15, label='5–95%')
    add_treatment_lines(ax, vd, ld, le)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_ylabel('CFU/mL', fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

axes1[1, 0].set_xlabel('Time (days)', fontsize=12)
axes1[1, 1].set_xlabel('Time (days)', fontsize=12)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig1.colorbar(sm, ax=axes1.ravel().tolist(), shrink=0.6, pad=0.02)
cbar.set_label('$E_{max,v}$  [h$^{-1}$]', fontsize=11)

plt.suptitle(
    'Other-Parameters Monte Carlo — Individual Trajectories Coloured by $E_{max,v}$\n'
    '(Fixed ICs, transfer rates, immune, and growth; PD, reservoir growth, capacity varied)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'OtherParams_trajectories_4panel.png'), dpi=150, bbox_inches='tight')
print('Saved: OtherParams_trajectories_4panel.png')
plt.close()

# =========================================================================
# Figure 2: 4-panel percentile bands
# =========================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

for ax in axes2.flat:
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim([0, t_days.max()])

for ax, results, label, color in zip(
    axes2.flat,
    [S_b_results, R_b_results, S_res_results, R_res_results],
    ['Sensitive Blood (S_b)', 'Resistant Blood (R_b)',
     'Sensitive Reservoir (S_res)', 'Resistant Reservoir (R_res)'],
    ['blue', 'orange', 'purple', 'green']
):
    med, p25, p75, p5, p95 = percentile_bands(results)
    ax.semilogy(t_days, log_plot_values(med),  color=color, lw=2.5, label='Median')
    ax.fill_between(t_days, log_plot_values(p25), log_plot_values(p75), color=color, alpha=0.35, label='IQR 25–75%')
    ax.fill_between(t_days, log_plot_values(p5),  log_plot_values(p95), color=color, alpha=0.15, label='5–95%')
    add_treatment_lines(ax, vd, ld, le)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_ylabel('CFU/mL', fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)

axes2[1, 0].set_xlabel('Time (days)', fontsize=12)
axes2[1, 1].set_xlabel('Time (days)', fontsize=12)
plt.suptitle(
    'Other-Parameters Monte Carlo — Percentile Bands\n'
    '(Fixed ICs, transfer rates, immune, and growth; 7 parameters varied log-normally)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'OtherParams_percentile_4panel.png'), dpi=150, bbox_inches='tight')
print('Saved: OtherParams_percentile_4panel.png')
plt.close()

# =========================================================================
# Figure 3: Boxplots at key timepoints
# =========================================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
colors = {'S_b': '#3498db', 'R_b': '#ff8c42', 'S_res': '#9b59b6', 'R_res': '#2ecc71'}
labels_bp = [f'{d:.1f}d' for d in boxplot_times_days]

for ax in axes3.flat:
    ax.tick_params(axis='both', which='major', labelsize=12)

def make_boxplot(ax, box_data, color, title):
    bp = ax.boxplot(
        [box_data[t] for t in boxplot_times],
        labels=labels_bp, patch_artist=True, widths=0.6
    )
    for patch in bp['boxes']:
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('log$_{10}$ CFU/mL', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

make_boxplot(axes3[0, 0], S_b_box,   colors['S_b'],   'Sensitive Blood (log₁₀ CFU/mL)')
make_boxplot(axes3[0, 1], R_b_box,   colors['R_b'],   'Resistant Blood (log₁₀ CFU/mL)')
make_boxplot(axes3[1, 0], S_res_box, colors['S_res'], 'Sensitive Reservoir (log₁₀ CFU/mL)')
make_boxplot(axes3[1, 1], R_res_box, colors['R_res'], 'Resistant Reservoir (log₁₀ CFU/mL)')

plt.suptitle('Other-Parameters Monte Carlo — Boxplot Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'OtherParams_boxplots.png'), dpi=150, bbox_inches='tight')
print('Saved: OtherParams_boxplots.png')
plt.close()

# =========================================================================
# Figure 4: Parameter sampling distributions (2x4 grid)
# =========================================================================
fig4, axes4 = plt.subplots(2, 4, figsize=(18, 7))

hist_specs = [
    # (row, col, data_key, xlabel, title, log_scale)
    (0, 0, 'rho_res_S',       '$\\rho_{res,S}$  [h$^{-1}$]', 'Reservoir Growth (S)',  False),
    (0, 1, 'Emax_v',          '$E_{max,v}$  [h$^{-1}$]',     'Vanco $E_{max}$',       False),
    (0, 2, 'EC50_V',          'EC$_{50,V}$  [mg/L]',         'Vanco EC$_{50}$',       False),
    (0, 3, 'Emax_l',          '$E_{max,l}$  [h$^{-1}$]',     'LZD $E_{max}$',         False),
    (1, 0, 'EC50_L',          'EC$_{50,L}$  [mg/L]',         'LZD EC$_{50}$',         False),
    (1, 1, 'B_max_blood',     'log$_{10}$ B$_{max,blood}$',  'Blood Capacity',        True),
    (1, 2, 'B_max_reservoir', 'log$_{10}$ B$_{max,res}$',    'Reservoir Capacity',    True),
]

hist_colors = ['mediumpurple', 'steelblue', 'coral', 'seagreen',
               'mediumpurple', 'steelblue', 'coral']

for (row, col, key, xlabel, title, use_log), hc in zip(hist_specs, hist_colors):
    ax = axes4[row, col]
    data = np.array(param_log[key])
    if use_log:
        data = np.log10(data)
    ax.hist(data, bins=20, color=hc, alpha=0.8, edgecolor='white')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

axes4[1, 3].set_visible(False)

plt.suptitle('Other-Parameters Monte Carlo — Sampled Parameter Distributions',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'OtherParams_sampling_distributions.png'), dpi=150, bbox_inches='tight')
print('Saved: OtherParams_sampling_distributions.png')
plt.close()

print('\nAll figures saved to Figures/OtherParams_*.png')
