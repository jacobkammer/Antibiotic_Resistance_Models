"""
run_MC_transfer_rates.py
========================
Monte Carlo exploration of reservoir-to-blood and blood-to-reservoir transfer
rate parameters (f_r_b, f_b_r).

Design differences from run_MC_model.py:
  - Initial conditions are FIXED (constant across all simulations) to isolate
    the effect of transfer rates from IC variability.
  - Only f_r_b and f_b_r are varied; all other parameters are held at their
    nominal values.
  - f_r_b and f_b_r are sampled independently from log-uniform distributions
    spanning four orders of magnitude, covering the full plausible biological
    range (~1e-6 to 1e-2 h⁻¹ for f_r_b; Kostakioti et al. 2013).
  - Output figures are saved to Figures/TransferRate_*.png.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — required for headless run
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import importlib.util
import os
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


# --- Load model ---
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, "model.transfer_rate_exploration.py")

spec = importlib.util.spec_from_file_location("model.transfer_rate_exploration", MODEL_PATH)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

fig_dir = os.path.join(THIS_DIR, "Figures")
os.makedirs(fig_dir, exist_ok=True)

PharmacokineticModel = model_module.PharmacokineticModel
ImmuneResponse       = model_module.ImmuneResponse
dual_reservoir_model = model_module.dual_reservoir_model

print('Transfer-Rate Monte Carlo Exploration')
print('Using: model.transfer_rate_exploration.py')
print('=' * 70)

# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
np.random.seed(42)

# -------------------------------------------------------------------------
# Fixed simulation settings
# -------------------------------------------------------------------------
resistance_factor = 0.25
rho_S  = 0.30
rho_R  = (1 - resistance_factor) * rho_S

n_simulations = 100
total_h       = 1344   # 56 days
vanco_start   = 504    # h (day 21)

print(f"Resistance factor: {resistance_factor}  →  rho_R = {rho_R:.4f}  (rho_S = {rho_S})")
print(f"Simulations: {n_simulations}  |  Total time: {total_h} h ({total_h/24:.0f} days)")
print(f"Vancomycin start: {vanco_start} h (day {vanco_start/24:.0f})")
print()

# -------------------------------------------------------------------------
# Fixed nominal parameters (all except transfer rates)
# -------------------------------------------------------------------------
nominal_params = {
    'B_max_blood':      5e5,
    'B_max_reservoir':  1e7,
    'rho_S':            rho_S,
    'rho_R':            rho_R,
    'rho_res_S':        0.07,
    'rho_res_R':        0.07 * (1 - resistance_factor),
    'Emax_v':  0.40,
    'EC50_V':  1.5,
    'Emax_l':  0.30,   # h⁻¹ (= rho_S; net effect model)
    'EC50_L':  1.0,
    'biofilm_factor': 100.0,
    # f_r_b and f_b_r are NOT set here — varied per simulation below
}

# -------------------------------------------------------------------------
# Transfer rate sampling range (log-uniform)
# -------------------------------------------------------------------------
# f_r_b (reservoir → blood): plausible range 1e-6 to 1e-2 h⁻¹
# f_b_r (blood → reservoir): plausible range 1e-7 to 1e-3 h⁻¹
# Sampled independently; log-uniform gives equal density per decade.
F_R_B_MIN, F_R_B_MAX = 1e-6, 1e-2   # h⁻¹
F_B_R_MIN, F_B_R_MAX = 1e-7, 1e-3   # h⁻¹

print(f"f_r_b range: {F_R_B_MIN:.0e} – {F_R_B_MAX:.0e} h⁻¹  (log-uniform)")
print(f"f_b_r range: {F_B_R_MIN:.0e} – {F_B_R_MAX:.0e} h⁻¹  (log-uniform)")
print()

# -------------------------------------------------------------------------
# FIXED initial conditions (same for every simulation)
# -------------------------------------------------------------------------
y0_fixed = [5.0, 1.0, 1e4, 1e3]   # [S_b, R_b, S_res, R_res] CFU/mL
print(f"Fixed ICs: S_b={y0_fixed[0]:.0f}, R_b={y0_fixed[1]:.0f}, "
      f"S_res={y0_fixed[2]:.0e}, R_res={y0_fixed[3]:.0e} CFU/mL")
print()

# -------------------------------------------------------------------------
# Time grid
# -------------------------------------------------------------------------
t_eval = np.linspace(0, total_h, 400)   # 400 pts sufficient for smooth curves at 56 days
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

f_r_b_log  = []   # sampled f_r_b values for each simulation
f_b_r_log  = []   # sampled f_b_r values

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

    # --- Sample transfer rates log-uniformly ---
    f_r_b = 10 ** np.random.uniform(np.log10(F_R_B_MIN), np.log10(F_R_B_MAX))
    f_b_r = 10 ** np.random.uniform(np.log10(F_B_R_MIN), np.log10(F_B_R_MAX))

    f_r_b_log.append(f_r_b)
    f_b_r_log.append(f_b_r)

    # --- Build parameter dict (nominal + sampled transfer rates) ---
    params = dict(nominal_params)
    params['f_r_b'] = f_r_b
    params['f_b_r'] = f_b_r

    # --- PK and immune (fixed, no variability) ---
    pk = PharmacokineticModel()
    immune_model = ImmuneResponse(k_immune=0.12)

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

        S_b_results.append(np.clip(solution[:, 0], 0.1, None))
        R_b_results.append(np.clip(solution[:, 1], 0.1, None))
        S_res_results.append(np.clip(solution[:, 2], 0.1, None))
        R_res_results.append(np.clip(solution[:, 3], 0.1, None))

        for i, t in enumerate(boxplot_times):
            idx = boxplot_indices[i]
            S_b_box[t].append(np.log10(max(solution[idx, 0], 0.1)))
            R_b_box[t].append(np.log10(max(solution[idx, 1], 0.1)))
            S_res_box[t].append(np.log10(max(solution[idx, 2], 0.1)))
            R_res_box[t].append(np.log10(max(solution[idx, 3], 0.1)))

        successful_runs += 1
    except Exception:
        # Pad storage so indexing stays aligned
        f_r_b_log.pop(); f_b_r_log.pop()
        continue

print(f"\nComplete: {successful_runs}/{n_simulations} runs successful.")

# -------------------------------------------------------------------------
# Save transfer rate log
# -------------------------------------------------------------------------
tr_df = pd.DataFrame({
    'Simulation': range(1, successful_runs + 1),
    'f_r_b': f_r_b_log,
    'f_b_r': f_b_r_log,
    'log10_f_r_b': np.log10(f_r_b_log),
    'log10_f_b_r': np.log10(f_b_r_log),
})
tr_df.to_csv('transfer_rate_MC_log.csv', index=False)
print(f"Transfer rates saved to: transfer_rate_MC_log.csv")
print(tr_df[['f_r_b', 'f_b_r']].describe())

# -------------------------------------------------------------------------
# Convert to arrays
# -------------------------------------------------------------------------
S_b_results   = np.array(S_b_results)
R_b_results   = np.array(R_b_results)
S_res_results = np.array(S_res_results)
R_res_results = np.array(R_res_results)
f_r_b_arr     = np.array(f_r_b_log)
f_b_r_arr     = np.array(f_b_r_log)

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
# Figure 1: 4-panel percentile bands (coloured by log10 f_r_b)
# =========================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axS, axR, axSr, axRr = axes1[0, 0], axes1[0, 1], axes1[1, 0], axes1[1, 1]

for ax in axes1.flat:
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim([0, t_days.max()])

# Colour each simulation trajectory by log10(f_r_b)
norm  = plt.Normalize(np.log10(F_R_B_MIN), np.log10(F_R_B_MAX))
cmap  = cm.viridis
log10_frb = np.log10(f_r_b_arr)

for i in range(successful_runs):
    colour = cmap(norm(log10_frb[i]))
    alpha  = 0.25
    for ax, arr in zip([axS, axR, axSr, axRr],
                       [S_b_results, R_b_results, S_res_results, R_res_results]):
        ax.semilogy(t_days, arr[i], color=colour, lw=0.6, alpha=alpha)

# Overlay median and IQR
for ax, results, label, color in zip(
    [axS, axR, axSr, axRr],
    [S_b_results, R_b_results, S_res_results, R_res_results],
    ['Sensitive Blood (S_b)', 'Resistant Blood (R_b)',
     'Sensitive Reservoir (S_res)', 'Resistant Reservoir (R_res)'],
    ['blue', 'orange', 'purple', 'green']
):
    med, p25, p75, p5, p95 = percentile_bands(results)
    ax.semilogy(t_days, med, color=color, lw=2.5, label='Median', zorder=5)
    ax.fill_between(t_days, p25, p75, color=color, alpha=0.35, label='IQR 25–75%')
    ax.fill_between(t_days, p5,  p95, color=color, alpha=0.15, label='5–95%')
    add_treatment_lines(ax, vd, ld, le)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_ylabel('CFU/mL', fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

axes1[1, 0].set_xlabel('Time (days)', fontsize=12)
axes1[1, 1].set_xlabel('Time (days)', fontsize=12)

# Colourbar for f_r_b
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig1.colorbar(sm, ax=axes1.ravel().tolist(), shrink=0.6, pad=0.02)
cbar.set_label('log$_{10}$(f$_{r\\to b}$)  [h$^{-1}$]', fontsize=11)

plt.suptitle(
    'Transfer-Rate Monte Carlo — Individual Trajectories Coloured by $f_{r\\to b}$\n'
    '(Fixed ICs; all other parameters nominal)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'TransferRate_trajectories_4panel.png'), dpi=150, bbox_inches='tight')
print('Saved: TransferRate_trajectories_4panel.png')
plt.close()

# =========================================================================
# Figure 2: 4-panel percentile bands (standard shaded bands)
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
    ax.semilogy(t_days, med,  color=color, lw=2.5, label='Median')
    ax.fill_between(t_days, p25, p75, color=color, alpha=0.35, label='IQR 25–75%')
    ax.fill_between(t_days, p5,  p95, color=color, alpha=0.15, label='5–95%')
    add_treatment_lines(ax, vd, ld, le)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_ylabel('CFU/mL', fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)

axes2[1, 0].set_xlabel('Time (days)', fontsize=12)
axes2[1, 1].set_xlabel('Time (days)', fontsize=12)
plt.suptitle(
    'Transfer-Rate Monte Carlo — Percentile Bands\n'
    '(Fixed ICs; f$_{r\\to b}$ and f$_{b\\to r}$ varied log-uniformly)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'TransferRate_percentile_4panel.png'), dpi=150, bbox_inches='tight')
print('Saved: TransferRate_percentile_4panel.png')
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

plt.suptitle('Transfer-Rate Monte Carlo — Boxplot Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'TransferRate_boxplots.png'), dpi=150, bbox_inches='tight')
print('Saved: TransferRate_boxplots.png')
plt.close()

# =========================================================================
# Figure 4: Transfer rate sampling distribution
# =========================================================================
fig4, (ax_frb, ax_fbr, ax_scatter) = plt.subplots(1, 3, figsize=(15, 4))

ax_frb.hist(np.log10(f_r_b_arr), bins=20, color='steelblue', alpha=0.8, edgecolor='white')
ax_frb.set_xlabel('log$_{10}$(f$_{r\\to b}$)  [h$^{-1}$]', fontsize=11)
ax_frb.set_ylabel('Count', fontsize=11)
ax_frb.set_title('Reservoir → Blood Rate', fontsize=12)
ax_frb.grid(True, alpha=0.3)

ax_fbr.hist(np.log10(f_b_r_arr), bins=20, color='coral', alpha=0.8, edgecolor='white')
ax_fbr.set_xlabel('log$_{10}$(f$_{b\\to r}$)  [h$^{-1}$]', fontsize=11)
ax_fbr.set_ylabel('Count', fontsize=11)
ax_fbr.set_title('Blood → Reservoir Rate', fontsize=12)
ax_fbr.grid(True, alpha=0.3)

sc = ax_scatter.scatter(
    np.log10(f_r_b_arr), np.log10(f_b_r_arr),
    c=np.arange(successful_runs), cmap='plasma', alpha=0.7, s=40
)
ax_scatter.set_xlabel('log$_{10}$(f$_{r\\to b}$)  [h$^{-1}$]', fontsize=11)
ax_scatter.set_ylabel('log$_{10}$(f$_{b\\to r}$)  [h$^{-1}$]', fontsize=11)
ax_scatter.set_title('Joint Sampling Distribution', fontsize=12)
ax_scatter.grid(True, alpha=0.3)
fig4.colorbar(sc, ax=ax_scatter, label='Simulation index')

plt.suptitle('Transfer Rate Sampling — Monte Carlo Input Distribution',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'TransferRate_sampling_distribution.png'), dpi=150, bbox_inches='tight')
print('Saved: TransferRate_sampling_distribution.png')
plt.close()

print('\nAll figures saved to Figures/TransferRate_*.png')
