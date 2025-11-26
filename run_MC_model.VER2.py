import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
import os
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# --- Dynamically load model.Ver10.py ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, 'model.Ver10.py')

spec = importlib.util.spec_from_file_location('model.Ver10.py', MODEL_PATH)
no_bm_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(no_bm_model)

PharmacokineticModel = no_bm_model.PharmacokineticModel
ImmuneResponse = no_bm_model.ImmuneResponse
dual_reservoir_model = no_bm_model.dual_reservoir_model

print('Starting Monte Carlo for dual-reservoir model (model.Ver10.py)')
print('=' * 70)

# Reproducibility
np.random.seed(42)

# --- Base parameters (aligned with model.Ver10.py)

base_params = {
    # Blood carrying capacity and reservoir capacity
    'B_max_blood': 4e12,
    'K_res_total': 1e6,
    # Blood bacteria
    'rho_S': 1.47,
    'rho_R': 1.47,
    'delta': 0.179,
    # Reservoir bacteria
    'rho_res_S': 2,
    'rho_res_R': 1.98,#fitness cost for being resistant
    'delta_res': 0.06,
    # PD
    'Emax_v': 1.74,
    'Emax_l': 1.97,
    'EC50_V': 0.245,
    'EC50_L': 0.56,
    # Exchange (same for S and R)
    'f_r_b': 1e-18,    # reservoir -> blood
    'f_b_r': 1e-20,  # blood -> reservoir
}

# --- Simulation settings ---
n_simulations = 1000
cv = 0.10  # coefficient of variation
total_h = 2500
vanco_start = 504 # 42 days

# time grid
t_eval = np.linspace(0, total_h, 800)
t_days = t_eval / 24.0  # Convert to days for plotting
vanco_start_days = vanco_start / 24.0

# --- Storage ---
S_b_results = []
R_b_results = []
S_res_results = []
R_res_results = []
N_results = []
# Storage for initial conditions
initial_conditions_log = []
# Boxplot times (align to new grid) - in hours for indexing
boxplot_times = [12, 24, 72, 168, 336, 504, 720, 1008]
boxplot_times_days = [t / 24.0 for t in boxplot_times]  # Convert to days for labels
boxplot_indices = [np.argmin(np.abs(t_eval - t)) for t in boxplot_times]

S_b_box = {t: [] for t in boxplot_times}
R_b_box = {t: [] for t in boxplot_times}
S_res_box = {t: [] for t in boxplot_times}
R_res_box = {t: [] for t in boxplot_times}
N_box = {t: [] for t in boxplot_times}

print(f"\nRunning {n_simulations} simulations with {cv*100:.0f}% parameter variation...")
print('-' * 40)

successful_runs = 0
for sim in range(n_simulations):
    if sim % 20 == 0:
        print(f"Progress: {sim}/{n_simulations} simulations completed...")

    # Sample parameters (log-normal for positive)
    params = {}
    for key, value in base_params.items():
        if key in ['EC50_V', 'EC50_L', 'f_r_b', 'f_b_r']:
            params[key] = np.random.lognormal(np.log(value), cv/2)
        else:
            params[key] = np.random.lognormal(np.log(value), cv)

    # Initialize models
    pk = PharmacokineticModel()
    immune_model = ImmuneResponse(N0=5000)

    # PK variability (mild)
    pk.van_dose = max(100, np.random.normal(pk.van_dose, pk.van_dose * cv/4))
    pk.lzd_dose = max(100, np.random.normal(pk.lzd_dose, pk.lzd_dose * cv/4))

    lzd_start = vanco_start + pk.van_duration

    van_func = pk.concentration_function('vancomycin', total_h, vanco_start)
    lzd_func = pk.concentration_function('linezolid', total_h, lzd_start)
    # Each of the 1000 Monte Carlo simulations starts
    # with a different random initial bacterial load
    # in the reservoir, uniformly distributed between 10-100 CFU/mL.
    S_res_init = np.random.uniform(10, 100)
    R_res_init = np.random.uniform(10, 100)
    
    # Initial conditions [S_b, R_b, S_res, R_res, N]
    y0_base = [0, 0, S_res_init, R_res_init, immune_model.N0]
    
    
    
    # normal variability to neutrophils (as before).
    y0 = y0_base.copy()
    # Ensure bacteria values are at least 1 CFU/mL for numerical stability
    y0[:4] = [max(1, val) for val in y0[:4]]
    # Apply normal variability only to neutrophils (keep a minimum of 500)
    y0[4] = max(500, np.random.normal(y0_base[4], y0_base[4] * cv / 4))

    # Log-normal sampling for bacteria, normal for N0
    # Log initial conditions for this simulation
    initial_conditions_log.append({
        'Simulation': sim + 1,
        'S_b_init': y0[0],
        'R_b_init': y0[1],
        'S_res_init': y0[2],
        'R_res_init': y0[3],
        'N_init': y0[4],
        'S_res_base': S_res_init,
        'R_res_base': R_res_init
    })
    
    try:
        solution = odeint(
            dual_reservoir_model,
            y0,
            t_eval,
            args=(params, van_func, lzd_func, immune_model),
            rtol=1e-6,
            atol=1e-9,
            mxstep=5000,
        )

        # Store (clip bacteria for log plots only later)
        S_b_results.append(np.clip(solution[:, 0], 1, None))
        R_b_results.append(np.clip(solution[:, 1], 1, None))
        S_res_results.append(np.clip(solution[:, 2], 1, None))
        R_res_results.append(np.clip(solution[:, 3], 1, None))
        N_results.append(solution[:, 4])

        # Boxplot collections (log10 for bacteria)


        for i, t in enumerate(boxplot_times):
            idx = boxplot_indices[i]
            S_b_box[t].append(np.log10(max(solution[idx, 0], 1)))
            R_b_box[t].append(np.log10(max(solution[idx, 1], 1)))
            S_res_box[t].append(np.log10(max(solution[idx, 2], 1)))
            R_res_box[t].append(np.log10(max(solution[idx, 3], 1)))
            N_box[t].append(solution[idx, 4])
        successful_runs += 1
    except Exception:
        # skip failed simulations
        continue

print(f"\nSimulation complete! {successful_runs}/{n_simulations} runs successful.")
# Save initial conditions to CSV
ic_df = pd.DataFrame(initial_conditions_log)
ic_df.to_csv('initial_conditions_log.csv', index=False)
print(f"\nInitial conditions saved to: initial_conditions_log.csv")

# Display summary statistics
print('\n' + '=' * 70)
print('INITIAL CONDITIONS SUMMARY')
print('=' * 70)
print(ic_df.describe())
# Initial Conditions Visualizations
print('Generating initial conditions plots...')

# S_res from uniform dist
fig_ic1, ax_ic1 = plt.subplots(figsize=(12, 7))
ax_ic1.hist(ic_df['S_res_base'], bins=30, color='mediumpurple', alpha=0.7, edgecolor='black', linewidth=1.5)
ax_ic1.axvline(ic_df['S_res_base'].median(), color='indigo', linestyle='--', linewidth=2.5, 
               label=f'Median: {ic_df["S_res_base"].median():.1f}')
ax_ic1.set_xlabel('S_res Initial Value (CFU/mL)', fontsize=12)
ax_ic1.set_ylabel('Frequency', fontsize=12)
ax_ic1.set_title('S_res Initial Conditions - Monte Carlo', fontsize=14, fontweight='bold')
ax_ic1.grid(True, alpha=0.3, linestyle='--')
ax_ic1.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig('Figures/Ver10_IC_S_res_uniform.png', dpi=300, bbox_inches='tight')
print('Saved: Ver10_IC_S_res_uniform.png')
plt.show()

# R_res from uniform dist
fig_ic2, ax_ic2 = plt.subplots(figsize=(12, 7))
ax_ic2.hist(ic_df['R_res_base'], bins=30, color='limegreen', alpha=0.7, edgecolor='black', linewidth=1.5)
ax_ic2.axvline(ic_df['R_res_base'].median(), color='forestgreen', linestyle='--', linewidth=2.5, 
               label=f'Median: {ic_df["R_res_base"].median():.1f}')
ax_ic2.set_xlabel('R_res Initial Value (CFU/mL)', fontsize=12)
ax_ic2.set_ylabel('Frequency', fontsize=12)
ax_ic2.set_title('R_res Initial Conditions - Monte Carlo', fontsize=14, fontweight='bold')
ax_ic2.grid(True, alpha=0.3, linestyle='--')
ax_ic2.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig('Figures/Ver10_IC_R_res_uniform.png', dpi=300, bbox_inches='tight')
print('Saved: Ver10_IC_R_res_uniform.png')
plt.show()


# Neutrophils Initial
fig_ic5, ax_ic5 = plt.subplots(figsize=(12, 7))
ax_ic5.hist(ic_df['N_init'], bins=30, color='seagreen', alpha=0.7, edgecolor='black', linewidth=1.5)
ax_ic5.axvline(ic_df['N_init'].median(), color='black', linestyle='--', linewidth=2.5, 
               label=f'Median: {ic_df["N_init"].median():.0f}')
ax_ic5.set_xlabel('Neutrophils Initial Value (cells/μL)', fontsize=12)
ax_ic5.set_ylabel('Frequency', fontsize=12)
ax_ic5.set_title('Neutrophil Initial Conditions - Monte Carlo', fontsize=14, fontweight='bold')
ax_ic5.grid(True, alpha=0.3, linestyle='--')
ax_ic5.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig('Figures/Ver10_IC_neutrophils.png', dpi=300, bbox_inches='tight')
print('Saved: Ver10_IC_neutrophils.png')
plt.show()

print('\nGenerating bacterial population visualizations...')
# Convert lists to arrays
S_b_results = np.array(S_b_results)
R_b_results = np.array(R_b_results)
S_res_results = np.array(S_res_results)
R_res_results = np.array(R_res_results)
N_results = np.array(N_results)
print(N_results.shape)

# Find the absolute maximum neutrophil value and its time point (across all simulations)
max_flat_index = np.argmax(N_results)
sim_idx, time_idx = np.unravel_index(max_flat_index, N_results.shape)
max_time_days = t_days[time_idx]
max_value = N_results[sim_idx, time_idx]
print(f"Maximum neutrophils: {max_value:,.0f} cells/μL at time: {max_time_days:.2f} days (simulation {sim_idx}, index {time_idx})")
# Find indices where neutrophils > 25000
""" indices_above_25000 = np.where(N_results > 25000)[0]
#print(indices_above_25000)
#determine the first index when neutrophils are > 25000
first_index = indices_above_25000[0] if len(indices_above_25000) > 0 else None
#print(first_index)
#Find the corresponding time when neutrophils > 25000
max_time = t_eval[first_index]
print(f"Time when nonBMstim neutrophils are greater than 25000: {max_time} h") """

print('\nGenerating visualizations...')
plt.style.use('seaborn-v0_8-darkgrid')

# Helper to compute percentile bands
def percentile_bands(arr):
    return ( 
        np.percentile(arr, 50, axis=0),
        np.percentile(arr, 25, axis=0),
        np.percentile(arr, 75, axis=0),
        np.percentile(arr, 5, axis=0),
        np.percentile(arr, 95, axis=0),
    )

lzd_start = vanco_start + pk.van_duration  # from last pk used; only for plotting ref
lzd_start_days = lzd_start / 24.0

# --- Plot 1: Sensitive Blood (S_b) ---
S_med, S_p25, S_p75, S_p5, S_p95 = percentile_bands(S_b_results)
figS, axS = plt.subplots(figsize=(12, 7))
axS.semilogy(t_days, S_med, color='blue', linewidth=2.5, label='S_b Median')
axS.fill_between(t_days, S_p25, S_p75, color='blue', alpha=0.35, label='S_b 25-75%')
axS.fill_between(t_days, S_p5, S_p95, color='blue', alpha=0.18, label='S_b 5-95%')
axS.axvline(vanco_start_days, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Vancomycin Start')
axS.axvline(lzd_start_days, color='darkblue', linestyle='--', alpha=0.7, linewidth=2, label='Linezolid Start')
axS.set_xlabel('Time (days)')
axS.set_ylabel('Sensitive Blood Bacteria (CFU/mL)')
axS.set_title('Sensitive Blood Bacteria - Monte Carlo ')
axS.grid(True, which='both', ls='-', lw=0.3, alpha=0.3)
axS.legend(loc='best')
axS.set_ylim([1e0, 1e13])
plt.tight_layout()
plt.savefig('Figures/Ver_10_S_b.png', dpi=300, bbox_inches='tight')
print('Saved: Ver_10_S_b.png')
plt.show()

# --- Plot 2: Resistant Blood (R_b) ---
R_med, R_p25, R_p75, R_p5, R_p95 = percentile_bands(R_b_results)
figR, axR = plt.subplots(figsize=(12, 7))
axR.semilogy(t_days, R_med, color='orange', linewidth=2.5, label='R_b Median')
axR.fill_between(t_days, R_p25, R_p75, color='orange', alpha=0.35, label='R_b 25-75%')
axR.fill_between(t_days, R_p5, R_p95, color='orange', alpha=0.18, label='R_b 5-95%')
axR.axvline(vanco_start_days, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Vancomycin Start')
axR.axvline(lzd_start_days, color='darkblue', linestyle='--', alpha=0.7, linewidth=2, label='Linezolid Start')
axR.set_xlabel('Time (days)')
axR.set_ylabel('Resistant Blood Bacteria (CFU/mL)')
axR.set_title('Resistant Blood Bacteria - Monte Carlo')
axR.grid(True, which='both', ls='-', lw=0.3, alpha=0.3)
axR.legend(loc='best')
axR.set_ylim([1e0, 1e13])
plt.tight_layout()
plt.savefig('Figures/Ver10_R_b.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Plot 3: Sensitive Reservoir (S_res) ---
Sr_med, Sr_p25, Sr_p75, Sr_p5, Sr_p95 = percentile_bands(S_res_results)
figSr, axSr = plt.subplots(figsize=(12, 7))
axSr.semilogy(t_days, Sr_med, color='purple', linewidth=2.5, label='S_res Median')
axSr.fill_between(t_days, Sr_p25, Sr_p75, color='purple', alpha=0.35, label='S_res 25-75%')
axSr.fill_between(t_days, Sr_p5, Sr_p95, color='purple', alpha=0.18, label='S_res 5-95%')
axSr.axvline(vanco_start_days, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Vancomycin Start')
axSr.axvline(lzd_start_days, color='darkblue', linestyle='--', alpha=0.7, linewidth=2, label='Linezolid Start')
axSr.set_xlabel('Time (days)')
axSr.set_ylabel('Sensitive Reservoir Bacteria (CFU/mL)')
axSr.set_title('Sensitive Reservoir Bacteria - Monte Carlo ')
axSr.grid(True, which='both', ls='-', lw=0.3, alpha=0.3)
axSr.legend(loc='best')
axSr.set_ylim([1e0, 1e6])
plt.tight_layout()
plt.savefig('Figures/Ver10_S_res.png', dpi=300, bbox_inches='tight')
print('  Saved: Ver10_S_res.png')
plt.show()

# --- Plot 4: Resistant Reservoir (R_res) ---
Rr_med, Rr_p25, Rr_p75, Rr_p5, Rr_p95 = percentile_bands(R_res_results)
figRr, axRr = plt.subplots(figsize=(12, 7))
axRr.semilogy(t_days, Rr_med, color='green', linewidth=2.5, label='R_res Median')
axRr.fill_between(t_days, Rr_p25, Rr_p75, color='green', alpha=0.35, label='R_res 25-75%')
axRr.fill_between(t_days, Rr_p5, Rr_p95, color='green', alpha=0.18, label='R_res 5-95%')
axRr.axvline(vanco_start_days, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Vancomycin Start')
axRr.axvline(lzd_start_days, color='darkblue', linestyle='--', alpha=0.7, linewidth=2, label='Linezolid Start')
axRr.set_xlabel('Time (days)')
axRr.set_ylabel('Resistant Reservoir Bacteria (CFU/mL)')
axRr.set_title('Resistant Reservoir Bacteria - Monte Carlo')
axRr.grid(True, which='both', ls='-', lw=0.3, alpha=0.3)
axRr.legend(loc='best')
axRr.set_ylim([1e0, 1e6])
plt.tight_layout()
plt.savefig('Figures/Ver10_R_res.png', dpi=300, bbox_inches='tight')
print(' Saved: Ver10_R_res.png')
plt.show()

# --- Plot 5: Neutrophils ---
fig3, ax3 = plt.subplots(figsize=(12, 7))
N_med, N_p25, N_p75, N_p5, N_p95 = percentile_bands(N_results)
ax3.plot(t_days, N_med, color='darkgreen', linewidth=2.5, label='Median')
ax3.fill_between(t_days, N_p25, N_p75, color='darkgreen', alpha=0.35, label='25-75%')
ax3.fill_between(t_days, N_p5, N_p95, color='darkgreen', alpha=0.18, label='5-95%')
ax3.axvline(vanco_start_days, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Vancomycin Start')
ax3.axvline(lzd_start_days, color='darkblue', linestyle='--', alpha=0.7, linewidth=2, label='Linezolid Start')
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Neutrophils (cells/μL)')
ax3.set_title('Neutrophil Dynamics - Monte Carlo')
ax3.grid(True, which='both', ls='-', lw=0.3, alpha=0.3)
ax3.legend(loc='best')
ax3.set_ylim([0, max(N_p95)*1.1])
plt.tight_layout()
plt.savefig('Figures/Ver10_neutrophils.png', dpi=300, bbox_inches='tight')
print('  Saved: Ver10_neutrophils.png')
plt.show()

# --- Boxplots at key times ---
fig4, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = {'S_b': '#3498db', 'R_b': '#ff8c42', 'S_res': '#9b59b6', 'R_res': '#2ecc71'}

# Blood S
ax = axes[0, 0]
bp = ax.boxplot([S_b_box[t] for t in boxplot_times], labels=[f'{t_d:.1f}d' for t_d in boxplot_times_days],
                patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor(colors['S_b']); patch.set_alpha(0.7)
ax.set_title('Sensitive Blood Bacteria (log10 CFU/mL)')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([-1, 13])

# Blood R
ax = axes[0, 1]
bp = ax.boxplot([R_b_box[t] for t in boxplot_times], labels=[f'{t_d:.1f}d' for t_d in boxplot_times_days],
                patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor(colors['R_b']); patch.set_alpha(0.7)
ax.set_title('Resistant Blood Bacteria (log10 CFU/mL)')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([-1, 13])

# Reservoir S
ax = axes[0, 2]
bp = ax.boxplot([S_res_box[t] for t in boxplot_times], labels=[f'{t_d:.1f}d' for t_d in boxplot_times_days],
                patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor(colors['S_res']); patch.set_alpha(0.7)
ax.set_title('Sensitive Reservoir Bacteria (log10 CFU/mL)')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([-1, 6])

# Reservoir R
ax = axes[1, 0]
bp = ax.boxplot([R_res_box[t] for t in boxplot_times], labels=[f'{t_d:.1f}d' for t_d in boxplot_times_days],
                patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor(colors['R_res']); patch.set_alpha(0.7)
ax.set_title('Resistant Reservoir Bacteria (log10 CFU/mL)')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([-1, 6])

# Neutrophils
ax = axes[1, 1]
bp = ax.boxplot([N_box[t] for t in boxplot_times], labels=[f'{t_d:.1f}d' for t_d in boxplot_times_days],
                patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor('green'); patch.set_alpha(0.7)
ax.set_title('Neutrophils (cells/μL)')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([0, max([max(v) if v else 0 for v in N_box.values()]) * 1.1 or 1])

# Hide unused subplot
axes[1, 2].axis('off')

plt.suptitle('Monte Carlo Summary', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/Ver10_boxplots.png', dpi=300, bbox_inches='tight')
print('Saved: Ver10_boxplots.png')
plt.show()

# --- Summary statistics ---
""" print('\n' + '=' * 60)
print('SUMMARY STATISTICS')
print('=' * 60)

for t in boxplot_times:
    print(f"\nTime = {t} hours:\n" + '-'*40)
    if S_b_box[t]:
        print('  Sensitive Blood Bacteria (log10 CFU/mL):')
        print(f"    • Median: {np.median(S_b_box[t]):.2f}")
        print(f"    • Mean: {np.mean(S_b_box[t]):.2f}")
        print(f"    • IQR: [{np.percentile(S_b_box[t],25):.2f}, {np.percentile(S_b_box[t],75):.2f}]")
        print(f"    • Range: [{np.min(S_b_box[t]):.2f}, {np.max(S_b_box[t]):.2f}]")
    if R_b_box[t]:
        print('\n  Resistant Blood Bacteria (log10 CFU/mL):')
        print(f"    • Median: {np.median(R_b_box[t]):.2f}")
        print(f"    • Mean: {np.mean(R_b_box[t]):.2f}")
        print(f"    • IQR: [{np.percentile(R_b_box[t],25):.2f}, {np.percentile(R_b_box[t],75):.2f}]")
        print(f"    • Range: [{np.min(R_b_box[t]):.2f}, {np.max(R_b_box[t]):.2f}]")
    if S_res_box[t]:
        print('\n  Sensitive Reservoir Bacteria (log10 CFU/mL):')
        print(f"    • Median: {np.median(S_res_box[t]):.2f}")
        print(f"    • Mean: {np.mean(S_res_box[t]):.2f}")
        print(f"    • IQR: [{np.percentile(S_res_box[t],25):.2f}, {np.percentile(S_res_box[t],75):.2f}]")
        print(f"    • Range: [{np.min(S_res_box[t]):.2f}, {np.max(S_res_box[t]):.2f}]")
    if R_res_box[t]:
        print('\n  Resistant Reservoir Bacteria (log10 CFU/mL):')
        print(f"    • Median: {np.median(R_res_box[t]):.2f}")
        print(f"    • Mean: {np.mean(R_res_box[t]):.2f}")
        print(f"    • IQR: [{np.percentile(R_res_box[t],25):.2f}, {np.percentile(R_res_box[t],75):.2f}]")
        print(f"    • Range: [{np.min(R_res_box[t]):.2f}, {np.max(R_res_box[t]):.2f}]")
    if N_box[t]:
        print('\n  Neutrophils (cells/μL):')
        print(f"    • Median: {np.median(N_box[t]):.0f}")
        print(f"    • Mean: {np.mean(N_box[t]):.0f}")
        print(f"    • IQR: [{np.percentile(N_box[t],25):.0f}, {np.percentile(N_box[t],75):.0f}]")
        print(f"    • Range: [{np.min(N_box[t]):.0f}, {np.max(N_box[t]):.0f}]")
 """
""" print('\n' + '=' * 60)
print('All visualizations have been generated and saved!')
print('Files saved with "NoBM" prefix to distinguish from BM stimulation results.')
print('=' * 60) """
