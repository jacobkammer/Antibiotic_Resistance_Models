import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
import os
import pandas as pd
import warnings
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
    'rho_res_S': 1.47,
    'rho_res_R': 1.47,
    'delta_res': 0.179,
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
total_h = 600
vanco_start = 300

# time grid
t_eval = np.linspace(0, total_h, 800)

# --- Storage ---
S_b_results = []
R_b_results = []
S_res_results = []
R_res_results = []
N_results = []

# NEW: Storage for initial conditions
initial_conditions_log = []

# Boxplot times (align to new grid)
boxplot_times = [10, 15, 25, 300, 400, 500]
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

    # Initial conditions [S_b, R_b, S_res, R_res, N]
    # Randomly select S_res and R_res between 10 and 100
    S_res_init = np.random.uniform(10, 100)
    R_res_init = np.random.uniform(10, 100)
    
    y0_base = [0, 0, S_res_init, R_res_init, immune_model.N0]
    # Log-normal sampling for bacteria, normal for N0
    y0 = [max(1, np.random.lognormal(np.log(y0_base[i]), cv/2)) for i in range(4)]
    y0.append(max(500, np.random.normal(y0_base[4], y0_base[4]*cv/4)))

    # NEW: Log the initial conditions for this simulation
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

# NEW: Save initial conditions to CSV
ic_df = pd.DataFrame(initial_conditions_log)
ic_df.to_csv('initial_conditions_log.csv', index=False)
print(f"\nInitial conditions saved to: initial_conditions_log.csv")

# NEW: Display summary statistics of initial conditions
print('\n' + '=' * 70)
print('INITIAL CONDITIONS SUMMARY')
print('=' * 70)
print(ic_df.describe())

# NEW: Create visualization of initial conditions distribution
fig_ic, axes_ic = plt.subplots(2, 3, figsize=(15, 10))

# S_res initial (base values before log-normal)
axes_ic[0, 0].hist(ic_df['S_res_base'], bins=30, color='purple', alpha=0.7, edgecolor='black')
axes_ic[0, 0].set_xlabel('S_res Base Initial Value')
axes_ic[0, 0].set_ylabel('Frequency')
axes_ic[0, 0].set_title('S_res Base (Uniform 10-100)')
axes_ic[0, 0].grid(True, alpha=0.3)

# R_res initial (base values before log-normal)
axes_ic[0, 1].hist(ic_df['R_res_base'], bins=30, color='green', alpha=0.7, edgecolor='black')
axes_ic[0, 1].set_xlabel('R_res Base Initial Value')
axes_ic[0, 1].set_ylabel('Frequency')
axes_ic[0, 1].set_title('R_res Base (Uniform 10-100)')
axes_ic[0, 1].grid(True, alpha=0.3)

# N initial
axes_ic[0, 2].hist(ic_df['N_init'], bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
axes_ic[0, 2].set_xlabel('Neutrophils Initial Value')
axes_ic[0, 2].set_ylabel('Frequency')
axes_ic[0, 2].set_title('Neutrophils Initial (N)')
axes_ic[0, 2].grid(True, alpha=0.3)

# S_res after log-normal sampling
axes_ic[1, 0].hist(ic_df['S_res_init'], bins=30, color='purple', alpha=0.7, edgecolor='black')
axes_ic[1, 0].set_xlabel('S_res After Sampling')
axes_ic[1, 0].set_ylabel('Frequency')
axes_ic[1, 0].set_title('S_res After Log-Normal Variation')
axes_ic[1, 0].grid(True, alpha=0.3)

# R_res after log-normal sampling
axes_ic[1, 1].hist(ic_df['R_res_init'], bins=30, color='green', alpha=0.7, edgecolor='black')
axes_ic[1, 1].set_xlabel('R_res After Sampling')
axes_ic[1, 1].set_ylabel('Frequency')
axes_ic[1, 1].set_title('R_res After Log-Normal Variation')
axes_ic[1, 1].grid(True, alpha=0.3)

# Scatter: S_res vs R_res (base values)
axes_ic[1, 2].scatter(ic_df['S_res_base'], ic_df['R_res_base'], alpha=0.5, s=10)
axes_ic[1, 2].set_xlabel('S_res Base')
axes_ic[1, 2].set_ylabel('R_res Base')
axes_ic[1, 2].set_title('S_res vs R_res Base Values')
axes_ic[1, 2].grid(True, alpha=0.3)

plt.suptitle('Initial Conditions Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/initial_conditions_distribution.png', dpi=300, bbox_inches='tight')
print('Saved: initial_conditions_distribution.png')
plt.show()

# Continue with rest of the Monte Carlo analysis...
# [Rest of the visualization code remains the same]