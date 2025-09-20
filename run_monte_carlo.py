import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Pharmacokinetic Model (returns functions directly) ---
class PharmacokineticModel:
    def __init__(self):
        # Vancomycin
        self.van_dose = 1200
        self.van_interval = 6
        self.van_duration = 72
        self.van_ke = 0.173
        self.van_volume = 50
        # Linezolid
        self.lzd_dose = 800
        self.lzd_interval = 12
        self.lzd_duration = 192
        self.lzd_ke = 0.116
        self.lzd_volume = 45

    def concentration_function(self, drug_type, total_time_h, start_h=0):
        """Returns a function conc(t) for the drug concentration over time."""
        if drug_type == 'vancomycin':
            dose, interval, duration, ke, volume = self.van_dose, self.van_interval, self.van_duration, self.van_ke, self.van_volume
        else:
            dose, interval, duration, ke, volume = self.lzd_dose, self.lzd_interval, self.lzd_duration, self.lzd_ke, self.lzd_volume

        t_points = np.linspace(0, total_time_h, int(total_time_h*10)+1)
        conc = np.zeros_like(t_points)
        for dt in np.arange(start_h, start_h + duration, interval):
            mask = t_points >= dt
            conc[mask] += (dose/volume) * np.exp(-ke*(t_points[mask]-dt))
        conc[t_points < start_h] = 0

        def conc_func(t):
            return np.interp(t, t_points, conc)#
        return conc_func

# --- Immune Response Model ---
class ImmuneResponse:
    def __init__(self, rho_N=1e-6, N_MAX=30000, delta_N=0.03, kill_N=5e-5, N0=5000):
        self.rho_N = rho_N#neutrophil growth rate
        self.N_MAX = N_MAX#neutrophil max population
        self.delta_N = delta_N#neutrophil death rate
        self.kill_N = kill_N#neutrophil killing rate
        self.N0 = N0#initial neutrophil population

    def compute(self, N, A_total, t=None):
        """Compute neutrophil growth and effective killing (full killing immediately)."""
        dN = self.rho_N * N * A_total * (1 - N/self.N_MAX) - self.delta_N * N
        immune_effect = self.kill_N * N
        return dN, immune_effect

# --- ODE system ---
def immune_and_pd_model(y, t, params, van_func, lzd_func, immune_model):
    S, R, A_res, N = y
    V = max(0, van_func(t))
    L = max(0, lzd_func(t))
    S, R, A_res, N = max(S,0), max(R,0), max(A_res,0), max(N,0)
    A_total = S + R + A_res

    h_V = h_L = 1
    linezolid_inhibition = (params['Emax_l'] * L**h_L) / (params['EC50_L']**h_L + L**h_L)
    vancomycin_kill = (params['Emax_v'] * V**h_V) / (params['EC50_V']**h_V + V**h_V)
    logistic = (1 - A_total / params['B_max'])

    # --- Immune Response ---
    '''
    **Line 71**: [immune_model.compute(N, A_total, t)]
    returns two values:
    - `dN`: The rate of change of neutrophil population (recruitment/death)
    - `immune_effect`: The killing rate per bacterium (based on neutrophil count)

**Lines 72-74**: Calculate the actual bacterial killing by neutrophils:
- `immune_S`: Number of sensitive bacteria killed per hour = killing rate × sensitive population
- `immune_R`: Number of resistant bacteria killed per hour = killing rate × resistant population  
- `immune_Ares`: Number of reservoir bacteria killed per hour = killing rate × reservoir population

This represents neutrophil-mediated bacterial clearance. The immune system kills bacteria proportionally to:
1. **Neutrophil count** (N) - more neutrophils = more killing
2. **Bacterial population size** - more bacteria available to kill
3. **Killing efficiency** (kill_N parameter = 5e-5 from the ImmuneResponse class)

These immune killing terms are then subtracted from each bacterial population in the differential equations (lines 77-80), representing how the immune system reduces bacterial growth alongside antibiotic effects.
    '''
    dN, immune_effect = immune_model.compute(N, A_total, t)
    immune_S = immune_effect * S
    immune_R = immune_effect * R
    immune_Ares = immune_effect * A_res

    # --- Sensitive & Resistant Bacterial population dynamics ---
    dS = params['rho_S']*S*logistic - params['delta']*S - immune_S - vancomycin_kill*S - linezolid_inhibition*S
    dR = params['rho_R']*R*logistic - params['delta']*R - immune_R - linezolid_inhibition*R + params['f_r_b']*A_res - params['f_b_r']*R
    dA_res = (params['rho_res']*A_res*(1-A_res/params['k_res']) - params['delta_res']*A_res
              - params['f_r_b']*A_res + params['f_b_r']*R - immune_Ares)

    return [dS, dR, dA_res, dN]

# --- Main Execution ---
print("Starting Monte Carlo Simulation for Antibiotic Resistance Model")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# Base parameters
base_params = {
    'rho_S': 1.47,  # rate of growth for sensitive bacteria
    'rho_R': 1.47,  # rate of growth for resistant bacteria
    'B_max': 4e12,  # bacterial carry capacity (CFU/ml)
    'delta': 0.179,  # Sensitive and Resistant Natural death rate (h^-1)
    'Emax_v': 1.74,
    'Emax_l': 1.97,
    'EC50_V': 0.245,  # Vancomycin concentration for 50% max effect (mg/L)
    'EC50_L': 0.56,  # Linezolid concentration for 50% max effect (mg/L)
    'rho_res': 1.47,  # Growth rate of reservoir bacteria (h^-1)
    'k_res': 1e4,  # reservoir carry capacity
    'delta_res': 0.179,  # Natural death rate of reservoir bacteria (h^-1)
    'f_r_b': 0.02,  # transfer rate from reservoir compartment to blood compartment
    'f_b_r': 0.02  # transfer rate from blood compartment to reservoir compartment
}

# Simulation settings
n_simulations = 3000
cv = 0.2  # coefficient of variation
total_h = 300
vanco_start = 48
t_eval = np.linspace(0, total_h, 500)

# Store results
S_results = []
R_results = []
Ares_results = []
N_results = []

# Time points for boxplots
boxplot_times = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250]
boxplot_indices = [np.argmin(np.abs(t_eval - t)) for t in boxplot_times]

S_boxplot_data = {t: [] for t in boxplot_times}
R_boxplot_data = {t: [] for t in boxplot_times}
Ares_boxplot_data = {t: [] for t in boxplot_times}

print(f"\nRunning {n_simulations} simulations with {cv*100:.0f}% parameter variation...")
print("-" * 40)
N_boxplot_data = {t: [] for t in boxplot_times}
successful_runs = 0
for sim in range(n_simulations):
    if sim % 20 == 0:
        print(f"Progress: {sim}/{n_simulations} simulations completed...")
    
    # Sample parameters with variation
    params = {}#stores the parameters for each simulation
    for key, value in base_params.items():
        # Use log-normal distribution for positive parameters
        if key in ['EC50_V', 'EC50_L', 'f_r_b', 'f_b_r']:
            # For these sensitive parameters, use smaller variation
            params[key] = np.random.lognormal(np.log(value), cv/2)
        else:
            params[key] = np.random.lognormal(np.log(value), cv)
    
    # Initialize models
    pk = PharmacokineticModel()
    immune_model = ImmuneResponse(N0=5000)
    
    # Vary PK parameters slightly
    pk.van_dose = max(100, np.random.normal(pk.van_dose, pk.van_dose * cv/4))
    pk.lzd_dose = max(100, np.random.normal(pk.lzd_dose, pk.lzd_dose * cv/4))
    
    lzd_start = vanco_start + pk.van_duration
    
    van_func = pk.concentration_function('vancomycin', total_h, vanco_start)
    lzd_func = pk.concentration_function('linezolid', total_h, lzd_start)
    
    # Vary initial conditions
    y0_base = [1e3, 1e3, 1e1, immune_model.N0]
    y0 = [max(1, np.random.lognormal(np.log(y0_base[i]), cv/2)) for i in range(3)]
    y0.append(max(500, np.random.normal(y0_base[3], y0_base[3] * cv/4)))
    
    try:
        # Solve ODE
        solution = odeint(immune_and_pd_model, y0, t_eval, 
                        args=(params, van_func, lzd_func, immune_model),
                        rtol=1e-6, atol=1e-9, mxstep=5000)
        
        # Store results (clip at 1 to avoid log(0))
        S_results.append(np.clip(solution[:, 0], 1, None))
        R_results.append(np.clip(solution[:, 1], 1, None))
        Ares_results.append(np.clip(solution[:, 2], 1, None))
        N_results.append(solution[:, 3])
        
        # Store data for boxplots
        for i, t in enumerate(boxplot_times):
            idx = boxplot_indices[i]
            S_boxplot_data[t].append(np.log10(max(solution[idx, 0], 1)))
            R_boxplot_data[t].append(np.log10(max(solution[idx, 1], 1)))
            Ares_boxplot_data[t].append(np.log10(max(solution[idx, 2], 1)))
            N_boxplot_data[t].append(solution[idx, 3])
        successful_runs += 1
            
    except Exception as e:
        continue

print(f"\nSimulation complete! {successful_runs}/{n_simulations} runs successful.")

# Convert to arrays
S_results = np.array(S_results)
R_results = np.array(R_results)
Ares_results = np.array(Ares_results)
N_results = np.array(N_results)

print("\nGenerating visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# --- Plot 1: Sensitive Bacteria ---
fig1, ax1 = plt.subplots(figsize=(12, 7))

# Calculate percentiles
S_median = np.percentile(S_results, 50, axis=0)
S_p25 = np.percentile(S_results, 25, axis=0)
S_p75 = np.percentile(S_results, 75, axis=0)
S_p5 = np.percentile(S_results, 5, axis=0)
S_p95 = np.percentile(S_results, 95, axis=0)

# Plot
ax1.semilogy(t_eval, S_median, color='blue', linewidth=2.5, label='Median')
ax1.fill_between(t_eval, S_p25, S_p75, alpha=0.4, color='blue', label='25-75 percentile')
ax1.fill_between(t_eval, S_p5, S_p95, alpha=0.2, color='blue', label='5-95 percentile')

# Add treatment lines
ax1.axvline(vanco_start, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Vancomycin Start')
ax1.axvline(lzd_start, color='darkblue', linestyle='--', alpha=0.7, linewidth=2, label='Linezolid Start')

# Formatting
ax1.set_xlabel('Time (hours)', fontsize=13)
ax1.set_ylabel('Population (CFU/ml)', fontsize=13)
ax1.set_title('Sensitive Bacteria Population - Monte Carlo Simulation', fontsize=15, fontweight='bold')
ax1.grid(True, which="both", ls="-", lw=0.3, alpha=0.3)
ax1.legend(loc='best', fontsize=11, framealpha=0.9)
ax1.set_ylim([1e0, 1e13])

plt.tight_layout()
plt.savefig('sensitive_bacteria_monte_carlo.png', dpi=300, bbox_inches='tight')
print("  Saved: sensitive_bacteria_monte_carlo.png")
plt.show()

# --- Plot 2: Resistant Bacteria ---
fig2, ax2 = plt.subplots(figsize=(12, 7))

# Calculate percentiles
R_median = np.percentile(R_results, 50, axis=0)
R_p25 = np.percentile(R_results, 25, axis=0)
R_p75 = np.percentile(R_results, 75, axis=0)
R_p5 = np.percentile(R_results, 5, axis=0)
R_p95 = np.percentile(R_results, 95, axis=0)

# Plot
ax2.semilogy(t_eval, R_median, color='orange', linewidth=2.5, label='Median')
ax2.fill_between(t_eval, R_p25, R_p75, alpha=0.4, color='orange', label='25-75 percentile')
ax2.fill_between(t_eval, R_p5, R_p95, alpha=0.2, color='orange', label='5-95 percentile')

# Add treatment lines
ax2.axvline(vanco_start, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Vancomycin Start')
ax2.axvline(lzd_start, color='darkblue', linestyle='--', alpha=0.7, linewidth=2, label='Linezolid Start')

# Formatting
ax2.set_xlabel('Time (hours)', fontsize=13)
ax2.set_ylabel('Population (CFU/ml)', fontsize=13)
ax2.set_title('Resistant Bacteria Population - Monte Carlo Simulation', fontsize=15, fontweight='bold')
ax2.grid(True, which="both", ls="-", lw=0.3, alpha=0.3)
ax2.legend(loc='best', fontsize=11, framealpha=0.9)
ax2.set_ylim([1e0, 1e13])

plt.tight_layout()
plt.savefig('resistant_bacteria_monte_carlo.png', dpi=300, bbox_inches='tight')
print("  Saved: resistant_bacteria_monte_carlo.png")
plt.show()

# --- Plot 3: Reservoir Bacteria ---
fig3, ax3 = plt.subplots(figsize=(12, 7))

# Calculate percentiles
A_median = np.percentile(Ares_results, 50, axis=0)
A_p25 = np.percentile(Ares_results, 25, axis=0)
A_p75 = np.percentile(Ares_results, 75, axis=0)
A_p5 = np.percentile(Ares_results, 5, axis=0)
A_p95 = np.percentile(Ares_results, 95, axis=0)

# Plot
ax3.semilogy(t_eval, A_median, color='purple', linewidth=2.5, label='Median')
ax3.fill_between(t_eval, A_p25, A_p75, alpha=0.4, color='purple', label='25-75 percentile')
ax3.fill_between(t_eval, A_p5, A_p95, alpha=0.2, color='purple', label='5-95 percentile')

# Add treatment lines
ax3.axvline(vanco_start, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Vancomycin Start')
ax3.axvline(lzd_start, color='darkblue', linestyle='--', alpha=0.7, linewidth=2, label='Linezolid Start')

# Formatting
ax3.set_xlabel('Time (hours)', fontsize=13)
ax3.set_ylabel('Population (CFU/ml)', fontsize=13)
ax3.set_title('Reservoir Bacteria Population - Monte Carlo Simulation', fontsize=15, fontweight='bold')
ax3.grid(True, which="both", ls="-", lw=0.3, alpha=0.3)
ax3.legend(loc='best', fontsize=11, framealpha=0.9)
ax3.set_ylim([1e0, 1e5])

plt.tight_layout()
plt.savefig('reservoir_bacteria_monte_carlo.png', dpi=300, bbox_inches='tight')
print("  Saved: reservoir_bacteria_monte_carlo.png")
plt.show()

# --- Plot 4: Neutrophil Dynamics ---
fig4, ax4 = plt.subplots(figsize=(12, 7))

# Calculate percentiles for neutrophils
N_median = np.percentile(N_results, 50, axis=0)
N_p25 = np.percentile(N_results, 25, axis=0)
N_p75 = np.percentile(N_results, 75, axis=0)
N_p5 = np.percentile(N_results, 5, axis=0)
N_p95 = np.percentile(N_results, 95, axis=0)

# Plot neutrophils
ax4.plot(t_eval, N_median, color='green', linewidth=2.5, label='Median')
ax4.fill_between(t_eval, N_p25, N_p75, alpha=0.4, color='green', label='25-75 percentile')
ax4.fill_between(t_eval, N_p5, N_p95, alpha=0.2, color='green', label='5-95 percentile')

# Add treatment lines
ax4.axvline(vanco_start, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Vancomycin Start')
ax4.axvline(lzd_start, color='darkblue', linestyle='--', alpha=0.7, linewidth=2, label='Linezolid Start')

# Formatting
ax4.set_xlabel('Time (hours)', fontsize=13)
ax4.set_ylabel('Neutrophil Count (cells/μL)', fontsize=13)
ax4.set_title('Neutrophil Population - Monte Carlo Simulation', fontsize=15, fontweight='bold')
ax4.grid(True, which="both", ls="-", lw=0.3, alpha=0.3)
ax4.legend(loc='best', fontsize=11, framealpha=0.9)
ax4.set_ylim([0, max(N_p95) * 1.1])

plt.tight_layout()
plt.savefig('neutrophil_monte_carlo.png', dpi=300, bbox_inches='tight')
print("  Saved: neutrophil_monte_carlo.png")
plt.show()

# --- Plot 5: Boxplots ---
fig5, axes = plt.subplots(2, 2, figsize=(16, 12))

# Color scheme
colors = {'S': '#3498db', 'R': '#ff8c42', 'A': '#9b59b6'}

# Sensitive bacteria boxplot
ax5_1 = axes[0, 0]
bp1 = ax5_1.boxplot([S_boxplot_data[t] for t in boxplot_times], 
                   labels=[f'{t}h' for t in boxplot_times],
                   patch_artist=True, widths=0.6,
                   boxprops=dict(linewidth=1.5),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5),
                   medianprops=dict(linewidth=2, color='darkred'))
for patch in bp1['boxes']:
    patch.set_facecolor(colors['S'])
    patch.set_alpha(0.7)
ax5_1.set_ylabel('Log₁₀(CFU/ml)', fontsize=12)
ax5_1.set_xlabel('Time Point', fontsize=12)
ax5_1.set_title('Sensitive Bacteria (S)', fontsize=13, fontweight='bold')
ax5_1.grid(True, alpha=0.3, linestyle='--')
ax5_1.set_ylim([-1, 13])

# Resistant bacteria boxplot
ax5_2 = axes[0, 1]
bp2 = ax5_2.boxplot([R_boxplot_data[t] for t in boxplot_times], 
                   labels=[f'{t}h' for t in boxplot_times],
                   patch_artist=True, widths=0.6,
                   boxprops=dict(linewidth=1.5),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5),
                   medianprops=dict(linewidth=2, color='darkred'))
for patch in bp2['boxes']:
    patch.set_facecolor(colors['R'])
    patch.set_alpha(0.7)
ax5_2.set_ylabel('Log₁₀(CFU/ml)', fontsize=12)
ax5_2.set_xlabel('Time Point', fontsize=12)
ax5_2.set_title('Resistant Bacteria (R)', fontsize=13, fontweight='bold')
ax5_2.grid(True, alpha=0.3, linestyle='--')
ax5_2.set_ylim([-1, 13])

# Reservoir bacteria boxplot
ax5_3 = axes[1, 0]
bp3 = ax5_3.boxplot([Ares_boxplot_data[t] for t in boxplot_times], 
                   labels=[f'{t}h' for t in boxplot_times],
                   patch_artist=True, widths=0.6,
                   boxprops=dict(linewidth=1.5),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5),
                   medianprops=dict(linewidth=2, color='darkred'))
for patch in bp3['boxes']:
    patch.set_facecolor(colors['A'])
    patch.set_alpha(0.7)
ax5_3.set_ylabel('Log₁₀(CFU/ml)', fontsize=12)
ax5_3.set_xlabel('Time Point', fontsize=12)
ax5_3.set_title('Reservoir Bacteria (A_res)', fontsize=13, fontweight='bold')
ax5_3.grid(True, alpha=0.3, linestyle='--')
ax5_3.set_ylim([-1, 5])

# Neutrophil boxplot
ax5_4 = axes[1, 1]
bp4 = ax5_4.boxplot([N_boxplot_data[t] for t in boxplot_times], 
                   labels=[f'{t}h' for t in boxplot_times],
                   patch_artist=True, widths=0.6,
                   boxprops=dict(linewidth=1.5),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5),
                   medianprops=dict(linewidth=2, color='darkred'))
for patch in bp4['boxes']:
    patch.set_facecolor('green')
    patch.set_alpha(0.7)
ax5_4.set_ylabel('Neutrophil Count (cells/μL)', fontsize=12)
ax5_4.set_xlabel('Time Point', fontsize=12)
ax5_4.set_title('Neutrophil Population', fontsize=13, fontweight='bold')
ax5_4.grid(True, alpha=0.3, linestyle='--')
# Handle empty neutrophil data gracefully
neutrophil_max_values = [max(N_boxplot_data[t]) if N_boxplot_data[t] else 0 for t in boxplot_times]
max_neutrophil = max(neutrophil_max_values) if neutrophil_max_values else 10000
ax5_4.set_ylim([0, max_neutrophil * 1.1])

plt.suptitle('Bacterial Population Distribution at Key Time Points', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('bacterial_and_neutrophil_boxplots.png', dpi=300, bbox_inches='tight')
print("  Saved: bacterial_and_neutrophil_boxplots.png")
plt.show()

# Print summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

for t in boxplot_times:
    print(f"\nTime = {t} hours:")
    print("-" * 40)
    
    if len(S_boxplot_data[t]) > 0:
        print(f"  Sensitive bacteria (log₁₀ CFU/ml):")
        print(f"    • Median: {np.median(S_boxplot_data[t]):.2f}")
        print(f"    • Mean: {np.mean(S_boxplot_data[t]):.2f}")
        print(f"    • IQR: [{np.percentile(S_boxplot_data[t], 25):.2f}, {np.percentile(S_boxplot_data[t], 75):.2f}]")
        print(f"    • Range: [{np.min(S_boxplot_data[t]):.2f}, {np.max(S_boxplot_data[t]):.2f}]")
    
    if len(R_boxplot_data[t]) > 0:
        print(f"\n  Resistant bacteria (log₁₀ CFU/ml):")
        print(f"    • Median: {np.median(R_boxplot_data[t]):.2f}")
        print(f"    • Mean: {np.mean(R_boxplot_data[t]):.2f}")
        print(f"    • IQR: [{np.percentile(R_boxplot_data[t], 25):.2f}, {np.percentile(R_boxplot_data[t], 75):.2f}]")
        print(f"    • Range: [{np.min(R_boxplot_data[t]):.2f}, {np.max(R_boxplot_data[t]):.2f}]")
    
    if len(Ares_boxplot_data[t]) > 0:
        print(f"\n  Reservoir bacteria (log₁₀ CFU/ml):")
        print(f"    • Median: {np.median(Ares_boxplot_data[t]):.2f}")
        print(f"    • Mean: {np.mean(Ares_boxplot_data[t]):.2f}")
        print(f"    • IQR: [{np.percentile(Ares_boxplot_data[t], 25):.2f}, {np.percentile(Ares_boxplot_data[t], 75):.2f}]")
        print(f"    • Range: [{np.min(Ares_boxplot_data[t]):.2f}, {np.max(Ares_boxplot_data[t]):.2f}]")
    
    if len(N_boxplot_data[t]) > 0:
        print(f"\n  Neutrophils (cells/μL):")
        print(f"    • Median: {np.median(N_boxplot_data[t]):.0f}")
        print(f"    • Mean: {np.mean(N_boxplot_data[t]):.0f}")
        print(f"    • IQR: [{np.percentile(N_boxplot_data[t], 25):.0f}, {np.percentile(N_boxplot_data[t], 75):.0f}]")
        print(f"    • Range: [{np.min(N_boxplot_data[t]):.0f}, {np.max(N_boxplot_data[t]):.0f}]")

print("\n" + "=" * 60)
print("All visualizations have been generated and saved!")
print("=" * 60)
