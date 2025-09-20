import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns

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
            return np.interp(t, t_points, conc)
        return conc_func

# --- Immune Response Model ---
class ImmuneResponse:
    def __init__(self, rho_N=1e-6, N_MAX=30000, delta_N=0.03, kill_N=5e-5, N0=5000):
        self.rho_N = rho_N
        self.N_MAX = N_MAX
        self.delta_N = delta_N
        self.kill_N = kill_N
        self.N0 = N0

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

# --- Monte Carlo Simulation Function ---
def run_monte_carlo_simulation(n_simulations=500, cv=0.2):
    """
    Run Monte Carlo simulation with parameter variation
    cv: coefficient of variation for parameter uncertainty (default 20%)
    """
    
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
    total_h = 300
    vanco_start = 48
    t_eval = np.linspace(0, total_h, 500)
    
    # Store results
    S_results = []
    R_results = []
    Ares_results = []
    N_results = []
    
    # Time points for boxplots
    boxplot_times = [25, 100, 150]
    boxplot_indices = [np.argmin(np.abs(t_eval - t)) for t in boxplot_times]
    
    S_boxplot_data = {t: [] for t in boxplot_times}
    R_boxplot_data = {t: [] for t in boxplot_times}
    Ares_boxplot_data = {t: [] for t in boxplot_times}
    
    print(f"Running {n_simulations} Monte Carlo simulations...")
    
    for sim in range(n_simulations):
        if sim % 10 == 0:
            print(f"  Simulation {sim}/{n_simulations}...")
        # Sample parameters with variation
        params = {}
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
        pk.van_dose = np.random.normal(pk.van_dose, pk.van_dose * cv/4)
        pk.lzd_dose = np.random.normal(pk.lzd_dose, pk.lzd_dose * cv/4)
        
        lzd_start = vanco_start + pk.van_duration
        
        van_func = pk.concentration_function('vancomycin', total_h, vanco_start)
        lzd_func = pk.concentration_function('linezolid', total_h, lzd_start)
        
        # Vary initial conditions
        y0_base = [1e3, 1e3, 1e1, immune_model.N0]
        y0 = [np.random.lognormal(np.log(y0_base[i]), cv/2) for i in range(3)]
        y0.append(np.random.normal(y0_base[3], y0_base[3] * cv/4))
        
        try:
            # Solve ODE
            solution = odeint(immune_and_pd_model, y0, t_eval, 
                            args=(params, van_func, lzd_func, immune_model),
                            rtol=1e-6, atol=1e-9)
            
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
                
        except Exception as e:
            print(f"Simulation {sim} failed: {e}")
            continue
    
    # Convert to arrays
    S_results = np.array(S_results)
    R_results = np.array(R_results)
    Ares_results = np.array(Ares_results)
    N_results = np.array(N_results)
    
    return (t_eval, S_results, R_results, Ares_results, N_results, 
            vanco_start, lzd_start, S_boxplot_data, R_boxplot_data, Ares_boxplot_data)

# --- Plotting Functions ---
def plot_population_with_uncertainty(t_eval, results, title, color, vanco_start, lzd_start):
    """Plot bacterial population with uncertainty bands"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate percentiles
    median = np.percentile(results, 50, axis=0)
    p25 = np.percentile(results, 25, axis=0)
    p75 = np.percentile(results, 75, axis=0)
    p5 = np.percentile(results, 5, axis=0)
    p95 = np.percentile(results, 95, axis=0)
    
    # Plot
    ax.semilogy(t_eval, median, color=color, linewidth=2, label='Median')
    ax.fill_between(t_eval, p25, p75, alpha=0.3, color=color, label='25-75 percentile')
    ax.fill_between(t_eval, p5, p95, alpha=0.15, color=color, label='5-95 percentile')
    
    # Add treatment lines
    ax.axvline(vanco_start, color='red', linestyle='--', alpha=0.7, label='Vancomycin Start')
    ax.axvline(lzd_start, color='blue', linestyle='--', alpha=0.7, label='Linezolid Start')
    
    # Formatting
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Population (CFU/ml)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_boxplots(S_data, R_data, Ares_data, time_points):
    """Create boxplots for all three bacterial populations at specific time points"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data for boxplots
    data_list = []
    labels = []
    colors = []
    
    for t in time_points:
        data_list.extend([S_data[t], R_data[t], Ares_data[t]])
        labels.extend([f'S (t={t}h)', f'R (t={t}h)', f'A_res (t={t}h)'])
        colors.extend(['blue', 'orange', 'purple'])
    
    # Sensitive bacteria boxplot
    ax1 = axes[0]
    bp1 = ax1.boxplot([S_data[t] for t in time_points], 
                       labels=[f't={t}h' for t in time_points],
                       patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('blue')
        patch.set_alpha(0.7)
    ax1.set_ylabel('Log10(CFU/ml)', fontsize=12)
    ax1.set_title('Sensitive Bacteria (S)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Resistant bacteria boxplot
    ax2 = axes[1]
    bp2 = ax2.boxplot([R_data[t] for t in time_points], 
                       labels=[f't={t}h' for t in time_points],
                       patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('orange')
        patch.set_alpha(0.7)
    ax2.set_ylabel('Log10(CFU/ml)', fontsize=12)
    ax2.set_title('Resistant Bacteria (R)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Reservoir bacteria boxplot
    ax3 = axes[2]
    bp3 = ax3.boxplot([Ares_data[t] for t in time_points], 
                       labels=[f't={t}h' for t in time_points],
                       patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('purple')
        patch.set_alpha(0.7)
    ax3.set_ylabel('Log10(CFU/ml)', fontsize=12)
    ax3.set_title('Reservoir Bacteria (A_res)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Bacterial Population Distribution at Key Time Points', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

# --- Main Execution ---
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run Monte Carlo simulation
    (t_eval, S_results, R_results, Ares_results, N_results, 
     vanco_start, lzd_start, S_boxplot_data, R_boxplot_data, Ares_boxplot_data) = run_monte_carlo_simulation(n_simulations=100)
    
    print(f"\nSimulation complete. Generated {len(S_results)} successful runs.")
    
    # Create individual plots for each bacterial population
    print("\nGenerating plots...")
    
    # Plot Sensitive Bacteria
    fig1 = plot_population_with_uncertainty(t_eval, S_results, 
                                           'Sensitive Bacteria Population - Monte Carlo Simulation',
                                           'blue', vanco_start, lzd_start)
    plt.savefig('sensitive_bacteria_monte_carlo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot Resistant Bacteria
    fig2 = plot_population_with_uncertainty(t_eval, R_results,
                                           'Resistant Bacteria Population - Monte Carlo Simulation',
                                           'orange', vanco_start, lzd_start)
    plt.savefig('resistant_bacteria_monte_carlo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot Reservoir Bacteria
    fig3 = plot_population_with_uncertainty(t_eval, Ares_results,
                                           'Reservoir Bacteria Population - Monte Carlo Simulation',
                                           'purple', vanco_start, lzd_start)
    plt.savefig('reservoir_bacteria_monte_carlo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create boxplots
    fig4 = plot_boxplots(S_boxplot_data, R_boxplot_data, Ares_boxplot_data, [25, 100, 150])
    plt.savefig('bacteria_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for t in [25, 100, 150]:
        print(f"\nTime = {t} hours:")
        print(f"  Sensitive bacteria (log10 CFU/ml):")
        print(f"    Median: {np.median(S_boxplot_data[t]):.2f}")
        print(f"    IQR: [{np.percentile(S_boxplot_data[t], 25):.2f}, {np.percentile(S_boxplot_data[t], 75):.2f}]")
        print(f"  Resistant bacteria (log10 CFU/ml):")
        print(f"    Median: {np.median(R_boxplot_data[t]):.2f}")
        print(f"    IQR: [{np.percentile(R_boxplot_data[t], 25):.2f}, {np.percentile(R_boxplot_data[t], 75):.2f}]")
        print(f"  Reservoir bacteria (log10 CFU/ml):")
        print(f"    Median: {np.median(Ares_boxplot_data[t]):.2f}")
        print(f"    IQR: [{np.percentile(Ares_boxplot_data[t], 25):.2f}, {np.percentile(Ares_boxplot_data[t], 75):.2f}]")
