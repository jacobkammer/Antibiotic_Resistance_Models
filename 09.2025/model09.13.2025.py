import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt



# --- Pharmacokinetic Model (returns functions directly) ---
class PharmacokineticModel:
    def __init__(self):
        # Vancomycin
        self.van_dose = 1200
        self.van_interval = 6
        self.van_duration = 96
        self.van_ke = 0.173
        self.van_volume = 50
        # Linezolid
        self.lzd_dose = 800
        self.lzd_interval = 12
        self.lzd_duration = 300
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
    def __init__(self, rho_N=3e-6, N_MAX=30000, delta_N=0.02, kill_N=5e-5, N0=5000):
        self.rho_N = rho_N
        self.N_MAX = N_MAX
        self.delta_N = delta_N
        self.kill_N = kill_N
        self.N0 = N0
        
    def compute(self, N, B_total, t=None):
        """Compute neutrophil growth and effective killing.
        - Recruitment increases with bacterial load (B_total) and saturates as N approaches N_MAX.
        - Immune killing is saturable with bacteria: kill_N * N * B_total / (K_imm + B_total)
        """
        dN = self.rho_N * N * B_total * (1 - N/self.N_MAX) - self.delta_N * N
        immune_effect = self.kill_N * N 
        return dN, immune_effect


print("Script starting...")
# --- ODE system ---
def immune_and_pd_model(y, t, params, van_func, lzd_func, immune_model):
    S, R, A_res, N = y
    V = max(0, van_func(t))
    L = max(0, lzd_func(t))
    S, R, A_res, N = max(S,0), max(R,0), max(A_res,0), max(N,0)
    B_total = S + R + A_res

    h_V = h_L = 1
    linezolid_inhibition = (params['Emax_l'] * L**h_L) / (params['EC50_L']**h_L + L**h_L)
    vancomycin_kill = (params['Emax_v'] * V**h_V) / (params['EC50_V']**h_V + V**h_V)
    logistic = (1 - B_total / params['B_max'])

    # --- Immune Response ---
    dN, immune_effect = immune_model.compute(N, B_total, t)
    immune_S = immune_effect * S
    immune_R = immune_effect * R
    immune_Ares = immune_effect * A_res

    # --- Sensitive & Resistant Bacterial population dynamics ---
    dS = params['rho_S']*S*logistic - params['delta']*S - immune_S - vancomycin_kill*S - linezolid_inhibition*S
    dR = params['rho_R']*R*logistic - params['delta']*R - immune_R - linezolid_inhibition*R + params['f_r_b']*A_res - params['f_b_r']*R
    dA_res = (params['rho_res']*A_res*(1-A_res/params['k_res']) - params['delta_res']*A_res
              - params['f_r_b']*A_res + params['f_b_r']*R - immune_Ares)

    return [dS, dR, dA_res, dN]

# --- Simulation ---
total_h = 600
pk = PharmacokineticModel()
immune_model = ImmuneResponse(N0=5000)  # full immune killing immediately

vanco_start = 300
lzd_start = vanco_start + pk.van_duration

van_func = pk.concentration_function('vancomycin', total_h, vanco_start)
lzd_func = pk.concentration_function('linezolid', total_h, lzd_start)

params = {
    'rho_S':1.47,#rate of growth for sensitive bacteria
    'rho_R':1.47,#rate of growth for resistant bacteria
    'B_max':4e12,# bacterial carry capacity (CFU/ml)
    'delta':0.179, # Sensitive and Resistant Natural death rate (h^-1)
    'Emax_v':1.74,
    'Emax_l':1.97,
    'EC50_V':0.245,# Vancomycin concentration for 50% max effect (mg/L)
    'EC50_L':0.56,# Linezolid concentration for 50% max effect (mg/L)
    'rho_res':1.47, # Growth rate of reservoir bacteria (h^-1)
    'k_res':1e4,# reservoir carry capacity
    'delta_res':0.179, # Natural death rate of reservoir bacteria (h^-1)
    'f_r_b':0.02,# transfer rate from reservoir compartment to blood compartment
    'f_b_r':0.0002# transfer rate from blood compartment to reservoir compartment
}

y0 = [1e1, 1e1, 1e4, immune_model.N0]
t_eval = np.linspace(0, total_h, 500)
solution = odeint(immune_and_pd_model, y0, t_eval, args=(params, van_func, lzd_func, immune_model))



# --- Prepare bacterial data (clip at 1 to avoid log(0)) ---
S_plot = np.clip(solution[:,0], 1, None)
R_plot = np.clip(solution[:,1], 1, None)
Ares_plot = np.clip(solution[:,2], 1, None)

# --- Plot Sensitive Bacteria ---
plt.figure(figsize=(10,6))
plt.plot(t_eval, S_plot, label='Sensitive Bacteria (S)', color='blue')
plt.yscale('log')
plt.xlabel('Time (h)')
plt.ylabel('Sensitive Population (CFU/ml)')
plt.title('Sensitive Bacteria')
plt.axvline(vanco_start, color='red', linestyle='--', label='Vancomycin Start')
plt.axvline(lzd_start, color='blue', linestyle='--', label='Linezolid Start')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.show()

# --- Plot Resistant Bacteria ---
plt.figure(figsize=(10,6))
plt.plot(t_eval, R_plot, label='Resistant Bacteria (R)', color='orange')
plt.yscale('log')
plt.xlabel('Time (h)')
plt.ylabel('Resistant Population (CFU/ml)')
plt.title('Resistant Bacteria')
plt.axvline(vanco_start, color='red', linestyle='--', label='Vancomycin Start')
plt.axvline(lzd_start, color='blue', linestyle='--', label='Linezolid Start')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.show()

# --- Plot Reservoir ---
plt.figure(figsize=(10,6))
plt.plot(t_eval, Ares_plot, label='Reservoir (A_res)', color='purple')
plt.yscale('log')
plt.xlabel('Time (h)')
plt.ylabel('Reservoir Population (CFU/ml)')
plt.title('Reservoir Bacteria ')
plt.axvline(vanco_start, color='red', linestyle='--', label='Vancomycin Start')
plt.axvline(lzd_start, color='blue', linestyle='--', label='Linezolid Start')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.show()

# --- Plot Neutrophils Separately ---
plt.figure(figsize=(10,6))
plt.plot(t_eval, solution[:,3], color='green', label='Neutrophils (N)')
plt.xlabel('Time (h)') 
plt.ylabel(r'Neutrophils (cells / $\mu$L)')
plt.title('Neutrophil Dynamics')
plt.axvline(vanco_start, color='red', linestyle='--', label='Vancomycin Start')
plt.axvline(lzd_start, color='blue', linestyle='--', label='Linezolid Start')
plt.grid(True, ls="--", lw=0.5)
plt.legend()
plt.show()


print(f"Vancomycin starts at: {vanco_start} hours")
print(f"Linezolid starts at: {lzd_start} hours") 

