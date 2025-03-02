import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class PharmacokineticModel:
    def __init__(self):
        # Vancomycin parameters
        self.van_dose = 1000  # mg (increased to 1000 mg)
        self.van_interval = 8   # hours (reduced from 12 to 8 hours)
        self.van_duration = 72  # hours
        self.van_ke = 0.173  # elimination rate constant (based on ~4h half-life)
        self.van_volume = 50  # distribution volume (L)

        # Linezolid parameters - updated to reflect clinical dosing
        self.lzd_dose = 600  # mg (standard clinical dose)
        self.lzd_interval = 12  # hours (twice daily dosing)
        self.lzd_duration = 192  # hours
        self.lzd_ke = 0.116  # elimination rate constant (based on ~6h half-life)
        self.lzd_volume = 40  # distribution volume (L) - closer to clinical values

        # Therapeutic ranges
        self.van_therapeutic_min = 10  # mg/L (trough)
        self.van_therapeutic_max = 20  # mg/L (trough)
        self.lzd_therapeutic_min = 2   # mg/L (trough)
        self.lzd_therapeutic_max = 8   # mg/L (trough)

    def calculate_concentrations(self, drug_type, start_time=0):
        if drug_type == 'vancomycin':
            dose = self.van_dose
            interval = self.van_interval
            duration = self.van_duration
            ke = self.van_ke
            volume = self.van_volume
        else:
            dose = self.lzd_dose
            interval = self.lzd_interval
            duration = self.lzd_duration
            ke = self.lzd_ke
            volume = self.lzd_volume

        # Time array including the no-drug period
        t = np.linspace(0, start_time + duration, 500)
        
        # Initialize concentration array
        concentrations = np.zeros_like(t)
        
        # Simulate multiple doses only after start_time
        for dose_time in np.arange(start_time, start_time + duration, interval):
            # Concentration calculation for IV bolus
            dose_mask = t >= dose_time
            concentrations[dose_mask] += (dose / volume) * np.exp(-ke * (t[dose_mask] - dose_time))
        
        return t, concentrations

def simulate_drug_dynamics(
    EC_50_vanco,
    EC_50_linez,
    fitness_cost=0.2,
    initial_sensitive=50,
    initial_resistant=50,
    no_drug_period=48,
    total_simulation_time=300,
    show_plots=False,
):
    # Create PK model and get drug concentrations
    pk_model = PharmacokineticModel()
    
    # Get concentrations with the no-drug period
    van_t, conc_vancomycin = pk_model.calculate_concentrations('vancomycin', start_time=no_drug_period)
    lzd_t, conc_linezolid = pk_model.calculate_concentrations('linezolid', start_time=no_drug_period + pk_model.van_duration)

    # Create time array for the full simulation
    time = np.linspace(0, total_simulation_time, total_simulation_time)
    
    # Interpolate drug concentrations to match simulation time points
    conc_vancomycin_interp = np.interp(time, van_t, conc_vancomycin)
    conc_linezolid_interp = np.interp(time, lzd_t, conc_linezolid)
    
    # Population Dynamics Parameters
    rho_sensitive = 0.03
    rho_resistant = rho_sensitive * (1 - fitness_cost)
    delta_sensitive, delta_resistant = 0.006, 0.006
    max_drug_effect_vanco, max_drug_effect_linez = 0.8, 0.95  # Increased linezolid efficacy 
    k = 1e5

    def population_ode(t, r):
        S, R = r
        if t <= no_drug_period:
            # No drug inhibition during no-drug period
            total_inhibition_sensitive = 0
            total_inhibition_resistant = 0
        else:
            # Get drug concentrations at current time point through interpolation
            conc_vanco_t = np.interp(t, time, conc_vancomycin_interp)
            conc_linez_t = np.interp(t, time, conc_linezolid_interp)

            # Calculate drug effects
            vanco_effect = max_drug_effect_vanco * (conc_vanco_t / (conc_vanco_t + EC_50_vanco))
            linez_effect_sensitive = max_drug_effect_linez * (conc_linez_t / (conc_linez_t + EC_50_linez))
            linez_effect_resistant = max_drug_effect_linez * (conc_linez_t / (conc_linez_t + EC_50_linez))

            total_inhibition_sensitive = vanco_effect + linez_effect_sensitive
            total_inhibition_resistant = linez_effect_resistant

        # Logistic growth with drug inhibition
        dSdt = rho_sensitive * S * (1 - (S + R) / k) * (1 - total_inhibition_sensitive) - delta_sensitive * S
        dRdt = rho_resistant * R * (1 - (S + R) / k) * (1 - total_inhibition_resistant) - delta_resistant * R
        return [dSdt, dRdt]

    # Solve ODE
    solution = solve_ivp(
        population_ode,
        [0, total_simulation_time],
        [initial_sensitive, initial_resistant],
        t_eval=time    
    )

    # Plot Results if requested
    if show_plots:
        plt.figure(figsize=(15, 10))

        plt.subplot(3, 1, 1)
        plt.plot(time, solution.y[0], label=f'Sensitive Population (ρ={rho_sensitive:.2f})')
        plt.plot(time, solution.y[1], label=f'Resistant Population (ρ={rho_resistant:.2f})')
        plt.axvline(no_drug_period, color='g', linestyle='--', label='Vancomycin Start')
        plt.axvline(no_drug_period + pk_model.van_duration, color='r', linestyle='--', label='Linezolid Start')
        plt.title(f'Population Dynamics (Resistance Fitness Cost: {fitness_cost*100}%)')
        plt.xlabel('Time (hours)')
        plt.ylabel('Population')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(time, conc_vancomycin_interp, label='Vancomycin Concentration')
        plt.plot(time, conc_linezolid_interp, label='Linezolid Concentration')
        plt.axvline(no_drug_period, color='g', linestyle='--')
        plt.axvline(no_drug_period + pk_model.van_duration, color='r', linestyle='--')
        plt.title('Drug Concentrations')
        plt.xlabel('Time (hours)')
        plt.ylabel('Concentration (mg/L)')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(time, conc_vancomycin_interp / (conc_vancomycin_interp + EC_50_vanco), 
                label='Vancomycin Inhibition')
        plt.plot(time, conc_linezolid_interp / (conc_linezolid_interp + EC_50_linez), 
                label='Linezolid Inhibition')
        plt.axvline(no_drug_period, color='g', linestyle='--')
        plt.axvline(no_drug_period + pk_model.van_duration, color='r', linestyle='--')
        plt.title('Drug Inhibition Over Time')
        plt.xlabel('Time (hours)')
        plt.ylabel('Inhibition')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return solution, time, conc_vancomycin_interp, conc_linezolid_interp

# Run the simulation with the calculated concentrations with plots enabled
solution, time, van_conc, lzd_conc = simulate_drug_dynamics(
    EC_50_vanco=0.8,
    EC_50_linez=1.2,  # Updated to reflect clinical MIC values for linezolid
    fitness_cost=0.3,
    no_drug_period=48,  # 48-hour no-drug period
    show_plots=True
)