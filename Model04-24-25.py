import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class PharmacokineticModel:
    def __init__(self):
        # Vancomycin parameters
        self.van_dose = 1500  
        self.van_interval = 6   # hours
        self.van_duration = 72  # hours
        self.van_ke = 0.173  # elimination rate constant (based on ~4h half-life)
        self.van_volume = 50  # distribution volume (L)

        # Linezolid parameters - updated to reflect clinical dosing
        self.lzd_dose = 750  # mg (increased from 600 for higher efficacy)
        self.lzd_interval = 8  # hours (more frequent dosing than standard 12h)
        self.lzd_duration = 192  # hours
        self.lzd_ke = 0.116  # elimination rate constant (based on ~6h half-life)
        self.lzd_volume = 40  # distribution volume (L) - original clinical value

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
        t = np.linspace(0, start_time + duration + 100, 500)  # Extended time to allow decay to zero
        
        # Initialize concentration array
        concentrations = np.zeros_like(t)
        
        # Simulate multiple doses only after start_time
        for dose_time in np.arange(start_time, start_time + duration, interval):
            # Concentration calculation for IV bolus
            dose_mask = t >= dose_time
            concentrations[dose_mask] += (dose / volume) * np.exp(-ke * (t[dose_mask] - dose_time))
        
        # Ensure concentration is zero before start_time
        concentrations[t < start_time] = 0.0
        
        return t, concentrations

def simulate_drug_dynamics(
    EC_50_vanco,
    EC_50_linez,
    fitness_cost=0.2,
    initial_sensitive=100,#CFU/ml
    initial_resistant=100,#CFU/ml
    no_drug_period=48,
    total_simulation_time=300,
    show_plots=False,
):
    """
    Simulate bacterial population dynamics under sequential vancomycin and linezolid treatment.
    
    Parameters:
    - EC_50_vanco: Concentration of vancomycin at which effect is 50% of maximum
    - EC_50_linez: Concentration of linezolid at which effect is 50% of maximum
    - fitness_cost: Growth rate reduction in resistant bacteria (0-1)
    - initial_sensitive: Initial population of sensitive bacteria
    - initial_resistant: Initial population of resistant bacteria
    - no_drug_period: Time (hours) before starting vancomycin treatment
    - total_simulation_time: Total simulation duration (hours)
    - show_plots: Whether to display simulation plots
    
    Returns:
    - solution: ODE solution object
    - time: Time points of simulation
    - conc_vancomycin_interp: Interpolated vancomycin concentrations
    - conc_linezolid_interp: Interpolated linezolid concentrations
    """
    # Create PK model and get drug concentrations
    pk_model = PharmacokineticModel()
    
    # Create time array for the full simulation
    time = np.linspace(0, total_simulation_time, total_simulation_time)
    
    # Get concentrations with the no-drug period
    van_t, conc_vancomycin = pk_model.calculate_concentrations('vancomycin', start_time=no_drug_period)
    lzd_t, conc_linezolid = pk_model.calculate_concentrations('linezolid', start_time=no_drug_period + pk_model.van_duration)
    
    # Interpolate drug concentrations to match simulation time points
    conc_vancomycin_interp = np.interp(time, van_t, conc_vancomycin)
    conc_linezolid_interp = np.interp(time, lzd_t, conc_linezolid)
    
    # Population Dynamics Parameters
    rho_sensitive = 0.03  # Growth rate for sensitive bacteria
    rho_resistant = rho_sensitive * (1 - fitness_cost)  # Reduced growth rate for resistant bacteria
    delta_sensitive, delta_resistant = 0.02, 0.008  # Death rates
    max_drug_effect_vanco, max_drug_effect_linez = 0.8, 1.0  # Maximum drug effect
    k = 1e5  # Carrying capacity
    rho_source = 0.03  # Growth rate for source bacteria
    k_source = 1e5  # Carrying capacity for source bacteria
    delta_source = 0.02  # Death rate for source bacteria
    transfer_rate = 0.01  # Transfer rate from source to resistant
    
    # Define the linezolid start time once (avoid recalculating)
    linezolid_start_time = no_drug_period + pk_model.van_duration
    
    def population_ode(t, y):
        S, R, resistant_reservoir = y
        # 
        """
        ODE system for bacterial population dynamics under antibiotic treatment.
        
        Parameters:
        - t: current time point
        - y: current population state [S, R, resistant_reservoir]
        
        Returns:
        - List containing rate of change [dSdt, dRdt, dResistantReservoirdt]
        """
        
        # Calculate drug inhibition effects
        if t < no_drug_period:
            # No drug period
            total_inhibition_sensitive = 0
            total_inhibition_resistant = 0
        else:
            # Get current drug concentrations at time t
            conc_vanco_t = np.interp(t, time, conc_vancomycin_interp)
            conc_linez_t = np.interp(t, time, conc_linezolid_interp)
            
            # Calculate vancomycin effect
            vanco_effect = calculate_vancomycin_effect(conc_vanco_t, max_drug_effect_vanco, EC_50_vanco)
            
            # Calculate linezolid effect (now one function for both populations)
            linez_effect = calculate_linezolid_effect(
                t, conc_linez_t, linezolid_start_time, max_drug_effect_linez, EC_50_linez)
            
            # Use the maximum of the two drug effects for sensitive bacteria
            # and linezolid effect for resistant bacteria
            total_inhibition_sensitive = max(vanco_effect, linez_effect)
            total_inhibition_resistant = linez_effect
        
        # Calculate population dynamics with constant death rates and transfer from resistant reservoir
        dSdt = calculate_growth_rate(S, total_inhibition_sensitive, rho_sensitive, k, S + R) - (delta_sensitive * S)
        # Resistant population receives constant transfer from the resistant reservoir compartment
        dRdt = (calculate_growth_rate(R, total_inhibition_resistant, rho_resistant, k, S + R) -
                (delta_resistant * R) + transfer_rate * resistant_reservoir)
        # Resistant reservoir compartment (acts as a source of resistant bacteria, no drug effect)
        dResistantReservoirdt = (calculate_growth_rate(resistant_reservoir, 0, rho_source, k_source, resistant_reservoir) -
                                 (delta_source * resistant_reservoir) - transfer_rate * resistant_reservoir)
        
        return [dSdt, dRdt, dResistantReservoirdt]
    def calculate_growth_rate(population, inhibition, growth_rate, k, total_population):
        """Calculate logistic growth rate with drug inhibition"""
        return growth_rate * population * (1 - total_population / k) * (1 - inhibition)

    def hill_equation(concentration, EC_50, max_effect, hill_coefficient=1):
        """
        Calculate drug effect using the Hill equation
        
        Parameters:
        - concentration: Drug concentration
        - EC_50: Concentration at which effect is 50% of max
        - max_effect: Maximum possible effect
        - hill_coefficient: Hill coefficient for steepness of response
        
        Returns:
        - Drug effect value between 0 and max_effect
        """
        if concentration <= 0:
            return 0
        return max_effect * (concentration**hill_coefficient / 
                            (EC_50**hill_coefficient + concentration**hill_coefficient))

    def calculate_vancomycin_effect(conc_vanco, max_effect, EC_50):
        """Calculate the effect of vancomycin based on concentration"""
        return hill_equation(conc_vanco, EC_50, max_effect, hill_coefficient=1)

    def calculate_linezolid_effect(t, conc_linez, linez_start_time, max_effect, EC_50):
        """Calculate linezolid effect on bacteria (identical for both populations)"""
        # If linezolid hasn't started yet
        if t < linez_start_time:
            return 0  # No effect before linezolid starts
        
        # Use Hill equation with a Hill coefficient of 3
        return hill_equation(conc_linez, EC_50, max_effect, hill_coefficient=3)

    # Solve ODE
    initial_resistant_reservoir = 100
    y0 = [initial_sensitive, initial_resistant, initial_resistant_reservoir]
    solution = solve_ivp(
        population_ode,
        [0, total_simulation_time],
        y0,
        t_eval=time,
        method='RK45'
    )

    # Plot Results if requested
    if show_plots:
        plt.figure(figsize=(15, 10))

        plt.subplot(3, 1, 1)
        plt.plot(time, solution.y[0], label=f'Sensitive Population (ρ={rho_sensitive:.2f})')
        plt.plot(time, solution.y[1], label=f'Resistant Population (ρ={rho_resistant:.2f})')
        plt.plot(time, solution.y[2], label=f'Source Population (ρ={rho_source:.2f})')
        plt.axvline(no_drug_period, color='g', linestyle='--', label='Vancomycin Start')
        plt.axvline(no_drug_period + pk_model.van_duration, color='r', linestyle='--', label='Linezolid Start')
        plt.title(f'Population Dynamics (Resistance Fitness Cost: {fitness_cost*100}%)')
        plt.xlabel('Time (hours)')
        plt.ylabel('CFU/ml')
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
        plt.plot(time, conc_vancomycin_interp / (EC_50_vanco + conc_vancomycin_interp), 
                label='Vancomycin Inhibition')
        plt.plot(time, conc_linezolid_interp / (EC_50_linez + conc_linezolid_interp), 
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
    EC_50_vanco=0.25,
    EC_50_linez=0.15,  # Further reduced from 0.2 to 0.15 for higher potency
    fitness_cost=0.3,
    no_drug_period=48,
    show_plots=True
)