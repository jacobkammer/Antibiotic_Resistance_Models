import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def combined_pk_pd_model(
    EC_50_vanco=0.8,
    EC_50_linez=2.0,
    initial_sensitive=50,
    initial_resistant=50,
    no_drug_period=48,
    total_time=216  # 48h drug-free + 72h vanco + 96h linez
):
    # Drug parameters
    vanco_dose = 500  # mg
    linez_dose = 800  # mg
    volume = 50  # L
    vanco_half_life = 4  # hours
    linez_half_life = 5  # hours
    
    # Calculate elimination rates
    ke_vanco = np.log(2) / vanco_half_life
    ke_linez = np.log(2) / linez_half_life
    
    # Create time points for simulation
    t = np.linspace(0, total_time, total_time)
    
    # Initialize concentration arrays
    vanco = np.zeros_like(t, dtype=float)
    linez = np.zeros_like(t, dtype=float)
    
    # Add doses and calculate concentrations
    # Vancomycin: 500mg q12h for 72h starting at 48h
    vanco_conc = vanco_dose / volume
    for dose_time in range(no_drug_period, no_drug_period+72, 12):
        idx = dose_time
        vanco[idx:] += vanco_conc * np.exp(-ke_vanco * (t[idx:] - t[idx]))
    
    # Linezolid: 800mg q6h for 96h starting at 120h
    linez_conc = linez_dose / volume
    for dose_time in range(no_drug_period+72, no_drug_period+72+96, 6):
        idx = dose_time
        linez[idx:] += linez_conc * np.exp(-ke_linez * (t[idx:] - t[idx]))
    
    # Population dynamics parameters
    rho = 0.03  # Base growth rate for both populations
    delta = 0.006  # Death rate for both populations
    max_drug_effect_vanco = 0.9
    max_drug_effect_linez = 0.9
    k = 1e5  # Carrying capacity
    
    # EC50 values for resistant strain
    EC_50_vanco_sensitive = EC_50_vanco * 0.5  # Sensitive bacteria need less vancomycin
    EC_50_linez_sensitive = EC_50_linez * 0.5  # Sensitive bacteria need less linezolid
    EC_50_linez_resistant = EC_50_linez * 2  # Resistant bacteria need more linezolid
    
    def population_ode(t, y):
        S, R = y
        
        # Get drug effects at current time
        if t <= no_drug_period:
            total_inhibition_sensitive = 0
            total_inhibition_resistant = 0
        else:
            # Get current drug concentrations
            t_idx = int(t)
            if t_idx >= len(vanco):
                t_idx = len(vanco) - 1
            
            conc_vanco_t = vanco[t_idx]
            conc_linez_t = linez[t_idx]
            
            # Calculate drug effects with Hill function
            vanco_effect_sensitive = max_drug_effect_vanco * (conc_vanco_t**2 / (conc_vanco_t**2 + EC_50_vanco_sensitive**2))
            linez_effect_sensitive = max_drug_effect_linez * (conc_linez_t**2 / (conc_linez_t**2 + EC_50_linez_sensitive**2))
            linez_effect_resistant = max_drug_effect_linez * 0.8 * (conc_linez_t**2 / (conc_linez_t**2 + EC_50_linez_resistant**2))
            
            # Cap total inhibition at 1.0 to prevent negative growth
            total_inhibition_sensitive = min(1.0, vanco_effect_sensitive + linez_effect_sensitive)
            total_inhibition_resistant = min(1.0, linez_effect_resistant)
        
        # Population dynamics with logistic growth and drug inhibition
        dSdt = rho * S * (1 - (S + R) / k) * (1 - total_inhibition_sensitive) - delta * S
        dRdt = rho * R * (1 - (S + R) / k) * (1 - total_inhibition_resistant) - delta * R
        
        return [dSdt, dRdt]
    
    # Solve population dynamics ODE
    solution = solve_ivp(
        population_ode,
        [0, total_time],
        [initial_sensitive, initial_resistant],
        t_eval=t,
        method='RK45'
    )
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot populations
    plt.subplot(3, 1, 1)
    plt.plot(t, solution.y[0], label=f'Sensitive Population (ρ={rho:.2f})')
    plt.plot(t, solution.y[1], label=f'Resistant Population (ρ={rho:.2f})')
    plt.axvline(x=no_drug_period, color='g', linestyle='--', label='Start Vancomycin')
    plt.axvline(x=no_drug_period+72, color='r', linestyle='--', label='Start Linezolid')
    plt.title('Population Dynamics')
    plt.xlabel('Time (hours)')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    
    # Plot drug concentrations
    plt.subplot(3, 1, 2)
    plt.plot(t, vanco, label='Vancomycin')
    plt.plot(t, linez, label='Linezolid')
    plt.axvline(x=no_drug_period, color='g', linestyle='--', label='Start Vancomycin')
    plt.axvline(x=no_drug_period+72, color='r', linestyle='--', label='Start Linezolid')
    plt.title('Drug Concentrations')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mg/L)')
    plt.legend()
    plt.grid(True)
    
    # Plot drug effects
    plt.subplot(3, 1, 3)
    vanco_effect_sensitive = max_drug_effect_vanco * (vanco**2 / (vanco**2 + EC_50_vanco_sensitive**2))
    linez_effect_sensitive = max_drug_effect_linez * (linez**2 / (linez**2 + EC_50_linez_sensitive**2))
    linez_effect_resistant = max_drug_effect_linez * 0.8 * (linez**2 / (linez**2 + EC_50_linez_resistant**2))
    plt.plot(t, vanco_effect_sensitive, label='Vancomycin Effect')
    plt.plot(t, linez_effect_sensitive, label='Linezolid Effect (Sensitive)')
    plt.plot(t, linez_effect_resistant, label='Linezolid Effect (Resistant)')
    plt.axvline(x=no_drug_period, color='g', linestyle='--', label='Start Vancomycin')
    plt.axvline(x=no_drug_period+72, color='r', linestyle='--', label='Start Linezolid')
    plt.title('Drug Effects')
    plt.xlabel('Time (hours)')
    plt.ylabel('Effect Magnitude')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nModel Summary:")
    print("--------------")
    print(f"Vancomycin: {vanco_dose}mg every 12h for 72h (start: {no_drug_period}h)")
    print(f"Linezolid: {linez_dose}mg every 6h for 96h (start: {no_drug_period+72}h)")
    print(f"\nHalf-lives:")
    print(f"Vancomycin: {vanco_half_life} hours")
    print(f"Linezolid: {linez_half_life} hours")
    print(f"\nPopulation Parameters:")
    print(f"Growth rate (both populations): {rho}")
    print(f"Death rate (both populations): {delta}")
    
    return solution, t, vanco, linez

# Run simulation
if __name__ == '__main__':
    solution, time, vanco, linez = combined_pk_pd_model(
        EC_50_vanco=0.8,
        EC_50_linez=2.0,
        no_drug_period=48
    )