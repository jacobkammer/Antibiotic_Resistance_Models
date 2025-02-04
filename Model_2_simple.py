import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def combined_pk_pd_model(
    EC_50_vanco,
    EC_50_linez,
    initial_sensitive=50,
    initial_resistant=5,
    no_drug_period=72,
    total_time=312  # 72h drug-free + 72h vanco + 168h linez
):
    # Drug parameters
    vanco_dose = 1500  # mg
    linez_dose = 800  # mg
    volume = 50  # L
    vanco_half_life = 4  # hours(normal kidney: 4 - 6 hours)
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
    # Vancomycin: 1500mg q12h for 72h starting at 72h
    vanco_conc = vanco_dose / volume
    for dose_time in range(no_drug_period, no_drug_period+72, 12):
        idx = dose_time
        vanco[idx:] += vanco_conc * np.exp(-ke_vanco * (t[idx:] - t[idx]))
    
    # Linezolid: 800mg q6h for 168h starting at 144h
    linez_conc = linez_dose / volume
    for dose_time in range(no_drug_period+72, no_drug_period+72+168, 6):
        idx = dose_time
        linez[idx:] += linez_conc * np.exp(-ke_linez * (t[idx:] - t[idx]))
    
    # Population dynamics parameters
    rho_sensitive = 0.028  # Base growth rate for sensitive bacteria
    rho_resistant = 0.024  # Lower growth rate for resistant bacteria (fitness cost=20%)
    delta_sensitive = 0.012  # Increased death rate for sensitive bacteria
    delta_resistant = 0.004  # Lower death rate for resistant bacteria 
    max_drug_effect_vanco = 0.98  # Nearly complete maximum effect for vancomycin on sensitive
    max_drug_effect_vanco_resistant = 0.2  # Very low maximum effect on resistant bacteria
    max_drug_effect_linez = 0.9
    k = 1e5  # Carrying capacity
    
    # EC50 values
    EC_50_vanco_sensitive = EC_50_vanco * 0.15  # Make sensitive bacteria much more susceptible to vancomycin
    EC_50_vanco_resistant = EC_50_vanco * 8.0   # Make resistant bacteria much less susceptible to vancomycin
    EC_50_linez_both = EC_50_linez  # Same linezolid effect for both populations
    
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
            
            # Calculate drug effects with Hill coefficient of 2
            vanco_effect_sensitive = max_drug_effect_vanco * (conc_vanco_t / (conc_vanco_t + EC_50_vanco_sensitive))
            vanco_effect_resistant = max_drug_effect_vanco_resistant * (conc_vanco_t / (conc_vanco_t + EC_50_vanco_resistant))
            linez_effect = max_drug_effect_linez * (conc_linez_t / (conc_linez_t + EC_50_linez_both))
            
            # Cap total inhibition at 1.0 to prevent negative growth
            total_inhibition_sensitive = min(1.0, vanco_effect_sensitive + linez_effect)
            total_inhibition_resistant = min(1.0, vanco_effect_resistant + linez_effect)
        
        # Population dynamics with logistic growth and drug inhibition
        dSdt = rho_sensitive * S * (1 - (S + R) / k) * (1 - total_inhibition_sensitive) - delta_sensitive * S
        dRdt = rho_resistant * R * (1 - (S + R) / k) * (1 - total_inhibition_resistant) - delta_resistant * R
        
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
    plt.figure(figsize=(15, 15))
    
    # Plot populations (log scale)
    plt.subplot(3, 1, 1)
    total_population = solution.y[0] + solution.y[1]
    plt.semilogy(t, solution.y[0], label='Sensitive Bacteria', linewidth=2)
    plt.semilogy(t, solution.y[1], label='Resistant Bacteria', linewidth=2)
    plt.semilogy(t, total_population, '--', label='Total Population', color='gray', alpha=0.7)
    plt.axvline(x=no_drug_period, color='g', linestyle='--', label='Start Vancomycin')
    plt.axvline(x=no_drug_period+72, color='r', linestyle='--', label='Start Linezolid')
    plt.title('Bacterial Population Dynamics')
    plt.xlabel('Time (hours)')
    plt.ylabel('Population (log scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.fill_between([no_drug_period, no_drug_period+72], plt.ylim()[0], plt.ylim()[1], 
                    color='green', alpha=0.1, label='Vancomycin Period')
    plt.fill_between([no_drug_period+72, total_time], plt.ylim()[0], plt.ylim()[1], 
                    color='red', alpha=0.1, label='Linezolid Period')
    
    # Plot populations (linear scale)
    plt.subplot(3, 1, 2)
    total_population = solution.y[0] + solution.y[1]
    plt.plot(t, solution.y[0], label='Sensitive Bacteria', linewidth=2)
    plt.plot(t, solution.y[1], label='Resistant Bacteria', linewidth=2)
    plt.plot(t, total_population, '--', label='Total Population', color='gray', alpha=0.7)
    plt.axvline(x=no_drug_period, color='g', linestyle='--', label='Start Vancomycin')
    plt.axvline(x=no_drug_period+72, color='r', linestyle='--', label='Start Linezolid')
    plt.title('Bacterial Population Dynamics (Linear Scale)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Population')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.fill_between([no_drug_period, no_drug_period+72], plt.ylim()[0], plt.ylim()[1], 
                    color='green', alpha=0.1, label='Vancomycin Period')
    plt.fill_between([no_drug_period+72, total_time], plt.ylim()[0], plt.ylim()[1], 
                    color='red', alpha=0.1, label='Linezolid Period')
    
    # Plot drug concentrations
    plt.subplot(3, 1, 3)
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
    plt.figure(figsize=(15, 5))
    vanco_effect_sensitive = max_drug_effect_vanco * (vanco**2 / (vanco**2 + EC_50_vanco_sensitive**2))
    vanco_effect_resistant = max_drug_effect_vanco_resistant * (vanco**2 / (vanco**2 + EC_50_vanco_resistant**2))
    linez_effect = max_drug_effect_linez * (linez**2 / (linez**2 + EC_50_linez_both**2))
    plt.plot(t, vanco_effect_sensitive, label='Vancomycin Effect on Sensitive')
    plt.plot(t, vanco_effect_resistant, label='Vancomycin Effect on Resistant')
    plt.plot(t, linez_effect, label='Linezolid Effect (Both populations)')
    plt.axvline(x=no_drug_period, color='g', linestyle='--', label='Start Vancomycin')
    plt.axvline(x=no_drug_period+72, color='r', linestyle='--', label='Start Linezolid')
    plt.title('Drug Effects')
    plt.xlabel('Time (hours)')
    plt.ylabel('Effect Magnitude')
    plt.legend()
    plt.grid(True)
    
    # Plot drug effect vs normalized populations over time
    plt.figure(figsize=(15, 5))
    
    # Calculate total drug effects
    total_effect_sensitive = np.zeros_like(t)
    total_effect_resistant = np.zeros_like(t)
    for i, time in enumerate(t):
        if time <= no_drug_period:
            continue
        t_idx = int(time)
        if t_idx >= len(vanco):
            t_idx = len(vanco) - 1
            
        conc_vanco_t = vanco[t_idx]
        conc_linez_t = linez[t_idx]
        
        vanco_effect_sensitive = max_drug_effect_vanco * (conc_vanco_t / (conc_vanco_t + EC_50_vanco_sensitive))
        vanco_effect_resistant = max_drug_effect_vanco_resistant * (conc_vanco_t / (conc_vanco_t + EC_50_vanco_resistant))
        linez_effect = max_drug_effect_linez * (conc_linez_t / (conc_linez_t + EC_50_linez_both))
        
        total_effect_sensitive[i] = min(1.0, vanco_effect_sensitive + linez_effect)
        total_effect_resistant[i] = min(1.0, vanco_effect_resistant + linez_effect)
    
    plt.plot(t, total_effect_sensitive, 'g-', label='Drug Effect on Sensitive')
    plt.plot(t, total_effect_resistant, 'r-', label='Drug Effect on Resistant')
    plt.plot(t, solution.y[0] / np.max(solution.y[0]), 'b--', label='Normalized Sensitive Pop.')
    plt.plot(t, solution.y[1] / np.max(solution.y[1]), 'k--', label='Normalized Resistant Pop.')
    plt.title('Drug Effect vs Normalized Population')
    plt.xlabel('Time (hours)')
    plt.ylabel('Relative Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nModel Summary:")
    print("--------------")
    print(f"Vancomycin: {vanco_dose}mg every 12h for 72h (start: {no_drug_period}h)")
    print(f"Linezolid: {linez_dose}mg every 6h for 168h (start: {no_drug_period+72}h)")
    print(f"\nHalf-lives:")
    print(f"Vancomycin: {vanco_half_life} hours")
    print(f"Linezolid: {linez_half_life} hours")
    print(f"\nPopulation Parameters:")
    print(f"Sensitive bacteria growth rate: {rho_sensitive}")
    print(f"Resistant bacteria growth rate: {rho_resistant}")
    print(f"Sensitive bacteria death rate: {delta_sensitive}")
    print(f"Resistant bacteria death rate: {delta_resistant}")
    print(f"Carrying capacity: {k}")
    
    return solution, t, vanco, linez

# Run simulation
if __name__ == '__main__':
    solution, time, vanco, linez = combined_pk_pd_model(
        EC_50_vanco=4.0,
        EC_50_linez=0.5,
        no_drug_period=72
    )