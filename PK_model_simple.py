import numpy as np
import matplotlib.pyplot as plt

def simple_pk_model():
    # Time points (1 point per hour)
    total_time = 216  # 48h drug-free + 72h vanco + 96h linez
    t = np.arange(0, total_time, 1)
    
    # Drug parameters
    vanco_dose = 500  # mg
    linez_dose = 800  # mg
    volume = 50  # L
    
    # Convert doses to concentrations
    vanco_conc = vanco_dose / volume  # mg/L
    linez_conc = linez_dose / volume  # mg/L
    
    # Half-lives
    vanco_half_life = 4  # hours
    linez_half_life = 5  # hours
    
    # Calculate elimination rates
    ke_vanco = np.log(2) / vanco_half_life
    ke_linez = np.log(2) / linez_half_life
    
    # Initialize concentration arrays
    vanco = np.zeros_like(t, dtype=float)
    linez = np.zeros_like(t, dtype=float)
    
    # Add doses
    # Vancomycin: 500mg q12h for 72h starting at 48h
    for dose_time in range(48, 48+72, 12):
        idx = dose_time
        vanco[idx:] += vanco_conc * np.exp(-ke_vanco * (t[idx:] - t[idx]))
    
    # Linezolid: 800mg q6h for 96h starting at 120h
    for dose_time in range(120, 120+96, 6):
        idx = dose_time
        linez[idx:] += linez_conc * np.exp(-ke_linez * (t[idx:] - t[idx]))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t, vanco, label='Vancomycin')
    plt.plot(t, linez, label='Linezolid')
    plt.axvline(x=48, color='g', linestyle='--', label='Start Vancomycin')
    plt.axvline(x=120, color='r', linestyle='--', label='Start Linezolid')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mg/L)')
    plt.title('Drug Concentrations Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print summary
    print("\nPharmacokinetic Summary:")
    print("-----------------------")
    print(f"Vancomycin: {vanco_dose}mg every 12h for 72h (start: 48h)")
    print(f"Linezolid: {linez_dose}mg every 6h for 96h (start: 120h)")
    print(f"\nHalf-lives:")
    print(f"Vancomycin: {vanco_half_life} hours")
    print(f"Linezolid: {linez_half_life} hours")

# Run simulation
if __name__ == '__main__':
    simple_pk_model()