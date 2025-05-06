import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class DaptomycinPKModel:
    def __init__(self):
        # PK parameters for daptomycin
        self.volume_dist = 0.1  # L/kg (volume of distribution)
        self.clearance = 0.9    # L/h/kg (total body clearance)
        self.half_life = 8.0    # hours
        self.weight = 70        # kg (standard body weight)
        self.bioavailability = 1.0  # 100% bioavailable (IV administration)
        
        # Derived parameters
        self.ke = np.log(2) / self.half_life  # elimination rate constant
        self.vd = self.volume_dist * self.weight  # total volume of distribution
        
    def single_dose_concentration(self, t, dose):
        """Calculate concentration after a single dose at time t."""
        conc = (dose / self.vd) * np.exp(-self.ke * t)
        return conc
    
    def simulate_multiple_doses(self, dose_mg_per_kg, dosing_interval, duration, sample_points=1000):
        """
        Simulate multiple doses of daptomycin.
        
        Parameters:
        -----------
        dose_mg_per_kg : float
            Dose in mg/kg
        dosing_interval : float
            Time between doses in hours
        duration : float
            Total simulation time in hours
        sample_points : int
            Number of time points to sample
        
        Returns:
        --------
        tuple
            (time points, concentrations)
        """
        dose = dose_mg_per_kg * self.weight  # total dose in mg
        t = np.linspace(0, duration, sample_points)
        concentrations = np.zeros_like(t)
        
        # Calculate number of doses
        num_doses = int(duration / dosing_interval) + 1
        
        # Add contribution from each dose
        for i in range(num_doses):
            dose_time = i * dosing_interval
            mask = t >= dose_time
            concentrations[mask] += self.single_dose_concentration(t[mask] - dose_time, dose)
        
        return t, concentrations
    
    def plot_concentrations(self, t, concentrations, mic_range=None):
        """Plot concentration-time curve with optional MIC range."""
        plt.figure(figsize=(12, 8))
        
        # Plot concentration curve
        plt.plot(t, concentrations, 'b-', label='Daptomycin concentration')
        
        # Add MIC range if provided
        if mic_range is not None:
            plt.axhline(y=mic_range[0], color='r', linestyle='--', 
                       label=f'MIC range: {mic_range[0]}-{mic_range[1]} mg/L')
            plt.axhline(y=mic_range[1], color='r', linestyle='--')
            plt.fill_between(t, mic_range[0], mic_range[1], color='r', alpha=0.1)
        
        # Calculate and plot key PK parameters
        cmax = np.max(concentrations)
        plt.axhline(y=cmax, color='g', linestyle=':', label=f'Cmax = {cmax:.1f} mg/L')
        
        # Add labels and title
        plt.title('Daptomycin Serum Concentration vs Time')
        plt.xlabel('Time (hours)')
        plt.ylabel('Concentration (mg/L)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add PK parameters text box
        pk_text = f'PK Parameters:\nVd = {self.volume_dist:.2f} L/kg\n'
        pk_text += f'T1/2 = {self.half_life:.1f} h\n'
        pk_text += f'Ke = {self.ke:.3f} h⁻¹'
        plt.text(0.02, 0.98, pk_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()
    
    def calculate_pk_metrics(self, t, concentrations, mic=1.0):
        """Calculate key PK/PD metrics."""
        cmax = np.max(concentrations)
        cmin = np.min(concentrations[concentrations > 0])
        auc = np.trapz(concentrations, t)  # AUC calculation using trapezoidal rule
        time_above_mic = np.sum(concentrations > mic) / len(concentrations) * 100
        
        print("\nPK/PD Metrics:")
        print(f"Cmax: {cmax:.1f} mg/L")
        print(f"Cmin: {cmin:.1f} mg/L")
        print(f"AUC(0-24): {auc:.1f} mg·h/L")
        print(f"Time above MIC: {time_above_mic:.1f}%")
        print(f"Cmax/MIC ratio: {cmax/mic:.1f}")
        print(f"AUC/MIC ratio: {auc/mic:.1f}")

if __name__ == '__main__':
    # Create model instance
    model = DaptomycinPKModel()
    
    # Simulation parameters
    dose = 6  # mg/kg/day (typical dose)
    duration = 72  # hours (3 days)
    dosing_interval = 24  # hours (once daily)
    
    # Run simulation
    t, concentrations = model.simulate_multiple_doses(dose, dosing_interval, duration)
    
    # Plot results with typical MIC range for S. aureus
    mic_range = (0.5, 1.0)  # mg/L
    model.plot_concentrations(t, concentrations, mic_range)
    
    # Calculate PK metrics using the upper MIC value
    model.calculate_pk_metrics(t, concentrations, mic=mic_range[1])
