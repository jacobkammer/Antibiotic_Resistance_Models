import numpy as np
import matplotlib.pyplot as plt

def gillespie_birth_process(lambda_birth, initial_pop, t_max, num_simulations):
    """
    Simulate a stochastic birth process using the Gillespie algorithm.
    
    Parameters:
    -----------
    lambda_birth : float
        Birth rate parameter
    initial_pop : int
        Initial population size
    t_max : float
        Maximum simulation time
    num_simulations : int
        Number of independent simulations to run
    
    Returns:
    --------
    list of tuples
        Each tuple contains (times, populations) for one simulation
    """
    simulations = []
    
    for sim in range(num_simulations):
        # Initialize
        t = 0
        population = initial_pop
        times = [0]
        populations = [initial_pop]
        
        while t < t_max:
            # Calculate total propensity (birth rate * current population)
            total_rate = lambda_birth * population
            
            # Generate random time until next event
            tau = np.random.exponential(1/total_rate) if total_rate > 0 else float('inf')
            
            # Update time and check if we've exceeded t_max
            if t + tau > t_max:
                break
                
            t += tau
            
            # Birth event
            population += 1
            
            # Record the new state
            times.append(t)
            populations.append(population)
        
        simulations.append((np.array(times), np.array(populations)))
    
    return simulations

def plot_birth_process(simulations, lambda_birth, initial_pop):
    """Plot the results of multiple birth process simulations."""
    plt.figure(figsize=(12, 8))
    
    # Plot each simulation trajectory
    for times, populations in simulations:
        plt.plot(times, populations, alpha=0.3, color='blue')
    
    # Plot the deterministic solution
    t = np.linspace(0, max(sim[0][-1] for sim in simulations), 100)
    deterministic = initial_pop * np.exp(lambda_birth * t)
    plt.plot(t, deterministic, 'r--', label='Deterministic solution', linewidth=2)
    
    plt.title(f'Stochastic Birth Process\n(λ={lambda_birth}, Initial Pop={initial_pop})')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_statistics(simulations, t_points):
    """Calculate mean and standard deviation across simulations at specified time points."""
    populations_at_t = []
    
    for t in t_points:
        pops = []
        for times, populations in simulations:
            # Find the population at or just before time t
            idx = np.searchsorted(times, t)
            if idx == 0:
                pops.append(populations[0])
            else:
                pops.append(populations[idx-1])
        populations_at_t.append(pops)
    
    means = np.mean(populations_at_t, axis=1)
    stds = np.std(populations_at_t, axis=1)
    
    return means, stds

if __name__ == '__main__':
    # Simulation parameters
    lambda_birth = 0.1  # Birth rate
    initial_pop = 10   # Initial population
    t_max = 50        # Maximum simulation time
    num_simulations = 20  # Number of simulations
    
    # Run simulations
    simulations = gillespie_birth_process(lambda_birth, initial_pop, t_max, num_simulations)
    
    # Plot results
    plot_birth_process(simulations, lambda_birth, initial_pop)
    
    # Calculate and print statistics at specific time points
    t_points = [0, 10, 20, 30, 40, 50]
    means, stds = calculate_statistics(simulations, t_points)
    
    print("\nPopulation Statistics:")
    print("Time\tMean\tStd Dev")
    print("-" * 30)
    for t, mean, std in zip(t_points, means, stds):
        print(f"{t:4.1f}\t{mean:4.1f}\t{std:4.1f}")
