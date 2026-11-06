import numpy as np
from scipy.integrate import solve_ivp

# ==============================================================================
# 1. BIOLOGICAL ODE SYSTEM (SIR Model)
# ==============================================================================
# State vector y:
#   y[0] = S (Susceptible individuals)
#   y[1] = I (Infected individuals)
#   y[2] = R (Recovered / Immune individuals)
def sir_model(t, y, beta=0.3, gamma=0.1, N=1000.0):
    S, I, R = y
    
    dS_dt = -beta * S * I / N               # Rate of new infections
    dI_dt = (beta * S * I / N) - gamma * I  # Net change in active infections
    dR_dt = gamma * I                        # Rate of recovery
    
    return [dS_dt, dI_dt, dR_dt]


# ==============================================================================
# 2. EVENT FUNCTIONS
# ==============================================================================

# Event 0: Peak Infection Rate (dI/dt = 0)
def peak_infection_event(t, y, beta=0.3, gamma=0.1, N=1000.0):
    S, I, R = y
    # Evaluates the derivative dI/dt directly
    return (beta * S * I / N) - gamma * I

# direction = -1: Detects when dI/dt transitions from positive (growing) to negative (declining)
peak_infection_event.direction = -1
# terminal = False: Continue simulating past the peak
peak_infection_event.terminal = False


# Event 1: Epidemic Clearance Threshold (Active infections I = 1)
def epidemic_clearance_event(t, y):
    I = y[1]
    return I - 1.0  # Triggers zero-crossing when I == 1

# direction = -1: Detects dropping down to 1 active case
epidemic_clearance_event.direction = -1
# terminal = True: Stop the solver when active cases drop below 1 person
epidemic_clearance_event.terminal = True


# ==============================================================================
# 3. INITIAL CONDITIONS & SIMULATION
# ==============================================================================
N = 1000.0
y0 = [999.0, 1.0, 0.0]  # 999 Susceptible, 1 Infected, 0 Recovered
t_span = (0, 200.0)     # Simulate up to 200 days

sol = solve_ivp(
    sir_model,
    t_span,
    y0,
    events=[peak_infection_event, epidemic_clearance_event]
)


# ==============================================================================
# 4. RESULTS
# ==============================================================================
print(f"Simulation Status: {sol.message}\n")

# Event 0: Peak Infection
peak_day = sol.t_events[0][0]
peak_S, peak_I, peak_R = sol.y_events[0][0]

print("--- Event 0: Peak Active Outbreak ---")
print(f"Peak Day: t = {peak_day:.2f} days")
print(f"Active Infections (I): {peak_I:.1f} individuals")
print(f"Susceptible Remaining (S): {peak_S:.1f} individuals\n")

# Event 1: Outbreak End
end_day = sol.t_events[1][0]
end_S, end_I, end_R = sol.y_events[1][0]

print("--- Event 1: Epidemic Clearance ---")
print(f"End Day (I <= 1): t = {end_day:.2f} days")
print(f"Total Population Recovered/Immune (R): {end_R:.1f} individuals")