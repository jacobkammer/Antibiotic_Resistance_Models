import numpy as np
from scipy.integrate import solve_ivp


# ==============================================================================
# 1. DAMPED HARMONIC OSCILLATOR ODE
# ==============================================================================
# State vector y:
#   y[0] = Position x (meters)
#   y[1] = Velocity v (meters / second)
#
# Equation of motion: m*(d^2x/dt^2) + c*(dx/dt) + k*x = 0
def damped_oscillator(t, y, m=1.0, k=10.0, c=0.5):
    x, v = y
    dxdt = v
    dvdt = -(c / m) * v - (k / m) * x
    return [dxdt, dvdt]


# ==============================================================================
# 2. EVENT FUNCTION WITH direction = 0
# ==============================================================================
def equilibrium_crossing_event(t, y):
    # Triggers zero-crossing when position x == 0
    return y[0]


# direction = 0: DETECTS ALL ZERO-CROSSINGS (both + -> - and - -> +)
equilibrium_crossing_event.direction = 0

# terminal = False: Allows the solver to keep running through multiple swings
equilibrium_crossing_event.terminal = False


# ==============================================================================
# 3. INITIAL CONDITIONS & SOLVER EXECUTION
# ==============================================================================
# Start pulled back at x = 2.0 meters, released from rest (v = 0.0 m/s)
y0 = [2.0, 0.0]
t_span = (0, 10.0)  # Integrate for 10 seconds

sol = solve_ivp(
    damped_oscillator,
    t_span,
    y0,
    events=[equilibrium_crossing_event]
)


# ==============================================================================
# 4. RESULTS
# ==============================================================================
crossing_times = sol.t_events[0]
crossing_states = sol.y_events[0]

print(f"Total equilibrium crossings detected: {len(crossing_times)}\n")
print(f"{'Pass #':<8}{'Time (s)':<12}{'Position (m)':<16}{'Velocity (m/s)':<16}{'Direction'}")
print("-" * 65)

for i, (t_cross, state) in enumerate(zip(crossing_times, crossing_states), 1):
    pos, vel = state
    # Notice how velocity alternates signs on each pass
    direction_label = "Moving Right (+ -> -)" if vel > 0 else "Moving Left (- -> +)"
    print(f"{i:<8}{t_cross:<12.3f}{pos:<16.2f}{vel:<16.3f}{direction_label}")


    #Why direction = 0 Matters HereAlternating Signs: As the mass oscillates back and forth,
    #its velocity alternates between positive and negative at each center-crossing:Pass 1: 
    #Moving left ($v \approx -5.55\text{ m/s}$)Pass 2: Moving right ($v \approx +4.32\text{ m/s}$)Pass 3:
    # oving left ($v \approx -3.37\text{ m/s}$)Complete Capture: If direction were set to -1 or +1, solve_ivp would
    #ignore half of the equilibrium passings. With direction = 0, it records every single time the spring passes
    #through $x = 0$.