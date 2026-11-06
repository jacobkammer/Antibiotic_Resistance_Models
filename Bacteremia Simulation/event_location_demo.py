# =============================================================================
# Event location demo: a simple discontinuous ODE
#
# A stripped-down version of what solver.py does, using problem (3) from
# Shampine & Thompson, "Event Location for Ordinary Differential Equations",
# Computers and Mathematics with Applications 39 (2000) 43-54.
#
#       y' =  y      while the switch is ON   (y grows)
#       y' = -y/2    while the switch is OFF  (y decays)
#
#   - the switch starts ON with y(0) = 1
#   - when y rises to 2 the switch turns OFF
#   - when y falls to 1 the switch turns ON again
#
# The right-hand side changes abruptly at each switch, so the solution is a
# sawtooth. The exact switch times are known, which lets us check the answer:
# growth from 1 to 2 takes ln(2), decay from 2 to 1 takes 2*ln(2).
#
# How this maps onto solver.py:
#   switch ON/OFF          <->  population active/extinct
#   thresholds 2 and 1     <->  the LOD
#   one event per segment  <->  one event per population per segment
# =============================================================================
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

T_END = 10.0
UPPER = 2.0   # switch turns OFF when y rises to this value
LOWER = 1.0   # switch turns ON when y falls to this value

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
#the ODE. Which formula is used depends on the switch, not on y, so within one
#segment the right-hand side is smooth and the solver never sees a discontinuity.
#The switch determines which equation is being solved; the events decide when the
#switch flips. switch_on is passed in from outside and stays fixed for a whole segment.
#In solver.py the switch is segment_active (four flags instead of one) and hybrid_rhs
#plays the role of this function.
def rhs(t, y, switch_on):
    return y if switch_on else -y / 2.0


#-------------------------------------------------------------------------------------
# Segment loop: integrate until the threshold is reached, flip the switch, restart.
#
# The solver is active the whole time, on the way up and on the way down. What turns
# off at y = 2 is the switch, not the solver:
#   - on the way up (switch ON) the solver integrates y' = y
#   - at y = 2 the event fires, the solver stops for an instant, the switch flips OFF,
#     and the solver is restarted immediately from that same time and value
#   - on the way down (switch OFF) the solver integrates y' = -y/2
#   - at y = 1 the event fires again and the switch flips back ON
# Each rising or falling stretch of the sawtooth is one solve_ivp call. The pause at an
# event takes no simulated time; it exists only so the equation can be changed at the
# exact crossing instead of partway through a step. This is the whole idea of event
# location: one discontinuous problem becomes a sequence of smooth ones, joined at
# precisely located switching times.
#-------------------------------------------------------------------------------------
t_now = 0.0
y_now = np.array([1.0])
switch_on = True

segments = []      #continuous solution for each segment, used for plotting
event_times = []   #time of every switch

while t_now < T_END:
    #event function: zero when y reaches the threshold for the current switch state.
    #It tells solve_ivp when the switch should flip, and plays the same role as
    #"boundary" in solver.py (there the loop builds four of these, one per population,
    #all against the LOD).

    #the value to watch for in this segment: 2 while the switch is ON and y is rising,
    #1 while it is OFF and y is falling
    threshold = UPPER if switch_on else LOWER

    #returns the distance from y to the threshold: negative below it, positive above,
    #zero exactly at it. solve_ivp checks the sign after every step and, when it changes,
    #searches within that step for the time the value is zero.
    #  - t and y are supplied by solve_ivp. y[0] is used because solve_ivp always passes
    #    the state as an array, even with only one variable.
    #  - switch_on is not used here. It must be accepted because solve_ivp passes
    #    args=(switch_on,) to the event function as well as to rhs.
    #  - threshold=threshold freezes the current threshold into this copy of the
    #    function. The function is redefined on every pass through the loop, and this
    #    makes sure each one keeps its own value.
    def crossing(t, y, switch_on, threshold=threshold):
        return y[0] - threshold

    crossing.terminal = True                         #stop the integration at the crossing
    #direction restricts which crossings count. ON: only y rising through 2 (+1).
    #OFF: only y falling through 1 (-1). This matters at each restart: right after the
    #switch turns ON, y sits at exactly 1, the value the previous segment was watching
    #for. Because the new segment watches only for a rise through 2, nothing fires
    #spuriously at the starting point.
    crossing.direction = 1.0 if switch_on else -1.0

    #integrate toward T_END. solve_ivp stops early if the event fires and locates the
    #crossing time inside the step (it uses brentq internally). This one call integrates
    #one stretch of the sawtooth.
    #  rhs               the ODE function: returns y or -y/2 depending on the switch
    #  (t_now, T_END)    time span: start of this segment to the end of the simulation.
    #                    The call always aims for T_END; the event usually stops it sooner.
    #  y_now             starting value for this segment: 1 at the very start, then
    #                    whatever y was at the last crossing
    #  args=(switch_on,) extra input passed to rhs (and to crossing) on every call. This
    #                    is how the fixed switch setting for the segment reaches the equation.
    #  events=crossing   the threshold crossing to watch for
    #  dense_output=True returns a continuous solution that can be evaluated at any time
    #                    in the segment
    #  rtol, atol        error tolerances, set very tight so the located switch times can
    #                    be compared with the exact ones to about nine digits
    #No method is given, so solve_ivp uses its default, RK45, an explicit Runge-Kutta
    #method suited to a simple non-stiff problem like this one.
    #
    #Compared with solver.py the call has the same structure, with three differences:
    #  method:    RK45 here; LSODA there, because the bacteremia model is stiff
    #  max_step:  not set here; 4 h there, so a dose is not stepped over
    #  warnings and failures: not checked here; turned into IntegrationError there
    #The demo can skip those because its equation is smooth within each segment and has
    #no external inputs such as drug doses.
    result = solve_ivp(
        rhs, (t_now, T_END), y_now, args=(switch_on,),
        events=crossing, dense_output=True, rtol=1e-9, atol=1e-12,
    )
    #the parts of result used below:
    #  result.t[-1]         time the segment actually ended
    #  result.y[:, -1]      value of y at that time; becomes the start of the next segment
    #  result.sol           the continuous solution, stored for plotting
    #  result.t_events[0]   times the event fired: empty if the segment ran to T_END,
    #                       otherwise it holds the crossing time
    #save the start time, end time and continuous solution of this segment; the plotting
    #code at the bottom loops over these to draw each rising or falling stretch
    segments.append((t_now, result.t[-1], result.sol))

    #decide whether the simulation is finished or another segment is needed, by checking
    #whether the event fired during the segment that just ran.
    #result.t_events is a list with one entry per event function, and each entry is an
    #array of the times that event fired. There is only one event function here
    #(crossing), so result.t_events[0] is its array of firing times.

    #no event: the array is empty, so the event never fired. The only other way for
    #solve_ivp to stop is by reaching the end of the time span, so the segment ran all the
    #way to T_END and the simulation is finished. With T_END = 10 this happens on the
    #tenth segment: after the ninth switch (about 9.011) y decays from 2 but has only
    #reached about 1.22 by time 10, so it never gets down to 1.
    if len(result.t_events[0]) == 0:
        break

    #an event fired: record it, flip the switch, and restart from the crossing
    #the crossing time. The second [0] picks the first, and only, time in the array;
    #because the event is terminal there can never be more than one. This becomes the
    #start time of the next segment.
    t_now = result.t_events[0][0]
    #the value of y at the end of the segment, which is the value at the crossing (2 or 1,
    #to within the solver's tolerance). This becomes the starting value of the next segment.
    y_now = result.y[:, -1]
    event_times.append(t_now)   #record the switch time for the comparison table and the plot
    #flip the switch, so the next segment uses the other equation and watches the other
    #threshold. The loop then goes back to the top and builds a new crossing function.
    #solver.py has the same logic with one extra step: because it has four event
    #functions, it first searches result.t_events to find which of the four fired.
    switch_on = not switch_on


#-------------------------------------------------------------------------------------
# Compare the located switch times with the exact values
#
# This block is only for comparison. It plays no part in solving the problem: delete it
# and the simulation, the located switch times and the plot are unchanged. It computes
# the exact switch times by hand, with no numerical solver, so there is a known-correct
# answer to check solve_ivp against.
#
# Both pieces of the ODE can be solved exactly:
#   switch ON,  y' = y:     y = e^t from 1, reaches 2 when e^t = 2          -> takes ln(2)
#   switch OFF, y' = -y/2:  y = 2e^(-t/2) from 2, reaches 1 when e^(-t/2) = 1/2 -> takes 2*ln(2)
# so the switch times are a running total of those two durations, alternating:
# ln(2), 3ln(2), 4ln(2), 6ln(2), ... The pattern repeats every 3*ln(2).
#
# A check like this is possible only because the demo problem is simple. The bacteremia
# model has no exact solution, which is why test_solver_convergence.py uses indirect
# checks instead (coarse vs fine output grid, LSODA vs BDF).
#-------------------------------------------------------------------------------------
exact_times = []                 #collects the exact switch times
t_exact, on = 0.0, True          #start the clock at 0 with the switch ON, as in the simulation
while True:                      #the break below is the only way out
    #advance the clock to the next switch: ln(2) if y is growing, 2*ln(2) if it is decaying
    t_exact += np.log(2.0) if on else 2.0 * np.log(2.0)
    #stop without recording if that switch would fall after the end of the simulation
    if t_exact > T_END:
        break
    exact_times.append(t_exact)
    #"on = not on" flips a True/False value to its opposite: True becomes False and
    #False becomes True. Python first works out "not on" (the opposite of the current
    #value) and then stores that result back into on. Here it makes the two durations
    #alternate: pass 1 adds ln(2) and turns on to False, pass 2 adds 2*ln(2) and turns
    #on back to True, and so on. The segment loop above uses the same idiom,
    #"switch_on = not switch_on", to flip the switch each time an event fires.
    on = not on

print("Event   Switch turns   Located time     Exact time       Error")
for i, (found, exact) in enumerate(zip(event_times, exact_times), start=1):
    turns = "OFF" if i % 2 == 1 else "ON "
    print(f"{i:>3}     {turns}            {found:.10f}     {exact:.10f}     {abs(found - exact):.1e}")
print(f"\nEvents located: {len(event_times)}   Events expected: {len(exact_times)}")


#-------------------------------------------------------------------------------------
# FIGURE: the sawtooth solution with the two thresholds and the switch times
#-------------------------------------------------------------------------------------
#create the figure and one set of axes, 9 by 4.5 inches. constrained_layout=True adjusts
#the margins automatically so labels are not cut off.
fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)

#draw the solution one segment at a time. Each entry in segments holds a start time, an
#end time and the continuous solution saved by the loop.
#  np.linspace(start, stop, 100)  100 evenly spaced times across the segment
#  sol(t_plot)                    evaluates the continuous solution at those times. It
#                                 returns one row per variable, so [0] picks out y.
#Plotting per segment is what gives the sharp corners at each switch: each solution
#object is valid only for its own segment, and drawing each one right up to its end
#point puts a plotted point exactly on the crossing.
for start, stop, sol in segments:
    t_plot = np.linspace(start, stop, 100)
    ax.plot(t_plot, sol(t_plot)[0], color="steelblue", lw=2.0)

#dotted horizontal lines at the two thresholds, 2 and 1. Only the first has a label, so
#the legend shows one entry for both.
ax.axhline(UPPER, color="black", ls=":", lw=1.0, label="Thresholds (y = 1 and y = 2)")
ax.axhline(LOWER, color="black", ls=":", lw=1.0)

#dashed red vertical line at each located switch time. enumerate supplies a counter i so
#that only the first line (i == 0) gets a label; without that the legend would list the
#same entry nine times. alpha=0.6 makes the lines slightly transparent.
for i, te in enumerate(event_times):
    ax.axvline(te, color="firebrick", ls="--", lw=0.8, alpha=0.6,
               label="Located switch times" if i == 0 else None)

#axis labels, title and legend
ax.set_xlabel("Time")
ax.set_ylabel("y")
ax.set_title("Event location on a simple discontinuous ODE")
ax.legend(loc="upper right")

#make sure the figures folder exists (exist_ok=True means no error if it already does),
#build the file path, save at 150 dots per inch, and print where it went.
#This uses plain fig.savefig and not save_figure from parameters.py, so the demo figure
#is not recorded in figure_provenance.tsv. That is deliberate: it is not a result from
#the bacteremia model.
os.makedirs("figures", exist_ok=True)
out_file = os.path.join("figures", "event_location_demo.png")
fig.savefig(out_file, dpi=150)
print(f"Saved: {out_file}")
