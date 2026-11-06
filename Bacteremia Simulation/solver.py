"""Validated hybrid integration with a hard limit-of-detection boundary.

The four bacterial burdens follow the continuous ODE while they are at or
above ``LOD``. A burden crossing downward through LOD is set to exact zero and
cannot grow autonomously. An extinct compartment may still accumulate incoming
transfer from another compartment, such as the reservoir. The
extinct compartment becomes active again only when those arrivals reach LOD.
This makes reservoir reseeding explicit without allowing a fractional
deterministic remnant to manufacture relapse.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import solve_ivp

from parameters import LOD, SOLVER_CONFIG


class IntegrationError(RuntimeError):
    """Raised when an ODE solution fails validation."""

#guard in solver.py that checks the list of output times before any integration starts, and returns a clean numpy array.
#If times are unusable it stops with an error so solver does not fail later in the integration.
#integrate_model uses the first and last value as the integration span and the whole array as the
#grid on which results are reported, so the times must be sorted for np.searchsorted to work.
def _validate_times(t_eval):
    times = np.asarray(t_eval, dtype=float) #accepts lists, tuples, integer arrays; converts to a float array
    #must be a flat sequence with at least a start and an end time
    #code below checks that the array is 1D and has at least 2 elements
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("t_eval must be a one-dimensional array with at least two points")
    #no NaN/inf, and every gap between consecutive times must be positive (strictly increasing)
    #code below checks that all values are finite and that the difference between consecutive times is positive
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("t_eval must contain finite, strictly increasing values")
    return times


#-------------------------------------------------------------------------------------
# Runs one simulation of the four-population model with the hard-LOD rule.
# The integration is done in segments: solve_ivp runs until a population crosses the
# LOD, stops, that population is switched off (or back on), and the integration
# restarts from the crossing time. Every analysis script calls this function instead
# of calling solve_ivp directly.
#-------------------------------------------------------------------------------------
def integrate_model(model_rhs, y0, t_eval, args=(), *, method=None):
    """Integrate the four-compartment model with exact LOD extinction events.

    Inactive states are hidden from the biological RHS, so they have no local
    growth, killing, immune clearance, or outgoing transfer. The RHS can still
    supply their incoming transfer term from an active paired compartment.
    The returned array has shape ``(len(t_eval), len(y0))``.
    """
    #----- Setup and input checks -----
    #numerical settings come only from parameters.SOLVER_CONFIG. The one exception is the
    #integration method, which a caller may override (test_solver_convergence.py does this
    #to compare LSODA against BDF).
    cfg = SOLVER_CONFIG
    #use the caller's method if one was given, otherwise the default from parameters.py.
    #Only the LSODA-versus-BDF test passes one; that test calls this function, not the reverse.
    selected_method = method or cfg["method"]
    times = _validate_times(t_eval)
    #starting state must be exactly four finite, nonnegative burdens [S_b, R_b, S_res, R_res]
    state = np.asarray(y0, dtype=float)
    #code below checks that state is a 1D array of exactly 4 finite, non-negative values
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError("y0 must be a finite one-dimensional state vector")
    if np.any(state < 0):
        raise ValueError("y0 must be nonnegative")
    if len(state) != 4:
        raise ValueError("the bacteremia model requires four state variables")

    #the LOD rule applies at time zero: a population that starts below the LOD is set to
    #zero and begins extinct (inactive). Blood starts sterile, so both blood populations
    #begin inactive and first appear when reservoir seeding accumulates to the LOD.
    #
    #compare each of the four starting burdens with the LOD and record the answers as four
    #True/False flags, one per population. state is a numpy array, so >= is applied to
    #every element at once. With the baseline start [0, 0, 100, 100] and LOD = 10, active
    #becomes [False, False, True, True]. The comparison is >=, so a population at exactly
    #the LOD counts as active.
    #active is the set of switches the rest of this function runs on (the four-population
    #version of switch_on in event_location_demo.py):
    #  - the next line uses it to zero any population that starts below the LOD
    #  - each segment copies it into segment_active, which decides what hybrid_rhs hides
    #    from the model and which direction each boundary event watches
    #  - event handling is the only place that changes it, flipping one flag each time a
    #    population goes extinct or is reactivated
    active = state >= LOD
    state = np.where(active, state, 0.0)
    output = np.zeros((len(times), len(state)), dtype=float) #one row per output time, one column per population, 4 in this case
    output[0] = state
    next_output = 1 #index of the next output row still to be filled
    current = float(times[0]) #start time of the current segment
    end = float(times[-1]) #end of the simulation
    event_count = 0
    max_events = int(cfg.get("max_extinction_events", 10000))
    nudge = float(cfg.get("reactivation_nudge", 1e-6))

    #----- Segment loop: one pass per LOD crossing -----
    while current < end:
        #snapshot of which populations are active. It stays fixed while solve_ivp runs;
        #status only changes between segments, when an event fires.
        segment_active = active.copy()

        #wrapper that solve_ivp calls in place of the biological model.
        #It makes extinct populations behave as extinct.
        def hybrid_rhs(t, y):
            #hide extinct populations from the model: they are passed in as zero, so they
            #have no growth, drug killing, immune clearance, or outgoing transfer, and take
            #up no carrying capacity. Any sub-LOD arrivals they are accumulating are
            #invisible to the model, so that pool is not exposed to killing or clearance.
            biological_state = np.where(segment_active, y, 0.0)
            #solve_ivp passes (t, y); the model is written as (y, t, ...)
            derivative = np.asarray(model_rhs(biological_state, t, *args), dtype=float)
            if derivative.shape != state.shape or not np.all(np.isfinite(derivative)):
                raise IntegrationError("model RHS returned an invalid derivative")
            # Extinct states can only accumulate positive arrivals from transfer.
            derivative[~segment_active] = np.maximum(derivative[~segment_active], 0.0)
            return derivative

        #one event function per population, each watching y[i] - LOD.
        #solve_ivp checks the sign after every step and, when it changes, locates the
        #crossing time within that step (it uses brentq internally to do this).
        events = []
        for index in range(len(state)):
            #state_index=index binds this function to its own population; without it
            #all four functions would watch the last loop index.
            def boundary(t, y, state_index=index):
                return y[state_index] - LOD

            boundary.terminal = True #stop the integration when this event fires
            #active population: watch only for a downward crossing (extinction)
            #extinct population: watch only for an upward crossing (reactivation)
            boundary.direction = -1.0 if segment_active[index] else 1.0
            events.append(boundary)

        #integrate from the current time toward the end of the simulation. The call stops
        #early if an LOD event fires. dense_output=True returns a continuous solution, so
        #t_eval is not passed here; the output grid is filled in below. Warnings are
        #recorded instead of printed because LSODA reports some numerical trouble as warnings.
        #
        #the solve_ivp arguments, one by one:
        #  hybrid_rhs        the wrapped model. Supplies the rates of change; extinct
        #                    populations are hidden from the biology and can only
        #                    accumulate incoming transfer.
        #  (current, end)    time span: start of this segment and the end of the
        #                    simulation. The call always aims for the end; an event
        #                    usually stops it sooner.
        #  state             the four starting burdens for this segment: the initial
        #                    condition on the first pass, then whatever the previous
        #                    segment ended with, after the reset.
        #  method            the integrator, LSODA by default. It switches automatically
        #                    between stiff and non-stiff methods, which suits the fast
        #                    killing and slow regrowth in this model.
        #  events=events     the four boundary functions. This turns on event detection:
        #                    solve_ivp watches each y[i] - LOD for a sign change and
        #                    stops at the crossing.
        #  dense_output=True returns a continuous solution, result.sol, that can be
        #                    evaluated at any time in the segment.
        #  rtol              relative error allowed per step (1e-6)
        #  atol              absolute error allowed per step (1e-8 CFU/mL); governs
        #                    accuracy when a burden is near zero
        #  max_step          largest step allowed (4 h), so the integrator cannot step
        #                    over a dose
        #
        #two things are deliberately absent:
        #  no t_eval   a segment usually covers only part of the output grid, so
        #              integrate_model evaluates result.sol at the relevant times itself,
        #              a few lines further down.
        #  no args     hybrid_rhs is defined inside the loop and reads segment_active and
        #              the model inputs directly, so nothing extra needs passing. (In
        #              event_location_demo.py the switch is passed with args=(switch_on,).)
        #This is the same call as in event_location_demo.py, with LSODA and max_step added
        #because the bacteremia model is stiff and has dosing.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = solve_ivp(
                hybrid_rhs, (current, end), state, method=selected_method,
                events=events, dense_output=True, rtol=cfg["rtol"],
                atol=cfg["atol"], max_step=cfg["max_step"],#cfg values come from SOLVER_CONFIG in parameters.py
            )
        #refuse to return a trajectory the integrator could not vouch for
        if caught:
            messages = "; ".join(str(item.message) for item in caught)
            raise IntegrationError(f"ODE solver warning: {messages}")
        if not result.success:
            raise IntegrationError(f"ODE solver failed: {result.message}")
        if result.sol is None or not np.all(np.isfinite(result.y)):
            raise IntegrationError("ODE solver returned an invalid segment")

        #fill every requested output time that falls inside this segment by evaluating the
        #continuous solution. Extinct populations, and any value below the LOD, are
        #reported as exact zero.
        stop = float(result.t[-1]) #time this segment actually ended
        stop_output = int(np.searchsorted(times, stop, side="right"))
        if stop_output > next_output:
            values = result.sol(times[next_output:stop_output]).T
            values[:, ~segment_active] = 0.0
            values[values < LOD] = 0.0
            output[next_output:stop_output] = values
            next_output = stop_output

        #find which event stopped the segment, if any
        event_index = next(
            (i for i, event_times in enumerate(result.t_events) if len(event_times)),
            None,
        )
        #no event fired: the segment reached the end of the simulation
        if event_index is None:
            state = result.y[:, -1]
            current = stop
            break

        #an event fired: switch that population's status and restart from the crossing time
        state = result.y[:, -1]
        if segment_active[event_index]:
            #extinction: set to exact zero and mark inactive
            state[event_index] = 0.0
            active[event_index] = False
        else:
            #reactivation: restart a hair above the LOD so the downward event watched in
            #the next segment does not fire immediately
            state[event_index] = LOD * (1.0 + nudge)
            active[event_index] = True
        current = stop
        event_count += 1
        #guard against a population flickering across the LOD indefinitely
        if event_count > max_events:
            raise IntegrationError("too many LOD extinction/reactivation events")

    #----- Finish -----
    #fill any output rows not yet written with the final state, check the whole array,
    #and report anything still below the LOD as zero
    if next_output < len(times):
        output[next_output:] = np.where(active & (state >= LOD), state, 0.0)
    if not np.all(np.isfinite(output)) or np.any(output < 0):
        raise IntegrationError("hybrid-state reconstruction failed")
    output[output < LOD] = 0.0
    return output
