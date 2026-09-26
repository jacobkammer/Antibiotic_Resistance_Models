"""Validated hybrid integration with a hard limit-of-detection boundary.

The four bacterial burdens follow the continuous ODE while they are at or
above ``LOD``. A burden crossing downward through LOD is set to exact zero and
cannot grow autonomously. An extinct compartment may still accumulate incoming
transfer from a detectable donor; it becomes active again only when those
arrivals reach LOD. This makes reservoir reseeding explicit without allowing a
fractional deterministic remnant to manufacture relapse.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import solve_ivp

from parameters import LOD, SOLVER_CONFIG


class IntegrationError(RuntimeError):
    """Raised when an ODE solution fails validation."""


def _validate_times(t_eval):
    times = np.asarray(t_eval, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("t_eval must be a one-dimensional array with at least two points")
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("t_eval must contain finite, strictly increasing values")
    return times


def integrate_model(model_rhs, y0, t_eval, args=(), *, method=None, config=None):
    """Integrate the four-compartment model with exact LOD extinction events.

    Inactive states are hidden from the biological RHS, so they have no local
    growth, killing, immune clearance, or outgoing transfer. The RHS can still
    supply their incoming transfer term from an active paired compartment. The
    The returned array has shape ``(len(t_eval), len(y0))``.
    """
    cfg = dict(SOLVER_CONFIG if config is None else config)
    selected_method = method or cfg["method"]
    times = _validate_times(t_eval)
    state = np.asarray(y0, dtype=float)
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError("y0 must be a finite one-dimensional state vector")
    if np.any(state < 0):
        raise ValueError("y0 must be nonnegative")
    if len(state) != 4:
        raise ValueError("the bacteremia model requires four state variables")

    active = state >= LOD
    state = np.where(active, state, 0.0)
    output = np.zeros((len(times), len(state)), dtype=float)
    output[0] = state
    next_output = 1
    current = float(times[0])
    end = float(times[-1])
    event_count = 0
    max_events = int(cfg.get("max_extinction_events", 10000))
    nudge = float(cfg.get("reactivation_nudge", 1e-6))

    while current < end:
        segment_active = active.copy()

        def hybrid_rhs(t, y):
            biological_state = np.where(segment_active, y, 0.0)
            derivative = np.asarray(model_rhs(biological_state, t, *args), dtype=float)
            if derivative.shape != state.shape or not np.all(np.isfinite(derivative)):
                raise IntegrationError("model RHS returned an invalid derivative")
            # Extinct states can only accumulate positive arrivals from transfer.
            derivative[~segment_active] = np.maximum(derivative[~segment_active], 0.0)
            return derivative

        events = []
        for index in range(len(state)):
            def boundary(t, y, state_index=index):
                return y[state_index] - LOD

            boundary.terminal = True
            boundary.direction = -1.0 if segment_active[index] else 1.0
            events.append(boundary)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = solve_ivp(
                hybrid_rhs, (current, end), state, method=selected_method,
                events=events, dense_output=True, rtol=cfg["rtol"],
                atol=cfg["atol"], max_step=cfg["max_step"],
            )
        if caught:
            messages = "; ".join(str(item.message) for item in caught)
            raise IntegrationError(f"ODE solver warning: {messages}")
        if not result.success:
            raise IntegrationError(f"ODE solver failed: {result.message}")
        if result.sol is None or not np.all(np.isfinite(result.y)):
            raise IntegrationError("ODE solver returned an invalid segment")

        stop = float(result.t[-1])
        stop_output = int(np.searchsorted(times, stop, side="right"))
        if stop_output > next_output:
            values = result.sol(times[next_output:stop_output]).T
            values[:, ~segment_active] = 0.0
            values[values < LOD] = 0.0
            output[next_output:stop_output] = values
            next_output = stop_output

        event_index = next(
            (i for i, event_times in enumerate(result.t_events) if len(event_times)),
            None,
        )
        if event_index is None:
            state = result.y[:, -1]
            current = stop
            break

        state = result.y[:, -1]
        if segment_active[event_index]:
            state[event_index] = 0.0
            active[event_index] = False
        else:
            state[event_index] = LOD * (1.0 + nudge)
            active[event_index] = True
        current = stop
        event_count += 1
        if event_count > max_events:
            raise IntegrationError("too many LOD extinction/reactivation events")

    if next_output < len(times):
        output[next_output:] = np.where(active & (state >= LOD), state, 0.0)
    if not np.all(np.isfinite(output)) or np.any(output < 0):
        raise IntegrationError("hybrid-state reconstruction failed")
    output[output < LOD] = 0.0
    return output
