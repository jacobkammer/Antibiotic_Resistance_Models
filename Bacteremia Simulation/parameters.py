"""
Single source of truth for the dual-reservoir model's parameterisation.

WHY THIS FILE EXISTS
--------------------
Every parameter used to be written out independently in each of the eleven
scripts in this project. That duplication caused real errors: EC50_L silently
carried two different values (1.0 as the fixed baseline, 0.6 as the Monte Carlo
sampling mean) while a docstring asserted they were equal, and each
recalibration required the same number to be edited in ten places, with the
LaTeX write-ups drifting out of step behind them.

Import from here instead of restating values:

    from parameters import BASE_PARAMS, LOD, TOTAL_H, VANCO_START, Y0

A script that sweeps or samples one parameter overrides it per call, e.g.

    params_i = {**BASE_PARAMS, "EC50_L": ec50_value}

so BASE_PARAMS can carry a value for every key even where a given script
varies one of them.

CHANGING A PARAMETER
--------------------
Change it here, then rerun the affected scripts. The escape thresholds are
re-derived by root-finding rather than hardcoded, so they will follow -- but
they are NOT stable under parameter change, and the Monte Carlo escape
fractions are highly sensitive to where each sampling mean sits relative to
its threshold. Rerun the deterministic threshold sweeps and check the values
in the "Reference values" block below before trusting any downstream figure
or manuscript number.
"""

# ---------------------------------------------------------------------------
# Treatment protocol and observation
# ---------------------------------------------------------------------------
TOTAL_H     = 1944      # h: 21 d pre-treatment + 4 d vancomycin + 42 d linezolid + 14 d follow-up
VANCO_START = 504       # h: vancomycin begins on day 21
LOD         = 10.       # CFU/mL: hard extinction boundary; sub-LOD burden is exactly zero

PK_PARAMS = {
    "van_dose": 1.0e3, "van_interval": 8.0, "van_duration": 96.,
    "van_volume": 50., "van_fu": 0.50, "van_cl": 5.0,
    "lzd_dose": 6.0e2, "lzd_interval": 12., "lzd_duration": 1008.,
    "lzd_volume": 45., "lzd_fu": 0.70, "lzd_cl": 6.0,
}
HILL_COEFFICIENTS = {"vancomycin": 1.0, "linezolid": 1.0}

# Initial condition [S_b, R_b, S_res, R_res]: blood sterile, reservoir seeded
Y0 = [0.0, 0.0, 1.0e2, 1.0e2]

# ---------------------------------------------------------------------------
# Baseline model parameters
# ---------------------------------------------------------------------------
BASE_PARAMS = {
    # --- growth rates (h^-1) ---
    "rho_S":            1.1,      # sensitive, blood
    "rho_R":            0.99,     # resistant, blood -- 10% fitness cost
    "rho_res_S":        0.50,     # sensitive, reservoir
    "rho_res_R":        0.45,     # resistant, reservoir -- 10% fitness cost

    # --- pharmacodynamics ---
    "Emax_v":           1.7,      # maximal vancomycin effect (acts on the sensitive strain only)
    "EC50_V":           0.25,     # mg/L
    "Emax_l":           1.65,     # maximal linezolid effect (Schmidt et al.); also the Monte Carlo mean
    "EC50_L":           1.13,     # mg/L (Schmidt et al.). Single project-wide value: this is both the fixed
                                   # baseline and MC_ec50_lzd.py's sampling mean.

    # --- carrying capacities (CFU/mL), shared by both strains within a compartment ---
    "B_max_blood":      6.0e3,
    "B_max_reservoir":  1.0e7,

    # --- reservoir tissue penetration (fraction of blood concentration) ---
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.40,    # lower end of the reported 0.40-0.60 bone penetration range
    # reservoir linezolid efficacy (alpha): Emax_l_res = Emax_l * lzd_res_efficacy,
    # reduced activity against slow-growing biofilm/SCV bacteria in bone. An
    # independent calibrated factor, NOT derived from the growth rates (it
    # replaced the former rho_res_S / rho_S scaling). Read only by
    # model_Bacteremia.dual_reservoir_model, so every script that integrates the
    # model applies it through BASE_PARAMS; no script varies it. The Emax_l
    # analyses (Emax_lzd_threshold_sweep.py, MC_emax_lzd.py,
    # study_summary_figures.py) vary Emax_l, so the reservoir ceiling scales
    # with each sampled Emax_l at this fixed alpha.
    "lzd_res_efficacy": 0.41,

    # --- inter-compartmental exchange (h^-1): one-way, reservoir -> blood only ---
    "k_r_b":            1.0e-7,   # reservoir -> blood; scaled with the 1e7 reservoir capacity
}

# ---------------------------------------------------------------------------
# Host immune response (lives on the ImmuneResponse object, not in BASE_PARAMS)
# ---------------------------------------------------------------------------
K_IMMUNE_DEFAULT = 0.10   # fixed baseline and Monte Carlo sampling mean
EFF_BLOOD        = 1.0    # immune efficacy multiplier, blood
EFF_RES          = 0.50   # reservoir immune access, relative to blood

# ---------------------------------------------------------------------------
# Monte Carlo sampling distributions: name -> (mean, CV)
# Log-normal, mean-corrected so E[X] equals the mean given here.
# ---------------------------------------------------------------------------
SAMPLING = {
    "EC50_L":   (1.13,  0.60),    # MC_ec50_lzd.py -- equals the fixed baseline
    "Emax_l":   (1.65,  0.080),   # log-normal; equals the fixed baseline
    "k_immune": (0.10,  0.80),    # MC_immune_response.py -- equals the fixed baseline
    "S_res_0":  (1.0e2, 6.0e1),   # MC_initial_conditions.py -- CV deliberately extreme
    "R_res_0":  (1.0e2, 6.0e1),   # MC_initial_conditions.py
}

# Literature-supported range used for deterministic sweeps and root bracketing.
# It does not truncate the log-normal Monte Carlo distribution.
EMAX_L_BOUNDS = (0.70, 2.2)

NUM_ITERATIONS = 1000
TIME_GRID_POINTS = 1450


def canonical_time_grid():
    """Shared output grid containing every treatment boundary exactly."""
    import numpy as np

    van_end = VANCO_START + PK_PARAMS["van_duration"]
    lzd_end = van_end + PK_PARAMS["lzd_duration"]
    return np.unique(np.append(
        np.linspace(0.0, TOTAL_H, TIME_GRID_POINTS),
        [VANCO_START, van_end, lzd_end],
    ))


def bacteremia_outcome_masks(resistant_blood_history, times):
    """Return relapse, persistent-treatment-failure, and resolution masks."""
    import numpy as np

    lzd_end = VANCO_START + PK_PARAMS["van_duration"] + PK_PARAMS["lzd_duration"]
    end_index = int(np.searchsorted(times, lzd_end))
    if end_index >= len(times) or not np.isclose(times[end_index], lzd_end):
        raise ValueError("time grid does not contain linezolid completion")
    # Plotting scripts may represent extinct states as NaN on logarithmic axes.
    # Restore those display masks to the model's exact-zero extinction state.
    history = np.nan_to_num(np.asarray(resistant_blood_history), nan=0.0)
    cleared = history[:, end_index] < LOD
    returned = np.any(history[:, end_index + 1:] >= LOD, axis=1)
    relapse = cleared & returned
    persistent = ~cleared
    resolved = ~(relapse | persistent)
    return relapse, persistent, resolved


def bacteremia_relapse_mask(resistant_blood_history, times):
    """True when blood clears by linezolid end and later becomes detectable."""
    return bacteremia_outcome_masks(resistant_blood_history, times)[0]

# ---------------------------------------------------------------------------
# Canonical numerical integration policy
# ---------------------------------------------------------------------------
# Hybrid extinction/reactivation events enforce LOD as a hard boundary. A
# sub-LOD compartment is zero and cannot grow autonomously; incoming transfer
# from a detectable donor can re-establish it upon reaching 10 CFU/mL.
SOLVER_CONFIG = {
    "method": "LSODA",
    "rtol": 1e-6,
    "atol": 1e-8,        # CFU/mL absolute tolerance within each hybrid segment
    "max_step": 4.0,     # h; at least two internal steps per shortest dosing interval
    "reactivation_nudge": 1e-6,
    "max_extinction_events": 10000,
}

# Reproducible random-number streams used by each Monte Carlo analysis.
RNG_SEEDS = {
    "MC_ec50_lzd": 44,
    "MC_emax_lzd": 44,
    "MC_immune_response": 42,
    "MC_immune_response_traces": 0,
    "MC_initial_conditions": 42,
}

# ---------------------------------------------------------------------------
# Reference values at the parameterisation above (regenerate after any change)
#
# Calibration candidate under full Monte Carlo validation:
#   EC50_L transition   1.3 mg/L (1.349); validated relapse 29% (289/1000)
#   Emax_l transition   1.5 h^-1 (1.549); validated relapse 22% (220/1000)
#   k_immune transition 0.047 h^-1; validated relapse 24% (240/1000)
# (Schmidt linezolid PD: Emax_l = 1.65, EC50_L = 1.13; bone penetration 0.40;
#  independent reservoir efficacy lzd_res_efficacy = 0.41; k_immune = 0.10;
#  B_max_reservoir = 1e7, k_r_b = 1e-7; rho_res_S = 0.50 / rho_res_R = 0.45.)
# Relapse requires R_b below LOD at linezolid completion and R_b >= LOD later
# in follow-up. Reservoir persistence alone is reported separately.
# ---------------------------------------------------------------------------


# ===========================================================================
# Figure provenance
# ===========================================================================
# Figures are gitignored, so git cannot tell when a PNG on disk was produced
# by an older parameterisation than the code that is now checked in. That
# happened repeatedly during development: manuscript tables and figures were
# citing results from a superseded calibration.
#
# Every figure is therefore stamped twice:
#   1. invisibly, in the PNG's own tEXt metadata, so the file itself carries
#      the parameter hash that produced it (visible via `exiftool`, PIL, or
#      `miktex`-independent tooling);
#   2. as a row in figure_provenance.tsv, which IS tracked by git, so a stale
#      figure shows up as an uncommitted diff.
#
# To check for stale figures:   python parameters.py --check
# ===========================================================================
import datetime as _dt
import hashlib as _hashlib
import json as _json
import os as _os
import pathlib as _pathlib
import time as _time

_MANIFEST = "figure_provenance.tsv"


def param_hash(script=None):
    """Hash configuration plus the model and optional generating script."""
    payload = {
        "BASE_PARAMS": BASE_PARAMS, "TOTAL_H": TOTAL_H, "VANCO_START": VANCO_START,
        "LOD": LOD, "Y0": Y0, "K_IMMUNE_DEFAULT": K_IMMUNE_DEFAULT,
        "EFF_BLOOD": EFF_BLOOD, "EFF_RES": EFF_RES, "SAMPLING": SAMPLING,
        "PK_PARAMS": PK_PARAMS, "HILL_COEFFICIENTS": HILL_COEFFICIENTS,
        "EMAX_L_BOUNDS": EMAX_L_BOUNDS, "TIME_GRID_POINTS": TIME_GRID_POINTS,
        "NUM_ITERATIONS": NUM_ITERATIONS, "RNG_SEEDS": RNG_SEEDS,
        "SOLVER_CONFIG": SOLVER_CONFIG,
    }
    source_hashes = {}
    for source in ("model_Bacteremia.py", "solver.py", script):
        if not source:
            continue
        path = _pathlib.Path(source)
        if not path.is_absolute():
            path = _pathlib.Path(__file__).resolve().parent / path.name
        if path.exists():
            source_hashes[path.name] = _hashlib.sha256(path.read_bytes()).hexdigest()
    payload["SOURCE_HASHES"] = source_hashes
    blob = _json.dumps(payload, sort_keys=True, default=str).encode()
    return _hashlib.sha256(blob).hexdigest()[:12]


def save_figure(target, out_file, script):
    """savefig() with provenance. `target` may be a Figure or the pyplot module.

    Writes the parameter hash into the PNG metadata and records a row in
    figure_provenance.tsv so stale figures are detectable from git alone.
    """
    h = param_hash(script)
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    name = _os.path.basename(script)
    target.savefig(
        out_file, dpi=300, bbox_inches="tight",
        metadata={"Software": name, "Comment": f"param_hash={h} generated={stamp}"},
    )
    _record(_os.path.basename(out_file), h, name, stamp)
    return h


def _record(figure, h, script, stamp):
    """Upsert one row of the manifest, keeping it sorted by figure name."""
    lock_path = _MANIFEST + ".lock"
    deadline = _time.monotonic() + 60.0
    while True:
        try:
            descriptor = _os.open(lock_path, _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY)
            _os.close(descriptor)
            break
        except FileExistsError:
            if _time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {lock_path}")
            _time.sleep(0.05)

    try:
        rows = {}
        if _os.path.exists(_MANIFEST):
            with open(_MANIFEST, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip() and not line.startswith("figure\t"):
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) == 4:
                            rows[parts[0]] = parts
        rows[figure] = [figure, h, script, stamp]
        with open(_MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("figure\tparam_hash\tscript\tgenerated\n")
            for k in sorted(rows):
                fh.write("\t".join(rows[k]) + "\n")
    finally:
        _os.unlink(lock_path)


def _check():
    """Report figures not built by the current model and generating script."""
    print(f"current configuration hash: {param_hash()}\n")
    if not _os.path.exists(_MANIFEST):
        print(f"no {_MANIFEST}; run the scripts to generate one")
        return 1
    stale, missing, ok = [], [], 0
    with open(_MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("figure\t"):
                continue
            fig, h, script, stamp = line.rstrip("\n").split("\t")
            if not _os.path.exists(_os.path.join(FIGURES_DIR, fig)):
                missing.append((fig, script))
            elif h != param_hash(script):
                stale.append((fig, h, script))
            else:
                ok += 1
    # PNGs on disk that no script has ever stamped -- the case that let stale
    # figures survive recalibration unnoticed
    recorded = set()
    with open(_MANIFEST, encoding='utf-8') as fh:
        for line in fh:
            if line.strip() and not line.startswith('figure'):
                recorded.add(line.split(chr(9))[0])
    on_disk = ({f.name for f in _pathlib.Path(FIGURES_DIR).glob('*.png')}
               if _os.path.isdir(FIGURES_DIR) else set())
    unstamped = sorted(on_disk - recorded)

    for fig, h, script in stale:
        print(f'  STALE      {fig:42s} built at {h} by {script}')
    for fig, script in missing:
        print(f'  ABSENT     {fig:42s} (recorded, not on disk; rerun {script})')
    for fig in unstamped:
        print(f'  UNSTAMPED  {fig:42s} (on disk, provenance unknown)')
    summary = (f'{ok} current, {len(stale)} stale, {len(missing)} absent, '
               f'{len(unstamped)} unstamped')
    print()
    print('  ' + summary)
    return 1 if (stale or missing or unstamped) else 0


FIGURES_DIR = "figures"

if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        sys.exit(_check())
    print(f"param_hash: {param_hash()}")
