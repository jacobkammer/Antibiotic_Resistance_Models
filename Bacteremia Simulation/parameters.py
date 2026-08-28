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
LOD         = 10.0      # CFU/mL: limit of detection; below this a compartment is "cleared"

# Initial condition [S_b, R_b, S_res, R_res]: blood sterile, reservoir seeded
Y0 = [0.0, 0.0, 100.0, 100.0]

# ---------------------------------------------------------------------------
# Baseline model parameters
# ---------------------------------------------------------------------------
BASE_PARAMS = {
    # --- growth rates (h^-1) ---
    "rho_S":            0.72,     # sensitive, blood. Raised from 0.61 to offset EC50_L 1.0 -> 0.6:
                                   # reservoir kill scales as Emax_l * (rho_res_S / rho_S), so a
                                   # higher rho_S weakens reservoir killing.
    "rho_R":            0.649,    # resistant, blood -- 9.9% fitness cost relative to rho_S
    "rho_res_S":        0.19415,  # sensitive, reservoir -- 1.1x rho_res_R
    "rho_res_R":        0.1765,   # resistant, reservoir -- 9.1% fitness cost. Held in a narrow
                                   # window: just above the reservoir's own persistence threshold
                                   # (0.17594055), below which R_res dies out on its own, and below
                                   # the point where R_b wins the pre-treatment blood race outright
                                   # and no visible S_b infection establishes.

    # --- pharmacodynamics ---
    "Emax_v":           1.7,      # maximal vancomycin effect (acts on the sensitive strain only)
    "EC50_V":           0.25,     # mg/L
    "Emax_l":           0.8,      # maximal linezolid effect (acts on both strains). Fixed;
                                   # decoupled from rho_S. NOTE: MC_emax_lzd.py samples around
                                   # 1.085, not this value -- see SAMPLING below.
    "EC50_L":           0.6,      # mg/L. Single project-wide value: this is both the fixed
                                   # baseline and MC_ec50_lzd.py's sampling mean.

    # --- carrying capacities (CFU/mL), shared by both strains within a compartment ---
    "B_max_blood":      6000,
    "B_max_reservoir":  1e4,

    # --- reservoir tissue penetration (fraction of blood concentration) ---
    "van_res_fraction": 0.15,
    "lzd_res_fraction": 0.30,

    # --- inter-compartmental exchange (h^-1) ---
    "f_r_b":            5e-5,     # reservoir -> blood
    "f_b_r":            1e-5,     # blood -> reservoir (1/5 of efflux)
}

# ---------------------------------------------------------------------------
# Host immune response (lives on the ImmuneResponse object, not in BASE_PARAMS)
# ---------------------------------------------------------------------------
K_IMMUNE_DEFAULT = 0.12   # model_Bacteremia.py's own class default
EFF_BLOOD        = 1.0    # immune efficacy multiplier, blood
EFF_RES          = 0.1    # immune efficacy multiplier, reservoir (10x reduced access)

# ---------------------------------------------------------------------------
# Monte Carlo sampling distributions: name -> (mean, CV)
# Log-normal, mean-corrected so E[X] equals the mean given here.
# ---------------------------------------------------------------------------
SAMPLING = {
    "EC50_L":   (0.6,   1.1),     # MC_ec50_lzd.py          -- equals the BASE_PARAMS value
    "Emax_l":   (1.085, 0.307),   # MC_emax_lzd.py          -- deliberately above the 0.8 baseline
    "k_immune": (0.243, 0.25),    # MC_immune_response.py   -- above the 0.12 class default
    "S_res_0":  (100.0, 60.0),    # MC_initial_conditions.py -- CV deliberately extreme
    "R_res_0":  (100.0, 60.0),    # MC_initial_conditions.py
}

NUM_ITERATIONS = 1000

# ---------------------------------------------------------------------------
# Reference values at the parameterisation above (regenerate after any change)
#
#   EC50_L    escape threshold 0.488 mg/L    Monte Carlo escape  43.6%  (436/1000)
#   Emax_l    escape threshold 0.843 h^-1    Monte Carlo escape  23.8%  (238/1000)
#   k_immune  escape threshold 0.203 h^-1    Monte Carlo relapse 26.0%  (260/1000)
#   f_r_b     establishment 2.230e-5 h^-1, clearance 8.329e-3 h^-1
#   S_res_0   no threshold in [1, 2000] -- R_res escapes at every value tested
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

_MANIFEST = "figure_provenance.tsv"


def param_hash():
    """Short, stable hash of everything that can change a simulation result."""
    payload = {
        "BASE_PARAMS": BASE_PARAMS, "TOTAL_H": TOTAL_H, "VANCO_START": VANCO_START,
        "LOD": LOD, "Y0": Y0, "K_IMMUNE_DEFAULT": K_IMMUNE_DEFAULT,
        "EFF_BLOOD": EFF_BLOOD, "EFF_RES": EFF_RES, "SAMPLING": SAMPLING,
        "NUM_ITERATIONS": NUM_ITERATIONS,
    }
    blob = _json.dumps(payload, sort_keys=True, default=str).encode()
    return _hashlib.sha256(blob).hexdigest()[:12]


def save_figure(target, out_file, script):
    """savefig() with provenance. `target` may be a Figure or the pyplot module.

    Writes the parameter hash into the PNG metadata and records a row in
    figure_provenance.tsv so stale figures are detectable from git alone.
    """
    h = param_hash()
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


def _check():
    """Report figures whose recorded hash differs from the current parameters."""
    current = param_hash()
    print(f"current param_hash: {current}\n")
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
            elif h != current:
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
