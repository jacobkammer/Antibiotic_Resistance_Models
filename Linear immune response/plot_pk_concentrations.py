"""
Plot total serum (blood) concentration and estimated reservoir (tissue)
concentration for vancomycin and linezolid over the simulated treatment course.

Blood compartment: total drug concentration (bound + unbound) -- what a
clinical serum assay reports. The PK model tracks only the *free* fraction
internally (used for pharmacodynamic killing in model.LinearIM.py), so
total = free / fu.

Reservoir compartment: not an independent PK compartment in the model. It is
estimated the same way dual_reservoir_model() estimates it for PD purposes --
a fixed penetration fraction of the free serum concentration
(van_res_fraction, lzd_res_fraction).
"""

import importlib.util

import matplotlib.pyplot as plt
import numpy as np

# ── Load model module ────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("model_mod", "model.LinearIM.py")
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)

# ── Simulation timeline (matches model.LinearIM.py) ──────────────────────────
TOTAL_H     = 1944   # 21d pre-tx + 4d vancomycin + 42d linezolid + 14d post-tx follow-up
VANCO_START = 504    # h
t_eval = np.linspace(0, TOTAL_H, 4000)
t_days = t_eval / 24.0

pk = model_mod.PharmacokineticModel()

lzd_start = VANCO_START + pk.van_duration
lzd_end   = lzd_start + pk.lzd_duration

van_func = pk.concentration_function("vancomycin", TOTAL_H, VANCO_START)
lzd_func = pk.concentration_function("linezolid",  TOTAL_H, lzd_start)

# Free (unbound) serum concentration -- what the PD equations act on
van_free = van_func(t_eval)
lzd_free = lzd_func(t_eval)

# Total serum concentration = free / unbound fraction
van_total = van_free / pk.van_fu
lzd_total = lzd_free / pk.lzd_fu

# ── Reservoir (tissue) concentration estimate ────────────────────────────────
# Penetration fractions match the baseline params in model.LinearIM.py
VAN_RES_FRACTION = 0.15
LZD_RES_FRACTION = 0.45

van_reservoir = VAN_RES_FRACTION * van_free
lzd_reservoir = LZD_RES_FRACTION * lzd_free

# ── Treatment window boundaries (days) ────────────────────────────────────────
vanco_start_d = VANCO_START / 24.0
lzd_start_d   = lzd_start / 24.0
lzd_end_d     = lzd_end / 24.0

VAN_COLOR = "steelblue"
LZD_COLOR = "darkorange"


def _shade_windows(ax):
    ax.axvspan(vanco_start_d, lzd_start_d, color=VAN_COLOR, alpha=0.08, label="Vancomycin window")
    ax.axvspan(lzd_start_d, lzd_end_d,     color=LZD_COLOR, alpha=0.08, label="Linezolid window")


# ── Figure 1: Blood / serum compartment ───────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
_shade_windows(ax1)
ax1.plot(t_days, van_total, color=VAN_COLOR, lw=1.6, label="Vancomycin (total serum)")
ax1.plot(t_days, lzd_total, color=LZD_COLOR, lw=1.6, label="Linezolid (total serum)")

ax1.set_xlim(0, t_days[-1])
ax1.set_ylim(bottom=0)
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Total serum concentration (mg/L)")
ax1.set_title("Blood Compartment — Total Serum Antibiotic Concentration")
ax1.grid(True, ls=":", alpha=0.4)
ax1.legend(loc="upper right", fontsize=9, framealpha=0.85)

fig1.savefig("blood_serum_concentration.png", dpi=300, bbox_inches="tight")
print("Saved: blood_serum_concentration.png")

# ── Figure 2: Reservoir compartment (estimated) ───────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
_shade_windows(ax2)
ax2.plot(t_days, van_reservoir, color=VAN_COLOR, lw=1.6,
         label=f"Vancomycin (reservoir, {VAN_RES_FRACTION:.0%} penetration)")
ax2.plot(t_days, lzd_reservoir, color=LZD_COLOR, lw=1.6,
         label=f"Linezolid (reservoir, {LZD_RES_FRACTION:.0%} penetration)")

ax2.set_xlim(0, t_days[-1])
ax2.set_ylim(bottom=0)
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Estimated reservoir concentration (mg/L)")
ax2.set_title("Reservoir Compartment — Estimated Antibiotic Concentration")
ax2.grid(True, ls=":", alpha=0.4)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.85)

fig2.savefig("reservoir_concentration.png", dpi=300, bbox_inches="tight")
print("Saved: reservoir_concentration.png")

# ── Summary stats ──────────────────────────────────────────────────────────────
def _peak_in_window(conc, t, window_start_d, window_end_d):
    mask = (t >= window_start_d) & (t <= window_end_d)
    return np.max(conc[mask]) if mask.any() else float("nan")


print("\nPeak concentrations during each drug's own dosing window (mg/L):")
print(f"  Vancomycin  serum total  : {_peak_in_window(van_total, t_days, vanco_start_d, lzd_start_d):.2f}")
print(f"  Vancomycin  reservoir    : {_peak_in_window(van_reservoir, t_days, vanco_start_d, lzd_start_d):.2f}")
print(f"  Linezolid   serum total  : {_peak_in_window(lzd_total, t_days, lzd_start_d, lzd_end_d):.2f}")
print(f"  Linezolid   reservoir    : {_peak_in_window(lzd_reservoir, t_days, lzd_start_d, lzd_end_d):.2f}")

plt.show()
