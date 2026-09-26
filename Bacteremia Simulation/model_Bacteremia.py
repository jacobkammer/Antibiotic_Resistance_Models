# =============================================================================
# Dual-reservoir model: PK, immune response, and the ODE right-hand side.
# Imported by every analysis script (threshold sweeps, Monte Carlo analyses,
# baseline/initial-condition checks, study_summary_figures.py, and
# test_solver_convergence.py); all parameters come from parameters.py.
# =============================================================================
import numpy as np
import os
import sys

from parameters import (
    BASE_PARAMS, EFF_BLOOD, EFF_RES, HILL_COEFFICIENTS, K_IMMUNE_DEFAULT,
    PK_PARAMS, TOTAL_H, VANCO_START, Y0,
)
from solver import integrate_model


class PharmacokineticModel:
    """
    Two-compartment PK model with fixed clearance values.
    Vancomycin: CL = 5.0 L/h, Vd = 50 L → ke = 0.10 h⁻¹, t½ = 6.9 h
    Linezolid:  CL = 6.0 L/h, Vd = 45 L → ke = 0.133 h⁻¹, t½ = 5.2 h

    Literature basis:
      - Moise-Broder PA et al. Clin Pharmacokinet 2004 (vancomycin)
      - Rayner CR et al. Antimicrob Agents Chemother 2003 (linezolid)
    """

    def __init__(self):
        for name, value in PK_PARAMS.items():
            setattr(self, name, value)

    def concentration_function(self, drug_type: str, total_time_h: float, start_h: float = 0):
        if drug_type == 'vancomycin':
            dose, interval, duration = self.van_dose, self.van_interval, self.van_duration
            cl, volume, fu           = self.van_cl,   self.van_volume,   self.van_fu
        else:
            dose, interval, duration = self.lzd_dose, self.lzd_interval, self.lzd_duration
            cl, volume, fu           = self.lzd_cl,   self.lzd_volume,   self.lzd_fu

        ke       = cl / volume
        t_points = np.linspace(0, total_time_h, int(total_time_h * 10) + 1)
        conc     = np.zeros_like(t_points)

        for dt in np.arange(start_h, start_h + duration, interval):
            mask = t_points >= dt
            conc[mask] += (dose / volume) * np.exp(-ke * (t_points[mask] - dt))

        conc[t_points < start_h] = 0

        return lambda t: np.interp(t, t_points, conc * fu)


class ImmuneResponse:
    """Fixed immune clearance model."""

    def __init__(self, k_immune=K_IMMUNE_DEFAULT, eff_blood=EFF_BLOOD,
                 eff_res=EFF_RES):
        # A single blood efficacy multiplier is applied to both strains; the
        # reservoir multiplier represents reduced compartmental immune access.
        self.k_immune  = k_immune
        self.eff_blood = eff_blood
        self.eff_res   = eff_res

    def compute(self, t=None):
        return self.k_immune


# --- ODE system: Blood and Reservoir for S and R ---
def dual_reservoir_model(y, t, params, van_func, lzd_func, immune_model):
    S_b, R_b, S_res, R_res = y

    S_b   = max(0.0, S_b)
    R_b   = max(0.0, R_b)
    S_res = max(0.0, S_res)
    R_res = max(0.0, R_res)

    V = max(0.0, van_func(t))
    L = max(0.0, lzd_func(t))

    # -------------------------------------------------------------------------
    # Pharmacodynamics
    # -------------------------------------------------------------------------
    # Vancomycin: bactericidal
    h_V = HILL_COEFFICIENTS["vancomycin"]
    vancomycin_kill = (params['Emax_v'] * V**h_V) / (params['EC50_V']**h_V + V**h_V)

    # Linezolid: BACTERIOSTATIC
    h_L = HILL_COEFFICIENTS["linezolid"]
    E_L = (params['Emax_l'] * L**h_L) / (params['EC50_L']**h_L + L**h_L)

    # -------------------------------------------------------------------------
    # Logistic carrying capacity
    # -------------------------------------------------------------------------
    logistic_blood = max(0.0, 1.0 - (S_b + R_b) / params['B_max_blood'])
    logistic_res   = max(0.0, 1.0 - (S_res + R_res) / params['B_max_reservoir'])

    # -------------------------------------------------------------------------
    # Immune killing
    # -------------------------------------------------------------------------
    immune_eff = immune_model.compute(t)
    imm_S_b   = immune_model.eff_blood * immune_eff * S_b
    imm_R_b   = immune_model.eff_blood * immune_eff * R_b
    imm_S_res = immune_model.eff_res   * immune_eff * S_res
    imm_R_res = immune_model.eff_res   * immune_eff * R_res

    # -------------------------------------------------------------------------
    # Exchange: one-way seeding from the reservoir into blood. There is no
    # blood -> reservoir return term.
    # -------------------------------------------------------------------------
    k_r_b = params['k_r_b']

    # -------------------------------------------------------------------------
    # Differential equations
    # -------------------------------------------------------------------------
    # Blood — Sensitive (S_b): Susceptible to both V and L
    dS_b = (
        params['rho_S'] * S_b * logistic_blood
        - E_L * S_b
        - imm_S_b
        - vancomycin_kill * S_b
        + k_r_b * S_res
    )

    # Blood — Resistant (R_b): Susceptible ONLY to linezolid (L)
    dR_b = (
        params['rho_R'] * R_b * logistic_blood
        - E_L * R_b
        - imm_R_b
        + k_r_b * R_res
    )

    # Reservoir tissue penetration effects
    van_res_fraction = params.get('van_res_fraction', 1.0)
    V_res = van_res_fraction * V
    vancomycin_kill_res = (params['Emax_v'] * V_res**h_V) / \
                          (params['EC50_V']**h_V + V_res**h_V)

    lzd_res_fraction = params['lzd_res_fraction']
    # Reduced linezolid efficacy against slow-growing, biofilm/SCV bone bacteria:
    # alpha = params['lzd_res_efficacy'] (parameters.py), an independent
    # calibrated factor deliberately not tied to the reservoir growth rates.
    # This is the only place alpha is used.
    Emax_l_res = params['Emax_l'] * params['lzd_res_efficacy']

    L_res = lzd_res_fraction * L   
    E_L_res_S = (Emax_l_res * L_res**h_L) / (params['EC50_L']**h_L   + L_res**h_L)
    E_L_res_R = (Emax_l_res * L_res**h_L) / (params['EC50_L']**h_L   + L_res**h_L)

    # Reservoir — Sensitive (S_res)
    dS_res = (
        params['rho_res_S'] * S_res * logistic_res
        - E_L_res_S * S_res
        - imm_S_res
        - vancomycin_kill_res * S_res
        - k_r_b * S_res
    )

    # Reservoir — Resistant (R_res)
    dR_res = (
        params['rho_res_R'] * R_res * logistic_res
        - E_L_res_R * R_res
        - imm_R_res
        - k_r_b * R_res
    )

    return [dS_b, dR_b, dS_res, dR_res]


if __name__ == "__main__":
    print("Dual-reservoir model (decoupled growth) starting...", flush=True)

    total_h = TOTAL_H
    vanco_start = VANCO_START

    pk           = PharmacokineticModel()
    immune_model = ImmuneResponse()

    lzd_start = vanco_start + pk.van_duration
    lzd_end   = lzd_start   + pk.lzd_duration

    van_func = pk.concentration_function('vancomycin', total_h, vanco_start)
    lzd_func = pk.concentration_function('linezolid',  total_h, lzd_start)

    params = dict(BASE_PARAMS)
    y0 = list(Y0)
    t_eval = np.linspace(0, total_h, 1150)

    solution = integrate_model(
        dual_reservoir_model, y0, t_eval,
        (params, van_func, lzd_func, immune_model),
    )

    S_b   = np.clip(solution[:, 0], 1, None)
    
    t_days           = t_eval / 24.0
    vanco_start_days = vanco_start / 24.0

    net_growth_blood = params['rho_S'] - immune_model.eff_blood * immune_model.k_immune
    print(f"Net blood growth before Abx: {net_growth_blood:.3f} h^-1", flush=True)
    assert net_growth_blood > 0, "ERROR: Host clears infection without antibiotics!"

    print(f"Reservoir -> blood exchange rate: k_r_b = {params['k_r_b']}", flush=True)
    pre_tx_idx = t_eval <= vanco_start
    peak_S_b   = np.max(S_b[pre_tx_idx])
    print(f"Peak S_b before vancomycin: {peak_S_b:.2e} CFU/mL", flush=True)

    sys.exit(0)
