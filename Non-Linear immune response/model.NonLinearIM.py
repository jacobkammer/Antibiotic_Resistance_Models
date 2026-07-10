# =============================================================================
# Model for Monte Carlo exploration (Saturable Immune Version)
# Used by run_MC_transfer_rates.py / run_MC_4_panels.py
# =============================================================================
import numpy as np
from scipy.integrate import odeint
import os
import sys


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
        # Vancomycin
        self.van_dose     = 1000    # mg (15 mg/kg for 67 kg patient)
        self.van_interval = 8       # h (q8h dosing)
        self.van_duration = 96      # h (4-day course)
        self.van_volume   = 50      # L (Vd)
        self.van_fu       = 0.50    # free fraction (protein binding ~50%)
        self.van_cl       = 5.0     # L/h (CrCl ~80 mL/min)

        # Linezolid
        self.lzd_dose     = 600     # mg (standard 600 mg q12h)
        self.lzd_interval = 12      # h (q12h dosing)
        self.lzd_duration = 336     # h (14-day course)
        self.lzd_volume   = 45      # L
        self.lzd_fu       = 0.70    # free fraction (~70%)
        self.lzd_cl       = 6.0     # L/h → ke = 0.133 h⁻¹, t½ ≈ 5.2 h

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
    """
    Saturable immune clearance model based on Mikkaichi et al. (2019).
    Host defense capacity scales non-linearly using a Hill function driven 
    by local total bacterial burdens (T).
    """

    def __init__(self, c_N=0.12, a=0.05, h=1.1, Im=1.0, eff_blood=1.0, eff_res=0.1):
        self.c_N       = c_N        # Baseline clearance rate (h⁻¹)
        self.a         = a          # Saturation constant
        self.h         = h          # Hill coefficient
        self.Im        = Im         # Immune status (1 = Active)
        self.eff_blood = eff_blood  # Blood performance modifier
        self.eff_res   = eff_res    # Reservoir/Biofilm shield factor

    def compute(self, T: float):
        """Returns dynamic saturable clearance rate based on bacterial density."""
        T_safe = max(0.0, T)
        saturation_modifier = 1.0 / (1.0 + self.a * (T_safe ** self.h))
        return self.c_N * saturation_modifier * self.Im


# --- ODE system: Blood and Reservoir for S and R ---
def dual_reservoir_model(y, t, params, van_func, lzd_func, immune_model):
    S_b, R_b, S_res, R_res = y

    S_b   = max(0.0, S_b)
    R_b   = max(0.0, R_b)
    S_res = max(0.0, S_res)
    R_res = max(0.0, R_res)

    # Smooth floor transitions to prevent solver stiffness
    _nf = 10.0
    _ns = 2
    smooth_S_b   = S_b**_ns   / (_nf**_ns + S_b**_ns)
    smooth_R_b   = R_b**_ns   / (_nf**_ns + R_b**_ns)
    smooth_S_res = S_res**_ns / (_nf**_ns + S_res**_ns)
    smooth_R_res = R_res**_ns / (_nf**_ns + R_res**_ns)

    V = max(0.0, van_func(t))
    L = max(0.0, lzd_func(t))

    # -------------------------------------------------------------------------
    # Pharmacodynamics
    # -------------------------------------------------------------------------
    # Vancomycin: bactericidal
    h_V = 1.0
    vancomycin_kill = (params['Emax_v'] * V**h_V) / (params['EC50_V']**h_V + V**h_V)

    # Linezolid: BACTERIOSTATIC
    h_L = 1.0
    E_L = (params['Emax_l'] * L**h_L) / (params['EC50_L']**h_L + L**h_L)

    # -------------------------------------------------------------------------
    # Logistic carrying capacity
    # -------------------------------------------------------------------------
    logistic_blood = max(0.0, 1.0 - (S_b + R_b) / params['B_max_blood'])
    logistic_res   = max(0.0, 1.0 - (S_res + R_res) / params['B_max_reservoir'])

    # -------------------------------------------------------------------------
    # Immune killing (Updated to calculate locally based on total local burdens)
    # -------------------------------------------------------------------------
    immune_eff_blood = immune_model.compute(S_b + R_b)
    immune_eff_res   = immune_model.compute(S_res + R_res)

    imm_S_b   = immune_model.eff_blood * immune_eff_blood * S_b
    imm_R_b   = immune_model.eff_blood * immune_eff_blood * R_b
    imm_S_res = immune_model.eff_res   * immune_eff_res * S_res
    imm_R_res = immune_model.eff_res   * immune_eff_res * R_res

    # -------------------------------------------------------------------------
    # Exchange rates
    # -------------------------------------------------------------------------
    f_r_b = params['f_r_b']   
    f_b_r = params['f_b_r']   

    # -------------------------------------------------------------------------
    # Differential equations
    # -------------------------------------------------------------------------
    # Blood — Sensitive (S_b)
    dS_b = (
        params['rho_S'] * S_b * logistic_blood * smooth_S_b
        - E_L * S_b
        - imm_S_b
        - vancomycin_kill * S_b
        + f_r_b * S_res - f_b_r * S_b
    )

    # Blood — Resistant (R_b)
    dR_b = (
        params['rho_R'] * R_b * logistic_blood * smooth_R_b
        - E_L * R_b
        - imm_R_b
        + f_r_b * R_res - f_b_r * R_b
    )

    # Reservoir tissue penetration effects
    van_res_fraction = params.get('van_res_fraction', 1.0)
    V_res = van_res_fraction * V
    vancomycin_kill_res = (params['Emax_v'] * V_res**h_V) / \
                          (params['EC50_V']**h_V + V_res**h_V)

    lzd_res_fraction = params.get('lzd_res_fraction', 0.45)
    Emax_l_res = params['Emax_l'] * (params['rho_res_S'] / params['rho_S'])

    L_res = lzd_res_fraction * L   
    E_L_res_S = (Emax_l_res * L_res**h_L) / (params['EC50_L']**h_L   + L_res**h_L)
    E_L_res_R = (Emax_l_res * L_res**h_L) / (params['EC50_L']**h_L   + L_res**h_L)

    # Reservoir — Sensitive (S_res)
    dS_res = (
        params['rho_res_S'] * S_res * logistic_res * smooth_S_res
        - E_L_res_S * S_res
        - imm_S_res
        - vancomycin_kill_res * S_res
        - f_r_b * S_res + f_b_r * S_b
    )

    # Reservoir — Resistant (R_res)
    dR_res = (
        params['rho_res_R'] * R_res * logistic_res * smooth_R_res
        - E_L_res_R * R_res
        - imm_R_res
        - f_r_b * R_res + f_b_r * R_b
    )

    return [dS_b, dR_b, dS_res, dR_res]


if __name__ == "__main__":
    print("Dual-reservoir model (Saturable Immune Response) starting...", flush=True)

    total_h = 1344   
    fitness_cost = 0.20   
    vanco_start = 504  

    pk           = PharmacokineticModel()
    immune_model = ImmuneResponse()  # Uses the new defaults: c_N=0.12, a=0.05, h=1.1

    lzd_start = vanco_start + pk.van_duration
    lzd_end   = lzd_start   + pk.lzd_duration

    van_func = pk.concentration_function('vancomycin', total_h, vanco_start)
    lzd_func = pk.concentration_function('linezolid',  total_h, lzd_start)

    # Growth Parameters
    rho_S = 0.16        
    rho_R = (1 - fitness_cost) * rho_S  

    params = {
        'rho_S':           rho_S,   
        'rho_R':           rho_R,   
        'rho_res_S':       0.035,   
        'rho_res_R':       0.035 * (1 - fitness_cost),  
        'Emax_v':          0.40,   
        'EC50_V':          1.5,    
        'Emax_l':          rho_S,   
        'EC50_L':          1.0,    
        'B_max_blood':     5e5,   
        'B_max_reservoir': 1e7,   
        'van_res_fraction':0.15,  
        'lzd_res_fraction':0.45,  
        'f_r_b':           5e-5,   
        'f_b_r':           1e-5,    
    }

    y0 = [0, 0, 100, 100]   
    t_eval = np.linspace(0, total_h, 800)

    solution = odeint(dual_reservoir_model, y0, t_eval,
                      args=(params, van_func, lzd_func, immune_model),
                      rtol=1e-8, atol=1e-10, mxstep=5000)

    S_b = np.clip(solution[:, 0], 1, None)
    
    # Calculate baseline growth at t=0 where loads are empty, making saturation multiplier = 1.0
    net_growth_blood = rho_S - immune_model.c_N
    print(f"Net baseline blood growth before saturation: {net_growth_blood:.3f} h⁻¹", flush=True)

    sys.exit(0)