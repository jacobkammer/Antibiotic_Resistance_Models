# =============================================================================
# Model for Monte Carlo exploration
# Used by run_MC_transfer_rates.py
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
        self.lzd_duration = 1008    # h (42-day / 6-week course, standard for endocarditis)
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
    """Fixed immune clearance model."""

    def __init__(self, k_immune=0.12, eff_blood=1.0, eff_res=0.1):
        # Strain-specific eff_blood_S/eff_blood_R (S=1.0, R=5.0) was tried to
        # make R_b clear without relapsing, but R_b never establishes a real,
        # visible infection under any constant R-specific value -- it's a
        # switch between "never appears" and "relapses to ~4e3+ CFU/mL after
        # treatment ends" with no stable middle ground (S_b is crowded out of
        # blood by S_b's own carrying-capacity dominance until vancomycin/
        # linezolid clear it, then R_b's fate purely depends on whether it can
        # out-run reservoir reseeding after linezolid stops). Reverted to a
        # single eff_blood so R_b shows a real, treated infection and accepts
        # the post-treatment relapse -- see baseline_reservoir_clearance.py.
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
    # Immune killing
    # -------------------------------------------------------------------------
    immune_eff = immune_model.compute(t)
    imm_S_b   = immune_model.eff_blood * immune_eff * S_b
    imm_R_b   = immune_model.eff_blood * immune_eff * R_b
    imm_S_res = immune_model.eff_res   * immune_eff * S_res
    imm_R_res = immune_model.eff_res   * immune_eff * R_res

    # -------------------------------------------------------------------------
    # Exchange rates
    # -------------------------------------------------------------------------
    f_r_b = params['f_r_b']   
    f_b_r = params['f_b_r']   

    # -------------------------------------------------------------------------
    # Differential equations
    # -------------------------------------------------------------------------
    # Blood — Sensitive (S_b): Susceptible to both V and L
    dS_b = (
        params['rho_S'] * S_b * logistic_blood * smooth_S_b
        - E_L * S_b
        - imm_S_b
        - vancomycin_kill * S_b
        + f_r_b * S_res - f_b_r * S_b
    )

    # Blood — Resistant (R_b): Susceptible ONLY to linezolid (L)
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
    print("Dual-reservoir model (decoupled growth) starting...", flush=True)

    total_h = 1944   # 21d pre-tx + 4d vancomycin + 42d linezolid + 14d post-tx follow-up
    vanco_start = 504

    pk           = PharmacokineticModel()
    immune_model = ImmuneResponse(k_immune=0.12)

    lzd_start = vanco_start + pk.van_duration
    lzd_end   = lzd_start   + pk.lzd_duration

    van_func = pk.concentration_function('vancomycin', total_h, vanco_start)
    lzd_func = pk.concentration_function('linezolid',  total_h, lzd_start)

    # =========================================================================
    # Adjusted Growth Parameters for Smooth Trajectories
    # =========================================================================
    rho_S = 0.60         # h⁻¹ Lowered from 0.80 for slower post-treatment R_b regrowth; Emax_l auto-follows
    rho_R = 0.55         # h⁻¹ Directly tuned (~8.3% fitness cost relative to rho_S, down from 20%)

    params = {
        'rho_S':           rho_S,
        'rho_R':           rho_R,
        'rho_res_R':       0.1765,  # narrow window just above the reservoir persistence threshold (0.17594055) so S_b can also establish in blood -- see model_Bacteremia.py
        'rho_res_S':       0.19415, # 1.1x rho_res_R -- sensitive strain grows 10% faster in the reservoir (fitness cost for R, mirroring the blood compartment)
        'Emax_v':          0.40,
        'EC50_V':          0.245,
        'Emax_l':          0.8,    # Fixed, decoupled from rho_S (was tied for "perfect bacteriostasis")
        'EC50_L':          1.0,
        'B_max_blood':     6000,
        'B_max_reservoir': 1e4,     # Lowered from 4.5e6 so escaped R_res plateaus well above the LOD but far below its old level
        'van_res_fraction':0.15,
        'lzd_res_fraction':0.30,    # Lowered from 0.45 -- Emax_l_res scales with rho_res_S, so raising rho_res_S 10% above rho_res_R also raises the reservoir linezolid kill rate; without this reduction R_res is wiped out entirely rather than persisting
        'f_r_b':           5e-5,   # Original value
        'f_b_r':           1e-5,
    }

    y0 = [0, 0, 100, 100]   
    t_eval = np.linspace(0, total_h, 1150)

    solution = odeint(dual_reservoir_model, y0, t_eval,
                      args=(params, van_func, lzd_func, immune_model),
                      rtol=1e-8, atol=1e-10, mxstep=5000)

    S_b   = np.clip(solution[:, 0], 1, None)
    
    t_days           = t_eval / 24.0
    vanco_start_days = vanco_start / 24.0

    net_growth_blood = rho_S - immune_model.eff_blood * immune_model.k_immune
    print(f"Net blood growth before Abx: {net_growth_blood:.3f} h^-1", flush=True)
    assert net_growth_blood > 0, "ERROR: Host clears infection without antibiotics!"

    print(f"Exchange rates: f_r_b = {params['f_r_b']}, f_b_r = {params['f_b_r']}", flush=True)
    pre_tx_idx = t_eval <= vanco_start
    peak_S_b   = np.max(S_b[pre_tx_idx])
    print(f"Peak S_b before vancomycin: {peak_S_b:.2e} CFU/mL", flush=True)

    sys.exit(0)