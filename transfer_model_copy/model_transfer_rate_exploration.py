# =============================================================================
# model.transfer_rate_exploration.py
# Copy of model.MRSA_dual_reservoir.py for transfer-rate Monte Carlo exploration.
# Used by run_MC_transfer_rates.py, which varies f_r_b and f_b_r across their
# full plausible range while holding all other parameters and initial conditions
# fixed. See run_MC_transfer_rates.py for details.
# =============================================================================
import numpy as np
from scipy.integrate import odeint
import os
import sys

# =============================================================================
# CORRECTIONS SUMMARY (vs model.06042026.py)
# =============================================================================
# 1. EC50_V raised from 0.384 → 1.5 mg/L (above vancomycin MIC for MRSA)
# 2. Emax_v raised from 0.096 → 2.0 h⁻¹ (bactericidal max kill, literature-based)
# 3. Emax_l corrected to = rho_S; linezolid modeled as BACTERIOSTATIC
#    (growth inhibition), not bactericidal (kill term)
# 4. k_immune lowered from 0.3 → 0.12 h⁻¹ so rho_S > k_immune and infection
#    can establish (net growth positive before antibiotics)
# 5. Exchange rates raised from ~1e-24 → 5e-4 h⁻¹ (biofilm seeding literature)
# 6. B_max_blood lowered from 4e7 → 5e5 CFU/mL (realistic peak bacteremia)
# 7. B_max_reservoir raised to 1e7 CFU/mL (tissue/biofilm capacity)
# 8. y0: S_b initialized to a small positive value so blood compartment
#    develops realistically (seeding + initial inoculum)
# 9. rho_S kept at 0.63 h⁻¹ (doubling time ~1.1 h, consistent with in vivo
#    S. aureus estimates; see Campion et al. 2005)
# =============================================================================

class PharmacokineticModel:
    """
    One-compartment PK model with fixed clearance values.
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
        # CORRECTION: was 800 mg — standard dose is 600 mg (Zyvox label)
        self.lzd_interval = 12      # h (q12h dosing)
        self.lzd_duration = 336     # h (14-day course; was 912 h = 38 days which is unusually long)
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
    """Fixed immune clearance model.

    k_immune is a first-order clearance rate (h⁻¹) applied to each
    bacterial compartment. It must satisfy:
        rho_S > k_immune  (blood)
    so that infection can establish before antibiotic treatment.

    Literature basis:
      - Regoes RR et al. Antimicrob Agents Chemother 2004 (immune killing)
      - Net in vivo clearance of S. aureus by phagocytes: ~0.1–0.15 h⁻¹
        (Drusano GL, Antimicrob Agents Chemother 2004)

    The current model assumes a constant immune response appropriate for a
    stable, immunocompetent host.
    """

    def __init__(self, k_immune=0.12, eff_blood=1.0, eff_res=0.3):
        self.k_immune  = k_immune
        self.eff_blood = eff_blood
        # CORRECTED eff_res: 0.8 → 0.3
        # Biofilm physically shields bacteria from phagocytosis; immune killing
        # in the reservoir is substantially reduced vs. planktonic blood bacteria.
        # eff_res = 0.3 gives reservoir net growth = rho_res_S - 0.12*0.3 = +0.034 h⁻¹
        # (doubling ~20 h), consistent with slow-growing persistent biofilm.
        self.eff_res = eff_res

    def compute(self, t=None):
        """Return a constant immune clearance rate (h⁻¹)."""
        return self.k_immune


# --- ODE system: Blood and Reservoir for S and R ---
# State vector y = [S_b, R_b, S_res, R_res]   (all in CFU/mL)
def dual_reservoir_model(y, t, params, van_func, lzd_func, immune_model):
    S_b, R_b, S_res, R_res = y

    # Non-negativity clamp
    S_b   = max(S_b,   0.0)
    R_b   = max(R_b,   0.0)
    S_res = max(S_res, 0.0)
    R_res = max(R_res, 0.0)

    # Drug free concentrations (mg/L)
    V = max(0.0, van_func(t))
    L = max(0.0, lzd_func(t))

    # -------------------------------------------------------------------------
    # Pharmacodynamics
    # -------------------------------------------------------------------------
    # Vancomycin: bactericidal — Hill equation → kill rate (h⁻¹)
    #   EC50_V ≈ 1.5 mg/L (near MIC for MRSA, 1–2 mg/L)
    #   Emax_v ≈ 2.0 h⁻¹ (max kill rate; Zhi et al. 2007, Campion et al. 2005)
    #   Hill coefficient h_V = 1 (conservative; some models use 2–4)
    h_V = 1.0
    vancomycin_kill = (params['Emax_v'] * V**h_V) / (params['EC50_V']**h_V + V**h_V)

    # Linezolid: BACTERIOSTATIC — net effect model (clinical PK/PD convention)
    #   E_L is a bacteriostatic effect rate (h⁻¹) subtracted from the growth rate.
    #   Emax_l = rho_S: complete growth arrest of sensitive strain at saturation.
    #   E_L ∈ [0, Emax_l]; (rho_S - E_L) ≥ 0 — drug cannot directly kill via this term.
    #
    # NEW: separate EC50 for resistant strain (EC50_L_R > EC50_L).
    # Higher EC50 means the resistant strain needs higher drug concentrations
    # to be inhibited — this is what makes a strain "resistant" in PD terms.
    # Typical fold-change for linezolid resistance: 4–32×; we use 10× as default.
    h_L = 1.0
    E_L_S = (params['Emax_l'] * L**h_L) / (params['EC50_L']**h_L   + L**h_L)
    E_L_R = (params['Emax_l'] * L**h_L) / (params['EC50_L_R']**h_L + L**h_L)

    # -------------------------------------------------------------------------
    # Logistic carrying capacity (separate for blood vs reservoir)
    # -------------------------------------------------------------------------
    logistic_blood = max(0.0, 1.0 - (S_b + R_b) / params['B_max_blood'])
    logistic_res   = max(0.0, 1.0 - (S_res + R_res) / params['B_max_reservoir'])

    # -------------------------------------------------------------------------
    # Immune killing (first-order, constant)
    # -------------------------------------------------------------------------
    immune_eff = immune_model.compute(t)
    imm_S_b   = immune_model.eff_blood * immune_eff * S_b
    imm_R_b   = immune_model.eff_blood * immune_eff * R_b
    imm_S_res = immune_model.eff_res   * immune_eff * S_res
    imm_R_res = immune_model.eff_res   * immune_eff * R_res

    # -------------------------------------------------------------------------
    # Exchange rates (reservoir ↔ blood)
    # Literature: biofilm seeding to blood ~1e-4 to 1e-3 h⁻¹
    # CORRECTION: raised from ~1e-24 (physically nonsensical) to 5e-4 h⁻¹
    # -------------------------------------------------------------------------
    f_r_b = params['f_r_b']   # reservoir → blood (h⁻¹)
    f_b_r = params['f_b_r']   # blood → reservoir (h⁻¹)

    # -------------------------------------------------------------------------
    # Differential equations
    # -------------------------------------------------------------------------

    # Blood — Sensitive (S_b): linezolid as separate density-independent term
    dS_b = (
        params['rho_S'] * S_b * logistic_blood
        - E_L_S * S_b
        - imm_S_b
        - vancomycin_kill * S_b
        + f_r_b * S_res - f_b_r * S_b
    )

    # Blood — Resistant (R_b): vancomycin-resistant; linezolid effect reduced
    # via elevated EC50_L_R.
    dR_b = (
        params['rho_R'] * R_b * logistic_blood
        - E_L_R * R_b
        - imm_R_b
        + f_r_b * R_res - f_b_r * R_b
    )

    # Reservoir — vancomycin with biofilm factor; linezolid with penetration + scaled Emax
    biofilm_factor = params.get('biofilm_factor', 100.0)
    vancomycin_kill_res = (params['Emax_v'] * V**h_V) / \
                          ((biofilm_factor * params['EC50_V'])**h_V + V**h_V)

    lzd_res_fraction = 0.60
    Emax_l_res = params['Emax_l'] * (params['rho_res_S'] / params['rho_S'])
    L_res = lzd_res_fraction * L   # effective linezolid concentration in reservoir
    E_L_res_S = (Emax_l_res * L_res**h_L) / (params['EC50_L']**h_L   + L_res**h_L)
    E_L_res_R = (Emax_l_res * L_res**h_L) / (params['EC50_L_R']**h_L + L_res**h_L)

    dS_res = (
        params['rho_res_S'] * S_res * logistic_res
        - E_L_res_S * S_res
        - imm_S_res
        - vancomycin_kill_res * S_res
        - f_r_b * S_res + f_b_r * S_b
    )

    dR_res = (
        params['rho_res_R'] * R_res * logistic_res
        - E_L_res_R * R_res
        - imm_R_res
        - f_r_b * R_res + f_b_r * R_b
    )

    return [dS_b, dR_b, dS_res, dR_res]


if __name__ == "__main__":
    print("Dual-reservoir model (corrected) starting...", flush=True)

    # -------------------------------------------------------------------------
    # Adjustable parameters
    # -------------------------------------------------------------------------
    total_h = 1344   # 56 days (21d pre-treatment + 4d vanco + 14d LZD + ~17d follow-up)

    resistance_factor = 0.25   # 25% fitness cost; MRSA literature: 10–30%

    vanco_start = 504  # hours (21 days after infection onset)
    # -------------------------------------------------------------------------

    pk           = PharmacokineticModel()
    immune_model = ImmuneResponse(k_immune=0.12)

    lzd_start = vanco_start + pk.van_duration
    lzd_end   = lzd_start   + pk.lzd_duration

    van_func = pk.concentration_function('vancomycin', total_h, vanco_start)
    lzd_func = pk.concentration_function('linezolid',  total_h, lzd_start)

    # =========================================================================
    # Parameters — corrected and literature-grounded
    # =========================================================================
    rho_S = 1.10 * immune_model.eff_blood * immune_model.k_immune
    # h⁻¹  blood sensitive growth set to 10% above immune killing rate.
    #   immune killing in blood = eff_blood * k_immune = 1.0 * 0.12 = 0.12 h⁻¹
    #   → rho_S = 1.10 * 0.12 = 0.132 h⁻¹
    #   → net blood growth (no Abx) = 0.132 - 0.12 = 0.012 h⁻¹
    #   → doubling time ≈ 58 h (~2.4 days)
    # This is much slower than typical in vivo S. aureus and is intentionally
    # close to the immune clearance ceiling — the infection only narrowly wins
    # against immune killing, so dynamics are dominated by bone seeding (f_r_b)
    # rather than autonomous blood proliferation.
    rho_R = (1 - resistance_factor) * rho_S
    # WARNING: with rho_R < k_immune, resistant blood bacteria have NEGATIVE
    # net growth in the absence of antibiotics and cannot establish on their own.
    # They persist only via continuous seeding from the reservoir.

    params = {
        # ---- Growth rates ----
        'rho_S':     rho_S,   # 0.63 h⁻¹ (blood planktonic)
        'rho_R':     rho_R,   # 0.504 h⁻¹
        'rho_res_S': 0.07,    # h⁻¹  biofilm/tissue (5–10× slower than planktonic)
        'rho_res_R': 0.07 * (1 - resistance_factor),  # fitness cost applies in reservoir too

        # ---- Vancomycin PD ----
        # Emax_v: maximum bactericidal rate (h⁻¹, natural-log units)
        #   Clinical target: blood clearance in 5–9 days for MRSA bacteremia
        #   (Fowler et al. 2006 JAMA; Chang et al. 2003 Medicine)
        #   With rho_S=0.30, k_immune=0.12, free vanco trough ~5 mg/L (Hill~0.77):
        #     net removal = Emax_v*0.77 - rho_S + k_immune
        #     For 7-day clearance: net removal ≈ 0.078 h⁻¹ → Emax_v ≈ 0.36 h⁻¹
        #   Previous value of 2.0 h⁻¹ cleared bacteria in ~13 h (far too fast).
        'Emax_v':  0.40,   # h⁻¹  (~7-day clearance for sensitive MRSA)
        # EC50_V: concentration at 50% max kill; MRSA MIC ~1–2 mg/L (EUCAST)
        'EC50_V':  1.5,    # mg/L (unchanged)

        # ---- Linezolid PD ----
        # Linezolid is BACTERIOSTATIC — at maximum effect, growth is arrested
        # but the drug cannot directly drive populations negative.
        # Setting Emax_l = rho_S ensures (rho_S - E_L) ≥ 0 at all linezolid
        # concentrations; killing only occurs via immune clearance, not the drug.
        'Emax_l':  rho_S,  # h⁻¹  = rho_S for true bacteriostasis (was 1.0 h⁻¹ = bactericidal)
        # EC50_L lowered 3.0 → 1.0 mg/L: EUCAST susceptible S. aureus peaks at 1–2 mg/L.
        # At EC50=3.0, trough free conc (1.88 mg/L) gave only 39% inhibition → S_b
        # net growth positive (+0.064 h⁻¹) despite linezolid → visually wrong.
        # EC50=1.0 gives 65% inhibition at trough → net = -0.016 h⁻¹ (suppressed).
        'EC50_L':  1.0,    # mg/L (sensitive S. aureus; EUCAST susceptible range)
        # EC50_L_R: linezolid EC50 for resistant strain.
        # Linezolid resistance (cfr / 23S rRNA G2576T) raises MIC ~4–32× above sensitive.
        # We use 10× as a moderate, clinically realistic default.
        'EC50_L_R': 10.0,  # mg/L (10× sensitive EC50)

        # ---- Carrying capacities ----
        # B_max_blood: peak bacteremia ~1e4–1e6 CFU/mL in severe MRSA
        #   (Nolan CM & Beaty HN, Am J Med 1976; Wisplinghoff H et al.)
        'B_max_blood':      5e5,   # CFU/mL (was 4e7 — unrealistically high)

        # B_max_reservoir: tissue/biofilm capacity ~1e6–1e9 CFU/g
        'B_max_reservoir':  1e7,   # CFU/mL equiv (was 1e2–1e4 — too small)

        # ---- Biofilm protection ----
        # Biofilm MIC is 100–1000× higher than planktonic MIC for vancomycin
        # IN VITRO. In vivo, achievable tissue concentrations and biofilm
        # heterogeneity make the effective protection factor lower —
        # clinical PK/PD models often use 5–50× (Lewis 2001 AAC; Ceri 1999 AAC;
        # Garrigós 2010 AAC for bone-PK adjustment).
        # Reduced 100 → 10 so vancomycin can partially penetrate the reservoir
        # and clear some bone bacteria in lucky simulations → partial relapse.
        'biofilm_factor': 10.0,

        # ---- Exchange rates ----
        # f_r_b lowered to 5e-5: at 5e-4, seeding from S_res=2.4e5 = 120 CFU/mL/h,
        # overwhelming linezolid's bacteriostatic effect on S_b in blood.
        # 5e-5 gives seeding = 12 CFU/mL/h (within literature range 1e-5–1e-3).
        'f_r_b': 5e-5,    # h⁻¹  reservoir → blood (Kostakioti et al. 2013)
        'f_b_r': 1e-5,    # h⁻¹  blood → reservoir
    }

    # =========================================================================
    # Initial conditions [S_b, R_b, S_res, R_res]
    # =========================================================================
    # INFECTION ORIGIN: bone (reservoir) is the primary site.
    # Bacteria seed bone first during an initial bacteremic event; the blood
    # compartment starts empty and is subsequently seeded from bone via f_r_b.
    # This produces the correct temporal sequence for MRSA osteomyelitis:
    #   bone peaks first → blood peaks later as f_r_b * S_res drives seeding.
    #
    # S_b = 0:   no blood inoculum at t=0; blood builds entirely from bone seeding
    # R_b = 0:   same for resistant strain
    # S_res = 1e4 CFU/mL: established bone infection (sensitive)
    # R_res = 1e3 CFU/mL: small resistant subpopulation in bone (~10% of S_res)
    #
    # Literature basis for bone IC:
    #   MRSA osteomyelitis tissue burden: 1e3–1e6 CFU/g bone
    #   (Norden CW, J Infect Dis 1970; Calhoun JH et al., Clin Orthop 2003)
    y0 = [0, 0, 1e4, 1e3]   # [S_b, R_b, S_res, R_res] in CFU/mL

    # Time grid
    t_eval = np.linspace(0, total_h, 800)

    # Solve
    solution = odeint(dual_reservoir_model, y0, t_eval,
                      args=(params, van_func, lzd_func, immune_model),
                      rtol=1e-8, atol=1e-10, mxstep=5000)

    # Extract (clip at 1 for log-scale plotting only)
    S_b   = np.clip(solution[:, 0], 1, None)
    R_b   = np.clip(solution[:, 1], 1, None)
    S_res = np.clip(solution[:, 2], 1, None)
    R_res = np.clip(solution[:, 3], 1, None)

    t_days           = t_eval / 24.0
    vanco_start_days = vanco_start / 24.0
    lzd_start_days   = lzd_start   / 24.0
    lzd_end_days     = lzd_end     / 24.0

    # Verify net growth before treatment is positive (infection can establish)
    net_growth_blood = rho_S - immune_model.k_immune
    print(f"Net blood growth before Abx: {net_growth_blood:.3f} h⁻¹  "
          f"(must be > 0 for infection to establish)", flush=True)
    assert net_growth_blood > 0, \
        "ERROR: k_immune >= rho_S — bacteria always cleared even without antibiotics!"

    print(f"Resistance factor:  {resistance_factor}  →  rho_R = {rho_R:.4f}  (rho_S = {rho_S})", flush=True)
    print(f"Exchange rates: f_r_b = {params['f_r_b']}, f_b_r = {params['f_b_r']}", flush=True)
    print(f"Initial conditions: S_b={y0[0]:.1f}, R_b={y0[1]:.1f}, S_res={y0[2]:.1f}, R_res={y0[3]:.1f}", flush=True)
    print(f"Vancomycin starts at: {vanco_start_days:.2f} days", flush=True)
    print(f"Linezolid starts at:  {lzd_start_days:.2f} days", flush=True)
    print(f"Linezolid ends at:    {lzd_end_days:.2f} days", flush=True)

    # Peak blood bacteria before treatment
    pre_tx_idx = t_eval <= vanco_start
    peak_S_b   = np.max(S_b[pre_tx_idx])
    print(f"\nPeak S_b before vancomycin: {peak_S_b:.2e} CFU/mL "
          f"(expected: 1e3–1e6 for clinical bacteremia)", flush=True)

    sys.exit(0)
