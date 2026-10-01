# MRSA osteomyelitis / bacteremia PK-PD model — specification

Dual-compartment (bone, blood) ODE model of vancomycin + linezolid combination
therapy against MRSA osteomyelitis, with a two-phenotype (active / persister)
bacterial population.

**Backbone: Schmidt et al. 2009** — the susceptibility-based two-subpopulation
model, fitted to static and dynamic oxazolidinone time-kill curves against MRSA
and validated by nonparametric bootstrap. That structure already is the
two-state design this model needs, so it is adopted wholesale and modified in
three specific places (§4 M1–M3) rather than assembled from parts of several
papers.

Where a parameter has no literature value, that is stated explicitly rather than
filled in with a guess.

---

## 1. Model structure

### Bacterial states

Four state variables. Phenotype is preserved across compartments; a persister
that sheds into blood arrives in blood as a persister.

| Symbol | Meaning |
|---|---|
| `S_bone` | Susceptible (growing, drug-susceptible) bacteria in bone |
| `P_bone` | Persister (dormant, drug-insusceptible) bacteria in bone |
| `S_blood` | Susceptible bacteria in blood |
| `P_blood` | Persisters in blood |

Note Schmidt's convention: growth and carrying capacity apply to the
**susceptible pool only**. Persisters neither replicate nor contribute to
saturation. `N_tot = S + P` is the observable.

### Drug states

Vancomycin and linezolid each have their own blood and bone PK compartments.
**These are separate from the bacterial compartments.** Drug exchanges
bidirectionally between blood and bone; bacteria use a different transfer
structure. Keep the two systems clearly named in code to avoid sign errors.

---

## 2. Equations

Following Schmidt eq. 2–3, extended to two compartments and two drugs.

```
dS_bone/dt  = [ ks · (1 − S_bone/Nmax) · (1 − e^(−dgs·t))
                − Σ_drugs ( kmax_i · C_i / (EC50_i + C_i) ) · (1 − e^(−dks·t))
                − ksp − kd ] · S_bone
              + kps · P_bone
              − φ · c_S · S_bone
              − k_shed · S_bone + k_seed · S_blood

dP_bone/dt  = ksp · S_bone − kps · P_bone − kd · P_bone
              − φ · c_P · P_bone
              − k_shed · P_bone + k_seed · P_blood

dS_blood/dt = k_shed · S_bone − k_seed · S_blood
              − Σ_drugs ( kmax_i · C_i / (EC50_i + C_i) ) · S_blood
              − kd · S_blood − c_S · S_blood

dP_blood/dt = k_shed · P_bone − k_seed · P_blood
              − kd · P_blood − c_P · P_blood
```

`C_i` is the **free** drug concentration of drug *i* at the effect site.

### Notes on each term

- **Plain Emax, not sigmoidal.** Schmidt tested a Hill factor and it did not
  significantly improve the fit; the final model uses a maximum-effect form.
  **γ = 1. Do not carry a Hill coefficient over from another paper while using
  Schmidt's EC50.**
- **Growth applies to the susceptible pool only**, with `(1 − S/Nmax)`
  saturation. Persisters are dormant by construction.
- **Onset delays.** `dgs` (growth) and `dks` (kill) are exponential-approach
  terms. Dropping `dks` while keeping `kmax` overstates killing in the first
  few hours. Retain both.
- **Natural death `kd`** applies to both pools. This is the only loss term
  acting on persisters in the original in vitro model.
- **Switching** occurs only in bone. Switch rates are ≤1e-2 /h and blood
  residence is under ~1 h, so the product is ≪1: negligible phenotype
  conversion happens in circulation. Saves two parameters; quantitatively
  justified, not hand-waved.
- **Drug effect on persisters is zero** in the baseline. See §5 for the
  sensitivity test that must be run on this assumption.
- **Immune clearance** uses rate constants `c_S`, `c_P` in both compartments,
  scaled in bone by a single access factor `φ < 1` for restricted phagocyte
  penetration of the biofilm. One parameter rather than a separate rate per
  compartment. Absent from Schmidt (in vitro); added from Mikkaichi.
- **Transfer** is bidirectional and phenotype-preserving. `k_seed` retained
  because bacteremia is a modeled endpoint (§4 D3).

### Drug combination

Kill terms are summed across drugs. This assumes additivity. Vancomycin (cell
wall) and linezolid (ribosomal) are mechanistically distinct, and additivity is
the appropriate null — but any predicted combination benefit is then a
consequence of the assumption, not evidence for synergy. State this explicitly
in the methods.

---

## 3. Parameters

### 3.1 Backbone — Schmidt et al. 2009

MRSA OC2878. Linezolid MIC 1.0 µg/mL. Inoculum ~5×10⁵ CFU/mL. NONMEM VI,
simultaneous fit to static + dynamic time-kill curves; bootstrap n = 1000.

| Parameter | Estimate ± SE | Bootstrap (95% CI) | Notes |
|---|---|---|---|
| `ks` | 1.19 ± 0.061 /h | 1.21 (1.11–1.35) | Growth rate |
| `kmax` | 1.65 ± 0.065 /h | 1.65 (1.51–1.82) | **Shared across both oxazolidinones** |
| `EC50` linezolid | 1.39 ± 0.207 µg/mL | 1.13 (0.80–1.53) | Only drug-specific parameter |
| `EC50` RWJ-416457 | 0.41 ± 0.057 µg/mL | 0.32 (0.22–0.45) | For reference |
| `Nmax` | fixed 3.39e9 CFU/mL | — | From growth control |
| `dgs` | fixed 0.24 /h | — | Growth onset delay |
| `dks` | 0.50 ± 0.049 /h | 0.54 (0.47–0.63) | Kill onset delay |
| `ksp` (S→P) | 0.004 ± 0.002 /h | 0.001 (0.000–0.003) | |
| `kps` (P→S) | **fixed 0** | — | See M1 |
| `kd` | fixed 0.015 /h | — | Natural death |
| `F` | 0.83 ± 0.023 | 0.83 (0.79–0.88) | Initial susceptible fraction |

**`kmax` is not a linezolid-specific parameter.** One parameter set was fitted
to both compounds, with EC50 the only drug-specific term. So `kmax` = 1.65 /h
means "maximum oxazolidinone kill rate in this dataset." It provides no
vancomycin value, and pairing it with a vancomycin Emax from a different paper
imports two incompatible model structures into one equation. See §3.2.

**`F` = 0.83 implies ~17% persisters at t = 0.** That is two orders of
magnitude above Mikkaichi's 0.1% and four to five above the classical ~1e-6
persister frequency (Bigger 1944, via Lewis 2001). It is an in vitro fitted
quantity at a 5×10⁵ CFU/mL inoculum and is doing work in Schmidt's model that
the persister *fraction* does elsewhere. **Decide which quantity is meant
before carrying either into an in vivo model; they are not the same thing.**

Both compounds were bacteriostatic here: counts fell by less than 3 log at any
concentration, with ~2 to 2.5 log reduction at >8× MIC after 24 h.

### 3.2 Vancomycin kill parameters — unresolved

Schmidt provides no vancomycin parameters. Options, in preference order:

1. Fit or source vancomycin `kmax`/`EC50` from a time-kill study using the
   **same** two-subpopulation structure, so the pair is internally consistent.
2. Use Vera-Yunca's vancomycin values and accept the structural mismatch,
   documenting it. Their in vitro `Emax,VAN` = 1.74 /h, `EC50,VAN,0` =
   0.245 mg/L, `γ_VAN` fixed at 20 (imprecise estimate, hit upper bound) — note
   the fixed γ = 20 is incompatible with Schmidt's γ = 1 form.
3. Treat vancomycin kill as a free parameter and ensemble it.

**Do not silently mix.** This is currently the weakest point in the parameter
set.

### 3.3 In vitro → in vivo attenuation

Schmidt's values are broth. Vera-Yunca 2025 estimated in vivo (mouse lung)
`Emax` at 69.7% lower for linezolid and 74.9% lower for vancomycin than the
corresponding in vitro values (ratio parameters 0.303 and 0.251; SIR CIs
0.269–0.330 and 0.219–0.279). They also found in vivo max growth 42% below in
vitro (0.845 vs 1.47 /h).

**Apply an attenuation factor to Schmidt's `kmax` rather than using it raw.**
Using 1.65 /h unattenuated in a bone compartment will overpredict kill roughly
threefold. The ~0.3 ratio is the transferable quantity here; the absolute lung
values are not bone values.

### 3.4 Switching and persister parameters — in vivo ranges

From Mikkaichi et al. 2019 (ensemble ranges, not point estimates):

| Parameter | Range | Units |
|---|---|---|
| `ksp` (S→P) | 1e-6 to 1e-2 | /h |
| `kps` (P→S) | 1e-6 to 1e-1 | /h |
| `g_P` (persister growth) | 1e-3 to 1e-1 | /h |
| Initial persister fraction | 0.1% of susceptible | — |

Schmidt's fitted `ksp` = 0.004 /h sits inside the Mikkaichi range, which is
reassuring. `kps` is the disagreement (§4 M1).

### 3.5 Immune clearance

| Parameter | Range | Source |
|---|---|---|
| `c_S` | 1e-0.5 to 1e1 /h | Mikkaichi 2019 |
| `c_P` | `f · c_S`, `f` ∈ [1e-4, 1e0] | Mikkaichi 2019 |
| `φ` (bone immune access) | no literature value | Ensemble it (§5) |

`c_P` was the single strongest discriminator of resolving vs. persistent
bacteremia in Mikkaichi's ensemble — 93.2% classification accuracy from that
one feature alone vs. 97.0% using all seven. `g_P` entered only when relapse was
the outcome. Expect this model to be similarly sensitive to `c_P`.

Independent support: Liu 2024 found tedizolid achieved only 0.28 ± 0.19 log10
kill in neutropenic mice at 20× the human-equivalent dose, versus a target
fAUC24/MIC of 2.40 for stasis in immunocompetent mice. Immune status dominates
oxazolidinone efficacy in bone.

### 3.6 Carrying capacity

Schmidt fixed `Nmax` to 3.39e9 CFU/mL from growth-control data. **Units are per
mL of broth — not transferable to per-tibia burden.** Apply the *method*, not
the number: fix `Nmax` at the observed control maximum in the target system.

- Liu 2024 tibia: 7.37 log10 CFU/tibia (neutropenic), 6.20 (immunocompetent)
- Vera-Yunca did the same in lung (fixed to 7.80 log10 CFU/total lung) because
  `Bmax` could not be identified in vivo even with 135 mice

`Nmax` is not identifiable from terminal-sampling data. Fixing it is standard
practice here, not a shortcut.

### 3.7 Observed in vivo growth (validation targets, not fitting targets)

Liu 2024 untreated tibia controls, 24 h:

- Neutropenic: +0.77 ± 0.19 log10 CFU/tibia → ~0.074 /h net
- Immunocompetent: +0.84 ± 0.90 log10 CFU/tibia → ~0.081 /h net

Single 24 h intervals, n = 3, at burdens near carrying capacity; the
immunocompetent SD exceeds its mean. These are *net* rates including immune
clearance and saturation, not intrinsic growth rates. Use to check the untreated
trajectory, not to fit `ks`.

### 3.8 Bone penetration

| Method | Ratio | Notes |
|---|---|---|
| Microdialysis, cancellous bone (Traunmüller, diabetic foot) | fAUC ~1.09 ± 0.11 | Free drug, interstitial fluid |
| Pooled review median | ~0.40 | Whole-bone homogenate |
| Spinal TB, trough | 0.48 (range 0.30–0.67) | Diseased bone homogenate |
| Broad review range | 0.3–1.2 | Substantial between-study variability |

**Use the microdialysis value (~1.0, free-to-free) if the bone compartment
represents interstitial fluid where bacteria live.** The ~0.4 figures come from
homogenizing whole bone, diluting drug across mineralized matrix no bacterium
occupies. Using 0.4 for a free-drug interstitial compartment double-counts the
protein-binding correction and understates exposure ~2.5-fold.

No tertiary reference gives a bone penetration value. AHFS DI and the Zyvox
label report only "readily distributes to well-perfused tissues," CSF at 36–58%
of plasma peak, and protein binding ~31%. Osteomyelitis is off-label, so bone
penetration was never in the registration package. **Cite the primary study and
its method explicitly.**

Tedizolid comparison (Liu 2024): Kp 0.28 (normal tibia), 0.32 (infected), 0.28
(thigh), no significant tissue difference. Total-drug homogenate Kp in mice —
same category as linezolid's 0.4, **not** comparable to the microdialysis 1.09.

Infection status does not move penetration much, so a single `Kp` rather than an
infection-dependent one is defensible.

### 3.9 Transfer

| Parameter | Status |
|---|---|
| `k_shed` | No literature value |
| `k_seed` | No literature value |

With bacteremia as an endpoint and blood culture data as an observable, these
are no longer a lumped ratio — a positive culture that clears and returns
constrains them differently from one that persists. Estimate against blood
culture timecourse if available; otherwise ensemble.

### 3.10 PK/PD targets (validation)

Liu 2024, MRSA osteomyelitis mouse model:

| | Vancomycin (neutropenic) | Tedizolid (immunocompetent) |
|---|---|---|
| Static effect | fAUC24/MIC 91.29 | 2.40 |
| 1 log10 kill | 430.03 | 49.20 |

MICs in that study: vancomycin 1.0 µg/mL, tedizolid 0.5 µg/mL.

---

## 4. Modifications to the Schmidt backbone, and design decisions

### Modifications

**M1 — Unfix `kps`.** Schmidt fixed persister→susceptible reversion to zero.
Defensible for a 24 h in vitro experiment with no washout, but it makes relapse
**structurally impossible**: the persister pool can only accumulate and die at
`kd`. Since relapse after therapy withdrawal is a behavior this model must
reproduce, `kps` must be > 0. Use the Mikkaichi range (1e-6 to 1e-1 /h) and
ensemble it.

**M2 — Attenuate `kmax` for in vivo.** See §3.3. Apply ~0.3 ratio, or ensemble
the attenuation factor across a range.

**M3 — Add immune clearance.** Schmidt is in vitro and has no immune term. Add
`c_S`, `c_P`, `φ` per §3.5. Given Mikkaichi's finding, this is likely the
dominant determinant of outcome, so it is not an optional refinement.

### Decisions

**D1 — Shed from which state?** Both, proportional to each pool. But the
biology favors susceptible-only shedding — persisters are matrix-embedded and
non-dispersing, and Lewis (2001) describes persisters reforming the biofilm
which *then* sheds planktonic cells. Test `k_shed,P` = 0 as a variant; it makes
relapsing bacteremia fall out of the structure (susceptible pool collapses under
therapy → seeding stops → persisters repopulate after therapy → seeding
resumes).

**D2 — Switching in blood?** Omitted. Timescale separation (§2).

**D3 — Retain `k_seed` (blood → bone)?** Yes. For a bone-burden-only endpoint
this term is a rounding error and should be dropped to keep the coupling
one-way. Because bacteremia is a modeled endpoint here, it stays.

**D4 — Biofilm tolerance as elevated EC50 or reduced Emax?** Rejected, and
Schmidt's data are the reason the two-state structure is right instead. Lewis
(2001): by the conventional criterion biofilm cells do **not** generally show
increased MIC; what is measured as biofilm "resistance" is resistance *to
killing*, and killing curves are biphasic with a floor — after an initial 3–4
log drop, further concentration increases had no effect. Schmidt reached the
same structure independently from time-kill data, choosing a persistence rather
than a resistance model because the starting inoculum was below the natural
mutation rate (1 in 1e7 to 1e8) and no regrowth occurred at constant ≥2× MIC.

**D5 — Phenotype interpretation.** Mikkaichi hypothesize persister/normal maps
onto intracellular/extracellular *S. aureus*, noting intracellular SCVs
recovered from chronic osteomyelitis patients years after initial infection. In
bone there are at least three candidate reservoirs — implant biofilm,
intracellular in osteoblasts, extracellular in marrow — collapsing into two
states. **State as an assumption; do not inherit silently.**

**D6 — EC50 source.** Schmidt 1.39 µg/mL vs. Vera-Yunca 0.560 mg/L, both
linezolid against MRSA, strains with MIC 1–2 mg/L. That 2.5× gap is the ordinary
cross-study spread; Vera-Yunca noted it themselves. Neither is wrong — different
strain, medium, and model structure. **This spec uses Schmidt**, for structural
consistency with the rest of the model. Rescale if the modeled isolate's MIC
differs from 1.0 µg/mL.

---

## 5. Analysis plan

Parameter identifiability is the central problem. There is no published growth
rate for *S. aureus* in bone, no bone carrying capacity, and no
attachment/detachment rates. Fitting is not the right frame.

**Use ensemble modeling** (Mikkaichi's approach):

1. Randomize parameters log-uniformly over the ranges in §3.
2. Filter to sets satisfying a qualitative biological constraint — a minimum
   inoculum threshold, or reproducing the observed untreated control trajectory
   (+0.77 to +0.84 log10 over 24 h).
3. Simulate the retained sets under the therapy regimens of interest.
4. Classify outcomes (cleared / persistent / relapsing) and use feature
   selection to find which parameters discriminate.

This converts "these parameters are unidentifiable" from a limitation into the
method, and produces a result (which parameter governs outcome) rather than
a fit.

### Specific sensitivity tests

- **Nonzero persister drug susceptibility.** Baseline assumes persisters are
  untouched. Vancomycin's kill requires active division; linezolid is
  bacteriostatic against *S. aureus*, so its whole effect is growth inhibition,
  which does nothing to a non-growing population. The persister pool therefore
  sets the floor on what any regimen achieves — a real prediction, or an
  artifact of the zero-susceptibility assumption. Give persisters 1–5% of the
  susceptible kill rate and see whether conclusions change. Mikkaichi found a
  hypothetical "persister killer" at 0.1 /h (one-third the vancomycin kill rate
  on normal cells) cured persistent bacteremia — so the threshold where
  persister killing starts to matter is not far from where linezolid might
  plausibly sit.
- **`kps`.** Sweep the full Mikkaichi range. This governs relapse timing and is
  fixed to zero in the source model, so it is pure extrapolation.
- **`kmax` attenuation factor.** Sweep 0.2–1.0 (in vivo ratio to in vitro).
- **`F` / initial persister fraction.** Sweep 1e-6 to 0.17, spanning the
  classical persister frequency to Schmidt's fitted value.
- **`φ`.** No literature value; sweep across its plausible range and report.
- **`k_seed` magnitude.** Confirm bone burden behavior; two lines of code,
  converts an assumption into a finding.

---

## 6. Key references

- Schmidt S, Sabarinath SN, Barbour A, et al. *Antimicrob Agents Chemother*
  2009;53:5039–5045. **Structural backbone**: two-subpopulation model, linezolid
  EC50/kmax, switching rates, delays, bootstrap validation.
- Mikkaichi T, Yeaman MR, Hoffmann A. *PLoS Comput Biol* 2019;15(7):e1007087.
  In vivo two-phenotype model, immune clearance terms, ensemble method, `c_P` as
  key determinant.
- Vera-Yunca D, Matias C, Vingsbo Lundberg C, Friberg LE. *J Antimicrob
  Chemother* 2025;80:1860–1868. In vitro → in vivo attenuation factors;
  vancomycin parameters (structurally incompatible — see §3.2).
- Liu X, Enoki Y, Taguchi K, Matsumoto K. *J Pharm Pharmacol* 2025;77:291–298.
  Osteomyelitis PK/PD targets, tibia burdens, tedizolid Kp.
- Lewis K. *Antimicrob Agents Chemother* 2001;45:999–1007. Persisters as the
  basis of biofilm tolerance; biphasic killing.
- Nielsen EI, Viberg A, Löwdin E, et al. *Antimicrob Agents Chemother*
  2007;51:128–136. Semi-mechanistic PKPD model family that Schmidt builds on.

---

## 7. Open items

- **Vancomycin kill parameters** in a structure compatible with Schmidt's
  (§3.2). Highest-priority gap.
- Reconcile `F` (0.83) against the intended initial persister fraction (§3.1).
- Source a value or range for `φ`.
- Decide whether the bone compartment represents interstitial fluid (→ Kp ~1.0,
  free-to-free) or bulk bone tissue (→ Kp ~0.4, total drug). Determines which
  penetration figure is correct; state explicitly in methods.
- Confirm whether blood culture timecourse data exist to constrain `k_shed` and
  `k_seed` separately.
- Confirm the modeled isolate's linezolid MIC; rescale EC50 if it differs from
  Schmidt's 1.0 µg/mL.
