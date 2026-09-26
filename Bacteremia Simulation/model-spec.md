# MRSA osteomyelitis / bacteremia PK-PD model — specification

Dual-compartment (bone, blood) ODE model of vancomycin + linezolid combination
therapy against MRSA osteomyelitis, with a two-phenotype (active / persister)
bacterial population.

This document is the reference for implementation. Where a parameter has no
literature value, that is stated explicitly rather than filled in with a guess.

---

## 1. Model structure

### Bacterial states

Four state variables. Phenotype is preserved across compartments; a persister
that sheds into blood arrives in blood as a persister.

| Symbol | Meaning |
|---|---|
| `A_bone` | Active (growing, drug-susceptible) bacteria in bone |
| `P_bone` | Persister (dormant, drug-tolerant) bacteria in bone |
| `A_blood` | Active bacteria in blood |
| `P_blood` | Persisters in blood |

`T_bone = A_bone + P_bone` (shared carrying capacity — the two phenotypes
compete for the same resources).

### Drug states

Vancomycin and linezolid each have their own blood and bone PK compartments.
**These are separate from the bacterial compartments.** Drug exchanges
bidirectionally between blood and bone; bacteria do not use the same transfer
structure. Keep the two systems clearly named in code to avoid sign errors.

---

## 2. Equations

```
dA_bone/dt  = g_A · A_bone · (1 − T_bone/Bmax)
              − sw_AP · A_bone + sw_PA · P_bone
              − E_drug(C_bone) · A_bone
              − φ · c_A · A_bone
              − k_shed · A_bone + k_seed · A_blood

dP_bone/dt  = g_P · P_bone · (1 − T_bone/Bmax)
              + sw_AP · A_bone − sw_PA · P_bone
              − φ · c_P · P_bone
              − k_shed · P_bone + k_seed · P_blood

dA_blood/dt = k_shed · A_bone − k_seed · A_blood
              − E_drug(C_blood) · A_blood
              − c_A · A_blood

dP_blood/dt = k_shed · P_bone − k_seed · P_blood
              − c_P · P_blood
```

### Notes on each term

- **Growth** applies to both phenotypes, but `g_P ≪ g_A`. Logistic with shared
  `Bmax`.
- **Switching** occurs only in bone. Switch rates are 1e-6 to 1e-2 /h and blood
  residence time is under ~1 h, so the product is ≪ 1: negligible phenotype
  conversion happens in circulation. This saves two parameters and is
  quantitatively justified, not hand-waved.
- **Drug effect** `E_drug` applies only to active bacteria in the baseline
  model (see §5 for the sensitivity test that should be run on this).
- **Immune clearance** uses the same rate constants `c_A`, `c_P` in both
  compartments, scaled in bone by a single access factor `φ < 1` representing
  restricted phagocyte penetration of the biofilm. One parameter rather than a
  separate clearance rate per compartment.
- **Transfer** is bidirectional and phenotype-preserving. `k_seed` is retained
  because bacteremia is a modeled endpoint (see §4, decision D3).

---

## 3. Parameters

### Bacterial dynamics

| Parameter | Value / range | Units | Source |
|---|---|---|---|
| `g_A` | 1.0 | /h | Mikkaichi 2019, from in vitro growth study |
| `g_P` | 1e-3 to 1e-1 | /h | Mikkaichi 2019 (ensemble range) |
| `sw_AP` | 1e-6 to 1e-2 | /h | Mikkaichi 2019 |
| `sw_PA` | 1e-6 to 1e-1 | /h | Mikkaichi 2019 |
| `Bmax` | see note | CFU/tibia | Not identifiable — see §4 D5 |
| Initial persister fraction | 0.1% of active | — | Mikkaichi 2019 |

Alternative in vivo growth anchor: Vera-Yunca 2025 estimated `k_g,max` = 0.845 /h
in mouse **lung**, 42% below their in vitro value of 1.47 /h. Their model also
modulates growth by a time-dependent sigmoid (`t50,g` = 10.3 h), so 0.845 /h is
a ceiling approached over ~half a day, not the rate at t = 0. **This is a lung
value from neutropenic mice — not a bone value.** Use it only as an anchor for
the in vitro → in vivo correction factor (~40% reduction), not as a bone growth
rate.

Observed net growth in untreated tibia controls (Liu 2024), for validation
rather than fitting:

- Neutropenic: +0.77 ± 0.19 log10 CFU/tibia over 24 h → ~0.074 /h net
- Immunocompetent: +0.84 ± 0.90 log10 CFU/tibia over 24 h → ~0.081 /h net

Both are single 24 h intervals, n = 3, at burdens near carrying capacity. The
immunocompetent SD exceeds its mean. These are *net* rates including immune
clearance, not intrinsic growth rates.

### Immune clearance

| Parameter | Value / range | Units | Source |
|---|---|---|---|
| `c_A` | 1e-0.5 to 1e1 | /h | Mikkaichi 2019 |
| `c_P` | randomize as `f · c_A`, `f` ∈ [1e-4, 1e0] | /h | Mikkaichi 2019 |
| `φ` (bone immune access) | no literature value | — | Ensemble it (§5) |

`c_P` was the single strongest discriminator of resolving vs. persistent
bacteremia in Mikkaichi's ensemble (93.2% classification accuracy from that one
feature alone vs. 97.0% using all seven). `g_P` entered only when relapse was
the outcome of interest. Expect this model to be similarly sensitive to `c_P`.

Independent support: Liu 2024 found tedizolid achieved only 0.28 ± 0.19 log10
kill in neutropenic mice even at 20× the human-equivalent dose, versus a target
fAUC24/MIC of 2.40 for stasis in immunocompetent mice. Immune status dominates.

### Drug effect

Sigmoidal Emax, driven by **free** drug concentration at the effect site.

Vera-Yunca 2025 (MRSA 160079; LZD MIC 2 mg/L, VAN MIC 1 mg/L):

| Parameter | In vitro | In vivo (lung) |
|---|---|---|
| `Emax,LZD` | 1.97 /h (RSE 8.5%) | ≈0.597 /h (ratio 0.303) |
| `Emax,VAN` | 1.74 /h (RSE 3.0%) | ≈0.437 /h (ratio 0.251) |
| `EC50,LZD` | 0.560 mg/L (RSE 3.8%) | not re-estimated — inherits in vitro value |
| `EC50,VAN,0` | 0.245 mg/L (RSE 2.0%) | not re-estimated |
| `γ_LZD` | 2.19 | — |
| `γ_VAN` | 20 (fixed at upper bound; estimate was imprecise) | — |
| `k_e0,LZD` | — | 0.0129 /h (RSE 29.5%) |
| `k_e0,VAN` | — | 0.0318 /h (RSE 6.1%) |

Caveats that travel with these numbers:

1. The in vivo Emax values are **derived** (in vitro Emax × ratio), not
   independently estimated.
2. EC50 was **not** re-estimated in vivo — sparse terminal sampling made it
   unidentifiable. The in vivo model carries the in vitro value forward.
3. EC50 is strain-specific and was scaled with MIC for clinical simulation. If
   our isolate has a different MIC, EC50 needs rescaling.
4. `k_e0,LZD` = 0.0129 /h is an equilibration half-life of ~54 h. Dropping Emax
   and EC50 into a model driven by plasma concentration with no distribution
   delay will substantially overpredict early killing.
5. Emax in this model family is anchored to the growth rate in the same fit.
   Importing these values into a model with a different growth rate breaks the
   relationship that made them coherent. Note that `Emax,LZD` (0.597 /h) sits
   *below* the in vivo `k_g,max` (0.845 /h) they were fitted alongside.

Do not mix Emax for one drug from one source with Emax for the other drug from
a different source — that imports two incompatible structural assumptions into
the same equation.

### Bone penetration

| Method | Ratio | Notes |
|---|---|---|
| Microdialysis, cancellous bone (Traunmüller, diabetic foot) | fAUC ratio ~1.09 ± 0.11 | Free drug, interstitial fluid |
| Pooled review median | ~0.40 | Whole-bone homogenate |
| Spinal TB, trough | 0.48 (range 0.30–0.67) | Diseased bone homogenate |
| Broad review range | 0.3–1.2 | Substantial between-study variability |

**Use the microdialysis value (~1.0, free-to-free) if the bone compartment
represents interstitial fluid where bacteria actually live.** The ~0.4 figures
come from homogenizing whole bone, diluting drug across mineralized matrix no
bacterium occupies. Using 0.4 for a free-drug interstitial compartment
double-counts the protein-binding correction and understates exposure ~2.5-fold.

No tertiary reference gives a bone penetration value. AHFS DI and the Zyvox
label both report only "readily distributes to well-perfused tissues," CSF at
36–58% of plasma peak, and protein binding ~31%. Osteomyelitis is an off-label
indication, so bone penetration was never part of the registration package.
**Cite the primary study and its method explicitly; there is no authoritative
value to point at.**

Tedizolid comparison (Liu 2024): Kp 0.28 (normal tibia), 0.32 (infected tibia),
0.28 (thigh), no significant difference between tissues. This is a total-drug
homogenate Kp in mice — same category as linezolid's 0.4, **not** comparable to
the microdialysis 1.09. Comparing 0.28 against 1.09 would be a method artifact.

Infection status does not appear to move penetration much (Liu found no tissue
difference; the TB study sampled diseased bone and still got ~0.48), so a single
`Kp` rather than an infection-dependent one is defensible.

### Transfer

| Parameter | Status |
|---|---|
| `k_shed` | No literature value |
| `k_seed` | No literature value |

With bacteremia as a modeled endpoint and blood culture data as an observable,
these are no longer a lumped ratio — a positive culture that clears and returns
constrains them differently from one that persists. Estimate against blood
culture timecourse if available; otherwise ensemble.

### PK/PD targets (for validation, not parameters)

Liu 2024, MRSA osteomyelitis mouse model:

| | Vancomycin (neutropenic) | Tedizolid (immunocompetent) |
|---|---|---|
| Static effect | fAUC24/MIC 91.29 | 2.40 |
| 1 log10 kill | 430.03 | 49.20 |

MICs in that study: vancomycin 1.0 µg/mL, tedizolid 0.5 µg/mL.

---

## 4. Design decisions

**D1 — Shed from the active state only?** No: shed from both, proportional to
each pool. But note the *rationale* that originally motivated active-only
shedding still holds biologically — persisters are matrix-embedded and
non-dispersing, and Lewis (2001) describes persisters reforming the biofilm
which *then* sheds planktonic cells. Consider testing `k_shed,P` = 0 as a
variant; it makes relapsing bacteremia fall out of the structure (active pool
collapses under therapy → seeding stops → persisters repopulate after therapy →
seeding resumes).

**D2 — Switching in blood?** Omitted. Justified by timescale separation (§2).

**D3 — Retain `k_seed` (blood → bone)?** Yes. For a bone-burden-only endpoint
this term is a rounding error and should be dropped to keep the coupling
one-way. Because bacteremia is a modeled endpoint here, it stays.

**D4 — Biofilm tolerance as elevated EC50 or reduced Emax?** Rejected. Lewis
(2001) is explicit that by the conventional criterion biofilm cells do **not**
generally show increased MIC; what is measured as biofilm "resistance" is
resistance *to killing*. Killing curves are biphasic with a floor — after an
initial 3–4 log drop, further concentration increases had no effect — not a
right-shifted sigmoid. A two-subpopulation structure is the correct
representation; a shifted EC50 is not.

**D5 — `Bmax`.** Not identifiable from terminal-sampling data. Vera-Yunca could
not identify it even with 135 mice and multiple timepoints; they fixed it to the
median maximum observed burden in the vehicle group (7.80 log10 CFU/total lung).
Apply the same method to our system: fix `Bmax` at the observed control maximum
in tibia (Liu: 7.37 log10 CFU/tibia neutropenic, 6.20 immunocompetent). Do not
import 7.80 — units differ (per total lung vs. per tibia) and so does the
tissue.

**D6 — Phenotype interpretation.** Mikkaichi hypothesize that persister and
normal cells map onto intracellular and extracellular *S. aureus* respectively,
noting intracellular SCVs recovered from chronic osteomyelitis patients years
after initial infection. In bone there are at least three candidate reservoirs —
implant biofilm, intracellular in osteoblasts, extracellular in marrow —
collapsing into two states. **State this as an assumption; do not inherit it
silently.**

---

## 5. Analysis plan

Parameter identifiability is the central problem. There is no published growth
rate for *S. aureus* in bone, no bone carrying capacity, and no
attachment/detachment rates. Fitting is not the right frame.

**Use ensemble modeling** (Mikkaichi's approach):

1. Randomize parameters log-uniformly over the ranges in §3.
2. Filter to sets satisfying a qualitative biological constraint — e.g. a
   minimum inoculum threshold, or reproducing the observed untreated control
   trajectory (+0.77 to +0.84 log10 over 24 h).
3. Simulate the retained sets under the therapy regimens of interest.
4. Classify outcomes (cleared / persistent / relapsing) and use feature
   selection to find which parameters discriminate.

This converts "these parameters are unidentifiable" from a limitation into the
method, and produces a result (which parameter governs outcome) rather than a
fit.

### Specific sensitivity tests to run

- **Growth rate.** Fix `g_A`, sweep 0.1–1.5 /h. If conclusions are exposure-
  driven, this should barely move them — report that as a robustness result.
- **`k_seed` magnitude.** Confirm bone burden is insensitive to it even at
  generous values. Two lines of code; converts an assumption into a finding.
- **Nonzero persister drug susceptibility.** Baseline assumes persisters are
  untouched by either drug. Vancomycin's kill term requires active division;
  linezolid is bacteriostatic against *S. aureus*, so its whole effect is growth
  inhibition, which does nothing to a non-growing population. The persister pool
  therefore sets the floor on what any regimen achieves — which may be a real
  prediction or an artifact of the zero-susceptibility assumption. Give
  persisters 1–5% of the active kill rate and see whether conclusions change.
  Mikkaichi found a hypothetical "persister killer" at 0.1 /h (one-third the
  vancomycin kill rate on normal cells) cured persistent bacteremia — so the
  threshold where persister killing starts to matter is not far from where
  linezolid might plausibly sit.
- **`φ`.** No literature value; sweep across its plausible range and report.

---

## 6. Key references

- Liu X, Enoki Y, Taguchi K, Matsumoto K. *J Pharm Pharmacol* 2025;77:291–298.
  PK/PD model for osteomyelitis; tedizolid vs. vancomycin targets; tedizolid Kp.
- Vera-Yunca D, Matias C, Vingsbo Lundberg C, Friberg LE. *J Antimicrob
  Chemother* 2025;80:1860–1868. In vitro → in vivo PKPD translation for LZD and
  VAN against MRSA; all Emax/EC50/growth parameters.
- Mikkaichi T, Yeaman MR, Hoffmann A. *PLoS Comput Biol* 2019;15(7):e1007087.
  Two-phenotype persister model, ensemble approach, `c_P` as key determinant.
- Lewis K. *Antimicrob Agents Chemother* 2001;45:999–1007. Persisters as the
  basis of biofilm tolerance; biphasic killing.
- Kostakioti M, Hadjifrangiskou M, Hultgren SJ. *Cold Spring Harb Perspect Med*
  2013;3:a010306. Biofilm development and dispersal (qualitative; no rate
  constants).
- Nielsen EI, Viberg A, Löwdin E, et al. *Antimicrob Agents Chemother*
  2007;51:128–136. Semi-mechanistic PKPD model structure (two-state A/D).

---

## 7. Open items

- Source a value or range for `φ`.
- Decide whether the bone compartment represents interstitial fluid (→ Kp ~1.0,
  free-to-free) or bulk bone tissue (→ Kp ~0.4, total drug). This choice
  determines which penetration figure is correct and must be stated explicitly
  in the methods.
- Determine whether blood culture timecourse data are available to constrain
  `k_shed` and `k_seed` separately.
- Confirm MIC values for the specific isolate being modeled; EC50 must be
  rescaled if they differ from Vera-Yunca's strain (LZD 2 mg/L, VAN 1 mg/L).
