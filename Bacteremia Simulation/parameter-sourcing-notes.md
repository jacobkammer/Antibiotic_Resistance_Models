# Parameter sourcing and modeling decisions — working notes

Companion to `model-spec.md`. The spec says *what* the model is; this says *why*,
what was rejected, and what the evidence base does and does not contain.

Written to survive a handoff. Where a number came from a different tissue,
different immune state, or a different model structure than the one it is being
used in, that is recorded here rather than left to be rediscovered.

---

## 1. Drug background

### Linezolid vs. tedizolid

Both oxazolidinones binding 23S rRNA of the 50S subunit, blocking initiation
complex formation; both generally bacteriostatic against *S. aureus*.

| | Linezolid | Tedizolid |
|---|---|---|
| Dose | 600 mg q12h | 200 mg daily |
| Half-life | ~5–7 h | ~12 h |
| Protein binding | ~30% | ~70–90% |
| Oral bioavailability | ~100% | ~90% |
| MIC vs MRSA | baseline | ~4–8× lower |
| cfr resistance | lost | usually retained |
| Approved use | broad | SSTI only (6-day course) |

Structural basis: tedizolid has an extra D-ring (more ribosome contacts → higher
potency) and a C5 hydroxymethyl in place of linezolid's acetamide (cfr
methyltransferase acts at A2503 and doesn't knock it out).

**Modeling implications if tedizolid is ever substituted in:** the much higher
protein binding makes the fu correction dominant; tedizolid's fAUC/MIC targets
are far lower but its activity collapses without neutrophils, so an immune term
matters more for it than for linezolid; bone data are sparse (animal + small
case series), so parameterization carries more uncertainty.

### Vancomycin vs. linezolid in the model

Vancomycin is the bactericidal half of the regimen and linezolid the
bacteriostatic half. In a two-state model this asymmetry matters: linezolid's
whole effect against *S. aureus* is growth inhibition, which does nothing to a
non-growing persister pool. Vancomycin's kill requires active division. **Both
drugs are therefore near-inert against persisters by construction** — see the
sensitivity test in spec §5.

---

## 2. What the literature does not contain

Established over the course of this review; recorded so it is not re-searched.

- **No growth rate for *S. aureus* in bone.** The osteomyelitis literature is
  histology, cytokine profiling, imaging, and endpoint CFU counts. Serial
  time-course data over multiple timepoints — the minimum for fitting a growth
  parameter — does not exist.
- **No bone carrying capacity.** Structurally unidentifiable from terminal
  sampling: each timepoint is a different set of animals, and burdens sit near
  saturation from day one, so even good serial data would constrain N/Nmax
  rather than the growth rate.
- **No biofilm attachment/detachment or dispersal rate constants.** Kostakioti
  2013 (a standard biofilm review) contains no rate constants at all — only
  UPEC doubling times (30–35 min early IBC, >60 min after 6–8 h), IBC counts
  (3–700 per bladder, 10⁴–10⁵ bacteria each), matrix as ~90% of biomass,
  up to 1000-fold antibiotic tolerance, and *B. subtilis* biofilm integrity loss
  at 5–8 days. None of it is kinetic, and none is *S. aureus* in bone.
- **No authoritative bone penetration value.** AHFS DI and the Zyvox label give
  only "readily distributes to well-perfused tissues," CSF 36–58% of plasma
  peak, protein binding ~31%. Osteomyelitis is off-label for linezolid, so bone
  penetration was never in the registration package. Every available number is
  investigator-initiated primary literature.
- **No tertiary-source PK/PD target for bone.** The vancomycin TDM consensus
  guideline (Rybak 2020, ASHP/IDSA/PIDS/SIDP) sets AUC/MIC targets but gives no
  bone penetration coefficient. No guideline does, for either drug.

**Consequence:** fitting is the wrong frame for this model. Ensemble modeling
over plausible ranges, filtered by a qualitative biological constraint, is the
approach — see spec §5.

---

## 3. Source papers and what each is used for

| Paper | Used for | Not used for |
|---|---|---|
| Schmidt 2009 | S→P switching rate; justification of persistence over resistance structure | Drug parameters (superseded — see §4) |
| Vera-Yunca 2025 | Growth rate, Emax, EC50, Hill, ke0 for **both** drugs | Bmax (units are per total lung) |
| Mikkaichi 2019 | Immune clearance terms, switch-rate ranges, persister fraction, ensemble method | Drug effect (binary on/off, no EC50) |
| Liu 2024 | PK/PD validation targets, tibia burdens, tedizolid Kp | Growth rate (net, n=3, single interval) |
| Lewis 2001 | Justification for two-state structure over shifted EC50 | Any rate constant (contains none) |
| Kostakioti 2013 | Qualitative biofilm dispersal mechanism | Any rate constant (contains none) |

---

## 4. The Schmidt → Vera-Yunca decision

Initially Schmidt 2009 was adopted as the structural backbone, on the grounds
that its two-subpopulation model is exactly the structure needed. **That was
reversed.** Reasoning, recorded because the first choice is defensible and
someone may revisit it:

**For Schmidt:** fitted `ksp` = 0.004 /h (bootstrap 0.001, CI 0.000–0.003);
fitted initial susceptible fraction F = 0.83; bootstrap-validated (n=1000);
plain Emax (no Hill) so fewer parameters; explicitly chose persistence over
resistance on two empirical grounds — starting inoculum below the natural
mutation rate (1 in 10⁷–10⁸), and no regrowth at constant ≥2× MIC where
resistant mutants would have produced it.

**Against Schmidt, decisively:**

1. **No vancomycin.** One parameter set was fitted to linezolid *and*
   RWJ-416457, with EC50 the only drug-specific term. `kmax` = 1.65 /h is
   "maximum oxazolidinone kill rate in this dataset," not a linezolid
   parameter. Nothing in the paper speaks to vancomycin.
2. **In vitro only.** Requires an attenuation factor borrowed from elsewhere,
   with double-counting risk against an explicit Kp term.
3. **`kps` fixed to zero.** Persisters never revert, so relapse is structurally
   impossible — the persister pool can only accumulate and die at `kd`. Since
   relapse after therapy withdrawal is a target behavior, this had to be
   modified regardless.
4. **F = 0.83 implies ~17% persisters**, two orders above Mikkaichi's 0.1% and
   four to five above the classical ~10⁻⁶ (Bigger 1944 via Lewis 2001). It is
   an in vitro fitted quantity at 5×10⁵ CFU/mL doing work that the persister
   *fraction* does elsewhere. Not the same quantity; not transferable.

**For Vera-Yunca:** both drugs from one simultaneous fit; already a two-state
model (active A / dormant D with kAD transfer), so the structural argument for
Schmidt was never actually decisive; in vivo, so no attenuation factor to
justify; SIR-based uncertainty on every estimate.

**Net:** Vera-Yunca supplies the entire drug-and-growth package. Schmidt is kept
only for the S→P switching rate and the structural justification. `kps` and the
persister fraction come from Mikkaichi either way.

---

## 5. Parameter values, with provenance and caveats

### Growth

| Source | Value | Context |
|---|---|---|
| Schmidt | 1.19 ± 0.061 /h (bootstrap 1.21, CI 1.11–1.35) | MRSA OC2878, MHB, in vitro |
| Vera-Yunca | 1.47 /h (RSE 3.4%) | MRSA 160079, in vitro |
| Vera-Yunca | **0.845 /h (RSE 5.0%, SIR 0.782–0.862)** | Same strain, mouse lung, in vivo |
| Mikkaichi | 1.0 /h (fixed) | From in vitro literature |
| Wicha (via Vera-Yunca) | 1.56 /h | MSSA, in vitro |
| Liu controls | ~0.074–0.081 /h net | Tibia, 24 h, n=3 — see caveat |

The three in vitro values cluster at 1.0–1.5 /h; this is the one parameter where
independent sources agree. The in vivo value is the outlier, and it is lung.

**Vera-Yunca's growth is not a bare constant.** A time-dependent sigmoid
modulates it: `t50,g` = 10.3 h (RSE 7.8%), Hill 1.53. So 0.845 /h is a ceiling
approached over roughly half a day, not the rate at t=0. Implementing it as a
constant overpredicts early expansion. Schmidt has the analogous term (`dgs`
fixed at 0.24 /h) and notes the delay appeared despite 2 h preincubation.

**Liu control caveat:** +0.77 ± 0.19 (neutropenic) and +0.84 ± 0.90
(immunocompetent) log10 CFU/tibia over 24 h. Single intervals, n=3, burdens near
carrying capacity, and the immunocompetent SD exceeds its mean. These are *net*
rates including immune clearance and saturation. Use to check the untreated
trajectory; do not fit to them.

**Bone scaling:** blood 1.19 /h (Schmidt) scaled to bone by Vera-Yunca's
in vitro→in vivo ratio 0.845/1.47 = 0.575 → ~0.68 /h. Note this uses an
*in vivo vs in vitro* correction as a *bone vs blood* correction, which
implicitly treats blood as in-vitro-like. Defensible (blood is well-mixed and
nutrient-replete) but must be stated, not inherited. Better: parameterize as
`ks_bone = ρ · ks_blood`, ρ ∈ (0,1], point 0.575, ensembled 0.1–1.0.

### Drug effect — Vera-Yunca (recommended set)

| Parameter | In vitro | In vivo |
|---|---|---|
| Emax,LZD | 1.97 /h (RSE 8.5%) | 0.597 /h (ratio 0.303, SIR 0.269–0.330) |
| Emax,VAN | 1.74 /h (RSE 3.0%) | 0.437 /h (ratio 0.251, SIR 0.219–0.279) |
| EC50,LZD | 0.560 mg/L (RSE 3.8%) | **not re-estimated** |
| EC50,VAN,0 | 0.245 mg/L (RSE 2.0%) | **not re-estimated** |
| γ_LZD | 2.19 (RSE 15.0%) | unchanged |
| γ_VAN | 20 (fixed at upper bound) | unchanged |
| ke0,LZD | — | 0.0129 /h (RSE 29.5%) |
| ke0,VAN | — | 0.0318 /h (RSE 6.1%) |

### Drug effect — Schmidt (superseded, retained for reference)

ks 1.19 /h · kmax 1.65 /h (shared across both oxazolidinones) · EC50 LZD 1.39 ±
0.207 (bootstrap 1.13, CI 0.80–1.53) · EC50 RWJ-416457 0.41 · Nmax fixed
3.39e9 CFU/mL · dgs fixed 0.24 /h · dks 0.50 /h · ksp 0.004 /h · kps fixed 0 ·
kd fixed 0.015 /h · F 0.83. MIC (OC2878) linezolid 1.0 µg/mL. Both compounds
bacteriostatic: <3 log reduction at any concentration, ~2–2.5 log at >8× MIC.

**If mixing summaries, be consistent.** EC50 1.13 is the bootstrap median, not
the point estimate 1.39. `ksp` differs fourfold between the two summaries (0.004
vs 0.001). Pick one convention across all parameters from a given fit.

### Immune clearance — Mikkaichi ranges

`c_N` 10⁻⁰·⁵ to 10¹ /h · `c_P` = f·c_N with f log-uniform 10⁻⁴ to 10⁰ ·
`a` (saturation) 10⁻¹·⁵ to 10⁰·⁵ · `h` (Hill) 0.2–2.0.

`c_P` was the **sole** discriminator of resolving vs. persistent bacteremia:
93.2% classification accuracy from that one feature vs. 97.0% from all seven.
`g_P` entered only for relapse. Switch rates had essentially no effect in vivo —
the opposite of the in vitro result in the same paper.

Independent support from Liu: tedizolid achieved only 0.28 ± 0.19 log10 kill in
neutropenic mice at 20× the human-equivalent dose, against a target fAUC24/MIC
of 2.40 for stasis in immunocompetent mice.

### Switching — Mikkaichi ranges

`sw_N2P` 10⁻⁶ to 10⁻² /h · `sw_P2N` 10⁻⁶ to 10⁻¹ /h · `g_P` 10⁻³ to 10⁻¹ /h ·
initial persister fraction 0.1%.

Schmidt's fitted `ksp` = 0.004 /h sits inside the sw_N2P range — reassuring.
`kps` is the disagreement: Schmidt fixes it at 0, Mikkaichi ranges over five
logs. Vera-Yunca could not estimate the D→A transfer at all (no washout period).

### Carrying capacity

Not identifiable in vivo. Vera-Yunca could not identify Bmax even with 135 mice
and multiple sacrifice timepoints, and fixed it to the median maximum observed
burden in the vehicle group: 7.80 log10 CFU/total lung. **Apply the method, not
the number** — units differ (per total lung vs per tibia) and so does the tissue.
Liu tibia observed maxima: 7.37 log10 CFU/tibia (neutropenic), 6.20
(immunocompetent). Schmidt's 3.39e9 CFU/mL is per mL of broth; also not
transferable.

### Bone penetration (linezolid)

| Method | Ratio | Notes |
|---|---|---|
| Microdialysis, cancellous bone | fAUC ~1.09 ± 0.11 | Free drug, interstitial fluid |
| Pooled review median | ~0.40 | Whole-bone homogenate |
| Spinal TB, trough (24 h post-dose) | 0.48 (range 0.30–0.67) | Diseased bone homogenate |
| Broad review range | 0.3–1.2 | Between-study variability |

The ~0.4 figures homogenize whole bone, diluting drug across mineralized matrix
no bacterium occupies. **If the bone compartment is interstitial fluid, use
~1.0 free-to-free.** Using 0.4 there double-counts the protein-binding
correction and understates exposure ~2.5-fold.

Tedizolid Kp (Liu): 0.28 normal tibia, 0.32 infected tibia, 0.28 thigh, no
significant tissue difference. Total-drug homogenate in mice — same category as
linezolid's 0.4, **not** comparable to the microdialysis 1.09. Comparing 0.28
against 1.09 would be a method artifact, not a real penetration difference.

Infection status barely moves these ratios, so a single Kp rather than an
infection-dependent one is defensible.

---

## 6. Structural decisions and what was rejected

### Rejected: biofilm tolerance as elevated EC50 or reduced Emax

Lewis 2001 is explicit that by the conventional criterion biofilm cells do not
generally show increased MIC; what gets reported as biofilm "resistance" is
resistance *to killing*, because most susceptibility studies measure survival in
a preformed biofilm rather than growth. The killing curve is biphasic with a
floor — the majority of cells die at ordinary concentrations (≤5 µg/mL), then
after a 3–4 log drop further concentration increases have no effect. That is
not a right-shifted sigmoid. A two-subpopulation structure is the correct
representation.

Schmidt reached the same structure independently from time-kill data.

### Rejected: separate biofilm compartment with attach/detach kinetics

Three unidentifiable parameters (kattach, kdetach, separate susceptibility) for a
phenomenon no available data constrains. The two-state persister structure
captures the same behavior with one.

### Rejected: resistant subpopulation with fitness cost

No supporting evidence in this literature base. Schmidt explicitly chose
persistence over resistance (see §4). Vera-Yunca modeled resistance but as
*adaptive* resistance for vancomycin only — a reversible, concentration-driven
EC50 shift via kON = 0.0635 and SlpAR = 3.43, not a heritable mutant paying a
growth penalty. A fitness-cost term implies a mechanism the evidence base does
not contain.

**Retained** as separate and legitimate: the persister growth penalty
(g_P < g_S), which Mikkaichi identified as discriminating relapsing from
non-relapsing persistent bacteremia; and the bone growth attenuation (0.68),
which is environmental and applies to all bacteria in the compartment, not a
penalty on one phenotype. **Do not name the latter a "fitness cost" in code** —
it reads as the rejected mechanism.

### Attenuation: Emax only, never EC50

Vera-Yunca tested each drug-effect parameter in turn against the mouse data.
Only the Emax ratios improved the fit significantly (OFV −300.1). EC50, Hill,
and the rest stayed at in vitro values, so Table 2 has no EC50 row — the in vivo
model silently inherits Table 1's values.

**Read this as a data limitation, not a finding.** The in vivo studies were
terminal, one timepoint per mouse; the paper states the in vitro parameters
could not be identified from in vivo data alone, which is why parameters were
tested one at a time. "EC50 unchanged in vivo" means the mouse data could not
distinguish a different EC50 — not that potency is preserved across the
in vitro–in vivo boundary.

**They added an effect compartment rather than shifting EC50.** ke0,LZD 0.0129
/h, ke0,VAN 0.0318 /h, OFV −147.9. Structurally cleaner: drug arrives late and
lower, rather than becoming intrinsically less potent. ke0,LZD = 0.0129 /h is an
equilibration half-life of ~54 h, so at 26 h the effect-site concentration is
still well below plasma.

**If mirroring their approach, the effect compartment is part of the package.**
Attenuated Emax + unattenuated EC50 + no distribution delay will overpredict
early killing badly.

**Double-counting warning:** the 0.303 ratio absorbed everything making mouse
lung kill less than broth. The paper attributes it not to ELF exposure (which is
adequate for both drugs) but to "other unexplored factors such as drug
concentrations not reaching the infected lung tissue." If the model has an
explicit Kp and effect compartment, part of what 0.303 absorbed is already
represented. Treat the attenuation as an ensembled parameter (~0.2–0.6), not a
fixed constant, and sample the upper end more heavily where Kp is doing real
work.

### Blood ↔ bone bacterial transfer

Bidirectional and phenotype-preserving. Switching is omitted in blood: switch
rates ≤10⁻² /h against blood residence under ~1 h means the product is ≪1, so
negligible conversion occurs in circulation. Saves two parameters on a
quantitative argument.

`k_seed` (blood→bone) would be droppable for a bone-burden-only endpoint — the
flux is proportional to the much smaller pool and cannot move bone burden. It is
retained **only** because bacteremia is a modeled endpoint. Mikkaichi dropped the
blood compartment entirely for this reason, using a detection threshold on
infection-site burden as a bacteremia proxy, and showed conclusions held across
a >300-fold range of that threshold.

**Shedding from which state:** biology favors susceptible-only. Persisters are
matrix-embedded and non-dispersing; Lewis describes persisters reforming the
biofilm which *then* sheds planktonic cells. Test `k_shed,P` = 0 as a variant —
it makes relapsing bacteremia fall out of the structure rather than needing to
be built in.

### Combination assumed additive

Kill terms summed across drugs. Mechanistically distinct agents (cell wall vs.
ribosomal), so additivity is the appropriate null — but any predicted
combination benefit is then a consequence of the assumption, not evidence for
synergy. State explicitly.

---

## 7. Cross-paper comparison is mostly not possible

Recorded because it will be tempting to build a comparison table.

**Emax across papers is not comparable.** Vera-Yunca's Emax is a rate constant
(h⁻¹) in an ODE. Liu's Imax is the *endpoint span* of a static dose–response
curve at 24 h, in log10 CFU/tibia — and it changes with the index plotted
against (vancomycin: 7.51, 3.34, 4.13 for fAUC/MIC, fCmax/MIC, fT>MIC — same
experiment, same drug). Different quantities; putting them in one table is a
units error.

**Even among rate-based Emax values, comparison is weak.** Emax trades off
against γ and EC50 during estimation, and depends on whether the model has a
dormant state, an effect compartment, an adaptive-resistance term. Add a dormant
compartment and Emax must rise to produce the same observed kill.

Note what Vera-Yunca actually compared in their discussion: growth rate against
Wicha (1.56) and Schmidt (1.19–1.21), and EC50 against both (0.68 and 1.39
mg/L). They did **not** compare Emax to anyone, despite citing models that
report it — and they add that their data could not be characterized by Wicha's
model at all.

**Transfers across papers, roughly in order of reliability:** MIC → EC50
relative to MIC (a ratio, less structure-dependent) → growth rate → in
vitro/in vivo attenuation factor. Emax is near the bottom.

**EC50 spread:** Schmidt 1.39 mg/L vs. Vera-Yunca 0.560 mg/L, both linezolid vs.
MRSA, strains with MIC 1–2 mg/L. A 2.5× gap between two competent in vitro
studies of the same drug and organism. This is the scale of uncertainty before
the tissue question is even reached.

**Similar Emax across drugs is normal, not suspicious.** Vera-Yunca's 1.97 (LZD)
vs 1.74 (VAN) in vitro both sit just above the in vitro growth rate of 1.47 /h.
At saturating concentration the net rate is ≈ Emax − kg − kd, and the experiment
measures the steepest observed decline; for any drug clearing to the detection
limit within 24 h, fitted Emax lands near (growth rate) + (observed max decline).
Emax here is as much a property of the assay as of the drug. The pharmacology is
in EC50 (VAN 0.245 vs LZD 0.560) and γ (VAN fixed 20, a near-step response, vs
LZD 2.19).

---

## 8. Validation targets

Liu 2024, MRSA osteomyelitis mouse model (MICs: vancomycin 1.0, tedizolid
0.5 µg/mL):

| | Vancomycin (neutropenic) | Tedizolid (immunocompetent) |
|---|---|---|
| Static effect | fAUC24/MIC 91.29 | 2.40 |
| 1 log10 kill | 430.03 | 49.20 |

Their vancomycin AUC24/MIC_BMD for bactericidal effect: 477.8–716.7, against
Gawronski's clinical AUC24/MIC_Etest > 293 (converted to BMD 533) — close enough
that Liu argue the mouse model extrapolates to humans.

Note the nephrotoxicity ceiling: vancomycin AUC above 600 µg·h/mL significantly
increases risk, so the therapeutic window for bone is narrow and partly
overlapped by the toxic one. Relevant if the model is used to propose regimens.

Vera-Yunca in vivo maxima for reference: 1.9-log (LZD 40 mg/kg or 30 mg/kg q8h)
and 2-log (VAN 10–30 mg/kg q4h) reductions; stasis at LZD 8 mg/kg q8h or VAN
5 mg/kg q4h.

---

## 9. Open items

1. **Effect compartment.** Confirm the model has one (ke0 terms, or an explicit
   bone PK compartment doing that work). Attenuated Emax without a distribution
   delay overpredicts early killing.
2. **Attenuation double-counting.** If Kp and an effect compartment are both
   explicit, the 0.303/0.251 ratios are partly redundant. Ensemble the factor
   rather than fixing it.
3. **`φ` (bone immune access).** No literature value. Ensemble.
4. **Bone compartment definition.** Interstitial fluid (→ Kp ~1.0, free-to-free)
   or bulk tissue (→ Kp ~0.4, total drug)? Determines which penetration figure
   is correct. State in methods.
5. **Blood culture timecourse data** — do they exist? They would constrain
   `k_shed` and `k_seed` separately rather than as a ratio.
6. **Isolate MIC.** Rescale EC50 if it differs from the source strain
   (Vera-Yunca: LZD 2 mg/L, VAN 1 mg/L).
7. **Blood growth rate.** 1.19 /h treats circulating bacteria as growing at
   broth rates. Blood is nutrient-restricted and has complement, opsonins, and
   platelet microbicidal proteins beyond the phagocytic clearance in `c_S`. If
   the immune term captures only phagocytosis, 1.19 /h is optimistic toward
   overpredicting bacteremia. Worth a sensitivity run given bacteremia is the
   endpoint.
