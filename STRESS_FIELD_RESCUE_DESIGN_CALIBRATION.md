# Biological Calibration of CEMA Model Parameters

Companion document to `FIELD_RESCUE_DESIGN.md`. Maps the computational parameters in `runGroupRescue.py` and `stressBistableSwitch.py` to biological units and assesses plausibility for each layer of the CEMA model.

---

## 1. Timestep Calibration

### 1.1 The Multi-Clock Architecture

The CEMA simulation loop (`runGroupRescue.py`, line 643) advances three subsystems per **bio step**, each with a different Euler integration step size:

```python
dt_ca    = 0.01   # Ca2+ integration step (line 620)
dt_stress = 0.1    # Stress S integration step (line 621)
# Field F: 10 substeps of dt_sub = 0.1 each -> total 1.0 per bio step (line 572)
```

This means per bio step, each subsystem advances by a **different amount of ODE time**:

| Subsystem | dt per bio step | ODE time advanced |
|---|---|---|
| Vmem (bioelectric) | 0.01 | 0.01 |
| Ca2+ | 0.01 | 0.01 |
| Stress S | 0.1 | 0.1 |
| Field F | 10 x 0.1 | 1.0 |

The bio step is an **abstract synchronization barrier**, not a literal time unit. Each subsystem has its own effective "tick rate" relative to the bio step clock. This is a common pattern in multiscale simulations but means that parameters like `tau_S` and `tau_ca` are not directly comparable -- they live on different clocks.

**Parameter source note:** Throughout this document, Ca2+ parameters are the fixed learned values from `data/bestLearnedCaMKIIParams_0.dat`, and stress parameters are the learned values from `data/bestLearnedStressParams_6.dat`. The learned stress parameters are stored as raw (unconstrained) values and mapped to bounded ranges via sigmoid: `actual = min + (max - min) * sigmoid(raw)`. See Section 8.3 for the full raw-to-actual mapping.

### 1.2 E-folding and Tau

For an exponential decay process `x(t) = x_0 * exp(-t/tau)`, one **e-folding** (`t = tau`) is the time for the quantity to decay by a factor of `1/e`, retaining ~37% of its initial value (~63% decayed). After `n` e-foldings (`t = n * tau`):

| Time | Remaining | Decayed | Interpretation |
|---|---|---|---|
| 1 tau | e^-1 = 37% | 63% | One e-folding |
| 2 tau | e^-2 = 13.5% | 86.5% | |
| 3 tau | e^-3 = 5% | 95% | |
| 5 tau | e^-5 = 0.67% | 99.3% | Practically complete |

Throughout this document, **e-folding** refers to the time for 63% completion, and **5x tau** is used as the "practically finished" benchmark (~99%).

### 1.3 Effective Timescales in Bio Steps

What matters for calibration is the **e-folding time in bio steps** for each process, which is `tau / dt`:

| Process | Time constant (ODE) | dt per bio step | e-folding (bio steps) | 5x tau (~99%, bio steps) |
|---|---|---|---|---|
| Vmem | ~1 (O(1)) | 0.01 | ~100 | ~500 |
| Ca2+ | tau_ca = 2.60 | 0.01 | **260** | 1,300 |
| Stress S | tau_S = 1.49 (learned) | 0.1 | **14.9** | 75 |
| Field F | 1/gamma_F = 10,000 | 1.0 | **10,000** | 50,000 |

**Note on tau vs 1/gamma:** For Ca2+ and S, the time constant `tau` appears explicitly in the ODE as `dx/dt = .../tau`, so `tau` directly gives the e-folding time. For the field F, the decay term is `dF/dt = ... - gamma_F * F`, so the e-folding time is `1/gamma_F`. In both cases, the e-folding time answers the same question: **how long does the signal persist?**

- Large `gamma_F` = F is degraded quickly = short-lived signal (eATP broken down fast by ectonucleotidases)
- Small `gamma_F` = F persists for a long time = long-lived signal
- Zero `gamma_F` = F never decays = signal accumulates indefinitely

This is the **lifetime** of the signal, not the speed of some process. A long-lived eATP field (large `1/gamma_F`) means the inter-embryo communication channel has long memory, not that it acts slowly.

**Note on D_F and effective lifetime:** The `1/gamma_F` value is the bulk decay lifetime in the absence of spatial effects. However, the field F is also removed by **diffusive loss to the boundaries**. The model uses absorbing boundary conditions (`runGroupRescue.py` line 575: Laplacian computed with `max_degree`, so boundary cells diffuse to a virtual exterior at F=0). This creates two loss mechanisms that act **additively**:

1. **Bulk decay** (gamma_F): removes F everywhere at rate `gamma_F`
2. **Diffusive boundary loss** (D_F + absorbing BC): diffusion carries F down concentration gradients toward the boundary, where it is absorbed. Even interior cells (which have no direct boundary contact) lose F indirectly through this transport.

For the diffusion-reaction equation `dF/dt = D_F * laplacian(F) - gamma_F * F + emission` on a finite domain with absorbing BC, the system-level decay rate of the slowest-decaying spatial mode is:

```
gamma_system = gamma_F + C * D_F / L²
```

where L is the domain half-width (center-to-edge distance in lattice spacings) and C is a geometric constant (~pi²/4 ≈ 2.5 for 1D; pi²/2 ≈ 5 for 2D). The effective signal lifetime is `1/gamma_system`. Both loss mechanisms contribute additively -- they always act together, not as competing alternatives. See [Appendix A](#appendix-a-why-gamma_system--gamma_f--c--d_f--l) for a step-by-step derivation.

| Grid size | L | C*D_F/L² | gamma_F | gamma_system | Lifetime (bio steps) | Lifetime (real) | Dominant term |
|---|---|---|---|---|---|---|---|
| 3×3 | ~1 | ~5.0 | 0.0001 | ~5.0 | ~0.2 | ~12 ms | Diffusion |
| 5×5 | ~2 | ~1.2 | 0.0001 | ~1.2 | ~0.8 | ~46 ms | Diffusion |
| 10×10 | ~5 | ~0.20 | 0.0001 | ~0.20 | ~5 | ~0.3 s | Diffusion |
| 17×17 | ~8 | ~0.08 | 0.0001 | ~0.08 | ~13 | ~0.7 s | Diffusion |
| Infinite | inf | 0 | 0.0001 | 0.0001 | 10,000 | ~10 min | Bulk decay |

(Using C ≈ 5 as an approximation for 2D grids with absorbing BC.)

**Diffusive boundary loss dominates bulk decay for all realistic grid sizes** (gamma_F = 0.0001 is negligible compared to `C*D_F/L²` for grids smaller than ~200×200). The effective signal lifetime is sub-second for typical embryo groups.

This is a key feature of the model architecture: **larger groups retain F for longer** (`L²` scaling), naturally producing the group-size-dependent rescue that matches the CEMA experimental finding. The `1/gamma_F = 10,000` bio step bulk lifetime is the asymptotic limit for an infinitely large group, never approached in practice.

Note: the default `tau_S = 50` in `stressBistableSwitch.py` would give an e-folding of 500 bio steps, but the optimizer in `learnStressBistableSwitch.py` pushed `tau_S` to near its lower bound (1.49 vs min 1.0), resulting in only 14.9 bio steps. This means **S equilibrates ~17x faster than Ca2+**, inverting the intended biological hierarchy where stress kinases should be slower than Ca2+ transients.

### 1.4 Anchoring to Real Time

The most reliable biological anchor is Ca2+ clearance dynamics, since intracellular calcium signaling is the best-characterized layer experimentally. Cytoplasmic Ca2+ clearance in embryonic cells (via SERCA pumps, mitochondrial uptake, and plasma membrane Ca2+-ATPase) operates on timescales of 5-30 seconds for sustained elevations relevant to stress kinase cascades. Taking ~15 seconds for the e-folding time:

**Ca2+ e-folds in 260 bio steps = 15 seconds**

**Calibration: 1 bio step = 15 / 260 = 0.058 seconds (58 ms).**

### 1.5 Resulting Timescale Mapping

| Process | e-folding (bio steps) | Real time (e-fold) | Real time (5x tau) | Expected biological range | Verdict |
|---|---|---|---|---|---|
| Vmem | ~100 | ~5.8 s | ~29 s | Seconds (non-excitable cells) | Correct |
| Ca2+ | 260 | **15 s** | 75 s | 5-30 s | **Anchor** |
| Stress S (learned) | 14.9 | **0.86 s** | 4.3 s | 3-15 min (p38/JNK peak) | **~200-1000x too fast** |
| Stress S (default) | 500 | 29 s | 2.4 min | 3-15 min (p38/JNK peak) | Too fast (~5-10x) |
| Field F (bulk 1/gamma_F) | 10,000 | 9.6 min | 48 min | 2-10 min (eATP half-life in bulk) | **~3-5x too slow** |
| Field F (effective, 5×5) | ~0.8 | **~46 ms** | ~230 ms | Seconds-minutes (small cluster eATP) | **Too fast** — D_F is ~200,000x above molecular diffusion |
| Field F (effective, 10×10) | ~5 | **~0.3 s** | ~1.5 s | Seconds-minutes (medium cluster eATP) | **Too fast** — diffusion-dominated, boundary loss drains F rapidly |
| Field F (effective, 17×17) | ~13 | **~0.7 s** | ~3.7 s | Minutes (large cluster eATP) | **Too fast** — but group-size scaling direction is correct |
| Full sim (2000 steps) | 2000 | **1.9 min** | -- | Hours (Tung et al.) | **Too short** |

**Note on Field F:** The bulk lifetime (1/gamma_F = 9.6 min) is in the right ballpark for eATP degradation by ectonucleotidases. However, the effective lifetime is dominated by diffusive boundary loss (see Section 1.3), which depends on grid size via `gamma_system = gamma_F + C*D_F/L²` (see Appendix A). The effective lifetimes are far too short because D_F = 0.5 corresponds to a physical diffusion rate of ~78,000,000 um²/s — about 200,000x faster than molecular eATP diffusion (~300-400 um²/s). This means eATP floods to the boundary and escapes almost instantly. The **qualitative** group-size scaling (larger groups retain F longer) is correct and biologically meaningful, but the **quantitative** timescales are compressed by orders of magnitude. See Section 5 for detailed D_F analysis.

### 1.6 Key Findings from Corrected Calibration

**The corrected calibration reverses the previous assessment of Layer 3.** The inter-embryo field F now has a biologically plausible ~10 minute e-folding, consistent with eATP degradation by ectonucleotidases (CD39/CD73). The previous analysis (which incorrectly assumed 1 bio step = 1 ODE time unit) had concluded gamma_F was ~10-20x too low; this turns out to be approximately correct.

**Three concerns emerge:**

1. **Stress S is far too fast (learned parameters).** With the learned tau_S = 1.49, S e-folds in only 14.9 bio steps (~0.86 seconds), making it **17x faster than Ca2+**. This completely inverts the biological hierarchy where stress kinases (p38/JNK, timescale 3-15 minutes) should be ~10-50x *slower* than Ca2+ transients. Even with the default tau_S = 50, the S/Ca2+ ratio is only 1.9x due to the multi-clock architecture. The optimizer pushed tau_S to near its lower bound because the short simulation window rewards fast convergence.

2. **Simulation window is too short.** The 2000-step simulation covers only ~2 minutes of real time. The Tung et al. experiments involve hours of embryo incubation. The inter-embryo field hasn't even reached steady state by step 2000 (requires ~50,000 steps = ~48 min for 5x tau).

3. **Multiple learned parameters hit their bounds.** The optimizer pushed 4 of 11 stress parameters to their lower bounds (tau_S, k_off_S, sigma_ca, D_S, K_decay), suggesting it is fighting the timescale architecture to make S respond as fast as possible within the simulation window. See Section 3 for details.

### 1.7 Alternative Calibration Anchor

If the Ca2+ in this model represents a *slow* developmental calcium signal (minutes-scale store-operated Ca2+ entry or calcineurin-pathway engagement rather than fast VGCC transients), using tau_ca ~ 2 minutes gives:

**1 bio step = 120 s / 260 = 0.46 seconds**

| Process | Real time (e-fold) | Real time (5x tau) | Verdict |
|---|---|---|---|
| Vmem | ~46 s | ~3.8 min | Plausible (non-excitable cells) |
| Ca2+ | **2 min** | 10 min | **Anchor** |
| Stress S | **3.8 min** | 19 min | Plausible (p38/JNK range) |
| Field F (bulk 1/gamma_F) | 77 min | 6.4 hr | Too slow (~8-40x above eATP half-life of 2-10 min) |
| Field F (effective, 5×5 grid) | **0.37 s** | ~1.8 s | Too fast (D_F too large) |
| Field F (effective, 10×10 grid) | **2.3 s** | ~12 s | Too fast (D_F too large) |
| Field F (effective, 17×17 grid) | **6.0 s** | ~30 s | Too fast (D_F too large) |
| Full sim (2000 steps) | **15.4 min** | -- | Too short |

This makes stress S more reasonable (~4 min, within p38/JNK range) but pushes the full simulation to only ~15 minutes and makes Vmem equilibration slow (~46 s, still within range for non-excitable cells).

**Important caveat on Field F:** The 77-minute value is the **bulk decay** lifetime (`1/gamma_F`), which assumes an infinitely large domain where only ectonucleotidase degradation removes eATP. In practice, diffusive loss to absorbing boundaries dominates (see Section 1.3), giving effective lifetimes of seconds for realistic grid sizes. The 77-minute value should be interpreted as the biological property of the molecule (ectonucleotidase half-life) rather than the effective signal persistence in the simulation. The short effective lifetimes are physically reasonable -- eATP released by a small cluster of cells dissipates rapidly into the surrounding medium, just as it would in vivo. The `1/gamma_F` bulk lifetime would only apply in a very large tissue volume or sealed chamber where boundary effects are negligible.

### 1.8 Injury Wave Calibration Anchor

Tung et al. (2024, Fig. 8) directly measured the speed of calcium/ATP injury waves propagating between embryos in culture:

| Measurement | Speed | N |
|---|---|---|
| Within injured embryo | 6.7 ± 3.48 μm/s | 7 |
| **Between embryos (through medium)** | **5.28 ± 1.89 μm/s** | **8** |
| Within receiver (uninjured) embryo | 2.36 ± 1.66 μm/s | 9 |

The wave crosses >1 mm in the medium, and suramin (P2 receptor antagonist) attenuates the inter-embryo transfer, confirming ATP signaling as the mechanism.

This is a powerful calibration anchor because it simultaneously constrains **both** spatial and temporal scales from a direct measurement in the same experimental system. Unlike the Ca²⁺ clearance anchor (Section 1.4), which only constrains time, the wave speed constrains the ratio D/k — linking the field's diffusion rate to its reaction kinetics.

#### What the wave speed constrains

For a reaction-diffusion wave (where each embryo receives eATP, amplifies the signal via stress-induced ATP release, and re-emits), the wave speed is approximately:

```
v_wave ≈ 2 * sqrt(D_physical * k_reaction)
```

where D_physical is the eATP diffusion coefficient (μm²/s) and k_reaction is the effective rate of the regenerative step (1/s). Using the inter-embryo speed of ~5 μm/s:

```
5 = 2 * sqrt(D_physical * k_reaction)
D_physical * k_reaction ≈ 6.25 μm²/s²
```

**If D_physical ~ 350 μm²/s** (molecular eATP diffusion):
```
k_reaction ≈ 6.25 / 350 ≈ 0.018 /s  (timescale ~56 seconds)
```

This is a reasonable regenerative timescale — an embryo receiving eATP takes about 1 minute to amplify and re-release it (consistent with Ca²⁺ transduction + pannexin-1 release).

**If D_physical ~ 78,000,000 μm²/s** (the model's current value):
```
k_reaction ≈ 6.25 / 78,000,000 ≈ 8 × 10⁻⁸ /s  (timescale ~145 days)
```

This is absurd — the regenerative step would take months. The model's D_F is far too large to reproduce the observed wave speed.

#### Back-calculating D_F from the wave speed

Using the wave speed to set D_physical, and then converting to dimensionless D_F (see Appendix B):

With v = 5 μm/s and k_reaction ~ 0.018 /s (56 s regenerative timescale):
```
D_physical = v² / (4 * k_reaction) = 25 / 0.072 ≈ 350 μm²/s
```

Converting to model units (delta_x = 3000 μm, T = 0.058 s per ODE time unit):
```
D_F = D_physical * T / delta_x² = 350 * 0.058 / (3000)² ≈ 2.3 × 10⁻⁶
```

This is **~200,000x smaller** than the current D_F = 0.5, confirming the assessment in Section 5 that the model's diffusion is far too fast for molecular transport.

#### Time to cross one embryo spacing

An independent check: the wave takes ~3000 μm / 5 μm/s = **600 seconds (~10 minutes)** to travel between adjacent embryos. In the model, signal crosses one lattice spacing in roughly 1/D_F = 2 ODE time units = 2 bio steps = 0.116 seconds. This is **~5000x too fast**.

#### Implications for model calibration

The injury wave provides the most direct experimental constraint on the field layer because it was measured in the same system (Xenopus embryo cultures) and involves the same signaling mechanism (eATP/P2 receptors). It strongly suggests that:

1. **D_F should be reduced by ~5 orders of magnitude** (from 0.5 to ~10⁻⁶) if the lattice spacing represents literal physical distance
2. **Alternatively**, if D_F = 0.5 is retained as an abstract connectivity parameter, the model should be understood as operating in a **well-mixed regime** that does not capture the spatial propagation dynamics of the injury wave
3. The current model cannot reproduce the ~10-minute inter-embryo propagation timescale — it would need either much smaller D_F or much larger lattice spacing

---

## 2. Layer 1: Intracellular Ca2+ Dynamics (Grade: B+)

Each cell has voltage-gated Ca2+ channels:
```
I_ca = g_ca * sigmoid((Vmem - V_half_ca) / k_ca) * (E_ca - Vmem) / 0.1
dCa/dt = I_ca - (1/tau_ca) * Ca - k_decay_ca
```

### Biological Mapping

**E_ca = +130 mV (Ca2+ reversal potential).** The Nernst potential for Ca2+ with extracellular [Ca2+] ~ 1-2 mM and intracellular [Ca2+] ~ 100 nM is +128 to +132 mV. This is essentially exact.

**V_half_ca = -75.3 mV (half-activation voltage).** Voltage-gated Ca2+ channels in embryonic cells have half-activation voltages of -60 to -40 mV for T-type (Cav3.1/3.2) and -40 to -20 mV for L-type (Cav1.2). At -75.3 mV, the channel is substantially activated at resting potential, which is appropriate if this represents a tonically active conductance modulated by Vmem changes rather than a sharply gated channel. Acceptable but on the hyperpolarized end; -60 to -50 mV would be more canonical for T-type channels.

**k_ca = 2.1 mV (Boltzmann slope factor).** This is the primary concern. Boltzmann slope factors for voltage-gated Ca2+ channels are typically 5-10 mV (L-type: ~6-8 mV, T-type: ~5-7 mV). A 2.1 mV slope produces an almost switch-like activation curve, transitioning from 10% to 90% activation over only ~9 mV (compared to ~40 mV for a typical channel). This makes Ca2+ influx behave as a binary switch rather than a graded conductance. **Too steep by 2-3x.** For a paper, this can be framed as representing the aggregate effect of multiple cooperative voltage-sensing mechanisms rather than a single channel species.

**g_ca = 5.34 (dimensionless conductance).** Combined with the steep slope factor and saturation clipping at Ca = 10, the effective Ca2+ dynamics are binary: cells are either near-zero or saturated at the clamp boundary. This is a simplification but not unreasonable for a bistable switch model where the downstream stress pathway has its own threshold.

**k_decay_ca = 4.33 (constant baseline consumption).** Represents constitutive Ca2+ buffering and sequestration. Combined with the proportional 1/tau_ca term, the two-component clearance mirrors SERCA pump Michaelis-Menten kinetics (constant-rate pumping at high [Ca2+] above Km ~ 0.1-0.4 uM, proportional clearance near baseline). Reasonable architecture.

### Assessment

The architecture is sound. E_ca is excellent and V_half_ca is acceptable. The main concern is the steep slope factor k_ca creating effectively binary Ca2+ dynamics, which is a deliberate abstraction rather than a biological error.

---

## 3. Layer 2: Stress Bistable Switch (Grade: B+)

Each cell has a stress variable S in [0,1] governed by bistable reaction-diffusion dynamics:
```
ca_drive = sigmoid((Ca - Ca_stress_threshold) / sigma_ca)
self_activation = (S^2 - K_S^2) / (S^2 + K_S^2)
or_gate = sigmoid(gain_S * ca_drive + self_activation - or_threshold_S)
reaction = (k_on_S * or_gate * (1 - S) - k_off_S * S) / tau_S
decay = -gamma * S / (K_decay + S)
diffusion = D_S * laplacian(S)
```

### Biological Identity of S

The best candidate is the **p38 MAPK / JNK pathway**, for four reasons:
1. p38 and JNK are canonical stress-activated kinases with well-documented bistability via positive feedback through upstream kinases (ASK1, MLK3) and ROS amplification
2. They operate on the correct timescale (minutes to tens of minutes)
3. They directly regulate ATP release via connexin hemichannels and pannexin-1
4. They have documented roles in embryonic stress responses

NFAT is also viable but operates somewhat more slowly (nuclear translocation takes 15-30 minutes, NFAT-dependent transcription takes hours), which would push tau_S to longer values than modeled.

### Learned Parameters (from `data/bestLearnedStressParams_6.dat`)

The learning script (`learnStressBistableSwitch.py`) stores raw unconstrained values that are mapped to bounded ranges via sigmoid: `actual = min + (max - min) * sigmoid(raw)`.

| Parameter | Raw | Bounds [min, max] | Actual (learned) | Default | At bound? |
|---|---|---|---|---|---|
| tau_S | -2.856 | [1.0, 10.0] | **1.49** | 50.0 | Near min |
| k_on_S | -1.906 | [0.5, 10.0] | **1.73** | 3.0 | |
| k_off_S | -90.67 | [0.001, 1.0] | **0.001** | 0.02 | **At min** |
| K_S | -1.821 | [0.1, 0.8] | **0.20** | 0.4 | Near min |
| Ca_stress_threshold | 1.424 | [0.001, 10.0] | **8.06** | 0.8 | Near max |
| sigma_ca | -91.07 | [0.005, 2.0] | **0.005** | 0.2 | **At min** |
| gain_S | -3.434 | [1.0, 6.0] | **1.16** | 2.0 | Near min |
| or_threshold_S | 1.811 | [0.1, 2.0] | **1.73** | 0.6 | Near max |
| D_S | -90.04 | [0.01, 0.3] | **0.01** | 0.15 | **At min** |
| gamma | -0.043 | [0.01, 0.5] | **0.25** | 0.08 | |
| K_decay | -95.57 | [0.01, 0.5] | **0.01** | 0.3 | **At min** |

**5 of 11 parameters hit their bounds** (k_off_S, sigma_ca, D_S, K_decay at minimum; tau_S near minimum). This is a strong signal that the optimizer is fighting the model architecture -- it pushes S to respond as fast as possible (low tau_S), with no spatial diffusion (D_S -> 0), no deactivation (k_off_S -> 0), minimal decay (K_decay -> 0), and a sharp binary Ca2+ threshold (sigma_ca -> 0). The result is a stress switch that behaves as a fast, non-spatial, nearly irreversible binary detector, which is effective computationally but loses the biological properties (temporal integration, spatial filtering, reversibility) that motivate the reaction-diffusion architecture.

### Biological Mapping

**tau_S = 1.49 (learned) vs 50.0 (default).** With dt_stress = 0.1, the learned value gives an e-folding of only 14.9 bio steps (~0.86 seconds), making S **17x faster than Ca2+**. This completely inverts the biological hierarchy: p38/JNK activation peaks at 5-15 minutes after stress exposure, so it should be ~10-50x *slower* than Ca2+ (e-folding ~15 s), not 17x faster. Even the default tau_S = 50 gives only a 1.9x ratio (S slightly slower than Ca2+) due to the multi-clock architecture (see Section 7.1). The optimizer pushed tau_S to near its lower bound because the short simulation window (2000 steps = ~2 min) rewards fast convergence.

**k_on_S/k_off_S ratio = 1730:1 (learned) vs 150:1 (default).** The optimizer pushed k_off_S to its minimum (0.001), creating an essentially irreversible switch. While the default 150:1 asymmetry is biologically plausible (strong hysteresis), a 1730:1 ratio makes the switch nearly one-way. For p38/JNK, deactivation via MKP-1 phosphatases does occur (timescale 30-60 minutes), so some nonzero k_off_S is needed for biological realism.

**K_S = 0.20 (learned) vs 0.4 (default).** The lower threshold means the bistable switch triggers at only 20% kinase activation (vs 40% default). This is at the low end of the plausible range (0.3-0.5) and makes the switch easier to trigger.

**Ca_stress_threshold = 8.06 (learned) vs 0.8 (default), sigma_ca = 0.005 (learned) vs 0.2 (default).** The optimizer set a high Ca2+ threshold (8.06 out of [0, 10] range) with an extremely sharp sigmoid (sigma_ca = 0.005, effectively a step function). This creates a binary all-or-nothing Ca2+ gate: cells with Ca2+ above ~8 activate stress; those below do not. Biologically, stress kinase activation by Ca2+ occurs at ~0.5-2 uM with a graded dose-response, not a binary threshold.

**D_S = 0.01 (learned) vs 0.15 (default).** The optimizer minimized spatial coupling, pushing D_S to its lower bound. This effectively disables the spatial frequency filtering that was a key design feature of the reaction-diffusion architecture (see `FIELD_RESCUE_DESIGN.md` Section 7). Without diffusion, S operates as an independent per-cell switch with no intercellular communication within the embryo.

**gamma = 0.25 (learned) vs 0.08 (default), K_decay = 0.01 (learned) vs 0.3 (default).** The optimizer increased the phosphatase Vmax (gamma: 0.08 -> 0.25) while minimizing Km (K_decay: 0.3 -> 0.01). A very low Km means the phosphatase is saturated even at low S, producing nearly constant-rate decay. Combined with the high Vmax, this creates strong decay that keeps S from drifting upward in the healthy case, compensating for the near-zero k_off_S.

**D_S interpretation (when non-zero).** Whether using the default (0.15) or learned (0.01) value, D_S cannot represent diffusion of phosphorylated p38 itself (38 kDa, too large for gap junctions with ~1 kDa cutoff). It must represent effective intercellular spread via **gap junction-permeable second messengers**:
- IP3 (420 Da): well-documented gap junction permeability, creates intercellular Ca2+ waves
- H2O2 (34 Da): freely permeates gap junctions, activates ASK1 -> p38/JNK
- Ca2+ itself (40 Da): can spread through gap junctions

On an 11x11 grid with cell spacing ~10-20 um, D_S = 0.15 (default) translates to ~2.5-10 um^2/s, somewhat slow compared to IP3 diffusion (~100-300 um^2/s) but reasonable with gap junction barriers.

### Assessment

The bistable switch architecture is well-conceived in principle. The competitive self-activation, Ca2+ drive, phosphatase decay, and spatial diffusion capture the essential biology of stress kinase cascades. However, the learned parameters substantially deviate from the design intent: the optimizer collapsed the reaction-diffusion spatial filter into a fast, non-spatial, binary switch. This is effective computationally (achieves the learning objective) but loses the biological properties -- temporal integration, spatial filtering, graded sensitivity, reversibility -- that motivated the architecture. The grade reflects the architecture quality tempered by the learned parameter issues.

---

## 4. Layer 3: Inter-Embryo Stress Field (Grade: B+)

Stress field F diffuses between embryos on the 2D lattice:
```
dF/dt = D_F * laplacian(F) - gamma_F * F + emission
emission_i = mean(S_i)
```

### Biological Mapping

The inter-embryo stress field represents **extracellular ATP (eATP)** released by stressed embryos via pannexin-1 channels, diffusing through the shared culture medium.

**gamma_F = 0.0001 (eATP bulk decay).** The bulk decay alone gives an e-folding of `1/gamma_F` = 10,000 bio steps = ~9.6 minutes. However, this is an upper bound that is never reached in practice. With absorbing boundary conditions, diffusive loss dominates: the effective system decay rate is `gamma_system = gamma_F + C * D_F / L²` (see Section 1.3 and [Appendix A](#appendix-a-why-gamma_system--gamma_f--c--d_f--l) for derivation), giving effective lifetimes of seconds for typical grid sizes. eATP is degraded biologically by ectonucleotidases (CD39/CD73). The half-life depends strongly on context: **2-10 minutes** in dense tissue or intact embryos (high ectonucleotidase activity, high cell-surface-area-to-volume ratio), or **30 minutes to 2+ hours** in cell-free medium or sparse cultures (non-enzymatic hydrolysis only). For Xenopus embryo cultures — the relevant context for this model — the **2-10 minute** range is appropriate. The model's bulk e-folding (~10 min) falls at the upper end of this range. However, the effective eATP lifetime (seconds, dominated by diffusive boundary loss) is much shorter than the biological value; the absorbing boundaries represent eATP escaping the embryo group into the surrounding medium, which is an additional loss pathway beyond enzymatic degradation.

**D_F = 0.5 (effective inter-embryo diffusion).** eATP diffusion in aqueous medium has D_physical ~ 300-400 um^2/s. On a lattice with spacing L ~ 3 mm (typical Xenopus embryo spacing), mapping to physical units depends on the effective time base of the field ODE. With the field advancing 1.0 ODE time units per bio step (= 0.058 s real time):

D_physical = D_F * delta_x^2 / T = 0.5 * (3000 um)^2 / 0.058 s ~ 78,000,000 um^2/s

(See [Appendix B](#appendix-b-converting-dimensionless-diffusion-to-physical-units) for the derivation of this conversion formula.)

This is unrealistically high for pure molecular diffusion. However, three mitigating factors apply:
1. The embryo lattice is coarse-grained; "diffusion" integrates over many molecular diffusion events per timestep
2. Convective mixing (thermal convection, embryo ciliary flow in Xenopus) substantially enhances effective transport beyond pure diffusion
3. The lattice spacing in the model is abstract -- it represents connectivity, not literal physical distance

**Communication range: lambda = sqrt(D_F/gamma_F) ~ 70 lattice spacings.** This creates an effectively **well-mixed mean-field** regime where each embryo senses the group-average stress. In a well-mixed dish, eATP released by one embryo does reach all others via convective mixing. This is consistent with the experimental observation that group size (not spatial arrangement) determines rescue efficacy (Tung et al. 2024).

### eATP Concentration Mapping

Tung et al. measured eATP of 1.073 uM (treated) vs 0.989 uM (control) -- a difference of only ~84 nM (~8%). P2X receptors (EC50 ~ 1-10 uM) would show ~8% change in occupancy from this increment. This supports the idea that rescue operates through **temporal integration** of small concentration differences over hours rather than instantaneous concentration sensing.

### Assessment

The architecture (diffusion-degradation with absorbing boundaries) is appropriate and produces a desirable emergent property: group-size-dependent signal retention. The effective signal lifetime is controlled by diffusive boundary loss (seconds, grid-size-dependent), not bulk decay (minutes). This means F reaches a dynamic steady state relatively quickly (within hundreds of bio steps), where emission balances diffusive loss. The high D_F creates rapid spatial equilibration within the group.

---

## 5. Layer 4: Rescue Modulation (Grade: B)

```
effective_damping = sigmoid(logit(base_damping) + alpha * F_i)
```

### Biological Mapping

The mechanism (eATP acting through P2 receptors to modulate intracellular signaling) is well-established:
- P2Y receptors -> PLC-beta -> IP3 -> Ca2+ release -> calcineurin -> NFAT nuclear translocation -> protective gene expression
- P2X receptors -> direct Ca2+ influx -> CREB, NF-kappaB activation

**alpha = 10 (rescue signal strength).** A stress field value of F = 0.1 shifts log-odds of damping by 1.0 (a large effect: 50% -> 73% in logistic terms). At F = 0.5, the shift saturates the sigmoid. The magnitude of alpha determines how much accumulated eATP is needed for rescue. Given the short simulation window (~2 min), F may not accumulate to large values, so a high alpha may be necessary to produce observable rescue within the simulated timeframe.

### Assessment

The pathway architecture is plausible. "GRN damping" is abstract -- biologically, the rescue signal would more likely modulate a specific protective pathway. Candidates:
- HSP70/90 upregulation (stabilizes developmental signaling proteins)
- Bcl-2 anti-apoptotic pathway
- Nrf2-mediated antioxidant response
- Direct enhancement of morphogenetic gene expression via NFAT/CREB

---

## 6. Timescale Hierarchy

### 6.1 Corrected Hierarchy with Learned Parameters (1 bio step = 58 ms)

```
  S (~0.9 s)  F_eff (~3 s)  Vmem (~6 s)   Ca2+ (15 s)
      |            |              |              |
      v            v              v              v
    --|----|----|----|----|----|----|----|----|----|---->  time
   Stress   eATP field   Bioelectric   Calcium
  (INVERTED: (diffusion-              (anchor)
   17x faster  dominated)
   than Ca2+!)
```

With the learned tau_S = 1.49 and effective F lifetime for a 10×10 grid:
- **S/Ca2+ ratio = 14.9/260 = 0.057x** -- S is **17x faster** than Ca2+. This completely inverts the biological hierarchy. Stress kinases (p38/JNK) should be 10-50x *slower* than Ca2+, not faster.
- **F_eff/Ca2+ ratio = ~50/260 = 0.19x** -- the field also equilibrates faster than Ca2+ for typical grid sizes, due to diffusive boundary loss.

### 6.2 Hierarchy with Default Parameters (for comparison)

```
    F_eff (~3 s)  Vmem (~6 s)   Ca2+ (15 s)   S (~29 s)
        |              |             |                |
        v              v             v                v
    ----|----|----|----|----|----|----|----|----|------|---->  time
    eATP field   Bioelectric   Calcium    Stress kinase
   (diffusion-                            (too fast:
    dominated)                             should be 3-15 min)
```

With default tau_S = 50:
- **S/Ca2+ ratio = 500/260 = 1.9x** -- better than learned, but still far below the biological 10-50x.

### 6.3 The Hierarchy Ratios Matter More Than Absolute Times

For pattern formation and rescue dynamics, the **ratio between timescales** is more important than their absolute values. These ratios are intrinsic to the model (determined by tau values and dt choices):

| Ratio | Learned Value | Default Value | Biological Target | Assessment |
|---|---|---|---|---|
| S / Ca2+ | 0.057x (S faster!) | 1.9x | 10-50x (S slower) | **Inverted (learned) / Too small (default)** |
| F_eff / S (10×10 grid) | 3.4x | 0.1x | 5-20x | Depends on grid size |
| F_eff / Ca2+ (10×10 grid) | 0.19x | 0.19x | 10-100x | Too small |

Note: F_eff is the effective field lifetime dominated by diffusive boundary loss (`L²/D_F`), not bulk decay (`1/gamma_F`). The F_eff ratios are grid-size-dependent and scale as `L²`.

---

## 7. Key Concerns and Recommendations

### 7.1 Stress S responds too quickly relative to Ca2+ (Priority 1)

With the learned tau_S = 1.49, S e-folds in 14.9 bio steps -- **17x faster than Ca2+** (260 bio steps). Even with the default tau_S = 50, the S/Ca2+ ratio is only 1.9x. Biologically, the stress kinase pathway should be ~10-50x *slower* than Ca2+, acting as a temporal integrator.

**Root cause (two compounding factors):**
1. **Multi-clock architecture:** Ca2+ uses dt_ca = 0.01 while S uses dt_stress = 0.1, so a 10x ODE time ratio only produces a 1x bio step ratio. The effective ratio is `(tau_S / dt_stress) / (tau_ca / dt_ca)` = `tau_S / (10 * tau_ca)`.
2. **Optimizer pressure:** The learning bounds for tau_S are [1.0, 10.0], giving effective ratios of 0.038x to 0.38x in bio steps. The optimizer pushed tau_S to near its lower bound (1.49) because the short simulation window (2000 steps) rewards fast convergence. **No value within the current bounds can make S slower than Ca2+.**

**Fix options:**

| Option | Change | Resulting S/Ca2+ ratio | Pros | Cons |
|---|---|---|---|---|
| A: Unify dt | Set dt_stress = 0.01 (same as dt_ca) | tau_S/tau_ca directly | Clean, transparent | S converges ~10x slower (needs more bio steps) |
| B: Widen tau_S bounds | tau_S bounds [100, 1500] | 3.8x to 57x | Allows bio plausible range | Opaque (different dt values) |
| C: Both A + relearn | Unify dt, bounds [10, 150], relearn | 3.8x to 57x | Best of both worlds | Requires rerunning learning |

**Recommended: Option C.** Unify dt values so tau_S/tau_ca directly gives the timescale ratio, set biologically-motivated bounds [10, 150] (giving ratios of 3.8x to 57x), and relearn. The current learned parameters are not biologically meaningful because the optimizer was fighting an architecture that cannot express the correct timescale hierarchy.

Files: `runGroupRescue.py` lines 620-621, `learnStressBistableSwitch.py` lines 226 and 302-303.

#### Why does the model still work despite the timescale inversion?

The S switch correctly discriminates between undamped and damped Ca2+ patterns even though S is 17x faster than Ca2+. This is because discrimination depends on **bistability**, not on **timescale separation**. These are two different properties.

The learned parameters reveal the strategy the optimizer found:

| Parameter | Value | Significance |
|---|---|---|
| tau_S | 1.49 | Near minimum — S responds almost instantly |
| k_off_S | 0.001 | At minimum — but Michaelis-Menten decay (gamma = 0.25) dominates, so S still turns OFF when Ca drops |
| K_S | 0.20 | Near minimum — low threshold for self-activation lock-in |

The optimizer built a **fast, sharp Ca2+ threshold detector** rather than a true bistable switch. Analysis of the learned parameters reveals:

**S is not bistable.** With Ca = 0, there is only one steady state: S ≈ 0.004 (OFF). The Michaelis-Menten decay (gamma = 0.25) overwhelms the self-activation at all Ca levels below threshold. The self-activation machinery (K_S, competitive feedback) is barely contributing.

**The discrimination comes from an ultra-sharp Ca2+ sigmoid.** With sigma_ca = 0.005 (at its lower bound) and Ca_stress_threshold = 8.06, the ca_drive is effectively a binary step function:

| Ca level | ca_drive | S steady state |
|---|---|---|
| < 8.05 | ≈ 0 | S ≈ 0.004 (OFF) |
| ≈ 8.06–8.10 | 0.14–1.0 | Narrow bistable window (3 steady states) |
| > 8.10 | ≈ 1 | S ≈ 0.62 (ON) |

The bistable window is only ~0.04 Ca units wide — too narrow to matter in practice. S is ON if and only if Ca > ~8.06, and OFF otherwise.

**What happens temporally:**

1. Ca2+ pattern develops slowly (over ~260 bio steps)
2. S tracks Ca2+ almost instantaneously (e-folds in ~15 bio steps), acting as a **nonlinear snapshot** of the current Ca2+ level
3. In cells where Ca2+ > 8.06, S snaps to ~0.62 (ON)
4. In cells where Ca2+ < 8.06, S stays near 0 (OFF)
5. If Ca2+ later drops below 8.06, **S turns OFF** — there is no persistent memory

A slower S (biologically realistic) would reach the **same final pattern** — it would just take longer to get there. The fast S reaches the same answer sooner.

**What is lost by being fast:** In a biological system with noisy Ca2+ signals, a slow S would act as a **temporal low-pass filter**, integrating Ca2+ over time and smoothing out transient fluctuations. The fast S responds to instantaneous Ca2+ values, making it sensitive to noise. This doesn't matter in the current model (clean, deterministic Ca2+ patterns), but would matter in real cells where Ca2+ oscillates stochastically. The biological argument for slow stress kinases is **robustness to noise**, not computational necessity.

**What is lost by not being truly bistable:** A genuine bistable S would maintain its pattern even after Ca2+ decays (analogous to CaMKII in the facial patterning model). The current S requires sustained Ca2+ to stay ON. This means the stress pattern is entirely dependent on the Ca2+ pattern at every moment — if Ca2+ homogenizes (e.g., through gap junction coupling in a rescue scenario), S will homogenize too, with no hysteresis or memory.

### 7.2 Simulation window is too short (Priority 2)

At 2000 bio steps = ~2 minutes (fast anchor) or ~15 minutes (slow anchor), the simulation does not cover the hours-long timescale of the Tung et al. experiments. The inter-embryo field F requires ~50,000 bio steps to fully equilibrate.

**Options:**
- Increase num_bio_steps to 20,000-60,000 (computationally expensive with many embryos)
- Accept the model simulates only the initial rescue dynamics, not the full experimental timecourse
- Adopt the slow-Ca2+ anchor, arguing the model's "Ca2+" represents a slow store-operated pathway

### 7.3 Ca2+ voltage sensitivity (k_ca) is ~2-3x too steep (Priority 3)

The 2.1 mV Boltzmann slope creates a near-binary activation curve, whereas biological voltage-gated Ca2+ channels have slopes of 5-10 mV. This makes Ca2+ dynamics effectively binary (off/saturated), reducing the graded information available to the stress switch.

| | Current | Recommended | Biological Basis |
|---|---|---|---|
| k_ca | 2.1 mV | 4-6 mV | Boltzmann slope for T-type/L-type VGCCs |
| g_ca | 5.34 | Reduce proportionally | Maintain same effective Ca2+ at midpoint |

Files: `stressBistableSwitch.py` defaults (line 65), learned parameters in `.dat` files.

### 7.4 Diffusion of S requires explicit second-messenger interpretation (Priority 4 -- paper text)

The stress variable S (representing p38/JNK phosphorylation) cannot itself traverse gap junctions. D_S must be interpreted as effective spread via small second messengers (IP3, H2O2) that activate the pathway in neighboring cells. Suggested paper text: *"D_S represents the effective intercellular spread of stress signaling via gap junction-permeable second messengers (IP3, molecular weight 420 Da; or H2O2, molecular weight 34 Da) that activate the stress kinase pathway in neighboring cells."*

### 7.5 Rescue modulation is abstract (Priority 5 -- paper text)

"GRN damping" is not a specific molecular target. A more biologically specific mechanism (e.g., HSP70/90 upregulation, Bcl-2 induction via eATP -> P2Y -> NFAT pathway) would strengthen the model's biological grounding and generate testable predictions.

### 7.6 Testable prediction: spacing-dependent rescue (Priority 6 -- optional)

If the well-mixed assumption is relaxed (by reducing D_F), the model predicts **spacing-dependent rescue**: closely packed embryos should show more rescue than widely spaced ones. This is experimentally testable by varying culture density and would distinguish this model from simple mean-field theories.

---

## 8. Parameter Reference

### 8.1 Multi-Clock Architecture (`runGroupRescue.py` lines 620-621, 572)

| Subsystem | dt per bio step | ODE time per bio step | e-folding (bio steps) | Real time (fast anchor) |
|---|---|---|---|---|
| Ca2+ | 0.01 | 0.01 | tau_ca/0.01 = 260 | **15 s** |
| Stress S (learned tau_S=1.49) | 0.1 | 0.1 | 1.49/0.1 = **14.9** | **0.86 s** |
| Stress S (default tau_S=50) | 0.1 | 0.1 | 50/0.1 = 500 | 29 s |
| Field F | 10 x 0.1 = 1.0 | 1.0 | 1/(gamma_F x 1.0) = 10,000 | **9.6 min** |

### 8.2 Ca2+ Dynamics (`stressBistableSwitch.py`)

| Parameter | Default | Learned | Biological Unit | Biological Value |
|---|---|---|---|---|
| tau_ca | 3.0 | 2.60 | seconds | ~15 s (fast anchor), ~2 min (slow anchor) |
| g_ca | 0.5 | 5.34 | dimensionless | Ca2+ conductance |
| V_half_ca | -0.04 V | -0.0753 V | mV | -75.3 mV |
| k_ca | 0.01 V | 0.0021 V | mV | 2.1 mV (too steep; biological: 5-10 mV) |
| k_decay_ca | 0.3 | 4.33 | conc/stu | constitutive SERCA Vmax |
| E_ca | 0.13 V | 0.13 V | mV | +130 mV (Nernst -- exact) |

### 8.3 Stress Bistable Switch (`stressBistableSwitch.py`, learned: `data/bestLearnedStressParams_6.dat`)

| Parameter | Default | Learned | Bounds | Biological Mapping | At bound? |
|---|---|---|---|---|---|
| tau_S | 50.0 | **1.49** | [1, 10] | p38/JNK activation timescale (bio: 3-15 min) | Near min |
| k_on_S | 3.0 | **1.73** | [0.5, 10] | Kinase activation rate | |
| k_off_S | 0.02 | **0.001** | [0.001, 1] | Phosphatase-limited deactivation (MKP-1) | **At min** |
| K_S | 0.4 | **0.20** | [0.1, 0.8] | Bistable decision boundary (~40% kinase pool) | Near min |
| Ca_stress_threshold | 0.8 | **8.06** | [0.001, 10] | Ca2+ level for kinase activation (~0.5-2 uM) | Near max |
| sigma_ca | 0.2 | **0.005** | [0.005, 2] | Sigmoid width (graded transition zone) | **At min** |
| gain_S | 2.0 | **1.16** | [1, 6] | Ca2+ drive amplification | Near min |
| or_threshold_S | 0.6 | **1.73** | [0.1, 2] | OR gate threshold (spatial selectivity) | Near max |
| D_S | 0.15 | **0.01** | [0.01, 0.3] | IP3/H2O2 gap junction diffusion (~2.5-10 um^2/s) | **At min** |
| gamma | 0.08 | **0.25** | [0.01, 0.5] | Phosphatase Vmax (MKP-1) | |
| K_decay | 0.3 | **0.01** | [0.01, 0.5] | Phosphatase Km (~0.5-5 uM for phospho-p38) | **At min** |

### 8.4 Inter-Embryo Stress Field (`runGroupRescue.py`)

| Parameter | Default | Biological Mapping | Assessment |
|---|---|---|---|
| D_F | 0.5 | eATP effective transport in medium | High (convective mixing assumed) |
| gamma_F | 0.0001 | Ectonucleotidase degradation (CD39/CD73) | **Plausible** (eff. ~10 min e-fold) |
| alpha | 10.0 | P2 receptor -> GRN modulation | May need adjustment with longer sims |
| lambda = sqrt(D_F/gamma_F) | ~70 | Communication range (lattice spacings) | Effectively well-mixed |

### 8.5 Overall Assessment

| Layer | Architecture | Default Params | Learned Params | Timescale | Grade |
|---|---|---|---|---|---|
| Ca2+ dynamics | Sound | Reasonable | k_ca too steep | Correct (anchor) | **B+** |
| Stress switch | Excellent | Well-designed | 5/11 at bounds, hierarchy inverted | S 17x faster than Ca2+ | **B-** (learned) / **B+** (default) |
| Inter-embryo field | Appropriate | Plausible | Plausible | ~10 min (plausible) | **B+** |
| Rescue modulation | Plausible | alpha context-dependent | -- | N/A | **B** |
| **Overall** | | | | | **B** |

The overall grade is reduced from B+ to B because the learned parameters (which are what the model actually uses in simulations) substantially deviate from biological plausibility. The architecture is sound but the optimizer exploited the multi-clock design and narrow tau_S bounds to collapse the reaction-diffusion spatial filter into a fast binary switch. Fixing the dt unification and relearning (Section 7.1, Option C) is the highest-priority improvement.

---

## Appendix A: Why gamma_system = gamma_F + C * D_F / L²

This appendix explains why the effective decay rate of a diffusing-and-decaying signal on a finite domain is the sum of two terms: bulk decay (gamma_F) and diffusive boundary loss (C * D_F / L²). The derivation uses only basic calculus concepts.

### A.1 The Setup

The full field equation is:

```
dF/dt = D_F * (spatial spreading) - gamma_F * F + emission
```

The signal decays everywhere at rate gamma_F (like a chemical being degraded by enzymes). It also spreads out by diffusion (D_F), and at the boundaries it escapes into the exterior where F = 0 (absorbing boundary conditions). The emission term is the source — each embryo adds signal at a rate proportional to its stress.

We want to find: **how fast does F respond to changes?** Specifically, what is the system's time constant?

### A.1.1 Why We Can Ignore the Emission Term

The emission term determines **what value F converges to** (the steady state), but not **how fast it gets there** (the time constant). Here's why:

The general solution of the full equation (with emission) is:

```
F(x, t) = F_steady(x) + F_transient(x, t)
```

where F_steady is the steady-state solution (where emission, decay, and diffusion are all in balance), and F_transient is the deviation from steady state. Substituting into the full equation:

```
dF_transient/dt = D_F * laplacian(F_steady + F_transient) - gamma_F * (F_steady + F_transient) + emission
```

Since F_steady already satisfies `0 = D_F * laplacian(F_steady) - gamma_F * F_steady + emission` (that's what "steady state" means), all the F_steady and emission terms cancel, leaving:

```
dF_transient/dt = D_F * laplacian(F_transient) - gamma_F * F_transient
```

The emission has vanished. The transient part obeys the **homogeneous** equation — meaning there is no external forcing term; the equation equals zero when F_transient is zero, and the system evolves purely from its own internal dynamics (decay and diffusion). In dynamical systems terminology, this is an **autonomous** system: its evolution depends only on its current state, not on any external input. The original equation with the emission term is **non-autonomous** (or **inhomogeneous**) because it is driven by an external source.

The intuition: the transient is the *deviation from equilibrium*. Nothing external pushes this deviation — the source is already fully accounted for in the steady state. The deviation simply relaxes back to zero under the system's own dynamics, and the rate of that relaxation is what we want to find.

This means the rate at which F approaches its steady state is entirely determined by the decay and diffusion terms — the emission is irrelevant to the timescale.

**Analogy:** Consider filling a bathtub. The faucet (emission) determines the final water level (steady state). The drain size (decay + diffusion) determines how fast the water level adjusts after you change the faucet. If you suddenly turn the faucet from low to high, the time it takes to reach the new level depends on the drain, not the faucet.

So we only need to analyze the homogeneous equation:

```
dF/dt = D_F * (spatial spreading) - gamma_F * F
```

### A.2 The Key Idea: Spatial Shapes That Preserve Themselves

Most initial patterns of F will change shape as they evolve — some parts decay faster than others, diffusion smooths out bumps, etc. But there are special spatial patterns that **keep their shape** as they decay. These are called **eigenmodes**.

An eigenmode shrinks uniformly over time, like a photograph fading — the spatial pattern stays the same, only the overall brightness decreases. If we can find these self-preserving shapes, the problem simplifies enormously: each eigenmode just fades at its own rate, and the slowest-fading one determines the effective lifetime of the signal.

### A.3 Finding the Eigenmodes (1D Example)

Consider a 1D domain of width W (positions 0 to W), with F = 0 at both ends.

We look for solutions where space and time separate:

```
F(x, t) = T(t) * phi(x)
```

Here phi(x) is the spatial shape, and T(t) is the overall amplitude.

#### Why look for solutions of this form?

We are not *assuming* the full solution has this product form — most solutions don't. Instead, we are searching for the special **self-preserving shapes** (eigenmodes) described in A.2. By definition, an eigenmode keeps its spatial shape over time — only its amplitude changes. That's exactly what `F(x,t) = T(t) * phi(x)` means: the shape phi(x) stays fixed while the amplitude T(t) grows or shrinks.

Once we find all the eigenmodes, we can express **any** initial pattern as a sum of them (this is guaranteed by a mathematical theorem — the eigenmodes form a "complete basis," much like any sound can be decomposed into a sum of pure tones). Each eigenmode then evolves independently, fading at its own rate. So the product-form ansatz is a tool for finding the building blocks, not a restriction on the solution.

Plugging into the equation:

```
T'(t) * phi(x) = D_F * T(t) * phi''(x) - gamma_F * T(t) * phi(x)
```

where phi''(x) means the second derivative of phi with respect to x (which measures how curved the shape is — this is what the Laplacian computes).

Dividing both sides by T(t) * phi(x):

```
T'(t) / T(t) = D_F * phi''(x) / phi(x) - gamma_F
```

The left side depends only on time. The right side depends only on space (apart from the constant gamma_F). A function of time can only equal a function of space if both sides equal the same constant. Call it -gamma:

```
T'(t) / T(t) = -gamma          ... (time equation)
D_F * phi''(x) / phi(x) = -gamma + gamma_F    ... (space equation)
```

### A.4 Solving the Time Equation

```
T'(t) = -gamma * T(t)
```

This is simple exponential decay:

```
T(t) = T(0) * exp(-gamma * t)
```

So gamma is the decay rate we're looking for. Now we need to find what values of gamma are allowed.

### A.5 Solving the Space Equation

Rearranging the space equation:

```
phi''(x) = -(gamma - gamma_F) / D_F * phi(x)
```

Let's define mu = (gamma - gamma_F) / D_F. Then:

```
phi''(x) = -mu * phi(x)
```

This equation says: "the second derivative of phi is proportional to negative phi." What function has this property? **Sine and cosine.** Specifically:

```
phi(x) = sin(sqrt(mu) * x)
```

(We exclude cosine because cos(0) = 1, which would violate our boundary condition F = 0 at x = 0.)

### A.6 Applying the Boundary Conditions

We need F = 0 at x = 0 and at x = W.

At x = 0: sin(0) = 0. This is automatically satisfied.

At x = W: sin(sqrt(mu) * W) = 0. This requires:

```
sqrt(mu) * W = n * pi       (where n = 1, 2, 3, ...)
```

So:

```
mu_n = n² * pi² / W²
```

### A.7 The Decay Rates

Recalling that mu = (gamma - gamma_F) / D_F, we get:

```
gamma_n = gamma_F + D_F * mu_n = gamma_F + D_F * n² * pi² / W²
```

Each integer n gives a different eigenmode with its own decay rate:

| Mode | Spatial shape | Decay rate |
|---|---|---|
| n = 1 | sin(pi*x/W) — one gentle arch | gamma_F + D_F * pi²/W² |
| n = 2 | sin(2*pi*x/W) — two arches | gamma_F + D_F * 4*pi²/W² |
| n = 3 | sin(3*pi*x/W) — three arches | gamma_F + D_F * 9*pi²/W² |

Higher modes (more wiggly shapes) decay faster because they have steeper gradients near the boundary, so diffusion carries signal out more quickly.

### A.8 The Slowest Mode Wins

Any initial pattern of F can be written as a sum of these eigenmodes (like decomposing a sound into its harmonics). Each mode decays independently at its own rate. As time passes, the faster modes die out first, and eventually only the **slowest mode (n = 1)** remains. This is why the system-level decay rate is:

```
gamma_system = gamma_1 = gamma_F + D_F * pi² / W²
```

In terms of the half-width L = W/2:

```
gamma_system = gamma_F + D_F * pi² / (4 * L²) = gamma_F + C * D_F / L²
```

where C = pi²/4 ≈ 2.5 for 1D.

### A.9 Extending to 2D

For a square domain W × W with absorbing BC on all sides, the eigenmodes are products of 1D sine functions:

```
phi(x, y) = sin(n * pi * x / W) * sin(m * pi * y / W)
```

The decay rate for mode (n, m) is:

```
gamma_{n,m} = gamma_F + D_F * (n² + m²) * pi² / W²
```

The slowest mode is (n=1, m=1):

```
gamma_system = gamma_F + D_F * 2 * pi² / W² = gamma_F + D_F * pi² / (2 * L²)
```

So C = pi²/2 ≈ 5 for 2D, which is the value used in the calibration tables.

### A.10 Physical Intuition

Why are the two terms additive? Because they represent **independent loss mechanisms** that act simultaneously:

- **gamma_F** removes signal everywhere in the bulk (like enzymes degrading a chemical throughout a solution)
- **C * D_F / L²** removes signal by transporting it to the boundary where it escapes (like a chemical diffusing out of a container through open sides)

Both processes happen at the same time, so the total loss rate is their sum — just as two drains in a bathtub empty it faster than either one alone.

The L² in the denominator explains why group size matters: a larger domain means a longer journey from center to boundary, so the diffusive loss rate decreases as L² — the signal persists longer in larger groups.

### A.11 Why This Matters for the Model

The formula tells us that for small embryo groups (small L), the C * D_F / L² term dominates and signal disappears quickly. For large groups (large L), this term shrinks and the bulk decay gamma_F takes over. This creates the group-size-dependent rescue effect: small groups cannot accumulate enough eATP to reach the rescue threshold, while large groups can.

---

## Appendix B: Converting Dimensionless Diffusion to Physical Units

The model uses a dimensionless diffusion coefficient D_F on a lattice with unit spacing and abstract time units. This appendix shows how to convert it to physical units (um^2/s).

### B.1 The Two Equations

The **continuous physical** diffusion equation is:

```
dF/dt_real = D_physical * d²F/dx²
```

where t_real is in seconds, x is in micrometers, and D_physical has units of um²/s.

Discretized on a grid with physical spacing delta_x (the real distance between neighboring embryos):

```
dF/dt_real = D_physical * (F_{i+1} + F_{i-1} - 2*F_i) / delta_x²
```

The **model** equation uses unit lattice spacing (distance between neighbors = 1) and abstract ODE time units:

```
dF/dt_ode = D_F * (F_{i+1} + F_{i-1} - 2*F_i)
```

### B.2 Matching the Two

The finite difference term `(F_{i+1} + F_{i-1} - 2*F_i)` is the same in both equations — it's just the values at neighboring grid points. What differs is the coefficient in front and the time units.

Let T be the real time (in seconds) per ODE time unit. Then `dt_real = T * dt_ode`, so:

```
dF/dt_real = (1/T) * dF/dt_ode = (D_F / T) * (F_{i+1} + F_{i-1} - 2*F_i)
```

Comparing with the physical discretization:

```
D_physical / delta_x² = D_F / T
```

Solving for D_physical:

```
D_physical = D_F * delta_x² / T
```

### B.3 Units Check

```
[D_physical] = [dimensionless] * [um²] / [s] = um²/s    ✓
```

### B.4 Application to the CEMA Model

For the inter-embryo field:
- D_F = 0.5 (dimensionless, from model)
- delta_x ~ 3 mm = 3000 um (typical Xenopus embryo spacing)
- T = 0.058 s (real time per ODE time unit, from the calibration 1 bio step = 0.058 s, and the field advances 1.0 ODE time units per bio step)

```
D_physical = 0.5 * (3000 um)² / 0.058 s = 0.5 * 9,000,000 / 0.058 ~ 78,000,000 um²/s
```

This is vastly higher than pure molecular diffusion of eATP in water (~300-400 um²/s). As discussed in Section 4, this reflects the coarse-grained nature of the lattice and the role of convective mixing in the culture medium, not literal molecular diffusion.

### B.5 The General Principle

Any dimensionless parameter in the model that has physical dimensions can be converted using the same approach: identify what physical scales (length, time) the model's abstract units correspond to, then multiply by the appropriate powers:

| Model quantity | Physical units | Conversion |
|---|---|---|
| D_F (diffusion) | um²/s | D_F * delta_x² / T |
| gamma_F (decay rate) | 1/s | gamma_F / T |
| emission rate | concentration/s | emission * C_scale / T |

where delta_x is the physical lattice spacing, T is real seconds per ODE time unit, and C_scale converts dimensionless concentration to physical units (e.g., uM).
