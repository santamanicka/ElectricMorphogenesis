# Timescale Hierarchy in Facial Integration Model

## Overview

The refined facial integration model implements a **3-tier timescale separation** that mirrors biological reality in embryonic development. This guide explains how timescales are implemented and how to interpret simulation dynamics.

---

## The Three Timescales

### 1. **Bioelectric Dynamics** (FASTEST)
**Physical Timescale**: Milliseconds to seconds
**Simulation Timestep**: `dt = 0.01` (10 ms)
**Iterations per cycle**: 100 steps = 1 second of simulated time

#### Implementation Location
- **File**: `bioelectricTransduction.py`
- **Key Parameters**:
  ```python
  self.tau_ca = 1.0              # Ca²⁺ decay timescale (1 second)
  self.alpha_lowpass = 0.8       # Voltage smoothing (0.2 weight on new values)
  ```

#### Physical Processes
1. **Voltage-gated Ca²⁺ channels** open/close in response to membrane voltage changes
2. **Ca²⁺ influx** through channels (fast response: < 100 ms)
3. **Ca²⁺ decay** via pumps and buffers (tau = 1 second)
4. **Voltage filtering** to smooth rapid fluctuations

#### Update Equation
```python
# bioelectricTransduction.py line 90-92
dCa_dt = I_ca - Ca / tau_ca
Ca = Ca + dt * dCa_dt  # dt = 0.01 seconds
```

#### Biological Meaning
- Bioelectric signals provide **rapid, dynamic permissiveness** for gene activation
- Ca²⁺ acts as a **temporal integrator** of voltage signals
- Patterns stabilize within seconds, providing a stable backdrop for slower processes

---

### 2. **Morphogen Dynamics** (INTERMEDIATE)
**Physical Timescale**: Minutes to hours
**Simulation Timestep**: `dt = 0.01` (same units as bioelectric, but effective timescale differs)
**Iterations per cycle**: 500 steps = 5 seconds of simulated time

#### Implementation Location
- **File**: `refinedFacialGRN.py` - `update_morphogens()` method
- **Key Parameters**:
  ```python
  self.timestep = 0.01                     # Integration timestep
  diffusion_rate = 0.1                     # Spatial diffusion coefficient
  degradation_rate = 0.05                  # Baseline degradation (20s half-life)
  fgf8_degradation_factor = 10.0           # FGF8 degrades 10× faster (2s half-life)
  ```

#### Physical Processes
1. **Secretion** from spatial sources (midline, lateral, posterior)
2. **Diffusion** across tissue (Laplacian operator)
3. **Degradation** (first-order kinetics)
4. **Mutual inhibition** (SHH ↔ FGF8 cross-repression)

#### Update Equations
```python
# refinedFacialGRN.py lines 257-263 (SHH example)
dshh_dt = (shh_secretion +
           D * laplacian(shh) -
           k_deg * shh * (1 - shh_inhibition) +
           inhibition_strength * shh * (1 - fgf8))
shh = shh + dt * dshh_dt
```

#### Biological Meaning
- Morphogens form **spatial gradients** that pattern the tissue
- Gradients stabilize over minutes-to-hours timescale
- Faster than gene expression but slower than bioelectric changes
- **Pre-equilibration**: 1000 steps run before genes to establish steady gradients

---

### 3. **Gene Expression Dynamics** (SLOWEST)
**Physical Timescale**: Hours to days
**Simulation Timestep**: `dt = 0.01` (same integration step)
**Iterations per cycle**: 500 steps (same as morphogens, but genes respond slower due to rate constants)

#### Implementation Location
- **File**: `refinedFacialGRN.py` - `update_genes()` method
- **Key Parameters**:
  ```python
  k_activation = 0.10      # Gene activation rate (10 seconds to reach 63% of target)
  k_degradation = 0.01     # Gene degradation rate (100 seconds half-life)
  ```

#### Physical Processes
1. **Transcription** (DNA → mRNA) gated by morphogens AND bioelectric signals
2. **Translation** (mRNA → protein)
3. **Protein degradation** (proteasome, dilution from growth)
4. **Gene cascades** (e.g., rx → six3 → pax6 → lhx2)

#### Update Equations
```python
# refinedFacialGRN.py lines 207-210 (bioelectric-gated mode)
initiation = AND(morph_signal, bio_gate)
maintenance = hill(current_gene, K_self, n_self)
activation = OR(w_init * initiation, w_maint * maintenance)
dgene_dt = k_activation * activation - k_degradation * current_gene
```

#### Biological Meaning
- Genes integrate **both** morphogen and bioelectric inputs
- Activation requires BOTH inputs to be high (AND logic)
- Once activated, genes can self-maintain (OR logic)
- Slowest timescale provides **long-term memory** of patterning decisions

---

## Timescale Separation in Practice

### Iteration Counts (from `run_refined_facial_integration.py`)

```python
# Per developmental cycle:
bio_steps = 100     # 100 × 0.01s = 1 second bioelectric time
grn_steps = 500     # 500 × 0.01s = 5 seconds morphogen/gene time

# Typical simulation:
num_cycles = 5      # Total: 5 cycles
```

### Effective Time Ratios

| Process | Steps/Cycle | Simulated Time | Relative Speed |
|---------|-------------|----------------|----------------|
| **Bioelectric** | 100 | 1 second | 1× (baseline) |
| **Morphogen** | 500 | 5 seconds | 5× slower |
| **Gene Expression** | 500 | 5 seconds | 5× slower (but with slower rate constants) |

### Why This Hierarchy Matters

#### 1. **Bioelectric Pre-Pattern** (Cycle 0)
- First, run bioelectric simulation alone to establish spatial voltage pattern
- Stigmergic field feedback creates **self-organized domains**
- Ca²⁺ integrates voltage over ~1 second
- **Result**: Stable bioelectric "scaffold" for morphogen patterning

#### 2. **Morphogen Equilibration** (Pre-GRN)
```python
# refinedFacialGRN.py - before gene dynamics
for _ in range(1000):
    grn.update_morphogens()
```
- Run morphogen dynamics **without genes** for 1000 steps (10 seconds)
- Allows SHH, FGF8, EDN1 gradients to reach steady state
- Mutual inhibition creates sharp complementary domains
- **Result**: Stable spatial gradients ready to activate genes

#### 3. **Coupled Dynamics** (Main Simulation)
```python
for grn_step in range(500):
    grn.update_morphogens()        # Update gradients
    grn.update_genes(bio_signals)  # Update genes (both inputs)
```
- Morphogens and genes co-evolve
- Bioelectric signals remain quasi-static (updated between cycles)
- Genes integrate morphogen × bioelectric over hours-days timescale

---

## Interpreting Simulation Outputs

### Time Unit Conversion

**Internal timestep**: `dt = 0.01` (unitless, but calibrated to ~10ms bioelectric time)

**To convert to biological time**:

| Process | Formula | Example (500 steps) |
|---------|---------|---------------------|
| **Bioelectric** | `steps × 0.01 seconds` | 5 seconds |
| **Morphogen** | `steps × 0.1 seconds` | 50 seconds (~1 min) |
| **Gene** | `steps × 1.0 seconds` | 500 seconds (~8 min) |

These are **effective biological timescales** based on rate constants, not absolute time.

### Key Diagnostic Metrics

#### 1. **Ca²⁺ Temporal Integration**
```python
# Check how fast Ca²⁺ responds to voltage changes
diagnostics = transduction.get_diagnostics()
ca_history = diagnostics['Ca_mean_history']  # Track mean Ca²⁺ over time
```

**Expected behavior**:
- Ca²⁺ rises over ~10-100 steps (0.1-1 second) after voltage depolarization
- Decays with tau = 1 second when voltage returns to baseline

#### 2. **Morphogen Gradient Formation**
```python
# Check when gradients stabilize
shh = grn.grid['shh']
fgf8 = grn.grid['fgf8']
```

**Expected behavior**:
- Initial transient (steps 0-200): Gradients build from sources
- Quasi-steady state (steps 200-1000): Gradients fluctuate around stable pattern
- Mutual inhibition creates complementary SHH (midline) vs FGF8 (lateral) domains

#### 3. **Gene Expression Dynamics**
```python
# Check gene activation timecourse
pax6 = grn.grid['pax6']
```

**Expected behavior**:
- Lag phase (steps 0-50): Genes accumulate slowly from zero
- Growth phase (steps 50-300): Exponential rise toward target
- Saturation (steps 300+): Approach steady state with characteristic time = 1/k_activation

---

## Parameter Sensitivity and Tuning

### Critical Timescale Parameters

#### 1. **Ca²⁺ Decay Time** (`tau_ca`)
**Default**: 1.0 seconds
**Effect**: Controls how long Ca²⁺ "remembers" voltage changes

- **Lower** (0.1s): Fast decay → Ca²⁺ tracks instantaneous voltage → noisy
- **Higher** (10s): Slow decay → Ca²⁺ averages over long time → sluggish

**Tuning guideline**: Should be ~10× slower than voltage fluctuations but ~10× faster than morphogen changes

#### 2. **Morphogen Degradation** (`degradation_rate`, `fgf8_degradation_factor`)
**Defaults**: 0.05 (baseline), 10.0 (FGF8 multiplier)
**Effect**: Controls gradient sharpness and response time

- **Lower degradation**: Broader gradients, slower to equilibrate
- **Higher degradation**: Sharper gradients, faster equilibration

**Why FGF8 is special**: 10× faster degradation (+ 10× slower diffusion) creates **sharp lateral peaks** complementary to SHH midline

#### 3. **Gene Activation Rate** (`k_activation`)
**Default**: 0.10 (10 second timescale)
**Effect**: How fast genes respond to morphogen/bioelectric inputs

- **Lower** (0.01): Genes change slowly → more stable, less responsive
- **Higher** (1.0): Genes track inputs rapidly → more dynamic, less memory

**Tuning guideline**: Should be slower than morphogen equilibration to avoid chasing transients

#### 4. **Gene Degradation Rate** (`k_degradation`)
**Default**: 0.01 (100 second timescale)
**Effect**: Baseline gene turnover

- **Higher degradation**: Genes decay faster when input removed → less memory
- **Lower degradation**: Genes persist longer → more hysteresis

**Typical ratio**: `k_activation / k_degradation ≈ 10` for reasonable Hill-like response

---

## Common Pitfalls and Solutions

### Problem 1: Genes Don't Activate
**Symptom**: All gene expression remains near zero after 500 steps

**Possible Causes**:
1. **Morphogens not equilibrated** → Run more pre-equilibration steps (increase from 1000 to 2000)
2. **AND gate too strict** → Lower `and_threshold` (try 1.0 instead of 1.25)
3. **Ca²⁺ gate too strict** → Lower `ca_threshold_percentile` (try 0.30 instead of 0.45)
4. **Activation too slow** → Increase `k_activation` (try 0.20 instead of 0.10)

### Problem 2: Genes Saturate Immediately
**Symptom**: All genes reach 1.0 within 50 steps

**Possible Causes**:
1. **AND gate too permissive** → Raise `and_threshold` or lower `and_sharpness`
2. **Activation too fast** → Decrease `k_activation`
3. **No bioelectric constraint** → Check that `bio_gate` has spatial variation

### Problem 3: Morphogen Gradients Don't Form
**Symptom**: SHH and FGF8 look similar or uniform

**Possible Causes**:
1. **Insufficient mutual inhibition** → Increase `inhibition_strength` (try 1.0 instead of 0.8)
2. **Diffusion too high** → Decrease `diffusion_rate`
3. **Sources too weak** → Increase `shh_strength` or `fgf8_strength`
4. **Need more steps** → Run longer pre-equilibration (2000+ steps)

### Problem 4: Ca²⁺ Doesn't Track Voltage
**Symptom**: Ca²⁺ remains uniform despite voltage patterns

**Possible Causes**:
1. **Voltage range wrong** → Check that voltage is in mV (e.g., -60mV = -0.06 V)
2. **V_half_ca mismatched** → Adjust to match voltage distribution (default: -40mV = -0.04)
3. **Tau too short** → Ca²⁺ decays before accumulating (increase `tau_ca`)

---

## Recommended Workflow for New Simulations

### Step 1: Establish Bioelectric Pattern (Fast)
```python
# Run bioelectric simulation alone
for bio_step in range(100):
    bio_model.simulate(numSimIters=1)
    transduction.update(vmem_grid, dt=0.01)
```

**Check**: Ca²⁺ should have clear spatial structure (not uniform)

### Step 2: Equilibrate Morphogens (Intermediate)
```python
# Pre-equilibrate without genes
for _ in range(1000):
    grn.update_morphogens()
```

**Check**: SHH high at midline, FGF8 high laterally, EDN1 gradient A→P

### Step 3: Run Coupled Dynamics (Slow)
```python
# Co-evolve morphogens + genes
for _ in range(500):
    grn.update_morphogens()
    grn.update_genes(bioelectric_signals=bio_signals)
```

**Check**: Genes activate in expected regions (eye lateral, nose midline, mouth posterior)

### Step 4: Validate Timescale Separation
```python
# Plot timecourses
import matplotlib.pyplot as plt

plt.plot(ca_history, label='Ca²⁺ (fast)')
plt.plot(shh_history, label='SHH (intermediate)')
plt.plot(pax6_history, label='Pax6 (slow)')
plt.legend()
plt.xlabel('Simulation steps')
plt.ylabel('Normalized concentration')
```

**Expected**: Ca²⁺ fluctuates rapidly, SHH changes moderately, Pax6 rises slowly

---

## Summary: Why Timescale Hierarchy Exists

### Biological Justification
1. **Ion channels** (bioelectric) operate on millisecond timescales
2. **Protein diffusion** (morphogens) operates on minute-to-hour timescales
3. **Transcription/translation** (genes) operates on hour-to-day timescales

This separation is **fundamental to development**: fast signals provide dynamic context, slow signals provide stable memory.

### Computational Advantages
1. **Stability**: Slow processes don't chase fast noise
2. **Efficiency**: Can run fast processes at higher resolution without slowing entire simulation
3. **Modularity**: Can pre-equilibrate subsystems independently
4. **Interpretability**: Clear cause-effect relationships across timescales

### Model Design Principle

> **Bioelectric signals** provide spatial *permissiveness* (fast gates)
> **Morphogen gradients** provide spatial *instructions* (slow fields)
> **Gene expression** integrates both via AND logic (slowest memory)

This architecture ensures:
- **Robust patterning**: Requires agreement between fast and slow signals
- **Spatial precision**: Morphogens define position, bioelectric refines boundaries
- **Temporal stability**: Genes don't fluctuate with transient bioelectric noise

---

## Advanced: Operator Splitting and Quasi-Static Forcing

The timescale separation enables **operator splitting** - updating different subsystems sequentially rather than simultaneously. This is NOT a true adiabatic approximation, but rather a computational strategy that exploits timescale differences.

### What is Operator Splitting?

**Operator splitting** divides the full system dynamics into separate update steps:

1. **Update bioelectric** → establish spatial pattern
2. **Freeze bioelectric** → use as static forcing for GRN
3. **Update morphogen + gene** → evolve under fixed bioelectric input
4. **Repeat cycle** → update bioelectric based on new GRN state

This differs from a **true adiabatic approximation**, which would require slow variables to remain constant while fast variables equilibrate to quasi-steady-state.

### Quasi-Static Forcing (Not Adiabatic Approximation)

Between developmental cycles, **bioelectric patterns are treated as constant forcing** while morphogens and genes evolve:

```python
# Bioelectric updates every cycle
for cycle in range(num_cycles):
    # Update bioelectric (100 steps)
    for bio_step in range(bio_steps):
        transduction.update(vmem, dt=0.01)

    # Snapshot bioelectric state as quasi-static forcing
    bio_signals = transduction.get_gene_modulation_signals()  # Fixed for this cycle

    # Evolve morphogen+gene under constant bioelectric forcing (500 steps)
    for grn_step in range(grn_steps):
        grn.update(bioelectric_signals=bio_signals)  # Bioelectric frozen
```

**Why this works**:
- Bioelectric pattern stabilizes in ~100 steps (stigmergic field creates stable domains)
- GRN dynamics occur over 500 steps under quasi-constant bioelectric input
- Bioelectric changes slowly enough between cycles that GRN sees approximately constant forcing

**Why this is NOT adiabatic approximation**:
- In standard adiabatic approximation, SLOW variables are constant while FAST variables equilibrate
- Here, we freeze a FAST variable (bioelectric) to use as forcing for SLOWER variables (morphogen/gene)
- This is better described as **quasi-static forcing** or **operator splitting**

### Context-Dependent "Fast" vs "Slow"

The confusion arises because "fast" and "slow" are context-dependent:

| Perspective | Fast Variables | Slow Variables | Approximation |
|-------------|---------------|----------------|---------------|
| **Physical timescales** | Bioelectric (ms-s) | Morphogen (min-hr), Gene (hr-days) | — |
| **Within GRN dynamics** | Morphogen (equilibrates in ~1000 steps) | Gene (accumulates over 500+ steps) | Quasi-steady morphogen |
| **Across cycles** | Bioelectric (updated every cycle) | GRN (evolves within cycle) | Quasi-static forcing |

**Key insight**: Bioelectric is physically FAST but acts as SLOW forcing from the GRN's perspective (updated infrequently, held constant during GRN evolution).

### Quasi-Steady Morphogen Assumption (True Separation of Scales)

When analyzing gene dynamics, can assume **morphogens are at quasi-steady-state**:

```python
# Pre-equilibrate morphogens before gene dynamics
for _ in range(1000):
    grn.update_morphogens()  # No gene feedback yet
```

**Validity**:
- Morphogen equilibration time (~1000 steps) << gene activation time (~5000 steps to saturation)
- Morphogens reach ~90% steady state before genes significantly accumulate
- This IS a proper separation of scales: fast (morphogen) equilibrates to slow (gene)

**This approximation is closer to adiabatic**: Morphogens (fast) equilibrate while genes (slow) remain approximately constant at zero.

---

## References to Code

| Concept | File | Lines |
|---------|------|-------|
| Ca²⁺ timescale (`tau_ca`) | `bioelectricTransduction.py` | 38 |
| Morphogen timestep | `refinedFacialGRN.py` | 87 |
| Gene rate constants | `refinedFacialGRN.py` | 66-67 |
| Simulation loop structure | `run_refined_facial_integration.py` | 135-165 |
| Pre-equilibration | `learnRefinedFacialIntegration.py` | 328-330 |
| Operator splitting (quasi-static bioelectric) | `run_refined_facial_integration.py` | 162 |

---

## Conclusion

The timescale hierarchy is implemented through:
1. **Different rate constants** (tau_ca vs k_activation vs k_degradation)
2. **Separate iteration loops** (bio_steps vs grn_steps)
3. **Pre-equilibration** (morphogens before genes)
4. **Operator splitting** (update subsystems sequentially, using quasi-static forcing)

**Key terminology clarification**:
- The model uses **operator splitting** and **quasi-static forcing**, NOT true adiabatic approximation
- Bioelectric acts as slowly-varying external forcing from the GRN's perspective
- Morphogen pre-equilibration IS a proper separation of scales (fast morphogen equilibrates to slow gene)

**Interpretation**: Simulation time is **not absolute** - each process has its own effective timescale determined by rate constants. The hierarchy ensures bioelectric patterns guide morphogen gradients, which in turn activate genes in a stable, robust manner.