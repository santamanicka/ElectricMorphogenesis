# Concurrent Bioelectric-Biochemical Dynamics Design

## Overview

This document describes the design for **concurrent bioelectric and biochemical dynamics** where all processes run simultaneously with proper timescale separation, replacing the current sequential (pre-compute then use) architecture.

**Key Principle**: Timescale separation achieved through **tau parameters**, not different dt values or sequential execution.

---

## Current Architecture (Sequential)

```python
# Phase 1: Pre-compute bioelectric pattern
bio_model.simulate(numSimIters=1000)  # Run to convergence
vmem_snapshot = bio_model.Vmem        # Extract final state

# Phase 2: Use static snapshot for GRN
for grn_step in range(5000):
    Ca_gate = compute_gate(vmem_snapshot)  # Fixed voltage
    grn.update(Ca_gate)
```

**Problems:**
- ❌ Bioelectric and biochemical processes artificially separated
- ❌ No ongoing interaction during GRN evolution
- ❌ Bioelectric pattern is "frozen" rather than dynamic
- ❌ Not biologically realistic

---

## New Architecture (Concurrent)

```python
# All processes run simultaneously with different timescales
dt = 0.01  # Same timestep for all (numerical stability)

for t in range(total_timesteps):  # e.g., 0-3000

    # 1. FAST: Bioelectric dynamics (tau_vmem ~ 0.1)
    bio_model.step(dt=dt)
    vmem = bio_model.electricNetwork.Vmem

    # 2. INTERMEDIATE: Ca²⁺ transduction (tau_ca = 20.0)
    bio_signals = transduction.update(vmem, dt=dt)
    # Ca²⁺ integrates voltage over ~20 time units
    # Pattern formed at t=1000 persists at t=2000 via slow decay

    # 3. SLOW: Morphogen + Gene dynamics (tau_morph ~ 20, tau_gene ~ 50)
    Ca_gate = compute_gate(bio_signals['Ca'])
    grn.update(Ca_gate, dt=dt)
    # Morphogens and genes change slowly due to large tau values
```

**Advantages:**
- ✅ All processes concurrent (biologically realistic)
- ✅ Temporal gating emerges from Ca²⁺ memory (tau_ca >> tau_vmem)
- ✅ Pattern persistence without artificial "freezing"
- ✅ Numerical stability (all use same dt=0.01)

---

## Key Design Principle: Timescale Separation via Tau

### **CRITICAL INSIGHT**

**Timescale separation comes from tau values in the differential equations, NOT from different dt or update frequencies.**

```python
# Ca²⁺ dynamics (intermediate timescale)
dCa/dt = I_ca - Ca / tau_ca    # tau_ca = 20.0
Ca = Ca + dt * dCa_dt           # dt = 0.01

# Gene dynamics (slow timescale)
dgene/dt = k_on * activation - k_off * gene
# Equivalent to: dgene/dt = (production - gene) / tau_gene
# where tau_gene = 1/k_off = 1/0.02 = 50.0
```

**With same dt=0.01:**
- Ca²⁺ changes by ~0.05% per step (slow)
- Gene changes by ~0.02% per step (very slow)
- Timescale ratio maintained: tau_gene / tau_ca = 50/20 = 2.5×

### Why NOT Use Different dt Values?

**Question:** Why not update GRN with dt_grn=0.1 every 10 steps?

**Answer:** Two reasons:

1. **Equivalence**: Updating every 10 steps with dt=0.1 gives the same total time advance as updating every step with dt=0.01
   - 10 steps × 0 updates + 1 × 0.1 = 0.1 time units
   - 10 steps × 0.01 = 0.1 time units
   - No mathematical difference!

2. **Numerical Stability**: For explicit Euler integration, stability requires `dt < tau / C` where C is stability factor (~2-5)
   - With tau_gene = 50, safe dt ~ 0.01-0.1
   - Using dt=0.1 is at the edge of stability
   - Can cause oscillations, negative values, or blow-up

**Conclusion:** Use **same dt=0.01 for everything**. Timescale separation emerges from tau values, not dt.

---

## Temporal Gating Mechanism

### How Early Bioelectric Pattern Gates Later Gene Expression

**Timeline:**
1. **t = 0-1000**: Bioelectric pattern forms via stigmergic field dynamics
   - Vmem stabilizes into spatial domains (~-60mV with variations)
   - Ca²⁺ accumulates via voltage-gated channels
   - With tau_ca=20, Ca²⁺ reaches ~90% of steady state by t=100

2. **t = 1000**: Bioelectric pattern established
   - Vmem has spatial structure (depolarized/hyperpolarized regions)
   - Ca²⁺ reflects this pattern with ~20 time unit memory

3. **t = 1000-2000**: Ca²⁺ memory persists while genes activate
   - Ca²⁺ decays slowly: `Ca(t) = Ca(1000) * exp(-(t-1000)/tau_ca)`
   - At t=2000: `Ca(2000) ≈ Ca(1000) * exp(-1000*0.01/20) = Ca(1000) * exp(-0.5) ≈ 0.6 * Ca(1000)`
   - **Ca²⁺ still retains 60% of pattern from t=1000!**
   - This "frozen" Ca²⁺ pattern gates gene activation

4. **t = 2000+**: Genes express in gated regions
   - Gene activation requires BOTH morphogen AND Ca²⁺ gate
   - Ca²⁺ gate reflects early bioelectric pattern
   - Spatial features emerge from this dual gating

### Mathematical Formulation

**Ca²⁺ dynamics:**
```python
dCa/dt = I_ca(Vmem) - Ca / tau_ca

# I_ca depends on Vmem(t), which changes rapidly
# But Ca integrates over time window ~ tau_ca
# Result: Ca is "low-pass filtered" version of Vmem
```

**Gene gating:**
```python
activation = AND(morphogen_signal, Ca_gate)
dgene/dt = k_on * activation - k_off * gene

# At t=2000:
# - morphogen_signal: current morphogen gradient
# - Ca_gate: reflects Vmem pattern from ~20 time units ago (t≈1980)
# - If Vmem stabilized at t=1000, Ca still "remembers" that pattern
```

**Key insight:** tau_ca >> tau_vmem creates temporal memory without artificial freezing.

---

## Implementation Parameters

### Timescale Hierarchy

| Component | Tau Parameter | Value | Physical Meaning |
|-----------|--------------|-------|------------------|
| **Vmem** | tau_vmem | ~0.1 | Membrane time constant (capacitance/conductance) |
| **Ca²⁺** | tau_ca | **20.0** | Ca²⁺ buffering + extrusion (INCREASED from 1.0) |
| **Morphogen** | 1/k_deg | ~20.0 | Morphogen degradation half-life |
| **Gene** | 1/k_off | ~50.0 | mRNA/protein degradation half-life |

**Ratios:**
- tau_ca / tau_vmem = 20 / 0.1 = 200× (Ca²⁺ much slower than voltage)
- tau_gene / tau_ca = 50 / 20 = 2.5× (genes slower than Ca²⁺)
- tau_gene / tau_vmem = 50 / 0.1 = 500× (genes much slower than voltage)

### Key Parameter Changes

**File: `bioelectricTransduction.py`, line 38**
```python
# OLD
self.tau_ca = torch.tensor(1.0, device=device, dtype=dtype)

# NEW
self.tau_ca = torch.tensor(20.0, device=device, dtype=dtype)
```

**Rationale:** Increase Ca²⁺ memory timescale to bridge bioelectric → gene gap.

---

## Simulation Loop Structure

### Option A: Simple Concurrent Loop (Recommended)

```python
def run_concurrent_integration(bio_model, transduction, grn, num_timesteps=3000):
    """
    All components update every timestep with same dt=0.01.
    Timescale separation via tau parameters.
    """
    dt = 0.01

    for t in range(num_timesteps):
        # 1. Update bioelectric (fast: tau ~ 0.1)
        bio_model.step(dt=dt)
        vmem = bio_model.electricNetwork.Vmem

        # 2. Update Ca²⁺ transduction (intermediate: tau = 20.0)
        bio_signals = transduction.update(vmem, dt=dt)

        # 3. Update GRN (slow: tau ~ 20-50)
        Ca_gate = compute_gate(bio_signals['Ca'])
        grn.update(Ca_gate, dt=dt)

        # Optional: Record diagnostics
        if t % 100 == 0:
            record_state(t, vmem, bio_signals['Ca'], grn.grid)
```

**Advantages:**
- ✅ Simple implementation
- ✅ Numerically stable (small dt throughout)
- ✅ True concurrent dynamics

**Disadvantages:**
- ⚠️ Computationally expensive (updates GRN every step)

### Option B: Optimized with Substepping (If Needed)

```python
def run_concurrent_integration_optimized(bio_model, transduction, grn, num_timesteps=3000):
    """
    Update GRN less frequently for computational savings.
    Must verify numerical stability.
    """
    dt_bio = 0.01
    dt_grn = 0.05  # 5× larger, but check stability!

    for t in range(num_timesteps):
        # Always update bioelectric and Ca²⁺
        bio_model.step(dt=dt_bio)
        vmem = bio_model.electricNetwork.Vmem
        bio_signals = transduction.update(vmem, dt=dt_bio)

        # Update GRN less frequently
        if t % 5 == 0:
            Ca_gate = compute_gate(bio_signals['Ca'])
            grn.update(Ca_gate, dt=dt_grn)
```

**Advantages:**
- ✅ Computational savings (80% fewer GRN updates)

**Disadvantages:**
- ⚠️ Must verify stability with larger dt_grn
- ⚠️ More complex logic

**Recommendation:** Start with Option A. Optimize later only if needed.

---

## Biological Realism

### Why Ca²⁺ Memory?

**Biological mechanisms:**
1. **Ca²⁺ buffering proteins** (calmodulin, calbindin, etc.) slow Ca²⁺ dynamics
2. **Ca²⁺ pumps** (PMCA, SERCA) have finite capacity → slow extrusion
3. **Ca²⁺-dependent kinases** (CaMKII) can maintain active state after Ca²⁺ drops

**Result:** Intracellular Ca²⁺ has effective tau ~ 100ms to 10s depending on buffering capacity.

**In development:**
- Bioelectric prepatterns form on seconds-minutes timescale
- Gene expression occurs on hours-days timescale
- Ca²⁺ bridges this gap: integrates fast signals, persists during slow processes

### Alternative: CaMKII Bistability (Future Extension)

For even longer memory, can add bistable kinase:

```python
# In BioelectricTransduction class:
self.CaMKII_active = torch.zeros(grid_size, grid_size)

def update(self, vmem, dt):
    # Ca²⁺ dynamics (fast)
    dCa_dt = I_ca - Ca / tau_ca
    Ca = Ca + dt * dCa_dt

    # CaMKII bistable switch (slow, persistent)
    activation = sigmoid((Ca - Ca_thresh) / Ca_sens)
    dCaMKII_dt = activation * (1 - CaMKII) - k_inact * CaMKII
    CaMKII = CaMKII + dt * dCaMKII_dt

    return {'Ca': Ca, 'gate': CaMKII}  # Use CaMKII for gating
```

**Advantages:**
- ✅ Bistability: once ON, stays ON even if Ca²⁺ drops
- ✅ Highly realistic molecular memory mechanism

**Disadvantages:**
- ⚠️ Additional state variable and equations
- ⚠️ More parameters to tune

**Recommendation:** Start with simple Ca²⁺ memory (tau_ca=20). Add CaMKII if pattern doesn't persist long enough.

---

## Expected Behavior

### Timeline of Spatial Pattern Formation

**Phase 1: Bioelectric Patterning (t = 0-1000)**
- Stigmergic field dynamics create spatial Vmem domains
- Ca²⁺ accumulates in depolarized regions
- By t=1000: Both Vmem and Ca²⁺ have stable spatial structure

**Phase 2: Morphogen Equilibration (t = 1000-2000)**
- Morphogen gradients diffuse and reach quasi-steady state
- Ca²⁺ decays slowly but retains spatial pattern
- Vmem may fluctuate, but Ca²⁺ filters out rapid changes

**Phase 3: Gene Activation (t = 2000-3000)**
- Genes activate where BOTH morphogen AND Ca²⁺ are high
- Ca²⁺ gate reflects pattern from t~1000 (due to slow decay)
- Spatial features emerge from dual gating

### Diagnostic Checks

**1. Ca²⁺ Memory Persistence:**
```python
# At t=1000
Ca_1000 = transduction.Ca.clone()

# At t=2000
Ca_2000 = transduction.Ca.clone()

# Check retention
correlation = torch.corrcoef(torch.stack([Ca_1000.flatten(), Ca_2000.flatten()]))[0,1]
print(f"Ca²⁺ pattern correlation (t=1000 vs t=2000): {correlation:.3f}")
# Should be > 0.6 if tau_ca=20
```

**2. Timescale Separation:**
```python
# Plot timecourses
plt.plot(vmem_history, label='Vmem (fast)', alpha=0.3)
plt.plot(ca_history, label='Ca²⁺ (intermediate)')
plt.plot(gene_history, label='Gene (slow)')
# Should see: Vmem fluctuates rapidly, Ca²⁺ smooth, Gene very smooth
```

**3. Spatial Gating:**
```python
# Check that features only appear where Ca²⁺ gate is high
feature_map = classifier.classify(grn.grid)
ca_gate = compute_gate(transduction.Ca)

overlap = (feature_map > 0) & (ca_gate > 0.5)
print(f"Feature-gate overlap: {overlap.sum() / (feature_map > 0).sum():.2%}")
# Should be > 80%
```

---

## Implementation Checklist

- [ ] Increase `tau_ca` from 1.0 to 20.0 in `bioelectricTransduction.py`
- [ ] Create new simulation script `run_concurrent_facial_integration.py`
- [ ] Implement concurrent loop with same dt=0.01 for all components
- [ ] Add diagnostics to track Ca²⁺ pattern persistence
- [ ] Verify temporal gating: Ca²⁺ at t=2000 correlates with pattern at t=1000
- [ ] Compare results to current sequential approach
- [ ] Optional: Add CaMKII bistability if needed
- [ ] Optional: Optimize with substepping if computational cost is high

---

## Expected Modifications to Existing Code

### Minimal Changes Required

**1. `bioelectricTransduction.py` (1 line change)**
```python
# Line 38
self.tau_ca = torch.tensor(20.0, device=device, dtype=dtype)  # Was 1.0
```

**2. New simulation script: `run_concurrent_facial_integration.py`**
- Clone `run_refined_facial_integration.py`
- Replace sequential phases with concurrent loop
- Add Ca²⁺ persistence diagnostics

**3. Optional: Add getter method to `BioelectricTransduction`**
```python
def get_ca_for_gating(self, percentile=45):
    """Compute Ca²⁺ gate from current state"""
    threshold = torch.quantile(self.Ca, percentile / 100.0)
    return (self.Ca > threshold).float()
```

### No Changes Needed

- ✅ `refinedFacialGRN.py` - gene/morphogen dynamics unchanged
- ✅ `cellularFieldNetwork.py` - bioelectric model unchanged
- ✅ `geneBasedFeatureClassifier.py` - feature classification unchanged

---

## Testing Strategy

### Unit Tests

1. **Ca²⁺ decay timescale**
   - Set Vmem to constant value
   - Check Ca²⁺ reaches steady state with time constant ~tau_ca
   - Verify exponential decay after Vmem drops

2. **GRN numerical stability**
   - Run GRN with dt=0.01 for 5000 steps
   - Check for oscillations, negative values, or blow-up
   - If unstable, reduce tau values or implement implicit integration

3. **Pattern persistence**
   - Establish Ca²⁺ pattern at t=1000
   - Turn off Vmem input (set to baseline)
   - Verify Ca²⁺ decays with tau=20 (not faster)

### Integration Tests

1. **Compare to sequential approach**
   - Run both sequential and concurrent simulations
   - Check if final feature patterns are similar
   - Concurrent should show more dynamic intermediate states

2. **Timescale hierarchy validation**
   - Plot Ca²⁺ vs gene timecourses
   - Verify Ca²⁺ equilibrates in ~100 steps
   - Verify genes accumulate over ~5000 steps

3. **Biological realism check**
   - At t=2000, check Ca²⁺ correlation with t=1000 pattern
   - Should be > 0.6 for tau_ca=20
   - If too low, increase tau_ca; if too high (>0.95), decrease tau_ca

---

## Critical Assumption: Does Vmem Pattern Persist After t=1000?

### The Key Question

**The concurrent design assumes:**
- Bioelectric pattern forms and **stabilizes** by t=1000
- Ca²⁺ integrates this stable pattern
- Ca²⁺ memory persists (tau_ca=20) while genes activate at t=2000

**But what if the Vmem pattern doesn't persist after t=1000?**

This section analyzes when the simple Ca²⁺ memory design works, and when we need more robust mechanisms.

---

### Scenario Analysis

#### Scenario 1: Vmem Fluctuates Randomly After t=1000

```python
# t < 1000: Vmem has spatial structure (eye/nose/jaw regions)
# t > 1000: Vmem becomes noisy/random (no stable pattern)
```

**What happens to Ca²⁺:**
- Ca²⁺ will track the changing voltage
- With tau_ca=20, Ca²⁺ will slowly follow Vmem fluctuations
- Pattern in Ca²⁺ will degrade over ~100 timesteps
- By t=2000, Ca²⁺ gate has no spatial structure

**Result:** ❌ **Simple Ca²⁺ memory design FAILS**

#### Scenario 2: Vmem Pattern Shifts/Drifts

```python
# t=1000: Eye regions at locations A
# t=2000: Eye regions drift to locations B
```

**What happens to Ca²⁺:**
- Ca²⁺ will show weighted average of patterns at A and B
- Weights depend on time constants: newer pattern has more weight
- Spatial features become blurred/smeared

**Result:** ⚠️ **PARTIAL FAILURE** - Features unclear, reduced fidelity

#### Scenario 3: Vmem Decays to Uniform Baseline

```python
# t < 1000: Vmem has structure (~-50mV eyes, ~-70mV elsewhere)
# t > 1000: All cells relax to ~-60mV (no spatial variation)
```

**What happens to Ca²⁺:**
- Ca²⁺ decays: `Ca(t) = Ca(1000) * exp(-(t-1000)*dt/tau_ca)`
- At t=2000: `Ca(2000) ≈ Ca(1000) * exp(-0.5) ≈ 0.6 * Ca(1000)`
- **Spatial pattern retained** even though Vmem is uniform

**Result:** ✓ **Simple Ca²⁺ memory design WORKS** - Pattern persists via Ca²⁺ memory

#### Scenario 4: Vmem Pattern Remains Stable (Ideal Case)

```python
# t=1000: Pattern established
# t=2000: Same pattern, small fluctuations around fixed point
```

**What happens to Ca²⁺:**
- Ca²⁺ equilibrates to steady state reflecting Vmem pattern
- Small fluctuations filtered by tau_ca
- Pattern highly stable

**Result:** ✓✓ **Simple Ca²⁺ memory design WORKS PERFECTLY**

---

### The Key Insight

**Simple Ca²⁺ memory (tau_ca=20) works if:**
1. Vmem pattern **stabilizes** by t=1000 (doesn't keep changing structure), OR
2. Vmem **decays uniformly** after establishing pattern (amplitude decreases but spatial structure preserved in Ca²⁺), OR
3. tau_ca is **long enough** that Ca²⁺ memory dominates over new fluctuating inputs

**Simple Ca²⁺ memory FAILS if:**
- Vmem continues to **change spatial structure** after t=1000
- Pattern **drifts or shifts** over time
- Fluctuations are **strong enough** to overwhelm Ca²⁺ memory

---

### Solution for Unstable Vmem: CaMKII Bistability

If voltage pattern doesn't persist, we need a **bistable molecular memory** that "locks in" the pattern:

```python
class BioelectricTransduction:
    def __init__(self, grid_size, use_bistability=False, device='cpu', dtype=torch.float64):
        self.grid_size = grid_size
        self.device = device
        self.dtype = dtype
        self.use_bistability = use_bistability

        # Ca²⁺ dynamics (always present)
        self.tau_ca = torch.tensor(20.0, device=device, dtype=dtype)
        self.Ca = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)

        # Optional: CaMKII bistable switch
        if use_bistability:
            self.CaMKII_active = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)
            self.k_camkii_activation = torch.tensor(1.0, device=device, dtype=dtype)
            self.k_camkii_inactivation = torch.tensor(0.01, device=device, dtype=dtype)  # VERY slow
            self.ca_threshold = torch.tensor(0.5, device=device, dtype=dtype)
            self.ca_sensitivity = torch.tensor(0.1, device=device, dtype=dtype)

    def update(self, vmem, dt):
        # 1. Ca²⁺ dynamics (fast, tracks voltage)
        I_ca = self.compute_ca_current(vmem)
        dCa_dt = I_ca - self.Ca / self.tau_ca
        self.Ca = self.Ca + dt * dCa_dt
        self.Ca = torch.clamp(self.Ca, min=0.0, max=10.0)

        if self.use_bistability:
            # 2. CaMKII bistable switch
            # Activates when Ca > threshold
            activation_signal = torch.sigmoid((self.Ca - self.ca_threshold) / self.ca_sensitivity)

            # Dynamics: fast activation, very slow inactivation
            dCaMKII_dt = (self.k_camkii_activation * activation_signal * (1 - self.CaMKII_active) -
                          self.k_camkii_inactivation * self.CaMKII_active)

            self.CaMKII_active = self.CaMKII_active + dt * dCaMKII_dt
            self.CaMKII_active = torch.clamp(self.CaMKII_active, min=0.0, max=1.0)

            # Use CaMKII for gating (bistable, persistent)
            gate = self.CaMKII_active
        else:
            # Use Ca directly (simple memory)
            gate = self.Ca

        return {
            'Ca': self.Ca,
            'gate': gate,
            'vmem': vmem,
            'camkii': self.CaMKII_active if self.use_bistability else None
        }
```

**Key properties of CaMKII bistability:**

```python
# Timeline:
# t < 1000: High Vmem → High Ca²⁺ → CaMKII activates
# t = 1000: CaMKII ≈ 1.0 in patterned regions
# t = 1500: Ca²⁺ drops (if Vmem unstable) BUT CaMKII stays high
# t = 2000: CaMKII still ≈ 0.95 (only 5% decay over 1000 steps)

# Effective memory timescale:
tau_camkii = 1 / k_inactivation = 1 / 0.01 = 100 time units

# This is 5× longer than Ca²⁺ alone (tau_ca = 20)
```

---

### Comparison: Ca²⁺ Memory vs CaMKII Bistability

| Mechanism | Memory Timescale | Requires Stable Vmem? | Pattern Persistence | Biological Basis |
|-----------|------------------|----------------------|---------------------|------------------|
| **Ca²⁺ alone (tau_ca=20)** | ~20 time units | ⚠️ Yes (after t=1000) | Decays exponentially | Ca²⁺ buffering |
| **CaMKII bistable (k_inact=0.01)** | ~100 time units | ✓ No (locks at t=1000) | Nearly constant | Autophosphorylation |

**Biological realism:**

**Ca²⁺ buffering:**
- Intracellular Ca²⁺ buffering proteins (calmodulin, calbindin)
- Ca²⁺ pumps (PMCA, SERCA) with finite capacity
- Effective tau ~ 100ms to 10s

**CaMKII bistability:**
- CaMKII autophosphorylation creates persistent active state
- Once activated by Ca²⁺, can remain active for minutes-hours
- Well-documented molecular memory mechanism in neurons
- Involved in long-term potentiation (LTP) and cellular memory

---

### Decision Tree: Which Design to Use?

```
                    Characterize Vmem Dynamics
                              |
                              v
                    Run bioelectric for 2000 steps
                              |
                              v
                    Compute pattern correlation:
                    corr(Vmem[1000], Vmem[2000])
                              |
        ______________________|______________________
       |                      |                      |
       v                      v                      v
  corr > 0.8           0.5 < corr < 0.8         corr < 0.5
  (STABLE)             (DRIFTING)               (UNSTABLE)
       |                      |                      |
       v                      v                      v
  ✓ Use simple          ⚠️ Try longer          ✓ Use CaMKII
  Ca²⁺ memory          Ca²⁺ memory           bistability
  tau_ca = 20          tau_ca = 50           k_inact = 0.01
       |                      |                      |
       v                      v                      v
  Sufficient           Test carefully          Required
  Efficient            May work               Robust
```

---

### Diagnostic: Testing Vmem Pattern Stability

Before implementing, run this diagnostic to determine which design is needed:

```python
def test_vmem_stability(bio_model, num_timesteps=2000, checkpoint_interval=100):
    """
    Test whether bioelectric pattern is stable over time.

    Returns:
        dict with stability metrics
    """
    print("=== Testing Vmem Pattern Stability ===")

    # Run bioelectric simulation
    vmem_history = []
    for t in range(num_timesteps):
        bio_model.step(dt=0.01)
        if t % checkpoint_interval == 0:
            vmem_history.append(bio_model.electricNetwork.Vmem.clone())

    # Compute pattern correlations over time
    n_checkpoints = len(vmem_history)
    correlation_matrix = torch.zeros(n_checkpoints, n_checkpoints)

    for i in range(n_checkpoints):
        for j in range(n_checkpoints):
            vmem_i = vmem_history[i].flatten()
            vmem_j = vmem_history[j].flatten()
            corr = torch.corrcoef(torch.stack([vmem_i, vmem_j]))[0, 1]
            correlation_matrix[i, j] = corr

    # Key metrics
    corr_early_mid = correlation_matrix[5, 10].item()  # t=500 vs t=1000
    corr_mid_late = correlation_matrix[10, 20].item()  # t=1000 vs t=2000
    corr_early_late = correlation_matrix[5, 20].item()  # t=500 vs t=2000

    # Spatial variance over time (measure pattern strength)
    spatial_std = torch.tensor([v.std().item() for v in vmem_history])

    # Temporal stability (how much pattern changes between adjacent timepoints)
    temporal_changes = []
    for i in range(1, n_checkpoints):
        diff = (vmem_history[i] - vmem_history[i-1]).abs().mean().item()
        temporal_changes.append(diff)

    avg_temporal_change = np.mean(temporal_changes[10:])  # Average after t=1000

    results = {
        'correlation_matrix': correlation_matrix,
        'corr_early_mid': corr_early_mid,
        'corr_mid_late': corr_mid_late,
        'corr_early_late': corr_early_late,
        'spatial_std_history': spatial_std,
        'temporal_changes': temporal_changes,
        'avg_temporal_change_after_1000': avg_temporal_change,
        'vmem_checkpoints': vmem_history
    }

    # Recommendations
    print(f"\n=== Stability Analysis ===")
    print(f"Pattern correlation (t=1000 vs t=2000): {corr_mid_late:.3f}")
    print(f"Average temporal change after t=1000: {avg_temporal_change:.6f} V")
    print(f"Spatial structure strength at t=2000: {spatial_std[-1]:.4f} V")

    if corr_mid_late > 0.8 and avg_temporal_change < 0.001:
        print("\n✓ RECOMMENDATION: Simple Ca²⁺ memory (tau_ca=20) is SUFFICIENT")
        print("  Pattern is stable. Use original concurrent design.")
        results['recommendation'] = 'simple'
    elif corr_mid_late > 0.5:
        print("\n⚠️ RECOMMENDATION: Try longer Ca²⁺ memory (tau_ca=50)")
        print("  Pattern drifts slowly. May work with extended memory.")
        print("  Test and validate carefully.")
        results['recommendation'] = 'extended'
    else:
        print("\n✗ RECOMMENDATION: CaMKII bistability REQUIRED")
        print("  Pattern is unstable. Need molecular memory mechanism.")
        results['recommendation'] = 'bistable'

    return results
```

**Usage:**

```python
# Load stigmergic model
params = load_stigmergic_parameters('data/StigmergicModelParameters.dat')
bio_model = model(params, numBasicSamples=1)
bio_model.setExperimentalConditions((params['simParameters']['initialValues'], 1))

# Test stability
stability_results = test_vmem_stability(bio_model, num_timesteps=2000)

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Correlation matrix
im = axes[0].imshow(stability_results['correlation_matrix'], cmap='viridis')
axes[0].set_xlabel('Checkpoint')
axes[0].set_ylabel('Checkpoint')
axes[0].set_title('Vmem Pattern Correlation Over Time')
plt.colorbar(im, ax=axes[0])

# Spatial std over time
axes[1].plot(stability_results['spatial_std_history'])
axes[1].axvline(10, color='r', linestyle='--', label='t=1000')
axes[1].set_xlabel('Checkpoint (×100 steps)')
axes[1].set_ylabel('Spatial Std (V)')
axes[1].set_title('Pattern Strength Over Time')
axes[1].legend()

# Temporal changes
axes[2].plot(stability_results['temporal_changes'])
axes[2].axvline(10, color='r', linestyle='--', label='t=1000')
axes[2].set_xlabel('Checkpoint')
axes[2].set_ylabel('|ΔVmem| (V)')
axes[2].set_title('Pattern Changes Between Timepoints')
axes[2].legend()

plt.tight_layout()
plt.savefig('vmem_stability_analysis.png', dpi=150)
print("Saved: vmem_stability_analysis.png")
```

---

### Implementation Strategy

#### Phase 1: Diagnose Vmem Stability (Do First!)

```bash
# Run diagnostic
python diagnose_vmem_stability.py --params data/StigmergicModelParameters.dat
```

#### Phase 2A: If Stable (corr > 0.8)

```python
# Implement simple Ca²⁺ memory concurrent design
# As described in main document
transduction = BioelectricTransduction(grid_size=11, use_bistability=False)
transduction.tau_ca = torch.tensor(20.0, ...)
```

#### Phase 2B: If Drifting (0.5 < corr < 0.8)

```python
# Try extended Ca²⁺ memory
transduction = BioelectricTransduction(grid_size=11, use_bistability=False)
transduction.tau_ca = torch.tensor(50.0, ...)  # Longer memory

# Validate that pattern persists long enough
```

#### Phase 2C: If Unstable (corr < 0.5)

```python
# Implement CaMKII bistability
transduction = BioelectricTransduction(grid_size=11, use_bistability=True)
transduction.tau_ca = torch.tensor(20.0, ...)
transduction.k_camkii_inactivation = torch.tensor(0.01, ...)  # tau_camkii = 100

# CaMKII locks pattern at t=1000, persists to t=2000
```

---

### Expected Behavior: CaMKII Bistability

**Timeline of CaMKII activation:**

```python
# Phase 1: Pattern formation (t=0-1000)
# - Vmem develops spatial structure via stigmergic dynamics
# - High Vmem → High Ca²⁺ in specific regions
# - High Ca²⁺ → CaMKII activation in those regions
# - CaMKII transitions from 0 → 1 over ~100 timesteps where Ca²⁺ > threshold

# Phase 2: Pattern locking (t=1000)
# - CaMKII ≈ 1.0 in patterned regions (e.g., eye/nose domains)
# - CaMKII ≈ 0.0 in background regions
# - Pattern is "locked in" to CaMKII state

# Phase 3: Pattern maintenance (t=1000-2000)
# - Vmem may fluctuate/decay (doesn't matter anymore)
# - Ca²⁺ tracks Vmem changes
# - CaMKII decays VERY slowly: CaMKII(t) = CaMKII(1000) * exp(-(t-1000)/100)
# - At t=2000: CaMKII still ≈ 0.90 * CaMKII(1000) (90% retention)

# Phase 4: Gene gating (t=2000+)
# - Genes activate where CaMKII_gate > threshold
# - Pattern reflects bioelectric state at t=1000
# - Robust to voltage fluctuations after locking
```

**Diagnostic checks:**

```python
# At t=1000 (after pattern formation)
camkii_max = transduction.CaMKII_active.max().item()
camkii_mean_high = transduction.CaMKII_active[transduction.CaMKII_active > 0.5].mean().item()
print(f"CaMKII at t=1000: max={camkii_max:.3f}, mean(high)={camkii_mean_high:.3f}")
# Should be: max ≈ 0.8-1.0, mean(high) ≈ 0.6-0.9

# At t=2000 (during gene activation)
camkii_corr = torch.corrcoef(torch.stack([
    camkii_at_1000.flatten(),
    camkii_at_2000.flatten()
]))[0, 1].item()
print(f"CaMKII pattern retention: {camkii_corr:.3f}")
# Should be: > 0.85 (much higher than Ca²⁺ alone)
```

---

### Summary: Robustness Analysis

| Vmem Behavior After t=1000 | Simple Ca²⁺ (tau=20) | Extended Ca²⁺ (tau=50) | CaMKII Bistable |
|----------------------------|---------------------|----------------------|-----------------|
| **Stable (corr>0.8)** | ✓✓ Works perfectly | ✓ Works (overkill) | ✓ Works (overkill) |
| **Decays uniformly** | ✓ Works | ✓ Works | ✓ Works |
| **Drifts slowly (0.5<corr<0.8)** | ✗ May fail | ⚠️ May work | ✓ Works |
| **Fluctuates (corr<0.5)** | ✗ Fails | ✗ Likely fails | ✓ Works |

**Computational cost:**
- Simple Ca²⁺: **1 state variable** per cell (Ca)
- CaMKII bistable: **2 state variables** per cell (Ca + CaMKII)

**Implementation complexity:**
- Simple Ca²⁺: **Low** (already implemented)
- CaMKII bistable: **Medium** (add bistable dynamics, ~50 lines)

**Recommendation:**
1. **Always run diagnostic first** to characterize Vmem stability
2. **Start with simple Ca²⁺ design** if Vmem is stable (most efficient)
3. **Add CaMKII only if needed** based on diagnostic results

---

## Future Extensions

### 1. Voltage-Dependent Morphogen Secretion

Instead of (or in addition to) Ca²⁺ gating:

```python
# In refinedFacialGRN.update_morphogens():
shh_secretion = base_secretion * sigmoid((vmem + 0.055) / 0.01)
# Depolarized regions secrete more SHH
```

**Biological basis:** Some morphogens (e.g., Wnts) have voltage-sensitive secretion.

### 2. Bidirectional Feedback

Genes modulate bioelectric properties:

```python
# In embryo.py or cellularFieldNetwork.py:
G_dep_modulated = G_dep * (1 + alpha * gene_expression)
# Gene expression increases depolarizing conductance
```

**Already exists in codebase:** See `apply_gene_voltage_feedback()` in `cellularFieldNetwork.py`.

### 3. Multiple Ca²⁺ Compartments

Separate cytoplasmic vs nuclear Ca²⁺:

```python
self.Ca_cyto = ...  # Fast, follows voltage
self.Ca_nucleus = ...  # Slow, integrates cyto
# Gene gating uses nuclear Ca²⁺ (longer memory)
```

**Advantage:** Even longer memory without unrealistically slow cytoplasmic dynamics.

---

## Summary

**Key Design Decisions:**

1. ✅ **Concurrent dynamics:** All processes run simultaneously
2. ✅ **Same dt=0.01:** Numerical stability, no artificial timescale separation via update frequency
3. ✅ **Timescale separation via tau:** tau_ca=20 creates Ca²⁺ memory that bridges bioelectric → gene gap
4. ✅ **Temporal gating emerges naturally:** Ca²⁺ at t=2000 reflects pattern from t=1000 due to slow decay
5. ✅ **Minimal code changes:** Only need to increase tau_ca and create new simulation loop

**Biological Realism:**

- ✅ Ca²⁺ buffering provides 10-100s memory in real cells
- ✅ All processes concurrent, not artificially separated
- ✅ Pattern persistence via biophysical mechanism, not ad-hoc freezing

**Next Steps:**

1. Increase `tau_ca` to 20.0
2. Implement concurrent simulation loop
3. Verify Ca²⁺ pattern persistence
4. Compare results to current sequential approach

---

## Appendix A: Mathematical Relationship Between Tau and Degradation Rate Constants

### The Basic Relationship

For a **first-order decay/degradation process**:

```
dx/dt = -k * x
```

The solution is exponential decay:
```
x(t) = x(0) * exp(-k * t)
```

The **time constant tau** is defined as the time when x decays to **1/e ≈ 37%** of its initial value:
```
x(tau) = x(0) * exp(-k * tau) = x(0) / e

This occurs when: k * tau = 1
Therefore: tau = 1/k
```

### Application to Our Model

#### Morphogens

**Equation (simplified, ignoring production/diffusion):**
```python
dshh/dt = -k_deg * shh
```

**Time constant:**
```
tau_morph = 1 / k_deg = 1 / 0.05 = 20.0 time units
```

**Meaning:** Morphogen concentration decays to 37% of initial value in 20 time units.

#### Genes

**Equation (simplified, ignoring activation):**
```python
dgene/dt = -k_off * gene
```

**Time constant:**
```
tau_gene = 1 / k_off = 1 / 0.02 = 50.0 time units
```

**Meaning:** Gene expression decays to 37% of initial value in 50 time units.

### Important Caveat: This is an Oversimplification!

In the actual model, the equations are **NOT** simple first-order decay:

#### Actual Morphogen Equation
```python
dshh/dt = secretion + D * laplacian(shh) - k_deg * shh + inhibition_terms
```

**This includes:**
- Production (secretion)
- Diffusion (spatial coupling)
- Degradation (k_deg)
- Interactions (mutual inhibition)

#### Actual Gene Equation
```python
dgene/dt = k_activation * f(morph, bio) - k_degradation * gene
```

**This includes:**
- Activation (depends on morphogens and bioelectric signals)
- Degradation (k_degradation)

### Effective Timescale for Production-Degradation Systems

For systems with both production and degradation, the **relaxation timescale** (how fast the system approaches steady state) depends on the dynamics around equilibrium.

#### Gene Example

```python
dgene/dt = k_on * activation - k_off * gene
```

Near steady state where `gene_ss = (k_on * activation) / k_off`:

The system relaxes back to steady state with a characteristic timescale that depends on both production and degradation terms.

### More Accurate Statement

Instead of saying "tau = 1/k_deg" or "tau = 1/k_off", more accurately:

**"The characteristic timescale is dominated by the degradation/decay rate, which is approximately tau ~ 1/k_deg for morphogens and tau ~ 1/k_off for genes when the system is near steady state and considering small perturbations."**

Or more precisely:

**"The relaxation time constant for returning to steady state after a perturbation is approximately tau = 1/k_deg (morphogens) or tau = 1/k_off (genes) when production terms remain constant."**

### When Does tau = 1/k Apply?

The simple relationship `tau = 1/k` is accurate when:
1. **Pure decay** with no production: `dx/dt = -k*x`
2. **Relaxation with constant input**: Perturbations decay at rate k when inputs are held fixed
3. **Weak feedback**: Self-regulatory terms don't significantly modify the effective rate constant

For more complex systems with strong feedback or nonlinear interactions, the effective timescale must be determined by linearization analysis (see Appendix B).

---

## Appendix B: Linearization Analysis of Gene Dynamics with Self-Feedback

### The Full Nonlinear System

```python
dgene/dt = k_on * activation(morph, bio, gene) - k_off * gene
```

where `activation` is generally a **nonlinear function** of morphogens, bioelectric signals, and potentially the gene itself (self-maintenance).

### Step 1: Find the Steady State

At steady state, `dgene/dt = 0`:

```python
0 = k_on * activation_ss - k_off * gene_ss

gene_ss = (k_on * activation_ss) / k_off
```

where `activation_ss` is the value of the activation function at steady state.

### Step 2: Define Perturbation Variables

Let's perturb slightly around steady state:

```python
gene(t) = gene_ss + δgene(t)
activation(t) = activation_ss + δactivation(t)
```

where `δgene` and `δactivation` are **small perturbations**.

### Step 3: Substitute into the ODE

```python
d(gene_ss + δgene)/dt = k_on * (activation_ss + δactivation) - k_off * (gene_ss + δgene)

d(δgene)/dt = k_on * activation_ss + k_on * δactivation - k_off * gene_ss - k_off * δgene
```

Note: `d(gene_ss)/dt = 0` because steady state is constant.

### Step 4: Use Steady State Condition

We know from steady state: `k_on * activation_ss = k_off * gene_ss`

So those terms cancel:

```python
d(δgene)/dt = k_on * δactivation - k_off * δgene
```

### Step 5: Linearization of Activation Function

Here's where the **linearization approximation** comes in. We need to relate `δactivation` to `δgene`.

#### Case A: Activation doesn't depend on gene itself

If `activation = f(morph, bio)` with no self-feedback, then:

```python
δactivation ≈ 0  (if morph and bio are fixed at steady state)
```

So:
```python
d(δgene)/dt = -k_off * δgene
```

**Solution:** `δgene(t) = δgene(0) * exp(-k_off * t)`

**Time constant:** `tau = 1 / k_off` ✓

#### Case B: Activation depends on gene (self-maintenance)

If `activation = f(morph, bio, gene)`, we need to linearize the activation function using Taylor expansion:

```python
activation(gene) ≈ activation_ss + (∂activation/∂gene)|_ss * δgene

δactivation ≈ (∂activation/∂gene)|_ss * δgene
```

Let's call this partial derivative `α = ∂activation/∂gene|_ss`.

Then:
```python
d(δgene)/dt = k_on * α * δgene - k_off * δgene
            = (k_on * α - k_off) * δgene
```

**Solution:** `δgene(t) = δgene(0) * exp((k_on * α - k_off) * t)`

**Effective time constant:** `tau_eff = 1 / (k_off - k_on * α)`

### Application to RefinedFacialGRN

In `refinedFacialGRN.py`, genes have self-maintenance:

```python
# Simplified version of the actual code
initiation = AND(morph_signal, bio_gate)
maintenance = hill(gene, K_self, n_self)  # Self-feedback!
activation = w_init * initiation + w_maint * maintenance

dgene/dt = k_on * activation - k_off * gene
```

With self-maintenance, the sensitivity is:

```python
α = ∂activation/∂gene = w_maint * ∂hill/∂gene

# For Hill function: hill(x, K, n) = x^n / (K^n + x^n)
∂hill/∂gene = (n * K^n * gene^(n-1)) / (K^n + gene^n)^2
```

At steady state where `gene ≈ K_self`, the Hill function has value ~0.5 and:

```python
∂hill/∂gene|_{gene=K_self} = n / (4 * K_self)

α ≈ w_maint * n_self / (4 * K_self)
```

So the **effective time constant** is:

```python
tau_eff = 1 / (k_off - k_on * w_maint * n_self / (4 * K_self))
```

This is **different from** `1/k_off` when self-maintenance is strong!

### Physical Interpretation

**Positive feedback (α > 0):**
- Self-maintenance reduces effective degradation rate
- System relaxes more slowly: `tau_eff > 1/k_off`
- Genes "remember" their state longer
- Can lead to bistability if `k_on * α > k_off`

**Negative feedback (α < 0):**
- Self-inhibition increases effective degradation rate
- System relaxes more quickly: `tau_eff < 1/k_off`
- More stable, less hysteresis

**No feedback (α = 0):**
- Simple decay: `tau_eff = 1/k_off`

### Correct Statement for Our Model

**Without Self-Feedback:**
```python
dgene/dt = k_on * f(morph, bio) - k_off * gene

tau_relax = 1 / k_off  ✓ CORRECT
```

**With Self-Feedback (RefinedFacialGRN):**
```python
dgene/dt = k_on * [w_init * f(morph,bio) + w_maint * hill(gene)] - k_off * gene

tau_relax = 1 / (k_off - k_on * w_maint * ∂hill/∂gene)  ✓ MORE ACCURATE
```

### When is "tau ≈ 1/k_off" a Good Approximation?

The simple approximation `tau ≈ 1/k_off` holds when:

1. **Self-maintenance is weak:** `w_maint << 1` or `k_on * α << k_off`
2. **Far from bifurcation:** System is not near bistable regime
3. **Qualitative understanding:** We just need order-of-magnitude timescale estimates

For precise quantitative analysis of relaxation dynamics, use the full linearization with feedback terms.

### Numerical Example

With parameters from `refinedFacialGRN.py`:
- `k_activation = 0.10` (this is k_on)
- `k_degradation = 0.01` (this is k_off)
- `w_maint = 0.3`
- `n_self = 2.0`
- `K_self = 0.3`

At steady state (gene ≈ K_self = 0.3):

```python
α = w_maint * n_self / (4 * K_self)
  = 0.3 * 2.0 / (4 * 0.3)
  = 0.5

tau_eff = 1 / (k_off - k_on * α)
        = 1 / (0.01 - 0.10 * 0.5)
        = 1 / (0.01 - 0.05)
        = 1 / (-0.04)
        = -25.0  (UNSTABLE!)
```

**Wait, this is negative!** This means `k_on * α > k_off`, which indicates:
- **The system is in the bistable regime!**
- Positive feedback is strong enough that small perturbations can grow
- The steady state is unstable (or marginally stable with additional terms)

This suggests that in the actual model, either:
1. Other terms (like initiation signal) stabilize the dynamics
2. The effective k_on is smaller due to gating
3. The system intentionally has bistable behavior for memory

### Conclusion

The relationship `tau = 1/k_off` is a **useful approximation** for understanding timescale ordering, but the **exact relaxation timescale** depends on:
- Degradation rate (k_off)
- Production rate (k_on)
- Strength of self-feedback (α)
- Nonlinear activation functions

For systems with strong positive feedback (like our gene networks with self-maintenance), the effective timescale can be **much longer** than `1/k_off`, or the system may even be bistable.

**Key takeaway:** Use `tau ~ 1/k_off` for rough estimates and conceptual understanding, but perform proper linearization analysis for quantitative predictions of relaxation dynamics.
