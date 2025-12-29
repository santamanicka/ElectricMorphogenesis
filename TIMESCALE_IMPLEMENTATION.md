# Timescale Implementation in Refined Model

## Summary

The timescale hierarchy is implemented through **two mechanisms**:
1. **Explicit tau parameters** - for Ca²⁺ dynamics
2. **Implicit timescales** - through rate constants (k_activation, k_degradation, diffusion, etc.)
3. **Simulation step counts** - different numbers of iterations per cycle

---

## 1. Explicit Tau Parameters

### Ca²⁺ Dynamics (bioelectricTransduction.py)

**Location**: `bioelectricTransduction.py`, line 40
```python
self.tau_ca = torch.tensor(1.0, device=device, dtype=dtype)  # Ca²⁺ decay timescale
```

**Usage**: Line 101
```python
dCa_dt = I_ca - self.Ca / self.tau_ca
```

**Physical meaning**:
- tau_ca = 1.0 time units
- Ca²⁺ decays with time constant of 1.0
- After time t = tau_ca, Ca²⁺ drops to 1/e ≈ 37% of initial value

**Effective timescale**:
- With dt = 0.01, this means Ca²⁺ takes ~100 timesteps to equilibrate
- Real time: If dt = 0.1 sec, tau_ca = 0.1 sec (100 milliseconds)

---

## 2. Implicit Timescales (Through Rate Constants)

### Morphogen Dynamics (refinedFacialGRN.py)

**Parameters** (lines 47-48):
```python
'diffusion_rate': torch.tensor(0.1, device=device, dtype=dtype),
'degradation_rate': torch.tensor(0.05, device=device, dtype=dtype),
```

**Usage** (lines 207-210):
```python
dshh_dt = (shh_secretion +
           D * laplacian(shh) -
           k_deg * shh)
```

**Effective tau_morphogen**:
```
tau_morph ≈ 1 / k_degradation = 1 / 0.05 = 20.0 time units
```

**NOT explicitly named as "tau"**, but degradation_rate implicitly defines the timescale.

**Equilibration time**:
- With dt = 0.01, morphogens take ~2000 timesteps to equilibrate
- In practice, we run 500 GRN steps per cycle (includes morphogen updates)

---

### Gene Dynamics (refinedFacialGRN.py)

**Parameters** (lines 61-62):
```python
'k_activation': torch.tensor(0.05, device=device, dtype=dtype),
'k_degradation': torch.tensor(0.02, device=device, dtype=dtype),
```

**Usage** (example, line 269):
```python
drx_dt = k_on * activation_rx - k_off * self.grid['rx']
```

**Effective tau_gene**:
```
tau_gene ≈ 1 / k_degradation = 1 / 0.02 = 50.0 time units
```

**Again, NOT explicitly named**, but k_degradation sets the timescale.

**Equilibration time**:
- With dt = 0.01, genes take ~5000 timesteps to equilibrate
- In practice, we run 500 GRN steps per cycle (partial equilibration)

---

## 3. Timescale Separation via Step Counts

### In run_refined_facial_integration.py

**Different iteration counts per cycle** (lines 321-325):

```python
results = run_integrated_dynamics(
    bio_model, transduction, facial_grn, classifier,
    num_cycles=5,
    bio_steps=100,      # Bioelectric steps per cycle
    grn_steps=500,      # GRN steps per cycle (morphogens + genes)
    feedback_gain=0.02
)
```

**This creates effective timescale separation**:

| Layer | Steps per cycle | dt | Total time per cycle | Relative speed |
|-------|----------------|-----|----------------------|----------------|
| Bioelectric | 100 | 0.01 | 1.0 | **100x faster** |
| GRN (morphogen + gene) | 500 | 0.01 | 5.0 | 1x (baseline) |

**Note**: Morphogens and genes run in the **same** loop, but genes are slower because their effective tau is longer (50.0 vs 20.0).

---

## 4. Complete Timescale Hierarchy

Combining all mechanisms:

| Component | Explicit tau | Implicit tau (from rates) | Steps/cycle | Equilibration time |
|-----------|-------------|--------------------------|-------------|-------------------|
| **Bioelectric (Vmem)** | - | ~0.1 (from timestep) | 100 | ~10 timesteps |
| **Ca²⁺ transduction** | **1.0** | - | (within bio loop) | ~100 timesteps |
| **Morphogens** | - | **20.0** (1/k_deg) | 500 | ~2000 timesteps |
| **Genes** | - | **50.0** (1/k_off) | 500 | ~5000 timesteps |

**Effective ratio**: 1 : 10 : 200 : 500 (Vmem : Ca²⁺ : Morphogen : Gene)

---

## 5. Why This Design?

### Explicit tau for Ca²⁺:
- ✅ **Clear parameter**: Easy to tune
- ✅ **Physical meaning**: Directly corresponds to Ca²⁺ buffering/extrusion timescale
- ✅ **Separable from other dynamics**: Ca²⁺ is an intermediate signal, not part of morphogen/gene equations

### Implicit tau for morphogens/genes:
- ✅ **Standard in reaction-diffusion models**: Degradation rate is the typical parameter
- ✅ **Biologically meaningful**: k_degradation maps to protein/mRNA half-life
- ✅ **Coupled to other processes**: Morphogen/gene dynamics involve production, degradation, and diffusion - tau is emergent

### Step count separation:
- ✅ **Computational efficiency**: Don't need to simulate bioelectrics at gene timescales
- ✅ **Quasi-steady-state assumption**: Bioelectrics equilibrate fast, so genes see time-averaged bioelectric signals
- ✅ **Prevents numerical stiffness**: Different step sizes for different dynamics

---

## 6. How to Modify Timescales

### To make Ca²⁺ slower (more integration):
**File**: `bioelectricTransduction.py`, line 40
```python
self.tau_ca = torch.tensor(5.0, ...)  # Was 1.0
```

### To make morphogens slower:
**File**: `refinedFacialGRN.py`, line 48
```python
'degradation_rate': torch.tensor(0.01, ...),  # Was 0.05 → tau = 100.0
```

### To make genes slower:
**File**: `refinedFacialGRN.py`, line 62
```python
'k_degradation': torch.tensor(0.01, ...),  # Was 0.02 → tau = 100.0
```

### To change simulation step ratios:
**File**: `run_refined_facial_integration.py`, lines 322-323
```python
bio_steps=200,    # Was 100 → bioelectrics run 2x longer
grn_steps=1000,   # Was 500 → GRN runs 2x longer
```

---

## 7. Mathematical Equivalence

For a first-order linear ODE:
```
dX/dt = -X/tau + input
```

This is equivalent to:
```
dX/dt = -k * X + input,  where k = 1/tau
```

**In the code**:
- Ca²⁺: Uses explicit tau form: `dCa/dt = I_ca - Ca/tau_ca`
- Morphogens: Uses rate constant form: `dSHH/dt = secretion - k_deg * SHH + diffusion`
- Genes: Uses rate constant form: `drx/dt = k_on * activation - k_off * rx`

Both formulations are mathematically equivalent, just different parameterizations.

---

## 8. Actual Timescales in Current Model

Running the test (`run_refined_facial_integration.py`):

```
5 cycles × (100 bio + 500 GRN steps) = 5 × 600 = 3000 total timesteps
With dt = 0.01 → 30.0 time units total
```

**Effective tau values achieved**:
- Ca²⁺: ~1.0 (explicit)
- Morphogens: ~20.0 (from 1/0.05)
- Genes: ~50.0 (from 1/0.02)

**Separation factors**:
- Ca²⁺ vs Morphogen: 20x
- Ca²⁺ vs Gene: 50x
- Morphogen vs Gene: 2.5x

---

## 9. Biological Realism Check

Mapping to real time (assuming dt = 0.1 sec = 100 ms):

| Component | Model tau | Real time | Biological reality |
|-----------|-----------|-----------|-------------------|
| Vmem | 0.1 | 10 ms | ✅ Action potentials: ~1-5 ms |
| Ca²⁺ | 1.0 | 100 ms | ✅ Ca²⁺ transients: ~50-500 ms |
| Morphogens | 20.0 | 2 sec | ⚠️ SHH/FGF8 gradients: minutes-hours |
| Genes | 50.0 | 5 sec | ❌ mRNA/protein: 10-60 min |

**Conclusion**: Timescale ordering is correct, but absolute values are compressed (for computational tractability).

---

## 10. Where Explicit Tau Would Help

Consider adding explicit tau parameters for morphogens and genes to make tuning easier:

### Option A: Add tau_morph to morphogen updates
```python
# In refinedFacialGRN.py, morphogen_params:
'tau_morph': torch.tensor(20.0, ...),

# In update_morphogens:
dshh_dt = (secretion + D * laplacian(shh)) / tau_morph - shh / tau_morph
```

### Option B: Add tau_gene to gene updates
```python
# In gene_params:
'tau_gene': torch.tensor(50.0, ...),

# In update_genes:
drx_dt = (k_on * activation - k_off * rx) / tau_gene
```

This would make timescales **explicit** rather than **implicit**.

---

## Summary Table

| Where | Timescale | How implemented | File:line |
|-------|-----------|----------------|-----------|
| Ca²⁺ | tau_ca = 1.0 | **Explicit** parameter | bioelectricTransduction.py:40 |
| Morphogen | tau ≈ 20.0 | **Implicit** (1/k_deg) | refinedFacialGRN.py:48 |
| Gene | tau ≈ 50.0 | **Implicit** (1/k_off) | refinedFacialGRN.py:62 |
| Bioelectric | Fast | Step count (100/cycle) | run_refined_facial_integration.py:322 |
| GRN | Slow | Step count (500/cycle) | run_refined_facial_integration.py:323 |
