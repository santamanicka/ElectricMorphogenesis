# Refined Model Architecture Design

## Overview: Dual-Driver (Bioelectric + Morphogen) Facial Patterning

**Key Changes from Original:**
1. ✅ Bioelectric AND morphogen gradients both drive genes (parallel, similar logic)
2. ✅ No A-P voltage gradient (all features ~-60mV from stigmergic model)
3. ✅ Replace "detail" with: gap junction currents + temporal integration + separate channels
4. ✅ Fast bioelectric, slow genetic timescales
5. ✅ Features classified from gene expression only (not voltage thresholds)

---

## Architecture Layers

### Layer 1: Bioelectric Network (FAST - milliseconds to seconds)

**Source**: Stigmergic cellularFieldNetwork
- Produces Vmem pattern (~-60mV everywhere, with spatial structure)
- Gap junction currents: **I_gj = G_gj × (V_neighbor - V_cell)**
- Timescale: dt = 0.01, ~1000 iterations (~10 time units)

**Key Output**:
- `Vmem`: (numCells, 1) - membrane voltage
- `I_gj`: (numCells, 1) - **gap junction current (NEW)**

---

### Layer 2: Bioelectric Transduction (INTERMEDIATE - seconds to minutes)

**NEW Component**: `BioelectricTransduction` class

**Two parallel channels** (replacing "detail"):

#### Channel A: Voltage → Ca²⁺ Dynamics (Temporal Integration)
```python
# Voltage-gated Ca²⁺ influx
V_half = -40mV  # Ca²⁺ channel activation threshold
I_ca = g_ca × sigmoid((Vmem - V_half) / k_v)

# Temporal integration (provides memory)
dCa/dt = I_ca - Ca / tau_ca
Ca[t] = Ca[t-1] + dt × dCa/dt
```

**Purpose**: Accumulates voltage signals over time, prevents instantaneous flips

#### Channel B: Gap Junction Current → Metabolic State
```python
# Current magnitude = metabolic cost (ATP consumption)
I_gj_magnitude = |I_gj|.sum()

# Metabolic state (1.0 = healthy, <1.0 = stressed)
metabolic_state = 1.0 - beta × I_gj_magnitude
```

**Purpose**: Cells with high current flow (boundary regions) have different metabolic state than quiet cells

**Outputs**:
- `Ca`: (numCells,) - integrated Ca²⁺ concentration
- `metabolic_state`: (numCells,) - metabolic health proxy

**Timescale**: tau_ca = 1.0 (integrates over ~100 bioelectric steps)

---

### Layer 3: Morphogen Gradients (SLOW - minutes to hours)

**Source**: FacialGRN morphogen dynamics (SHH, FGF8, EDN1)

**Key Change**: Morphogen secretion modulated by bioelectric signals

```python
# Baseline secretion (spatially patterned sources)
shh_baseline = gaussian(ventral_midline)
fgf8_baseline = gaussian(anterior_neural_ridge)

# Bioelectric modulation
shh_secretion = shh_baseline × (1.0 + alpha_shh × Ca)
fgf8_secretion = fgf8_baseline × (1.0 + alpha_fgf × metabolic_state)

# Diffusion-degradation dynamics
dSHH/dt = shh_secretion + D∇²SHH - k_deg × SHH
```

**Rationale**:
- High Ca²⁺ (from depolarization) → increased morphogen secretion
- High metabolic state (low current, quiet regions) → stable morphogen production
- Low metabolic (high current, boundaries) → reduced secretion

**Outputs**:
- `SHH`, `FGF8`, `EDN1`: (grid_size, grid_size) - morphogen concentrations

**Timescale**: tau_morph = 10.0

---

### Layer 4: Gene Regulatory Network (SLOWEST - hours)

**Dual Driver Architecture** (both use Hill dynamics):

```python
# Driver 1: Morphogen input (PRIMARY - 70% weight)
morph_input_pax6 = Hill(FGF8, K=0.3, n=2.0) × inhibit(SHH, K=0.4, n=2.0)

# Driver 2: Bioelectric input (SECONDARY - 30% weight)
bio_input_pax6 = Hill(Ca, K=0.5, n=2.0)

# Combined activation
total_activation = w_morph × morph_input + w_bio × bio_input

# Gene dynamics
dpax6/dt = k_on × total_activation - k_off × pax6
```

**Gene Battery**:
- **Eye**: rx, six3, pax6, lhx2 (activated by FGF8 + Ca²⁺)
- **Nose**: alx (activated by SHH + intermediate metabolic)
- **Mouth**: dlx, hand2 (activated by EDN1 + metabolic stress)
- **Bone**: runx2 (default when others low)

**Timescale**: tau_gene = 50.0 (slowest)

---

### Layer 5: Feature Classification (FROM GENES ONLY)

**NEW Component**: `GeneBasedFeatureClassifier`

**Soft probabilistic classification**:
```python
# Compute feature scores from gene combinations
eye_score = pax6 × lhx2  # Cooperative binding
nose_score = alx
mouth_score = dlx × hand2
bone_score = runx2 + (1 - max(eye, nose, mouth))  # Default

# Probabilistic assignment (softmax)
scores = stack([bone, eye, nose, mouth])
feature_probs = softmax(scores / temperature)

# Hard classification for visualization
feature_id = argmax(scores)
```

**No voltage thresholds!** Features emerge purely from gene expression.

---

### Layer 6: Bidirectional Feedback (Genes → Bioelectrics)

**Existing mechanism** (from cellularFieldNetwork.apply_gene_voltage_feedback):

```python
# Eye genes → depolarizing channels
delta_G_dep = 0.02 × (pax6 + six3 + lhx2)

# Mouth genes → polarizing channels
delta_G_pol = 0.02 × (dlx + hand2 + alx)

# Update Vmem via modified ion channel conductances
# (This feeds back into gap junction currents)
```

**Weak coupling** (gain ~ 0.02) preserves bioelectric pattern structure.

---

## Timescale Hierarchy

**Mapping to real time** (if dt_bio = 0.1 seconds):

| Layer | Timescale Param | Real Time | Equilibration |
|-------|----------------|-----------|---------------|
| Bioelectrics | dt = 0.01 | 1 ms | 10-100 sec |
| Ca²⁺ dynamics | tau = 1.0 | 0.1 sec | 10 sec |
| Morphogens | tau = 10.0 | 1 sec | 100 sec (~2 min) |
| Genes | tau = 50.0 | 5 sec | 500 sec (~8 min) |

**Ratio**: Bioelectrics : Ca²⁺ : Morphogens : Genes = 1 : 100 : 1000 : 5000

This separation prevents:
- Genes instantly flipping when voltage fluctuates (temporal integration via Ca²⁺)
- Bioelectric pattern being destroyed by gene feedback (morphogens buffer)
- Loss of spatial structure (morphogen diffusion preserves gradients)

---

## Information Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ STIGMERGIC BIOELECTRIC PATTERN                              │
│ (~-60mV everywhere, spatial structure from field effects)   │
└──────────────┬─────────────────────────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
   ┌─────────┐  ┌─────────────┐
   │  Vmem   │  │ Gap Junction│
   │         │  │ Currents Igj│
   └────┬────┘  └──────┬──────┘
        │              │
        │ Ca²⁺ channel │ Metabolic
        │ activation   │ cost
        ▼              ▼
   ┌─────────────────────────┐
   │ BIOELECTRIC TRANSDUCTION│
   │ • Ca²⁺ dynamics         │
   │ • Metabolic state       │
   │ (Temporal integration)  │
   └────────┬────────────────┘
            │
            │ Modulates
            ▼
   ┌─────────────────────────┐
   │ MORPHOGEN GRADIENTS     │
   │ • SHH (ventral)         │
   │ • FGF8 (anterior)       │
   │ • EDN1 (lateral)        │
   │ (Diffusion-degradation) │
   └────┬────────────────────┘
        │
        │ Hill activation
        ▼
   ┌─────────────────────────┐
   │ GENE REGULATORY NETWORK │
   │ Dual drivers:           │
   │ • Morphogens (70%)      │
   │ • Bioelectrics (30%)    │
   │ Both use Hill functions │
   └────┬────────────────────┘
        │
        ▼
   ┌─────────────────────────┐
   │ FEATURE CLASSIFICATION  │
   │ (From gene expression)  │
   │ • Eye: pax6×lhx2        │
   │ • Nose: alx             │
   │ • Mouth: dlx×hand2      │
   └────┬────────────────────┘
        │
        │ Gene → ion channel
        │ expression (feedback)
        └──────────► Vmem (weak coupling)
```

---

## Key Design Decisions

### 1. Why Two Channels (Vmem + I_gj)?
- **Vmem (absolute)**: Cell's intrinsic state, metabolic activity, Ca²⁺ signaling
- **I_gj (gradient)**: Boundary detection, neighbor sensing, metabolic stress
- Biology uses both! Separating them is more realistic.

### 2. Why Ca²⁺ Integration?
- **Temporal buffering**: Prevents genes from instantly flipping when voltage fluctuates
- **Biological reality**: Ca²⁺ is THE second messenger linking voltage to genes
- **Memory**: Accumulates history, cells "remember" past voltage states

### 3. Why Modulate Morphogen Secretion (Not Just Gene Response)?
- **Bidirectionality at multiple levels**: Bioelectrics can influence morphogen landscape
- **Amplification**: Small voltage differences → large morphogen gradient changes
- **Biological precedent**: Voltage affects exocytosis, vesicle trafficking

### 4. Why 70% Morphogen, 30% Bioelectric?
- **Morphogens are primary** (well-established in facial development)
- **Bioelectrics modulate** (supported but less characterized)
- **Prevents electrical dominance** while allowing bioelectric guidance

### 5. Why Soft Feature Classification?
- **No hard thresholds**: Biology is graded, probabilistic
- **Differentiable**: Can be incorporated into optimization/learning
- **Realistic**: Cells at boundaries express multiple markers

---

## Expected Behavior

### Phase 1: Bioelectric Pattern Formation (0-100 time units)
- Stigmergic model creates spatial Vmem structure
- Gap junction currents flow where voltage differences exist
- Ca²⁺ begins accumulating in regions with specific voltage dynamics
- Metabolic state differentiates boundaries vs quiet regions

### Phase 2: Morphogen Gradient Establishment (100-1000 time units)
- Bioelectric signals modulate morphogen secretion
- SHH/FGF8/EDN1 gradients form, guided by bioelectric prepattern
- Morphogens diffuse, creating smooth spatial gradients

### Phase 3: Gene Expression Patterning (1000-5000 time units)
- Genes respond to both morphogens (70%) and bioelectrics (30%)
- Eye genes (pax6, lhx2) activate where FGF8 high + Ca²⁺ elevated
- Mouth genes (dlx, hand2) activate where EDN1 high + metabolic stress
- Gene patterns stabilize slowly

### Phase 4: Bidirectional Equilibration (5000+ time units)
- Gene expression feeds back to modulate Vmem (weak)
- Bioelectric pattern shifts slightly but preserves overall structure
- Morphogen gradients adjust to new bioelectric state
- System reaches dynamic equilibrium

**Outcome**: Biomolecular face pattern (genes) mirrors bioelectric face pattern (Vmem structure), despite bidirectional coupling. Morphogens act as intermediaries, preventing destructive feedback.

---

## Parameters Summary

```python
# Bioelectric Transduction
V_half_ca = -0.04        # -40mV Ca²⁺ activation
k_ca = 0.01              # 10mV voltage sensitivity
g_ca = 1.0               # Max Ca²⁺ conductance
tau_ca = 1.0             # Ca²⁺ decay timescale
beta_metabolic = 0.1     # Current → metabolic cost factor

# Morphogen Secretion Modulation
alpha_shh_ca = 0.3       # Ca²⁺ → SHH secretion gain
alpha_fgf_meta = 0.5     # Metabolic → FGF8 secretion gain

# Gene Dual Driver Weights
w_morphogen = 0.7        # Morphogen contribution
w_bioelectric = 0.3      # Bioelectric contribution

# Gene Dynamics
k_on = 0.05              # Gene activation rate
k_off = 0.02             # Gene degradation rate
tau_gene = 50.0          # Gene timescale

# Bidirectional Feedback
gene_feedback_gain = 0.02  # Gene → Vmem coupling strength
```

---

## Implementation Components

1. **BioelectricTransduction** (new class)
   - Inputs: Vmem, I_gj
   - Outputs: Ca, metabolic_state
   - Update: Ca²⁺ dynamics + metabolic computation

2. **cellularFieldNetwork.get_gap_junction_currents()** (new method)
   - Expose I_gj already computed internally

3. **FacialGRN (refactored)**
   - Morphogen secretion modulated by bioelectric signals
   - Gene activation from dual drivers (morphogen + bioelectric)
   - Remove all detail-based logic

4. **GeneBasedFeatureClassifier** (new class)
   - Inputs: Gene expression grids
   - Outputs: Feature classifications (soft + hard)
   - No voltage inputs!

5. **run_refined_facial_integration.py** (new script)
   - Orchestrates all components
   - Implements timescale hierarchy
   - Runs bidirectional coupling loops

---

## Success Criteria

✅ **Bioelectric pattern guides gene pattern**
- Without destroying it via feedback

✅ **Features emerge from genes, not voltage**
- Can measure success by gene expression, not Vmem thresholds

✅ **Morphogen gradients amplify bioelectric cues**
- Act as intermediate layer, prevent direct voltage→gene brittleness

✅ **Temporal integration provides robustness**
- Gene patterns don't flicker when voltage fluctuates

✅ **Biologically justifiable mechanisms**
- Ca²⁺ dynamics, gap junction currents, morphogen modulation
- All have literature support

✅ **No A-P voltage gradient assumption**
- All features at ~-60mV, distinguished by local dynamics (I_gj, Ca²⁺)
