# Electric ↔ GRN Transduction Logic

## Overview

The bidirectional coupling between the bioelectric layer (Vmem) and the gene regulatory network (GRN) occurs through two distinct transduction pathways:

1. **Electric → GRN**: Voltage patterns influence gene expression
2. **GRN → Electric**: Gene expression modulates ion channel conductances, which alter voltage

---

## 1. Electric → GRN Transduction

There are **two different mechanisms** depending on which GRN type is used:

### A. Generic GRN (geneRegulatoryNetwork class)

**Location**: `geneRegulatoryNetwork.py:232-296`

#### Step 1: Voltage Input Processing (`updateDynamicalParameters`)

```python
def updateDynamicalParameters(self, externalInputs=None):
    # externalInputs = Vmem with shape (numSamples, numCells, 1)

    # Repeat voltage for each gene variable
    self.tissueExternalInputs = torch.repeat_interleave(
        externalInputs,
        repeats=self.numGenes,
        dim=1
    ).view(self.numSamples, self.numVariables, 1)
    # Result: (numSamples, numCells*numGenes, 1)
```

**What this does**: Each cell's voltage becomes an input to all gene variables in that cell.

#### Step 2: Voltage-to-Gene Activation (`updateState`)

```python
# Hill activation for Vmem input (line 279-280)
vmem_input = torch.exp(self.VmemGain) * self.tissueExternalInputs + self.VmemBias
vmem_hill_activation = 2 * hill_activation(vmem_input, K=0.5, n=2.0) - 1
```

**Voltage transformation**:
1. **Exponential gain**: `exp(VmemGain)` - learned parameter that amplifies/attenuates voltage influence
2. **Bias shift**: `VmemBias` - learned parameter that shifts the activation threshold
3. **Hill activation**: Sigmoidal response with half-saturation K=0.5, cooperativity n=2.0
4. **Rescaling**: `2 * sigmoid - 1` maps output to [-1, 1] range

#### Step 3: Integration into Gene Dynamics

```python
self.dstate = -self.state + \
              torch.matmul(self.W, gene_hill_activation) + \
              self.tissueVmemToGRNWeights * vmem_hill_activation + \
              0.5 * nc_regulation_flat
```

**Components**:
- `-self.state`: Decay term (genes naturally decay to zero)
- `torch.matmul(self.W, gene_hill_activation)`: Gene-gene interactions
- `self.tissueVmemToGRNWeights * vmem_hill_activation`: **Voltage influence** (weighted by learned matrix)
- `0.5 * nc_regulation_flat`: Neural crest GRN upstream control

**Key parameters**:
- `VmemToGRNWeights`: (numGenes, 1) - learned weights specifying which genes respond to voltage
- `VmemGain`: scalar - learned gain factor
- `VmemBias`: scalar - learned bias term

**Biological interpretation**:
- Voltage acts as a **global signaling cue** to all genes in a cell
- Each gene has a learned sensitivity to voltage (via `VmemToGRNWeights`)
- Similar to voltage-gated transcription factors in real biology

---

### B. FacialGRN (FacialGRN class)

**Location**: `geneRegulatoryNetwork.py:682-722, 577-608`

FacialGRN uses a more sophisticated, multi-channel transduction mechanism:

#### Step 1: Voltage Signal Preprocessing (`updateDynamicalParameters`)

```python
def updateDynamicalParameters(self, externalInputs=None):
    vmem = externalInputs  # shape: (numSamples, numCells, 1)
    vmem_grid = vmem.view(self.numSamples, self.numRows, self.numCols)

    # 1. Normalized voltage (0 to 1 range)
    vmin = vmem_grid.amin(dim=(1, 2), keepdim=True)
    vmax = vmem_grid.amax(dim=(1, 2), keepdim=True)
    norm = torch.clamp((vmem_grid - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)

    # 2. Ca²⁺-gating signal (depolarization detector)
    V_half = -0.04  # -40 mV activation threshold
    k_v = 0.008     # 8 mV voltage sensitivity
    ca_gate = torch.sigmoid((vmem_grid - V_half) / k_v)
    # Higher when DEPOLARIZED (Vmem > -40mV)

    # 3. Import signal (hyperpolarization detector)
    V_rest = -0.07   # -70 mV resting potential
    delta_v = 0.07   # 70 mV range
    import_signal = torch.clamp((V_rest - vmem_grid) / delta_v, 0.0, 1.0)
    # Higher when HYPERPOLARIZED (Vmem < -70mV)

    # Low-pass filtering for temporal smoothing (alpha = 0.8)
    self.voltage_input = 0.8 * self.voltage_input + 0.2 * norm
    self.voltage_ca = 0.8 * self.voltage_ca + 0.2 * ca_gate
    self.voltage_import = 0.8 * self.voltage_import + 0.2 * import_signal
```

**Three distinct voltage signals**:
1. **`voltage_input`**: Normalized voltage (spatial pattern)
2. **`voltage_ca`**: Depolarization-gated signal (mimics voltage-gated Ca²⁺ channels)
3. **`voltage_import`**: Hyperpolarization-activated signal (mimics ion import/HCN channels)

#### Step 2: Compute Voltage Detail (Local Contrast)

```python
# In update_genes() around line 578
detail_map = self.voltage_detail if self.voltage_detail is not None else (self.voltage_input - 0.5)
# detail_map: normalized deviation from mean (0.5)
```

This extracts **local features** from the voltage pattern rather than absolute values.

#### Step 3: Apply Voltage Signals to Genes (`update_genes`)

**A. Detail-based spatial patterning** (lines 577-591):

```python
# Convert detail into feature-specific drives
eye_drive = torch.clamp(-detail_map, min=0.0, max=1.0)      # High where Vmem is LOW (hyperpolarized)
jaw_drive = torch.clamp(detail_map, min=0.0, max=1.0)       # High where Vmem is HIGH (depolarized)
nose_drive = torch.clamp(1.0 - detail_map.abs(), min=0.0)   # High at INTERMEDIATE voltages
bone_drive = torch.clamp(0.7 - detail_map.abs(), min=0.0)   # High in FLAT/neutral regions

# Apply to specific genes (gain = 0.05 by default)
for gene in ['rx', 'six3', 'pax6', 'lhx2']:  # Eye genes
    self.grid[gene] += gain * (eye_drive - 0.5)

self.grid['alx'] += 0.5 * gain * (nose_drive - 0.5)        # Nose gene

self.grid['dlx'] += gain * (jaw_drive - 0.5)                # Jaw genes
self.grid['hand2'] += gain * (jaw_drive - 0.5)

self.grid['runx2'] += 0.3 * gain * (bone_drive - 0.5)       # Bone gene
```

**Biological interpretation**:
- **Eye genes** activated by **hyperpolarized** regions (mimics neural tissue)
- **Jaw genes** activated by **depolarized** regions (mimics mesenchymal/active tissue)
- **Nose genes** prefer **intermediate** voltages (boundary regions)
- **Bone genes** prefer **uniform/flat** regions (differentiated/stable tissue)

**B. Ca²⁺-gated modulation** (lines 596-600):

```python
if ca_map is not None:
    ca_drive = (ca_map - 0.5).clamp(-0.5, 0.5)  # Centered, range [-0.5, 0.5]
    ca_gain = 0.15  # default
    for gene in ['rx', 'six3', 'pax6', 'lhx2']:  # Eye genes only
        self.grid[gene] += ca_gain * ca_drive
```

**Effect**: Depolarization (high Ca²⁺ signal) **boosts eye genes**, mimicking Ca²⁺-dependent transcription factors.

**C. Import signal modulation** (lines 602-607):

```python
if import_map is not None:
    imp_drive = (import_map - 0.5).clamp(-0.5, 0.5)
    imp_gain = 0.1  # default
    self.grid['alx'] += imp_gain * imp_drive         # Nose
    self.grid['dlx'] += 0.5 * imp_gain * imp_drive   # Jaw
    self.grid['hand2'] += 0.5 * imp_gain * imp_drive
```

**Effect**: Hyperpolarization (high import signal) **boosts nose/jaw genes**, simulating hyperpolarization-activated ion import promoting ventral/lateral fates.

---

## 2. GRN → Electric Transduction (Reverse Direction)

**Location**: `cellularFieldNetwork.py:322-397`

### Mechanism: Gene Expression Modulates Voltage

```python
def apply_gene_voltage_feedback(self, gene_fields=None, gain=0.02):
    # Gene-to-voltage coupling weights
    weights = {
        # Eye genes → DEPOLARIZING (increase Vmem)
        'pax6': {'dep': 0.25},   # Strong depolarizing
        'six3': {'dep': 0.15},
        'lhx2': {'dep': 0.1},
        'rx': {'dep': 0.1},

        # Nose/jaw genes → POLARIZING (decrease Vmem)
        'alx': {'pol': 0.2},     # Nose
        'dlx': {'pol': 0.15},    # Jaw
        'hand2': {'pol': 0.25},  # Strong hyperpolarizing

        # Bone gene → weak polarizing
        'runx2': {'pol': 0.1},
    }

    # Compute net depolarizing and hyperpolarizing signals
    dep_signal = sum(weights[gene]['dep'] * gene_field for gene, gene_field if 'dep' in weights[gene])
    pol_signal = sum(weights[gene]['pol'] * gene_field for gene, gene_field if 'pol' in weights[gene])

    # Net voltage change
    net_signal = dep_signal - pol_signal
    delta_v = gain * torch.clamp(net_signal, -1.0, 1.0)

    # Update voltage
    self.Vmem = torch.clamp(self.Vmem + delta_v.unsqueeze(2), min=-0.2, max=0.1)
```

### Biological Interpretation

**Eye genes (pax6, six3, etc.) cause DEPOLARIZATION**:
- Express depolarizing ion channels (e.g., Na⁺ channels)
- Open gap junctions (increase connectivity)
- Mimic neural precursor states

**Jaw/nose genes (dlx, hand2, alx) cause HYPERPOLARIZATION**:
- Express polarizing ion channels (e.g., K⁺ channels, Cl⁻ channels)
- Close gap junctions (decrease connectivity)
- Mimic mesenchymal/non-neural states

**Gain parameter** (default 0.02):
- Controls strength of genetic feedback on voltage
- Low values preserve bioelectric pattern stability
- High values allow genes to dramatically reshape voltage landscape

---

## Summary: Complete Bidirectional Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    INITIAL STATE                            │
│  Stigmergic Model: Field-driven voltage pattern emerges    │
│  Vmem pattern has spatial structure (eyes, nose, jaw)      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Electric → GRN        │
        │  (Transduction)        │
        ├────────────────────────┤
        │ FacialGRN:             │
        │ • Detail extraction    │
        │ • Ca²⁺ gating          │
        │ • Import signal        │
        │                        │
        │ Generic GRN:           │
        │ • Hill activation      │
        │ • Learned weights      │
        └────────┬───────────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │  GRN Dynamics         │
     │  • Morphogen gradients│
     │  • Gene networks      │
     │  • Bioelectric guide  │
     └───────┬───────────────┘
             │
             ▼
   ┌─────────────────────┐
   │  GRN → Electric     │
   │  (Feedback)         │
   ├─────────────────────┤
   │ Gene expression:    │
   │ • pax6 → depolarize │
   │ • dlx → hyperpolarize│
   │ • Modulate G_dep/pol│
   └──────┬──────────────┘
          │
          ▼
  ┌───────────────────────┐
  │ Voltage Re-patterns   │
  │ • Ion conductances    │
  │   change              │
  │ • Gap junctions shift │
  │ • New Vmem pattern    │
  └──────┬────────────────┘
         │
         └──────► LOOP back to Electric→GRN
```

### Key Design Principles

1. **Separation of Timescales**:
   - Electric: fast (milliseconds), threshold-based
   - Genetic: slow (minutes), continuous dynamics

2. **Soft Coupling**:
   - Neither layer absolutely controls the other
   - Weak gains (0.02-0.15) allow both autonomy and influence
   - Mimics biological signal integration

3. **Multiple Channels**:
   - FacialGRN uses 3 distinct voltage signals (detail, Ca²⁺, import)
   - Generic GRN uses learned weight matrices
   - Allows rich, context-dependent responses

4. **Gene-Specific Effects**:
   - Different genes have different voltage sensitivities
   - Eye genes prefer hyperpolarization + high Ca²⁺
   - Jaw genes prefer depolarization + import signal
   - Mimics real developmental gene batteries

5. **Biological Plausibility**:
   - Ca²⁺-gated transcription (common in neurons)
   - Voltage-dependent ion channel expression
   - Morphogen gradients + bioelectric prepatterns
   - All grounded in developmental biology literature