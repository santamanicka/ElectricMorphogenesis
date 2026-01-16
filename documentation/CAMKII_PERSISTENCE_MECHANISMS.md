# CaMKII Pattern Persistence Mechanisms

## Overview

This document describes biologically realistic mechanisms for enabling long-term persistence of CaMKII spatial patterns beyond the initial bioelectric patterning phase (t > 2000). The current model successfully captures Vmem patterns at t=1000 into CaMKII patterns at t=2000, but these patterns decay afterward without additional memory mechanisms.

## Current Model Status

**Working**: Pattern capture (Vmem t=1000 → CaMKII t=2000)
**Limitation**: Pattern does not persist beyond t=2000
**Goal**: Implement biologically realistic long-term memory mechanisms

---

## Mechanism 1: CaMKII Autophosphorylation & Cooperative Activation ⭐

### Biological Basis

**Mechanism**: CaMKII exhibits true bistability through cooperative autophosphorylation. Active CaMKII phosphorylates neighboring subunits within the holoenzyme complex (12 subunits), creating positive feedback that sustains activity even after Ca²⁺ levels drop. Critically, inactive subunits can allosterically inhibit the complex, creating **competitive dynamics** between active and inactive states.

**Key Biology**:
- CaMKII holoenzyme: 12 subunits arranged in two hexameric rings
- Autophosphorylation at Thr286 makes CaMKII Ca²⁺-independent
- Hill coefficient n=2 (cooperative binding)
- **Competitive dynamics**: Active subunits promote activity, inactive subunits inhibit
- Once ~50% of subunits are active (critical threshold), the complex becomes self-sustaining
- Below threshold: inactive subunits dominate → active suppression of activation
- Above threshold: active subunits dominate → self-maintenance

**References**:
- Lisman et al., "The molecular basis of CaMKII function in synaptic and behavioural memory", Nature Reviews Neuroscience (2002)
- Miller & Kennedy, "Regulation of brain type II Ca2+/calmodulin-dependent protein kinase by autophosphorylation", J Biol Chem (1986)
- Stratton et al., "Structural studies on the regulation of Ca2+/calmodulin dependent protein kinase II", Current Opinion in Structural Biology (2014)

### Implementation: Competitive Dynamics (Recommended)

**CRITICAL**: Map self_activation to **[-1, 1]** instead of [0, 1] to create true bistability with active inhibition of background cells.

```python
class SimpleCaMKII:
    def __init__(self, grid_size, device='cpu', dtype=torch.float64, learned_params=None):
        # ... existing parameters ...

        # Bistability parameters
        self.K_half = torch.tensor(0.5, device=device, dtype=dtype)  # Half-saturation for Hill function

    def update(self, vmem_grid, dt=0.01):
        # ... existing Ca²⁺ dynamics ...

        # Hill function with competitive dynamics (n=2 cooperativity)
        CaMKII_sq = self.CaMKII_active * self.CaMKII_active
        K_half_sq = self.K_half * self.K_half

        # COMPETITIVE DYNAMICS: active fraction - inactive fraction
        # Maps to [-1, 1] instead of [0, 1]
        active_fraction = CaMKII_sq / (K_half_sq + CaMKII_sq)
        inactive_fraction = K_half_sq / (K_half_sq + CaMKII_sq)
        self_activation = active_fraction - inactive_fraction

        # Simplified form:
        # self_activation = (CaMKII_sq - K_half_sq) / (K_half_sq + CaMKII_sq)

        # Behavior:
        # CaMKII=0:     self_activation = -1  (strong inhibition)
        # CaMKII=K_half: self_activation = 0   (neutral, unstable equilibrium)
        # CaMKII=1:     self_activation = +1  (strong excitation)

        # Combined activation: Ca²⁺ signal + self-activation (in [-1, 1])
        ca_signal = torch.sigmoid((self.Ca - self.ca_threshold) / self.ca_sensitivity)
        combined_signal = or_sharpness * (ca_signal + self_activation - or_threshold)
        activation = torch.relu(combined_signal) / or_sharpness

        # CaMKII dynamics with competitive self-activation
        dCaMKII_dt = (self.k_activation * activation -
                      self.k_inactivation * self.CaMKII_active)
        # ... update and clamp ...
```

### Why [-1, 1] Mapping is Critical

**Problem with [0, 1] mapping**:
```python
# INCORRECT (original formulation):
self_activation = CaMKII_sq / (K_half_sq + CaMKII_sq)  # Range: [0, 1]

# At CaMKII=0 (background): self_activation = 0 (neutral)
# → Background cells get NO inhibition
# → Gradual accumulation can occur
# → Loss of spatial contrast over time
```

**Solution with [-1, 1] mapping**:
```python
# CORRECT (competitive formulation):
self_activation = (CaMKII_sq - K_half_sq) / (K_half_sq + CaMKII_sq)  # Range: [-1, 1]

# At CaMKII=0 (background): self_activation = -1 (strong inhibition)
# → Background cells actively resist activation
# → Sharp spatial boundaries maintained
# → Prevents gradual spreading
```

### Comparison: [0, 1] vs [-1, 1]

| Aspect | [0, 1] Mapping | [-1, 1] Mapping (Recommended) |
|--------|----------------|-------------------------------|
| **Background cells (CaMKII≈0)** | self_activation = 0 (neutral) | self_activation = -1 (inhibited) |
| **Feature cells (CaMKII≈1)** | self_activation = 1 (excited) | self_activation = +1 (excited) |
| **Threshold behavior** | Gradual transition | Sharp bistable switch |
| **Pattern maintenance** | ⚠ Gradual spreading/blurring | ✓ Sharp boundaries maintained |
| **Spatial contrast** | ⚠ Decays over time | ✓ Preserved indefinitely |
| **Biology** | Only models active state | ✓ Models active vs inactive competition |

### Biological Justification for Competitive Dynamics

**In CaMKII holoenzymes:**

1. **Inactive subunits** (unphosphorylated):
   - Occupy space in the holoenzyme ring
   - Block access of Ca²⁺/CaM to neighboring subunits
   - Allosterically reduce catalytic activity
   - **Net effect**: Inhibition proportional to inactive fraction

2. **Active subunits** (phosphorylated at Thr286):
   - Trans-autophosphorylate neighboring subunits
   - Enhance catalytic efficiency allosterically
   - Maintain open conformation
   - **Net effect**: Activation proportional to active fraction

3. **Competition**:
   - At low phosphorylation: inactive > active → net inhibition
   - At high phosphorylation: active > inactive → net excitation
   - At K_half (50%): balanced → unstable equilibrium

This creates a **molecular switch** with hysteresis: once switched ON, it stays ON; once OFF, it stays OFF.

### Mathematical Analysis

The competitive formulation creates three fixed points in the dynamics:

```
dCaMKII/dt = 0  when:
  k_on * activation(CaMKII) = k_off * CaMKII

Where activation depends on self_activation(CaMKII):
  self_activation = (CaMKII² - K_half²) / (K_half² + CaMKII²)
```

**Fixed points** (with appropriate parameters):
1. **LOW state**: CaMKII ≈ 0 (stable) - background
2. **MIDDLE**: CaMKII ≈ K_half (unstable) - threshold
3. **HIGH state**: CaMKII ≈ 1 (stable) - features

The [-1, 1] mapping ensures the LOW state is **attracting** (not just neutral), creating robust bistability.

### Effect on Spatial Patterns

**With [0, 1] (original)**:
```
Background cell: CaMKII=0.01, self_activation=0.0004 (≈0)
→ Needs: ca_signal > 0.5 to activate
→ Weak noise can cause gradual drift upward
→ Pattern boundaries blur over time
```

**With [-1, 1] (competitive)**:
```
Background cell: CaMKII=0.01, self_activation=-0.96 (≈-1)
→ Needs: ca_signal > 1.5 to activate (very high!)
→ Strongly rejects activation
→ Pattern boundaries stay sharp
```

### Timescale & Parameters

- **Memory duration**: 100-1000 steps after Ca²⁺ removal
- **K_half = 0.5**: Half-maximal self-activation at 50% CaMKII activity
- **Threshold for bistability**: Need k_activation / k_inactivation > 5

### Advantages

✅ Strong experimental support
✅ Simple to implement (no additional state variables)
✅ True bistability (two stable fixed points)
✅ Commonly used in computational neuroscience models

### Limitations

⚠️ Memory is finite (~100-1000 steps)
⚠️ Requires continuous low-level Ca²⁺ or strong initial activation

---

## Mechanism 2: Chromatin/Epigenetic Memory ⭐⭐

### Biological Basis

**Mechanism**: Ca²⁺/CaMKII signaling triggers chromatin remodeling and epigenetic modifications (histone acetylation, DNA methylation) that persist for very long timescales. Open chromatin maintains accessibility of genes required for sustained CaMKII expression and activity.

**Key Biology**:
- CaMKII activates CREB and MEF2 transcription factors
- CREB recruits CBP/p300 (histone acetyltransferases)
- MEF2 recruits SWI/SNF (chromatin remodelers)
- Modified chromatin persists for hours to days (>10,000 timesteps)

**References**:
- West et al., "Calcium regulation of neuronal gene expression", PNAS (2001)
- Flavell & Greenberg, "Signaling mechanisms linking neuronal activity to gene expression and plasticity", Annual Review of Neuroscience (2008)

### Implementation

```python
class SimpleCaMKII:
    def __init__(self, grid_size, device='cpu', dtype=torch.float64, learned_params=None):
        # ... existing parameters ...

        # Chromatin state (0 = closed, 1 = open)
        self.chromatin_state = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)

        # Chromatin dynamics parameters
        defaults = {
            'k_chromatin_open': 0.001,   # Slow opening (tau_open ~ 1000)
            'k_chromatin_close': 0.0001, # Very slow closing (tau_close ~ 10,000)
        }
        # ... apply learned_params overrides ...

        self.k_chromatin_open = torch.tensor(defaults['k_chromatin_open'], device=device, dtype=dtype)
        self.k_chromatin_close = torch.tensor(defaults['k_chromatin_close'], device=device, dtype=dtype)

    def update(self, vmem_grid, dt=0.01):
        # ... existing Ca²⁺ and CaMKII dynamics ...

        # Chromatin state evolution (slow)
        dChromatin_dt = (self.k_chromatin_open * self.CaMKII_active * (1 - self.chromatin_state) -
                         self.k_chromatin_close * self.chromatin_state)
        self.chromatin_state = self.chromatin_state + dt * dChromatin_dt
        self.chromatin_state = torch.clamp(self.chromatin_state, min=0.0, max=1.0)

        # Chromatin feeds back to sustain CaMKII activation
        chromatin_boost = self.chromatin_state * 0.3  # 30% boost from open chromatin
        ca_signal = torch.sigmoid((self.Ca - self.ca_threshold) / self.ca_sensitivity)
        combined_activation = ca_signal + chromatin_boost

        # CaMKII dynamics
        dCaMKII_dt = (self.k_activation * combined_activation * (1 - self.CaMKII_active) -
                      self.k_inactivation * self.CaMKII_active)
        # ...

        return {
            'Ca': self.Ca,
            'CaMKII': self.CaMKII_active,
            'chromatin': self.chromatin_state,  # Track chromatin state
            'vmem': vmem_grid
        }
```

### Timescale & Parameters

- **Opening timescale**: tau_open = 1/k_chromatin_open ≈ 1000 steps
- **Closing timescale**: tau_close = 1/k_chromatin_close ≈ 10,000 steps
- **Memory duration**: 5,000-10,000+ steps after CaMKII activation

### Advantages

✅ Very long-term memory (>10,000 steps)
✅ Biologically well-documented
✅ Can be made essentially permanent (k_chromatin_close → 0)
✅ Provides "molecular memory" layer

### Limitations

⚠️ Requires additional state variable (chromatin_state)
⚠️ More parameters to learn/tune
⚠️ Chromatin state not directly observable in experiments

---

## Mechanism 3: Transcriptional Feedback Loops

### Biological Basis

**Mechanism**: CaMKII activates transcription factors (CREB, c-Fos, c-Jun) that drive expression of genes maintaining the active state. These include growth factors (BDNF, NGF), neurotrophic receptors, and CaMKII itself.

**Key Biology**:
- CaMKII → CREB → immediate early genes (IEG: c-Fos, c-Jun)
- IEGs → growth factors (BDNF, NGF)
- BDNF/NGF → TrkB/TrkA receptors → sustained signaling
- Positive feedback: growth factors maintain CaMKII expression

**References**:
- Deisseroth et al., "Signaling from synapse to nucleus: postsynaptic CREB phosphorylation", Neuron (2003)
- Ghosh et al., "Calcium signaling in neurons: molecular mechanisms and cellular consequences", Science (1994)

### Implementation

```python
class SimpleCaMKII:
    def __init__(self, grid_size, device='cpu', dtype=torch.float64, learned_params=None):
        # ... existing parameters ...

        # Transcription factor concentration
        self.TF_conc = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)

        defaults = {
            'k_transcribe': 0.01,      # TF production rate
            'k_degrade_TF': 0.001,     # TF degradation (tau ~ 1000)
            'tf_feedback_strength': 0.5, # How much TF boosts CaMKII activation
        }
        # ... apply learned_params overrides ...

    def update(self, vmem_grid, dt=0.01):
        # ... existing Ca²⁺ dynamics ...

        # Transcription factor dynamics
        dTF_dt = (self.k_transcribe * self.CaMKII_active -
                  self.k_degrade_TF * self.TF_conc)
        self.TF_conc = self.TF_conc + dt * dTF_dt
        self.TF_conc = torch.clamp(self.TF_conc, min=0.0)

        # TF feedback to CaMKII activation
        tf_feedback = torch.sigmoid((self.TF_conc - 0.3) / 0.1)
        ca_signal = torch.sigmoid((self.Ca - self.ca_threshold) / self.ca_sensitivity)
        combined_activation = ca_signal + self.tf_feedback_strength * tf_feedback

        # CaMKII dynamics
        dCaMKII_dt = (self.k_activation * combined_activation * (1 - self.CaMKII_active) -
                      self.k_inactivation * self.CaMKII_active)
        # ...

        return {
            'Ca': self.Ca,
            'CaMKII': self.CaMKII_active,
            'TF': self.TF_conc,
            'vmem': vmem_grid
        }
```

### Timescale & Parameters

- **TF production timescale**: tau_produce ≈ 100 steps
- **TF degradation timescale**: tau_degrade ≈ 1000 steps
- **Memory duration**: 500-2000 steps after CaMKII activation

### Advantages

✅ Intermediate memory duration (1000-2000 steps)
✅ Models protein synthesis explicitly
✅ Can couple to gene regulatory networks
✅ Testable with transcriptomics

### Limitations

⚠️ Requires additional state variable (TF_conc)
⚠️ More parameters to tune
⚠️ Memory is still finite (eventually decays)

---

## Mechanism 4: Gap Junction Remodeling 🔌

### Biological Basis

**Mechanism**: CaMKII phosphorylates connexin proteins, modulating gap junction formation and electrical coupling between cells. Cells with high CaMKII form stronger connections, stabilizing electrical domains.

**Key Biology**:
- CaMKII phosphorylates connexin-43 (Cx43) at Ser368
- Phosphorylation enhances gap junction assembly and conductance
- Creates positive feedback: synchronized cells maintain synchronization
- Well-documented in cardiac and neural tissue

**References**:
- Lampe & Lau, "The effects of connexin phosphorylation on gap junctional communication", Int J Biochem Cell Biol (2004)
- Beardslee et al., "Dephosphorylation and intracellular redistribution of ventricular connexin43", Circ Res (2000)

### Implementation

```python
class SimpleCaMKII:
    def __init__(self, grid_size, device='cpu', dtype=torch.float64, learned_params=None):
        # ... existing parameters ...

        # Gap junction modulation strength
        defaults = {
            'gj_modulation_strength': 2.0,  # How much CaMKII boosts gap junctions
            'gj_threshold': 0.5,            # CaMKII level for GJ enhancement
        }
        # ... apply learned_params overrides ...

    def compute_gap_junction_modulation(self):
        """
        Compute gap junction strength modulation based on CaMKII state.
        Returns modulation factor (1.0 = baseline, >1 = enhanced coupling)
        """
        # Cells with high CaMKII have stronger gap junctions
        gj_modulation = 1.0 + self.gj_modulation_strength * (self.CaMKII_active > self.gj_threshold).float()
        return gj_modulation
```

**Coupling to bioelectric network:**
```python
# In cellularFieldNetwork or embryo model:
# Get gap junction modulation from CaMKII state
gj_modulation = camkii_tracker.compute_gap_junction_modulation()

# Modify adjacency matrix (element-wise)
# Each cell's coupling to neighbors is scaled by its CaMKII state
adjacency_modulated = adjacency_matrix * gj_modulation.flatten().unsqueeze(1)

# Use modulated adjacency in electric field computation
# This requires integrating CaMKII back into the bioelectric model
```

### Timescale & Parameters

- **GJ assembly timescale**: 100-500 steps (connexin trafficking)
- **Memory duration**: Indefinite (as long as electrical domain persists)
- **Threshold**: Typically CaMKII > 0.5 for GJ enhancement

### Advantages

✅ Directly couples back to bioelectric network
✅ Creates spatial domain stability
✅ No new state variables needed
✅ Well-documented in cardiac tissue

### Limitations

⚠️ Requires bidirectional coupling to bioelectric model
⚠️ More complex integration with existing codebase
⚠️ Can create runaway positive feedback if not tuned carefully

---

## Mechanism 5: Metabolic Switching ⚡

### Biological Basis

**Mechanism**: CaMKII enters mitochondria and enhances oxidative phosphorylation, creating energetically distinct cellular states. High-ATP cells sustain higher CaMKII activity through positive feedback.

**Key Biology**:
- Mitochondrial CaMKII (mtCaMKII) enhances OXPHOS
- Ca²⁺ entry into mitochondria via CaMKII-gated channels
- High ATP/ADP ratio sustains cellular signaling
- Creates bistable "high energy" vs "low energy" states

**References**:
- Joiner et al., "CaMKII determines mitochondrial stress responses in heart", Nature (2012)
- Maier & Bers, "Calcium, calmodulin, and calcium-calmodulin kinase II: heartbeat to heartbeat and beyond", J Mol Cell Cardiol (2002)

### Implementation

```python
class SimpleCaMKII:
    def __init__(self, grid_size, device='cpu', dtype=torch.float64, learned_params=None):
        # ... existing parameters ...

        # Metabolic state (normalized ATP level, 0-1)
        self.ATP_state = torch.ones(grid_size, grid_size, device=device, dtype=dtype) * 0.5

        defaults = {
            'k_ATP_produce': 0.005,      # ATP production rate from CaMKII
            'k_ATP_consume': 0.002,      # Basal ATP consumption
            'atp_boost_strength': 0.4,   # How much ATP sustains CaMKII
        }
        # ... apply learned_params overrides ...

    def update(self, vmem_grid, dt=0.01):
        # ... existing Ca²⁺ dynamics ...

        # Metabolic state dynamics
        dATP_dt = (self.k_ATP_produce * self.CaMKII_active -
                   self.k_ATP_consume * (1 - self.CaMKII_active))
        self.ATP_state = self.ATP_state + dt * dATP_dt
        self.ATP_state = torch.clamp(self.ATP_state, min=0.0, max=1.0)

        # High ATP boosts CaMKII activation
        atp_boost = torch.relu(self.ATP_state - 0.5) * self.atp_boost_strength
        ca_signal = torch.sigmoid((self.Ca - self.ca_threshold) / self.ca_sensitivity)
        combined_activation = ca_signal + atp_boost

        # CaMKII dynamics
        dCaMKII_dt = (self.k_activation * combined_activation * (1 - self.CaMKII_active) -
                      self.k_inactivation * self.CaMKII_active)
        # ...

        return {
            'Ca': self.Ca,
            'CaMKII': self.CaMKII_active,
            'ATP': self.ATP_state,
            'vmem': vmem_grid
        }
```

### Timescale & Parameters

- **ATP production timescale**: tau_produce ≈ 200 steps
- **ATP depletion timescale**: tau_consume ≈ 500 steps
- **Memory duration**: 1000-2000 steps after CaMKII activation

### Advantages

✅ Creates metabolic memory
✅ Couples to cellular energy state
✅ Relatively simple to implement
✅ Explains metabolic differences in patterned tissues

### Limitations

⚠️ Less direct experimental support for this mechanism
⚠️ Requires additional state variable (ATP_state)
⚠️ Memory is still finite

---

## Mechanism 6: Protein Scaffolding/Clustering 🧬

### Biological Basis

**Mechanism**: Active CaMKII forms stable clusters with cytoskeletal proteins (actin, spectrin) or postsynaptic scaffolds (PSD-95 in neurons). Once clustered, CaMKII becomes resistant to dephosphorylation.

**Key Biology**:
- CaMKII binds to F-actin via multiple subunit-actin interactions
- Binding to actin inhibits phosphatase access
- PSD-95 scaffolds concentrate CaMKII in postsynaptic densities
- Clustered CaMKII has reduced turnover

**References**:
- Lisman & Zhabotinsky, "A model of synaptic memory: a CaMKII/PP1 switch that potentiates transmission", Neuron (2001)
- Okamoto et al., "Rapid and persistent modulation of actin dynamics regulates postsynaptic reorganization", Nature Neuroscience (2004)

### Implementation

```python
class SimpleCaMKII:
    def __init__(self, grid_size, device='cpu', dtype=torch.float64, learned_params=None):
        # ... existing parameters ...

        # Locked state (boolean: once locked, CaMKII doesn't decay)
        self.CaMKII_locked = torch.zeros(grid_size, grid_size, device=device, dtype=torch.bool)

        # Counter for sustained high activation
        self.lock_counter = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)

        defaults = {
            'lock_threshold': 0.7,       # CaMKII level to trigger locking
            'lock_duration': 100,        # Steps at high CaMKII to lock
        }
        # ... apply learned_params overrides ...

    def update(self, vmem_grid, dt=0.01):
        # ... existing Ca²⁺ dynamics ...

        # Update lock counter
        self.lock_counter += (self.CaMKII_active > self.lock_threshold).float()
        self.lock_counter *= (self.CaMKII_active > self.lock_threshold).float()  # Reset if drops below threshold

        # Lock cells that have been highly active for long enough
        newly_locked = (self.lock_counter > self.lock_duration) & (~self.CaMKII_locked)
        self.CaMKII_locked |= newly_locked

        # Locked CaMKII has zero inactivation
        effective_k_inactivation = self.k_inactivation * (~self.CaMKII_locked).float()

        # CaMKII dynamics with conditional inactivation
        ca_signal = torch.sigmoid((self.Ca - self.ca_threshold) / self.ca_sensitivity)
        dCaMKII_dt = (self.k_activation * ca_signal * (1 - self.CaMKII_active) -
                      effective_k_inactivation * self.CaMKII_active)
        # ...

        return {
            'Ca': self.Ca,
            'CaMKII': self.CaMKII_active,
            'locked': self.CaMKII_locked.float(),  # Track which cells are locked
            'vmem': vmem_grid
        }
```

### Timescale & Parameters

- **Lock threshold**: CaMKII > 0.7
- **Lock duration**: 100 steps of sustained high activation
- **Memory duration**: Essentially permanent (locked state doesn't decay)

### Advantages

✅ Essentially permanent memory
✅ Threshold-based (digital-like)
✅ Models irreversible structural changes
✅ Explains persistent synaptic potentiation

### Limitations

⚠️ Irreversible (no way to unlock)
⚠️ Requires boolean state tracking
⚠️ May be too rigid for developmental morphogenesis

---

## Recommended Implementation Strategy

### Option A: Dual-Mechanism (Short + Long Term Memory) ⭐⭐⭐ **RECOMMENDED**

Combine **Mechanism 1 (Autophosphorylation)** + **Mechanism 2 (Chromatin Memory)**:

**Rationale**:
- Autophosphorylation provides immediate bistability (100-1000 steps)
- Chromatin memory provides long-term persistence (5,000-10,000+ steps)
- Both are well-supported by experimental data
- Together they cover all timescales

**Implementation**:
```python
class SimpleCaMKII:
    def __init__(self, grid_size, device='cpu', dtype=torch.float64, learned_params=None):
        # ... existing parameters ...

        # Autophosphorylation bistability
        self.K_half = torch.tensor(0.5, device=device, dtype=dtype)

        # Chromatin state
        self.chromatin_state = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)
        self.k_chromatin_open = torch.tensor(0.001, device=device, dtype=dtype)
        self.k_chromatin_close = torch.tensor(0.0001, device=device, dtype=dtype)

    def update(self, vmem_grid, dt=0.01):
        # 1. Ca²⁺ dynamics (existing)
        # ...

        # 2. Hill function self-activation with competitive dynamics (short-term memory)
        # CRITICAL: Use [-1, 1] mapping, not [0, 1]
        CaMKII_sq = self.CaMKII_active * self.CaMKII_active
        K_half_sq = self.K_half * self.K_half

        # Competitive dynamics: active - inactive
        self_activation = (CaMKII_sq - K_half_sq) / (K_half_sq + CaMKII_sq)
        # Range: [-1, 1]
        # Background cells (CaMKII≈0): self_activation ≈ -1 (inhibited)
        # Feature cells (CaMKII≈1): self_activation ≈ +1 (excited)

        # 3. Chromatin state evolution (long-term memory)
        dChromatin_dt = (self.k_chromatin_open * self.CaMKII_active * (1 - self.chromatin_state) -
                         self.k_chromatin_close * self.chromatin_state)
        self.chromatin_state = self.chromatin_state + dt * dChromatin_dt
        self.chromatin_state = torch.clamp(self.chromatin_state, min=0.0, max=1.0)

        # 4. Combined activation: Ca²⁺ + self + chromatin
        ca_signal = torch.sigmoid((self.Ca - self.ca_threshold) / self.ca_sensitivity)
        chromatin_boost = self.chromatin_state * 0.3

        # With self_activation in [-1, 1], need to adjust threshold
        combined_signal = or_sharpness * (ca_signal + self_activation + chromatin_boost - or_threshold)
        activation = torch.relu(combined_signal) / or_sharpness

        # 5. CaMKII dynamics
        dCaMKII_dt = (self.k_activation * activation * (1 - self.CaMKII_active) -
                      self.k_inactivation * self.CaMKII_active)
        self.CaMKII_active = self.CaMKII_active + dt * dCaMKII_dt
        self.CaMKII_active = torch.clamp(self.CaMKII_active, min=0.0, max=1.0)

        return {
            'Ca': self.Ca,
            'CaMKII': self.CaMKII_active,
            'chromatin': self.chromatin_state,
            'vmem': vmem_grid
        }
```

### Option B: Simple Autophosphorylation Only

Use only **Mechanism 1** if you need simpler implementation and moderate memory (100-1000 steps).

### Option C: Permanent Locking

Use **Mechanism 6 (Scaffolding)** if you need essentially permanent patterns after development.

---

## Parameter Learning Considerations

### Learnable Parameters for Dual-Mechanism Model

```python
# Autophosphorylation (Mechanism 1)
- K_half: [0.3, 0.7]  # Half-saturation for Hill function

# Chromatin dynamics (Mechanism 2)
- k_chromatin_open: [0.0001, 0.01]   # Opening rate
- k_chromatin_close: [0.00001, 0.001] # Closing rate (keep < k_open)
- chromatin_boost_strength: [0.1, 0.5] # How much chromatin sustains CaMKII
```

### Testing Strategy

1. **Baseline test**: Run existing model to t=5000, observe decay
2. **Add autophosphorylation**: Expect persistence to t=2000-3000
3. **Add chromatin**: Expect persistence to t=5000-10000+
4. **Tune parameters**: Adjust k_chromatin_close for desired memory duration

### Expected Behavior

| Mechanism | Memory Duration | Pattern Quality at t=5000 |
|-----------|-----------------|---------------------------|
| Current (Ca²⁺ only) | 0-100 steps | ✗ Lost |
| + Autophosphorylation | 100-1000 steps | ⚠ Degraded |
| + Chromatin | 5000-10000 steps | ✓ Maintained |
| + Scaffolding | Permanent | ✓✓ Locked |

---

## Testing and Visualization Updates

### Required Changes to test_camkii_bistability.py

1. Add chromatin state visualization
2. Track chromatin state in checkpoints
3. Add chromatin time series to plots
4. Report chromatin activation levels

### Suggested Visualization Layout

```
Row 1: Vmem at t=early, t=1000 (lock), t=mid, t=final
Row 2: CaMKII at t=early, t=1000, t=mid, t=final
Row 3: Chromatin at t=early, t=1000, t=mid, t=final  [NEW]
Row 4: Time series + correlation analysis
```

---

## Future Directions

### Extensions

1. **Combinatorial mechanisms**: Test autophosphorylation + chromatin + transcriptional feedback together
2. **Spatial coupling**: Implement gap junction remodeling for domain stabilization
3. **Stochasticity**: Add noise to test robustness of memory mechanisms
4. **Reversibility**: Add "erasure" mechanisms (phosphatases, chromatin remodelers)

### Biological Validation

Test predictions:
- Does CaMKII pattern persist when Ca²⁺ channels are blocked after t=1000?
- Do histone marks (H3K27ac) appear in pattern-matched domains?
- Does chromatin remodeler inhibition (BRG1 knockout) prevent pattern persistence?

---

## References

1. Lisman, J., Schulman, H., & Cline, H. (2002). The molecular basis of CaMKII function in synaptic and behavioural memory. *Nature Reviews Neuroscience*, 3(3), 175-190.

2. West, A. E., Chen, W. G., Dalva, M. B., et al. (2001). Calcium regulation of neuronal gene expression. *PNAS*, 98(20), 11024-11031.

3. Deisseroth, K., Mermelstein, P. G., Xia, H., & Tsien, R. W. (2003). Signaling from synapse to nucleus: the logic behind the mechanisms. *Current Opinion in Neurobiology*, 13(3), 354-365.

4. Lampe, P. D., & Lau, A. F. (2004). The effects of connexin phosphorylation on gap junctional communication. *International Journal of Biochemistry & Cell Biology*, 36(7), 1171-1186.

5. Joiner, M. L. A., Koval, O. M., Li, J., et al. (2012). CaMKII determines mitochondrial stress responses in heart. *Nature*, 491(7423), 269-273.

6. Lisman, J. E., & Zhabotinsky, A. M. (2001). A model of synaptic memory: a CaMKII/PP1 switch that potentiates transmission by organizing an AMPA receptor anchoring assembly. *Neuron*, 31(2), 191-201.

7. Miller, S. G., & Kennedy, M. B. (1986). Regulation of brain type II Ca2+/calmodulin-dependent protein kinase by autophosphorylation: a Ca2+-triggered molecular switch. *Cell*, 44(6), 861-870.

8. Flavell, S. W., & Greenberg, M. E. (2008). Signaling mechanisms linking neuronal activity to gene expression and plasticity of the nervous system. *Annual Review of Neuroscience*, 31, 563-590.

---

## Quick Reference: Implementation Checklist

To implement **Option A (Dual-Mechanism - Recommended)**:

- [ ] Add `K_half` parameter (autophosphorylation)
- [ ] Add `chromatin_state` tensor
- [ ] Add `k_chromatin_open`, `k_chromatin_close` parameters
- [ ] Implement Hill function self-activation
- [ ] Implement chromatin dynamics
- [ ] Combine Ca²⁺ + self + chromatin signals
- [ ] Update visualization to show chromatin state
- [ ] Test persistence to t=5000-10000
- [ ] Tune `k_chromatin_close` for desired memory duration
- [ ] Update documentation with learned parameter values

---

**Document Version**: 1.0
**Date**: 2025-12-05
**Author**: AI Assistant
**Status**: Ready for implementation
