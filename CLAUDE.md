# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements bioelectric field models for developmental morphogenesis, integrating:
- **Bioelectric networks** (cellular field networks with ion channels and gap junctions)
- **Gene regulatory networks** (GRN) including hierarchical Neural Crest GRN and FacialGRN variants
- **Stigmergic/Mosaic models** for pattern formation via electric fields
- **Bidirectional coupling** between bioelectric patterns and gene expression
- **Dual-driver architecture** combining morphogen gradients and bioelectric signals

The system simulates how voltage membrane (Vmem) patterns emerge and interact with genetic networks to coordinate tissue-level morphogenesis, with a particular focus on craniofacial development. Two main architectural approaches exist: the original FacialGRN (stigmergic integration) and the newer RefinedFacialGRN (dual bioelectric-morphogen drivers).

## Core Architecture

### Three-Layer Model Structure

1. **Cellular Field Network** (`cellularFieldNetwork.py`)
   - Models multicellular bioelectric network with ion channels (polarizing/depolarizing) and gap junctions
   - Implements electric field computation and transduction
   - Supports voltage-dependent gene modulation
   - Key parameters: `G_pol`, `G_dep` (ion channel conductances), `G_0` (gap junction strength)

2. **Gene Regulatory Network** (`geneRegulatoryNetwork.py`)
   - Two-tier hierarchy: `NeuralCrestGRN` (upstream developmental control) and generic GRN (downstream)
   - `FacialGRN`: specialized craniofacial patterning with genes (Pax6, Sox9, Dlx, Hand2, etc.)
   - Uses Hill activation functions and cooperative binding dynamics
   - Can receive bioelectric prepattern as spatial constraints

3. **Embryo Model** (`embryo.py`)
   - Top-level orchestrator managing bioelectric ↔ GRN interactions
   - Handles ligand diffusion, ATP dynamics (optional), and face coupling
   - Coordinates between `cellularFieldNetwork` and `geneRegulatoryNetwork` instances

### Key Integration Components

- **FacePatternCoordinator** (`facePatternCoordinator.py`): Derives facial feature masks (eye/nose/jaw/bone) from Vmem snapshots and translates them into gene expression set points
- **BioelectricTransduction** (`bioelectricTransduction.py`): Converts Vmem into gene-regulatory signals via Ca²⁺ dynamics with temporal integration
- **CaMKII Bistability** (`learn_camkii_bistability.py`, `test_camkii_bistability.py`): Bistable pattern maintenance system with competitive self-activation dynamics
  - Vmem → Ca²⁺ → CaMKII signaling cascade with voltage-gated Ca²⁺ channels and temporal integration
  - Competitive dynamics: CaMKII self-activation (positive feedback) vs self-inhibition (negative feedback), mapped to [-1, 1] range
  - OR gate logic: combines external Ca²⁺ drive (gain-modulated sigmoid) with competitive self-activation via additive threshold
  - Ca²⁺ dynamics: `dCa/dt = I_ca - (1/tau_ca)*Ca - k_decay_ca`, where k_decay_ca is learnable constant baseline consumption
  - Enables pattern formation during Vmem phase and autonomous bistable maintenance after Ca²⁺ decays to low levels (~K_half)
- **RefinedFacialGRN** (`refinedFacialGRN.py`): Dual-driver architecture where genes respond to both morphogen gradients (SHH, FGF8, EDN1) and bioelectric signals (Ca²⁺)
- **GeneBasedFeatureClassifier** (`geneBasedFeatureClassifier.py`): Classifies facial features (bone/eye/nose/mouth) from gene expression patterns
- **Stigmergic Integration** (`run_stigmergic_facial_integration.py`): Original pipeline: electric pattern → face mask → GRN seeding → bidirectional feedback
- **Refined Integration** (`run_refined_facial_integration.py`): Newer pipeline using dual-driver architecture with morphogen gradients and bioelectric gating
- **CaMKII-Integrated Facial GRN** (`camkiiFacialGRN.py`): Newest architecture with concurrent CaMKII + GRN dynamics
  - `CaMKIIBistableSwitch`: Implements bistable pattern memory via competitive self-activation
  - `CaMKIIFacialGRN`: Combines morphogen gradients + CaMKII-derived bio_gate + gene expression
  - bio_gate comes from CaMKII activity (persistent, bistable) rather than raw Ca²⁺ (transient)
  - Pattern persists after Vmem decay through CaMKII bistable memory

## Common Development Commands

### Running Simulations

```bash
# Quick test of trained Stigmergic or Mosaic models
# Edit Model variable in file to "Stigmergic" or "Mosaic"
python simulateTrainedModel.py

# Custom configuration (edit parameters in script first)
python simulateCellularFieldNetwork.py

# Full Stigmergic + FacialGRN integration (original architecture)
python run_stigmergic_facial_integration.py

# Refined dual-driver facial integration (newer architecture)
python run_refined_facial_integration.py

# CaMKII-integrated facial integration (concurrent CaMKII + GRN dynamics)
python run_camkii_facial_integration.py
python run_camkii_facial_integration.py --camkii-params data/bestLearnedCaMKIIParams_0.dat

# Autonomous morphogen-only GRN (no bioelectric coupling)
python run_autonomous_morphogen_grn.py

# Embryo network pipeline
bash runSimulateEmbryoNetwork.sh
```

### Learning/Training

```bash
# Learn cellular field network parameters
bash runLearnCellularFieldNetwork.sh

# Or direct invocation with arguments
python learnCellularFieldNetwork.py --latticeDims "(11,11)" --fieldEnabled True --numLearnIters 100

# Learn refined facial integration parameters (dual-driver architecture)
bash runLearnRefinedFacialIntegration.sh

# Or direct invocation with custom parameters
python learnRefinedFacialIntegration.py --gridSize 11 --numGRNIters 5000 --numLearnIters 100 --lr 0.02 \
  --learnedParameters "['ca_threshold','ca_sensitivity','and_threshold']"

# Learn CaMKII bistability parameters with competitive dynamics
bash runLearnCaMKIIBistability.sh

# Or direct invocation with custom parameters
python learn_camkii_bistability.py --grid_size 11 --num_iters 3000 --num_learn_iters 100 \
  --target_type "ring"  # Options: "ring", "stripe", "checkerboard"

# Learn CaMKII-integrated facial patterning (concurrent CaMKII + GRN)
python learnCaMKIIFacialIntegration.py --numLearnIters 50 --lr 0.01

# With fixed CaMKII parameters (learn only GRN gating)
python learnCaMKIIFacialIntegration.py --fixedCaMKIIParams data/bestLearnedCaMKIIParams_0.dat
```

### Analysis and Visualization

```bash
# Plot analysis data from parameter sweeps
bash runPlotAnalysisData.sh
python plotAnalysisData.py

# Analyze parameter sweeps
python analyzeCellularFieldNetwork.py
python analyzeCellularFieldNetworkParameterSweep.py
```

### Testing

```bash
# Quick smoke tests for FacialGRN
python testFacialGRN.py                    # generates facial_grn_visualization.png
python testFacialGRN_compatibility.py      # prints interface checks
python quickTestFacialGRN.py

# Face-coupled demo
python run_face_coupled_demo.py

# CaMKII bistability tests
python test_camkii_bistability.py          # test bistable dynamics with learned parameters
python diagnose_camkii_dynamics.py         # diagnostic analysis of Ca²⁺ and CaMKII evolution

# Diagnostic scripts
python diagnose_eye_labels.py              # check eye feature detection
python diagnose_coupling_evolution.py      # trace bioelectric-GRN coupling dynamics
```

## Parameter Files

Models are seeded by `.dat` files in `data/`:
- `data/StigmergicModelParameters.dat`: Stigmergic model configuration
- `data/MosaicModelParameters.dat`: Mosaic model configuration
- `data/bestModelParameters_*.dat`: Learned bioelectric model parameters from training runs
- `data/bestLearnedFacialParams_*.dat`: Learned facial integration parameters (refined dual-driver model)
- `data/bestLearnedCaMKIIParams_*.dat`: Learned CaMKII bistability parameters with competitive dynamics
- `data/bestLearnedCaMKIIFacialParams_*.dat`: Learned CaMKII-integrated facial patterning parameters (concurrent dynamics)

### Standard Parameter File Structure

These files contain dictionaries with keys:
- `latticeDims`: tissue grid size (rows, cols)
- `GJParameters`: gap junction settings
- `fieldParameters`: bioelectric field configuration (enabled, resolution, strength, transduction weights)
- `GRNParameters`: gene network weights, biases, timescales
- `ligandParameters`: diffusion and gating parameters
- `simParameters`: initial values, external inputs, iteration counts
- `clampParameters`: voltage clamping indices and values
- `trainParameters`: target patterns, loss methods, learning rates

### Refined Facial Parameters File Structure

`bestLearnedFacialParams_*.dat` files contain learned parameters for the dual-driver architecture:
- Bioelectric gating: `ca_threshold`, `ca_sensitivity`, `and_threshold`, `and_sharpness`
- Morphogen gradients: `shh_decay_length`, `fgf8_decay_length`, `edn1_decay_length`, source strengths
- Gene dynamics: `k_activation`, `k_degradation`, Hill function parameters
- Loss and iteration metadata: `best_loss`, `best_iteration`

**Example learned values** (from `bestLearnedFacialParams_130.dat`):
```
Learned (gating) parameters:
  ca_threshold:   0.52    (range 0.0-1.0)
  ca_sensitivity: 0.03    (range 0.01-0.1)
  and_threshold:  1.18    (range 1.0-1.5)
  and_sharpness:  17.35   (range 10.0-25.0)

Fixed GRN parameters:
  k_activation:   0.17    (gene activation rate)
  k_degradation:  0.0066  (gene degradation rate)
  diffusion_rate: 0.02    (morphogen diffusion)
  K_self:         0.28    (self-maintenance Hill constant)
  n_self:         3.26    (self-maintenance cooperativity)

Morphogen decay lengths:
  shh_decay_length:  0.20
  fgf8_decay_length: 0.05
  edn1_decay_length: 2.00
```

### CaMKII Bistability Parameters File Structure

`bestLearnedCaMKIIParams_*.dat` files contain learned parameters for the bistable competitive dynamics model:
- Ca²⁺ dynamics: `tau_ca`, `g_ca`, `V_half_ca`, `k_ca`, `k_decay_ca` (constant baseline consumption)
- Ca²⁺ to activation: `ca_threshold`, `ca_sensitivity` (sigmoid parameters for external drive)
- CaMKII bistability: `k_on`, `k_off`, `K_half`, `tau_camkii` (competitive self-activation dynamics)
- OR gate logic: `or_threshold`, `or_sharpness`, `gain_ca` (combines Ca²⁺ drive with self-activation)
- Parameter bounds: `*_min`, `*_max` for each learnable parameter
- Training metadata: `best_loss`, `best_iteration`

**Example learned values** (from `bestLearnedCaMKIIParams_0.dat`):
```
Ca²⁺ dynamics:
  tau_ca:        2.60    (time constant, range 2.0-5.0)
  g_ca:          5.34    (conductance, range 0.1-20.0)
  k_decay_ca:    4.33    (baseline decay, range 0.0-5.0)

CaMKII bistability:
  tau_camkii:    61.07   (time constant, range 10.0-100.0)
  k_on:          3.77    (activation rate, range 0.5-5.0)
  k_off:         0.024   (inactivation rate, range 0.001-1.0)
  K_half:        0.24    (bistable threshold, range 0.2-0.8)

OR gate:
  ca_threshold:  8.28    (range 0.01-10.0)
  ca_sensitivity: 1.26   (range 0.01-2.0)
  or_threshold:  0.51    (range 0.2-1.5)
  gain_ca:       2.22    (range 1.5-3.0)
```

## Development Patterns

### Adding New Simulations

1. Clone an existing simulation script (e.g., `simulateTrainedModel.py`)
2. Adjust parameter loading or set custom parameters
3. Configure `initialValues`, `clampParameters`, `externalInputs` as needed
4. Call `model.simulate(...)` with appropriate flags (`fieldModulation`, `numSimIters`, etc.)
5. Use `visualize.py` or matplotlib for output plots

### Modifying the Bioelectric ↔ GRN Coupling

- **Electric → GRN**: `cellularFieldNetwork.apply_gene_voltage_feedback(gene_fields, gain)` modulates ion channel conductances based on gene expression
- **GRN → Electric (via FacePatternCoordinator)**:
  1. Extract Vmem snapshot: `vmem_snapshot = model.electricNetwork.Vmem.detach().clone()`
  2. Derive face set point: `set_point = coordinator.derive_set_point(vmem_snapshot)`
  3. Seed GRN: `facial_grn.register_bioelectric_prepattern(set_point, weight=...)`
- See `run_stigmergic_facial_integration.py` for full bidirectional coupling loops

### Parameter Sweeps

Use the `compute*` and `analyze*` scripts:
- `computeCellularFieldNetworkParameterSweep.py`: generates sweep data
- `analyzeCellularFieldNetworkParameterSweep.py`: processes results
- Shell wrappers (`run*.sh`) orchestrate batch jobs

### Coding Conventions

- Use 4-space indentation (PEP8)
- Preserve existing class names (e.g., `cellularFieldNetwork` in lowerCamel) to avoid API drift
- Place configuration variables at the top of scripts (e.g., `Model` in `simulateTrainedModel.py`)
- Add helper functions to `utilities.py` rather than duplicating logic
- New scripts should follow naming patterns: `simulate*.py`, `analyze*.py`, `run*.sh`

## Model Flow Architectures

### Original: Stigmergic ↔ FacialGRN

1. **Stigmergic run**: produces a voltage-based face pattern (via `run_stigmergic_facial_integration.py`)
2. **FacePatternCoordinator**: converts Vmem snapshot into a feature mask/set-point (bone/eye/nose/jaw)
3. **FacialGRN seeding**: GRN is pre-seeded with the bioelectric set-point to align morphogens/genes to the electric mask
4. **Optional bidirectional loop**:
   - GRN evolves features
   - Feeds back to electric model via `apply_gene_voltage_feedback`
   - New set-point derived from updated Vmem
   - GRN re-seeded; repeat for a few cycles
5. **Diagnostic output**: `stigmergic_facial_integration.png` shows Vmem, derived mask, GRN features, and Pax6 expression; `gene_timeseries_lines.pdf` shows per-cell gene trajectories

### Refined: Dual-Driver (Bioelectric + Morphogen)

1. **Bioelectric layer** (`cellularFieldNetwork`): Fast timescale (dt=0.01, ~1000 iters) produces Vmem pattern
2. **Transduction layer** (`BioelectricTransduction`):
   - Converts Vmem → Ca²⁺ via voltage-gated channels
   - Temporal integration provides memory (tau_ca timescale)
3. **Morphogen layer** (`RefinedFacialGRN`):
   - SHH, FGF8, EDN1 gradients form via diffusion/degradation
   - Spatial patterning from source locations and decay lengths
4. **Gene activation**: Dual drivers with Hill functions
   - Morphogen gradients activate genes (70% weight)
   - Ca²⁺ gates activation via AND logic (30% weight)
   - Gene_activation = (Morphogen AND Ca²⁺_gate) OR (Self_maintenance)
5. **Feature classification** (`GeneBasedFeatureClassifier`):
   - Classifies bone/eye/nose/mouth from gene expression only
   - No voltage thresholds used
6. **Diagnostic output**: `refined_facial_integration.png` shows Vmem, Ca²⁺, morphogens, genes, and classified features

### CaMKII Bistability: Pattern Persistence via Competitive Dynamics

#### Signal Flow Diagram

```
                              ┌─────────────────────────────────────┐
                              │         OR GATE LOGIC               │
                              │                                     │
    ┌─────────┐   ┌───────┐   │   ┌─────────────┐                   │
    │  Vmem   │──▶│  Ca²⁺ │──▶│──▶│  ca_signal  │──┐ gain_ca        │
    │ pattern │   │       │   │   │   [0, 1]    │  │   ×            │
    └─────────┘   └───────┘   │   └─────────────┘  │                │
         │                    │                    ▼                │
         │        ┌───────────────────────────────────┐             │
         │        │  combined = gain_ca × ca_signal   │             │
         │        │            + self_activation      │             │   ┌────────┐
         │        │            - or_threshold         │─────────────│──▶│ CaMKII │
         │        └───────────────────────────────────┘             │   │ active │
         │                    ▲                                     │   └────────┘
         │                    │                                     │        │
         │        ┌───────────────────────────┐                     │        │
         │        │    self_activation        │◀────────────────────│────────┘
         │        │       [-1, +1]            │     competitive     │
         │        │                           │      feedback       │
         │        │  (CaMKII²-K²)/(CaMKII²+K²)│                     │
         │        └───────────────────────────┘                     │
         │                                                          │
         └──────────────────────────────────────────────────────────┘
                        Bioelectric → Molecular → Bistable
```

#### Temporal Phases

```
    Vmem ─────┐
              │╲
              │ ╲
              │  ╲─────────────────────────────
              │
    Ca²⁺ ─────┼───────┐
              │       │╲
              │       │ ╲
              │       │  ╲───────────────────     ~K_half (low, uniform)
              │       │
   CaMKII ────┼───────┼─────────┬─────────────    Pattern persists!
              │       │        ╱│
              │       │      ╱  │
              │       │    ╱    │
              │       │  ╱      │
    ──────────┴───────┴─────────┴─────────────▶  time
              0     1000      2000
              │       │         │
         Stimulus   Decay   Autonomous
          Phase     Phase   Maintenance
              │       │         │
        (Vmem drives (Ca²⁺    (Bistable
         Ca²⁺ pattern) decays)  memory)
```

#### Competitive Dynamics (Bistability Mechanism)

```
   self_activation
        +1 ─┼─────────────────────────╱─────   ← High CaMKII: positive feedback (stays ON)
            │                       ╱
            │                     ╱
            │                   ╱
         0 ─┼─────────────────●─────────────   ← K_half: unstable equilibrium
            │               ╱ │
            │             ╱   │
            │           ╱     │
        -1 ─┼─────────╱───────┼─────────────   ← Low CaMKII: negative feedback (stays OFF)
            └─────────────────┴────────────▶
            0              K_half          1    CaMKII_active

   Result: Two stable states (ON/OFF) with K_half as decision boundary
```

1. **Bioelectric stimulus phase** (t < 1000): Vmem pattern drives spatial Ca²⁺ patterning
   - Voltage-gated Ca²⁺ channels: `I_ca = g_ca * sigmoid((Vmem - V_half_ca)/k_ca) * (E_ca - Vmem)`
   - Ca²⁺ dynamics: `dCa/dt = I_ca - (1/tau_ca)*Ca - k_decay_ca`
   - Spatially patterned Ca²⁺ provides external drive to CaMKII
2. **Transduction layer**: Ca²⁺ → CaMKII activation via OR gate
   - External drive: `ca_signal = sigmoid((Ca - ca_threshold)/ca_sensitivity)` → [0, 1]
   - Competitive self-activation: `self_activation = (CaMKII² - K_half²)/(CaMKII² + K_half²)` → [-1, 1]
   - OR gate: `or_gate = sigmoid(gain_ca * ca_signal + self_activation - or_threshold)`
3. **CaMKII bistable dynamics**: Activation/inactivation kinetics
   - `dCaMKII/dt = (or_gate * k_on - k_off)/tau_camkii`
   - Pattern forms during high Ca²⁺ phase, amplified by competitive feedback
4. **Decay phase** (t = 1000-2000): Ca²⁺ decays due to k_decay_ca constant leak
   - Vmem decays, so I_ca decreases
   - Constant decay k_decay_ca drives Ca down to low equilibrium level (~K_half)
5. **Autonomous maintenance** (t > 2000): CaMKII pattern persists via bistability with reduced Ca drive
   - Ca stabilizes at low level, external drive becomes weak but spatially uniform
   - Only competitive self-activation remains: high CaMKII regions stay ON, low regions stay OFF
   - True bistable memory independent of external stimulus
6. **Diagnostic output**: Time series plots of Ca²⁺ and CaMKII showing formation, decay, and persistence phases

### CaMKII-Integrated Facial Patterning: Concurrent Dynamics

This architecture extends the CaMKII bistability mechanism to run **concurrently** with the facial GRN:

#### Key Files

- `camkiiFacialGRN.py`: Module containing `CaMKIIBistableSwitch` and `CaMKIIFacialGRN` classes
- `run_camkii_facial_integration.py`: Simulation runner for concurrent dynamics
- `learnCaMKIIFacialIntegration.py`: Parameter learning for concurrent architecture

#### Architecture Comparison

| Aspect | RefinedFacialGRN | CaMKIIFacialGRN |
|--------|------------------|-----------------|
| Ca²⁺ handling | Pre-equilibrated, static | Concurrent with GRN |
| bio_gate source | Raw Ca²⁺ (transient) | CaMKII activity (bistable) |
| Pattern persistence | Requires sustained Vmem | Survives Vmem decay |
| Temporal phases | 1 (static equilibrium) | 3 (rise/decay/maintain) |

#### Signal Flow

```
                    ┌──────────────────────────────────────────────────┐
                    │           CONCURRENT DYNAMICS                     │
    ┌─────────┐     │                                                   │
    │  Vmem   │─────┼───▶ Ca²⁺ ───▶ OR gate ───▶ CaMKII ───▶ bio_gate │
    │ pattern │     │                   ▲              │                │
    └─────────┘     │                   │              │                │
         │          │           self_activation ◀──────┘                │
         │          │          (competitive [-1,+1])                    │
         │          └──────────────────────────────────────────────────┘
         │                              │
         │                              ▼
         │          ┌──────────────────────────────────────────────────┐
         │          │           CONCURRENT GRN                          │
         │          │                                                   │
         └──────────┼───▶ Morphogens ───┐                              │
                    │    (SHH,FGF8,EDN1) │                              │
                    │                    ▼                              │
                    │         Gene = AND(Morphogen, bio_gate)           │
                    │                  OR Self_maintenance              │
                    │                    │                              │
                    │                    ▼                              │
                    │              Features (bone/eye/nose/mouth)       │
                    └──────────────────────────────────────────────────┘
```

#### Three Temporal Phases

1. **Rise Phase** (t = 0 to 1000): Vmem pattern develops
   - Ca²⁺ tracks Vmem spatial pattern
   - CaMKII begins to activate in high-Ca regions
   - GRN responds to emerging bio_gate pattern

2. **Decay Phase** (t = 1000 to 2000): Vmem decays back to uniform
   - Ca²⁺ decays due to k_decay_ca constant consumption
   - CaMKII "locks in" via competitive bistability
   - GRN pattern stabilizes

3. **Maintenance Phase** (t = 2000 to 3000): Vmem uniform, pattern persists
   - Ca²⁺ settles to low uniform level (~K_half)
   - CaMKII pattern persists via bistable memory
   - GRN maintains features through CaMKII-derived bio_gate

#### Learnable Parameters

CaMKII-integrated learning (`learnCaMKIIFacialIntegration.py`) can optimize:
- CaMKII dynamics: tau_ca, g_ca, V_half_ca, k_ca, k_decay_ca
- CaMKII bistability: ca_threshold, ca_sensitivity, k_on, k_off, K_half, tau_camkii
- OR gate: or_threshold, or_sharpness, gain_ca
- AND gate: and_threshold, and_sharpness

Note: bio_gate uses raw CaMKII activity directly (already in [0,1]), matching RefinedFacialGRN's use of normalized Ca²⁺.

## Important Implementation Notes

- **Tensor shapes**:
  - `Vmem`: (numSamples, numCells, 1)
  - `externalInputs`: (numSamples, numCells, numGenes) for GRN inputs
  - `GRNWeights`: (numGenes, numGenes)
  - `tissueConnectivity`: (numCells, numCells) adjacency matrix

- **Lattice connectivity**: Generated by `utilities.computeLatticeAdjacencyMatrix(latticeDims, periodicBoundary)`. Set `periodicBoundary=True` for toroidal topology.

- **Timescales** (from learned parameters in `bestLearnedCaMKIIParams_0.dat` and `bestLearnedFacialParams_130.dat`):

  | Process | Key Parameter | Learned Value | Relative Speed |
  |---------|--------------|---------------|----------------|
  | Bioelectric (Vmem) | dt (timestep) | 0.01 | Fastest |
  | Calcium (Ca²⁺) | tau_ca | 2.60 | Fast |
  | CaMKII | tau_camkii | 61.07 | Medium |
  | Gene Network | 1/k_degradation | ~152 | Slowest |

  **Detailed parameter values:**
  - **Bioelectric**: `timestep = 0.01`; typical `numSimIters` 1000–5000; equilibrates in ~10-50 time units
  - **Ca²⁺ dynamics**: `tau_ca = 2.60`, `k_decay_ca = 4.33`, `g_ca = 5.34`; equilibrates in ~10-20 time units
  - **CaMKII dynamics**: `tau_camkii = 61.07`, `k_on = 3.77`, `k_off = 0.024`, `K_half = 0.24`; **~24x slower than Ca²⁺**
  - **Gene dynamics**: `k_activation = 0.17`, `k_degradation = 0.0066`, `diffusion_rate = 0.02`; **~10x slower than CaMKII**

  **Timescale separation is critical for pattern persistence:**
  1. Vmem establishes spatial pattern quickly (~1000 iterations)
  2. Ca²⁺ transduces this pattern with slight lag (tau ~2.6)
  3. CaMKII integrates over longer time (tau ~61), enabling bistable "lock-in"
  4. Genes respond slowest, stabilizing final feature identities

  The **24x ratio between tau_camkii and tau_ca** allows CaMKII to maintain its state even after Ca²⁺ decays—this is the key to pattern persistence.

  See `TIMESCALE_IMPLEMENTATION.md` for detailed timescale hierarchy

- **Field modulation**: When `fieldModulation=True`, the electric field influences ion channel conductances via transduction weights. Use `apply_gene_voltage_feedback` for GRN-mediated modulation.

- **Dual-driver gene activation**: In `RefinedFacialGRN`, genes respond to both morphogen gradients (Hill activation) and bioelectric gating (Ca²⁺ threshold). The AND-OR logic combines initiation (morphogen AND bioelectric) with self-maintenance.

- **Feature classification**: `GeneBasedFeatureClassifier` uses gene expression thresholds to identify features. In refined model, features are classified from genes only (not voltage thresholds).

- **Random seeds**: Not always set explicitly in current code. For reproducibility in stochastic runs, add `torch.manual_seed(...)` or `np.random.seed(...)` at the top of scripts.

## Gene and Morphogen Reference

### Neural Crest GRN genes (upstream developmental control)
- **Pax3, Zic1, Msx1**: Early neural crest specification
- **Sox9, FoxD3, Snail2, Sox10**: Neural crest migration and differentiation

### Facial GRN genes (craniofacial patterning)
- **rx, six3, pax6, lhx2**: Eye field specification and development
- **alx**: Nasal/olfactory development
- **dlx, hand2**: Jaw/mandibular development
- **runx2**: Bone/osteogenic differentiation

### Morphogens (spatial patterning signals)
- **SHH (Sonic Hedgehog)**: Midline patterning, ventral/dorsal axis
- **FGF8 (Fibroblast Growth Factor 8)**: Anterior neural boundary, facial prominence
- **EDN1 (Endothelin 1)**: Ventral/mandibular specification

## Key Documentation Files

Several design and analysis documents provide architectural context:

- **REFINED_MODEL_DESIGN.md**: Architecture specification for dual-driver (bioelectric + morphogen) facial patterning model
- **TIMESCALE_IMPLEMENTATION.md**: Detailed timescale hierarchy implementation across bioelectric, transduction, morphogen, and gene layers
- **TIMESCALE_HIERARCHY_GUIDE.md**: Guide to understanding and tuning temporal dynamics
- **LEARNING_GUIDE.md**: Parameter learning workflow and optimization strategies
- **electric_grn_transduction_explained.md**: Explanation of bioelectric-to-GRN transduction mechanisms
- **biological_realism_assessment.md**: Assessment of biological plausibility of model components
- **voltage_gradient_critique.md**: Analysis of voltage gradient patterns and their biological interpretation

## Environment

- **Python 3** with PyTorch, NumPy, SciPy, Matplotlib, PIL
- **GPU support**: PyTorch will use GPU if available, but not required
- No `requirements.txt` or `setup.py` currently in repo; install dependencies manually

## Git Workflow

- **Main branch**: `main`
- **Current working branch**: `multiembryo`
- Commit messages: concise, capitalized summaries (e.g., "Add stability check for field gating"), subject lines under ~72 characters
- In PRs: include intent summary, commands run, pointers to generated artifacts (plots, `.dat` outputs), and note any backward-incompatible parameter changes