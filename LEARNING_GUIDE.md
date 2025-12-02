# Refined Facial Integration Parameter Learning Guide

## Overview

The `learnRefinedFacialIntegration.py` script learns bioelectric-morphogen-gene parameters to match a target facial feature pattern. It uses backpropagation through the entire simulation pipeline to optimize parameters.

## Quick Start

```bash
# Quick test (20 iterations, ~5-10 minutes)
bash runLearnRefinedFacialIntegration.sh test

# Full bioelectric optimization (100 iterations, ~30-60 minutes)
bash runLearnRefinedFacialIntegration.sh bioelectric

# Run all configurations
bash runLearnRefinedFacialIntegration.sh all
```

## Available Configurations

### 1. Bioelectric Gating Only (Fastest)
```bash
bash runLearnRefinedFacialIntegration.sh bioelectric [file_number]
```
**Parameters learned**: 3
- `ca_threshold_percentile`: Where to threshold Ca²⁺ distribution (0.2-0.6)
- `ca_sensitivity`: Sigmoid sharpness for Ca²⁺ gating (0.01-0.1)
- `and_threshold`: AND gate activation threshold (1.0-1.5)

**Use when**: You want to optimize just the bioelectric gating mechanism

### 2. Bioelectric + AND Sharpness
```bash
bash runLearnRefinedFacialIntegration.sh bioelectric_sharp [file_number]
```
**Parameters learned**: 4 (bioelectric + `and_sharpness`)
- All from config 1, plus:
- `and_sharpness`: AND gate sigmoid sharpness (10.0-25.0)

**Use when**: AND gate nonlinearity needs tuning

### 3. Bioelectric + FGF8
```bash
bash runLearnRefinedFacialIntegration.sh bioelectric_fgf8 [file_number]
```
**Parameters learned**: 5 (bioelectric + FGF8 dynamics)
- All from config 1, plus:
- `fgf8_strength`: FGF8 secretion strength (0.1-0.5)
- `fgf8_degradation_factor`: FGF8 degradation multiplier (5.0-15.0)

**Use when**: Morphogen gradient needs refinement

### 4. Full Model
```bash
bash runLearnRefinedFacialIntegration.sh full [file_number]
```
**Parameters learned**: 8 (all systems)
- All from config 3, plus:
- `and_sharpness`: AND gate sharpness
- `k_activation`: Gene activation rate (0.05-0.2)
- `k_degradation`: Gene degradation rate (0.005-0.02)

**Use when**: Maximum flexibility needed

### 5. Feature Classification Threshold
```bash
bash runLearnRefinedFacialIntegration.sh feature_threshold [file_number]
```
**Parameters learned**: 4 (bioelectric + mouth threshold)
- All from config 1, plus:
- `min_mouth_expr`: Mouth classification threshold (0.3-0.9)

**Use when**: Mouth feature detection needs adjustment

### 6. Quick Test
```bash
bash runLearnRefinedFacialIntegration.sh test [file_number]
```
**Parameters learned**: 3 (bioelectric only)
**Iterations**: 20 (vs 100 default)
**GRN iterations**: 2000 (vs 5000 default)

**Use when**: Testing changes or debugging

### 7. Long Training
```bash
bash runLearnRefinedFacialIntegration.sh long [file_number]
```
**Parameters learned**: 5 (bioelectric + FGF8)
**Iterations**: 200 (vs 100 default)
**GRN iterations**: 8000 (vs 5000 default)
**Learning rate**: 0.01 (vs 0.02 default)

**Use when**: You need better convergence

## Parameter Selection Rationale

### Why AND sharpness but not OR sharpness?
- **AND gate is critical**: Bottleneck for feature formation (requires BOTH morphogen AND bioelectric permissiveness)
- **OR gate is permissive**: Already allows either input, so sharpness has minimal impact

### Why FGF8 parameters but not SHH/EDN1?
- **FGF8 is most sensitive**: 10× faster degradation + 10× slower diffusion = high parameter sensitivity
- **SHH is robust**: Simpler midline geometry, auto-balanced via mutual inhibition with FGF8
- **EDN1 is static**: No dynamics (linear gradient only), degradation/diffusion not applicable

### Minimal parameterization strategy
Following Occam's Razor: Include only parameters with highest impact on pattern formation.

## Output Files

Each configuration saves:
- **Parameters**: `data/bestLearnedFacialParams_<file_number>.dat`
- **Visualization**: `learned_facial_comparison_<file_number>.png`

The visualization shows:
- Target feature map
- Learned feature map
- SHH morphogen gradient
- FGF8 morphogen gradient
- Ca²⁺ signal
- Bioelectric gate
- Per-feature gene expression

## Understanding the Loss

The script uses **cross-entropy loss** on continuous feature scores:

```python
loss = cross_entropy(feature_scores, target_features)
```

Where:
- `feature_scores`: (4, grid_size, grid_size) - continuous [0,1] for each feature type
- `target_features`: (grid_size, grid_size) - discrete labels (0=bone, 1=eye, 2=nose, 3=mouth)

Lower loss means better match to target pattern.

## Simulation Pipeline

Each learning iteration runs:

1. **Phase 1**: Stigmergic bioelectric simulation (1000 iterations)
   - Self-organizing voltage patterns emerge

2. **Phase 2**: Bioelectric transduction (100 iterations)
   - Voltage → Ca²⁺ dynamics

3. **Phase 3**: Morphogen dynamics (5000 iterations)
   - SHH, FGF8, EDN1 gradients form
   - Mutual inhibition creates spatial domains

4. **Phase 4**: Gene regulatory network (5000 iterations)
   - AND-OR logic: (Morphogen AND Bio_gate) OR Self_maintenance
   - Gene expression patterns stabilize
   - Feature scores computed from expression levels

5. **Loss computation**: Feature map vs target
   - Gradients backpropagate through entire pipeline
   - Parameters updated via Rprop optimizer

## Recommended Workflow

1. **Start with quick test**:
   ```bash
   bash runLearnRefinedFacialIntegration.sh test 0
   ```
   Verifies setup and provides baseline (5-10 min)

2. **Optimize bioelectric gating**:
   ```bash
   bash runLearnRefinedFacialIntegration.sh bioelectric 1
   ```
   Most impactful parameters (30-60 min)

3. **Add morphogen tuning** (if needed):
   ```bash
   bash runLearnRefinedFacialIntegration.sh bioelectric_fgf8 2
   ```
   Refines gradients (45-90 min)

4. **Extended training** (if convergence needed):
   ```bash
   bash runLearnRefinedFacialIntegration.sh long 3
   ```
   Better convergence (2-4 hours)

## Troubleshooting

### Loss not decreasing
- Try lower learning rate (edit script: `LEARNING_RATE=0.01`)
- Increase iterations (use `long` config)
- Check if target pattern is achievable with current model

### Gradients vanishing/exploding
- Reduce learning rate
- Adjust sigmoid sharpness parameters (they affect gradient flow)

### Features not forming
- Check bioelectric transduction output (Ca²⁺ should have spatial variation)
- Verify morphogen gradients have sufficient contrast
- Ensure AND gate threshold isn't too high

### MPS/GPU issues
- Script automatically falls back to CPU if MPS/CUDA unavailable
- For Mac: Ensure macOS ≥ 12.3 for MPS support

## Advanced: Custom Parameter Sets

Edit the script to create custom configurations:

```bash
config_custom() {
    FILE_NUMBER=$1
    LEARNED_PARAMS="['ca_threshold_percentile','fgf8_strength']"  # Your params

    python learnRefinedFacialIntegration.py \
        --gridSize 11 \
        --numSimIters 1000 \
        --numGRNIters 5000 \
        --numLearnIters 100 \
        --lr 0.02 \
        --lossMethod featureMap \
        --learnedParameters "$LEARNED_PARAMS" \
        --idealFacePath IdealFace.png \
        --stigmergicParamsPath data/StigmergicModelParameters.dat \
        --fileNumber $FILE_NUMBER \
        --verbose True
}
```

Then add to the case statement in the script.

## Performance Expectations

### Iteration Times (M1/M2 Mac, MPS)
- Quick test (20 iter): 5-10 minutes
- Bioelectric (100 iter): 30-60 minutes
- Full model (100 iter): 45-90 minutes
- Long training (200 iter): 2-4 hours

### CPU Performance
Expect 2-3× slower than MPS/GPU.

## Next Steps After Learning

Once you have learned parameters, use them in simulations:

```python
# Load learned parameters
import pickle
with open('data/bestLearnedFacialParams_1.dat', 'rb') as f:
    learned_params = pickle.load(f)

# Apply to simulation
# (See run_refined_facial_integration.py for usage)
```

## Technical Details

### Optimizer: Rprop
- Resilient backpropagation with adaptive per-parameter learning rates
- More robust to gradient magnitudes than SGD/Adam
- Default step sizes: `step_sizes=[0.01, 0.5]`

### Device Selection
Automatic fallback chain:
1. MPS (Mac GPU) - if available
2. CUDA (NVIDIA GPU) - if available
3. CPU - always available

### Dtype Handling
- **MPS**: Uses `float32` (float64 not supported)
- **CPU/CUDA**: Uses `float64` for numerical precision

### Gradient Flow
All phases maintain gradients:
- Stigmergic simulation: No gradients (loaded from file)
- Transduction: Gradients through Ca²⁺ dynamics
- Morphogen: Gradients through reaction-diffusion
- GRN: Gradients through AND-OR logic
- Feature scores: Gradients through softmax/classification

Parameter updates minimize cross-entropy loss through this full pipeline.