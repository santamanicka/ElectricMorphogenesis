# Disconnect Between Gene Expression and Feature Labels

## The Problem

Cells at positions (3, 3) and (3, 7) express high levels of eye genes (like pax6) but are labeled as "bone" in the feature mask. This seems contradictory, but it reveals a fundamental architectural design of the bidirectional coupling system.

## Root Cause: Two Different Time Scales and Mechanisms

### 1. **Feature Labels** (from Vmem → instantaneous, memoryless)
- Derived from `FacePatternCoordinator.derive_set_point()`
- **Threshold-based classification** of current Vmem pattern
- Computed from the "detail" (local contrast) of voltage:
  ```python
  eye_mask = (detail >= 0.35) & (eye_template > 0.15)
  ```
- **No memory**: labels change instantly when Vmem crosses thresholds
- **Snapshot classification**: "What does the bioelectric pattern look like RIGHT NOW?"

### 2. **Gene Expression** (from FacialGRN → dynamical, has momentum)
- Governed by differential equations with timescales
- **Soft constraint from bioelectric prepattern**, not hard forcing
- Key code (geneRegulatoryNetwork.py:573-575):
  ```python
  if self.bioelectric_targets is not None:
      for gene_name, target in self.bioelectric_targets.items():
          self.grid[gene_name] += self.bioelectric_weight * (target - self.grid[gene_name])
  ```
- `bioelectric_weight = 0.4`: only 40% pull toward bioelectric target per timestep
- Gene expression also driven by **morphogen dynamics** (shh, fgf8, edn1) which compete with bioelectric guidance
- **Has history**: genes don't instantly snap to new targets, they evolve gradually

## Timeline of What Happens

### Before Coupling (Initial State)
- Vmem detail at (3,3) and (3,7): **0.374** (above 0.35 threshold) ✓
- Feature label: **"eye"**
- Bioelectric target for pax6: **0.95** (high)
- GRN initialized with pax6 = 0.95 at these cells

### Cycles 1-2: Everything Aligned
- Vmem detail: **0.421 → 0.392** (still above threshold) ✓
- Feature label: **"eye"**
- Bioelectric target: **0.95**
- pax6 expression: **~0.9-0.95** (maintained high)

### Cycle 3: THE DISCONNECT APPEARS
- **Vmem changes due to GRN feedback** → detail drops to **0.079** ✗
- Feature label **instantly switches** to: **"bone"**
- Bioelectric target **instantly changes** to: **0.05** (bone level for pax6)
- BUT pax6 expression: **still ~0.8-0.9** (high from previous state!)

  Why? Because the GRN update equation is:
  ```
  pax6(new) = pax6(old) + morphogen_dynamics + 0.4 * (target - pax6(old))
  ```

  If pax6(old) = 0.9 and target suddenly changes to 0.05:
  ```
  pax6(new) = 0.9 + morphogen_term + 0.4 * (0.05 - 0.9)
              = 0.9 + morphogen_term - 0.34
              ≈ 0.56 + morphogen_term
  ```

  It takes **multiple timesteps** to decay to bone levels!

### Cycle 4: Still Disconnected
- Vmem detail: **-0.085** (even more bone-like) ✗
- Feature label: **"bone"**
- Bioelectric target: **0.05**
- pax6 expression: **~0.5-0.7** (still declining, but not yet at bone level)

## Why This Design Makes Biological Sense

This is actually a **feature, not a bug**! It models realistic developmental biology:

1. **Bioelectric patterns change quickly** (milliseconds to seconds)
   - Ion channels open/close rapidly
   - Gap junctions modulate quickly

2. **Gene expression changes slowly** (minutes to hours)
   - Transcription, translation, protein degradation have longer timescales
   - Genes have regulatory momentum from their network dynamics

3. **Hysteresis and robustness**
   - Cells don't instantly switch identity every time voltage flickers
   - Gene networks maintain commitment to a fate for some time
   - Allows for "developmental memory" even as bioelectric signals fluctuate

## Solutions

### Option 1: Accept It as Biologically Realistic
- This temporal mismatch reflects real biology
- Label plots as "bioelectric prediction" vs "genetic commitment"

### Option 2: Increase Bioelectric Weight
In `run_stigmergic_facial_integration.py:273`:
```python
prepattern_weight=0.8,  # Was 0.4 - stronger bioelectric control
```
Makes genes snap faster to bioelectric targets.

### Option 3: Reduce Feedback That Disrupts Vmem
```python
feedback_strength=0.05,  # Was 0.2 - less GRN→electric feedback
```
Prevents Vmem from drifting away from the initial eye pattern.

### Option 4: Use Two Different Feature Masks for Plotting
```python
# In plotting code:
bioelectric_mask = coordinator.derive_set_point(vmem)  # Current Vmem
genetic_mask = infer_from_gene_expression(facial_grn)  # Based on gene levels
```
Show both on the plot to illustrate the temporal lag.

## Key Insight

The disconnect shows that **the bioelectric layer and genetic layer are semi-autonomous** with different dynamics. The bioelectric layer provides **guidance signals** (prepattern), not absolute commands. The genetic layer **interprets and integrates** these signals with its own regulatory logic and timescales.

This is conceptually similar to how neural activity patterns (fast) guide synaptic plasticity and gene expression changes (slow) in the brain.