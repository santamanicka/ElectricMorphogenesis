# FacialGRN Compatibility Guide

## Overview

The `FacialGRN` class now supports **two usage modes**:

1. **Standalone Mode**: Simple interface for quick facial patterning simulations
2. **Framework Mode**: Compatible with `geneRegulatoryNetwork` parameter structure for integration into existing bioelectric simulation framework

## Key Design Decisions

### Cell-Autonomous Behavior (InterGRNWeights = None)

**Decision**: FacialGRN operates in **cell-autonomous** mode with `InterGRNWeights` forced to `None`.

**Biological Justification**:
- Morphogen gradients alone are sufficient for facial patterning
- Cells independently interpret local morphogen concentrations
- No direct gene-gene communication between neighboring cells required
- Supported by transplantation experiments and in vitro differentiation studies

**Implementation**: Even if you provide `InterGRNWeights` in parameters, it will be automatically set to `None` with a warning message.

## Usage Modes

### Mode 1: Standalone (Original Interface)

Simple interface for standalone simulations:

```python
from geneRegulatoryNetwork import FacialGRN

# Create model
grn = FacialGRN(grid_size=40, device='cpu')

# Run simulation
grn.simulate(num_steps=100)

# Get results
state = grn.get_state()
print(f"Features: {state['features']}")
print(f"Genes: {state['genes'].keys()}")
```

**Parameters**:
- `grid_size`: Grid dimension (default=40)
- `device`: 'cpu' or 'cuda'

### Mode 2: Framework (geneRegulatoryNetwork Compatible)

Compatible interface for integration with electric field models:

```python
from geneRegulatoryNetwork import FacialGRN
import torch

# Create parameters dict (similar to geneRegulatoryNetwork)
params = {
    'latticeDims': (30, 30),  # Grid dimensions
    'GRNParameters': {
        'tissueConnectivity': None,  # Optional, not used for FacialGRN
        'InterGRNWeights': None,  # Will be enforced to None (cell-autonomous)
        'facialParams': {
            # Optional: Override morphogen parameters
            'shhStrength': 1.2,
            'fgf8Strength': 0.9,
            'edn1Strength': 1.1,
            'diffusionRate': 0.12,
            'degradationRate': 0.06,
            'inhibitionStrength': 0.35,
            'geneActivationRate': 0.06,
            'geneDegradationRate': 0.025,
        }
    }
}

# Create model
grn = FacialGRN(parameters=params, numSamples=1)

# Run simulation (framework interface)
vmem = torch.randn(1, 900, 1)  # External Vmem input (optional)
grn.simulate(electricNetworkState=vmem, numSimIters=100)

# Access state vector (compatible format)
print(f"State shape: {grn.state.shape}")  # (numSamples, numVariables, 1)
print(f"State range: [{grn.state.min():.3f}, {grn.state.max():.3f}]")

# Also access grid representation
state_dict = grn.get_state()
print(f"Pax6 expression: {state_dict['genes']['pax6']}")
```

## Parameter Structure

### Standalone Mode Parameters

```python
grn = FacialGRN(
    grid_size=40,      # Grid dimension
    device='cpu'       # 'cpu' or 'cuda'
)
```

### Framework Mode Parameters

```python
parameters = {
    'latticeDims': (rows, cols),  # Required: Grid dimensions
    'GRNParameters': {             # Required: GRN parameters dict
        # Optional parameters:
        'tissueConnectivity': None,      # Tissue connectivity matrix (not used)
        'InterGRNWeights': None,         # Inter-cell weights (forced to None)
        'facialParams': {                # Optional: Morphogen parameters
            'shhStrength': 1.0,          # Shh gradient strength
            'fgf8Strength': 1.0,         # Fgf8 gradient strength
            'edn1Strength': 1.0,         # Edn1 gradient strength
            'diffusionRate': 0.1,        # Morphogen diffusion rate
            'degradationRate': 0.05,     # Morphogen degradation
            'inhibitionStrength': 0.3,   # Mutual inhibition strength
            'geneActivationRate': 0.05,  # Gene activation rate
            'geneDegradationRate': 0.02, # Gene degradation rate
        }
    }
}
```

## Simulation Methods

Both modes support flexible `simulate()` interface:

```python
# Standalone style
grn.simulate(num_steps=100)

# Framework style
grn.simulate(electricNetworkState=vmem, ATPConc=None, numSimIters=100)

# Both work!
```

## State Representation

### Grid Representation (Primary)

FacialGRN internally uses **grid representation** for morphogens and genes:

```python
grn.grid = {
    # Morphogens (spatial gradients)
    'shh': torch.Tensor[grid_size, grid_size],
    'fgf8': torch.Tensor[grid_size, grid_size],
    'edn1': torch.Tensor[grid_size, grid_size],

    # Genes (expression patterns)
    'rx': torch.Tensor[grid_size, grid_size],
    'six3': torch.Tensor[grid_size, grid_size],
    'pax6': torch.Tensor[grid_size, grid_size],
    'lhx2': torch.Tensor[grid_size, grid_size],
    'alx': torch.Tensor[grid_size, grid_size],
    'dlx': torch.Tensor[grid_size, grid_size],
    'hand2': torch.Tensor[grid_size, grid_size],

    # Features (final pattern)
    'feature': torch.Tensor[grid_size, grid_size]  # 0=undiff, 1=eye, 2=nose, 3=jaw
}
```

### State Vector (Framework Compatibility)

For framework compatibility, genes are also synced to **state vector**:

```python
grn.state.shape = (numSamples, numVariables, 1)
# where numVariables = numCells * numGenes = grid_size^2 * 7
```

**Ordering**: `[cell0_gene0, cell0_gene1, ..., cell0_gene6, cell1_gene0, ...]`

**Synchronization**:
- `sync_grid_to_state()`: Grid → State vector
- `sync_state_to_grid()`: State vector → Grid
- Automatically called in `updateState()`

## Interface Methods

### Required Methods (Framework Compatible)

```python
# Initialization
grn.defineParameters()        # Parse parameters dict
grn.defineVariables()         # Initialize state vectors

# Simulation
grn.updateDynamicalParameters(externalInputs)  # Set external inputs (Vmem)
grn.updateState(ATPConc)      # Update one timestep
grn.simulate(...)             # Run multiple steps

# Properties
grn.numCells                  # Total cells
grn.numGenes                  # Number of genes (7)
grn.numVariables              # numCells * numGenes
grn.state                     # State vector (numSamples, numVariables, 1)
grn.InterGRNWeights           # Always None (cell-autonomous)
```

### Additional Methods (Standalone)

```python
# Grid access
grn.get_state()              # Returns dict with morphogens, genes, features
grn.initialize_grid()        # Reset morphogen gradients
grn.reset()                  # Reset to initial state

# Parameter modification
grn.set_parameters(shhStrength=1.5, fgf8Strength=0.8)

# Direct grid updates
grn.update_morphogens()      # Update morphogen diffusion
grn.update_genes()           # Update gene expression
```

## Example: Integration with Electric Field Model

```python
import torch
from geneRegulatoryNetwork import FacialGRN

# Setup parameters for 20x20 grid
params = {
    'latticeDims': (20, 20),
    'GRNParameters': {
        'InterGRNWeights': None,  # Cell-autonomous
        'facialParams': {
            'shhStrength': 1.0,
            'fgf8Strength': 1.0,
        }
    }
}

# Create FacialGRN
facial_grn = FacialGRN(parameters=params, numSamples=1)

# Simulate with electric field input
num_cells = 20 * 20
for t in range(100):
    # Get Vmem from electric field model (example)
    vmem = torch.randn(1, num_cells, 1) * 0.1  # Mock Vmem

    # Update FacialGRN
    facial_grn.updateDynamicalParameters(externalInputs=vmem)
    facial_grn.updateState(ATPConc=None)

    if t % 10 == 0:
        state = facial_grn.get_state()
        print(f"t={t}: Pax6 range [{state['genes']['pax6'].min():.3f}, "
              f"{state['genes']['pax6'].max():.3f}]")
```

## Testing

Run comprehensive compatibility tests:

```bash
python testFacialGRN_compatibility.py
```

**Tests verify**:
- ✓ Standalone mode works
- ✓ Framework mode works
- ✓ InterGRNWeights enforced to None
- ✓ Both simulate() interfaces work
- ✓ Custom parameters work
- ✓ State vector synchronization works

## Comparison to geneRegulatoryNetwork

| Feature | geneRegulatoryNetwork | FacialGRN |
|---------|----------------------|-----------|
| **Parameter dict** | Required | Optional |
| **InterGRNWeights** | Optional | Forced to None |
| **Tissue connectivity** | Required | Optional (not used) |
| **State representation** | Vector only | Grid + Vector |
| **Morphogens** | No | Yes (Shh, Fgf8, Edn1) |
| **External inputs** | Vmem | Vmem (optional) |
| **ATP modulation** | Yes | No |
| **Gene regulation** | Generic | Facial-specific |

## Design Rationale

### Why Cell-Autonomous?

1. **Biological accuracy**: Morphogen-based patterning doesn't require cell-cell gene coupling
2. **Computational efficiency**: No need for connectivity matrix operations
3. **Model simplicity**: Clearer interpretation of results
4. **Experimental support**: Consistent with transplantation and in vitro studies

### Why Two Modes?

1. **Standalone mode**: Quick prototyping, parameter exploration, visualization
2. **Framework mode**: Integration with complex multi-scale models, electric fields, ATP dynamics

### State Synchronization

Grid ↔ State vector synchronization enables:
- Natural spatial operations (morphogen diffusion, gradient computation)
- Framework compatibility (state vector for coupled models)
- Best of both worlds

## Morphogen Parameters

Default values match biological observations:

```python
'shhStrength': 1.0,           # Medial gradient strength
'fgf8Strength': 1.0,          # Lateral gradient strength
'edn1Strength': 1.0,          # Posterior gradient strength
'diffusionRate': 0.1,         # Morphogen diffusion
'degradationRate': 0.05,      # Morphogen decay
'inhibitionStrength': 0.3,    # Shh-Fgf8 mutual inhibition
'geneActivationRate': 0.05,   # Gene expression dynamics
'geneDegradationRate': 0.02,  # Gene decay
```

## Common Usage Patterns

### Pattern 1: Quick Exploration

```python
grn = FacialGRN(grid_size=30)
grn.set_parameters(shhStrength=1.5)
grn.simulate(num_steps=100)
state = grn.get_state()
# Visualize state['features']
```

### Pattern 2: Framework Integration

```python
params = create_facial_params(grid_size=25, custom_morphogens=...)
grn = FacialGRN(parameters=params)
grn.simulate(electricNetworkState=vmem_field, numSimIters=200)
final_pattern = grn.state  # Use in coupled model
```

### Pattern 3: Parameter Sweep

```python
for shh_strength in [0.8, 1.0, 1.2, 1.5]:
    grn = FacialGRN(grid_size=20)
    grn.set_parameters(shhStrength=shh_strength)
    grn.simulate(num_steps=100)
    analyze_features(grn.get_state())
```

## Files

- `geneRegulatoryNetwork.py`: Main implementation
- `testFacialGRN.py`: Standalone visualization tests
- `testFacialGRN_compatibility.py`: Compatibility test suite
- `runFacialGRN_interactive.py`: Interactive visualization
- `quickTestFacialGRN.py`: Quick verification
- `FacialGRN_README.md`: General documentation
- `FacialGRN_Compatibility_Guide.md`: This file

## Summary

The FacialGRN compatibility layer provides:

✓ **Dual interface**: Standalone + Framework modes
✓ **Cell-autonomous**: Biologically justified, InterGRNWeights=None
✓ **Flexible parameters**: Via facialParams in GRNParameters
✓ **State synchronization**: Grid ↔ State vector
✓ **Framework compatible**: Works with electricNetworkState, simulate() interface
✓ **Backward compatible**: All standalone features preserved

Use standalone mode for exploration, framework mode for integration!