# ElectricMorphogenesis

Computational model of endogenous bioelectric pattern formation in multicellular tissues, demonstrating how electric fields mediate the emergence of spatial voltage patterns.

## Paper

> Manicka, S., & Levin, M. (2025). Field-mediated bioelectric basis of morphogenetic prepatterning. *Cell Reports Physical Science*, 102865.

## Overview

This project models how multicellular tissues generate and maintain bioelectric patterns through the interplay of ion channels, gap junctions, and self-generated electric fields. The model implements two complementary architectures:

- **Stigmergic model:** Cells collectively coordinate their membrane voltages through stigmergic feedback — each cell modifies the shared electric field, which in turn influences neighboring cells' ion channel dynamics. This indirect coordination mechanism produces emergent spatial voltage patterns without explicit cell-to-cell signaling programs.

- **Mosaic model:** Cells establish voltage patterns through a mosaic of distinct ion channel expression profiles, creating spatial heterogeneity in membrane potential.

Both models are built on a **cellular field network** — a lattice of cells coupled by gap junctions and immersed in a self-generated electric field that feeds back on ion channel conductances.

## Repository Structure

```
├── cellularFieldNetwork.py        # Core bioelectric network model
├── geneRegulatoryNetwork.py       # Gene regulatory network module
├── model.py                       # Top-level simulation model
├── utilities.py                   # Lattice generation and helper functions
├── visualize.py                   # Visualization utilities
├── simulateTrainedModel.py        # Simulate pre-trained Stigmergic/Mosaic models
├── simulateCellularFieldNetwork.py # Simulate with custom parameters
├── simulateModel.py               # General model simulation
├── simulateSingleCell.py          # Single-cell dynamics exploration
├── learnCellularFieldNetwork.py   # Learn network parameters via backpropagation
├── analyzeCellularFieldNetwork.py                   # Analyze trained models
├── analyzeCellularFieldNetworkParameterSweep.py     # Parameter sweep analysis
├── computeCellularFieldNetworkParameterSweep.py     # Generate parameter sweep data
├── computeCellularFieldNetworkEntropyRate.py        # Entropy rate computation
├── plotAnalysisData.py            # Plot analysis results
├── run*.sh                        # Shell scripts for batch execution
└── data/
    ├── StigmergicModelParameters.dat   # Trained Stigmergic model
    ├── MosaicModelParameters.dat       # Trained Mosaic model
    └── bestModelParameters_*.dat       # Additional trained models
```

## Getting Started

### Requirements

- Python 3
- PyTorch
- NumPy, SciPy
- Matplotlib

### Simulating Pre-trained Models

Edit the `Model` variable in `simulateTrainedModel.py` to select a model:

```python
Model = "Stigmergic"  # or "Mosaic"
```

Then run:

```bash
python simulateTrainedModel.py
```

### Custom Simulations

Configure parameters in `simulateCellularFieldNetwork.py` and run:

```bash
python simulateCellularFieldNetwork.py
```

### Training New Models

```bash
bash runLearnCellularFieldNetwork.sh

# Or directly:
python learnCellularFieldNetwork.py --latticeDims "(11,11)" --fieldEnabled True --numLearnIters 100
```

### Parameter Sweeps

```bash
# Generate sweep data
python computeCellularFieldNetworkParameterSweep.py

# Analyze results
python analyzeCellularFieldNetworkParameterSweep.py

# Plot
python plotAnalysisData.py
```

## License

MIT License. See [LICENSE](LICENSE) for details.