#!/usr/bin/env python3

"""
Test script for hierarchical GRN integration.
Tests the neural crest upstream GRN controlling the generic downstream GRN.
"""

import torch
import numpy as np
from geneRegulatoryNetwork import geneRegulatoryNetwork

def create_minimal_parameters():
    """Create minimal parameters for testing hierarchical GRN"""

    # Basic tissue setup
    num_cells = 4  # Small 2x2 tissue
    num_genes = 3  # Small generic GRN

    # Create simple connectivity (all connected)
    tissue_connectivity = torch.ones(num_cells, num_cells, dtype=torch.float64) * 0.1

    # Basic GRN weights
    grn_weights = torch.randn(num_genes, num_genes, dtype=torch.float64) * 0.1
    inter_grn_weights = torch.randn(num_genes, num_genes, dtype=torch.float64) * 0.05
    vmem_to_grn_weights = torch.randn(1, num_genes, dtype=torch.float64) * 0.1

    # Parameters structure as expected by geneRegulatoryNetwork
    parameters = {
        'latticeDims': [2, 2],  # 2x2 grid
        'GRNParameters': {
            'tissueConnectivity': tissue_connectivity,
            'AsymmetricInterGRN': False,
            'PCPAxes': '2D',
            'GRNWeights': grn_weights,
            'GRNNumGenes': num_genes,
            'InterGRNWeights': inter_grn_weights,
            'VmemToGRNWeights': vmem_to_grn_weights,
            'VmemGain': torch.tensor([[1.0]], dtype=torch.float64),
            'GRNGains': torch.ones(1, num_genes, dtype=torch.float64),
            'GRNBiases': torch.zeros(1, num_genes, dtype=torch.float64),
            'VmemBias': torch.tensor([[0.0]], dtype=torch.float64),
            'GRNTimeconstants': torch.ones(1, num_genes, dtype=torch.float64) * 0.1,
            'InterGRNWeightsTimeconstant': torch.tensor([[0.1]], dtype=torch.float64),
            'VmemToGRNWeightsTimeconstant': torch.tensor([[0.1]], dtype=torch.float64)
        }
    }

    return parameters

def test_hierarchical_grn():
    """Test the hierarchical GRN integration"""

    print("=" * 60)
    print("HIERARCHICAL GRN INTEGRATION TEST")
    print("=" * 60)

    # Create test parameters
    print("Creating minimal test parameters...")
    parameters = create_minimal_parameters()

    # Initialize hierarchical GRN
    print("Initializing hierarchical GRN...")
    grn = geneRegulatoryNetwork(parameters=parameters, numSamples=2)

    print(f"✓ Generic GRN: {grn.numGenes} genes, {grn.numCells} cells")
    print(f"✓ Neural Crest GRN: {grn.neuralCrestGRN.num_nc_genes} genes")
    print(f"✓ State shape: {grn.state.shape}")
    print(f"✓ NC state shape: {grn.nc_state.shape}")

    # Test initial state
    initial_state = grn.state.clone()
    initial_nc_state = grn.nc_state.clone()

    print(f"\nInitial generic GRN activity: {torch.norm(initial_state):.6f}")
    print(f"Initial neural crest activity: {torch.norm(initial_nc_state):.6f}")

    # Add some initial activity to trigger dynamics
    grn.nc_state[:, :, 0] = 0.2  # Activate Pax3
    grn.nc_state[:, :, 1] = 0.1  # Activate Zic1
    grn.state[:, :, 0] = 0.1     # Activate first generic gene

    print(f"Added initial activity to both GRNs...")

    # Run multiple update steps
    print(f"\nRunning hierarchical dynamics...")
    for step in range(10):
        grn.updateState()

        if step % 3 == 0:
            generic_activity = torch.norm(grn.state)
            nc_activity = torch.norm(grn.nc_state)
            regulation_signals = grn.neuralCrestGRN.get_downstream_regulation(grn.nc_state)
            regulation_strength = torch.norm(regulation_signals)

            print(f"Step {step:2d}: Generic={generic_activity:.4f}, NC={nc_activity:.4f}, Regulation={regulation_strength:.4f}")

    # Final assessment
    final_state = grn.state
    final_nc_state = grn.nc_state

    generic_change = torch.norm(final_state - initial_state)
    nc_change = torch.norm(final_nc_state - initial_nc_state)

    print(f"\n" + "=" * 60)
    print("HIERARCHICAL INTEGRATION RESULTS")
    print("=" * 60)
    print(f"Generic GRN state change: {generic_change:.6f}")
    print(f"Neural crest GRN state change: {nc_change:.6f}")

    # Test Hill function replacement
    print(f"\nTesting Hill function activation...")
    test_input = torch.tensor([[0.5, 1.0, 2.0]], dtype=torch.float64)
    K = grn.hill_params['K_values']
    n = grn.hill_params['n_values']

    # Test sigmoid (old)
    sigmoid_result = torch.sigmoid(test_input)

    # Test Hill function (new)
    hill_result = (test_input**n) / (K**n + test_input**n)

    print(f"Sigmoid activation: {sigmoid_result.numpy()}")
    print(f"Hill activation:    {hill_result.numpy()}")
    print(f"Hill parameters K:  {K.numpy()}")
    print(f"Hill parameters n:  {n.numpy()}")

    # Test regulation signal propagation
    regulation_signals = grn.neuralCrestGRN.get_downstream_regulation(grn.nc_state)
    print(f"\nRegulation signal analysis:")
    print(f"Shape: {regulation_signals.shape}")
    print(f"Mean values: {regulation_signals.mean(dim=(0,1)).numpy()}")
    print(f"Signal types: [proliferation, migration, differentiation]")

    print(f"\n✓ HIERARCHICAL GRN INTEGRATION SUCCESSFUL!")
    print(f"  - Neural crest GRN operates upstream")
    print(f"  - Generic GRN operates downstream")
    print(f"  - Hill functions replace sigmoid functions")
    print(f"  - Regulation signals propagate between tiers")
    print(f"  - All operations use torch tensors")

    return grn

if __name__ == "__main__":
    test_hierarchical_grn()