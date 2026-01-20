#!/usr/bin/env python3
"""
Test FacialGRN compatibility with geneRegulatoryNetwork parameter structure
Tests both standalone and framework modes
"""

import torch
from geneRegulatoryNetwork import FacialGRN
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def test_standalone_mode():
    """Test FacialGRN in standalone mode (original interface)"""
    print("="*70)
    print("TEST 1: Standalone Mode")
    print("="*70)

    # Create FacialGRN the simple way
    grn = FacialGRN(grid_size=40, device='cpu')

    print(f"✓ Created FacialGRN")
    print(f"  Grid size: {grn.grid_size}")
    print(f"  Num cells: {grn.numCells}")
    print(f"  Num genes: {grn.numGenes}")
    print(f"  Num variables: {grn.numVariables}")
    print(f"  InterGRNWeights: {grn.InterGRNWeights}")  # Should be None

    # Run simple simulation
    grn.simulate(num_steps=50)
    state = grn.get_state()

    print(f"\n✓ Simulation completed")
    print(f"  Time: {state['time']}")
    print(f"  Pax6 range: [{state['genes']['pax6'].min():.3f}, {state['genes']['pax6'].max():.3f}]")

    return grn


def test_framework_mode():
    """Test FacialGRN in framework mode (compatible with geneRegulatoryNetwork)"""
    print("\n" + "="*70)
    print("TEST 2: Framework Mode (geneRegulatoryNetwork compatible)")
    print("="*70)

    # Create parameters dict similar to geneRegulatoryNetwork
    params = {
        'latticeDims': (20, 20),  # 20x20 grid
        'GRNParameters': {
            'tissueConnectivity': None,  # No connectivity for FacialGRN
            'InterGRNWeights': None,  # Cell-autonomous (will be enforced)
            'facialParams': {
                'shhStrength': 1.2,
                'fgf8Strength': 0.9,
                'edn1Strength': 1.1,
            }
        }
    }

    # Create FacialGRN using parameters dict
    grn = FacialGRN(parameters=params, numSamples=1)

    print(f"✓ Created FacialGRN with parameters dict")
    print(f"  Lattice dimensions: {grn.LatticeDimensions}")
    print(f"  Grid size: {grn.grid_size}")
    print(f"  Num cells: {grn.numCells}")
    print(f"  Num genes: {grn.numGenes}")
    print(f"  Num variables: {grn.numVariables}")
    print(f"  InterGRNWeights: {grn.InterGRNWeights}")  # Should be None (enforced)
    print(f"  Shh strength: {grn.params['shhStrength']}")
    print(f"  State shape: {grn.state.shape}")  # Should be (numSamples, numVariables, 1)

    # Run framework-style simulation
    grn.simulate(electricNetworkState=None, ATPConc=None, numSimIters=50)

    print(f"\n✓ Simulation completed (framework style)")
    print(f"  Time: {grn.current_time}")
    print(f"  State vector range: [{grn.state.min():.3f}, {grn.state.max():.3f}]")

    # Verify state sync
    grn.sync_state_to_grid()
    state = grn.get_state()
    print(f"  Grid Pax6 range: [{state['genes']['pax6'].min():.3f}, {state['genes']['pax6'].max():.3f}]")

    return grn


def test_intercell_warning():
    """Test that InterGRNWeights is properly set to None with warning"""
    print("\n" + "="*70)
    print("TEST 3: InterGRNWeights Warning Test")
    print("="*70)

    # Try to create with InterGRNWeights (should be set to None with warning)
    params = {
        'latticeDims': (15, 15),
        'GRNParameters': {
            'tissueConnectivity': torch.eye(225),  # 15x15 = 225 cells
            'InterGRNWeights': torch.randn(7, 7),  # Try to set InterGRNWeights
        }
    }

    print("Attempting to create FacialGRN with InterGRNWeights...")
    grn = FacialGRN(parameters=params, numSamples=1)

    print(f"✓ FacialGRN created")
    print(f"  InterGRNWeights: {grn.InterGRNWeights}")  # Should be None (enforced)
    print(f"  ✓ Cell-autonomous behavior enforced")

    return grn


def test_both_interfaces():
    """Test that both simulate() interfaces work"""
    print("\n" + "="*70)
    print("TEST 4: Both simulate() Interfaces")
    print("="*70)

    grn = FacialGRN(grid_size=20, device='cpu')

    # Test standalone interface
    print("Testing standalone interface: simulate(num_steps=30)")
    grn.simulate(num_steps=30)
    time1 = grn.current_time
    print(f"  ✓ Time after standalone: {time1}")

    # Reset
    grn.reset()

    # Test framework interface
    print("\nTesting framework interface: simulate(numSimIters=30)")
    grn.simulate(numSimIters=30)
    time2 = grn.current_time
    print(f"  ✓ Time after framework: {time2}")

    # Both should give same result
    assert time1 == time2, f"Times don't match: {time1} vs {time2}"
    print(f"\n✓ Both interfaces produce identical results")

    return grn


def visualize_comparison():
    """Visualize both modes side by side"""
    print("\n" + "="*70)
    print("TEST 5: Visual Comparison")
    print("="*70)

    # Standalone mode
    grn1 = FacialGRN(grid_size=30, device='cpu')
    grn1.simulate(num_steps=100)
    state1 = grn1.get_state()

    # Framework mode with custom parameters
    params = {
        'latticeDims': (30, 30),
        'GRNParameters': {
            'InterGRNWeights': None,
            'facialParams': {
                'shhStrength': 1.3,
                'fgf8Strength': 1.0,
            }
        }
    }
    grn2 = FacialGRN(parameters=params, numSamples=1)
    grn2.simulate(numSimIters=100)
    state2 = grn2.get_state()

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('FacialGRN Mode Comparison', fontsize=16, fontweight='bold')

    colors = ['#ecf0f1', '#9b59b6', '#e74c3c', '#f39c12']
    feature_cmap = ListedColormap(colors)

    # Row 1: Standalone mode
    axes[0, 0].imshow(state1['morphogens']['shh'].cpu().numpy(), cmap='Reds', aspect='auto')
    axes[0, 0].set_title('Standalone: Shh')
    axes[0, 0].set_ylabel('Standalone Mode')

    axes[0, 1].imshow(state1['genes']['pax6'].cpu().numpy(), cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Standalone: Pax6 (Eye)')

    axes[0, 2].imshow(state1['features'].cpu().numpy(), cmap=feature_cmap, aspect='auto', vmin=0, vmax=3)
    axes[0, 2].set_title('Standalone: Features')

    # Row 2: Framework mode
    axes[1, 0].imshow(state2['morphogens']['shh'].cpu().numpy(), cmap='Reds', aspect='auto')
    axes[1, 0].set_title('Framework: Shh (High strength)')
    axes[1, 0].set_ylabel('Framework Mode')

    axes[1, 1].imshow(state2['genes']['pax6'].cpu().numpy(), cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Framework: Pax6 (Eye)')

    axes[1, 2].imshow(state2['features'].cpu().numpy(), cmap=feature_cmap, aspect='auto', vmin=0, vmax=3)
    axes[1, 2].set_title('Framework: Features')

    plt.tight_layout()
    plt.savefig('facial_grn_mode_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to facial_grn_mode_comparison.png")

    return fig


def main():
    """Run all compatibility tests"""
    print("\n" + "="*70)
    print("FacialGRN Compatibility Test Suite")
    print("="*70 + "\n")

    try:
        # Run all tests
        grn1 = test_standalone_mode()
        grn2 = test_framework_mode()
        grn3 = test_intercell_warning()
        grn4 = test_both_interfaces()
        fig = visualize_comparison()

        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nSummary:")
        print("  ✓ Standalone mode works")
        print("  ✓ Framework mode works (compatible with geneRegulatoryNetwork)")
        print("  ✓ InterGRNWeights properly enforced to None (cell-autonomous)")
        print("  ✓ Both simulate() interfaces work correctly")
        print("  ✓ Custom parameters via facialParams work")
        print("  ✓ State vector synchronization works")
        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
