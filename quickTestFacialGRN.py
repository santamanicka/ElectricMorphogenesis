#!/usr/bin/env python3
"""
Quick test of FacialGRN functionality
Runs a simple simulation and displays basic results
"""

from geneRegulatoryNetwork import FacialGRN
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def quick_test():
    """Quick test showing basic FacialGRN functionality"""

    print("Initializing FacialGRN...")
    grn = FacialGRN(grid_size=40, device='cpu')

    print("Initial state:")
    state = grn.get_state()
    print(f"  Time: {state['time']}")
    print(f"  Shh range: [{state['morphogens']['shh'].min():.3f}, {state['morphogens']['shh'].max():.3f}]")
    print(f"  Fgf8 range: [{state['morphogens']['fgf8'].min():.3f}, {state['morphogens']['fgf8'].max():.3f}]")
    print(f"  Edn1 range: [{state['morphogens']['edn1'].min():.3f}, {state['morphogens']['edn1'].max():.3f}]")

    print("\nRunning simulation for 100 steps...")
    grn.simulate(num_steps=100)

    print("\nFinal state:")
    state = grn.get_state()
    print(f"  Time: {state['time']}")
    print(f"  Pax6 (eye) range: [{state['genes']['pax6'].min():.3f}, {state['genes']['pax6'].max():.3f}]")
    print(f"  Alx (nose) range: [{state['genes']['alx'].min():.3f}, {state['genes']['alx'].max():.3f}]")
    print(f"  Hand2 (jaw) range: [{state['genes']['hand2'].min():.3f}, {state['genes']['hand2'].max():.3f}]")

    # Count features
    features = state['features'].cpu().numpy()
    unique, counts = features.flatten(), {}
    for val in unique:
        counts[val] = counts.get(val, 0) + 1

    print("\nFacial feature distribution:")
    feature_names = {0: 'Undifferentiated', 1: 'Eye', 2: 'Nose', 3: 'Jaw'}
    for feature_id in sorted(set(unique)):
        name = feature_names.get(feature_id, 'Unknown')
        count = sum(1 for x in unique if x == feature_id)
        percentage = (count / len(unique)) * 100
        print(f"  {name}: {count} cells ({percentage:.1f}%)")

    # Create simple visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Morphogens
    axes[0].imshow(state['morphogens']['shh'].cpu().numpy(), cmap='Reds', aspect='auto')
    axes[0].set_title('Shh Gradient')
    axes[0].set_ylabel('Anterior → Posterior')

    # Gene expression (Pax6 for eye)
    axes[1].imshow(state['genes']['pax6'].cpu().numpy(), cmap='viridis', aspect='auto')
    axes[1].set_title('Pax6 (Eye Gene)')

    # Features
    colors = ['#ecf0f1', '#9b59b6', '#e74c3c', '#f39c12']
    feature_cmap = ListedColormap(colors)
    axes[2].imshow(state['features'].cpu().numpy(), cmap=feature_cmap, aspect='auto', vmin=0, vmax=3)
    axes[2].set_title('Facial Features')

    plt.tight_layout()
    plt.savefig('quick_test_result.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to quick_test_result.png")

    plt.show()

    print("\nTest complete!")

if __name__ == '__main__':
    quick_test()
