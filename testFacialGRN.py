#!/usr/bin/env python3
"""
Test script for FacialGRN - Craniofacial Patterning Model
Visualizes morphogen gradients, gene expression, and facial feature patterns
Similar to the interactive HTML visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from geneRegulatoryNetwork import FacialGRN

def visualize_facial_grn(grn, save_path='facial_grn_visualization.png'):
    """Create comprehensive visualization of FacialGRN state"""

    # Get current state
    state = grn.get_state()
    gs = grn.grid_size

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs_layout = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Color maps
    shh_cmap = plt.cm.Reds
    fgf8_cmap = plt.cm.cool
    edn1_cmap = plt.cm.YlOrBr

    # Row 1: Individual morphogen gradients
    ax1 = fig.add_subplot(gs_layout[0, 0])
    shh_data = state['morphogens']['shh'].cpu().numpy()
    im1 = ax1.imshow(shh_data, cmap=shh_cmap, aspect='auto', origin='upper')
    ax1.set_title('Shh Gradient (Medial)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Lateral Position')
    ax1.set_ylabel('Anterior → Posterior')
    plt.colorbar(im1, ax=ax1, label='Concentration')

    ax2 = fig.add_subplot(gs_layout[0, 1])
    fgf8_data = state['morphogens']['fgf8'].cpu().numpy()
    im2 = ax2.imshow(fgf8_data, cmap=fgf8_cmap, aspect='auto', origin='upper')
    ax2.set_title('Fgf8 Gradient (Lateral)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Lateral Position')
    ax2.set_ylabel('Anterior → Posterior')
    plt.colorbar(im2, ax=ax2, label='Concentration')

    ax3 = fig.add_subplot(gs_layout[0, 2])
    edn1_data = state['morphogens']['edn1'].cpu().numpy()
    im3 = ax3.imshow(edn1_data, cmap=edn1_cmap, aspect='auto', origin='upper')
    ax3.set_title('Edn1 Gradient (Posterior)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Lateral Position')
    ax3.set_ylabel('Anterior → Posterior')
    plt.colorbar(im3, ax=ax3, label='Concentration')

    # Combined morphogens (RGB composite)
    ax4 = fig.add_subplot(gs_layout[0, 3])
    rgb_image = np.zeros((gs, gs, 3))
    rgb_image[:, :, 0] = shh_data + edn1_data  # Red channel: Shh + Edn1
    rgb_image[:, :, 1] = fgf8_data * 0.7 + edn1_data * 0.6  # Green channel: Fgf8 + Edn1
    rgb_image[:, :, 2] = fgf8_data * 0.8  # Blue channel: Fgf8
    rgb_image = np.clip(rgb_image, 0, 1)
    ax4.imshow(rgb_image, aspect='auto', origin='upper')
    ax4.set_title('Combined Morphogens (RGB)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Lateral Position')
    ax4.set_ylabel('Anterior → Posterior')

    # Row 2: Gene expression patterns
    gene_names = ['pax6', 'lhx2', 'alx', 'dlx']
    gene_titles = ['Pax6 (Eye)', 'Lhx2 (Eye)', 'Alx (Nose)', 'Dlx (Jaw)']

    for i, (gene, title) in enumerate(zip(gene_names, gene_titles)):
        ax = fig.add_subplot(gs_layout[1, i])
        gene_data = state['genes'][gene].cpu().numpy()
        im = ax.imshow(gene_data, cmap='viridis', aspect='auto', origin='upper', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Lateral Position')
        ax.set_ylabel('Anterior → Posterior')
        plt.colorbar(im, ax=ax, label='Expression')

    # Row 3: Facial features and gene expression profiles
    ax_features = fig.add_subplot(gs_layout[2, 0:2])
    feature_data = state['features'].cpu().numpy()

    # Create custom colormap for features
    from matplotlib.colors import ListedColormap
    colors = ['#ecf0f1', '#9b59b6', '#e74c3c', '#f39c12']  # undifferentiated, eye, nose, jaw
    feature_cmap = ListedColormap(colors)

    im_features = ax_features.imshow(feature_data, cmap=feature_cmap, aspect='auto',
                                     origin='upper', vmin=0, vmax=3)
    ax_features.set_title('Facial Features Pattern', fontsize=14, fontweight='bold')
    ax_features.set_xlabel('Lateral Position', fontsize=11)
    ax_features.set_ylabel('Anterior → Posterior', fontsize=11)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ecf0f1', label='Undifferentiated'),
        Patch(facecolor='#9b59b6', label='Eye'),
        Patch(facecolor='#e74c3c', label='Nose'),
        Patch(facecolor='#f39c12', label='Jaw')
    ]
    ax_features.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Gene expression profiles along midline
    ax_profiles = fig.add_subplot(gs_layout[2, 2:])
    mid_x = gs // 2

    # Extract midline profiles
    profiles = {
        'Shh': shh_data[:, mid_x],
        'Fgf8': fgf8_data[:, mid_x],
        'Edn1': edn1_data[:, mid_x],
        'Pax6': state['genes']['pax6'].cpu().numpy()[:, mid_x],
        'Alx': state['genes']['alx'].cpu().numpy()[:, mid_x],
        'Hand2': state['genes']['hand2'].cpu().numpy()[:, mid_x],
    }

    colors_dict = {
        'Shh': '#FF0000',
        'Fgf8': '#00FFFF',
        'Edn1': '#FF00FF',
        'Pax6': '#00FF00',
        'Alx': '#FFFF00',
        'Hand2': '#0000FF'
    }

    linestyles = {
        'Shh': '-',
        'Fgf8': '-',
        'Edn1': '-',
        'Pax6': '--',
        'Alx': '--',
        'Hand2': '--'
    }

    linewidths = {
        'Shh': 3,
        'Fgf8': 3,
        'Edn1': 3,
        'Pax6': 2,
        'Alx': 2,
        'Hand2': 2
    }

    for name, profile in profiles.items():
        ax_profiles.plot(profile, label=name, color=colors_dict[name],
                        linestyle=linestyles[name], linewidth=linewidths[name])

    ax_profiles.set_xlabel('Anterior → Posterior Position', fontsize=11)
    ax_profiles.set_ylabel('Expression Level', fontsize=11)
    ax_profiles.set_title('Gene Expression Profiles (Midline)', fontsize=14, fontweight='bold')
    ax_profiles.legend(loc='upper right', fontsize=10)
    ax_profiles.grid(True, alpha=0.3)
    ax_profiles.set_ylim([0, 1.1])

    # Main title
    fig.suptitle(f'Craniofacial Patterning Model - Time Step: {state["time"]}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")

    return fig


def create_animation(grn, num_steps=100, interval=50, save_path='facial_grn_animation.mp4'):
    """Create animation showing temporal evolution of facial patterning"""

    print(f"Creating animation for {num_steps} steps...")

    fig = plt.figure(figsize=(16, 10))
    gs_layout = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Initialize subplots
    axes = {
        'shh': fig.add_subplot(gs_layout[0, 0]),
        'fgf8': fig.add_subplot(gs_layout[0, 1]),
        'edn1': fig.add_subplot(gs_layout[0, 2]),
        'pax6': fig.add_subplot(gs_layout[1, 0]),
        'alx': fig.add_subplot(gs_layout[1, 1]),
        'features': fig.add_subplot(gs_layout[1, 2]),
    }

    # Color maps
    from matplotlib.colors import ListedColormap
    colors = ['#ecf0f1', '#9b59b6', '#e74c3c', '#f39c12']
    feature_cmap = ListedColormap(colors)

    # Initialize images
    images = {}

    def init():
        state = grn.get_state()

        images['shh'] = axes['shh'].imshow(state['morphogens']['shh'].cpu().numpy(),
                                           cmap='Reds', aspect='auto', origin='upper')
        axes['shh'].set_title('Shh Gradient', fontweight='bold')

        images['fgf8'] = axes['fgf8'].imshow(state['morphogens']['fgf8'].cpu().numpy(),
                                             cmap='cool', aspect='auto', origin='upper')
        axes['fgf8'].set_title('Fgf8 Gradient', fontweight='bold')

        images['edn1'] = axes['edn1'].imshow(state['morphogens']['edn1'].cpu().numpy(),
                                             cmap='YlOrBr', aspect='auto', origin='upper')
        axes['edn1'].set_title('Edn1 Gradient', fontweight='bold')

        images['pax6'] = axes['pax6'].imshow(state['genes']['pax6'].cpu().numpy(),
                                             cmap='viridis', aspect='auto', origin='upper',
                                             vmin=0, vmax=1)
        axes['pax6'].set_title('Pax6 (Eye)', fontweight='bold')

        images['alx'] = axes['alx'].imshow(state['genes']['alx'].cpu().numpy(),
                                           cmap='viridis', aspect='auto', origin='upper',
                                           vmin=0, vmax=1)
        axes['alx'].set_title('Alx (Nose)', fontweight='bold')

        images['features'] = axes['features'].imshow(state['features'].cpu().numpy(),
                                                     cmap=feature_cmap, aspect='auto',
                                                     origin='upper', vmin=0, vmax=3)
        axes['features'].set_title('Facial Features', fontweight='bold')

        return list(images.values())

    def update(frame):
        grn.update_state()
        state = grn.get_state()

        images['shh'].set_array(state['morphogens']['shh'].cpu().numpy())
        images['fgf8'].set_array(state['morphogens']['fgf8'].cpu().numpy())
        images['edn1'].set_array(state['morphogens']['edn1'].cpu().numpy())
        images['pax6'].set_array(state['genes']['pax6'].cpu().numpy())
        images['alx'].set_array(state['genes']['alx'].cpu().numpy())
        images['features'].set_array(state['features'].cpu().numpy())

        fig.suptitle(f'Facial Patterning - Step: {state["time"]}',
                    fontsize=14, fontweight='bold')

        return list(images.values())

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=num_steps,
                                  interval=interval, blit=True)

    # Save animation
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("Displaying animation instead...")
        plt.show()

    return anim


def run_simulation_and_visualize():
    """Main function to run FacialGRN simulation and create visualizations"""

    print("="*60)
    print("FacialGRN Test - Craniofacial Patterning Model")
    print("="*60)

    # Initialize the model
    print("\nInitializing FacialGRN model...")
    grn = FacialGRN(grid_size=11, device='cpu')

    # Save initial state
    print("Saving initial state visualization...")
    visualize_facial_grn(grn, save_path='facial_grn_initial.png')

    # Run simulation
    print("\nRunning simulation for 100 steps...")
    grn.simulate(num_steps=100)

    # Save final state
    print("Saving final state visualization...")
    visualize_facial_grn(grn, save_path='facial_grn_final.png')

    # Create time series plots
    print("\nCreating time series analysis...")
    create_time_series_plot(grn)

    # Test parameter variations
    print("\nTesting parameter variations...")
    test_parameter_variations()

    print("\n" + "="*60)
    print("Test complete! Check the generated PNG files.")
    print("="*60)


def create_time_series_plot(grn):
    """Create time series showing evolution of gene expression"""

    # Reset and run with state tracking
    grn.reset()

    time_points = []
    midline_pax6 = []
    midline_alx = []
    midline_dlx = []

    num_steps = 150
    mid_x = grn.grid_size // 2
    mid_y = grn.grid_size // 4  # Sample from anterior region

    for step in range(num_steps):
        state = grn.get_state()
        time_points.append(state['time'])
        midline_pax6.append(state['genes']['pax6'][mid_y, mid_x].item())
        midline_alx.append(state['genes']['alx'][mid_y, mid_x].item())
        midline_dlx.append(state['genes']['dlx'][mid_y, mid_x].item())
        grn.update_state()

    # Plot time series
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_points, midline_pax6, label='Pax6 (Eye)', color='#9b59b6', linewidth=2)
    ax.plot(time_points, midline_alx, label='Alx (Nose)', color='#e74c3c', linewidth=2)
    ax.plot(time_points, midline_dlx, label='Dlx (Jaw)', color='#f39c12', linewidth=2)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Gene Expression Level', fontsize=12)
    ax.set_title('Gene Expression Dynamics at Anterior Midline', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.savefig('facial_grn_timeseries.png', dpi=150, bbox_inches='tight')
    print("Time series plot saved to facial_grn_timeseries.png")


def test_parameter_variations():
    """Test effect of varying morphogen strengths"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parameter Variation Effects on Facial Features',
                 fontsize=16, fontweight='bold')

    from matplotlib.colors import ListedColormap
    colors = ['#ecf0f1', '#9b59b6', '#e74c3c', '#f39c12']
    feature_cmap = ListedColormap(colors)

    # Test different parameter combinations
    param_sets = [
        {'shhStrength': 1.0, 'fgf8Strength': 1.0, 'edn1Strength': 1.0, 'title': 'Default'},
        {'shhStrength': 1.5, 'fgf8Strength': 1.0, 'edn1Strength': 1.0, 'title': 'High Shh'},
        {'shhStrength': 0.5, 'fgf8Strength': 1.0, 'edn1Strength': 1.0, 'title': 'Low Shh'},
        {'shhStrength': 1.0, 'fgf8Strength': 1.5, 'edn1Strength': 1.0, 'title': 'High Fgf8'},
        {'shhStrength': 1.0, 'fgf8Strength': 0.5, 'edn1Strength': 1.0, 'title': 'Low Fgf8'},
        {'shhStrength': 1.0, 'fgf8Strength': 1.0, 'edn1Strength': 1.5, 'title': 'High Edn1'},
    ]

    for idx, params in enumerate(param_sets):
        ax = axes[idx // 3, idx % 3]

        # Create and run model
        grn = FacialGRN(grid_size=40, device='cpu')
        grn.set_parameters(**{k: v for k, v in params.items() if k != 'title'})
        grn.simulate(num_steps=100)

        # Plot features
        state = grn.get_state()
        im = ax.imshow(state['features'].cpu().numpy(), cmap=feature_cmap,
                      aspect='auto', origin='upper', vmin=0, vmax=3)
        ax.set_title(params['title'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Lateral')
        ax.set_ylabel('A-P')

    # Add shared colorbar
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ecf0f1', label='Undiff'),
        Patch(facecolor='#9b59b6', label='Eye'),
        Patch(facecolor='#e74c3c', label='Nose'),
        Patch(facecolor='#f39c12', label='Jaw')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig('facial_grn_parameter_variations.png', dpi=150, bbox_inches='tight')
    print("Parameter variations plot saved to facial_grn_parameter_variations.png")


if __name__ == '__main__':
    # Run main simulation and visualization
    run_simulation_and_visualize()

    # Optionally create animation (uncomment if ffmpeg is available)
    # print("\nCreating animation...")
    # grn = FacialGRN(grid_size=40, device='cpu')
    # create_animation(grn, num_steps=100, save_path='facial_grn_animation.mp4')
