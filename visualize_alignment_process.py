"""
Visualization of Field Alignment Process

This script creates intuitive diagrams explaining how field alignment works
at different resolutions and with different coarse-graining modes.

The visualization is designed to be understandable without prior knowledge
of the technical details.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from fieldAlignment import (
    FieldCoarseGrainer,
    create_radial_external_field,
    extract_field_2d,
)


def create_sample_fields(field_shape=(12, 12)):
    """Create sample main and reference fields for demonstration."""
    H, W = field_shape

    # Reference field: clean radial pattern (outward from center)
    ref_field = create_radial_external_field(field_shape, magnitude=1.0)

    # Main field: slightly rotated/perturbed version
    # Create a field that's similar but not identical
    Y, X = np.mgrid[0:H, 0:W]
    center_y, center_x = (H - 1) / 2, (W - 1) / 2

    # Add some rotation and noise to create a "misaligned" field
    dy = Y - center_y
    dx = X - center_x

    # Rotate by ~30 degrees and add noise
    angle = np.radians(30)
    rotated_dx = dx * np.cos(angle) - dy * np.sin(angle)
    rotated_dy = dx * np.sin(angle) + dy * np.cos(angle)

    # Normalize
    mag = np.sqrt(rotated_dx**2 + rotated_dy**2) + 1e-10
    main_field = torch.tensor(np.stack([
        rotated_dx / mag,
        rotated_dy / mag
    ]), dtype=torch.float32)

    return main_field, ref_field


def plot_field_with_arrows(ax, field, title, cmap='viridis', show_colorbar=True):
    """Plot a field with magnitude coloring and direction arrows."""
    H, W = field.shape[1], field.shape[2]
    Y, X = np.mgrid[0:H, 0:W]

    # Compute magnitude
    mag = torch.sqrt(field[0]**2 + field[1]**2).numpy()

    # Plot magnitude as background
    im = ax.imshow(mag, cmap=cmap, origin='lower', aspect='equal',
                   vmin=0, vmax=mag.max() if mag.max() > 0 else 1)

    # Plot arrows
    scale = mag.max() * 12 if mag.max() > 0 else 1
    ax.quiver(X, Y, field[0].numpy(), field[1].numpy(),
              color='white', alpha=0.8, scale=scale, width=0.02)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

    if show_colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8, label='Magnitude')

    return im


def plot_coarse_field(ax, field, resolution, title, cmap='viridis'):
    """Plot a coarse-grained field with grid overlay."""
    res_h, res_w = resolution

    # Compute magnitude
    mag = torch.sqrt(field[0]**2 + field[1]**2).numpy()

    # Plot magnitude
    im = ax.imshow(mag, cmap=cmap, origin='lower', aspect='equal',
                   interpolation='nearest',
                   vmin=0, vmax=mag.max() if mag.max() > 0 else 1)

    # Plot arrows
    Y, X = np.mgrid[0:res_h, 0:res_w]
    scale = mag.max() * 8 if mag.max() > 0 else 1
    ax.quiver(X, Y, field[0].numpy(), field[1].numpy(),
              color='white', alpha=0.9, scale=scale, width=0.03)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks(range(res_w))
    ax.set_yticks(range(res_h))
    ax.grid(True, color='white', alpha=0.3, linewidth=0.5)

    return im


def create_alignment_explanation_figure():
    """
    Create a comprehensive figure explaining field alignment.

    Layout:
    - Top: Concept overview
    - Middle: Average vs Sample mode comparison
    - Bottom: Resolution comparison
    """
    fig = plt.figure(figsize=(20, 24))

    # Create main and reference fields
    field_shape = (12, 12)
    main_field, ref_field = create_sample_fields(field_shape)
    coarsener = FieldCoarseGrainer(field_shape)

    # =========================================================================
    # SECTION 1: Concept Overview
    # =========================================================================
    gs_top = GridSpec(2, 5, figure=fig, top=0.90, bottom=0.72,
                      left=0.05, right=0.95, hspace=0.3, wspace=0.3)

    # Title for section
    fig.text(0.5, 0.97, 'Field Alignment: How Two Embryos Synchronize Their Electric Patterns',
             ha='center', va='top', fontsize=16, fontweight='bold')

    fig.text(0.5, 0.95,
             'Electric fields guide embryo development. When embryos communicate, their fields can align.',
             ha='center', va='top', fontsize=11, style='italic', color='gray')

    # 1a. Main embryo field (original)
    ax1 = fig.add_subplot(gs_top[0, 0])
    plot_field_with_arrows(ax1, main_field, 'Main Embryo\n(Original Field)',
                          cmap='plasma', show_colorbar=False)
    ax1.text(0.5, -0.15, 'Arrows show\nfield direction', transform=ax1.transAxes,
             ha='center', fontsize=8, color='gray')

    # 1b. Reference embryo field
    ax2 = fig.add_subplot(gs_top[0, 1])
    plot_field_with_arrows(ax2, ref_field, 'Reference Embryo\n(Target Field)',
                          cmap='viridis', show_colorbar=False)
    ax2.text(0.5, -0.15, 'Clean radial\npattern', transform=ax2.transAxes,
             ha='center', fontsize=8, color='gray')

    # Arrow and question between panels 1 and 2
    fig.text(0.24, 0.85, '\u2192', fontsize=16, ha='center', va='center',
             fontweight='bold', color='gray')

    # 1c. Coarse-graining step
    ax3 = fig.add_subplot(gs_top[0, 2])
    resolution = (3, 3)
    main_coarse = coarsener.coarsen(main_field, resolution, mode='average')
    ref_coarse = coarsener.coarsen(ref_field, resolution, mode='average')

    # Show coarse reference
    plot_coarse_field(ax3, ref_coarse, resolution,
                     f'Step 1: Simplify\n({resolution[0]}x{resolution[1]} regions)')
    ax3.text(0.5, -0.23, 'Divide into\nregions', transform=ax3.transAxes,
             ha='center', fontsize=8, color='gray')

    # 1d. Alignment computation
    ax4 = fig.add_subplot(gs_top[0, 3])
    # Show angle difference
    dot = (main_coarse[0] * ref_coarse[0] + main_coarse[1] * ref_coarse[1])
    main_mag = torch.sqrt(main_coarse[0]**2 + main_coarse[1]**2)
    ref_mag = torch.sqrt(ref_coarse[0]**2 + ref_coarse[1]**2)
    cos_angle = dot / (main_mag * ref_mag + 1e-10)
    angles = torch.acos(torch.clamp(cos_angle, -1, 1)) * 180 / np.pi

    im = ax4.imshow(angles.numpy(), cmap='RdYlGn_r', origin='lower',
                    vmin=0, vmax=90, aspect='equal')
    plt.colorbar(im, ax=ax4, shrink=0.8, label='Angle (degrees)')
    ax4.set_title('Step 2: Measure\nMisalignment', fontsize=10, fontweight='bold')
    ax4.set_xticks(range(resolution[1]))
    ax4.set_yticks(range(resolution[0]))
    ax4.text(0.5, -0.24, 'Green = aligned\nRed = misaligned', transform=ax4.transAxes,
             ha='center', fontsize=8, color='gray')

    # 1e. Result after alignment
    ax5 = fig.add_subplot(gs_top[0, 4])
    # Simulate partial alignment
    alpha = 0.7  # alignment strength effect
    aligned_field = (1 - alpha) * main_field + alpha * ref_field
    aligned_mag = torch.sqrt(aligned_field[0]**2 + aligned_field[1]**2)
    aligned_field = aligned_field / (aligned_mag + 1e-10)  # normalize

    plot_field_with_arrows(ax5, aligned_field, 'Step 3: Adjust\nMain Field',
                          cmap='plasma', show_colorbar=False)
    ax5.text(0.5, -0.15, 'Fields become\nmore similar', transform=ax5.transAxes,
             ha='center', fontsize=8, color='gray')

    # Add flow arrows between steps using text arrows
    for x in [0.22, 0.38, 0.54, 0.70]:
        fig.text(x+0.03, 0.85, '\u2192', fontsize=20, ha='center', va='center',
                fontweight='bold', color='black')

    # =========================================================================
    # SECTION 2: Average vs Sample Mode
    # =========================================================================
    gs_middle = GridSpec(2, 6, figure=fig, top=0.68, bottom=0.40,
                        left=0.05, right=0.95, hspace=0.4, wspace=0.25)

    fig.text(0.5, 0.70, 'Two Ways to Simplify: Average vs Sample Mode',
             ha='center', va='top', fontsize=14, fontweight='bold')

    # Show both modes at resolution 3x3
    resolution = (3, 3)

    # AVERAGE MODE explanation
    fig.text(0.27, 0.665, 'AVERAGE Mode', ha='center', fontsize=12,
             fontweight='bold', color='#2E86AB')
    fig.text(0.27, 0.645, 'Combines all values in each region',
             ha='center', fontsize=9, color='gray')

    # Original with grid overlay showing regions
    ax_avg1 = fig.add_subplot(gs_middle[0, 0])
    plot_field_with_arrows(ax_avg1, ref_field, 'Original (12x12)',
                          cmap='viridis', show_colorbar=False)
    # Draw region boundaries
    for i in range(1, 3):
        ax_avg1.axhline(y=i*4-0.5, color='yellow', linewidth=2, linestyle='--')
        ax_avg1.axvline(x=i*4-0.5, color='yellow', linewidth=2, linestyle='--')
    ax_avg1.text(0.5, -0.12, '4x4 cells per region', transform=ax_avg1.transAxes,
                ha='center', fontsize=8, color='gray')

    # Coarse-grained (average)
    ax_avg2 = fig.add_subplot(gs_middle[0, 1])
    ref_coarse_avg = coarsener.coarsen(ref_field, resolution, mode='average')
    plot_coarse_field(ax_avg2, ref_coarse_avg, resolution,
                     f'Coarse ({resolution[0]}x{resolution[1]})')
    ax_avg2.text(0.5, -0.19, 'Each cell = average\nof 16 original cells',
                transform=ax_avg2.transAxes, ha='center', fontsize=8, color='gray')

    # Upscaled back
    ax_avg3 = fig.add_subplot(gs_middle[0, 2])
    ref_upscaled_avg = coarsener.upscale(ref_coarse_avg, field_shape, mode='nearest')
    plot_field_with_arrows(ax_avg3, ref_upscaled_avg, 'Upscaled (12x12)',
                          cmap='viridis', show_colorbar=False)
    ax_avg3.text(0.5, -0.12, 'Blocky appearance\n(nearest neighbor)',
                transform=ax_avg3.transAxes, ha='center', fontsize=8, color='gray')

    # SAMPLE MODE explanation
    fig.text(0.73, 0.665, 'SAMPLE Mode', ha='center', fontsize=12,
             fontweight='bold', color='#A23B72')
    fig.text(0.73, 0.645, 'Samples at region corners only',
             ha='center', fontsize=9, color='gray')

    # Original with sample points
    ax_smp1 = fig.add_subplot(gs_middle[0, 3])
    plot_field_with_arrows(ax_smp1, ref_field, 'Original (12x12)',
                          cmap='viridis', show_colorbar=False)
    # Mark sample points
    sample_y = [0, 4, 8, 11]
    sample_x = [0, 4, 8, 11]
    for y in sample_y:
        for x in sample_x:
            ax_smp1.plot(x, y, 'ro', markersize=8, markeredgecolor='yellow',
                        markeredgewidth=2)
    ax_smp1.text(0.5, -0.12, 'Red dots = sample\npoints (4x4 grid)',
                transform=ax_smp1.transAxes, ha='center', fontsize=8, color='gray')

    # Coarse-grained (sample) - show 4x4 sampled grid
    ax_smp2 = fig.add_subplot(gs_middle[0, 4])
    ref_coarse_smp = coarsener.coarsen(ref_field, resolution, mode='sample')
    # Sample mode returns (res+1) x (res+1) for corners
    smp_shape = ref_coarse_smp.shape[1:]
    plot_coarse_field(ax_smp2, ref_coarse_smp, smp_shape,
                     f'Sampled ({smp_shape[0]}x{smp_shape[1]})')
    ax_smp2.text(0.5, -0.18, 'Corner values only\n(no averaging)',
                transform=ax_smp2.transAxes, ha='center', fontsize=8, color='gray')

    # Upscaled back (interpolate)
    ax_smp3 = fig.add_subplot(gs_middle[0, 5])
    ref_upscaled_smp = coarsener.upscale(ref_coarse_smp, field_shape, mode='interpolate')
    plot_field_with_arrows(ax_smp3, ref_upscaled_smp, 'Upscaled (12x12)',
                          cmap='viridis', show_colorbar=False)
    ax_smp3.text(0.5, -0.12, 'Smooth appearance\n(bilinear interpolation)',
                transform=ax_smp3.transAxes, ha='center', fontsize=8, color='gray')

    # Add comparison summary with cleaner layout
    ax_compare = fig.add_subplot(gs_middle[1, 0:3])
    ax_compare.axis('off')
    ax_compare.text(0.5, 0.8, 'AVERAGE Mode', ha='center', fontsize=11,
                   fontweight='bold', color='#2E86AB', transform=ax_compare.transAxes)
    avg_bullets = [
        '• Combines ALL cell values in each region',
        '• Preserves overall pattern energy',
        '• Creates blocky upscaled patterns',
        '• Best for global alignment detection'
    ]
    for i, text in enumerate(avg_bullets):
        ax_compare.text(0.5, 0.65 - i*0.12, text, ha='center', fontsize=9,
                       transform=ax_compare.transAxes)

    ax_compare_smp = fig.add_subplot(gs_middle[1, 3:6])
    ax_compare_smp.axis('off')
    ax_compare_smp.text(0.5, 0.8, 'SAMPLE Mode', ha='center', fontsize=11,
                        fontweight='bold', color='#A23B72', transform=ax_compare_smp.transAxes)
    smp_bullets = [
        '• Samples only at region CORNERS',
        '• Captures boundary behavior',
        '• Creates smooth interpolated patterns',
        '• Best for smooth field transitions'
    ]
    for i, text in enumerate(smp_bullets):
        ax_compare_smp.text(0.5, 0.65 - i*0.12, text, ha='center', fontsize=9,
                           transform=ax_compare_smp.transAxes)

    # =========================================================================
    # SECTION 3: Resolution Comparison
    # =========================================================================
    gs_bottom = GridSpec(2, 6, figure=fig, top=0.36, bottom=0.05,
                        left=0.05, right=0.95, hspace=0.35, wspace=0.25)

    fig.text(0.5, 0.38, 'Resolution: How Much Detail to Keep?',
             ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.37,
             'Lower resolution = simpler representation = faster alignment but less precise',
             ha='center', va='top', fontsize=10, style='italic', color='gray')

    # Show different resolutions
    resolutions = [(1, 1), (2, 2), (3, 3), (6, 6), (12, 12)]
    labels = ['1x1\n(Global)', '2x2\n(Quadrants)', '3x3\n(Regions)',
              '6x6\n(Local)', '12x12\n(Full)']
    descriptions = [
        'Single direction\nfor whole tissue',
        'Four regions\n(coarse)',
        'Nine regions\n(moderate)',
        '36 regions\n(detailed)',
        'Cell-by-cell\n(maximum detail)'
    ]

    for i, (res, label, desc) in enumerate(zip(resolutions, labels, descriptions)):
        # Coarse-grained field
        ax = fig.add_subplot(gs_bottom[0, i])

        if res == (12, 12):
            # Native resolution - no coarsening
            plot_field_with_arrows(ax, ref_field, label, cmap='viridis', show_colorbar=False)
        else:
            coarse = coarsener.coarsen(ref_field, res, mode='average')
            plot_coarse_field(ax, coarse, res, label)

        ax.text(0.5, -0.19, desc, transform=ax.transAxes,
               ha='center', fontsize=8, color='gray')

        # Add info box below
        if i == 0:
            ax.text(0.5, -0.35, 'Fastest alignment\nLeast precise',
                   transform=ax.transAxes, ha='center', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        elif i == len(resolutions) - 1:
            ax.text(0.5, -0.35, 'Slowest alignment\nMost precise',
                   transform=ax.transAxes, ha='center', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Add arrow showing resolution spectrum
    ax_arrow = fig.add_subplot(gs_bottom[1, :])
    ax_arrow.axis('off')

    # Draw gradient arrow
    ax_arrow.annotate('', xy=(0.9, 0.7), xytext=(0.1, 0.7),
                     arrowprops=dict(arrowstyle='->', color='black', lw=3))
    ax_arrow.text(0.1, 0.5, 'COARSE\n(Tissue-level)', ha='center', fontsize=10, fontweight='bold')
    ax_arrow.text(0.9, 0.5, 'FINE\n(Cell-level)', ha='center', fontsize=10, fontweight='bold')
    ax_arrow.text(0.5, 0.85, 'Increasing Resolution', ha='center', fontsize=11)

    # Add key insight
    insight_text = """
    KEY INSIGHT: Field alignment at coarse resolution captures "tissue-level" coordination
    (like two embryos agreeing on overall body plan), while fine resolution captures
    "cell-level" details (like precise organ boundaries).
    """
    ax_arrow.text(0.5, 0.15, insight_text, ha='center', va='top', fontsize=10,
                 style='italic', wrap=True,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    return fig


def create_alignment_dynamics_figure():
    """
    Create a figure showing how alignment evolves over time.
    """
    fig = plt.figure(figsize=(16, 10))

    fig.text(0.5, 0.97, 'Field Alignment Over Time: How Patterns Converge',
             ha='center', va='top', fontsize=14, fontweight='bold')

    # Create fields
    field_shape = (12, 12)
    main_field, ref_field = create_sample_fields(field_shape)

    # Simulate alignment steps
    gs = GridSpec(2, 5, figure=fig, top=0.90, bottom=0.15,
                 left=0.05, right=0.95, hspace=0.3, wspace=0.2)

    alignment_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]

    for i, alpha in enumerate(alignment_strengths):
        # Interpolate between main and reference
        if alpha == 0:
            current_field = main_field.clone()
        elif alpha == 1:
            current_field = ref_field.clone()
        else:
            current_field = (1 - alpha) * main_field + alpha * ref_field
            # Normalize to unit vectors
            mag = torch.sqrt(current_field[0]**2 + current_field[1]**2) + 1e-10
            current_field = current_field / mag

        # Top row: field visualization
        ax_field = fig.add_subplot(gs[0, i])
        plot_field_with_arrows(ax_field, current_field,
                              f'Step {i}: {int(alpha*100)}% aligned',
                              cmap='plasma', show_colorbar=False)

        if i == 0:
            ax_field.text(0.5, -0.12, 'Initial\n(misaligned)',
                         transform=ax_field.transAxes, ha='center', fontsize=9)
        elif i == len(alignment_strengths) - 1:
            ax_field.text(0.5, -0.12, 'Final\n(aligned)',
                         transform=ax_field.transAxes, ha='center', fontsize=9)

        # Bottom row: angle heatmap
        ax_angle = fig.add_subplot(gs[1, i])

        # Compute angle to reference
        dot = (current_field[0] * ref_field[0] + current_field[1] * ref_field[1])
        cur_mag = torch.sqrt(current_field[0]**2 + current_field[1]**2)
        ref_mag = torch.sqrt(ref_field[0]**2 + ref_field[1]**2)
        cos_angle = dot / (cur_mag * ref_mag + 1e-10)
        angles = torch.acos(torch.clamp(cos_angle, -1, 1)) * 180 / np.pi

        im = ax_angle.imshow(angles.numpy(), cmap='RdYlGn_r', origin='lower',
                            vmin=0, vmax=45, aspect='equal')
        ax_angle.set_title(f'Misalignment: {angles.mean():.1f}° avg', fontsize=9)
        ax_angle.set_xticks([])
        ax_angle.set_yticks([])

        if i == len(alignment_strengths) - 1:
            plt.colorbar(im, ax=ax_angle, shrink=0.8, label='Angle (°)')

    # Add arrows between steps using text arrows
    for i in range(4):
        x_start = 0.17 + i * 0.175
        fig.text(x_start, 0.72, '\u2192', fontsize=20, ha='center', va='center',
                fontweight='bold', color='blue')

    # Add explanation
    fig.text(0.5, 0.08,
             'As alignment progresses, the main embryo\'s field gradually rotates to match the reference.\n'
             'Green regions indicate good alignment; red regions indicate remaining misalignment.',
             ha='center', va='top', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    return fig


def main():
    """Generate all visualization figures."""
    print("Generating Field Alignment Process Visualization...")

    # Figure 1: Complete explanation
    fig1 = create_alignment_explanation_figure()
    fig1.savefig('./data/field_alignment_explanation.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: ./data/field_alignment_explanation.png")

    # Figure 2: Alignment dynamics
    fig2 = create_alignment_dynamics_figure()
    fig2.savefig('./data/field_alignment_dynamics.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: ./data/field_alignment_dynamics.png")

    plt.show()
    print("\nDone!")


if __name__ == '__main__':
    main()