"""
High-Level Schematic: Embryo Communication via Electric Field Alignment

This script creates an intuitive, accessible visualization showing how
embryos can communicate and coordinate their development through
electric field alignment.

Designed to be understandable by anyone without technical background.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Ellipse
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects


def draw_embryo(ax, center, size, field_pattern='radial', color='#FFB6C1',
                label=None, field_color='#4169E1', alpha=0.8):
    """
    Draw a stylized embryo with electric field arrows.

    Parameters:
    -----------
    ax : matplotlib axis
    center : (x, y) center position
    size : radius of embryo
    field_pattern : 'radial', 'rotated', 'aligned', or 'random'
    color : fill color for embryo body
    label : optional label text
    field_color : color for field arrows
    alpha : transparency
    """
    cx, cy = center

    # Draw embryo body (oval shape)
    embryo = Ellipse(center, width=size*2, height=size*1.6,
                     facecolor=color, edgecolor='#8B4513',
                     linewidth=3, alpha=alpha, zorder=2)
    ax.add_patch(embryo)

    # Draw cell grid inside embryo
    n_cells = 5
    cell_positions = []
    for i in range(n_cells):
        for j in range(n_cells):
            # Position cells in a grid within the embryo
            x = cx + (i - n_cells//2) * size * 0.35
            y = cy + (j - n_cells//2) * size * 0.28
            # Check if inside ellipse
            if ((x - cx)**2 / (size*0.9)**2 + (y - cy)**2 / (size*0.7)**2) < 1:
                cell_positions.append((x, y))
                cell = Circle((x, y), size*0.12, facecolor='white',
                              edgecolor='#D3D3D3', linewidth=1, alpha=0.7, zorder=3)
                ax.add_patch(cell)

    # Draw field arrows based on pattern
    arrow_scale = size * 0.15
    for x, y in cell_positions:
        dx, dy = 0, 0

        if field_pattern == 'radial':
            # Outward from center
            dx = (x - cx)
            dy = (y - cy)
        elif field_pattern == 'rotated':
            # Rotated ~45 degrees from radial
            rx, ry = x - cx, y - cy
            angle = np.radians(45)
            dx = rx * np.cos(angle) - ry * np.sin(angle)
            dy = rx * np.sin(angle) + ry * np.cos(angle)
        elif field_pattern == 'aligned':
            # Same as radial (aligned to reference)
            dx = (x - cx)
            dy = (y - cy)
        elif field_pattern == 'random':
            # Random directions
            angle = np.random.uniform(0, 2*np.pi)
            dx = np.cos(angle)
            dy = np.sin(angle)

        # Normalize and scale
        mag = np.sqrt(dx**2 + dy**2) + 1e-10
        dx, dy = dx/mag * arrow_scale, dy/mag * arrow_scale

        ax.arrow(x, y, dx, dy, head_width=size*0.06, head_length=size*0.04,
                fc=field_color, ec=field_color, linewidth=1.5, zorder=4)

    # Add label
    if label:
        ax.text(cx, cy - size*1.1, label, ha='center', va='top',
               fontsize=12, fontweight='bold', color='#333333')

    return embryo


def draw_communication_waves(ax, start, end, n_waves=3, color='#FFD700', alpha=0.6):
    """Draw wavy lines representing communication/signaling between embryos."""
    sx, sy = start
    ex, ey = end

    for i in range(n_waves):
        offset = (i - n_waves//2) * 0.15
        # Create wavy path
        t = np.linspace(0, 1, 50)
        x = sx + t * (ex - sx)
        y = sy + t * (ey - sy) + offset + 0.1 * np.sin(t * 4 * np.pi)

        ax.plot(x, y, color=color, linewidth=2.5, alpha=alpha - i*0.1, zorder=1)


def create_embryo_communication_schematic():
    """
    Create the main schematic showing embryo communication concept.
    """
    fig = plt.figure(figsize=(18, 14))

    # Main title
    fig.suptitle('How Embryos "Talk" Through Electric Fields',
                fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94,
             'Embryos can coordinate their development by aligning their bioelectric patterns',
             ha='center', fontsize=13, style='italic', color='#555555')

    # =========================================================================
    # PANEL 1: The Concept (Top)
    # =========================================================================
    ax1 = fig.add_axes([0.05, 0.55, 0.9, 0.35])
    ax1.set_xlim(-0.5, 11.5)
    ax1.set_ylim(-1, 3.5)
    ax1.axis('off')
    ax1.set_title('The Big Picture: Two Embryos Learning from Each Other',
                 fontsize=14, fontweight='bold', pad=10)

    # Draw "Before" state (left side)
    ax1.text(2.5, 3.2, 'BEFORE', fontsize=12, fontweight='bold',
            ha='center', color='#CC0000')
    ax1.text(2.5, 2.9, '(Fields misaligned)', fontsize=10,
            ha='center', color='#666666')

    # Embryo A (reference - healthy)
    draw_embryo(ax1, (1, 1.5), 0.8, field_pattern='radial',
               color='#90EE90', label='Healthy Embryo\n(Reference)',
               field_color='#228B22')

    # Embryo B (main - misaligned)
    draw_embryo(ax1, (4, 1.5), 0.8, field_pattern='rotated',
               color='#FFB6C1', label='Developing Embryo\n(Needs guidance)',
               field_color='#DC143C')

    # Communication waves (weak/starting)
    draw_communication_waves(ax1, (1.9, 1.5), (3.1, 1.5), n_waves=2,
                            color='#0051FF', alpha=0.3)

    # Arrow showing time progression
    ax1.annotate('', xy=(6.3, 1.5), xytext=(5.0, 1.5),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=3))
    ax1.text(5.6, 1.7, 'Field\nAlignment', ha='center', fontsize=10,
            fontweight='bold', color='#333333')

    # Draw "After" state (right side)
    ax1.text(8.5, 3.2, 'AFTER', fontsize=12, fontweight='bold',
            ha='center', color='#006400')
    ax1.text(8.5, 2.9, '(Fields aligned)', fontsize=10,
            ha='center', color='#666666')

    # Embryo A (reference - still healthy)
    draw_embryo(ax1, (7.2, 1.5), 0.8, field_pattern='radial',
               color='#90EE90', label='Healthy Embryo',
               field_color='#228B22')

    # Embryo B (main - now aligned)
    draw_embryo(ax1, (9.8, 1.5), 0.8, field_pattern='aligned',
               color='#98FB98', label='Corrected Embryo',
               field_color='#228B22')

    # Strong communication waves
    draw_communication_waves(ax1, (8.1, 1.5), (8.9, 1.5), n_waves=3,
                            color="#0051FF", alpha=0.7)

    # Add "synchronized" indicator
    ax1.plot([7.2, 9.8], [0.4, 0.4], 'g-', linewidth=2, alpha=0.7)
    ax1.text(8.5, 0.15, 'Synchronized!', ha='center', fontsize=10,
            color='#006400', fontweight='bold')

    # =========================================================================
    # PANEL 2: Step-by-step process (Bottom)
    # =========================================================================
    ax2 = fig.add_axes([0.05, 0.08, 0.9, 0.42])
    ax2.set_xlim(-0.5, 11.5)
    ax2.set_ylim(-0.5, 4)
    ax2.axis('off')
    ax2.set_title('How It Works: The Alignment Process',
                 fontsize=14, fontweight='bold', pad=10)

    # Step 1: Sense
    step_y = 2.5
    box1 = FancyBboxPatch((0.2, step_y-0.6), 2.2, 1.4,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor='#E6F3FF', edgecolor='#4169E1', linewidth=2)
    ax2.add_patch(box1)
    ax2.text(1.3, step_y+0.5, 'STEP 1', fontsize=11, fontweight='bold',
            ha='center', color='#4169E1')
    ax2.text(1.3, step_y+0.15, 'SENSE', fontsize=13, fontweight='bold', ha='center')
    ax2.text(1.3, step_y-0.25, 'Each embryo detects\nthe other\'s electric\nfield pattern',
            ha='center', fontsize=9, color='#333333')

    # Arrow 1->2
    ax2.annotate('', xy=(3.0, step_y), xytext=(2.5, step_y),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    # Step 2: Compare
    box2 = FancyBboxPatch((3.1, step_y-0.6), 2.2, 1.4,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor='#FFF3E6', edgecolor='#FF8C00', linewidth=2)
    ax2.add_patch(box2)
    ax2.text(4.2, step_y+0.5, 'STEP 2', fontsize=11, fontweight='bold',
            ha='center', color='#FF8C00')
    ax2.text(4.2, step_y+0.15, 'COMPARE', fontsize=13, fontweight='bold', ha='center')
    ax2.text(4.2, step_y-0.25, 'Measure how much\nthe patterns differ\n(angle between fields)',
            ha='center', fontsize=9, color='#333333')

    # Arrow 2->3
    ax2.annotate('', xy=(5.9, step_y), xytext=(5.4, step_y),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    # Step 3: Adjust
    box3 = FancyBboxPatch((6.0, step_y-0.6), 2.2, 1.4,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor='#E6FFE6', edgecolor='#228B22', linewidth=2)
    ax2.add_patch(box3)
    ax2.text(7.1, step_y+0.5, 'STEP 3', fontsize=11, fontweight='bold',
            ha='center', color='#228B22')
    ax2.text(7.1, step_y+0.15, 'ADJUST', fontsize=13, fontweight='bold', ha='center')
    ax2.text(7.1, step_y-0.25, 'Gradually rotate\nfield to match\nthe reference',
            ha='center', fontsize=9, color='#333333')

    # Arrow 3->4
    ax2.annotate('', xy=(8.8, step_y), xytext=(8.3, step_y),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    # Step 4: Coordinate
    box4 = FancyBboxPatch((8.9, step_y-0.6), 2.2, 1.4,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor='#FFE6FF', edgecolor='#8B008B', linewidth=2)
    ax2.add_patch(box4)
    ax2.text(10.0, step_y+0.5, 'STEP 4', fontsize=11, fontweight='bold',
            ha='center', color='#8B008B')
    ax2.text(10.0, step_y+0.15, 'COORDINATE', fontsize=13, fontweight='bold', ha='center')
    ax2.text(10.0, step_y-0.25, 'Both embryos now\ndevelop with\naligned patterns',
            ha='center', fontsize=9, color='#333333')

    # Add repeat loop arrow
    ax2.annotate('', xy=(9.1, step_y-0.65), xytext=(1.3, step_y-0.65),
                arrowprops=dict(arrowstyle='<-', color='#666666', lw=1.5,
                               connectionstyle='arc3,rad=0.22'))
    ax2.text(5.3, step_y-1.4, 'Repeat until aligned', ha='center',
            fontsize=9, style='italic', color='#666666')

    # =========================================================================
    # Key insight box at bottom
    # =========================================================================
    insight_box = FancyBboxPatch((1.5, -0.1), 7.5, 0.9,
                                 boxstyle="round,pad=0.05,rounding_size=0.15",
                                 facecolor='#FFFACD', edgecolor='#DAA520',
                                 linewidth=2, alpha=0.9)
    ax2.add_patch(insight_box)

    ax2.text(5.25, 0.6, 'KEY INSIGHT', fontsize=15, fontweight='bold',
            ha='center', color='#B8860B')
    ax2.text(5.25, 0.2,
            'Electric fields act like a "language" that embryos use to coordinate development.\n'
            'A healthy embryo can guide a damaged one by sharing its correct pattern.',
            ha='center', fontsize=14, color='#333333')

    return fig


def create_biological_context_figure():
    """
    Create a figure showing the biological context and real-world implications.
    """
    fig = plt.figure(figsize=(16, 12))

    fig.suptitle('Why This Matters: Electric Fields in Development',
                fontsize=18, fontweight='bold', y=0.96)

    # =========================================================================
    # Left panel: What are bioelectric fields?
    # =========================================================================
    ax1 = fig.add_axes([0.05, 0.5, 0.42, 0.4])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    ax1.set_title('What Are Bioelectric Fields?', fontsize=14, fontweight='bold')

    # Draw a cell with ions
    cell_center = (5, 4)
    cell = Circle(cell_center, 2.5, facecolor='#E6F3FF', edgecolor='#4169E1',
                 linewidth=3, zorder=2)
    ax1.add_patch(cell)

    # Inside label
    ax1.text(5, 4.5, 'CELL', ha='center', fontsize=12, fontweight='bold')
    ax1.text(5, 3.5, 'Vmem = -50mV', ha='center', fontsize=10,
            style='italic', color='#4169E1')

    # Ion symbols outside
    ion_positions = [(2, 4), (8, 4), (5, 1.2), (5, 6.8), (2.5, 2), (7.5, 6)]
    for i, (x, y) in enumerate(ion_positions):
        sign = '+' if i % 2 == 0 else '-'
        color = '#DC143C' if sign == '+' else '#4169E1'
        ax1.text(x, y, sign, fontsize=16, fontweight='bold', ha='center',
                color=color, zorder=3)

    # Explanation text
    ax1.text(5, -0.5,
            'Every cell maintains a voltage difference\n'
            'across its membrane (like a tiny battery)',
            ha='center', fontsize=10, color='#333333')

    # =========================================================================
    # Right panel: How fields guide development
    # =========================================================================
    ax2 = fig.add_axes([0.53, 0.5, 0.42, 0.4])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title('How Fields Guide Development', fontsize=14, fontweight='bold')

    # Draw tissue with voltage pattern
    n = 5
    for i in range(n):
        for j in range(n):
            x = 2 + i * 1.2
            y = 2 + j * 1.0
            # Voltage varies with position (center depolarized)
            dist = np.sqrt((i - n//2)**2 + (j - n//2)**2)
            voltage = -50 + 30 * np.exp(-dist/1.5)

            # Color by voltage
            if voltage > -30:
                color = '#FFB6C1'  # Depolarized (pink)
            elif voltage > -45:
                color = '#FFFACD'  # Intermediate (yellow)
            else:
                color = '#90EE90'  # Polarized (green)

            cell = Circle((x, y), 0.4, facecolor=color, edgecolor='#333333',
                         linewidth=1, zorder=2)
            ax2.add_patch(cell)

    # Legend
    ax2.text(8.5, 6, 'Voltage Pattern', fontsize=10, fontweight='bold')
    colors = ['#90EE90', '#FFFACD', '#FFB6C1']
    labels = ['Polarized\n(-50mV)', 'Transition\n(-40mV)', 'Depolarized\n(-20mV)']
    for i, (c, l) in enumerate(zip(colors, labels)):
        y = 5 - i * 1.2
        patch = Circle((8.2, y), 0.25, facecolor=c, edgecolor='#333333')
        ax2.add_patch(patch)
        ax2.text(9, y, l, fontsize=8, va='center')

    ax2.text(5, -0.5,
            'Voltage patterns across tissues act as\n'
            '"blueprints" that tell cells where to go',
            ha='center', fontsize=10, color='#333333')

    # =========================================================================
    # Bottom panel: Real-world applications
    # =========================================================================
    ax3 = fig.add_axes([0.1, 0.08, 0.8, 0.35])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 4)
    ax3.axis('off')
    ax3.set_title('Potential Applications', fontsize=14, fontweight='bold', pad=10)

    # Application boxes
    apps = [
        ('Regeneration', '#E6F3FF', '#4169E1',
         'Help damaged tissues\nregenerate by restoring\ncorrect electric patterns'),
        ('Birth Defects', '#FFE6E6', '#DC143C',
         'Prevent developmental\ndefects by guiding\nembryos to correct patterns'),
        ('Cancer', '#E6FFE6', '#228B22',
         'Normalize tumor cells\nby exposing them to\nhealthy electric fields'),
        ('Tissue Engineering', '#FFF3E6', '#FF8C00',
         'Grow complex organs\nby using electric fields\nas organizing signals')
    ]

    for i, (title, bg, edge, desc) in enumerate(apps):
        x = 0.5 + i * 2.4
        box = FancyBboxPatch((x, 0.8), 2.0, 2.8,
                            boxstyle="round,pad=0.05,rounding_size=0.15",
                            facecolor=bg, edgecolor=edge, linewidth=2)
        ax3.add_patch(box)
        ax3.text(x + 1, 3.2, title, ha='center', fontsize=11,
                fontweight='bold', color=edge)
        ax3.text(x + 1, 2.1, desc, ha='center', fontsize=9,
                color='#333333', linespacing=1.3)

    return fig


def main():
    """Generate all visualization figures."""
    print("Generating Embryo Communication Schematic...")

    # Figure 1: Main communication concept
    fig1 = create_embryo_communication_schematic()
    fig1.savefig('./data/embryo_communication_schematic.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print("Saved: ./data/embryo_communication_schematic.png")

    # Figure 2: Biological context
    fig2 = create_biological_context_figure()
    fig2.savefig('./data/embryo_bioelectric_context.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print("Saved: ./data/embryo_bioelectric_context.png")

    plt.show()
    print("\nDone!")


if __name__ == '__main__':
    main()