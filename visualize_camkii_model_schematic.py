#!/usr/bin/env python3
"""
High-Quality Schematic: CaMKII Facial Patterning Model
Publication/Presentation quality diagram.

Key insight to communicate:
- Bioelectric patterns are TRANSIENT (Vmem decays)
- CaMKII bistability provides PERSISTENT memory
- Pattern "locks in" even after the original signal is gone
- Dual drivers: Morphogens provide WHAT, Bioelectrics provide WHERE

Usage:
    python visualize_camkii_model_schematic_v2.py
    python visualize_camkii_model_schematic_v2.py --save
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true', help='Save the schematic as PNG and PDF')
parser.add_argument('--output', type=str, default='camkii_model_schematic_v2', help='Output filename')
args = parser.parse_args()

# Create figure with golden ratio proportions
fig = plt.figure(figsize=(16, 9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('white')

# ============================================================
# COLOR PALETTE (Professional, accessible)
# ============================================================
colors = {
    'vmem': '#2E86AB',        # Steel blue - bioelectric
    'ca': '#F6AE2D',          # Amber - calcium
    'camkii': '#E94F37',      # Vermilion - CaMKII (key player)
    'morphogen': '#7B68EE',   # Medium purple - morphogens
    'gene': '#2ECC71',        # Emerald - genes
    'feature': '#1ABC9C',     # Turquoise - features
    'bg_light': '#F8F9FA',    # Light gray background
    'bg_dark': '#343A40',     # Dark gray
    'text': '#2C3E50',        # Dark blue-gray text
    'arrow': '#5D6D7E',       # Gray arrows
}

# ============================================================
# TITLE
# ============================================================
ax.text(8, 8.6, 'CaMKII-Integrated Facial Patterning Model',
        fontsize=22, fontweight='bold', ha='center', va='top',
        color=colors['text'], fontfamily='sans-serif')
ax.text(8, 8.25, 'Bistable Memory for Developmental Pattern Persistence',
        fontsize=12, ha='center', va='top', style='italic',
        color='#7f8c8d', fontfamily='sans-serif')

# ============================================================
# MAIN FLOW: Three columns
# ============================================================

# --- Column 1: INPUT (Bioelectric Signal) ---
col1_x = 2.2
col1_y = 5.5

# Vmem pattern visualization (small grid showing pattern)
def draw_vmem_pattern(ax, x, y, size=1.2):
    """Draw a small voltage pattern grid - uniform color for all features"""
    n = 5
    cell_size = size / n
    # Binary pattern: 1 = feature (high Vmem), 0 = background (low Vmem)
    pattern = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],  # eyes
        [0, 0, 1, 0, 0],  # nose
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],  # mouth
    ])
    for i in range(n):
        for j in range(n):
            # Uniform colors: depolarized (warm) for features, polarized (cool) for background
            if pattern[i, j] == 1:
                color = plt.cm.RdYlBu_r(0.85)  # Warm red/orange for high Vmem
            else:
                color = plt.cm.RdYlBu_r(0.25)  # Cool blue for low Vmem
            rect = Rectangle((x + j*cell_size, y - i*cell_size - cell_size),
                            cell_size*0.9, cell_size*0.9,
                            facecolor=color,
                            edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)

# Vmem box with pattern
box_vmem = FancyBboxPatch((col1_x - 1, col1_y - 0.8), 2, 2.2,
                          boxstyle="round,pad=0.08,rounding_size=0.3",
                          edgecolor=colors['vmem'],
                          facecolor='white',
                          linewidth=2.5)
ax.add_patch(box_vmem)
ax.text(col1_x, col1_y + 1.15, 'Bioelectric\nPattern', fontsize=11,
        fontweight='bold', ha='center', va='center', color=colors['vmem'])
draw_vmem_pattern(ax, col1_x - 0.6, col1_y + 0.4, size=1.2)
# ax.text(col1_x, col1_y - 0.55, 'Vmem', fontsize=10, ha='center',
#         va='center', color=colors['vmem'], style='italic')

# "TRANSIENT" label with fade effect
ax.text(col1_x, col1_y - 1.1, 'TRANSIENT', fontsize=9, ha='center',
        va='center', color='#e74c3c', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fdecea',
                  edgecolor='#e74c3c', linewidth=1))

# --- Column 2: PROCESSING (CaMKII Bistability) - THE KEY ---
col2_x = 6.5
col2_y = 5.5

# Large processing box
box_process = FancyBboxPatch((col2_x - 2.3, col2_y - 1.8), 4.6, 4.2,
                             boxstyle="round,pad=0.1,rounding_size=0.4",
                             edgecolor=colors['camkii'],
                             facecolor='#fef5f4',
                             linewidth=3)
ax.add_patch(box_process)
ax.text(col2_x, col2_y + 2.1, 'Bistable Memory Module',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=colors['camkii'])

# Ca2+ transduction
box_ca = FancyBboxPatch((col2_x - 1.8, col2_y + 0.9), 1.4, 0.7,
                        boxstyle="round,pad=0.05",
                        edgecolor=colors['ca'],
                        facecolor=colors['ca'],
                        alpha=0.3, linewidth=2)
ax.add_patch(box_ca)
ax.text(col2_x - 1.1, col2_y + 1.25, 'Ca²⁺', fontsize=10,
        fontweight='bold', ha='center', va='center', color=colors['ca'])

# Arrow Ca -> CaMKII
arrow_ca_camkii = FancyArrowPatch((col2_x - 0.4, col2_y + 1.25),
                                  (col2_x + 0.4, col2_y + 1.25),
                                  arrowstyle='->', mutation_scale=20,
                                  linewidth=2.5, color=colors['ca'])
ax.add_patch(arrow_ca_camkii)

# CaMKII box
box_camkii = FancyBboxPatch((col2_x + 0.4, col2_y + 0.9), 1.6, 0.7,
                            boxstyle="round,pad=0.05",
                            edgecolor=colors['camkii'],
                            facecolor=colors['camkii'],
                            alpha=0.4, linewidth=2)
ax.add_patch(box_camkii)
ax.text(col2_x + 1.2, col2_y + 1.25, 'CaMKII', fontsize=10,
        fontweight='bold', ha='center', va='center', color=colors['camkii'])

# Draw energy landscape (bistability visualization)
def draw_energy_landscape(ax, x, y, width=3.0, height=1.0):
    """Draw the double-well potential showing bistability"""
    t = np.linspace(0, 1, 100)
    # Double well: minima at 0.15 and 0.85
    energy = 4 * (t - 0.15)**2 * (t - 0.85)**2
    energy = energy / energy.max() * height * 0.8

    # Scale and position
    x_plot = x - width/2 + t * width
    # FIXED: Wells should go DOWN (low energy = low y position)
    y_plot = y - height + energy  # Now minima are at bottom, barrier at top

    # Fill the landscape (from bottom up to the curve)
    ax.fill_between(x_plot, y - height - 0.1, y_plot, color='#ecf0f1', alpha=0.8)
    ax.plot(x_plot, y_plot, color=colors['camkii'], linewidth=2)

    # Draw decision boundary (K_half) at the barrier peak
    k_half_x = x
    barrier_y = y - height + height * 0.8  # Top of the barrier
    ax.plot([k_half_x, k_half_x], [y - height - 0.1, barrier_y + 0.1],
            color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(k_half_x, barrier_y + 0.2, 'K½', fontsize=8, ha='center',
            color='#e74c3c', fontweight='bold')

    # Draw balls in wells (at the bottom of wells)
    well_bottom = y - height
    ball_off = Circle((x - width*0.35, well_bottom + 0.12), 0.12,
                      facecolor='#3498db', edgecolor='white', linewidth=1.5)
    ball_on = Circle((x + width*0.35, well_bottom + 0.12), 0.12,
                     facecolor='#2ecc71', edgecolor='white', linewidth=1.5)
    ax.add_patch(ball_off)
    ax.add_patch(ball_on)

    # Labels below the wells
    ax.text(x - width*0.35, y - height - 0.25, 'OFF', fontsize=8,
            ha='center', color='#3498db', fontweight='bold')
    ax.text(x + width*0.35, y - height - 0.25, 'ON', fontsize=8,
            ha='center', color='#2ecc71', fontweight='bold')

draw_energy_landscape(ax, col2_x, col2_y - 0.2, width=3.2, height=1.0)

# Self-activation feedback loop at the BOTTOM of CaMKII box
# Using a single FancyArrowPatch with arc connectionstyle for a clean integrated arrow

# CaMKII box bottom edge is at col2_y + 0.9
camkii_bottom = col2_y + 0.9
camkii_center_x = col2_x + 1.2

# Single curved arrow from right side of bottom to left side of bottom
loop_start = (camkii_center_x + 0.55, camkii_bottom)   # Right side of bottom
loop_end = (camkii_center_x - 0.55, camkii_bottom)     # Left side of bottom

# Use FancyArrowPatch with arc3 connection for smooth integrated arrow
self_loop_arrow = FancyArrowPatch(
    loop_start, loop_end,
    connectionstyle='arc3,rad=-0.8',  # Negative rad curves downward
    arrowstyle='-|>',
    mutation_scale=20,
    linewidth=2.5,
    color=colors['camkii']
)
ax.add_patch(self_loop_arrow)

# Label below the loop
loop_bottom_y = camkii_bottom - 0.55
ax.text(camkii_center_x, loop_bottom_y, 'Self-activation', fontsize=8,
        ha='center', va='top', color=colors['camkii'],
        fontweight='bold', style='italic')

# "PERSISTENT" label
ax.text(col2_x, col2_y - 1.5, 'PERSISTENT', fontsize=9, ha='center',
        va='center', color='#27ae60', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#e8f8f0',
                  edgecolor='#27ae60', linewidth=1))

# --- Column 3: OUTPUT (Gene Expression & Features) ---
col3_x = 11.5
col3_y = 5.5

# Morphogen gradients (top of output)
box_morph = FancyBboxPatch((col3_x - 1.5, col3_y + 1.3), 3, 0.9,
                           boxstyle="round,pad=0.05",
                           edgecolor=colors['morphogen'],
                           facecolor='white',
                           linewidth=2)
ax.add_patch(box_morph)
ax.text(col3_x, col3_y + 1.75, 'Morphogen Gradients', fontsize=10,
        fontweight='bold', ha='center', va='center', color=colors['morphogen'])
ax.text(col3_x, col3_y + 1.45, 'SHH • FGF8 • EDN1', fontsize=8,
        ha='center', va='center', color=colors['morphogen'])

# Gene activation box
box_gene = FancyBboxPatch((col3_x - 1.5, col3_y - 0.3), 3, 1.2,
                          boxstyle="round,pad=0.05",
                          edgecolor=colors['gene'],
                          facecolor='white',
                          linewidth=2)
ax.add_patch(box_gene)
ax.text(col3_x, col3_y + 0.6, 'Gene Network', fontsize=10,
        fontweight='bold', ha='center', va='center', color=colors['gene'])
ax.text(col3_x, col3_y + 0.25, '(Morphogen ∧ CaMKII_gate)', fontsize=8,
        ha='center', va='center', color=colors['gene'], style='italic')
ax.text(col3_x, col3_y - 0.1, 'Pax6 • Alx • Dlx • Runx2', fontsize=7,
        ha='center', va='center', color=colors['gene'])

# Face output visualization
def draw_face_output(ax, x, y, size=1.0):
    """Draw stylized face pattern with different colors for each feature"""
    n = 5
    cell_size = size / n
    # Feature-specific patterns: 1=eye, 2=nose, 3=mouth, 0=background
    pattern = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],  # eyes (1)
        [0, 0, 2, 0, 0],  # nose (2)
        [0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0],  # mouth (3)
    ])
    # Different colors for each feature type
    feature_colors = {
        0: '#ecf0f1',    # Background - light gray
        1: '#3498db',    # Eyes - blue
        2: '#2ecc71',    # Nose - green
        3: '#e74c3c',    # Mouth - red
    }
    for i in range(n):
        for j in range(n):
            color = feature_colors[pattern[i, j]]
            rect = Rectangle((x + j*cell_size, y - i*cell_size - cell_size),
                            cell_size*0.9, cell_size*0.9,
                            facecolor=color, edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)

# Feature output box
box_feature = FancyBboxPatch((col3_x - 1.5, col3_y - 2.0), 3, 1.4,
                             boxstyle="round,pad=0.05",
                             edgecolor=colors['feature'],
                             facecolor='white',
                             linewidth=2)
ax.add_patch(box_feature)
ax.text(col3_x, col3_y - 0.75, 'Facial Features', fontsize=10,
        fontweight='bold', ha='center', va='center', color=colors['feature'])
draw_face_output(ax, col3_x - 0.5, col3_y - 0.9, size=1.0)
# ax.text(col3_x, col3_y - 1.8, 'Eye • Nose • Mouth', fontsize=8,
#         ha='center', va='center', color=colors['feature'])

# ============================================================
# ARROWS connecting the columns
# ============================================================

# Vmem -> Ca2+ (arrow ends at left side of Ca box)
arrow1 = FancyArrowPatch((col1_x + 1.1, col1_y + 0.5), (col2_x - 1.8, col2_y + 1.25),
                         arrowstyle='-|>', mutation_scale=25,
                         linewidth=3, color=colors['arrow'],
                         connectionstyle='arc3,rad=0.1')
ax.add_patch(arrow1)
ax.text((col1_x + col2_x - 1)/2 - 0.15, col1_y + 1.0, 'V-gated\nCa²⁺ channels',
        fontsize=8, ha='center', va='center', color=colors['arrow'],
        style='italic')

# CaMKII -> CaMKII_gate -> Genes (start at right side of CaMKII box, end at left side of Gene Network box)
arrow2 = FancyArrowPatch((col2_x + 2.0, col2_y + 1.25), (col3_x - 1.5, col3_y + 0.3),
                         arrowstyle='-|>', mutation_scale=25,
                         linewidth=3, color=colors['camkii'],
                         connectionstyle='arc3,rad=-0.15')
ax.add_patch(arrow2)
ax.text((col2_x + col3_x)/2 + 0.5, col2_y + 1.1, 'CaMKII_gate',
        fontsize=9, ha='center', va='center', color=colors['camkii'],
        fontweight='bold', style='italic')

# Morphogens -> Genes
arrow3 = FancyArrowPatch((col3_x, col3_y + 1.3), (col3_x, col3_y + 0.95),
                         arrowstyle='-|>', mutation_scale=20,
                         linewidth=2.5, color=colors['morphogen'])
ax.add_patch(arrow3)

# Genes -> Features
arrow4 = FancyArrowPatch((col3_x, col3_y - 0.35), (col3_x, col3_y - 0.6),
                         arrowstyle='-|>', mutation_scale=20,
                         linewidth=2.5, color=colors['gene'])
ax.add_patch(arrow4)

# ============================================================
# TEMPORAL DYNAMICS (Bottom panel)
# ============================================================
y_timeline = 1.8

# Timeline box
timeline_box = FancyBboxPatch((1, y_timeline - 0.9), 14, 1.6,
                              boxstyle="round,pad=0.1",
                              edgecolor='#bdc3c7',
                              facecolor=colors['bg_light'],
                              linewidth=1.5)
ax.add_patch(timeline_box)

ax.text(8, y_timeline + 0.55, 'Temporal Dynamics: Pattern Persistence',
        fontsize=11, fontweight='bold', ha='center', va='center',
        color=colors['text'])

# Draw timeline with three phases
timeline_y = y_timeline - 0.15
phase_colors = [colors['vmem'], colors['ca'], colors['camkii']]
phase_labels = ['Initiation', 'Decay of input', 'Maintenance of output']
phase_descs = [
    'Vmem → Ca²⁺ → CaMKII',
    'Vmem↓ Ca²⁺↓ CaMKII locks',
    'Pattern persists!'
]
phase_starts = [1.5, 5.5, 10]
phase_widths = [3.5, 4, 4]

for i, (start, width) in enumerate(zip(phase_starts, phase_widths)):
    # Phase box
    phase_box = FancyBboxPatch((start, timeline_y - 0.55), width, 0.9,
                               boxstyle="round,pad=0.03",
                               edgecolor=phase_colors[i],
                               facecolor='white',
                               linewidth=2)
    ax.add_patch(phase_box)
    ax.text(start + width/2, timeline_y + 0.15, phase_labels[i],
            fontsize=9, fontweight='bold', ha='center', va='center',
            color=phase_colors[i])
    ax.text(start + width/2, timeline_y - 0.25, phase_descs[i],
            fontsize=8, ha='center', va='center', color=phase_colors[i])

# Time arrow
ax.annotate('', xy=(14.5, timeline_y - 0.7), xytext=(1.5, timeline_y - 0.7),
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
ax.text(14.6, timeline_y - 0.7, 'time', fontsize=9, ha='left',
        va='center', color='#7f8c8d', style='italic')

plt.tight_layout()

if args.save:
    # Save as PNG (high res) and PDF (vector)
    plt.savefig(f'{args.output}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{args.output}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {args.output}.png (300 DPI)")
    print(f"Saved: {args.output}.pdf (vector)")
else:
    plt.show()

print("\nSchematic complete!")
print("\nKey message: CaMKII bistability converts TRANSIENT bioelectric")
print("signals into PERSISTENT developmental memory.")
