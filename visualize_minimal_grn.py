"""
Visualization of the Minimal Literature-Based Neural Crest GRN

Based on:
- Simões-Costa & Bronner (2015): Neural crest specification
- Akiyama et al. (2002): Chondrogenesis
- Komori (2019): Osteogenesis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np

def get_node_boundary_point(pos, target_pos, shape='rectangle', width=0.5, height=0.25):
    """Calculate the point on the boundary of a node shape towards a target"""
    x, y = pos
    target_x, target_y = target_pos

    dx = target_x - x
    dy = target_y - y
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 1e-6:
        return x, y

    dx_norm = dx / dist
    dy_norm = dy / dist

    if shape == 'rectangle':
        # Rectangle boundary intersection
        t_x = width / (2 * abs(dx_norm)) if dx_norm != 0 else float('inf')
        t_y = height / (2 * abs(dy_norm)) if dy_norm != 0 else float('inf')
        t = min(t_x, t_y)
        return x + t * dx_norm, y + t * dy_norm
    elif shape == 'ellipse':
        # Ellipse boundary intersection
        a = width / 2  # Semi-major axis
        b = height / 2  # Semi-minor axis
        # Parametric form: x = a*cos(θ), y = b*sin(θ)
        theta = np.arctan2(dy_norm * a, dx_norm * b)
        return x + a * np.cos(theta), y + b * np.sin(theta)
    else:
        return x + 0.3 * dx_norm, y + 0.3 * dy_norm


def draw_arrow(ax, start_pos, end_pos, color='green', linestyle='-',
               start_shape='rectangle', end_shape='rectangle',
               arrow_type='positive', linewidth=2):
    """Draw an arrow from start to end with proper boundary intersections"""
    start_x, start_y = get_node_boundary_point(start_pos, end_pos, start_shape)
    end_x, end_y = get_node_boundary_point(end_pos, start_pos, end_shape)

    if arrow_type == 'positive':
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                  linestyle=linestyle, shrinkA=0, shrinkB=0))
    elif arrow_type == 'negative':
        # Use '|-|' style (bar-bar) for inhibition
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='|-|', color=color, lw=linewidth,
                                  linestyle=linestyle, shrinkA=0, shrinkB=0))


def draw_self_loop(ax, pos, color='green', radius=0.35, label=''):
    """Draw a self-loop using a Bezier curve"""
    x, y = pos

    # Control points for Bezier curve (loop above the node)
    start_x, start_y = x - 0.15, y + 0.125
    end_x, end_y = x + 0.15, y + 0.125
    ctrl1_x, ctrl1_y = x - 0.15, y + radius
    ctrl2_x, ctrl2_y = x + 0.15, y + radius

    verts = [(start_x, start_y), (ctrl1_x, ctrl1_y),
             (ctrl2_x, ctrl2_y), (end_x, end_y)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color,
                               lw=2, linestyle='-')
    ax.add_patch(patch)

    # Add arrowhead
    ax.annotate('', xy=(end_x, end_y), xytext=(end_x + 0.05, end_y + 0.15),
               arrowprops=dict(arrowstyle='->', color=color, lw=2))


def visualize_minimal_grn():
    """Create visualization of minimal literature-based neural crest GRN"""
    fig, ax = plt.subplots(figsize=(18, 12))

    # Define gene categories and positions
    genes = {
        # INPUT SIGNALS
        'BMP': {'pos': (0, 5), 'type': 'signal', 'label': 'BMP'},
        'Wnt': {'pos': (0, 3), 'type': 'signal', 'label': 'Wnt'},

        # PHASE 1: NEURAL CREST SPECIFICATION
        'Pax3': {'pos': (3, 5), 'type': 'border', 'label': 'Pax3'},
        'Zic1': {'pos': (3, 3), 'type': 'border', 'label': 'Zic1'},
        'FoxD3': {'pos': (6, 4), 'type': 'nc_specifier', 'label': 'FoxD3'},
        'Snail2': {'pos': (9, 2), 'type': 'nc_specifier', 'label': 'Snail2'},
        'Sox9_NC': {'pos': (9, 5), 'type': 'nc_specifier', 'label': 'Sox9'},
        'Sox10': {'pos': (9, 6.5), 'type': 'nc_specifier', 'label': 'Sox10'},

        # PHASE 2: CHONDROGENIC PATHWAY
        'Sox9': {'pos': (13, 5), 'type': 'chondro_tf', 'label': 'Sox9'},
        'Sox5': {'pos': (13, 7), 'type': 'chondro_tf', 'label': 'Sox5'},
        'Sox6': {'pos': (13, 3), 'type': 'chondro_tf', 'label': 'Sox6'},
        'Col2a1': {'pos': (16, 5), 'type': 'chondro_marker', 'label': 'Col2a1'},

        # PHASE 2: OSTEOGENIC PATHWAY
        'Runx2': {'pos': (13, -1), 'type': 'osteo_tf', 'label': 'Runx2'},
        'Osterix': {'pos': (16, -1), 'type': 'osteo_tf', 'label': 'Osterix'},
        'Col1a1': {'pos': (19, -1), 'type': 'osteo_marker', 'label': 'Col1a1'},
    }

    # Node colors by type
    colors = {
        'signal': '#FFD700',        # Gold
        'border': '#87CEEB',        # Sky blue
        'nc_specifier': '#9370DB',  # Medium purple
        'chondro_tf': '#32CD32',    # Lime green
        'chondro_marker': '#006400', # Dark green
        'osteo_tf': '#FF6347',      # Tomato red
        'osteo_marker': '#8B0000',  # Dark red
    }

    # Draw nodes
    for gene, info in genes.items():
        x, y = info['pos']
        color = colors[info['type']]

        if info['type'] in ['signal']:
            # Ellipse for signals
            ellipse = mpatches.Ellipse((x, y), width=0.8, height=0.4,
                                      facecolor=color, edgecolor='black', lw=2)
            ax.add_patch(ellipse)
        else:
            # Rectangle for genes
            rect = mpatches.Rectangle((x - 0.35, y - 0.2), 0.7, 0.4,
                                     facecolor=color, edgecolor='black', lw=2)
            ax.add_patch(rect)

        ax.text(x, y, info['label'], ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')

    # PHASE 1 REGULATIONS
    # BMP/Wnt → Pax3
    draw_arrow(ax, genes['BMP']['pos'], genes['Pax3']['pos'],
              color='green', start_shape='ellipse', arrow_type='positive')
    draw_arrow(ax, genes['Wnt']['pos'], genes['Pax3']['pos'],
              color='green', start_shape='ellipse', arrow_type='positive')

    # BMP/Wnt → Zic1
    draw_arrow(ax, genes['BMP']['pos'], genes['Zic1']['pos'],
              color='green', start_shape='ellipse', arrow_type='positive')
    draw_arrow(ax, genes['Wnt']['pos'], genes['Zic1']['pos'],
              color='green', start_shape='ellipse', arrow_type='positive')

    # Pax3 → FoxD3
    draw_arrow(ax, genes['Pax3']['pos'], genes['FoxD3']['pos'],
              color='green', arrow_type='positive')

    # Zic1 → FoxD3
    draw_arrow(ax, genes['Zic1']['pos'], genes['FoxD3']['pos'],
              color='green', arrow_type='positive')

    # FoxD3 → Snail2
    draw_arrow(ax, genes['FoxD3']['pos'], genes['Snail2']['pos'],
              color='green', arrow_type='positive')

    # FoxD3 → Sox9_NC
    draw_arrow(ax, genes['FoxD3']['pos'], genes['Sox9_NC']['pos'],
              color='green', arrow_type='positive')

    # FoxD3 → Sox10
    draw_arrow(ax, genes['FoxD3']['pos'], genes['Sox10']['pos'],
              color='green', arrow_type='positive')

    # Transition arrow from Sox9_NC to Sox9
    ax.annotate('', xy=genes['Sox9']['pos'], xytext=genes['Sox9_NC']['pos'],
               arrowprops=dict(arrowstyle='->', color='purple', lw=3,
                              linestyle='--', shrinkA=5, shrinkB=5))
    ax.text(11, 5.5, 'Phase\nTransition', ha='center', va='center',
           fontsize=9, style='italic', color='purple',
           bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))

    # PHASE 2 REGULATIONS: CHONDROGENIC PATHWAY
    # Sox9 self-loop
    draw_self_loop(ax, genes['Sox9']['pos'], color='green', radius=0.45)

    # Sox9 → Sox5
    draw_arrow(ax, genes['Sox9']['pos'], genes['Sox5']['pos'],
              color='green', arrow_type='positive')

    # Sox9 → Sox6
    draw_arrow(ax, genes['Sox9']['pos'], genes['Sox6']['pos'],
              color='green', arrow_type='positive')

    # Sox5 → Col2a1
    draw_arrow(ax, genes['Sox5']['pos'], genes['Col2a1']['pos'],
              color='green', arrow_type='positive')

    # Sox6 → Col2a1
    draw_arrow(ax, genes['Sox6']['pos'], genes['Col2a1']['pos'],
              color='green', arrow_type='positive')

    # PHASE 2 REGULATIONS: OSTEOGENIC PATHWAY
    # BMP → Runx2 (curved arrow to avoid clutter)
    bmp_x, bmp_y = genes['BMP']['pos']
    runx2_x, runx2_y = genes['Runx2']['pos']
    # Draw curved connection
    ax.annotate('', xy=(runx2_x - 0.35, runx2_y + 0.2),
               xytext=(bmp_x + 0.4, bmp_y - 0.2),
               arrowprops=dict(arrowstyle='->', color='green', lw=2,
                              connectionstyle='arc3,rad=0.3'))

    # Runx2 → Osterix
    draw_arrow(ax, genes['Runx2']['pos'], genes['Osterix']['pos'],
              color='green', arrow_type='positive')

    # Osterix → Col1a1
    draw_arrow(ax, genes['Osterix']['pos'], genes['Col1a1']['pos'],
              color='green', arrow_type='positive')

    # MUTUAL EXCLUSION: Sox9 ⊣ Runx2
    sox9_pos = genes['Sox9']['pos']
    runx2_pos = genes['Runx2']['pos']
    # Start from Sox9 bottom edge
    start_x, start_y = sox9_pos[0], sox9_pos[1] - 0.2
    # End at Runx2 top edge
    end_x, end_y = runx2_pos[0], runx2_pos[1] + 0.2

    # Draw inhibition line with blunt end
    ax.plot([start_x, end_x], [start_y, end_y], 'r-', lw=3)
    # Add blunt end marker
    ax.plot([end_x - 0.15, end_x + 0.15], [end_y, end_y], 'r-', lw=3)

    ax.text((start_x + end_x)/2 - 0.5, (start_y + end_y)/2, 'Mutual\nExclusion',
           ha='center', va='center', fontsize=9, style='italic', color='red',
           bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    # Add phase labels
    ax.text(6, 8.5, 'PHASE 1: Neural Crest Specification',
           fontsize=16, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='lavender', edgecolor='blue', lw=2))

    ax.text(15.5, 8.5, 'PHASE 2: Craniofacial Differentiation',
           fontsize=16, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', lw=2))

    # Add pathway labels
    ax.text(14.5, 7.5, 'Chondrogenic\nPathway',
           fontsize=12, fontweight='bold', ha='center', color='darkgreen',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

    ax.text(16, -2.5, 'Osteogenic\nPathway',
           fontsize=12, fontweight='bold', ha='center', color='darkred',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['signal'], edgecolor='black', label='Input Signal'),
        mpatches.Patch(facecolor=colors['border'], edgecolor='black', label='Border Specifier'),
        mpatches.Patch(facecolor=colors['nc_specifier'], edgecolor='black', label='Neural Crest Specifier'),
        mpatches.Patch(facecolor=colors['chondro_tf'], edgecolor='black', label='Chondrogenic TF'),
        mpatches.Patch(facecolor=colors['chondro_marker'], edgecolor='black', label='Chondrogenic Marker'),
        mpatches.Patch(facecolor=colors['osteo_tf'], edgecolor='black', label='Osteogenic TF'),
        mpatches.Patch(facecolor=colors['osteo_marker'], edgecolor='black', label='Osteogenic Marker'),
        mpatches.Patch(facecolor='white', edgecolor='green', label='Positive Regulation →'),
        mpatches.Patch(facecolor='white', edgecolor='red', label='Negative Regulation ⊣'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             framealpha=0.95, edgecolor='black', fancybox=True)

    # Add literature citations
    citation_text = (
        "Literature-Based Model:\n"
        "• Phase 1: Simões-Costa & Bronner (2015)\n"
        "• Chondrogenesis: Akiyama et al. (2002)\n"
        "• Osteogenesis: Komori (2019)\n"
        "• Mutual Exclusion: Zhou et al. (2006)"
    )
    ax.text(19.5, 7, citation_text, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Set axis properties
    ax.set_xlim(-1.5, 21)
    ax.set_ylim(-3.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.title('Minimal Literature-Based Neural Crest GRN\n' +
             'Neural Crest Specification → Craniofacial Differentiation',
             fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('minimal_grn_architecture.png', dpi=300, bbox_inches='tight')
    print("GRN architecture saved to 'minimal_grn_architecture.png'")
    plt.show()


if __name__ == '__main__':
    visualize_minimal_grn()
