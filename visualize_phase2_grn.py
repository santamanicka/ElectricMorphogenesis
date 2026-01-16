#!/usr/bin/env python3

"""
Phase 2 Gene Regulatory Network Visualization

Creates a directed graph showing the regulatory relationships in the craniofacial
skeletogenesis phase, with clear distinction between positive and negative regulations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patches as patches

def create_phase2_grn_graph():
    """Create the Phase 2 GRN network graph"""

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes (genes/factors)
    genes = {
        # Core transcription factors
        'Sox9': {'type': 'chondro_tf', 'pos': (0, 0)},
        'Sox5': {'type': 'chondro_tf', 'pos': (-1.5, -1)},
        'Sox6': {'type': 'chondro_tf', 'pos': (1, -1)},
        'Runx2': {'type': 'osteo_tf', 'pos': (3, 0)},
        'Osterix': {'type': 'osteo_tf', 'pos': (4, -1)},

        # Antagonists
        'Msx1': {'type': 'antagonist', 'pos': (-2, 1)},
        'Msx2': {'type': 'antagonist', 'pos': (-1, 1)},

        # Matrix proteins
        'Col2a1': {'type': 'chondro_matrix', 'pos': (-2, -2)},
        'Aggrecan': {'type': 'chondro_matrix', 'pos': (0, -2)},
        'Col1a1': {'type': 'osteo_matrix', 'pos': (4, -2)},  # Moved closer to Osterix

        # Signals
        'BMP': {'type': 'signal', 'pos': (1, 2)},
    }

    # Add nodes to graph
    for gene, attrs in genes.items():
        G.add_node(gene, **attrs)

    # Define regulatory relationships
    # Format: (source, target, regulation_type, strength)
    regulations = [
        # Positive regulations (activations)
        ('Sox9', 'Sox9', 'positive', 'strong'),      # Auto-regulation
        ('Sox9', 'Sox5', 'positive', 'strong'),      # Sox9 → Sox5
        ('Sox9', 'Sox6', 'positive', 'strong'),      # Sox9 → Sox6
        ('Sox9', 'Col2a1', 'positive', 'medium'),    # Sox9 → Col2a1
        ('Sox9', 'Aggrecan', 'positive', 'medium'),  # Sox9 → Aggrecan
        ('Sox5', 'Col2a1', 'positive', 'medium'),    # Sox5 → Col2a1 (cooperative)
        ('BMP', 'Runx2', 'positive', 'strong'),      # BMP → Runx2
        ('BMP', 'Msx1', 'positive', 'medium'),       # BMP → Msx1
        ('BMP', 'Msx2', 'positive', 'medium'),       # BMP → Msx2
        ('Runx2', 'Osterix', 'positive', 'strong'),  # Runx2 → Osterix
        ('Osterix', 'Col1a1', 'positive', 'strong'), # Osterix → Col1a1

        # Negative regulations (inhibitions)
        ('Sox9', 'Runx2', 'negative', 'strong'),     # Sox9 ⊣ Runx2
        ('Msx1', 'Sox9', 'negative', 'medium'),      # Msx1 ⊣ Sox9
        ('Msx2', 'Sox9', 'negative', 'medium'),      # Msx2 ⊣ Sox9
    ]

    # Add edges with attributes
    for source, target, reg_type, strength in regulations:
        G.add_edge(source, target, regulation=reg_type, strength=strength)

    return G, genes

def get_node_boundary_point(gene, genes, pos, target_x, target_y):
    """Calculate the point on the boundary of a node shape towards a target"""
    x, y = pos[gene]
    gene_type = genes[gene]['type']

    # Calculate direction vector
    dx = target_x - x
    dy = target_y - y

    # Normalize direction
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return x, y

    dx_norm = dx / length
    dy_norm = dy / length

    # Calculate boundary point based on shape
    if gene_type in ['chondro_tf', 'osteo_tf']:
        # Rectangle: find intersection with rectangle boundary
        # Half-widths: 0.5, half-height: 0.25
        t_x = 0.5 / abs(dx_norm) if dx_norm != 0 else float('inf')
        t_y = 0.25 / abs(dy_norm) if dy_norm != 0 else float('inf')
        t = min(t_x, t_y)
        return x + t * dx_norm - 0.2, y + t * dy_norm

    elif (gene_type == 'antagonist') or (gene == 'Sox9'):
        # Diamond: radius 0.4
        return x + 0.4 * dx_norm, y + 0.4 * dy_norm

    elif gene_type in ['chondro_matrix', 'osteo_matrix']:
        # Ellipse: semi-major axis 0.4, semi-minor axis 0.2
        # Approximate with average radius
        return x + 0.3 * dx_norm, y + 0.3 * dy_norm

    else:  # signal
        # Hexagon: radius 0.35
        return x + 0.35 * dx_norm, y + 0.35 * dy_norm

def plot_grn_network(G, genes, save_path='phase2_grn_network.png'):
    """Plot the GRN network with clear visual distinctions"""

    # Create figure with more space
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Define colors for different gene types
    colors = {
        'chondro_tf': '#4CAF50',      # Green for chondrogenic TFs
        'osteo_tf': '#FF5722',        # Red for osteogenic TFs
        'antagonist': '#9C27B0',      # Purple for antagonists
        'chondro_matrix': '#81C784',   # Light green for cartilage matrix
        'osteo_matrix': '#FF8A65',     # Light red for bone matrix
        'signal': '#2196F3'           # Blue for signals
    }

    # Get positions
    pos = {gene: attrs['pos'] for gene, attrs in genes.items()}

    # Scale positions for better visualization and more spacing
    for gene in pos:
        pos[gene] = (pos[gene][0] * 3.0, pos[gene][1] * 2.5)

    # Draw nodes with different shapes and colors
    for gene, attrs in genes.items():
        x, y = pos[gene]
        color = colors[attrs['type']]

        # Different shapes for different types (larger for better visibility)
        if attrs['type'] in ['chondro_tf', 'osteo_tf']:
            # Rectangles for transcription factors
            bbox = FancyBboxPatch((x-0.5, y-0.25), 1.0, 0.5,
                                boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(bbox)
        elif attrs['type'] == 'antagonist':
            # Diamonds for antagonists
            diamond = mpatches.RegularPolygon((x, y), 4, radius=0.4,
                                            orientation=np.pi/4,
                                            facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(diamond)
        elif attrs['type'] in ['chondro_matrix', 'osteo_matrix']:
            # Ellipses for matrix proteins
            ellipse = mpatches.Ellipse((x, y), 0.8, 0.4,
                                     facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(ellipse)
        else:  # signal
            # Hexagon for signals
            hexagon = mpatches.RegularPolygon((x, y), 6, radius=0.35,
                                            facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(hexagon)

        # Add gene labels (larger font)
        ax.text(x, y, gene, ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw edges with different styles for positive/negative regulation
    for edge in G.edges(data=True):
        source, target, attrs = edge
        reg_type = attrs['regulation']
        strength = attrs['strength']

        # Get coordinates
        x1, y1 = pos[source]
        x2, y2 = pos[target]

        # Handle self-loops (Sox9 auto-regulation)
        if source == target:
            # Draw smaller, cleaner self-loop above the node
            loop_radius = 0.6
            loop_center_x = x1
            loop_center_y = y1 + 0.7  # Smaller offset above the node

            # Connection points at the top corners of the rectangle
            start_x = x1 - 0.3  # Left corner of top edge
            start_y = y1 + 0.25  # Top edge of rectangle
            end_x = x1 + 0.3   # Right corner of top edge
            end_y = y1 + 0.25  # Top edge of rectangle

            # Draw the curved path using a bezier-like curve
            # Create control points for smooth curve
            ctrl1_x = start_x - 0.2
            ctrl1_y = start_y + 0.8
            ctrl2_x = end_x + 0.2
            ctrl2_y = end_y + 0.8

            # Draw the self-loop as a smooth curve
            # Create a curved path
            verts = [
                (start_x, start_y),     # Start point
                (ctrl1_x, ctrl1_y),     # Control point 1
                (ctrl2_x, ctrl2_y),     # Control point 2
                (end_x, end_y),         # End point
            ]

            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', edgecolor='green', linewidth=3)
            ax.add_patch(patch)

            # Add arrowhead at the end point
            ax.annotate('', xy=(end_x, end_y), xytext=(end_x - 0.1, end_y + 0.1),
                       arrowprops=dict(arrowstyle='->', color='green', lw=3))
            continue

        # Choose color and style based on regulation type
        if reg_type == 'positive':
            color = 'green'
            linestyle = '-'
            arrowstyle = '->'
        else:  # negative
            color = 'red'
            linestyle = '-'
            arrowstyle = '-['  # Blunt arrow for inhibition

        # Choose line width based on strength
        linewidth = 3 if strength == 'strong' else 2

        # Calculate boundary points for proper arrow positioning
        start_x, start_y = get_node_boundary_point(source, genes, pos, x2, y2)
        end_x, end_y = get_node_boundary_point(target, genes, pos, x1, y1)

        # Draw edge from boundary to boundary
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                 lw=linewidth, linestyle=linestyle))

    # Add title and labels
    ax.set_title('Phase 2 Gene Regulatory Network\nCraniofacial Skeletogenesis',
                fontsize=16, fontweight='bold', pad=20)

    # Create legend
    legend_elements = [
        # Gene types
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['chondro_tf'],
                          edgecolor='black', label='Chondrogenic TFs'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['osteo_tf'],
                          edgecolor='black', label='Osteogenic TFs'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['antagonist'],
                          edgecolor='black', label='Antagonists'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['chondro_matrix'],
                          edgecolor='black', label='Cartilage Matrix'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['osteo_matrix'],
                          edgecolor='black', label='Bone Matrix'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['signal'],
                          edgecolor='black', label='Signals'),

        # Regulation types
        plt.Line2D([0], [0], color='green', lw=3, label='Positive regulation'),
        plt.Line2D([0], [0], color='red', lw=3, label='Negative regulation'),
    ]

    # Position legend outside the plot area
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    # Set equal aspect ratio and remove axes with more space
    ax.set_aspect('equal')
    ax.set_xlim(-9, 15)  # Extended to ensure Col1a1 is fully visible
    ax.set_ylim(-7, 7)
    ax.axis('off')

    # Add text annotations for key regulatory modules (repositioned for larger layout)
    ax.text(-3, -6, 'Chondrogenic Module', fontsize=12, fontweight='bold',
           ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

    ax.text(9, -6, 'Osteogenic Module', fontsize=12, fontweight='bold',
           ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

    ax.text(-3, 5, 'Antagonistic Regulation', fontsize=12, fontweight='bold',
           ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='plum', alpha=0.7))

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Phase 2 GRN network saved to {save_path}")

    return fig

def create_regulatory_summary():
    """Create a summary table of all regulatory relationships"""

    regulations = [
        # Positive regulations
        ('Sox9', 'Sox9', 'Positive', 'Auto-regulation for maintenance'),
        ('Sox9', 'Sox5', 'Positive', 'Master chondrogenic regulator'),
        ('Sox9', 'Sox6', 'Positive', 'Master chondrogenic regulator'),
        ('Sox9', 'Col2a1', 'Positive', 'Cartilage matrix production'),
        ('Sox9', 'Aggrecan', 'Positive', 'Cartilage matrix production'),
        ('Sox5', 'Col2a1', 'Positive', 'Cooperative cartilage matrix activation'),
        ('BMP', 'Runx2', 'Positive', 'Osteogenic specification'),
        ('BMP', 'Msx1', 'Positive', 'Antagonist activation'),
        ('BMP', 'Msx2', 'Positive', 'Antagonist activation'),
        ('Runx2', 'Osterix', 'Positive', 'Osteogenic differentiation'),
        ('Osterix', 'Col1a1', 'Positive', 'Bone matrix production'),

        # Negative regulations
        ('Sox9', 'Runx2', 'Negative', 'Cell fate decision (mutual exclusion)'),
        ('Msx1', 'Sox9', 'Negative', 'Anti-chondrogenic activity'),
        ('Msx2', 'Sox9', 'Negative', 'Anti-chondrogenic activity'),
    ]

    print("\\n" + "="*80)
    print("PHASE 2 REGULATORY RELATIONSHIPS SUMMARY")
    print("="*80)
    print(f"{'Source':<10} {'Target':<10} {'Type':<10} {'Description':<40}")
    print("-"*80)

    for source, target, reg_type, description in regulations:
        print(f"{source:<10} {target:<10} {reg_type:<10} {description:<40}")

    print("\\nKey Regulatory Principles:")
    print("1. Sox9 acts as master chondrogenic regulator")
    print("2. Runx2-Osterix pathway drives osteogenesis")
    print("3. Sox9 and Runx2 mutually inhibit each other")
    print("4. Msx proteins antagonize chondrogenesis")
    print("5. BMP signal activates both osteogenic and antagonistic pathways")

def main():
    """Main function to create and display the Phase 2 GRN"""

    print("Creating Phase 2 Gene Regulatory Network Visualization...")

    # Create the network graph
    G, genes = create_phase2_grn_graph()

    # Plot the network
    fig = plot_grn_network(G, genes)

    # Show regulatory summary
    create_regulatory_summary()

    # Display the plot
    # plt.show()
    plt.savefig('./Phase2GRN.png', dpi=300, bbox_inches='tight', facecolor='white')

    print("\\nNetwork visualization complete!")
    print("Key features:")
    print("- Green arrows: Positive regulation (activation)")
    print("- Red arrows: Negative regulation (inhibition)")
    print("- Line thickness: Regulatory strength")
    print("- Node shapes: Gene/factor types")
    print("- Color coding: Functional modules")

if __name__ == "__main__":
    main()