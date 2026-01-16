"""
Field as Language: Translation from Electric Patterns to Biochemical Signals

This visualization shows how embryos "translate" electric field patterns
into their native biochemical code (ions, genes, proteins, morphogens).

Think of it like translating between languages:
- Electric field patterns = the "spoken" language (communication medium)
- Ion channels, voltage sensors = the "ears" (receptors)
- Ca²⁺, CaMKII, genes = the "native code" (internal processing)
- Proteins, morphogens = the "actions" (developmental outcomes)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle, Wedge
from matplotlib.patches import Polygon, Ellipse
import matplotlib.patheffects as path_effects


def draw_wavy_arrow(ax, start, end, n_waves=2, color='black', linewidth=2,
                    arrowstyle='->', label=None, label_offset=(0, 0)):
    """Draw a wavy arrow between two points."""
    sx, sy = start
    ex, ey = end

    # Create wavy path
    t = np.linspace(0, 1, 50)
    x = sx + t * (ex - sx)
    y = sy + t * (ey - sy)

    # Add wave perpendicular to main direction
    dx, dy = ex - sx, ey - sy
    length = np.sqrt(dx**2 + dy**2)
    perp_x, perp_y = -dy/length, dx/length

    wave_amp = 0.1
    y_wave = y + wave_amp * np.sin(t * n_waves * 2 * np.pi) * perp_x
    x_wave = x + wave_amp * np.sin(t * n_waves * 2 * np.pi) * perp_y

    ax.plot(x_wave, y_wave, color=color, linewidth=linewidth, alpha=0.7)

    # Add arrowhead at end
    ax.annotate('', xy=(ex, ey), xytext=(x_wave[-5], y_wave[-5]),
                arrowprops=dict(arrowstyle=arrowstyle, color=color, lw=linewidth))

    # Add label if provided
    if label:
        mid_x, mid_y = (sx + ex) / 2 + label_offset[0], (sy + ey) / 2 + label_offset[1]
        ax.text(mid_x, mid_y, label, fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=color, linewidth=1.5))


def create_translation_layers_figure():
    """
    Main figure showing the translation from field to biochemistry.

    Layout: Five layers from top to bottom
    1. Electric Field Pattern (external "language")
    2. Voltage Sensors (receptors/"ears")
    3. Ion Dynamics (immediate response)
    4. Molecular Signaling (signal processing)
    5. Gene Expression & Morphogens (developmental "actions")
    """
    fig = plt.figure(figsize=(20, 16))

    fig.suptitle('Electric Fields as Language: From Pattern to Protein',
                fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.95,
             'How embryos translate external electric "words" into internal biochemical "meaning"',
             ha='center', fontsize=13, style='italic', color='#555555')

    # Create main axis
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.82])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 22)
    ax.axis('off')

    # =========================================================================
    # LAYER 1: Electric Field Pattern (Top) - "THE LANGUAGE"
    # =========================================================================
    layer1_y = 19.5

    # Background box for layer 1
    layer1_box = FancyBboxPatch((0.5, layer1_y-0.8), 19, 2.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#E6F3FF', edgecolor='#4169E1',
                                linewidth=3, alpha=0.3)
    ax.add_patch(layer1_box)

    ax.text(1, layer1_y+1.5, 'LAYER 1: ELECTRIC FIELD PATTERN',
            fontsize=13, fontweight='bold', color='#4169E1')
    ax.text(1, layer1_y+1.0, '"The Communication Language"',
            fontsize=10, style='italic', color='#4169E1')

    # Draw field pattern visualization
    n_arrows = 8
    for i in range(n_arrows):
        x = 3 + i * 2.0
        y = layer1_y + 0.2

        # Field varies with position (radial-like pattern)
        center_x = 10
        dx = (x - center_x) * 0.15
        dy = 0.3 if abs(x - center_x) < 3 else -0.2

        # Field magnitude shown by color intensity
        mag = np.sqrt(dx**2 + dy**2)
        color = plt.cm.PiYG(mag / 0.5)

        ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.15,
                fc=color, ec=color, linewidth=2, zorder=10)

    # Label
    ax.text(19, layer1_y+0.3, 'Vector Field\n(direction + magnitude)',
            ha='right', fontsize=9, color='#4169E1',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # =========================================================================
    # TRANSLATION ZONE 1: Field → Voltage (receptors)
    # =========================================================================
    trans1_y = 17.5

    # Translation arrow
    for x in [5, 15]:
        ax.annotate('', xy=(x, trans1_y+0.3), xytext=(x, layer1_y-1.0),
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=2))

    ax.text(10, trans1_y+0.7, 'TRANSLATION 1: Cells "read" field',
            ha='center', fontsize=10, fontweight='bold', color='#888888')

    # =========================================================================
    # LAYER 2: Membrane Voltage (Vmem) - "THE RECEPTOR"
    # =========================================================================
    layer2_y = 15.5

    layer2_box = FancyBboxPatch((0.5, layer2_y-0.8), 19, 2.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFF3E6', edgecolor='#FF8C00',
                                linewidth=3, alpha=0.3)
    ax.add_patch(layer2_box)

    ax.text(1, layer2_y+1.5, 'LAYER 2: MEMBRANE VOLTAGE (Vmem)',
            fontsize=13, fontweight='bold', color='#FF8C00')
    ax.text(1, layer2_y+1.0, '"The Receptor / Ears"',
            fontsize=10, style='italic', color='#FF8C00')

    # Draw cell membranes with voltages
    for i in range(7):
        x = 3.5 + i * 2.3
        y = layer2_y + 0.2

        # Voltage varies with position
        vmem = -50 + 20 * np.sin(i * np.pi / 3)

        # Color by voltage
        if vmem > -35:
            color = '#FFB6C1'  # Depolarized
        elif vmem > -45:
            color = '#FFFACD'  # Intermediate
        else:
            color = '#90EE90'  # Hyperpolarized

        # Draw cell
        cell = Circle((x, y), 0.4, facecolor=color, edgecolor='#333333',
                     linewidth=2, zorder=5)
        ax.add_patch(cell)

        # Voltage label
        ax.text(x, y-0.7, f'{int(vmem)} mV', ha='center', fontsize=7,
               color='#333333')

    # Legend
    ax.text(19, layer2_y+0.8, 'Depolarized', ha='right', fontsize=8,
            color='#FF1493', fontweight='bold')
    ax.text(19, layer2_y+0.4, 'Intermediate', ha='right', fontsize=8,
            color='#DAA520', fontweight='bold')
    ax.text(19, layer2_y+0.0, 'Hyperpolarized', ha='right', fontsize=8,
            color='#228B22', fontweight='bold')

    # =========================================================================
    # TRANSLATION ZONE 2: Voltage → Ion Channels
    # =========================================================================
    trans2_y = 13.5

    for x in [5, 15]:
        ax.annotate('', xy=(x, trans2_y+0.3), xytext=(x, layer2_y-1.0),
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=2))

    ax.text(10, trans2_y+0.7, 'TRANSLATION 2: Voltage-gated channels open',
            ha='center', fontsize=10, fontweight='bold', color='#888888')

    # =========================================================================
    # LAYER 3: Ion Dynamics (Ca²⁺ influx) - "IMMEDIATE RESPONSE"
    # =========================================================================
    layer3_y = 11.5

    layer3_box = FancyBboxPatch((0.5, layer3_y-0.8), 19, 2.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFE6F0', edgecolor='#DC143C',
                                linewidth=3, alpha=0.3)
    ax.add_patch(layer3_box)

    ax.text(1, layer3_y+1.5, 'LAYER 3: CALCIUM (Ca²⁺) DYNAMICS',
            fontsize=13, fontweight='bold', color='#DC143C')
    ax.text(1, layer3_y+1.0, '"Immediate Response / Signal Amplification"',
            fontsize=10, style='italic', color='#DC143C')

    # Draw Ca²⁺ concentration waves
    x_vals = np.linspace(3, 18, 100)
    ca_pattern = 0.5 + 0.4 * np.sin(x_vals * 0.8) * np.exp(-(x_vals-10)**2/20)

    ax.fill_between(x_vals, layer3_y+0.1, layer3_y+0.1 + ca_pattern,
                    color='#DC143C', alpha=0.5)
    ax.plot(x_vals, layer3_y+0.1 + ca_pattern, color='#DC143C', linewidth=2)

    # Ca²⁺ ions
    for i in range(8):
        x = 4 + i * 2.0
        ca_height = 0.5 + 0.4 * np.sin(x * 0.8) * np.exp(-(x-10)**2/20)
        y = layer3_y + 0.2 + ca_height + 0.1
        ax.text(x, y, 'Ca²⁺', fontsize=10, ha='center', color='#DC143C',
               fontweight='bold',
               bbox=dict(boxstyle='circle', facecolor='white',
                        edgecolor='#DC143C', linewidth=2))

    ax.text(19.4, layer3_y+0.3, '[Ca²⁺]\nConcentration',
            ha='right', fontsize=9, color='#DC143C',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # =========================================================================
    # TRANSLATION ZONE 3: Ca²⁺ → Molecular Signaling
    # =========================================================================
    trans3_y = 9.5

    for x in [5, 15]:
        ax.annotate('', xy=(x, trans3_y+0.3), xytext=(x, layer3_y-1.0),
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=2))

    ax.text(10, trans3_y+0.7, 'TRANSLATION 3: Ca²⁺ activates molecular switches',
            ha='center', fontsize=10, fontweight='bold', color='#888888')

    # =========================================================================
    # LAYER 4: Molecular Signaling (CaMKII, PKC, etc.) - "SIGNAL PROCESSING"
    # =========================================================================
    layer4_y = 7.5

    layer4_box = FancyBboxPatch((0.5, layer4_y-0.8), 19, 2.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#F0E6FF', edgecolor='#8B008B',
                                linewidth=3, alpha=0.3)
    ax.add_patch(layer4_box)

    ax.text(1, layer4_y+1.5, 'LAYER 4: MOLECULAR SIGNALING',
            fontsize=13, fontweight='bold', color='#8B008B')
    ax.text(1, layer4_y+1.0, '"Signal Processing / Memory Formation"',
            fontsize=10, style='italic', color='#8B008B')

    # Draw molecular switches
    molecules = [
        ('CaMKII', 5, 'Bistable\nMemory'),
        ('PKC', 8.5, 'Signal\nAmplification'),
        ('CREB', 12, 'Gene\nActivation'),
        ('NFκB', 15.5, 'Stress\nResponse')
    ]

    for mol_name, x, subtitle in molecules:
        y = layer4_y + 0.3

        # Draw molecule as hexagon
        angles = np.linspace(0, 2*np.pi, 7)
        radius = 0.65
        hex_x = x + radius * np.cos(angles)
        hex_y = y + radius * np.sin(angles)

        hexagon = Polygon(list(zip(hex_x, hex_y)),
                         facecolor='#DDA0DD', edgecolor='#8B008B',
                         linewidth=2)
        ax.add_patch(hexagon)

        ax.text(x, y+0.15, mol_name, ha='center', fontsize=9,
               fontweight='bold', color='#4B0082')
        ax.text(x, y-0.15, subtitle.replace('\n', ' '), ha='center',
               fontsize=7, color='#4B0082')

    # =========================================================================
    # TRANSLATION ZONE 4: Molecular → Genetic/Protein
    # =========================================================================
    trans4_y = 5.5

    for x in [5, 15]:
        ax.annotate('', xy=(x, trans4_y+0.3), xytext=(x, layer4_y-1.0),
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=2))

    ax.text(10, trans4_y+0.7, 'TRANSLATION 4: Activate genes & produce proteins',
            ha='center', fontsize=10, fontweight='bold', color='#888888')

    # =========================================================================
    # LAYER 5: Gene Expression & Morphogens - "THE ACTION"
    # =========================================================================
    layer5_y = 3.5

    layer5_box = FancyBboxPatch((0.5, layer5_y-0.8), 19, 2.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#E6FFE6', edgecolor='#228B22',
                                linewidth=3, alpha=0.3)
    ax.add_patch(layer5_box)

    ax.text(1, layer5_y+1.5, 'LAYER 5: GENES & MORPHOGENS',
            fontsize=13, fontweight='bold', color='#228B22')
    ax.text(1, layer5_y+1.0, '"Developmental Actions / Physical Changes"',
            fontsize=10, style='italic', color='#228B22')

    # Draw gene expression and outcomes
    outcomes = [
        ('Pax6\nGene', 4, 'Eye\nFormation', '#4169E1'),
        ('Dlx\nGene', 7.5, 'Jaw\nDevelopment', '#FF8C00'),
        ('Runx2\nGene', 11, 'Bone\nFormation', '#8B4513'),
        ('SHH\nMorphogen', 14.5, 'Pattern\nOrganizer', '#9370DB'),
        ('FGF8\nMorphogen', 18, 'Growth\nSignal', '#DC143C')
    ]

    for gene_name, x, outcome, color in outcomes:
        y = layer5_y + 0.3

        # Gene/protein symbol
        gene_box = Rectangle((x-0.6, y-0.3), 1.2, 0.6,
                            facecolor=color, edgecolor='#333333',
                            linewidth=2, alpha=0.6)
        ax.add_patch(gene_box)

        ax.text(x, y, gene_name.replace('\n', ' '), ha='center', fontsize=8,
               fontweight='bold', color='white')

        # Outcome arrow and label
        ax.annotate('', xy=(x, y-0.7), xytext=(x, y-0.4),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.text(x, y-0.95, outcome.replace('\n', ' '), ha='center',
               fontsize=7, color=color, fontweight='bold')

    # =========================================================================
    # Final outcome banner
    # =========================================================================
    outcome_y = 1.0
    outcome_box = FancyBboxPatch((2, outcome_y-0.3), 16, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor='#FFFACD', edgecolor='#DAA520',
                                linewidth=2, alpha=0.9)
    ax.add_patch(outcome_box)

    ax.text(10, outcome_y+0.15, 'RESULT: Coordinated Development',
            ha='center', fontsize=12, fontweight='bold', color='#B8860B')
    ax.text(10, outcome_y-0.15, 'Tissues form in right place, at right time, with right shape',
            ha='center', fontsize=9, color='#333333')

    # =========================================================================
    # Side annotation: "Like a translation chain"
    # =========================================================================
    # Draw vertical bracket on right side
    bracket_x = 20
    ax.plot([bracket_x, bracket_x], [layer1_y-0.5, layer5_y-0.5],
           'k-', linewidth=2)
    ax.plot([bracket_x, bracket_x-0.2], [layer1_y-0.5, layer1_y-0.5],
           'k-', linewidth=2)
    ax.plot([bracket_x, bracket_x-0.2], [layer5_y-0.5, layer5_y-0.5],
           'k-', linewidth=2)

    # Annotation text rotated
    ax.text(bracket_x+0.3, (layer1_y + layer5_y)/2,
           'Multi-layer Translation:\n High-Dim \u2190 Low-Dim',
            rotation=90, va='center', ha='left', fontsize=11,
            fontweight='bold', color='#333333')

    return fig


def create_language_analogy_figure():
    """
    Create a figure using language/communication analogy.
    Shows parallel between human language translation and field translation.
    """
    fig = plt.figure(figsize=(18, 10))

    fig.suptitle('Electric Fields as Communication: A Language Analogy',
                fontsize=18, fontweight='bold', y=0.96)

    # =========================================================================
    # Left side: Human Language Translation
    # =========================================================================
    ax_left = fig.add_axes([0.05, 0.15, 0.42, 0.75])
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 12)
    ax_left.axis('off')
    ax_left.set_title('Human Language Translation', fontsize=14,
                     fontweight='bold', pad=10)

    # Layer 1: Spoken words
    y = 10
    box = FancyBboxPatch((1, y-0.5), 8, 1.5, boxstyle="round,pad=0.1",
                        facecolor='#E6F3FF', edgecolor='#4169E1', linewidth=2)
    ax_left.add_patch(box)
    ax_left.text(5, y+0.5, 'SOUND WAVES', fontsize=11, fontweight='bold',
                ha='center', color='#4169E1')
    ax_left.text(5, y-0.1, '"Hello, friend!"', fontsize=10, ha='center',
                style='italic')

    # Arrow
    ax_left.annotate('', xy=(5, y-1.3), xytext=(5, y-0.7),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#888'))
    ax_left.text(7, y-0.9, 'Ears hear', fontsize=9, color='#888')

    # Layer 2: Ear reception
    y = 7.5
    box = FancyBboxPatch((1, y-0.5), 8, 1.5, boxstyle="round,pad=0.1",
                        facecolor='#FFF3E6', edgecolor='#FF8C00', linewidth=2)
    ax_left.add_patch(box)
    ax_left.text(5, y+0.5, 'SOUND → NERVE SIGNALS', fontsize=11,
                fontweight='bold', ha='center', color='#FF8C00')
    ax_left.text(5, y-0.1, 'Vibrations → Neural spikes', fontsize=9,
                ha='center')

    # Arrow
    ax_left.annotate('', xy=(5, y-1.3), xytext=(5, y-0.7),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#888'))
    ax_left.text(7, y-0.9, 'Process', fontsize=9, color='#888')

    # Layer 3: Brain processing
    y = 5
    box = FancyBboxPatch((1, y-0.5), 8, 1.5, boxstyle="round,pad=0.1",
                        facecolor='#FFE6F0', edgecolor='#DC143C', linewidth=2)
    ax_left.add_patch(box)
    ax_left.text(5, y+0.5, 'BRAIN DECODING', fontsize=11, fontweight='bold',
                ha='center', color='#DC143C')
    ax_left.text(5, y-0.1, 'Extract meaning & context', fontsize=9,
                ha='center')

    # Arrow
    ax_left.annotate('', xy=(5, y-1.3), xytext=(5, y-0.7),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#888'))
    ax_left.text(7, y-0.9, 'Understand', fontsize=9, color='#888')

    # Layer 4: Understanding & action
    y = 2.5
    box = FancyBboxPatch((1, y-0.5), 8, 1.5, boxstyle="round,pad=0.1",
                        facecolor='#E6FFE6', edgecolor='#228B22', linewidth=2)
    ax_left.add_patch(box)
    ax_left.text(5, y+0.5, 'MEANING → ACTION', fontsize=11, fontweight='bold',
                ha='center', color='#228B22')
    ax_left.text(5, y-0.1, 'Smile & wave back', fontsize=9, ha='center')

    # Bottom label
    ax_left.text(5, 0.5, 'External sound → Internal meaning → Physical action',
                ha='center', fontsize=9, style='italic', color='#666',
                bbox=dict(boxstyle='round', facecolor='#FFFACD', alpha=0.8))

    # =========================================================================
    # Right side: Embryo Field Translation
    # =========================================================================
    ax_right = fig.add_axes([0.53, 0.15, 0.42, 0.75])
    ax_right.set_xlim(0, 10)
    ax_right.set_ylim(0, 12)
    ax_right.axis('off')
    ax_right.set_title('Embryo Field Translation', fontsize=14,
                      fontweight='bold', pad=10)

    # Layer 1: Electric field
    y = 10
    box = FancyBboxPatch((1, y-0.5), 8, 1.5, boxstyle="round,pad=0.1",
                        facecolor='#E6F3FF', edgecolor='#4169E1', linewidth=2)
    ax_right.add_patch(box)
    ax_right.text(5, y+0.5, 'ELECTRIC FIELD', fontsize=11, fontweight='bold',
                 ha='center', color='#4169E1')
    ax_right.text(5, y-0.1, 'Vector pattern from neighbor', fontsize=10,
                 ha='center', style='italic')

    # Arrow
    ax_right.annotate('', xy=(5, y-1.3), xytext=(5, y-0.7),
                     arrowprops=dict(arrowstyle='->', lw=2, color='#888'))
    ax_right.text(7, y-0.9, 'Cells sense', fontsize=9, color='#888')

    # Layer 2: Voltage response
    y = 7.5
    box = FancyBboxPatch((1, y-0.5), 8, 1.5, boxstyle="round,pad=0.1",
                        facecolor='#FFF3E6', edgecolor='#FF8C00', linewidth=2)
    ax_right.add_patch(box)
    ax_right.text(5, y+0.5, 'FIELD → VOLTAGE (Vmem)', fontsize=11,
                 fontweight='bold', ha='center', color='#FF8C00')
    ax_right.text(5, y-0.1, 'Membrane depolarizes', fontsize=9, ha='center')

    # Arrow
    ax_right.annotate('', xy=(5, y-1.3), xytext=(5, y-0.7),
                     arrowprops=dict(arrowstyle='->', lw=2, color='#888'))
    ax_right.text(7, y-0.9, 'Ca²⁺ enters', fontsize=9, color='#888')

    # Layer 3: Molecular processing
    y = 5
    box = FancyBboxPatch((1, y-0.5), 8, 1.5, boxstyle="round,pad=0.1",
                        facecolor='#FFE6F0', edgecolor='#DC143C', linewidth=2)
    ax_right.add_patch(box)
    ax_right.text(5, y+0.5, 'Ca²⁺ → CaMKII → GENES', fontsize=11,
                 fontweight='bold', ha='center', color='#DC143C')
    ax_right.text(5, y-0.1, 'Decode pattern → gene activation', fontsize=9,
                 ha='center')

    # Arrow
    ax_right.annotate('', xy=(5, y-1.3), xytext=(5, y-0.7),
                     arrowprops=dict(arrowstyle='->', lw=2, color='#888'))
    ax_right.text(7, y-0.9, 'Express', fontsize=9, color='#888')

    # Layer 4: Developmental outcome
    y = 2.5
    box = FancyBboxPatch((1, y-0.5), 8, 1.5, boxstyle="round,pad=0.1",
                        facecolor='#E6FFE6', edgecolor='#228B22', linewidth=2)
    ax_right.add_patch(box)
    ax_right.text(5, y+0.5, 'GENES → MORPHOGENESIS', fontsize=11,
                 fontweight='bold', ha='center', color='#228B22')
    ax_right.text(5, y-0.1, 'Form eye/jaw/bone', fontsize=9, ha='center')

    # Bottom label
    ax_right.text(5, 0.5, 'External field → Internal signals → Developmental action',
                 ha='center', fontsize=9, style='italic', color='#666',
                 bbox=dict(boxstyle='round', facecolor='#FFFACD', alpha=0.8))

    # =========================================================================
    # Central comparison arrow
    # =========================================================================
    fig.text(0.5, 0.5, '⇄', fontsize=40, ha='center', va='center',
            color='#333333', fontweight='bold')
    fig.text(0.5, 0.45, 'Same Logic!', fontsize=12, ha='center', va='top',
            color='#333333', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    return fig


def main():
    """Generate all visualization figures."""
    print("Generating Field Translation Visualizations...")

    # Figure 1: Translation layers
    fig1 = create_translation_layers_figure()
    fig1.savefig('./data/field_translation_layers.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print("Saved: ./data/field_translation_layers.png")

    # Figure 2: Language analogy
    fig2 = create_language_analogy_figure()
    fig2.savefig('./data/field_language_analogy.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print("Saved: ./data/field_language_analogy.png")

    plt.show()
    print("\nDone!")


if __name__ == '__main__':
    main()