#!/usr/bin/env python3
"""
CaMKII Bistability - Static Multi-Panel Figure for Presentations

Creates a publication-quality figure showing the "drag and drop" mechanism
in 4 key phases, ideal for biology presentations.

Usage:
    python visualize_camkii_static.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# ============================================================
# Parameters
# ============================================================
K_half = 0.5

# ============================================================
# Helper functions
# ============================================================
def energy_landscape(x, K=K_half):
    """Bistable energy landscape"""
    return 2 * (x - 0) ** 2 * (x - 1) ** 2

def self_activation(x, K=K_half):
    """Competitive self-activation: (x² - K²) / (x² + K²)"""
    return (x**2 - K**2) / (x**2 + K**2 + 1e-10)

# ============================================================
# Create figure
# ============================================================
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0f0f23')

# Create grid for subplots
gs = gridspec.GridSpec(2, 4, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

# Top row: 4 phases of the mechanism
# Bottom row: Energy landscape + Self-activation function + Summary

# ============================================================
# Define the 4 phases
# ============================================================
phases = [
    {
        'title': 'Phase 1: Initial State',
        'subtitle': 'All cells start in OFF state',
        'feature_pos': [0.15, 0.12, 0.18, 0.14, 0.16],
        'background_pos': [0.13, 0.11, 0.15, 0.12, 0.14],
        'ca_feature': 0.1,
        'ca_background': 0.1,
        'arrow': False
    },
    {
        'title': 'Phase 2: Ca²⁺ Rising',
        'subtitle': 'Vmem activates Ca²⁺ channels',
        'feature_pos': [0.35, 0.32, 0.38, 0.34, 0.36],
        'background_pos': [0.18, 0.16, 0.20, 0.17, 0.19],
        'ca_feature': 0.8,
        'ca_background': 0.2,
        'arrow': True
    },
    {
        'title': 'Phase 3: Ca²⁺ "Drops" Cells',
        'subtitle': 'Feature cells cross K_half',
        'feature_pos': [0.65, 0.62, 0.68, 0.64, 0.66],
        'background_pos': [0.25, 0.22, 0.28, 0.24, 0.26],
        'ca_feature': 0.9,
        'ca_background': 0.2,
        'arrow': True
    },
    {
        'title': 'Phase 4: Bistable Memory',
        'subtitle': 'Pattern persists without Ca²⁺',
        'feature_pos': [0.88, 0.85, 0.90, 0.87, 0.89],
        'background_pos': [0.12, 0.10, 0.14, 0.11, 0.13],
        'ca_feature': 0.15,
        'ca_background': 0.15,
        'arrow': False
    }
]

# Plot each phase
for i, phase in enumerate(phases):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor('#1a1a2e')

    # Draw energy landscape
    x_range = np.linspace(-0.05, 1.05, 200)
    y_energy = energy_landscape(x_range)
    y_scaled = y_energy / y_energy.max() * 0.4
    ax.fill_between(x_range, 0, y_scaled, color='#16213e', alpha=0.9)
    ax.plot(x_range, y_scaled, color='#4a69bd', linewidth=2)

    # K_half line
    ax.axvline(K_half, color='#e74c3c', linestyle='--', linewidth=2.5, alpha=0.9)
    if i == 0:
        ax.text(K_half + 0.03, 0.75, 'K_half', color='#e74c3c',
                fontsize=11, fontweight='bold', rotation=90, va='center')

    # Draw cells
    for pos in phase['feature_pos']:
        y = energy_landscape(pos) / energy_landscape(x_range).max() * 0.4
        circle = Circle((pos, y + 0.05), 0.035, color='#2ecc71',
                        ec='white', linewidth=2, zorder=10)
        ax.add_patch(circle)

    for pos in phase['background_pos']:
        y = energy_landscape(pos) / energy_landscape(x_range).max() * 0.4
        circle = Circle((pos, y + 0.05), 0.035, color='#3498db',
                        ec='white', linewidth=2, zorder=10)
        ax.add_patch(circle)

    # Ca²⁺ indicator bar
    ca_color = plt.cm.YlOrRd(phase['ca_feature'])
    rect = Rectangle((0.02, 0.85), 0.15, 0.08, color=ca_color,
                     ec='white', linewidth=1)
    ax.add_patch(rect)
    ax.text(0.17, 0.89, f"Ca²⁺: {phase['ca_feature']:.1f}",
            color='white', fontsize=9, va='center')

    # Arrow showing Ca push
    if phase['arrow'] and phase['ca_feature'] > 0.5:
        mean_feat = np.mean(phase['feature_pos'])
        ax.annotate('', xy=(min(mean_feat + 0.12, 0.95), 0.5),
                    xytext=(mean_feat - 0.02, 0.5),
                    arrowprops=dict(arrowstyle='->', color='#f39c12', lw=3))

    # Labels
    ax.text(0.1, -0.08, 'OFF', color='#3498db', fontsize=10,
            fontweight='bold', ha='center')
    ax.text(0.9, -0.08, 'ON', color='#2ecc71', fontsize=10,
            fontweight='bold', ha='center')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.0)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', 'K_half', '1'], color='white')
    ax.set_yticks([])

    ax.set_title(f'{phase["title"]}\n{phase["subtitle"]}',
                 fontsize=12, fontweight='bold', color='white', pad=10)

    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(0.5)
    ax.tick_params(colors='white')

    # Phase number circle
    circle_bg = Circle((0.93, 0.93), 0.06, color='#e74c3c',
                       transform=ax.transAxes, zorder=20)
    ax.add_patch(circle_bg)
    ax.text(0.93, 0.93, str(i+1), transform=ax.transAxes,
            color='white', fontsize=12, fontweight='bold',
            ha='center', va='center', zorder=21)

# ============================================================
# Bottom left: Self-activation function
# ============================================================
ax_sa = fig.add_subplot(gs[1, 0:2])
ax_sa.set_facecolor('#1a1a2e')

x = np.linspace(0, 1, 100)
sa = self_activation(x)

ax_sa.axhline(0, color='white', linewidth=0.5, alpha=0.5)
ax_sa.axvline(K_half, color='#e74c3c', linestyle='--', linewidth=2, label='K_half')

# Fill regions
ax_sa.fill_between(x[x < K_half], sa[x < K_half], 0, color='#3498db', alpha=0.3)
ax_sa.fill_between(x[x >= K_half], sa[x >= K_half], 0, color='#2ecc71', alpha=0.3)

ax_sa.plot(x, sa, color='white', linewidth=3)

# Annotations
ax_sa.annotate('Self-INHIBITION\n(stays OFF)', xy=(0.2, -0.4),
               fontsize=11, color='#3498db', ha='center', fontweight='bold')
ax_sa.annotate('Self-ACTIVATION\n(stays ON)', xy=(0.8, 0.4),
               fontsize=11, color='#2ecc71', ha='center', fontweight='bold')

ax_sa.set_xlim(0, 1)
ax_sa.set_ylim(-1.1, 1.1)
ax_sa.set_xlabel('CaMKII Activity', fontsize=12, color='white')
ax_sa.set_ylabel('Self-Activation\n(CaMKII² - K²) / (CaMKII² + K²)', fontsize=11, color='white')
ax_sa.set_title('Competitive Dynamics: The Bistability Engine',
                fontsize=13, fontweight='bold', color='white')

for spine in ax_sa.spines.values():
    spine.set_color('white')
ax_sa.tick_params(colors='white')
ax_sa.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='white',
             labelcolor='white', fontsize=10)

# ============================================================
# Bottom right: Summary diagram
# ============================================================
ax_sum = fig.add_subplot(gs[1, 2:4])
ax_sum.set_facecolor('#1a1a2e')
ax_sum.axis('off')

summary_text = """
┌─────────────────────────────────────────────────────────────┐
│                   THE "DRAG AND DROP" MECHANISM             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Bioelectric        Molecular          Bistable            │
│     Signal          Transduction         Memory             │
│                                                             │
│   ┌───────┐        ┌───────────┐       ┌─────────┐         │
│   │ Vmem  │ ──────▶│   Ca²⁺    │──────▶│ CaMKII  │         │
│   │pattern│        │  channels │       │ activity│         │
│   └───────┘        └───────────┘       └─────────┘         │
│                                              │              │
│                                              ▼              │
│                                        ┌──────────┐        │
│                                        │Self-     │        │
│                                        │Activation│        │
│                    ◀───────────────────│(+/-)     │        │
│                    Feedback loop       └──────────┘        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  KEY INSIGHT:                                               │
│  • Ca²⁺ "drags" CaMKII across K_half threshold              │
│  • Once past K_half: self-activation takes over (+)         │
│  • Below K_half: self-inhibition keeps it OFF (-)           │
│  • Pattern persists even after Ca²⁺ decays!                 │
│                                                             │
│  BIOLOGICAL RELEVANCE:                                      │
│  • Converts transient bioelectric signals → stable memory   │
│  • Enables developmental pattern maintenance                │
│  • Similar to synaptic LTP/LTD mechanisms                   │
└─────────────────────────────────────────────────────────────┘
"""

ax_sum.text(0.5, 0.5, summary_text, transform=ax_sum.transAxes,
            fontsize=10, family='monospace', color='white',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e',
                     edgecolor='#4a69bd', linewidth=2))

# ============================================================
# Legend at bottom
# ============================================================
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=12, label='Feature cells (high Vmem → high Ca²⁺)', linestyle='None'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=12, label='Background cells (low Vmem → low Ca²⁺)', linestyle='None'),
    plt.Line2D([0], [0], color='#e74c3c', linewidth=2, linestyle='--',
               label='K_half: Decision boundary')
]

fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=11, facecolor='#0f0f23', edgecolor='white',
           labelcolor='white', framealpha=0.9,
           bbox_to_anchor=(0.5, 0.02))

# Main title
fig.suptitle('Ca²⁺ "Drags and Drops" CaMKII into Bistable States:\nA Mechanism for Bioelectric Pattern Memory',
             fontsize=18, fontweight='bold', color='white', y=0.98)

plt.tight_layout(rect=[0, 0.06, 1, 0.94])

# Save figure
output_file = 'camkii_mechanism_static.png'
plt.savefig(output_file, dpi=200, facecolor='#0f0f23', edgecolor='none',
            bbox_inches='tight')
print(f"Saved: {output_file}")

output_pdf = 'camkii_mechanism_static.pdf'
plt.savefig(output_pdf, facecolor='#0f0f23', edgecolor='none',
            bbox_inches='tight')
print(f"Saved: {output_pdf}")

plt.show()
