import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Create a comprehensive visualization of the electricmorphogenesis codebase
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors for different categories
colors = {
    'core': '#3498db',        # Blue - Core model files
    'simulation': '#e74c3c',  # Red - Simulation files
    'analysis': '#2ecc71',    # Green - Analysis files
    'utility': '#f39c12',     # Orange - Utility files
    'neural': '#9b59b6',      # Purple - Neural crest files
    'data': '#95a5a6',        # Gray - Data files
    'scripts': '#1abc9c'      # Teal - Shell scripts
}

# Define file categories and their positions
file_groups = {
    'Core Models': {
        'files': ['embryo.py', 'cellularFieldNetwork.py', 'geneRegulatoryNetwork.py', 'embryoNetwork.py'],
        'position': (2, 8),
        'color': colors['core'],
        'description': 'Main model implementations'
    },
    'Simulation': {
        'files': ['simulateTrainedModel.py', 'simulateCellularFieldNetwork.py', 'simulateEmbryoNetwork.py', 'simulateModel.py', 'simulateSingleCell.py'],
        'position': (7, 8),
        'color': colors['simulation'],
        'description': 'Simulation execution files'
    },
    'Neural Crest Models': {
        'files': ['NeuralCrestGRNModel1.py', 'NeuralCrestGRNModel2HillOptim.py', 'NeuralCrestGRNModel2SigmoidOptim.py'],
        'position': (2, 6),
        'color': colors['neural'],
        'description': 'Gene regulatory network models'
    },
    'Analysis & Learning': {
        'files': ['learnCellularFieldNetwork.py', 'analyzeCellularFieldNetwork.py', 'analyzeCellularFieldNetworkParameterSweep.py', 'analyzeCellularFieldNetworkScreeningParameterSweep.py'],
        'position': (7, 6),
        'color': colors['analysis'],
        'description': 'Model training and analysis'
    },
    'Computation & Processing': {
        'files': ['computeCellularFieldNetworkParameterSweep.py', 'computeCellularFieldNetworkScreeningParameterSweep.py', 'computeCellularFieldNetworkEntropyRate.py'],
        'position': (2, 4),
        'color': colors['analysis'],
        'description': 'Parameter sweeps and computations'
    },
    'Evaluation & Metrics': {
        'files': ['evaluateEmbryoNetworkSims.py', 'analyzeSensitivityTSEComplexity.py', 'analyzeSensitivityDistance.py'],
        'position': (7, 4),
        'color': colors['analysis'],
        'description': 'Model evaluation and metrics'
    },
    'Utilities & Support': {
        'files': ['utilities.py', 'visualize.py', 'plotAnalysisData.py', 'combineEmbryoNetworkSimFiles.py', 'fixLoss.py', 'correctParameterSweepFile.py'],
        'position': (4.5, 2),
        'color': colors['utility'],
        'description': 'Helper functions and visualization'
    }
}

# Draw file groups as boxes with file lists
def draw_file_group(ax, group_name, group_info, box_width=2.2, box_height=1.2):
    x, y = group_info['position']

    # Create rounded rectangle
    rect = FancyBboxPatch((x-box_width/2, y-box_height/2), box_width, box_height,
                         boxstyle="round,pad=0.1",
                         facecolor=group_info['color'],
                         edgecolor='black',
                         alpha=0.7,
                         linewidth=2)
    ax.add_patch(rect)

    # Add group title
    ax.text(x, y + box_height/2 - 0.15, group_name,
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Add file count
    file_count = len(group_info['files'])
    ax.text(x, y + 0.2, f"{file_count} files",
            ha='center', va='center', fontsize=9, color='white', style='italic')

    # Add description
    ax.text(x, y - 0.1, group_info['description'],
            ha='center', va='center', fontsize=8, color='white', wrap=True)

    return x, y

# Draw all file groups
group_positions = {}
for group_name, group_info in file_groups.items():
    x, y = draw_file_group(ax, group_name, group_info)
    group_positions[group_name] = (x, y)

# Draw arrows showing relationships between components
def draw_arrow(ax, start_pos, end_pos, label="", color='black', style='->', alpha=0.6):
    ax.annotate('', xy=end_pos, xytext=start_pos,
                arrowprops=dict(arrowstyle=style, color=color, alpha=alpha, lw=2))
    if label:
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Define relationships
relationships = [
    ('Core Models', 'Simulation', 'uses', '#2c3e50'),
    ('Core Models', 'Neural Crest Models', 'implements', '#8e44ad'),
    ('Simulation', 'Analysis & Learning', 'generates data', '#27ae60'),
    ('Analysis & Learning', 'Evaluation & Metrics', 'feeds into', '#e67e22'),
    ('Computation & Processing', 'Analysis & Learning', 'supports', '#34495e'),
    ('Utilities & Support', 'Simulation', 'supports', '#f39c12'),
    ('Utilities & Support', 'Analysis & Learning', 'supports', '#f39c12'),
]

# Draw relationship arrows
for source, target, label, color in relationships:
    if source in group_positions and target in group_positions:
        start_pos = group_positions[source]
        end_pos = group_positions[target]
        draw_arrow(ax, start_pos, end_pos, label, color)

# Add title and subtitle
ax.text(5, 9.5, 'Electric Morphogenesis Codebase Architecture',
        ha='center', va='center', fontsize=16, fontweight='bold')
ax.text(5, 9.2, 'Bioelectric Field Model for Embryonic Development',
        ha='center', va='center', fontsize=12, style='italic', color='gray')

# Add legend for file types
legend_x = 0.5
legend_y = 1
legend_items = [
    ('Core Models', colors['core']),
    ('Simulation', colors['simulation']),
    ('Analysis', colors['analysis']),
    ('Neural Crest', colors['neural']),
    ('Utilities', colors['utility'])
]

ax.text(legend_x, legend_y + 0.5, 'Component Types:', fontsize=10, fontweight='bold')
for i, (label, color) in enumerate(legend_items):
    y_pos = legend_y - i * 0.2
    rect = patches.Rectangle((legend_x, y_pos-0.05), 0.15, 0.1,
                           facecolor=color, alpha=0.7, edgecolor='black')
    ax.add_patch(rect)
    ax.text(legend_x + 0.2, y_pos, label, fontsize=9, va='center')

# Add data directory information
data_box = FancyBboxPatch((8.5, 0.5), 1.4, 1, boxstyle="round,pad=0.1",
                         facecolor=colors['data'], alpha=0.7, edgecolor='black', linewidth=2)
ax.add_patch(data_box)
ax.text(9.2, 1.2, 'Data Directory', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
ax.text(9.2, 0.9, '~625 files', ha='center', va='center', fontsize=9, color='white')
ax.text(9.2, 0.7, 'Model parameters', ha='center', va='center', fontsize=8, color='white')
ax.text(9.2, 0.5, 'Simulation results', ha='center', va='center', fontsize=8, color='white')

# Add project description
description_text = """This codebase implements a bioelectric model of embryonic morphogenesis.
Key Features:
• Cellular field networks modeling bioelectric signaling
• Gene regulatory networks (GRN) for neural crest development
• Parameter optimization and sensitivity analysis
• Multi-scale simulations from single cells to tissues
• Stigmergic and Mosaic model variants"""

ax.text(5, 0.8, description_text, ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('/Users/santoshmanicka/PycharmProjects/electricmorphogenesis/codebase_structure_graph.png',
           dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Codebase visualization saved as 'codebase_structure_graph.png'")
print("\nCodebase Summary:")
print("=" * 50)
print(f"Total Python files: ~40")
print(f"Core model files: {len(file_groups['Core Models']['files'])}")
print(f"Simulation files: {len(file_groups['Simulation']['files'])}")
print(f"Analysis files: {len(file_groups['Analysis & Learning']['files']) + len(file_groups['Evaluation & Metrics']['files'])}")
print(f"Data files: ~625 (parameter sets, results)")
print(f"Shell scripts: ~10 (execution scripts)")
print("\nMain Dependencies: PyTorch, NumPy, SciPy, Matplotlib, Pandas")