#!/usr/bin/env python3
"""
Stigmergic Computation - The Field as Shared Memory

Visualizes the stigmergic patterning mechanism as a form of computation where:
- Cells are READ/WRITE agents
- The deformable field is shared memory
- WRITE: Cells "pinch" the field based on internal state (hyperpolarization)
- READ: Cells "sense" the local field deformation (gradient)
- COMPUTE: The field physics (diffusion, superposition) performs computation

This creates a static figure suitable for presentations that captures
the computational interpretation of stigmergic dynamics.

Usage:
    python visualize_stigmergic_computation.py
    python visualize_stigmergic_computation.py --output stigmergic_computation.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as path_effects
import torch
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='stigmergic_computation.png',
                    help='Output filename (supports .png, .pdf)')
parser.add_argument('--params', type=str, default='data/StigmergicModelParameters.dat',
                    help='Parameter file for simulation')
args = parser.parse_args()

# ============================================================
# Run simulation to get realistic data
# ============================================================
print("Loading Stigmergic model and running simulation...")

from embryo import model

# Load parameters
parameters = torch.load(args.params, weights_only=False)
if 'ATPParameters' not in parameters:
    parameters['ATPParameters'] = None

latticeDims = parameters['latticeDims']
numRows, numCols = latticeDims
numCells = numRows * numCols
numSamples = parameters['simParameters']['numSamples']
initialValues = copy.deepcopy(parameters['simParameters']['initialValues'])
externalInputs = copy.deepcopy(parameters['simParameters']['externalInputs'])
clampParameters = copy.deepcopy(parameters['clampParameters'])
numSimIters = parameters['simParameters']['numSimIters']

# Run simulation
torch.manual_seed(42)
sim = model(parameters, numBasicSamples=numSamples)
sim.setExperimentalConditions((initialValues, numSamples))
sim.simulate(
    externalInputs=externalInputs,
    clampParameters=clampParameters,
    perturbation=None,
    fieldModulation=False,
    numSimIters=numSimIters,
)

print(f"Simulation complete: {numSimIters} iterations, {numRows}x{numCols} grid")

# Extract a mid-simulation frame with good pattern
t_frame = 200  # Good frame with developed pattern

vmem = sim.timeseriesVmem[t_frame, 0, :, 0].detach().numpy()

# Get field vectors
field_vec_x = sim.timeserieseVforceVector[t_frame, 0, 0, :, 0].detach().numpy()
field_vec_y = sim.timeserieseVforceVector[t_frame, 1, 0, :, 0].detach().numpy()
field_index_grid = sim.electricNetwork.extracellularIndexGrid
field_rows, field_cols = field_index_grid.shape

# Compute field magnitude
field_mag = np.sqrt(field_vec_x**2 + field_vec_y**2)

# Compute pinch strength for each cell
def compute_cell_pinch(cell_row, cell_col):
    field_grid = field_mag[field_index_grid.astype(int)]
    r_min = max(0, cell_row)
    r_max = min(field_rows, cell_row + 2)
    c_min = max(0, cell_col)
    c_max = min(field_cols, cell_col + 2)
    neighborhood = field_grid[r_min:r_max, c_min:c_max]
    return neighborhood.mean() if neighborhood.size > 0 else 0

cell_pinches = []
for cell_idx in range(numCells):
    cell_row = cell_idx // numCols
    cell_col = cell_idx % numCols
    pinch = compute_cell_pinch(cell_row, cell_col)
    cell_pinches.append(pinch)
cell_pinches = np.array(cell_pinches)
pinch_max = cell_pinches.max()

print(f"Data extracted. Creating visualization...")

# ============================================================
# Create figure
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Layout:
# Top row: 3D deformable surface (large, center)
# Bottom row: Conceptual diagrams (Write | Physics/Compute | Read)

# Main 3D panel
ax_3d = fig.add_axes([0.05, 0.38, 0.9, 0.52], projection='3d')

# Bottom conceptual panels - more space
ax_write = fig.add_axes([0.03, 0.03, 0.30, 0.30])
ax_compute = fig.add_axes([0.35, 0.03, 0.30, 0.30])
ax_read = fig.add_axes([0.67, 0.03, 0.30, 0.30])

# ============================================================
# 3D Deformable Surface with Cells
# ============================================================
mesh_res = 80
mesh_x = np.linspace(0, numCols, mesh_res)
mesh_y = np.linspace(0, numRows, mesh_res)
mesh_X, mesh_Y = np.meshgrid(mesh_x, mesh_y)

# Compute mesh deformation
mesh_Z = np.zeros_like(mesh_X)
for cell_idx in range(numCells):
    cell_row = cell_idx // numCols
    cell_col = cell_idx % numCols
    cx = cell_col + 0.5
    cy = cell_row + 0.5
    pinch = cell_pinches[cell_idx] / pinch_max
    sigma = 0.8
    dist_sq = (mesh_X - cx)**2 + (mesh_Y - cy)**2
    deformation = -pinch * 2.0 * np.exp(-dist_sq / (2 * sigma**2))
    mesh_Z += deformation

# Style the 3D axis
ax_3d.set_facecolor('white')
ax_3d.xaxis.pane.fill = False
ax_3d.yaxis.pane.fill = False
ax_3d.zaxis.pane.fill = False
ax_3d.xaxis.pane.set_edgecolor('#cccccc')
ax_3d.yaxis.pane.set_edgecolor('#cccccc')
ax_3d.zaxis.pane.set_edgecolor('#cccccc')
ax_3d.grid(True, color='#e0e0e0', alpha=0.5)

# Plot surface
surf = ax_3d.plot_surface(mesh_X, mesh_Y, mesh_Z,
                          cmap='Blues', alpha=0.6, linewidth=0,
                          antialiased=True, shade=True)

# Plot wireframe
ax_3d.plot_wireframe(mesh_X, mesh_Y, mesh_Z,
                     color='#3b82f6', alpha=0.3, linewidth=0.4,
                     rstride=3, cstride=3)

# Plot cells on the surface
vmem_min, vmem_max = vmem.min(), vmem.max()
for cell_idx in range(numCells):
    cell_row = cell_idx // numCols
    cell_col = cell_idx % numCols
    cx = cell_col + 0.5
    cy = cell_row + 0.5

    xi = int(cx / numCols * (mesh_res - 1))
    yi = int(cy / numRows * (mesh_res - 1))
    xi = min(xi, mesh_res - 1)
    yi = min(yi, mesh_res - 1)
    cz = mesh_Z[yi, xi]

    vmem_norm = (vmem[cell_idx] - vmem_min) / (vmem_max - vmem_min + 1e-10)
    color = plt.cm.coolwarm(vmem_norm)

    pinch = cell_pinches[cell_idx] / pinch_max
    size = 60 + 100 * pinch

    ax_3d.scatter([cx], [cy], [cz], c=[color], s=size,
                  edgecolors='white', linewidths=1.5, zorder=10)

# Add WRITE and READ annotations on the 3D plot
# Find a strongly writing cell (high pinch) and a sensing cell nearby
high_pinch_idx = np.argmax(cell_pinches)
high_pinch_row = high_pinch_idx // numCols
high_pinch_col = high_pinch_idx % numCols

# Find a nearby cell for READ annotation
read_idx = None
for offset in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)]:
    test_row = high_pinch_row + offset[0]
    test_col = high_pinch_col + offset[1]
    if 0 <= test_row < numRows and 0 <= test_col < numCols:
        test_idx = test_row * numCols + test_col
        if cell_pinches[test_idx] < cell_pinches[high_pinch_idx] * 0.5:
            read_idx = test_idx
            break

if read_idx is None:
    read_idx = (high_pinch_row + 1) * numCols + high_pinch_col if high_pinch_row + 1 < numRows else high_pinch_idx + 1

# Annotate WRITE cell
write_cx = high_pinch_col + 0.5
write_cy = high_pinch_row + 0.5
write_xi = int(write_cx / numCols * (mesh_res - 1))
write_yi = int(write_cy / numRows * (mesh_res - 1))
write_cz = mesh_Z[min(write_yi, mesh_res-1), min(write_xi, mesh_res-1)]

ax_3d.text(write_cx, write_cy, write_cz + 0.8, 'WRITE\n(pinch)',
           fontsize=11, fontweight='bold', color='#dc2626', ha='center', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#dc2626', alpha=0.9))

# Annotate READ cell
read_row = read_idx // numCols
read_col = read_idx % numCols
read_cx = read_col + 0.5
read_cy = read_row + 0.5
read_xi = int(read_cx / numCols * (mesh_res - 1))
read_yi = int(read_cy / numRows * (mesh_res - 1))
read_cz = mesh_Z[min(read_yi, mesh_res-1), min(read_xi, mesh_res-1)]

ax_3d.text(read_cx + 1.5, read_cy, read_cz + 0.5, 'READ\n(sense gradient)',
           fontsize=11, fontweight='bold', color='#16a34a', ha='center', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#16a34a', alpha=0.9))

ax_3d.set_xlim(0, numCols)
ax_3d.set_ylim(numRows, 0)
ax_3d.set_zlim(-3, 0.5)
ax_3d.set_xticks([])
ax_3d.set_yticks([])
ax_3d.set_zticks([])
ax_3d.view_init(elev=30, azim=-45)

ax_3d.set_title('CELLS "COMPUTE" THROUGH A DEFORMABLE SHARED MEDIUM\n',
                fontsize=16, fontweight='bold', color='#1f2937', pad=10, y=0.96)

# ============================================================
# Bottom Panel 1: WRITE Operation
# ============================================================
ax_write.set_xlim(0, 10)
ax_write.set_ylim(0, 10)
ax_write.set_aspect('equal')
ax_write.axis('off')

# Draw a cell with arrow pointing down into the field
cell_x, cell_y = 5, 7.2
field_y = 3

# Field surface (wavy line) - deeper deformation
x_wave = np.linspace(0.5, 9.5, 100)
y_base = 4.0
y_wave = y_base - 1.5 * np.exp(-((x_wave - cell_x)**2) / 2)
ax_write.fill_between(x_wave, 0, y_wave, color='#dbeafe', alpha=0.8)
ax_write.plot(x_wave, y_wave, color='#3b82f6', linewidth=3)

# Cell (circle) - larger
cell = Circle((cell_x, cell_y), 1.0, facecolor='#ef4444', edgecolor='white', linewidth=3)
ax_write.add_patch(cell)
ax_write.text(cell_x, cell_y, 'Cell', fontsize=12, fontweight='bold',
              color='white', ha='center', va='center')

# Arrow from cell to field (WRITE) - thicker
arrow = FancyArrowPatch((cell_x, cell_y - 1.1), (cell_x, y_wave[50] + 0.4),
                        arrowstyle='-|>', mutation_scale=25,
                        color='#dc2626', linewidth=4)
ax_write.add_patch(arrow)

# Labels
ax_write.text(cell_x + 1.5, 5.5, 'WRITE', fontsize=16, fontweight='bold', color='#dc2626')
ax_write.text(cell_x + 1.5, 4.6, '(pinch field)', fontsize=12, color='#6b7280')

ax_write.text(5, 9.5, 'WRITE OPERATION', fontsize=14, fontweight='bold',
              color='#1f2937', ha='center', va='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef2f2', edgecolor='#dc2626'))

ax_write.text(5, 0.8, 'Cell state (Vmem) determines\npinch strength', fontsize=11,
              color='#6b7280', ha='center', va='bottom', style='italic')

# ============================================================
# Bottom Panel 2: COMPUTE (Field Physics)
# ============================================================
ax_compute.set_xlim(0, 10)
ax_compute.set_ylim(0, 10)
ax_compute.set_aspect('equal')
ax_compute.axis('off')

# Multiple cells writing to create superposition
cell_positions = [(2.5, 7.2), (5, 7.2), (7.5, 7.2)]
pinch_strengths = [0.6, 1.0, 0.4]

# Field surface with multiple deformations (superposition)
x_wave = np.linspace(0.5, 9.5, 200)
y_base = 4.0
y_wave = np.ones_like(x_wave) * y_base
for (cx, cy), strength in zip(cell_positions, pinch_strengths):
    y_wave -= strength * 1.2 * np.exp(-((x_wave - cx)**2) / 1.5)

ax_compute.fill_between(x_wave, 0, y_wave, color='#dbeafe', alpha=0.8)
ax_compute.plot(x_wave, y_wave, color='#3b82f6', linewidth=3)

# Cells - larger
colors = ['#f97316', '#ef4444', '#f59e0b']
for (cx, cy), strength, col in zip(cell_positions, pinch_strengths, colors):
    cell = Circle((cx, cy), 0.65 + 0.35 * strength, facecolor=col, edgecolor='white', linewidth=3)
    ax_compute.add_patch(cell)

# Arrows showing diffusion/superposition - more visible
ax_compute.annotate('', xy=(4, 1.9), xytext=(2.5, 1.9),
                   arrowprops=dict(arrowstyle='-|>', color='#8b5cf6', lw=3, mutation_scale=15))
ax_compute.annotate('', xy=(6, 1.9), xytext=(7.5, 1.9),
                   arrowprops=dict(arrowstyle='-|>', color='#8b5cf6', lw=3, mutation_scale=15))

ax_compute.text(5, 2.4, 'diffusion\nsuperposition', fontsize=11, color='#8b5cf6',
                ha='center', va='center', fontweight='bold')

ax_compute.text(5, 9.5, 'COMPUTE (PHYSICS)', fontsize=14, fontweight='bold',
                color='#1f2937', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f3e8ff', edgecolor='#8b5cf6'))

ax_compute.text(5, 0.8, 'Field physics integrates multiple\nwrites via superposition', fontsize=11,
                color='#6b7280', ha='center', va='bottom', style='italic')

# ============================================================
# Bottom Panel 3: READ Operation
# ============================================================
ax_read.set_xlim(0, 10)
ax_read.set_ylim(0, 10)
ax_read.set_aspect('equal')
ax_read.axis('off')

# Field surface with gradient
x_wave = np.linspace(0.5, 9.5, 100)
y_base = 3.5
# Gradient from left (deep) to right (shallow)
y_wave = y_base - 1.2 * np.exp(-((x_wave - 2)**2) / 3)

ax_read.fill_between(x_wave, 0, y_wave, color='#dbeafe', alpha=0.7)
ax_read.plot(x_wave, y_wave, color='#3b82f6', linewidth=2)

# Reading cell on the gradient
read_x = 5
read_y_surface = y_base - 1.2 * np.exp(-((read_x - 2)**2) / 3)
cell_y = read_y_surface + 1.5

cell = Circle((read_x, cell_y), 0.8, facecolor='#22c55e', edgecolor='white', linewidth=2)
ax_read.add_patch(cell)
ax_read.text(read_x, cell_y, 'Cell', fontsize=10, fontweight='bold',
             color='white', ha='center', va='center')

# Arrow from field to cell (READ)
arrow = FancyArrowPatch((read_x, read_y_surface + 0.3), (read_x, cell_y - 0.9),
                        arrowstyle='-|>', mutation_scale=20,
                        color='#16a34a', linewidth=3)
ax_read.add_patch(arrow)

# Gradient arrow showing direction sensed
ax_read.annotate('', xy=(7, read_y_surface - 0.5), xytext=(3, read_y_surface - 0.5),
                arrowprops=dict(arrowstyle='->', color='#0891b2', lw=2))
ax_read.text(5, read_y_surface - 1, 'gradient', fontsize=11, color='#0891b2',
             ha='center', va='top', fontweight='bold')

ax_read.text(read_x + 1.5, 5, 'READ', fontsize=16, fontweight='bold', color='#16a34a')
ax_read.text(read_x + 1.5, 4.2, '(sense local)', fontsize=12, color='#6b7280')

ax_read.text(5, 9.5, 'READ OPERATION', fontsize=14, fontweight='bold',
             color='#1f2937', ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0fdf4', edgecolor='#16a34a'))

ax_read.text(5, 0.8, 'Cell senses local field gradient\nto infer global pattern', fontsize=11,
             color='#6b7280', ha='center', va='bottom', style='italic')

# ============================================================
# Main title and equation
# ============================================================
fig.suptitle('STIGMERGIC COMPUTATION: The Field as Shared Memory',
             fontsize=22, fontweight='bold', color='#1f2937', y=0.91)

# Computational equation/summary
eq_text = ('READ(cell) = gradient( WRITE(all cells) )\n Pattern emerges from distributed read-write cycles')
fig.text(0.3, 0.4, eq_text, fontsize=13, ha='center', va='top',
         color='#374151', family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f3f4f6', edgecolor='#9ca3af'))

# ============================================================
# Save figure
# ============================================================
# plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig(args.output, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {args.output}")

# Also save PDF if PNG was saved
if args.output.endswith('.png'):
    pdf_output = args.output.replace('.png', '.pdf')
    plt.savefig(pdf_output, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_output}")

plt.show()

# ============================================================
# Print conceptual summary
# ============================================================
print("\n" + "="*70)
print("STIGMERGIC COMPUTATION: READ-WRITE INTERPRETATION")
print("="*70)
print("""
The electric field serves as a SHARED MEMORY that all cells can access:

WRITE (Pinch):
  - Each cell "writes" to the field by modulating local electric potential
  - Hyperpolarized cells create stronger fields (deeper pinches)
  - The write operation is LOCAL but has GLOBAL effects through diffusion

COMPUTE (Physics):
  - The field integrates multiple writes through superposition
  - Diffusion spreads information spatially
  - No central processor - computation is distributed in the medium
  - This is analogous to how associative memory or attention works

READ (Sense):
  - Cells sense the LOCAL gradient of the field
  - This gradient encodes information about ALL other cells' writes
  - Cells infer global state from local measurement

EMERGENCE:
  - Pattern formation = iterative read-write cycles
  - Each cell reads the collective pattern, updates its state, writes back
  - The stable pattern is a fixed point of this distributed computation

This is "computation through physics" - the medium itself processes information.
""")
print("="*70)
