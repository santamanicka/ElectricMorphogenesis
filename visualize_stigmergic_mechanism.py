#!/usr/bin/env python3
"""
Stigmergic Patterning - Essence Visualization

Core concept: The electric field is a deformable "body" that cells pinch.
- Hyperpolarized cells create stronger fields → deeper "pinches" in the mesh
- The mesh visualizes how cells write their state onto a shared medium
- Other cells sense these deformations (read from the body)

Uses actual simulation data from the Stigmergic model.

Usage:
    python visualize_stigmergic_mechanism.py
    python visualize_stigmergic_mechanism.py --save_gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import torch
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--save_gif', action='store_true', help='Save as animated GIF')
parser.add_argument('--save_mp4', action='store_true', help='Save as MP4 video')
parser.add_argument('--output', type=str, default='stigmergic_mechanism', help='Output filename')
parser.add_argument('--params', type=str, default='data/StigmergicModelParameters.dat', help='Parameter file')
parser.add_argument('--mode', type=str, default='both', choices=['2d', '3d', 'both'],
                    help='Visualization mode: 2d, 3d, or both')
args = parser.parse_args()

# ============================================================
# Run actual Stigmergic simulation
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

startTimeIndex = 102  # ignore the initial stimulation

# Extract time series data
vmem_series = sim.timeseriesVmem[startTimeIndex:, 0, :, 0].detach().numpy()  # (time, cells)
field_series = sim.timeserieseV[startTimeIndex:, 0, :, 0].detach().numpy()   # (time, field_points)

# Field vectors for computing magnitude
field_vec_x = sim.timeserieseVforceVector[startTimeIndex:, 0, 0, :, 0].detach().numpy()
field_vec_y = sim.timeserieseVforceVector[startTimeIndex:, 1, 0, :, 0].detach().numpy()

# Filtered timeseries length
timeSeriesLength = vmem_series.shape[0]

# Get field grid shape
field_index_grid = sim.electricNetwork.extracellularIndexGrid
field_rows, field_cols = field_index_grid.shape

# Compute field magnitude at each point
field_mag_series = np.sqrt(field_vec_x**2 + field_vec_y**2)  # (time, field_points)

print(f"Field grid: {field_rows}x{field_cols}")

# ============================================================
# Compute pinch strength for each cell (average field magnitude nearby)
# ============================================================
def compute_cell_pinch(t, cell_row, cell_col):
    """Compute the average field magnitude around a cell."""
    # Field grid coordinates around this cell
    # Each cell is at position (col+0.5, row+0.5) in field coordinates
    # Sample field points in a small neighborhood

    field_flat = field_mag_series[t]
    field_grid = field_flat[field_index_grid.astype(int)]

    # Get the field points near this cell (cell center is between field grid points)
    r_min = max(0, cell_row)
    r_max = min(field_rows, cell_row + 2)
    c_min = max(0, cell_col)
    c_max = min(field_cols, cell_col + 2)

    neighborhood = field_grid[r_min:r_max, c_min:c_max]
    return neighborhood.mean() if neighborhood.size > 0 else 0

# ============================================================
# Sample frames for animation
# ============================================================
num_frames = 500
frame_indices = np.linspace(0, timeSeriesLength - 1, num_frames, dtype=int)

# ============================================================
# Create figure based on mode
# ============================================================
if args.mode == 'both':
    fig = plt.figure(figsize=(16, 8))
    ax_2d = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122, projection='3d')
    axes = [ax_2d, ax_3d]
elif args.mode == '2d':
    fig = plt.figure(figsize=(6,6))
    ax_2d = fig.add_subplot(111)
    ax_3d = None
    axes = [ax_2d]
else:  # 3d
    fig = plt.figure(figsize=(12, 10))
    ax_3d = fig.add_subplot(111, projection='3d')
    ax_2d = None
    axes = [ax_3d]

fig.patch.set_facecolor('#0d1117')

# Mesh resolution (higher than cell grid for smooth deformation)
mesh_res = 100  # orig: 40
mesh_x = np.linspace(0, numCols, mesh_res)
mesh_y = np.linspace(0, numRows, mesh_res)
mesh_X, mesh_Y = np.meshgrid(mesh_x, mesh_y)

# Pre-compute global pinch range for consistent scaling
all_pinches = []
for t in frame_indices:
    pinches = []
    for cell_idx in range(numCells):
        cell_row = cell_idx // numCols
        cell_col = cell_idx % numCols
        pinch = compute_cell_pinch(t, cell_row, cell_col)
        pinches.append(pinch)
    all_pinches.extend(pinches)
pinch_max = np.percentile(all_pinches, 98) if all_pinches else 1.0
pinch_max = max(pinch_max, 1e-10)

print(f"Pinch range: 0 to {pinch_max:.6f}")

# ============================================================
# Animation
# ============================================================
def compute_2d_displacement(mesh_X, mesh_Y, cell_pinches, pinch_max):
    """
    Compute 2D displacement field - points are pulled towards pinch centers.
    Stronger pinch = more stretching towards that cell.
    """
    disp_X = np.zeros_like(mesh_X)
    disp_Y = np.zeros_like(mesh_Y)

    for cell_idx in range(numCells):
        cell_row = cell_idx // numCols
        cell_col = cell_idx % numCols

        cx = cell_col + 0.5
        cy = cell_row + 0.5

        pinch = cell_pinches[cell_idx] / pinch_max

        # Vector from each mesh point to cell center
        dx = cx - mesh_X
        dy = cy - mesh_Y
        dist = np.sqrt(dx**2 + dy**2) + 0.1  # avoid div by zero

        # Displacement magnitude: stronger near cell, scaled by pinch
        # Points get pulled towards the cell (stretch effect)
        # sigma = 1.2
        # magnitude = pinch * 0.4 * np.exp(-dist**2 / (2 * sigma**2))
        sigma = 1.3
        magnitude = pinch * 0.4 * np.exp(-dist ** 2 / (2 * sigma ** 2))

        # Normalize direction and apply magnitude
        disp_X += magnitude * dx / dist
        disp_Y += magnitude * dy / dist

    return disp_X, disp_Y


def animate(frame_num):
    t = frame_indices[frame_num]

    # Get pinch strength for each cell
    cell_pinches = []
    for cell_idx in range(numCells):
        cell_row = cell_idx // numCols
        cell_col = cell_idx % numCols
        pinch = compute_cell_pinch(t, cell_row, cell_col)
        cell_pinches.append(pinch)

    # Compute mesh deformation (Z for 3D, displacement for 2D)
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

    # 2D displacement
    disp_X, disp_Y = compute_2d_displacement(mesh_X, mesh_Y, cell_pinches, pinch_max)
    mesh_X_deformed = mesh_X + disp_X
    mesh_Y_deformed = mesh_Y + disp_Y

    # ==================== 2D Panel ====================
    if ax_2d is not None:
        ax_2d.clear()
        ax_2d.set_facecolor('#0d1117')

        # Draw deformed grid lines
        # Horizontal lines
        for i in range(0, mesh_res, 2):
            ax_2d.plot(mesh_X_deformed[i, :], mesh_Y_deformed[i, :],
                      color='#60a5fa', alpha=0.5, linewidth=0.8)
        # Vertical lines
        for j in range(0, mesh_res, 2):
            ax_2d.plot(mesh_X_deformed[:, j], mesh_Y_deformed[:, j],
                      color='#60a5fa', alpha=0.5, linewidth=0.8)

        # Plot cells
        for cell_idx in range(numCells):
            cell_row = cell_idx // numCols
            cell_col = cell_idx % numCols
            cx = cell_col + 0.5
            cy = cell_row + 0.5

            # Cell color based on Vmem
            vmem = vmem_series[t, cell_idx]
            vmem_norm = (vmem - vmem_series.min()) / (vmem_series.max() - vmem_series.min() + 1e-10)
            color = plt.cm.Greys(vmem_norm)

            # Size based on pinch strength
            pinch = cell_pinches[cell_idx] / pinch_max
            size = 500 + 250 * pinch  # orig: 150 + 250 * pinch

            ax_2d.scatter([cx], [cy], c=[color], s=size,
                          linewidths=1.5, zorder=10, alpha=0.5)  # edgecolors="white"

        ax_2d.set_xlim(-0.5, numCols + 0.5)
        ax_2d.set_ylim(numRows + 0.5, -0.5)  # Inverted Y-axis
        ax_2d.set_aspect('equal')
        ax_2d.set_xticks([])
        ax_2d.set_yticks([])
        for spine in ax_2d.spines.values():
            spine.set_color('#60a5fa')
            spine.set_linewidth(2)

        ax_2d.set_title('The field as a deformable body',
                       color='white', fontsize=12, fontweight='bold', pad=10)

    # ==================== 3D Panel ====================
    if ax_3d is not None:
        ax_3d.clear()
        ax_3d.set_facecolor('#0d1117')
        ax_3d.xaxis.pane.fill = False
        ax_3d.yaxis.pane.fill = False
        ax_3d.zaxis.pane.fill = False
        ax_3d.xaxis.pane.set_edgecolor('#30363d')
        ax_3d.yaxis.pane.set_edgecolor('#30363d')
        ax_3d.zaxis.pane.set_edgecolor('#30363d')
        ax_3d.grid(True, color='#30363d', alpha=0.3)

        # Plot wireframe
        ax_3d.plot_wireframe(mesh_X, mesh_Y, mesh_Z,
                            color='#60a5fa', alpha=0.4, linewidth=0.5,
                            rstride=2, cstride=2)

        # Plot surface
        ax_3d.plot_surface(mesh_X, mesh_Y, mesh_Z,
                          cmap='Blues', alpha=0.3, linewidth=0,
                          antialiased=True)

        # Plot cells
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

            vmem = vmem_series[t, cell_idx]
            vmem_norm = (vmem - vmem_series.min()) / (vmem_series.max() - vmem_series.min() + 1e-10)
            color = plt.cm.coolwarm(vmem_norm)

            pinch = cell_pinches[cell_idx] / pinch_max
            size = 80 + 120 * pinch

            ax_3d.scatter([cx], [cy], [cz], c=[color], s=size,
                         edgecolors='white', linewidths=1, zorder=10)

        ax_3d.set_xlim(0, numCols)
        ax_3d.set_ylim(numRows, 0)  # Inverted Y-axis
        ax_3d.set_zlim(-3, 0.5)
        ax_3d.set_xticks([])
        ax_3d.set_yticks([])
        ax_3d.set_zticks([])
        ax_3d.view_init(elev=35, azim=-60 + frame_num * 0.3)

        ax_3d.set_title('3D: Cells pinch the shared medium',
                       color='white', fontsize=12, fontweight='bold', pad=10)

    # Main title
    fig.suptitle(f'THE FIELD AS A DEFORMABLE "BODY" • t = {t}/{numSimIters}',
                fontsize=14, fontweight='bold', color='white', y=0.95)

    # Legend (position depends on mode)
    if args.mode == 'both':
        legend_x = 0.5
    else:
        legend_x = 0.85

    # Clear old text by redrawing
    for txt in fig.texts:
        txt.remove()

    fig.text(0.51, 0.07,
             'Stronger field → more deformation • Cells sense each other through the shared body',
             ha='center', va='bottom', fontsize=8, color='#8b949e', style='italic')

    fig.suptitle(f'THE FIELD AS A DEFORMABLE "BODY" • t = {t}/{numSimIters}',
                fontsize=14, fontweight='bold', color='white', y=0.95)

    return []

# ============================================================
# Run
# ============================================================
print("Creating animation...")

if args.save_gif or args.save_mp4:
    anim = animation.FuncAnimation(fig, animate, frames=num_frames,
                                    interval=80, blit=False)

    if args.save_gif:
        output_file = f'{args.output}.gif'
        print(f"Saving as {output_file}...")
        anim.save(output_file, writer='pillow', fps=15, dpi=100)
        print(f"Saved: {output_file}")

    if args.save_mp4:
        output_file = f'{args.output}.mp4'
        print(f"Saving as {output_file}...")
        # Use ffmpeg writer for MP4
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, bitrate=3000)
        anim.save(output_file, writer=writer, dpi=120)
        print(f"Saved: {output_file}")
else:
    anim = animation.FuncAnimation(fig, animate, frames=num_frames,
                                    interval=80, blit=False)
    print("Showing animation...")
    plt.show()

print("\n" + "="*60)
print("THE DEFORMABLE BODY METAPHOR")
print("="*60)
print("The electric field is a shared 'fabric' that all cells touch.")
print("Each cell pinches this fabric based on its bioelectric state.")
print("Other cells feel these pinches - that's how they 'read' the field.")
print("The pattern emerges because the fabric is shared.")
print("="*60)
