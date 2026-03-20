#!/usr/bin/env python3
"""
Multi-Embryo Stress Rescue Network Visualization.

Creates a two-level field visualization showing:
- Top: Collective stress field as a 3D deformable surface over the embryo grid
  (YlOrRd warm colormap -- each perturbed embryo creates a "well" in the field)
- Bottom: Individual embryo bioelectric fields as 3D deformable surfaces
  (Blues colormap -- each cell "pinches" the field based on its activity)

The analogy: within each embryo, cells read/write a shared electric field to
coordinate pattern formation. Between embryos, stress signals create a shared
extracellular field that enables collective rescue.

Usage:
    python visualize_multiembryo_network.py
    python visualize_multiembryo_network.py --output data/multiembryo_network.pdf
    python visualize_multiembryo_network.py --numBioSteps 500
"""

import argparse
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D

from embryo import model
from stressBistableSwitch import StressBistableSwitch
import utilities

parser = argparse.ArgumentParser(
    description='Visualize multi-embryo stress rescue network'
)
parser.add_argument('--output', type=str, default='data/multiembryo_network.png',
                    help='Output filename (supports .png, .pdf)')
parser.add_argument('--numBioSteps', type=int, default=500,
                    help='Bioelectric simulation steps (default: 500)')
parser.add_argument('--stressParamsFile', type=str, default=None,
                    help='Path to learned stress parameters file (.dat)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
args = parser.parse_args()


# ============================================================
# Utility functions (adapted from runGroupRescue.py)
# ============================================================
def apply_sigmoid_constraint(raw_param, min_val, max_val):
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)


def load_stress_params(params_file):
    data = torch.load(params_file, weights_only=False)
    stress_params = {}
    param_bounds = data.get('parameter_bounds', {})
    raw_params = data.get('parameters', {})
    for param_name, raw_value in raw_params.items():
        min_key = f'{param_name}_min'
        max_key = f'{param_name}_max'
        if min_key in param_bounds and max_key in param_bounds:
            constrained = apply_sigmoid_constraint(
                raw_value, param_bounds[min_key], param_bounds[max_key])
            stress_params[param_name] = float(constrained.item())
        else:
            stress_params[param_name] = float(raw_value.item())
    fixed_ca_params = data.get('fixed_ca_params', None)
    if fixed_ca_params is None:
        fixed_ca_params = get_default_ca_params()
    return stress_params, fixed_ca_params


def get_default_ca_params():
    return {
        'tau_ca': 2.5964, 'g_ca': 5.3437, 'V_half_ca': -0.0753,
        'k_ca': 0.0021, 'k_decay_ca': 4.3346,
    }


def get_default_stress_params():
    # Ca threshold tuned to differentiate healthy (Ca~8.2) vs perturbed (Ca~9.5)
    stress_params = {
        'tau_S': 50.0, 'k_on_S': 3.0, 'k_off_S': 0.02, 'K_S': 0.4,
        'Ca_stress_threshold': 8.8, 'sigma_ca': 0.5, 'gain_S': 2.0,
        'or_threshold_S': 0.6, 'D_S': 0.15, 'gamma': 0.08, 'K_decay': 0.3,
    }
    return stress_params, get_default_ca_params()


def load_model_parameters(grn_damping=1.0):
    path = './data/bestModelParameters_fieldVector_Ligand_GRN_253.dat'
    params = torch.load(path, weights_only=False)
    if "ATPParameters" not in params:
        params["ATPParameters"] = None
    if grn_damping != 1.0 and 'GRNParameters' in params and params['GRNParameters'] is not None:
        grn_params = params['GRNParameters']
        if 'GRNWeights' in grn_params and grn_params['GRNWeights'] is not None:
            grn_params['GRNWeights'] = grn_params['GRNWeights'] * grn_damping
        if 'InterGRNWeights' in grn_params and grn_params['InterGRNWeights'] is not None:
            grn_params['InterGRNWeights'] = grn_params['InterGRNWeights'] * grn_damping
        if 'GRNtoLigandWeights' in grn_params and grn_params['GRNtoLigandWeights'] is not None:
            grn_params['GRNtoLigandWeights'] = grn_params['GRNtoLigandWeights'] * grn_damping
        if grn_damping == 0.0:
            grn_params['GRNEnabled'] = False
    return params


def build_vonneumann_adjacency(rows, cols):
    n = rows * cols
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    nidx = ni * cols + nj
                    adj[idx, nidx] = 1.0
    return adj


def compute_effective_damping(base_damping, neighbor_stress_mean, alpha):
    base_clamped = max(min(base_damping, 0.999), 0.001)
    base_logit = np.log(base_clamped / (1.0 - base_clamped))
    return 1.0 / (1.0 + np.exp(-(base_logit + alpha * neighbor_stress_mean)))


# ============================================================
# Data collection: Group simulation (simplified)
# ============================================================
def run_group_simulation(num_bio_steps, stress_params, ca_params):
    """Run a 3x3 embryo grid and return stress/Vmem data."""
    grid_rows, grid_cols = 3, 3
    num_embryos = grid_rows * grid_cols
    # Damping d: 1.0=healthy, 0.85=mild, 0.7=strong perturbation
    # Stress s=1-d: 0.0=healthy, 0.15=mild, 0.30=strong
    damping_map = np.array([
        [1.0, 0.85, 1.0],
        [0.85, 0.7, 0.85],
        [1.0, 0.85, 1.0],
    ])

    # Build embryo adjacency
    embryo_adj = build_vonneumann_adjacency(grid_rows, grid_cols)
    max_degree = 4
    D_F, gamma_F = 0.5, 0.0001
    n_substeps = 10

    # Load cell adjacency
    ref_params = load_model_parameters(grn_damping=1.0)
    cell_grid_size = ref_params['latticeDims'][0]
    num_cells = cell_grid_size * cell_grid_size
    utils = utilities.utilities()
    cell_adjacency = utils.computeLatticeAdjacencyMatrix(
        latticeDims=(cell_grid_size, cell_grid_size), periodicBoundary=False)

    print(f"Instantiating {num_embryos} embryos ({grid_rows}x{grid_cols})...")
    embryos = []
    damping_flat = damping_map.flatten()

    for idx in range(num_embryos):
        d = damping_flat[idx]
        params = load_model_parameters(grn_damping=d)
        num_samples = params["simParameters"]["numSamples"]
        initial_values = copy.deepcopy(params["simParameters"]["initialValues"])
        external_inputs = copy.deepcopy(params["simParameters"]["externalInputs"])
        clamp_params = copy.deepcopy(params["clampParameters"])

        bio_model = model(params, numBasicSamples=num_samples)
        bio_model.setExperimentalConditions((initial_values, num_samples))

        stress_switch = StressBistableSwitch(
            num_cells=num_cells,
            adjacency_matrix=cell_adjacency,
            params=ca_params,
            device='cpu', dtype=torch.float32)
        stress_switch.set_params_from_tensors(
            **{k: torch.tensor(v, dtype=torch.float32) for k, v in stress_params.items()})

        embryos.append({
            'bio_model': bio_model,
            'stress_switch': stress_switch,
            'external_inputs': external_inputs,
            'clamp_params': clamp_params,
        })

    # Run simulation
    F = np.zeros(num_embryos)
    dt_ca, dt_stress = 0.01, 0.1

    print(f"Running group simulation ({num_bio_steps} steps)...")
    for t in range(num_bio_steps):
        # Step bio
        for idx in range(num_embryos):
            e = embryos[idx]
            e['bio_model'].simulate(
                externalInputs=e['external_inputs'],
                clampParameters=e['clamp_params'],
                perturbation=None, fieldModulation=False,
                numSimIters=1, outerIter=t)

        # Step stress
        for idx in range(num_embryos):
            vmem_flat = embryos[idx]['bio_model'].electricNetwork.Vmem[0, :, 0]
            embryos[idx]['stress_switch'].compute_ca_from_vmem(
                vmem_flat.to(dtype=torch.float32), dt_ca)
            embryos[idx]['stress_switch'].step(dt_stress)

        # Diffuse stress field
        emission = np.array([embryos[i]['stress_switch'].get_embryo_stress().item()
                             for i in range(num_embryos)])
        dt_sub = 1.0 / n_substeps
        for _ in range(n_substeps):
            laplacian = embryo_adj @ F - max_degree * F
            dF_dt = D_F * laplacian - gamma_F * F + emission
            F = F + dt_sub * dF_dt
            F = np.clip(F, 0.0, None)

        if (t + 1) % 100 == 0:
            print(f"  Step {t+1}/{num_bio_steps}: mean_stress={emission.mean():.4f}, mean_F={F.mean():.4f}")

    # Stress equilibration (100 steps)
    print("  Stress equilibration (100 steps)...")
    for _ in range(100):
        for idx in range(num_embryos):
            ca_final = embryos[idx]['stress_switch'].Ca.detach().clone()
            embryos[idx]['stress_switch'].step(dt_stress, Ca=ca_final)
        emission = np.array([embryos[i]['stress_switch'].get_embryo_stress().item()
                             for i in range(num_embryos)])
        dt_sub = 1.0 / n_substeps
        for _ in range(n_substeps):
            laplacian = embryo_adj @ F - max_degree * F
            dF_dt = D_F * laplacian - gamma_F * F + emission
            F = F + dt_sub * dF_dt
            F = np.clip(F, 0.0, None)

    # Collect results
    final_stress = np.array([embryos[i]['stress_switch'].get_embryo_stress().item()
                             for i in range(num_embryos)])
    final_vmem = [embryos[i]['bio_model'].electricNetwork.Vmem[0, :, 0].detach().cpu().numpy()
                  for i in range(num_embryos)]
    final_S_per_cell = [embryos[i]['stress_switch'].S.detach().cpu().numpy()
                        for i in range(num_embryos)]
    final_Ca_per_cell = [embryos[i]['stress_switch'].Ca.detach().cpu().numpy()
                         for i in range(num_embryos)]

    return {
        'final_stress': final_stress,
        'final_field': F.copy(),
        'final_vmem': final_vmem,
        'final_S_per_cell': final_S_per_cell,
        'final_Ca_per_cell': final_Ca_per_cell,
        'damping_map': damping_map,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'cell_grid_size': cell_grid_size,
    }


# ============================================================
# Data collection: Individual embryo with field vectors
# ============================================================
def run_representative_embryos(num_bio_steps, stress_params, ca_params):
    """Run 3 standalone embryos to get Vmem field data for bottom panels."""
    dampings = [1.0, 0.85, 0.7]
    results = []

    ref_params = load_model_parameters(grn_damping=1.0)
    cell_grid_size = ref_params['latticeDims'][0]
    num_cells = cell_grid_size * cell_grid_size
    utils = utilities.utilities()
    cell_adjacency = utils.computeLatticeAdjacencyMatrix(
        latticeDims=(cell_grid_size, cell_grid_size), periodicBoundary=False)

    for d in dampings:
        print(f"  Running representative embryo (damping={d})...")
        params = load_model_parameters(grn_damping=d)
        num_samples = params["simParameters"]["numSamples"]
        initial_values = copy.deepcopy(params["simParameters"]["initialValues"])
        external_inputs = copy.deepcopy(params["simParameters"]["externalInputs"])
        clamp_params = copy.deepcopy(params["clampParameters"])

        sim = model(params, numBasicSamples=num_samples)
        sim.setExperimentalConditions((initial_values, num_samples))
        sim.simulate(
            externalInputs=external_inputs,
            clampParameters=clamp_params,
            perturbation=None, fieldModulation=False,
            numSimIters=num_bio_steps)

        # Extract data at a late frame (GRN feedback needs time to affect Vmem)
        t_frame = num_bio_steps - 1
        vmem = sim.timeseriesVmem[t_frame, 0, :, 0].detach().numpy()
        numRows, numCols = cell_grid_size, cell_grid_size

        # Compute cell pinch strengths from field vectors
        field_vec_x = sim.timeserieseVforceVector[t_frame, 0, 0, :, 0].detach().numpy()
        field_vec_y = sim.timeserieseVforceVector[t_frame, 1, 0, :, 0].detach().numpy()
        field_index_grid = sim.electricNetwork.extracellularIndexGrid
        field_rows, field_cols = field_index_grid.shape
        field_mag = np.sqrt(field_vec_x**2 + field_vec_y**2)

        cell_pinches = np.zeros(num_cells)
        for cell_idx in range(num_cells):
            cell_row = cell_idx // numCols
            cell_col = cell_idx % numCols
            field_grid = field_mag[field_index_grid.astype(int)]
            r_min = max(0, cell_row)
            r_max = min(field_rows, cell_row + 2)
            c_min = max(0, cell_col)
            c_max = min(field_cols, cell_col + 2)
            neighborhood = field_grid[r_min:r_max, c_min:c_max]
            cell_pinches[cell_idx] = neighborhood.mean() if neighborhood.size > 0 else 0

        pinch_max = cell_pinches.max() if cell_pinches.max() > 0 else 1.0

        # Run stress switch to get per-cell S
        stress_switch = StressBistableSwitch(
            num_cells=num_cells, adjacency_matrix=cell_adjacency,
            params=ca_params, device='cpu', dtype=torch.float32)
        stress_switch.set_params_from_tensors(
            **{k: torch.tensor(v, dtype=torch.float32) for k, v in stress_params.items()})

        dt_ca, dt_stress = 0.01, 0.1
        for t in range(num_bio_steps):
            t_idx = min(t, sim.timeseriesVmem.shape[0] - 1)
            vmem_t = sim.timeseriesVmem[t_idx, 0, :, 0].to(dtype=torch.float32)
            stress_switch.compute_ca_from_vmem(vmem_t, dt_ca)
            stress_switch.step(dt_stress)
        # Equilibrate
        for _ in range(100):
            ca_final = stress_switch.Ca.detach().clone()
            stress_switch.step(dt_stress, Ca=ca_final)

        embryo_stress = stress_switch.get_embryo_stress().item()
        stress_S = stress_switch.S.detach().cpu().numpy()

        results.append({
            'vmem': vmem,
            'cell_pinches': cell_pinches,
            'pinch_max': pinch_max,
            'numRows': numRows,
            'numCols': numCols,
            'damping': d,
            'stress_S': stress_S,
            'embryo_stress': embryo_stress,
        })
        print(f"    Vmem: [{vmem.min():.3f}, {vmem.max():.3f}], "
              f"Stress S={embryo_stress:.4f}")

    return results


# ============================================================
# 3D surface builders
# ============================================================
def build_collective_stress_surface(grid_rows, grid_cols, embryo_stresses, mesh_res=100):
    """Build 3D mesh over embryo grid with Gaussian wells proportional to stress."""
    mesh_x = np.linspace(0, grid_cols, mesh_res)
    mesh_y = np.linspace(0, grid_rows, mesh_res)
    mesh_X, mesh_Y = np.meshgrid(mesh_x, mesh_y)
    mesh_Z = np.zeros_like(mesh_X)

    stress_max = max(embryo_stresses.max(), 0.01)
    for idx in range(grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        cx, cy = col + 0.5, row + 0.5
        stress_norm = embryo_stresses[idx] / stress_max
        sigma = 0.6
        dist_sq = (mesh_X - cx)**2 + (mesh_Y - cy)**2
        mesh_Z += stress_norm * 2.5 * np.exp(-dist_sq / (2 * sigma**2))

    return mesh_X, mesh_Y, mesh_Z


def build_vmem_surface(cell_pinches, pinch_max, numRows, numCols, mesh_res=80):
    """Build 3D mesh for an individual embryo's electric field."""
    mesh_x = np.linspace(0, numCols, mesh_res)
    mesh_y = np.linspace(0, numRows, mesh_res)
    mesh_X, mesh_Y = np.meshgrid(mesh_x, mesh_y)
    mesh_Z = np.zeros_like(mesh_X)

    numCells = numRows * numCols
    for cell_idx in range(numCells):
        cell_row = cell_idx // numCols
        cell_col = cell_idx % numCols
        cx, cy = cell_col + 0.5, cell_row + 0.5
        pinch = cell_pinches[cell_idx] / pinch_max
        sigma = 0.8
        dist_sq = (mesh_X - cx)**2 + (mesh_Y - cy)**2
        mesh_Z -= pinch * 2.0 * np.exp(-dist_sq / (2 * sigma**2))

    return mesh_X, mesh_Y, mesh_Z


# ============================================================
# Plotting: Collective stress field (top panel)
# ============================================================
def plot_collective_stress_field(ax, mesh_X, mesh_Y, mesh_Z,
                                  grid_rows, grid_cols,
                                  embryo_stresses, damping_map, mesh_res=100):
    """Render the collective stress field as a warm-colored 3D deformable surface."""
    # Style 3D axis
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#cccccc')
    ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(True, color='#e0e0e0', alpha=0.5)

    # Surface (warm colormap)
    ax.plot_surface(mesh_X, mesh_Y, mesh_Z,
                    cmap='YlOrRd', alpha=0.6, linewidth=0,
                    antialiased=True, shade=True)
    # Wireframe
    ax.plot_wireframe(mesh_X, mesh_Y, mesh_Z,
                      color='#e65100', alpha=0.25, linewidth=0.4,
                      rstride=3, cstride=3)

    # Plot embryos on surface
    damping_flat = damping_map.flatten()
    for idx in range(grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        cx, cy = col + 0.5, row + 0.5

        xi = int(cx / grid_cols * (mesh_res - 1))
        yi = int(cy / grid_rows * (mesh_res - 1))
        xi, yi = min(xi, mesh_res - 1), min(yi, mesh_res - 1)
        cz = mesh_Z[yi, xi]

        d = damping_flat[idx]
        s = 1.0 - d  # stress level: high s = high perturbation
        color = plt.cm.RdYlGn(1.0 - s)  # green=low stress, red=high stress
        stress = embryo_stresses[idx]
        stress_max = max(embryo_stresses.max(), 0.01)
        size = 120 + 200 * (stress / stress_max)

        ax.scatter([cx], [cy], [cz], c=[color], s=size,
                   edgecolors='white', linewidths=2.0, zorder=10,
                   depthshade=False)

        # Annotate stress level
        ax.text(cx, cy, cz + 0.4, f's={s:.2f}',
                fontsize=8, ha='center', va='bottom', color='#374151')

    # Annotate the tallest peak (highest stress)
    max_stress_idx = np.argmax(embryo_stresses)
    max_row = max_stress_idx // grid_cols
    max_col = max_stress_idx % grid_cols
    mcx, mcy = max_col + 0.5, max_row + 0.5
    mxi = int(mcx / grid_cols * (mesh_res - 1))
    myi = int(mcy / grid_rows * (mesh_res - 1))
    mcz = mesh_Z[min(myi, mesh_res - 1), min(mxi, mesh_res - 1)]
    ax.text(mcx + 1.2, mcy, mcz + 0.5,
            'High stress\n(eATP emission)',
            fontsize=10, fontweight='bold', color='#dc2626', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#dc2626', alpha=0.9))

    # Annotate diffusion region
    ax.text(grid_cols * 0.8, grid_rows * 0.15, mesh_Z.min() + 0.2,
            'Diffusion\n(shared medium)',
            fontsize=10, fontweight='bold', color='#ea580c', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#ea580c', alpha=0.9))

    ax.view_init(elev=35, azim=-50)
    ax.set_xlim(0, grid_cols)
    ax.set_ylim(grid_rows, 0)
    z_max = mesh_Z.max() + 0.5
    ax.set_zlim(-0.5, z_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_title('COLLECTIVE STRESS FIELD\n'
                 'Embryos communicate through shared extracellular medium',
                 fontsize=14, fontweight='bold', color='#1f2937', pad=10, y=0.96)


# ============================================================
# Plotting: Individual Vmem field (bottom panels)
# ============================================================
def plot_individual_vmem_field(ax, embryo_data, label, label_color):
    """Render one embryo's Vmem as a blue 3D deformable surface with cells."""
    vmem = embryo_data['vmem']
    numRows, numCols = embryo_data['numRows'], embryo_data['numCols']
    mesh_res = 80

    mesh_X, mesh_Y, mesh_Z = build_vmem_surface(
        embryo_data['cell_pinches'], embryo_data['pinch_max'],
        numRows, numCols, mesh_res)

    # Style
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#cccccc')
    ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(True, color='#e0e0e0', alpha=0.5)

    # Surface (blue)
    ax.plot_surface(mesh_X, mesh_Y, mesh_Z,
                    cmap='Blues', alpha=0.6, linewidth=0,
                    antialiased=True, shade=True)
    ax.plot_wireframe(mesh_X, mesh_Y, mesh_Z,
                      color='#3b82f6', alpha=0.3, linewidth=0.4,
                      rstride=3, cstride=3)

    # Cells as scatter
    vmem_min, vmem_max = vmem.min(), vmem.max()
    numCells = numRows * numCols
    for cell_idx in range(numCells):
        cell_row = cell_idx // numCols
        cell_col = cell_idx % numCols
        cx, cy = cell_col + 0.5, cell_row + 0.5

        xi = int(cx / numCols * (mesh_res - 1))
        yi = int(cy / numRows * (mesh_res - 1))
        xi, yi = min(xi, mesh_res - 1), min(yi, mesh_res - 1)
        cz = mesh_Z[yi, xi]

        vmem_norm = (vmem[cell_idx] - vmem_min) / (vmem_max - vmem_min + 1e-10)
        color = plt.cm.coolwarm(vmem_norm)
        pinch = embryo_data['cell_pinches'][cell_idx] / embryo_data['pinch_max']
        size = 30 + 60 * pinch

        ax.scatter([cx], [cy], [cz], c=[color], s=size,
                   edgecolors='white', linewidths=1.0, zorder=10,
                   depthshade=False)

    ax.view_init(elev=30, azim=-45)
    ax.set_xlim(0, numCols)
    ax.set_ylim(numRows, 0)
    ax.set_zlim(-3, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_title(label, fontsize=11, fontweight='bold', color='#1f2937', pad=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=label_color))

    stress_val = embryo_data['embryo_stress']
    ax.text2D(0.5, -0.02, f'Stress emission: S = {stress_val:.3f}',
              transform=ax.transAxes, fontsize=10, ha='center', color='#6b7280')


# ============================================================
# Legend panel
# ============================================================
def add_legend_panel(ax, damping_map, embryo_stresses, grid_rows, grid_cols):
    """Render legend with health status key and stress values."""
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')

    ax.text(5, 9.8, 'Embryo Stress Level', fontsize=13,
            fontweight='bold', ha='center', va='top', color='#1f2937')

    y = 8.8
    for s_val, label_text, color_hex in [
        (0.0, 'Healthy (s = 0.0)', '#16a34a'),
        (0.15, 'Mildly stressed (s = 0.15)', '#b45309'),
        (0.3, 'Strongly stressed (s = 0.30)', '#dc2626'),
    ]:
        circle = Circle((1.5, y), 0.3, facecolor=plt.cm.RdYlGn(1.0 - s_val),
                         edgecolor='white', linewidth=2, transform=ax.transData)
        ax.add_patch(circle)
        ax.text(2.5, y, label_text, fontsize=10, va='center', color=color_hex)
        y -= 1.1

    y -= 0.4
    ax.text(5, y, 'Stress Field Equation', fontsize=12,
            fontweight='bold', ha='center', color='#1f2937')
    y -= 0.7
    ax.text(5, y, r'$\frac{dF}{dt} = D_F \nabla^2 F - \gamma_F F + \mathrm{emission}$',
            fontsize=13, ha='center', color='#9a3412', family='serif')
    y -= 0.7
    ax.text(5, y, 'emission = mean(S) per embryo',
            fontsize=9, ha='center', color='#6b7280', style='italic')

    y -= 1.0
    ax.text(5, y, 'Rescue Mechanism', fontsize=12,
            fontweight='bold', ha='center', color='#1f2937')
    y -= 0.7
    ax.text(5, y,
            'Stressed embryos emit eATP\n'
            'into the shared medium.\n'
            'Collective signal reshapes\n'
            "each receiver's ATP dynamics\n"
            'via saddle-node bifurcation.',
            fontsize=9, ha='center', va='top', color='#6b7280',
            linespacing=1.4)


# ============================================================
# Combined visualization: embryo fields overlaid on stress landscape
# ============================================================
def plot_combined_field(ax, group_data, representative_data):
    """
    Single 3D scene: warm stress landscape with blue embryo Vmem surfaces on top.

    The stress field is a smooth warm surface spanning the embryo grid.
    At each embryo position, a small blue Vmem surface sits on the stress
    landscape, showing the internal bioelectric pattern.
    """
    grid_rows = group_data['grid_rows']
    grid_cols = group_data['grid_cols']
    embryo_stresses = group_data['final_stress']
    damping_map = group_data['damping_map']
    damping_flat = damping_map.flatten()
    cell_grid_size = group_data['cell_grid_size']
    num_embryos = grid_rows * grid_cols

    # --- Style 3D axis ---
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#cccccc')
    ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(True, color='#e0e0e0', alpha=0.5)

    # --- 1. Stress landscape (warm surface) ---
    stress_res = 120
    mesh_X, mesh_Y, mesh_Z = build_collective_stress_surface(
        grid_rows, grid_cols, embryo_stresses, mesh_res=stress_res)

    ax.plot_surface(mesh_X, mesh_Y, mesh_Z,
                    cmap='YlOrRd', alpha=0.45, linewidth=0,
                    antialiased=True, shade=True)
    ax.plot_wireframe(mesh_X, mesh_Y, mesh_Z,
                      color='#e65100', alpha=0.15, linewidth=0.3,
                      rstride=4, cstride=4)

    # --- 2. Embryo Vmem surfaces (blue) overlaid at each grid position ---
    # Build a mapping from damping -> representative data for pinch info
    repr_by_damping = {}
    for rd in representative_data:
        repr_by_damping[rd['damping']] = rd

    # Inset scale: each embryo occupies a fraction of one grid cell
    inset_margin = 0.12  # margin from cell edges
    inset_size = 1.0 - 2 * inset_margin  # size of inset within cell

    vmem_res = 40  # mesh resolution per embryo inset
    vmem_amplitude = 1.2  # scale for Vmem deformations on top of stress

    for idx in range(num_embryos):
        row = idx // grid_cols
        col = idx % grid_cols
        d = damping_flat[idx]

        # Find matching representative for pinch data (closest damping)
        closest_d = min(repr_by_damping.keys(), key=lambda k: abs(k - d))
        repr_data = repr_by_damping[closest_d]
        cell_pinches = repr_data['cell_pinches']
        pinch_max = repr_data['pinch_max']

        # Local mesh within this embryo's footprint
        x0 = col + inset_margin
        y0 = row + inset_margin
        local_x = np.linspace(x0, x0 + inset_size, vmem_res)
        local_y = np.linspace(y0, y0 + inset_size, vmem_res)
        lX, lY = np.meshgrid(local_x, local_y)

        # Compute stress landscape height at this embryo's center
        cx, cy = col + 0.5, row + 0.5
        sxi = int(cx / grid_cols * (stress_res - 1))
        syi = int(cy / grid_rows * (stress_res - 1))
        sxi, syi = min(sxi, stress_res - 1), min(syi, stress_res - 1)
        z_base = mesh_Z[syi, sxi]

        # Build local Vmem deformation sitting on top of the stress surface
        lZ = np.full_like(lX, z_base)
        numCells = cell_grid_size * cell_grid_size
        for cell_idx in range(numCells):
            cell_row = cell_idx // cell_grid_size
            cell_col = cell_idx % cell_grid_size
            cell_cx = x0 + (cell_col + 0.5) / cell_grid_size * inset_size
            cell_cy = y0 + (cell_row + 0.5) / cell_grid_size * inset_size
            pinch = cell_pinches[cell_idx] / pinch_max
            sigma = inset_size / cell_grid_size * 0.6
            dist_sq = (lX - cell_cx)**2 + (lY - cell_cy)**2
            lZ -= pinch * vmem_amplitude * np.exp(-dist_sq / (2 * sigma**2))

        # Plot this embryo's Vmem surface (blue)
        ax.plot_surface(lX, lY, lZ,
                        cmap='Blues', alpha=0.75, linewidth=0,
                        antialiased=True, shade=True)
        ax.plot_wireframe(lX, lY, lZ,
                          color='#3b82f6', alpha=0.3, linewidth=0.3,
                          rstride=2, cstride=2)

        # Annotate stress level
        s = 1.0 - d
        ax.text(cx, cy, z_base + vmem_amplitude * 0.3 + 0.3, f's={s:.2f}',
                fontsize=7, ha='center', va='bottom', color='#374151',
                fontweight='bold')

    # --- Camera and limits ---
    ax.view_init(elev=40, azim=-45)
    ax.set_xlim(0, grid_cols)
    ax.set_ylim(grid_rows, 0)
    z_max = mesh_Z.max() + vmem_amplitude + 1.0
    ax.set_zlim(-0.5, z_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def create_combined_figure(group_data, representative_data, output_path):
    """Single combined visualization: embryo fields overlaid on stress landscape."""
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('white')

    ax_3d = fig.add_axes([0.02, 0.08, 0.68, 0.82], projection='3d')
    ax_legend = fig.add_axes([0.72, 0.15, 0.26, 0.70])

    plot_combined_field(ax_3d, group_data, representative_data)

    ax_3d.set_title(
        'Embryo bioelectric fields (blue) on collective stress landscape (warm)',
        fontsize=13, fontweight='bold', color='#1f2937', pad=10, y=0.96)

    add_legend_panel(
        ax_legend, group_data['damping_map'],
        group_data['final_stress'],
        group_data['grid_rows'], group_data['grid_cols'])

    fig.suptitle(
        'MULTI-EMBRYO STRESS RESCUE NETWORK',
        fontsize=18, fontweight='bold', color='#1f2937', y=0.97)

    combined_path = output_path.replace('.png', '_combined.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {combined_path}")

    combined_pdf = combined_path.replace('.png', '.pdf')
    plt.savefig(combined_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {combined_pdf}")

    plt.close(fig)


# ============================================================
# Main figure assembly
# ============================================================
def create_figure(group_data, representative_data, output_path):
    """Assemble the full two-level field visualization."""
    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor('white')

    # Top: Collective stress field
    ax_stress_3d = fig.add_axes([0.05, 0.48, 0.60, 0.47], projection='3d')
    ax_legend = fig.add_axes([0.68, 0.48, 0.29, 0.47])

    # Bottom: Individual embryo fields
    ax_vmem_0 = fig.add_axes([0.03, 0.02, 0.30, 0.38], projection='3d')
    ax_vmem_1 = fig.add_axes([0.35, 0.02, 0.30, 0.38], projection='3d')
    ax_vmem_2 = fig.add_axes([0.67, 0.02, 0.30, 0.38], projection='3d')

    # Build and plot collective stress surface
    mesh_X, mesh_Y, mesh_Z = build_collective_stress_surface(
        group_data['grid_rows'], group_data['grid_cols'],
        group_data['final_stress'])
    plot_collective_stress_field(
        ax_stress_3d, mesh_X, mesh_Y, mesh_Z,
        group_data['grid_rows'], group_data['grid_cols'],
        group_data['final_stress'], group_data['damping_map'])

    # Legend
    add_legend_panel(
        ax_legend, group_data['damping_map'],
        group_data['final_stress'],
        group_data['grid_rows'], group_data['grid_cols'])

    # Bottom panels
    labels_colors = [
        ('HEALTHY (s=0.0)\nIntact bioelectric field', '#16a34a'),
        ('MILDLY STRESSED (s=0.15)\nPartially disrupted field', '#b45309'),
        ('STRONGLY STRESSED (s=0.30)\nCollapsed field pattern', '#dc2626'),
    ]
    for ax, data, (label, color) in zip(
        [ax_vmem_0, ax_vmem_1, ax_vmem_2],
        representative_data,
        labels_colors):
        plot_individual_vmem_field(ax, data, label, color)

    # Connection annotation band
    fig.text(0.5, 0.445,
             'Individual cell stress (S)  \u2192  Collective extracellular field (F)',
             fontsize=14, ha='center', va='center', color='#374151',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff7ed',
                       edgecolor='#ea580c', alpha=0.9))

    # Subtitle for bottom row
    fig.text(0.5, 0.42,
             'INDIVIDUAL EMBRYO BIOELECTRIC FIELDS (11\u00d711 cells)',
             fontsize=12, ha='center', va='center', color='#6b7280',
             fontweight='bold')

    # Main title
    fig.suptitle(
        'MULTI-EMBRYO STRESS RESCUE NETWORK\n'
        'Bioelectric fields within embryos generate stress signals '
        'that diffuse between embryos',
        fontsize=16, fontweight='bold', color='#1f2937', y=0.98)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    if output_path.endswith('.png'):
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"Saved: {pdf_path}")

    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("MULTI-EMBRYO NETWORK VISUALIZATION")
    print("=" * 60)

    if args.stressParamsFile is not None:
        stress_params, ca_params = load_stress_params(args.stressParamsFile)
    else:
        stress_params, ca_params = get_default_stress_params()

    print("\n--- Running 3x3 group simulation ---")
    group_data = run_group_simulation(args.numBioSteps, stress_params, ca_params)

    print(f"\nGroup stress values: {group_data['final_stress']}")
    print(f"Damping map:\n{group_data['damping_map']}")

    print("\n--- Running representative embryos ---")
    representative_data = run_representative_embryos(
        args.numBioSteps, stress_params, ca_params)

    print("\n--- Creating split figure ---")
    create_figure(group_data, representative_data, args.output)

    print("\n--- Creating combined figure ---")
    create_combined_figure(group_data, representative_data, args.output)

    print("\nDone!")


if __name__ == '__main__':
    main()
