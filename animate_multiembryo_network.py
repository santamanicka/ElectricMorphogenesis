#!/usr/bin/env python3
"""
Animated Multi-Embryo Stress Rescue Network.

Runs a group rescue simulation and renders an animation of the combined view:
warm stress landscape with blue embryo Vmem surfaces riding on top.
As stress builds, embryo surfaces rise; as patterns collapse, surfaces flatten.

Usage:
    python animate_multiembryo_network.py
    python animate_multiembryo_network.py --groupSize 9 --dampingGaussian "0.5,0.01" \\
        --alpha 100.0 --D_F 0.5 --gamma_F 0.0001 --numBioSteps 2000 \\
        --stressParamsFile data/bestLearnedStressParams_6.dat --initialStress 0.0

Output: data/multiembryo_animation.gif (and .mp4 if ffmpeg available)
"""

import argparse
import copy
import math
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from embryo import model
from stressBistableSwitch import StressBistableSwitch
import utilities


# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description='Animate multi-embryo stress rescue')
parser.add_argument('--groupSize', type=int, default=9)
parser.add_argument('--gridDims', type=str, default=None)
parser.add_argument('--alpha', type=float, default=100.0)
parser.add_argument('--numBioSteps', type=int, default=2000)
parser.add_argument('--numStressSteps', type=int, default=500)
parser.add_argument('--neighborhood', type=str, default='vonNeumann')
parser.add_argument('--stressParamsFile', type=str, default=None)
parser.add_argument('--D_F', type=float, default=0.5)
parser.add_argument('--gamma_F', type=float, default=0.0001)
parser.add_argument('--diffusion_substeps', type=int, default=10)
parser.add_argument('--initialStress', type=float, default=0.0)
# Damping modes
damping_group = parser.add_mutually_exclusive_group()
damping_group.add_argument('--dampingLevels', type=str, default=None)
damping_group.add_argument('--dampingRange', type=str, default=None)
damping_group.add_argument('--dampingMap', type=str, default=None)
damping_group.add_argument('--dampingCenter', type=float, default=None)
damping_group.add_argument('--dampingGaussian', type=str, default=None)
# Animation
parser.add_argument('--frameInterval', type=int, default=20,
                    help='Record a frame every N bio steps (default: 20)')
parser.add_argument('--fps', type=int, default=15, help='Animation FPS (default: 15)')
parser.add_argument('--output', type=str, default='data/multiembryo_animation.gif')
parser.add_argument('--embryoSurface', type=str, default='field',
                    choices=['field', 'vmem'],
                    help='Embryo surface mode: "field" for electric field magnitude (default), '
                         '"vmem" for Vmem deviation from mean')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


# ============================================================
# Utility functions (from runGroupRescue.py / visualize_multiembryo_network.py)
# ============================================================
def apply_sigmoid_constraint(raw_param, min_val, max_val):
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)


def load_stress_params(params_file):
    data = torch.load(params_file, weights_only=False)
    stress_params = {}
    param_bounds = data.get('parameter_bounds', {})
    raw_params = data.get('parameters', {})
    for pname, raw_val in raw_params.items():
        mk, xk = f'{pname}_min', f'{pname}_max'
        if mk in param_bounds and xk in param_bounds:
            stress_params[pname] = float(
                apply_sigmoid_constraint(raw_val, param_bounds[mk], param_bounds[xk]).item())
        else:
            stress_params[pname] = float(raw_val.item())
    fixed_ca = data.get('fixed_ca_params', None)
    if fixed_ca is None:
        fixed_ca = get_default_ca_params()
    return stress_params, fixed_ca


def get_default_ca_params():
    return {'tau_ca': 2.5964, 'g_ca': 5.3437, 'V_half_ca': -0.0753,
            'k_ca': 0.0021, 'k_decay_ca': 4.3346}


def get_default_stress_params():
    return {
        'tau_S': 50.0, 'k_on_S': 3.0, 'k_off_S': 0.02, 'K_S': 0.4,
        'Ca_stress_threshold': 8.8, 'sigma_ca': 0.5, 'gain_S': 2.0,
        'or_threshold_S': 0.6, 'D_S': 0.15, 'gamma': 0.08, 'K_decay': 0.3,
    }, get_default_ca_params()


def load_model_parameters(grn_damping=1.0):
    path = './data/bestModelParameters_fieldVector_Ligand_GRN_253.dat'
    params = torch.load(path, weights_only=False)
    if "ATPParameters" not in params:
        params["ATPParameters"] = None
    if grn_damping != 1.0 and 'GRNParameters' in params and params['GRNParameters'] is not None:
        grn = params['GRNParameters']
        for k in ['GRNWeights', 'InterGRNWeights', 'GRNtoLigandWeights']:
            if k in grn and grn[k] is not None:
                grn[k] = grn[k] * grn_damping
        if grn_damping == 0.0:
            grn['GRNEnabled'] = False
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
                    adj[idx, ni * cols + nj] = 1.0
    return adj


def build_moore_adjacency(rows, cols):
    n = rows * cols
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        adj[idx, ni * cols + nj] = 1.0
    return adj


def compute_effective_damping(base_damping, neighbor_stress_mean, alpha):
    base_clamped = max(min(base_damping, 0.999), 0.001)
    base_logit = np.log(base_clamped / (1.0 - base_clamped))
    return 1.0 / (1.0 + np.exp(-(base_logit + alpha * neighbor_stress_mean)))


# ============================================================
# Grid / damping setup
# ============================================================
def derive_grid_dims(group_size):
    side = int(math.ceil(math.sqrt(group_size)))
    rows, cols = side, side
    while rows * cols > group_size and rows > 1:
        rows -= 1
    if rows * cols < group_size:
        rows += 1
    return rows, cols


def build_damping_map(args, rows, cols):
    n = rows * cols
    if args.dampingMap is not None:
        vals = [float(x) for x in args.dampingMap.split(',')]
        return np.array(vals).reshape(rows, cols)
    elif args.dampingGaussian is not None:
        mean_d, std_d = [float(x) for x in args.dampingGaussian.split(',')]
        return np.clip(np.random.normal(mean_d, std_d, (rows, cols)), 0.01, 1.0)
    elif args.dampingRange is not None:
        lo, hi = [float(x) for x in args.dampingRange.split(',')]
        return np.random.uniform(lo, hi, (rows, cols))
    elif args.dampingCenter is not None:
        dmap = np.ones((rows, cols))
        dmap[rows // 2, cols // 2] = args.dampingCenter
        return dmap
    elif args.dampingLevels is not None:
        levels = [float(x) for x in args.dampingLevels.split(',')]
        dmap = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                dmap[i, j] = levels[(i * cols + j) % len(levels)]
        return dmap
    else:
        return np.full((rows, cols), 0.5)


# ============================================================
# Simulation with timeseries recording
# ============================================================
def run_simulation_with_recording(args, stress_params, ca_params):
    """Run group rescue and record snapshots every frameInterval steps."""
    if args.gridDims:
        grid_rows, grid_cols = [int(x) for x in args.gridDims.split(',')]
    else:
        grid_rows, grid_cols = derive_grid_dims(args.groupSize)
    num_embryos = grid_rows * grid_cols

    damping_map = build_damping_map(args, grid_rows, grid_cols)
    damping_flat = damping_map.flatten()
    print(f"Grid: {grid_rows}x{grid_cols}, damping range: [{damping_flat.min():.3f}, {damping_flat.max():.3f}]")

    # Adjacency
    if args.neighborhood == 'moore':
        embryo_adj = build_moore_adjacency(grid_rows, grid_cols)
        max_degree = 8
    else:
        embryo_adj = build_vonneumann_adjacency(grid_rows, grid_cols)
        max_degree = 4

    # Cell-level setup
    ref_params = load_model_parameters(grn_damping=1.0)
    cell_grid_size = ref_params['latticeDims'][0]
    num_cells = cell_grid_size * cell_grid_size
    utils_obj = utilities.utilities()
    cell_adjacency = utils_obj.computeLatticeAdjacencyMatrix(
        (cell_grid_size, cell_grid_size), False)

    # Instantiate embryos
    print(f"Instantiating {num_embryos} embryos...")
    embryos = []
    for idx in range(num_embryos):
        d = damping_flat[idx]
        params = load_model_parameters(grn_damping=d)
        ns = params["simParameters"]["numSamples"]
        iv = copy.deepcopy(params["simParameters"]["initialValues"])
        ei = copy.deepcopy(params["simParameters"]["externalInputs"])
        cp = copy.deepcopy(params["clampParameters"])

        bio_model = model(params, numBasicSamples=ns)
        bio_model.setExperimentalConditions((iv, ns))

        orig_grn_w = bio_model.geneNetwork.tissueGRNWeights.clone()
        has_lig = (hasattr(bio_model.electricNetwork, 'GRNtoLigandWeights') and
                   bio_model.electricNetwork.GRNtoLigandWeights is not None and
                   not isinstance(bio_model.electricNetwork.GRNtoLigandWeights, bool))
        orig_lig_w = bio_model.electricNetwork.GRNtoLigandWeights.clone() if has_lig else None

        ss = StressBistableSwitch(num_cells, cell_adjacency, ca_params,
                                  device='cpu', dtype=torch.float32)
        ss.set_params_from_tensors(
            **{k: torch.tensor(v, dtype=torch.float32) for k, v in stress_params.items()})

        if args.initialStress > 0:
            ss.S = torch.full((num_cells,), args.initialStress, dtype=torch.float32)

        embryos.append({
            'bio_model': bio_model, 'stress_switch': ss,
            'external_inputs': ei, 'clamp_params': cp,
            'orig_grn_w': orig_grn_w, 'orig_lig_w': orig_lig_w, 'has_lig': has_lig,
        })

    # Diffusive field
    F = np.zeros(num_embryos)
    D_F, gamma_F = args.D_F, args.gamma_F
    n_sub = args.diffusion_substeps
    dt_ca, dt_stress = 0.01, 0.1

    # Recording
    fi = args.frameInterval
    total_steps = args.numBioSteps + args.numStressSteps
    record_times = list(range(0, total_steps, fi))
    if (total_steps - 1) not in record_times:
        record_times.append(total_steps - 1)

    frames_field = []      # per-embryo diffusive field F at each recorded frame
    frames_vmem = []       # per-embryo Vmem arrays at each recorded frame
    frames_efield = []     # per-embryo per-cell electric field magnitude at each recorded frame
    frames_damping = []    # effective damping at each recorded frame
    frames_time = []

    def get_cell_field_magnitude(bio_model):
        """Compute mean electric field magnitude at each cell from surrounding field grid points."""
        net = bio_model.electricNetwork
        # eV: (numSamples, numFieldGridPoints, 1)
        # fieldScreenMatrixIn: (numSamples, numFieldGridPoints, numCells)
        # Replicate the eVneighborsMean computation from cellularFieldNetwork.py:261
        cell_field = (net.eV * net.fieldScreenMatrixIn).sum(1) / max(net.numFieldNeighbors, 1)
        # cell_field: (numSamples, numCells)
        return cell_field[0].detach().cpu().numpy()  # (numCells,)

    def record_frame(t):
        vmem_list = [embryos[i]['bio_model'].electricNetwork.Vmem[0, :, 0].detach().cpu().numpy().copy()
                     for i in range(num_embryos)]
        efield_list = [get_cell_field_magnitude(embryos[i]['bio_model'])
                       for i in range(num_embryos)]
        eff_damping = np.array([compute_effective_damping(damping_flat[i], F[i], args.alpha)
                                for i in range(num_embryos)])
        frames_field.append(F.copy())
        frames_vmem.append(vmem_list)
        frames_efield.append(efield_list)
        frames_damping.append(eff_damping)
        frames_time.append(t)

    # --- Bio + stress concurrent phase ---
    t_start = time.time()
    print(f"Running simulation ({args.numBioSteps} bio + {args.numStressSteps} equil steps)...")

    for t in range(args.numBioSteps):
        # Bio step
        for idx in range(num_embryos):
            e = embryos[idx]
            e['bio_model'].simulate(
                externalInputs=e['external_inputs'], clampParameters=e['clamp_params'],
                perturbation=None, fieldModulation=False, numSimIters=1, outerIter=t)

        # Stress step
        for idx in range(num_embryos):
            vmem_flat = embryos[idx]['bio_model'].electricNetwork.Vmem[0, :, 0]
            embryos[idx]['stress_switch'].compute_ca_from_vmem(
                vmem_flat.to(dtype=torch.float32), dt_ca)
            embryos[idx]['stress_switch'].step(dt_stress)

        # Diffuse
        emission = np.array([embryos[i]['stress_switch'].get_embryo_stress().item()
                             for i in range(num_embryos)])
        dt_sub = 1.0 / n_sub
        for _ in range(n_sub):
            lap = embryo_adj @ F - max_degree * F
            F = F + dt_sub * (D_F * lap - gamma_F * F + emission)
            F = np.clip(F, 0.0, None)

        # Update damping
        for idx in range(num_embryos):
            eff_d = compute_effective_damping(damping_flat[idx], F[idx], args.alpha)
            e = embryos[idx]
            e['bio_model'].geneNetwork.tissueGRNWeights = e['orig_grn_w'] * eff_d
            if e['has_lig']:
                e['bio_model'].electricNetwork.GRNtoLigandWeights = e['orig_lig_w'] * eff_d

        # Record
        if t in record_times:
            record_frame(t)

        if (t + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate = (t + 1) / elapsed
            print(f"  Bio step {t+1}/{args.numBioSteps}: "
                  f"mean_stress={emission.mean():.4f}, mean_F={F.mean():.4f} "
                  f"[{rate:.0f} steps/s]")

    # --- Stress equilibration phase ---
    final_emission = np.array([embryos[i]['stress_switch'].get_embryo_stress().item()
                               for i in range(num_embryos)])
    for t_eq in range(args.numStressSteps):
        t_total = args.numBioSteps + t_eq
        for idx in range(num_embryos):
            ca_final = embryos[idx]['stress_switch'].Ca.detach().clone()
            embryos[idx]['stress_switch'].step(dt_stress, Ca=ca_final)
        dt_sub = 1.0 / n_sub
        for _ in range(n_sub):
            lap = embryo_adj @ F - max_degree * F
            F = F + dt_sub * (D_F * lap - gamma_F * F + final_emission)
            F = np.clip(F, 0.0, None)

        if t_total in record_times:
            record_frame(t_total)

    elapsed = time.time() - t_start
    print(f"  Simulation complete: {elapsed:.1f}s, {len(frames_field)} frames recorded")

    return {
        'frames_field': frames_field,
        'frames_vmem': frames_vmem,
        'frames_efield': frames_efield,
        'frames_damping': frames_damping,
        'frames_time': frames_time,
        'damping_map': damping_map,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'cell_grid_size': cell_grid_size,
    }


# ============================================================
# 3D surface builders (from visualize_multiembryo_network.py)
# ============================================================
def build_stress_surface(grid_rows, grid_cols, embryo_stresses, stress_max_global, mesh_res=80):
    """Build convex stress landscape; normalized to global max for consistent scale."""
    mx = np.linspace(0, grid_cols, mesh_res)
    my = np.linspace(0, grid_rows, mesh_res)
    mX, mY = np.meshgrid(mx, my)
    mZ = np.zeros_like(mX)
    norm = max(stress_max_global, 1e-6)
    for idx in range(grid_rows * grid_cols):
        row, col = idx // grid_cols, idx % grid_cols
        cx, cy = col + 0.5, row + 0.5
        s_norm = embryo_stresses[idx] / norm
        dist_sq = (mX - cx)**2 + (mY - cy)**2
        mZ += s_norm * 2.5 * np.exp(-dist_sq / (2 * 0.6**2))
    return mX, mY, mZ


def compute_vmem_pinches(vmem, cell_grid_size):
    """Compute per-cell pinch from Vmem deviation (no field vectors needed)."""
    vmem_mean = vmem.mean()
    deviation = np.abs(vmem - vmem_mean)
    dev_max = deviation.max()
    if dev_max < 1e-10:
        return np.zeros_like(vmem), 1.0
    return deviation, dev_max


# ============================================================
# Frame rendering
# ============================================================
def render_frame(ax, sim_data, frame_idx, stress_max_global, z_max_global):
    """Render one frame of the combined visualization."""
    ax.cla()

    grid_rows = sim_data['grid_rows']
    grid_cols = sim_data['grid_cols']
    cell_grid_size = sim_data['cell_grid_size']
    damping_flat = sim_data['damping_map'].flatten()
    num_embryos = grid_rows * grid_cols

    embryo_stresses = sim_data['frames_field'][frame_idx]
    vmem_list = sim_data['frames_vmem'][frame_idx]
    t = sim_data['frames_time'][frame_idx]

    # Style
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#cccccc')
    ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(True, color='#e0e0e0', alpha=0.5)

    stress_res = 80
    mX, mY, mZ = build_stress_surface(
        grid_rows, grid_cols, embryo_stresses, stress_max_global, mesh_res=stress_res)

    # Stress landscape (warm)
    ax.plot_surface(mX, mY, mZ, cmap='YlOrRd', alpha=0.45,
                    linewidth=0, antialiased=True, shade=True)
    ax.plot_wireframe(mX, mY, mZ, color='#e65100', alpha=0.15,
                      linewidth=0.3, rstride=4, cstride=4)

    # Embryo Vmem insets (blue)
    inset_margin = 0.12
    inset_size = 1.0 - 2 * inset_margin
    vmem_res = 30
    vmem_amplitude = 1.2

    embryo_surface_mode = sim_data.get('embryo_surface_mode', 'field')
    efield_list = sim_data['frames_efield'][frame_idx] if embryo_surface_mode == 'field' else None

    for idx in range(num_embryos):
        row, col = idx // grid_cols, idx % grid_cols
        d = damping_flat[idx]
        vmem = vmem_list[idx]

        if embryo_surface_mode == 'field':
            efield = efield_list[idx]
            ef_max = efield.max()
            if ef_max < 1e-10:
                pinches, pinch_max = np.zeros_like(efield), 1.0
            else:
                pinches, pinch_max = efield, ef_max
        else:
            pinches, pinch_max = compute_vmem_pinches(vmem, cell_grid_size)

        x0 = col + inset_margin
        y0 = row + inset_margin
        lx = np.linspace(x0, x0 + inset_size, vmem_res)
        ly = np.linspace(y0, y0 + inset_size, vmem_res)
        lX, lY = np.meshgrid(lx, ly)

        # Z base from stress surface
        cx, cy = col + 0.5, row + 0.5
        sxi = min(int(cx / grid_cols * (stress_res - 1)), stress_res - 1)
        syi = min(int(cy / grid_rows * (stress_res - 1)), stress_res - 1)
        z_base = mZ[syi, sxi]

        lZ = np.full_like(lX, z_base)
        num_cells = cell_grid_size * cell_grid_size
        for ci in range(num_cells):
            cr, cc = ci // cell_grid_size, ci % cell_grid_size
            ccx = x0 + (cc + 0.5) / cell_grid_size * inset_size
            ccy = y0 + (cr + 0.5) / cell_grid_size * inset_size
            p = pinches[ci] / pinch_max
            sigma = inset_size / cell_grid_size * 0.6
            dsq = (lX - ccx)**2 + (lY - ccy)**2
            lZ -= p * vmem_amplitude * np.exp(-dsq / (2 * sigma**2))

        ax.plot_surface(lX, lY, lZ, cmap='Blues', alpha=0.75,
                        linewidth=0, antialiased=True, shade=True)
        ax.plot_wireframe(lX, lY, lZ, color='#3b82f6', alpha=0.25,
                          linewidth=0.3, rstride=2, cstride=2)

        # Field label
        f_val = sim_data['frames_field'][frame_idx][idx]
        ax.text(cx, cy, z_base + vmem_amplitude * 0.2 + 0.25, f'F={f_val:.2f}',
                fontsize=6, ha='center', va='bottom', color='#374151',
                fontweight='bold')

    # Camera and limits (fixed across frames for smooth animation)
    ax.view_init(elev=40, azim=-45)
    ax.set_xlim(0, grid_cols)
    ax.set_ylim(grid_rows, 0)
    ax.set_zlim(-0.3, z_max_global)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_title(f't = {t}', fontsize=14, fontweight='bold', color='#1f2937', pad=5)


# ============================================================
# Main
# ============================================================
def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("MULTI-EMBRYO NETWORK ANIMATION")
    print("=" * 60)

    if args.stressParamsFile:
        stress_params, ca_params = load_stress_params(args.stressParamsFile)
    else:
        stress_params, ca_params = get_default_stress_params()

    print("\n--- Running simulation with timeseries recording ---")
    sim_data = run_simulation_with_recording(args, stress_params, ca_params)
    sim_data['embryo_surface_mode'] = args.embryoSurface

    # Compute global scale for consistent axis limits across all frames
    all_stresses = np.array(sim_data['frames_field'])
    stress_max_global = all_stresses.max()
    # Z max = stress surface peak + vmem amplitude + margin
    z_max_global = (stress_max_global / max(stress_max_global, 1e-6)) * 2.5 + 1.2 + 0.5
    print(f"Global stress max: {stress_max_global:.4f}, z_max: {z_max_global:.2f}")

    num_frames = len(sim_data['frames_field'])
    print(f"\n--- Rendering {num_frames} frames ---")

    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.90], projection='3d')

    fig.suptitle('MULTI-EMBRYO STRESS RESCUE NETWORK',
                 fontsize=16, fontweight='bold', color='#1f2937', y=0.97)

    def update(frame_idx):
        render_frame(ax, sim_data, frame_idx, stress_max_global, z_max_global)
        if (frame_idx + 1) % 10 == 0:
            print(f"  Rendered frame {frame_idx + 1}/{num_frames}")
        return []

    anim = FuncAnimation(fig, update, frames=num_frames, blit=False)

    # Build parameter suffix for output filename
    damping_map = sim_data['damping_map'].flatten()
    d_lo, d_hi = damping_map.min(), damping_map.max()
    suffix = (f"_g{sim_data['grid_rows']*sim_data['grid_cols']}"
              f"_a{args.alpha:.1f}"
              f"_d{d_lo:.2f}-{d_hi:.2f}"
              f"_t{args.numBioSteps}"
              f"_D{args.D_F}_g{args.gamma_F}")
    base, ext = os.path.splitext(args.output)
    output_path = base + suffix + ext

    # Save GIF
    print(f"\n--- Saving animation ---")
    gif_path = output_path
    writer = PillowWriter(fps=args.fps)
    anim.save(gif_path, writer=writer, dpi=100)
    print(f"Saved: {gif_path}")

    # Try MP4
    try:
        mp4_path = gif_path.replace('.gif', '.mp4')
        from matplotlib.animation import FFMpegWriter
        mp4_writer = FFMpegWriter(fps=args.fps, bitrate=2000)
        anim.save(mp4_path, writer=mp4_writer, dpi=100)
        print(f"Saved: {mp4_path}")
    except Exception as e:
        print(f"MP4 save skipped ({e})")

    plt.close(fig)
    print("\nDone!")


if __name__ == '__main__':
    main()