#!/usr/bin/env python3
"""
Algebraic connectivity analysis for multi-embryo group rescue simulations.

Computes the Fiedler eigenvalue (lambda_2) of the functional connectivity
graph built from pairwise Vmem timeseries correlations.  Two normalizations
are reported:

  1. lambda_2 / N  -- removes trivial linear scaling with group size
  2. lambda_2 / lambda_2_null  -- ratio to null model (shuffled timeseries)

Modes:
  simulate  -- run simulation and save all data to .dat file
  analyze   -- load saved .dat file, compute metrics, and visualize
  (default) -- simulate + analyze in one shot

Usage:
    # One-shot (simulate + analyze):
    python analyzeGroupRescue.py --groupSize 25 \\
        --dampingGaussian "0.5,0.01" --alpha 10.0 \\
        --D_F 0.5 --gamma_F 0.0001 --numBioSteps 2000 \\
        --stressParamsFile data/bestLearnedStressParams_6.dat \\
        --initialStress 1.0

    # Simulate only (save data for later analysis):
    python analyzeGroupRescue.py --mode simulate --groupSize 200 \\
        --dampingGaussian "0.5,0.01" --alpha 10.0 --numBioSteps 2000 \\
        --stressParamsFile data/bestLearnedStressParams_6.dat \\
        --initialStress 1.0 --saveData data/sim_N200.dat

    # Analyze from saved data (fast, no simulation):
    python analyzeGroupRescue.py --mode analyze --loadData data/sim_N200.dat
"""

import argparse
import math
import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# --------------- imports from runGroupRescue (utility functions) -----------
from runGroupRescue import (
    GroupRescueSimulation,
    build_vonneumann_adjacency,
    compute_vmem_similarity,
    derive_grid_dims,
    load_stress_params,
    get_default_stress_params,
    run_reference_sim,
)

# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser(
    description='Algebraic connectivity analysis of group rescue simulations'
)
parser.add_argument('--groupSize', type=int, default=25)
parser.add_argument('--gridDims', type=str, default=None)
parser.add_argument('--alpha', type=float, default=10.0)
parser.add_argument('--numBioSteps', type=int, default=2000)
parser.add_argument('--numStressSteps', type=int, default=500)
parser.add_argument('--stressParamsFile', type=str, default=None)
parser.add_argument('--D_F', type=float, default=0.5)
parser.add_argument('--gamma_F', type=float, default=0.0001)
parser.add_argument('--diffusion_substeps', type=int, default=10)
parser.add_argument('--initialStress', type=float, default=0.0)
parser.add_argument('--dampingGaussian', type=str, default=None)
parser.add_argument('--dampingRange', type=str, default=None)
parser.add_argument('--dampingCenter', type=float, default=None)
parser.add_argument('--dampingLevels', type=str, default=None)
parser.add_argument('--neighborhood', type=str, default='vonNeumann',
                    choices=['vonNeumann', 'moore'])
parser.add_argument('--rescueThreshold', type=float, default=0.5)
parser.add_argument('--outputFile', type=str, default=None)
parser.add_argument('--vmemSampleInterval', type=int, default=10,
                    help='Record Vmem every N bio-steps for correlation (default: 10)')
parser.add_argument('--numNullShuffles', type=int, default=50,
                    help='Number of null-model shuffles (default: 50)')
parser.add_argument('--mode', type=str, default='both',
                    choices=['simulate', 'analyze', 'both'],
                    help='simulate=run+save, analyze=load+viz, both=all (default: both)')
parser.add_argument('--saveData', type=str, default=None,
                    help='Path to save simulation data (.dat)')
parser.add_argument('--loadData', type=str, default=None,
                    help='Path to load saved simulation data (.dat)')

# ============================================================
# Damping map (simplified from runGroupRescue -- avoids argparse conflict)
# ============================================================

def build_damping_map_from_args(args, rows, cols):
    """Build (rows, cols) damping map from parsed args."""
    if args.dampingGaussian is not None:
        mean_d, std_d = [float(x) for x in args.dampingGaussian.split(',')]
        dmap = np.random.normal(mean_d, std_d, (rows, cols))
        return np.clip(dmap, 0.01, 1.0)
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
# Algebraic connectivity computation
# ============================================================

def build_functional_connectivity(vmem_history):
    """
    Build a functional connectivity matrix from Vmem timeseries.

    Args:
        vmem_history: (T, N, C) array -- T timepoints, N embryos, C cells per embryo

    Returns:
        W: (N, N) correlation-based weight matrix (non-negative)
    """
    T, N, C = vmem_history.shape
    mean_vmem = vmem_history.mean(axis=2)  # (T, N) -- mean voltage per embryo per time

    # Pearson correlation matrix across embryos
    centered = mean_vmem - mean_vmem.mean(axis=0, keepdims=True)  # (T, N)
    norms = np.sqrt((centered ** 2).sum(axis=0, keepdims=True))   # (1, N)
    norms = np.maximum(norms, 1e-12)
    normed = centered / norms  # (T, N)
    corr = normed.T @ normed   # (N, N)

    # Clamp to non-negative (negative correlations -> 0 weight)
    W = np.maximum(corr, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def build_full_functional_connectivity(vmem_history):
    """
    Build functional connectivity from full spatiotemporal Vmem timeseries
    (no dimensionality reduction).

    For each embryo, flatten all (T, C) into a single T*C vector, then
    compute pairwise Pearson correlation.  This preserves spatial structure
    unlike the mean-Vmem method.

    Args:
        vmem_history: (T, N, C) array -- T timepoints, N embryos, C cells

    Returns:
        W: (N, N) non-negative weight matrix
    """
    T, N, C = vmem_history.shape
    # Reshape to (N, T*C) -- each embryo is a single long vector
    flat = vmem_history.transpose(1, 0, 2).reshape(N, T * C)  # (N, T*C)

    # Pearson correlation
    centered = flat - flat.mean(axis=1, keepdims=True)
    norms = np.sqrt((centered ** 2).sum(axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-12)
    normed = centered / norms
    corr = normed @ normed.T  # (N, N)

    W = np.maximum(corr, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def build_reference_functional_connectivity(vmem_history, vmem_ref):
    """
    Build functional connectivity from similarity-to-reference timeseries.

    For each embryo at each snapshot, compute Pearson correlation to the
    healthy reference Vmem.  Then build inter-embryo connectivity from
    correlations of these "rescue trajectory" timeseries.

    This measures whether embryos *rescue together* rather than merely
    whether they are synchronized (which can also mean synchronized failure).

    Args:
        vmem_history: (T, N, C) array
        vmem_ref: (C,) reference Vmem pattern (numpy)

    Returns:
        W: (N, N) non-negative weight matrix
        sim_timeseries: (T, N) per-embryo similarity to reference over time
    """
    T, N, C = vmem_history.shape
    ref = vmem_ref.flatten()
    ref_std = ref.std()

    # Compute per-embryo, per-snapshot similarity to reference
    sim_ts = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        for emb in range(N):
            v = vmem_history[t, emb, :]
            if np.std(v) < 1e-10 or ref_std < 1e-10:
                sim_ts[t, emb] = 0.0
            else:
                sim_ts[t, emb] = np.corrcoef(v, ref)[0, 1]

    # Pearson correlation of rescue trajectories across time
    centered = sim_ts - sim_ts.mean(axis=0, keepdims=True)
    norms = np.sqrt((centered ** 2).sum(axis=0, keepdims=True))
    norms = np.maximum(norms, 1e-12)
    normed = centered / norms
    corr = normed.T @ normed  # (N, N)

    W = np.maximum(corr, 0.0)
    np.fill_diagonal(W, 0.0)
    return W, sim_ts


def compute_algebraic_connectivity(W):
    """
    Compute the Fiedler eigenvalue (lambda_2) and eigenvector from a
    weighted adjacency matrix W.

    Returns:
        lambda_2: second-smallest eigenvalue of the graph Laplacian
        fiedler_vec: corresponding eigenvector
    """
    N = W.shape[0]
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals, eigvecs = linalg.eigh(L)
    # eigenvalues are sorted ascending; lambda_0 ~ 0, lambda_1 = Fiedler value
    lambda_2 = eigvals[1]
    fiedler_vec = eigvecs[:, 1]
    return lambda_2, fiedler_vec


def compute_null_algebraic_connectivity(vmem_history, num_shuffles=50,
                                        vmem_ref=None, method='inter'):
    """
    Compute null-model algebraic connectivity by shuffling embryo timeseries.

    For each shuffle, independently permute each embryo's temporal order,
    breaking inter-embryo correlations while preserving per-embryo statistics.

    Args:
        vmem_history: (T, N, C) array
        num_shuffles: number of random permutations
        vmem_ref: if provided and method='reference', use reference-based connectivity
        method: 'inter' (mean Vmem), 'full' (spatiotemporal), or 'reference'

    Returns:
        mean_lambda2_null, std_lambda2_null
    """
    T, N, C = vmem_history.shape
    null_lambda2s = []

    for _ in range(num_shuffles):
        shuffled = vmem_history.copy()
        for emb in range(N):
            perm = np.random.permutation(T)
            shuffled[:, emb, :] = vmem_history[perm, emb, :]
        if method == 'reference' and vmem_ref is not None:
            W_null, _ = build_reference_functional_connectivity(shuffled, vmem_ref)
        elif method == 'full':
            W_null = build_full_functional_connectivity(shuffled)
        else:
            W_null = build_functional_connectivity(shuffled)
        lam2, _ = compute_algebraic_connectivity(W_null)
        null_lambda2s.append(lam2)

    return np.mean(null_lambda2s), np.std(null_lambda2s)


# ============================================================
# Run simulation with Vmem recording
# ============================================================

def run_with_vmem_recording(sim, num_bio_steps, num_stress_equil_steps,
                            vmem_ref, sample_interval):
    """
    Run GroupRescueSimulation while recording Vmem snapshots at intervals.

    Returns:
        results: dict from sim.run() (patched to include vmem_history)
        vmem_history: (T, N, C) numpy array of Vmem snapshots
    """
    from concurrent.futures import ThreadPoolExecutor, wait

    dt_ca = 0.01
    dt_stress = 0.1
    total_stress_steps = num_bio_steps + num_stress_equil_steps
    N = sim.num_embryos

    # Figure out number of cells from first embryo
    num_cells = sim.embryos[0]['bio_model'].electricNetwork.Vmem.shape[1]

    # Pre-allocate storage
    num_snapshots = num_bio_steps // sample_interval
    vmem_history = np.zeros((num_snapshots, N, num_cells), dtype=np.float32)
    stress_history = np.zeros((total_stress_steps, N))
    damping_history = np.zeros((num_bio_steps, N))
    field_history = np.zeros((total_stress_steps, N))
    similarity_history = np.zeros((num_bio_steps, N)) if vmem_ref is not None else None

    use_threads = sim.use_parallel
    max_workers = min(8, N) if use_threads else 1
    if use_threads:
        print(f"Using ThreadPoolExecutor with {max_workers} workers")

    t_start = time.time()
    snap_idx = 0

    # ---- Bioelectric + Stress concurrent phase ----
    for t in range(num_bio_steps):
        # Step 1: All embryos advance bioelectric sim by 1 step
        if use_threads:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(sim._step_single_embryo_bio, idx, t)
                    for idx in range(N)
                ]
                wait(futures)
                for f in futures:
                    f.result()
        else:
            for idx in range(N):
                sim._step_single_embryo_bio(idx, t)

        # Step 2: Update stress
        for idx in range(N):
            vmem_flat = sim.embryos[idx]['bio_model'].electricNetwork.Vmem[0, :, 0]
            sim.embryos[idx]['stress_switch'].compute_ca_from_vmem(
                vmem_flat.to(dtype=torch.float32), dt_ca
            )
            sim.embryos[idx]['stress_switch'].step(dt_stress)

        # Step 3: Rescue signal and damping
        if sim.use_diffusive_field:
            sim._diffuse_stress_field()
            rescue_signal = sim.F
        else:
            rescue_signal = sim._compute_mean_neighbor_stress()

        from runGroupRescue import compute_effective_damping
        for idx in range(N):
            base_d = sim.damping_map_flat[idx]
            eff_d = compute_effective_damping(base_d, rescue_signal[idx], sim.alpha)
            sim._apply_effective_damping(idx, eff_d)
            damping_history[t, idx] = eff_d
            stress_history[t, idx] = (
                sim.embryos[idx]['stress_switch'].get_embryo_stress().item()
            )
            field_history[t, idx] = rescue_signal[idx]

        # Track similarity
        if vmem_ref is not None:
            for idx in range(N):
                vmem_now = sim.embryos[idx]['bio_model'].electricNetwork.Vmem[0, :, 0].detach().cpu()
                similarity_history[t, idx] = compute_vmem_similarity(vmem_now, vmem_ref)

        # Record Vmem snapshot
        if t % sample_interval == 0 and snap_idx < num_snapshots:
            for idx in range(N):
                vmem_history[snap_idx, idx, :] = (
                    sim.embryos[idx]['bio_model'].electricNetwork.Vmem[0, :, 0]
                    .detach().cpu().numpy()
                )
            snap_idx += 1

        # Progress
        if (t + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (t + 1) / elapsed
            eta = (num_bio_steps - t - 1) / rate if rate > 0 else 0
            print(f"  Step {t+1}/{num_bio_steps}: "
                  f"mean_stress={stress_history[t].mean():.4f}, "
                  f"mean_eff_damping={damping_history[t].mean():.4f} "
                  f"[{rate:.1f} steps/s, ETA {eta:.0f}s]")

    # ---- Stress equilibration phase ----
    print(f"  Stress equilibration ({num_stress_equil_steps} steps)...")
    final_emission = np.array([
        sim.embryos[idx]['stress_switch'].get_embryo_stress().item()
        for idx in range(N)
    ])
    for t_eq in range(num_stress_equil_steps):
        for idx in range(N):
            ca_final = sim.embryos[idx]['stress_switch'].Ca.detach().clone()
            sim.embryos[idx]['stress_switch'].step(dt_stress, Ca=ca_final)
            stress_history[num_bio_steps + t_eq, idx] = (
                sim.embryos[idx]['stress_switch'].get_embryo_stress().item()
            )
        if sim.use_diffusive_field:
            sim._diffuse_stress_field_with_emission(final_emission)
            field_history[num_bio_steps + t_eq] = sim.F
        else:
            field_history[num_bio_steps + t_eq] = sim._compute_mean_neighbor_stress()

    # Collect final Vmem
    final_vmem = []
    for idx in range(N):
        final_vmem.append(
            sim.embryos[idx]['bio_model'].electricNetwork.Vmem[0, :, 0].detach().cpu()
        )

    elapsed_total = time.time() - t_start
    print(f"  Total simulation time: {elapsed_total:.1f}s")

    results = {
        'stress_history': stress_history,
        'damping_history': damping_history,
        'field_history': field_history,
        'final_vmem': final_vmem,
        'final_stress': stress_history[-1],
        'final_field': field_history[-1],
        'similarity_history': similarity_history,
        'num_bio_steps': num_bio_steps,
    }

    # Trim to actual recorded snapshots
    vmem_history = vmem_history[:snap_idx]

    return results, vmem_history


# ============================================================
# Visualization
# ============================================================

def visualize_connectivity_analysis(
        W_inter, W_full, W_ref, sim_ts,
        lam2_inter, fiedler_inter, lam2_inter_over_N,
        lam2_inter_null_mean, lam2_inter_null_std, lam2_inter_ratio,
        lam2_full, fiedler_full, lam2_full_over_N,
        lam2_full_null_mean, lam2_full_null_std, lam2_full_ratio,
        lam2_ref, fiedler_ref, lam2_ref_over_N,
        lam2_ref_null_mean, lam2_ref_null_std, lam2_ref_ratio,
        damping_map, results, vmem_ref,
        grid_rows, grid_cols, output_path, rescue_threshold=0.5):
    """
    Multi-panel figure comparing three connectivity methods.

    Row 1: Inter-embryo W | Inter-embryo Fiedler | Damping map
    Row 2: Full 121D W    | Full Fiedler          | (empty)
    Row 3: Reference W    | Reference Fiedler     | Final Vmem similarity
    Row 4: Stress ts      | Rescue trajectory ts  | Metrics summary
    """
    N = grid_rows * grid_cols
    damping_flat = damping_map.flatten()
    num_bio_steps = results['num_bio_steps']

    fig = plt.figure(figsize=(16, 18))

    # --- Row 1: Inter-embryo connectivity (mean Vmem) ---
    ax1 = fig.add_subplot(4, 3, 1)
    im1 = ax1.imshow(W_inter, cmap='hot', aspect='equal')
    ax1.set_title(f'Mean-Vmem W  (lam2={lam2_inter:.2f})')
    ax1.set_xlabel('Embryo')
    ax1.set_ylabel('Embryo')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(4, 3, 2)
    im2 = ax2.imshow(fiedler_inter.reshape(grid_rows, grid_cols),
                     cmap='RdBu_r', aspect='equal')
    ax2.set_title(f'Mean-Vmem Fiedler  (lam2/N={lam2_inter_over_N:.4f})')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(4, 3, 3)
    im3 = ax3.imshow(damping_map, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
    ax3.set_title('Base damping map')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # --- Row 2: Full spatiotemporal connectivity (121D) ---
    ax4 = fig.add_subplot(4, 3, 4)
    im4 = ax4.imshow(W_full, cmap='hot', aspect='equal')
    ax4.set_title(f'Full 121D W  (lam2={lam2_full:.2f})')
    ax4.set_xlabel('Embryo')
    ax4.set_ylabel('Embryo')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    ax5 = fig.add_subplot(4, 3, 5)
    im5 = ax5.imshow(fiedler_full.reshape(grid_rows, grid_cols),
                     cmap='RdBu_r', aspect='equal')
    ax5.set_title(f'Full 121D Fiedler  (lam2/N={lam2_full_over_N:.4f})')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # Empty panel for row 2, col 3 -- or show W_full vs W_inter difference
    ax6_empty = fig.add_subplot(4, 3, 6)
    ax6_empty.axis('off')

    # --- Row 3: Reference-based connectivity ---
    ax7 = fig.add_subplot(4, 3, 7)
    im7 = ax7.imshow(W_ref, cmap='hot', aspect='equal')
    ax7.set_title(f'Reference W  (lam2={lam2_ref:.2f})')
    ax7.set_xlabel('Embryo')
    ax7.set_ylabel('Embryo')
    plt.colorbar(im7, ax=ax7, fraction=0.046)

    ax8 = fig.add_subplot(4, 3, 8)
    im8 = ax8.imshow(fiedler_ref.reshape(grid_rows, grid_cols),
                     cmap='RdBu_r', aspect='equal')
    ax8.set_title(f'Reference Fiedler  (lam2/N={lam2_ref_over_N:.4f})')
    plt.colorbar(im8, ax=ax8, fraction=0.046)

    # Final Vmem similarity heatmap
    ax9 = fig.add_subplot(4, 3, 9)
    if results['similarity_history'] is not None:
        final_sims = results['similarity_history'][-1]
        rescue_rate = (final_sims > rescue_threshold).mean()
    else:
        final_sims = np.zeros(N)
        rescue_rate = np.nan
    if vmem_ref is not None:
        sim_grid = np.zeros(N)
        for idx in range(N):
            sim_grid[idx] = compute_vmem_similarity(results['final_vmem'][idx], vmem_ref)
        im9 = ax9.imshow(sim_grid.reshape(grid_rows, grid_cols), cmap='RdYlGn',
                         vmin=-0.5, vmax=1.0, aspect='equal')
        ax9.set_title(f'Final Vmem sim (rescue={rescue_rate:.0%})')
        plt.colorbar(im9, ax=ax9, fraction=0.046)
    else:
        ax9.set_title('Final Vmem sim (N/A)')

    # --- Row 4: Timeseries + metrics ---
    # Stress timeseries
    ax10 = fig.add_subplot(4, 3, 10)
    unique_dampings = sorted(set(np.round(damping_flat, 3)))
    if len(unique_dampings) > 10:
        quartiles = np.percentile(damping_flat, [0, 25, 50, 75, 100])
        labels = [f'd<{quartiles[1]:.2f}', f'd<{quartiles[2]:.2f}',
                  f'd<{quartiles[3]:.2f}', f'd>{quartiles[3]:.2f}']
        masks = [
            damping_flat < quartiles[1],
            (damping_flat >= quartiles[1]) & (damping_flat < quartiles[2]),
            (damping_flat >= quartiles[2]) & (damping_flat < quartiles[3]),
            damping_flat >= quartiles[3],
        ]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, 4))
        for i, (mask, label) in enumerate(zip(masks, labels)):
            if mask.any():
                ax10.plot(results['stress_history'][:, mask].mean(axis=1),
                          color=colors[i], label=label, alpha=0.8)
    else:
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(unique_dampings)))
        for i, d in enumerate(unique_dampings):
            mask = np.abs(damping_flat - d) < 1e-4
            ax10.plot(results['stress_history'][:, mask].mean(axis=1),
                      color=colors[i], label=f'd={d:.2f}', alpha=0.8)
    ax10.set_title('Mean stress')
    ax10.set_xlabel('Step')
    ax10.set_ylabel('Stress')
    ax10.legend(fontsize=7, loc='best')
    ax10.axvline(num_bio_steps, color='gray', linestyle='--', alpha=0.5)

    # Rescue trajectory timeseries (from sim_ts)
    ax11 = fig.add_subplot(4, 3, 11)
    T_snap = sim_ts.shape[0]
    mean_sim_ts = sim_ts.mean(axis=1)
    std_sim_ts = sim_ts.std(axis=1)
    snap_times = np.arange(T_snap)
    ax11.plot(snap_times, mean_sim_ts, color='steelblue', linewidth=2,
              label='Mean sim-to-ref')
    ax11.fill_between(snap_times, mean_sim_ts - std_sim_ts,
                      mean_sim_ts + std_sim_ts, color='steelblue', alpha=0.2)
    n_show = min(5, N)
    for emb in np.linspace(0, N - 1, n_show, dtype=int):
        ax11.plot(snap_times, sim_ts[:, emb], alpha=0.3, linewidth=0.8)
    ax11.set_title('Rescue trajectory (sim-to-ref)')
    ax11.set_xlabel('Snapshot index')
    ax11.set_ylabel('Pearson r to ref')
    ax11.legend(fontsize=7, loc='best')

    # Metrics text box
    ax12 = fig.add_subplot(4, 3, 12)
    ax12.axis('off')
    metrics_text = (
        f"Algebraic Connectivity Analysis\n"
        f"{'='*42}\n"
        f"N = {N} embryos ({grid_rows}x{grid_cols})\n\n"
        f"MEAN-VMEM (1D projection):\n"
        f"  lam2/N = {lam2_inter_over_N:.6f}\n"
        f"  lam2/null = {lam2_inter_ratio:.2f}\n\n"
        f"FULL 121D (no reduction):\n"
        f"  lam2/N = {lam2_full_over_N:.6f}\n"
        f"  lam2/null = {lam2_full_ratio:.2f}\n\n"
        f"REFERENCE (rescue traj.):\n"
        f"  lam2/N = {lam2_ref_over_N:.6f}\n"
        f"  lam2/null = {lam2_ref_ratio:.2f}\n\n"
        f"Rescue rate = {rescue_rate:.1%}\n"
        f"Mean final stress = {results['final_stress'].mean():.4f}"
    )
    ax12.text(0.05, 0.95, metrics_text, transform=ax12.transAxes,
              fontsize=9, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(
        f'Algebraic Connectivity: N={N} | '
        f'Mean: lam2/N={lam2_inter_over_N:.4f} | '
        f'Full: lam2/N={lam2_full_over_N:.4f} | '
        f'Ref: lam2/N={lam2_ref_over_N:.4f}',
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved analysis figure to {output_path}")
    plt.close(fig)


def visualize_fiedler_network(W_ref, fiedler_ref, damping_map,
                              grid_rows, grid_cols, output_path,
                              vmem_ref=None, results=None,
                              rescue_threshold=0.5):
    """
    Network graph visualization of the Fiedler partition on the embryo grid.

    Left:  Nodes colored by Fiedler vector, edges by weight.
    Right: Same layout but nodes colored by final Vmem similarity.

    Edge thickness and opacity encode functional connectivity weight.
    Cross-cluster edges (connecting opposite Fiedler signs) drawn dashed.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize, TwoSlopeNorm

    N = grid_rows * grid_cols

    # Node positions: grid layout (col = x, row = y inverted so row 0 is top)
    pos = np.zeros((N, 2))
    for i in range(grid_rows):
        for j in range(grid_cols):
            idx = i * grid_cols + j
            pos[idx] = [j, grid_rows - 1 - i]

    # Fiedler sign partition
    fiedler_sign = np.sign(fiedler_ref)
    fiedler_sign[fiedler_sign == 0] = 1  # break ties

    # Build edge list from W_ref (upper triangle only)
    edges = []
    weights = []
    for i in range(N):
        for j in range(i + 1, N):
            if W_ref[i, j] > 1e-6:
                edges.append((i, j))
                weights.append(W_ref[i, j])
    edges = np.array(edges) if edges else np.empty((0, 2), dtype=int)
    weights = np.array(weights)

    # For large N, only draw the strongest edges to keep it readable
    if len(weights) > 0:
        # Keep top fraction of edges (adaptive to N)
        max_edges = min(len(weights), N * 4)
        if len(weights) > max_edges:
            threshold = np.sort(weights)[-max_edges]
            mask = weights >= threshold
            edges = edges[mask]
            weights = weights[mask]

    # Normalize edge weights for drawing
    if len(weights) > 0:
        w_norm = weights / weights.max()
    else:
        w_norm = np.array([])

    # Classify edges: same-cluster vs cross-cluster
    same_cluster = []
    cross_cluster = []
    same_w = []
    cross_w = []
    for k, (i, j) in enumerate(edges):
        seg = [pos[i], pos[j]]
        if fiedler_sign[i] == fiedler_sign[j]:
            same_cluster.append(seg)
            same_w.append(w_norm[k])
        else:
            cross_cluster.append(seg)
            cross_w.append(w_norm[k])

    # --- Figure with 2 panels ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # ---- Panel 1: Fiedler-colored network ----
    ax = axes[0]

    # Draw cross-cluster edges (dashed, gray)
    if cross_cluster:
        cross_w_arr = np.array(cross_w)
        lc_cross = LineCollection(
            cross_cluster,
            linewidths=0.3 + 2.0 * cross_w_arr,
            colors=[(0.6, 0.6, 0.6, 0.15 + 0.35 * a) for a in cross_w_arr],
            linestyles='dashed',
            zorder=1,
        )
        ax.add_collection(lc_cross)

    # Draw same-cluster edges (solid)
    if same_cluster:
        same_w_arr = np.array(same_w)
        # Color by which cluster: positive=warm, negative=cool
        same_colors = []
        for k, seg in enumerate(same_cluster):
            # Find which edge this corresponds to
            alpha = 0.2 + 0.6 * same_w_arr[k]
            # Determine cluster from first endpoint
            i_idx = np.argmin(np.sum((pos - seg[0]) ** 2, axis=1))
            if fiedler_sign[i_idx] > 0:
                same_colors.append((0.8, 0.2, 0.2, alpha))  # red cluster
            else:
                same_colors.append((0.2, 0.3, 0.8, alpha))  # blue cluster
        lc_same = LineCollection(
            same_cluster,
            linewidths=0.5 + 2.5 * same_w_arr,
            colors=same_colors,
            zorder=2,
        )
        ax.add_collection(lc_same)

    # Draw nodes colored by Fiedler value
    fmax = max(abs(fiedler_ref.min()), abs(fiedler_ref.max()))
    fnorm = TwoSlopeNorm(vmin=-fmax, vcenter=0, vmax=fmax)
    node_sizes = 80 if N <= 50 else max(15, 600 // grid_rows)
    sc = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=fiedler_ref, cmap='RdBu_r', norm=fnorm,
        s=node_sizes, edgecolors='black', linewidths=0.5,
        zorder=3,
    )
    plt.colorbar(sc, ax=ax, label='Fiedler value', fraction=0.046, pad=0.04)

    # Label cluster membership counts
    n_pos = (fiedler_sign > 0).sum()
    n_neg = (fiedler_sign < 0).sum()
    ax.set_title(
        f'Fiedler partition: {n_pos} interior (red) / {n_neg} boundary (blue)\n'
        f'Solid=within-cluster, Dashed=cross-cluster',
        fontsize=10,
    )
    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(-0.5, grid_rows - 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # ---- Panel 2: Same layout, colored by rescue outcome ----
    ax2 = axes[1]

    # Draw edges (all gray, same topology)
    if same_cluster or cross_cluster:
        all_segs = same_cluster + cross_cluster
        all_w = list(same_w) + list(cross_w)
        all_w_arr = np.array(all_w) if all_w else np.array([])
        if len(all_segs) > 0:
            lc_all = LineCollection(
                all_segs,
                linewidths=0.3 + 1.5 * all_w_arr,
                colors=[(0.5, 0.5, 0.5, 0.1 + 0.3 * a) for a in all_w_arr],
                zorder=1,
            )
            ax2.add_collection(lc_all)

    # Node color = final Vmem similarity
    if results is not None and vmem_ref is not None:
        sim_vals = np.zeros(N)
        for idx in range(N):
            sim_vals[idx] = compute_vmem_similarity(
                results['final_vmem'][idx], vmem_ref
            )
        rescue_mask = sim_vals > rescue_threshold
        rescue_rate = rescue_mask.mean()
    else:
        sim_vals = np.zeros(N)
        rescue_rate = 0.0

    sc2 = ax2.scatter(
        pos[:, 0], pos[:, 1],
        c=sim_vals, cmap='RdYlGn', vmin=-0.5, vmax=1.0,
        s=node_sizes, edgecolors='black', linewidths=0.5,
        zorder=3,
    )
    plt.colorbar(sc2, ax=ax2, label='Similarity to ref', fraction=0.046, pad=0.04)

    ax2.set_title(
        f'Rescue outcome (rate={rescue_rate:.0%})\n'
        f'Green=rescued, Red=failed',
        fontsize=10,
    )
    ax2.set_xlim(-0.5, grid_cols - 0.5)
    ax2.set_ylim(-0.5, grid_rows - 0.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')

    fig.suptitle(
        f'Fiedler Network Clustering: N={N} ({grid_rows}x{grid_cols})',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    net_path = output_path.replace('.png', '_network.png')
    fig.savefig(net_path, dpi=150, bbox_inches='tight')
    print(f"Saved network figure to {net_path}")
    plt.close(fig)


# ============================================================
# Save / Load simulation data
# ============================================================

def save_simulation_data(path, results, vmem_history, vmem_ref,
                         damping_map, grid_rows, grid_cols, sim_params):
    """
    Save all simulation data needed for analysis and visualization.

    Converts torch tensors in results['final_vmem'] to numpy so the
    saved file has no torch dependency for loading.
    """
    # Convert final_vmem list of tensors to numpy
    final_vmem_np = []
    for v in results['final_vmem']:
        if hasattr(v, 'numpy'):
            final_vmem_np.append(v.detach().cpu().numpy())
        else:
            final_vmem_np.append(np.asarray(v))

    vmem_ref_np = vmem_ref.numpy() if hasattr(vmem_ref, 'numpy') else np.asarray(vmem_ref)

    save_dict = {
        # Simulation results (numpy arrays)
        'stress_history': results['stress_history'],
        'damping_history': results['damping_history'],
        'field_history': results['field_history'],
        'final_vmem': final_vmem_np,
        'final_stress': results['final_stress'],
        'final_field': results['final_field'],
        'similarity_history': results['similarity_history'],
        'num_bio_steps': results['num_bio_steps'],
        # Vmem timeseries for connectivity analysis
        'vmem_history': vmem_history,
        # Reference and layout
        'vmem_ref': vmem_ref_np,
        'damping_map': damping_map,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        # Simulation parameters (for reproducibility)
        'sim_params': sim_params,
    }
    torch.save(save_dict, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Saved simulation data to {path} ({size_mb:.1f} MB)")


def load_simulation_data(path):
    """
    Load saved simulation data.

    Returns:
        results: dict compatible with visualization functions
        vmem_history: (T, N, C) numpy array
        vmem_ref: (C,) numpy array
        damping_map: (rows, cols) numpy array
        grid_rows, grid_cols: int
        sim_params: dict
    """
    print(f"Loading simulation data from {path}...")
    data = torch.load(path, weights_only=False)

    results = {
        'stress_history': data['stress_history'],
        'damping_history': data['damping_history'],
        'field_history': data['field_history'],
        'final_vmem': data['final_vmem'],  # list of numpy arrays
        'final_stress': data['final_stress'],
        'final_field': data['final_field'],
        'similarity_history': data['similarity_history'],
        'num_bio_steps': data['num_bio_steps'],
    }

    grid_rows = int(data['grid_rows'])
    grid_cols = int(data['grid_cols'])
    N = grid_rows * grid_cols
    print(f"  Grid: {grid_rows}x{grid_cols} = {N} embryos")
    print(f"  Vmem history: {data['vmem_history'].shape}")
    print(f"  Sim params: {data.get('sim_params', {})}")

    return (results, data['vmem_history'], data['vmem_ref'],
            data['damping_map'], grid_rows, grid_cols,
            data.get('sim_params', {}))


# ============================================================
# Main
# ============================================================

def run_simulation(args):
    """Run the group rescue simulation and return all data."""
    # Derive grid dims
    if args.gridDims:
        grid_rows, grid_cols = [int(x) for x in args.gridDims.split(',')]
    else:
        grid_rows, grid_cols = derive_grid_dims(args.groupSize)
    N = grid_rows * grid_cols

    print(f"\n{'='*60}")
    print(f"Simulation Phase")
    print(f"  Group size: {N} ({grid_rows}x{grid_cols})")
    print(f"  Alpha: {args.alpha}, D_F: {args.D_F}, gamma_F: {args.gamma_F}")
    print(f"  Bio steps: {args.numBioSteps}, Vmem sample interval: {args.vmemSampleInterval}")
    print(f"{'='*60}\n")

    # Build damping map
    np.random.seed(42)
    damping_map = build_damping_map_from_args(args, grid_rows, grid_cols)
    print(f"Damping map: mean={damping_map.mean():.3f}, "
          f"std={damping_map.std():.4f}, "
          f"range=[{damping_map.min():.3f}, {damping_map.max():.3f}]")

    # Load stress params
    if args.stressParamsFile:
        stress_params, ca_params = load_stress_params(args.stressParamsFile)
    else:
        stress_params, ca_params = get_default_stress_params()

    # Run reference embryo
    vmem_ref, cell_grid_size = run_reference_sim(args.numBioSteps)

    # Create and run simulation
    sim = GroupRescueSimulation(
        grid_rows=grid_rows, grid_cols=grid_cols,
        damping_map=damping_map, alpha=args.alpha,
        stress_params=stress_params, ca_params=ca_params,
        neighborhood=args.neighborhood,
        D_F=args.D_F, gamma_F=args.gamma_F,
        diffusion_substeps=args.diffusion_substeps,
        initial_stress=args.initialStress,
    )

    print(f"\nRunning simulation with Vmem recording (interval={args.vmemSampleInterval})...")
    results, vmem_history = run_with_vmem_recording(
        sim, args.numBioSteps, args.numStressSteps,
        vmem_ref, args.vmemSampleInterval,
    )
    print(f"Vmem history shape: {vmem_history.shape} (snapshots, embryos, cells)")

    # Store simulation parameters for reproducibility
    sim_params = {
        'groupSize': N, 'grid_rows': grid_rows, 'grid_cols': grid_cols,
        'alpha': args.alpha, 'D_F': args.D_F, 'gamma_F': args.gamma_F,
        'numBioSteps': args.numBioSteps, 'numStressSteps': args.numStressSteps,
        'neighborhood': args.neighborhood, 'initialStress': args.initialStress,
        'vmemSampleInterval': args.vmemSampleInterval,
        'stressParamsFile': args.stressParamsFile,
    }

    # Save if requested
    save_path = args.saveData
    if save_path is None and args.mode == 'simulate':
        save_path = f"data/sim_N{N}.dat"
    if save_path is None and args.mode == 'both':
        save_path = f"data/sim_N{N}.dat"
    if save_path is not None:
        save_simulation_data(save_path, results, vmem_history, vmem_ref,
                             damping_map, grid_rows, grid_cols, sim_params)

    return results, vmem_history, vmem_ref, damping_map, grid_rows, grid_cols


def run_analysis(results, vmem_history, vmem_ref, damping_map,
                 grid_rows, grid_cols, args):
    """Compute algebraic connectivity metrics and visualize."""
    N = grid_rows * grid_cols

    vmem_ref_np = vmem_ref.numpy() if hasattr(vmem_ref, 'numpy') else np.asarray(vmem_ref)

    print(f"\n{'='*60}")
    print(f"Analysis Phase  (N = {N})")
    print(f"  Null shuffles: {args.numNullShuffles}")
    print(f"{'='*60}")

    # ---- Inter-embryo connectivity (mean Vmem) ----
    print(f"\n--- Inter-embryo connectivity (mean Vmem correlation) ---")
    W_inter = build_functional_connectivity(vmem_history)
    lam2_inter, fiedler_inter = compute_algebraic_connectivity(W_inter)
    lam2_inter_over_N = lam2_inter / N

    print(f"  Raw lambda_2 = {lam2_inter:.6f}")
    print(f"  lambda_2 / N = {lam2_inter_over_N:.8f}")

    print(f"  Null model ({args.numNullShuffles} shuffles)...")
    lam2_inter_null_mean, lam2_inter_null_std = compute_null_algebraic_connectivity(
        vmem_history, num_shuffles=args.numNullShuffles, method='inter',
    )
    lam2_inter_ratio = lam2_inter / lam2_inter_null_mean if lam2_inter_null_mean > 1e-12 else float('inf')
    print(f"  lambda_2_null = {lam2_inter_null_mean:.6f} +/- {lam2_inter_null_std:.6f}")
    print(f"  lambda_2 / lambda_2_null = {lam2_inter_ratio:.4f}")

    # ---- Full spatiotemporal connectivity (121D, no reduction) ----
    print(f"\n--- Full spatiotemporal connectivity (121D Vmem correlation) ---")
    W_full = build_full_functional_connectivity(vmem_history)
    lam2_full, fiedler_full = compute_algebraic_connectivity(W_full)
    lam2_full_over_N = lam2_full / N

    print(f"  Raw lambda_2 = {lam2_full:.6f}")
    print(f"  lambda_2 / N = {lam2_full_over_N:.8f}")

    print(f"  Null model ({args.numNullShuffles} shuffles)...")
    lam2_full_null_mean, lam2_full_null_std = compute_null_algebraic_connectivity(
        vmem_history, num_shuffles=args.numNullShuffles, method='full',
    )
    lam2_full_ratio = lam2_full / lam2_full_null_mean if lam2_full_null_mean > 1e-12 else float('inf')
    print(f"  lambda_2_null = {lam2_full_null_mean:.6f} +/- {lam2_full_null_std:.6f}")
    print(f"  lambda_2 / lambda_2_null = {lam2_full_ratio:.4f}")

    # ---- Reference-based connectivity ----
    print(f"\n--- Reference-based connectivity (rescue trajectory correlation) ---")
    W_ref, sim_ts = build_reference_functional_connectivity(vmem_history, vmem_ref_np)
    lam2_ref, fiedler_ref = compute_algebraic_connectivity(W_ref)
    lam2_ref_over_N = lam2_ref / N

    print(f"  Raw lambda_2 = {lam2_ref:.6f}")
    print(f"  lambda_2 / N = {lam2_ref_over_N:.8f}")

    print(f"  Null model ({args.numNullShuffles} shuffles)...")
    lam2_ref_null_mean, lam2_ref_null_std = compute_null_algebraic_connectivity(
        vmem_history, num_shuffles=args.numNullShuffles, vmem_ref=vmem_ref_np,
        method='reference',
    )
    lam2_ref_ratio = lam2_ref / lam2_ref_null_mean if lam2_ref_null_mean > 1e-12 else float('inf')
    print(f"  lambda_2_null = {lam2_ref_null_mean:.6f} +/- {lam2_ref_null_std:.6f}")
    print(f"  lambda_2 / lambda_2_null = {lam2_ref_ratio:.4f}")

    # ---- Summary ----
    if results['similarity_history'] is not None:
        final_sims = results['similarity_history'][-1]
        rescue_rate = (final_sims > args.rescueThreshold).mean()
    else:
        rescue_rate = np.nan

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY  (N = {N})")
    print(f"{'='*60}")
    print(f"  Inter-embryo (mean Vmem correlation):")
    print(f"    Raw lambda_2:          {lam2_inter:.6f}")
    print(f"    Norm 1 (lambda_2/N):   {lam2_inter_over_N:.8f}")
    print(f"    Null lambda_2:         {lam2_inter_null_mean:.6f} +/- {lam2_inter_null_std:.6f}")
    print(f"    Norm 2 (ratio):        {lam2_inter_ratio:.4f}")
    print(f"  Full spatiotemporal (121D Vmem):")
    print(f"    Raw lambda_2:          {lam2_full:.6f}")
    print(f"    Norm 1 (lambda_2/N):   {lam2_full_over_N:.8f}")
    print(f"    Null lambda_2:         {lam2_full_null_mean:.6f} +/- {lam2_full_null_std:.6f}")
    print(f"    Norm 2 (ratio):        {lam2_full_ratio:.4f}")
    print(f"  Reference-based (rescue trajectory):")
    print(f"    Raw lambda_2:          {lam2_ref:.6f}")
    print(f"    Norm 1 (lambda_2/N):   {lam2_ref_over_N:.8f}")
    print(f"    Null lambda_2:         {lam2_ref_null_mean:.6f} +/- {lam2_ref_null_std:.6f}")
    print(f"    Norm 2 (ratio):        {lam2_ref_ratio:.4f}")
    if not np.isnan(rescue_rate):
        print(f"  Rescue rate:             {rescue_rate:.1%}")
        print(f"  Mean final similarity:   {final_sims.mean():.4f}")
    print(f"  Mean final stress:       {results['final_stress'].mean():.4f}")
    print(f"{'='*60}\n")

    # ---- Visualization ----
    if args.outputFile:
        output_path = args.outputFile
    else:
        output_path = f"data/algebraic_connectivity_N{N}.png"

    visualize_connectivity_analysis(
        W_inter, W_full, W_ref, sim_ts,
        lam2_inter, fiedler_inter, lam2_inter_over_N,
        lam2_inter_null_mean, lam2_inter_null_std, lam2_inter_ratio,
        lam2_full, fiedler_full, lam2_full_over_N,
        lam2_full_null_mean, lam2_full_null_std, lam2_full_ratio,
        lam2_ref, fiedler_ref, lam2_ref_over_N,
        lam2_ref_null_mean, lam2_ref_null_std, lam2_ref_ratio,
        damping_map, results, vmem_ref_np,
        grid_rows, grid_cols, output_path,
        rescue_threshold=args.rescueThreshold,
    )

    visualize_fiedler_network(
        W_ref, fiedler_ref, damping_map,
        grid_rows, grid_cols, output_path,
        vmem_ref=vmem_ref_np, results=results,
        rescue_threshold=args.rescueThreshold,
    )

    return {
        'N': N,
        'inter_lambda_2': lam2_inter,
        'inter_lambda_2_over_N': lam2_inter_over_N,
        'inter_lambda_2_null_mean': lam2_inter_null_mean,
        'inter_lambda_2_ratio': lam2_inter_ratio,
        'full_lambda_2': lam2_full,
        'full_lambda_2_over_N': lam2_full_over_N,
        'full_lambda_2_null_mean': lam2_full_null_mean,
        'full_lambda_2_ratio': lam2_full_ratio,
        'ref_lambda_2': lam2_ref,
        'ref_lambda_2_over_N': lam2_ref_over_N,
        'ref_lambda_2_null_mean': lam2_ref_null_mean,
        'ref_lambda_2_ratio': lam2_ref_ratio,
    }


def main():
    args = parser.parse_args()

    if args.mode == 'analyze':
        # Load-only mode: skip simulation entirely
        if args.loadData is None:
            raise ValueError("--loadData is required when --mode=analyze")
        (results, vmem_history, vmem_ref, damping_map,
         grid_rows, grid_cols, sim_params) = load_simulation_data(args.loadData)
        return run_analysis(results, vmem_history, vmem_ref, damping_map,
                            grid_rows, grid_cols, args)

    elif args.mode == 'simulate':
        # Simulate-only mode: run and save, no analysis
        (results, vmem_history, vmem_ref,
         damping_map, grid_rows, grid_cols) = run_simulation(args)
        print("Simulation complete. Use --mode analyze --loadData <path> to visualize.")
        return None

    else:
        # Default: both simulate and analyze
        if args.loadData is not None:
            # User provided saved data -- skip simulation
            (results, vmem_history, vmem_ref, damping_map,
             grid_rows, grid_cols, sim_params) = load_simulation_data(args.loadData)
        else:
            (results, vmem_history, vmem_ref,
             damping_map, grid_rows, grid_cols) = run_simulation(args)
        return run_analysis(results, vmem_history, vmem_ref, damping_map,
                            grid_rows, grid_cols, args)


if __name__ == '__main__':
    main()
