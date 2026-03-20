# Positional information analysis of the Stigmergic bioelectric field model
#
# Tests whether the field patterning system generates unique positional
# identities for cells in an 11x11 tissue — WITHOUT any external clamping.
#
# The idea: if the emergent bioelectric dynamics give each cell a distinct
# temporal fingerprint (mean voltage, transition rate, distribution shape),
# then the field system is encoding 2D positional information.
#
# ── Per-cell dynamical metrics (compute_per_cell_metrics) ──
#   For each of the 121 cells (11x11 grid), computed from the full Vmem timeseries:
#
#   1. mean_vmem       Time-averaged membrane voltage. Cells with different mean
#                      voltages have distinguishable resting states — this is the
#                      most basic positional signal.
#
#   2. std_vmem        Voltage fluctuation magnitude. High std indicates an
#                      oscillating cell; low std indicates a steady-state cell.
#                      Captures dynamical complexity beyond the mean.
#
#   3. pol_fraction    Fraction of time the cell spends below a voltage threshold
#                      (default -40 mV), i.e. in the "polarized" state. Measures
#                      how often a cell is polarized vs depolarized. A cell at 0.5
#                      spends equal time in both states; 1.0 = always polarized.
#
#   4. transition_rate How frequently a cell switches between polarized and
#                      depolarized states, measured as state flips per timestep.
#                      High rate = rapidly switching cell; zero = locked in one state.
#                      Computed as mean(|diff(state)|) where state is binary.
#
#   5. entropy         Shannon entropy of the Vmem histogram (in bits). The cell's
#                      full voltage trace is binned into a histogram; entropy of
#                      that distribution measures dynamical complexity. High entropy
#                      = voltage spread across many bins (rich dynamics); low entropy
#                      = concentrated in few bins (simple/steady dynamics).
#
#   6. autocorr_tau    Temporal autocorrelation decay timescale — the lag (in
#                      iterations) at which the autocorrelation function drops below
#                      1/e. Short tau = fast-decorrelating signal (noisy or rapidly
#                      oscillatory); long tau = slowly-changing (persistent state).
#                      Capped at min(T/2, 500) if autocorrelation never decays.
#
# ── Positional uniqueness metrics (compute_positional_uniqueness) ──
#   These quantify whether cell dynamics encode spatial position. Because the
#   grid has two-fold reflection symmetry about both midlines, analysis is
#   restricted to the symmetry-independent upper-left quadrant (6×6 = 36 cells,
#   including the center row and column) to avoid counting mirror duplicates.
#
#   1. Pairwise KS distance  For every pair of quadrant cells, the two-sample
#                             Kolmogorov–Smirnov statistic compares their full
#                             Vmem distributions (all T timepoints). KS = 0 means
#                             identical distributions; KS = 1 means completely
#                             non-overlapping. A high mean KS across all pairs
#                             means cells are dynamically distinguishable.
#
#   2. Unique fingerprints   Each quadrant cell gets a fingerprint tuple:
#                             (mean_vmem, std_vmem, transition_rate), each rounded
#                             to 0.1 mV precision. The fraction of unique tuples
#                             out of 36 cells measures how many have distinct
#                             identities. 100% = every cell is unique.
#
#   3. Position decodability  Pearson correlation |r| between each dynamical metric
#                             and three spatial coordinates within the quadrant:
#                             row index, column index, and distance from center.
#                             High |r| means a cell's position is predictable from
#                             its dynamics alone. Also reports r2_total = r_row² +
#                             r_col² as an overall 2D decodability score.
#
#   4. Mutual information     Estimates MI(position; Vmem) by discretizing mean Vmem
#                             into quantile bins across quadrant cells, then computing
#                             Shannon entropy H(Vmem_bin). Compared against the
#                             theoretical max H_pos = log2(36) ≈ 5.17 bits (which
#                             would require every cell to be perfectly distinguishable).
#
#   5. Uniqueness over time   Tracks the unique fingerprint fraction at progressive
#                             time checkpoints (from t=100 up to full run length).
#                             Shows when positional identities emerge — whether they
#                             appear early or require long dynamics to differentiate.
#
# ── Interpretation ──
#   Strong positional encoding: >50% unique fingerprints, high mean KS, high MI,
#   and strong correlations between metrics and spatial coordinates.
#   This would mean the stigmergic field gives each cell a distinct temporal
#   "fingerprint" that encodes its 2D position without external clamping.
#
# Usage:
#   python positional_information_analysis.py                      # default (random G_pol, 5000 iters)
#   python positional_information_analysis.py --init_mode homogeneous
#   python positional_information_analysis.py --num_iters 2000     # quick test

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import ks_2samp, entropy as scipy_entropy
from itertools import chain

from embryo import model
import utilities


def load_and_run(parameters, num_iters, init_mode, seed=42):
    """Load stigmergic model, set up unclamped conditions, and run."""
    latticeDims = parameters['latticeDims']
    numRows, numCols = latticeDims
    numCells = numRows * numCols
    numSamples = parameters['simParameters']['numSamples']

    modelinstance = model(parameters, numSamples)
    initialValues = parameters['simParameters']['initialValues']

    if 'ligandConc' not in initialValues:
        initialValues['ligandConc'] = torch.zeros(
            (numSamples, numCells, 1), dtype=torch.float64)

    # Set up unclamped initial conditions
    torch.manual_seed(seed)
    np.random.seed(seed)

    if init_mode == 'homogeneous':
        # Uniform depolarized state + small noise to break symmetry
        initVmem = -9.2e-3  # depolarized equilibrium
        initialValues['Vmem'] = (
            torch.ones(numSamples, numCells, 1, dtype=torch.float64) * initVmem
            + torch.randn(numSamples, numCells, 1, dtype=torch.float64) * 1e-4
        )
        initialValues['eV'] = torch.zeros(
            (numSamples, modelinstance.electricNetwork.numFieldGridPoints, 1),
            dtype=torch.float64)
        initialValues['ligandConc'] = torch.ones(
            (numSamples, numCells, 1), dtype=torch.float64) * 0.5
        AllCells = list(range(numCells))
        initialValues['G_pol']['cells'] = [[AllCells]] * numSamples
        initialValues['G_pol']['values'] = [
            torch.DoubleTensor([1.0] * numCells)] * numSamples

    elif init_mode == 'random_gpol':
        # Homogeneous Vmem but randomized ion channel conductances
        initVmem = -9.2e-3
        initialValues['Vmem'] = (
            torch.ones(numSamples, numCells, 1, dtype=torch.float64) * initVmem
        )
        initialValues['eV'] = torch.zeros(
            (numSamples, modelinstance.electricNetwork.numFieldGridPoints, 1),
            dtype=torch.float64)
        initialValues['ligandConc'] = torch.ones(
            (numSamples, numCells, 1), dtype=torch.float64) * 0.5
        AllCells = list(range(numCells))
        initialValues['G_pol']['cells'] = [[AllCells]] * numSamples
        initialValues['G_pol']['values'] = [
            torch.rand(numCells, dtype=torch.float64) * 2
            for _ in range(numSamples)
        ]

    circuit = modelinstance.electricNetwork
    circuit.initVariables(initialValues)
    circuit.initParameters(initialValues)

    # Run unclamped simulation
    print(f"  Running {num_iters} iterations (init_mode={init_mode})...")
    externalInputs = parameters['simParameters']['externalInputs']
    modelinstance.simulate(
        externalInputs=externalInputs,
        clampParameters=None,
        perturbation=None,
        numSimIters=num_iters,
    )

    # Extract Vmem timeseries: (T, numCells)
    vmem_ts = modelinstance.timeseriesVmem[:, 0, :, 0].detach().numpy()
    return vmem_ts, latticeDims


def compute_per_cell_metrics(vmem_ts, threshold, num_bins):
    """
    Compute per-cell dynamical metrics from Vmem timeseries.

    Args:
        vmem_ts: (T, N) array of membrane voltages
        threshold: voltage threshold for polarized/depolarized classification (V)
        num_bins: number of bins for Vmem histogram / entropy

    Returns:
        dict of metric arrays, each shape (N,)
    """
    T, N = vmem_ts.shape

    # 1. Mean and Std
    mean_vmem = vmem_ts.mean(axis=0)
    std_vmem = vmem_ts.std(axis=0)

    # 2. Polarization fraction (fraction of time below threshold = polarized)
    polarized = vmem_ts < threshold
    pol_fraction = polarized.mean(axis=0)

    # 3. Transition rate (state switches per timestep)
    state = polarized.astype(int)
    transitions = np.abs(np.diff(state, axis=0))
    transition_rate = transitions.mean(axis=0)

    # 4. Shannon entropy of Vmem distribution
    vmin, vmax = vmem_ts.min(), vmem_ts.max()
    bin_edges = np.linspace(vmin - 1e-10, vmax + 1e-10, num_bins + 1)
    entropies = np.zeros(N)
    for i in range(N):
        counts, _ = np.histogram(vmem_ts[:, i], bins=bin_edges)
        p = counts / counts.sum()
        p = p[p > 0]
        entropies[i] = scipy_entropy(p, base=2)

    # 5. Temporal autocorrelation decay timescale
    autocorr_tau = np.zeros(N)
    for i in range(N):
        v = vmem_ts[:, i] - mean_vmem[i]
        var = np.var(v)
        if var < 1e-20:
            autocorr_tau[i] = T  # constant signal
            continue
        # Find lag where autocorrelation drops below 1/e
        max_lag = min(T // 2, 500)
        found = False
        for lag in range(1, max_lag):
            acf = np.mean(v[:T - lag] * v[lag:]) / var
            if acf < 1.0 / np.e:
                autocorr_tau[i] = lag
                found = True
                break
        if not found:
            autocorr_tau[i] = max_lag

    return {
        'mean_vmem': mean_vmem,
        'std_vmem': std_vmem,
        'pol_fraction': pol_fraction,
        'transition_rate': transition_rate,
        'entropy': entropies,
        'autocorr_tau': autocorr_tau,
    }


def compute_quadrant_indices(lattice_dims):
    """
    Compute indices of the symmetry-independent quadrant.

    For an 11x11 grid with two-fold reflection symmetry about both midlines,
    the independent quadrant is rows 0..center_r, cols 0..center_c (inclusive),
    i.e. a 6x6 = 36 cell region that includes the center row and column.

    Returns:
        quadrant_indices: 1D array of flat cell indices in the quadrant
        quadrant_rows: number of rows in the quadrant
        quadrant_cols: number of cols in the quadrant
    """
    rows, cols = lattice_dims
    center_r = rows // 2
    center_c = cols // 2
    q_rows = center_r + 1  # 0 to center inclusive
    q_cols = center_c + 1
    indices = []
    for r in range(q_rows):
        for c in range(q_cols):
            indices.append(r * cols + c)
    return np.array(indices), q_rows, q_cols


def compute_positional_uniqueness(vmem_ts, metrics, lattice_dims):
    """
    Compute metrics that quantify how well cell dynamics encode position.

    With homogeneous initial conditions on a square grid, the system has
    two-fold reflection symmetry about both midlines. Uniqueness is therefore
    computed only within the symmetry-independent quadrant (upper-left,
    including center row/col).

    Returns:
        dict with uniqueness metrics
    """
    T, _ = vmem_ts.shape
    rows, cols = lattice_dims

    # Quadrant: the symmetry-independent region
    quad_idx, q_rows, q_cols = compute_quadrant_indices(lattice_dims)
    N_quad = len(quad_idx)

    # 1. Pairwise KS distance matrix (quadrant cells only)
    ks_matrix = np.zeros((N_quad, N_quad))
    for i in range(N_quad):
        for j in range(i + 1, N_quad):
            stat, _ = ks_2samp(vmem_ts[:, quad_idx[i]], vmem_ts[:, quad_idx[j]])
            ks_matrix[i, j] = stat
            ks_matrix[j, i] = stat

    # 2. Unique fingerprint fraction (quadrant cells only)
    fingerprints = np.column_stack([
        np.round(metrics['mean_vmem'][quad_idx] * 1000, 1),  # round to 0.1 mV
        np.round(metrics['std_vmem'][quad_idx] * 1000, 1),
        np.round(metrics['transition_rate'][quad_idx] * 1000, 1),
    ])
    unique_fps = len(set(map(tuple, fingerprints)))
    uniqueness_fraction = unique_fps / N_quad

    # 3. Position decodability within the quadrant
    q_row_coords = np.array([idx // cols for idx in quad_idx])
    q_col_coords = np.array([idx % cols for idx in quad_idx])
    center_r, center_c = (rows - 1) / 2, (cols - 1) / 2
    q_dist = np.sqrt((q_row_coords - center_r) ** 2 +
                     (q_col_coords - center_c) ** 2)

    decodability = {}
    for name, values in metrics.items():
        q_values = values[quad_idx]
        r_row = np.corrcoef(q_row_coords, q_values)[0, 1]
        r_col = np.corrcoef(q_col_coords, q_values)[0, 1]
        r_dist = np.corrcoef(q_dist, q_values)[0, 1]
        decodability[name] = {
            'r_row': r_row, 'r_col': r_col, 'r_dist': r_dist,
            'r2_total': r_row ** 2 + r_col ** 2,
        }

    # 4. Mutual information estimate (quadrant cells)
    q_mean = metrics['mean_vmem'][quad_idx]
    n_pos_bins = min(N_quad, int(np.sqrt(N_quad)) + 1)
    mean_bins = np.quantile(q_mean, np.linspace(0, 1, n_pos_bins + 1))
    mean_bins[0] -= 1e-10
    mean_bins[-1] += 1e-10
    vmem_discrete = np.digitize(q_mean, mean_bins) - 1

    # P(vmem_bin)
    joint = np.zeros((N_quad, n_pos_bins))
    for i in range(N_quad):
        joint[i, vmem_discrete[i]] = 1
    p_vmem = joint.sum(axis=0) / N_quad
    p_vmem_pos = p_vmem[p_vmem > 0]
    H_vmem = scipy_entropy(p_vmem_pos, base=2)
    H_pos = np.log2(N_quad)
    mutual_info = H_vmem

    # 5. Uniqueness over time (quadrant cells)
    checkpoints = np.linspace(100, T, min(20, T // 100)).astype(int)
    uniqueness_over_time = []
    for t_end in checkpoints:
        q_ts = vmem_ts[:t_end, :][:, quad_idx]
        fp = np.column_stack([
            np.round(q_ts.mean(axis=0) * 1000, 1),
            np.round(q_ts.std(axis=0) * 1000, 1),
        ])
        unique = len(set(map(tuple, fp)))
        uniqueness_over_time.append((t_end, unique / N_quad))

    return {
        'ks_matrix': ks_matrix,
        'uniqueness_fraction': uniqueness_fraction,
        'decodability': decodability,
        'mutual_info': mutual_info,
        'H_pos': H_pos,
        'uniqueness_over_time': np.array(uniqueness_over_time),
        'quad_idx': quad_idx,
        'q_rows': q_rows,
        'q_cols': q_cols,
        'N_quad': N_quad,
    }


def plot_results(vmem_ts, metrics, uniqueness, lattice_dims, init_mode,
                 filename='data/positional_information.png'):
    """Generate the main analysis figure."""
    T, _ = vmem_ts.shape
    rows, cols = lattice_dims
    N_quad = uniqueness['N_quad']
    q_rows = uniqueness['q_rows']
    q_cols = uniqueness['q_cols']
    quad_idx = uniqueness['quad_idx']

    fig = plt.figure(figsize=(20, 16))
    # Outer grid: 3 rows. Row 1 has its own 5-col layout; rows 2-3 use 6-col.
    outer_gs = GridSpec(3, 1, figure=fig, hspace=0.45)

    # === Row 1: Spatial heatmaps of per-cell metrics (own 5-column sub-grid) ===
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_row1 = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[0], wspace=0.5)
    metric_plots = [
        ('mean_vmem', 'Mean Vmem (V)', 'RdBu'),
        ('std_vmem', 'Std Vmem (V)', 'hot'),
        ('pol_fraction', 'Polarization Fraction', 'coolwarm'),
        ('transition_rate', 'Transition Rate', 'YlOrRd'),
        ('entropy', 'Entropy (bits)', 'viridis'),
    ]

    for i, (key, title, cmap) in enumerate(metric_plots):
        ax = fig.add_subplot(gs_row1[0, i])
        data = metrics[key].reshape(rows, cols)
        im = ax.imshow(data, cmap=cmap, aspect='equal')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('col')
        ax.set_ylabel('row')
        # Draw quadrant boundary (dashed white lines at midlines)
        ax.axhline(y=q_rows - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=q_cols - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Sub-grids for rows 2 (6 columns) and 3 (5 columns, matching row 1)
    gs_row2 = GridSpecFromSubplotSpec(1, 6, subplot_spec=outer_gs[1], wspace=0.4)
    gs_row3 = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[2], wspace=0.5)

    # === Row 2: Positional encoding analysis (quadrant only) ===

    # KS distance matrix (quadrant cells)
    ax_ks = fig.add_subplot(gs_row2[0, 0:2])
    ks_upper = uniqueness['ks_matrix'][np.triu_indices(N_quad, 1)]
    im = ax_ks.imshow(uniqueness['ks_matrix'], cmap='magma', aspect='equal')
    ax_ks.set_title(f"Pairwise KS Distance (quadrant {q_rows}x{q_cols}={N_quad} cells)\n"
                    f"mean={ks_upper.mean():.3f}",
                    fontsize=10)
    ax_ks.set_xlabel('Quadrant cell index')
    ax_ks.set_ylabel('Quadrant cell index')
    plt.colorbar(im, ax=ax_ks, shrink=0.8, label='KS stat')

    # Uniqueness over time
    ax_uniq = fig.add_subplot(gs_row2[0, 2])
    uot = uniqueness['uniqueness_over_time']
    ax_uniq.plot(uot[:, 0], uot[:, 1], 'o-', markersize=4, color='teal')
    ax_uniq.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    ax_uniq.set_xlabel('Simulation time (iters)')
    ax_uniq.set_ylabel('Unique fraction (quadrant)')
    ax_uniq.set_title(f'Identity Emergence\n({q_rows}x{q_cols} quadrant)', fontsize=10)
    ax_uniq.set_ylim(0, 1.05)
    ax_uniq.grid(True, alpha=0.3)

    # Position decodability bar chart with r2_total (quadrant)
    ax_decode = fig.add_subplot(gs_row2[0, 3:5])
    decode = uniqueness['decodability']
    metric_names = list(decode.keys())
    r2_row = [abs(decode[m]['r_row']) for m in metric_names]
    r2_col = [abs(decode[m]['r_col']) for m in metric_names]
    r2_dist = [abs(decode[m]['r_dist']) for m in metric_names]
    r2_total = [decode[m]['r2_total'] for m in metric_names]
    x = np.arange(len(metric_names))
    w = 0.2
    ax_decode.bar(x - 1.5*w, r2_row, w, label='|r(row)|', color='steelblue', alpha=0.8)
    ax_decode.bar(x - 0.5*w, r2_col, w, label='|r(col)|', color='coral', alpha=0.8)
    ax_decode.bar(x + 0.5*w, r2_dist, w, label='|r(dist)|', color='seagreen', alpha=0.8)
    ax_decode.bar(x + 1.5*w, r2_total, w, label='r²_total', color='mediumpurple', alpha=0.8)
    ax_decode.set_xticks(x)
    ax_decode.set_xticklabels([n.replace('_', '\n') for n in metric_names],
                               fontsize=7, rotation=0)
    ax_decode.set_ylabel('|Correlation| / r² (quadrant)')
    ax_decode.set_title('Position Decodability (quadrant)', fontsize=10)
    ax_decode.legend(fontsize=7, loc='upper right')
    ax_decode.grid(True, alpha=0.3, axis='y')

    # Mutual information bar
    ax_mi = fig.add_subplot(gs_row2[0, 5])
    mi = uniqueness['mutual_info']
    h_pos = uniqueness['H_pos']
    bars = ax_mi.bar(['MI', 'H_pos'], [mi, h_pos],
                     color=['darkorange', 'lightgray'], edgecolor='black', width=0.5)
    ax_mi.set_ylabel('bits')
    ax_mi.set_title(f'Mutual Information\n{mi:.2f} / {h_pos:.2f} bits '
                    f'({mi / h_pos * 100:.0f}%)', fontsize=9)
    ax_mi.bar_label(bars, fmt='%.2f', fontsize=9, padding=3)
    ax_mi.set_ylim(0, h_pos * 1.25)
    ax_mi.grid(True, alpha=0.3, axis='y')

    # === Row 3: Quadrant cell trajectories and snapshots ===

    # Select representative cells from the quadrant
    center_idx = (rows // 2) * cols + cols // 2
    q_set = set(quad_idx)
    q_representative = sorted(set(idx for idx in [
        0,                                          # (0,0) corner
        cols // 2,                                  # (0,5) top-center edge
        (rows // 2) * cols,                         # (5,0) left-center edge
        center_idx,                                 # (5,5) center
        1,                                          # (0,1) near corner
        cols // 2 + cols,                           # (1,5) near top-center
        (rows // 4) * cols + cols // 4,             # (2,2) inner quadrant
    ] if idx in q_set))
    labels = []
    for idx in q_representative:
        r, c = idx // cols, idx % cols
        labels.append(f'({r},{c})')

    ax_traj = fig.add_subplot(gs_row3[0, 0:3])
    cmap_traj = plt.cm.tab10
    for k, idx in enumerate(q_representative):
        ax_traj.plot(vmem_ts[:, idx] * 1000, alpha=0.7, linewidth=0.8,
                     color=cmap_traj(k / max(len(q_representative), 1)),
                     label=labels[k])
    ax_traj.set_xlabel('Iteration')
    ax_traj.set_ylabel('Vmem (mV)')
    ax_traj.set_title('Quadrant Cell Trajectories', fontsize=10)
    ax_traj.legend(fontsize=7, ncol=3, loc='upper right')
    ax_traj.grid(True, alpha=0.3)

    # Vmem snapshots with quadrant boundary
    ax_snap1 = fig.add_subplot(gs_row3[0, 3])
    t_early = min(200, T - 1)
    im = ax_snap1.imshow(vmem_ts[t_early].reshape(rows, cols) * 1000,
                         cmap='RdBu', aspect='equal')
    ax_snap1.set_title(f'Vmem at t={t_early} (mV)', fontsize=9)
    ax_snap1.axhline(y=q_rows - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.7)
    ax_snap1.axvline(x=q_cols - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im, ax=ax_snap1, shrink=0.8)

    ax_snap2 = fig.add_subplot(gs_row3[0, 4])
    im = ax_snap2.imshow(vmem_ts[-1].reshape(rows, cols) * 1000,
                         cmap='RdBu', aspect='equal')
    ax_snap2.set_title(f'Vmem at t={T - 1} (mV)', fontsize=9)
    ax_snap2.axhline(y=q_rows - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.7)
    ax_snap2.axvline(x=q_cols - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im, ax=ax_snap2, shrink=0.8)

    # Summary annotation
    uf = uniqueness['uniqueness_fraction']
    fig.suptitle(
        f'Positional Information Analysis — Stigmergic Model (init={init_mode}, T={T})\n'
        f'Quadrant ({q_rows}x{q_cols}={N_quad} cells): '
        f'Unique fingerprints: {uf:.1%} | Mean pairwise KS: {ks_upper.mean():.3f} | '
        f'MI(pos;Vmem): {mi:.2f} / {h_pos:.2f} bits',
        fontsize=13, fontweight='bold')

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Positional information analysis of stigmergic field model')
    parser.add_argument('--num_iters', type=int, default=5000,
                        help='Number of simulation iterations (default: 5000)')
    parser.add_argument('--init_mode', type=str, default='random_gpol',
                        choices=['homogeneous', 'random_gpol'],
                        help='Initial condition mode (default: random_gpol)')
    parser.add_argument('--threshold', type=float, default=-0.04,
                        help='Polarization threshold in V (default: -0.04)')
    parser.add_argument('--num_bins', type=int, default=50,
                        help='Histogram bins for entropy (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    print("=" * 60)
    print("Positional Information Analysis")
    print("Stigmergic Bioelectric Field Model (unclamped)")
    print("=" * 60)
    print(f"  Init mode: {args.init_mode}")
    print(f"  Iterations: {args.num_iters}")
    print(f"  Polarization threshold: {args.threshold * 1000:.1f} mV")
    print(f"  Seed: {args.seed}")

    # Load parameters
    print("\n[1/4] Loading Stigmergic model...")
    parameters = torch.load('./data/StigmergicModelParameters.dat',
                            weights_only=False)
    parameters['latticePeriodicBoundaryGJ'] = False
    if 'ATPParameters' not in parameters:
        parameters['ATPParameters'] = None

    # Run simulation
    print("\n[2/4] Running unclamped simulation...")
    vmem_ts, lattice_dims = load_and_run(
        parameters, args.num_iters, args.init_mode, args.seed)
    print(f"  Vmem range: [{vmem_ts.min() * 1000:.1f}, {vmem_ts.max() * 1000:.1f}] mV")

    # Compute per-cell metrics
    print("\n[3/4] Computing per-cell metrics...")
    metrics = compute_per_cell_metrics(vmem_ts, args.threshold, args.num_bins)
    for name, values in metrics.items():
        print(f"  {name:20s}: mean={values.mean():.4f}, "
              f"std={values.std():.4f}, "
              f"range=[{values.min():.4f}, {values.max():.4f}]")

    # Compute positional uniqueness
    print("\n[4/4] Computing positional uniqueness...")
    uniqueness = compute_positional_uniqueness(
        vmem_ts, metrics, lattice_dims)

    N_quad = uniqueness['N_quad']
    q_rows = uniqueness['q_rows']
    q_cols = uniqueness['q_cols']
    ks_upper = uniqueness['ks_matrix'][np.triu_indices(N_quad, 1)]
    print(f"  Symmetry-independent quadrant: {q_rows}x{q_cols} = {N_quad} cells")
    print(f"  Unique fingerprint fraction: {uniqueness['uniqueness_fraction']:.1%}")
    print(f"  Mean pairwise KS distance: {ks_upper.mean():.4f}")
    print(f"  Min pairwise KS distance: {ks_upper.min():.4f}")
    print(f"  Mutual information: {uniqueness['mutual_info']:.2f} bits "
          f"(of {uniqueness['H_pos']:.2f} bits max)")

    # Position decodability
    print("\n  Position decodability within quadrant (|correlation|):")
    for name, d in uniqueness['decodability'].items():
        print(f"    {name:20s}: row={abs(d['r_row']):.3f}, "
              f"col={abs(d['r_col']):.3f}, dist={abs(d['r_dist']):.3f}")

    # Plot
    plot_results(vmem_ts, metrics, uniqueness, lattice_dims,
                 args.init_mode)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Quadrant:           {q_rows}x{q_cols} = {N_quad} cells "
          f"(from {lattice_dims[0]}x{lattice_dims[1]} grid with 2-fold reflection symmetry)")
    print(f"  Unique identities:  {uniqueness['uniqueness_fraction']:.1%} "
          f"({int(uniqueness['uniqueness_fraction'] * N_quad)}/{N_quad} cells)")
    print(f"  Positional MI:      {uniqueness['mutual_info']:.2f} / "
          f"{uniqueness['H_pos']:.2f} bits")
    print(f"  Mean KS distance:   {ks_upper.mean():.3f}")
    if uniqueness['uniqueness_fraction'] > 0.5:
        print("  --> Strong positional encoding: majority of quadrant cells are distinguishable")
    elif uniqueness['uniqueness_fraction'] > 0.1:
        print("  --> Moderate positional encoding: some quadrant cells are distinguishable")
    else:
        print("  --> Weak positional encoding: quadrant cells are largely indistinguishable")


if __name__ == '__main__':
    main()