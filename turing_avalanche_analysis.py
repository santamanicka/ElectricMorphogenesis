# Avalanche analysis of a Schnakenberg Turing reaction-diffusion system
#
# Comparison to the Stigmergic bioelectric model (avalanche_analysis.py).
# Uses the same reference-subtraction methodology:
#
#   1. Run a reference simulation (no perturbation) from t=0 to T
#   2. At t_perturb, fork the state and apply a small kick to the activator u
#   3. Run the perturbed simulation from t_perturb to T
#   4. Avalanche = deviation of perturbed trajectory from reference trajectory
#      delta_u(t) = u_perturbed(t) - u_reference(t)
#
# Schnakenberg model:
#   du/dt = a - u + u^2 * v + D_u * Laplacian(u)
#   dv/dt = b - u^2 * v     + D_v * Laplacian(v)
#
# Turing instability requires D_v >> D_u (differential diffusion).
#
# Usage:
#   python turing_avalanche_analysis.py                          # default analysis
#   python turing_avalanche_analysis.py --kick_amplitude 0.01    # custom kick
#   python turing_avalanche_analysis.py --perturb_times 3000,5000,7000
#   python turing_avalanche_analysis.py --Du 0.005 --Dv 0.5      # custom diffusion

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from avalanche_analysis import (
    compute_avalanche_metrics,
    fit_power_law_mle,
    power_law_ks_test,
    compute_ccdf,
    plot_power_law_analysis,
)

# ============================================================
# SCHNAKENBERG MODEL
# ============================================================

def build_laplacian_matrix(lattice_dims):
    """
    Build the discrete Laplacian matrix for a 2D grid with no-flux (Neumann)
    boundary conditions using the 5-point stencil.

    Returns:
        L: (N, N) numpy array where N = rows * cols
    """
    rows, cols = lattice_dims
    N = rows * cols
    L = np.zeros((N, N))

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            n_neighbors = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nidx = nr * cols + nc
                    L[idx, nidx] = 1.0
                    n_neighbors += 1
            L[idx, idx] = -n_neighbors

    return L


def schnakenberg_rhs(u, v, a, b, Du, Dv, L):
    """Compute du/dt, dv/dt for the Schnakenberg system."""
    u2v = u * u * v
    du = a - u + u2v + Du * (L @ u)
    dv = b - u2v + Dv * (L @ v)
    return du, dv


def run_schnakenberg(lattice_dims, a, b, Du, Dv, dt, num_iters,
                     u_init=None, v_init=None, record_every=1):
    """
    Run Schnakenberg simulation and return timeseries.

    Returns:
        u_ts: (T, N) array of activator timeseries (recorded every `record_every` steps)
        v_ts: (T, N) array of inhibitor timeseries
        u_final, v_final: final state vectors
    """
    N = np.prod(lattice_dims)
    L = build_laplacian_matrix(lattice_dims)

    # Initial conditions: uniform steady state + small noise
    u_ss = a + b
    v_ss = b / (a + b) ** 2
    if u_init is None:
        u = u_ss + 0.01 * np.random.randn(N)
    else:
        u = u_init.copy()
    if v_init is None:
        v = v_ss + 0.01 * np.random.randn(N)
    else:
        v = v_init.copy()

    n_records = num_iters // record_every
    u_ts = np.zeros((n_records, N))
    v_ts = np.zeros((n_records, N))
    rec_idx = 0

    for i in range(num_iters):
        du, dv = schnakenberg_rhs(u, v, a, b, Du, Dv, L)
        u = u + dt * du
        v = v + dt * dv
        # Clamp to non-negative
        u = np.maximum(u, 0.0)
        v = np.maximum(v, 0.0)

        if (i + 1) % record_every == 0 and rec_idx < n_records:
            u_ts[rec_idx] = u
            v_ts[rec_idx] = v
            rec_idx += 1

    return u_ts[:rec_idx], v_ts[:rec_idx], u, v


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Avalanche analysis of Schnakenberg Turing system')
    parser.add_argument('--num_total_iters', type=int, default=10000,
                        help='Total simulation iterations (default: 10000)')
    parser.add_argument('--perturb_times', type=str, default='3000,5000,7000',
                        help='Comma-separated perturbation times (default: 3000,5000,7000)')
    parser.add_argument('--kick_amplitude', type=float, default=0.01,
                        help='Perturbation amplitude for activator u (default: 0.01)')
    parser.add_argument('--num_kick_cells', type=int, default=1,
                        help='Number of cells to kick (default: 1)')
    parser.add_argument('--num_kick_locations', type=int, default=20,
                        help='Number of different kick locations to test (default: 20)')
    parser.add_argument('--response_window', type=int, default=500,
                        help='Timesteps to track after perturbation (default: 500)')
    parser.add_argument('--kick_amplitudes_sweep', type=str, default=None,
                        help='Comma-separated amplitudes for sweep')
    # Schnakenberg parameters
    parser.add_argument('--a', type=float, default=0.1,
                        help='Schnakenberg parameter a (default: 0.1)')
    parser.add_argument('--b', type=float, default=0.9,
                        help='Schnakenberg parameter b (default: 0.9)')
    parser.add_argument('--Du', type=float, default=0.01,
                        help='Activator diffusion coefficient (default: 0.01)')
    parser.add_argument('--Dv', type=float, default=1.0,
                        help='Inhibitor diffusion coefficient (default: 1.0)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Integration timestep (default: 0.01)')
    parser.add_argument('--lattice_dims', type=str, default='11,11',
                        help='Grid dimensions (default: 11,11)')
    args = parser.parse_args()

    lattice_dims = tuple(int(x) for x in args.lattice_dims.split(','))
    perturb_times = [int(x) for x in args.perturb_times.split(',')]
    num_total_iters = max(args.num_total_iters, max(perturb_times) + args.response_window)
    num_cells = np.prod(lattice_dims)

    print("=" * 60)
    print("Avalanche Analysis of Schnakenberg Turing System")
    print("=" * 60)
    print(f"  Lattice: {lattice_dims[0]}x{lattice_dims[1]} ({num_cells} cells)")
    print(f"  Schnakenberg params: a={args.a}, b={args.b}")
    print(f"  Diffusion: D_u={args.Du}, D_v={args.Dv} (ratio={args.Dv/args.Du:.0f})")
    print(f"  dt={args.dt}, total iters: {num_total_iters}")
    print(f"  Perturbation times: {perturb_times}")
    print(f"  Kick amplitude: {args.kick_amplitude}")
    print(f"  Kick cells: {args.num_kick_cells}")
    print(f"  Kick locations: {args.num_kick_locations}")
    print(f"  Response window: {args.response_window} iters")

    # Verify Turing instability conditions
    u_ss = args.a + args.b
    v_ss = args.b / u_ss ** 2
    fu = -1 + 2 * u_ss * v_ss  # df/du at steady state
    fv = u_ss ** 2              # df/dv at steady state
    gu = -2 * u_ss * v_ss       # dg/du at steady state
    gv = -u_ss ** 2             # dg/dv at steady state
    print(f"\n  Steady state: u*={u_ss:.4f}, v*={v_ss:.4f}")
    print(f"  Jacobian: fu={fu:.3f}, fv={fv:.3f}, gu={gu:.3f}, gv={gv:.3f}")
    print(f"  tr(J)={fu+gv:.3f} (need <0), det(J)={fu*gv-fv*gu:.3f} (need >0)")
    if fu + gv >= 0:
        print("  WARNING: Steady state is unstable without diffusion (tr >= 0)")
    if fu * gv - fv * gu <= 0:
        print("  WARNING: det(J) <= 0, steady state is a saddle point")

    # Step 1: Run reference simulation
    print(f"\n[1/3] Running reference simulation ({num_total_iters} iters)...")
    np.random.seed(42)
    ref_u_ts, ref_v_ts, _, _ = run_schnakenberg(
        lattice_dims, args.a, args.b, args.Du, args.Dv, args.dt,
        num_total_iters, record_every=1,
    )
    print(f"  Reference done. Final u range: [{ref_u_ts[-1].min():.4f}, {ref_u_ts[-1].max():.4f}]")
    print(f"  Final v range: [{ref_v_ts[-1].min():.4f}, {ref_v_ts[-1].max():.4f}]")

    # Step 2: Perturbation experiments
    print(f"\n[2/3] Running perturbation experiments...")

    np.random.seed(42)
    all_cells = np.arange(num_cells)
    kick_locations = np.random.choice(
        all_cells, size=min(args.num_kick_locations, num_cells), replace=False)

    all_metrics = {}

    for t_perturb in perturb_times:
        print(f"\n  t_perturb = {t_perturb}:")
        remaining = min(args.response_window, num_total_iters - t_perturb)

        # Reference window for this perturbation time
        ref_window = ref_u_ts[t_perturb:t_perturb + remaining]  # (T, N)

        for loc_idx, cell in enumerate(kick_locations):
            # Select kick cells
            if args.num_kick_cells == 1:
                kick_cells = [cell]
            else:
                row, col = cell // lattice_dims[1], cell % lattice_dims[1]
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < lattice_dims[0] and 0 <= nc < lattice_dims[1]:
                        neighbors.append(nr * lattice_dims[1] + nc)
                kick_cells = neighbors[:args.num_kick_cells]

            # Fork state at t_perturb and apply kick
            u_init = ref_u_ts[t_perturb].copy()
            v_init = ref_v_ts[t_perturb].copy()
            u_init[kick_cells] += args.kick_amplitude

            # Run perturbed simulation
            pert_u_ts, _, _, _ = run_schnakenberg(
                lattice_dims, args.a, args.b, args.Du, args.Dv, args.dt,
                remaining, u_init=u_init, v_init=v_init, record_every=1,
            )

            # Compute deviation
            delta_u = pert_u_ts - ref_window[:len(pert_u_ts)]

            # Compute metrics (reuse from avalanche_analysis)
            metrics = compute_avalanche_metrics(delta_u, lattice_dims)
            all_metrics[(t_perturb, cell)] = metrics

            if loc_idx % 5 == 0:
                print(f"    cell {cell:3d}: size={metrics['size']:.4e}, "
                      f"dur={metrics['duration']:3d}, spatial={metrics['spatial_extent']:3d}, "
                      f"branch={metrics['branching_ratio']:.3f}")

    # Step 3: Aggregate and analyze
    print(f"\n[3/3] Analyzing avalanche statistics...")

    results_by_time = {}
    for t_perturb in perturb_times:
        sizes = []
        durations = []
        spatial_extents = []
        branching_ratios = []
        activities = []
        for cell in kick_locations:
            m = all_metrics[(t_perturb, cell)]
            if m['size'] > 0:
                sizes.append(m['size'])
                durations.append(m['duration'])
                spatial_extents.append(m['spatial_extent'])
                if not np.isnan(m['branching_ratio']):
                    branching_ratios.append(m['branching_ratio'])
                activities.append(m['activity'])

        results_by_time[t_perturb] = {
            'sizes': np.array(sizes),
            'durations': np.array(durations),
            'spatial_extents': np.array(spatial_extents),
            'branching_ratios': np.array(branching_ratios),
            'activities': activities,
        }

        print(f"\n  t_perturb = {t_perturb} ({len(sizes)} avalanches):")
        if len(sizes) > 0:
            print(f"    Size:     mean={np.mean(sizes):.4e}, median={np.median(sizes):.4e}, "
                  f"range=[{np.min(sizes):.4e}, {np.max(sizes):.4e}]")
            print(f"    Duration: mean={np.mean(durations):.1f}, median={np.median(durations):.1f}, "
                  f"range=[{np.min(durations)}, {np.max(durations)}]")
            print(f"    Spatial:  mean={np.mean(spatial_extents):.1f}, "
                  f"range=[{np.min(spatial_extents)}, {np.max(spatial_extents)}]")
        if len(branching_ratios) > 0:
            br = np.mean(branching_ratios)
            print(f"    Branching ratio: {br:.4f} (=1.0 at criticality)")

    # Save results
    save_data = {
        'all_metrics': {str(k): v for k, v in all_metrics.items()},
        'results_by_time': results_by_time,
        'perturb_times': perturb_times,
        'kick_locations': kick_locations,
        'kick_amplitude': args.kick_amplitude,
        'lattice_dims': lattice_dims,
        'ref_u_ts': ref_u_ts,
        'ref_v_ts': ref_v_ts,
        'schnakenberg_params': {'a': args.a, 'b': args.b, 'Du': args.Du, 'Dv': args.Dv, 'dt': args.dt},
    }
    np.savez_compressed('data/turing_avalanche_results.npz', **{
        k: v for k, v in save_data.items() if isinstance(v, np.ndarray)
    })

    # ============================================================
    # PLOTTING
    # ============================================================

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- Row 1: Reference activator u pattern snapshots ---
    snap_times = [0, num_total_iters // 4, num_total_iters // 2, num_total_iters - 1]
    snap_times = [t for t in snap_times if t < len(ref_u_ts)]
    for i, t in enumerate(snap_times[:3]):
        ax = fig.add_subplot(gs[0, i])
        u_2d = ref_u_ts[t].reshape(lattice_dims)
        im = ax.imshow(u_2d, cmap='viridis', aspect='equal')
        ax.set_title(f'Reference u(x)\nt={t}', fontsize=10)
        ax.set_xlabel('col')
        ax.set_ylabel('row')
        plt.colorbar(im, ax=ax, label='u', shrink=0.8)

    # --- Row 2: Avalanche size/duration/spatial distributions ---
    ax_size = fig.add_subplot(gs[1, 0])
    ax_dur = fig.add_subplot(gs[1, 1])
    ax_spatial = fig.add_subplot(gs[1, 2])

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(perturb_times)))
    for i, t_perturb in enumerate(perturb_times):
        r = results_by_time[t_perturb]
        if len(r['sizes']) > 2:
            ax_size.hist(r['sizes'], bins=15, alpha=0.5, color=colors[i],
                         label=f't={t_perturb}', edgecolor='k', linewidth=0.5)
            ax_dur.hist(r['durations'], bins=15, alpha=0.5, color=colors[i],
                        label=f't={t_perturb}', edgecolor='k', linewidth=0.5)
            ax_spatial.hist(r['spatial_extents'], bins=15, alpha=0.5, color=colors[i],
                            label=f't={t_perturb}', edgecolor='k', linewidth=0.5)

    ax_size.set_xlabel('Avalanche Size')
    ax_size.set_ylabel('Count')
    ax_size.set_title('Size Distribution')
    ax_size.legend(fontsize=8)

    ax_dur.set_xlabel('Duration (iters)')
    ax_dur.set_ylabel('Count')
    ax_dur.set_title('Duration Distribution')
    ax_dur.legend(fontsize=8)

    ax_spatial.set_xlabel('Spatial Extent (cells)')
    ax_spatial.set_ylabel('Count')
    ax_spatial.set_title('Spatial Extent Distribution')
    ax_spatial.legend(fontsize=8)

    # --- Row 3: Activity profiles and branching ratio ---
    ax_activity = fig.add_subplot(gs[2, 0:2])
    for i, t_perturb in enumerate(perturb_times):
        r = results_by_time[t_perturb]
        if len(r['activities']) > 0:
            max_len = max(len(a) for a in r['activities'])
            padded = np.zeros((len(r['activities']), max_len))
            for j, a in enumerate(r['activities']):
                padded[j, :len(a)] = a
            mean_activity = padded.mean(axis=0)
            std_activity = padded.std(axis=0)
            t_axis = np.arange(max_len)
            ax_activity.semilogy(t_axis, mean_activity, color=colors[i], label=f't={t_perturb}')
            ax_activity.fill_between(t_axis,
                                     np.maximum(mean_activity - std_activity, 1e-10),
                                     mean_activity + std_activity,
                                     alpha=0.2, color=colors[i])

    ax_activity.set_xlabel('Time after perturbation (iters)')
    ax_activity.set_ylabel(r'Total $|\Delta u|$')
    ax_activity.set_title('Average Avalanche Activity Profile')
    ax_activity.legend(fontsize=8)
    ax_activity.grid(True, alpha=0.3)

    # Branching ratio by perturbation time
    ax_branch = fig.add_subplot(gs[2, 2])
    br_means = []
    br_stds = []
    for t_perturb in perturb_times:
        r = results_by_time[t_perturb]
        if len(r['branching_ratios']) > 0:
            br_means.append(np.mean(r['branching_ratios']))
            br_stds.append(np.std(r['branching_ratios']))
        else:
            br_means.append(np.nan)
            br_stds.append(0)
    ax_branch.errorbar(perturb_times, br_means, yerr=br_stds, fmt='o-', capsize=5, markersize=8)
    ax_branch.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Critical (BR=1)')
    ax_branch.set_xlabel('Perturbation Time')
    ax_branch.set_ylabel('Branching Ratio')
    ax_branch.set_title('Branching Ratio vs Time')
    ax_branch.legend(fontsize=8)
    ax_branch.grid(True, alpha=0.3)

    plt.suptitle(f'Schnakenberg Turing Avalanche Analysis (kick={args.kick_amplitude})',
                 fontsize=14, fontweight='bold')
    plt.savefig('data/turing_avalanche_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: data/turing_avalanche_analysis.png")

    # Power-law analysis
    print(f"\n{'='*60}")
    print("Power-Law Fitting (Clauset et al. 2009)")
    print(f"{'='*60}")
    fit_results = plot_power_law_analysis(results_by_time, perturb_times,
                                          'data/turing_avalanche_powerlaw.png')
    for key, fr in fit_results.items():
        print(f"  {key}:")
        print(f"    alpha = {fr['alpha']:.3f} +/- {fr['sigma']:.3f}")
        print(f"    x_min = {fr['xmin']:.4f}, n_tail = {fr['n_tail']}")
        print(f"    KS stat = {fr['ks_stat']:.4f}, KS p-value = {fr['ks_p']:.3f}")
        if fr['ks_p'] >= 0.1:
            print(f"    --> Power-law PLAUSIBLE (p >= 0.1)")
        else:
            print(f"    --> Power-law REJECTED (p < 0.1)")
    print(f"\nSaved: data/turing_avalanche_powerlaw.png")

    # ============================================================
    # AMPLITUDE SWEEP (optional)
    # ============================================================
    if args.kick_amplitudes_sweep:
        amplitudes = [float(x) for x in args.kick_amplitudes_sweep.split(',')]
        t_probe = perturb_times[0]
        remaining = min(args.response_window, num_total_iters - t_probe)
        ref_window = ref_u_ts[t_probe:t_probe + remaining]

        print(f"\n{'='*60}")
        print(f"Amplitude Sweep at t={t_probe}")
        print(f"{'='*60}")

        sweep_sizes = []
        sweep_durations = []
        sweep_spatial = []
        test_cell = kick_locations[0]

        for amp in amplitudes:
            u_init = ref_u_ts[t_probe].copy()
            v_init = ref_v_ts[t_probe].copy()
            u_init[test_cell] += amp

            pert_u_ts, _, _, _ = run_schnakenberg(
                lattice_dims, args.a, args.b, args.Du, args.Dv, args.dt,
                remaining, u_init=u_init, v_init=v_init, record_every=1,
            )
            delta = pert_u_ts - ref_window[:len(pert_u_ts)]
            m = compute_avalanche_metrics(delta, lattice_dims)
            sweep_sizes.append(m['size'])
            sweep_durations.append(m['duration'])
            sweep_spatial.append(m['spatial_extent'])
            print(f"  amp={amp:10.4f}: size={m['size']:.4e}, "
                  f"dur={m['duration']:3d}, spatial={m['spatial_extent']:3d}")

        # Plot amplitude scaling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        amps = np.array(amplitudes)
        ax1.loglog(amps, sweep_sizes, 'o-', markersize=8)
        ax1.set_xlabel('Kick Amplitude')
        ax1.set_ylabel('Avalanche Size')
        ax1.set_title('Size vs Kick Amplitude')
        ax1.grid(True, alpha=0.3)
        if len(amplitudes) >= 3:
            log_a = np.log(amps)
            log_s = np.log(np.array(sweep_sizes) + 1e-20)
            valid = np.isfinite(log_s)
            if valid.sum() >= 2:
                slope, intercept = np.polyfit(log_a[valid], log_s[valid], 1)
                a_fit = np.linspace(amps.min(), amps.max(), 50)
                ax1.loglog(a_fit, np.exp(intercept) * a_fit ** slope, '--',
                           label=f'slope = {slope:.2f}')
                ax1.legend()

        ax2.semilogx(amps, sweep_spatial, 's-', markersize=8, color='C1')
        ax2.set_xlabel('Kick Amplitude')
        ax2.set_ylabel('Spatial Extent (cells)')
        ax2.set_title('Spatial Extent vs Kick Amplitude')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Amplitude Scaling (cell={test_cell}, t={t_probe})', fontsize=13)
        plt.tight_layout()
        plt.savefig('data/turing_avalanche_amplitude_sweep.png', dpi=150)
        plt.close()
        print(f"\nSaved: data/turing_avalanche_amplitude_sweep.png")


if __name__ == '__main__':
    main()