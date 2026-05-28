# Avalanche analysis of the trained Stigmergic bioelectric patterning model
#
# The Stigmergic model does NOT reach a steady state — the face pattern continues
# evolving well beyond t=1000. To isolate perturbation-induced avalanches from
# ongoing dynamics, we use a REFERENCE SUBTRACTION approach:
#
#   1. Run a reference simulation (no perturbation) from t=0 to T
#   2. At t_perturb (after t=500), fork the state and apply a small Vmem kick
#      to a single cell (or small group)
#   3. Run the perturbed simulation from t_perturb to T
#   4. Avalanche = deviation of perturbed trajectory from reference trajectory
#      delta_V(t) = Vmem_perturbed(t) - Vmem_reference(t)
#
# This cleanly separates the perturbation response from ongoing pattern dynamics.
#
# Measured quantities:
#   - Avalanche size: sum of |delta_V| across all cells and timesteps
#   - Avalanche duration: time until max|delta_V| drops below threshold
#   - Spatial extent: number of cells significantly affected
#   - Branching ratio: <activity(t+1)> / <activity(t)> — =1 at criticality
#
# Usage:
#   python avalanche_analysis.py                          # default analysis
#   python avalanche_analysis.py --kick_amplitude 0.005   # 5 mV kick
#   python avalanche_analysis.py --num_kick_cells 1       # single-cell perturbation
#   python avalanche_analysis.py --perturb_times 500,700,900  # multiple probe times

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from embryo import model
from scipy.ndimage import label
from scipy.stats import kstest

# ============================================================
# SIMULATION HELPERS
# ============================================================

def load_stigmergic_model():
    """Load the trained Stigmergic model parameters."""
    parameters = torch.load('./data/StigmergicModelParameters.dat', weights_only=False)
    parameters['latticePeriodicBoundaryGJ'] = False
    if 'ATPParameters' not in parameters:
        parameters['ATPParameters'] = None
    return parameters


def setup_model(parameters, num_samples=1):
    """Create and initialize the embryo model with trained parameters."""
    initial_values = parameters['simParameters']['initialValues']
    clamp_params = parameters['clampParameters']
    external_inputs = parameters['simParameters']['externalInputs']

    # Replicate initial conditions for multiple samples if needed
    if num_samples > 1:
        num_cells = np.prod(parameters['latticeDims'])
        iv = initial_values
        iv['Vmem'] = iv['Vmem'].repeat(num_samples, 1, 1)
        iv['eV'] = iv['eV'].repeat(num_samples, 1, 1)
        iv['ligandConc'] = iv['ligandConc'].repeat(num_samples, 1, 1)
        iv['G_pol']['cells'] = iv['G_pol']['cells'] * num_samples
        iv['G_pol']['values'] = iv['G_pol']['values'] * num_samples
        # Replicate clamp indices for multiple samples
        orig_sample_idx, orig_clamp_idx = clamp_params['clampIndices']
        n_clamp = len(orig_clamp_idx)
        new_sample_idx = np.concatenate([np.full(n_clamp, s) for s in range(num_samples)])
        new_clamp_idx = np.tile(orig_clamp_idx, num_samples)
        clamp_params['clampIndices'] = (new_sample_idx, new_clamp_idx)
        clamp_params['clampValues'] = clamp_params['clampValues'].repeat(1, num_samples)

    m = model(parameters, num_samples)
    m.setExperimentalConditions((initial_values, num_samples))
    return m, initial_values, clamp_params, external_inputs


def run_reference(parameters, num_total_iters):
    """Run the full reference (unperturbed) simulation and return the timeseries."""
    m, iv, clamp_params, ext_inputs = setup_model(parameters, num_samples=1)
    m.simulate(
        externalInputs=ext_inputs,
        clampParameters=clamp_params,
        perturbation=None,
        numSimIters=num_total_iters,
    )
    # Return Vmem timeseries: shape (T, 1, numCells, 1)
    return m.timeseriesVmem.detach().clone(), m


def run_perturbed_from_snapshot(parameters, ref_model, t_perturb, num_remaining_iters,
                                 kick_cells, kick_amplitude):
    """
    Fork from reference state at t_perturb, apply kick, and run forward.

    Args:
        parameters: model parameters dict
        ref_model: the reference model (already simulated)
        t_perturb: iteration at which to fork
        num_remaining_iters: how many iterations to run after forking
        kick_cells: list of cell indices to perturb
        kick_amplitude: voltage kick in Volts (e.g., 0.005 = 5 mV)

    Returns:
        vmem_ts: Vmem timeseries of the perturbed run, shape (num_remaining_iters, 1, numCells, 1)
    """
    # Create a fresh model
    m, iv, clamp_params, ext_inputs = setup_model(parameters, num_samples=1)
    circuit = m.electricNetwork

    # Copy the reference state at t_perturb
    circuit.Vmem = ref_model.timeseriesVmem[t_perturb].clone()
    circuit.eV = ref_model.timeserieseV[t_perturb].clone()
    circuit.G_pol = ref_model.timeseriesGpol[t_perturb].clone()
    circuit.ligandConc = ref_model.timeseriesLigandConc[t_perturb].clone()
    circuit.eVforceVector[0] = ref_model.timeserieseVforceVector[t_perturb, 0].clone()
    circuit.eVforceVector[1] = ref_model.timeserieseVforceVector[t_perturb, 1].clone()
    circuit.GapJunctionCurrent = ref_model.timeseriesGJcurrent[t_perturb].clone()
    circuit.G_ij = ref_model.timeseriesGij[t_perturb].clone()

    # Apply the perturbation kick
    circuit.Vmem[0, kick_cells, 0] += kick_amplitude

    # Run forward — no clamping (clamping ended at iter 100)
    m.simulate(
        externalInputs=ext_inputs,
        clampParameters=None,
        perturbation=None,
        numSimIters=num_remaining_iters,
        outerIter=t_perturb,
    )
    return m.timeseriesVmem.detach().clone()


# ============================================================
# AVALANCHE METRICS
# ============================================================

def compute_avalanche_metrics(delta_V_ts, lattice_dims, threshold_frac=0.05):
    """
    Compute avalanche metrics from the deviation timeseries.

    Args:
        delta_V_ts: shape (T, numCells) — deviation from reference per cell per timestep
        lattice_dims: (rows, cols)
        threshold_frac: fraction of max|delta_V| to use as activity threshold

    Returns:
        dict with size, duration, spatial_extent, branching_ratio, activity_profile
    """
    T, N = delta_V_ts.shape
    abs_delta = np.abs(delta_V_ts)

    # Activity profile: total |delta_V| at each timestep
    activity = abs_delta.sum(axis=1)  # shape (T,)

    # Peak deviation
    peak_delta = abs_delta.max()
    if peak_delta < 1e-15:
        return {
            'size': 0, 'duration': 0, 'spatial_extent': 0,
            'branching_ratio': np.nan, 'activity': activity,
            'peak_delta': 0, 'active_cells_ts': np.zeros(T),
        }

    # Threshold for "active" cells
    threshold = threshold_frac * peak_delta

    # Active cells per timestep
    active_mask = abs_delta > threshold  # (T, N)
    active_cells_ts = active_mask.sum(axis=1)  # (T,)

    # Avalanche size: total integrated activity above threshold
    size = abs_delta[active_mask].sum()

    # Duration: time from first to last active timestep
    active_times = np.where(active_cells_ts > 0)[0]
    if len(active_times) > 0:
        duration = active_times[-1] - active_times[0] + 1
    else:
        duration = 0

    # Spatial extent: max number of simultaneously active cells
    spatial_extent = active_cells_ts.max()

    # Branching ratio: <n_active(t+1)> / <n_active(t)> for active timesteps
    active_t = active_cells_ts[:-1]
    active_t1 = active_cells_ts[1:]
    mask = active_t > 0
    if mask.sum() > 0:
        branching_ratio = np.mean(active_t1[mask] / active_t[mask])
    else:
        branching_ratio = np.nan

    return {
        'size': float(size),
        'duration': int(duration),
        'spatial_extent': int(spatial_extent),
        'branching_ratio': float(branching_ratio),
        'activity': activity,
        'peak_delta': float(peak_delta),
        'active_cells_ts': active_cells_ts,
    }


# ============================================================
# POWER-LAW FITTING (Clauset, Shalizi, Newman 2009)
# ============================================================

def fit_power_law_mle(data, x_min=None):
    """
    Maximum likelihood estimation of power-law exponent.
    alpha_hat = 1 + n / sum(ln(x_i / x_min))
    Returns (alpha, sigma, x_min, n_tail).
    """
    data = np.array(data, dtype=float)
    data = data[data > 0]
    if len(data) < 5:
        return np.nan, np.nan, np.nan, 0

    if x_min is None:
        # Search for optimal x_min by minimizing KS distance
        sorted_data = np.sort(data)
        best_ks = np.inf
        best_xmin = sorted_data[0]
        # Test each unique value as candidate x_min (up to 90th percentile)
        candidates = np.unique(sorted_data)
        max_idx = max(1, int(0.9 * len(candidates)))
        for xm in candidates[:max_idx]:
            tail = data[data >= xm]
            if len(tail) < 5:
                continue
            n = len(tail)
            alpha = 1.0 + n / np.sum(np.log(tail / xm))
            # KS statistic for this fit
            theoretical_cdf = 1.0 - (np.sort(tail) / xm)**(1 - alpha)
            empirical_cdf = np.arange(1, n + 1) / n
            ks = np.max(np.abs(theoretical_cdf - empirical_cdf))
            if ks < best_ks:
                best_ks = ks
                best_xmin = xm
        x_min = best_xmin

    tail = data[data >= x_min]
    n = len(tail)
    if n < 5:
        return np.nan, np.nan, x_min, n
    alpha = 1.0 + n / np.sum(np.log(tail / x_min))
    sigma = (alpha - 1.0) / np.sqrt(n)
    return alpha, sigma, x_min, n


def power_law_ks_test(data, alpha, x_min):
    """
    Kolmogorov-Smirnov test: compare data tail to fitted power law.
    Returns KS statistic and p-value (via parametric bootstrap).
    """
    tail = np.sort(data[data >= x_min])
    n = len(tail)
    if n < 5 or np.isnan(alpha):
        return np.nan, np.nan

    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n
    # Theoretical CDF: P(X <= x) = 1 - (x/x_min)^(1-alpha)
    tcdf = 1.0 - (tail / x_min)**(1 - alpha)
    ks_stat = np.max(np.abs(ecdf - tcdf))

    # Bootstrap p-value: generate synthetic power-law samples and measure KS
    n_bootstrap = 500
    count_worse = 0
    for _ in range(n_bootstrap):
        # Generate power-law sample: x = x_min * u^(-1/(alpha-1))
        u = np.random.uniform(0, 1, n)
        synthetic = x_min * u**(-1.0 / (alpha - 1.0))
        # Fit MLE to synthetic
        alpha_syn = 1.0 + n / np.sum(np.log(synthetic / x_min))
        # KS of synthetic vs its own fit
        syn_sorted = np.sort(synthetic)
        syn_ecdf = np.arange(1, n + 1) / n
        syn_tcdf = 1.0 - (syn_sorted / x_min)**(1 - alpha_syn)
        ks_syn = np.max(np.abs(syn_ecdf - syn_tcdf))
        if ks_syn >= ks_stat:
            count_worse += 1
    p_value = count_worse / n_bootstrap
    return ks_stat, p_value


def compute_ccdf(data):
    """Compute complementary CDF: P(X >= x) for each unique x."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    ccdf = 1.0 - np.arange(n) / n
    return sorted_data, ccdf


def plot_power_law_analysis(results_by_time, perturb_times, filename):
    """Generate dedicated power-law analysis figure with CCDF plots and fits."""
    n_times = len(perturb_times)
    fig, axes = plt.subplots(2, n_times, figsize=(6 * n_times, 10))
    if n_times == 1:
        axes = axes.reshape(2, 1)

    all_fit_results = {}

    for col, t_perturb in enumerate(perturb_times):
        r = results_by_time[t_perturb]
        sizes = r['sizes']
        durations = r['durations']

        # --- Size CCDF (top row) ---
        ax = axes[0, col]
        if len(sizes) >= 5:
            x_s, ccdf_s = compute_ccdf(sizes)
            ax.loglog(x_s, ccdf_s, 'o', markersize=4, alpha=0.7, label='Data')

            alpha_s, sigma_s, xmin_s, n_tail_s = fit_power_law_mle(sizes)
            ks_s, p_s = power_law_ks_test(sizes, alpha_s, xmin_s)

            if not np.isnan(alpha_s) and np.isfinite(alpha_s) and np.isfinite(sigma_s):
                # Plot fitted power law on CCDF
                x_fit = np.logspace(np.log10(xmin_s), np.log10(sizes.max()), 100)
                ccdf_fit = (x_fit / xmin_s)**(1 - alpha_s)
                # Scale to match empirical CCDF at x_min
                n_above_xmin = np.sum(sizes >= xmin_s)
                scale = n_above_xmin / len(sizes)
                ax.loglog(x_fit, ccdf_fit * scale, 'r--', linewidth=2,
                          label=rf'$\alpha={alpha_s:.2f} \pm {sigma_s:.2f}$')
                ax.axvline(xmin_s, color='gray', linestyle=':', alpha=0.5, label=f'$x_{{min}}={xmin_s:.3f}$')

                fit_label = f'alpha={alpha_s:.2f}+/-{sigma_s:.2f}, KS p={p_s:.2f}, n_tail={n_tail_s}'
            else:
                fit_label = 'fit failed'

            all_fit_results[f'size_t{t_perturb}'] = {
                'alpha': alpha_s, 'sigma': sigma_s, 'xmin': xmin_s,
                'n_tail': n_tail_s, 'ks_stat': ks_s, 'ks_p': p_s,
            }
        else:
            fit_label = f'too few data ({len(sizes)})'

        ax.set_xlabel('Avalanche Size')
        ax.set_ylabel('P(S >= s)')
        ax.set_title(f't={t_perturb}: Size CCDF\n{fit_label}', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

        # --- Duration CCDF (bottom row) ---
        ax = axes[1, col]
        if len(durations) >= 5:
            x_d, ccdf_d = compute_ccdf(durations.astype(float))
            ax.loglog(x_d, ccdf_d, 's', markersize=4, alpha=0.7, color='C1', label='Data')

            alpha_d, sigma_d, xmin_d, n_tail_d = fit_power_law_mle(durations.astype(float))
            ks_d, p_d = power_law_ks_test(durations.astype(float), alpha_d, xmin_d)

            if not np.isnan(alpha_d) and np.isfinite(alpha_d) and np.isfinite(sigma_d):
                x_fit = np.logspace(np.log10(xmin_d), np.log10(durations.max()), 100)
                ccdf_fit = (x_fit / xmin_d)**(1 - alpha_d)
                n_above_xmin = np.sum(durations >= xmin_d)
                scale = n_above_xmin / len(durations)
                ax.loglog(x_fit, ccdf_fit * scale, 'r--', linewidth=2,
                          label=rf'$\alpha={alpha_d:.2f} \pm {sigma_d:.2f}$')
                ax.axvline(xmin_d, color='gray', linestyle=':', alpha=0.5, label=f'$x_{{min}}={xmin_d:.0f}$')

                fit_label = f'alpha={alpha_d:.2f}+/-{sigma_d:.2f}, KS p={p_d:.2f}, n_tail={n_tail_d}'
            else:
                fit_label = 'fit failed'

            all_fit_results[f'duration_t{t_perturb}'] = {
                'alpha': alpha_d, 'sigma': sigma_d, 'xmin': xmin_d,
                'n_tail': n_tail_d, 'ks_stat': ks_d, 'ks_p': p_d,
            }
        else:
            fit_label = f'too few data ({len(durations)})'

        ax.set_xlabel('Avalanche Duration (iters)')
        ax.set_ylabel('P(T >= t)')
        ax.set_title(f't={t_perturb}: Duration CCDF\n{fit_label}', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    plt.suptitle('Power-Law Analysis of Avalanche Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return all_fit_results


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Avalanche analysis of trained Stigmergic model')
    parser.add_argument('--num_total_iters', type=int, default=1500,
                        help='Total simulation iterations (default: 1500)')
    parser.add_argument('--perturb_times', type=str, default='500,700,900',
                        help='Comma-separated perturbation times (default: 500,700,900)')
    parser.add_argument('--kick_amplitude', type=float, default=0.005,
                        help='Perturbation amplitude in Volts (default: 0.005 = 5 mV)')
    parser.add_argument('--num_kick_cells', type=int, default=1,
                        help='Number of cells to kick (default: 1)')
    parser.add_argument('--num_kick_locations', type=int, default=20,
                        help='Number of different kick locations to test (default: 20)')
    parser.add_argument('--response_window', type=int, default=200,
                        help='Timesteps to track after perturbation (default: 200)')
    parser.add_argument('--kick_amplitudes_sweep', type=str, default=None,
                        help='Comma-separated amplitudes for sweep (e.g., 0.001,0.005,0.01,0.05)')
    args = parser.parse_args()

    perturb_times = [int(x) for x in args.perturb_times.split(',')]
    num_total_iters = max(args.num_total_iters, max(perturb_times) + args.response_window)

    print("=" * 60)
    print("Avalanche Analysis of Trained Stigmergic Model")
    print("=" * 60)
    print(f"  Total iters: {num_total_iters}")
    print(f"  Perturbation times: {perturb_times}")
    print(f"  Kick amplitude: {args.kick_amplitude*1000:.1f} mV")
    print(f"  Kick cells: {args.num_kick_cells}")
    print(f"  Kick locations: {args.num_kick_locations}")
    print(f"  Response window: {args.response_window} iters")

    # Step 1: Run reference simulation
    print(f"\n[1/3] Running reference simulation ({num_total_iters} iters)...")
    parameters = load_stigmergic_model()
    lattice_dims = parameters['latticeDims']
    num_cells = np.prod(lattice_dims)
    ref_vmem_ts, ref_model = run_reference(parameters, num_total_iters)
    print(f"  Reference done. Final Vmem range: [{ref_vmem_ts[-1].min():.4f}, {ref_vmem_ts[-1].max():.4f}]")

    # Step 2: For each perturb_time x kick_location, run perturbed simulation
    print(f"\n[2/3] Running perturbation experiments...")

    # Select kick locations (random cells, or systematic grid)
    np.random.seed(42)
    all_cells = np.arange(num_cells)
    kick_locations = np.random.choice(all_cells, size=min(args.num_kick_locations, num_cells), replace=False)

    all_metrics = {}  # keyed by (t_perturb, cell_idx)

    for t_perturb in perturb_times:
        print(f"\n  t_perturb = {t_perturb}:")
        remaining = min(args.response_window, num_total_iters - t_perturb)

        # Reference Vmem for this window
        ref_window = ref_vmem_ts[t_perturb:t_perturb + remaining, 0, :, 0].numpy()  # (T, N)

        for loc_idx, cell in enumerate(kick_locations):
            # Select kick cells (single cell or neighbors)
            if args.num_kick_cells == 1:
                kick_cells = [cell]
            else:
                # Pick the cell and its nearest neighbors
                row, col = cell // lattice_dims[1], cell % lattice_dims[1]
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < lattice_dims[0] and 0 <= nc < lattice_dims[1]:
                        neighbors.append(nr * lattice_dims[1] + nc)
                kick_cells = neighbors[:args.num_kick_cells]

            # Run perturbed simulation
            parameters_fresh = load_stigmergic_model()
            pert_vmem_ts = run_perturbed_from_snapshot(
                parameters_fresh, ref_model, t_perturb, remaining,
                kick_cells, args.kick_amplitude,
            )

            # Compute deviation
            pert_window = pert_vmem_ts[:, 0, :, 0].numpy()  # (T, N)
            delta_V = pert_window - ref_window

            # Compute metrics
            metrics = compute_avalanche_metrics(delta_V, lattice_dims)
            all_metrics[(t_perturb, cell)] = metrics

            if loc_idx % 5 == 0:
                print(f"    cell {cell:3d}: size={metrics['size']:.4e}, "
                      f"dur={metrics['duration']:3d}, spatial={metrics['spatial_extent']:3d}, "
                      f"branch={metrics['branching_ratio']:.3f}")

    # Step 3: Aggregate and analyze
    print(f"\n[3/3] Analyzing avalanche statistics...")

    # Collect all sizes, durations, spatial extents
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
        'ref_vmem_ts': ref_vmem_ts.numpy(),
    }
    torch.save(save_data, 'data/stigmergic_avalanche_results.dat')

    # ============================================================
    # PLOTTING
    # ============================================================

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- Row 1: Reference Vmem pattern snapshots ---
    snap_times = [0, 100, 500, num_total_iters - 1]
    snap_times = [t for t in snap_times if t < num_total_iters]
    for i, t in enumerate(snap_times[:3]):
        ax = fig.add_subplot(gs[0, i])
        vmem_2d = ref_vmem_ts[t, 0, :, 0].numpy().reshape(lattice_dims)
        im = ax.imshow(vmem_2d * 1000, cmap='RdBu_r', aspect='equal')
        ax.set_title(f'Reference Vmem\nt={t}', fontsize=10)
        ax.set_xlabel('col')
        ax.set_ylabel('row')
        plt.colorbar(im, ax=ax, label='mV', shrink=0.8)

    # --- Row 2: Avalanche size distributions ---
    ax_size = fig.add_subplot(gs[1, 0])
    ax_dur = fig.add_subplot(gs[1, 1])
    ax_spatial = fig.add_subplot(gs[1, 2])

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(perturb_times)))
    for i, t_perturb in enumerate(perturb_times):
        r = results_by_time[t_perturb]
        if len(r['sizes']) > 2:
            # Size histogram
            ax_size.hist(r['sizes'], bins=15, alpha=0.5, color=colors[i],
                         label=f't={t_perturb}', edgecolor='k', linewidth=0.5)
            # Duration histogram
            ax_dur.hist(r['durations'], bins=15, alpha=0.5, color=colors[i],
                        label=f't={t_perturb}', edgecolor='k', linewidth=0.5)
            # Spatial extent histogram
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
            # Average activity profile across kick locations
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
    ax_activity.set_ylabel(r'Total $|\Delta V_{mem}|$ (V)')
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

    plt.suptitle(f'Stigmergic Model Avalanche Analysis (kick={args.kick_amplitude*1000:.1f} mV)',
                 fontsize=14, fontweight='bold')
    plt.savefig('data/stigmergic_avalanche_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: data/stigmergic_avalanche_analysis.png")
    print(f"Saved: data/stigmergic_avalanche_results.dat")

    # Power-law analysis
    print(f"\n{'='*60}")
    print("Power-Law Fitting (Clauset et al. 2009)")
    print(f"{'='*60}")
    fit_results = plot_power_law_analysis(results_by_time, perturb_times,
                                          'data/stigmergic_avalanche_powerlaw.png')
    for key, fr in fit_results.items():
        print(f"  {key}:")
        print(f"    alpha = {fr['alpha']:.3f} +/- {fr['sigma']:.3f}")
        print(f"    x_min = {fr['xmin']:.4f}, n_tail = {fr['n_tail']}")
        print(f"    KS stat = {fr['ks_stat']:.4f}, KS p-value = {fr['ks_p']:.3f}")
        if fr['ks_p'] >= 0.1:
            print(f"    --> Power-law PLAUSIBLE (p >= 0.1)")
        else:
            print(f"    --> Power-law REJECTED (p < 0.1)")
    print(f"\nSaved: data/stigmergic_avalanche_powerlaw.png")

    # ============================================================
    # AMPLITUDE SWEEP (optional)
    # ============================================================
    if args.kick_amplitudes_sweep:
        amplitudes = [float(x) for x in args.kick_amplitudes_sweep.split(',')]
        t_probe = perturb_times[0]  # use first perturbation time
        remaining = min(args.response_window, num_total_iters - t_probe)
        ref_window = ref_vmem_ts[t_probe:t_probe + remaining, 0, :, 0].numpy()

        print(f"\n{'='*60}")
        print(f"Amplitude Sweep at t={t_probe}")
        print(f"{'='*60}")

        sweep_sizes = []
        sweep_durations = []
        sweep_spatial = []
        test_cell = kick_locations[0]

        for amp in amplitudes:
            params_fresh = load_stigmergic_model()
            pert_ts = run_perturbed_from_snapshot(
                params_fresh, ref_model, t_probe, remaining,
                [test_cell], amp,
            )
            delta = pert_ts[:, 0, :, 0].numpy() - ref_window
            m = compute_avalanche_metrics(delta, lattice_dims)
            sweep_sizes.append(m['size'])
            sweep_durations.append(m['duration'])
            sweep_spatial.append(m['spatial_extent'])
            print(f"  amp={amp*1000:8.2f} mV: size={m['size']:.4e}, "
                  f"dur={m['duration']:3d}, spatial={m['spatial_extent']:3d}")

        # Plot amplitude scaling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        amps_mV = np.array(amplitudes) * 1000
        ax1.loglog(amps_mV, sweep_sizes, 'o-', markersize=8)
        ax1.set_xlabel('Kick Amplitude (mV)')
        ax1.set_ylabel('Avalanche Size')
        ax1.set_title('Size vs Kick Amplitude')
        ax1.grid(True, alpha=0.3)
        # Fit power law if enough points
        if len(amplitudes) >= 3:
            log_a = np.log(amps_mV)
            log_s = np.log(np.array(sweep_sizes) + 1e-20)
            valid = np.isfinite(log_s)
            if valid.sum() >= 2:
                slope, intercept = np.polyfit(log_a[valid], log_s[valid], 1)
                a_fit = np.linspace(amps_mV.min(), amps_mV.max(), 50)
                ax1.loglog(a_fit, np.exp(intercept) * a_fit**slope, '--',
                           label=f'slope = {slope:.2f}')
                ax1.legend()

        ax2.semilogx(amps_mV, sweep_spatial, 's-', markersize=8, color='C1')
        ax2.set_xlabel('Kick Amplitude (mV)')
        ax2.set_ylabel('Spatial Extent (cells)')
        ax2.set_title('Spatial Extent vs Kick Amplitude')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Amplitude Scaling (cell={test_cell}, t={t_probe})', fontsize=13)
        plt.tight_layout()
        plt.savefig('data/stigmergic_avalanche_amplitude_sweep.png', dpi=150)
        plt.close()
        print(f"\nSaved: data/stigmergic_avalanche_amplitude_sweep.png")


if __name__ == '__main__':
    main()
