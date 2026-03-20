# Criticality analysis of the Stigmergic bioelectric patterning model
#
# Tests whether the field-based pattern formation exhibits signatures of criticality:
# - Order parameter transition as function of gap junction strength
# - Binder cumulant crossing (finite-size scaling)
# - Critical exponents (beta/nu, gamma/nu, eta)
# - Avalanche size/duration distributions
#
# Usage:
#   python criticality_analysis.py --phase coarse      # Phase 1: quick survey
#   python criticality_analysis.py --phase binder      # Phase 2: Binder cumulant crossing
#   python criticality_analysis.py --phase exponents   # Phase 3: critical exponents at g_c
#   python criticality_analysis.py --phase avalanche   # Phase 4: avalanche statistics
#   python criticality_analysis.py --phase all         # Run all phases sequentially

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import label
from cellularFieldNetwork import cellularFieldNetwork

# ============================================================
# CONFIGURATION
# ============================================================

# Physics constants (must match cellularFieldNetwork.py)
E_POL = -55e-3   # reversal potential of hyperpolarizing channel
E_DEP = -5e-3    # reversal potential of depolarizing channel
V_TH = -27e-3    # threshold voltage for order parameter mapping
G_REF = 1.0e-9   # reference conductance

# Order parameter mapping scale (10 mV)
ORDER_PARAM_SCALE = 0.010

# Default field parameters (matching Stigmergic model)
FIELD_PARAMS = {
    'fieldEnabled': True,
    'fieldResolution': 1,
    'fieldStrength': 1.0,
    'fieldAggregation': 'average',
    'fieldScreenSize': 4,
    'fieldRangeSymmetric': False,
    'fieldVector': True,
    'fieldTransductionWeight': torch.DoubleTensor([1000.0]),
    'fieldTransductionBias': torch.DoubleTensor([0.0005]),
    'fieldTransductionGain': torch.DoubleTensor([-1.0]),
    'fieldTransductionTimeConstant': torch.DoubleTensor([10.0]),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_circuit(L, gj_strength, num_samples=1):
    """Create a cellularFieldNetwork with given lattice size and GJ strength."""
    parameters = {
        'fieldParameters': FIELD_PARAMS.copy(),
        'GJParameters': {'GJStrength': gj_strength},
        'GRNParameters': None,
        'ligandParameters': None,
        'ATPParameters': None,
    }
    circuit = cellularFieldNetwork(
        latticeDims=(L, L),
        latticePeriodicBoundary=True,
        parameters=parameters,
        numSamples=num_samples,
    )
    return circuit


def init_random_vmem(circuit, num_samples):
    """Initialize Vmem with random values in [E_pol, E_dep] across samples."""
    num_cells = circuit.numCells
    num_field_pts = circuit.numFieldGridPoints

    vmem_init = (E_DEP - E_POL) * torch.rand(num_samples, num_cells, 1, dtype=torch.float64) + E_POL
    initial_values = {
        'Vmem': vmem_init,
        'eV': torch.zeros(num_samples, num_field_pts, 1, dtype=torch.float64),
        'ligandConc': torch.zeros(num_samples, num_cells, 1, dtype=torch.float64),
    }
    circuit.initVariables(initial_values)

    # Set G_pol to bistable value (1.0 * G_ref) for all cells
    g_pol_init = {
        'cells': [[list(range(num_cells))]] * num_samples,
        'values': [torch.DoubleTensor([1.0])] * num_samples,
    }
    g_dep_init = {'cells': [], 'values': torch.DoubleTensor([])}
    circuit.initParameters({'G_pol': g_pol_init, 'G_dep': g_dep_init})


def compute_order_parameter(vmem):
    """
    Map continuous Vmem to pseudo-spin order parameter.
    m = (1/N) * sum_i tanh((Vmem_i - V_th) / scale)

    Args:
        vmem: tensor of shape (num_samples, num_cells, 1)

    Returns:
        m: array of shape (num_samples,) — order parameter per sample
    """
    sigma = torch.tanh((vmem.squeeze(-1) - V_TH) / ORDER_PARAM_SCALE)
    m = sigma.mean(dim=1).numpy()
    return m


def compute_binder_cumulant(m_samples):
    """U_4 = 1 - <m^4> / (3 * <m^2>^2)"""
    m = np.array(m_samples)
    m2 = np.mean(m**2)
    m4 = np.mean(m**4)
    if m2 < 1e-20:
        return 0.0
    return 1.0 - m4 / (3.0 * m2**2)


def compute_susceptibility(m_samples, N):
    """chi = N * (<m^2> - <|m|>^2)"""
    m = np.array(m_samples)
    return N * (np.mean(m**2) - np.mean(np.abs(m))**2)


def compute_correlation_function(vmem_2d, max_r=None):
    """
    Radial correlation function C(r) from 2D Vmem pattern.
    C(r) = <delta_V(x) * delta_V(x+r)> / <delta_V^2>
    """
    L = vmem_2d.shape[0]
    if max_r is None:
        max_r = L // 2
    delta_V = vmem_2d - vmem_2d.mean()
    var = (delta_V**2).mean()
    if var < 1e-20:
        return np.zeros(max_r), np.arange(max_r)

    C = np.zeros(max_r)
    counts = np.zeros(max_r)
    for dx in range(-max_r + 1, max_r):
        for dy in range(-max_r + 1, max_r):
            r = int(np.sqrt(dx**2 + dy**2))
            if 0 <= r < max_r:
                shifted = np.roll(np.roll(delta_V, dx, axis=0), dy, axis=1)
                C[r] += (delta_V * shifted).mean()
                counts[r] += 1
    mask = counts > 0
    C[mask] /= counts[mask]
    C /= var
    return C, np.arange(max_r)


def run_single_trial(L, gj_strength, num_iters, num_samples=1, save_data=False, stochastic=False):
    """
    Run a single simulation trial and return final Vmem.

    Uses num_samples to batch multiple independent runs with same parameters.

    Returns:
        vmem: tensor of shape (num_samples, num_cells, 1)
        circuit: the cellularFieldNetwork instance (for accessing timeseries if save_data=True)
    """
    circuit = create_circuit(L, gj_strength, num_samples=num_samples)
    init_random_vmem(circuit, num_samples)

    external_inputs = {'gene': None, 'ATP': None}
    circuit.simulate(
        externalInputs=external_inputs,
        numSimIters=num_iters,
        stochasticIonChannels=stochastic,
        fieldModulation=False,
        saveData=save_data,
    )
    return circuit.Vmem.detach().clone(), circuit


def power_law(x, a, alpha):
    """Power law: f(x) = a * x^(-alpha)"""
    return a * x**(-alpha)


def fit_power_law_mle(data, x_min=None):
    """
    Maximum likelihood power-law exponent estimation (Clauset et al. 2009).
    alpha_hat = 1 + n / sum(ln(x_i / x_min))
    """
    data = np.array(data, dtype=float)
    data = data[data > 0]
    if len(data) < 10:
        return np.nan, np.nan
    if x_min is None:
        x_min = np.min(data)
    data = data[data >= x_min]
    n = len(data)
    if n < 5:
        return np.nan, np.nan
    alpha = 1.0 + n / np.sum(np.log(data / x_min))
    sigma = (alpha - 1.0) / np.sqrt(n)
    return alpha, sigma


# ============================================================
# PHASE 1: COARSE PARAMETER SWEEP
# ============================================================

def phase_coarse(args):
    """Sweep GJStrength to find approximate transition region."""
    L = args.lattice_size
    num_iters = args.num_iters
    num_trials = args.num_trials
    gj_values = np.linspace(args.gj_min, args.gj_max, args.num_gj_points)

    print(f"Phase 1: Coarse sweep — L={L}, {num_iters} iters, {num_trials} trials")
    print(f"  GJStrength range: [{args.gj_min:.3f}, {args.gj_max:.3f}], {args.num_gj_points} points")

    mean_abs_m = np.zeros(len(gj_values))
    mean_var_vmem = np.zeros(len(gj_values))
    std_abs_m = np.zeros(len(gj_values))

    for i, gj in enumerate(gj_values):
        # Batch all trials in a single simulation call
        vmem, _ = run_single_trial(L, gj, num_iters, num_samples=num_trials)
        m = compute_order_parameter(vmem)
        mean_abs_m[i] = np.mean(np.abs(m))
        std_abs_m[i] = np.std(np.abs(m))
        vmem_np = vmem.squeeze(-1).numpy()
        mean_var_vmem[i] = np.mean(np.var(vmem_np, axis=1))
        print(f"  GJ={gj:.4f}: <|m|>={mean_abs_m[i]:.4f} +/- {std_abs_m[i]:.4f}, <var(V)>={mean_var_vmem[i]:.2e}")

    # Save results
    results = {
        'gj_values': gj_values,
        'mean_abs_m': mean_abs_m,
        'std_abs_m': std_abs_m,
        'mean_var_vmem': mean_var_vmem,
        'L': L,
        'num_iters': num_iters,
        'num_trials': num_trials,
    }
    torch.save(results, 'data/criticality_coarse_results.dat')

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(gj_values, mean_abs_m, yerr=std_abs_m, fmt='o-', capsize=3)
    ax1.set_xlabel('GJStrength')
    ax1.set_ylabel(r'$\langle |m| \rangle$')
    ax1.set_title(f'Order Parameter (L={L})')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(gj_values, mean_var_vmem, 'o-')
    ax2.set_xlabel('GJStrength')
    ax2.set_ylabel(r'$\langle \mathrm{Var}(V_{\mathrm{mem}}) \rangle$')
    ax2.set_title(f'Vmem Variance (L={L})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/criticality_coarse_sweep.png', dpi=150)
    plt.close()
    print(f"  Saved: data/criticality_coarse_sweep.png")
    return results


# ============================================================
# PHASE 2: BINDER CUMULANT CROSSING
# ============================================================

def phase_binder(args):
    """Compute Binder cumulant and susceptibility across lattice sizes."""
    lattice_sizes = [int(x) for x in args.lattice_sizes.split(',')]
    num_iters = args.num_iters
    num_trials = args.num_trials
    gj_values = np.linspace(args.gj_min, args.gj_max, args.num_gj_points)

    print(f"Phase 2: Binder cumulant — L={lattice_sizes}, {num_iters} iters, {num_trials} trials")

    all_results = {}
    for L in lattice_sizes:
        print(f"\n  L = {L}:")
        binder = np.zeros(len(gj_values))
        suscept = np.zeros(len(gj_values))
        N = L * L

        for i, gj in enumerate(gj_values):
            # Run trials in batches to avoid memory issues
            batch_size = min(num_trials, 50)
            m_all = []
            for batch_start in range(0, num_trials, batch_size):
                batch_n = min(batch_size, num_trials - batch_start)
                vmem, _ = run_single_trial(L, gj, num_iters, num_samples=batch_n)
                m_batch = compute_order_parameter(vmem)
                m_all.extend(m_batch.tolist())

            binder[i] = compute_binder_cumulant(m_all)
            suscept[i] = compute_susceptibility(m_all, N)
            if i % 10 == 0:
                print(f"    GJ={gj:.4f}: U4={binder[i]:.4f}, chi={suscept[i]:.4f}")

        all_results[L] = {
            'binder': binder,
            'susceptibility': suscept,
        }

    all_results['gj_values'] = gj_values
    all_results['lattice_sizes'] = lattice_sizes
    all_results['num_iters'] = num_iters
    all_results['num_trials'] = num_trials
    torch.save(all_results, 'data/criticality_binder_results.dat')

    # Plot Binder cumulant
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for L in lattice_sizes:
        ax1.plot(gj_values, all_results[L]['binder'], 'o-', label=f'L={L}', markersize=3)
    ax1.axhline(y=2/3, color='k', linestyle='--', alpha=0.3, label='U₄=2/3')
    ax1.set_xlabel('GJStrength')
    ax1.set_ylabel(r'$U_4$')
    ax1.set_title('Binder Cumulant')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for L in lattice_sizes:
        ax2.plot(gj_values, all_results[L]['susceptibility'], 'o-', label=f'L={L}', markersize=3)
    ax2.set_xlabel('GJStrength')
    ax2.set_ylabel(r'$\chi$')
    ax2.set_title('Susceptibility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/criticality_binder_crossing.png', dpi=150)
    plt.close()
    print(f"\n  Saved: data/criticality_binder_crossing.png")
    return all_results


# ============================================================
# PHASE 3: CRITICAL EXPONENTS
# ============================================================

def phase_exponents(args):
    """Extract critical exponents at g_c via finite-size scaling."""
    lattice_sizes = [int(x) for x in args.lattice_sizes.split(',')]
    gj_c = args.gj_critical
    num_iters = args.num_iters
    num_trials = args.num_trials

    print(f"Phase 3: Critical exponents at GJ_c={gj_c}")
    print(f"  L = {lattice_sizes}, {num_iters} iters, {num_trials} trials")

    mean_abs_m_list = []
    chi_list = []
    corr_functions = {}

    for L in lattice_sizes:
        N = L * L
        print(f"\n  L = {L}:")
        m_all = []
        corr_accum = None
        corr_count = 0

        batch_size = min(num_trials, 50)
        for batch_start in range(0, num_trials, batch_size):
            batch_n = min(batch_size, num_trials - batch_start)
            vmem, _ = run_single_trial(L, gj_c, num_iters, num_samples=batch_n)
            m_batch = compute_order_parameter(vmem)
            m_all.extend(m_batch.tolist())

            # Compute correlation function for each sample
            for s in range(batch_n):
                vmem_2d = vmem[s, :, 0].numpy().reshape(L, L)
                C_r, r_vals = compute_correlation_function(vmem_2d)
                if corr_accum is None:
                    corr_accum = np.zeros_like(C_r)
                corr_accum += C_r
                corr_count += 1

        mean_abs_m = np.mean(np.abs(m_all))
        chi = compute_susceptibility(m_all, N)
        mean_abs_m_list.append(mean_abs_m)
        chi_list.append(chi)
        corr_functions[L] = corr_accum / corr_count
        print(f"    <|m|> = {mean_abs_m:.4f}, chi = {chi:.4f}")

    # Fit scaling relations
    L_arr = np.array(lattice_sizes, dtype=float)
    m_arr = np.array(mean_abs_m_list)
    chi_arr = np.array(chi_list)

    # <|m|> ~ L^(-beta/nu)
    try:
        log_L = np.log(L_arr)
        log_m = np.log(m_arr)
        slope_m, intercept_m = np.polyfit(log_L, log_m, 1)
        beta_over_nu = -slope_m
        print(f"\n  beta/nu = {beta_over_nu:.3f} (2D Ising: 0.125, MF: 1.0)")
    except Exception:
        beta_over_nu = np.nan

    # chi ~ L^(gamma/nu)
    try:
        log_chi = np.log(chi_arr)
        slope_chi, intercept_chi = np.polyfit(log_L, log_chi, 1)
        gamma_over_nu = slope_chi
        print(f"  gamma/nu = {gamma_over_nu:.3f} (2D Ising: 1.75, MF: 2.0)")
    except Exception:
        gamma_over_nu = np.nan

    results = {
        'lattice_sizes': lattice_sizes,
        'gj_critical': gj_c,
        'mean_abs_m': m_arr,
        'susceptibility': chi_arr,
        'beta_over_nu': beta_over_nu,
        'gamma_over_nu': gamma_over_nu,
        'correlation_functions': corr_functions,
    }
    torch.save(results, 'data/criticality_exponents_results.dat')

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Order parameter scaling
    ax = axes[0, 0]
    ax.loglog(L_arr, m_arr, 'o-', markersize=8)
    if not np.isnan(beta_over_nu):
        L_fit = np.linspace(L_arr.min(), L_arr.max(), 50)
        ax.loglog(L_fit, np.exp(intercept_m) * L_fit**slope_m, '--', label=rf'$\beta/\nu = {beta_over_nu:.3f}$')
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\langle |m| \rangle$')
    ax.set_title(r'Order Parameter Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Susceptibility scaling
    ax = axes[0, 1]
    ax.loglog(L_arr, chi_arr, 's-', markersize=8, color='C1')
    if not np.isnan(gamma_over_nu):
        L_fit = np.linspace(L_arr.min(), L_arr.max(), 50)
        ax.loglog(L_fit, np.exp(intercept_chi) * L_fit**slope_chi, '--', color='C1',
                  label=rf'$\gamma/\nu = {gamma_over_nu:.3f}$')
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\chi$')
    ax.set_title('Susceptibility Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correlation functions
    ax = axes[1, 0]
    for L in lattice_sizes:
        C = corr_functions[L]
        r = np.arange(len(C))
        mask = (r > 0) & (C > 0)
        if mask.any():
            ax.loglog(r[mask], C[mask], 'o-', label=f'L={L}', markersize=4)
    ax.set_xlabel('r')
    ax.set_ylabel('C(r)')
    ax.set_title('Spatial Correlation Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Eta extraction from largest L
    ax = axes[1, 1]
    L_max = max(lattice_sizes)
    C = corr_functions[L_max]
    r = np.arange(len(C))
    mask = (r >= 1) & (r <= L_max // 3) & (C > 0)
    if mask.sum() >= 3:
        try:
            log_r = np.log(r[mask])
            log_C = np.log(C[mask])
            slope_eta, intercept_eta = np.polyfit(log_r, log_C, 1)
            eta = -slope_eta  # C(r) ~ r^(-eta) in 2D (d-2+eta = eta)
            ax.loglog(r[mask], C[mask], 'o', markersize=6, label=f'L={L_max} data')
            r_fit = np.linspace(r[mask].min(), r[mask].max(), 50)
            ax.loglog(r_fit, np.exp(intercept_eta) * r_fit**slope_eta, '--',
                      label=rf'$\eta = {eta:.3f}$ (2D Ising: 0.25)')
            ax.legend()
            print(f"  eta = {eta:.3f} (2D Ising: 0.25, MF: 0)")
            results['eta'] = eta
        except Exception:
            pass
    ax.set_xlabel('r')
    ax.set_ylabel('C(r)')
    ax.set_title(rf'$\eta$ Extraction (L={L_max})')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Critical Exponents at GJ_c = {gj_c:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig('data/criticality_exponents.png', dpi=150)
    plt.close()
    print(f"\n  Saved: data/criticality_exponents.png")
    return results


# ============================================================
# PHASE 4: AVALANCHE STATISTICS
# ============================================================

def detect_avalanches(dVmem_timeseries, threshold_k=2.5, lattice_dims=None):
    """
    Detect avalanches from |dVmem| timeseries using spatiotemporal clustering.

    Args:
        dVmem_timeseries: array of shape (T, N_cells) — voltage change per timestep
        threshold_k: activation threshold in units of std(|dVmem|)
        lattice_dims: (rows, cols) for spatial adjacency

    Returns:
        sizes: list of avalanche sizes (number of active cell-timesteps)
        durations: list of avalanche durations (timesteps)
    """
    T, N = dVmem_timeseries.shape
    abs_dV = np.abs(dVmem_timeseries)

    # Global threshold based on std of |dVmem|
    threshold = threshold_k * abs_dV.std()
    if threshold < 1e-15:
        return [], []

    # Binary activation matrix
    active = abs_dV > threshold  # shape (T, N)

    if lattice_dims is not None:
        L_r, L_c = lattice_dims
        # Reshape to (T, L_r, L_c) for spatial clustering
        active_3d = active.reshape(T, L_r, L_c)

        # Create 3D structure for spatiotemporal connectivity (6-connected: 4 spatial + 2 temporal)
        struct = np.zeros((3, 3, 3), dtype=int)
        struct[1, 1, 0] = 1  # left
        struct[1, 1, 2] = 1  # right
        struct[1, 0, 1] = 1  # up
        struct[1, 2, 1] = 1  # down
        struct[0, 1, 1] = 1  # previous time
        struct[2, 1, 1] = 1  # next time
        struct[1, 1, 1] = 1  # self

        labeled, num_features = label(active_3d, structure=struct)
    else:
        # Pure temporal clustering (no spatial structure)
        labeled, num_features = label(active.reshape(T, 1, N))

    if num_features == 0:
        return [], []

    sizes = []
    durations = []
    for k in range(1, num_features + 1):
        coords = np.where(labeled == k)
        size = len(coords[0])
        duration = coords[0].max() - coords[0].min() + 1
        sizes.append(size)
        durations.append(duration)

    return sizes, durations


def phase_avalanche(args):
    """Compute avalanche statistics at the critical point."""
    L = args.lattice_size
    gj_c = args.gj_critical
    num_iters = args.num_iters_avalanche
    num_runs = args.num_avalanche_runs

    print(f"Phase 4: Avalanche statistics at GJ_c={gj_c}")
    print(f"  L={L}, {num_iters} iters/run, {num_runs} runs, stochastic=True")

    all_sizes = []
    all_durations = []

    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}...", end=' ')
        vmem_final, circuit = run_single_trial(
            L, gj_c, num_iters,
            num_samples=1, save_data=True, stochastic=True,
        )

        # Extract dVmem from timeseries
        vmem_ts = circuit.timeseriesVmem[:, 0, :, 0].numpy()  # shape (T, N)

        # Discard equilibration (first 20%)
        equil = int(0.2 * num_iters)
        vmem_ts = vmem_ts[equil:]

        # Compute dVmem
        dVmem = np.diff(vmem_ts, axis=0)  # shape (T-1, N)

        sizes, durations = detect_avalanches(dVmem, threshold_k=2.5, lattice_dims=(L, L))
        all_sizes.extend(sizes)
        all_durations.extend(durations)
        print(f"found {len(sizes)} avalanches")

    print(f"\n  Total avalanches: {len(all_sizes)}")

    if len(all_sizes) < 10:
        print("  WARNING: Too few avalanches for statistical analysis.")
        print("  Try increasing num_iters_avalanche or adjusting threshold_k.")
        results = {'sizes': all_sizes, 'durations': all_durations, 'L': L, 'gj_c': gj_c}
        torch.save(results, 'data/criticality_avalanche_results.dat')
        return results

    sizes_arr = np.array(all_sizes)
    durations_arr = np.array(all_durations)

    # Fit power laws
    alpha_s, sigma_s = fit_power_law_mle(sizes_arr, x_min=2)
    alpha_t, sigma_t = fit_power_law_mle(durations_arr, x_min=2)
    print(f"  Size exponent: tau_s = {alpha_s:.3f} +/- {sigma_s:.3f}")
    print(f"  Duration exponent: tau_t = {alpha_t:.3f} +/- {sigma_t:.3f}")
    print(f"  (2D Ising: tau_s ~ 1.27; Mean-field: tau_s ~ 1.5)")

    results = {
        'sizes': sizes_arr,
        'durations': durations_arr,
        'tau_s': alpha_s,
        'sigma_s': sigma_s,
        'tau_t': alpha_t,
        'sigma_t': sigma_t,
        'L': L,
        'gj_c': gj_c,
    }
    torch.save(results, 'data/criticality_avalanche_results.dat')

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Size distribution
    s_unique, s_counts = np.unique(sizes_arr, return_counts=True)
    s_freq = s_counts / s_counts.sum()
    ax1.loglog(s_unique, s_freq, 'o', markersize=4, alpha=0.7)
    if not np.isnan(alpha_s):
        s_fit = np.linspace(2, s_unique.max(), 100)
        p_fit = power_law(s_fit, s_freq[0] * 2**alpha_s, alpha_s)
        ax1.loglog(s_fit, p_fit / p_fit.sum() * s_freq.sum(), '--', color='r',
                   label=rf'$\tau_s = {alpha_s:.2f} \pm {sigma_s:.2f}$')
    ax1.set_xlabel('Avalanche Size S')
    ax1.set_ylabel('P(S)')
    ax1.set_title('Avalanche Size Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Duration distribution
    t_unique, t_counts = np.unique(durations_arr, return_counts=True)
    t_freq = t_counts / t_counts.sum()
    ax2.loglog(t_unique, t_freq, 's', markersize=4, alpha=0.7, color='C1')
    if not np.isnan(alpha_t):
        t_fit = np.linspace(2, t_unique.max(), 100)
        p_fit = power_law(t_fit, t_freq[0] * 2**alpha_t, alpha_t)
        ax2.loglog(t_fit, p_fit / p_fit.sum() * t_freq.sum(), '--', color='r',
                   label=rf'$\tau_t = {alpha_t:.2f} \pm {sigma_t:.2f}$')
    ax2.set_xlabel('Avalanche Duration T')
    ax2.set_ylabel('P(T)')
    ax2.set_title('Avalanche Duration Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Avalanche Statistics at GJ_c = {gj_c:.4f}, L={L}', fontsize=14)
    plt.tight_layout()
    plt.savefig('data/criticality_avalanches.png', dpi=150)
    plt.close()
    print(f"  Saved: data/criticality_avalanches.png")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Criticality analysis of Stigmergic bioelectric model')
    parser.add_argument('--phase', type=str, default='coarse',
                        choices=['coarse', 'binder', 'exponents', 'avalanche', 'all'],
                        help='Analysis phase to run')

    # Common parameters
    parser.add_argument('--num_iters', type=int, default=3000,
                        help='Simulation iterations per trial (default: 3000)')
    parser.add_argument('--num_trials', type=int, default=50,
                        help='Number of independent trials per parameter point (default: 50)')

    # Phase 1 parameters
    parser.add_argument('--lattice_size', type=int, default=11,
                        help='Lattice size L for phases 1 and 4 (default: 11)')
    parser.add_argument('--gj_min', type=float, default=0.01,
                        help='Minimum GJStrength (default: 0.01)')
    parser.add_argument('--gj_max', type=float, default=1.0,
                        help='Maximum GJStrength (default: 1.0)')
    parser.add_argument('--num_gj_points', type=int, default=20,
                        help='Number of GJStrength values to sweep (default: 20)')

    # Phase 2/3 parameters
    parser.add_argument('--lattice_sizes', type=str, default='7,11,15,21',
                        help='Comma-separated lattice sizes for phases 2 and 3 (default: 7,11,15,21)')
    parser.add_argument('--gj_critical', type=float, default=0.15,
                        help='Critical GJStrength for phases 3 and 4 (default: 0.15)')

    # Phase 4 parameters
    parser.add_argument('--num_iters_avalanche', type=int, default=20000,
                        help='Iterations per avalanche run (default: 20000)')
    parser.add_argument('--num_avalanche_runs', type=int, default=10,
                        help='Number of independent avalanche runs (default: 10)')

    args = parser.parse_args()

    if args.phase == 'coarse' or args.phase == 'all':
        phase_coarse(args)

    if args.phase == 'binder' or args.phase == 'all':
        phase_binder(args)

    if args.phase == 'exponents' or args.phase == 'all':
        phase_exponents(args)

    if args.phase == 'avalanche' or args.phase == 'all':
        phase_avalanche(args)


if __name__ == '__main__':
    main()
