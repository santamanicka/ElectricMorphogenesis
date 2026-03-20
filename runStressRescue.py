#!/usr/bin/env python3
"""
Run stress-based rescue mechanism: donor stress profile modulates recipient GRN damping.

A donor embryo's stress temporal profile is added to the recipient's GRN damping
(in logit space) and passed through a sigmoid, dynamically modulating GRN weights
at each bioelectric timestep.

    effective_damping(t) = sigmoid(logit(base_damping) + alpha * donor_stress(t))

Performs a pairwise sweep of donor x recipient damping levels and visualizes
the rescue effect as a heatmap plus stress timeseries.

Usage:
    # Quick test with 2 levels
    python runStressRescue.py --dampingLevels "1.0,0.9" --alpha 3.0 --numBioSteps 500

    # Full sweep with learned stress parameters
    python runStressRescue.py --stressParamsFile data/bestLearnedStressParams_5.dat

    # Custom alpha and damping levels
    python runStressRescue.py --alpha 5.0 --dampingLevels "1.0,0.95,0.9,0.5,0.1"
"""

import argparse
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

from embryo import model
from stressBistableSwitch import StressBistableSwitch
import utilities


# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser(description='Run stress-based rescue mechanism')
parser.add_argument('--numBioSteps', type=int, default=2000,
                    help='Number of bioelectric simulation steps (default: 2000)')
parser.add_argument('--numStressSteps', type=int, default=500,
                    help='Additional stress equilibration steps after bio sim (default: 500)')
parser.add_argument('--dampingLevels', type=str, default='1.0,0.95,0.9,0.5,0.1',
                    help='Comma-separated GRN damping levels (default: 1.0,0.95,0.9,0.5,0.1)')
parser.add_argument('--alpha', type=float, default=3.0,
                    help='Rescue rate parameter (default: 3.0)')
parser.add_argument('--stressParamsFile', type=str, default=None,
                    help='Path to learned stress parameters file (.dat)')
parser.add_argument('--outputFile', type=str, default='data/stress_rescue_test.png',
                    help='Output visualization filename')
parser.add_argument('--donor', type=float, default=None,
                    help='Donor damping level for single pair mode')
parser.add_argument('--recipient', type=float, default=None,
                    help='Recipient damping level for single pair mode')
parser.add_argument('--spatioTemporalStress', action='store_true',
                    help='Visualize spatiotemporal stress dynamics for stressed vs unstressed systems')
parser.add_argument('--stressedDamping', type=float, default=0.5,
                    help='GRN damping for the stressed system in spatiotemporalStress mode (default: 0.5)')
parser.add_argument('--unstressedDamping', type=float, default=1.0,
                    help='GRN damping for the unstressed system in spatiotemporalStress mode (default: 1.0)')
args = parser.parse_args()

damping_levels = [float(x) for x in args.dampingLevels.split(',')]


# ============================================================
# Parameter loading (reused from runStressBistableSwitch.py)
# ============================================================
def apply_sigmoid_constraint(raw_param, min_val, max_val):
    """Map unbounded raw parameter to bounded range via sigmoid."""
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)


def load_stress_params(params_file):
    """Load learned stress parameters and fixed Ca2+ parameters from file."""
    print(f"Loading learned stress parameters from: {params_file}")
    data = torch.load(params_file, weights_only=False)

    stress_params = {}
    param_bounds = data.get('parameter_bounds', {})
    raw_params = data.get('parameters', {})

    for param_name, raw_value in raw_params.items():
        min_key = f'{param_name}_min'
        max_key = f'{param_name}_max'
        if min_key in param_bounds and max_key in param_bounds:
            constrained = apply_sigmoid_constraint(
                raw_value, param_bounds[min_key], param_bounds[max_key]
            )
            stress_params[param_name] = float(constrained.item())
        else:
            stress_params[param_name] = float(raw_value.item())

    print(f"  Learned stress parameters ({len(stress_params)}):")
    for name, value in stress_params.items():
        print(f"    {name}: {value:.4f}")

    fixed_ca_params = data.get('fixed_ca_params', None)
    if fixed_ca_params is not None:
        print(f"  Fixed Ca2+ parameters ({len(fixed_ca_params)}):")
        for name, value in fixed_ca_params.items():
            print(f"    {name}: {value:.4f}")
    else:
        print("  WARNING: No fixed_ca_params in file, using defaults")
        fixed_ca_params = get_default_ca_params()

    return stress_params, fixed_ca_params


def get_default_ca_params():
    """Return default Ca2+ parameters (from learned CaMKII file)."""
    return {
        'tau_ca': 2.5964,
        'g_ca': 5.3437,
        'V_half_ca': -0.0753,
        'k_ca': 0.0021,
        'k_decay_ca': 4.3346,
    }


def get_default_stress_params():
    """Return default stress-specific parameters and Ca2+ parameters."""
    stress_params = {
        'tau_S': 50.0,
        'k_on_S': 3.0,
        'k_off_S': 0.02,
        'K_S': 0.4,
        'Ca_stress_threshold': 0.8,
        'sigma_ca': 0.2,
        'gain_S': 2.0,
        'or_threshold_S': 0.6,
        'D_S': 0.15,
        'gamma': 0.08,
        'K_decay': 0.3,
    }
    ca_params = get_default_ca_params()
    return stress_params, ca_params


def load_model_parameters(grn_damping=1.0):
    """Load Model 253 parameters with specified GRN damping."""
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


# ============================================================
# Vmem pattern similarity
# ============================================================
def compute_vmem_similarity(vmem, vmem_ref):
    """
    Compute Pearson correlation between a Vmem pattern and the healthy reference.

    Args:
        vmem: (num_cells,) tensor or numpy array
        vmem_ref: (num_cells,) tensor or numpy array, healthy (damping=1.0) Vmem

    Returns:
        similarity: float in [-1, 1], where 1 = identical pattern
    """
    if isinstance(vmem, torch.Tensor):
        vmem = vmem.detach().cpu().numpy()
    if isinstance(vmem_ref, torch.Tensor):
        vmem_ref = vmem_ref.detach().cpu().numpy()

    vmem_flat = vmem.flatten()
    ref_flat = vmem_ref.flatten()

    # Handle degenerate case (uniform pattern -> zero std)
    if np.std(vmem_flat) < 1e-10 or np.std(ref_flat) < 1e-10:
        # If both are uniform and close, similarity = 1; otherwise 0
        if np.std(vmem_flat) < 1e-10 and np.std(ref_flat) < 1e-10:
            return 1.0 if np.abs(vmem_flat.mean() - ref_flat.mean()) < 1e-6 else 0.0
        return 0.0

    r = np.corrcoef(vmem_flat, ref_flat)[0, 1]
    return float(r)


# ============================================================
# Rescue formula
# ============================================================
def compute_effective_damping(base_damping, donor_stress_t, alpha):
    """
    Compute effective GRN damping at a given timestep.

    effective_damping = sigmoid(logit(base_damping) + alpha * donor_stress_t)

    Args:
        base_damping: float in (0, 1), recipient's native damping level
        donor_stress_t: float in [0, 1], donor's stress signal at time t
        alpha: float, rescue rate parameter

    Returns:
        effective_damping: float in (0, 1)
    """
    base_clamped = max(min(base_damping, 0.999), 0.001)
    base_logit = np.log(base_clamped / (1.0 - base_clamped))
    return 1.0 / (1.0 + np.exp(-(base_logit + alpha * donor_stress_t)))


# ============================================================
# Run bioelectric simulation (static damping — for donors)
# ============================================================
def run_bioelectric_sim(grn_damping, num_bio_steps):
    """Run Model 253 with static GRN damping and return Vmem timeseries."""
    params = load_model_parameters(grn_damping)
    grid_size = params['latticeDims'][0]

    num_samples = params["simParameters"]["numSamples"]
    initial_values = copy.deepcopy(params["simParameters"]["initialValues"])
    external_inputs = copy.deepcopy(params["simParameters"]["externalInputs"])
    clamp_params = copy.deepcopy(params["clampParameters"])

    bio_model = model(params, numBasicSamples=num_samples)
    bio_model.setExperimentalConditions((initial_values, num_samples))

    vmem_timeseries = []
    vmem_timeseries.append(bio_model.electricNetwork.Vmem[0, :, 0].clone())

    for iter_idx in range(num_bio_steps):
        bio_model.simulate(
            externalInputs=external_inputs,
            clampParameters=clamp_params,
            perturbation=None,
            fieldModulation=False,
            numSimIters=1,
            outerIter=iter_idx
        )
        vmem_timeseries.append(bio_model.electricNetwork.Vmem[0, :, 0].clone())

        if (iter_idx + 1) % 500 == 0:
            v = vmem_timeseries[-1]
            print(f"      Iter {iter_idx+1}/{num_bio_steps}: "
                  f"Vmem mean={v.mean().item():.4f}V, std={v.std().item():.4f}V")

    return vmem_timeseries, grid_size


# ============================================================
# Run bioelectric simulation with dynamic damping (for recipients)
# ============================================================
def run_bioelectric_sim_with_dynamic_damping(base_damping, donor_stress_profile,
                                              alpha, num_bio_steps):
    """
    Run Model 253 with time-varying GRN damping based on donor stress profile.

    At each timestep t:
        effective_damping(t) = sigmoid(logit(base_damping) + alpha * donor_stress(t))
        GRN weights are scaled by effective_damping(t) / 1.0 (since model loaded undamped)

    Args:
        base_damping: recipient's native GRN damping level
        donor_stress_profile: list of floats, donor's stress at each timestep
        alpha: rescue rate parameter
        num_bio_steps: number of simulation steps

    Returns:
        vmem_timeseries: list of (num_cells,) tensors
        damping_history: list of effective damping values over time
        grid_size: tissue grid size
    """
    # Load model with FULL (undamped) weights
    params = load_model_parameters(grn_damping=1.0)
    grid_size = params['latticeDims'][0]

    num_samples = params["simParameters"]["numSamples"]
    initial_values = copy.deepcopy(params["simParameters"]["initialValues"])
    external_inputs = copy.deepcopy(params["simParameters"]["externalInputs"])
    clamp_params = copy.deepcopy(params["clampParameters"])

    bio_model = model(params, numBasicSamples=num_samples)
    bio_model.setExperimentalConditions((initial_values, num_samples))

    # Store original undamped weights (after model construction)
    original_tissueGRNWeights = bio_model.geneNetwork.tissueGRNWeights.clone()
    has_ligand_weights = (hasattr(bio_model.electricNetwork, 'GRNtoLigandWeights') and
                          bio_model.electricNetwork.GRNtoLigandWeights is not None and
                          not isinstance(bio_model.electricNetwork.GRNtoLigandWeights, bool))
    if has_ligand_weights:
        original_GRNtoLigandWeights = bio_model.electricNetwork.GRNtoLigandWeights.clone()

    vmem_timeseries = []
    damping_history = []
    vmem_timeseries.append(bio_model.electricNetwork.Vmem[0, :, 0].clone())

    for iter_idx in range(num_bio_steps):
        # Get donor stress at this timestep (clamp to available range)
        t = min(iter_idx, len(donor_stress_profile) - 1)
        donor_stress_t = donor_stress_profile[t]

        # Compute effective damping
        eff_damp = compute_effective_damping(base_damping, donor_stress_t, alpha)
        damping_history.append(eff_damp)

        # Apply dynamic damping to GRN weights
        bio_model.geneNetwork.tissueGRNWeights = original_tissueGRNWeights * eff_damp
        if has_ligand_weights:
            bio_model.electricNetwork.GRNtoLigandWeights = original_GRNtoLigandWeights * eff_damp

        # Run one simulation step
        bio_model.simulate(
            externalInputs=external_inputs,
            clampParameters=clamp_params,
            perturbation=None,
            fieldModulation=False,
            numSimIters=1,
            outerIter=iter_idx
        )
        vmem_timeseries.append(bio_model.electricNetwork.Vmem[0, :, 0].clone())

        if (iter_idx + 1) % 500 == 0:
            v = vmem_timeseries[-1]
            print(f"      Iter {iter_idx+1}/{num_bio_steps}: "
                  f"eff_damp={eff_damp:.4f}, "
                  f"Vmem mean={v.mean().item():.4f}V, std={v.std().item():.4f}V")

    return vmem_timeseries, damping_history, grid_size


# ============================================================
# Run stress system on Vmem timeseries
# ============================================================
def run_stress_system(vmem_timeseries, adjacency_matrix, stress_params,
                      ca_params, num_stress_steps, grid_size, device='cpu'):
    """
    Run stress bistable switch concurrently with Ca2+ on Vmem timeseries.

    Returns:
        dict with stress_history, final_stress, ca_history, final_state
    """
    num_cells = grid_size * grid_size
    dtype = torch.float32

    stress_switch = StressBistableSwitch(
        num_cells=num_cells,
        adjacency_matrix=adjacency_matrix,
        params=ca_params,
        device=device,
        dtype=dtype,
    )
    stress_switch.set_params_from_tensors(
        **{k: torch.tensor(v, dtype=dtype, device=device) for k, v in stress_params.items()}
    )

    dt_ca = 0.01
    dt_stress = 0.1
    stress_history = []
    ca_history = []

    # Concurrent phase: Ca2+ + S evolve together during Vmem drive
    for vmem_flat in vmem_timeseries:
        with torch.no_grad():
            stress_switch.compute_ca_from_vmem(vmem_flat.to(device=device, dtype=dtype), dt_ca)
        ca_history.append(stress_switch.Ca.mean().item())
        ca_now = stress_switch.Ca.detach()
        stress_switch.step(dt_stress, Ca=ca_now)
        stress_history.append(stress_switch.get_embryo_stress().item())

    # Equilibration phase: S continues with final Ca2+ held constant
    ca_final = stress_switch.Ca.detach().clone()
    for t in range(num_stress_steps):
        stress_switch.step(dt_stress, Ca=ca_final)
        stress_history.append(stress_switch.get_embryo_stress().item())

    final_state = stress_switch.get_state()

    return {
        'stress_history': stress_history,
        'ca_history': ca_history,
        'final_stress': stress_switch.get_embryo_stress().item(),
        'final_state': final_state,
    }


# ============================================================
# Spatiotemporal stress analysis
# ============================================================
def run_stress_system_with_snapshots(vmem_timeseries, adjacency_matrix, stress_params,
                                      ca_params, num_stress_steps, grid_size,
                                      n_snapshots=5, device='cpu'):
    """
    Like run_stress_system but also captures per-cell S and Ca tensors at
    n_snapshot timepoints spread evenly across the full concurrent + equilibration
    duration.

    Returns same keys as run_stress_system plus:
        'S_snapshots'    : list of n_snapshots (num_cells,) tensors
        'Ca_snapshots'   : list of n_snapshots (num_cells,) tensors
        'snapshot_steps' : list of int step indices where snapshots were taken
        'n_concurrent'   : int, number of concurrent-phase steps (= len(vmem_timeseries))
    """
    num_cells = grid_size * grid_size
    dtype = torch.float32

    stress_switch = StressBistableSwitch(
        num_cells=num_cells,
        adjacency_matrix=adjacency_matrix,
        params=ca_params,
        device=device,
        dtype=dtype,
    )
    stress_switch.set_params_from_tensors(
        **{k: torch.tensor(v, dtype=dtype, device=device) for k, v in stress_params.items()}
    )

    dt_ca = 0.01
    dt_stress = 0.1
    stress_history = []
    ca_history = []

    n_concurrent = len(vmem_timeseries)
    total_steps = n_concurrent + num_stress_steps

    # Evenly-spaced snapshot indices, always including step 0 and the last step
    if n_snapshots >= total_steps:
        snap_set = set(range(total_steps))
    else:
        snap_set = set([0,110,300,500,1999])
        # snap_set = set(
        #     int(round(i * (total_steps - 1) / (n_snapshots - 1)))
        #     for i in range(n_snapshots)
        # )

    S_snapshots = []
    Ca_snapshots = []
    snapshot_steps = []
    step = 0

    # --- Concurrent phase: Ca + S evolve with Vmem drive ---
    for vmem_flat in vmem_timeseries:
        with torch.no_grad():
            stress_switch.compute_ca_from_vmem(vmem_flat.to(device=device, dtype=dtype), dt_ca)
        ca_history.append(stress_switch.Ca.mean().item())
        ca_now = stress_switch.Ca.detach()
        stress_switch.step(dt_stress, Ca=ca_now)
        stress_history.append(stress_switch.get_embryo_stress().item())

        if step in snap_set:
            S_snapshots.append(stress_switch.S.detach().cpu().clone())
            Ca_snapshots.append(stress_switch.Ca.detach().cpu().clone())
            snapshot_steps.append(step)
        step += 1

    # --- Equilibration phase: S continues with Ca held constant ---
    ca_final = stress_switch.Ca.detach().clone()
    for _ in range(num_stress_steps):
        stress_switch.step(dt_stress, Ca=ca_final)
        stress_history.append(stress_switch.get_embryo_stress().item())

        if step in snap_set:
            S_snapshots.append(stress_switch.S.detach().cpu().clone())
            Ca_snapshots.append(stress_switch.Ca.detach().cpu().clone())
            snapshot_steps.append(step)
        step += 1

    final_state = stress_switch.get_state()

    return {
        'stress_history': stress_history,
        'ca_history': ca_history,
        'final_stress': stress_switch.get_embryo_stress().item(),
        'final_state': final_state,
        'S_snapshots': S_snapshots,
        'Ca_snapshots': Ca_snapshots,
        'snapshot_steps': snapshot_steps,
        'n_concurrent': n_concurrent,
    }


def run_spatiotemporalStress(stressed_d, unstressed_d, stress_params, ca_params,
                       adjacency_matrix, num_bio_steps, num_stress_steps,
                       grid_size, n_snapshots=5, device='cpu'):
    """Run stressed and unstressed conditions, collecting spatiotemporal stress snapshots."""
    results = {}
    for label, damping in [('unstressed', unstressed_d), ('stressed', stressed_d)]:
        print(f"\n[{label.upper()}] damping={damping:.1f}: running bioelectric sim...")
        vmem_ts, _ = run_bioelectric_sim(damping, num_bio_steps)
        print(f"[{label.upper()}] running stress system with snapshots...")
        results[label] = run_stress_system_with_snapshots(
            vmem_timeseries=vmem_ts,
            adjacency_matrix=adjacency_matrix,
            stress_params=stress_params,
            ca_params=ca_params,
            num_stress_steps=num_stress_steps,
            grid_size=grid_size,
            n_snapshots=n_snapshots,
            device=device,
        )
        print(f"[{label.upper()}] final stress: {results[label]['final_stress']:.4f}")
    return results


def visualize_spatiotemporalStress(results, stressed_d, unstressed_d, grid_size, output_path):
    """
    Figure layout (4 heatmap rows + 1 timeseries row):

        Row 0  Ca²⁺  unstressed   | t0 | t1 | t2 | t3 | t4 |
        Row 1  Ca²⁺  stressed     | t0 | t1 | t2 | t3 | t4 |
        Row 2  Stress S unstressed| t0 | t1 | t2 | t3 | t4 |
        Row 3  Stress S stressed  | t0 | t1 | t2 | t3 | t4 |
        Row 4  [mean Ca timeseries (L) | mean S timeseries (R)]

    Concurrent phase (Vmem-driven) and equilibration phase are separated by a
    dashed vertical line in the timeseries panels.
    """
    unstressed = results['unstressed']
    stressed   = results['stressed']

    n_snaps       = len(unstressed['S_snapshots'])
    snap_steps    = unstressed['snapshot_steps']   # same for both (identical params)
    n_concurrent  = unstressed['n_concurrent']

    # ---- Shared colour scales ----
    ca_arrays = [t.numpy() for t in unstressed['Ca_snapshots'] + stressed['Ca_snapshots']]
    ca_min = min(a.min() for a in ca_arrays)
    ca_max = max(a.max() for a in ca_arrays)
    s_min, s_max = 0.0, 1.0   # S is clamped to [0, 1]

    # ---- Figure & GridSpec (via fig.add_gridspec — no separate import needed) ----
    fig = plt.figure(figsize=(n_snaps * 2.6 + 0.6, 15))
    gs  = fig.add_gridspec(5, n_snaps,
                           height_ratios=[1, 1, 1, 1, 1.5],
                           hspace=0.50, wspace=0.22,
                           left=0.10, right=0.97, top=0.93, bottom=0.05)

    row_meta = [
        ('Ca²⁺',     f'unstressed  d={unstressed_d:.1f}', 'YlOrRd',
         unstressed['Ca_snapshots'], ca_min, ca_max),
        ('Ca²⁺',     f'stressed  d={stressed_d:.1f}',     'YlOrRd',
         stressed['Ca_snapshots'],   ca_min, ca_max),
        ('Stress S', f'unstressed  d={unstressed_d:.1f}', 'hot',
         unstressed['S_snapshots'],  s_min,  s_max),
        ('Stress S', f'stressed  d={stressed_d:.1f}',     'hot',
         stressed['S_snapshots'],    s_min,  s_max),
    ]

    # ---- Heatmap rows 0–3 ----
    for row_idx, (var_label, cond_label, cmap, snaps, vmin, vmax) in enumerate(row_meta):
        for col, (snap, step) in enumerate(zip(snaps, snap_steps)):
            ax = fig.add_subplot(gs[row_idx, col])
            im = ax.imshow(snap.numpy().reshape(grid_size, grid_size),
                           cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])

            # Column title: show step number + phase tag on first row only
            if row_idx == 0:
                phase = 'drive' if step < n_concurrent else 'equil'
                ax.set_title(f'step {step}\n({phase})', fontsize=9)

            # Row label on leftmost column
            if col == 0:
                ax.set_ylabel(f'{var_label}\n{cond_label}', fontsize=10, labelpad=5)

            # Colorbar on rightmost column
            if col == n_snaps - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Timeseries row 4 ----
    split = n_snaps // 2                        # columns for Ca panel
    ax_ca = fig.add_subplot(gs[4, :split])
    ax_s  = fig.add_subplot(gs[4, (split+1):])

    # Ca history: concurrent phase only; extend flat through equilibration
    def full_ca(data):
        h = data['ca_history']
        n_eq = len(data['stress_history']) - len(h)
        return h + ([h[-1]] * n_eq if h else [])

    colors = {'unstressed': 'steelblue', 'stressed': 'firebrick'}
    for label, data in [('unstressed', unstressed), ('stressed', stressed)]:
        c = colors[label]
        t = list(range(len(data['stress_history'])))
        ax_ca.plot(full_ca(data), color=c, lw=1.5, label=label)
        ax_s.plot(t, data['stress_history'], color=c, lw=1.5, label=label)

    for ax, ylabel, title in [
        (ax_ca, 'Ca²⁺ (mean)',   'Mean Ca²⁺ over time'),
        (ax_s,  'S (mean)',       'Mean Stress S over time'),
    ]:
        ax.axvline(n_concurrent, color='gray', lw=0.9, ls='--', label='equil. start')
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f'Spatiotemporal Stress Dynamics\n'
        f'Unstressed (teratogen={1-unstressed_d:.1f})  vs  Stressed (teratogen={1-stressed_d:.1f})',
        fontsize=12, fontweight='bold',
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved spatiotemporal visualization: {output_path}")


# ============================================================
# Pairwise rescue sweep
# ============================================================
def run_pairwise_sweep(damping_levels, alpha, stress_params, ca_params,
                        adjacency_matrix, num_bio_steps, num_stress_steps,
                        grid_size, device='cpu'):
    """
    Run ALL donor-recipient pairs.

    Phase 1: Compute donor stress profiles (static damping bio sim + stress system)
    Phase 2: For each (donor, recipient) pair, run rescued bio sim + stress system

    Returns:
        results: dict keyed by (donor_d, recipient_d) with rescue data
        donor_data: dict keyed by damping level with donor baseline data
    """
    num_levels = len(damping_levels)

    # ---- Phase 1: Donor stress profiles ----
    print("\n" + "=" * 60)
    print("PHASE 1: Computing donor stress profiles (static damping)")
    print("=" * 60)

    donor_data = {}
    for i, damping in enumerate(damping_levels):
        print(f"\n  [{i+1}/{num_levels}] Donor damping = {damping:.2f}")
        print(f"    Running bioelectric simulation...")
        vmem_ts, _ = run_bioelectric_sim(damping, num_bio_steps)

        print(f"    Running stress bistable switch...")
        stress_result = run_stress_system(
            vmem_timeseries=vmem_ts,
            adjacency_matrix=adjacency_matrix,
            stress_params=stress_params,
            ca_params=ca_params,
            num_stress_steps=num_stress_steps,
            grid_size=grid_size,
            device=device,
        )

        donor_data[damping] = {
            'vmem_timeseries': vmem_ts,
            'vmem_final': vmem_ts[-1].detach().cpu(),
            'stress_history': stress_result['stress_history'],
            'final_stress': stress_result['final_stress'],
            'final_state': stress_result['final_state'],
        }
        print(f"    Final stress: {stress_result['final_stress']:.4f}")

    # ---- Reference Vmem: healthy (damping=1.0) pattern for similarity ----
    ref_damping = max(damping_levels)  # highest damping = healthiest
    vmem_ref = donor_data[ref_damping]['vmem_final']
    print(f"\n  Reference Vmem for similarity: damping={ref_damping:.2f}")

    # ---- Precompute unrescued baseline Vmem similarity for each damping level ----
    baseline_vmem_sims = {}
    for d in damping_levels:
        baseline_vmem_sims[d] = compute_vmem_similarity(
            donor_data[d]['vmem_final'], vmem_ref
        )
        print(f"    Baseline Vmem similarity (damping={d:.2f}): {baseline_vmem_sims[d]:.4f}")

    # ---- Phase 2: Pairwise rescue ----
    print("\n" + "=" * 60)
    print(f"PHASE 2: Pairwise rescue sweep ({num_levels}x{num_levels} = {num_levels**2} pairs)")
    print(f"  alpha = {alpha}")
    print("=" * 60)

    results = {}
    pair_count = 0
    total_pairs = num_levels * num_levels

    # Compute ALL pairs (including diagonal — donor==recipient still rescues)
    for recip_d in damping_levels:
        for donor_d in damping_levels:
            pair_count += 1
            is_diag = (donor_d == recip_d)
            diag_tag = " (SELF-RESCUE)" if is_diag else ""

            print(f"\n  [{pair_count}/{total_pairs}] donor={donor_d:.2f} -> recipient={recip_d:.2f}{diag_tag}")

            # Get donor stress profile
            donor_profile = donor_data[donor_d]['stress_history']

            # Run recipient bio sim with dynamic damping
            print(f"    Running rescued bioelectric simulation...")
            rescued_vmem_ts, damping_hist, _ = run_bioelectric_sim_with_dynamic_damping(
                base_damping=recip_d,
                donor_stress_profile=donor_profile,
                alpha=alpha,
                num_bio_steps=num_bio_steps,
            )

            # Run stress system on rescued Vmem
            print(f"    Running stress system on rescued Vmem...")
            rescued_stress_result = run_stress_system(
                vmem_timeseries=rescued_vmem_ts,
                adjacency_matrix=adjacency_matrix,
                stress_params=stress_params,
                ca_params=ca_params,
                num_stress_steps=num_stress_steps,
                grid_size=grid_size,
                device=device,
            )

            rescued_vmem_final = rescued_vmem_ts[-1].detach().cpu()
            sim = compute_vmem_similarity(rescued_vmem_final, vmem_ref)
            baseline_sim = baseline_vmem_sims[recip_d]
            sim_gain = sim - baseline_sim

            results[(donor_d, recip_d)] = {
                'final_stress': rescued_stress_result['final_stress'],
                'stress_history': rescued_stress_result['stress_history'],
                'damping_history': damping_hist,
                'vmem_final': rescued_vmem_final,
                'vmem_similarity': sim,
                'baseline_vmem_similarity': baseline_sim,
                'vmem_similarity_gain': sim_gain,
                'is_baseline': is_diag,
            }

            baseline_stress = donor_data[recip_d]['final_stress']
            rescued_stress = rescued_stress_result['final_stress']
            stress_delta = baseline_stress - rescued_stress
            print(f"    Rescued stress: {rescued_stress:.4f} (unrescued: {baseline_stress:.4f}, delta: {stress_delta:+.4f})")
            print(f"    Vmem similarity: {sim:.4f} (unrescued: {baseline_sim:.4f}, gain: {sim_gain:+.4f})")

    return results, donor_data


# ============================================================
# Visualization
# ============================================================
def visualize_rescue(results, donor_data, damping_levels, alpha, output_path, vmem_ref=None):
    """
    6-panel visualization:
    Row 1: Similarity heatmap, Stress heatmap, Vmem similarity bar chart
    Row 2: Stress timeseries, Effective damping curves, Rescue magnitude bar chart
    """
    num_levels = len(damping_levels)
    # Compute vmem_ref if not provided
    if vmem_ref is None:
        ref_damping = max(damping_levels)
        vmem_ref = donor_data[ref_damping]['vmem_final']
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)
    most_stressed = min(damping_levels)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_levels))

    # ---- Panel 1: Vmem similarity GAIN heatmap ----
    ax1 = fig.add_subplot(gs[0, 0])
    gain_matrix = np.zeros((num_levels, num_levels))
    for i, recip_d in enumerate(damping_levels):
        for j, donor_d in enumerate(damping_levels):
            gain_matrix[i, j] = results[(donor_d, recip_d)]['vmem_similarity_gain']

    gain_max = max(abs(gain_matrix.min()), abs(gain_matrix.max()), 0.1)
    im1 = ax1.imshow(gain_matrix, cmap='RdYlGn', vmin=-gain_max, vmax=gain_max, aspect='equal')
    ax1.set_xticks(range(num_levels))
    ax1.set_xticklabels([f'{d:.2f}' for d in damping_levels], fontsize=9)
    ax1.set_yticks(range(num_levels))
    ax1.set_yticklabels([f'{d:.2f}' for d in damping_levels], fontsize=9)
    ax1.set_xlabel('Donor Damping', fontsize=11)
    ax1.set_ylabel('Recipient Damping', fontsize=11)
    ax1.set_title(f'Vmem Similarity Gain\n(rescued - unrescued, alpha={alpha:.1f})',
                  fontsize=12, fontweight='bold')
    for i in range(num_levels):
        for j in range(num_levels):
            val = gain_matrix[i, j]
            color = 'white' if abs(val) > gain_max * 0.6 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax1.text(j, i, f'{val:+.2f}', ha='center', va='center',
                     fontsize=9, color=color, fontweight=weight)
    plt.colorbar(im1, ax=ax1, fraction=0.046, label='Similarity Gain')

    # ---- Panel 2: Raw Vmem similarity heatmap ----
    ax2 = fig.add_subplot(gs[0, 1])
    sim_matrix = np.zeros((num_levels, num_levels))
    for i, recip_d in enumerate(damping_levels):
        for j, donor_d in enumerate(damping_levels):
            sim_matrix[i, j] = results[(donor_d, recip_d)]['vmem_similarity']

    im2 = ax2.imshow(sim_matrix, cmap='RdYlGn', vmin=-0.2, vmax=1, aspect='equal')
    ax2.set_xticks(range(num_levels))
    ax2.set_xticklabels([f'{d:.2f}' for d in damping_levels], fontsize=9)
    ax2.set_yticks(range(num_levels))
    ax2.set_yticklabels([f'{d:.2f}' for d in damping_levels], fontsize=9)
    ax2.set_xlabel('Donor Damping', fontsize=11)
    ax2.set_ylabel('Recipient Damping', fontsize=11)
    ax2.set_title(f'Vmem Similarity to Healthy\n(Pearson r, alpha={alpha:.1f})',
                  fontsize=12, fontweight='bold')
    for i in range(num_levels):
        for j in range(num_levels):
            val = sim_matrix[i, j]
            color = 'white' if val < 0.4 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=9, color=color, fontweight=weight)
    plt.colorbar(im2, ax=ax2, fraction=0.046, label='Pearson r')

    # ---- Panel 3: Similarity gain bar chart per recipient ----
    ax3 = fig.add_subplot(gs[0, 2])
    recipient_labels = []
    baseline_sims = []
    best_gains = []
    best_gain_donors = []

    for recip_d in damping_levels:
        baseline_sim = results[(damping_levels[0], recip_d)]['baseline_vmem_similarity']
        best_gain = -float('inf')
        best_donor = None
        for donor_d in damping_levels:
            gain = results[(donor_d, recip_d)]['vmem_similarity_gain']
            if gain > best_gain:
                best_gain = gain
                best_donor = donor_d
        recipient_labels.append(f'{recip_d:.2f}')
        baseline_sims.append(baseline_sim)
        best_gains.append(best_gain)
        best_gain_donors.append(best_donor)

    x = np.arange(len(recipient_labels))
    width = 0.5
    bar_colors = ['#2ECC71' if g > 0.005 else ('#E74C3C' if g < -0.005 else '#95A5A6')
                  for g in best_gains]
    ax3.bar(x, best_gains, width, color=bar_colors, alpha=0.8)
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_xlabel('Recipient Damping')
    ax3.set_ylabel('Best Similarity Gain')
    ax3.set_title('Best Rescue Gain per Recipient', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(recipient_labels)
    ax3.grid(alpha=0.3, axis='y')
    for i, (bd, bg) in enumerate(zip(best_gain_donors, best_gains)):
        if abs(bg) > 0.005:
            va = 'bottom' if bg > 0 else 'top'
            offset = 0.02 if bg > 0 else -0.02
            ax3.text(i, bg + offset, f'd={bd:.2f}\n{bg:+.2f}',
                     ha='center', va=va, fontsize=7, color='black')

    # ---- Panel 4: Stress timeseries for most stressed recipient ----
    ax4 = fig.add_subplot(gs[1, 0])
    for j, donor_d in enumerate(damping_levels):
        history = results[(donor_d, most_stressed)]['stress_history']
        is_baseline = results[(donor_d, most_stressed)]['is_baseline']
        style = '--' if is_baseline else '-'
        label_suffix = ' (baseline)' if is_baseline else ''
        ax4.plot(history, color=colors[j], linestyle=style, linewidth=1.5,
                 label=f'donor={donor_d:.2f}{label_suffix}')
    num_vmem = len(donor_data[most_stressed]['stress_history']) - args.numStressSteps
    ax4.axvline(num_vmem, color='gray', linestyle=':', alpha=0.5, label='Vmem drive end')
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Mean Stress')
    ax4.set_title(f'Stress Timeseries (recipient={most_stressed:.2f})',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper left')
    ax4.grid(alpha=0.3)
    ax4.set_ylim(-0.05, 1.05)

    # ---- Panel 5: Effective damping curves ----
    ax5 = fig.add_subplot(gs[1, 1])
    for j, donor_d in enumerate(damping_levels):
        damp_hist = results[(donor_d, most_stressed)]['damping_history']
        style = '--' if donor_d == most_stressed else '-'
        label_suffix = ' (self)' if donor_d == most_stressed else ''
        ax5.plot(damp_hist, color=colors[j], linewidth=1.5, linestyle=style,
                 label=f'donor={donor_d:.2f}{label_suffix}')
    ax5.axhline(most_stressed, color='red', linestyle='--', alpha=0.7,
                label=f'base damping={most_stressed:.2f}')
    ax5.set_xlabel('Bio Timestep')
    ax5.set_ylabel('Effective GRN Damping')
    ax5.set_title(f'Dynamic Damping (recipient={most_stressed:.2f}, alpha={alpha:.1f})',
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    ax5.set_ylim(-0.05, 1.05)

    # ---- Panel 6: Rescue magnitude bar chart (stress) ----
    ax6 = fig.add_subplot(gs[1, 2])
    recipient_labels2 = []
    baseline_stresses = []
    best_rescued_stresses = []
    best_stress_donors = []

    for recip_d in damping_levels:
        # Unrescued baseline: static damping from Phase 1 (no donor)
        baseline = donor_data[recip_d]['final_stress']
        best_stress = float('inf')
        best_donor = None
        for donor_d in damping_levels:
            rescued = results[(donor_d, recip_d)]['final_stress']
            if rescued < best_stress:
                best_stress = rescued
                best_donor = donor_d
        recipient_labels2.append(f'{recip_d:.2f}')
        baseline_stresses.append(baseline)
        best_rescued_stresses.append(best_stress)
        best_stress_donors.append(best_donor)

    x2 = np.arange(len(recipient_labels2))
    ax6.bar(x2 - width/2, baseline_stresses, width, label='Baseline', color='#E74C3C', alpha=0.8)
    ax6.bar(x2 + width/2, best_rescued_stresses, width, label='Best Rescue', color='#2ECC71', alpha=0.8)
    ax6.set_xlabel('Recipient Damping')
    ax6.set_ylabel('Final Stress')
    ax6.set_title('Stress Reduction: Baseline vs Best Donor', fontsize=12, fontweight='bold')
    ax6.set_xticks(x2)
    ax6.set_xticklabels(recipient_labels2)
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3, axis='y')
    ax6.set_ylim(0, 1.1)
    for i, (bd, bs, br) in enumerate(zip(best_stress_donors, baseline_stresses, best_rescued_stresses)):
        if bs - br > 0.005:
            delta = bs - br
            ax6.text(i + width/2, br + 0.03, f'd={bd:.1f}\n-{delta:.2f}',
                     ha='center', va='bottom', fontsize=7, color='green')

    plt.suptitle(f'Stress-Based Rescue Mechanism (alpha={alpha:.1f})',
                 fontsize=14, fontweight='bold', y=0.99)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")


# ============================================================
# Single pair mode
# ============================================================
def run_single_pair_mode(donor_d, recip_d, alpha, stress_params, ca_params,
                         adjacency_matrix, num_bio_steps, num_stress_steps,
                         grid_size, device='cpu'):
    """
    Run a single donor → recipient rescue pair.

    Returns a dict with initial/final Vmem for both the unrescued and rescued cases,
    plus the donor stress profile and effective damping history.
    """
    # --- Donor: static damping bio sim → stress profile ---
    print(f"\n[Donor] damping={donor_d:.2f}: running bioelectric simulation...")
    donor_vmem_ts, _ = run_bioelectric_sim(donor_d, num_bio_steps)

    print(f"[Donor] damping={donor_d:.2f}: running stress bistable switch...")
    donor_stress_result = run_stress_system(
        vmem_timeseries=donor_vmem_ts,
        adjacency_matrix=adjacency_matrix,
        stress_params=stress_params,
        ca_params=ca_params,
        num_stress_steps=num_stress_steps,
        grid_size=grid_size,
        device=device,
    )
    donor_stress_profile = donor_stress_result['stress_history']
    print(f"[Donor] final stress: {donor_stress_result['final_stress']:.4f}")

    # --- Recipient unrescued: static damping ---
    print(f"\n[Unrescued] recipient damping={recip_d:.2f}: running bioelectric simulation...")
    recip_vmem_ts, _ = run_bioelectric_sim(recip_d, num_bio_steps)
    initial_vmem = recip_vmem_ts[0].detach().cpu()
    final_vmem_unrescued = recip_vmem_ts[-1].detach().cpu()

    unrescued_stress_result = run_stress_system(
        vmem_timeseries=recip_vmem_ts,
        adjacency_matrix=adjacency_matrix,
        stress_params=stress_params,
        ca_params=ca_params,
        num_stress_steps=num_stress_steps,
        grid_size=grid_size,
        device=device,
    )
    print(f"[Unrescued] final stress: {unrescued_stress_result['final_stress']:.4f}")

    # --- Recipient rescued: dynamic damping from donor ---
    print(f"\n[Rescued] donor={donor_d:.2f} -> recipient={recip_d:.2f}: "
          f"running bioelectric simulation with dynamic damping...")
    rescued_vmem_ts, damping_hist, _ = run_bioelectric_sim_with_dynamic_damping(
        base_damping=recip_d,
        donor_stress_profile=donor_stress_profile,
        alpha=alpha,
        num_bio_steps=num_bio_steps,
    )
    final_vmem_rescued = rescued_vmem_ts[-1].detach().cpu()

    rescued_stress_result = run_stress_system(
        vmem_timeseries=rescued_vmem_ts,
        adjacency_matrix=adjacency_matrix,
        stress_params=stress_params,
        ca_params=ca_params,
        num_stress_steps=num_stress_steps,
        grid_size=grid_size,
        device=device,
    )
    print(f"[Rescued]  final stress: {rescued_stress_result['final_stress']:.4f}")

    return {
        'initial_vmem': initial_vmem,
        'final_vmem_unrescued': final_vmem_unrescued,
        'final_vmem_rescued': final_vmem_rescued,
        'unrescued_stress': unrescued_stress_result['final_stress'],
        'rescued_stress': rescued_stress_result['final_stress'],
        'donor_stress_profile': donor_stress_profile,
        'damping_history': damping_hist,
    }


def visualize_single_pair(data, donor_d, recip_d, alpha, grid_size, output_path):
    """
    2-row × 2-column figure showing initial and final Vmem patterns.

    Row 1: without rescue  (recipient static damping = recip_d)
    Row 2: with rescue     (donor static damping = donor_d, dynamic coupling)
    """
    initial = data['initial_vmem'].numpy().reshape(grid_size, grid_size)
    unrescued = data['final_vmem_unrescued'].numpy().reshape(grid_size, grid_size)
    rescued = data['final_vmem_rescued'].numpy().reshape(grid_size, grid_size)

    # Shared colour scale across all panels
    vmin = min(initial.min(), unrescued.min(), rescued.min())
    vmax = max(initial.max(), unrescued.max(), rescued.max())

    fig, axes = plt.subplots(2, 2, figsize=(9, 8),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.35})

    def _plot(ax, arr, title, xlabel=''):
        im = ax.imshow(arr, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Vmem (V)')
        return im

    # --- Row 1: no rescue ---
    _plot(axes[0, 0], initial,
          'Initial Vmem')
    _plot(axes[0, 1], unrescued,
          f'Final Vmem (no rescue)\nstress={data["unrescued_stress"]:.3f}')

    # --- Row 2: with rescue ---
    _plot(axes[1, 0], initial,
          'Initial Vmem')
    _plot(axes[1, 1], rescued,
          f'Final Vmem (rescued)\nstress={data["rescued_stress"]:.3f}')

    # Row labels on left side
    for row_idx, label in enumerate(['No rescue', 'With rescue']):
        axes[row_idx, 0].set_ylabel(label, fontsize=12, fontweight='bold',
                                    rotation=90, labelpad=8)

    stress_delta = data['rescued_stress'] - data['unrescued_stress']
    fig.suptitle(
        f'Stress-Based Rescue  |  Teratogen: donor={1-donor_d:.2f}  →  recipient={1-recip_d:.2f}'
        f'  |  signal strength={alpha:.1f}  |  stress delta={stress_delta:+.3f}',
        fontsize=12, fontweight='bold', y=1.01
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device('cpu')

    print("=" * 60)
    print("STRESS-BASED RESCUE MECHANISM")
    print("=" * 60)
    print(f"Bioelectric steps: {args.numBioSteps}")
    print(f"Stress equilibration steps: {args.numStressSteps}")
    print(f"Damping levels: {damping_levels}")
    print(f"Rescue alpha: {args.alpha}")

    # Load stress parameters
    if args.stressParamsFile is not None:
        stress_params, ca_params = load_stress_params(args.stressParamsFile)
    else:
        stress_params, ca_params = get_default_stress_params()
        print("\nUsing DEFAULT stress parameters:")
        for name, val in stress_params.items():
            print(f"  {name}: {val}")
        print("\nUsing DEFAULT Ca2+ parameters:")
        for name, val in ca_params.items():
            print(f"  {name}: {val}")

    # Compute adjacency matrix
    test_params = load_model_parameters(1.0)
    grid_size = test_params['latticeDims'][0]
    del test_params

    utils = utilities.utilities()
    adjacency_matrix = utils.computeLatticeAdjacencyMatrix(
        latticeDims=(grid_size, grid_size), periodicBoundary=False
    )
    print(f"\nGrid size: {grid_size}x{grid_size}")

    # ---- Spatiotemporal mode ----
    if args.spatioTemporalStress:
        print(f"\nSpatiotemporalStress mode: "
              f"unstressed d={args.unstressedDamping:.1f}, "
              f"stressed d={args.stressedDamping:.1f}")
        st_results = run_spatiotemporalStress(
            stressed_d=args.stressedDamping,
            unstressed_d=args.unstressedDamping,
            stress_params=stress_params,
            ca_params=ca_params,
            adjacency_matrix=adjacency_matrix,
            num_bio_steps=args.numBioSteps,
            num_stress_steps=args.numStressSteps,
            grid_size=grid_size,
            device=device,
        )
        visualize_spatiotemporalStress(st_results, args.stressedDamping, args.unstressedDamping,
                                 grid_size, args.outputFile)
        print(f"\nDone!")
        return st_results

    # ---- Single pair mode ----
    if args.donor is not None and args.recipient is not None:
        print(f"\nSingle pair mode: donor={args.donor:.2f}, recipient={args.recipient:.2f}")
        data = run_single_pair_mode(
            donor_d=args.donor,
            recip_d=args.recipient,
            alpha=args.alpha,
            stress_params=stress_params,
            ca_params=ca_params,
            adjacency_matrix=adjacency_matrix,
            num_bio_steps=args.numBioSteps,
            num_stress_steps=args.numStressSteps,
            grid_size=grid_size,
            device=device,
        )
        visualize_single_pair(data, args.donor, args.recipient, args.alpha,
                              grid_size, args.outputFile)
        print(f"\nDone!")
        return data

    # ---- Pairwise sweep mode ----
    results, donor_data = run_pairwise_sweep(
        damping_levels=damping_levels,
        alpha=args.alpha,
        stress_params=stress_params,
        ca_params=ca_params,
        adjacency_matrix=adjacency_matrix,
        num_bio_steps=args.numBioSteps,
        num_stress_steps=args.numStressSteps,
        grid_size=grid_size,
        device=device,
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    # Compute vmem_ref for summary
    ref_damping = max(damping_levels)
    vmem_ref_summary = donor_data[ref_damping]['vmem_final']

    print(f"\nUnrescued baseline (no donor, static damping):")
    for d in damping_levels:
        sim = compute_vmem_similarity(donor_data[d]['vmem_final'], vmem_ref_summary)
        print(f"  damping={d:.2f}: stress={donor_data[d]['final_stress']:.4f}, "
              f"Vmem similarity={sim:.4f}")

    header = "recip\\donor  " + "  ".join([f"{d:>6.2f}" for d in damping_levels])

    print(f"\nVmem Similarity GAIN matrix (rescued - unrescued baseline):")
    print(f"  {header}")
    for recip_d in damping_levels:
        row = f"  {recip_d:.2f}        "
        for donor_d in damping_levels:
            gain = results[(donor_d, recip_d)]['vmem_similarity_gain']
            row += f"  {gain:+.3f}"
        print(row)

    print(f"\nVmem Similarity matrix (Pearson r with healthy ref):")
    print(f"  {header}")
    for recip_d in damping_levels:
        row = f"  {recip_d:.2f}        "
        for donor_d in damping_levels:
            sim = results[(donor_d, recip_d)]['vmem_similarity']
            row += f"  {sim:6.2f}"
        print(row)

    print(f"\nStress matrix (donor -> recipient):")
    print(f"  {header}")
    for recip_d in damping_levels:
        row = f"  {recip_d:.2f}        "
        for donor_d in damping_levels:
            stress = results[(donor_d, recip_d)]['final_stress']
            row += f"  {stress:6.2f}"
        print(row)

    # Visualize
    visualize_rescue(results, donor_data, damping_levels, args.alpha, args.outputFile)

    print(f"\nDone!")
    return results, donor_data


if __name__ == "__main__":
    main()
