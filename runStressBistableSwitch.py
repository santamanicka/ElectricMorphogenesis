#!/usr/bin/env python3
"""
Run and visualize the Stress Bistable Switch on Model 253.

Runs the bioelectric model at one or more GRN damping levels, feeds the
resulting Vmem timeseries through the RD bistable stress system, and
visualizes the stress patterns and timeseries.

Usage:
    # Single run with default (healthy) parameters
    python runStressBistableSwitch.py

    # Compare healthy vs perturbed
    python runStressBistableSwitch.py --dampingLevels "1.0,0.95,0.9"

    # Use learned stress parameters
    python runStressBistableSwitch.py --stressParamsFile data/bestLearnedStressParams_0.dat

    # Custom bio and stress steps
    python runStressBistableSwitch.py --numBioSteps 1000 --numStressSteps 500
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
parser = argparse.ArgumentParser(description='Run stress bistable switch on Model 253')
parser.add_argument('--numBioSteps', type=int, default=2000,
                    help='Number of bioelectric simulation steps (default: 2000)')
parser.add_argument('--numStressSteps', type=int, default=500,
                    help='Additional stress equilibration steps after bio sim (default: 500)')
parser.add_argument('--dampingLevels', type=str, default='1.0,0.95,0.9',
                    help='Comma-separated GRN damping levels (default: 1.0,0.95,0.9)')
parser.add_argument('--stressParamsFile', type=str, default=None,
                    help='Path to learned stress parameters file (.dat)')
parser.add_argument('--outputFile', type=str, default='data/stress_bistable_test.png',
                    help='Output visualization filename')
args = parser.parse_args()

damping_levels = [float(x) for x in args.dampingLevels.split(',')]

# Extract file number suffix from stressParamsFile and append to output filename
if args.stressParamsFile is not None:
    match = re.search(r'_(\d+)\.dat$', args.stressParamsFile)
    if match and args.outputFile == 'data/stress_bistable_test.png':
        suffix = match.group(1)
        args.outputFile = f'data/stress_bistable_test_{suffix}.png'


# ============================================================
# Parameter loading
# ============================================================
def apply_sigmoid_constraint(raw_param, min_val, max_val):
    """Map unbounded raw parameter to bounded range via sigmoid."""
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)


def load_stress_params(params_file):
    """
    Load learned stress parameters and fixed Ca²⁺ parameters from file.

    The file contains:
    - 'parameters': raw stress params (sigmoid-encoded)
    - 'parameter_bounds': min/max for each stress param
    - 'fixed_ca_params': Ca²⁺ params from learned CaMKII (plain floats)

    Returns:
        (stress_params, fixed_ca_params) tuple of dicts
    """
    print(f"Loading learned stress parameters from: {params_file}")
    data = torch.load(params_file, weights_only=False)

    # Decode stress parameters from raw sigmoid-encoded values
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

    # Load fixed Ca²⁺ parameters
    fixed_ca_params = data.get('fixed_ca_params', None)
    if fixed_ca_params is not None:
        print(f"  Fixed Ca²⁺ parameters ({len(fixed_ca_params)}):")
        for name, value in fixed_ca_params.items():
            print(f"    {name}: {value:.4f}")
    else:
        print("  WARNING: No fixed_ca_params in file, using defaults")
        fixed_ca_params = get_default_ca_params()

    return stress_params, fixed_ca_params


def get_default_ca_params():
    """Return default Ca²⁺ parameters (from learned CaMKII file)."""
    return {
        'tau_ca': 2.5964,
        'g_ca': 5.3437,
        'V_half_ca': -0.0753,
        'k_ca': 0.0021,
        'k_decay_ca': 4.3346,
    }


def get_default_stress_params():
    """Return default stress-specific parameters and Ca²⁺ parameters."""
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
# Run bioelectric simulation
# ============================================================
def run_bioelectric_sim(grn_damping, num_bio_steps):
    """Run Model 253 and return Vmem timeseries."""
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

        if (iter_idx + 1) % 200 == 0:
            v = vmem_timeseries[-1]
            print(f"    Iter {iter_idx+1}/{num_bio_steps}: "
                  f"Vmem mean={v.mean().item():.4f}V, std={v.std().item():.4f}V")

    return vmem_timeseries, grid_size


# ============================================================
# Run stress system
# ============================================================
def run_stress_system(vmem_timeseries, adjacency_matrix, stress_params,
                      ca_params, num_stress_steps, grid_size, device='cpu'):
    """
    Run stress bistable switch concurrently with Ca²⁺ on Vmem timeseries.

    Matches learnStressBistableSwitch.py:
      Concurrent phase: Ca²⁺ + S evolve together during Vmem drive
      Equilibration phase: S continues with final Ca²⁺ held constant
    """
    num_cells = grid_size * grid_size
    dtype = torch.float32

    # Create stress switch with fixed Ca²⁺ params
    stress_switch = StressBistableSwitch(
        num_cells=num_cells,
        adjacency_matrix=adjacency_matrix,
        params=ca_params,  # Fixed Ca²⁺ params via constructor
        device=device,
        dtype=dtype,
    )
    # Set stress-specific parameters
    stress_switch.set_params_from_tensors(
        **{k: torch.tensor(v, dtype=dtype, device=device) for k, v in stress_params.items()}
    )

    dt_ca = 0.01
    dt_stress = 0.1
    stress_history = []
    ca_history = []
    S_snapshots = {}

    # Concurrent phase: Ca²⁺ + S evolve together during Vmem drive
    num_vmem = len(vmem_timeseries)
    for t, vmem_flat in enumerate(vmem_timeseries):
        # Update Ca²⁺ from Vmem
        with torch.no_grad():
            stress_switch.compute_ca_from_vmem(vmem_flat.to(device=device, dtype=dtype), dt_ca)
        ca_history.append(stress_switch.Ca.mean().item())

        # Step S using current Ca²⁺
        ca_now = stress_switch.Ca.detach()
        stress_switch.step(dt_stress, Ca=ca_now)
        stress_history.append(stress_switch.get_embryo_stress().item())

        # Record snapshots at key times
        if t in [0, num_vmem // 2, num_vmem - 1]:
            S_snapshots[f'concurrent_t{t}'] = stress_switch.get_state()

    print(f"    Ca²⁺ after Vmem drive: mean={stress_switch.Ca.mean().item():.4f}, "
          f"std={stress_switch.Ca.std().item():.4f}")
    print(f"    S after Vmem drive: mean={stress_switch.S.mean().item():.4f}")

    # Equilibration phase: S continues with final Ca²⁺ held constant
    ca_final = stress_switch.Ca.detach().clone()
    for t in range(num_stress_steps):
        stress_switch.step(dt_stress, Ca=ca_final)
        stress_history.append(stress_switch.get_embryo_stress().item())

    final_state = stress_switch.get_state()
    S_snapshots['final'] = final_state

    return {
        'stress_history': stress_history,
        'ca_history': ca_history,
        'final_stress': stress_switch.get_embryo_stress().item(),
        'S_snapshots': S_snapshots,
        'final_state': final_state,
    }


# ============================================================
# Visualization
# ============================================================
def visualize(all_results, damping_levels, grid_size, output_path):
    """Visualize stress results across damping levels."""
    num_damping = len(damping_levels)

    fig = plt.figure(figsize=(5 * max(num_damping, 3), 14))
    gs = fig.add_gridspec(4, max(num_damping, 3), hspace=0.4, wspace=0.35)

    # Row 1: Final Vmem patterns
    for i, d in enumerate(damping_levels):
        ax = fig.add_subplot(gs[0, i])
        vmem_final = all_results[d]['vmem_final']
        vmem_2d = vmem_final.reshape(grid_size, grid_size).numpy()
        im = ax.imshow(vmem_2d, cmap='coolwarm')
        ax.set_title(f'Vmem (t={args.numBioSteps})\ndamping={d:.2f}', fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: Final Ca2+ patterns
    for i, d in enumerate(damping_levels):
        ax = fig.add_subplot(gs[1, i])
        ca = all_results[d]['final_state']['Ca'].detach().cpu()
        ca_2d = ca.reshape(grid_size, grid_size).numpy()
        im = ax.imshow(ca_2d, cmap='YlOrRd')
        ax.set_title(f'Ca2+ (final)\ndamping={d:.2f}\nmean={ca.mean().item():.3f}',
                     fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 3: Final S patterns
    for i, d in enumerate(damping_levels):
        ax = fig.add_subplot(gs[2, i])
        S = all_results[d]['final_state']['S'].detach().cpu()
        S_2d = S.reshape(grid_size, grid_size).numpy()
        im = ax.imshow(S_2d, cmap='YlOrRd', vmin=0, vmax=1)
        stress = all_results[d]['final_stress']
        ax.set_title(f'Stress S (final)\ndamping={d:.2f}\nmean(S)={stress:.3f}',
                     fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 4: Timeseries
    ax_ts = fig.add_subplot(gs[3, :])
    colors = ['#2166AC', '#F4A582', '#B2182B']  # blue -> red for healthy -> perturbed
    if len(damping_levels) > 3:
        colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, num_damping))

    for i, d in enumerate(damping_levels):
        history = all_results[d]['stress_history']
        color = colors[i] if i < len(colors) else colors[-1]
        ax_ts.plot(history, color=color,
                   label=f'damping={d:.2f} (final={all_results[d]["final_stress"]:.3f})',
                   linewidth=1.5)

    # Mark end of concurrent Vmem drive phase
    num_vmem = len(list(all_results.values())[0]['ca_history'])
    ax_ts.axvline(num_vmem, color='gray', linestyle='--', alpha=0.5,
                  label=f'Vmem drive end (t={num_vmem})')
    ax_ts.set_xlabel('Time step (concurrent Ca²⁺ + S, then S equilibration)')
    ax_ts.set_ylabel('Mean Stress (embryo level)')
    ax_ts.set_title('Stress Evolution: Healthy vs Perturbed', fontsize=12, fontweight='bold')
    ax_ts.legend(fontsize=9, loc='upper left')
    ax_ts.grid(alpha=0.3)
    ax_ts.set_ylim(-0.05, 1.05)

    plt.suptitle('Stress Bistable Switch Test (Model 253)',
                 fontsize=14, fontweight='bold', y=0.99)

    plt.savefig(f'./{output_path}', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: ./{output_path}")


# ============================================================
# Main
# ============================================================
def main():
    # CPU is faster for small grids (11x11 = 121 cells)
    device = torch.device('cpu')

    print("=" * 60)
    print("Stress Bistable Switch Test")
    print("=" * 60)
    print(f"Bioelectric steps: {args.numBioSteps}")
    print(f"Stress equilibration steps: {args.numStressSteps}")
    print(f"Damping levels: {damping_levels}")

    # Load stress parameters and fixed Ca²⁺ parameters
    if args.stressParamsFile is not None:
        stress_params, ca_params = load_stress_params(args.stressParamsFile)
    else:
        stress_params, ca_params = get_default_stress_params()
        print("\nUsing DEFAULT stress parameters:")
        for name, val in stress_params.items():
            print(f"  {name}: {val}")
        print("\nUsing DEFAULT Ca²⁺ parameters:")
        for name, val in ca_params.items():
            print(f"  {name}: {val}")

    print()

    # Compute adjacency matrix
    # Load one set of params to get grid size
    test_params = load_model_parameters(1.0)
    grid_size = test_params['latticeDims'][0]
    del test_params

    utils = utilities.utilities()
    adjacency_matrix = utils.computeLatticeAdjacencyMatrix(
        latticeDims=(grid_size, grid_size), periodicBoundary=False
    )
    print(f"Grid size: {grid_size}x{grid_size}")

    # Run for each damping level
    all_results = {}
    for i, damping in enumerate(damping_levels):
        print(f"\n{'='*40}")
        print(f"[{i+1}/{len(damping_levels)}] GRN Damping = {damping:.2f}")
        print(f"{'='*40}")

        # Run bioelectric sim
        print("  Running bioelectric simulation...")
        vmem_ts, _ = run_bioelectric_sim(damping, args.numBioSteps)
        print(f"  Vmem timeseries: {len(vmem_ts)} frames")

        # Run stress system
        print("  Running stress bistable switch...")
        results = run_stress_system(
            vmem_timeseries=vmem_ts,
            adjacency_matrix=adjacency_matrix,
            stress_params=stress_params,
            ca_params=ca_params,
            num_stress_steps=args.numStressSteps,
            grid_size=grid_size,
            device=device,
        )
        results['vmem_final'] = vmem_ts[-1].detach().cpu()

        all_results[damping] = results
        print(f"  Final stress: {results['final_stress']:.4f}")
        print(f"  Final Ca2+ mean: {results['ca_history'][-1]:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for d in damping_levels:
        print(f"  damping={d:.2f}: stress={all_results[d]['final_stress']:.4f}")

    # Visualize
    visualize(all_results, damping_levels, grid_size, args.outputFile)

    print(f"\nDone!")
    return all_results


if __name__ == "__main__":
    all_results = main()
