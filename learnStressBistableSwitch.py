#!/usr/bin/env python3
"""
Learn Stress Bistable Switch Parameters.

Optimizes the RD bistable stress system parameters so that:
- Healthy embryos (GRN damping = 1.0) produce mean stress ~ 0
- Perturbed embryos (GRN damping = 0.9) produce mean stress ~ 1.0
- Intermediate damping produces intermediate stress

The bioelectric simulations are run ONCE per damping level (expensive),
then the Vmem timeseries are replayed through the stress system each
learning iteration (cheap).

Usage:
    # Default settings (3 damping levels, 100 learning iterations)
    python learnStressBistableSwitch.py

    # Custom damping levels and targets
    python learnStressBistableSwitch.py --dampingLevels "1.0,0.95,0.9" --targetStress "0.0,0.5,1.0"

    # More learning iterations with lower learning rate
    python learnStressBistableSwitch.py --numLearnIters 200 --lr 0.005

    # Custom bio steps and stress steps
    python learnStressBistableSwitch.py --numBioSteps 1000 --numStressSteps 500
"""

import argparse
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
parser = argparse.ArgumentParser(description='Learn stress bistable switch parameters')
parser.add_argument('--numBioSteps', type=int, default=1000,
                    help='Number of bioelectric simulation steps (default: 1000)')
parser.add_argument('--numStressSteps', type=int, default=500,
                    help='Number of stress system iterations on Ca2+ pattern (default: 500)')
parser.add_argument('--numLearnIters', type=int, default=100,
                    help='Number of learning iterations (default: 100)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate for Rprop optimizer (default: 0.01)')
parser.add_argument('--fileNumber', type=int, default=0,
                    help='File number for output (default: 0)')
parser.add_argument('--dampingLevels', type=str, default='1.0,0.95,0.9',
                    help='Comma-separated GRN damping levels (default: 1.0,0.95,0.9)')
parser.add_argument('--targetStress', type=str, default='0.0,0.5,1.0',
                    help='Comma-separated target stress values (default: 0.0,0.5,1.0)')
parser.add_argument('--verbose', type=str, default='True',
                    help='Print progress (default: True)')
args = parser.parse_args()

# Parse arguments
num_bio_steps = args.numBioSteps
num_stress_steps = args.numStressSteps
num_learn_iters = args.numLearnIters
lr = args.lr
file_number = args.fileNumber
damping_levels = [float(x) for x in args.dampingLevels.split(',')]
target_stress = [float(x) for x in args.targetStress.split(',')]
verbose = args.verbose.lower() in ('true', '1', 'yes')

assert len(damping_levels) == len(target_stress), \
    f"dampingLevels ({len(damping_levels)}) and targetStress ({len(target_stress)}) must have same length"


# ============================================================
# Parameter transformation utilities
# ============================================================
def inverse_sigmoid(x, min_val, max_val):
    """Compute raw parameter that maps to x via sigmoid."""
    normalized = (x - min_val) / (max_val - min_val)
    normalized = torch.clamp(normalized, 1e-6, 1.0 - 1e-6)
    return torch.logit(normalized)


def apply_sigmoid_constraint(raw_param, min_val, max_val):
    """Map unbounded raw parameter to bounded range via sigmoid."""
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)


# ============================================================
# Load model parameters with GRN damping
# ============================================================
def load_model_parameters(grn_damping=1.0):
    """
    Load Model 253 parameters with specified GRN damping.

    Args:
        grn_damping: GRN weight damping factor [0,1]. 1=native weights.

    Returns:
        dict with model parameters
    """
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
# Run bioelectric simulation and collect Vmem timeseries
# ============================================================
def run_bioelectric_sim(grn_damping, num_bio_steps):
    """
    Run Model 253 with specified GRN damping and return Vmem timeseries.

    Args:
        grn_damping: GRN weight damping factor
        num_bio_steps: number of simulation iterations

    Returns:
        vmem_timeseries: list of (num_cells,) tensors, one per iteration
        grid_size: tissue grid size
    """
    params = load_model_parameters(grn_damping)
    grid_size = params['latticeDims'][0]

    num_samples = params["simParameters"]["numSamples"]
    initial_values = copy.deepcopy(params["simParameters"]["initialValues"])
    external_inputs = copy.deepcopy(params["simParameters"]["externalInputs"])
    clamp_params = copy.deepcopy(params["clampParameters"])

    bio_model = model(params, numBasicSamples=num_samples)
    bio_model.setExperimentalConditions((initial_values, num_samples))

    vmem_timeseries = []

    # Record initial Vmem
    initial_vmem = bio_model.electricNetwork.Vmem[0, :, 0].clone()
    vmem_timeseries.append(initial_vmem)

    # Run iteratively to capture Vmem at each step
    for iter_idx in range(num_bio_steps):
        bio_model.simulate(
            externalInputs=external_inputs,
            clampParameters=clamp_params,
            perturbation=None,
            fieldModulation=False,
            numSimIters=1,
            outerIter=iter_idx
        )
        current_vmem = bio_model.electricNetwork.Vmem[0, :, 0].clone()
        vmem_timeseries.append(current_vmem)

    return vmem_timeseries, grid_size


def load_fixed_ca_params(camkii_params_path='./data/bestLearnedCaMKIIParams_0.dat'):
    """
    Load learned Ca²⁺ channel parameters from the CaMKII params file.

    These are fixed (not learned) and passed to StressBistableSwitch so it
    uses the same proven Ca²⁺ transduction as the CaMKII system.

    Returns:
        dict mapping Ca²⁺ param names to scalar float values
    """
    data = torch.load(camkii_params_path, weights_only=False)
    raw_params = data['parameters']
    bounds = data['parameter_bounds']

    ca_param_names = ['tau_ca', 'g_ca', 'V_half_ca', 'k_ca', 'k_decay_ca']
    fixed_ca = {}
    for name in ca_param_names:
        raw = raw_params[name]
        mn = bounds[f'{name}_min']
        mx = bounds[f'{name}_max']
        fixed_ca[name] = float(mn + (mx - mn) * torch.sigmoid(raw))

    return fixed_ca


# ============================================================
# Initialize learnable parameters
# ============================================================
STRESS_PARAM_NAMES = [
    'tau_S', 'k_on_S', 'k_off_S', 'K_S', 'Ca_stress_threshold',
    'sigma_ca', 'gain_S', 'or_threshold_S', 'D_S', 'gamma', 'K_decay'
]


def initialize_parameters(dtype=torch.float32, device='cpu'):
    """
    Initialize learnable stress system parameters using sigmoid parameterization.

    Only stress-specific parameters are learned here. Ca²⁺ dynamics are handled
    by SimpleCaMKII with fixed learned parameters (loaded separately).

    Returns:
        params dict with raw (learnable) parameters and bounds
    """
    params = {}

    def add_param(name, min_val, max_val):
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype, device=device), min_val, max_val)
        params[f'{name}_raw'] = raw_param.clone().requires_grad_(True)
        params[f'{name}_min'] = min_val
        params[f'{name}_max'] = max_val

    # --- Stress RD bistable dynamics (all learnable) ---
    # add_param('tau_S', 5.0, 100.0)
    add_param('tau_S', 1.0, 10.0)
    add_param('k_on_S', 0.5, 10.0)
    add_param('k_off_S', 0.001, 1.0)
    add_param('K_S', 0.1, 0.8)
    add_param('Ca_stress_threshold', 0.001, 10.0)
    add_param('sigma_ca', 0.005, 2.0)
    add_param('gain_S', 1.0, 6.0)
    add_param('or_threshold_S', 0.1, 2.0)
    add_param('D_S', 0.01, 0.3)
    add_param('gamma', 0.01, 0.5)
    add_param('K_decay', 0.01, 0.5)   # phosphatase Km (half-saturation for decay)

    return params


def get_constrained_params(params):
    """Extract all constrained parameter values from raw parameters."""
    constrained = {}
    for name in STRESS_PARAM_NAMES:
        constrained[name] = apply_sigmoid_constraint(
            params[f'{name}_raw'],
            params[f'{name}_min'],
            params[f'{name}_max']
        )
    return constrained


# ============================================================
# Run stress system on Vmem timeseries with fixed Ca²⁺ params
# ============================================================
def run_stress_on_vmem(vmem_timeseries, adjacency_matrix, fixed_ca_params,
                       constrained_stress_params, num_stress_steps, device, dtype):
    """
    Run the stress bistable switch concurrently with Ca²⁺ on a Vmem timeseries.

    Ca²⁺ dynamics use fixed parameters from the learned CaMKII file.
    Only stress (S) parameters are learnable.

    At each Vmem timestep:
      1. Update Ca²⁺ from Vmem (fixed params, no grad)
      2. Detach Ca²⁺ and step S dynamics (learnable params, with grad)

    This is biologically more plausible: stress develops alongside the
    calcium signal rather than only seeing the final equilibrated pattern.

    After the Vmem drive, run additional S equilibration steps with the
    final Ca²⁺ pattern held constant.

    Args:
        vmem_timeseries: list of (num_cells,) Vmem tensors
        adjacency_matrix: (num_cells, num_cells) adjacency
        fixed_ca_params: dict of fixed Ca²⁺ parameter values (floats)
        constrained_stress_params: dict of learnable stress parameter tensors
        num_stress_steps: extra S equilibration steps after Vmem drive
        device: torch device
        dtype: torch dtype

    Returns:
        final_stress: scalar embryo-level stress = mean(S)
        stress_history: list of mean(S) over time
        final_state: dict with S, Ca, etc.
    """
    num_cells = vmem_timeseries[0].shape[0]

    # Create stress switch with fixed Ca²⁺ params
    stress_switch = StressBistableSwitch(
        num_cells=num_cells,
        adjacency_matrix=adjacency_matrix,
        params=fixed_ca_params,  # Fixed Ca²⁺ params set via constructor
        device=device,
        dtype=dtype,
    )

    # Set learnable stress parameters
    stress_switch.set_params_from_tensors(**constrained_stress_params)

    dt = 0.01
    dt_stress = 0.1
    stress_history = []

    # Concurrent phase: Ca²⁺ + S evolve together during Vmem drive
    for vmem_flat in vmem_timeseries:
        # Update Ca²⁺ from Vmem (fixed params, no grad)
        with torch.no_grad():
            vmem_on_device = vmem_flat.to(device=device, dtype=dtype)
            stress_switch.compute_ca_from_vmem(vmem_on_device, dt)

        # Step S using current Ca²⁺ (detached so grad flows only through S params)
        ca_now = stress_switch.Ca.detach()
        stress_switch.step(dt_stress, Ca=ca_now)
        stress_history.append(stress_switch.get_embryo_stress().item())

    # Equilibration phase: S continues with final Ca²⁺ held constant
    ca_final = stress_switch.Ca.detach().clone()
    for _ in range(num_stress_steps):
        stress_switch.step(dt_stress, Ca=ca_final)
        stress_history.append(stress_switch.get_embryo_stress().item())

    final_stress = stress_switch.get_embryo_stress()
    final_state = stress_switch.get_state()

    return final_stress, stress_history, final_state


# ============================================================
# Visualization
# ============================================================
def visualize_results(all_results, damping_levels, target_stress,
                      best_loss_history, output_path):
    """
    Visualize stress vs damping curve and S spatial patterns.

    Args:
        all_results: dict mapping damping -> (final_stress, stress_history, final_state)
        damping_levels: list of damping values
        target_stress: list of target stress values
        best_loss_history: list of (iteration, loss) tuples
        output_path: path to save figure
    """
    num_damping = len(damping_levels)
    fig = plt.figure(figsize=(6 + 4 * num_damping, 10))
    gs = fig.add_gridspec(2, 1 + num_damping, hspace=0.35, wspace=0.3)

    # --- Top-left: Stress vs Damping curve ---
    ax_curve = fig.add_subplot(gs[0, 0])
    actual_stress = [all_results[d][0] for d in damping_levels]
    ax_curve.plot(damping_levels, actual_stress, 'bo-', markersize=8, label='Learned', linewidth=2)
    ax_curve.plot(damping_levels, target_stress, 'rx--', markersize=10, label='Target', linewidth=2)
    ax_curve.set_xlabel('GRN Damping Factor')
    ax_curve.set_ylabel('Mean Stress (embryo level)')
    ax_curve.set_title('Stress vs GRN Damping', fontweight='bold')
    ax_curve.legend()
    ax_curve.grid(alpha=0.3)
    ax_curve.set_xlim(min(damping_levels) - 0.02, max(damping_levels) + 0.02)
    ax_curve.set_ylim(-0.1, 1.1)

    # --- Top row: S spatial patterns at each damping level ---
    for i, d in enumerate(damping_levels):
        ax = fig.add_subplot(gs[0, 1 + i])
        state = all_results[d][2]
        S_grid = state['S'].detach().cpu()
        grid_size = int(np.sqrt(len(S_grid)))
        S_2d = S_grid.reshape(grid_size, grid_size)
        im = ax.imshow(S_2d.numpy(), cmap='YlOrRd', vmin=0, vmax=1)
        actual = all_results[d][0]
        if isinstance(actual, torch.Tensor):
            actual = actual.item()
        ax.set_title(f'S pattern\ndamping={d:.2f}\nstress={actual:.3f}',
                     fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # --- Bottom-left: Stress timeseries for each damping level ---
    ax_ts = fig.add_subplot(gs[1, 0:2])
    colors = plt.cm.viridis(np.linspace(0, 1, num_damping))
    for i, d in enumerate(damping_levels):
        history = all_results[d][1]
        ax_ts.plot(history, color=colors[i], label=f'damping={d:.2f}', linewidth=1.5)
    ax_ts.set_xlabel('Time step')
    ax_ts.set_ylabel('Mean Stress')
    ax_ts.set_title('Stress Evolution Over Time', fontweight='bold')
    ax_ts.legend(fontsize=8)
    ax_ts.grid(alpha=0.3)

    # --- Bottom-right: Loss history ---
    if len(best_loss_history) > 0:
        ax_loss = fig.add_subplot(gs[1, 2:])
        iters, losses = zip(*best_loss_history)
        ax_loss.plot(iters, losses, 'g-o', markersize=3, linewidth=1.5)
        ax_loss.set_xlabel('Learning Iteration')
        ax_loss.set_ylabel('Best Loss')
        ax_loss.set_title('Learning Progress', fontweight='bold')
        ax_loss.grid(alpha=0.3)
        ax_loss.set_yscale('log')

    plt.suptitle('Stress Bistable Switch: Learned Parameters',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")


# ============================================================
# Main learning loop
# ============================================================
def main():
    # Device setup — CPU is faster for small grids (11x11 = 121 cells)
    # due to per-step transfer overhead with MPS/CUDA
    device = torch.device('cpu')
    device_name = 'CPU'
    dtype = torch.float32

    print("=" * 70)
    print("LEARNING STRESS BISTABLE SWITCH PARAMETERS")
    print("=" * 70)
    print(f"Device: {device_name}")
    print(f"Bioelectric steps: {num_bio_steps}")
    print(f"Stress equilibration steps: {num_stress_steps}")
    print(f"Learning iterations: {num_learn_iters}")
    print(f"Learning rate: {lr}")
    print(f"Damping levels: {damping_levels}")
    print(f"Target stress:  {target_stress}")
    print("=" * 70 + "\n")

    # --------------------------------------------------------
    # Step 1: Load fixed Ca²⁺ params from learned CaMKII file
    # --------------------------------------------------------
    print("Step 1: Loading fixed Ca²⁺ parameters from learned CaMKII file...\n")
    fixed_ca_params = load_fixed_ca_params()
    print("  Fixed Ca²⁺ parameters (from bestLearnedCaMKIIParams_0.dat):")
    for name, val in fixed_ca_params.items():
        print(f"    {name}: {val:.4f}")
    print()

    # --------------------------------------------------------
    # Step 2: Run bioelectric sims ONCE for each damping level
    # --------------------------------------------------------
    print("Step 2: Running bioelectric simulations (one-time cost)...\n")
    vmem_data = {}  # damping -> (vmem_timeseries, grid_size)

    for i, damping in enumerate(damping_levels):
        print(f"  [{i+1}/{len(damping_levels)}] Damping = {damping:.2f} ...")
        vmem_ts, grid_size = run_bioelectric_sim(damping, num_bio_steps)
        vmem_data[damping] = (vmem_ts, grid_size)
        final_vmem = vmem_ts[-1]
        print(f"    Vmem at t={num_bio_steps}: mean={final_vmem.mean().item():.4f}V, "
              f"std={final_vmem.std().item():.4f}V")

    # Compute adjacency matrix (same for all, since grid is the same)
    num_cells = grid_size * grid_size
    utils = utilities.utilities()
    adjacency_matrix = utils.computeLatticeAdjacencyMatrix(
        latticeDims=(grid_size, grid_size), periodicBoundary=False
    )

    # Diagnostic: show Ca²⁺ patterns from fixed params to verify they differ
    print(f"\n  Grid size: {grid_size}x{grid_size} = {num_cells} cells")
    print(f"\n  Ca²⁺ patterns from fixed CaMKII params:")
    for damping in damping_levels:
        vmem_ts, _ = vmem_data[damping]
        # Run Ca²⁺ forward pass with fixed params to check
        tmp_switch = StressBistableSwitch(
            num_cells=num_cells, adjacency_matrix=adjacency_matrix,
            params=fixed_ca_params, device=device, dtype=dtype)
        with torch.no_grad():
            for vmem_flat in vmem_ts:
                tmp_switch.compute_ca_from_vmem(vmem_flat.to(device=device, dtype=dtype), dt=0.01)
        ca = tmp_switch.Ca
        print(f"    damping={damping:.2f}: Ca mean={ca.mean().item():.4f}, std={ca.std().item():.4f}")
    print(f"\n  Bioelectric simulations complete.\n")

    # --------------------------------------------------------
    # Step 3: Initialize learnable stress parameters
    # --------------------------------------------------------
    print("Step 3: Initializing learnable stress parameters...\n")
    params = initialize_parameters(dtype=dtype, device=device)

    print("  Initial parameter values:")
    for name in STRESS_PARAM_NAMES:
        val = apply_sigmoid_constraint(
            params[f'{name}_raw'], params[f'{name}_min'], params[f'{name}_max']
        ).item()
        print(f"    {name}: {val:.4f} (range [{params[f'{name}_min']}, {params[f'{name}_max']}])")
    print()

    # Collect raw parameters for optimizer (stress params only)
    learned_params_list = [params[f'{name}_raw'] for name in STRESS_PARAM_NAMES]
    optimizer = torch.optim.Rprop(learned_params_list, lr=lr)

    # --------------------------------------------------------
    # Step 4: Learning loop
    # --------------------------------------------------------
    print("=" * 70)
    print("STARTING LEARNING LOOP")
    print("=" * 70)
    print(f"Objective: stress(damping={damping_levels}) -> target={target_stress}\n")

    best_loss = float('inf')
    best_params = {}
    best_loss_history = []

    for iter_idx in range(num_learn_iters):
        # Get current constrained stress parameters
        constrained = get_constrained_params(params)

        # Run stress system for each damping level
        total_loss = torch.tensor(0.0, dtype=dtype, device=device)
        stress_values = []
        stress_tensors = []

        # Weight healthy target (0.0) more heavily to ensure OFF state reaches ~0
        loss_weights = [3.0 if t == 0.0 else 1.0 for t in target_stress]

        for damping, target, weight in zip(damping_levels, target_stress, loss_weights):
            vmem_ts, _ = vmem_data[damping]

            final_stress, _, _ = run_stress_on_vmem(
                vmem_timeseries=vmem_ts,
                adjacency_matrix=adjacency_matrix,
                fixed_ca_params=fixed_ca_params,
                constrained_stress_params=constrained,
                num_stress_steps=num_stress_steps,
                device=device,
                dtype=dtype,
            )

            target_tensor = torch.tensor(target, dtype=dtype, device=device)
            total_loss = total_loss + weight * (final_stress - target_tensor) ** 2
            stress_values.append(final_stress.item())
            stress_tensors.append(final_stress)

        # Variance penalty: penalize outputs collapsing to the same value
        if len(stress_tensors) > 1:
            stress_stack = torch.stack(stress_tensors)
            target_var = torch.tensor(target_stress, dtype=dtype, device=device).var()
            actual_var = stress_stack.var()
            variance_loss = torch.relu(target_var - actual_var)
            total_loss = total_loss + 2.0 * variance_loss

        current_loss = total_loss.item()

        # Track best
        if current_loss < best_loss:
            best_loss = current_loss
            best_loss_history.append((iter_idx, best_loss))

            # Save best raw parameters
            best_params = {}
            for name in STRESS_PARAM_NAMES:
                best_params[name] = params[f'{name}_raw'].detach().clone()

            # Save to file
            save_data = {
                'parameters': best_params,
                'parameter_bounds': {},
                'fixed_ca_params': fixed_ca_params,
                'best_loss': best_loss,
                'best_iteration': iter_idx,
                'damping_levels': damping_levels,
                'target_stress': target_stress,
                'num_bio_steps': num_bio_steps,
                'num_stress_steps': num_stress_steps,
                'grid_size': grid_size,
            }
            for name in STRESS_PARAM_NAMES:
                save_data['parameter_bounds'][f'{name}_min'] = params[f'{name}_min']
                save_data['parameter_bounds'][f'{name}_max'] = params[f'{name}_max']

            torch.save(save_data, f'./data/bestLearnedStressParams_{file_number}.dat')

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if verbose and ((iter_idx + 1) % 5 == 0 or iter_idx == 0):
            stress_str = ', '.join([f'{s:.3f}' for s in stress_values])
            target_str = ', '.join([f'{t:.1f}' for t in target_stress])
            print(f"  Iter {iter_idx+1:4d}/{num_learn_iters}: "
                  f"loss={current_loss:.4f} (best={best_loss:.4f}) | "
                  f"stress=[{stress_str}] target=[{target_str}]")

            if (iter_idx + 1) % 20 == 0:
                print("    Parameters:")
                for name in STRESS_PARAM_NAMES:
                    val = apply_sigmoid_constraint(
                        params[f'{name}_raw'], params[f'{name}_min'], params[f'{name}_max']
                    ).item()
                    print(f"      {name}: {val:.4f}")

    # --------------------------------------------------------
    # Step 5: Final evaluation and visualization
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("LEARNING COMPLETE")
    print("=" * 70)

    # Load best parameters and run final evaluation
    best_constrained = {}
    for name in STRESS_PARAM_NAMES:
        best_constrained[name] = apply_sigmoid_constraint(
            best_params[name],
            params[f'{name}_min'],
            params[f'{name}_max']
        )

    print(f"\nBest loss: {best_loss:.6f}")
    print(f"\nBest parameter values:")
    for name in STRESS_PARAM_NAMES:
        print(f"  {name}: {best_constrained[name].item():.4f}")

    # Run final evaluation with best parameters
    print(f"\nFinal evaluation:")
    all_results = {}
    for damping, target in zip(damping_levels, target_stress):
        vmem_ts, _ = vmem_data[damping]
        final_stress, stress_history, final_state = run_stress_on_vmem(
            vmem_timeseries=vmem_ts,
            adjacency_matrix=adjacency_matrix,
            fixed_ca_params=fixed_ca_params,
            constrained_stress_params=best_constrained,
            num_stress_steps=num_stress_steps,
            device=device,
            dtype=dtype,
        )
        all_results[damping] = (final_stress.item(), stress_history, final_state)
        print(f"  damping={damping:.2f}: stress={final_stress.item():.4f} (target={target:.1f})")

    # Visualize
    output_path = f'./data/learned_stress_bistable_{file_number}.png'
    visualize_results(all_results, damping_levels, target_stress,
                      best_loss_history, output_path)

    print(f"\nParameters saved to: ./data/bestLearnedStressParams_{file_number}.dat")
    print(f"Visualization saved to: {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
