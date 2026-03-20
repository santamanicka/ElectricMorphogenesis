#!/usr/bin/env python3
"""
Test CaMKII Bistability for Pattern Locking in Bioelectric Models

Tests whether CaMKII can "lock in" the bioelectric face pattern at t=1000
and retain it at t=2000, even if the voltage pattern changes or degrades.

Key Question: Does CaMKII pattern at t=2000 still resemble the face pattern
from Vmem at t=1000?

Supported models:
    - Stigmergic: Uses data/StigmergicModelParameters.dat
    - 253: Uses data/bestModelParameters_fieldVector_Ligand_GRN_253.dat

Usage:
    # Test with default parameters (Stigmergic model)
    python test_camkii_bistability.py

    # Test with Model 253
    python test_camkii_bistability.py --model 253

    # Test with learned CaMKII parameters
    python test_camkii_bistability.py --paramsFile data/bestLearnedCaMKIIParams_0.dat

    # Test Model 253 with learned parameters
    python test_camkii_bistability.py --model 253 --paramsFile data/bestLearnedCaMKIIParams_0.dat

    # Test with Vmem perturbation at pattern lock time (t=1000)
    python test_camkii_bistability.py --perturb-vmem 0.01 --perturb-seed 42

    # Test with perturbation at a specific iteration
    python test_camkii_bistability.py --perturb-vmem 0.02 --perturb-iter 500

    # Test with sustained perturbation over multiple iterations
    python test_camkii_bistability.py --perturb-vmem 0.005 --perturb-iter 1000 --perturb-duration 100

    # Test with GRN weights damped to 50%
    python test_camkii_bistability.py --model 253 --grn-damping 0.5

    # Test with GRN completely disabled (damping=0)
    python test_camkii_bistability.py --model 253 --grn-damping 0.0

    # Test with progressive GRN damping (0.0 -> 1.0 over full simulation)
    python test_camkii_bistability.py --model 253 --grn-progressive

    # Test with progressive damping over custom range (iter 200-800)
    python test_camkii_bistability.py --model 253 --grn-progressive --grn-prog-start 200 --grn-prog-end 800
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy

from embryo import model


# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser(description='Test CaMKII bistability with optional learned parameters')
parser.add_argument('--paramsFile', type=str, default=None,
                    help='Path to learned parameters file (.dat format)')
parser.add_argument('--outputFile', type=str, default='data/camkii_bistability_test.png',
                    help='Output visualization filename (default: data/camkii_bistability_test.png)')
parser.add_argument('--numBioSteps', type=int, default=1000,
                    help='Number of bioelectric formation steps (default: 1000)')
parser.add_argument('--numTotalSteps', type=int, default=2000,
                    help='Total number of steps including decay test (default: 2000)')
parser.add_argument('--recordInterval', type=int, default=100,
                    help='Checkpoint recording interval (default: 100)')
parser.add_argument('--model', type=str, default='Stigmergic',
                    choices=['Stigmergic', '253'],
                    help='Model to use: Stigmergic or 253 (default: Stigmergic)')
parser.add_argument('--perturb-vmem', type=float, default=None, dest='perturb_vmem',
                    help='Add Gaussian noise to Vmem with specified std dev (e.g., 0.01 for 10mV noise)')
parser.add_argument('--perturb-iter', type=int, default=None, dest='perturb_iter',
                    help='Iteration at which to apply Vmem perturbation (default: numBioSteps, i.e., at pattern lock time)')
parser.add_argument('--perturb-duration', type=int, default=1, dest='perturb_duration',
                    help='Number of iterations to apply perturbation (default: 1, single application)')
parser.add_argument('--perturb-seed', type=int, default=None, dest='perturb_seed',
                    help='Random seed for Vmem perturbation (for reproducibility)')
parser.add_argument('--grn-damping', type=float, default=1.0, dest='grn_damping',
                    help='GRN weight damping factor [0,1]: 0=disable GRN, 1=native weights (default: 1.0)')
parser.add_argument('--grn-progressive', action='store_true', dest='grn_progressive',
                    help='Enable progressive GRN damping: gradually increase from 0.0 to 1.0 over time range')
parser.add_argument('--grn-prog-start', type=int, default=0, dest='grn_prog_start',
                    help='Start iteration for progressive GRN damping (default: 0)')
parser.add_argument('--grn-prog-end', type=int, default=None, dest='grn_prog_end',
                    help='End iteration for progressive GRN damping (default: numBioSteps)')
parser.add_argument('--visualize-timeseries', action='store_true', dest='visualize_timeseries',
                    help='Generate per-cell timeseries heatmaps for Vmem, Ca, and CaMKII')
args = parser.parse_args()


# ============================================================
# Parameter loading utilities
# ============================================================
def apply_sigmoid_constraint(raw_param, min_val, max_val):
    """Map unbounded raw parameter to bounded range via sigmoid"""
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)


def load_learned_parameters(params_file):
    """
    Load learned CaMKII parameters from file.

    Args:
        params_file: Path to .dat file with learned parameters

    Returns:
        dict with parameter names -> constrained values
    """
    print(f"Loading learned parameters from: {params_file}")
    data = torch.load(params_file, weights_only=False)

    learned_params = {}
    param_bounds = data.get('parameter_bounds', {})
    raw_params = data.get('parameters', {})

    # Apply sigmoid constraint to convert raw parameters to constrained values
    for param_name, raw_value in raw_params.items():
        min_key = f'{param_name}_min'
        max_key = f'{param_name}_max'

        if min_key in param_bounds and max_key in param_bounds:
            constrained = apply_sigmoid_constraint(
                raw_value,
                param_bounds[min_key],
                param_bounds[max_key]
            )
            learned_params[param_name] = float(constrained.item())
        else:
            # Use raw value if bounds not available
            learned_params[param_name] = float(raw_value.item())

    print(f"Loaded {len(learned_params)} parameters:")
    for name, value in learned_params.items():
        print(f"  {name}: {value:.4f}")
    print()

    return learned_params


class SimpleCaMKII:
    """
    CaMKII bistable switch with Mechanism 1: Autophosphorylation with Competitive Dynamics.

    Architecture:
    1. Vmem → Ca²⁺ (with tau_ca for temporal integration/delay)
    2. Ca²⁺ → external drive (sigmoid activation, range [0,1])
    3. CaMKII → self-activation (competitive dynamics [-1, 1])
    4. OR gate: additive threshold combining gain_ca * ca_signal + self_activation
       - Learnable gain_ca allows Ca²⁺ to overcome initial inhibition
    5. CaMKII bistable dynamics: activation drives k_on, competitive self_activation modulates state
    """

    def __init__(self, grid_size, device='cpu', dtype=torch.float64, learned_params=None):
        self.grid_size = grid_size
        self.device = device
        self.dtype = dtype

        # CaMKII state (0 = inactive, 1 = active)
        # Initialize near 0 (background/inhibited state)
        self.CaMKII_active = torch.rand(grid_size, grid_size, device=device, dtype=dtype) * 0.01

        # Ca²⁺ state (for transduction)
        self.Ca = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)

        # Default parameters (can be overridden by learned_params)
        # Note: These defaults match the initial values in learn_camkii_bistability.py
        defaults = {
            # Ca²⁺ dynamics
            'tau_ca': 20.0,           # Ca²⁺ decay time constant (CRITICAL for delayed patterning)
            'V_half_ca': -0.04,       # Voltage for half-maximal Ca²⁺ activation
            'k_ca': 0.01,             # Voltage sensitivity of Ca²⁺ channels
            'g_ca': 1.0,              # Ca²⁺ channel conductance
            'k_decay_ca': 0.0,        # Additional Ca²⁺ decay rate (allows faster decay than 1/tau_ca)
            # CaMKII bistable dynamics (learning range: k_on [0.5, 5.0])
            'k_on': 1.0,              # Activation rate
            'k_off': 0.01,            # Inactivation rate (CRITICAL for bistability persistence)
            'ca_threshold': 0.3,      # Ca²⁺ threshold for external drive
            'ca_sensitivity': 0.1,    # Sharpness of Ca²⁺ activation
            'K_half': 0.5,            # Bistability threshold for self-activation
            'tau_camkii': 100.0,      # CaMKII time constant (slows dynamics relative to Ca²⁺)
            # OR gate parameters (tighter ranges to prevent saturation)
            'or_threshold': 0.5,      # Threshold for combined activation (learning range: [0.2, 1.5])
            'or_sharpness': 5.0,      # Sharpness of OR gate (learning range: [1.0, 20.0])
            'gain_ca': 2.0,           # Gain on Ca²⁺ signal (learning range: [1.5, 3.0])
        }

        # Override with learned parameters if provided
        if learned_params is not None:
            for key in defaults.keys():
                if key in learned_params:
                    defaults[key] = learned_params[key]

        # Set parameters
        self.tau_ca = torch.tensor(defaults['tau_ca'], device=device, dtype=dtype)
        self.V_half_ca = torch.tensor(defaults['V_half_ca'], device=device, dtype=dtype)
        self.k_ca = torch.tensor(defaults['k_ca'], device=device, dtype=dtype)
        self.g_ca = torch.tensor(defaults['g_ca'], device=device, dtype=dtype)
        self.k_decay_ca = torch.tensor(defaults['k_decay_ca'], device=device, dtype=dtype)
        self.E_ca = torch.tensor(0.13, device=device, dtype=dtype)  # +130mV (fixed)
        self.k_on = torch.tensor(defaults['k_on'], device=device, dtype=dtype)
        self.k_off = torch.tensor(defaults['k_off'], device=device, dtype=dtype)
        self.ca_threshold = torch.tensor(defaults['ca_threshold'], device=device, dtype=dtype)
        self.ca_sensitivity = torch.tensor(defaults['ca_sensitivity'], device=device, dtype=dtype)
        self.K_half = torch.tensor(defaults['K_half'], device=device, dtype=dtype)
        self.tau_camkii = torch.tensor(defaults['tau_camkii'], device=device, dtype=dtype)
        self.or_threshold = torch.tensor(defaults['or_threshold'], device=device, dtype=dtype)
        self.or_sharpness = torch.tensor(defaults['or_sharpness'], device=device, dtype=dtype)
        self.gain_ca = torch.tensor(defaults['gain_ca'], device=device, dtype=dtype)

        # History for diagnostics
        self.ca_history = []
        self.camkii_history = []

    def update(self, vmem_grid, dt=0.01):
        """
        Update Ca²⁺ and CaMKII states with competitive dynamics.

        Args:
            vmem_grid: (grid_size, grid_size) membrane voltage in Volts
            dt: timestep

        Returns:
            dict with Ca²⁺, CaMKII, and activation components
        """
        # 1. Ca²⁺ dynamics (voltage-gated channels with temporal integration + additional decay)
        ca_activation = torch.sigmoid((vmem_grid - self.V_half_ca) / self.k_ca)
        driving_force = self.E_ca - vmem_grid
        I_ca = self.g_ca * ca_activation * (driving_force / 0.1)
        I_ca = torch.clamp(I_ca, min=0.0)

        # Ca²⁺ decay: baseline (1/tau_ca) + additional learnable decay (k_decay_ca)
        # This allows Ca²⁺ to decay faster than passive diffusion when needed
        dCa_dt = I_ca - (1.0 / self.tau_ca) * self.Ca - self.k_decay_ca
        self.Ca = self.Ca + dt * dCa_dt
        self.Ca = torch.clamp(self.Ca, min=0.0, max=10.0)

        # 2. External drive: Ca²⁺ activation (range: [0, 1])
        ca_signal = torch.sigmoid((self.Ca - self.ca_threshold) / self.ca_sensitivity)

        # 3. Internal drive: Competitive self-activation (range: [-1, 1])
        # Active subunits promote activity (+), inactive subunits inhibit (-)
        CaMKII_sq = self.CaMKII_active * self.CaMKII_active
        K_half_sq = self.K_half * self.K_half

        # Competitive dynamics: (active - inactive) / (active + inactive)
        # Maps: CaMKII=0 → -1 (inhibited), CaMKII=K_half → 0 (unstable), CaMKII=1 → +1 (excited)
        self_activation = (CaMKII_sq - K_half_sq) / (K_half_sq + CaMKII_sq)

        # 4. OR gate with learnable Ca²⁺ gain
        # Gain allows Ca²⁺ signal to overcome initial inhibition (self_activation = -1)
        # Both drivers contribute additively, gradients flow through both branches
        combined_signal = self.gain_ca * ca_signal + self_activation - self.or_threshold

        # Use softplus instead of ReLU to maintain gradient flow even when saturated
        activation = torch.nn.functional.softplus(combined_signal * self.or_sharpness, beta=1.0) / self.or_sharpness

        # 5. Update CaMKII with bistable dynamics
        dCaMKII_dt = (self.k_on * activation - self.k_off * self.CaMKII_active) / self.tau_camkii

        self.CaMKII_active = self.CaMKII_active + dt * dCaMKII_dt
        self.CaMKII_active = torch.clamp(self.CaMKII_active, min=0.0, max=1.0)

        return {
            'Ca': self.Ca,
            'CaMKII': self.CaMKII_active,
            'ca_signal': ca_signal,
            'self_activation': self_activation,
            'combined_signal': combined_signal,
            'activation': activation,
            'vmem': vmem_grid
        }

    def record_state(self):
        """Record current state for diagnostics"""
        self.ca_history.append(self.Ca.mean().item())
        self.camkii_history.append(self.CaMKII_active.mean().item())


def load_model_parameters(model_type='Stigmergic', grn_damping=1.0):
    """
    Load model parameters based on model type.

    Args:
        model_type: 'Stigmergic' or '253'
        grn_damping: GRN weight damping factor [0,1]. 0=disable GRN, 1=native weights.

    Returns:
        dict with model parameters
    """
    from torch.serialization import add_safe_globals
    add_safe_globals([np.core.multiarray._reconstruct])

    if model_type == 'Stigmergic':
        path = 'data/StigmergicModelParameters.dat'
    elif model_type == '253':
        path = './data/bestModelParameters_fieldVector_Ligand_GRN_253.dat'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Loading {model_type} model parameters from: {path}")
    params = torch.load(path, weights_only=False)

    if "ATPParameters" not in params:
        params["ATPParameters"] = None

    # Apply GRN damping if specified
    if grn_damping != 1.0 and 'GRNParameters' in params and params['GRNParameters'] is not None:
        grn_params = params['GRNParameters']
        print(f"Applying GRN damping factor: {grn_damping:.2f}")

        # Damp GRN weights
        if 'GRNWeights' in grn_params and grn_params['GRNWeights'] is not None:
            grn_params['GRNWeights'] = grn_params['GRNWeights'] * grn_damping
            print(f"  GRNWeights *= {grn_damping:.2f}")

        # Damp inter-GRN weights
        if 'InterGRNWeights' in grn_params and grn_params['InterGRNWeights'] is not None:
            grn_params['InterGRNWeights'] = grn_params['InterGRNWeights'] * grn_damping
            print(f"  InterGRNWeights *= {grn_damping:.2f}")

        # Damp GRN-to-ligand weights
        if 'GRNtoLigandWeights' in grn_params and grn_params['GRNtoLigandWeights'] is not None:
            grn_params['GRNtoLigandWeights'] = grn_params['GRNtoLigandWeights'] * grn_damping
            print(f"  GRNtoLigandWeights *= {grn_damping:.2f}")

        # If damping is 0, also disable GRN entirely
        if grn_damping == 0.0:
            grn_params['GRNEnabled'] = False
            print(f"  GRNEnabled = False (damping=0)")

    return params


def run_bioelectric_with_camkii(params, num_bio_steps=1000, num_total_steps=2000, record_interval=100,
                                 learned_params=None, model_name='Stigmergic',
                                 perturb_vmem=None, perturb_iter=None, perturb_duration=1, perturb_seed=None,
                                 grn_progressive=False, grn_prog_start=0, grn_prog_end=None):
    """
    Run bioelectric model with CaMKII tracking.

    Phase 1 (0-num_bio_steps): Run bioelectric simulation with CaMKII tracking
    Phase 2 (num_bio_steps-num_total_steps): Hold Vmem constant, let CaMKII evolve

    Args:
        params: Model parameters dictionary
        num_bio_steps: Number of bioelectric formation steps
        num_total_steps: Total simulation steps including decay test
        record_interval: Checkpoint recording interval
        learned_params: Optional learned CaMKII parameters
        model_name: Name of model being used
        perturb_vmem: Standard deviation of Gaussian noise to add to Vmem (None to skip)
        perturb_iter: Iteration at which to start perturbation (default: num_bio_steps)
        perturb_duration: Number of iterations to apply perturbation (default: 1)
        perturb_seed: Random seed for perturbation (for reproducibility)
        grn_progressive: If True, gradually increase GRN damping from 0.0 to 1.0
        grn_prog_start: Start iteration for progressive damping (default: 0)
        grn_prog_end: End iteration for progressive damping (default: num_bio_steps)

    Returns:
        dict with vmem and camkii states at checkpoints
    """
    print(f"=== Running {model_name} Model with CaMKII Tracking ===")
    if learned_params is not None:
        print(f"Using LEARNED parameters from file")
    else:
        print(f"Using DEFAULT parameters")
    print(f"Phase 1: Bioelectric formation with CaMKII tracking (0-{num_bio_steps} steps)")
    print(f"Phase 2: Fixed Vmem, CaMKII decay test ({num_bio_steps}-{num_total_steps} steps)")
    print(f"Record interval: {record_interval}")

    # Setup perturbation parameters
    if perturb_vmem is not None:
        # Default perturbation iteration is at pattern lock time (num_bio_steps)
        if perturb_iter is None:
            perturb_iter = num_bio_steps
        perturb_end = perturb_iter + perturb_duration - 1
        print(f"Vmem perturbation: std={perturb_vmem:.4f}V, iter=[{perturb_iter}, {perturb_end}], seed={perturb_seed}")

    # Setup progressive GRN damping
    if grn_prog_end is None:
        grn_prog_end = num_bio_steps

    # Initialize bioelectric model first (weights are transformed during init)
    num_samples = params["simParameters"]["numSamples"]
    initial_values = copy.deepcopy(params["simParameters"]["initialValues"])
    external_inputs = copy.deepcopy(params["simParameters"]["externalInputs"])
    clamp_params = copy.deepcopy(params["clampParameters"])

    bio_model = model(params, numBasicSamples=num_samples)
    bio_model.setExperimentalConditions((initial_values, num_samples))

    # Store original GRN weights for progressive damping AFTER model init
    # (model transforms weights during initialization - transpose, divide by timeconstant)
    original_grn_weights = {}
    if grn_progressive and bio_model.GRNEnabled and hasattr(bio_model, 'geneNetwork'):
        grn = bio_model.geneNetwork
        if grn.GRNWeights is not None:
            original_grn_weights['GRNWeights'] = grn.GRNWeights.clone()
        if grn.InterGRNWeights is not None:
            original_grn_weights['InterGRNWeights'] = grn.InterGRNWeights.clone()
        # GRNtoLigandWeights is in electricNetwork, already transposed
        if hasattr(bio_model.electricNetwork, 'GRNtoLigandWeights') and bio_model.electricNetwork.GRNtoLigandWeights is not None:
            original_grn_weights['GRNtoLigandWeights'] = bio_model.electricNetwork.GRNtoLigandWeights.clone()
        print(f"Progressive GRN damping: 0.0 -> 1.0 over iter=[{grn_prog_start}, {grn_prog_end}]")

    # Initialize CaMKII tracker with learned parameters
    grid_size = params['latticeDims'][0]  # Assume square
    camkii_tracker = SimpleCaMKII(grid_size=grid_size, device='cpu', learned_params=learned_params)

    # Storage for checkpoints
    checkpoints = {
        'times': [],
        'vmem': [],
        'ca': [],
        'camkii': [],
        'vmem_mean': [],
        'vmem_std': [],
        'ca_mean': [],
        'ca_std': [],
        'camkii_mean': [],
        'camkii_std': [],
        # Perturbation info
        'perturb_vmem': perturb_vmem,
        'perturb_iter': perturb_iter if perturb_vmem is not None else None,
        'perturb_duration': perturb_duration if perturb_vmem is not None else None,
        'perturb_seed': perturb_seed,
        # Progressive GRN damping info
        'grn_progressive': grn_progressive,
        'grn_prog_start': grn_prog_start if grn_progressive else None,
        'grn_prog_end': grn_prog_end if grn_progressive else None,
        # Clamp info
        'clamp_end_iter': params['clampParameters'].get('clampEndIter', None) if params.get('clampParameters') else None,
    }

    # Get initial Vmem pattern before simulation
    initial_vmem_grid = bio_model.electricNetwork.Vmem[0, :, 0].reshape(grid_size, grid_size).clone()

    # Phase 1: Run bioelectric simulation to establish pattern
    # Record Vmem timeseries for use in Phase 2
    print("\n  Phase 1: Running bioelectric simulation to establish face pattern...")
    vmem_timeseries = [initial_vmem_grid.clone()]  # Store Vmem at each timestep

    # Always run iteratively to capture Vmem timeseries
    print(f"  (Running iteratively to capture Vmem timeseries)")

    if grn_progressive and len(original_grn_weights) > 0:
        print(f"  (Progressive GRN damping enabled)")
        # Check if model has GRN enabled
        if not bio_model.GRNEnabled:
            print("  WARNING: GRN is not enabled in model, progressive damping will have no effect")

    for iter_idx in range(num_bio_steps):
    # for iter_idx in range(num_total_steps):
        # Apply progressive GRN damping if enabled
        if grn_progressive and len(original_grn_weights) > 0:
            # Compute current damping factor
            if (iter_idx < grn_prog_start) or (iter_idx > grn_prog_end):
                damping = 1.0
            else:
                # Linear interpolation from minDamp to maxDamp
                minDamp, maxDamp = 0.0, 1.0
                damping = ((maxDamp - minDamp) * (iter_idx - grn_prog_start) / max(1, grn_prog_end - grn_prog_start)) + minDamp

            # Apply damping to GRN weights - must update the model's internal GRN weights
            # and recompute tissueGRNWeights, not just the params dict
            if bio_model.GRNEnabled and hasattr(bio_model, 'geneNetwork'):
                grn = bio_model.geneNetwork
                for key, orig_weight in original_grn_weights.items():
                    damped_weight = orig_weight * damping
                    # Update the GRN's internal weight attribute
                    if key == 'GRNWeights':
                        grn.GRNWeights = damped_weight
                    elif key == 'InterGRNWeights':
                        grn.InterGRNWeights = damped_weight
                    elif key == 'GRNtoLigandWeights':
                        # GRNtoLigandWeights is stored in electricNetwork, not geneNetwork
                        bio_model.electricNetwork.GRNtoLigandWeights = damped_weight
                # Recompute tissue-level GRN weights from updated weights
                grn.composeTissueLevelGRN()

        # Run one simulation step
        bio_model.simulate(
            externalInputs=external_inputs,
            clampParameters=clamp_params,
            perturbation=None,
            fieldModulation=False,
            numSimIters=1,
            outerIter=iter_idx
        )

        # Record current Vmem
        current_vmem = bio_model.electricNetwork.Vmem[0, :, 0].reshape(grid_size, grid_size).clone()
        vmem_timeseries.append(current_vmem)

        # Progress update
        if (iter_idx + 1) % 200 == 0:
            if grn_progressive and len(original_grn_weights) > 0:
                print(f"    Iter {iter_idx+1}/{num_bio_steps}: damping={damping:.3f}, "
                      f"Vmem std={current_vmem.std().item():.4f}V")
            else:
                print(f"    Iter {iter_idx+1}/{num_bio_steps}: "
                      f"Vmem std={current_vmem.std().item():.4f}V")

    # Restore original weights after progressive phase
    if grn_progressive and len(original_grn_weights) > 0:
        if bio_model.GRNEnabled and hasattr(bio_model, 'geneNetwork'):
            grn = bio_model.geneNetwork
            for key, orig_weight in original_grn_weights.items():
                if key == 'GRNWeights':
                    grn.GRNWeights = orig_weight
                elif key == 'InterGRNWeights':
                    grn.InterGRNWeights = orig_weight
                elif key == 'GRNtoLigandWeights':
                    bio_model.electricNetwork.GRNtoLigandWeights = orig_weight
            grn.composeTissueLevelGRN()

    # Get final Vmem pattern from bioelectric simulation
    vmem_final = vmem_timeseries[-1]
    print(f"  Bioelectric simulation complete!")
    print(f"  Initial Vmem: mean={initial_vmem_grid.mean().item():.4f}V, std={initial_vmem_grid.std().item():.4f}V")
    print(f"  Final Vmem: mean={vmem_final.mean().item():.4f}V, std={vmem_final.std().item():.4f}V")

    # Phase 2: Track CaMKII response and persistence
    print(f"\n  Phase 2: Tracking CaMKII from t=0 to t={num_total_steps}...")
    print(f"  (Using bioelectric pattern from Phase 1, then testing decay)")

    dt = 0.01
    for t in range(num_total_steps):
        # For t < num_bio_steps: use actual recorded Vmem from bioelectric simulation
        # For t >= num_bio_steps: test if CaMKII persists even if Vmem changes

        if t < num_bio_steps:
        # if t < num_total_steps:
            # Use actual Vmem from recorded timeseries
            # vmem_timeseries has num_bio_steps + 1 entries (initial + each step)
            vmem_grid = vmem_timeseries[t]
        else:
            # After num_bio_steps: decay back toward initial (uniform) pattern
            # This tests whether CaMKII can retain the face pattern even when Vmem loses it
            decay_progress = (t - num_bio_steps) / (num_total_steps - num_bio_steps)
            decay_progress = min(1.0, decay_progress)
            vmem_grid = (1 - decay_progress) * vmem_final + decay_progress * initial_vmem_grid

        # Apply Vmem perturbation if requested
        if perturb_vmem is not None:
            perturb_end = perturb_iter + perturb_duration - 1
            if perturb_iter <= t <= perturb_end:
                # Set random seed only on first perturbation iteration (for reproducibility)
                if t == perturb_iter and perturb_seed is not None:
                    torch.manual_seed(perturb_seed)
                    np.random.seed(perturb_seed)

                # Apply Gaussian noise to Vmem
                noise = torch.randn_like(vmem_grid) * perturb_vmem
                vmem_grid = vmem_grid + noise

                if t == perturb_iter:
                    print(f"  [t={t}] Applied Vmem perturbation: std={perturb_vmem:.4f}V, "
                          f"noise range=[{noise.min().item():.4f}, {noise.max().item():.4f}]V")

        # Update CaMKII
        states = camkii_tracker.update(vmem_grid, dt=dt)
        camkii_tracker.record_state()

        # Record checkpoint
        if t % record_interval == 0 or t == num_total_steps - 1:
            checkpoints['times'].append(t)
            checkpoints['vmem'].append(vmem_grid.clone())
            checkpoints['ca'].append(states['Ca'].clone())
            checkpoints['camkii'].append(states['CaMKII'].clone())
            checkpoints['vmem_mean'].append(vmem_grid.mean().item())
            checkpoints['vmem_std'].append(vmem_grid.std().item())
            checkpoints['ca_mean'].append(states['Ca'].mean().item())
            checkpoints['ca_std'].append(states['Ca'].std().item())
            checkpoints['camkii_mean'].append(states['CaMKII'].mean().item())
            checkpoints['camkii_std'].append(states['CaMKII'].std().item())

            print(f"  t={t:4d}: Vmem std={vmem_grid.std().item():.4f}V, "
                  f"Ca mean={states['Ca'].mean().item():.3f}, "
                  f"CaMKII mean={states['CaMKII'].mean().item():.3f}")

    # Add full history
    checkpoints['ca_history'] = camkii_tracker.ca_history
    checkpoints['camkii_history'] = camkii_tracker.camkii_history

    return checkpoints, bio_model, camkii_tracker


def analyze_pattern_retention(checkpoints, t_lock=1000, t_test=2000):
    """
    Analyze whether CaMKII retains pattern from t_lock at t_test.

    Args:
        checkpoints: dict from run_stigmergic_with_camkii
        t_lock: time when pattern should be locked (default 1000)
        t_test: time to test retention (default 2000)

    Returns:
        dict with analysis results
    """
    print(f"\n=== Pattern Retention Analysis ===")

    # Get indices for target times (find closest checkpoint to each target time)
    times = checkpoints['times']
    idx_lock = min(range(len(times)), key=lambda i: abs(times[i] - t_lock))
    idx_test = min(range(len(times)), key=lambda i: abs(times[i] - t_test))

    # Extract states
    vmem_lock = checkpoints['vmem'][idx_lock]
    vmem_test = checkpoints['vmem'][idx_test]
    camkii_lock = checkpoints['camkii'][idx_lock]
    camkii_test = checkpoints['camkii'][idx_test]

    # Compute correlations
    vmem_corr = torch.corrcoef(torch.stack([
        vmem_lock.flatten(),
        vmem_test.flatten()
    ]))[0, 1].item()

    camkii_corr = torch.corrcoef(torch.stack([
        camkii_lock.flatten(),
        camkii_test.flatten()
    ]))[0, 1].item()

    # Correlation between Vmem at t_lock and CaMKII at t_test
    cross_corr = torch.corrcoef(torch.stack([
        vmem_lock.flatten(),
        camkii_test.flatten()
    ]))[0, 1].item()

    # Check for uniform patterns (zero variance) - calculate BEFORE creating results
    vmem_lock_std = vmem_lock.std().item()
    vmem_test_std = vmem_test.std().item()
    camkii_lock_std = camkii_lock.std().item()
    camkii_test_std = camkii_test.std().item()

    results = {
        'vmem_correlation': vmem_corr,
        'camkii_correlation': camkii_corr,
        'cross_correlation': cross_corr,
        'vmem_lock': vmem_lock,
        'vmem_test': vmem_test,
        'camkii_lock': camkii_lock,
        'camkii_test': camkii_test,
        'vmem_lock_std': vmem_lock_std,
        'vmem_test_std': vmem_test_std,
        'camkii_lock_std': camkii_lock_std,
        'camkii_test_std': camkii_test_std
    }

    print(f"\nSpatial Variance Check:")
    print(f"  Vmem(t={t_lock}):   mean={vmem_lock.mean().item():.4f}, std={vmem_lock_std:.4f}")
    print(f"  Vmem(t={t_test}):   mean={vmem_test.mean().item():.4f}, std={vmem_test_std:.4f}")
    print(f"  CaMKII(t={t_lock}): mean={camkii_lock.mean().item():.4f}, std={camkii_lock_std:.4f}")
    print(f"  CaMKII(t={t_test}): mean={camkii_test.mean().item():.4f}, std={camkii_test_std:.4f}")

    # Warn about degenerate correlations
    if camkii_lock_std < 0.01 or camkii_test_std < 0.01:
        print(f"\n  ⚠⚠ WARNING: CaMKII pattern has very low variance (std < 0.01)!")
        print(f"      Pattern is nearly UNIFORM - correlation is meaningless!")
        camkii_corr = 0.0  # Override with 0 to indicate failure

    print(f"\nPattern Correlations:")
    print(f"  Vmem(t={t_lock}) vs Vmem(t={t_test}):   {vmem_corr:.3f}")
    print(f"  CaMKII(t={t_lock}) vs CaMKII(t={t_test}): {camkii_corr:.3f}")
    print(f"  Vmem(t={t_lock}) vs CaMKII(t={t_test}):  {cross_corr:.3f}")

    print(f"\nInterpretation:")
    if vmem_corr > 0.8:
        print(f"  ✓ Vmem pattern is STABLE (corr={vmem_corr:.3f} > 0.8)")
    elif vmem_corr > 0.5:
        print(f"  ⚠ Vmem pattern DRIFTS slowly (corr={vmem_corr:.3f})")
    else:
        print(f"  ✗ Vmem pattern is UNSTABLE (corr={vmem_corr:.3f} < 0.5)")

    if camkii_corr > 0.85:
        print(f"  ✓ CaMKII pattern is HIGHLY STABLE (corr={camkii_corr:.3f} > 0.85)")
    else:
        print(f"  ⚠ CaMKII pattern degrades (corr={camkii_corr:.3f} < 0.85)")

    if cross_corr > 0.7:
        print(f"  ✓✓ CaMKII at t={t_test} RETAINS Vmem pattern from t={t_lock} (corr={cross_corr:.3f})")
        print(f"     → Pattern locking SUCCESSFUL!")
    else:
        print(f"  ✗ CaMKII does not retain original pattern (corr={cross_corr:.3f} < 0.7)")

    return results


def visualize_results(checkpoints, analysis, t_lock=1000, output_path='camkii_bistability_test.png'):
    """
    Visualize Vmem, Ca²⁺, and CaMKII patterns at key timepoints.

    Args:
        checkpoints: dict with simulation data
        analysis: dict with analysis results
        t_lock: time when pattern is locked (default 1000)
        output_path: path to save visualization
    """
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)

    # Get key timepoints dynamically
    times = checkpoints['times']
    t_final = times[-1]

    # Find closest indices to desired timepoints
    def find_closest_idx(target_time):
        return min(range(len(times)), key=lambda i: abs(times[i] - target_time))

    # Five timepoints: t=0, early (t_lock/2), lock (t_lock), mid (between lock and final), final
    t_early = t_lock // 2
    t_mid = (t_lock + t_final) // 2

    idx_zero = 0  # t=0
    idx_early = find_closest_idx(t_early)
    idx_lock = find_closest_idx(t_lock)
    idx_mid = find_closest_idx(t_mid)
    idx_final = -1  # Last timepoint

    # Color maps
    cmap_vmem = 'coolwarm'
    cmap_ca = 'coolwarm'
    cmap_camkii = 'coolwarm'

    # Row 1: Vmem at different times
    vmem_vmin = torch.stack(checkpoints['vmem']).min().item()
    vmem_vmax = torch.stack(checkpoints['vmem']).max().item()

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(checkpoints['vmem'][idx_zero].cpu(), cmap=cmap_vmem, vmin=vmem_vmin, vmax=vmem_vmax)
    ax0.set_title(f'Vmem at t={times[idx_zero]}\n(Initial)', fontsize=10, fontweight='bold')
    ax0.axis('off')
    plt.colorbar(im0, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(checkpoints['vmem'][idx_early].cpu(), cmap=cmap_vmem, vmin=vmem_vmin, vmax=vmem_vmax)
    ax1.set_title(f'Vmem at t={times[idx_early]}', fontsize=10, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(checkpoints['vmem'][idx_lock].cpu(), cmap=cmap_vmem, vmin=vmem_vmin, vmax=vmem_vmax)
    ax2.set_title(f'Vmem at t={times[idx_lock]}\n(Pattern Lock Time)', fontsize=10, fontweight='bold', color='red')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(checkpoints['vmem'][idx_mid].cpu(), cmap=cmap_vmem, vmin=vmem_vmin, vmax=vmem_vmax)
    ax3.set_title(f'Vmem at t={times[idx_mid]}', fontsize=10, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    ax4 = fig.add_subplot(gs[0, 4])
    im4 = ax4.imshow(checkpoints['vmem'][idx_final].cpu(), cmap=cmap_vmem, vmin=vmem_vmin, vmax=vmem_vmax)
    ax4.set_title(f'Vmem at t={t_final}\n(Final Time)', fontsize=10, fontweight='bold', color='blue')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # Row 2: Ca²⁺ at different times
    ca_vmin = torch.stack(checkpoints['ca']).min().item()
    ca_vmax = torch.stack(checkpoints['ca']).max().item()

    ax_ca0 = fig.add_subplot(gs[1, 0])
    im_ca0 = ax_ca0.imshow(checkpoints['ca'][idx_zero].cpu(), cmap=cmap_ca, vmin=ca_vmin, vmax=ca_vmax)
    ax_ca0.set_title(f'Ca²⁺ at t={times[idx_zero]}\n(Initial)', fontsize=10, fontweight='bold')
    ax_ca0.axis('off')
    plt.colorbar(im_ca0, ax=ax_ca0, fraction=0.046)

    ax_ca1 = fig.add_subplot(gs[1, 1])
    im_ca1 = ax_ca1.imshow(checkpoints['ca'][idx_early].cpu(), cmap=cmap_ca, vmin=ca_vmin, vmax=ca_vmax)
    ax_ca1.set_title(f'Ca²⁺ at t={times[idx_early]}', fontsize=10, fontweight='bold')
    ax_ca1.axis('off')
    plt.colorbar(im_ca1, ax=ax_ca1, fraction=0.046)

    ax_ca2 = fig.add_subplot(gs[1, 2])
    im_ca2 = ax_ca2.imshow(checkpoints['ca'][idx_lock].cpu(), cmap=cmap_ca, vmin=ca_vmin, vmax=ca_vmax)
    ax_ca2.set_title(f'Ca²⁺ at t={times[idx_lock]}\n(Transduction)', fontsize=10, fontweight='bold', color='red')
    ax_ca2.axis('off')
    plt.colorbar(im_ca2, ax=ax_ca2, fraction=0.046)

    ax_ca3 = fig.add_subplot(gs[1, 3])
    im_ca3 = ax_ca3.imshow(checkpoints['ca'][idx_mid].cpu(), cmap=cmap_ca, vmin=ca_vmin, vmax=ca_vmax)
    ax_ca3.set_title(f'Ca²⁺ at t={times[idx_mid]}', fontsize=10, fontweight='bold')
    ax_ca3.axis('off')
    plt.colorbar(im_ca3, ax=ax_ca3, fraction=0.046)

    ax_ca4 = fig.add_subplot(gs[1, 4])
    im_ca4 = ax_ca4.imshow(checkpoints['ca'][idx_final].cpu(), cmap=cmap_ca, vmin=ca_vmin, vmax=ca_vmax)
    ax_ca4.set_title(f'Ca²⁺ at t={t_final}\n(Decayed)', fontsize=10, fontweight='bold', color='blue')
    ax_ca4.axis('off')
    plt.colorbar(im_ca4, ax=ax_ca4, fraction=0.046)

    # Row 3: CaMKII at different times
    ax_ck0 = fig.add_subplot(gs[2, 0])
    im_ck0 = ax_ck0.imshow(checkpoints['camkii'][idx_zero].cpu(), cmap=cmap_camkii, vmin=0, vmax=1)
    ax_ck0.set_title(f'CaMKII at t={times[idx_zero]}\n(Initial)', fontsize=10, fontweight='bold')
    ax_ck0.axis('off')
    plt.colorbar(im_ck0, ax=ax_ck0, fraction=0.046)

    ax5 = fig.add_subplot(gs[2, 1])
    im5 = ax5.imshow(checkpoints['camkii'][idx_early].cpu(), cmap=cmap_camkii, vmin=0, vmax=1)
    ax5.set_title(f'CaMKII at t={times[idx_early]}', fontsize=10, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    ax6 = fig.add_subplot(gs[2, 2])
    im6 = ax6.imshow(checkpoints['camkii'][idx_lock].cpu(), cmap=cmap_camkii, vmin=0, vmax=1)
    ax6.set_title(f'CaMKII at t={times[idx_lock]}\n(Locked Pattern)', fontsize=10, fontweight='bold', color='red')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    ax7 = fig.add_subplot(gs[2, 3])
    im7 = ax7.imshow(checkpoints['camkii'][idx_mid].cpu(), cmap=cmap_camkii, vmin=0, vmax=1)
    ax7.set_title(f'CaMKII at t={times[idx_mid]}', fontsize=10, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046)

    ax8 = fig.add_subplot(gs[2, 4])
    im8 = ax8.imshow(checkpoints['camkii'][idx_final].cpu(), cmap=cmap_camkii, vmin=0, vmax=1)
    ax8.set_title(f'CaMKII at t={t_final}\n(Retained Pattern?)', fontsize=10, fontweight='bold', color='blue')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046)

    # Row 4: Time series - Normalized Mean and Variance for Vmem, Ca²⁺, CaMKII
    ax9 = fig.add_subplot(gs[3, 0:2])
    x_checkpoints = np.arange(len(checkpoints['vmem_mean']))

    # Convert to numpy for fill_between
    vmem_mean = np.array(checkpoints['vmem_mean'])
    vmem_std = np.array(checkpoints['vmem_std'])
    ca_mean = np.array(checkpoints['ca_mean'])
    ca_std = np.array(checkpoints['ca_std'])
    camkii_mean = np.array(checkpoints['camkii_mean'])
    camkii_std = np.array(checkpoints['camkii_std'])

    # Normalize each signal to [0, 1] range using min-max normalization
    def normalize(mean_arr, std_arr):
        """Normalize mean and std to [0, 1] range based on mean's min/max"""
        min_val = (mean_arr - std_arr).min()
        max_val = (mean_arr + std_arr).max()
        range_val = max_val - min_val
        if range_val < 1e-10:
            range_val = 1.0  # Avoid division by zero
        norm_mean = (mean_arr - min_val) / range_val
        norm_std = std_arr / range_val
        return norm_mean, norm_std

    vmem_mean_norm, vmem_std_norm = normalize(vmem_mean, vmem_std)
    ca_mean_norm, ca_std_norm = normalize(ca_mean, ca_std)
    camkii_mean_norm, camkii_std_norm = normalize(camkii_mean, camkii_std)

    # Plot Vmem (blue)
    ax9.plot(x_checkpoints, vmem_mean_norm, 'b-', label='Vmem', linewidth=2)
    ax9.fill_between(x_checkpoints,
                     vmem_mean_norm - vmem_std_norm,
                     vmem_mean_norm + vmem_std_norm,
                     color='blue', alpha=0.2)

    # Plot Ca²⁺ (orange)
    ax9.plot(x_checkpoints, ca_mean_norm, 'orange', label='Ca²⁺', linewidth=2)
    ax9.fill_between(x_checkpoints,
                     ca_mean_norm - ca_std_norm,
                     ca_mean_norm + ca_std_norm,
                     color='orange', alpha=0.2)

    # Plot CaMKII (green)
    ax9.plot(x_checkpoints, camkii_mean_norm, 'g-', label='CaMKII', linewidth=2)
    ax9.fill_between(x_checkpoints,
                     camkii_mean_norm - camkii_std_norm,
                     camkii_mean_norm + camkii_std_norm,
                     color='green', alpha=0.2)

    # Vertical lines for key time points
    ax9.axvline(idx_lock, color='r', linestyle='--', label=f't={times[idx_lock]} (lock)', alpha=0.7, linewidth=1.5)
    ax9.axvline(len(times)-1, color='purple', linestyle='--', label=f't={t_final} (final)', alpha=0.7, linewidth=1.5)

    ax9.set_xlabel('Checkpoint (×100 steps)')
    ax9.set_ylabel('Normalized Value [0-1]')
    ax9.legend(loc='upper left', bbox_to_anchor=(1.0, 1), fontsize=8, framealpha=0.9)
    ax9.set_title('Normalized Mean ± Std: Vmem, Ca²⁺, CaMKII', fontsize=10, fontweight='bold')
    ax9.grid(alpha=0.3)
    ax9.set_xlim(0, len(x_checkpoints)-1)
    ax9.set_ylim(-0.1, 1.1)  # Slight padding around [0, 1]

    # Correlation text
    ax10 = fig.add_subplot(gs[3, 2:4])
    ax10.axis('off')

    corr_text = f"""
Pattern Retention Analysis
{'='*40}

Vmem Stability (t={t_lock} → t={t_final}):
    Correlation: {analysis['vmem_correlation']:.3f}
    {'✓ STABLE' if analysis['vmem_correlation'] > 0.8 else '⚠ DRIFTING' if analysis['vmem_correlation'] > 0.5 else '✗ UNSTABLE'}

CaMKII Persistence (t={t_lock} → t={t_final}):
    Correlation: {analysis['camkii_correlation']:.3f}
    {'✓ HIGHLY STABLE' if analysis['camkii_correlation'] > 0.85 else '⚠ DEGRADES'}

Pattern Locking Success:
    Vmem(t={t_lock}) ↔ CaMKII(t={t_final}): {analysis['cross_correlation']:.3f}
    {'✓✓ SUCCESS - CaMKII retains face pattern!' if abs(analysis['cross_correlation']) > 0.7 else '✗ FAILURE - Pattern not retained'}

Interpretation:
    {
    f'CaMKII successfully locked the bioelectric\\nface pattern at t={t_lock} and retained it at\\nt={t_final}, even though Vmem may have changed.\\nThis validates the bistability mechanism!'
    if abs(analysis['cross_correlation']) > 0.7
    else 'CaMKII did not retain the original pattern.\\nMay need to adjust parameters (threshold,\\nrates) or Vmem pattern is too weak.'
    }
    """

    ax10.text(0.1, 0.5, corr_text, fontsize=10, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('CaMKII Bistability Test: Pattern Locking in Stigmergic Model',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_path}")


def visualize_timeseries(checkpoints, t_lock=1000, output_path='camkii_timeseries.png'):
    """
    Visualize per-cell timeseries of Vmem, Ca²⁺, and CaMKII as heatmaps.

    Layout:
    - Top row: Heatmaps (cells × time) for Vmem, Ca²⁺, CaMKII
    - Bottom row: Tissue snapshots at t=0, t_lock, t_final for spatial reference

    Args:
        checkpoints: dict with simulation data (vmem, ca, camkii lists of grids)
        t_lock: time when pattern is locked
        output_path: path to save visualization
    """
    times = checkpoints['times']
    num_checkpoints = len(times)
    t_final = times[-1]

    # Stack checkpoint grids into (num_checkpoints, grid_size, grid_size) tensors
    vmem_stack = torch.stack(checkpoints['vmem'])   # (T, H, W)
    ca_stack = torch.stack(checkpoints['ca'])
    camkii_stack = torch.stack(checkpoints['camkii'])

    grid_size = vmem_stack.shape[1]
    num_cells = grid_size * grid_size

    # Flatten spatial dims: (T, H, W) -> (T, num_cells), then transpose to (num_cells, T)
    vmem_cells = vmem_stack.reshape(num_checkpoints, num_cells).T.cpu().numpy()
    ca_cells = ca_stack.reshape(num_checkpoints, num_cells).T.cpu().numpy()
    camkii_cells = camkii_stack.reshape(num_checkpoints, num_cells).T.cpu().numpy()

    # Find index of t_lock and clamp_end in checkpoint times
    idx_lock = min(range(num_checkpoints), key=lambda i: abs(times[i] - t_lock))
    clamp_end_iter = checkpoints.get('clamp_end_iter', None)
    idx_clamp_end = None
    if clamp_end_iter is not None:
        idx_clamp_end = min(range(num_checkpoints), key=lambda i: abs(times[i] - clamp_end_iter))

    # --- Figure layout ---
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.35, wspace=0.3)

    # Time labels for x-axis
    time_labels = np.array(times)

    signals = [
        ('Vmem (V)', vmem_cells, 'coolwarm', vmem_stack),
        ('Ca²⁺', ca_cells, 'YlOrRd', ca_stack),
        ('CaMKII', camkii_cells, 'YlGnBu', camkii_stack),
    ]

    for col, (label, cell_data, cmap, grid_stack) in enumerate(signals):
        # --- Top row: per-cell heatmap (kymograph) ---
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(cell_data, aspect='auto', cmap=cmap, interpolation='nearest')
        ax.set_xlabel('Time (iteration)')
        ax.set_ylabel('Cell index (row-major)')
        ax.set_title(f'{label} per cell over time', fontsize=12, fontweight='bold')

        # X-axis: show time values at tick positions
        num_ticks = min(10, num_checkpoints)
        tick_positions = np.linspace(0, num_checkpoints - 1, num_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(time_labels[i]) for i in tick_positions], rotation=45, fontsize=8)

        # Y-axis: show a few cell indices
        num_yticks = min(10, num_cells)
        ytick_positions = np.linspace(0, num_cells - 1, num_yticks, dtype=int)
        ax.set_yticks(ytick_positions)

        # Mark clamp end with a vertical line
        if idx_clamp_end is not None:
            ax.axvline(idx_clamp_end, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.text(idx_clamp_end + 0.5, num_cells * 0.05, f'clamp end\nt={clamp_end_iter}', color='orange',
                    fontsize=7, fontweight='bold', va='top')

        # Mark t_lock with a vertical line
        ax.axvline(idx_lock, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.text(idx_lock + 0.5, num_cells * 0.02, f't={t_lock}', color='red',
                fontsize=8, fontweight='bold', va='top')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Bottom row: spatial snapshots at t=0, t_lock, t_final ---
        ax_bottom = fig.add_subplot(gs[1, col])

        # Create a 1x3 subplot within this axes area
        # Use inset axes for the 3 snapshots
        snapshot_indices = [0, idx_lock, -1]
        snapshot_labels = [f't={times[0]}', f't={times[idx_lock]}', f't={t_final}']
        snapshot_colors = ['black', 'red', 'blue']

        vmin = cell_data.min()
        vmax = cell_data.max()

        for i, (s_idx, s_label, s_color) in enumerate(zip(snapshot_indices, snapshot_labels, snapshot_colors)):
            # Position inset axes within the bottom panel
            inset_x = 0.02 + i * 0.34
            inset_ax = ax_bottom.inset_axes([inset_x, 0.05, 0.28, 0.85])
            inset_ax.imshow(grid_stack[s_idx].cpu().numpy(), cmap=cmap,
                           vmin=vmin, vmax=vmax)
            inset_ax.set_title(s_label, fontsize=9, fontweight='bold', color=s_color)
            inset_ax.axis('off')

        ax_bottom.set_title(f'{label} spatial snapshots', fontsize=10)
        ax_bottom.axis('off')

    plt.suptitle('Per-Cell Timeseries: Vmem, Ca²⁺, CaMKII',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved timeseries heatmap: {output_path}")
    plt.close(fig)

    # --- Spatial sparkline grid ---
    _visualize_timeseries_spatial(checkpoints, t_lock, output_path)


def _visualize_timeseries_spatial(checkpoints, t_lock, output_path):
    """
    Visualize per-cell timeseries spatially: a grid of sparklines matching the
    tissue layout. Each cell shows normalized Vmem (blue), Ca²⁺ (orange), and
    CaMKII (green) traces over time.

    Args:
        checkpoints: dict with simulation data
        t_lock: time when pattern is locked
        output_path: base output path (will append '_spatial' before extension)
    """
    times = checkpoints['times']
    num_checkpoints = len(times)
    t_final = times[-1]

    vmem_stack = torch.stack(checkpoints['vmem']).cpu().numpy()  # (T, H, W)
    ca_stack = torch.stack(checkpoints['ca']).cpu().numpy()
    camkii_stack = torch.stack(checkpoints['camkii']).cpu().numpy()

    grid_size = vmem_stack.shape[1]
    idx_lock = min(range(num_checkpoints), key=lambda i: abs(times[i] - t_lock))
    clamp_end_iter = checkpoints.get('clamp_end_iter', None)
    idx_clamp_end = None
    if clamp_end_iter is not None:
        idx_clamp_end = min(range(num_checkpoints), key=lambda i: abs(times[i] - clamp_end_iter))
    x = np.arange(num_checkpoints)

    # Normalize each signal globally to [0, 1] for comparable sparklines
    def norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1e-10)

    vmem_norm = norm(vmem_stack)
    ca_norm = norm(ca_stack)
    camkii_norm = norm(camkii_stack)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.2, grid_size * 1.6),
                             sharex=True, sharey=True)

    for row in range(grid_size):
        for col in range(grid_size):
            ax = axes[row, col]

            # Extract this cell's timeseries across checkpoints
            v_trace = vmem_norm[:, row, col]
            ca_trace = ca_norm[:, row, col]
            ck_trace = camkii_norm[:, row, col]

            ax.plot(x, v_trace, color='#4477AA', linewidth=0.8, alpha=0.9)   # Vmem
            ax.plot(x, ca_trace, color='#EE7733', linewidth=0.8, alpha=0.9)  # Ca
            ax.plot(x, ck_trace, color='#228833', linewidth=0.8, alpha=0.9)  # CaMKII

            # Mark clamp end and t_lock
            if idx_clamp_end is not None:
                ax.axvline(idx_clamp_end, color='orange', linewidth=0.5, alpha=0.5, linestyle=':')
            ax.axvline(idx_lock, color='red', linewidth=0.5, alpha=0.5, linestyle=':')

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, num_checkpoints - 1)

            # Remove all tick labels for compactness
            ax.set_xticks([])
            ax.set_yticks([])

            # Light border to delineate cells
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color('#CCCCCC')

    # Add row/col labels on edges
    for row in range(grid_size):
        axes[row, 0].set_ylabel(f'{row}', fontsize=7, rotation=0, labelpad=10, va='center')
    for col in range(grid_size):
        axes[grid_size - 1, col].set_xlabel(f'{col}', fontsize=7)

    # Legend in top-right corner
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#4477AA', linewidth=1.5, label='Vmem'),
        Line2D([0], [0], color='#EE7733', linewidth=1.5, label='Ca²⁺'),
        Line2D([0], [0], color='#228833', linewidth=1.5, label='CaMKII'),
        Line2D([0], [0], color='red', linewidth=0.8, linestyle=':', label=f't={t_lock} (lock)'),
    ]
    if clamp_end_iter is not None:
        legend_elements.append(
            Line2D([0], [0], color='orange', linewidth=0.8, linestyle=':', label=f't={clamp_end_iter} (clamp end)')
        )
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9,
               bbox_to_anchor=(0.99, 0.99), framealpha=0.9)

    fig.suptitle(f'Spatial Timeseries: each cell shows Vmem/Ca²⁺/CaMKII (t=0 to {t_final})',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    spatial_path = output_path.replace('.png', '_spatial.png')
    fig.savefig(spatial_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved spatial timeseries: {spatial_path}")
    plt.close(fig)


def main():
    """Main test function"""
    print("="*60)
    print("CaMKII Bistability Test")
    print("="*60)
    print(f"\nObjective: Test whether CaMKII can lock in the bioelectric")
    print(f"face pattern at t={args.numBioSteps} and retain it at t={args.numTotalSteps}, even if the")
    print(f"voltage pattern changes or degrades.")
    print(f"\nParameters:")
    print(f"  Bioelectric formation steps: {args.numBioSteps}")
    print(f"  Total steps (with decay): {args.numTotalSteps}")
    print(f"  Record interval: {args.recordInterval}")
    if args.perturb_vmem is not None:
        perturb_start = args.perturb_iter if args.perturb_iter is not None else args.numBioSteps
        perturb_end = perturb_start + args.perturb_duration - 1
        print(f"  Vmem perturbation: std={args.perturb_vmem:.4f}V, iter=[{perturb_start}, {perturb_end}], seed={args.perturb_seed}")
    if args.grn_damping != 1.0:
        print(f"  GRN damping factor: {args.grn_damping:.2f}")
    if args.grn_progressive:
        prog_end = args.grn_prog_end if args.grn_prog_end is not None else args.numBioSteps
        print(f"  Progressive GRN damping: 0.0 -> 1.0 over iter=[{args.grn_prog_start}, {prog_end}]")
    print()

    # Load learned parameters if provided
    learned_params = None
    if args.paramsFile is not None:
        learned_params = load_learned_parameters('./data/'+args.paramsFile)

    # Load model parameters
    print(f"Loading {args.model} model parameters...")
    params = load_model_parameters(args.model, grn_damping=args.grn_damping)
    print(f"Grid size: {params['latticeDims']}")

    # Run simulation with CaMKII tracking
    checkpoints, bio_model, camkii_tracker = run_bioelectric_with_camkii(
        params,
        num_bio_steps=args.numBioSteps,
        num_total_steps=args.numTotalSteps,
        record_interval=args.recordInterval,
        learned_params=learned_params,  # Pass learned parameters
        model_name=args.model,
        perturb_vmem=args.perturb_vmem,
        perturb_iter=args.perturb_iter,
        perturb_duration=args.perturb_duration,
        perturb_seed=args.perturb_seed,
        grn_progressive=args.grn_progressive,
        grn_prog_start=args.grn_prog_start,
        grn_prog_end=args.grn_prog_end
    )

    # Analyze pattern retention
    analysis = analyze_pattern_retention(
        checkpoints,
        t_lock=args.numBioSteps,
        t_test=args.numTotalSteps
    )

    # Visualize results
    visualize_results(checkpoints, analysis, t_lock=args.numBioSteps, output_path='./data/'+args.outputFile)

    # Visualize per-cell timeseries if requested
    if args.visualize_timeseries:
        ts_output = './data/' + args.outputFile.replace('.png', '_timeseries.png')
        visualize_timeseries(checkpoints, t_lock=args.numBioSteps, output_path=ts_output)

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

    # Return results for further analysis if needed
    return checkpoints, analysis, bio_model, camkii_tracker


if __name__ == "__main__":
    checkpoints, analysis, bio_model, camkii_tracker = main()
