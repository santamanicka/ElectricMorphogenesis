#!/usr/bin/env python3
"""
Learn CaMKII-Integrated Facial Patterning Parameters

Optimizes parameters for concurrent CaMKII + GRN dynamics to match
target facial feature patterns.

Key differences from learnRefinedFacialIntegration.py:
1. CaMKII runs concurrently with GRN (not pre-equilibrated Ca)
2. Learns CaMKII bistability parameters alongside GRN parameters
3. Loss computed at end of maintenance phase (pattern persistence)

Learnable parameters:
- CaMKII dynamics: tau_ca, g_ca, V_half_ca, k_ca, k_decay_ca
- CaMKII bistability: ca_threshold, ca_sensitivity, k_on, k_off, K_half, tau_camkii
- OR gate: or_threshold, or_sharpness, gain_ca
- CaMKII gating: camkii_gate_threshold, camkii_gate_sensitivity
- GRN parameters (if not fixed): morphogen strengths, gene activation rates, etc.

Usage:
    python learnCaMKIIFacialIntegration.py
    python learnCaMKIIFacialIntegration.py --numLearnIters 100 --lr 0.02
    python learnCaMKIIFacialIntegration.py --fixedCaMKIIParams data/bestLearnedCaMKIIParams_0.dat
"""

import argparse
import ast
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from embryo import model
from camkiiFacialGRN import CaMKIIFacialGRN, CaMKIIBistableSwitch
from geneBasedFeatureClassifier import GeneBasedFeatureClassifier


# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--gridSize', type=int, default=11)
parser.add_argument('--numRise', type=int, default=1000)
parser.add_argument('--numDecay', type=int, default=1000)
parser.add_argument('--numMaintain', type=int, default=1000)
parser.add_argument('--numLearnIters', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--stigmergicParamsPath', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--fixedCaMKIIParams', type=str, default='',
                    help='Path to pre-learned CaMKII parameters (fixes CaMKII, learns GRN gating only)')
parser.add_argument('--fixedGRNParams', type=str, default='',
                    help='Path to pre-learned GRN parameters (fixes GRN, learns CaMKII only)')
parser.add_argument('--fileNumber', type=int, default=0)
parser.add_argument('--verbose', type=str, default='True')

args = parser.parse_args()

grid_size = args.gridSize
num_rise = args.numRise
num_decay = args.numDecay
num_maintain = args.numMaintain
num_learn_iters = args.numLearnIters
lr = args.lr
stigmergic_params_path = args.stigmergicParamsPath
fixed_camkii_params_path = args.fixedCaMKIIParams
fixed_grn_params_path = args.fixedGRNParams
file_number = args.fileNumber
verbose = ast.literal_eval(args.verbose)


# ============================================================
# Target feature map definition
# ============================================================
def define_target_features(grid_size, mode='bioelectric'):
    """Define target feature map using explicit cell indices"""
    target_features = torch.zeros(grid_size, grid_size, dtype=torch.long)

    if grid_size == 11:
        if mode == 'bioelectric':
            # Fine-grained pattern for bioelectric learning
            left_eye_indices = [(2, 2), (2, 3), (3, 2), (3, 3)]
            right_eye_indices = [(2, 7), (2, 8), (3, 7), (3, 8)]
            eye_indices = left_eye_indices + right_eye_indices

            nose_indices = [(3, 5), (4, 5), (5, 5), (6, 5)]
            mouth_indices = [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
                             (8, 3), (8, 4), (8, 5), (8, 6), (8, 7)]

            for (row, col) in eye_indices:
                target_features[row, col] = 1
            for (row, col) in nose_indices:
                target_features[row, col] = 2
            for (row, col) in mouth_indices:
                target_features[row, col] = 3
    else:
        raise ValueError(f"Target feature map not defined for grid_size={grid_size}")

    return target_features


# ============================================================
# Parameter transformation utilities
# ============================================================
def inverse_sigmoid(x, min_val, max_val):
    """Compute raw parameter that maps to x via sigmoid"""
    normalized = (x - min_val) / (max_val - min_val)
    normalized = torch.clamp(normalized, 1e-6, 1.0 - 1e-6)
    return torch.logit(normalized)


def apply_sigmoid_constraint(raw_param, min_val, max_val):
    """Map unbounded raw parameter to bounded range via sigmoid"""
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)


# ============================================================
# Load pre-learned parameters
# ============================================================
def load_camkii_params(path):
    """Load pre-learned CaMKII parameters"""
    print(f"Loading CaMKII parameters from: {path}")
    data = torch.load(path, weights_only=False)

    learned_params = {}
    param_bounds = data.get('parameter_bounds', {})

    for param_name, raw_value in data['parameters'].items():
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
            learned_params[param_name] = float(raw_value.item()) if hasattr(raw_value, 'item') else float(raw_value)

    print(f"Loaded {len(learned_params)} CaMKII parameters")
    return learned_params


def load_grn_params(path):
    """Load pre-learned GRN parameters"""
    print(f"Loading GRN parameters from: {path}")
    data = torch.load(path, weights_only=False)

    learned_params = {}
    param_bounds = data.get('parameter_bounds', {})

    for param_name, raw_value in data['parameters'].items():
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
            learned_params[param_name] = float(raw_value.item()) if hasattr(raw_value, 'item') else float(raw_value)

    if 'fixed_grn_params' in data:
        for param_name, param_value in data['fixed_grn_params'].items():
            learned_params[param_name] = float(param_value) if isinstance(param_value, torch.Tensor) else param_value

    print(f"Loaded {len(learned_params)} GRN parameters")
    return learned_params


# ============================================================
# Initialize learnable parameters
# ============================================================
def initialize_parameters(fixed_camkii_params=None, fixed_grn_params=None, dtype=torch.float32):
    """
    Initialize learnable parameters using sigmoid parameterization.

    Args:
        fixed_camkii_params: If provided, CaMKII params are fixed (learn only gating)
        fixed_grn_params: If provided, GRN params are fixed (learn only CaMKII)
        dtype: Torch dtype

    Returns:
        params dict with raw (learnable) and fixed parameters
    """
    params = {}

    # ==========================================
    # CaMKII parameters (learn if not fixed)
    # ==========================================
    if fixed_camkii_params is None:
        # tau_ca
        min_val, max_val = 2.0, 5.0
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['tau_ca_raw'] = raw_param.clone().requires_grad_(True)
        params['tau_ca_min'] = min_val
        params['tau_ca_max'] = max_val

        # g_ca
        min_val, max_val = 0.1, 20.0
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['g_ca_raw'] = raw_param.clone().requires_grad_(True)
        params['g_ca_min'] = min_val
        params['g_ca_max'] = max_val

        # V_half_ca
        min_val, max_val = -0.08, -0.01
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['V_half_ca_raw'] = raw_param.clone().requires_grad_(True)
        params['V_half_ca_min'] = min_val
        params['V_half_ca_max'] = max_val

        # k_ca
        min_val, max_val = 0.001, 0.05
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['k_ca_raw'] = raw_param.clone().requires_grad_(True)
        params['k_ca_min'] = min_val
        params['k_ca_max'] = max_val

        # k_decay_ca
        min_val, max_val = 0.0, 5.0
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['k_decay_ca_raw'] = raw_param.clone().requires_grad_(True)
        params['k_decay_ca_min'] = min_val
        params['k_decay_ca_max'] = max_val

        # ca_threshold
        min_val, max_val = 0.01, 10.0
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['ca_threshold_raw'] = raw_param.clone().requires_grad_(True)
        params['ca_threshold_min'] = min_val
        params['ca_threshold_max'] = max_val

        # ca_sensitivity
        min_val, max_val = 0.01, 2.0
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['ca_sensitivity_raw'] = raw_param.clone().requires_grad_(True)
        params['ca_sensitivity_min'] = min_val
        params['ca_sensitivity_max'] = max_val

        # k_on
        min_val, max_val = 0.5, 5.0
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['k_on_raw'] = raw_param.clone().requires_grad_(True)
        params['k_on_min'] = min_val
        params['k_on_max'] = max_val

        # k_off
        min_val, max_val = 0.001, 1.0
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['k_off_raw'] = raw_param.clone().requires_grad_(True)
        params['k_off_min'] = min_val
        params['k_off_max'] = max_val

        # K_half
        min_val, max_val = 0.2, 0.8
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['K_half_raw'] = raw_param.clone().requires_grad_(True)
        params['K_half_min'] = min_val
        params['K_half_max'] = max_val

        # tau_camkii
        min_val, max_val = 10.0, 100.0
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['tau_camkii_raw'] = raw_param.clone().requires_grad_(True)
        params['tau_camkii_min'] = min_val
        params['tau_camkii_max'] = max_val

        # or_threshold
        min_val, max_val = 0.2, 1.5
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['or_threshold_raw'] = raw_param.clone().requires_grad_(True)
        params['or_threshold_min'] = min_val
        params['or_threshold_max'] = max_val

        # or_sharpness
        min_val, max_val = 1.0, 20.0
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['or_sharpness_raw'] = raw_param.clone().requires_grad_(True)
        params['or_sharpness_min'] = min_val
        params['or_sharpness_max'] = max_val

        # gain_ca
        min_val, max_val = 1.5, 3.0
        initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
        raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
        params['gain_ca_raw'] = raw_param.clone().requires_grad_(True)
        params['gain_ca_min'] = min_val
        params['gain_ca_max'] = max_val

    else:
        # Store fixed CaMKII parameters
        params['fixed_camkii'] = fixed_camkii_params

    # ==========================================
    # AND gate parameters (always learnable)
    # ==========================================
    min_val, max_val = 1.0, 1.5
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['and_threshold_raw'] = raw_param.clone().requires_grad_(True)
    params['and_threshold_min'] = min_val
    params['and_threshold_max'] = max_val

    min_val, max_val = 10.0, 25.0
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['and_sharpness_raw'] = raw_param.clone().requires_grad_(True)
    params['and_sharpness_min'] = min_val
    params['and_sharpness_max'] = max_val

    # Store fixed GRN parameters if provided
    if fixed_grn_params is not None:
        params['fixed_grn'] = fixed_grn_params

    return params


# ============================================================
# Run simulation with current parameters
# ============================================================
def run_simulation(params, vmem_trajectory, target_features, device, dtype):
    """
    Run one complete simulation cycle with current parameters.

    Returns:
        predicted_features: Final feature map
        loss: Scalar loss value
    """
    # Extract CaMKII parameters
    if 'fixed_camkii' in params:
        camkii_params = params['fixed_camkii']
    else:
        camkii_params = {
            'tau_ca': apply_sigmoid_constraint(params['tau_ca_raw'], params['tau_ca_min'], params['tau_ca_max']).item(),
            'g_ca': apply_sigmoid_constraint(params['g_ca_raw'], params['g_ca_min'], params['g_ca_max']).item(),
            'V_half_ca': apply_sigmoid_constraint(params['V_half_ca_raw'], params['V_half_ca_min'], params['V_half_ca_max']).item(),
            'k_ca': apply_sigmoid_constraint(params['k_ca_raw'], params['k_ca_min'], params['k_ca_max']).item(),
            'k_decay_ca': apply_sigmoid_constraint(params['k_decay_ca_raw'], params['k_decay_ca_min'], params['k_decay_ca_max']).item(),
            'ca_threshold': apply_sigmoid_constraint(params['ca_threshold_raw'], params['ca_threshold_min'], params['ca_threshold_max']).item(),
            'ca_sensitivity': apply_sigmoid_constraint(params['ca_sensitivity_raw'], params['ca_sensitivity_min'], params['ca_sensitivity_max']).item(),
            'k_on': apply_sigmoid_constraint(params['k_on_raw'], params['k_on_min'], params['k_on_max']).item(),
            'k_off': apply_sigmoid_constraint(params['k_off_raw'], params['k_off_min'], params['k_off_max']).item(),
            'K_half': apply_sigmoid_constraint(params['K_half_raw'], params['K_half_min'], params['K_half_max']).item(),
            'tau_camkii': apply_sigmoid_constraint(params['tau_camkii_raw'], params['tau_camkii_min'], params['tau_camkii_max']).item(),
            'or_threshold': apply_sigmoid_constraint(params['or_threshold_raw'], params['or_threshold_min'], params['or_threshold_max']).item(),
            'or_sharpness': apply_sigmoid_constraint(params['or_sharpness_raw'], params['or_sharpness_min'], params['or_sharpness_max']).item(),
            'gain_ca': apply_sigmoid_constraint(params['gain_ca_raw'], params['gain_ca_min'], params['gain_ca_max']).item(),
        }

    # Extract AND gate parameters (always learnable)
    and_threshold = apply_sigmoid_constraint(
        params['and_threshold_raw'],
        params['and_threshold_min'],
        params['and_threshold_max']
    )
    and_sharpness = apply_sigmoid_constraint(
        params['and_sharpness_raw'],
        params['and_sharpness_min'],
        params['and_sharpness_max']
    )

    # Create GRN
    grn = CaMKIIFacialGRN(
        grid_size=grid_size,
        device=device,
        dtype=dtype
    )

    # Apply CaMKII parameters
    grn.camkii_switch.load_learned_parameters(camkii_params)

    # Apply AND gate overrides
    grn.and_threshold_override = and_threshold
    grn.and_sharpness_override = and_sharpness

    # Apply fixed GRN parameters if provided
    if 'fixed_grn' in params:
        fixed_grn = params['fixed_grn']
        if 'fgf8_strength' in fixed_grn:
            grn.morphogen_params['fgf8_strength'] = torch.tensor(fixed_grn['fgf8_strength'], device=device, dtype=dtype)
        if 'k_activation' in fixed_grn:
            grn.gene_params['k_activation'] = torch.tensor(fixed_grn['k_activation'], device=device, dtype=dtype)
        if 'k_degradation' in fixed_grn:
            grn.gene_params['k_degradation'] = torch.tensor(fixed_grn['k_degradation'], device=device, dtype=dtype)
        if 'K_self' in fixed_grn:
            grn.gene_params['K_self'] = torch.tensor(fixed_grn['K_self'], device=device, dtype=dtype)
        if 'n_self' in fixed_grn:
            grn.gene_params['n_self'] = torch.tensor(fixed_grn['n_self'], device=device, dtype=dtype)

    # Reset and pre-equilibrate morphogens
    grn.reset()
    for _ in range(1000):
        grn.update_morphogens()

    # Run concurrent simulation
    dt = 0.01
    for vmem_grid in vmem_trajectory:
        grn.update_concurrent(vmem_grid, dt=dt)

    # Classify features
    classifier = GeneBasedFeatureClassifier(
        grid_size=grid_size,
        device=device,
        dtype=dtype
    )

    # Get feature scores
    feature_scores = classifier.compute_feature_scores(grn.grid)
    scores_tensor = torch.stack([
        feature_scores['bone'],
        feature_scores['eye'],
        feature_scores['nose'],
        feature_scores['mouth']
    ], dim=0)

    # Get hard classification
    classification = classifier.classify(grn.grid, mode='hard')
    predicted_features = classification['features']

    # Compute loss (cross-entropy with class balancing)
    unique, counts = torch.unique(target_features, return_counts=True)
    total_cells = grid_size * grid_size
    class_weights = torch.ones(4, device=device, dtype=dtype)
    for label, count in zip(unique, counts):
        class_weights[label] = torch.sqrt(torch.tensor(total_cells / (count.float() * 4.0), device=device, dtype=dtype))
    class_weights = class_weights / class_weights.mean()

    loss = torch.nn.functional.cross_entropy(
        scores_tensor.unsqueeze(0),
        target_features.unsqueeze(0),
        weight=class_weights,
        reduction='mean'
    )

    return predicted_features, loss, grn


# ============================================================
# Main learning loop
# ============================================================
def main():
    # Device setup
    device = 'cpu'
    dtype = torch.float32

    print("=" * 70)
    print("LEARNING CAMKII-INTEGRATED FACIAL PATTERNING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Phases: Rise={num_rise}, Decay={num_decay}, Maintain={num_maintain}")
    print(f"Learning iterations: {num_learn_iters}")
    print(f"Learning rate: {lr}")
    print("=" * 70 + "\n")

    # Load fixed parameters if provided
    fixed_camkii = None
    if fixed_camkii_params_path:
        fixed_camkii = load_camkii_params(fixed_camkii_params_path)

    fixed_grn = None
    if fixed_grn_params_path:
        fixed_grn = load_grn_params(fixed_grn_params_path)

    # Define target features
    print("Defining target feature map...")
    target_features = define_target_features(grid_size, mode='bioelectric').to(device)
    feature_names = ['bone', 'eye', 'nose', 'mouth']
    unique, counts = torch.unique(target_features, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {feature_names[label]}: {count.item()} cells")

    # Load and run bioelectric simulation
    print("\nRunning Stigmergic bioelectric simulation...")
    stig_params = torch.load(stigmergic_params_path, weights_only=False)
    if "ATPParameters" not in stig_params:
        stig_params["ATPParameters"] = None

    num_samples = stig_params["simParameters"]["numSamples"]
    initial_values = copy.deepcopy(stig_params["simParameters"]["initialValues"])
    external_inputs = copy.deepcopy(stig_params["simParameters"]["externalInputs"])
    clamp_params = copy.deepcopy(stig_params["clampParameters"])

    bio_model = model(stig_params, numBasicSamples=num_samples)
    bio_model.setExperimentalConditions((initial_values, num_samples))

    initial_vmem = bio_model.electricNetwork.Vmem[0, :, 0].reshape(grid_size, grid_size).clone()

    bio_model.simulate(
        externalInputs=external_inputs,
        clampParameters=clamp_params,
        perturbation=None,
        fieldModulation=False,
        numSimIters=stig_params['simParameters']['numSimIters']
    )

    final_vmem = bio_model.electricNetwork.Vmem[0, :, 0].reshape(grid_size, grid_size).clone()
    print(f"✓ Stigmergic simulation complete")
    print(f"  Initial Vmem: mean={initial_vmem.mean().item():.4f}V")
    print(f"  Final Vmem: mean={final_vmem.mean().item():.4f}V, std={final_vmem.std().item():.4f}V\n")

    # Move to device
    initial_vmem_grid = initial_vmem.to(device=device, dtype=dtype)
    final_vmem_grid = final_vmem.to(device=device, dtype=dtype)

    # Create Vmem trajectory
    print("Creating Vmem trajectory...")
    vmem_trajectory = []
    total_steps = num_rise + num_decay + num_maintain

    for t in range(total_steps):
        if t < num_rise:
            alpha = t / num_rise
            vmem = (1 - alpha) * initial_vmem_grid + alpha * final_vmem_grid
        elif t < num_rise + num_decay:
            decay_progress = (t - num_rise) / num_decay
            vmem = (1 - decay_progress) * final_vmem_grid + decay_progress * initial_vmem_grid
        else:
            vmem = initial_vmem_grid.clone()
        vmem_trajectory.append(vmem)

    print(f"  Total steps: {len(vmem_trajectory)}")

    # Initialize parameters
    params = initialize_parameters(
        fixed_camkii_params=fixed_camkii,
        fixed_grn_params=fixed_grn,
        dtype=dtype
    )

    # Collect learnable parameters
    learned_params_list = []
    param_names_to_learn = []

    # CaMKII parameters (if not fixed)
    if 'fixed_camkii' not in params:
        camkii_param_names = ['tau_ca', 'g_ca', 'V_half_ca', 'k_ca', 'k_decay_ca',
                              'ca_threshold', 'ca_sensitivity',
                              'k_on', 'k_off', 'K_half', 'tau_camkii',
                              'or_threshold', 'or_sharpness', 'gain_ca']
        for pname in camkii_param_names:
            if f'{pname}_raw' in params:
                learned_params_list.append(params[f'{pname}_raw'])
                param_names_to_learn.append(pname)

    # AND gate parameters (always learnable)
    gating_param_names = ['and_threshold', 'and_sharpness']
    for pname in gating_param_names:
        if f'{pname}_raw' in params:
            learned_params_list.append(params[f'{pname}_raw'])
            param_names_to_learn.append(pname)

    print(f"\nLearning {len(param_names_to_learn)} parameters: {param_names_to_learn}")

    # Setup optimizer
    optimizer = torch.optim.Rprop(learned_params_list, lr=lr)

    # Learning loop
    best_loss = 999999.0
    best_params = {}
    best_history = []

    print("\n" + "=" * 70)
    print("STARTING LEARNING LOOP")
    print("=" * 70 + "\n")

    for iter_idx in range(num_learn_iters):
        predicted_features, loss, grn = run_simulation(
            params, vmem_trajectory, target_features, device, dtype
        )

        current_loss = loss.item()

        # Track best
        if current_loss < best_loss:
            best_loss = current_loss
            best_history.append((iter_idx, best_loss))

            # Save best parameters
            best_params = {}
            best_param_bounds = {}
            for pname in param_names_to_learn:
                raw_name = f'{pname}_raw'
                if raw_name in params:
                    best_params[pname] = params[raw_name].detach().clone()
                    if f'{pname}_min' in params:
                        best_param_bounds[f'{pname}_min'] = params[f'{pname}_min']
                    if f'{pname}_max' in params:
                        best_param_bounds[f'{pname}_max'] = params[f'{pname}_max']

            # Save to file
            save_data = {
                'parameters': best_params,
                'parameter_bounds': best_param_bounds,
                'loss': best_loss,
                'loss_history': best_history,
                'target_features': target_features,
                'predicted_features': predicted_features.detach(),
                'learned_parameter_names': param_names_to_learn,
                'grid_size': grid_size,
                'fixed_camkii_params': fixed_camkii,
                'fixed_grn_params': fixed_grn,
            }
            torch.save(save_data, f'./data/bestLearnedCaMKIIFacialParams_{file_number}.dat')

        # Backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if verbose and ((iter_idx + 1) % 5 == 0 or iter_idx == 0):
            # Per-class accuracy
            feature_accuracy = {}
            for label in range(4):
                mask = target_features == label
                if mask.sum() > 0:
                    correct = ((predicted_features == label) & mask).sum().item()
                    total = mask.sum().item()
                    feature_accuracy[feature_names[label]] = correct / total

            acc_str = ", ".join([f"{k}:{v:.2f}" for k, v in feature_accuracy.items()])
            print(f"Iter {iter_idx+1:3d}/{num_learn_iters}: loss={current_loss:.4f}, best={best_loss:.4f} | acc=[{acc_str}]")

            # Print key parameters
            if (iter_idx + 1) % 10 == 0:
                print("  Key parameters:")
                for pname in ['K_half', 'gain_ca', 'camkii_gate_threshold', 'and_threshold']:
                    raw_name = f'{pname}_raw'
                    if raw_name in params:
                        val = apply_sigmoid_constraint(
                            params[raw_name],
                            params[f'{pname}_min'],
                            params[f'{pname}_max']
                        )
                        print(f"    {pname}: {val.item():.4f}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION WITH BEST PARAMETERS")
    print("=" * 70)

    # Restore best parameters
    for pname in param_names_to_learn:
        raw_name = f'{pname}_raw'
        if raw_name in params and pname in best_params:
            params[raw_name].data = best_params[pname]

    predicted_features, final_loss, final_grn = run_simulation(
        params, vmem_trajectory, target_features, device, dtype
    )

    print(f"\nBest loss: {best_loss:.4f}")
    print("\nBest parameters:")
    for pname in param_names_to_learn:
        if pname in best_params:
            constrained_val = apply_sigmoid_constraint(
                best_params[pname],
                best_param_bounds[f'{pname}_min'],
                best_param_bounds[f'{pname}_max']
            )
            print(f"  {pname}: {constrained_val.item():.4f}")

    # Feature comparison
    print("\nFeature distribution:")
    print("Target:")
    unique, counts = torch.unique(target_features, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {feature_names[label]}: {count.item()} cells")

    print("\nPredicted:")
    unique, counts = torch.unique(predicted_features, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {feature_names[label]}: {count.item()} cells")

    # Accuracy
    correct = (predicted_features == target_features).float().sum()
    accuracy = correct / (grid_size * grid_size)
    print(f"\nAccuracy: {accuracy.item():.2%}")

    # Visualize
    visualize_results(target_features, predicted_features, final_grn, file_number)

    print("\n" + "=" * 70)
    print("✅ LEARNING COMPLETE!")
    print("=" * 70)
    print(f"\nSaved best parameters to: ./data/bestLearnedCaMKIIFacialParams_{file_number}.dat")


def visualize_results(target_features, predicted_features, grn, file_number):
    """Create visualization comparing target and predicted features"""
    feature_cmap = ListedColormap(["#f9f9f9", "#9b59b6", "#e67e22", "#2ecc71"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Target, CaMKII, Predicted
    ax = axes[0, 0]
    im = ax.imshow(target_features.cpu().numpy(), cmap=feature_cmap, vmin=0, vmax=3)
    ax.set_title('Target Features', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 1]
    im = ax.imshow(grn.camkii_switch.CaMKII_active.detach().cpu().numpy(), cmap='plasma', vmin=0, vmax=1)
    ax.set_title('Final CaMKII (Bistable)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 2]
    im = ax.imshow(predicted_features.detach().cpu().numpy(), cmap=feature_cmap, vmin=0, vmax=3)
    ax.set_title('Predicted Features', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046)
    cbar.set_ticklabels(['Bone', 'Eye', 'Nose', 'Mouth'])
    ax.set_xticks([])
    ax.set_yticks([])

    # Row 2: Key genes
    genes = grn.get_gene_grids()
    gene_display = ['pax6', 'alx', 'dlx']
    gene_titles = ['Pax6 (Eye)', 'Alx (Nose)', 'Dlx (Mouth)']

    for idx, (gene, title) in enumerate(zip(gene_display, gene_titles)):
        ax = axes[1, idx]
        im = ax.imshow(genes[gene].detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('CaMKII-Integrated Facial Patterning: Learned Parameters',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'learned_camkii_facial_{file_number}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved visualization to: learned_camkii_facial_{file_number}.png")


if __name__ == "__main__":
    main()
