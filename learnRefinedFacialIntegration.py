#!/usr/bin/env python3
"""
Learn Bioelectric-Morphogen Facial Patterning Parameters

Optimizes parameters to match gene feature map to IdealFace.png using Rprop optimizer.
Similar style to learnCellularFieldNetwork.py.

Learnable parameters:
- Bioelectric: Ca²⁺ gating threshold percentile, sensitivity, AND gate threshold
- Morphogen: Source strengths, decay lengths, degradation rates
- Gene: Activation/degradation rates, Hill function parameters
"""

import argparse
import ast
import copy
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from embryo import model
from refinedFacialGRN import RefinedFacialGRN
from geneBasedFeatureClassifier import GeneBasedFeatureClassifier
from bioelectricTransduction import BioelectricTransduction


# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--gridSize', type=int, default=11)
parser.add_argument('--numSimIters', type=int, default=1000)
parser.add_argument('--numGRNIters', type=int, default=5000)
parser.add_argument('--numLearnIters', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--lossMethod', type=str, default='featureMap')
parser.add_argument('--learnedParameters', type=str, default="['ca_threshold_percentile','ca_sensitivity','and_threshold']")
parser.add_argument('--idealFacePath', type=str, default='IdealFace.png')
parser.add_argument('--stigmergicParamsPath', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--fileNumber', type=int, default=0)
parser.add_argument('--verbose', type=str, default='True')

args = parser.parse_args()

grid_size = args.gridSize
num_sim_iters = args.numSimIters
num_grn_iters = args.numGRNIters
num_learn_iters = args.numLearnIters
lr = args.lr
loss_method = args.lossMethod
learned_parameter_names = ast.literal_eval(args.learnedParameters)
ideal_face_path = args.idealFacePath
stigmergic_params_path = args.stigmergicParamsPath
file_number = args.fileNumber
verbose = ast.literal_eval(args.verbose)


# ============================================================
# Define target feature map using explicit indices
# ============================================================
def define_target_features(grid_size):
    """
    Define target feature map using explicit cell indices (similar to defineTargetVmem).

    For an 11x11 grid:
    - Bone (0): Outer border/frame (default/background)
    - Eye (1): Two square patches in anterior-lateral positions
    - Nose (2): Vertical stripe at midline
    - Mouth (3): Horizontal stripe in posterior

    Returns:
        target_features: (grid_size, grid_size) tensor with feature labels
            0=bone, 1=eye, 2=nose, 3=mouth
    """
    # Initialize all cells as bone (default/background)
    target_features = torch.zeros(grid_size, grid_size, dtype=torch.long)

    if grid_size == 11:
        # Define feature regions for 11x11 grid matching IdealFace.png

        # Eye indices: Two square patches (2x2 or 3x2) in anterior-lateral positions
        # Left eye: rows 2-3, cols 2-3
        left_eye_indices = [
            (2, 2), (2, 3),
            (3, 2), (3, 3),
        ]

        # Right eye: rows 2-3, cols 7-8
        right_eye_indices = [
            (2, 7), (2, 8),
            (3, 7), (3, 8),
        ]

        eye_indices = left_eye_indices + right_eye_indices

        # Nose indices: Vertical stripe at midline (col 5), rows 3-6
        nose_indices = [
            (3, 5),
            (4, 5),
            (5, 5),
            (6, 5),
        ]

        # Mouth indices: Horizontal stripe in posterior (row 7-8), cols 3-7
        mouth_indices = [
            (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
            (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
        ]

        # Set feature labels
        for (row, col) in eye_indices:
            target_features[row, col] = 1  # eye

        for (row, col) in nose_indices:
            target_features[row, col] = 2  # nose

        for (row, col) in mouth_indices:
            target_features[row, col] = 3  # mouth

        # All other cells remain 0 (bone/background)

    else:
        # For other grid sizes, could define scaled indices or raise error
        raise ValueError(f"Target feature map not defined for grid_size={grid_size}")

    return target_features


# ============================================================
# Initialize parameters
# ============================================================
def initialize_parameters(learned_params, dtype=torch.float32):
    """Initialize learnable parameters with reasonable ranges"""
    params = {}

    # Bioelectric gating parameters (matching run_refined_facial_integration.py / refinedFacialGRN.py)
    # Add random noise to initial values for learned parameters
    if 'ca_threshold_percentile' in learned_params:
        # Range: 0.20 to 0.60 (20th to 60th percentile)
        # Initial: 0.45 ± noise (from refinedFacialGRN.py line 313)
        min_val, max_val = 0.20, 0.60
        noise = (torch.rand(1, dtype=dtype) - 0.5) * 0.1  # ±5% of range
        params['ca_threshold_percentile'] = torch.tensor(0.45, dtype=dtype) + noise
        params['ca_threshold_percentile'] = torch.clamp(params['ca_threshold_percentile'], min_val, max_val)
        params['ca_threshold_percentile'].requires_grad = True
        params['ca_threshold_percentile_min'] = min_val
        params['ca_threshold_percentile_max'] = max_val

    if 'ca_sensitivity' in learned_params:
        # Range: 0.01 to 0.10 (sharpness of sigmoid)
        # Initial: 0.04 ± noise (from refinedFacialGRN.py line 314)
        min_val, max_val = 0.01, 0.10
        noise = (torch.rand(1, dtype=dtype) - 0.5) * 0.02  # ±10% of range
        params['ca_sensitivity'] = torch.tensor(0.04, dtype=dtype) + noise
        params['ca_sensitivity'] = torch.clamp(params['ca_sensitivity'], min_val, max_val)
        params['ca_sensitivity'].requires_grad = True
        params['ca_sensitivity_min'] = min_val
        params['ca_sensitivity_max'] = max_val

    if 'and_threshold' in learned_params:
        # Range: 1.0 to 1.5 (AND gate threshold)
        # Initial: 1.25 ± noise (from refinedFacialGRN.py line 152)
        min_val, max_val = 1.0, 1.5
        noise = (torch.rand(1, dtype=dtype) - 0.5) * 0.1  # ±10% of range
        params['and_threshold'] = torch.tensor(1.25, dtype=dtype) + noise
        params['and_threshold'] = torch.clamp(params['and_threshold'], min_val, max_val)
        params['and_threshold'].requires_grad = True
        params['and_threshold_min'] = min_val
        params['and_threshold_max'] = max_val

    if 'and_sharpness' in learned_params:
        # Range: 10.0 to 25.0 (AND gate sharpness)
        # Initial: 20.0 ± noise (from refinedFacialGRN.py line 152)
        min_val, max_val = 10.0, 25.0
        noise = (torch.rand(1, dtype=dtype) - 0.5) * 3.0  # ±10% of range
        params['and_sharpness'] = torch.tensor(20.0, dtype=dtype) + noise
        params['and_sharpness'] = torch.clamp(params['and_sharpness'], min_val, max_val)
        params['and_sharpness'].requires_grad = True
        params['and_sharpness_min'] = min_val
        params['and_sharpness_max'] = max_val

    # Morphogen parameters
    if 'fgf8_strength' in learned_params:
        # Range: 0.1 to 0.5
        params['fgf8_strength'] = torch.tensor(0.2, dtype=dtype, requires_grad=True)
        params['fgf8_strength_min'] = 0.1
        params['fgf8_strength_max'] = 0.5

    if 'fgf8_degradation_factor' in learned_params:
        # Range: 5.0 to 15.0 (multiplier on base degradation)
        params['fgf8_degradation_factor'] = torch.tensor(10.0, dtype=dtype, requires_grad=True)
        params['fgf8_degradation_factor_min'] = 5.0
        params['fgf8_degradation_factor_max'] = 15.0

    # Gene activation parameters
    if 'k_activation' in learned_params:
        # Range: 0.05 to 0.20
        params['k_activation'] = torch.tensor(0.10, dtype=dtype, requires_grad=True)
        params['k_activation_min'] = 0.05
        params['k_activation_max'] = 0.20

    if 'k_degradation' in learned_params:
        # Range: 0.005 to 0.02
        params['k_degradation'] = torch.tensor(0.01, dtype=dtype, requires_grad=True)
        params['k_degradation_min'] = 0.005
        params['k_degradation_max'] = 0.02

    # Feature classification parameters
    if 'min_mouth_expr' in learned_params:
        # Range: 0.3 to 0.9
        params['min_mouth_expr'] = torch.tensor(0.85, dtype=dtype, requires_grad=True)
        params['min_mouth_expr_min'] = 0.3
        params['min_mouth_expr_max'] = 0.9

    return params


def clip_parameters(params, learned_params):
    """Clip parameters to valid ranges"""
    for param_name in learned_params:
        if param_name in params:
            param = params[param_name]
            min_val = params[f'{param_name}_min']
            max_val = params[f'{param_name}_max']
            param.data = torch.clamp(param.data, min_val, max_val)


# ============================================================
# Run simulation with current parameters
# ============================================================
def run_simulation(params, stig_model, transduction, target_features, device, dtype):
    """
    Run one complete simulation cycle with current parameters.

    Returns:
        predicted_features: (grid_size, grid_size) feature map
        loss: scalar loss value
    """
    # Extract current parameter values
    ca_threshold_pct = params.get('ca_threshold_percentile', 0.35)
    ca_sensitivity = params.get('ca_sensitivity', 0.03)
    and_threshold = params.get('and_threshold', 1.3)
    and_sharpness = params.get('and_sharpness', 18.0)
    fgf8_strength = params.get('fgf8_strength', 0.2)
    fgf8_deg_factor = params.get('fgf8_degradation_factor', 10.0)
    k_activation = params.get('k_activation', 0.10)
    k_degradation = params.get('k_degradation', 0.01)
    min_mouth_expr = params.get('min_mouth_expr', 0.85)

    # Create GRN with current parameters
    grn = RefinedFacialGRN(
        grid_size=grid_size,
        device=device,
        dtype=dtype
    )

    # Update GRN parameters if learnable
    if 'fgf8_strength' in params:
        grn.morphogen_params['fgf8_strength'] = fgf8_strength
    if 'k_activation' in params:
        grn.gene_params['k_activation'] = k_activation
    if 'k_degradation' in params:
        grn.gene_params['k_degradation'] = k_degradation

    # Override AND gate parameters in gene dynamics
    # Ensure these are scalars or on the correct device
    if isinstance(and_threshold, torch.Tensor):
        grn.and_threshold_override = and_threshold.to(device).item()  # Convert to scalar
    else:
        grn.and_threshold_override = and_threshold
    if isinstance(and_sharpness, torch.Tensor):
        grn.and_sharpness_override = and_sharpness.to(device).item()  # Convert to scalar
    else:
        grn.and_sharpness_override = and_sharpness

    # Get bioelectric signals from transduction module
    vmem_grid = stig_model.electricNetwork.Vmem.view(grid_size, grid_size).detach()
    bio_signals = transduction.get_gene_modulation_signals()

    # Override Ca²⁺ gating with current parameters
    Ca = bio_signals['Ca'].to(device)  # Ensure Ca is on the correct device
    # Ensure all parameter tensors are on the same device
    if isinstance(ca_threshold_pct, torch.Tensor):
        ca_threshold_pct_device = ca_threshold_pct.to(device)
    else:
        ca_threshold_pct_device = ca_threshold_pct
    if isinstance(ca_sensitivity, torch.Tensor):
        ca_sensitivity_device = ca_sensitivity.to(device)
    else:
        ca_sensitivity_device = ca_sensitivity
    Ca_threshold = torch.quantile(Ca, ca_threshold_pct_device)
    bio_gate = torch.sigmoid((Ca_threshold - Ca) / ca_sensitivity_device)
    bio_signals_override = {
        'Ca': bio_gate,  # Use computed bio_gate directly
        'metabolic': bio_signals['metabolic'].to(device)
    }

    # Pre-equilibrate morphogens
    for _ in range(1000):
        grn.update_morphogens(bioelectric_signals=None)

    # Run GRN dynamics with bioelectric gating
    for _ in range(num_grn_iters):
        grn.update_morphogens(bioelectric_signals=None)
        grn.update_genes(bioelectric_signals=bio_signals_override)

    # Classify features
    classifier = GeneBasedFeatureClassifier(
        grid_size=grid_size,
        device=device,
        dtype=dtype
    )

    # Override mouth threshold if learnable
    if 'min_mouth_expr' in params:
        classifier.min_mouth_expr = min_mouth_expr

    # Get feature scores (continuous, differentiable)
    feature_scores = classifier.compute_feature_scores(grn.grid)

    # Stack scores into (4, grid_size, grid_size) tensor
    scores_tensor = torch.stack([
        feature_scores['bone'],
        feature_scores['eye'],
        feature_scores['nose'],
        feature_scores['mouth']
    ], dim=0)  # Shape: (4, grid_size, grid_size)

    # Get hard classification for evaluation
    classification = classifier.classify(grn.grid, mode='hard')
    predicted_features = classification['features']

    # Compute loss using continuous scores
    if loss_method == 'featureMap':
        # Cross-entropy loss using continuous scores
        # scores_tensor: (4, grid_size, grid_size) -> (1, 4, grid_size, grid_size)
        # target_features: (grid_size, grid_size) -> (1, grid_size, grid_size)
        loss = torch.nn.functional.cross_entropy(
            scores_tensor.unsqueeze(0),
            target_features.unsqueeze(0),
            reduction='mean'
        )
    elif loss_method == 'featureMapMSE':
        # MSE on soft scores vs one-hot target
        target_onehot = torch.nn.functional.one_hot(target_features, num_classes=4).float()
        target_onehot = target_onehot.permute(2, 0, 1)  # (grid, grid, 4) -> (4, grid, grid)

        # Softmax on scores to get probabilities
        probs = torch.softmax(scores_tensor, dim=0)
        loss = ((probs - target_onehot) ** 2).mean()
    elif loss_method == 'accuracy':
        # Accuracy-based loss (1 - accuracy) - not differentiable, use for logging only
        correct = (predicted_features == target_features).float().sum()
        total = grid_size * grid_size
        accuracy = correct / total
        loss = 1.0 - accuracy

    return predicted_features, loss


# ============================================================
# Main learning loop
# ============================================================
def main():
    # Detect and set device (Mac GPU if available)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'Mac GPU (MPS)'
        dtype = torch.float32  # MPS doesn't support float64
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = 'CUDA GPU'
        dtype = torch.float32  # Use float32 for GPU
    else:
        device = torch.device('cpu')
        device_name = 'CPU'
        dtype = torch.float64  # CPU can use float64

    print("=" * 70)
    print("LEARNING REFINED FACIAL INTEGRATION PARAMETERS")
    print("=" * 70)
    print(f"Device: {device_name}")
    print(f"Data type: {dtype}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Bioelectric simulation: {num_sim_iters} iterations")
    print(f"GRN simulation: {num_grn_iters} iterations per learning step")
    print(f"Learning iterations: {num_learn_iters}")
    print(f"Learning rate: {lr}")
    print(f"Loss method: {loss_method}")
    print(f"Learned parameters: {learned_parameter_names}")
    print("=" * 70 + "\n")

    # Define target face using explicit indices
    print("Defining target feature map from explicit indices...")
    target_features = define_target_features(grid_size).to(device)
    print(f"Target feature counts:")
    unique, counts = torch.unique(target_features, return_counts=True)
    feature_names = ['bone', 'eye', 'nose', 'mouth']
    for label, count in zip(unique, counts):
        print(f"  {feature_names[label]}: {count.item()} cells")
    print()

    # Load and run Stigmergic bioelectric model (once, fixed)
    print("Running Stigmergic bioelectric simulation...")
    from torch.serialization import add_safe_globals
    add_safe_globals([np.core.multiarray._reconstruct])
    stig_params = torch.load(stigmergic_params_path, weights_only=False)
    if "ATPParameters" not in stig_params:
        stig_params["ATPParameters"] = None

    num_samples = stig_params["simParameters"]["numSamples"]
    initial_values = copy.deepcopy(stig_params["simParameters"]["initialValues"])
    external_inputs = copy.deepcopy(stig_params["simParameters"]["externalInputs"])
    clamp_params = copy.deepcopy(stig_params["clampParameters"])

    stig_model = model(stig_params, numBasicSamples=num_samples)
    stig_model.setExperimentalConditions((initial_values, num_samples))
    stig_model.simulate(
        externalInputs=external_inputs,
        clampParameters=clamp_params,
        perturbation=None,
        fieldModulation=True,
        numSimIters=num_sim_iters,
    )
    print("✓ Stigmergic simulation complete\n")

    # Extract bioelectric signals (fixed)
    transduction = BioelectricTransduction(grid_size=grid_size, device=device, dtype=dtype)
    rows, cols = stig_params["latticeDims"]
    # Convert dtype first, then move to device (MPS doesn't support float64)
    vmem_grid = stig_model.electricNetwork.Vmem.view(rows, cols).detach().to(dtype=dtype).to(device=device)

    # Run transduction to get Ca²⁺
    for _ in range(100):
        transduction.update(vmem_grid, I_gj_grid=None, dt=0.01)
    print("✓ Bioelectric signals extracted\n")

    # Initialize learnable parameters
    params = initialize_parameters(learned_parameter_names, dtype=dtype)
    learned_params_list = [params[name] for name in learned_parameter_names if name in params]

    # Setup optimizer
    optimizer = torch.optim.Rprop(learned_params_list, lr=lr)

    # Learning loop
    best_loss = 999999.0
    best_params = {}
    best_loss_history = []

    print("=" * 70)
    print("STARTING LEARNING LOOP")
    print("=" * 70 + "\n")

    for iter_idx in range(num_learn_iters):
        # Clip parameters to valid ranges
        clip_parameters(params, learned_parameter_names)

        # Run simulation with current parameters
        predicted_features, loss = run_simulation(params, stig_model, transduction, target_features, device, dtype)

        current_loss = loss.item()

        # Track best parameters
        if current_loss < best_loss:
            best_loss = current_loss
            best_loss_history.append((iter_idx, best_loss))

            # Save best parameters
            for param_name in learned_parameter_names:
                if param_name in params:
                    best_params[param_name] = params[param_name].detach().clone()

            # Save best model
            save_data = {
                'parameters': best_params,
                'loss': best_loss,
                'loss_history': best_loss_history,
                'target_features': target_features,
                'predicted_features': predicted_features.detach(),
                'learned_parameter_names': learned_parameter_names,
                'grid_size': grid_size,
            }
            torch.save(save_data, f'./data/bestLearnedFacialParams_{file_number}.dat')

        # Backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if verbose and ((iter_idx + 1) % 10 == 0 or iter_idx == 0):
            print(f"Iter {iter_idx+1:3d}/{num_learn_iters}: loss={current_loss:.6f}, best={best_loss:.6f}")

            # Print current parameter values
            if (iter_idx + 1) % 20 == 0:
                print("  Current parameters:")
                for param_name in learned_parameter_names:
                    if param_name in params:
                        print(f"    {param_name}: {params[param_name].item():.4f}")

    # Final evaluation with best parameters
    print("\n" + "=" * 70)
    print("FINAL EVALUATION WITH BEST PARAMETERS")
    print("=" * 70)

    # Restore best parameters
    for param_name in learned_parameter_names:
        if param_name in params and param_name in best_params:
            params[param_name].data = best_params[param_name]

    predicted_features, final_loss = run_simulation(params, stig_model, transduction, target_features, device, dtype)

    print(f"\nBest loss: {best_loss:.6f}")
    print("\nBest parameters:")
    for param_name in learned_parameter_names:
        if param_name in best_params:
            print(f"  {param_name}: {best_params[param_name].item():.4f}")

    # Feature comparison
    print("\nFeature distribution comparison:")
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

    # Visualize results
    visualize_results(target_features, predicted_features, file_number)

    print("\n" + "=" * 70)
    print("✅ LEARNING COMPLETE!")
    print("=" * 70)
    print(f"\nSaved best parameters to: ./data/bestLearnedFacialParams_{file_number}.dat")
    print(f"Saved visualization to: learned_facial_comparison_{file_number}.png")


def visualize_results(target_features, predicted_features, file_number):
    """Create visualization comparing target and predicted features"""
    feature_cmap = ListedColormap(["#f9f9f9", "#9b59b6", "#e67e22", "#2ecc71"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    im = ax.imshow(target_features.cpu().numpy(), cmap=feature_cmap, vmin=0, vmax=3)
    ax.set_title('Target (Defined by Indices)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    im = ax.imshow(predicted_features.detach().cpu().numpy(), cmap=feature_cmap, vmin=0, vmax=3)
    ax.set_title('Predicted (Learned)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.set_label('0=bone, 1=eye, 2=nose, 3=mouth', rotation=270, labelpad=20)

    fig.suptitle('Learned Facial Pattern Comparison', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(f'learned_facial_comparison_{file_number}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
