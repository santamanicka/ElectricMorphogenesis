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
parser.add_argument('--verbose', type=str, default='False')
parser.add_argument('--visualize', type=str, default='False')
parser.add_argument('--grnOnly', type=str, default='False')  # GRN-only mode (no bioelectric gating)

args = parser.parse_args()

grid_size = args.gridSize
num_sim_iters = args.numSimIters
num_grn_iters = args.numGRNIters
num_learn_iters = args.numLearnIters
lr = args.lr
loss_method = args.lossMethod
learned_parameter_names = ast.literal_eval(args.learnedParameters)
# ideal_face_path not used - targets defined programmatically in define_target_features()
stigmergic_params_path = args.stigmergicParamsPath
file_number = args.fileNumber
verbose = ast.literal_eval(args.verbose)
visualize = ast.literal_eval(args.visualize)
grn_only = ast.literal_eval(args.grnOnly)


# ============================================================
# Define target feature map using explicit indices
# ============================================================
def define_target_features(grid_size, mode='bioelectric'):
    """
    Define target feature map using explicit cell indices (similar to defineTargetVmem).

    Args:
        grid_size: Size of the grid (e.g., 11 for 11x11)
        mode: 'bioelectric' for fine-grained pattern, 'grn_only' for coarse-grained pattern

    For an 11x11 grid:
    - Bioelectric mode (fine-grained):
        - Bone (0): Outer border/frame (default/background)
        - Eye (1): Two small square patches in anterior-lateral positions
        - Nose (2): Vertical stripe at midline
        - Mouth (3): Horizontal stripe in posterior

    - GRN-only mode (coarse-grained):
        - Left Eye (1): rows 0-5, cols 0-4
        - Right Eye (1): rows 0-5, cols 6-10
        - Nose (2): rows 0-5, col 5
        - Mouth (3): rows 6-10, cols 0-10
        - Bone (0): None (all cells assigned to features)

    Returns:
        target_features: (grid_size, grid_size) tensor with feature labels
            0=bone, 1=eye, 2=nose, 3=mouth
    """
    # Initialize all cells as bone (default/background)
    target_features = torch.zeros(grid_size, grid_size, dtype=torch.long)

    if grid_size == 11:
        if mode == 'bioelectric':
            # Fine-grained pattern for bioelectric learning
            # Eye indices: Two square patches (2x2) in anterior-lateral positions
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

        elif mode == 'grn_only':
            # Coarse-grained pattern for GRN-only learning
            # Left eye: rows 0-5, cols 0-4
            for row in range(0, 6):
                for col in range(0, 5):
                    target_features[row, col] = 1  # left eye

            # Right eye: rows 0-5, cols 6-10
            for row in range(0, 6):
                for col in range(6, 11):
                    target_features[row, col] = 1  # right eye

            # Nose: rows 0-5, col 5
            for row in range(0, 6):
                target_features[row, 5] = 2  # nose

            # Mouth: rows 6-10, cols 0-10
            for row in range(6, 11):
                for col in range(0, 11):
                    target_features[row, col] = 3  # mouth

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'bioelectric' or 'grn_only'")

    else:
        # For other grid sizes, could define scaled indices or raise error
        raise ValueError(f"Target feature map not defined for grid_size={grid_size}")

    return target_features


# ============================================================
# Parameter transformation utilities
# ============================================================
def inverse_sigmoid(x, min_val, max_val):
    """
    Compute the raw (unbounded) parameter that maps to x via sigmoid.

    sigmoid(raw) * (max - min) + min = x
    sigmoid(raw) = (x - min) / (max - min)
    raw = logit((x - min) / (max - min))
    """
    normalized = (x - min_val) / (max_val - min_val)
    normalized = torch.clamp(normalized, 1e-6, 1.0 - 1e-6)  # Avoid log(0)
    return torch.logit(normalized)


def apply_sigmoid_constraint(raw_param, min_val, max_val):
    """
    Map unbounded raw parameter to bounded range [min_val, max_val] via sigmoid.

    Guarantees output is always in valid range regardless of raw_param value.
    """
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)


# ============================================================
# Initialize parameters
# ============================================================
def initialize_parameters(learned_params, dtype=torch.float32):
    """
    Initialize learnable parameters using sigmoid parameterization.

    All learned parameters are stored as UNBOUNDED raw values, then transformed
    to their constrained ranges via sigmoid when used. This guarantees constraints
    are always satisfied without needing explicit clipping.
    """
    params = {}

    # Bioelectric gating parameters (matching run_refined_facial_integration.py / refinedFacialGRN.py)
    if 'ca_threshold_percentile' in learned_params:
        # Range: 0.20 to 0.60 (20th to 60th percentile)
        min_val, max_val = 0.20, 0.60
        # Old: initial_val = 0.45 + (torch.rand(1, dtype=dtype) - 0.5) * 0.04  # ±2% noise
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)  # Random from full range
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['ca_threshold_percentile_raw'] = raw_param.clone().requires_grad_(True)
        params['ca_threshold_percentile_min'] = min_val
        params['ca_threshold_percentile_max'] = max_val

    if 'ca_sensitivity' in learned_params:
        # Range: 0.01 to 0.10 (sharpness of sigmoid)
        min_val, max_val = 0.01, 0.10
        # Old: initial_val = 0.04 + (torch.rand(1, dtype=dtype) - 0.5) * 0.01  # ±5% noise
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)  # Random from full range
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['ca_sensitivity_raw'] = raw_param.clone().requires_grad_(True)
        params['ca_sensitivity_min'] = min_val
        params['ca_sensitivity_max'] = max_val

    if 'and_threshold' in learned_params:
        # Range: 1.0 to 1.5 (AND gate threshold)
        min_val, max_val = 1.0, 1.5
        # Old: initial_val = 1.25 + (torch.rand(1, dtype=dtype) - 0.5) * 0.05  # ±2% noise
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)  # Random from full range
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['and_threshold_raw'] = raw_param.clone().requires_grad_(True)
        params['and_threshold_min'] = min_val
        params['and_threshold_max'] = max_val

    if 'and_sharpness' in learned_params:
        # Range: 10.0 to 25.0 (AND gate sharpness)
        min_val, max_val = 10.0, 25.0
        # Old: initial_val = 20.0 + (torch.rand(1, dtype=dtype) - 0.5) * 2.0  # ±5% noise
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)  # Random from full range
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['and_sharpness_raw'] = raw_param.clone().requires_grad_(True)
        params['and_sharpness_min'] = min_val
        params['and_sharpness_max'] = max_val

    # Morphogen parameters
    if 'shh_strength' in learned_params:
        # Range: 0.3 to 2.0 (expanded for GRN-only learning)
        min_val, max_val = 0.3, 2.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['shh_strength_raw'] = raw_param.clone().requires_grad_(True)
        params['shh_strength_min'] = min_val
        params['shh_strength_max'] = max_val

    if 'fgf8_strength' in learned_params:
        # Range: 0.05 to 1.0 (expanded from 0.1-0.5 due to saturation)
        min_val, max_val = 0.05, 1.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['fgf8_strength_raw'] = raw_param.clone().requires_grad_(True)
        params['fgf8_strength_min'] = min_val
        params['fgf8_strength_max'] = max_val

    if 'fgf8_degradation_factor' in learned_params:
        # Range: 2.0 to 30.0 (expanded from 5.0-15.0 due to saturation)
        min_val, max_val = 2.0, 30.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['fgf8_degradation_factor_raw'] = raw_param.clone().requires_grad_(True)
        params['fgf8_degradation_factor_min'] = min_val
        params['fgf8_degradation_factor_max'] = max_val

    # Morphogen shape parameters (decay lengths)
    if 'shh_decay_length' in learned_params:
        # Range: 0.2 to 2.0 (expanded from 0.4-1.2 due to saturation)
        min_val, max_val = 0.2, 2.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['shh_decay_length_raw'] = raw_param.clone().requires_grad_(True)
        params['shh_decay_length_min'] = min_val
        params['shh_decay_length_max'] = max_val

    if 'fgf8_decay_length' in learned_params:
        # Range: 0.05 to 1.0 (expanded from 0.1-0.6 due to saturation)
        min_val, max_val = 0.05, 1.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['fgf8_decay_length_raw'] = raw_param.clone().requires_grad_(True)
        params['fgf8_decay_length_min'] = min_val
        params['fgf8_decay_length_max'] = max_val

    if 'edn1_decay_length' in learned_params:
        # Range: 0.15 to 2.0 (expanded from 0.3-1.0 due to saturation)
        min_val, max_val = 0.15, 2.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['edn1_decay_length_raw'] = raw_param.clone().requires_grad_(True)
        params['edn1_decay_length_min'] = min_val
        params['edn1_decay_length_max'] = max_val

    if 'edn1_strength' in learned_params:
        # Range: 0.3 to 2.5 (expanded from 0.5-1.5 due to saturation)
        min_val, max_val = 0.3, 2.5
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['edn1_strength_raw'] = raw_param.clone().requires_grad_(True)
        params['edn1_strength_min'] = min_val
        params['edn1_strength_max'] = max_val

    if 'edn1_degradation_factor' in learned_params:
        # Range: 0.5 to 10.0 (expanded from 1.0-5.0 due to saturation)
        min_val, max_val = 0.5, 10.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['edn1_degradation_factor_raw'] = raw_param.clone().requires_grad_(True)
        params['edn1_degradation_factor_min'] = min_val
        params['edn1_degradation_factor_max'] = max_val

    if 'diffusion_rate' in learned_params:
        # Range: 0.02 to 0.30 (expanded from 0.05-0.20 due to saturation)
        min_val, max_val = 0.02, 0.30
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['diffusion_rate_raw'] = raw_param.clone().requires_grad_(True)
        params['diffusion_rate_min'] = min_val
        params['diffusion_rate_max'] = max_val

    # Gene activation parameters
    if 'k_activation' in learned_params:
        # Range: 0.02 to 0.40 (expanded from 0.05-0.20 due to saturation)
        min_val, max_val = 0.02, 0.40
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['k_activation_raw'] = raw_param.clone().requires_grad_(True)
        params['k_activation_min'] = min_val
        params['k_activation_max'] = max_val

    if 'k_degradation' in learned_params:
        # Range: 0.001 to 0.05 (expanded from 0.005-0.02 due to saturation)
        min_val, max_val = 0.001, 0.05
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['k_degradation_raw'] = raw_param.clone().requires_grad_(True)
        params['k_degradation_min'] = min_val
        params['k_degradation_max'] = max_val

    # Hill function parameters
    if 'K_morph' in learned_params:
        # Range: 0.05 to 0.8 (expanded from 0.1-0.5 due to saturation)
        min_val, max_val = 0.05, 0.8
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['K_morph_raw'] = raw_param.clone().requires_grad_(True)
        params['K_morph_min'] = min_val
        params['K_morph_max'] = max_val

    if 'n_morph' in learned_params:
        # Range: 0.5 to 6.0 (expanded from 1.0-4.0 due to saturation)
        min_val, max_val = 0.5, 6.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['n_morph_raw'] = raw_param.clone().requires_grad_(True)
        params['n_morph_min'] = min_val
        params['n_morph_max'] = max_val

    if 'K_self' in learned_params:
        # Range: 0.05 to 0.8 (expanded from 0.1-0.5 due to saturation)
        min_val, max_val = 0.05, 0.8
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['K_self_raw'] = raw_param.clone().requires_grad_(True)
        params['K_self_min'] = min_val
        params['K_self_max'] = max_val

    if 'n_self' in learned_params:
        # Range: 0.5 to 6.0 (expanded from 1.0-4.0 due to saturation)
        min_val, max_val = 0.5, 6.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['n_self_raw'] = raw_param.clone().requires_grad_(True)
        params['n_self_min'] = min_val
        params['n_self_max'] = max_val

    # Nose-specific morphogen parameters
    if 'nose_shh_threshold' in learned_params:
        # Range: 0.3 to 0.9 (SHH activation threshold for nose)
        min_val, max_val = 0.3, 0.9
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['nose_shh_threshold_raw'] = raw_param.clone().requires_grad_(True)
        params['nose_shh_threshold_min'] = min_val
        params['nose_shh_threshold_max'] = max_val

    if 'nose_shh_cooperativity' in learned_params:
        # Range: 1.0 to 6.0 (Hill cooperativity for nose SHH response)
        min_val, max_val = 1.0, 6.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['nose_shh_cooperativity_raw'] = raw_param.clone().requires_grad_(True)
        params['nose_shh_cooperativity_min'] = min_val
        params['nose_shh_cooperativity_max'] = max_val

    if 'nose_edn1_threshold' in learned_params:
        # Range: 0.1 to 0.6 (EDN1 inhibition threshold for nose)
        min_val, max_val = 0.1, 0.6
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['nose_edn1_threshold_raw'] = raw_param.clone().requires_grad_(True)
        params['nose_edn1_threshold_min'] = min_val
        params['nose_edn1_threshold_max'] = max_val

    # Mouth-specific morphogen parameters
    if 'mouth_edn1_threshold' in learned_params:
        # Range: 0.2 to 0.8 (EDN1 activation threshold for mouth - higher pushes mouth more posterior)
        min_val, max_val = 0.2, 0.8
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['mouth_edn1_threshold_raw'] = raw_param.clone().requires_grad_(True)
        params['mouth_edn1_threshold_min'] = min_val
        params['mouth_edn1_threshold_max'] = max_val

    if 'mouth_edn1_cooperativity' in learned_params:
        # Range: 1.0 to 6.0 (Hill cooperativity for mouth EDN1 response - higher makes sharper boundary)
        min_val, max_val = 1.0, 6.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['mouth_edn1_cooperativity_raw'] = raw_param.clone().requires_grad_(True)
        params['mouth_edn1_cooperativity_min'] = min_val
        params['mouth_edn1_cooperativity_max'] = max_val

    # Feature classification parameters
    if 'min_mouth_expr' in learned_params:
        # Range: 0.3 to 0.9
        min_val, max_val = 0.3, 0.9
        # Old: initial_val = 0.85
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)  # Random from full range
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['min_mouth_expr_raw'] = raw_param.clone().requires_grad_(True)
        params['min_mouth_expr_min'] = min_val
        params['min_mouth_expr_max'] = max_val

    # ==========================================
    # DEFAULT VALUES (for parameters not being learned)
    # ==========================================
    # These match refinedFacialGRN.py default parameters

    # Morphogen defaults (if not learning them)
    if 'shh_strength' not in params:
        params['shh_strength'] = torch.tensor(1.0, dtype=dtype)
    if 'fgf8_strength' not in params:
        params['fgf8_strength'] = torch.tensor(0.2, dtype=dtype)
    if 'fgf8_degradation_factor' not in params:
        params['fgf8_degradation_factor'] = torch.tensor(10.0, dtype=dtype)
    if 'edn1_strength' not in params:
        params['edn1_strength'] = torch.tensor(1.0, dtype=dtype)
    if 'diffusion_rate' not in params:
        params['diffusion_rate'] = torch.tensor(0.1, dtype=dtype)
    if 'degradation_rate' not in params:
        params['degradation_rate'] = torch.tensor(0.05, dtype=dtype)

    # Gene activation defaults (if not learning them)
    if 'k_activation' not in params:
        params['k_activation'] = torch.tensor(0.10, dtype=dtype)
    if 'k_degradation' not in params:
        params['k_degradation'] = torch.tensor(0.01, dtype=dtype)
    if 'w_initiation' not in params:
        params['w_initiation'] = torch.tensor(0.7, dtype=dtype)  # Changed from 1.0 to match refinedFacialGRN.py
    if 'w_maintenance' not in params:
        params['w_maintenance'] = torch.tensor(0.3, dtype=dtype)  # Changed from 0.0 to 0.3 to enable self-maintenance
    if 'K_morph' not in params:
        params['K_morph'] = torch.tensor(0.3, dtype=dtype)
    if 'n_morph' not in params:
        params['n_morph'] = torch.tensor(2.0, dtype=dtype)
    if 'K_self' not in params:
        params['K_self'] = torch.tensor(0.3, dtype=dtype)
    if 'n_self' not in params:
        params['n_self'] = torch.tensor(2.0, dtype=dtype)

    return params


# ============================================================
# Run simulation with current parameters
# ============================================================
def run_simulation(params, stig_model, transduction, target_features, device, dtype, grn_only_mode=False):
    """
    Run one complete simulation cycle with current parameters.

    Uses sigmoid parameterization to extract constrained parameter values
    from unbounded raw parameters.

    Args:
        grn_only_mode: If True, run GRN without bioelectric gating

    Returns:
        predicted_features: (grid_size, grid_size) feature map
        loss: scalar loss value
    """
    # Apply sigmoid constraints to extract bounded parameter values from raw parameters
    if 'ca_threshold_percentile_raw' in params:
        ca_threshold_pct = apply_sigmoid_constraint(
            params['ca_threshold_percentile_raw'],
            params['ca_threshold_percentile_min'],
            params['ca_threshold_percentile_max']
        )
    else:
        ca_threshold_pct = 0.35

    if 'ca_sensitivity_raw' in params:
        ca_sensitivity = apply_sigmoid_constraint(
            params['ca_sensitivity_raw'],
            params['ca_sensitivity_min'],
            params['ca_sensitivity_max']
        )
    else:
        ca_sensitivity = 0.03

    if 'and_threshold_raw' in params:
        and_threshold = apply_sigmoid_constraint(
            params['and_threshold_raw'],
            params['and_threshold_min'],
            params['and_threshold_max']
        )
    else:
        and_threshold = 1.3

    if 'and_sharpness_raw' in params:
        and_sharpness = apply_sigmoid_constraint(
            params['and_sharpness_raw'],
            params['and_sharpness_min'],
            params['and_sharpness_max']
        )
    else:
        and_sharpness = 18.0

    if 'fgf8_strength_raw' in params:
        fgf8_strength = apply_sigmoid_constraint(
            params['fgf8_strength_raw'],
            params['fgf8_strength_min'],
            params['fgf8_strength_max']
        )
    else:
        fgf8_strength = 0.2

    if 'fgf8_degradation_factor_raw' in params:
        fgf8_deg_factor = apply_sigmoid_constraint(
            params['fgf8_degradation_factor_raw'],
            params['fgf8_degradation_factor_min'],
            params['fgf8_degradation_factor_max']
        )
    else:
        fgf8_deg_factor = 10.0

    if 'k_activation_raw' in params:
        k_activation = apply_sigmoid_constraint(
            params['k_activation_raw'],
            params['k_activation_min'],
            params['k_activation_max']
        )
    else:
        k_activation = 0.10

    if 'k_degradation_raw' in params:
        k_degradation = apply_sigmoid_constraint(
            params['k_degradation_raw'],
            params['k_degradation_min'],
            params['k_degradation_max']
        )
    else:
        k_degradation = 0.01

    if 'min_mouth_expr_raw' in params:
        min_mouth_expr = apply_sigmoid_constraint(
            params['min_mouth_expr_raw'],
            params['min_mouth_expr_min'],
            params['min_mouth_expr_max']
        )
    else:
        min_mouth_expr = 0.85

    # Extract morphogen shape parameters
    if 'shh_decay_length_raw' in params:
        shh_decay_length = apply_sigmoid_constraint(
            params['shh_decay_length_raw'],
            params['shh_decay_length_min'],
            params['shh_decay_length_max']
        )
    else:
        shh_decay_length = 0.8

    if 'fgf8_decay_length_raw' in params:
        fgf8_decay_length = apply_sigmoid_constraint(
            params['fgf8_decay_length_raw'],
            params['fgf8_decay_length_min'],
            params['fgf8_decay_length_max']
        )
    else:
        fgf8_decay_length = 0.3

    if 'edn1_decay_length_raw' in params:
        edn1_decay_length = apply_sigmoid_constraint(
            params['edn1_decay_length_raw'],
            params['edn1_decay_length_min'],
            params['edn1_decay_length_max']
        )
    else:
        edn1_decay_length = 0.6

    if 'edn1_strength_raw' in params:
        edn1_strength = apply_sigmoid_constraint(
            params['edn1_strength_raw'],
            params['edn1_strength_min'],
            params['edn1_strength_max']
        )
    else:
        edn1_strength = 1.0

    if 'edn1_degradation_factor_raw' in params:
        edn1_deg_factor = apply_sigmoid_constraint(
            params['edn1_degradation_factor_raw'],
            params['edn1_degradation_factor_min'],
            params['edn1_degradation_factor_max']
        )
    else:
        edn1_deg_factor = 2.0

    if 'diffusion_rate_raw' in params:
        diffusion_rate = apply_sigmoid_constraint(
            params['diffusion_rate_raw'],
            params['diffusion_rate_min'],
            params['diffusion_rate_max']
        )
    else:
        diffusion_rate = 0.1

    # Extract Hill function parameters
    if 'K_morph_raw' in params:
        K_morph = apply_sigmoid_constraint(
            params['K_morph_raw'],
            params['K_morph_min'],
            params['K_morph_max']
        )
    else:
        K_morph = 0.3

    if 'n_morph_raw' in params:
        n_morph = apply_sigmoid_constraint(
            params['n_morph_raw'],
            params['n_morph_min'],
            params['n_morph_max']
        )
    else:
        n_morph = 2.0

    if 'K_self_raw' in params:
        K_self = apply_sigmoid_constraint(
            params['K_self_raw'],
            params['K_self_min'],
            params['K_self_max']
        )
    else:
        K_self = 0.3

    if 'n_self_raw' in params:
        n_self = apply_sigmoid_constraint(
            params['n_self_raw'],
            params['n_self_min'],
            params['n_self_max']
        )
    else:
        n_self = 2.0

    # Extract nose-specific parameters
    if 'nose_shh_threshold_raw' in params:
        nose_shh_K = apply_sigmoid_constraint(
            params['nose_shh_threshold_raw'],
            params['nose_shh_threshold_min'],
            params['nose_shh_threshold_max']
        )
    else:
        nose_shh_K = 0.7  # Default from refinedFacialGRN.py

    if 'nose_shh_cooperativity_raw' in params:
        nose_shh_n = apply_sigmoid_constraint(
            params['nose_shh_cooperativity_raw'],
            params['nose_shh_cooperativity_min'],
            params['nose_shh_cooperativity_max']
        )
    else:
        nose_shh_n = 4.0  # Default from refinedFacialGRN.py

    if 'nose_edn1_threshold_raw' in params:
        nose_edn1_K = apply_sigmoid_constraint(
            params['nose_edn1_threshold_raw'],
            params['nose_edn1_threshold_min'],
            params['nose_edn1_threshold_max']
        )
    else:
        nose_edn1_K = 0.2  # Default from refinedFacialGRN.py

    # Extract mouth-specific parameters
    if 'mouth_edn1_threshold_raw' in params:
        mouth_edn1_K = apply_sigmoid_constraint(
            params['mouth_edn1_threshold_raw'],
            params['mouth_edn1_threshold_min'],
            params['mouth_edn1_threshold_max']
        )
    else:
        mouth_edn1_K = 0.2  # Default from refinedFacialGRN.py

    if 'mouth_edn1_cooperativity_raw' in params:
        mouth_edn1_n = apply_sigmoid_constraint(
            params['mouth_edn1_cooperativity_raw'],
            params['mouth_edn1_cooperativity_min'],
            params['mouth_edn1_cooperativity_max']
        )
    else:
        mouth_edn1_n = 2.0  # Default from refinedFacialGRN.py

    # Convert tensor parameters to device once (avoid redundant transfers)
    if isinstance(ca_threshold_pct, torch.Tensor):
        ca_threshold_pct = ca_threshold_pct.to(device)
    if isinstance(ca_sensitivity, torch.Tensor):
        ca_sensitivity = ca_sensitivity.to(device)

    # Pass decay lengths to GRN (as tensors if learnable, scalars otherwise)
    # CRITICAL: Pass tensors with gradients so sources can be recomputed during forward pass
    if 'shh_decay_length_raw' in params:
        shh_decay_for_grn = shh_decay_length.to(device) if isinstance(shh_decay_length, torch.Tensor) else shh_decay_length
    else:
        shh_decay_for_grn = 0.8

    if 'fgf8_decay_length_raw' in params:
        fgf8_decay_for_grn = fgf8_decay_length.to(device) if isinstance(fgf8_decay_length, torch.Tensor) else fgf8_decay_length
    else:
        fgf8_decay_for_grn = 0.3

    if 'edn1_decay_length_raw' in params:
        edn1_decay_for_grn = edn1_decay_length.to(device) if isinstance(edn1_decay_length, torch.Tensor) else edn1_decay_length
    else:
        edn1_decay_for_grn = 0.6

    # Create GRN with decay lengths (will be stored as tensors if learnable)
    grn = RefinedFacialGRN(
        grid_size=grid_size,
        device=device,
        dtype=dtype,
        shh_decay_length=shh_decay_for_grn,
        fgf8_decay_length=fgf8_decay_for_grn,
        edn1_decay_length=edn1_decay_for_grn
    )

    # Update GRN parameters if learnable
    # IMPORTANT: Use in-place operations or ensure tensors maintain gradient connection
    # Dictionary assignment CAN work if we're careful about maintaining the tensor references
    if 'fgf8_strength_raw' in params:
        grn.morphogen_params['fgf8_strength'] = fgf8_strength.to(device) if isinstance(fgf8_strength, torch.Tensor) else fgf8_strength
        grn.morphogen_params['fgf8_degradation_factor'] = fgf8_deg_factor.to(device) if isinstance(fgf8_deg_factor, torch.Tensor) else fgf8_deg_factor
    if 'edn1_strength_raw' in params:
        grn.morphogen_params['edn1_strength'] = edn1_strength.to(device) if isinstance(edn1_strength, torch.Tensor) else edn1_strength
    if 'edn1_degradation_factor_raw' in params:
        grn.morphogen_params['edn1_degradation_factor'] = edn1_deg_factor.to(device) if isinstance(edn1_deg_factor, torch.Tensor) else edn1_deg_factor
    if 'diffusion_rate_raw' in params:
        grn.morphogen_params['diffusion_rate'] = diffusion_rate.to(device) if isinstance(diffusion_rate, torch.Tensor) else diffusion_rate
    if 'k_activation_raw' in params:
        grn.gene_params['k_activation'] = k_activation.to(device) if isinstance(k_activation, torch.Tensor) else k_activation
    if 'k_degradation_raw' in params:
        grn.gene_params['k_degradation'] = k_degradation.to(device) if isinstance(k_degradation, torch.Tensor) else k_degradation

    # Update Hill function parameters if learnable
    # These WILL have gradients because they're extracted via sigmoid constraint above
    if 'K_morph_raw' in params:
        grn.gene_params['K_morph'] = K_morph.to(device) if isinstance(K_morph, torch.Tensor) else K_morph
    if 'n_morph_raw' in params:
        grn.gene_params['n_morph'] = n_morph.to(device) if isinstance(n_morph, torch.Tensor) else n_morph
    if 'K_self_raw' in params:
        grn.gene_params['K_self'] = K_self.to(device) if isinstance(K_self, torch.Tensor) else K_self
    if 'n_self_raw' in params:
        grn.gene_params['n_self'] = n_self.to(device) if isinstance(n_self, torch.Tensor) else n_self

    # Update nose-specific parameters if learnable
    if 'nose_shh_threshold_raw' in params:
        grn.gene_params['nose_shh_K'] = nose_shh_K.to(device) if isinstance(nose_shh_K, torch.Tensor) else nose_shh_K
    if 'nose_shh_cooperativity_raw' in params:
        grn.gene_params['nose_shh_n'] = nose_shh_n.to(device) if isinstance(nose_shh_n, torch.Tensor) else nose_shh_n
    if 'nose_edn1_threshold_raw' in params:
        grn.gene_params['nose_edn1_K'] = nose_edn1_K.to(device) if isinstance(nose_edn1_K, torch.Tensor) else nose_edn1_K

    # Update mouth-specific parameters if learnable
    if 'mouth_edn1_threshold_raw' in params:
        grn.gene_params['mouth_edn1_K'] = mouth_edn1_K.to(device) if isinstance(mouth_edn1_K, torch.Tensor) else mouth_edn1_K
    if 'mouth_edn1_cooperativity_raw' in params:
        grn.gene_params['mouth_edn1_n'] = mouth_edn1_n.to(device) if isinstance(mouth_edn1_n, torch.Tensor) else mouth_edn1_n

    # Override AND gate parameters (extract scalar values once)
    grn.and_threshold_override = and_threshold.item() if isinstance(and_threshold, torch.Tensor) else and_threshold
    grn.and_sharpness_override = and_sharpness.item() if isinstance(and_sharpness, torch.Tensor) else and_sharpness

    # Pre-equilibrate morphogens
    for _ in range(1000):
        grn.update_morphogens()

    # Run GRN dynamics
    if grn_only_mode:
        # GRN-only mode: No bioelectric gating
        for _ in range(num_grn_iters):
            grn.update_morphogens()
            grn.update_genes(bioelectric_signals=None)
    else:
        # Bioelectric mode: Use Ca²⁺ gating
        # Get bioelectric signals from transduction module
        bio_signals = transduction.get_gene_modulation_signals()

        # Override Ca²⁺ gating with current parameters
        Ca = bio_signals['Ca'].to(device)
        Ca_threshold = torch.quantile(Ca, ca_threshold_pct)
        bio_gate = torch.sigmoid((Ca_threshold - Ca) / ca_sensitivity)
        bio_signals_override = {
            'Ca': bio_gate,  # Use computed bio_gate directly
        }

        for _ in range(num_grn_iters):
            grn.update_morphogens()
            grn.update_genes(bioelectric_signals=bio_signals_override)

    # Classify features
    classifier = GeneBasedFeatureClassifier(
        grid_size=grid_size,
        device=device,
        dtype=dtype
    )

    # Override mouth threshold if learnable
    if 'min_mouth_expr_raw' in params:
        classifier.min_mouth_expr = min_mouth_expr.to(device) if isinstance(min_mouth_expr, torch.Tensor) else min_mouth_expr

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
    print(f"Mode: {'GRN-only (no bioelectric gating)' if grn_only else 'Bioelectric + GRN'}")
    print(f"Bioelectric simulation: {num_sim_iters} iterations")
    print(f"GRN simulation: {num_grn_iters} iterations per learning step")
    print(f"Learning iterations: {num_learn_iters}")
    print(f"Learning rate: {lr}")
    print(f"Loss method: {loss_method}")
    print(f"Learned parameters: {learned_parameter_names}")
    print("=" * 70 + "\n")

    # Define target face using explicit indices
    target_mode = 'grn_only' if grn_only else 'bioelectric'
    print(f"Defining target feature map from explicit indices (mode: {target_mode})...")
    target_features = define_target_features(grid_size, mode=target_mode).to(device)
    print(f"Target feature counts:")
    unique, counts = torch.unique(target_features, return_counts=True)
    feature_names = ['bone', 'eye', 'nose', 'mouth']
    for label, count in zip(unique, counts):
        print(f"  {feature_names[label]}: {count.item()} cells")
    print()

    # Load and run Stigmergic bioelectric model (once, fixed)
    print("Running Stigmergic bioelectric simulation...")
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
        numSimIters=stig_params['simParameters']['numSimIters'],
    )
    print("✓ Stigmergic simulation complete\n")

    # Extract bioelectric signals (fixed)
    transduction = BioelectricTransduction(grid_size=grid_size, device=device, dtype=dtype)
    rows, cols = stig_params["latticeDims"]
    # Convert dtype first, then move to device (MPS doesn't support float64)
    vmem_grid = stig_model.electricNetwork.Vmem.view(rows, cols).detach().to(dtype=dtype).to(device=device)

    # Run transduction to get Ca²⁺
    for _ in range(100):
        transduction.update(vmem_grid, dt=0.01)
    print("✓ Bioelectric signals extracted\n")

    # Initialize learnable parameters (sigmoid parameterization)
    params = initialize_parameters(learned_parameter_names, dtype=dtype)

    # Collect raw parameters for optimizer (all end with "_raw")
    learned_params_list = [params[f'{name}_raw'] for name in learned_parameter_names
                           if f'{name}_raw' in params]

    # Setup optimizer (operates on unbounded raw parameters)
    optimizer = torch.optim.Rprop(learned_params_list, lr=lr)

    # Learning loop
    best_loss = 999999.0
    best_params = {}
    best_loss_history = []

    print("=" * 70)
    print("STARTING LEARNING LOOP")
    print("=" * 70 + "\n")

    for iter_idx in range(num_learn_iters):
        # Run simulation with current parameters (sigmoid automatically constrains)
        predicted_features, loss = run_simulation(params, stig_model, transduction, target_features, device, dtype, grn_only_mode=grn_only)

        current_loss = loss.item()

        # Track best parameters
        if current_loss < best_loss:
            best_loss = current_loss
            best_loss_history.append((iter_idx, best_loss))

            # Save best RAW parameters (they will be transformed via sigmoid when loaded)
            best_param_bounds = {}
            for param_name in learned_parameter_names:
                raw_name = f'{param_name}_raw'
                if raw_name in params:
                    best_params[param_name] = params[raw_name].detach().clone()
                    # Also save min/max bounds for proper reconstruction
                    if f'{param_name}_min' in params:
                        best_param_bounds[f'{param_name}_min'] = params[f'{param_name}_min']
                    if f'{param_name}_max' in params:
                        best_param_bounds[f'{param_name}_max'] = params[f'{param_name}_max']

            # Save best model
            save_data = {
                'parameters': best_params,
                'parameter_bounds': best_param_bounds,
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
        if verbose and ((iter_idx + 1) % 1 == 0 or iter_idx == 0):
            print(f"Iter {iter_idx+1:3d}/{num_learn_iters}: loss={current_loss:.6f}, best={best_loss:.6f}")

            # Print current CONSTRAINED parameter values (apply sigmoid)
            if (iter_idx + 1) % 1 == 0:
                print("  Current parameters:")
                for param_name in learned_parameter_names:
                    raw_name = f'{param_name}_raw'
                    if raw_name in params:
                        # Apply sigmoid constraint to get actual value
                        constrained_val = apply_sigmoid_constraint(
                            params[raw_name],
                            params[f'{param_name}_min'],
                            params[f'{param_name}_max']
                        )
                        print(f"    {param_name}: {constrained_val.item():.4f}")

    # Final evaluation with best parameters
    print("\n" + "=" * 70)
    print("FINAL EVALUATION WITH BEST PARAMETERS")
    print("=" * 70)

    # Restore best RAW parameters
    for param_name in learned_parameter_names:
        raw_name = f'{param_name}_raw'
        if raw_name in params and param_name in best_params:
            params[raw_name].data = best_params[param_name]

    predicted_features, final_loss = run_simulation(params, stig_model, transduction, target_features, device, dtype, grn_only_mode=grn_only)

    print(f"\nBest loss: {best_loss:.6f}")
    print("\nBest parameters (constrained values):")
    for param_name in learned_parameter_names:
        if param_name in best_params:
            # Apply sigmoid to display actual constrained value using saved bounds
            min_key = f'{param_name}_min'
            max_key = f'{param_name}_max'
            if min_key in best_param_bounds and max_key in best_param_bounds:
                constrained_val = apply_sigmoid_constraint(
                    best_params[param_name],
                    best_param_bounds[min_key],
                    best_param_bounds[max_key]
                )
                print(f"  {param_name}: {constrained_val.item():.4f}")
            else:
                # Fallback to current params dict if bounds not saved (old format)
                if min_key in params and max_key in params:
                    constrained_val = apply_sigmoid_constraint(
                        best_params[param_name],
                        params[min_key],
                        params[max_key]
                    )
                    print(f"  {param_name}: {constrained_val.item():.4f}")
                else:
                    print(f"  {param_name}: [bounds missing, showing raw value: {best_params[param_name].item():.4f}]")

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
    if visualize:
        visualize_results(target_features, predicted_features, file_number)

    print("\n" + "=" * 70)
    print("✅ LEARNING COMPLETE!")
    print("=" * 70)
    print(f"\nSaved best parameters to: ./data/bestLearnedFacialParams_{file_number}.dat")
    print(f"Saved visualization to: learned_facial_comparison_{file_number}.png")


def visualize_results(target_features, predicted_features, file_number):
    """Create visualization comparing target and predicted features"""
    feature_cmap = ListedColormap(["#f9f9f9", "#9b59b6", "#e67e22", "#2ecc71"])

    # Create figure with 3 subplots: left plot, colorbar, right plot
    fig = plt.figure(figsize=(14, 5))

    # Left subplot
    ax1 = plt.subplot(1, 3, 1)
    im = ax1.imshow(target_features.cpu().numpy(), cmap=feature_cmap, vmin=0, vmax=3)
    ax1.set_title('Target (Defined by Indices)', fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Middle subplot for colorbar (invisible axes)
    ax_cbar = plt.subplot(1, 3, 2)
    ax_cbar.axis('off')

    # Add colorbar in the middle
    cbar = fig.colorbar(im, ax=ax_cbar, ticks=[0, 1, 2, 3],
                        fraction=0.6, aspect=15)
    cbar.ax.set_yticklabels(['Bone', 'Eye', 'Nose', 'Mouth'])

    # Right subplot
    ax2 = plt.subplot(1, 3, 3)
    im = ax2.imshow(predicted_features.detach().cpu().numpy(), cmap=feature_cmap, vmin=0, vmax=3)
    ax2.set_title('Predicted (Learned)', fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Add suptitle
    fig.suptitle('Learned Facial Pattern Comparison', fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'learned_facial_comparison_{file_number}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
