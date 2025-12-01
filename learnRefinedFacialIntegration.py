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
parser.add_argument('--learnedParameters', type=str, default="['ca_threshold','ca_sensitivity','and_threshold']")
parser.add_argument('--idealFacePath', type=str, default='IdealFace.png')
parser.add_argument('--stigmergicParamsPath', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--fileNumber', type=int, default=0)
parser.add_argument('--verbose', type=str, default='False')
parser.add_argument('--visualize', type=str, default='False')
parser.add_argument('--grnOnly', type=str, default='False')  # GRN-only mode (no bioelectric gating)
parser.add_argument('--grnParamsPath', type=str, default='')  # Path to pre-learned GRN parameters (.dat file)

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
grn_params_path = args.grnParamsPath


# ============================================================
# Load pre-learned GRN parameters
# ============================================================
def load_learned_grn_params(path):
    """
    Load pre-learned GRN parameters from .dat file.

    Applies sigmoid constraint to convert raw (unbounded) parameters to
    constrained values using saved bounds.

    Args:
        path: Path to learned parameters file (.dat format)

    Returns:
        dict with learned parameter names → constrained values
    """
    print(f"Loading learned GRN parameters from: {path}")
    data = torch.load(path, weights_only=False)

    # Extract parameters and bounds
    learned_params = {}
    param_bounds = data.get('parameter_bounds', {})

    for param_name, raw_value in data['parameters'].items():
        min_key = f'{param_name}_min'
        max_key = f'{param_name}_max'

        if min_key in param_bounds and max_key in param_bounds:
            # Apply sigmoid transformation
            constrained = apply_sigmoid_constraint(
                raw_value,
                param_bounds[min_key],
                param_bounds[max_key]
            )
            learned_params[param_name] = float(constrained.item())
        else:
            # Use raw value if bounds not available
            learned_params[param_name] = float(raw_value.item())

    print(f"Loaded {len(learned_params)} learned GRN parameters")
    return learned_params


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
    - Bioelectric mode (original fine-grained):
        - Bone (0): Outer border/frame (default/background)
        - Eye (1): Two small square patches (2x2) in anterior-lateral positions
        - Nose (2): Vertical stripe at midline (4 cells)
        - Mouth (3): Horizontal stripe in posterior (10 cells)

    - Bioelectric_fine mode (very fine-grained with pre-learned GRN):
        - Bone (0): Background (104 cells)
        - Eye (1): Two tiny square patches (2x2) at rows 2-3, cols 2-3 and 7-8 (8 cells total)
        - Nose (2): Minimal vertical stripe at rows 3-5, col 5 (3 cells)
        - Mouth (3): Narrow horizontal stripe at row 7, cols 3-8 (6 cells)

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

        elif mode == 'bioelectric_fine':
            # Very fine-grained pattern for bioelectric learning with pre-learned GRN
            # This uses much smaller features to test bioelectric gating precision

            # Eye indices: Two tiny square patches (2x2) in anterior-lateral positions
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

            # Nose indices: Minimal vertical stripe at midline (col 5), rows 3-5 (3 cells)
            nose_indices = [
                (3, 5),
                (4, 5),
                (5, 5),
            ]

            # Mouth indices: Narrow horizontal stripe at row 7, cols 3-8 (6 cells)
            mouth_indices = [
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
def initialize_parameters(learned_params, fixed_grn_params=None, dtype=torch.float32):
    """
    Initialize learnable parameters using sigmoid parameterization.

    All learned parameters are stored as UNBOUNDED raw values, then transformed
    to their constrained ranges via sigmoid when used. This guarantees constraints
    are always satisfied without needing explicit clipping.

    Args:
        learned_params: List of parameter names to learn (will create _raw versions)
        fixed_grn_params: Optional dict of pre-learned GRN parameter values to fix
        dtype: Torch dtype for parameters

    Returns:
        params dict with raw (learnable) and fixed (non-learnable) parameters
    """
    params = {}
    fixed_grn_params = fixed_grn_params or {}

    # Bioelectric gating parameters (matching run_refined_facial_integration.py / refinedFacialGRN.py)
    # These are ALWAYS learnable, never fixed
    if 'ca_threshold' in learned_params:
        # Range: 0.0 to 1.0 (direct Ca threshold value)
        # Ca is normalized to [0,1], so threshold should be in same range
        min_val, max_val = 0.0, 1.0
        # Initialize near 0.5 with small noise (±10%)
        initial_val = 0.5 + (torch.rand(1, dtype=dtype) - 0.5) * 0.1
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['ca_threshold_raw'] = raw_param.clone().requires_grad_(True)
        params['ca_threshold_min'] = min_val
        params['ca_threshold_max'] = max_val

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

    # ==========================================
    # GRN parameters (may be fixed or learnable)
    # ==========================================
    # If a parameter is in fixed_grn_params, use the fixed value (no _raw version)
    # Otherwise, if it's in learned_params, create a learnable _raw version

    # Morphogen parameters
    if 'shh_strength' in fixed_grn_params:
        # Use fixed pre-learned value
        params['shh_strength'] = torch.tensor(fixed_grn_params['shh_strength'], dtype=dtype)
    elif 'shh_strength' in learned_params:
        # Range: 0.3 to 2.0 (expanded for GRN-only learning)
        min_val, max_val = 0.3, 2.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['shh_strength_raw'] = raw_param.clone().requires_grad_(True)
        params['shh_strength_min'] = min_val
        params['shh_strength_max'] = max_val

    if 'fgf8_strength' in fixed_grn_params:
        params['fgf8_strength'] = torch.tensor(fixed_grn_params['fgf8_strength'], dtype=dtype)
    elif 'fgf8_strength' in learned_params:
        # Range: 0.05 to 1.0 (expanded from 0.1-0.5 due to saturation)
        min_val, max_val = 0.05, 1.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['fgf8_strength_raw'] = raw_param.clone().requires_grad_(True)
        params['fgf8_strength_min'] = min_val
        params['fgf8_strength_max'] = max_val

    if 'fgf8_degradation_factor' in fixed_grn_params:
        params['fgf8_degradation_factor'] = torch.tensor(fixed_grn_params['fgf8_degradation_factor'], dtype=dtype)
    elif 'fgf8_degradation_factor' in learned_params:
        # Range: 2.0 to 30.0 (expanded from 5.0-15.0 due to saturation)
        min_val, max_val = 2.0, 30.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['fgf8_degradation_factor_raw'] = raw_param.clone().requires_grad_(True)
        params['fgf8_degradation_factor_min'] = min_val
        params['fgf8_degradation_factor_max'] = max_val

    # Morphogen shape parameters (decay lengths)
    if 'shh_decay_length' in fixed_grn_params:
        params['shh_decay_length'] = torch.tensor(fixed_grn_params['shh_decay_length'], dtype=dtype)
    elif 'shh_decay_length' in learned_params:
        # Range: 0.2 to 2.0 (expanded from 0.4-1.2 due to saturation)
        min_val, max_val = 0.2, 2.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['shh_decay_length_raw'] = raw_param.clone().requires_grad_(True)
        params['shh_decay_length_min'] = min_val
        params['shh_decay_length_max'] = max_val

    if 'fgf8_decay_length' in fixed_grn_params:
        params['fgf8_decay_length'] = torch.tensor(fixed_grn_params['fgf8_decay_length'], dtype=dtype)
    elif 'fgf8_decay_length' in learned_params:
        # Range: 0.05 to 1.0 (expanded from 0.1-0.6 due to saturation)
        min_val, max_val = 0.05, 1.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['fgf8_decay_length_raw'] = raw_param.clone().requires_grad_(True)
        params['fgf8_decay_length_min'] = min_val
        params['fgf8_decay_length_max'] = max_val

    if 'edn1_decay_length' in fixed_grn_params:
        params['edn1_decay_length'] = torch.tensor(fixed_grn_params['edn1_decay_length'], dtype=dtype)
    elif 'edn1_decay_length' in learned_params:
        # Range: 0.15 to 2.0 (expanded from 0.3-1.0 due to saturation)
        min_val, max_val = 0.15, 2.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['edn1_decay_length_raw'] = raw_param.clone().requires_grad_(True)
        params['edn1_decay_length_min'] = min_val
        params['edn1_decay_length_max'] = max_val

    if 'edn1_strength' in fixed_grn_params:
        params['edn1_strength'] = torch.tensor(fixed_grn_params['edn1_strength'], dtype=dtype)
    elif 'edn1_strength' in learned_params:
        # Range: 0.3 to 2.5 (expanded from 0.5-1.5 due to saturation)
        min_val, max_val = 0.3, 2.5
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['edn1_strength_raw'] = raw_param.clone().requires_grad_(True)
        params['edn1_strength_min'] = min_val
        params['edn1_strength_max'] = max_val

    if 'edn1_degradation_factor' in fixed_grn_params:
        params['edn1_degradation_factor'] = torch.tensor(fixed_grn_params['edn1_degradation_factor'], dtype=dtype)
    elif 'edn1_degradation_factor' in learned_params:
        # Range: 0.5 to 10.0 (expanded from 1.0-5.0 due to saturation)
        min_val, max_val = 0.5, 10.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['edn1_degradation_factor_raw'] = raw_param.clone().requires_grad_(True)
        params['edn1_degradation_factor_min'] = min_val
        params['edn1_degradation_factor_max'] = max_val

    if 'diffusion_rate' in fixed_grn_params:
        params['diffusion_rate'] = torch.tensor(fixed_grn_params['diffusion_rate'], dtype=dtype)
    elif 'diffusion_rate' in learned_params:
        # Range: 0.02 to 0.30 (expanded from 0.05-0.20 due to saturation)
        min_val, max_val = 0.02, 0.30
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['diffusion_rate_raw'] = raw_param.clone().requires_grad_(True)
        params['diffusion_rate_min'] = min_val
        params['diffusion_rate_max'] = max_val

    # Gene activation parameters
    if 'k_activation' in fixed_grn_params:
        params['k_activation'] = torch.tensor(fixed_grn_params['k_activation'], dtype=dtype)
    elif 'k_activation' in learned_params:
        # Range: 0.02 to 0.40 (expanded from 0.05-0.20 due to saturation)
        min_val, max_val = 0.02, 0.40
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['k_activation_raw'] = raw_param.clone().requires_grad_(True)
        params['k_activation_min'] = min_val
        params['k_activation_max'] = max_val

    if 'k_degradation' in fixed_grn_params:
        params['k_degradation'] = torch.tensor(fixed_grn_params['k_degradation'], dtype=dtype)
    elif 'k_degradation' in learned_params:
        # Range: 0.001 to 0.05 (expanded from 0.005-0.02 due to saturation)
        min_val, max_val = 0.001, 0.05
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['k_degradation_raw'] = raw_param.clone().requires_grad_(True)
        params['k_degradation_min'] = min_val
        params['k_degradation_max'] = max_val

    # Hill function parameters
    if 'K_morph' in fixed_grn_params:
        params['K_morph'] = torch.tensor(fixed_grn_params['K_morph'], dtype=dtype)
    elif 'K_morph' in learned_params:
        # Range: 0.05 to 0.8 (expanded from 0.1-0.5 due to saturation)
        min_val, max_val = 0.05, 0.8
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['K_morph_raw'] = raw_param.clone().requires_grad_(True)
        params['K_morph_min'] = min_val
        params['K_morph_max'] = max_val

    if 'n_morph' in fixed_grn_params:
        params['n_morph'] = torch.tensor(fixed_grn_params['n_morph'], dtype=dtype)
    elif 'n_morph' in learned_params:
        # Range: 0.5 to 6.0 (expanded from 1.0-4.0 due to saturation)
        min_val, max_val = 0.5, 6.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['n_morph_raw'] = raw_param.clone().requires_grad_(True)
        params['n_morph_min'] = min_val
        params['n_morph_max'] = max_val

    if 'K_self' in fixed_grn_params:
        params['K_self'] = torch.tensor(fixed_grn_params['K_self'], dtype=dtype)
    elif 'K_self' in learned_params:
        # Range: 0.05 to 0.8 (expanded from 0.1-0.5 due to saturation)
        min_val, max_val = 0.05, 0.8
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['K_self_raw'] = raw_param.clone().requires_grad_(True)
        params['K_self_min'] = min_val
        params['K_self_max'] = max_val

    if 'n_self' in fixed_grn_params:
        params['n_self'] = torch.tensor(fixed_grn_params['n_self'], dtype=dtype)
    elif 'n_self' in learned_params:
        # Range: 0.5 to 6.0 (expanded from 1.0-4.0 due to saturation)
        min_val, max_val = 0.5, 6.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['n_self_raw'] = raw_param.clone().requires_grad_(True)
        params['n_self_min'] = min_val
        params['n_self_max'] = max_val

    # Nose-specific morphogen parameters
    if 'nose_shh_threshold' in fixed_grn_params:
        params['nose_shh_threshold'] = torch.tensor(fixed_grn_params['nose_shh_threshold'], dtype=dtype)
    elif 'nose_shh_threshold' in learned_params:
        # Range: 0.3 to 0.9 (SHH activation threshold for nose)
        min_val, max_val = 0.3, 0.9
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['nose_shh_threshold_raw'] = raw_param.clone().requires_grad_(True)
        params['nose_shh_threshold_min'] = min_val
        params['nose_shh_threshold_max'] = max_val

    if 'nose_shh_cooperativity' in fixed_grn_params:
        params['nose_shh_cooperativity'] = torch.tensor(fixed_grn_params['nose_shh_cooperativity'], dtype=dtype)
    elif 'nose_shh_cooperativity' in learned_params:
        # Range: 1.0 to 6.0 (Hill cooperativity for nose SHH response)
        min_val, max_val = 1.0, 6.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['nose_shh_cooperativity_raw'] = raw_param.clone().requires_grad_(True)
        params['nose_shh_cooperativity_min'] = min_val
        params['nose_shh_cooperativity_max'] = max_val

    if 'nose_edn1_threshold' in fixed_grn_params:
        params['nose_edn1_threshold'] = torch.tensor(fixed_grn_params['nose_edn1_threshold'], dtype=dtype)
    elif 'nose_edn1_threshold' in learned_params:
        # Range: 0.1 to 0.6 (EDN1 inhibition threshold for nose)
        min_val, max_val = 0.1, 0.6
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['nose_edn1_threshold_raw'] = raw_param.clone().requires_grad_(True)
        params['nose_edn1_threshold_min'] = min_val
        params['nose_edn1_threshold_max'] = max_val

    # Mouth-specific morphogen parameters
    if 'mouth_edn1_threshold' in fixed_grn_params:
        params['mouth_edn1_threshold'] = torch.tensor(fixed_grn_params['mouth_edn1_threshold'], dtype=dtype)
    elif 'mouth_edn1_threshold' in learned_params:
        # Range: 0.2 to 0.8 (EDN1 activation threshold for mouth - higher pushes mouth more posterior)
        min_val, max_val = 0.2, 0.8
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['mouth_edn1_threshold_raw'] = raw_param.clone().requires_grad_(True)
        params['mouth_edn1_threshold_min'] = min_val
        params['mouth_edn1_threshold_max'] = max_val

    if 'mouth_edn1_cooperativity' in fixed_grn_params:
        params['mouth_edn1_cooperativity'] = torch.tensor(fixed_grn_params['mouth_edn1_cooperativity'], dtype=dtype)
    elif 'mouth_edn1_cooperativity' in learned_params:
        # Range: 1.0 to 6.0 (Hill cooperativity for mouth EDN1 response - higher makes sharper boundary)
        min_val, max_val = 1.0, 6.0
        initial_val = min_val + torch.rand(1, dtype=dtype) * (max_val - min_val)
        raw_param = inverse_sigmoid(initial_val, min_val, max_val)
        params['mouth_edn1_cooperativity_raw'] = raw_param.clone().requires_grad_(True)
        params['mouth_edn1_cooperativity_min'] = min_val
        params['mouth_edn1_cooperativity_max'] = max_val

    # Feature classification parameters
    if 'min_mouth_expr' in fixed_grn_params:
        params['min_mouth_expr'] = torch.tensor(fixed_grn_params['min_mouth_expr'], dtype=dtype)
    elif 'min_mouth_expr' in learned_params:
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
# Helper function for parameter extraction
# ============================================================
def extract_parameter(params, param_name, default_value):
    """
    Extract parameter value from params dict.

    Handles three cases:
    1. Fixed value (param_name exists but param_name_raw doesn't)
    2. Learnable value (param_name_raw exists, needs sigmoid constraint)
    3. Default value (neither exists)

    Args:
        params: Parameter dictionary
        param_name: Base parameter name (without _raw suffix)
        default_value: Default value if parameter not found

    Returns:
        Parameter value (tensor or scalar)
    """
    raw_key = f'{param_name}_raw'

    # Case 1: Fixed pre-learned value
    if param_name in params and raw_key not in params:
        return params[param_name]

    # Case 2: Learnable parameter (apply sigmoid constraint)
    elif raw_key in params:
        return apply_sigmoid_constraint(
            params[raw_key],
            params[f'{param_name}_min'],
            params[f'{param_name}_max']
        )

    # Case 3: Default value
    else:
        return default_value


# ============================================================
# Run simulation with current parameters
# ============================================================
def run_simulation(params, stig_model, transduction, target_features, device, dtype, vmem_grid=None, grn_only_mode=False):
    """
    Run one complete simulation cycle with current parameters.

    Uses sigmoid parameterization to extract constrained parameter values
    from unbounded raw parameters.

    Args:
        params: Parameter dictionary
        stig_model: Stigmergic bioelectric model
        transduction: BioelectricTransduction module
        target_features: Target feature map
        device: Torch device
        dtype: Torch dtype
        vmem_grid: Membrane voltage grid (required for Ca pre-equilibration)
        grn_only_mode: If True, run GRN without bioelectric gating

    Returns:
        predicted_features: (grid_size, grid_size) feature map
        loss: scalar loss value
    """
    # Apply sigmoid constraints to extract bounded parameter values from raw parameters
    if 'ca_threshold_raw' in params:
        # Direct Ca threshold value (0-1)
        ca_threshold = apply_sigmoid_constraint(
            params['ca_threshold_raw'],
            params['ca_threshold_min'],
            params['ca_threshold_max']
        )
    else:
        ca_threshold = 0.5  # Default direct threshold value

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

    # Extract GRN parameters (handles fixed/learnable/default cases)
    fgf8_strength = extract_parameter(params, 'fgf8_strength', 0.2)
    fgf8_deg_factor = extract_parameter(params, 'fgf8_degradation_factor', 10.0)
    k_activation = extract_parameter(params, 'k_activation', 0.10)
    k_degradation = extract_parameter(params, 'k_degradation', 0.01)
    min_mouth_expr = extract_parameter(params, 'min_mouth_expr', 0.85)

    # Morphogen shape parameters
    shh_decay_length = extract_parameter(params, 'shh_decay_length', 0.8)
    fgf8_decay_length = extract_parameter(params, 'fgf8_decay_length', 0.3)
    edn1_decay_length = extract_parameter(params, 'edn1_decay_length', 0.6)
    edn1_strength = extract_parameter(params, 'edn1_strength', 1.0)
    edn1_deg_factor = extract_parameter(params, 'edn1_degradation_factor', 2.0)
    diffusion_rate = extract_parameter(params, 'diffusion_rate', 0.1)

    # Hill function parameters
    K_morph = extract_parameter(params, 'K_morph', 0.3)
    n_morph = extract_parameter(params, 'n_morph', 2.0)
    K_self = extract_parameter(params, 'K_self', 0.3)
    n_self = extract_parameter(params, 'n_self', 2.0)

    # Nose-specific parameters
    nose_shh_K = extract_parameter(params, 'nose_shh_threshold', 0.7)
    nose_shh_n = extract_parameter(params, 'nose_shh_cooperativity', 4.0)
    nose_edn1_K = extract_parameter(params, 'nose_edn1_threshold', 0.2)

    # Mouth-specific parameters
    mouth_edn1_K = extract_parameter(params, 'mouth_edn1_threshold', 0.2)
    mouth_edn1_n = extract_parameter(params, 'mouth_edn1_cooperativity', 2.0)

    # Convert tensor parameters to device once (avoid redundant transfers)
    if isinstance(ca_threshold, torch.Tensor):
        ca_threshold = ca_threshold.to(device)
    if isinstance(ca_sensitivity, torch.Tensor):
        ca_sensitivity = ca_sensitivity.to(device)

    # Pass decay lengths to GRN (as tensors if learnable, scalars otherwise)
    # CRITICAL: Pass tensors with gradients so sources can be recomputed during forward pass
    # Check for _raw (learnable) OR direct parameter (fixed from file)
    if 'shh_decay_length_raw' in params or 'shh_decay_length' in params:
        shh_decay_for_grn = shh_decay_length.to(device) if isinstance(shh_decay_length, torch.Tensor) else shh_decay_length
    else:
        shh_decay_for_grn = 0.8

    if 'fgf8_decay_length_raw' in params or 'fgf8_decay_length' in params:
        fgf8_decay_for_grn = fgf8_decay_length.to(device) if isinstance(fgf8_decay_length, torch.Tensor) else fgf8_decay_length
    else:
        fgf8_decay_for_grn = 0.3

    if 'edn1_decay_length_raw' in params or 'edn1_decay_length' in params:
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

    # Update GRN parameters if learnable OR fixed
    # IMPORTANT: Use in-place operations or ensure tensors maintain gradient connection
    # Dictionary assignment CAN work if we're careful about maintaining the tensor references
    # Check for _raw (learnable) OR direct parameter (fixed from file)
    if 'fgf8_strength_raw' in params or 'fgf8_strength' in params:
        grn.morphogen_params['fgf8_strength'] = fgf8_strength.to(device) if isinstance(fgf8_strength, torch.Tensor) else fgf8_strength
    if 'fgf8_degradation_factor_raw' in params or 'fgf8_degradation_factor' in params:
        grn.morphogen_params['fgf8_degradation_factor'] = fgf8_deg_factor.to(device) if isinstance(fgf8_deg_factor, torch.Tensor) else fgf8_deg_factor
    if 'edn1_strength_raw' in params or 'edn1_strength' in params:
        grn.morphogen_params['edn1_strength'] = edn1_strength.to(device) if isinstance(edn1_strength, torch.Tensor) else edn1_strength
    if 'edn1_degradation_factor_raw' in params or 'edn1_degradation_factor' in params:
        grn.morphogen_params['edn1_degradation_factor'] = edn1_deg_factor.to(device) if isinstance(edn1_deg_factor, torch.Tensor) else edn1_deg_factor
    if 'diffusion_rate_raw' in params or 'diffusion_rate' in params:
        grn.morphogen_params['diffusion_rate'] = diffusion_rate.to(device) if isinstance(diffusion_rate, torch.Tensor) else diffusion_rate
    if 'k_activation_raw' in params or 'k_activation' in params:
        grn.gene_params['k_activation'] = k_activation.to(device) if isinstance(k_activation, torch.Tensor) else k_activation
    if 'k_degradation_raw' in params or 'k_degradation' in params:
        grn.gene_params['k_degradation'] = k_degradation.to(device) if isinstance(k_degradation, torch.Tensor) else k_degradation

    # Update Hill function parameters if learnable OR fixed
    # These WILL have gradients because they're extracted via sigmoid constraint above
    if 'K_morph_raw' in params or 'K_morph' in params:
        grn.gene_params['K_morph'] = K_morph.to(device) if isinstance(K_morph, torch.Tensor) else K_morph
    if 'n_morph_raw' in params or 'n_morph' in params:
        grn.gene_params['n_morph'] = n_morph.to(device) if isinstance(n_morph, torch.Tensor) else n_morph
    if 'K_self_raw' in params or 'K_self' in params:
        grn.gene_params['K_self'] = K_self.to(device) if isinstance(K_self, torch.Tensor) else K_self
    if 'n_self_raw' in params or 'n_self' in params:
        grn.gene_params['n_self'] = n_self.to(device) if isinstance(n_self, torch.Tensor) else n_self

    # Update nose-specific parameters if learnable OR fixed
    if 'nose_shh_threshold_raw' in params or 'nose_shh_threshold' in params:
        grn.gene_params['nose_shh_K'] = nose_shh_K.to(device) if isinstance(nose_shh_K, torch.Tensor) else nose_shh_K
    if 'nose_shh_cooperativity_raw' in params or 'nose_shh_cooperativity' in params:
        grn.gene_params['nose_shh_n'] = nose_shh_n.to(device) if isinstance(nose_shh_n, torch.Tensor) else nose_shh_n
    if 'nose_edn1_threshold_raw' in params or 'nose_edn1_threshold' in params:
        grn.gene_params['nose_edn1_K'] = nose_edn1_K.to(device) if isinstance(nose_edn1_K, torch.Tensor) else nose_edn1_K

    # Update mouth-specific parameters if learnable OR fixed
    if 'mouth_edn1_threshold_raw' in params or 'mouth_edn1_threshold' in params:
        grn.gene_params['mouth_edn1_K'] = mouth_edn1_K.to(device) if isinstance(mouth_edn1_K, torch.Tensor) else mouth_edn1_K
    if 'mouth_edn1_cooperativity_raw' in params or 'mouth_edn1_cooperativity' in params:
        grn.gene_params['mouth_edn1_n'] = mouth_edn1_n.to(device) if isinstance(mouth_edn1_n, torch.Tensor) else mouth_edn1_n

    # Override AND gate parameters (keep as tensors to maintain gradient flow)
    grn.and_threshold_override = and_threshold.to(device) if isinstance(and_threshold, torch.Tensor) else and_threshold
    grn.and_sharpness_override = and_sharpness.to(device) if isinstance(and_sharpness, torch.Tensor) else and_sharpness

    # Override Ca²⁺ gating parameters (keep as tensors to maintain gradient flow)
    grn.ca_threshold_override = ca_threshold.to(device) if isinstance(ca_threshold, torch.Tensor) else ca_threshold
    grn.ca_sensitivity_override = ca_sensitivity.to(device) if isinstance(ca_sensitivity, torch.Tensor) else ca_sensitivity

    # Pre-equilibrate morphogens
    for _ in range(1000):
        grn.update_morphogens()

    # Pre-equilibrate Ca²⁺ signals (only in bioelectric mode)
    if not grn_only_mode and vmem_grid is not None:
        # Run bioelectric transduction for 100 steps to equilibrate Ca²⁺
        for _ in range(100):
            transduction.update(vmem_grid, dt=0.01)

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

        # Pass raw Ca signal - GRN will apply learnable threshold/sensitivity via overrides
        Ca = bio_signals['Ca'].to(device)
        bio_signals_pass = {
            'Ca': Ca,  # Pass raw normalized Ca signal
        }

        for _ in range(num_grn_iters):
            grn.update_morphogens()
            grn.update_genes(bioelectric_signals=bio_signals_pass)

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
        # Cross-entropy loss using continuous scores with moderate class balancing
        # scores_tensor: (4, grid_size, grid_size) -> (1, 4, grid_size, grid_size)
        # target_features: (grid_size, grid_size) -> (1, grid_size, grid_size)

        # Use square root of inverse frequency for more moderate balancing
        # This gives rare classes more weight without making them dominate
        unique, counts = torch.unique(target_features, return_counts=True)
        total_cells = grid_size * grid_size
        class_weights = torch.ones(4, device=device, dtype=dtype)
        for label, count in zip(unique, counts):
            # Square root balancing: less extreme than linear inverse frequency
            class_weights[label] = torch.sqrt(torch.tensor(total_cells / (count.float() * 4.0), device=device, dtype=dtype))

        # Normalize so mean weight is 1.0
        class_weights = class_weights / class_weights.mean()

        loss = torch.nn.functional.cross_entropy(
            scores_tensor.unsqueeze(0),
            target_features.unsqueeze(0),
            weight=class_weights,
            reduction='mean'
        )
    elif loss_method == 'featureMapMSE':
        # MSE on soft scores vs one-hot target with class balancing
        target_onehot = torch.nn.functional.one_hot(target_features, num_classes=4).float()
        target_onehot = target_onehot.permute(2, 0, 1)  # (grid, grid, 4) -> (4, grid, grid)

        # Softmax on scores to get probabilities
        probs = torch.softmax(scores_tensor, dim=0)

        # Compute class weights (same as cross-entropy)
        unique, counts = torch.unique(target_features, return_counts=True)
        total_cells = grid_size * grid_size
        class_weights = torch.ones(4, device=device, dtype=dtype)
        for label, count in zip(unique, counts):
            class_weights[label] = torch.sqrt(torch.tensor(total_cells / (count.float() * 4.0), device=device, dtype=dtype))
        class_weights = class_weights / class_weights.mean()

        # Apply per-pixel weighting based on target class
        # For each pixel, weight the MSE by the class weight of its target label
        pixel_weights = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)
        for label in range(4):
            mask = (target_features == label)
            pixel_weights[mask] = class_weights[label]

        # Compute weighted MSE: weight each pixel's contribution
        squared_errors = ((probs - target_onehot) ** 2).sum(dim=0)  # Sum over class dimension
        loss = (squared_errors * pixel_weights).mean()
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

    # Load pre-learned GRN parameters if provided
    fixed_grn_params = None
    if grn_params_path:
        print(f"Loading pre-learned GRN parameters from: {grn_params_path}")
        fixed_grn_params = load_learned_grn_params(grn_params_path)
        print(f"✓ Loaded {len(fixed_grn_params)} fixed GRN parameters")
        print()

        # Filter learned_parameter_names to only bioelectric params when GRN is fixed
        bioelectric_param_names = ['ca_threshold', 'ca_sensitivity', 'and_threshold', 'and_sharpness']
        learned_parameter_names_filtered = [p for p in learned_parameter_names if p in bioelectric_param_names]
        print(f"GRN parameters fixed, learning only bioelectric parameters: {learned_parameter_names_filtered}")
    else:
        learned_parameter_names_filtered = learned_parameter_names

    print("=" * 70)
    print("LEARNING REFINED FACIAL INTEGRATION PARAMETERS")
    print("=" * 70)
    print(f"Device: {device_name}")
    print(f"Data type: {dtype}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Mode: {'GRN-only (no bioelectric gating)' if grn_only else 'Bioelectric + GRN (fixed GRN)' if fixed_grn_params else 'Bioelectric + GRN'}")
    print(f"Bioelectric simulation: {num_sim_iters} iterations")
    print(f"GRN simulation: {num_grn_iters} iterations per learning step")
    print(f"Learning iterations: {num_learn_iters}")
    print(f"Learning rate: {lr}")
    print(f"Loss method: {loss_method}")
    print(f"Learned parameters: {learned_parameter_names_filtered}")
    print("=" * 70 + "\n")

    # Define target face using explicit indices
    # Use 'bioelectric_fine' mode when GRN is fixed (smaller features for testing Ca gating precision)
    if fixed_grn_params:
        target_mode = 'bioelectric_fine'
    else:
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
    params = initialize_parameters(learned_parameter_names_filtered, fixed_grn_params=fixed_grn_params, dtype=dtype)

    # Collect raw parameters for optimizer (all end with "_raw")
    learned_params_list = [params[f'{name}_raw'] for name in learned_parameter_names_filtered
                           if f'{name}_raw' in params]

    # Setup optimizer (operates on unbounded raw parameters)
    optimizer = torch.optim.Rprop(learned_params_list, lr=lr)

    # Diagnostic: Print which parameters are being optimized
    print("\n" + "=" * 70)
    print("PARAMETERS IN OPTIMIZER:")
    for name in learned_parameter_names_filtered:
        if f'{name}_raw' in params:
            print(f"  {name}: requires_grad={params[f'{name}_raw'].requires_grad}")
    print("=" * 70 + "\n")

    # Learning loop
    best_loss = 999999.0
    best_params = {}
    best_loss_history = []

    print("=" * 70)
    print("STARTING LEARNING LOOP")
    print("=" * 70 + "\n")

    # Diagnostic: Check what fixed GRN produces WITHOUT bioelectric gating (iteration -1)
    if fixed_grn_params and verbose:
        print("=" * 70)
        print("DIAGNOSTIC: Fixed GRN output WITHOUT bioelectric gating")
        print("=" * 70)

        # Create GRN with the SAME fixed parameters from file 44
        # Extract decay lengths from fixed params
        shh_decay = fixed_grn_params.get('shh_decay_length', 0.8)
        fgf8_decay = fixed_grn_params.get('fgf8_decay_length', 0.3)
        edn1_decay = fixed_grn_params.get('edn1_decay_length', 0.6)

        test_grn = RefinedFacialGRN(
            grid_size=grid_size,
            device=device,
            dtype=dtype,
            shh_decay_length=shh_decay,
            fgf8_decay_length=fgf8_decay,
            edn1_decay_length=edn1_decay
        )

        # Set all other fixed GRN parameters
        if 'fgf8_strength' in fixed_grn_params:
            test_grn.morphogen_params['fgf8_strength'] = fixed_grn_params['fgf8_strength']
        if 'fgf8_degradation_factor' in fixed_grn_params:
            test_grn.morphogen_params['fgf8_degradation_factor'] = fixed_grn_params['fgf8_degradation_factor']
        if 'edn1_strength' in fixed_grn_params:
            test_grn.morphogen_params['edn1_strength'] = fixed_grn_params['edn1_strength']
        if 'edn1_degradation_factor' in fixed_grn_params:
            test_grn.morphogen_params['edn1_degradation_factor'] = fixed_grn_params['edn1_degradation_factor']
        if 'diffusion_rate' in fixed_grn_params:
            test_grn.morphogen_params['diffusion_rate'] = fixed_grn_params['diffusion_rate']
        if 'k_activation' in fixed_grn_params:
            test_grn.gene_params['k_activation'] = fixed_grn_params['k_activation']
        if 'k_degradation' in fixed_grn_params:
            test_grn.gene_params['k_degradation'] = fixed_grn_params['k_degradation']
        if 'K_self' in fixed_grn_params:
            test_grn.gene_params['K_self'] = fixed_grn_params['K_self']
        if 'n_self' in fixed_grn_params:
            test_grn.gene_params['n_self'] = fixed_grn_params['n_self']
        if 'nose_shh_threshold' in fixed_grn_params:
            test_grn.gene_params['nose_shh_K'] = fixed_grn_params['nose_shh_threshold']
        if 'nose_shh_cooperativity' in fixed_grn_params:
            test_grn.gene_params['nose_shh_n'] = fixed_grn_params['nose_shh_cooperativity']
        if 'nose_edn1_threshold' in fixed_grn_params:
            test_grn.gene_params['nose_edn1_K'] = fixed_grn_params['nose_edn1_threshold']
        if 'mouth_edn1_threshold' in fixed_grn_params:
            test_grn.gene_params['mouth_edn1_K'] = fixed_grn_params['mouth_edn1_threshold']
        if 'mouth_edn1_cooperativity' in fixed_grn_params:
            test_grn.gene_params['mouth_edn1_n'] = fixed_grn_params['mouth_edn1_cooperativity']

        # Equilibrate morphogens and run GRN without bioelectric gating
        for _ in range(1000):
            test_grn.update_morphogens()
        for _ in range(num_grn_iters):
            test_grn.update(bioelectric_signals=None)

        test_classifier = GeneBasedFeatureClassifier(grid_size=grid_size, device=device, dtype=dtype)
        test_classification = test_classifier.classify(test_grn.get_gene_grids(), mode='both')
        test_features = test_classification['features']

        print("Fixed GRN predictions (no bioelectric gating):")
        unique, counts = torch.unique(test_features, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {feature_names[label]}: {count.item()} cells")

        # Check overlap with target
        print("\nOverlap with fine-grained target positions:")
        for target_label in range(4):
            target_mask = target_features == target_label
            if target_mask.sum() > 0:
                overlap = ((test_features == target_label) & target_mask).sum().item()
                total_target = target_mask.sum().item()
                pct = 100.0 * overlap / total_target
                print(f"  {feature_names[target_label]}: {overlap}/{total_target} ({pct:.0f}%) target cells match GRN output")

        # Check Ca²⁺ spatial pattern
        print("\nCa²⁺ spatial pattern analysis:")
        bio_signals = transduction.get_gene_modulation_signals()
        Ca = bio_signals['Ca'].to(device)
        print(f"  Ca mean: {Ca.mean().item():.4f}, std: {Ca.std().item():.4f}")
        print(f"  Ca min: {Ca.min().item():.4f}, max: {Ca.max().item():.4f}")

        # Sample Ca at target positions
        print("\n  Ca values at target feature positions:")
        for target_label in [1, 2, 3]:  # eye, nose, mouth
            target_mask = target_features == target_label
            if target_mask.sum() > 0:
                ca_at_targets = Ca[target_mask]
                print(f"    {feature_names[target_label]}: mean={ca_at_targets.mean().item():.4f}, range=[{ca_at_targets.min().item():.4f}, {ca_at_targets.max().item():.4f}]")

        # Check bio_gate with default parameters
        ca_pct_default = 0.45  # Default from refinedFacialGRN.py
        ca_sens_default = 0.04
        Ca_threshold = torch.quantile(Ca, ca_pct_default)
        bio_gate_default = torch.sigmoid((Ca_threshold - Ca) / ca_sens_default)
        print(f"\n  With default params (pct=0.45, sens=0.04):")
        print(f"    Ca_threshold: {Ca_threshold.item():.4f}")
        print(f"    bio_gate mean: {bio_gate_default.mean().item():.4f}, range=[{bio_gate_default.min().item():.4f}, {bio_gate_default.max().item():.4f}]")

        # Check if bio_gate varies spatially
        bone_mask = target_features == 0
        if bone_mask.sum() > 0:
            gate_at_bone = bio_gate_default[bone_mask].mean().item()
            gate_at_features = bio_gate_default[~bone_mask].mean().item()
            print(f"    bio_gate at bone targets: {gate_at_bone:.4f}")
            print(f"    bio_gate at feature targets: {gate_at_features:.4f}")
            print(f"    Difference: {abs(gate_at_bone - gate_at_features):.4f}")

        print("=" * 70 + "\n")

    for iter_idx in range(num_learn_iters):
        # Run simulation with current parameters (sigmoid automatically constrains)
        predicted_features, loss = run_simulation(params, stig_model, transduction, target_features, device, dtype, vmem_grid=vmem_grid, grn_only_mode=grn_only)

        current_loss = loss.item()

        # Track best parameters
        if current_loss < best_loss:
            best_loss = current_loss
            best_loss_history.append((iter_idx, best_loss))

            # Save best RAW parameters (they will be transformed via sigmoid when loaded)
            best_param_bounds = {}
            for param_name in learned_parameter_names_filtered:
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
                'learned_parameter_names': learned_parameter_names_filtered,
                'grid_size': grid_size,
                'fixed_grn_params': fixed_grn_params,  # Record whether GRN was fixed
            }
            torch.save(save_data, f'./data/bestLearnedFacialParams_{file_number}.dat')

        # Backpropagation
        loss.backward(retain_graph=True)

        # # Diagnostic: Check gradients (only first iteration)
        # if verbose and iter_idx == 0:
        #     print("\n  Gradient diagnostics (iteration 0):")
        #     for param_name in learned_parameter_names_filtered:
        #         raw_name = f'{param_name}_raw'
        #         if raw_name in params:
        #             grad = params[raw_name].grad
        #             if grad is not None:
        #                 print(f"    {param_name}_raw.grad: {grad.item():.6e}")
        #             else:
        #                 print(f"    {param_name}_raw.grad: None")

        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if verbose and ((iter_idx + 1) % 1 == 0 or iter_idx == 0):
            # Per-class accuracy for diagnostics
            feature_accuracy = {}
            for label in range(4):
                mask = target_features == label
                if mask.sum() > 0:
                    correct = ((predicted_features == label) & mask).sum().item()
                    total = mask.sum().item()
                    feature_accuracy[feature_names[label]] = correct / total

            acc_str = ", ".join([f"{k}:{v:.2f}" for k, v in feature_accuracy.items()])
            print(f"Iter {iter_idx+1:3d}/{num_learn_iters}: loss={current_loss:.6f}, best={best_loss:.6f} | acc=[{acc_str}]")

            # Print current CONSTRAINED parameter values (apply sigmoid)
            if (iter_idx + 1) % 1 == 0 or iter_idx == 0:
                print("  Current parameters:")
                for param_name in learned_parameter_names_filtered:
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
    for param_name in learned_parameter_names_filtered:
        raw_name = f'{param_name}_raw'
        if raw_name in params and param_name in best_params:
            params[raw_name].data = best_params[param_name]

    predicted_features, final_loss = run_simulation(params, stig_model, transduction, target_features, device, dtype, vmem_grid=vmem_grid, grn_only_mode=grn_only)

    print(f"\nBest loss: {best_loss:.6f}")
    print("\nBest parameters (constrained values):")
    for param_name in learned_parameter_names_filtered:
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
