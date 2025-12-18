#!/usr/bin/env python3
"""
Learn CaMKII Bistability Parameters (Mechanism 1: Competitive Dynamics)

Implements autophosphorylation with competitive dynamics for CaMKII bistability.
Optimizes parameters to maximize pattern retention when Vmem decays.
Uses Rprop optimizer with sigmoid parameterization.

Architecture:
1. Vmem → Ca²⁺ (voltage-gated channels with tau_ca for temporal integration)
2. Ca²⁺ → external drive (sigmoid activation, range [0,1])
3. CaMKII → self-activation (competitive dynamics mapping to [-1, 1])
4. OR gate: additive threshold combining gain_ca * ca_signal + self_activation
   - Learnable gain_ca allows Ca²⁺ to overcome initial inhibition
5. CaMKII bistable dynamics: activation drives k_on, competitive self_activation modulates state

Learnable parameters (13 total):
- Ca²⁺ dynamics: tau_ca, g_ca, V_half_ca, k_ca
- CaMKII external drive: ca_threshold, ca_sensitivity
- CaMKII bistability: k_on, k_off, K_half, tau_camkii
- OR gate: or_threshold, or_sharpness, gain_ca
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy

from embryo import model


# ============================================================
# Define target face pattern (adapted from learnRefinedFacialIntegration.py)
# ============================================================
def define_target_face_pattern(grid_size=11):
    """
    Define target face pattern using bioelectric_fine mode from learnRefinedFacialIntegration.py.

    Returns binary masks for each facial feature.

    Returns:
        dict with 'eye', 'nose', 'mouth' masks (True = feature should be present)
    """
    eye_mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    nose_mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    mouth_mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)

    if grid_size == 11:
        # Eye indices: Two tiny square patches (2x2)
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

        for (row, col) in left_eye_indices + right_eye_indices:
            eye_mask[row, col] = True

        # Nose indices: Minimal vertical stripe at rows 3-5, col 5 (3 cells)
        nose_indices = [
            (3, 5),
            (4, 5),
            (5, 5),
        ]

        for (row, col) in nose_indices:
            nose_mask[row, col] = True

        # Mouth indices: Narrow horizontal stripe at row 8, cols 3-7 (5 cells)
        mouth_indices = [
            (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
        ]

        for (row, col) in mouth_indices:
            mouth_mask[row, col] = True

    else:
        raise ValueError(f"Target face pattern not defined for grid_size={grid_size}")

    return {
        'eye': eye_mask,
        'nose': nose_mask,
        'mouth': mouth_mask
    }


# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--gridSize', type=int, default=11)
parser.add_argument('--numBioSteps', type=int, default=1000)
parser.add_argument('--numTotalSteps', type=int, default=2000)
parser.add_argument('--numLearnIters', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--stigmergicParamsPath', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--fileNumber', type=int, default=0)
parser.add_argument('--verbose', type=str, default='True')

args = parser.parse_args()

grid_size = args.gridSize
num_bio_steps = args.numBioSteps
num_total_steps = args.numTotalSteps
num_learn_iters = args.numLearnIters
lr = args.lr
stigmergic_params_path = args.stigmergicParamsPath
file_number = args.fileNumber
verbose = args.verbose.lower() == 'true'


# ============================================================
# CaMKII model with learnable parameters (Mechanism 1: Competitive Dynamics)
# ============================================================
class LearnableCaMKII:
    """
    CaMKII bistable switch with Mechanism 1: Autophosphorylation with Competitive Dynamics.

    Architecture:
    1. Vmem → Ca²⁺ (with tau_ca for temporal integration/delay)
    2. Ca²⁺ → external drive (sigmoid activation, range [0,1])
    3. CaMKII → self-activation (competitive dynamics [-1, 1])
       - Positive: self_excitation = ReLU(self_activation) → contributes to OR gate
       - Negative: inhibition through enhanced k_off
    4. OR gate: activation from ca_signal OR self_excitation (additive threshold)
    5. CaMKII bistable dynamics with competitive inhibition

    Uses parameter overrides to allow external learning without modifying internal state.
    """

    def __init__(self, grid_size, device='cpu', dtype=torch.float32):
        self.grid_size = grid_size
        self.device = device
        self.dtype = dtype

        # CaMKII state (0 = inactive, 1 = active)
        self.CaMKII_active = None

        # Ca²⁺ state
        self.Ca = None

        # Default parameters (can be overridden)
        # --- Ca²⁺ dynamics (temporal integration) ---
        self.tau_ca = 20.0           # Ca²⁺ decay time constant (CRITICAL for delayed patterning)
        self.V_half_ca = -0.04       # Voltage for half-maximal Ca²⁺ activation
        self.k_ca = 0.01             # Voltage sensitivity of Ca²⁺ channels
        self.g_ca = 1.0              # Ca²⁺ channel conductance
        self.E_ca = 0.13             # Ca²⁺ reversal potential
        self.k_decay_ca = 0.0        # Additional Ca²⁺ decay rate (allows faster decay than 1/tau_ca)

        # --- CaMKII bistable dynamics ---
        self.k_on = 1.0              # Activation rate
        self.k_off = 0.01            # Inactivation rate
        self.ca_threshold = 0.3      # Ca²⁺ threshold for external drive
        self.ca_sensitivity = 0.1    # Sharpness of Ca²⁺ activation
        self.K_half = 0.5            # Bistability threshold for self-activation
        self.tau_camkii = 100.0      # CaMKII time constant (slows dynamics relative to Ca²⁺)

        # --- OR gate parameters (additive threshold) ---
        self.or_threshold = 0.5      # Threshold for combined activation
        self.or_sharpness = 10.0     # Sharpness of OR gate (higher = more OR-like)
        self.gain_ca = 2.0           # Gain on Ca²⁺ signal (allows boosting to overcome inhibition)

        # Parameter overrides (set by learner)
        self.tau_ca_override = None
        self.V_half_ca_override = None
        self.k_ca_override = None
        self.g_ca_override = None
        self.k_decay_ca_override = None
        self.k_on_override = None
        self.k_off_override = None
        self.ca_threshold_override = None
        self.ca_sensitivity_override = None
        self.K_half_override = None
        self.tau_camkii_override = None
        self.or_threshold_override = None
        self.or_sharpness_override = None
        self.gain_ca_override = None

    def reset(self):
        """Reset state to initial conditions (background state near 0)"""
        # Initialize CaMKII near 0 (background/inhibited state)
        self.CaMKII_active = torch.rand(self.grid_size, self.grid_size,
                                        device=self.device, dtype=self.dtype) * 0.01
        self.Ca = torch.zeros(self.grid_size, self.grid_size,
                             device=self.device, dtype=self.dtype)

    def update(self, vmem_grid, dt=0.01):
        """
        Update Ca²⁺ and CaMKII states with competitive dynamics.

        Args:
            vmem_grid: (grid_size, grid_size) membrane voltage in Volts
            dt: timestep

        Returns:
            dict with Ca²⁺, CaMKII, and activation components
        """
        # Get effective parameters (use overrides if available)
        tau_ca = self.tau_ca_override if self.tau_ca_override is not None else self.tau_ca
        V_half_ca = self.V_half_ca_override if self.V_half_ca_override is not None else self.V_half_ca
        k_ca = self.k_ca_override if self.k_ca_override is not None else self.k_ca
        g_ca = self.g_ca_override if self.g_ca_override is not None else self.g_ca
        k_decay_ca = self.k_decay_ca_override if self.k_decay_ca_override is not None else self.k_decay_ca
        k_on = self.k_on_override if self.k_on_override is not None else self.k_on
        k_off = self.k_off_override if self.k_off_override is not None else self.k_off
        ca_threshold = self.ca_threshold_override if self.ca_threshold_override is not None else self.ca_threshold
        ca_sensitivity = self.ca_sensitivity_override if self.ca_sensitivity_override is not None else self.ca_sensitivity
        K_half = self.K_half_override if self.K_half_override is not None else self.K_half
        tau_camkii = self.tau_camkii_override if self.tau_camkii_override is not None else self.tau_camkii
        or_threshold = self.or_threshold_override if self.or_threshold_override is not None else self.or_threshold
        or_sharpness = self.or_sharpness_override if self.or_sharpness_override is not None else self.or_sharpness
        gain_ca = self.gain_ca_override if self.gain_ca_override is not None else self.gain_ca

        # 1. Ca²⁺ dynamics (voltage-gated channels with temporal integration + additional decay)
        ca_activation = torch.sigmoid((vmem_grid - V_half_ca) / k_ca)
        driving_force = self.E_ca - vmem_grid
        I_ca = g_ca * ca_activation * (driving_force / 0.1)
        I_ca = torch.clamp(I_ca, min=0.0)

        # Ca²⁺ decay: baseline (1/tau_ca) + additional learnable decay (k_decay_ca)
        # This allows Ca²⁺ to decay faster than passive diffusion when needed
        dCa_dt = I_ca - (1.0 / tau_ca) * self.Ca - k_decay_ca
        self.Ca = self.Ca + dt * dCa_dt
        self.Ca = torch.clamp(self.Ca, min=0.0, max=10.0)

        # 2. External drive: Ca²⁺ activation (range: [0, 1])
        ca_signal = torch.sigmoid((self.Ca - ca_threshold) / ca_sensitivity)

        # 3. Internal drive: Competitive self-activation (range: [-1, 1])
        # Active subunits promote activity (+), inactive subunits inhibit (-)
        CaMKII_sq = self.CaMKII_active * self.CaMKII_active
        K_half_sq = K_half * K_half

        # Competitive dynamics: (active - inactive) / (active + inactive)
        # Maps: CaMKII=0 → -1 (inhibited), CaMKII=K_half → 0 (unstable), CaMKII=1 → +1 (excited)
        self_activation = (CaMKII_sq - K_half_sq) / (K_half_sq + CaMKII_sq)

        # 4. OR gate with learnable Ca²⁺ gain
        # Gain allows Ca²⁺ signal to overcome initial inhibition (self_activation = -1)
        # Both drivers contribute additively, gradients flow through both branches
        combined_signal = gain_ca * ca_signal + self_activation - or_threshold

        # Use softplus instead of ReLU to maintain gradient flow even when saturated
        # softplus(x) ≈ x when x >> 0, ≈ 0 when x << 0, but always has non-zero gradient
        activation = torch.nn.functional.softplus(combined_signal * or_sharpness, beta=1.0) / or_sharpness

        # 5. Update CaMKII with bistable dynamics
        dCaMKII_dt = (k_on * activation - k_off * self.CaMKII_active) / tau_camkii

        self.CaMKII_active = self.CaMKII_active + dt * dCaMKII_dt
        self.CaMKII_active = torch.clamp(self.CaMKII_active, min=0.0, max=1.0)

        return {
            'Ca': self.Ca,
            'CaMKII': self.CaMKII_active,
            'ca_signal': ca_signal,
            'self_activation': self_activation,
            'combined_signal': combined_signal,
            'activation': activation
        }


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
# Initialize learnable parameters
# ============================================================
def initialize_parameters(dtype=torch.float32):
    """
    Initialize learnable parameters using sigmoid parameterization with random initialization.

    Learnable parameters (Mechanism 1: Competitive Dynamics):
    - Ca²⁺ dynamics: tau_ca, g_ca, V_half_ca, k_ca
    - CaMKII external drive: ca_threshold, ca_sensitivity
    - CaMKII bistability: k_on, k_off, K_half, tau_camkii
    - OR gate: or_threshold, or_sharpness

    Returns:
        params dict with raw (learnable) parameters and bounds
    """
    params = {}

    # --- Ca²⁺ dynamics (temporal integration) ---
    # tau_ca: Ca²⁺ decay time constant (range: 10.0 to 100.0) - CRITICAL for delayed patterning
    min_val, max_val = 2.0, 5.0
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['tau_ca_raw'] = raw_param.clone().requires_grad_(True)
    params['tau_ca_min'] = min_val
    params['tau_ca_max'] = max_val

    # g_ca: Ca²⁺ channel conductance (range: 0.1 to 20.0)
    min_val, max_val = 0.1, 20.0
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['g_ca_raw'] = raw_param.clone().requires_grad_(True)
    params['g_ca_min'] = min_val
    params['g_ca_max'] = max_val

    # V_half_ca: Voltage for half-maximal Ca²⁺ activation (range: -0.08 to -0.01 V)
    min_val, max_val = -0.08, -0.01
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['V_half_ca_raw'] = raw_param.clone().requires_grad_(True)
    params['V_half_ca_min'] = min_val
    params['V_half_ca_max'] = max_val

    # k_ca: Voltage sensitivity of Ca²⁺ channels (range: 0.001 to 0.05 V)
    min_val, max_val = 0.001, 0.05
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['k_ca_raw'] = raw_param.clone().requires_grad_(True)
    params['k_ca_min'] = min_val
    params['k_ca_max'] = max_val

    # k_decay_ca: Additional Ca²⁺ decay rate (range: 0.0 to 1.0)
    # Allows Ca²⁺ to decay to 0 after Vmem stimulus ends
    # Total decay rate = 1/tau_ca + k_decay_ca
    min_val, max_val = 0.0, 5.0
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)  # Start with moderate additional decay
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['k_decay_ca_raw'] = raw_param.clone().requires_grad_(True)
    params['k_decay_ca_min'] = min_val
    params['k_decay_ca_max'] = max_val

    # --- CaMKII external drive (Ca²⁺ → activation) ---
    # ca_threshold: Ca²⁺ threshold for external drive (range: 0.01 to 10.0)
    min_val, max_val = 0.01, 10.0
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['ca_threshold_raw'] = raw_param.clone().requires_grad_(True)
    params['ca_threshold_min'] = min_val
    params['ca_threshold_max'] = max_val

    # ca_sensitivity: Sharpness of Ca²⁺ activation sigmoid (range: 0.01 to 2.0)
    min_val, max_val = 0.01, 2.0
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['ca_sensitivity_raw'] = raw_param.clone().requires_grad_(True)
    params['ca_sensitivity_min'] = min_val
    params['ca_sensitivity_max'] = max_val

    # --- CaMKII bistable dynamics ---
    # k_on: CaMKII activation rate (range: 0.5 to 5.0)
    # Tighter range to prevent runaway activation
    min_val, max_val = 0.5, 5.0
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)  # 1.0: Moderate activation rate
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['k_on_raw'] = raw_param.clone().requires_grad_(True)
    params['k_on_min'] = min_val
    params['k_on_max'] = max_val

    # k_off: CaMKII inactivation rate (range: 0.001 to 1.0) - CRITICAL for bistability persistence
    min_val, max_val = 0.001, 1.0
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['k_off_raw'] = raw_param.clone().requires_grad_(True)
    params['k_off_min'] = min_val
    params['k_off_max'] = max_val

    # K_half: Bistability threshold for competitive dynamics (range: 0.2 to 0.8)
    min_val, max_val = 0.2, 0.8
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['K_half_raw'] = raw_param.clone().requires_grad_(True)
    params['K_half_min'] = min_val
    params['K_half_max'] = max_val

    # tau_camkii: CaMKII time constant (range: 50.0 to 500.0) - slows dynamics for stability
    min_val, max_val = 10.0, 100.0
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['tau_camkii_raw'] = raw_param.clone().requires_grad_(True)
    params['tau_camkii_min'] = min_val
    params['tau_camkii_max'] = max_val

    # --- OR gate parameters (additive threshold) ---
    # or_threshold: Threshold for combined activation
    # With gain_ca ∈ [1.5,3], ca_signal ∈ [0,1], self_activation ∈ [-1,1]:
    # combined_signal ≈ [1.5*0 + (-1), 3*1 + 1] = [-1, 4]
    # Tighter range centered around expected operating point
    min_val, max_val = 0.2, 1.5
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)  # 0.5: Moderate threshold
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['or_threshold_raw'] = raw_param.clone().requires_grad_(True)
    params['or_threshold_min'] = min_val
    params['or_threshold_max'] = max_val

    # or_sharpness: Sharpness of OR gate (range: 1.0 to 20.0)
    # With softplus, we don't need extreme sharpness values
    min_val, max_val = 1.0, 20.0
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)  # 5.0: Moderate sharpness
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['or_sharpness_raw'] = raw_param.clone().requires_grad_(True)
    params['or_sharpness_min'] = min_val
    params['or_sharpness_max'] = max_val

    # gain_ca: Gain on Ca²⁺ signal in OR gate (range: 1.5 to 3.0)
    # Allows Ca²⁺ to overcome initial competitive inhibition (self_activation = -1)
    # Tighter range to prevent extreme saturation
    min_val, max_val = 1.5, 3.0
    initial_val = min_val + torch.rand(1, dtype=dtype).item() * (max_val - min_val)  # 2.0: Known reasonable starting point
    raw_param = inverse_sigmoid(torch.tensor(initial_val, dtype=dtype), min_val, max_val)
    params['gain_ca_raw'] = raw_param.clone().requires_grad_(True)
    params['gain_ca_min'] = min_val
    params['gain_ca_max'] = max_val

    return params


# ============================================================
# Run simulation with current parameters
# ============================================================
def run_simulation(params, bio_model, vmem_final, initial_vmem_grid, device, dtype):
    """
    Run CaMKII simulation with current parameters.

    Args:
        params: Parameter dictionary with raw learnable params
        bio_model: Bioelectric model (not used, kept for consistency)
        vmem_final: Final Vmem pattern at t=1000
        initial_vmem_grid: Initial uniform Vmem
        device: Torch device
        dtype: Torch dtype

    Returns:
        camkii_t1000: CaMKII pattern at t=1000
        camkii_t2000: CaMKII pattern at t=2000
        correlation: Spatial correlation between the two
    """
    # Extract constrained parameter values
    # --- Ca²⁺ dynamics ---
    tau_ca = apply_sigmoid_constraint(
        params['tau_ca_raw'],
        params['tau_ca_min'],
        params['tau_ca_max']
    ).to(device)

    g_ca = apply_sigmoid_constraint(
        params['g_ca_raw'],
        params['g_ca_min'],
        params['g_ca_max']
    ).to(device)

    V_half_ca = apply_sigmoid_constraint(
        params['V_half_ca_raw'],
        params['V_half_ca_min'],
        params['V_half_ca_max']
    ).to(device)

    k_ca = apply_sigmoid_constraint(
        params['k_ca_raw'],
        params['k_ca_min'],
        params['k_ca_max']
    ).to(device)

    k_decay_ca = apply_sigmoid_constraint(
        params['k_decay_ca_raw'],
        params['k_decay_ca_min'],
        params['k_decay_ca_max']
    ).to(device)

    # --- CaMKII external drive ---
    ca_threshold = apply_sigmoid_constraint(
        params['ca_threshold_raw'],
        params['ca_threshold_min'],
        params['ca_threshold_max']
    ).to(device)

    ca_sensitivity = apply_sigmoid_constraint(
        params['ca_sensitivity_raw'],
        params['ca_sensitivity_min'],
        params['ca_sensitivity_max']
    ).to(device)

    # --- CaMKII bistability ---
    k_on = apply_sigmoid_constraint(
        params['k_on_raw'],
        params['k_on_min'],
        params['k_on_max']
    ).to(device)

    k_off = apply_sigmoid_constraint(
        params['k_off_raw'],
        params['k_off_min'],
        params['k_off_max']
    ).to(device)

    K_half = apply_sigmoid_constraint(
        params['K_half_raw'],
        params['K_half_min'],
        params['K_half_max']
    ).to(device)

    tau_camkii = apply_sigmoid_constraint(
        params['tau_camkii_raw'],
        params['tau_camkii_min'],
        params['tau_camkii_max']
    ).to(device)

    # --- OR gate ---
    or_threshold = apply_sigmoid_constraint(
        params['or_threshold_raw'],
        params['or_threshold_min'],
        params['or_threshold_max']
    ).to(device)

    or_sharpness = apply_sigmoid_constraint(
        params['or_sharpness_raw'],
        params['or_sharpness_min'],
        params['or_sharpness_max']
    ).to(device)

    gain_ca = apply_sigmoid_constraint(
        params['gain_ca_raw'],
        params['gain_ca_min'],
        params['gain_ca_max']
    ).to(device)

    # Create CaMKII tracker with parameter overrides
    camkii = LearnableCaMKII(grid_size=grid_size, device=device, dtype=dtype)
    camkii.reset()

    # Set parameter overrides
    camkii.tau_ca_override = tau_ca
    camkii.g_ca_override = g_ca
    camkii.V_half_ca_override = V_half_ca
    camkii.k_ca_override = k_ca
    camkii.k_decay_ca_override = k_decay_ca
    camkii.ca_threshold_override = ca_threshold
    camkii.ca_sensitivity_override = ca_sensitivity
    camkii.k_on_override = k_on
    camkii.k_off_override = k_off
    camkii.K_half_override = K_half
    camkii.tau_camkii_override = tau_camkii
    camkii.or_threshold_override = or_threshold
    camkii.or_sharpness_override = or_sharpness
    camkii.gain_ca_override = gain_ca

    # Run simulation
    dt = 0.01
    camkii_t1000 = None

    # Diagnostic tracking
    diagnostics = {
        'ca_signal_t1000': None,
        'self_activation_t1000': None,
        'combined_signal_t1000': None,
        'activation_t1000': None,
        'ca_signal_t1500': None,
        'self_activation_t1500': None,
        'combined_signal_t1500': None,
        'activation_t1500': None,
        'ca_signal_t2000': None,
        'self_activation_t2000': None,
        'combined_signal_t2000': None,
        'activation_t2000': None,
    }

    for t in range(num_total_steps):
        # Interpolate Vmem from initial to final (0 to num_bio_steps)
        # Then decay back to initial (num_bio_steps to num_total_steps)
        if t < num_bio_steps:
            alpha = t / num_bio_steps
            vmem_grid = (1 - alpha) * initial_vmem_grid + alpha * vmem_final
        else:
            decay_progress = (t - num_bio_steps) / (num_total_steps - num_bio_steps)
            vmem_grid = (1 - decay_progress) * vmem_final + decay_progress * initial_vmem_grid

        # Update CaMKII and get diagnostic info
        state = camkii.update(vmem_grid, dt=dt)

        # Capture patterns at key timepoints (multiply by 1.0 to create new tensor with grad)
        if t == num_bio_steps - 1:
            camkii_t1000 = camkii.CaMKII_active * 1.0  # Create new tensor, keep gradient
            diagnostics['ca_signal_t1000'] = state['ca_signal'].clone()
            diagnostics['self_activation_t1000'] = state['self_activation'].clone()
            diagnostics['combined_signal_t1000'] = state['combined_signal'].clone()
            diagnostics['activation_t1000'] = state['activation'].clone()

        # Mid-decay point
        if t == (num_bio_steps + num_total_steps) // 2:
            diagnostics['ca_signal_t1500'] = state['ca_signal'].clone()
            diagnostics['self_activation_t1500'] = state['self_activation'].clone()
            diagnostics['combined_signal_t1500'] = state['combined_signal'].clone()
            diagnostics['activation_t1500'] = state['activation'].clone()

    camkii_t2000 = camkii.CaMKII_active * 1.0  # Create new tensor, keep gradient

    # Capture final state
    final_state = camkii.update(vmem_grid, dt=0)  # No update, just get state
    diagnostics['ca_signal_t2000'] = final_state['ca_signal'].clone()
    diagnostics['self_activation_t2000'] = final_state['self_activation'].clone()
    diagnostics['combined_signal_t2000'] = final_state['combined_signal'].clone()
    diagnostics['activation_t2000'] = final_state['activation'].clone()

    # Compute spatial correlation (differentiable)
    correlation = compute_spatial_correlation(camkii_t1000, camkii_t2000)

    return camkii_t1000, camkii_t2000, correlation, diagnostics


def compute_spatial_correlation(pattern1, pattern2):
    """
    Compute spatial correlation between two patterns.

    Returns Pearson correlation coefficient.
    """
    p1_flat = pattern1.flatten()
    p2_flat = pattern2.flatten()

    p1_mean = p1_flat.mean()
    p2_mean = p2_flat.mean()

    numerator = ((p1_flat - p1_mean) * (p2_flat - p2_mean)).sum()
    denominator = torch.sqrt(((p1_flat - p1_mean) ** 2).sum() *
                            ((p2_flat - p2_mean) ** 2).sum())

    if denominator < 1e-10:
        return torch.tensor(0.0, device=pattern1.device, dtype=pattern1.dtype)

    return numerator / denominator


# ============================================================
# Main learning loop
# ============================================================
def main():
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'Mac GPU (MPS)'
        dtype = torch.float32
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = 'CUDA GPU'
        dtype = torch.float32
    else:
        device = torch.device('cpu')
        device_name = 'CPU'
        dtype = torch.float32

    print("=" * 70)
    print("LEARNING CAMKII BISTABILITY PARAMETERS")
    print("=" * 70)
    print(f"Device: {device_name}")
    print(f"Data type: {dtype}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Bioelectric steps: {num_bio_steps}")
    print(f"Total steps (with decay): {num_total_steps}")
    print(f"Learning iterations: {num_learn_iters}")
    print(f"Learning rate: {lr}")
    print("=" * 70 + "\n")

    # Load and run stigmergic bioelectric model (once, fixed)
    print("Running Stigmergic bioelectric simulation...")
    stig_params = torch.load(stigmergic_params_path, weights_only=False)
    if "ATPParameters" not in stig_params:
        stig_params["ATPParameters"] = None

    num_samples = stig_params["simParameters"]["numSamples"]
    initial_values = copy.deepcopy(stig_params["simParameters"]["initialValues"])
    external_inputs = copy.deepcopy(stig_params["simParameters"]["externalInputs"])
    clamp_params = copy.deepcopy(stig_params["clampParameters"])

    bio_model = model(stig_params, numBasicSamples=num_samples)
    bio_model.setExperimentalConditions((initial_values, num_samples))

    # Get initial Vmem before simulation
    initial_vmem_grid = bio_model.electricNetwork.Vmem[0, :, 0].reshape(grid_size, grid_size).clone()

    # Run bioelectric simulation
    bio_model.simulate(
        externalInputs=external_inputs,
        clampParameters=clamp_params,
        perturbation=None,
        fieldModulation=False,
        numSimIters=num_bio_steps
    )

    # Get final Vmem pattern
    vmem_final = bio_model.electricNetwork.Vmem[0, :, 0].reshape(grid_size, grid_size).clone()
    print(f"✓ Stigmergic simulation complete")
    print(f"  Initial Vmem: mean={initial_vmem_grid.mean().item():.4f}V, std={initial_vmem_grid.std().item():.4f}V")
    print(f"  Final Vmem: mean={vmem_final.mean().item():.4f}V, std={vmem_final.std().item():.4f}V\n")

    # Move patterns to device
    vmem_final = vmem_final.to(device=device, dtype=dtype)
    initial_vmem_grid = initial_vmem_grid.to(device=device, dtype=dtype)

    # Define target face pattern
    print("Defining target face pattern...")
    target_masks = define_target_face_pattern(grid_size)
    target_masks = {k: v.to(device) for k, v in target_masks.items()}
    print(f"  Eye: {target_masks['eye'].sum().item()} cells")
    print(f"  Nose: {target_masks['nose'].sum().item()} cells")
    print(f"  Mouth: {target_masks['mouth'].sum().item()} cells\n")

    # Initialize learnable parameters
    params = initialize_parameters(dtype=dtype)

    # Print initial random values
    print("Initial random parameter values:")
    param_names_init = ['tau_ca', 'g_ca', 'V_half_ca', 'k_ca', 'k_decay_ca',
                        'ca_threshold', 'ca_sensitivity',
                        'k_on', 'k_off', 'K_half', 'tau_camkii',
                        'or_threshold', 'or_sharpness', 'gain_ca']
    for pname in param_names_init:
        init_val = apply_sigmoid_constraint(
            params[f'{pname}_raw'],
            params[f'{pname}_min'],
            params[f'{pname}_max']
        ).item()
        print(f"  {pname}: {init_val:.4f}")
    print()

    # Collect raw parameters for optimizer
    learned_params_list = [
        params['tau_ca_raw'],
        params['g_ca_raw'],
        params['V_half_ca_raw'],
        params['k_ca_raw'],
        params['k_decay_ca_raw'],
        params['ca_threshold_raw'],
        params['ca_sensitivity_raw'],
        params['k_on_raw'],
        params['k_off_raw'],
        params['K_half_raw'],
        params['tau_camkii_raw'],
        params['or_threshold_raw'],
        params['or_sharpness_raw'],
        params['gain_ca_raw']
    ]

    # Setup optimizer
    optimizer = torch.optim.Rprop(learned_params_list, lr=lr)

    # Learning loop
    best_correlation = -999.0
    best_loss = 999.0
    best_params = {}
    best_history = []

    print("=" * 70)
    print("STARTING LEARNING LOOP")
    print("=" * 70)
    print("Objective: CaMKII at t=2000 should match target face pattern")
    print("           (high at eye/nose/mouth, low elsewhere)\n")

    for iter_idx in range(num_learn_iters):
        # Run simulation
        camkii_t1000, camkii_t2000, correlation, diag = run_simulation(
            params, bio_model, vmem_final, initial_vmem_grid, device, dtype
        )

        # CONTRAST-BASED LOSS with spatial variance regularization
        # More robust than MSE - explicitly rewards spatial differentiation

        # Create feature mask
        all_features = target_masks['eye'] | target_masks['nose'] | target_masks['mouth']
        background = ~all_features

        # Track activation statistics for diagnostics
        eye_activation = camkii_t2000[target_masks['eye']].mean()
        nose_activation = camkii_t2000[target_masks['nose']].mean()
        mouth_activation = camkii_t2000[target_masks['mouth']].mean()
        feature_activation = camkii_t2000[all_features].mean()
        background_activation = camkii_t2000[background].mean()

        # === Loss Component 1: Push features toward 1.0 ===
        feature_loss = (1.0 - feature_activation) ** 2

        # === Loss Component 2: Push background toward 0.0 ===
        background_loss = background_activation ** 2

        # === Loss Component 3: Maximize contrast (feature - background → 1.0) ===
        contrast = feature_activation - background_activation
        contrast_loss = (1.0 - contrast) ** 2

        # === Loss Component 4: Spatial variance regularization ===
        # Penalize uniform patterns (saturation to all 0s or all 1s)
        spatial_variance = camkii_t2000.var()
        target_variance = 0.1  # Expect reasonable spatial variation
        variance_penalty = torch.relu(target_variance - spatial_variance)  # Only penalize if var < target

        # === Combined loss ===
        loss = feature_loss + background_loss + contrast_loss + 5.0 * variance_penalty

        # Track actual contrast for logging
        actual_contrast = torch.abs(feature_activation - background_activation)

        current_corr = correlation.item()
        current_loss = loss.item()

        # Track best parameters (based on lowest loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_correlation = current_corr
            best_history.append((iter_idx, best_loss))

            # Save best raw parameters
            best_params = {
                'tau_ca': params['tau_ca_raw'].detach().clone(),
                'g_ca': params['g_ca_raw'].detach().clone(),
                'V_half_ca': params['V_half_ca_raw'].detach().clone(),
                'k_ca': params['k_ca_raw'].detach().clone(),
                'k_decay_ca': params['k_decay_ca_raw'].detach().clone(),
                'ca_threshold': params['ca_threshold_raw'].detach().clone(),
                'ca_sensitivity': params['ca_sensitivity_raw'].detach().clone(),
                'k_on': params['k_on_raw'].detach().clone(),
                'k_off': params['k_off_raw'].detach().clone(),
                'K_half': params['K_half_raw'].detach().clone(),
                'tau_camkii': params['tau_camkii_raw'].detach().clone(),
                'or_threshold': params['or_threshold_raw'].detach().clone(),
                'or_sharpness': params['or_sharpness_raw'].detach().clone(),
                'gain_ca': params['gain_ca_raw'].detach().clone(),
            }

            # Save with bounds
            save_data = {
                'parameters': best_params,
                'parameter_bounds': {
                    'tau_ca_min': params['tau_ca_min'],
                    'tau_ca_max': params['tau_ca_max'],
                    'g_ca_min': params['g_ca_min'],
                    'g_ca_max': params['g_ca_max'],
                    'V_half_ca_min': params['V_half_ca_min'],
                    'V_half_ca_max': params['V_half_ca_max'],
                    'k_ca_min': params['k_ca_min'],
                    'k_ca_max': params['k_ca_max'],
                    'k_decay_ca_min': params['k_decay_ca_min'],
                    'k_decay_ca_max': params['k_decay_ca_max'],
                    'ca_threshold_min': params['ca_threshold_min'],
                    'ca_threshold_max': params['ca_threshold_max'],
                    'ca_sensitivity_min': params['ca_sensitivity_min'],
                    'ca_sensitivity_max': params['ca_sensitivity_max'],
                    'k_on_min': params['k_on_min'],
                    'k_on_max': params['k_on_max'],
                    'k_off_min': params['k_off_min'],
                    'k_off_max': params['k_off_max'],
                    'K_half_min': params['K_half_min'],
                    'K_half_max': params['K_half_max'],
                    'tau_camkii_min': params['tau_camkii_min'],
                    'tau_camkii_max': params['tau_camkii_max'],
                    'or_threshold_min': params['or_threshold_min'],
                    'or_threshold_max': params['or_threshold_max'],
                    'or_sharpness_min': params['or_sharpness_min'],
                    'or_sharpness_max': params['or_sharpness_max'],
                    'gain_ca_min': params['gain_ca_min'],
                    'gain_ca_max': params['gain_ca_max'],
                },
                'correlation': best_correlation,
                'correlation_history': best_history,
                'grid_size': grid_size,
                'best_loss': best_loss,
                'best_iteration': iter_idx,
            }
            torch.save(save_data, f'./data/bestLearnedCaMKIIParams_{file_number}.dat')

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if verbose and ((iter_idx + 1) % 5 == 0 or iter_idx == 0):
            # Get constrained values for all parameters
            param_names_print = ['tau_ca', 'g_ca', 'V_half_ca', 'k_ca', 'k_decay_ca', 'ca_threshold', 'ca_sensitivity',
                                'k_on', 'k_off', 'K_half', 'tau_camkii', 'or_threshold', 'or_sharpness', 'gain_ca']
            param_vals = {}
            for pname in param_names_print:
                param_vals[pname] = apply_sigmoid_constraint(
                    params[f'{pname}_raw'],
                    params[f'{pname}_min'],
                    params[f'{pname}_max']
                ).item()

            print(f"\nIter {iter_idx+1:3d}/{num_learn_iters}: loss={current_loss:.4f}, best_loss={best_loss:.4f}, best_corr={best_correlation:.4f}")
            print(f"  Loss components: feat={feature_loss.item():.3f}, bg={background_loss.item():.3f}, contrast={contrast_loss.item():.3f}, var_pen={variance_penalty.item():.3f}")
            print(f"  Activations: eye={eye_activation.item():.3f}, nose={nose_activation.item():.3f}, mouth={mouth_activation.item():.3f}, bg={background_activation.item():.3f}")
            print(f"  Contrast: {actual_contrast.item():.3f}, Spatial variance: {spatial_variance.item():.4f}")
            print(f"  Ca: tau={param_vals['tau_ca']:.1f}, g={param_vals['g_ca']:.2f}, V_half={param_vals['V_half_ca']:.4f}, k={param_vals['k_ca']:.4f}, k_decay={param_vals['k_decay_ca']:.3f}")
            print(f"  CaMKII: k_on={param_vals['k_on']:.2f}, k_off={param_vals['k_off']:.4f}, K_half={param_vals['K_half']:.3f}, tau={param_vals['tau_camkii']:.1f}")
            print(f"  Gate: ca_thr={param_vals['ca_threshold']:.3f}, ca_sens={param_vals['ca_sensitivity']:.3f}, or_thr={param_vals['or_threshold']:.3f}, or_sharp={param_vals['or_sharpness']:.1f}, gain={param_vals['gain_ca']:.2f}")

            # DIAGNOSTIC: Print activation component statistics
            print(f"  === DIAGNOSTICS ===")
            print(f"  t=1000: ca=[{diag['ca_signal_t1000'].min():.2f},{diag['ca_signal_t1000'].max():.2f}], self_act=[{diag['self_activation_t1000'].min():.2f},{diag['self_activation_t1000'].max():.2f}], comb=[{diag['combined_signal_t1000'].min():.1f},{diag['combined_signal_t1000'].max():.1f}], act=[{diag['activation_t1000'].min():.2f},{diag['activation_t1000'].max():.2f}]")
            print(f"  t=1500: ca=[{diag['ca_signal_t1500'].min():.2f},{diag['ca_signal_t1500'].max():.2f}], self_act=[{diag['self_activation_t1500'].min():.2f},{diag['self_activation_t1500'].max():.2f}], comb=[{diag['combined_signal_t1500'].min():.1f},{diag['combined_signal_t1500'].max():.1f}], act=[{diag['activation_t1500'].min():.2f},{diag['activation_t1500'].max():.2f}]")
            print(f"  t=2000: ca=[{diag['ca_signal_t2000'].min():.2f},{diag['ca_signal_t2000'].max():.2f}], self_act=[{diag['self_activation_t2000'].min():.2f},{diag['self_activation_t2000'].max():.2f}], comb=[{diag['combined_signal_t2000'].min():.1f},{diag['combined_signal_t2000'].max():.1f}], act=[{diag['activation_t2000'].min():.2f},{diag['activation_t2000'].max():.2f}]")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION WITH BEST PARAMETERS")
    print("=" * 70)

    # List of all learnable parameter names
    param_names = ['tau_ca', 'g_ca', 'V_half_ca', 'k_ca', 'k_decay_ca',
                   'ca_threshold', 'ca_sensitivity',
                   'k_on', 'k_off', 'K_half', 'tau_camkii',
                   'or_threshold', 'or_sharpness', 'gain_ca']

    # Restore best parameters
    for param_name in param_names:
        if param_name in best_params:
            params[f'{param_name}_raw'].data = best_params[param_name]

    camkii_t1000, camkii_t2000, final_corr, _ = run_simulation(
        params, bio_model, vmem_final, initial_vmem_grid, device, dtype
    )

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Best correlation: {best_correlation:.4f}")
    print("\nBest parameters:")
    for param_name in param_names:
        constrained_val = apply_sigmoid_constraint(
            best_params[param_name],
            params[f'{param_name}_min'],
            params[f'{param_name}_max']
        )
        print(f"  {param_name}: {constrained_val.item():.4f}")

    # Visualize results
    visualize_results(vmem_final, camkii_t1000, camkii_t2000, file_number)

    print("\n" + "=" * 70)
    print("✅ LEARNING COMPLETE!")
    print("=" * 70)
    print(f"\nSaved best parameters to: ./data/bestLearnedCaMKIIParams_{file_number}.dat")
    print(f"Saved visualization to: learned_camkii_bistability_{file_number}.png")


def visualize_results(vmem_final, camkii_t1000, camkii_t2000, file_number):
    """Create visualization of learned CaMKII pattern locking"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Vmem at t=1000
    im0 = axes[0].imshow(vmem_final.cpu().numpy(), cmap='RdBu_r', vmin=-0.05, vmax=-0.01)
    axes[0].set_title('Vmem at t=1000\n(Face Pattern)', fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], label='Voltage (V)')

    # CaMKII at t=1000
    im1 = axes[1].imshow(camkii_t1000.detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('CaMKII at t=1000\n(Locked Pattern)', fontweight='bold')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], label='CaMKII Activity')

    # CaMKII at t=2000
    im2 = axes[2].imshow(camkii_t2000.detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('CaMKII at t=2000\n(Retained Pattern)', fontweight='bold')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im2, ax=axes[2], label='CaMKII Activity')

    plt.suptitle('Learned CaMKII Bistability: Pattern Locking',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'learned_camkii_bistability_{file_number}.png', dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()