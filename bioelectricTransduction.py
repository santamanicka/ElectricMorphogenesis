"""
Bioelectric Transduction Module
Converts voltage and gap junction currents into gene-regulatory signals via Ca²⁺ dynamics and metabolic state.

Two parallel channels:
1. Voltage → Ca²⁺ dynamics (temporal integration)
2. Gap junction currents → metabolic state (boundary detection)
"""

import torch
import torch.nn.functional as F


class BioelectricTransduction:
    """
    Transduces bioelectric signals (Vmem, gap junction currents) into gene-regulatory signals.

    Key features:
    - Temporal integration via Ca²⁺ dynamics (provides memory)
    - Separate voltage and current channels (biological realism)
    - No "detail" computation (replaced with actual biophysical currents)
    """

    def __init__(self, grid_size, device='cpu', dtype=torch.float64):
        """
        Args:
            grid_size: Size of spatial grid (assumes square grid)
            device: 'cpu' or 'cuda'
            dtype: torch data type
        """
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size
        self.device = device
        self.dtype = dtype

        # Channel A: Voltage-gated Ca²⁺ dynamics parameters
        self.V_half_ca = torch.tensor(-0.04, device=device, dtype=dtype)  # -40mV activation threshold
        self.k_ca = torch.tensor(0.01, device=device, dtype=dtype)         # 10mV voltage sensitivity
        self.g_ca = torch.tensor(1.0, device=device, dtype=dtype)          # Max Ca²⁺ conductance
        self.tau_ca = torch.tensor(1.0, device=device, dtype=dtype)        # Ca²⁺ decay timescale
        self.E_ca = torch.tensor(0.13, device=device, dtype=dtype)         # +130mV Ca²⁺ reversal potential

        # Channel B: Gap junction current → metabolic state
        self.beta_metabolic = torch.tensor(0.1, device=device, dtype=dtype)  # Current→metabolic cost factor
        self.metabolic_baseline = torch.tensor(1.0, device=device, dtype=dtype)  # Healthy baseline

        # Low-pass filtering for temporal smoothing
        self.alpha_lowpass = torch.tensor(0.8, device=device, dtype=dtype)  # Smoothing factor (0.8 = 20% new)

        # State variables (grid format)
        self.Ca = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)
        self.metabolic_state = torch.ones(grid_size, grid_size, device=device, dtype=dtype)
        self.voltage_history = None  # For low-pass filtered voltage

        # Diagnostics
        self.I_ca_history = []
        self.I_gj_magnitude_history = []

    def sigmoid(self, x):
        """Numerically stable sigmoid"""
        return torch.sigmoid(x)

    def update(self, vmem_grid, I_gj_grid=None, dt=0.01):
        """
        Update bioelectric transduction state.

        Args:
            vmem_grid: (grid_size, grid_size) - membrane voltage
            I_gj_grid: (grid_size, grid_size) - gap junction currents (optional)
            dt: timestep

        Returns:
            dict with keys:
                'Ca': (grid_size, grid_size) - intracellular Ca²⁺
                'metabolic': (grid_size, grid_size) - metabolic state
                'Ca_activation': Ca²⁺ channel activation (diagnostic)
                'I_gj_magnitude': Gap junction current magnitude (diagnostic)
        """
        # Low-pass filter voltage (temporal smoothing)
        if self.voltage_history is None:
            self.voltage_history = vmem_grid.clone()
        else:
            self.voltage_history = (self.alpha_lowpass * self.voltage_history +
                                   (1 - self.alpha_lowpass) * vmem_grid)

        # ===============================
        # Channel A: Voltage → Ca²⁺ Dynamics
        # ===============================

        # Voltage-gated Ca²⁺ channel activation (sigmoid)
        ca_activation = self.sigmoid((self.voltage_history - self.V_half_ca) / self.k_ca)

        # Ca²⁺ current (Goldman-Hodgkin-Katz-like, but simplified)
        # I_ca ∝ activation × driving_force
        driving_force = self.E_ca - self.voltage_history  # mV
        I_ca = self.g_ca * ca_activation * (driving_force / 0.1)  # Normalized by 100mV
        I_ca = torch.clamp(I_ca, min=0.0)  # Ca²⁺ only flows inward

        # Ca²⁺ dynamics with temporal integration
        # dCa/dt = I_ca - Ca/tau_ca (influx minus decay)
        dCa_dt = I_ca - self.Ca / self.tau_ca
        self.Ca = self.Ca + dt * dCa_dt
        self.Ca = torch.clamp(self.Ca, min=0.0, max=10.0)  # Physiological bounds

        # ===============================
        # Channel B: Gap Junction Current → Metabolic State
        # ===============================

        I_gj_magnitude = torch.zeros_like(vmem_grid)

        if I_gj_grid is not None:
            # Compute total current magnitude per cell
            # If I_gj_grid is per-neighbor, sum; if total, use directly
            if I_gj_grid.dim() == 3:  # (grid, grid, neighbors)
                I_gj_magnitude = I_gj_grid.abs().sum(dim=-1)
            else:  # (grid, grid)
                I_gj_magnitude = I_gj_grid.abs()

            # Normalize by maximum to get relative current
            max_current = I_gj_magnitude.max()
            if max_current > 1e-9:
                I_gj_magnitude_norm = I_gj_magnitude / (max_current + 1e-9)
            else:
                I_gj_magnitude_norm = I_gj_magnitude

            # Metabolic cost: high current → low metabolic state
            # (ATP consumption for pumping ions against gradients)
            metabolic_cost = self.beta_metabolic * I_gj_magnitude_norm
            self.metabolic_state = self.metabolic_baseline - metabolic_cost
            self.metabolic_state = torch.clamp(self.metabolic_state, min=0.0, max=1.0)

        # Record diagnostics
        self.I_ca_history.append(I_ca.mean().item())
        self.I_gj_magnitude_history.append(I_gj_magnitude.mean().item())

        return {
            'vmem': vmem_grid,  # Raw voltage for direct use
            'Ca': self.Ca,
            'metabolic': self.metabolic_state,
            'Ca_activation': ca_activation,  # Diagnostic
            'I_ca': I_ca,                     # Diagnostic
            'I_gj_magnitude': I_gj_magnitude  # Diagnostic
        }

    def get_gene_modulation_signals(self):
        """
        Get signals that modulate gene expression.

        Returns:
            dict with normalized signals ready for gene regulation
        """
        # Normalize Ca²⁺ to [0, 1] range
        ca_norm = self.Ca / (self.Ca.max() + 1e-9)

        return {
            'Ca': ca_norm,
            'metabolic': self.metabolic_state
        }

    def reset(self):
        """Reset state variables to initial conditions"""
        self.Ca = torch.zeros(self.grid_size, self.grid_size, device=self.device, dtype=self.dtype)
        self.metabolic_state = torch.ones(self.grid_size, self.grid_size, device=self.device, dtype=self.dtype)
        self.voltage_history = None
        self.I_ca_history = []
        self.I_gj_magnitude_history = []

    def get_diagnostics(self):
        """Get diagnostic information for debugging/visualization"""
        return {
            'Ca_mean_history': self.I_ca_history,
            'I_gj_mean_history': self.I_gj_magnitude_history,
            'current_Ca_max': self.Ca.max().item(),
            'current_Ca_mean': self.Ca.mean().item(),
            'current_metabolic_mean': self.metabolic_state.mean().item()
        }
