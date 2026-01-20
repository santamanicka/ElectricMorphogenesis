"""
Bioelectric Transduction Module
Converts voltage into gene-regulatory signals via Ca²⁺ dynamics.

Key mechanism:
- Voltage → Ca²⁺ dynamics (temporal integration, provides memory)
"""

import torch
import torch.nn.functional as F


class BioelectricTransduction:
    """
    Transduces bioelectric signals (Vmem) into gene-regulatory signals via Ca²⁺ dynamics.

    Key features:
    - Temporal integration via Ca²⁺ dynamics (provides memory)
    - Voltage-gated Ca²⁺ channels transduce Vmem patterns into biochemical signals
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

        # Voltage-gated Ca²⁺ dynamics parameters
        self.V_half_ca = torch.tensor(-0.04, device=device, dtype=dtype)  # -40mV activation threshold
        self.k_ca = torch.tensor(0.01, device=device, dtype=dtype)         # 10mV voltage sensitivity
        self.g_ca = torch.tensor(1.0, device=device, dtype=dtype)          # Max Ca²⁺ conductance
        self.tau_ca = torch.tensor(1.0, device=device, dtype=dtype)        # Ca²⁺ decay timescale (1s)
        self.E_ca = torch.tensor(0.13, device=device, dtype=dtype)         # +130mV Ca²⁺ reversal potential

        # Low-pass filtering for temporal smoothing
        self.alpha_lowpass = torch.tensor(0.8, device=device, dtype=dtype)  # Smoothing factor (0.8 = 20% new)

        # State variables (grid format)
        self.Ca = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)
        self.voltage_history = None  # For low-pass filtered voltage

        # Diagnostics
        self.I_ca_history = []

    def sigmoid(self, x):
        """Numerically stable sigmoid"""
        return torch.sigmoid(x)

    def update(self, vmem_grid, dt=0.01):
        """
        Update bioelectric transduction state.

        Args:
            vmem_grid: (grid_size, grid_size) - membrane voltage
            dt: timestep

        Returns:
            dict with keys:
                'Ca': (grid_size, grid_size) - intracellular Ca²⁺
                'Ca_activation': Ca²⁺ channel activation (diagnostic)
                'I_ca': Ca²⁺ current (diagnostic)
        """
        # Low-pass filter voltage (temporal smoothing)
        if self.voltage_history is None:
            self.voltage_history = vmem_grid.clone()
        else:
            self.voltage_history = (self.alpha_lowpass * self.voltage_history +
                                   (1 - self.alpha_lowpass) * vmem_grid)

        # ===============================
        # Voltage → Ca²⁺ Dynamics
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

        # Record diagnostics
        self.I_ca_history.append(I_ca.mean().item())

        return {
            'vmem': vmem_grid,  # Raw voltage for direct use
            'Ca': self.Ca,
            'Ca_activation': ca_activation,  # Diagnostic
            'I_ca': I_ca                      # Diagnostic
        }

    def get_gene_modulation_signals(self):
        """
        Get signals that modulate gene expression.

        Returns:
            dict with normalized Ca²⁺ signal ready for gene regulation
        """
        # Normalize Ca²⁺ to [0, 1] range
        ca_norm = self.Ca / (self.Ca.max() + 1e-9)

        return {
            'Ca': ca_norm
        }

    def reset(self):
        """Reset state variables to initial conditions"""
        self.Ca = torch.zeros(self.grid_size, self.grid_size, device=self.device, dtype=self.dtype)
        self.voltage_history = None
        self.I_ca_history = []

    def get_diagnostics(self):
        """Get diagnostic information for debugging/visualization"""
        return {
            'Ca_mean_history': self.I_ca_history,
            'current_Ca_max': self.Ca.max().item(),
            'current_Ca_mean': self.Ca.mean().item()
        }
