"""
Decentralized Reaction-Diffusion Bistable Stress System.

Each cell has a stress variable S governed by bistable RD dynamics with decay.
Local Ca2+ (derived from Vmem) acts as bifurcation parameter. Diffusion couples
neighboring cells. Decay + diffusion + bistability create a spatial frequency
filter for pattern-sensitive stress detection.

Biological interpretation:
- S represents a Ca2+-dependent bistable pathway (ROS-Ca2+ feedback,
  NFAT switch, or p38/JNK stress kinase cascade)
- Diffusion via gap junction-permeable molecules (IP3, H2O2)
- Decay via protein degradation / phosphatase activity
- Output: S drives pannexin-1 -> eATP release [Tung et al. 2024]

See FIELD_RESCUE_DESIGN.md Section 7 for full design rationale.
"""

import torch


class StressBistableSwitch:
    """
    Decentralized reaction-diffusion bistable stress system.

    Architecture:
        1. Vmem -> Ca2+ via voltage-gated channels (temporal integration)
        2. Ca2+ -> bifurcation parameter (sigmoid activation)
        3. S -> self-activation (competitive dynamics [-1, 1])
        4. OR gate: combines gain_S * ca_drive + self_activation
        5. Bistable RD dynamics: reaction + decay + diffusion

    The embryo-level stress signal is mean(S), corresponding to total eATP
    released by all cells into the shared extracellular medium.
    """

    def __init__(self, num_cells, adjacency_matrix, params=None,
                 device='cpu', dtype=torch.float32):
        """
        Args:
            num_cells: number of cells in the tissue grid
            adjacency_matrix: (num_cells, num_cells) lattice connectivity
            params: dict of parameters (or use defaults)
            device: torch device
            dtype: torch dtype
        """
        self.num_cells = num_cells
        self.device = device
        self.dtype = dtype

        # Adjacency matrix for discrete Laplacian
        self.A = adjacency_matrix.to(device=device, dtype=dtype)
        # Precompute degree (number of neighbors per cell)
        self.degree = self.A.sum(dim=1)

        # State variables
        self.S = torch.zeros(num_cells, device=device, dtype=dtype)
        self.Ca = torch.zeros(num_cells, device=device, dtype=dtype)

        # Default parameters (from FIELD_RESCUE_DESIGN.md Section 7.8)
        defaults = {
            # Ca2+ channel dynamics (reused from SimpleCaMKII architecture)
            'tau_ca': 3.0,
            'V_half_ca': -0.04,
            'k_ca': 0.01,
            'g_ca': 0.5,            # low conductance (Ca2+ peaks ~0.5-1.5)
            'k_decay_ca': 0.3,      # moderate baseline decay
            # Stress RD bistable dynamics
            'tau_S': 50.0,
            'k_on_S': 3.0,
            'k_off_S': 0.02,
            'K_S': 0.4,
            'Ca_stress_threshold': 0.8,  # between healthy/perturbed Ca2+ ranges
            'sigma_ca': 0.2,        # sharp Ca2+ sensitivity
            'gain_S': 2.0,          # moderate Ca2+ drive gain
            'or_threshold_S': 0.6,  # raise threshold for spatial selectivity
            'D_S': 0.15,            # diffusion for spatial filtering
            'gamma': 0.08,          # max decay rate (V_max in Michaelis-Menten)
            'K_decay': 0.3,         # half-saturation for decay (phosphatase Km)
        }

        if params is not None:
            for key in defaults:
                if key in params:
                    defaults[key] = params[key]

        # Store as tensors for differentiable computation
        self.tau_ca = self._to_tensor(defaults['tau_ca'])
        self.V_half_ca = self._to_tensor(defaults['V_half_ca'])
        self.k_ca = self._to_tensor(defaults['k_ca'])
        self.g_ca = self._to_tensor(defaults['g_ca'])
        self.k_decay_ca = self._to_tensor(defaults['k_decay_ca'])
        self.E_ca = self._to_tensor(0.13)  # +130mV reversal potential (fixed)

        self.tau_S = self._to_tensor(defaults['tau_S'])
        self.k_on_S = self._to_tensor(defaults['k_on_S'])
        self.k_off_S = self._to_tensor(defaults['k_off_S'])
        self.K_S = self._to_tensor(defaults['K_S'])
        self.Ca_stress_threshold = self._to_tensor(defaults['Ca_stress_threshold'])
        self.sigma_ca = self._to_tensor(defaults['sigma_ca'])
        self.gain_S = self._to_tensor(defaults['gain_S'])
        self.or_threshold_S = self._to_tensor(defaults['or_threshold_S'])
        self.D_S = self._to_tensor(defaults['D_S'])
        self.gamma = self._to_tensor(defaults['gamma'])
        self.K_decay = self._to_tensor(defaults['K_decay'])

    def _to_tensor(self, val):
        """Convert scalar to tensor on correct device/dtype."""
        if isinstance(val, torch.Tensor):
            return val.to(device=self.device, dtype=self.dtype)
        return torch.tensor(val, device=self.device, dtype=self.dtype)

    def set_params_from_tensors(self, **kwargs):
        """Set parameters from tensor values (for differentiable learning)."""
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def reset(self):
        """Reset stress and Ca2+ state to zeros."""
        self.S = torch.zeros(self.num_cells, device=self.device, dtype=self.dtype)
        self.Ca = torch.zeros(self.num_cells, device=self.device, dtype=self.dtype)

    def compute_ca_from_vmem(self, vmem_flat, dt):
        """
        Update Ca2+ from Vmem via voltage-gated channels.

        Same channel model as SimpleCaMKII in test_camkii_bistability.py:
            I_ca = g_ca * sigmoid((Vmem - V_half) / k_ca) * (E_ca - Vmem) / 0.1
            dCa/dt = I_ca - (1/tau_ca) * Ca - k_decay_ca

        Args:
            vmem_flat: (num_cells,) membrane voltage in Volts
            dt: timestep
        """
        ca_activation = torch.sigmoid((vmem_flat - self.V_half_ca) / self.k_ca)
        driving_force = self.E_ca - vmem_flat
        I_ca = self.g_ca * ca_activation * (driving_force / 0.1)
        I_ca = torch.clamp(I_ca, min=0.0)

        dCa_dt = I_ca - (1.0 / self.tau_ca) * self.Ca - self.k_decay_ca
        self.Ca = self.Ca + dt * dCa_dt
        self.Ca = torch.clamp(self.Ca, min=0.0, max=10.0)

    def step(self, dt, Ca=None):
        """
        Advance stress variable S by one timestep.

        Args:
            dt: timestep
            Ca: (num_cells,) local Ca2+ at each cell. If None, uses self.Ca.
        """
        if Ca is not None:
            ca = Ca
        else:
            ca = self.Ca

        # Ca2+ drive: sigmoid activation -> [0, 1]
        ca_drive = torch.sigmoid(
            (ca - self.Ca_stress_threshold) / self.sigma_ca
        )

        # Competitive self-activation: (S^2 - K^2) / (S^2 + K^2) -> [-1, +1]
        S_sq = self.S * self.S
        K_sq = self.K_S * self.K_S
        self_activation = (S_sq - K_sq) / (S_sq + K_sq + 1e-10)

        # OR gate: combines Ca2+ drive with self-activation
        or_input = self.gain_S * ca_drive + self_activation - self.or_threshold_S
        or_gate = torch.sigmoid(or_input)

        # Bistable reaction
        reaction = (self.k_on_S * or_gate * (1.0 - self.S)
                    - self.k_off_S * self.S) / self.tau_S

        # Decay: Michaelis-Menten (phosphatase kinetics)
        # Low S: unsaturated phosphatase → strong decay rate per unit S
        # High S: saturated phosphatase → constant decay rate
        decay = -self.gamma * self.S / (self.K_decay + self.S)

        # Diffusion: laplacian_i = sum_j A(i,j) * (S_j - S_i)
        laplacian = torch.matmul(self.A, self.S) - self.degree * self.S
        diffusion = self.D_S * laplacian

        # Update
        self.S = self.S + dt * (reaction + decay + diffusion)
        self.S = torch.clamp(self.S, 0.0, 1.0)

    def step_with_vmem(self, vmem_flat, dt):
        """
        Combined step: update Ca2+ from Vmem, then advance S.

        Args:
            vmem_flat: (num_cells,) membrane voltage in Volts
            dt: timestep
        """
        self.compute_ca_from_vmem(vmem_flat, dt)
        self.step(dt)

    def get_embryo_stress(self):
        """Returns embryo-level stress = mean(S) = total eATP proxy."""
        return self.S.mean()

    def get_state(self):
        """Return current state dict for diagnostics."""
        ca_drive = torch.sigmoid(
            (self.Ca - self.Ca_stress_threshold) / self.sigma_ca
        )
        S_sq = self.S * self.S
        K_sq = self.K_S * self.K_S
        self_activation = (S_sq - K_sq) / (S_sq + K_sq + 1e-10)

        return {
            'S': self.S.clone(),
            'Ca': self.Ca.clone(),
            'ca_drive': ca_drive,
            'self_activation': self_activation,
            'embryo_stress': self.get_embryo_stress(),
        }
