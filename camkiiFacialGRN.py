"""
CaMKII-Integrated Facial GRN with Concurrent Dynamics

Extends RefinedFacialGRN by integrating CaMKII bistability mechanism.
CaMKII runs concurrently with the GRN, providing bistable pattern memory
that persists even after bioelectric stimulus decays.

Key architecture changes from refinedFacialGRN.py:
1. CaMKII dynamics run alongside GRN (not pre-equilibrated Ca)
2. CaMKII activity (not raw Ca) gates gene expression
3. Bistable memory allows pattern persistence after Vmem decay
4. Three temporal phases:
   - Phase 1 (Stimulus): Vmem pattern drives Ca → CaMKII pattern formation
   - Phase 2 (Decay): Vmem decays, Ca decays, CaMKII locks in via bistability
   - Phase 3 (Maintenance): CaMKII pattern persists, continues to gate GRN

Signal flow:
    Vmem → Ca²⁺ → [OR gate] → CaMKII_active → bio_gate → GRN
                      ↑
           self_activation (competitive feedback)
"""

import torch
import torch.nn.functional as F


class CaMKIIBistableSwitch:
    """
    CaMKII bistable switch with competitive self-activation dynamics.

    Implements the learned bistability mechanism from learn_camkii_bistability.py:
    - Vmem → Ca²⁺ (voltage-gated channels with tau_ca decay)
    - Ca²⁺ → ca_signal (sigmoid activation, range [0,1])
    - CaMKII → self_activation (competitive dynamics [-1, +1])
    - OR gate: gain_ca * ca_signal + self_activation - or_threshold
    - CaMKII bistable dynamics with k_on, k_off

    The key insight is the "drag and drop" mechanism:
    - Ca²⁺ "drags" CaMKII across K_half threshold during stimulus phase
    - Once past K_half, competitive self-activation "drops" it into stable ON state
    - Pattern persists even after Ca²⁺ decays to uniform low level (~K_half)
    """

    def __init__(self, grid_size, device='cpu', dtype=torch.float64):
        self.grid_size = grid_size
        self.device = device
        self.dtype = dtype

        # State variables
        self.Ca = None
        self.CaMKII_active = None

        # Default parameters (can be overridden by learned values)
        # --- Ca²⁺ dynamics ---
        self.tau_ca = 3.5           # Ca²⁺ decay time constant
        self.g_ca = 10.0            # Ca²⁺ channel conductance
        self.V_half_ca = -0.04      # Voltage for half-maximal Ca²⁺ activation
        self.k_ca = 0.01            # Voltage sensitivity of Ca²⁺ channels
        self.E_ca = 0.13            # Ca²⁺ reversal potential (fixed)
        self.k_decay_ca = 2.5       # Constant baseline Ca²⁺ consumption

        # --- CaMKII external drive (Ca → activation) ---
        self.ca_threshold = 5.0     # Ca²⁺ threshold for activation
        self.ca_sensitivity = 1.0   # Sharpness of Ca²⁺ sigmoid

        # --- CaMKII bistable dynamics ---
        self.k_on = 2.5             # Activation rate
        self.k_off = 0.5            # Inactivation rate
        self.K_half = 0.5           # Bistability threshold
        self.tau_camkii = 50.0      # CaMKII time constant

        # --- OR gate parameters ---
        self.or_threshold = 0.8     # Threshold for combined activation
        self.or_sharpness = 10.0    # Sharpness of OR gate
        self.gain_ca = 2.25         # Gain on Ca²⁺ signal

    def reset(self):
        """Reset state to initial conditions (near inactive state)"""
        self.Ca = torch.zeros(self.grid_size, self.grid_size,
                              device=self.device, dtype=self.dtype)
        # Initialize CaMKII near 0 with small noise
        self.CaMKII_active = torch.rand(self.grid_size, self.grid_size,
                                        device=self.device, dtype=self.dtype) * 0.01

    def load_learned_parameters(self, param_dict):
        """
        Load learned parameters from a dictionary.

        Args:
            param_dict: Dictionary with parameter values (already constrained)
        """
        param_map = {
            'tau_ca': 'tau_ca',
            'g_ca': 'g_ca',
            'V_half_ca': 'V_half_ca',
            'k_ca': 'k_ca',
            'k_decay_ca': 'k_decay_ca',
            'ca_threshold': 'ca_threshold',
            'ca_sensitivity': 'ca_sensitivity',
            'k_on': 'k_on',
            'k_off': 'k_off',
            'K_half': 'K_half',
            'tau_camkii': 'tau_camkii',
            'or_threshold': 'or_threshold',
            'or_sharpness': 'or_sharpness',
            'gain_ca': 'gain_ca',
        }

        for param_name, attr_name in param_map.items():
            if param_name in param_dict:
                setattr(self, attr_name, param_dict[param_name])

    def update(self, vmem_grid, dt=0.01):
        """
        Update Ca²⁺ and CaMKII states.

        Args:
            vmem_grid: (grid_size, grid_size) membrane voltage in Volts
            dt: timestep

        Returns:
            dict with state variables for diagnostics
        """
        # 1. Ca²⁺ dynamics (voltage-gated channels)
        ca_activation = torch.sigmoid((vmem_grid - self.V_half_ca) / self.k_ca)
        driving_force = self.E_ca - vmem_grid
        I_ca = self.g_ca * ca_activation * (driving_force / 0.1)
        I_ca = torch.clamp(I_ca, min=0.0)

        # Ca²⁺ decay: passive (1/tau_ca) + constant consumption (k_decay_ca)
        dCa_dt = I_ca - (1.0 / self.tau_ca) * self.Ca - self.k_decay_ca
        self.Ca = self.Ca + dt * dCa_dt
        self.Ca = torch.clamp(self.Ca, min=0.0, max=10.0)

        # 2. External drive: Ca²⁺ activation (range: [0, 1])
        ca_signal = torch.sigmoid((self.Ca - self.ca_threshold) / self.ca_sensitivity)

        # 3. Internal drive: Competitive self-activation (range: [-1, 1])
        CaMKII_sq = self.CaMKII_active * self.CaMKII_active
        K_half_sq = self.K_half * self.K_half
        self_activation = (CaMKII_sq - K_half_sq) / (K_half_sq + CaMKII_sq + 1e-10)

        # 4. OR gate with learnable Ca²⁺ gain
        combined_signal = self.gain_ca * ca_signal + self_activation - self.or_threshold

        # Use softplus for smooth gradient flow
        activation = torch.nn.functional.softplus(
            combined_signal * self.or_sharpness, beta=1.0
        ) / self.or_sharpness

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
            'activation': activation
        }

    def get_bio_gate(self, threshold_percentile=0.5, sensitivity=0.1):
        """
        Get bioelectric gate signal from CaMKII activity.

        Unlike raw Ca²⁺, CaMKII activity is bistable and persists.
        High CaMKII regions (features) → bio_gate ≈ 1.0
        Low CaMKII regions (background) → bio_gate ≈ 0.0

        Args:
            threshold_percentile: Percentile threshold for gate activation
            sensitivity: Sigmoid sensitivity

        Returns:
            bio_gate: (grid_size, grid_size) tensor in [0, 1]
        """
        # Use CaMKII activity threshold (not Ca)
        # High CaMKII → high gate (features)
        threshold = torch.quantile(self.CaMKII_active, threshold_percentile)
        bio_gate = torch.sigmoid((self.CaMKII_active - threshold) / sensitivity)
        return bio_gate


class CaMKIIFacialGRN:
    """
    Facial Gene Regulatory Network with concurrent CaMKII bistability.

    Combines:
    1. Morphogen gradients (SHH, FGF8, EDN1) - spatial patterning
    2. CaMKII bistable switch - bioelectric pattern memory
    3. Gene expression - feature classification

    Key differences from RefinedFacialGRN:
    - CaMKII runs concurrently (not pre-equilibrated Ca)
    - bio_gate comes from CaMKII activity (bistable, persistent)
    - Pattern survives Vmem decay via CaMKII memory
    """

    def __init__(self, grid_size, device='cpu', dtype=torch.float64,
                 shh_decay_length=0.8, fgf8_decay_length=0.3, edn1_decay_length=0.6):
        self.grid_size = grid_size
        self.device = device
        self.dtype = dtype

        # Store decay lengths as tensors for gradient flow
        if isinstance(shh_decay_length, torch.Tensor):
            self.shh_decay_length = shh_decay_length
        else:
            self.shh_decay_length = torch.tensor(shh_decay_length, device=device, dtype=dtype)

        if isinstance(fgf8_decay_length, torch.Tensor):
            self.fgf8_decay_length = fgf8_decay_length
        else:
            self.fgf8_decay_length = torch.tensor(fgf8_decay_length, device=device, dtype=dtype)

        if isinstance(edn1_decay_length, torch.Tensor):
            self.edn1_decay_length = edn1_decay_length
        else:
            self.edn1_decay_length = torch.tensor(edn1_decay_length, device=device, dtype=dtype)

        # Gene and morphogen names
        self.morphogen_names = ['shh', 'fgf8', 'edn1']
        self.gene_names = ['rx', 'six3', 'pax6', 'lhx2', 'alx', 'dlx', 'hand2', 'runx2']
        self.feature_names = ['bone', 'eye', 'nose', 'mouth']
        self.numGenes = len(self.gene_names)

        # CaMKII bistable switch (concurrent with GRN)
        self.camkii_switch = CaMKIIBistableSwitch(grid_size, device, dtype)

        # Parameter overrides for learning
        self.and_threshold_override = None
        self.and_sharpness_override = None

        # Morphogen parameters (from RefinedFacialGRN)
        self.morphogen_params = {
            'shh_strength': torch.tensor(1.0, device=device, dtype=dtype),
            'fgf8_strength': torch.tensor(0.2, device=device, dtype=dtype),
            'fgf8_degradation_factor': torch.tensor(10.0, device=device, dtype=dtype),
            'edn1_strength': torch.tensor(1.0, device=device, dtype=dtype),
            'edn1_degradation_factor': torch.tensor(2.0, device=device, dtype=dtype),
            'diffusion_rate': torch.tensor(0.1, device=device, dtype=dtype),
            'degradation_rate': torch.tensor(0.05, device=device, dtype=dtype),
            'inhibition_strength': torch.tensor(0.3, device=device, dtype=dtype),
        }

        # Gene parameters (from RefinedFacialGRN)
        self.gene_params = {
            'k_activation': torch.tensor(0.10, device=device, dtype=dtype),
            'k_degradation': torch.tensor(0.01, device=device, dtype=dtype),
            'w_initiation': torch.tensor(0.7, device=device, dtype=dtype),
            'w_maintenance': torch.tensor(0.3, device=device, dtype=dtype),
            'K_morph': torch.tensor(0.3, device=device, dtype=dtype),
            'n_morph': torch.tensor(2.0, device=device, dtype=dtype),
            'K_self': torch.tensor(0.3, device=device, dtype=dtype),
            'n_self': torch.tensor(2.0, device=device, dtype=dtype),
        }

        self.timestep = 0.01
        self.current_time = 0

        # Initialize grids
        self.initialize_grids()
        self.initialize_morphogen_sources()

    def initialize_grids(self):
        """Initialize all state grids"""
        gs = self.grid_size
        self.grid = {}

        # Morphogens
        for morph in self.morphogen_names:
            self.grid[morph] = torch.zeros(gs, gs, device=self.device, dtype=self.dtype)

        # Genes
        for gene in self.gene_names:
            self.grid[gene] = torch.zeros(gs, gs, device=self.device, dtype=self.dtype)

        self.grid['features'] = torch.zeros(gs, gs, device=self.device, dtype=self.dtype)

    def initialize_morphogen_sources(self):
        """Initialize spatial patterns for morphogen secretion"""
        gs = self.grid_size

        self.y_coords = torch.linspace(0.0, 1.0, gs, device=self.device, dtype=self.dtype).view(gs, 1).expand(gs, gs)
        self.x_coords = torch.linspace(0.0, 1.0, gs, device=self.device, dtype=self.dtype).view(1, gs).expand(gs, gs)
        self.dist_from_midline = torch.abs(self.x_coords - 0.5)

        self.compute_morphogen_sources()
        self.grid['edn1'] = self.edn1_source.clone()

    def compute_morphogen_sources(self):
        """Compute morphogen source patterns using current decay length parameters"""
        gs = self.grid_size

        # SHH: High at midline, decays laterally
        shhAnteriorBoost = 1.0 - self.y_coords * 0.5
        shh_decay_length_scaled = self.shh_decay_length / gs
        self.shh_source = shhAnteriorBoost * torch.exp(-self.dist_from_midline / shh_decay_length_scaled)

        # FGF8: High laterally, low at midline
        fgf8AnteriorBoost = 1.0 - self.y_coords * 0.5
        fgf8_decay_length_scaled = self.fgf8_decay_length / gs
        self.fgf8_source = fgf8AnteriorBoost * (1.0 - torch.exp(-self.dist_from_midline / fgf8_decay_length_scaled))

        # EDN1: Gradient from anterior to posterior
        # Changed from 0.8 to 1.0 to match SHH and FGF8 strength
        edn1_decay_length_scaled = self.edn1_decay_length / gs
        self.edn1_source = 1.0 * (1.0 - torch.exp(-self.y_coords / edn1_decay_length_scaled))

    def hill_activation(self, x, K, n):
        """Hill activation function: x^n / (K^n + x^n)"""
        return (x**n) / (K**n + x**n + 1e-9)

    def hill_inhibition(self, x, K, n):
        """Hill inhibition: K^n / (K^n + x^n)"""
        return (K**n) / (K**n + x**n + 1e-9)

    def logic_AND(self, A, B, threshold=1.25, sharpness=20.0):
        """Sigmoid-based AND gate"""
        if self.and_threshold_override is not None:
            threshold = self.and_threshold_override
        if self.and_sharpness_override is not None:
            sharpness = self.and_sharpness_override
        return torch.sigmoid(sharpness * (A + B - threshold))

    def logic_OR(self, A, B, threshold=0.5, sharpness=10.0):
        """Sigmoid-based OR gate"""
        return torch.sigmoid(sharpness * (A + B - threshold))

    def laplacian_2d(self, field):
        """Compute 2D Laplacian for diffusion"""
        field_padded = F.pad(field.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular')
        center = field_padded[:, :, 1:-1, 1:-1]
        left = field_padded[:, :, 1:-1, :-2]
        right = field_padded[:, :, 1:-1, 2:]
        up = field_padded[:, :, :-2, 1:-1]
        down = field_padded[:, :, 2:, 1:-1]
        laplacian = (left + right + up + down - 4 * center).squeeze(0).squeeze(0)
        return laplacian

    def update_morphogens(self):
        """Update morphogen concentrations via diffusion-degradation"""
        if (isinstance(self.shh_decay_length, torch.Tensor) and self.shh_decay_length.requires_grad) or \
           (isinstance(self.fgf8_decay_length, torch.Tensor) and self.fgf8_decay_length.requires_grad) or \
           (isinstance(self.edn1_decay_length, torch.Tensor) and self.edn1_decay_length.requires_grad):
            self.compute_morphogen_sources()

        params = self.morphogen_params
        D = params['diffusion_rate']
        k_deg = params['degradation_rate']

        shh_secretion = params['shh_strength'] * self.shh_source
        fgf8_secretion = params['fgf8_strength'] * self.fgf8_source
        fgf8_deg_factor = params['fgf8_degradation_factor']
        edn1_secretion = params['edn1_strength'] * self.edn1_source

        dt = self.timestep

        # Mutual inhibition
        inhibition_strength = torch.tensor(0.8, device=self.device, dtype=self.dtype)
        shh_inhibition = self.hill_inhibition(self.grid['fgf8'], K=0.3, n=3.0)
        fgf8_inhibition = self.hill_inhibition(self.grid['shh'], K=0.3, n=3.0)

        # SHH dynamics
        dshh_dt = (shh_secretion +
                   D * self.laplacian_2d(self.grid['shh']) -
                   k_deg * self.grid['shh'] * (1.0 - shh_inhibition) +
                   inhibition_strength * self.grid['shh'] * (1.0 - self.grid['fgf8']))
        self.grid['shh'] = self.grid['shh'] + dt * dshh_dt
        self.grid['shh'] = torch.clamp(self.grid['shh'], min=0.0, max=1.0)

        # FGF8 dynamics
        D_fgf8 = D * 0.1
        k_deg_fgf8 = k_deg * fgf8_deg_factor
        dfgf8_dt = (fgf8_secretion +
                    D_fgf8 * self.laplacian_2d(self.grid['fgf8']) -
                    k_deg_fgf8 * self.grid['fgf8'] * (1.0 - fgf8_inhibition) +
                    inhibition_strength * self.grid['fgf8'] * (1.0 - self.grid['shh']))
        self.grid['fgf8'] = self.grid['fgf8'] + dt * dfgf8_dt
        self.grid['fgf8'] = torch.clamp(self.grid['fgf8'], min=0.0, max=1.0)

        # EDN1 dynamics
        if 'edn1_degradation_factor' in params:
            edn1_deg_factor = params['edn1_degradation_factor']
        else:
            edn1_deg_factor = 1.0  # Default degradation matching SHH (changed from 2.0)

        D_edn1 = D * 0.05
        k_deg_edn1 = k_deg * edn1_deg_factor
        dedn1_dt = (edn1_secretion +
                    D_edn1 * self.laplacian_2d(self.grid['edn1']) -
                    k_deg_edn1 * self.grid['edn1'])
        self.grid['edn1'] = self.grid['edn1'] + dt * dedn1_dt
        self.grid['edn1'] = torch.clamp(self.grid['edn1'], min=0.0, max=1.0)

    def update_genes(self, bio_gate=None):
        """
        Update gene expression using CaMKII-based bioelectric gating.

        Args:
            bio_gate: (grid_size, grid_size) bioelectric gate from CaMKII
                      High at features (CaMKII > K_half), low at background
                      If None, uses autonomous morphogen-only mode (no bioelectric gating)
        """
        params = self.gene_params
        K_morph = params['K_morph']
        n_morph = params['n_morph']
        k_on = params['k_activation']
        k_off = params['k_degradation']
        dt = self.timestep

        shh = self.grid['shh']
        fgf8 = self.grid['fgf8']
        edn1 = self.grid['edn1']

        w_init = params['w_initiation']
        w_maint = params['w_maintenance']
        K_self = params['K_self']
        n_self = params['n_self']

        # Determine if autonomous mode (no bioelectric gating)
        autonomous = (bio_gate is None)
        if autonomous:
            bio_gate = torch.ones_like(shh)

        # Eye genes
        morph_eye = (self.hill_activation(fgf8, 0.3, 2.0) *
                     self.hill_inhibition(shh, 0.6, 2.0) *
                     self.hill_inhibition(edn1, 0.2, 2.0))

        # Rx
        # Bioelectric-gated: AND-OR logic
        initiation = self.logic_AND(morph_eye, bio_gate)
        maintenance = self.hill_activation(self.grid['rx'], K_self, n_self)
        activation = self.logic_OR(w_init * initiation, w_maint * maintenance)
        drx_dt = k_on * activation - k_off * self.grid['rx']
        self.grid['rx'] = self.grid['rx'] + dt * drx_dt

        # Six3
        morph_six3 = self.hill_activation(self.grid['rx'], 0.3, 2.0)
        initiation = self.logic_AND(morph_six3, bio_gate)
        maintenance = self.hill_activation(self.grid['six3'], K_self, n_self)
        activation = self.logic_OR(w_init * initiation, w_maint * maintenance)
        dsix3_dt = k_on * activation - k_off * self.grid['six3']
        self.grid['six3'] = self.grid['six3'] + dt * dsix3_dt

        # Pax6
        morph_pax6 = self.hill_activation(self.grid['six3'], 0.3, 2.0)
        initiation = self.logic_AND(morph_pax6, bio_gate)
        maintenance = self.hill_activation(self.grid['pax6'], K_self, n_self)
        activation = self.logic_OR(w_init * initiation, w_maint * maintenance)
        dpax6_dt = k_on * activation - k_off * self.grid['pax6']
        self.grid['pax6'] = self.grid['pax6'] + dt * dpax6_dt

        # Lhx2
        morph_lhx2 = self.hill_activation(self.grid['pax6'], 0.3, 2.0)
        initiation = self.logic_AND(morph_lhx2, bio_gate)
        maintenance = self.hill_activation(self.grid['lhx2'], K_self, n_self)
        activation = self.logic_OR(w_init * initiation, w_maint * maintenance)
        dlhx2_dt = k_on * activation - k_off * self.grid['lhx2']
        self.grid['lhx2'] = self.grid['lhx2'] + dt * dlhx2_dt

        # Nose gene (alx)
        nose_shh_K = params.get('nose_shh_K', 0.7)
        nose_shh_n = params.get('nose_shh_n', 4.0)
        nose_edn1_K = params.get('nose_edn1_K', 0.2)

        morph_nose = (self.hill_activation(shh, nose_shh_K, nose_shh_n) *
                      self.hill_inhibition(fgf8, 0.4, 2.0) *
                      self.hill_inhibition(edn1, nose_edn1_K, 2.0))

        initiation = self.logic_AND(morph_nose, bio_gate)
        maintenance = self.hill_activation(self.grid['alx'], K_self, n_self)
        activation = self.logic_OR(w_init * initiation, w_maint * maintenance)
        dalx_dt = k_on * activation - k_off * self.grid['alx']
        self.grid['alx'] = self.grid['alx'] + dt * dalx_dt

        # Mouth genes
        mouth_edn1_K = params.get('mouth_edn1_K', 0.2)
        mouth_edn1_n = params.get('mouth_edn1_n', 2.0)

        morph_mouth = self.hill_activation(edn1, mouth_edn1_K, mouth_edn1_n)

        # Dlx
        initiation = self.logic_AND(morph_mouth, bio_gate)
        maintenance = self.hill_activation(self.grid['dlx'], K_self, n_self)
        activation = self.logic_OR(w_init * initiation, w_maint * maintenance)
        ddlx_dt = k_on * activation - k_off * self.grid['dlx']
        self.grid['dlx'] = self.grid['dlx'] + dt * ddlx_dt

        # Hand2
        morph_hand2 = self.hill_activation(self.grid['dlx'], 0.3, 2.0)
        initiation = self.logic_AND(morph_hand2, bio_gate)
        maintenance = self.hill_activation(self.grid['hand2'], K_self, n_self)
        activation = self.logic_OR(w_init * initiation, w_maint * maintenance)
        dhand2_dt = k_on * activation - k_off * self.grid['hand2']
        self.grid['hand2'] = self.grid['hand2'] + dt * dhand2_dt

        # Bone gene (runx2) - activated when others are low
        eye_signal = self.grid['pax6'] * self.grid['lhx2']
        nose_signal = self.grid['alx']
        mouth_signal = self.grid['dlx'] * self.grid['hand2']
        max_other = torch.maximum(torch.maximum(eye_signal, nose_signal), mouth_signal)

        morph_bone = self.hill_inhibition(max_other, 0.2, 2.0)
        initiation_runx2 = self.logic_AND(morph_bone, bio_gate)
        maintenance_runx2 = self.hill_activation(self.grid['runx2'], K_self, n_self)
        activation_runx2 = self.logic_OR(w_init * initiation_runx2, w_maint * maintenance_runx2)
        drunx2_dt = k_on * activation_runx2 - k_off * self.grid['runx2']
        self.grid['runx2'] = self.grid['runx2'] + dt * drunx2_dt

        # Clamp all genes
        for gene in self.gene_names:
            self.grid[gene] = torch.clamp(self.grid[gene], min=0.0, max=1.0)

    def update_concurrent(self, vmem_grid, dt=0.01):
        """
        Single timestep update with concurrent CaMKII and GRN dynamics.

        This is the key difference from RefinedFacialGRN:
        - CaMKII updates every timestep (tracks Vmem changes)
        - bio_gate comes from raw CaMKII activity (bistable, persistent)
        - GRN uses current CaMKII state for gating (already in [0,1])

        Args:
            vmem_grid: (grid_size, grid_size) membrane voltage
            dt: timestep
        """
        # Update CaMKII (tracks Vmem, maintains bistable state)
        self.camkii_switch.update(vmem_grid, dt=dt)

        # Get bioelectric gate from raw CaMKII activity (already in [0,1])
        # This is equivalent to how RefinedFacialGRN uses normalized Ca as bio_gate
        bio_gate = self.camkii_switch.CaMKII_active

        # Update morphogens
        self.update_morphogens()

        # Update genes with CaMKII-derived bio_gate
        self.update_genes(bio_gate)

        self.current_time += self.timestep

    def simulate_concurrent(self, vmem_trajectory, dt=0.01, checkpoint_interval=100):
        """
        Run simulation with concurrent CaMKII and GRN dynamics.

        Args:
            vmem_trajectory: List of (grid_size, grid_size) Vmem snapshots,
                             or callable(t) -> vmem_grid
            dt: timestep
            checkpoint_interval: How often to save history

        Returns:
            history: dict with time series data
        """
        history = {
            'time': [],
            'Ca_mean': [], 'Ca_std': [],
            'CaMKII_mean': [], 'CaMKII_std': [],
            'vmem_mean': [], 'vmem_std': [],
            'bio_gate_mean': [],
            'genes': {gene: [] for gene in self.gene_names},
            'morphogens': {morph: [] for morph in self.morphogen_names},
        }

        # Reset CaMKII state
        self.camkii_switch.reset()

        # Pre-equilibrate morphogens (1000 steps, matching refined model)
        print("Pre-equilibrating morphogens (1000 steps)...")
        for _ in range(1000):
            self.update_morphogens()

        morph_grids_eq = self.get_morphogen_grids()
        print(f"  SHH: max={morph_grids_eq['shh'].max():.4f}")
        print(f"  FGF8: max={morph_grids_eq['fgf8'].max():.4f}")
        print(f"  EDN1: max={morph_grids_eq['edn1'].max():.4f}")

        num_steps = len(vmem_trajectory) if isinstance(vmem_trajectory, list) else None

        t = 0
        while True:
            # Get current Vmem
            if callable(vmem_trajectory):
                vmem_grid = vmem_trajectory(t)
                if vmem_grid is None:
                    break
            else:
                if t >= len(vmem_trajectory):
                    break
                vmem_grid = vmem_trajectory[t]

            # Update (concurrent CaMKII + GRN)
            self.update_concurrent(vmem_grid, dt=dt)

            # Record history
            if t % checkpoint_interval == 0:
                history['time'].append(t * dt)
                history['Ca_mean'].append(self.camkii_switch.Ca.mean().item())
                history['Ca_std'].append(self.camkii_switch.Ca.std().item())
                history['CaMKII_mean'].append(self.camkii_switch.CaMKII_active.mean().item())
                history['CaMKII_std'].append(self.camkii_switch.CaMKII_active.std().item())
                history['vmem_mean'].append(vmem_grid.mean().item())
                history['vmem_std'].append(vmem_grid.std().item())

                # bio_gate is raw CaMKII activity (already in [0,1])
                bio_gate = self.camkii_switch.CaMKII_active
                history['bio_gate_mean'].append(bio_gate.mean().item())

                for gene in self.gene_names:
                    history['genes'][gene].append(self.grid[gene].mean().item())
                for morph in self.morphogen_names:
                    history['morphogens'][morph].append(self.grid[morph].mean().item())

            t += 1

        return history

    def get_gene_grids(self):
        """Get all gene expression grids as dict"""
        return {gene: self.grid[gene] for gene in self.gene_names}

    def get_morphogen_grids(self):
        """Get all morphogen grids as dict"""
        return {morph: self.grid[morph] for morph in self.morphogen_names}

    def get_state(self):
        """Get complete state for visualization/analysis"""
        return {
            'genes': self.get_gene_grids(),
            'morphogens': self.get_morphogen_grids(),
            'Ca': self.camkii_switch.Ca,
            'CaMKII': self.camkii_switch.CaMKII_active,
            'time': self.current_time
        }

    def reset(self):
        """Reset all grids to initial state"""
        self.initialize_grids()
        self.camkii_switch.reset()
        self.current_time = 0
