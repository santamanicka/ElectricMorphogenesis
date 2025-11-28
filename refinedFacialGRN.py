"""
Refined Facial GRN with Dual-Driver Architecture
Genes are activated by BOTH:
1. Morphogen gradients (SHH, FGF8, EDN1) - 70% weight
2. Bioelectric signals (Ca²⁺) - 30% weight

Both drivers use Hill function dynamics (as requested).
No "detail" logic - uses actual biophysical signals.
"""

import torch
import torch.nn.functional as F


class RefinedFacialGRN:
    """
    Facial Gene Regulatory Network with dual bioelectric + morphogen drivers.

    Key features:
    - Morphogen gradients (SHH, FGF8, EDN1) provide spatial patterning
    - Bioelectric signals (Ca²⁺) modulate gene activation via gating
    - Both pathways use Hill activation functions
    - Features classified from gene expression only (not voltage)
    """

    def __init__(self, grid_size, device='cpu', dtype=torch.float64, shh_decay_length=0.8, fgf8_decay_length=0.3, edn1_decay_length=0.6):
        self.grid_size = grid_size
        self.device = device
        self.dtype = dtype

        # Morphogen shape parameters (can be overridden for learning)
        # Store as tensors to support gradient flow
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

        # Parameter overrides for learning (set by external optimizer)
        self.and_threshold_override = None
        self.and_sharpness_override = None

        # =====================
        # Morphogen Parameters
        # =====================
        self.morphogen_params = {
            # Morphogen parameters from FacialGRN.html lines 271-280
            # Adapted for 11x11 grid - REDUCED FGF8 to prevent saturation and create sharp complementarity
            'shh_strength': torch.tensor(1.0, device=device, dtype=dtype),   # Applied to shh_source
            'fgf8_strength': torch.tensor(0.2, device=device, dtype=dtype),  # FURTHER REDUCED from 0.4 to 0.2
            'fgf8_degradation_factor': torch.tensor(10.0, device=device, dtype=dtype),   # 10× higher degradation than base rate
            'edn1_strength': torch.tensor(1.0, device=device, dtype=dtype),  # Applied to edn1_source

            # Diffusion and degradation (FacialGRN.html lines 276-277)
            'diffusion_rate': torch.tensor(0.1, device=device, dtype=dtype),     # Line 276
            'degradation_rate': torch.tensor(0.05, device=device, dtype=dtype),  # Line 277

            # Mutual inhibition strength (FacialGRN.html line 278)
            'inhibition_strength': torch.tensor(0.3, device=device, dtype=dtype),  # Line 278
        }

        # =====================
        # Gene Activation Parameters (from FacialGRN.html lines 279-280, 408-422)
        # =====================
        self.gene_params = {
            # Gene dynamics rates (FacialGRN.html lines 279-280)
            # INCREASED activation to allow genes to reach morphogen targets
            'k_activation': torch.tensor(0.10, device=device, dtype=dtype),    # geneActivationRate (was 0.05)
            'k_degradation': torch.tensor(0.01, device=device, dtype=dtype),   # geneDegradationRate (was 0.02)

            # AND-OR architecture weights for bioelectric gating
            # Gene_activation = (Morphogen AND Bio_gate) OR (Self_maintenance)
            'w_initiation': torch.tensor(0.7, device=device, dtype=dtype),   # Morphogen × Bio_gate strength
            'w_maintenance': torch.tensor(0.3, device=device, dtype=dtype),  # Self-maintenance allows persistence

            # Hill function parameters matching FacialGRN.html
            # Eye pathway (line 410): hill(fgf8, 0.3, 2) * inhibit(shh, 0.4, 2) * inhibit(edn1, 0.2, 2)
            # Nose pathway (line 417): hill(shh, 0.5, 2) * inhibit(fgf8, 0.4, 2) * inhibit(edn1, 0.2, 2)
            # Jaw pathway (line 422): hill(edn1, 0.3, 2) * hill(shh, 0.15, 1.5)
            'K_morph': torch.tensor(0.3, device=device, dtype=dtype),  # Base K for morphogens
            'n_morph': torch.tensor(2.0, device=device, dtype=dtype),  # Cooperativity

            # Self-maintenance parameters (for gene cascades like line 411-413)
            'K_self': torch.tensor(0.3, device=device, dtype=dtype),
            'n_self': torch.tensor(2.0, device=device, dtype=dtype),
        }

        # Timescale
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

        # Features (will be computed from genes, not stored in dynamics)
        self.grid['features'] = torch.zeros(gs, gs, device=self.device, dtype=self.dtype)

    def initialize_morphogen_sources(self):
        """
        Initialize spatial patterns for morphogen secretion.
        These are the BASELINE patterns (before bioelectric modulation).

        NOTE: For learnable decay lengths, sources are recomputed in compute_morphogen_sources()
        This is just initial setup.
        """
        gs = self.grid_size

        # Create normalized coordinate grids (0 to 1) - store for reuse
        self.y_coords = torch.linspace(0.0, 1.0, gs, device=self.device, dtype=self.dtype).view(gs, 1).expand(gs, gs)
        self.x_coords = torch.linspace(0.0, 1.0, gs, device=self.device, dtype=self.dtype).view(1, gs).expand(gs, gs)
        self.dist_from_midline = torch.abs(self.x_coords - 0.5)

        # Compute initial sources
        self.compute_morphogen_sources()

        # Initialize EDN1 grid from source (since it's static, just copy the pattern)
        self.grid['edn1'] = self.edn1_source.clone()

    def compute_morphogen_sources(self):
        """
        Compute morphogen source patterns using current decay length parameters.
        This is called during initialization and can be called again if decay lengths change.
        """
        gs = self.grid_size

        # SHH: High at midline, decays laterally (inverted V)
        # SHARPER decay for 11x11 grid to create spatial specificity
        shhAnteriorBoost = 1.0 - self.y_coords * 0.5  # Stronger in anterior
        shh_decay_length_scaled = self.shh_decay_length / gs
        self.shh_source = shhAnteriorBoost * torch.exp(-self.dist_from_midline / shh_decay_length_scaled)

        # FGF8: High laterally, low at midline (V-shaped)
        # MATCHING sharp spatial profile to SHH for complementarity
        fgf8AnteriorBoost = 1.0 - self.y_coords * 0.5  # Stronger in anterior
        fgf8_decay_length_scaled = self.fgf8_decay_length / gs
        self.fgf8_source = fgf8AnteriorBoost * (1.0 - torch.exp(-self.dist_from_midline / fgf8_decay_length_scaled))

        # EDN1: Exponential gradient from anterior (low) to posterior (high)
        # FacialGRN.html line 324-325: params.edn1Strength * 0.8 * normalizedY
        # Now with learnable decay length for steepness control
        edn1_decay_length_scaled = self.edn1_decay_length / gs
        self.edn1_source = 0.8 * (1.0 - torch.exp(-self.y_coords / edn1_decay_length_scaled))

    def hill_activation(self, x, K, n):
        """Hill activation function: x^n / (K^n + x^n)"""
        return (x**n) / (K**n + x**n + 1e-9)

    def hill_inhibition(self, x, K, n):
        """Hill inhibition: K^n / (K^n + x^n)"""
        return (K**n) / (K**n + x**n + 1e-9)

    def logic_AND(self, A, B, threshold=1.25, sharpness=20.0):
        """
        Sigmoid-based AND gate: both A and B must be high
        AND(A,B) = sigmoid(sharpness × (A + B - threshold))
        With threshold=1.25, requires A≈0.625 AND B≈0.625 to activate
        BALANCED: Moderate-high threshold (1.25) and sharp slope (20.0) for spatial segregation
        while allowing sufficient feature differentiation

        Can be overridden by setting self.and_threshold_override and self.and_sharpness_override
        """
        # Use overrides if set (for learning)
        if self.and_threshold_override is not None:
            threshold = self.and_threshold_override
        if self.and_sharpness_override is not None:
            sharpness = self.and_sharpness_override

        return torch.sigmoid(sharpness * (A + B - threshold))

    def logic_OR(self, A, B, threshold=0.5, sharpness=10.0):
        """
        Sigmoid-based OR gate: either A or B can be high
        OR(A,B) = sigmoid(sharpness × (A + B - threshold))
        With threshold=0.5, activates if A≈0.5 OR B≈0.5
        """
        return torch.sigmoid(sharpness * (A + B - threshold))

    def gene_dynamics(self, morph_signal, bio_gate, current_gene, bioelectric_signals, params, dt):
        """
        Unified gene dynamics: autonomous relaxation OR bioelectric-gated AND-OR logic.

        Args:
            morph_signal: Morphogen-derived target signal
            bio_gate: Bioelectric gate (1.0 for autonomous)
            current_gene: Current gene expression level
            bioelectric_signals: None for autonomous, dict for bioelectric-gated
            params: Gene parameters dict
            dt: Timestep

        Returns:
            Updated gene expression
        """
        k_on = params['k_activation']
        k_off = params['k_degradation']
        w_init = params['w_initiation']
        w_maint = params['w_maintenance']
        K_self = params['K_self']
        n_self = params['n_self']

        if bioelectric_signals is None:
            # Autonomous: morphogen-driven + self-maintenance (no bioelectric gating)
            # Uses self-maintenance to allow K_self and n_self to be learnable
            initiation = morph_signal  # Direct morphogen signal (no AND gate)
            maintenance = self.hill_activation(current_gene, K_self, n_self)
            activation = w_init * initiation + w_maint * maintenance
            dgene_dt = k_on * activation - k_off * current_gene
        else:
            # Bioelectric-gated: AND-OR logic
            initiation = self.logic_AND(morph_signal, bio_gate)
            maintenance = self.hill_activation(current_gene, K_self, n_self)
            activation = self.logic_OR(w_init * initiation, w_maint * maintenance)
            dgene_dt = k_on * activation - k_off * current_gene

        return current_gene + dt * dgene_dt

    def laplacian_2d(self, field):
        """
        Compute 2D Laplacian for diffusion.
        Uses 5-point stencil with periodic boundary conditions.
        """
        # Pad with periodic boundaries
        field_padded = F.pad(field.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular')

        # 5-point stencil
        center = field_padded[:, :, 1:-1, 1:-1]
        left = field_padded[:, :, 1:-1, :-2]
        right = field_padded[:, :, 1:-1, 2:]
        up = field_padded[:, :, :-2, 1:-1]
        down = field_padded[:, :, 2:, 1:-1]

        laplacian = (left + right + up + down - 4 * center).squeeze(0).squeeze(0)

        return laplacian

    def update_morphogens(self):
        """
        Update morphogen concentrations via diffusion-degradation.

        IMPORTANT: If decay lengths are learnable (have gradients), recompute sources
        to enable gradient flow during backpropagation.
        """
        # Recompute morphogen sources if decay lengths have gradients (for learning)
        if (isinstance(self.shh_decay_length, torch.Tensor) and self.shh_decay_length.requires_grad) or \
           (isinstance(self.fgf8_decay_length, torch.Tensor) and self.fgf8_decay_length.requires_grad) or \
           (isinstance(self.edn1_decay_length, torch.Tensor) and self.edn1_decay_length.requires_grad):
            self.compute_morphogen_sources()

        params = self.morphogen_params
        D = params['diffusion_rate']
        k_deg = params['degradation_rate']

        # Baseline secretion patterns (spatial)
        # NO bioelectric modulation - morphogens maintain independent steady gradients
        shh_secretion = params['shh_strength'] * self.shh_source
        fgf8_secretion = params['fgf8_strength'] * self.fgf8_source
        fgf8_deg_factor = params['fgf8_degradation_factor']
        edn1_secretion = params['edn1_strength'] * self.edn1_source

        # Diffusion-degradation-inhibition dynamics
        dt = self.timestep

        # Mutual inhibition between SHH and FGF8 (maintains complementary patterns)
        # STRONGER inhibition to enforce sharp spatial complementarity in 11x11 grid
        inhibition_strength = torch.tensor(0.8, device=self.device, dtype=self.dtype)  # Increased from 0.3
        shh_inhibition = self.hill_inhibition(self.grid['fgf8'], K=0.3, n=3.0)  # Sharper (was K=0.5, n=2.0)
        fgf8_inhibition = self.hill_inhibition(self.grid['shh'], K=0.3, n=3.0)  # Sharper

        # SHH dynamics (FacialGRN.html lines 375-378)
        dshh_dt = (shh_secretion +
                   D * self.laplacian_2d(self.grid['shh']) -
                   k_deg * self.grid['shh'] * (1.0 - shh_inhibition) +
                   inhibition_strength * self.grid['shh'] * (1.0 - self.grid['fgf8']))
        self.grid['shh'] = self.grid['shh'] + dt * dshh_dt
        self.grid['shh'] = torch.clamp(self.grid['shh'], min=0.0, max=1.0)

        # FGF8 dynamics - REDUCED diffusion and INCREASED degradation to maintain sharp spatial pattern
        # With source strength=0.4 and lateral source=0.75, secretion=0.30
        # Need higher degradation to prevent saturation: k_deg_fgf8 * fgf8 ≈ 0.30
        # For fgf8 ~ 0.5 at steady state: k_deg_fgf8 ≈ 0.60
        D_fgf8 = D * 0.1  # Much lower diffusion than SHH
        k_deg_fgf8 = k_deg * fgf8_deg_factor  # 10× higher degradation for FGF8 (0.5 instead of 0.05)
        dfgf8_dt = (fgf8_secretion +
                    D_fgf8 * self.laplacian_2d(self.grid['fgf8']) -
                    k_deg_fgf8 * self.grid['fgf8'] * (1.0 - fgf8_inhibition) +
                    inhibition_strength * self.grid['fgf8'] * (1.0 - self.grid['shh']))
        self.grid['fgf8'] = self.grid['fgf8'] + dt * dfgf8_dt
        self.grid['fgf8'] = torch.clamp(self.grid['fgf8'], min=0.0, max=1.0)

        # EDN1: Slower dynamics with learnable degradation (FacialGRN.html line 385-386)
        # "Edn1 doesn't diffuse much - just maintain posterior expression"
        # Check if edn1_degradation_factor exists in params for learning
        if 'edn1_degradation_factor' in params:
            edn1_deg_factor = params['edn1_degradation_factor']
        else:
            edn1_deg_factor = 2.0  # Default moderate degradation

        D_edn1 = D * 0.05  # Very low diffusion (even less than FGF8)
        k_deg_edn1 = k_deg * edn1_deg_factor  # Learnable degradation for EDN1
        dedn1_dt = (edn1_secretion +
                    D_edn1 * self.laplacian_2d(self.grid['edn1']) -
                    k_deg_edn1 * self.grid['edn1'])
        self.grid['edn1'] = self.grid['edn1'] + dt * dedn1_dt
        self.grid['edn1'] = torch.clamp(self.grid['edn1'], min=0.0, max=1.0)

    def update_genes(self, bioelectric_signals=None):
        """
        Update gene expression using DUAL DRIVERS:
        1. Morphogen input (70%) - Hill activation
        2. Bioelectric input (30%) - Hill activation

        Both pathways use Hill dynamics (as requested).
        """
        params = self.gene_params
        K_morph = params['K_morph']
        n_morph = params['n_morph']
        k_on = params['k_activation']
        k_off = params['k_degradation']
        dt = self.timestep

        # Extract morphogens
        shh = self.grid['shh']
        fgf8 = self.grid['fgf8']
        edn1 = self.grid['edn1']

        # Extract bioelectric signals (or use defaults)
        if bioelectric_signals is not None:
            # Use Ca²⁺ as the bioelectric gate (has better face-like spatial structure than Vmem)
            Ca = bioelectric_signals.get('Ca', torch.zeros_like(shh))

            # Bioelectric GATE: Use Ca²⁺ percentile threshold
            # Features appear at LOW Ca²⁺ (dark regions in visualization)
            # BALANCED gating for spatial segregation with feature differentiation
            # Use 45th percentile as threshold: Ca below this → high gate (balanced selectivity)
            Ca_threshold = torch.quantile(Ca, 0.45)  # 35th percentile (very selective)
            Ca_sensitivity = 0.04  # VERY SHARP sensitivity for crisp boundaries
            bio_gate = torch.sigmoid((Ca_threshold - Ca) / Ca_sensitivity)  # HIGH when Ca < threshold
            # Ca < threshold (face features) → bio_gate ≈ 1.0
            # Ca > threshold (background) → bio_gate ≈ 0.0
        else:
            bio_gate = torch.ones_like(shh)  # Permissive when no bioelectrics specified

        # ========================================
        # EYE GENES (morphogen AND bio_gate OR self)
        # ========================================
        # Morphogen signal from FacialGRN.html line 410:
        # targetRx = hill(fgf8, 0.3, 2) * inhibit(shh, 0.4, 2) * inhibit(edn1, 0.2, 2)
        # MODIFIED: Broader eyes - weaker SHH inhibition (K=0.6 instead of 0.4) allows eyes closer to midline
        morph_eye = (self.hill_activation(fgf8, 0.3, 2.0) *
                     self.hill_inhibition(shh, 0.6, 2.0) *  # Raised from 0.4 to 0.6
                     self.hill_inhibition(edn1, 0.2, 2.0))

        # AND-OR logic: (Morphogen AND Bio_gate) OR Self_maintenance
        w_init = params['w_initiation']
        w_maint = params['w_maintenance']
        K_self = params['K_self']
        n_self = params['n_self']

        # Rx: first eye gene (FacialGRN.html line 410)
        self.grid['rx'] = self.gene_dynamics(morph_eye, bio_gate, self.grid['rx'],
                                             bioelectric_signals, params, dt)

        # Six3: activated by rx (gene cascade, FacialGRN.html line 411)
        morph_six3 = self.hill_activation(self.grid['rx'], 0.3, 2.0)
        self.grid['six3'] = self.gene_dynamics(morph_six3, bio_gate, self.grid['six3'],
                                               bioelectric_signals, params, dt)

        # Pax6: activated by six3 (FacialGRN.html line 412)
        morph_pax6 = self.hill_activation(self.grid['six3'], 0.3, 2.0)
        self.grid['pax6'] = self.gene_dynamics(morph_pax6, bio_gate, self.grid['pax6'],
                                               bioelectric_signals, params, dt)

        # Lhx2: activated by pax6 (FacialGRN.html line 413)
        morph_lhx2 = self.hill_activation(self.grid['pax6'], 0.3, 2.0)
        self.grid['lhx2'] = self.gene_dynamics(morph_lhx2, bio_gate, self.grid['lhx2'],
                                               bioelectric_signals, params, dt)

        # ========================================
        # NOSE GENE (morphogen AND bio_gate OR self)
        # ========================================
        # Morphogen signal from FacialGRN.html line 417:
        # targetAlx = hill(shh, 0.5, 2) * inhibit(fgf8, 0.4, 2) * inhibit(edn1, 0.2, 2)
        # MODIFIED: Use learnable parameters for nose-specific thresholds and cooperativity
        # Allows optimizer to find appropriate values for proper nose formation
        nose_shh_K = params.get('nose_shh_K', 0.7)  # Learnable: default 0.7, range 0.3-0.9
        nose_shh_n = params.get('nose_shh_n', 4.0)  # Learnable: default 4.0, range 1.0-6.0
        nose_edn1_K = params.get('nose_edn1_K', 0.2)  # Learnable: default 0.2, range 0.1-0.6

        morph_nose = (self.hill_activation(shh, nose_shh_K, nose_shh_n) *
                      self.hill_inhibition(fgf8, 0.4, 2.0) *
                      self.hill_inhibition(edn1, nose_edn1_K, 2.0))

        # Sigmoid-based AND-OR logic
        self.grid['alx'] = self.gene_dynamics(morph_nose, bio_gate, self.grid['alx'],
                                              bioelectric_signals, params, dt)

        # ========================================
        # MOUTH GENES (morphogen AND bio_gate OR self)
        # ========================================
        # Morphogen signal: High EDN1 (posterior) spanning horizontally
        # MODIFIED: Remove SHH inhibition to allow mouth to span entire posterior horizontal region
        # Lower EDN1 threshold (K=0.2 instead of 0.3) to activate earlier in posterior
        morph_mouth = self.hill_activation(edn1, 0.2, 2.0)  # Removed SHH inhibition, lowered K from 0.3 to 0.2

        # Dlx: Sigmoid-based AND-OR logic
        self.grid['dlx'] = self.gene_dynamics(morph_mouth, bio_gate, self.grid['dlx'],
                                              bioelectric_signals, params, dt)

        # Hand2: activated by dlx (gene cascade, FacialGRN.html line 423)
        morph_hand2 = self.hill_activation(self.grid['dlx'], 0.3, 2.0)
        self.grid['hand2'] = self.gene_dynamics(morph_hand2, bio_gate, self.grid['hand2'],
                                                bioelectric_signals, params, dt)

        # ========================================
        # BONE GENE (default when others low, with self-maintenance)
        # ========================================
        # Activated when eye/nose/mouth scores are all low (default/background state)
        eye_signal = self.grid['pax6'] * self.grid['lhx2']
        nose_signal = self.grid['alx']
        mouth_signal = self.grid['dlx'] * self.grid['hand2']
        max_other = torch.maximum(torch.maximum(eye_signal, nose_signal), mouth_signal)

        morph_bone = self.hill_inhibition(max_other, 0.2, 2.0)  # High when others low
        initiation_runx2 = self.logic_AND(morph_bone, bio_gate)  # Sigmoid-based AND
        maintenance_runx2 = self.hill_activation(self.grid['runx2'], K_self, n_self)
        activation_runx2 = self.logic_OR(w_init * initiation_runx2, w_maint * maintenance_runx2)  # Sigmoid-based OR
        drunx2_dt = k_on * activation_runx2 - k_off * self.grid['runx2']
        self.grid['runx2'] = self.grid['runx2'] + dt * drunx2_dt

        # Clamp all genes to [0, 1]
        for gene in self.gene_names:
            self.grid[gene] = torch.clamp(self.grid[gene], min=0.0, max=1.0)

    def update(self, bioelectric_signals=None):
        """
        Single timestep update: morphogens then genes.

        Args:
            bioelectric_signals: dict with 'Ca' key
        """
        self.update_morphogens()
        self.update_genes(bioelectric_signals)
        self.current_time += self.timestep

    def simulate(self, num_steps, bioelectric_signals=None):
        """
        Run simulation for multiple timesteps.

        Args:
            num_steps: Number of timesteps
            bioelectric_signals: dict with 'Ca' key (can be time-varying)
        """
        for step in range(num_steps):
            self.update(bioelectric_signals)

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
            'time': self.current_time
        }

    def reset(self):
        """Reset all grids to initial state"""
        self.initialize_grids()
        self.current_time = 0
