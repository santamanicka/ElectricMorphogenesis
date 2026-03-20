#!/usr/bin/env python3
"""
Group-level stress-based rescue on a 2D embryo lattice.

Each embryo is placed on a 2D grid and exchanges stress signals with its
lattice neighbors (von Neumann or Moore connectivity). At each bioelectric
timestep, neighbor stress modulates each embryo's GRN damping:

    effective_damping(t) = sigmoid(logit(base_damping) + alpha * mean_neighbor_stress(t))

Usage:
    # Quick smoke test (4 embryos, short sim)
    python runGroupRescue.py --groupSize 4 --numBioSteps 100 --alpha 3.0 --dampingLevels "1.0,0.5"

    # 5x5 grid, center embryo stressed
    python runGroupRescue.py --groupSize 25 --dampingCenter 0.1 --alpha 5.0 --numBioSteps 500

    # 10x10 grid, random damping
    python runGroupRescue.py --groupSize 100 --dampingRange "0.3,1.0" --alpha 3.0 --numBioSteps 2000

    # Explicit grid dims
    python runGroupRescue.py --gridDims "5,10" --dampingRange "0.5,1.0" --alpha 3.0

    # Explicit per-embryo damping map (row-major order)
    python runGroupRescue.py --groupSize 9 --dampingMap "1.0,0.5,1.0,0.5,0.1,0.5,1.0,0.5,1.0"

    # With learned stress parameters
    python runGroupRescue.py --groupSize 25 --stressParamsFile data/bestLearnedStressParams_5.dat
"""

import argparse
import math
import copy
import time
from concurrent.futures import ThreadPoolExecutor, wait

import torch
import numpy as np
import matplotlib.pyplot as plt

from embryo import model
from stressBistableSwitch import StressBistableSwitch
import utilities


# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser(
    description='Group stress-based rescue on a 2D embryo lattice'
)
parser.add_argument('--groupSize', type=int, default=25,
                    help='Total number of embryos (auto-derives square grid, max 300)')
parser.add_argument('--gridDims', type=str, default=None,
                    help='Explicit grid dimensions as "rows,cols" (overrides groupSize)')
parser.add_argument('--alpha', type=float, default=3.0,
                    help='Rescue signal strength (default: 3.0)')
parser.add_argument('--numBioSteps', type=int, default=2000,
                    help='Bioelectric simulation steps (default: 2000)')
parser.add_argument('--numStressSteps', type=int, default=500,
                    help='Stress equilibration steps after bio sim (default: 500)')
parser.add_argument('--neighborhood', type=str, default='vonNeumann',
                    choices=['vonNeumann', 'moore'],
                    help='Neighbor connectivity type (default: vonNeumann)')
parser.add_argument('--stressParamsFile', type=str, default=None,
                    help='Path to learned stress parameters file (.dat)')
parser.add_argument('--outputFile', type=str, default='data/group_rescue_test.png',
                    help='Output visualization filename')
parser.add_argument('--parallelThreshold', type=int, default=16,
                    help='Use ThreadPoolExecutor if num_embryos >= this (default: 16)')
parser.add_argument('--rescueThreshold', type=float, default=0.5,
                    help='Vmem similarity threshold for counting an embryo as "rescued" (default: 0.5)')

# Diffusive stress field parameters
parser.add_argument('--D_F', type=float, default=1.0,
                    help='Diffusion rate of stress field on embryo lattice (default: 1.0). '
                         'Set to 0 to fall back to mean_neighbor_stress.')
parser.add_argument('--gamma_F', type=float, default=0.1,
                    help='Decay rate of stress field (default: 0.1). '
                         'lambda = sqrt(D_F/gamma_F) sets communication range in lattice spacings.')
parser.add_argument('--diffusion_substeps', type=int, default=10,
                    help='Number of sub-steps per bio step for field diffusion (default: 10)')

# Initial stress level
parser.add_argument('--initialStress', type=float, default=0.0,
                    help='Initial stress level S for all cells in all embryos (default: 0.0, range [0,1])')

# Shuffle experiment
parser.add_argument('--shuffleTimes', type=str, default=None,
                    help='Comma-separated bio step(s) at which to shuffle embryo grid positions '
                         '(e.g., "500,1000"). Each time point runs as a separate simulation '
                         'compared to the unperturbed baseline.')
parser.add_argument('--shufflePreserveField', action='store_true', default=False,
                    help='Keep the diffusive stress field F fixed in place during shuffle. '
                         'Embryos move to new positions but inherit the field value already '
                         'at their destination rather than carrying their old field value along.')

# Save/load simulation data
parser.add_argument('--mode', type=str, default='both',
                    choices=['simulate', 'visualize', 'both'],
                    help='simulate=run+save, visualize=load+viz, both=all (default: both)')
parser.add_argument('--saveData', type=str, default=None,
                    help='Path to save simulation data (.dat)')
parser.add_argument('--loadData', type=str, default=None,
                    help='Path to load saved simulation data (.dat) for visualization')

# Damping assignment modes (mutually exclusive)
damping_group = parser.add_mutually_exclusive_group()
damping_group.add_argument('--dampingLevels', type=str, default=None,
                           help='Comma-separated damping levels assigned in alternating pattern')
damping_group.add_argument('--dampingRange', type=str, default=None,
                           help='Uniform random damping in "min,max" range')
damping_group.add_argument('--dampingMap', type=str, default=None,
                           help='Explicit comma-separated damping per embryo (row-major)')
damping_group.add_argument('--dampingCenter', type=float, default=None,
                           help='Center embryo gets this damping, all others get 1.0')
damping_group.add_argument('--dampingGaussian', type=str, default=None,
                           help='Gaussian damping: "mean,std" (clipped to [0.01, 1.0])')


# ============================================================
# Utility functions (copied from runStressRescue.py to avoid
# module-level argparse conflict on import)
# ============================================================
def apply_sigmoid_constraint(raw_param, min_val, max_val):
    """Map unbounded raw parameter to bounded range via sigmoid."""
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)


def load_stress_params(params_file):
    """Load learned stress parameters and fixed Ca2+ parameters from file."""
    print(f"Loading learned stress parameters from: {params_file}")
    data = torch.load(params_file, weights_only=False)

    stress_params = {}
    param_bounds = data.get('parameter_bounds', {})
    raw_params = data.get('parameters', {})

    for param_name, raw_value in raw_params.items():
        min_key = f'{param_name}_min'
        max_key = f'{param_name}_max'
        if min_key in param_bounds and max_key in param_bounds:
            constrained = apply_sigmoid_constraint(
                raw_value, param_bounds[min_key], param_bounds[max_key]
            )
            stress_params[param_name] = float(constrained.item())
        else:
            stress_params[param_name] = float(raw_value.item())

    print(f"  Learned stress parameters ({len(stress_params)}):")
    for name, value in stress_params.items():
        print(f"    {name}: {value:.4f}")

    fixed_ca_params = data.get('fixed_ca_params', None)
    if fixed_ca_params is not None:
        print(f"  Fixed Ca2+ parameters ({len(fixed_ca_params)}):")
        for name, value in fixed_ca_params.items():
            print(f"    {name}: {value:.4f}")
    else:
        print("  WARNING: No fixed_ca_params in file, using defaults")
        fixed_ca_params = get_default_ca_params()

    return stress_params, fixed_ca_params


def get_default_ca_params():
    """Return default Ca2+ parameters."""
    return {
        'tau_ca': 2.5964,
        'g_ca': 5.3437,
        'V_half_ca': -0.0753,
        'k_ca': 0.0021,
        'k_decay_ca': 4.3346,
    }


def get_default_stress_params():
    """Return default stress-specific parameters and Ca2+ parameters."""
    stress_params = {
        'tau_S': 50.0,
        'k_on_S': 3.0,
        'k_off_S': 0.02,
        'K_S': 0.4,
        'Ca_stress_threshold': 0.8,
        'sigma_ca': 0.2,
        'gain_S': 2.0,
        'or_threshold_S': 0.6,
        'D_S': 0.15,
        'gamma': 0.08,
        'K_decay': 0.3,
    }
    ca_params = get_default_ca_params()
    return stress_params, ca_params


def load_model_parameters(grn_damping=1.0):
    """Load Model 253 parameters with specified GRN damping."""
    path = './data/bestModelParameters_fieldVector_Ligand_GRN_253.dat'
    params = torch.load(path, weights_only=False)

    if "ATPParameters" not in params:
        params["ATPParameters"] = None

    if grn_damping != 1.0 and 'GRNParameters' in params and params['GRNParameters'] is not None:
        grn_params = params['GRNParameters']
        if 'GRNWeights' in grn_params and grn_params['GRNWeights'] is not None:
            grn_params['GRNWeights'] = grn_params['GRNWeights'] * grn_damping
        if 'InterGRNWeights' in grn_params and grn_params['InterGRNWeights'] is not None:
            grn_params['InterGRNWeights'] = grn_params['InterGRNWeights'] * grn_damping
        if 'GRNtoLigandWeights' in grn_params and grn_params['GRNtoLigandWeights'] is not None:
            grn_params['GRNtoLigandWeights'] = grn_params['GRNtoLigandWeights'] * grn_damping
        if grn_damping == 0.0:
            grn_params['GRNEnabled'] = False

    return params


def compute_effective_damping(base_damping, neighbor_stress_mean, alpha):
    """
    Compute effective GRN damping at a given timestep.

    effective_damping = sigmoid(logit(base_damping) + alpha * neighbor_stress_mean)
    """
    base_clamped = max(min(base_damping, 0.999), 0.001)
    base_logit = np.log(base_clamped / (1.0 - base_clamped))
    return 1.0 / (1.0 + np.exp(-(base_logit + alpha * neighbor_stress_mean)))


def compute_vmem_similarity(vmem, vmem_ref):
    """Compute Pearson correlation between a Vmem pattern and healthy reference."""
    if isinstance(vmem, torch.Tensor):
        vmem = vmem.detach().cpu().numpy()
    if isinstance(vmem_ref, torch.Tensor):
        vmem_ref = vmem_ref.detach().cpu().numpy()

    vmem_flat = vmem.flatten()
    ref_flat = vmem_ref.flatten()

    if np.std(vmem_flat) < 1e-10 or np.std(ref_flat) < 1e-10:
        if np.std(vmem_flat) < 1e-10 and np.std(ref_flat) < 1e-10:
            return 1.0 if np.abs(vmem_flat.mean() - ref_flat.mean()) < 1e-6 else 0.0
        return 0.0

    r = np.corrcoef(vmem_flat, ref_flat)[0, 1]
    return float(r)


# ============================================================
# Damping map construction
# ============================================================
def build_damping_map(args, rows, cols):
    """Build (rows, cols) numpy array of per-embryo GRN damping levels."""
    n = rows * cols
    if args.dampingMap is not None:
        vals = [float(x) for x in args.dampingMap.split(',')]
        if len(vals) != n:
            raise ValueError(f"dampingMap has {len(vals)} values, need {n}")
        return np.array(vals).reshape(rows, cols)
    elif args.dampingGaussian is not None:
        parts = [float(x) for x in args.dampingGaussian.split(',')]
        mean_d, std_d = parts[0], parts[1]
        dmap = np.random.normal(mean_d, std_d, (rows, cols))
        return np.clip(dmap, 0.01, 1.0)
    elif args.dampingRange is not None:
        lo, hi = [float(x) for x in args.dampingRange.split(',')]
        return np.random.uniform(lo, hi, (rows, cols))
    elif args.dampingCenter is not None:
        dmap = np.ones((rows, cols))
        cr, cc = rows // 2, cols // 2
        dmap[cr, cc] = args.dampingCenter
        return dmap
    elif args.dampingLevels is not None:
        levels = [float(x) for x in args.dampingLevels.split(',')]
        dmap = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                idx = (i * cols + j) % len(levels)
                dmap[i, j] = levels[idx]
        return dmap
    else:
        # Default: half healthy, half mildly stressed
        levels = [1.0, 0.5]
        dmap = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                idx = (i * cols + j) % len(levels)
                dmap[i, j] = levels[idx]
        return dmap


# ============================================================
# Grid dimension derivation
# ============================================================
def derive_grid_dims(group_size):
    """Derive grid (rows, cols) from group size, preferring square layouts."""
    side = int(math.ceil(math.sqrt(group_size)))
    rows = side
    cols = side
    # Shrink rows if we have more cells than needed
    while rows * cols > group_size and rows > 1:
        rows -= 1
    # If still too few, bump rows back up
    if rows * cols < group_size:
        rows += 1
    return rows, cols


# ============================================================
# Embryo neighbor adjacency builders
# ============================================================
def build_vonneumann_adjacency(rows, cols):
    """4-connected adjacency for embryo grid (non-periodic)."""
    n = rows * cols
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    nidx = ni * cols + nj
                    adj[idx, nidx] = 1.0
    return adj


def build_moore_adjacency(rows, cols):
    """8-connected adjacency for embryo grid (non-periodic)."""
    n = rows * cols
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        nidx = ni * cols + nj
                        adj[idx, nidx] = 1.0
    return adj


# ============================================================
# GroupRescueSimulation
# ============================================================
class GroupRescueSimulation:
    """
    Manages a 2D grid of embryos with group-level stress-based rescue.

    Each embryo has:
    - A bioelectric model (Model 253 instance)
    - A StressBistableSwitch (Vmem -> Ca2+ -> stress S)
    - A base GRN damping level
    - An effective damping that is modulated by neighbor stress signals
    """

    def __init__(self, grid_rows, grid_cols, damping_map, alpha,
                 stress_params, ca_params, neighborhood='vonNeumann',
                 parallel_threshold=16, D_F=1.0, gamma_F=0.1,
                 diffusion_substeps=10, initial_stress=0.0):
        """
        Args:
            grid_rows, grid_cols: embryo grid dimensions
            damping_map: (grid_rows, grid_cols) numpy array of per-embryo base dampings
            alpha: rescue signal strength
            stress_params: dict for StressBistableSwitch
            ca_params: dict for Ca2+ dynamics
            neighborhood: 'vonNeumann' (4-connected) or 'moore' (8-connected)
            parallel_threshold: use thread parallelism if num_embryos >= this
            D_F: diffusion rate of stress field on embryo lattice
            gamma_F: decay rate of stress field
            diffusion_substeps: number of sub-steps per bio step for field diffusion
            initial_stress: initial S value for all cells in all embryos (default: 0.0)
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_embryos = grid_rows * grid_cols
        self.damping_map = damping_map
        self.damping_map_flat = damping_map.flatten()
        self.alpha = alpha
        self.stress_params = stress_params
        self.ca_params = ca_params
        self.parallel_threshold = parallel_threshold
        self.use_parallel = self.num_embryos >= parallel_threshold
        self.initial_stress = initial_stress

        # Diffusive stress field parameters
        self.D_F = D_F
        self.gamma_F = gamma_F
        self.n_substeps = diffusion_substeps
        self.use_diffusive_field = D_F > 0
        self.F = np.zeros(self.num_embryos)  # diffusive stress field

        # Build embryo-level adjacency (needed before diagnostics)
        if neighborhood == 'moore':
            self.embryo_adj = build_moore_adjacency(grid_rows, grid_cols)
            self.max_degree = 8  # Moore neighborhood
        else:
            self.embryo_adj = build_vonneumann_adjacency(grid_rows, grid_cols)
            self.max_degree = 4  # von Neumann neighborhood
        self.neighbor_count = self.embryo_adj.sum(axis=1)

        if self.use_diffusive_field:
            if gamma_F > 0:
                lam = np.sqrt(D_F / gamma_F)
                print(f"Diffusive field (absorbing BC): D_F={D_F}, gamma_F={gamma_F}, "
                      f"lambda={lam:.1f} lattice spacings, "
                      f"{diffusion_substeps} sub-steps/bio-step")
            else:
                print(f"Diffusive field (absorbing BC): D_F={D_F}, gamma_F=0 (no bulk decay), "
                      f"{diffusion_substeps} sub-steps/bio-step")
            # Diagnostic: estimate boundary leakage and critical alpha
            min_neighbors = min(self.neighbor_count)
            worst_k = self.max_degree - min_neighbors  # missing neighbors for worst-case cell
            gamma_eff_worst = gamma_F + worst_k * D_F
            gamma_eff_interior = gamma_F  # interior cells (k=0)
            print(f"  Boundary leakage: worst-case gamma_eff={gamma_eff_worst:.2f} "
                  f"(k={worst_k} missing neighbors), "
                  f"interior gamma_eff={gamma_eff_interior:.4f}")
            e_est = 0.5  # typical emission
            alpha_crit_worst = gamma_eff_worst / e_est
            print(f"  Critical alpha for worst-case cell: {alpha_crit_worst:.1f} "
                  f"(current alpha={alpha}{'  OK' if alpha > alpha_crit_worst else '  << SUBCRITICAL'})")
        else:
            print("Diffusive field DISABLED (D_F=0), using mean_neighbor_stress")

        # Instantiate embryos
        self._instantiate_embryos()

    def _instantiate_embryos(self):
        """Create model instances and stress switches for each embryo."""
        self.embryos = []

        # Load undamped parameters once to get grid_size and cell adjacency
        ref_params = load_model_parameters(grn_damping=1.0)
        self.cell_grid_size = ref_params['latticeDims'][0]
        num_cells = self.cell_grid_size * self.cell_grid_size

        utils = utilities.utilities()
        self.cell_adjacency = utils.computeLatticeAdjacencyMatrix(
            latticeDims=(self.cell_grid_size, self.cell_grid_size),
            periodicBoundary=False
        )

        print(f"Instantiating {self.num_embryos} embryos "
              f"({self.grid_rows}x{self.grid_cols} grid, "
              f"{self.cell_grid_size}x{self.cell_grid_size} cells each)...")

        for idx in range(self.num_embryos):
            # Load with full (undamped) weights -- we manage damping dynamically
            params = load_model_parameters(grn_damping=1.0)
            num_samples = params["simParameters"]["numSamples"]
            initial_values = copy.deepcopy(params["simParameters"]["initialValues"])
            external_inputs = copy.deepcopy(params["simParameters"]["externalInputs"])
            clamp_params = copy.deepcopy(params["clampParameters"])

            bio_model = model(params, numBasicSamples=num_samples)
            bio_model.setExperimentalConditions((initial_values, num_samples))

            # Store original undamped weights
            original_grn_weights = bio_model.geneNetwork.tissueGRNWeights.clone()
            has_ligand_weights = (
                hasattr(bio_model.electricNetwork, 'GRNtoLigandWeights') and
                bio_model.electricNetwork.GRNtoLigandWeights is not None and
                not isinstance(bio_model.electricNetwork.GRNtoLigandWeights, bool)
            )
            original_ligand_weights = None
            if has_ligand_weights:
                original_ligand_weights = bio_model.electricNetwork.GRNtoLigandWeights.clone()

            # Create stress switch for this embryo
            stress_switch = StressBistableSwitch(
                num_cells=num_cells,
                adjacency_matrix=self.cell_adjacency,
                params=self.ca_params,
                device='cpu',
                dtype=torch.float32,
            )
            stress_switch.set_params_from_tensors(
                **{k: torch.tensor(v, dtype=torch.float32) for k, v in self.stress_params.items()}
            )

            self.embryos.append({
                'bio_model': bio_model,
                'stress_switch': stress_switch,
                'external_inputs': external_inputs,
                'clamp_params': clamp_params,
                'original_grn_weights': original_grn_weights,
                'original_ligand_weights': original_ligand_weights,
                'has_ligand_weights': has_ligand_weights,
            })

            if (idx + 1) % 10 == 0 or idx == self.num_embryos - 1:
                print(f"  Instantiated {idx + 1}/{self.num_embryos} embryos")

        # Apply initial stress level to all cells in all embryos
        if self.initial_stress > 0.0:
            print(f"  Setting initial stress S = {self.initial_stress:.4f} for all embryos")
            for idx in range(self.num_embryos):
                self.embryos[idx]['stress_switch'].S = torch.full(
                    (num_cells,), self.initial_stress, dtype=torch.float32
                )

        # Apply initial damping (before first sim step)
        for idx in range(self.num_embryos):
            base_d = self.damping_map_flat[idx]
            self._apply_effective_damping(idx, base_d)

    def _apply_effective_damping(self, idx, eff_damp):
        """Scale GRN weights for embryo idx by effective damping."""
        embryo = self.embryos[idx]
        embryo['bio_model'].geneNetwork.tissueGRNWeights = (
            embryo['original_grn_weights'] * eff_damp
        )
        if embryo['has_ligand_weights']:
            embryo['bio_model'].electricNetwork.GRNtoLigandWeights = (
                embryo['original_ligand_weights'] * eff_damp
            )

    def shuffle_embryos(self, preserve_field=False):
        """Randomly permute embryo positions on the grid, preserving internal states.

        Each embryo keeps its full internal state (Vmem, stress, Ca2+, GRN weights)
        but is moved to a new grid position, changing its neighbors.

        Args:
            preserve_field: if True, the diffusive stress field F stays fixed
                in place — embryos inherit the field value at their new position.
                If False (default), F is permuted along with the embryos so each
                embryo carries its field value to the new position.

        Returns:
            perm: the permutation array used
        """
        perm = np.random.permutation(self.num_embryos)
        self.embryos = [self.embryos[i] for i in perm]
        self.damping_map_flat = self.damping_map_flat[perm]
        self.damping_map = self.damping_map_flat.reshape(self.grid_rows, self.grid_cols)
        if not preserve_field:
            self.F = self.F[perm]
        return perm

    def _compute_mean_neighbor_stress(self):
        """Compute mean stress of each embryo's lattice neighbors."""
        stress_vals = np.array([
            self.embryos[idx]['stress_switch'].get_embryo_stress().item()
            for idx in range(self.num_embryos)
        ])
        neighbor_sum = self.embryo_adj @ stress_vals
        mean_neighbor = neighbor_sum / np.maximum(self.neighbor_count, 1)
        return mean_neighbor

    def _diffuse_stress_field(self):
        """Solve reaction-diffusion for the stress field on the embryo lattice.

        dF/dt = D_F * laplacian(F) - gamma_F * F + emission

        emission_i = mean(S_i) for each embryo (includes self-contribution).

        Uses ABSORBING boundary conditions: boundary cells diffuse to the
        exterior (F=0 outside), so the Laplacian uses max_degree (4 for
        von Neumann) instead of actual degree. This creates group-size
        dependence: boundary cells lose signal to the exterior, while
        interior cells in large groups accumulate stronger F.
        """
        emission = np.array([
            self.embryos[idx]['stress_switch'].get_embryo_stress().item()
            for idx in range(self.num_embryos)
        ])

        dt_sub = 1.0 / self.n_substeps
        for _ in range(self.n_substeps):
            # Absorbing BC: use max_degree so boundary cells leak to exterior
            laplacian = self.embryo_adj @ self.F - self.max_degree * self.F
            dF_dt = self.D_F * laplacian - self.gamma_F * self.F + emission
            self.F = self.F + dt_sub * dF_dt
            self.F = np.clip(self.F, 0.0, None)

    def _diffuse_stress_field_with_emission(self, emission):
        """Diffuse stress field with a provided emission array (for equilibration)."""
        dt_sub = 1.0 / self.n_substeps
        for _ in range(self.n_substeps):
            # Absorbing BC: use max_degree so boundary cells leak to exterior
            laplacian = self.embryo_adj @ self.F - self.max_degree * self.F
            dF_dt = self.D_F * laplacian - self.gamma_F * self.F + emission
            self.F = self.F + dt_sub * dF_dt
            self.F = np.clip(self.F, 0.0, None)

    def _step_single_embryo_bio(self, idx, t):
        """Run one bioelectric sim step for a single embryo."""
        embryo = self.embryos[idx]
        embryo['bio_model'].simulate(
            externalInputs=embryo['external_inputs'],
            clampParameters=embryo['clamp_params'],
            perturbation=None,
            fieldModulation=False,
            numSimIters=1,
            outerIter=t,
        )

    def run(self, num_bio_steps, num_stress_equil_steps=500, vmem_ref=None,
            shuffle_time=None, shuffle_preserve_field=False,
            snapshot_times=None):
        """
        Main synchronized simulation loop.

        Args:
            num_bio_steps: number of bioelectric simulation steps
            num_stress_equil_steps: stress equilibration steps after bio sim
            vmem_ref: reference Vmem tensor for similarity tracking (optional)
            shuffle_time: if set, shuffle embryo grid positions at this bio step
            shuffle_preserve_field: if True, keep F in place during shuffle
            snapshot_times: list of bio steps at which to capture Vmem snapshots

        Returns:
            dict with stress_history, damping_history, field_history,
            similarity_history, final_vmem, final_stress, etc.
        """
        dt_ca = 0.01
        dt_stress = 0.1
        total_stress_steps = num_bio_steps + num_stress_equil_steps

        # Storage
        stress_history = np.zeros((total_stress_steps, self.num_embryos))
        damping_history = np.zeros((num_bio_steps, self.num_embryos))
        field_history = np.zeros((total_stress_steps, self.num_embryos))
        similarity_history = np.zeros((num_bio_steps, self.num_embryos)) if vmem_ref is not None else None

        use_threads = self.use_parallel
        max_workers = min(8, self.num_embryos) if use_threads else 1

        if use_threads:
            print(f"Using ThreadPoolExecutor with {max_workers} workers")

        t_start = time.time()

        # ---- Bioelectric + Stress concurrent phase ----
        shuffle_perm = None
        vmem_at_shuffle = None        # Vmem snapshot at the shuffle time point
        snapshot_set = set(snapshot_times) if snapshot_times else set()
        vmem_snapshots = {}             # {time: [tensor_per_embryo]}
        for t in range(num_bio_steps):
            # Capture Vmem snapshot at requested time points
            if t in snapshot_set:
                snap = []
                for idx in range(self.num_embryos):
                    snap.append(
                        self.embryos[idx]['bio_model'].electricNetwork.Vmem[0, :, 0].detach().cpu()
                    )
                vmem_snapshots[t] = snap

            # Shuffle embryo positions at the requested time point
            if shuffle_time is not None and t == shuffle_time:
                shuffle_perm = self.shuffle_embryos(preserve_field=shuffle_preserve_field)
                field_note = " (field preserved)" if shuffle_preserve_field else ""
                print(f"  *** SHUFFLED embryo grid positions at bio step {t}{field_note} ***")

            # Capture Vmem snapshot one step after shuffle (shows effect of new neighbors)
            if shuffle_time is not None and t == shuffle_time + 1:
                vmem_at_shuffle = []
                for idx in range(self.num_embryos):
                    vmem_at_shuffle.append(
                        self.embryos[idx]['bio_model'].electricNetwork.Vmem[0, :, 0].detach().cpu()
                    )
                vmem_snapshots[t] = vmem_at_shuffle

            # Step 1: All embryos advance bioelectric sim by 1 step
            if use_threads:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(self._step_single_embryo_bio, idx, t)
                        for idx in range(self.num_embryos)
                    ]
                    wait(futures)
                    # Propagate any exceptions
                    for f in futures:
                        f.result()
            else:
                for idx in range(self.num_embryos):
                    self._step_single_embryo_bio(idx, t)

            # Step 2: All embryos update stress (Ca2+ from Vmem, then stress step)
            for idx in range(self.num_embryos):
                vmem_flat = self.embryos[idx]['bio_model'].electricNetwork.Vmem[0, :, 0]
                self.embryos[idx]['stress_switch'].compute_ca_from_vmem(
                    vmem_flat.to(dtype=torch.float32), dt_ca
                )
                self.embryos[idx]['stress_switch'].step(dt_stress)

            # Step 3: Compute rescue signal and update damping
            if self.use_diffusive_field:
                self._diffuse_stress_field()
                rescue_signal = self.F
            else:
                rescue_signal = self._compute_mean_neighbor_stress()

            for idx in range(self.num_embryos):
                base_d = self.damping_map_flat[idx]
                eff_d = compute_effective_damping(base_d, rescue_signal[idx], self.alpha)
                self._apply_effective_damping(idx, eff_d)
                damping_history[t, idx] = eff_d
                stress_history[t, idx] = (
                    self.embryos[idx]['stress_switch'].get_embryo_stress().item()
                )
                field_history[t, idx] = rescue_signal[idx]

            # Track Vmem similarity to healthy reference
            if vmem_ref is not None:
                for idx in range(self.num_embryos):
                    vmem_now = self.embryos[idx]['bio_model'].electricNetwork.Vmem[0, :, 0].detach().cpu()
                    similarity_history[t, idx] = compute_vmem_similarity(vmem_now, vmem_ref)

            # Progress logging
            if (t + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = (t + 1) / elapsed
                eta = (num_bio_steps - t - 1) / rate if rate > 0 else 0
                field_str = (f"mean_F={self.F.mean():.4f}" if self.use_diffusive_field
                             else f"mean_nbr_S={rescue_signal.mean():.4f}")
                print(f"  Step {t+1}/{num_bio_steps}: "
                      f"mean_stress={stress_history[t].mean():.4f}, "
                      f"mean_eff_damping={damping_history[t].mean():.4f}, "
                      f"{field_str} "
                      f"[{rate:.1f} steps/s, ETA {eta:.0f}s]")

        # ---- Stress equilibration phase ----
        print(f"  Stress equilibration ({num_stress_equil_steps} steps)...")
        # Freeze emissions at final stress values for field diffusion
        final_emission = np.array([
            self.embryos[idx]['stress_switch'].get_embryo_stress().item()
            for idx in range(self.num_embryos)
        ])
        for t_eq in range(num_stress_equil_steps):
            for idx in range(self.num_embryos):
                ca_final = self.embryos[idx]['stress_switch'].Ca.detach().clone()
                self.embryos[idx]['stress_switch'].step(dt_stress, Ca=ca_final)
                stress_history[num_bio_steps + t_eq, idx] = (
                    self.embryos[idx]['stress_switch'].get_embryo_stress().item()
                )

            # Continue field diffusion during equilibration
            if self.use_diffusive_field:
                self._diffuse_stress_field_with_emission(final_emission)
                field_history[num_bio_steps + t_eq] = self.F
            else:
                field_history[num_bio_steps + t_eq] = self._compute_mean_neighbor_stress()

        # Collect final Vmem and similarity
        final_vmem = []
        for idx in range(self.num_embryos):
            final_vmem.append(
                self.embryos[idx]['bio_model'].electricNetwork.Vmem[0, :, 0].detach().cpu()
            )

        elapsed_total = time.time() - t_start
        print(f"  Total simulation time: {elapsed_total:.1f}s")

        return {
            'stress_history': stress_history,
            'damping_history': damping_history,
            'field_history': field_history,
            'final_vmem': final_vmem,
            'final_stress': stress_history[-1],
            'final_field': field_history[-1],
            'similarity_history': similarity_history,
            'num_bio_steps': num_bio_steps,
            'shuffle_perm': shuffle_perm,
            'shuffle_time': shuffle_time,
            'vmem_at_shuffle': vmem_at_shuffle,
            'vmem_snapshots': vmem_snapshots if vmem_snapshots else None,
        }


# ============================================================
# Reference embryo for Vmem similarity
# ============================================================
def run_reference_sim(num_bio_steps):
    """Run a healthy (damping=1.0) embryo to get reference Vmem."""
    print("Running healthy reference embryo (damping=1.0)...")
    params = load_model_parameters(grn_damping=1.0)
    num_samples = params["simParameters"]["numSamples"]
    initial_values = copy.deepcopy(params["simParameters"]["initialValues"])
    external_inputs = copy.deepcopy(params["simParameters"]["externalInputs"])
    clamp_params = copy.deepcopy(params["clampParameters"])

    bio_model = model(params, numBasicSamples=num_samples)
    bio_model.setExperimentalConditions((initial_values, num_samples))

    for t in range(num_bio_steps):
        bio_model.simulate(
            externalInputs=external_inputs,
            clampParameters=clamp_params,
            perturbation=None,
            fieldModulation=False,
            numSimIters=1,
            outerIter=t,
        )
        if (t + 1) % 500 == 0:
            v = bio_model.electricNetwork.Vmem[0, :, 0]
            print(f"  Ref iter {t+1}/{num_bio_steps}: "
                  f"Vmem mean={v.mean().item():.4f}V, std={v.std().item():.4f}V")

    vmem_ref = bio_model.electricNetwork.Vmem[0, :, 0].detach().cpu()
    grid_size = params['latticeDims'][0]
    print(f"  Reference Vmem: mean={vmem_ref.mean().item():.4f}V, "
          f"std={vmem_ref.std().item():.4f}V")
    return vmem_ref, grid_size


# ============================================================
# Save / Load simulation data
# ============================================================
def _convert_vmem_snapshots_to_numpy(snapshots):
    """Convert {time: [tensor, ...]} to {time: [ndarray, ...]} for saving."""
    if not snapshots:
        return None
    out = {}
    for t, vlist in snapshots.items():
        out[t] = [v.detach().cpu().numpy() if hasattr(v, 'numpy') else np.asarray(v)
                  for v in vlist]
    return out


def _convert_vmem_snapshots_from_numpy(snapshots):
    """Convert {time: [ndarray, ...]} back to {time: [tensor, ...]} for viz."""
    if not snapshots:
        return None
    out = {}
    for t, vlist in snapshots.items():
        out[t] = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                  for v in vlist]
    return out


def save_simulation_data(path, results, damping_map, grid_rows, grid_cols,
                         vmem_ref, cell_grid_size, sim_params):
    """
    Save all simulation data to a .dat file for later visualization.

    Converts torch tensors in results['final_vmem'] to numpy so the
    saved file has no torch dependency for loading.
    """
    import os

    # Convert final_vmem list of tensors to numpy
    final_vmem_np = []
    for v in results['final_vmem']:
        if hasattr(v, 'numpy'):
            final_vmem_np.append(v.detach().cpu().numpy())
        else:
            final_vmem_np.append(np.asarray(v))

    vmem_ref_np = vmem_ref.numpy() if hasattr(vmem_ref, 'numpy') else np.asarray(vmem_ref)

    save_dict = {
        'stress_history': results['stress_history'],
        'damping_history': results['damping_history'],
        'field_history': results['field_history'],
        'final_vmem': final_vmem_np,
        'final_stress': results['final_stress'],
        'final_field': results['final_field'],
        'similarity_history': results['similarity_history'],
        'num_bio_steps': results['num_bio_steps'],
        'shuffle_perm': results.get('shuffle_perm'),
        'shuffle_time': results.get('shuffle_time'),
        'vmem_snapshots': _convert_vmem_snapshots_to_numpy(results.get('vmem_snapshots')),
        'vmem_ref': vmem_ref_np,
        'damping_map': damping_map,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'cell_grid_size': cell_grid_size,
        'sim_params': sim_params,
    }
    torch.save(save_dict, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Saved simulation data to {path} ({size_mb:.1f} MB)")


def load_simulation_data(path):
    """
    Load saved simulation data.

    Returns:
        results: dict compatible with visualization functions
        damping_map: (rows, cols) numpy array
        grid_rows, grid_cols: int
        vmem_ref: numpy array (or torch tensor)
        cell_grid_size: int
        sim_params: dict
    """
    print(f"Loading simulation data from {path}...")
    data = torch.load(path, weights_only=False)

    # Convert final_vmem numpy arrays back to torch tensors for compatibility
    final_vmem = []
    for v in data['final_vmem']:
        if isinstance(v, np.ndarray):
            final_vmem.append(torch.from_numpy(v))
        else:
            final_vmem.append(v)

    results = {
        'stress_history': data['stress_history'],
        'damping_history': data['damping_history'],
        'field_history': data['field_history'],
        'final_vmem': final_vmem,
        'final_stress': data['final_stress'],
        'final_field': data['final_field'],
        'similarity_history': data['similarity_history'],
        'num_bio_steps': data['num_bio_steps'],
        'shuffle_perm': data.get('shuffle_perm'),
        'shuffle_time': data.get('shuffle_time'),
        'vmem_snapshots': _convert_vmem_snapshots_from_numpy(data.get('vmem_snapshots')),
    }

    grid_rows = int(data['grid_rows'])
    grid_cols = int(data['grid_cols'])
    cell_grid_size = int(data['cell_grid_size'])
    N = grid_rows * grid_cols

    vmem_ref = data['vmem_ref']
    if isinstance(vmem_ref, np.ndarray):
        vmem_ref = torch.from_numpy(vmem_ref)

    print(f"  Grid: {grid_rows}x{grid_cols} = {N} embryos")
    print(f"  Cell grid: {cell_grid_size}x{cell_grid_size}")
    print(f"  Bio steps: {data['num_bio_steps']}")
    print(f"  Sim params: {data.get('sim_params', {})}")

    return (results, data['damping_map'], grid_rows, grid_cols,
            vmem_ref, cell_grid_size, data.get('sim_params', {}))


# ============================================================
# Visualization
# ============================================================
def visualize_group_rescue(results, damping_map, grid_rows, grid_cols,
                           alpha, output_path, vmem_ref=None,
                           rescue_threshold=0.5):
    """
    Multi-panel visualization of group rescue results.

    Row 1: Base damping map | Final stress heatmap | Vmem similarity heatmap
    Row 2: Stress timeseries by group | Effective damping timeseries by group
    Row 3: Vmem similarity timeseries | Rescue rate timeseries
    """
    num_embryos = grid_rows * grid_cols
    damping_flat = damping_map.flatten()

    # Compute Vmem similarity for each embryo
    vmem_sims = np.zeros(num_embryos)
    if vmem_ref is not None:
        for idx in range(num_embryos):
            vmem_sims[idx] = compute_vmem_similarity(results['final_vmem'][idx], vmem_ref)

    # Unique damping levels for grouping
    unique_dampings = sorted(set(damping_flat))
    group_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(unique_dampings)))
    damping_to_color = {d: group_colors[i] for i, d in enumerate(unique_dampings)}

    has_similarity = (results.get('similarity_history') is not None)

    # Figure layout: 2 or 3 rows (heatmaps + timeseries + optional similarity)
    n_rows = 3 if has_similarity else 2
    fig = plt.figure(figsize=(16, 5 * n_rows))

    height_ratios = [1, 1, 1] if has_similarity else [1, 1]
    outer_gs = fig.add_gridspec(
        n_rows, 1,
        height_ratios=height_ratios,
        hspace=0.40,
        left=0.06, right=0.96, top=0.94, bottom=0.05,
    )

    # ---- Row 1: Summary heatmaps (3 panels) ----
    row1_gs = outer_gs[0].subgridspec(1, 3, wspace=0.40)

    # Panel 1: Base Damping Map
    ax1 = fig.add_subplot(row1_gs[0, 0])
    im1 = ax1.imshow(damping_map, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
    ax1.set_title('Base GRN Damping', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    txt_size = max(6, 11 - max(grid_rows, grid_cols))
    if num_embryos <= 100:
        for i in range(grid_rows):
            for j in range(grid_cols):
                ax1.text(j, i, f'{damping_map[i, j]:.2f}', ha='center', va='center',
                         fontsize=txt_size, color='black')
    plt.colorbar(im1, ax=ax1, fraction=0.046, label='Damping')

    # Panel 2: Final Stress Heatmap
    ax2 = fig.add_subplot(row1_gs[0, 1])
    stress_grid = results['final_stress'].reshape(grid_rows, grid_cols)
    im2 = ax2.imshow(stress_grid, cmap='hot', vmin=0, vmax=1, aspect='equal')
    ax2.set_title('Final Stress (S)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Column')
    if num_embryos <= 100:
        for i in range(grid_rows):
            for j in range(grid_cols):
                val = stress_grid[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=txt_size, color=color)
    plt.colorbar(im2, ax=ax2, fraction=0.046, label='Stress')

    # Panel 3: Vmem Similarity Heatmap
    ax3 = fig.add_subplot(row1_gs[0, 2])
    sim_grid = vmem_sims.reshape(grid_rows, grid_cols)
    im3 = ax3.imshow(sim_grid, cmap='RdYlGn', vmin=-0.2, vmax=1, aspect='equal')
    ax3.set_title('Vmem Similarity to Healthy', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Column')
    if num_embryos <= 100:
        for i in range(grid_rows):
            for j in range(grid_cols):
                val = sim_grid[i, j]
                color = 'white' if val < 0.4 else 'black'
                ax3.text(j, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=txt_size, color=color)
    plt.colorbar(im3, ax=ax3, fraction=0.046, label='Pearson r')

    # ---- Row 2: Timeseries (3 panels) ----
    row2_gs = outer_gs[1].subgridspec(1, 3, wspace=0.30)
    ax4 = fig.add_subplot(row2_gs[0, 0])
    ax5 = fig.add_subplot(row2_gs[0, 1])
    ax6 = fig.add_subplot(row2_gs[0, 2])

    num_bio_steps = results['num_bio_steps']

    # For many unique damping levels (e.g. dampingRange), bin them into groups
    max_legend_entries = 8
    if len(unique_dampings) > max_legend_entries:
        # Bin into quantile groups
        n_bins = max_legend_entries
        bin_edges = np.linspace(min(unique_dampings), max(unique_dampings), n_bins + 1)
        bin_labels = []
        bin_indices_list = []
        bin_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_bins))
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b < n_bins - 1:
                mask = (damping_flat >= lo) & (damping_flat < hi)
            else:
                mask = (damping_flat >= lo) & (damping_flat <= hi)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            bin_labels.append(f'd=[{lo:.2f},{hi:.2f}]')
            bin_indices_list.append(indices)
        plot_groups = list(zip(bin_labels, bin_indices_list,
                               bin_colors[:len(bin_labels)]))
    else:
        plot_groups = []
        for d_level in unique_dampings:
            mask = damping_flat == d_level
            indices = np.where(mask)[0]
            color = damping_to_color[d_level]
            plot_groups.append((f'd={d_level:.2f} (n={len(indices)})', indices, color))

    # Plot grouped timeseries
    for label, indices, color in plot_groups:
        # Stress timeseries
        stress_group = results['stress_history'][:, indices]
        mean_stress = stress_group.mean(axis=1)
        std_stress = stress_group.std(axis=1)
        t_axis = np.arange(len(mean_stress))
        ax4.plot(t_axis, mean_stress, color=color, lw=1.5, label=label)
        ax4.fill_between(t_axis, mean_stress - std_stress, mean_stress + std_stress,
                         color=color, alpha=0.15)

        # Effective damping timeseries (only during bio phase)
        damp_group = results['damping_history'][:, indices]
        mean_damp = damp_group.mean(axis=1)
        std_damp = damp_group.std(axis=1)
        t_damp = np.arange(num_bio_steps)
        ax5.plot(t_damp, mean_damp, color=color, lw=1.5, label=label)
        ax5.fill_between(t_damp, mean_damp - std_damp, mean_damp + std_damp,
                         color=color, alpha=0.15)

    # Field (rescue signal) timeseries
    has_field = 'field_history' in results
    if has_field:
        for label, indices, color in plot_groups:
            field_group = results['field_history'][:, indices]
            mean_field = field_group.mean(axis=1)
            std_field = field_group.std(axis=1)
            t_axis_f = np.arange(len(mean_field))
            ax6.plot(t_axis_f, mean_field, color=color, lw=1.5, label=label)
            ax6.fill_between(t_axis_f, mean_field - std_field, mean_field + std_field,
                             color=color, alpha=0.15)

    ax4.axvline(num_bio_steps, color='gray', ls='--', lw=0.8, label='equil. start')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Mean Stress')
    ax4.set_title('Stress Timeseries by Damping Group', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7, loc='upper left', ncol=max(1, len(plot_groups) // 4))
    ax4.grid(alpha=0.3)
    ax4.set_ylim(-0.05, 1.05)

    ax5.set_xlabel('Bio Step')
    ax5.set_ylabel('Effective Damping')
    ax5.set_title(f'Effective Damping (alpha={alpha:.1f})', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=7, ncol=max(1, len(plot_groups) // 4))
    ax5.grid(alpha=0.3)
    ax5.set_ylim(-0.05, 1.05)

    if has_field:
        ax6.axvline(num_bio_steps, color='gray', ls='--', lw=0.8, label='equil. start')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Rescue Signal (F)')
        ax6.set_title('Diffusive Field by Damping Group', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=7, ncol=max(1, len(plot_groups) // 4))
        ax6.grid(alpha=0.3)
    else:
        ax6.axis('off')

    # ---- Row 3: Vmem Similarity + Rescue Rate timeseries ----
    if has_similarity:
        row3_gs = outer_gs[2].subgridspec(1, 2, wspace=0.30)
        ax7 = fig.add_subplot(row3_gs[0, 0])
        ax8 = fig.add_subplot(row3_gs[0, 1])

        sim_hist = results['similarity_history']  # (num_bio_steps, num_embryos)
        t_sim = np.arange(num_bio_steps)

        for label, indices, color in plot_groups:
            sim_group = sim_hist[:, indices]
            mean_sim = sim_group.mean(axis=1)
            std_sim = sim_group.std(axis=1)
            ax7.plot(t_sim, mean_sim, color=color, lw=1.5, label=label)
            ax7.fill_between(t_sim, mean_sim - std_sim, mean_sim + std_sim,
                             color=color, alpha=0.15)

        # Report final values as text annotation
        final_texts = []
        for label, indices, color in plot_groups:
            final_mean = sim_hist[-1, indices].mean()
            final_std = sim_hist[-1, indices].std()
            final_texts.append(f'{label}: r={final_mean:.3f}\u00b1{final_std:.3f}')
        annotation = 'Final:  ' + '    '.join(final_texts)
        ax7.text(0.5, -0.15, annotation, transform=ax7.transAxes,
                 fontsize=9, ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           edgecolor='gray', alpha=0.9))

        ax7.axhline(rescue_threshold, color='red', ls=':', lw=1.0, alpha=0.6,
                     label=f'threshold={rescue_threshold}')
        ax7.set_xlabel('Bio Step')
        ax7.set_ylabel('Vmem Similarity (Pearson r)')
        ax7.set_title('Vmem Similarity to Healthy Reference Over Time',
                       fontsize=11, fontweight='bold')
        ax7.legend(fontsize=7, ncol=max(1, len(plot_groups) // 4))
        ax7.grid(alpha=0.3)
        ax7.set_ylim(-0.3, 1.05)

        # Rescue rate: fraction of embryos with similarity > threshold
        # Overall rescue rate (all embryos)
        rescue_rate_all = (sim_hist > rescue_threshold).mean(axis=1)
        ax8.plot(t_sim, rescue_rate_all, color='black', lw=2.0, label='All embryos')

        # Per damping group rescue rate
        for label, indices, color in plot_groups:
            sim_group = sim_hist[:, indices]
            rescue_rate_group = (sim_group > rescue_threshold).mean(axis=1)
            ax8.plot(t_sim, rescue_rate_group, color=color, lw=1.5,
                     ls='--', label=label)

        # Annotate final rescue rate
        final_rate = rescue_rate_all[-1]
        ax8.text(0.5, -0.15,
                 f'Final rescue rate: {final_rate:.1%} '
                 f'({int(final_rate * num_embryos)}/{num_embryos} embryos '
                 f'with r > {rescue_threshold})',
                 transform=ax8.transAxes, fontsize=9, ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           edgecolor='gray', alpha=0.9))

        ax8.set_xlabel('Bio Step')
        ax8.set_ylabel('Rescue Rate')
        ax8.set_title(f'Rescue Rate (similarity > {rescue_threshold})',
                       fontsize=11, fontweight='bold')
        ax8.legend(fontsize=7, ncol=max(1, (len(plot_groups) + 1) // 4))
        ax8.grid(alpha=0.3)
        ax8.set_ylim(-0.05, 1.05)

    fig.suptitle(
        f'Group Stress-Based Rescue  |  {grid_rows}x{grid_cols} grid  |  '
        f'alpha={alpha:.1f}  |  {num_bio_steps} bio steps',
        fontsize=13, fontweight='bold',
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")


def visualize_vmem_grid(results, damping_map, grid_rows, grid_cols,
                        cell_grid_size, alpha, output_path):
    """
    Separate figure showing only the embryo Vmem heatmap grid.

    Each embryo's final Vmem pattern is shown as a small heatmap, arranged
    to match its position in the 2D embryo lattice.
    """
    import os
    num_embryos = grid_rows * grid_cols
    damping_flat = damping_map.flatten()

    all_vmem_np = [v.numpy() for v in results['final_vmem']]
    vmin = min(v.min() for v in all_vmem_np)
    vmax = max(v.max() for v in all_vmem_np)

    # For large grids, subsample
    max_panels = 100
    if num_embryos <= max_panels:
        vr, vc = grid_rows, grid_cols
        show_all = True
    else:
        vr = min(grid_rows, 10)
        vc = min(grid_cols, 10)
        show_all = False

    panel_size = 1.8
    fig, axes = plt.subplots(
        vr, vc,
        figsize=(vc * panel_size + 1.5, vr * panel_size + 1.5),
        squeeze=False,
    )

    for vi in range(vr):
        for vj in range(vc):
            ax = axes[vi][vj]

            if show_all:
                idx = vi * grid_cols + vj
            else:
                ri = int(vi * (grid_rows - 1) / max(vr - 1, 1))
                ci = int(vj * (grid_cols - 1) / max(vc - 1, 1))
                idx = ri * grid_cols + ci

            if idx >= num_embryos:
                ax.axis('off')
                continue

            vmem_pattern = all_vmem_np[idx].reshape(cell_grid_size, cell_grid_size)
            im = ax.imshow(vmem_pattern, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                           aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            d = damping_flat[idx]
            ax.set_title(f'd={d:.2f}', fontsize=8, pad=3)

    # Single shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.40, wspace=0.20)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Vmem (V)')

    fig.suptitle(
        f'Final Vmem Patterns  |  {grid_rows}x{grid_cols} grid  |  alpha={alpha:.1f}',
        fontsize=13, fontweight='bold',
    )

    # Derive output path from main output file
    base, ext = os.path.splitext(output_path)
    grid_path = f'{base}_vmem_grid{ext}'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"Saved Vmem grid: {grid_path}")


def visualize_vmem_snapshot_grid(vmem_list, damping_map, grid_rows, grid_cols,
                                 cell_grid_size, alpha, snapshot_time,
                                 output_path, label_prefix=''):
    """
    Vmem grid at a specific snapshot time (e.g., at the moment of shuffle).

    Same layout as visualize_vmem_grid but for a snapshot rather than final state.
    """
    import os
    num_embryos = grid_rows * grid_cols
    damping_flat = damping_map.flatten()

    all_vmem_np = [v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
                   for v in vmem_list]
    vmin = min(v.min() for v in all_vmem_np)
    vmax = max(v.max() for v in all_vmem_np)

    max_panels = 100
    if num_embryos <= max_panels:
        vr, vc = grid_rows, grid_cols
        show_all = True
    else:
        vr = min(grid_rows, 10)
        vc = min(grid_cols, 10)
        show_all = False

    panel_size = 1.8
    fig, axes = plt.subplots(
        vr, vc,
        figsize=(vc * panel_size + 1.5, vr * panel_size + 1.5),
        squeeze=False,
    )

    for vi in range(vr):
        for vj in range(vc):
            ax = axes[vi][vj]
            if show_all:
                idx = vi * grid_cols + vj
            else:
                ri = int(vi * (grid_rows - 1) / max(vr - 1, 1))
                ci = int(vj * (grid_cols - 1) / max(vc - 1, 1))
                idx = ri * grid_cols + ci

            if idx >= num_embryos:
                ax.axis('off')
                continue

            vmem_pattern = all_vmem_np[idx].reshape(cell_grid_size, cell_grid_size)
            im = ax.imshow(vmem_pattern, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                           aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            d = damping_flat[idx]
            ax.set_title(f'd={d:.2f}', fontsize=8, pad=3)

    fig.subplots_adjust(right=0.88, hspace=0.40, wspace=0.20)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Vmem (V)')

    title_prefix = f'{label_prefix}  |  ' if label_prefix else ''
    fig.suptitle(
        f'{title_prefix}Vmem at t={snapshot_time}  |  '
        f'{grid_rows}x{grid_cols} grid  |  alpha={alpha:.1f}',
        fontsize=13, fontweight='bold',
    )

    base, ext = os.path.splitext(output_path)
    snap_path = f'{base}_vmem_t{snapshot_time}{ext}'
    plt.savefig(snap_path, dpi=150, bbox_inches='tight')
    print(f"Saved Vmem snapshot grid (t={snapshot_time}): {snap_path}")


# ============================================================
# Shuffle experiment comparison visualization
# ============================================================
def visualize_shuffle_comparison(baseline_results, shuffle_results, damping_map,
                                 grid_rows, grid_cols, cell_grid_size,
                                 vmem_ref, alpha, output_path,
                                 preserve_field=False):
    """
    Compare unperturbed vs shuffled embryo patterns.

    Shows for each condition (columns):
      Row 1: Vmem similarity heatmap on the embryo grid
      Row 2: Final stress heatmap on the embryo grid
      Row 3: Sample Vmem patterns (mean across embryos)
    """
    import os

    conditions = [('Unperturbed', baseline_results)]
    for st in sorted(shuffle_results.keys()):
        conditions.append((f'Shuffle @ t={st}', shuffle_results[st]))

    n_conds = len(conditions)
    num_embryos = grid_rows * grid_cols
    txt_size = max(6, 11 - max(grid_rows, grid_cols))

    fig, axes = plt.subplots(3, n_conds, figsize=(5 * n_conds, 14), squeeze=False)

    for ci, (label, res) in enumerate(conditions):
        # Row 0: Vmem similarity heatmap
        vmem_sims = np.zeros(num_embryos)
        if vmem_ref is not None:
            for idx in range(num_embryos):
                vmem_sims[idx] = compute_vmem_similarity(res['final_vmem'][idx], vmem_ref)
        sim_grid = vmem_sims.reshape(grid_rows, grid_cols)
        ax = axes[0][ci]
        im = ax.imshow(sim_grid, cmap='RdYlGn', vmin=-0.2, vmax=1, aspect='equal')
        ax.set_title(f'{label}\nVmem Similarity', fontsize=10, fontweight='bold')
        ax.set_xlabel('Column')
        if ci == 0:
            ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, fraction=0.046)
        if num_embryos <= 100:
            for i in range(grid_rows):
                for j in range(grid_cols):
                    val = sim_grid[i, j]
                    color = 'white' if val < 0.4 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=txt_size, color=color)

        # Row 1: Final stress heatmap
        stress_grid = res['final_stress'].reshape(grid_rows, grid_cols)
        ax = axes[1][ci]
        im = ax.imshow(stress_grid, cmap='hot', vmin=0, vmax=1, aspect='equal')
        ax.set_title('Final Stress', fontsize=10, fontweight='bold')
        ax.set_xlabel('Column')
        if ci == 0:
            ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, fraction=0.046)
        if num_embryos <= 100:
            for i in range(grid_rows):
                for j in range(grid_cols):
                    val = stress_grid[i, j]
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=txt_size, color=color)

        # Row 2: Mean Vmem pattern across all embryos
        all_vmem_np = [v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
                       for v in res['final_vmem']]
        vmin_v = min(v.min() for v in all_vmem_np)
        vmax_v = max(v.max() for v in all_vmem_np)
        mean_vmem = np.mean(
            [v.reshape(cell_grid_size, cell_grid_size) for v in all_vmem_np], axis=0
        )
        ax = axes[2][ci]
        im = ax.imshow(mean_vmem, cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v, aspect='equal')
        ax.set_title('Mean Vmem Pattern', fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, label='Vmem (V)')

        # Annotate rescue rate
        if vmem_ref is not None:
            rescued = (vmem_sims > 0.5).sum()
            mean_sim = vmem_sims.mean()
            ax.set_xlabel(f'Rescue: {rescued}/{num_embryos}, mean r={mean_sim:.3f}',
                          fontsize=9)

    fig.suptitle(
        f'Shuffle Experiment  |  {grid_rows}x{grid_cols} grid  |  alpha={alpha:.1f}',
        fontsize=13, fontweight='bold',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    base, ext = os.path.splitext(output_path)
    preserve_tag = '_preserveF' if preserve_field else ''
    shuffle_path = f'{base}_shuffle_comparison{preserve_tag}{ext}'
    plt.savefig(shuffle_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved shuffle comparison: {shuffle_path}")


# ============================================================
# Parameter suffix for file naming
# ============================================================
def build_param_suffix(num_embryos, alpha, damping_map, num_bio_steps, D_F, gamma_F):
    """Build a descriptive suffix string from run parameters."""
    damp_min = damping_map.min()
    damp_max = damping_map.max()
    diff_str = f"_D{D_F}_g{gamma_F}" if D_F > 0 else "_noDiff"
    return (f"_g{num_embryos}_a{alpha}"
            f"_d{damp_min:.2f}-{damp_max:.2f}"
            f"_t{num_bio_steps}{diff_str}")


# ============================================================
# Simulation runner (shared by simulate and both modes)
# ============================================================
def run_simulation(args):
    """Run the group rescue simulation and return all data needed for visualization."""
    # Validate group size
    if args.groupSize < 1 or args.groupSize > 300:
        raise ValueError(f"groupSize must be between 1 and 300, got {args.groupSize}")

    # Derive grid dimensions
    if args.gridDims is not None:
        grid_rows, grid_cols = [int(x) for x in args.gridDims.split(',')]
    else:
        grid_rows, grid_cols = derive_grid_dims(args.groupSize)

    num_embryos = grid_rows * grid_cols

    print("=" * 60)
    print("GROUP STRESS-BASED RESCUE")
    print("=" * 60)
    print(f"Grid: {grid_rows} x {grid_cols} = {num_embryos} embryos")
    print(f"Alpha: {args.alpha}")
    print(f"Bio steps: {args.numBioSteps}")
    print(f"Stress equil steps: {args.numStressSteps}")
    print(f"Neighborhood: {args.neighborhood}")
    print(f"Initial stress: {args.initialStress}")
    print(f"Rescue threshold: {args.rescueThreshold}")

    # Build damping map
    damping_map = build_damping_map(args, grid_rows, grid_cols)
    print(f"\nDamping map ({grid_rows}x{grid_cols}):")
    print(damping_map)

    # Load stress parameters
    if args.stressParamsFile is not None:
        stress_params, ca_params = load_stress_params(args.stressParamsFile)
    else:
        stress_params, ca_params = get_default_stress_params()
        print("\nUsing DEFAULT stress parameters")

    # Run healthy reference
    vmem_ref, cell_grid_size = run_reference_sim(args.numBioSteps)

    # Create and run group simulation
    sim = GroupRescueSimulation(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        damping_map=damping_map,
        alpha=args.alpha,
        stress_params=stress_params,
        ca_params=ca_params,
        neighborhood=args.neighborhood,
        parallel_threshold=args.parallelThreshold,
        D_F=args.D_F,
        gamma_F=args.gamma_F,
        diffusion_substeps=args.diffusion_substeps,
        initial_stress=args.initialStress,
    )

    # Parse shuffle times before running baseline so we can capture snapshots
    shuffle_times = []
    if args.shuffleTimes is not None:
        shuffle_times = sorted([int(x) for x in args.shuffleTimes.split(',')])
        for st in shuffle_times:
            if st < 0 or st >= args.numBioSteps:
                raise ValueError(
                    f"shuffleTime {st} out of range [0, {args.numBioSteps})")

    print(f"\n{'='*60}")
    print("RUNNING GROUP SIMULATION")
    print(f"{'='*60}")
    results = sim.run(
        num_bio_steps=args.numBioSteps,
        num_stress_equil_steps=args.numStressSteps,
        vmem_ref=vmem_ref,
        snapshot_times=shuffle_times if shuffle_times else None,
    )

    # Build save path with parameter suffix
    import os
    suffix = build_param_suffix(num_embryos, args.alpha, damping_map,
                                args.numBioSteps, args.D_F, args.gamma_F)
    if args.saveData is not None:
        base, ext = os.path.splitext(args.saveData)
        save_path = f"{base}{suffix}{ext}"
    else:
        save_path = f"data/group_rescue_sim{suffix}.dat"

    # Store simulation parameters for reproducibility
    sim_params = {
        'groupSize': num_embryos, 'alpha': args.alpha,
        'numBioSteps': args.numBioSteps, 'numStressSteps': args.numStressSteps,
        'neighborhood': args.neighborhood, 'D_F': args.D_F, 'gamma_F': args.gamma_F,
        'diffusion_substeps': args.diffusion_substeps,
        'initialStress': args.initialStress,
        'stressParamsFile': args.stressParamsFile,
        'rescueThreshold': args.rescueThreshold,
        'save_path': save_path,
        'shuffle_times': shuffle_times,
        'shufflePreserveField': args.shufflePreserveField,
    }

    # Shuffle file tag: includes _preserveF when the flag is set
    preserve_tag = '_preserveF' if args.shufflePreserveField else ''

    # Save baseline simulation data
    save_simulation_data(save_path, results, damping_map, grid_rows, grid_cols,
                         vmem_ref, cell_grid_size, sim_params)

    # Run shuffle experiments and save each
    shuffle_results = {}
    for st in shuffle_times:
        print(f"\n{'='*60}")
        print(f"SHUFFLE EXPERIMENT: shuffle at bio step {st}")
        print(f"{'='*60}")

        sim_shuffle = GroupRescueSimulation(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            damping_map=damping_map.copy(),
            alpha=args.alpha,
            stress_params=stress_params,
            ca_params=ca_params,
            neighborhood=args.neighborhood,
            parallel_threshold=args.parallelThreshold,
            D_F=args.D_F,
            gamma_F=args.gamma_F,
            diffusion_substeps=args.diffusion_substeps,
            initial_stress=args.initialStress,
        )

        shuffle_results[st] = sim_shuffle.run(
            num_bio_steps=args.numBioSteps,
            num_stress_equil_steps=args.numStressSteps,
            vmem_ref=vmem_ref,
            shuffle_time=st,
            shuffle_preserve_field=args.shufflePreserveField,
            snapshot_times=[st],
        )

        # Save shuffle result
        shuffle_save = save_path.replace('.dat', f'_shuffle{st}{preserve_tag}.dat')
        save_simulation_data(shuffle_save, shuffle_results[st],
                             sim_shuffle.damping_map, grid_rows, grid_cols,
                             vmem_ref, cell_grid_size,
                             {**sim_params, 'shuffle_time': st})

    return (results, damping_map, grid_rows, grid_cols, vmem_ref,
            cell_grid_size, sim_params, shuffle_results)


# ============================================================
# Visualization runner (shared by visualize and both modes)
# ============================================================
def run_visualization(results, damping_map, grid_rows, grid_cols,
                      vmem_ref, cell_grid_size, sim_params, args,
                      shuffle_results=None):
    """Run all visualization on simulation results."""
    import os

    num_embryos = grid_rows * grid_cols
    alpha = sim_params.get('alpha', args.alpha)
    rescue_threshold = args.rescueThreshold

    # Build output path with parameter suffix
    suffix = build_param_suffix(
        num_embryos, alpha, damping_map, results['num_bio_steps'],
        sim_params.get('D_F', args.D_F), sim_params.get('gamma_F', args.gamma_F),
    )
    base, ext = os.path.splitext(args.outputFile)
    output_path = f"{base}{suffix}{ext}"

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    unique_dampings = sorted(set(damping_map.flatten()))
    for d_level in unique_dampings:
        mask = damping_map.flatten() == d_level
        indices = np.where(mask)[0]
        stresses = results['final_stress'][indices]
        sims = np.array([
            compute_vmem_similarity(results['final_vmem'][i], vmem_ref)
            for i in indices
        ])
        print(f"  Damping={d_level:.2f} (n={len(indices)}): "
              f"stress={stresses.mean():.4f}+/-{stresses.std():.4f}, "
              f"Vmem_sim={sims.mean():.4f}+/-{sims.std():.4f}")

    if results.get('similarity_history') is not None:
        sim_hist = results['similarity_history']
        final_rescued = (sim_hist[-1] > rescue_threshold).sum()
        print(f"\n  Rescue rate (r > {rescue_threshold}): "
              f"{final_rescued}/{num_embryos} = {final_rescued/num_embryos:.1%}")

    # Main visualization
    visualize_group_rescue(
        results=results,
        damping_map=damping_map,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        alpha=alpha,
        output_path=output_path,
        vmem_ref=vmem_ref,
        rescue_threshold=rescue_threshold,
    )
    visualize_vmem_grid(
        results=results,
        damping_map=damping_map,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        cell_grid_size=cell_grid_size,
        alpha=alpha,
        output_path=output_path,
    )

    # Baseline Vmem snapshots at shuffle time points
    baseline_snapshots = results.get('vmem_snapshots')
    if baseline_snapshots:
        for snap_t, snap_vmem in sorted(baseline_snapshots.items()):
            visualize_vmem_snapshot_grid(
                vmem_list=snap_vmem,
                damping_map=damping_map,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                cell_grid_size=cell_grid_size,
                alpha=alpha,
                snapshot_time=snap_t,
                output_path=output_path,
                label_prefix='Baseline',
            )

    # Per-shuffle full visualizations (same panels as the baseline)
    if shuffle_results and len(shuffle_results) > 0:
        preserve_tag = '_preserveF' if sim_params.get('shufflePreserveField', False) else ''
        for st, sres in shuffle_results.items():
            shuffle_output = output_path.replace('.png', f'_shuffle{st}{preserve_tag}.png')

            # Reconstruct shuffled damping map from stored perm
            if sres.get('shuffle_perm') is not None:
                shuffled_dmap = damping_map.flatten()[sres['shuffle_perm']]
                shuffled_dmap = shuffled_dmap.reshape(grid_rows, grid_cols)
            else:
                shuffled_dmap = damping_map

            visualize_group_rescue(
                results=sres,
                damping_map=shuffled_dmap,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                alpha=alpha,
                output_path=shuffle_output,
                vmem_ref=vmem_ref,
                rescue_threshold=rescue_threshold,
            )
            visualize_vmem_grid(
                results=sres,
                damping_map=shuffled_dmap,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                cell_grid_size=cell_grid_size,
                alpha=alpha,
                output_path=shuffle_output,
            )

            # Vmem snapshot at the shuffle time for this shuffled run
            shuffle_snapshots = sres.get('vmem_snapshots')
            snap_t = st + 1  # snapshot captured one step after shuffle
            if shuffle_snapshots and snap_t in shuffle_snapshots:
                visualize_vmem_snapshot_grid(
                    vmem_list=shuffle_snapshots[snap_t],
                    damping_map=shuffled_dmap,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    cell_grid_size=cell_grid_size,
                    alpha=alpha,
                    snapshot_time=snap_t,
                    output_path=shuffle_output,
                    label_prefix=f'Shuffle@t={st}+1',
                )

        # Combined comparison heatmaps
        visualize_shuffle_comparison(
            baseline_results=results,
            shuffle_results=shuffle_results,
            damping_map=damping_map,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            cell_grid_size=cell_grid_size,
            vmem_ref=vmem_ref,
            alpha=alpha,
            output_path=output_path,
            preserve_field=sim_params.get('shufflePreserveField', False),
        )

    return output_path


# ============================================================
# Load baseline + associated shuffle files
# ============================================================
def _resolve_load_path(load_path, args=None):
    """
    Resolve a --loadData path to an actual file on disk.

    Resolution order:
      1. Exact path as given.
      2. If CLI args are available, append the parameter suffix derived from
         those args and check for that specific file.  This ensures that
         ``--loadData data/groupRescue.dat --groupSize 100 ...``
         resolves to the g100 file, not a g25 sibling.
      3. Blind glob ``{base}*{ext}`` (excluding ``*_shuffle*``).

    Returns the resolved path (str) or raises FileNotFoundError.
    """
    import os
    import glob

    # 1. Exact path
    if os.path.exists(load_path):
        return load_path

    base, ext = os.path.splitext(load_path)

    # 2. Try suffix derived from CLI args
    if args is not None:
        try:
            if args.gridDims is not None:
                rows, cols = [int(x) for x in args.gridDims.split(',')]
            else:
                rows, cols = derive_grid_dims(args.groupSize)
            num_embryos = rows * cols
            # Build a damping map just to get min/max for the suffix
            damping_map = build_damping_map(args, rows, cols)
            suffix = build_param_suffix(num_embryos, args.alpha, damping_map,
                                        args.numBioSteps, args.D_F, args.gamma_F)
            suffixed = f"{base}{suffix}{ext}"
            if os.path.exists(suffixed):
                print(f"  Resolved --loadData to: {suffixed}")
                return suffixed
        except Exception:
            pass  # fall through to glob

    # 3. Blind glob fallback
    candidates = sorted(glob.glob(f"{base}*{ext}"))
    candidates = [c for c in candidates if '_shuffle' not in c]

    if len(candidates) == 1:
        print(f"  Resolved --loadData to: {candidates[0]}")
        return candidates[0]
    elif len(candidates) > 1:
        print(f"  Multiple candidates for '{load_path}':")
        for c in candidates:
            print(f"    {c}")
        print(f"  Using most recent: {candidates[-1]}")
        return candidates[-1]
    else:
        raise FileNotFoundError(
            f"No data file matching '{load_path}' (also tried '{base}*{ext}')")


def _load_baseline_and_shuffles(load_path, args=None):
    """
    Load the baseline .dat file and auto-discover associated shuffle files.

    Shuffle file discovery order:
      1. shuffle_times recorded in sim_params (from the simulation run)
      2. Glob for *_shuffle*.dat siblings of the baseline file

    Returns the same 8-tuple as run_simulation().
    """
    import os
    import glob

    resolved_path = _resolve_load_path(load_path, args=args)

    (results, damping_map, grid_rows, grid_cols,
     vmem_ref, cell_grid_size, sim_params) = load_simulation_data(resolved_path)

    # Determine which shuffle times to look for
    shuffle_times = sim_params.get('shuffle_times', [])

    # Reconstruct preserve_field tag for file discovery.
    # Check CLI args first (user may pass --shufflePreserveField at visualize time),
    # then fall back to what was stored in sim_params during simulation.
    preserve_field = False
    if args is not None and hasattr(args, 'shufflePreserveField'):
        preserve_field = args.shufflePreserveField
    if not preserve_field:
        preserve_field = sim_params.get('shufflePreserveField', False)
    # Propagate into sim_params so downstream visualization picks it up
    sim_params['shufflePreserveField'] = preserve_field
    preserve_tag = '_preserveF' if preserve_field else ''

    # If none recorded, glob for sibling files matching the pattern
    if not shuffle_times:
        base_no_ext, ext = os.path.splitext(resolved_path)
        pattern = f"{base_no_ext}_shuffle*{ext}"
        shuffle_files = sorted(glob.glob(pattern))
        for sf in shuffle_files:
            # Extract shuffle time from filename:
            #   ..._shuffle500.dat -> 500
            #   ..._shuffle500_preserveF.dat -> 500
            name = os.path.splitext(os.path.basename(sf))[0]
            name = name.replace('_preserveF', '')
            parts = name.rsplit('_shuffle', 1)
            if len(parts) == 2 and parts[1].isdigit():
                shuffle_times.append(int(parts[1]))

    # Load each shuffle file
    shuffle_results = {}
    for st in sorted(shuffle_times):
        shuffle_path = resolved_path.replace('.dat', f'_shuffle{st}{preserve_tag}.dat')
        if os.path.exists(shuffle_path):
            (sres, _, _, _, _, _, _) = load_simulation_data(shuffle_path)
            shuffle_results[st] = sres
        else:
            print(f"  WARNING: Shuffle data not found: {shuffle_path}")

    if shuffle_results:
        print(f"  Loaded {len(shuffle_results)} shuffle result(s): "
              f"t={list(shuffle_results.keys())}")

    return (results, damping_map, grid_rows, grid_cols, vmem_ref,
            cell_grid_size, sim_params, shuffle_results)


# ============================================================
# Main
# ============================================================
def main():
    args = parser.parse_args()

    if args.mode == 'visualize':
        # ---- Visualize-only: load from saved data ----
        if args.loadData is None:
            raise ValueError("--loadData is required when --mode=visualize")

        results, damping_map, grid_rows, grid_cols, vmem_ref, \
            cell_grid_size, sim_params, shuffle_results = \
            _load_baseline_and_shuffles(args.loadData, args=args)

        run_visualization(results, damping_map, grid_rows, grid_cols,
                          vmem_ref, cell_grid_size, sim_params, args,
                          shuffle_results=shuffle_results)

    elif args.mode == 'simulate':
        # ---- Simulate-only: run and save, no visualization ----
        (results, damping_map, grid_rows, grid_cols, vmem_ref,
         cell_grid_size, sim_params, shuffle_results) = run_simulation(args)
        print("\nSimulation complete. Use --mode visualize --loadData <path> to visualize.")

    else:
        # ---- Both: simulate + visualize ----
        if args.loadData is not None:
            results, damping_map, grid_rows, grid_cols, vmem_ref, \
                cell_grid_size, sim_params, shuffle_results = \
                _load_baseline_and_shuffles(args.loadData, args=args)
        else:
            (results, damping_map, grid_rows, grid_cols, vmem_ref,
             cell_grid_size, sim_params, shuffle_results) = run_simulation(args)

        run_visualization(results, damping_map, grid_rows, grid_cols,
                          vmem_ref, cell_grid_size, sim_params, args,
                          shuffle_results=shuffle_results)

    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
