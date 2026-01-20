#!/usr/bin/env python3
"""
Refined Bioelectric-Morphogen-Gene Integration

Implements dual-driver facial patterning with proper timescale hierarchy:
1. Bioelectric pattern (fast) - stigmergic model
2. Ca²⁺ dynamics (intermediate) - temporal integration
3. Morphogen gradients (slow) - diffusion-degradation
4. Gene expression (slowest) - developmental timescale

Features emerge from GENE EXPRESSION only (not voltage thresholds).
No A-P voltage gradient assumption (all features ~-60mV).
"""

import copy
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.serialization import add_safe_globals

from embryo import model
from bioelectricTransduction import BioelectricTransduction
from refinedFacialGRN import RefinedFacialGRN
from geneBasedFeatureClassifier import GeneBasedFeatureClassifier


def load_stigmergic_parameters(path: str):
    """Load stigmergic model parameters"""
    add_safe_globals([np.core.multiarray._reconstruct])
    params = torch.load(path, weights_only=False)
    if "ATPParameters" not in params:
        params["ATPParameters"] = None
    return params


def load_learned_grn_parameters(path: str):
    """
    Load learned GRN parameters from file.

    Args:
        path: Path to learned parameters file (.dat format)

    Returns:
        dict with learned parameter values (includes both learned and fixed params)
    """
    print(f"Loading learned GRN parameters from: {path}")
    data = torch.load(path, weights_only=False)

    # Extract parameters and bounds
    learned_params = {}
    param_bounds = data.get('parameter_bounds', {})

    # Apply sigmoid constraint to get actual values
    def apply_sigmoid_constraint(raw_param, min_val, max_val):
        return min_val + (max_val - min_val) * torch.sigmoid(raw_param)

    # Load learned parameters (from optimization)
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

    # Load fixed GRN parameters if present (from fixed_grn mode)
    # These were kept constant during learning and are already in constrained form
    if 'fixed_grn_params' in data:
        fixed_grn_params = data['fixed_grn_params']
        print(f"Found {len(fixed_grn_params)} fixed GRN parameters")
        for param_name, param_value in fixed_grn_params.items():
            # Fixed params are already constrained, just convert to float
            learned_params[param_name] = float(param_value) if isinstance(param_value, torch.Tensor) else param_value

    print(f"Loaded {len(learned_params)} total parameters")
    return learned_params


def run_bioelectric_simulation(params, num_iters=1000):
    """
    Run stigmergic bioelectric simulation to establish spatial Vmem pattern.

    Returns bioelectric model with converged voltage pattern (~-60mV everywhere).
    """
    print("=== Phase 1: Bioelectric Pattern Formation ===")
    print(f"Running stigmergic simulation for {num_iters} iterations...")

    sim_params = copy.deepcopy(params)
    num_samples = sim_params["simParameters"]["numSamples"]
    initial_values = copy.deepcopy(sim_params["simParameters"]["initialValues"])
    external_inputs = copy.deepcopy(sim_params["simParameters"]["externalInputs"])
    clamp_params = copy.deepcopy(sim_params["clampParameters"])

    bio_model = model(sim_params, numBasicSamples=num_samples)
    bio_model.setExperimentalConditions((initial_values, num_samples))
    bio_model.simulate(
        externalInputs=external_inputs,
        clampParameters=clamp_params,
        perturbation=None,
        fieldModulation=False,
        numSimIters=num_iters
    )

    # Check final voltage statistics
    final_vmem = bio_model.electricNetwork.Vmem
    print(f"Final Vmem: mean={final_vmem.mean().item():.4f}V, "
          f"std={final_vmem.std().item():.4f}V, "
          f"range=[{final_vmem.min().item():.4f}, {final_vmem.max().item():.4f}]V")

    return bio_model


def save_equilibrated_morphogens(morph_grids, grid_size, output_path='data/equilibrated_morphogens.json'):
    """
    Save equilibrated morphogen grids to JSON for visualization.

    Args:
        morph_grids: Dict of morphogen tensors
        grid_size: Grid dimension
        output_path: Path to save JSON file
    """
    # Convert tensors to nested lists
    morphogens_data = {}
    for morph_name, morph_tensor in morph_grids.items():
        # Convert to numpy then to nested list
        morph_array = morph_tensor.cpu().numpy()
        morphogens_data[morph_name] = morph_array.tolist()

    output_data = {
        'grid_size': grid_size,
        'morphogens': morphogens_data,
        'description': 'Equilibrated morphogen gradients after 1000 pre-equilibration steps with diffusion'
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved equilibrated morphogens to {output_path}")


def initialize_transduction_and_grn(lattice_dims, learned_params=None):
    """
    Initialize bioelectric transduction and GRN modules

    Args:
        lattice_dims: Grid dimensions
        learned_params: Optional dict of learned parameters to apply, including:
            - Bioelectric transduction: V_half_ca, k_ca, g_ca, tau_ca, alpha_lowpass
            - Bioelectric gating: ca_threshold, ca_sensitivity, and_threshold, and_sharpness
            - Morphogen: fgf8_strength, fgf8_degradation_factor, edn1_strength, etc.
            - Gene regulation: k_activation, k_degradation, K_self, n_self, etc.

    Returns:
        tuple of (transduction, facial_grn, classifier)
    """
    grid_size = lattice_dims[0]  # Assume square grid

    # Bioelectric transduction (Ca²⁺ dynamics)
    transduction = BioelectricTransduction(grid_size=grid_size, device='cpu')

    # Apply learned transduction parameters if provided
    if learned_params is not None:
        transduction_param_map = {
            'V_half_ca': 'V_half_ca',
            'k_ca': 'k_ca',
            'g_ca': 'g_ca',
            'tau_ca': 'tau_ca',
            'alpha_lowpass': 'alpha_lowpass',
        }

        transduction_params_found = False
        for learned_name, attr_name in transduction_param_map.items():
            if learned_name in learned_params:
                setattr(transduction, attr_name, torch.tensor(
                    learned_params[learned_name],
                    device=transduction.device,
                    dtype=transduction.dtype
                ))
                print(f"  Set transduction.{attr_name} = {learned_params[learned_name]:.4f}")
                transduction_params_found = True

        if transduction_params_found:
            print(f"Applied learned bioelectric transduction parameters")

    # Extract morphogen decay lengths for GRN initialization if provided
    shh_decay = learned_params.get('shh_decay_length', 0.8) if learned_params else 0.8
    fgf8_decay = learned_params.get('fgf8_decay_length', 0.3) if learned_params else 0.3
    edn1_decay = learned_params.get('edn1_decay_length', 0.6) if learned_params else 0.6

    # Refined facial GRN (dual driver architecture)
    facial_grn = RefinedFacialGRN(
        grid_size=grid_size,
        device='cpu',
        shh_decay_length=shh_decay,
        fgf8_decay_length=fgf8_decay,
        edn1_decay_length=edn1_decay
    )

    # Apply learned parameters if provided
    if learned_params is not None:
        print(f"Applying {len(learned_params)} learned parameters to GRN...")

        # Morphogen parameters
        morph_param_map = {
            'fgf8_strength': 'fgf8_strength',
            'fgf8_degradation_factor': 'fgf8_degradation_factor',
            'edn1_strength': 'edn1_strength',
            'edn1_degradation_factor': 'edn1_degradation_factor',
            'diffusion_rate': 'diffusion_rate',
        }

        for learned_name, grn_name in morph_param_map.items():
            if learned_name in learned_params:
                facial_grn.morphogen_params[grn_name] = torch.tensor(
                    learned_params[learned_name],
                    device=facial_grn.device,
                    dtype=facial_grn.dtype
                )
                print(f"  Set {grn_name} = {learned_params[learned_name]:.4f}")

        # Gene parameters
        gene_param_map = {
            'k_activation': 'k_activation',
            'k_degradation': 'k_degradation',
            'K_self': 'K_self',
            'n_self': 'n_self',
            'nose_shh_threshold': 'nose_shh_K',
            'nose_shh_cooperativity': 'nose_shh_n',
            'nose_edn1_threshold': 'nose_edn1_K',
            'mouth_edn1_threshold': 'mouth_edn1_K',
            'mouth_edn1_cooperativity': 'mouth_edn1_n',
        }

        for learned_name, grn_name in gene_param_map.items():
            if learned_name in learned_params:
                facial_grn.gene_params[grn_name] = torch.tensor(
                    learned_params[learned_name],
                    device=facial_grn.device,
                    dtype=facial_grn.dtype
                )
                print(f"  Set {grn_name} = {learned_params[learned_name]:.4f}")

        # Bioelectric gating parameters (Ca threshold, sensitivity, AND gate)
        bioelectric_param_map = {
            'ca_threshold': 'ca_threshold_override',
            'ca_sensitivity': 'ca_sensitivity_override',
            'and_threshold': 'and_threshold_override',
            'and_sharpness': 'and_sharpness_override',
        }

        bioelectric_params_found = False
        for learned_name, grn_attr in bioelectric_param_map.items():
            if learned_name in learned_params:
                setattr(facial_grn, grn_attr, torch.tensor(
                    learned_params[learned_name],
                    device=facial_grn.device,
                    dtype=facial_grn.dtype
                ))
                print(f"  Set {grn_attr} = {learned_params[learned_name]:.4f}")
                bioelectric_params_found = True

        if bioelectric_params_found:
            print(f"Applied learned bioelectric gating parameters")

    # Gene-based feature classifier
    classifier = GeneBasedFeatureClassifier(grid_size=grid_size, device='cpu')

    return transduction, facial_grn, classifier


def run_grn_only_dynamics(facial_grn, classifier, grid_size, num_iters=10000):
    """
    Run GRN-only dynamics without bioelectric coupling.

    This mode runs the morphogen-gene system autonomously:
    - Morphogen gradients establish spatial patterns
    - Genes respond to morphogen combinations
    - No bioelectric gating (bio_gate = 1.0 everywhere)

    Args:
        facial_grn: RefinedFacialGRN instance
        classifier: GeneBasedFeatureClassifier instance
        grid_size: Grid dimension
        num_iters: Number of GRN update iterations

    Returns:
        dict with final state and history
    """
    print("\n=== GRN-Only Dynamics (No Bioelectric Coupling) ===")
    print(f"Running {num_iters} GRN iterations...")

    # History tracking (sample every 100 iterations)
    sample_interval = 100
    history = {
        'genes': {gene: [] for gene in facial_grn.gene_names},
        'morphogens': {morph: [] for morph in facial_grn.morphogen_names},
        'features': [],
        'feature_counts': []
    }

    # Bioelectric signals set to None for GRN-only mode (bio_gate defaults to 1.0)
    bio_signals = None

    # Pre-equilibrate morphogens (match learning script: 1000 steps)
    print("Pre-equilibrating morphogens (1000 steps)...")
    for pre_step in range(1000):
        facial_grn.update_morphogens()

    morph_grids_eq = facial_grn.get_morphogen_grids()
    print(f"Morphogen equilibration complete:")
    print(f"  SHH: max={morph_grids_eq['shh'].max():.4f}")
    print(f"  FGF8: max={morph_grids_eq['fgf8'].max():.4f}")
    print(f"  EDN1: max={morph_grids_eq['edn1'].max():.4f}")

    # Save equilibrated morphogens for visualization
    save_equilibrated_morphogens(morph_grids_eq, grid_size)

    # Run GRN dynamics
    print(f"\nRunning GRN dynamics ({num_iters} iterations)...")
    for iter_idx in range(num_iters):
        facial_grn.update(bioelectric_signals=bio_signals)

        # Sample history periodically
        if (iter_idx + 1) % sample_interval == 0:
            gene_grids = facial_grn.get_gene_grids()
            morph_grids = facial_grn.get_morphogen_grids()
            classification = classifier.classify(gene_grids, mode='both')
            feature_grid = classification['features']
            feature_counts = classifier.summarize_features(feature_grid)

            for gene in facial_grn.gene_names:
                history['genes'][gene].append(gene_grids[gene].detach().clone().cpu())

            for morph in facial_grn.morphogen_names:
                history['morphogens'][morph].append(morph_grids[morph].detach().clone().cpu())

            history['features'].append(feature_grid.detach().clone().cpu())
            history['feature_counts'].append(feature_counts)

            if (iter_idx + 1) % 1000 == 0:
                print(f"  Iteration {iter_idx + 1}/{num_iters} - Features: {feature_counts}")

    # Get final state
    gene_grids = facial_grn.get_gene_grids()
    morph_grids = facial_grn.get_morphogen_grids()
    classification = classifier.classify(gene_grids, mode='both')
    feature_grid = classification['features']
    feature_counts = classifier.summarize_features(feature_grid)

    print(f"\nFinal feature counts: {feature_counts}")

    return {
        'final_genes': {k: v.detach().cpu() for k, v in gene_grids.items()},
        'final_morphogens': {k: v.detach().cpu() for k, v in morph_grids.items()},
        'final_features': feature_grid.detach().cpu(),
        'final_feature_counts': feature_counts,
        'history': history
    }


def run_integrated_dynamics(bio_model, transduction, facial_grn, classifier,
                            num_grn_iters=2000):
    """
    Run integrated bioelectric-morphogen-gene dynamics matching the learning script logic.

    This follows the same approach as learnRefinedFacialIntegration.py:
    1. Bioelectric simulation has already converged (from run_bioelectric_simulation)
    2. Pre-equilibrate morphogens (1000 steps)
    3. Pre-equilibrate Ca²⁺ with static Vmem (100 steps)
    4. Run GRN dynamics with fixed bioelectric signals (num_grn_iters steps)

    Args:
        bio_model: Stigmergic bioelectric model (already converged)
        transduction: BioelectricTransduction instance
        facial_grn: RefinedFacialGRN instance
        classifier: GeneBasedFeatureClassifier instance
        num_grn_iters: Number of GRN iterations (default: 2000, matching learning)

    Returns:
        dict with final state and history
    """
    print("\n=== Phase 2: Integrated Bioelectric-Morphogen-Gene Dynamics ===")

    lattice_dims = bio_model.parameters["latticeDims"]
    rows, cols = lattice_dims

    # Get static Vmem from converged bioelectric simulation
    vmem_grid = bio_model.electricNetwork.Vmem.view(rows, cols)

    # ============================================
    # Step 1: Pre-equilibrate morphogens (1000 steps)
    # ============================================
    print("Pre-equilibrating morphogens (1000 steps)...")
    for _ in range(1000):
        facial_grn.update_morphogens()

    morph_grids_eq = facial_grn.get_morphogen_grids()
    print(f"Morphogen equilibration complete:")
    print(f"  SHH: max={morph_grids_eq['shh'].max():.4f}")
    print(f"  FGF8: max={morph_grids_eq['fgf8'].max():.4f}")
    print(f"  EDN1: max={morph_grids_eq['edn1'].max():.4f}")

    # ============================================
    # Step 2: Pre-equilibrate Ca²⁺ with static Vmem (100 steps)
    # ============================================
    print("Pre-equilibrating Ca²⁺ (100 steps with static Vmem)...")
    for _ in range(100):
        transduction.update(vmem_grid, dt=0.01)

    Ca_eq = transduction.Ca
    print(f"Ca²⁺ equilibration complete:")
    print(f"  Ca mean={Ca_eq.mean():.4f}, max={Ca_eq.max():.4f}")

    # ============================================
    # Step 3: Run GRN dynamics with fixed bioelectric signals
    # ============================================
    print(f"\nRunning GRN dynamics ({num_grn_iters} iterations with fixed bioelectric signals)...")

    # Get bioelectric signals (will remain constant during GRN iterations)
    bio_signals = transduction.get_gene_modulation_signals()

    # History tracking (sample every 100 iterations)
    sample_interval = 100
    history = {
        'vmem': [],
        'Ca': [],
        'genes': {gene: [] for gene in facial_grn.gene_names},
        'morphogens': {morph: [] for morph in facial_grn.morphogen_names},
        'features': [],
        'feature_counts': []
    }

    for iter_idx in range(num_grn_iters):
        facial_grn.update_morphogens()
        facial_grn.update_genes(bioelectric_signals=bio_signals)

        # Sample history periodically
        if (iter_idx + 1) % sample_interval == 0:
            gene_grids = facial_grn.get_gene_grids()
            morph_grids = facial_grn.get_morphogen_grids()
            classification = classifier.classify(gene_grids, mode='both')
            feature_grid = classification['features']
            feature_counts = classifier.summarize_features(feature_grid)

            history['vmem'].append(vmem_grid.detach().clone().cpu())
            history['Ca'].append(transduction.Ca.detach().clone().cpu())

            for gene in facial_grn.gene_names:
                history['genes'][gene].append(gene_grids[gene].detach().clone().cpu())

            for morph in facial_grn.morphogen_names:
                history['morphogens'][morph].append(morph_grids[morph].detach().clone().cpu())

            history['features'].append(feature_grid.detach().clone().cpu())
            history['feature_counts'].append(feature_counts)

            if (iter_idx + 1) % 1000 == 0:
                print(f"  Iteration {iter_idx + 1}/{num_grn_iters} - Features: {feature_counts}")

    # ============================================
    # Final State
    # ============================================
    gene_grids = facial_grn.get_gene_grids()
    morph_grids = facial_grn.get_morphogen_grids()
    classification = classifier.classify(gene_grids, mode='both')
    feature_grid = classification['features']
    feature_counts = classifier.summarize_features(feature_grid)

    print(f"\nFinal feature counts: {feature_counts}")

    return {
        'final_vmem': vmem_grid.detach().cpu(),
        'final_Ca': transduction.Ca.detach().cpu(),
        'final_genes': {k: v.detach().cpu() for k, v in gene_grids.items()},
        'final_morphogens': {k: v.detach().cpu() for k, v in morph_grids.items()},
        'final_features': feature_grid.detach().cpu(),
        'final_feature_counts': feature_counts,
        'history': history
    }


def visualize_results(results, output_path='refined_facial_integration.png', grn_only=False):
    """
    Visualize final state: bioelectrics, morphogens, genes, features.

    Args:
        results: Results dict from run_integrated_dynamics or run_grn_only_dynamics
        output_path: Path to save visualization
        grn_only: If True, skip bioelectric visualizations (Vmem, Ca)
    """
    print("\n=== Generating Visualization ===")

    # Feature colormap
    feature_cmap = ListedColormap(['#f9f9f9', '#9b59b6', '#e67e22', '#2ecc71'])  # bone, eye, nose, mouth

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Bioelectric signals (or features if GRN-only)
    if not grn_only:
        ax = axes[0, 0]
        im = ax.imshow(results['final_vmem'].numpy(), cmap='coolwarm')
        ax.set_title('Vmem (Bioelectric Pattern)', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[0, 1]
        im = ax.imshow(results['final_Ca'].numpy(), cmap='viridis')
        ax.set_title('Ca²⁺ (Integrated Signal)', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        # For GRN-only mode, show features in first position
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')

    ax = axes[0, 2]
    im = ax.imshow(results['final_features'].numpy(), cmap=feature_cmap, vmin=0, vmax=3)
    ax.set_title('Features (from Genes)', fontsize=10, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046)
    cbar.set_ticklabels(['Bone', 'Eye', 'Nose', 'Mouth'])

    # Empty subplot
    axes[0, 3].axis('off')

    # Row 2: Morphogen gradients
    morphogens = results['final_morphogens']
    morph_names = ['shh', 'fgf8', 'edn1']
    for idx, morph_name in enumerate(morph_names):
        ax = axes[1, idx]
        im = ax.imshow(morphogens[morph_name].numpy(), cmap='plasma')
        ax.set_title(f'{morph_name.upper()} Gradient', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Empty subplot
    axes[1, 3].axis('off')

    # Row 3: Key gene expressions
    genes = results['final_genes']
    gene_display = ['pax6', 'lhx2', 'alx', 'dlx']
    gene_titles = ['Pax6 (Eye)', 'Lhx2 (Eye)', 'Alx (Nose)', 'Dlx (Mouth)']

    for idx, (gene, title) in enumerate(zip(gene_display, gene_titles)):
        ax = axes[2, idx]
        im = ax.imshow(genes[gene].numpy(), cmap='viridis', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Remove axes ticks
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    # Overall title
    if grn_only:
        fig.suptitle('GRN-Only Facial Patterning\n'
                     'Morphogen Gradients → Gene Expression → Features',
                     fontsize=14, fontweight='bold')
    else:
        fig.suptitle('Refined Dual-Driver Facial Patterning\n'
                     'Bioelectric + Morphogen → Gene Expression → Features',
                     fontsize=14, fontweight='bold')

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved visualization to {output_path}")


def print_summary(results, grn_only=False):
    """
    Print summary statistics

    Args:
        results: Results dict from run_integrated_dynamics or run_grn_only_dynamics
        grn_only: If True, skip bioelectric statistics (Vmem, Ca)
    """
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    print("\nFeature Counts:")
    for feature, count in results['final_feature_counts'].items():
        print(f"  {feature:8s}: {count:3d} cells")

    if not grn_only:
        print("\nBioelectric Statistics:")
        vmem = results['final_vmem']
        print(f"  Vmem:      mean={vmem.mean():.4f}V, std={vmem.std():.4f}V")

        Ca = results['final_Ca']
        print(f"  Ca²⁺:      mean={Ca.mean():.4f}, max={Ca.max():.4f}")

    print("\nKey Gene Expression Levels:")
    genes = results['final_genes']
    for gene in ['pax6', 'lhx2', 'alx', 'dlx', 'hand2', 'runx2']:
        expr = genes[gene]
        print(f"  {gene:6s}: mean={expr.mean():.3f}, max={expr.max():.3f}")

    print("\n" + "="*60)


def main():
    """Main execution"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Refined Bioelectric-Morphogen Facial Patterning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python run_refined_facial_integration.py

  # Run with learned parameters from file 44
  python run_refined_facial_integration.py --learned-params data/bestLearnedFacialParams_44.dat

  # Run with custom stigmergic parameters and learned GRN parameters
  python run_refined_facial_integration.py \\
    --stigmergic-params data/StigmergicModelParameters.dat \\
    --learned-params data/bestLearnedFacialParams_44.dat

  # Run GRN-only mode (no bioelectric coupling)
  python run_refined_facial_integration.py --grn-only --learned-params data/bestLearnedFacialParams_44.dat

  # Run GRN-only with custom number of iterations
  python run_refined_facial_integration.py --grn-only --grn-iters 5000
        """
    )
    parser.add_argument(
        '--learned-params',
        type=str,
        default=None,
        help='Path to learned GRN parameters file (.dat format). If not specified, uses default parameter values.'
    )
    parser.add_argument(
        '--stigmergic-params',
        type=str,
        default='data/StigmergicModelParameters.dat',
        help='Path to stigmergic bioelectric model parameters (default: data/StigmergicModelParameters.dat)'
    )
    parser.add_argument(
        '--grn-only',
        action='store_true',
        help='Run GRN-only mode without bioelectric coupling (bio_gate = 1.0 everywhere)'
    )
    parser.add_argument(
        '--grn-iters',
        type=int,
        default=2000,
        help='Number of GRN iterations for GRN-only mode (default: 2000, matching learning script)'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=11,
        help='Grid size for GRN-only mode (default: 11). Ignored in integrated mode.'
    )
    args = parser.parse_args()

    print("="*60)
    if args.grn_only:
        print("GRN-ONLY FACIAL PATTERNING (NO BIOELECTRIC COUPLING)")
    else:
        print("REFINED BIOELECTRIC-MORPHOGEN FACIAL PATTERNING")
    print("="*60)

    # Load learned GRN parameters if specified
    learned_params = None
    if args.learned_params is not None:
        learned_params = load_learned_grn_parameters(args.learned_params)
        print(f"Using learned GRN parameters from: {args.learned_params}")
    else:
        print("Using default GRN parameter values")

    # ========================================================================
    # GRN-ONLY MODE: Skip bioelectric simulation
    # ========================================================================
    if args.grn_only:
        print(f"\nRunning in GRN-only mode (grid size: {args.grid_size}x{args.grid_size})")

        # Initialize GRN and classifier only (no bioelectric components)
        lattice_dims = (args.grid_size, args.grid_size)
        _, facial_grn, classifier = initialize_transduction_and_grn(
            lattice_dims,
            learned_params=learned_params
        )

        # Run GRN-only dynamics
        results = run_grn_only_dynamics(
            facial_grn,
            classifier,
            args.grid_size,
            num_iters=args.grn_iters
        )

        # Visualize and summarize
        visualize_results(results, output_path='grn_only_facial_integration.png', grn_only=True)
        print_summary(results, grn_only=True)

        print("\n✅ GRN-only simulation complete!")
        print("\nKey design features:")
        print("  ✓ Morphogen gradients only (no bioelectric coupling)")
        print("  ✓ Uniform permissive bio_gate (1.0 everywhere)")
        print("  ✓ Features from gene expression (not voltage thresholds)")

    # ========================================================================
    # INTEGRATED MODE: Full bioelectric-morphogen-gene dynamics
    # ========================================================================
    else:
        # Load stigmergic parameters
        print(f"\nLoading stigmergic parameters from: {args.stigmergic_params}")
        params = load_stigmergic_parameters(args.stigmergic_params)

        # Phase 1: Bioelectric simulation
        bio_model = run_bioelectric_simulation(params, num_iters=params['simParameters']['numSimIters'])

        # Initialize modules (with optional learned parameters)
        lattice_dims = params["latticeDims"]
        rows, cols = lattice_dims
        transduction, facial_grn, classifier = initialize_transduction_and_grn(
            lattice_dims,
            learned_params=learned_params
        )

        # Initialize transduction module with bioelectric state
        vmem_grid = bio_model.electricNetwork.Vmem.view(rows, cols)
        transduction.update(vmem_grid, dt=0.01)

        # Phase 2: Integrated dynamics (with pre-equilibration inside)
        # Matches learnRefinedFacialIntegration.py: pre-equilibrate morphogens & Ca, then run GRN
        results = run_integrated_dynamics(
            bio_model, transduction, facial_grn, classifier,
            num_grn_iters=2000  # Match learning script default
        )

        # Visualize and summarize
        visualize_results(results, output_path='refined_facial_integration.png')
        print_summary(results)

        print("\n✅ Refined integration complete!")
        print("\nKey design features:")
        print("  ✓ Bioelectric pattern converged first (stigmergic model)")
        print("  ✓ Morphogens pre-equilibrated (1000 steps)")
        print("  ✓ Ca²⁺ pre-equilibrated with static Vmem (100 steps)")
        print("  ✓ GRN dynamics run with fixed bioelectric signals")
        print("  ✓ Matches learning script logic exactly")
        print("  ✓ Features from gene expression (not voltage thresholds)")


if __name__ == "__main__":
    main()
