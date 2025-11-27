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


def initialize_transduction_and_grn(lattice_dims):
    """Initialize bioelectric transduction and GRN modules"""
    grid_size = lattice_dims[0]  # Assume square grid

    # Bioelectric transduction (Ca²⁺ dynamics)
    transduction = BioelectricTransduction(grid_size=grid_size, device='cpu')

    # Refined facial GRN (dual driver architecture)
    facial_grn = RefinedFacialGRN(grid_size=grid_size, device='cpu')

    # Gene-based feature classifier
    classifier = GeneBasedFeatureClassifier(grid_size=grid_size, device='cpu')

    return transduction, facial_grn, classifier


def run_integrated_dynamics(bio_model, transduction, facial_grn, classifier,
                            num_cycles=5, bio_steps=100, grn_steps=500,
                            feedback_gain=0.02):
    """
    Run integrated bioelectric-morphogen-gene dynamics with proper timescale hierarchy.

    Timescale ratios (relative):
    - Bioelectric: 1x (fastest - dt=0.01)
    - Ca²⁺: 100x (intermediate - tau=1.0)
    - Morphogen: 1000x (slow - tau_morph=10.0)
    - Genes: 5000x (slowest - tau_gene=50.0)

    Args:
        bio_model: Stigmergic bioelectric model
        transduction: BioelectricTransduction instance
        facial_grn: RefinedFacialGRN instance
        classifier: GeneBasedFeatureClassifier instance
        num_cycles: Number of bidirectional coupling cycles
        bio_steps: Bioelectric steps per cycle
        grn_steps: GRN steps per cycle
        feedback_gain: Gene → Vmem feedback strength

    Returns:
        dict with final state and history
    """
    print("\n=== Phase 2: Integrated Bioelectric-Morphogen-Gene Dynamics ===")

    lattice_dims = bio_model.parameters["latticeDims"]
    rows, cols = lattice_dims

    # History tracking
    history = {
        'vmem': [],
        'Ca': [],
        'genes': {gene: [] for gene in facial_grn.gene_names},
        'morphogens': {morph: [] for morph in facial_grn.morphogen_names},
        'features': [],
        'feature_counts': []
    }

    ext_inputs_electric = {"gene": None}

    for cycle in range(num_cycles):
        print(f"\n--- Cycle {cycle+1}/{num_cycles} ---")

        # ============================================
        # Step 1: Bioelectric Dynamics (FAST)
        # ============================================
        print(f"  Bioelectric simulation ({bio_steps} steps)...")

        for bio_step in range(bio_steps):
            # Get current state
            vmem_flat = bio_model.electricNetwork.Vmem  # (numSamples, numCells, 1)
            vmem_grid = vmem_flat.view(rows, cols)

            # Update bioelectric transduction (Ca²⁺ dynamics)
            transduction.update(vmem_grid, dt=0.01)

            # Bioelectric simulation step
            bio_model.electricNetwork.simulate(
                externalInputs=ext_inputs_electric,
                numSimIters=1,
                outerIter=0,
                stochasticIonChannels=False,
                fieldModulation=False,
                setGradient=False,
                retainGradients=False
            )

        # ============================================
        # Step 2: Morphogen + Gene Dynamics (SLOW)
        # ============================================
        print(f"  GRN simulation ({grn_steps} steps)...")

        # Get bioelectric signals for GRN
        bio_signals = transduction.get_gene_modulation_signals()

        for grn_step in range(grn_steps):
            facial_grn.update(bioelectric_signals=bio_signals)

        # ============================================
        # Step 3: Feature Classification (from genes)
        # ============================================
        gene_grids = facial_grn.get_gene_grids()
        classification = classifier.classify(gene_grids, mode='both')
        feature_grid = classification['features']
        feature_counts = classifier.summarize_features(feature_grid)

        print(f"  Feature counts: {feature_counts}")

        # ============================================
        # Step 4: Gene → Voltage Feedback (WEAK)
        # ============================================
        # Apply weak feedback to preserve bioelectric structure
        # DISABLED to test pure AND constraint without positive feedback loops
        # bio_model.electricNetwork.apply_gene_voltage_feedback(
        #     gene_fields=gene_grids,
        #     gain=feedback_gain
        # )

        # ============================================
        # Record History
        # ============================================
        history['vmem'].append(vmem_grid.detach().clone().cpu())
        history['Ca'].append(transduction.Ca.detach().clone().cpu())

        for gene in facial_grn.gene_names:
            history['genes'][gene].append(gene_grids[gene].detach().clone().cpu())

        for morph in facial_grn.morphogen_names:
            morph_grids = facial_grn.get_morphogen_grids()
            history['morphogens'][morph].append(morph_grids[morph].detach().clone().cpu())

        history['features'].append(feature_grid.detach().clone().cpu())
        history['feature_counts'].append(feature_counts)

    return {
        'final_vmem': vmem_grid.detach().cpu(),
        'final_Ca': transduction.Ca.detach().cpu(),
        'final_genes': {k: v.detach().cpu() for k, v in gene_grids.items()},
        'final_morphogens': {k: v.detach().cpu() for k, v in facial_grn.get_morphogen_grids().items()},
        'final_features': feature_grid.detach().cpu(),
        'final_feature_counts': feature_counts,
        'history': history
    }


def visualize_results(results, output_path='refined_facial_integration.png'):
    """
    Visualize final state: bioelectrics, morphogens, genes, features.
    """
    print("\n=== Generating Visualization ===")

    # Feature colormap
    feature_cmap = ListedColormap(['#f9f9f9', '#9b59b6', '#e67e22', '#2ecc71'])  # bone, eye, nose, mouth

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Bioelectric signals
    ax = axes[0, 0]
    im = ax.imshow(results['final_vmem'].numpy(), cmap='coolwarm')
    ax.set_title('Vmem (Bioelectric Pattern)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(results['final_Ca'].numpy(), cmap='viridis')
    ax.set_title('Ca²⁺ (Integrated Signal)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

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
    fig.suptitle('Refined Dual-Driver Facial Patterning\n'
                 'Bioelectric + Morphogen → Gene Expression → Features',
                 fontsize=14, fontweight='bold')

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved visualization to {output_path}")


def print_summary(results):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    print("\nFeature Counts:")
    for feature, count in results['final_feature_counts'].items():
        print(f"  {feature:8s}: {count:3d} cells")

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
    print("="*60)
    print("REFINED BIOELECTRIC-MORPHOGEN FACIAL PATTERNING")
    print("="*60)

    # Load parameters
    params_path = "data/StigmergicModelParameters.dat"
    print(f"\nLoading parameters from: {params_path}")
    params = load_stigmergic_parameters(params_path)

    # Phase 1: Bioelectric simulation
    bio_model = run_bioelectric_simulation(params, num_iters=params['simParameters']['numSimIters'])

    # Initialize modules
    lattice_dims = params["latticeDims"]
    rows, cols = lattice_dims
    transduction, facial_grn, classifier = initialize_transduction_and_grn(lattice_dims)

    # Initialize transduction module with bioelectric state
    vmem_grid = bio_model.electricNetwork.Vmem.view(rows, cols)
    transduction.update(vmem_grid, dt=0.01)

    # Phase 2a: PRE-EQUILIBRATE morphogens (NEW)
    print("\n=== Morphogen Pre-Equilibration ===")
    print("Running morphogen-only updates to establish gradients before gene activation...")
    bio_signals_initial = transduction.get_gene_modulation_signals()
    for pre_step in range(2000):  # 2000 steps to reach steady state
        facial_grn.update_morphogens()
    morph_grids_eq = facial_grn.get_morphogen_grids()
    print(f"Morphogen equilibration complete:")
    print(f"  SHH: max={morph_grids_eq['shh'].max():.4f}")
    print(f"  FGF8: max={morph_grids_eq['fgf8'].max():.4f}")
    print(f"  EDN1: max={morph_grids_eq['edn1'].max():.4f}")

    # Phase 2b: Integrated dynamics
    results = run_integrated_dynamics(
        bio_model, transduction, facial_grn, classifier,
        num_cycles=20,  # Increased from 5 to match autonomous runtime (20 × 500 = 10,000)
        bio_steps=100,
        grn_steps=500,
        feedback_gain=0.02
    )

    # Visualize and summarize
    visualize_results(results, output_path='refined_facial_integration.png')
    print_summary(results)

    print("\n✅ Refined integration complete!")
    print("\nKey design features:")
    print("  ✓ No A-P voltage gradient (all features ~-60mV)")
    print("  ✓ Dual drivers: Morphogen (70%) + Bioelectric (30%)")
    print("  ✓ Temporal integration via Ca²⁺ dynamics")
    print("  ✓ Gap junction currents (not 'detail')")
    print("  ✓ Features from gene expression (not voltage thresholds)")
    print("  ✓ Proper timescale hierarchy (bio << morph << genes)")


if __name__ == "__main__":
    main()
