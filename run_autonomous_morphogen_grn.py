#!/usr/bin/env python3
"""
Autonomous Morphogen-GRN Patterning (NO Bioelectrics)

Tests whether morphogen gradients + gene networks ALONE can produce facial features,
without bioelectric prepattern guidance.

This serves as a control to determine:
1. Can morphogens alone create spatial structure?
2. How much does bioelectric input contribute to final pattern?
3. What is the "baseline" morphogen-driven pattern?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from refinedFacialGRN import RefinedFacialGRN
from geneBasedFeatureClassifier import GeneBasedFeatureClassifier


def run_autonomous_morphogen_grn(grid_size=11, num_iterations=10000, save_interval=500):
    """
    Run morphogen-GRN dynamics without any bioelectric input.

    Args:
        grid_size: Spatial grid size (square)
        num_iterations: Total simulation steps
        save_interval: How often to save snapshots

    Returns:
        dict with final state and history
    """
    print("="*70)
    print("AUTONOMOUS MORPHOGEN-GRN PATTERNING (No Bioelectrics)")
    print("="*70)
    print(f"\nGrid size: {grid_size}x{grid_size}")
    print(f"Total iterations: {num_iterations}")
    print(f"Save interval: {save_interval}")

    # Initialize GRN (no bioelectric input)
    facial_grn = RefinedFacialGRN(grid_size=grid_size, device='cpu')
    classifier = GeneBasedFeatureClassifier(grid_size=grid_size, device='cpu')

    # History tracking
    history = {
        'time': [],
        'genes': {gene: [] for gene in facial_grn.gene_names},
        'morphogens': {morph: [] for morph in facial_grn.morphogen_names},
        'features': [],
        'feature_counts': []
    }

    print("\n" + "="*70)
    print("Running autonomous dynamics (morphogens + genes, NO bioelectric input)...")
    print("="*70)

    # Run dynamics WITHOUT bioelectric signals
    for iteration in range(num_iterations):
        # Update morphogens and genes (bioelectric_signals=None means no bioelectric input)
        facial_grn.update(bioelectric_signals=None)

        # Save snapshots periodically
        if (iteration + 1) % save_interval == 0:
            current_time = facial_grn.current_time

            # Get current state
            gene_grids = facial_grn.get_gene_grids()
            morph_grids = facial_grn.get_morphogen_grids()

            # Classify features from genes
            classification = classifier.classify(gene_grids, mode='both')
            feature_grid = classification['features']
            feature_counts = classifier.summarize_features(feature_grid)

            # Record
            history['time'].append(current_time)

            for gene in facial_grn.gene_names:
                history['genes'][gene].append(gene_grids[gene].detach().clone().cpu())

            for morph in facial_grn.morphogen_names:
                history['morphogens'][morph].append(morph_grids[morph].detach().clone().cpu())

            history['features'].append(feature_grid.detach().clone().cpu())
            history['feature_counts'].append(feature_counts)

            # Print progress
            print(f"\nIteration {iteration+1}/{num_iterations} (t={current_time:.1f})")
            print(f"  Feature counts: {feature_counts}")
            print(f"  Key genes: pax6={gene_grids['pax6'].mean():.3f}, "
                  f"alx={gene_grids['alx'].mean():.3f}, "
                  f"dlx={gene_grids['dlx'].mean():.3f}")
            print(f"  Morphogens: SHH={morph_grids['shh'].mean():.3f}, "
                  f"FGF8={morph_grids['fgf8'].mean():.3f}, "
                  f"EDN1={morph_grids['edn1'].mean():.3f}")

    # Final state
    final_genes = facial_grn.get_gene_grids()
    final_morphogens = facial_grn.get_morphogen_grids()
    final_classification = classifier.classify(final_genes, mode='both')
    final_features = final_classification['features']
    final_counts = classifier.summarize_features(final_features)

    print("\n" + "="*70)
    print("FINAL STATE")
    print("="*70)
    print(f"Final feature counts: {final_counts}")

    return {
        'final_genes': {k: v.detach().cpu() for k, v in final_genes.items()},
        'final_morphogens': {k: v.detach().cpu() for k, v in final_morphogens.items()},
        'final_features': final_features.detach().cpu(),
        'final_counts': final_counts,
        'history': history,
        'grid_size': grid_size
    }


def visualize_autonomous_results(results, output_path='autonomous_morphogen_grn.png'):
    """
    Visualize autonomous morphogen-GRN patterning results.
    """
    print("\n" + "="*70)
    print("Generating visualization...")
    print("="*70)

    feature_cmap = ListedColormap(['#f9f9f9', '#9b59b6', '#e67e22', '#2ecc71'])  # bone, eye, nose, mouth

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Morphogen gradients
    morphogens = results['final_morphogens']
    morph_names = ['shh', 'fgf8', 'edn1']

    for idx, morph_name in enumerate(morph_names):
        ax = axes[0, idx]
        im = ax.imshow(morphogens[morph_name].numpy(), cmap='plasma')
        ax.set_title(f'{morph_name.upper()} Gradient', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([])
        ax.set_yticks([])

    # Features (from genes alone)
    ax = axes[0, 3]
    im = ax.imshow(results['final_features'].numpy(), cmap=feature_cmap, vmin=0, vmax=3)
    ax.set_title('Features (from Genes)', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046)
    cbar.set_ticklabels(['Bone', 'Eye', 'Nose', 'Mouth'])
    ax.set_xticks([])
    ax.set_yticks([])

    # Row 2: Eye genes
    genes = results['final_genes']
    eye_genes = ['rx', 'six3', 'pax6', 'lhx2']
    eye_titles = ['Rx (Eye)', 'Six3 (Eye)', 'Pax6 (Eye)', 'Lhx2 (Eye)']

    for idx, (gene, title) in enumerate(zip(eye_genes, eye_titles)):
        ax = axes[1, idx]
        im = ax.imshow(genes[gene].numpy(), cmap='viridis', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 3: Other feature genes
    other_genes = ['alx', 'dlx', 'hand2', 'runx2']
    other_titles = ['Alx (Nose)', 'Dlx (Mouth)', 'Hand2 (Mouth)', 'Runx2 (Bone)']

    for idx, (gene, title) in enumerate(zip(other_genes, other_titles)):
        ax = axes[2, idx]
        im = ax.imshow(genes[gene].numpy(), cmap='viridis', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([])
        ax.set_yticks([])

    # Overall title
    fig.suptitle('Autonomous Morphogen-GRN Patterning (NO Bioelectric Input)\n'
                 'Pure Morphogen Gradient → Gene Expression → Features',
                 fontsize=14, fontweight='bold')

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved visualization to {output_path}")


def visualize_temporal_evolution(results, output_path='autonomous_timeseries.png'):
    """
    Show how features emerge over time in autonomous mode.
    """
    print("\nGenerating temporal evolution plots...")

    history = results['history']
    times = history['time']

    if len(times) == 0:
        print("No history data to plot!")
        return

    # Convert feature counts to arrays
    feature_names = ['bone', 'eye', 'nose', 'mouth']
    feature_arrays = {name: [] for name in feature_names}

    for counts in history['feature_counts']:
        for name in feature_names:
            feature_arrays[name].append(counts.get(name, 0))

    # Plot 1: Feature counts over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Feature evolution
    ax = axes[0, 0]
    for name in feature_names:
        ax.plot(times, feature_arrays[name], label=name.capitalize(), linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Cell Count', fontsize=11)
    ax.set_title('Feature Cell Counts Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Morphogen mean levels over time
    ax = axes[0, 1]
    for morph in ['shh', 'fgf8', 'edn1']:
        morph_means = [history['morphogens'][morph][i].mean().item() for i in range(len(times))]
        ax.plot(times, morph_means, label=morph.upper(), linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Mean Concentration', fontsize=11)
    ax.set_title('Morphogen Levels Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Key gene expression over time
    ax = axes[1, 0]
    key_genes = ['pax6', 'alx', 'dlx', 'runx2']
    for gene in key_genes:
        gene_means = [history['genes'][gene][i].mean().item() for i in range(len(times))]
        ax.plot(times, gene_means, label=gene.upper(), linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Mean Expression', fontsize=11)
    ax.set_title('Key Gene Expression Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Gene cascade (eye genes)
    ax = axes[1, 1]
    eye_genes = ['rx', 'six3', 'pax6', 'lhx2']
    for gene in eye_genes:
        gene_means = [history['genes'][gene][i].mean().item() for i in range(len(times))]
        ax.plot(times, gene_means, label=gene.upper(), linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Mean Expression', fontsize=11)
    ax.set_title('Eye Gene Cascade Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle('Temporal Evolution: Autonomous Morphogen-GRN Dynamics',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved timeseries to {output_path}")


def compare_with_bioelectric(autonomous_results):
    """
    Print comparison metrics to help understand bioelectric contribution.
    """
    print("\n" + "="*70)
    print("AUTONOMOUS (Morphogen-only) RESULTS")
    print("="*70)

    print("\nFinal Feature Distribution:")
    for feature, count in autonomous_results['final_counts'].items():
        percentage = 100 * count / (autonomous_results['grid_size']**2)
        print(f"  {feature:8s}: {count:3d} cells ({percentage:5.1f}%)")

    print("\nKey Gene Expression Levels:")
    genes = autonomous_results['final_genes']
    for gene in ['pax6', 'lhx2', 'alx', 'dlx', 'hand2', 'runx2']:
        expr = genes[gene]
        print(f"  {gene:6s}: mean={expr.mean():.3f}, max={expr.max():.3f}, std={expr.std():.3f}")

    print("\nMorphogen Gradient Statistics:")
    morphogens = autonomous_results['final_morphogens']
    for morph in ['shh', 'fgf8', 'edn1']:
        conc = morphogens[morph]
        print(f"  {morph:5s}: mean={conc.mean():.3f}, max={conc.max():.3f}, std={conc.std():.3f}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nThis autonomous run shows what morphogen gradients ALONE produce.")
    print("Compare with bioelectric-coupled results to see:")
    print("  • How much spatial structure comes from morphogens vs bioelectrics")
    print("  • Whether bioelectrics create NEW features or REFINE existing ones")
    print("  • If morphogen sources are sufficient for patterning")
    print("="*70)


def main():
    """Main execution"""
    # Run autonomous simulation
    results = run_autonomous_morphogen_grn(
        grid_size=11,
        num_iterations=10000,  # Long run to reach equilibrium
        save_interval=500       # Save every 500 steps
    )

    # Visualize
    visualize_autonomous_results(results, output_path='autonomous_morphogen_grn.png')
    visualize_temporal_evolution(results, output_path='autonomous_timeseries.png')

    # Analysis
    compare_with_bioelectric(results)

    print("\n✅ Autonomous morphogen-GRN simulation complete!")
    print("\nGenerated files:")
    print("  • autonomous_morphogen_grn.png  - Final spatial patterns")
    print("  • autonomous_timeseries.png     - Temporal evolution")
    print("\nNext step: Compare with refined_facial_integration.png to see bioelectric contribution!")


if __name__ == "__main__":
    main()
