#!/usr/bin/env python3
"""
CaMKII-Integrated Facial Patterning with Concurrent Dynamics

Runs the CaMKII bistability mechanism concurrently with the facial GRN,
enabling bioelectric pattern memory that persists after Vmem decay.

Key differences from run_refined_facial_integration.py:
1. CaMKII runs concurrently with GRN (not pre-equilibrated Ca)
2. bio_gate comes from CaMKII activity (bistable, persistent)
3. Three temporal phases:
   - Phase 1 (0-1000): Vmem pattern drives Ca → CaMKII pattern formation
   - Phase 2 (1000-2000): Vmem decays, Ca decays, CaMKII locks in via bistability
   - Phase 3 (2000-3000): Vmem uniform, CaMKII pattern persists, GRN continues

Usage:
    python run_camkii_facial_integration.py
    python run_camkii_facial_integration.py --camkii-params data/bestLearnedCaMKIIParams_0.dat
    python run_camkii_facial_integration.py --grn-params data/bestLearnedFacialParams_0.dat
"""

import argparse
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from embryo import model
from camkiiFacialGRN import CaMKIIFacialGRN
from geneBasedFeatureClassifier import GeneBasedFeatureClassifier


def load_stigmergic_parameters(path: str):
    """Load stigmergic model parameters"""
    params = torch.load(path, weights_only=False)
    if "ATPParameters" not in params:
        params["ATPParameters"] = None
    return params


def apply_sigmoid_constraint(raw_param, min_val, max_val):
    """Map unbounded raw parameter to bounded range via sigmoid"""
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)


def load_camkii_parameters(path: str):
    """
    Load learned CaMKII bistability parameters.

    Args:
        path: Path to bestLearnedCaMKIIParams_*.dat file

    Returns:
        dict with constrained parameter values
    """
    print(f"Loading CaMKII parameters from: {path}")
    data = torch.load(path, weights_only=False)

    learned_params = {}
    param_bounds = data.get('parameter_bounds', {})

    for param_name, raw_value in data['parameters'].items():
        min_key = f'{param_name}_min'
        max_key = f'{param_name}_max'

        if min_key in param_bounds and max_key in param_bounds:
            constrained = apply_sigmoid_constraint(
                raw_value,
                param_bounds[min_key],
                param_bounds[max_key]
            )
            learned_params[param_name] = float(constrained.item())
        else:
            learned_params[param_name] = float(raw_value.item()) if hasattr(raw_value, 'item') else float(raw_value)

    print(f"Loaded {len(learned_params)} CaMKII parameters")
    return learned_params


def load_grn_parameters(path: str):
    """
    Load learned GRN parameters.

    Args:
        path: Path to bestLearnedFacialParams_*.dat file

    Returns:
        dict with constrained parameter values
    """
    print(f"Loading GRN parameters from: {path}")
    data = torch.load(path, weights_only=False)

    learned_params = {}
    param_bounds = data.get('parameter_bounds', {})

    for param_name, raw_value in data['parameters'].items():
        min_key = f'{param_name}_min'
        max_key = f'{param_name}_max'

        if min_key in param_bounds and max_key in param_bounds:
            constrained = apply_sigmoid_constraint(
                raw_value,
                param_bounds[min_key],
                param_bounds[max_key]
            )
            learned_params[param_name] = float(constrained.item())
        else:
            learned_params[param_name] = float(raw_value.item()) if hasattr(raw_value, 'item') else float(raw_value)

    # Load fixed GRN parameters if present
    if 'fixed_grn_params' in data:
        fixed_grn_params = data['fixed_grn_params']
        print(f"Found {len(fixed_grn_params)} fixed GRN parameters")
        for param_name, param_value in fixed_grn_params.items():
            learned_params[param_name] = float(param_value) if isinstance(param_value, torch.Tensor) else param_value

    print(f"Loaded {len(learned_params)} GRN parameters")
    return learned_params


def run_bioelectric_simulation(params, num_iters=1000):
    """
    Run stigmergic bioelectric simulation to get Vmem pattern.

    Returns:
        bio_model: Converged bioelectric model
        initial_vmem: Initial uniform Vmem
        final_vmem: Final patterned Vmem
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

    # Get initial Vmem
    initial_vmem = bio_model.electricNetwork.Vmem.clone()

    # Run simulation
    bio_model.simulate(
        externalInputs=external_inputs,
        clampParameters=clamp_params,
        perturbation=None,
        fieldModulation=False,
        numSimIters=num_iters
    )

    final_vmem = bio_model.electricNetwork.Vmem
    print(f"Final Vmem: mean={final_vmem.mean().item():.4f}V, "
          f"std={final_vmem.std().item():.4f}V, "
          f"range=[{final_vmem.min().item():.4f}, {final_vmem.max().item():.4f}]V")

    return bio_model, initial_vmem, final_vmem


def create_vmem_trajectory(initial_vmem_grid, final_vmem_grid,
                           num_rise=1000, num_decay=1000, num_maintain=1000):
    """
    Create Vmem trajectory for concurrent simulation.

    Three phases:
    1. Rise (0 to num_rise): Vmem interpolates from initial to final
    2. Decay (num_rise to num_rise+num_decay): Vmem decays back to initial
    3. Maintain (num_rise+num_decay to total): Vmem stays at initial

    Args:
        initial_vmem_grid: Initial uniform Vmem (grid_size, grid_size)
        final_vmem_grid: Final patterned Vmem (grid_size, grid_size)
        num_rise: Steps for Vmem to reach pattern
        num_decay: Steps for Vmem to decay
        num_maintain: Steps to maintain uniform Vmem

    Returns:
        List of Vmem grids for each timestep
    """
    trajectory = []
    total_steps = num_rise + num_decay + num_maintain

    for t in range(total_steps):
        if t < num_rise:
            # Phase 1: Rise - interpolate from initial to final
            alpha = t / num_rise
            vmem = (1 - alpha) * initial_vmem_grid + alpha * final_vmem_grid
        elif t < num_rise + num_decay:
            # Phase 2: Decay - interpolate from final back to initial
            decay_progress = (t - num_rise) / num_decay
            vmem = (1 - decay_progress) * final_vmem_grid + decay_progress * initial_vmem_grid
        else:
            # Phase 3: Maintain - stay at initial (uniform)
            vmem = initial_vmem_grid.clone()

        trajectory.append(vmem)

    return trajectory


def run_concurrent_integration(camkii_grn, vmem_trajectory, classifier,
                                num_rise=1000, num_decay=1000, num_maintain=1000,
                                checkpoint_interval=100, verbose=True):
    """
    Run concurrent CaMKII + GRN dynamics.

    Key temporal separation:
    - Phase 1 (Rise, t < num_rise): Only CaMKII updates (bioelectric pattern formation)
    - Phase 2+ (Decay & Maintain, t >= num_rise): Both CaMKII and GRN update

    Args:
        camkii_grn: CaMKIIFacialGRN instance
        vmem_trajectory: List of Vmem grids
        classifier: GeneBasedFeatureClassifier instance
        num_rise: Steps in rise phase (GRN starts updating AFTER this)
        num_decay: Steps in decay phase
        num_maintain: Steps in maintain phase
        checkpoint_interval: How often to record history
        verbose: Print progress

    Returns:
        dict with final state and history
    """
    print("\n=== Running Concurrent CaMKII + GRN Dynamics ===")
    print(f"Total steps: {len(vmem_trajectory)}")
    print(f"Phase 1 (Rise, t=0-{num_rise}): CaMKII only (bioelectric pattern formation)")
    print(f"Phase 2+ (t>={num_rise}): CaMKII + GRN concurrent (morphogen-gene dynamics)")

    history = {
        'time': [],
        'vmem_mean': [], 'vmem_std': [],
        'Ca_mean': [], 'Ca_std': [],
        'CaMKII_mean': [], 'CaMKII_std': [],
        'bio_gate_mean': [],
        'genes': {gene: [] for gene in camkii_grn.gene_names},
        'morphogens': {morph: [] for morph in camkii_grn.morphogen_names},
        'features': [],
        'feature_counts': [],
    }

    # Reset state
    camkii_grn.reset()

    # Pre-equilibrate morphogens (they don't depend on Vmem)
    print("Pre-equilibrating morphogens (1000 steps)...")
    for _ in range(1000):
        camkii_grn.update_morphogens()

    morph_grids = camkii_grn.get_morphogen_grids()
    print(f"  SHH: max={morph_grids['shh'].max():.4f}")
    print(f"  FGF8: max={morph_grids['fgf8'].max():.4f}")
    print(f"  EDN1: max={morph_grids['edn1'].max():.4f}")

    # Run concurrent dynamics
    dt = 0.01
    for t, vmem_grid in enumerate(vmem_trajectory):
        # # Temporal separation: CaMKII forms pattern first, then GRN dynamics begin
        # if t < 1: # (num_rise):
        #     # Phase 1 (Rise): Only CaMKII updates (bioelectric pattern formation)
        #     camkii_grn.camkii_switch.update(vmem_grid, dt=dt)
        #     # Morphogens continue to equilibrate (already equilibrated)
        #     # Genes stay at initial low values
        # else:
        #     # Phase 2+ (Decay & Maintain): Full concurrent CaMKII + GRN dynamics
        camkii_grn.update_concurrent(vmem_grid, dt=dt)

        # Record history
        if t % checkpoint_interval == 0:
            history['time'].append(t * dt)
            history['vmem_mean'].append(vmem_grid.mean().item())
            history['vmem_std'].append(vmem_grid.std().item())
            history['Ca_mean'].append(camkii_grn.camkii_switch.Ca.mean().item())
            history['Ca_std'].append(camkii_grn.camkii_switch.Ca.std().item())
            history['CaMKII_mean'].append(camkii_grn.camkii_switch.CaMKII_active.mean().item())
            history['CaMKII_std'].append(camkii_grn.camkii_switch.CaMKII_active.std().item())

            # bio_gate is raw CaMKII activity (already in [0,1])
            history['bio_gate_mean'].append(camkii_grn.camkii_switch.CaMKII_active.mean().item())

            for gene in camkii_grn.gene_names:
                history['genes'][gene].append(camkii_grn.grid[gene].mean().item())
            for morph in camkii_grn.morphogen_names:
                history['morphogens'][morph].append(camkii_grn.grid[morph].mean().item())

            # Classify features
            gene_grids = camkii_grn.get_gene_grids()
            classification = classifier.classify(gene_grids, mode='both')
            feature_grid = classification['features']
            feature_counts = classifier.summarize_features(feature_grid)
            history['features'].append(feature_grid.detach().clone().cpu())
            history['feature_counts'].append(feature_counts)

            if verbose and t % (checkpoint_interval * 10) == 0:
                phase = "Rise" if t < num_rise else ("Decay" if t < num_rise + num_decay else "Maintain")
                print(f"  t={t:4d} [{phase:8s}]: Ca={history['Ca_mean'][-1]:.3f}, "
                      f"CaMKII={history['CaMKII_mean'][-1]:.3f}, "
                      f"Features={feature_counts}")

    # Get final state
    gene_grids = camkii_grn.get_gene_grids()
    morph_grids = camkii_grn.get_morphogen_grids()
    classification = classifier.classify(gene_grids, mode='both')
    feature_grid = classification['features']
    feature_counts = classifier.summarize_features(feature_grid)

    print(f"\nFinal feature counts: {feature_counts}")

    return {
        'final_vmem': vmem_trajectory[-1].detach().cpu(),
        'final_Ca': camkii_grn.camkii_switch.Ca.detach().cpu(),
        'final_CaMKII': camkii_grn.camkii_switch.CaMKII_active.detach().cpu(),
        'final_genes': {k: v.detach().cpu() for k, v in gene_grids.items()},
        'final_morphogens': {k: v.detach().cpu() for k, v in morph_grids.items()},
        'final_features': feature_grid.detach().cpu(),
        'final_feature_counts': feature_counts,
        'history': history
    }


def run_grn_only_dynamics(camkii_grn, classifier, num_iters=2000, checkpoint_interval=100):
    """
    Run GRN-only dynamics without bioelectric coupling.

    This mode runs the morphogen-gene system autonomously:
    - Morphogen gradients establish spatial patterns
    - Genes respond to morphogen combinations
    - No bioelectric gating (bio_gate defaults to autonomous mode)

    Args:
        camkii_grn: CaMKIIFacialGRN instance
        classifier: GeneBasedFeatureClassifier instance
        num_iters: Number of GRN update iterations
        checkpoint_interval: How often to record history

    Returns:
        dict with final state and history
    """
    print("\n=== GRN-Only Dynamics (No Bioelectric Coupling) ===")
    print(f"Running {num_iters} GRN iterations...")

    # History tracking (sample every checkpoint_interval iterations)
    history = {
        'genes': {gene: [] for gene in camkii_grn.gene_names},
        'morphogens': {morph: [] for morph in camkii_grn.morphogen_names},
        'features': [],
        'feature_counts': []
    }

    # Reset state
    # camkii_grn.reset()

    # Pre-equilibrate morphogens (match learning script: 1000 steps)
    print("Pre-equilibrating morphogens (1000 steps)...")
    for pre_step in range(1000):
        camkii_grn.update_morphogens()

    morph_grids_eq = camkii_grn.get_morphogen_grids()
    print(f"Morphogen equilibration complete:")
    print(f"  SHH: max={morph_grids_eq['shh'].max():.4f}")
    print(f"  FGF8: max={morph_grids_eq['fgf8'].max():.4f}")
    print(f"  EDN1: max={morph_grids_eq['edn1'].max():.4f}")

    # Run GRN dynamics
    print(f"\nRunning GRN dynamics ({num_iters} iterations)...")
    for iter_idx in range(num_iters):
        # Update morphogens
        camkii_grn.update_morphogens()

        # Update genes in autonomous mode (bio_gate=None)
        camkii_grn.update_genes(bio_gate=None)

        # Sample history periodically
        if (iter_idx + 1) % checkpoint_interval == 0:
            gene_grids = camkii_grn.get_gene_grids()
            morph_grids = camkii_grn.get_morphogen_grids()
            classification = classifier.classify(gene_grids, mode='both')
            feature_grid = classification['features']
            feature_counts = classifier.summarize_features(feature_grid)

            for gene in camkii_grn.gene_names:
                history['genes'][gene].append(gene_grids[gene].mean().item())

            for morph in camkii_grn.morphogen_names:
                history['morphogens'][morph].append(morph_grids[morph].mean().item())

            history['features'].append(feature_grid.detach().clone().cpu())
            history['feature_counts'].append(feature_counts)

            if (iter_idx + 1) % 1000 == 0:
                print(f"  Iteration {iter_idx + 1}/{num_iters} - Features: {feature_counts}")

    # Get final state
    gene_grids = camkii_grn.get_gene_grids()
    morph_grids = camkii_grn.get_morphogen_grids()
    classification = classifier.classify(gene_grids, mode='both')
    feature_grid = classification['features']
    feature_counts = classifier.summarize_features(feature_grid)

    print(f"\nFinal feature counts: {feature_counts}")

    # Create dummy uniform Vmem for visualization (not used in GRN-only mode)
    dummy_vmem = torch.full((camkii_grn.grid_size, camkii_grn.grid_size),
                            -0.04, device=camkii_grn.device, dtype=camkii_grn.dtype)

    return {
        'final_vmem': dummy_vmem.detach().cpu(),
        # 'final_Ca': camkii_grn.camkii_switch.Ca.detach().cpu(),
        # 'final_CaMKII': camkii_grn.camkii_switch.CaMKII_active.detach().cpu(),
        'final_genes': {k: v.detach().cpu() for k, v in gene_grids.items()},
        'final_morphogens': {k: v.detach().cpu() for k, v in morph_grids.items()},
        'final_features': feature_grid.detach().cpu(),
        'final_feature_counts': feature_counts,
        'history': history
    }


def visualize_results(results, output_path='camkii_facial_integration.png'):
    """
    Visualize concurrent CaMKII + GRN results.

    Shows:
    - Row 1: Vmem, Ca, CaMKII, Features
    - Row 2: Morphogens (SHH, FGF8, EDN1) + empty
    - Row 3: Key genes (Pax6, Lhx2, Alx, Dlx)
    - Row 4: Time series (Vmem, Ca, CaMKII means with variance)
    """
    print("\n=== Generating Visualization ===")

    feature_cmap = ListedColormap(['#f9f9f9', '#9b59b6', '#e67e22', '#2ecc71'])

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    # Row 1: Bioelectric signals and features
    ax = axes[0, 0]
    im = ax.imshow(results['final_vmem'].numpy(), cmap='coolwarm')
    ax.set_title('Final Vmem', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Check if Ca and CaMKII data exist (not available in GRN-only mode)
    if 'final_Ca' in results:
        ax = axes[0, 1]
        im = ax.imshow(results['final_Ca'].numpy(), cmap='viridis')
        ax.set_title('Final Ca²⁺', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        axes[0, 1].axis('off')

    if 'final_CaMKII' in results:
        ax = axes[0, 2]
        im = ax.imshow(results['final_CaMKII'].numpy(), cmap='plasma', vmin=0, vmax=1)
        ax.set_title('Final CaMKII (Bistable)', fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        axes[0, 2].axis('off')

    ax = axes[0, 3]
    im = ax.imshow(results['final_features'].numpy(), cmap=feature_cmap, vmin=0, vmax=3)
    ax.set_title('Features (from Genes)', fontsize=10, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046)
    cbar.set_ticklabels(['Bone', 'Eye', 'Nose', 'Mouth'])

    # Row 2: Morphogens
    morphogens = results['final_morphogens']
    morph_names = ['shh', 'fgf8', 'edn1']
    for idx, morph_name in enumerate(morph_names):
        ax = axes[1, idx]
        im = ax.imshow(morphogens[morph_name].numpy(), cmap='plasma')
        ax.set_title(f'{morph_name.upper()} Gradient', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    axes[1, 3].axis('off')

    # Row 3: Key genes
    genes = results['final_genes']
    gene_display = ['pax6', 'lhx2', 'alx', 'dlx']
    gene_titles = ['Pax6 (Eye)', 'Lhx2 (Eye)', 'Alx (Nose)', 'Dlx (Mouth)']
    for idx, (gene, title) in enumerate(zip(gene_display, gene_titles)):
        ax = axes[2, idx]
        im = ax.imshow(genes[gene].numpy(), cmap='viridis', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 4: Time series
    history = results['history']

    # Check if this is integrated mode (has time series data) or GRN-only mode
    has_time_series = 'time' in history

    if has_time_series:
        # Integrated mode: Show full time series with Vmem, Ca, CaMKII
        time = np.array(history['time'])

        # Normalize helper
        def normalize(mean_arr, std_arr):
            mean_arr = np.array(mean_arr)
            std_arr = np.array(std_arr)
            min_val = (mean_arr - std_arr).min()
            max_val = (mean_arr + std_arr).max()
            range_val = max_val - min_val
            if range_val < 1e-10:
                range_val = 1.0
            norm_mean = (mean_arr - min_val) / range_val
            norm_std = std_arr / range_val
            return norm_mean, norm_std

        # Combined time series plot
        ax = axes[3, 0]
        ax.set_title('Normalized Time Series', fontsize=10)

        vmem_mean, vmem_std = normalize(history['vmem_mean'], history['vmem_std'])
        ca_mean, ca_std = normalize(history['Ca_mean'], history['Ca_std'])
        camkii_mean, camkii_std = normalize(history['CaMKII_mean'], history['CaMKII_std'])

        ax.fill_between(time, vmem_mean - vmem_std, vmem_mean + vmem_std, alpha=0.2, color='blue')
        ax.plot(time, vmem_mean, 'b-', label='Vmem', linewidth=1.5)

        ax.fill_between(time, ca_mean - ca_std, ca_mean + ca_std, alpha=0.2, color='green')
        ax.plot(time, ca_mean, 'g-', label='Ca²⁺', linewidth=1.5)

        ax.fill_between(time, camkii_mean - camkii_std, camkii_mean + camkii_std, alpha=0.2, color='red')
        ax.plot(time, camkii_mean, 'r-', label='CaMKII', linewidth=1.5)

        # Phase markers
        ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
        ax.text(5, 0.95, 'Rise', ha='center', fontsize=8, color='gray')
        ax.text(15, 0.95, 'Decay', ha='center', fontsize=8, color='gray')
        ax.text(25, 0.95, 'Maintain', ha='center', fontsize=8, color='gray')

        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Value')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
        ax.set_ylim(-0.1, 1.1)
    else:
        # GRN-only mode: No time series data available
        ax = axes[3, 0]
        ax.text(0.5, 0.5, 'GRN-Only Mode\n(No bioelectric time series)',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')

        # Create iteration index for gene and feature plots
        time = np.arange(len(history['genes']['pax6']))

    # Gene time series (available in both modes)
    ax = axes[3, 1]
    ax.set_title('Gene Expression Over Time', fontsize=10)
    for gene in ['pax6', 'alx', 'dlx']:
        if gene in history['genes']:
            ax.plot(time, history['genes'][gene], label=gene, linewidth=1.5)
    ax.set_xlabel('Iteration' if not has_time_series else 'Time')
    ax.set_ylabel('Mean Expression')
    ax.legend(fontsize=8)

    # Feature counts over time (available in both modes)
    ax = axes[3, 2]
    ax.set_title('Feature Cell Counts', fontsize=10)
    feature_counts_arr = np.array([[fc.get('bone', 0), fc.get('eye', 0),
                                     fc.get('nose', 0), fc.get('mouth', 0)]
                                    for fc in history['feature_counts']])
    if len(feature_counts_arr) > 0:
        ax.plot(time, feature_counts_arr[:, 0], label='Bone', color='#f9f9f9', linewidth=2,
                markeredgecolor='black', marker='o', markersize=3)
        ax.plot(time, feature_counts_arr[:, 1], label='Eye', color='#9b59b6', linewidth=2)
        ax.plot(time, feature_counts_arr[:, 2], label='Nose', color='#e67e22', linewidth=2)
        ax.plot(time, feature_counts_arr[:, 3], label='Mouth', color='#2ecc71', linewidth=2)
    ax.set_xlabel('Iteration' if not has_time_series else 'Time')
    ax.set_ylabel('Cell Count')
    ax.legend(fontsize=8)

    axes[3, 3].axis('off')

    # Remove ticks from image plots
    for i in range(3):
        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    fig.suptitle('CaMKII-Integrated Facial Patterning with Concurrent Dynamics\n'
                 'Vmem → Ca²⁺ → CaMKII (bistable) → Gene Expression → Features',
                 fontsize=14, fontweight='bold')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved visualization to {output_path}")


def print_summary(results):
    """Print summary statistics"""
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    print("\nFeature Counts:")
    for feature, count in results['final_feature_counts'].items():
        print(f"  {feature:8s}: {count:3d} cells")

    print("\nBioelectric Statistics:")
    vmem = results['final_vmem']
    print(f"  Vmem:      mean={vmem.mean():.4f}V, std={vmem.std():.4f}V")

    # Ca = results['final_Ca']
    # print(f"  Ca²⁺:      mean={Ca.mean():.4f}, max={Ca.max():.4f}")
    #
    # CaMKII = results['final_CaMKII']
    # print(f"  CaMKII:    mean={CaMKII.mean():.4f}, max={CaMKII.max():.4f}")

    print("\nKey Gene Expression Levels:")
    genes = results['final_genes']
    for gene in ['pax6', 'lhx2', 'alx', 'dlx', 'hand2', 'runx2']:
        expr = genes[gene]
        print(f"  {gene:6s}: mean={expr.mean():.3f}, max={expr.max():.3f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='CaMKII-Integrated Facial Patterning with Concurrent Dynamics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python run_camkii_facial_integration.py

  # Run with learned CaMKII parameters
  python run_camkii_facial_integration.py --camkii-params data/bestLearnedCaMKIIParams_0.dat

  # Run with both learned CaMKII and GRN parameters
  python run_camkii_facial_integration.py \\
    --camkii-params data/bestLearnedCaMKIIParams_0.dat \\
    --grn-params data/bestLearnedFacialParams_0.dat

  # Run GRN-only mode (no bioelectric coupling)
  python run_camkii_facial_integration.py --grn-only --grn-params data/bestLearnedFacialParams_0.dat
        """
    )
    parser.add_argument('--stigmergic-params', type=str,
                        default='data/StigmergicModelParameters.dat',
                        help='Path to stigmergic bioelectric model parameters')
    parser.add_argument('--camkii-params', type=str, default=None,
                        help='Path to learned CaMKII bistability parameters')
    parser.add_argument('--grn-params', type=str, default=None,
                        help='Path to learned GRN parameters')
    parser.add_argument('--grn-only', action='store_true',
                        help='Run GRN-only mode without bioelectric coupling (uniform Vmem, CaMKII at baseline)')
    parser.add_argument('--grn-iters', type=int, default=2000,
                        help='Number of GRN iterations for GRN-only mode (default: 2000)')
    parser.add_argument('--grid-size', type=int, default=11,
                        help='Grid size (default: 11)')
    parser.add_argument('--num-rise', type=int, default=1000,
                        help='Steps for Vmem rise phase (default: 1000)')
    parser.add_argument('--num-decay', type=int, default=1000,
                        help='Steps for Vmem decay phase (default: 1000)')
    parser.add_argument('--num-maintain', type=int, default=1000,
                        help='Steps for maintenance phase (default: 1000)')
    parser.add_argument('--output', type=str, default='camkii_facial_integration.png',
                        help='Output visualization path')

    args = parser.parse_args()

    print("=" * 60)
    if args.grn_only:
        print("GRN-ONLY MODE (NO BIOELECTRIC COUPLING)")
    else:
        print("CAMKII-INTEGRATED FACIAL PATTERNING")
        print("WITH CONCURRENT DYNAMICS")
    print("=" * 60)

    # Device setup
    device = 'cpu'
    dtype = torch.float64

    # Load learned GRN parameters if provided
    grn_params = None
    if args.grn_params:
        grn_params = load_grn_parameters(args.grn_params)

    # ========================================================================
    # GRN-ONLY MODE: Skip bioelectric simulation
    # ========================================================================
    if args.grn_only:
        print(f"\nRunning in GRN-only mode (grid size: {args.grid_size}x{args.grid_size})")

        # Create CaMKII-integrated GRN
        print("\nInitializing CaMKII-integrated facial GRN...")

        # Extract decay lengths from learned params if available
        shh_decay = grn_params.get('shh_decay_length', 0.8) if grn_params else 0.8
        fgf8_decay = grn_params.get('fgf8_decay_length', 0.3) if grn_params else 0.3
        edn1_decay = grn_params.get('edn1_decay_length', 0.6) if grn_params else 0.6

        camkii_grn = CaMKIIFacialGRN(
            grid_size=args.grid_size,
            device=device,
            dtype=dtype,
            shh_decay_length=shh_decay,
            fgf8_decay_length=fgf8_decay,
            edn1_decay_length=edn1_decay
        )

        # Apply learned GRN parameters
        if grn_params:
            print("Applying learned GRN parameters...")
            # Morphogen parameters
            morph_map = {
                'fgf8_strength': 'fgf8_strength',
                'fgf8_degradation_factor': 'fgf8_degradation_factor',
                'edn1_strength': 'edn1_strength',
                'edn1_degradation_factor': 'edn1_degradation_factor',
                'diffusion_rate': 'diffusion_rate',
            }
            for learned_name, grn_name in morph_map.items():
                if learned_name in grn_params:
                    camkii_grn.morphogen_params[grn_name] = torch.tensor(
                        grn_params[learned_name], device=device, dtype=dtype)
                    print(f"  Set {grn_name} = {grn_params[learned_name]:.4f}")

            # Gene parameters
            gene_map = {
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
            for learned_name, grn_name in gene_map.items():
                if learned_name in grn_params:
                    camkii_grn.gene_params[grn_name] = torch.tensor(
                        grn_params[learned_name], device=device, dtype=dtype)
                    print(f"  Set {grn_name} = {grn_params[learned_name]:.4f}")

        # Bioelectric gating parameters (Ca threshold, sensitivity, AND gate)
        bioelectric_param_map = {
            'ca_threshold': 'ca_threshold_override',
            'ca_sensitivity': 'ca_sensitivity_override',
            'and_threshold': 'and_threshold_override',
            'and_sharpness': 'and_sharpness_override',
        }

        bioelectric_params_found = False
        for learned_name, grn_attr in bioelectric_param_map.items():
            if learned_name in grn_params:
                setattr(camkii_grn, grn_attr, torch.tensor(
                    grn_params[learned_name],
                    device=camkii_grn.device,
                    dtype=camkii_grn.dtype
                ))
                print(f"  Set {grn_attr} = {grn_params[learned_name]:.4f}")
                bioelectric_params_found = True

        if bioelectric_params_found:
            print(f"Applied learned bioelectric gating parameters")

        # Create feature classifier
        classifier = GeneBasedFeatureClassifier(grid_size=args.grid_size, device=device)

        # Run GRN-only dynamics
        results = run_grn_only_dynamics(
            camkii_grn, classifier,
            num_iters=args.grn_iters,
            checkpoint_interval=100
        )

        # Visualize and summarize
        visualize_results(results, output_path='grn_only_camkii_facial_integration.png')
        print_summary(results)

        print("\n✅ GRN-only simulation complete!")
        print("\nKey design features:")
        print("  ✓ Morphogen gradients only (no bioelectric coupling)")
        print("  ✓ Autonomous mode (bio_gate=None, direct morphogen activation)")
        print("  ✓ Matches RefinedFacialGRN autonomous behavior")
        print("  ✓ Features from gene expression (not voltage thresholds)")

    # ========================================================================
    # INTEGRATED MODE: Full bioelectric-CaMKII-GRN dynamics
    # ========================================================================
    else:
        # Load stigmergic parameters
        print(f"\nLoading stigmergic parameters from: {args.stigmergic_params}")
        stig_params = load_stigmergic_parameters(args.stigmergic_params)

        # Run bioelectric simulation to get Vmem pattern
        bio_model, initial_vmem, final_vmem = run_bioelectric_simulation(
            stig_params,
            num_iters=stig_params['simParameters']['numSimIters']
        )

        lattice_dims = stig_params["latticeDims"]
        rows, cols = lattice_dims
        grid_size = rows

        # Reshape Vmem grids
        initial_vmem_grid = initial_vmem[0, :, 0].view(rows, cols).to(device=device, dtype=dtype)
        final_vmem_grid = final_vmem[0, :, 0].view(rows, cols).to(device=device, dtype=dtype)

        # Load learned CaMKII parameters if provided
        camkii_params = None
        if args.camkii_params:
            camkii_params = load_camkii_parameters(args.camkii_params)

        # Create CaMKII-integrated GRN
        print("\nInitializing CaMKII-integrated facial GRN...")

        # Extract decay lengths from learned params if available
        shh_decay = grn_params.get('shh_decay_length', 0.8) if grn_params else 0.8
        fgf8_decay = grn_params.get('fgf8_decay_length', 0.3) if grn_params else 0.3
        edn1_decay = grn_params.get('edn1_decay_length', 0.6) if grn_params else 0.6

        camkii_grn = CaMKIIFacialGRN(
            grid_size=grid_size,
            device=device,
            dtype=dtype,
            shh_decay_length=shh_decay,
            fgf8_decay_length=fgf8_decay,
            edn1_decay_length=edn1_decay
        )

        # Apply learned CaMKII parameters
        if camkii_params:
            print("Applying learned CaMKII parameters...")
            camkii_grn.camkii_switch.load_learned_parameters(camkii_params)

        # Apply learned GRN parameters
        if grn_params:
            print("Applying learned GRN parameters...")
            # Morphogen parameters
            morph_map = {
                'fgf8_strength': 'fgf8_strength',
                'fgf8_degradation_factor': 'fgf8_degradation_factor',
                'edn1_strength': 'edn1_strength',
                'edn1_degradation_factor': 'edn1_degradation_factor',
                'diffusion_rate': 'diffusion_rate',
            }
            for learned_name, grn_name in morph_map.items():
                if learned_name in grn_params:
                    camkii_grn.morphogen_params[grn_name] = torch.tensor(
                        grn_params[learned_name], device=device, dtype=dtype)

            # Gene parameters
            gene_map = {
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
            for learned_name, grn_name in gene_map.items():
                if learned_name in grn_params:
                    camkii_grn.gene_params[grn_name] = torch.tensor(
                        grn_params[learned_name], device=device, dtype=dtype)
                    print(f"  Set {grn_name} = {grn_params[learned_name]:.4f}")

        # Bioelectric gating parameters (Ca threshold, sensitivity, AND gate)
        bioelectric_param_map = {
            'ca_threshold': 'ca_threshold_override',
            'ca_sensitivity': 'ca_sensitivity_override',
            'and_threshold': 'and_threshold_override',
            'and_sharpness': 'and_sharpness_override',
        }

        bioelectric_params_found = False
        for learned_name, grn_attr in bioelectric_param_map.items():
            if learned_name in grn_params:
                setattr(camkii_grn, grn_attr, torch.tensor(
                    grn_params[learned_name],
                    device=camkii_grn.device,
                    dtype=camkii_grn.dtype
                ))
                print(f"  Set {grn_attr} = {grn_params[learned_name]:.4f}")
                bioelectric_params_found = True

        if bioelectric_params_found:
            print(f"Applied learned bioelectric gating parameters")

        # Create feature classifier
        classifier = GeneBasedFeatureClassifier(grid_size=grid_size, device=device)

        # Create Vmem trajectory (rise → decay → maintain)
        print(f"\nCreating Vmem trajectory...")
        print(f"  Rise phase: {args.num_rise} steps")
        print(f"  Decay phase: {args.num_decay} steps")
        print(f"  Maintain phase: {args.num_maintain} steps")

        vmem_trajectory = create_vmem_trajectory(
            initial_vmem_grid, final_vmem_grid,
            num_rise=args.num_rise,
            num_decay=args.num_decay,
            num_maintain=args.num_maintain
        )

        # Run concurrent integration
        results = run_concurrent_integration(
            camkii_grn, vmem_trajectory, classifier,
            num_rise=args.num_rise,
            num_decay=args.num_decay,
            num_maintain=args.num_maintain,
            checkpoint_interval=100,
            verbose=True
        )

        # Visualize and summarize
        visualize_results(results, output_path=args.output)
        print_summary(results)

        print("\n✅ Concurrent integration complete!")
        print("\nKey design features:")
        print("  ✓ CaMKII runs concurrently with GRN")
        print("  ✓ bio_gate derived from CaMKII (bistable, persistent)")
        print("  ✓ Pattern survives Vmem decay via CaMKII memory")
        print("  ✓ Three temporal phases: Rise → Decay → Maintain")


if __name__ == "__main__":
    main()
