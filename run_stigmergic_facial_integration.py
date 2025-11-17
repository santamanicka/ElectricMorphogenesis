#!/usr/bin/env python3
"""
Integrate the Stigmergic bioelectric model with FacialGRN:
1. Run the original Stigmergic simulation (electric-only) until the face-like
   voltage pattern emerges via field stimulation.
2. Derive the bioelectric face mask using FacePatternCoordinator.
3. Run FacialGRN, snapping it to the electric set point so the biomolecular
   layer reproduces the same facial regions without altering morphogen logic.
4. Save a diagnostic figure comparing both layers.
"""

import copy

import matplotlib

# matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.serialization import add_safe_globals
import numpy

from embryo import model
from facePatternCoordinator import FacePatternCoordinator
from geneRegulatoryNetwork import FacialGRN


def load_stigmergic_parameters(path: str):
    add_safe_globals([numpy.core.multiarray._reconstruct])
    params = torch.load(path, weights_only=False)
    if "ATPParameters" not in params:
        params["ATPParameters"] = None
    return params


def run_stigmergic_simulation(params):
    """Run the original Stigmergic setup to obtain the bioelectric pattern."""
    sim_params = copy.deepcopy(params)
    num_samples = sim_params["simParameters"]["numSamples"]
    initial_values = copy.deepcopy(sim_params["simParameters"]["initialValues"])
    external_inputs = copy.deepcopy(sim_params["simParameters"]["externalInputs"])
    clamp_params = copy.deepcopy(sim_params["clampParameters"])
    num_iters = sim_params["simParameters"]["numSimIters"]

    stig_model = model(sim_params, numBasicSamples=num_samples)
    stig_model.setExperimentalConditions((initial_values, num_samples))
    stig_model.simulate(
        externalInputs=external_inputs,
        clampParameters=clamp_params,
        perturbation=None,
        fieldModulation=False,
        numSimIters=num_iters,
    )
    return stig_model


def derive_face_set_point(stig_model, snap_strength=0.35):
    lattice_dims = stig_model.parameters["latticeDims"]
    vmem_snapshot = stig_model.electricNetwork.Vmem.detach().clone()
    coordinator = FacePatternCoordinator(
        latticeDims=lattice_dims,
        gene_names=None,
        device=vmem_snapshot.device,
        dtype=vmem_snapshot.dtype,
    )
    set_point = coordinator.derive_set_point(vmem_snapshot)
    return coordinator, set_point


def run_facial_grn(set_point, lattice_dims, snap_strength=0.35, num_iters=200, bioelectric_prepattern=True):
    facial = FacialGRN(grid_size=lattice_dims[0], device="cpu")
    facial.bioelectric_prepattern_enabled = bioelectric_prepattern
    if bioelectric_prepattern:
        facial.register_face_set_point(set_point, snap_strength=0.0)
        facial.register_bioelectric_prepattern(set_point, weight=snap_strength)
    facial.simulate(numSimIters=num_iters)
    return facial


def summarize_features(feature_mask, label):
    unique, counts = torch.unique(feature_mask, return_counts=True)
    mapping = {0: "bone", 1: "eye", 2: "nose", 3: "jaw"}
    stats = {mapping.get(int(k.item()), "unknown"): int(v.item()) for k, v in zip(unique, counts)}
    print(f"{label} feature counts:")
    for name in ["bone", "eye", "nose", "jaw"]:
        print(f"  {name}: {stats.get(name, 0)}")


def plot_results(stig_model, set_point, facial_grn, output_path):
    rows, cols = stig_model.parameters["latticeDims"]
    vmem_grid = stig_model.electricNetwork.Vmem.view(rows, cols).cpu().numpy()
    feature_mask = set_point["feature_mask_grid"][0].cpu().numpy()
    grn_state = facial_grn.get_state()
    grn_feature = grn_state["features"].cpu().numpy()
    pax6 = grn_state["genes"]["pax6"].cpu().numpy()
    from matplotlib.colors import ListedColormap
    feature_cmap = ListedColormap(["#f9f9f9", "#9b59b6", "#e67e22", "#2ecc71"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax = axes[0, 0]
    vm_plot = ax.imshow(vmem_grid, cmap="coolwarm")
    ax.set_title("Stigmergic Vmem (final)")
    fig.colorbar(vm_plot, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    mask_plot = ax.imshow(feature_mask, cmap=feature_cmap, vmin=0, vmax=3)
    ax.set_title("Derived Face Mask (electric)")
    fig.colorbar(mask_plot, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    grn_plot = ax.imshow(grn_feature, cmap=feature_cmap, vmin=0, vmax=3)
    ax.set_title("FacialGRN Feature Map")
    fig.colorbar(grn_plot, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    pax_plot = ax.imshow(pax6, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("FacialGRN Pax6 Expression")
    fig.colorbar(pax_plot, ax=ax, fraction=0.046, pad=0.04)

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle("Stigmergic Bioelectric → FacialGRN Integration", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def blend_vmem_with_grn(stig_model, facial_grn, feedback_strength, target_map=None):
    feature_map = facial_grn.get_feature_map()
    stig_model.electricNetwork.apply_gene_voltage_feedback(feature_map, gain=feedback_strength)


def update_face_prepattern(facial_grn, set_point, weight, enabled=True):
    if not enabled:
        return
    facial_grn.face_set_point = set_point
    facial_grn.register_bioelectric_prepattern(set_point, weight=weight)


def bidirectional_coupling(
    stig_model,
    facial_grn,
    coordinator,
    initial_set_point,
    cycles=4,
    grn_steps=120,
    electric_steps=200,
    prepattern_weight=0.4,
    feedback_strength=0.2,
    bioelectric_prepattern=True,
):
    current_set_point = initial_set_point
    update_face_prepattern(facial_grn, current_set_point, prepattern_weight, enabled=bioelectric_prepattern)
    ext_inputs_electric = {"gene": None}
    for _ in range(cycles):
        facial_grn.simulate(numSimIters=grn_steps)
        blend_vmem_with_grn(stig_model, facial_grn, feedback_strength)
        stig_model.electricNetwork.simulate(
            externalInputs=ext_inputs_electric,
            numSimIters=electric_steps,
            outerIter=0,
            stochasticIonChannels=False,
            fieldModulation=False,
            setGradient=False,
            retainGradients=False,
            saveData=False,
        )
        current_set_point = coordinator.derive_set_point(stig_model.electricNetwork.Vmem.detach().clone())
        update_face_prepattern(facial_grn, current_set_point, prepattern_weight, enabled=bioelectric_prepattern)
    return current_set_point


def main():
    params = load_stigmergic_parameters("data/StigmergicModelParameters.dat")
    stig_model = run_stigmergic_simulation(params)
    coordinator, set_point = derive_face_set_point(stig_model, snap_strength=0.4)
    summarize_features(set_point["feature_mask"][0, :, 0], "Bioelectric")
    use_bioelectric_prepattern = True
    facial_grn = run_facial_grn(
        set_point,
        params["latticeDims"],
        snap_strength=0.4,
        num_iters=200,
        bioelectric_prepattern=use_bioelectric_prepattern,
    )
    final_set_point = bidirectional_coupling(
        stig_model,
        facial_grn,
        coordinator,
        set_point,
        cycles=4,
        grn_steps=120,
        electric_steps=200,
        prepattern_weight=0.4,
        feedback_strength=0.2,
        bioelectric_prepattern=use_bioelectric_prepattern,
    )
    summarize_features(facial_grn.get_state()["features"].flatten(), "FacialGRN")
    output_path = "stigmergic_facial_integration.png"
    print("Generating visualization...")
    plot_results(stig_model, final_set_point, facial_grn, output_path)
    print(f"\nSaved visualization to {output_path}")


if __name__ == "__main__":
    main()
