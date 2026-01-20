#!/usr/bin/env python3
"""
Track how feature assignments change through bidirectional coupling cycles.
"""

import copy
import torch
import numpy as np
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


def derive_face_set_point(stig_model):
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


def blend_vmem_with_grn(stig_model, facial_grn, feedback_strength):
    gene_fields = facial_grn.get_gene_fields()
    stig_model.electricNetwork.apply_gene_voltage_feedback(gene_fields=gene_fields, gain=feedback_strength)


def update_face_prepattern(facial_grn, set_point, weight):
    facial_grn.face_set_point = set_point
    facial_grn.register_bioelectric_prepattern(set_point, weight=weight)


def check_cell_features(set_point, coordinator, label):
    """Check and report feature assignments for problem cells."""
    rows, cols = coordinator.num_rows, coordinator.num_cols
    feature_grid = set_point["feature_mask_grid"][0]
    feature_map = {0: "bone", 1: "eye", 2: "nose", 3: "jaw"}

    problem_cells = [(3, 3), (3, 7)]
    print(f"\n{label}:")
    for r, c in problem_cells:
        feature_id = int(feature_grid[r, c].item())
        print(f"  Cell ({r}, {c}): {feature_map[feature_id]}")


def compute_detail_at_cells(vmem, coordinator, cells):
    """Compute detail values at specific cells."""
    import torch.nn.functional as F

    rows, cols = coordinator.num_rows, coordinator.num_cols
    vmem_grid = vmem.view(1, rows, cols)

    vnorm = coordinator._normalize(vmem_grid)
    blurred = F.avg_pool2d(vnorm.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    detail = vnorm - blurred
    denom = detail.abs().amax(dim=(1, 2), keepdim=True) + coordinator._eps
    detail = detail / denom

    results = {}
    for r, c in cells:
        results[(r, c)] = {
            'vmem': vmem[0, r*cols + c, 0].item(),
            'normalized': vnorm[0, r, c].item(),
            'blurred': blurred[0, r, c].item(),
            'detail': detail[0, r, c].item(),
        }
    return results


def main():
    # Initial simulation
    params = load_stigmergic_parameters("data/StigmergicModelParameters.dat")
    stig_model = run_stigmergic_simulation(params)

    # Initial set point
    coordinator, initial_set_point = derive_face_set_point(stig_model)
    check_cell_features(initial_set_point, coordinator, "INITIAL (before coupling)")

    problem_cells = [(3, 3), (3, 7)]
    detail_initial = compute_detail_at_cells(stig_model.electricNetwork.Vmem, coordinator, problem_cells)
    print("\n  Detail values:")
    for cell, vals in detail_initial.items():
        print(f"    {cell}: {vals['detail']:.6f}")

    # Create FacialGRN
    facial_grn = FacialGRN(grid_size=params["latticeDims"][0], device="cpu")
    facial_grn.bioelectric_prepattern_enabled = True
    facial_grn.register_face_set_point(initial_set_point, snap_strength=0.0)
    facial_grn.register_bioelectric_prepattern(initial_set_point, weight=0.4)
    facial_grn.simulate(numSimIters=200)

    # Run coupling cycles
    cycles = 4
    grn_steps = 120
    electric_steps = 200
    prepattern_weight = 0.4
    feedback_strength = 0.2
    ext_inputs_electric = {"gene": None}

    current_set_point = initial_set_point

    for cycle in range(cycles):
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle + 1}")
        print('='*60)

        # GRN evolution
        for _ in range(grn_steps):
            facial_grn.updateDynamicalParameters(externalInputs=None)
            facial_grn.updateState()

        # Apply GRN feedback to electric model
        blend_vmem_with_grn(stig_model, facial_grn, feedback_strength)

        # Electric simulation
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

        # Derive new set point
        current_set_point = coordinator.derive_set_point(stig_model.electricNetwork.Vmem.detach().clone())

        # Check features and detail
        check_cell_features(current_set_point, coordinator, f"After cycle {cycle + 1}")
        detail_vals = compute_detail_at_cells(stig_model.electricNetwork.Vmem, coordinator, problem_cells)
        print("\n  Detail values:")
        for cell, vals in detail_vals.items():
            detail_val = vals['detail']
            meets_thresh = "✓" if detail_val >= 0.35 else "✗"
            print(f"    {cell}: {detail_val:.6f} {meets_thresh}")

        # Update GRN with new set point
        update_face_prepattern(facial_grn, current_set_point, prepattern_weight)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Initial: both cells labeled as 'eye'")
    feature_grid = current_set_point["feature_mask_grid"][0]
    feature_map = {0: "bone", 1: "eye", 2: "nose", 3: "jaw"}
    final_labels = []
    for r, c in problem_cells:
        feature_id = int(feature_grid[r, c].item())
        final_labels.append(feature_map[feature_id])
    print(f"Final (after {cycles} cycles): {problem_cells[0]} = '{final_labels[0]}', {problem_cells[1]} = '{final_labels[1]}'")

    if final_labels[0] != "eye" or final_labels[1] != "eye":
        print("\n⚠️  Labels changed during bidirectional coupling!")
        print("   The electric-GRN feedback loop altered the Vmem pattern,")
        print("   causing detail values to drop below the threshold.")


if __name__ == "__main__":
    main()