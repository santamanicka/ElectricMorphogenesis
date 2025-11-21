#!/usr/bin/env python3
"""
Diagnostic script to investigate why cells at row 3, cols 3 and 7 aren't labeled as "eye".
"""

import copy
import torch
import numpy as np
from torch.serialization import add_safe_globals
import numpy

from embryo import model
from facePatternCoordinator import FacePatternCoordinator


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


def diagnose_feature_assignment():
    # Load and run simulation
    params = load_stigmergic_parameters("data/StigmergicModelParameters.dat")
    stig_model = run_stigmergic_simulation(params)

    # Get Vmem and reshape to grid
    lattice_dims = stig_model.parameters["latticeDims"]
    rows, cols = lattice_dims
    vmem_snapshot = stig_model.electricNetwork.Vmem.detach().clone()
    vmem_grid = vmem_snapshot.view(1, rows, cols)

    # Create coordinator and manually compute intermediate values
    coordinator = FacePatternCoordinator(
        latticeDims=lattice_dims,
        gene_names=None,
        device=vmem_snapshot.device,
        dtype=vmem_snapshot.dtype,
    )

    # Compute detail (reproducing the internal logic)
    vnorm = coordinator._normalize(vmem_grid)
    import torch.nn.functional as F
    blurred = F.avg_pool2d(vnorm.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    detail = vnorm - blurred
    denom = detail.abs().amax(dim=(1, 2), keepdim=True) + coordinator._eps
    detail = detail / denom

    # Get templates
    eye_template = coordinator.base_templates[0].to(vmem_grid.device)
    nose_template = coordinator.base_templates[1].to(vmem_grid.device)
    jaw_template = coordinator.base_templates[2].to(vmem_grid.device)

    # Thresholds
    pos_thresh = 0.35
    neg_thresh = -0.35
    template_thresh = 0.15

    # Problem cells
    problem_cells = [(3, 3), (3, 7)]

    print(f"Grid size: {rows}x{cols}")
    print(f"Eye centers (normalized): left=(0.32, 0.32), right=(0.32, 0.68)")
    print(f"Thresholds: pos_thresh={pos_thresh}, template_thresh={template_thresh}\n")

    # Check each problem cell
    for r, c in problem_cells:
        print(f"=== Cell ({r}, {c}) ===")
        print(f"Raw Vmem value: {vmem_snapshot[0, r*cols + c, 0].item():.6f}")
        print(f"Normalized Vmem: {vnorm[0, r, c].item():.6f}")
        print(f"Blurred value: {blurred[0, r, c].item():.6f}")
        print(f"Detail value: {detail[0, r, c].item():.6f}")
        print(f"Eye template value: {eye_template[r, c].item():.6f}")
        print(f"Nose template value: {nose_template[r, c].item():.6f}")
        print(f"Jaw template value: {jaw_template[r, c].item():.6f}")

        # Check conditions
        detail_val = detail[0, r, c].item()
        eye_temp_val = eye_template[r, c].item()

        eye_condition_1 = detail_val >= pos_thresh
        eye_condition_2 = eye_temp_val > template_thresh

        print(f"\nEye conditions:")
        print(f"  detail >= {pos_thresh}: {eye_condition_1} (detail={detail_val:.6f})")
        print(f"  eye_template > {template_thresh}: {eye_condition_2} (template={eye_temp_val:.6f})")
        print(f"  Both satisfied: {eye_condition_1 and eye_condition_2}")

        if not eye_condition_1:
            print(f"\n  ⚠️  PROBLEM: Detail value {detail_val:.6f} is below threshold {pos_thresh}")
            print(f"      Need to increase by {pos_thresh - detail_val:.6f}")

        print()

    # Show detail statistics
    print("=== Detail Map Statistics ===")
    print(f"Min detail: {detail.min().item():.6f}")
    print(f"Max detail: {detail.max().item():.6f}")
    print(f"Mean detail: {detail.mean().item():.6f}")
    print(f"Std detail: {detail.std().item():.6f}")

    # Show where detail exceeds threshold
    high_detail_mask = detail[0] >= pos_thresh
    high_detail_coords = torch.nonzero(high_detail_mask, as_tuple=False)
    print(f"\nCells with detail >= {pos_thresh}:")
    for coord in high_detail_coords:
        r, c = coord.tolist()
        print(f"  ({r}, {c}): detail={detail[0, r, c].item():.6f}")

    # Show actual feature assignment
    set_point = coordinator.derive_set_point(vmem_snapshot)
    feature_grid = set_point["feature_mask_grid"][0]

    print("\n=== Actual Feature Assignments ===")
    feature_map = {0: "bone", 1: "eye", 2: "nose", 3: "jaw"}
    for r, c in problem_cells:
        feature_id = int(feature_grid[r, c].item())
        print(f"Cell ({r}, {c}): {feature_map[feature_id]}")


if __name__ == "__main__":
    diagnose_feature_assignment()