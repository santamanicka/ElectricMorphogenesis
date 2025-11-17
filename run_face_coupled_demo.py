#!/usr/bin/env python3
"""
Run a lightweight coupled bioelectric + FacialGRN simulation to demonstrate
the new face set-point workflow and save a diagnostic figure.
"""

import math
import matplotlib

matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt

from cellularFieldNetwork import cellularFieldNetwork
from geneRegulatoryNetwork import FacialGRN
from facePatternCoordinator import FacePatternCoordinator


def build_parameters(lattice_dims):
    """Create a minimal parameter set for the cellular field network."""
    field_params = {
        'fieldEnabled': True,
        'fieldResolution': 1,
        'fieldStrength': 1.0,
        'fieldAggregation': 'average',
        'fieldScreenSize': 4,
        'fieldTransductionWeight': torch.DoubleTensor([600.0]),
        'fieldTransductionBias': torch.DoubleTensor([0.0005]),
        'fieldTransductionGain': -1.0,
        'fieldTransductionTimeConstant': torch.DoubleTensor([10.0]),
        'fieldRangeSymmetric': False,
        'fieldVector': True,
    }
    ligand_params = {
        'ligandEnabled': False,
        'ligandGatingWeight': torch.DoubleTensor([0.5]),
        'ligandGatingBias': torch.DoubleTensor([0.5]),
        'ligandDiffusionStrength': torch.DoubleTensor([1.0]),
        'vmemToLigandTransductionWeight': torch.DoubleTensor([1.0]),
        'tissueConnectivity': torch.zeros(lattice_dims[0] * lattice_dims[1], lattice_dims[0] * lattice_dims[1], dtype=torch.float64),
    }
    params = {
        'latticeDims': lattice_dims,
        'fieldParameters': field_params,
        'GJParameters': {'GJStrength': 0.12},
        'GRNParameters': None,
        'ligandParameters': ligand_params,
        'ATPParameters': None,
    }
    return params


def initialize_state(circuit, seed=0):
    """Seed the circuit with a structured Vmem profile that relaxes into a face-like pattern."""
    torch.manual_seed(seed)
    rows, cols = circuit.latticeDims
    num_cells = circuit.numCells
    num_field_pts = circuit.numFieldGridPoints

    y_coords = torch.linspace(0, 1, rows).unsqueeze(1).repeat(1, cols)
    x_coords = torch.linspace(0, 1, cols).unsqueeze(0).repeat(rows, 1)
    dorsal_gradient = -0.035 - 0.02 * y_coords
    medial_ridge = 0.01 * torch.exp(-((x_coords - 0.5) ** 2) / 0.02)
    noise = 0.002 * torch.randn_like(dorsal_gradient)
    vmem_grid = dorsal_gradient + medial_ridge + noise
    vmem_grid = vmem_grid.clamp(-0.08, -0.01)

    init_values = dict()
    init_values['Vmem'] = vmem_grid.reshape(1, num_cells, 1).contiguous().to(torch.float64)
    init_values['eV'] = torch.zeros((1, num_field_pts, 1), dtype=torch.float64)
    init_values['ligandConc'] = torch.zeros((1, num_cells, 1), dtype=torch.float64)
    init_values['G_pol'] = {
        'cells': [[list(range(num_cells))]],
        'values': [torch.DoubleTensor([1.0])],
    }
    init_values['G_dep'] = {'cells': [], 'values': torch.DoubleTensor([])}

    circuit.initVariables(init_values)
    circuit.initParameters(init_values)


def run_bioelectric_simulation(lattice_dims=(21, 21), num_iters=600, seed=0):
    """Run the bioelectric lattice and return the Vmem snapshot plus coordinator result."""
    params = build_parameters(lattice_dims)
    circuit = cellularFieldNetwork(latticeDims=lattice_dims, parameters=params, numSamples=1)
    initialize_state(circuit, seed=seed)

    external_inputs = {'gene': None, 'ATP': None}
    for iter_idx in range(num_iters):
        circuit.simulate(
            externalInputs=external_inputs,
            numSimIters=1,
            outerIter=iter_idx,
            stochasticIonChannels=False,
            fieldModulation=False,
            setGradient=False,
            retainGradients=False,
            saveData=False,
        )

    vmem_snapshot = circuit.Vmem.detach().clone()
    coordinator = FacePatternCoordinator(
        latticeDims=lattice_dims,
        gene_names=None,
        device=vmem_snapshot.device,
        dtype=vmem_snapshot.dtype,
    )
    set_point = coordinator.derive_set_point(vmem_snapshot)
    return circuit, set_point


def drive_facial_grn(lattice_dims, set_point, num_iters=120):
    """Run FacialGRN with the supplied face set point."""
    grn = FacialGRN(grid_size=lattice_dims[0], device='cpu')
    grn.register_face_set_point(set_point, snap_strength=0.35)
    grn.simulate(numSimIters=num_iters)
    return grn


def plot_results(circuit, set_point, grn, out_path):
    rows, cols = circuit.latticeDims
    vmem_grid = circuit.Vmem.view(rows, cols).cpu().numpy()
    feature_mask = set_point['feature_mask_grid'][0].cpu().numpy()
    grn_state = grn.get_state()
    grn_feature = grn_state['features'].cpu().numpy()
    pax6 = grn_state['genes']['pax6'].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax = axes[0, 0]
    vm = ax.imshow(vmem_grid, cmap='coolwarm')
    ax.set_title('Bioelectric Vmem (a.u.)')
    fig.colorbar(vm, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    fm1 = ax.imshow(feature_mask, cmap='Accent', vmin=1, vmax=3)
    ax.set_title('Bioelectric Feature Mask')
    fig.colorbar(fm1, ax=ax, ticks=[1, 2, 3], fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    fm2 = ax.imshow(grn_feature, cmap='Accent', vmin=1, vmax=3)
    ax.set_title('FacialGRN Feature Map')
    fig.colorbar(fm2, ax=ax, ticks=[1, 2, 3], fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    pax_plot = ax.imshow(pax6, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('FacialGRN Pax6 Expression')
    fig.colorbar(pax_plot, ax=ax, fraction=0.046, pad=0.04)

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle('Coupled Bioelectric ↔ FacialGRN Face Pattern Demo', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    lattice_dims = (21, 21)
    circuit, set_point = run_bioelectric_simulation(lattice_dims=lattice_dims, num_iters=600, seed=3)
    grn = drive_facial_grn(lattice_dims, set_point, num_iters=160)
    output_path = 'face_coupled_demo.png'
    plot_results(circuit, set_point, grn, output_path)

    feature_mask = set_point['feature_mask'][0, :, 0]
    unique, counts = torch.unique(feature_mask, return_counts=True)
    print('Bioelectric feature counts:')
    for feat, cnt in zip(unique.tolist(), counts.tolist()):
        label = {1: 'eye', 2: 'nose', 3: 'jaw'}.get(feat, 'unknown')
        print(f'  {label}: {cnt}')

    grn_features = grn.get_state()['features']
    unique_g, counts_g = torch.unique(grn_features, return_counts=True)
    print('FacialGRN feature counts:')
    for feat, cnt in zip(unique_g.tolist(), counts_g.tolist()):
        label = {1: 'eye', 2: 'nose', 3: 'jaw'}.get(feat, 'unknown')
        print(f'  {label}: {cnt}')

    print(f'\nSaved visualization to {output_path}')


if __name__ == '__main__':
    main()
