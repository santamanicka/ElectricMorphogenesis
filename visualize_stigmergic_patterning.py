#!/usr/bin/env python3
"""
Stigmergic Patterning Visualization

Creates a presentation-friendly view of how the stigmergic electric model
self-organizes a voltage pattern when run with data/StigmergicModelParameters.dat.
It runs the same simulation used by simulateTrainedModel.py (Stigmergic mode)
and renders three snapshots (clamped seed, early self-organization, stable
pattern) across Vmem, extracellular field magnitude, and G_pol.

Usage:
    python visualize_stigmergic_patterning.py
    python visualize_stigmergic_patterning.py --num_iters 400 --output stigmergic.png
"""

import argparse
import copy
from typing import List, Sequence

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import torch
from torch.serialization import add_safe_globals
import numpy  # required for safe torch.load

from embryo import model


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize stigmergic pattern formation.")
    parser.add_argument(
        "--params",
        type=str,
        default="data/StigmergicModelParameters.dat",
        help="Path to the stigmergic parameter file.",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=None,
        help="Override simulation length (defaults to file value).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stigmergic_patterning.png",
        help="Output image filename.",
    )
    parser.add_argument(
        "--quiver_stride",
        type=int,
        default=2,
        help="Stride for field vector arrows (larger = fewer arrows).",
    )
    return parser.parse_args()


def load_parameters(path: str):
    add_safe_globals([numpy.core.multiarray._reconstruct])
    params = torch.load(path, weights_only=False)
    if "ATPParameters" not in params:
        params["ATPParameters"] = None
    return params


def run_stigmergic_sim(params, num_iters: int):
    sim_params = copy.deepcopy(params)
    num_samples = sim_params["simParameters"]["numSamples"]
    initial_values = copy.deepcopy(sim_params["simParameters"]["initialValues"])
    external_inputs = copy.deepcopy(sim_params["simParameters"]["externalInputs"])
    clamp_params = copy.deepcopy(sim_params["clampParameters"])

    torch.manual_seed(7)
    model_instance = model(sim_params, numBasicSamples=num_samples)
    model_instance.setExperimentalConditions((initial_values, num_samples))
    model_instance.simulate(
        externalInputs=external_inputs,
        clampParameters=clamp_params,
        perturbation=None,
        fieldModulation=False,
        numSimIters=num_iters,
    )
    return model_instance


def to_cell_grid(flat: torch.Tensor, dims: Sequence[int]) -> np.ndarray:
    return flat.view(dims[0], dims[1]).detach().cpu().numpy()


def to_field_grid(flat: torch.Tensor, index_grid: np.ndarray) -> np.ndarray:
    flat_np = flat.detach().cpu().numpy()
    return flat_np[np.array(index_grid, dtype=int)]


def choose_timepoints(num_iters: int, clamp_end: int) -> List[int]:
    early = min(clamp_end, num_iters - 1)
    growth = min(clamp_end + 80, num_iters - 1)
    final = num_iters - 1
    picks = [0, early, growth, final]
    picks = sorted(set(picks))
    return picks


def add_feedback_diagram(fig, gs, facecolor: str):
    ax = fig.add_subplot(gs)
    ax.set_facecolor(facecolor)
    ax.axis("off")

    nodes = {
        "Vmem": (0.08, 0.5, "#2ecc71"),
        "E-field": (0.38, 0.5, "#00a8e8"),
        "G_pol": (0.68, 0.5, "#f39c12"),
    }
    for label, (x, y, color) in nodes.items():
        circ = Circle((x, y), 0.08, color=color, alpha=0.85, ec="white", lw=2, transform=ax.transAxes)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=12, fontweight="bold", transform=ax.transAxes)

    arrows = [
        ("Vmem", "E-field", "Charges build the local field"),
        ("E-field", "G_pol", "High field suppresses G_pol"),
        ("G_pol", "Vmem", "Lower G_pol depolarizes neighbors"),
    ]
    label_pos = {"Vmem": nodes["Vmem"], "E-field": nodes["E-field"], "G_pol": nodes["G_pol"]}
    for src, dst, text in arrows:
        (x1, y1, _), (x2, y2, _) = label_pos[src], label_pos[dst]
        arrow = FancyArrowPatch(
            (x1 + 0.08, y1),
            (x2 - 0.08, y2),
            arrowstyle="->",
            mutation_scale=14,
            color="white",
            lw=2,
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)
        xm = (x1 + x2) / 2
        ax.text(xm, y1 + 0.12, text, ha="center", va="center", color="#dfe6e9", fontsize=10, transform=ax.transAxes)

    ax.text(
        0.89,
        0.5,
        "Positive feedback → emergent pattern",
        ha="right",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )


def plot_stigmergic(model_instance, timepoints: List[int], output: str, quiver_stride: int):
    rows, cols = model_instance.parameters["latticeDims"]
    num_iters = model_instance.timeseriesVmem.shape[0]
    field_grid_shape = model_instance.electricNetwork.extracellularIndexGrid.shape

    vmem_series = model_instance.timeseriesVmem[:, 0, :, 0]
    gpol_series = model_instance.timeseriesGpol[:, 0, :, 0]
    efield_series = model_instance.timeserieseV[:, 0, :, 0]
    efield_vec_x = model_instance.timeserieseVforceVector[:, 0, 0, :, 0]
    efield_vec_y = model_instance.timeserieseVforceVector[:, 1, 0, :, 0]
    field_index_grid = model_instance.electricNetwork.extracellularIndexGrid

    vmem_frames = [to_cell_grid(vmem_series[t], (rows, cols)) * 1e3 for t in timepoints]  # convert to mV
    gpol_frames = [to_cell_grid(gpol_series[t], (rows, cols)) * 1e9 for t in timepoints]  # convert to nS
    efield_frames = [to_field_grid(efield_series[t], field_index_grid) for t in timepoints]
    efield_vecs = [
        (
            to_field_grid(efield_vec_x[t], field_index_grid),
            to_field_grid(efield_vec_y[t], field_index_grid),
        )
        for t in timepoints
    ]

    vmem_min, vmem_max = min(f.min() for f in vmem_frames), max(f.max() for f in vmem_frames)
    e_min, e_max = min(f.min() for f in efield_frames), max(f.max() for f in efield_frames)
    g_min, g_max = min(f.min() for f in gpol_frames), max(f.max() for f in gpol_frames)

    fig = plt.figure(figsize=(12, 3 * len(timepoints) + 3))
    fig.patch.set_facecolor("#0f0f23")
    gs = gridspec.GridSpec(
        len(timepoints) + 1, 3, height_ratios=[1] * len(timepoints) + [0.6], hspace=0.2, wspace=0.18
    )

    axes = []
    for r, t in enumerate(timepoints):
        row_axes = []
        vmem_grid = vmem_frames[r]
        efield_grid = efield_frames[r]
        gpol_grid = gpol_frames[r]
        vx_grid, vy_grid = efield_vecs[r]

        titles = [
            f"Vmem (t={t})",
            "Extracellular field (clamp + induced)",
            "G_pol (hyperpolarizing conductance)",
        ]
        data = [
            (vmem_grid, plt.cm.coolwarm, vmem_min, vmem_max),
            (efield_grid, plt.cm.magma, e_min, e_max),
            (gpol_grid, plt.cm.Greens, g_min, g_max),
        ]
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor("#0f0f23")
            img = ax.imshow(data[c][0], cmap=data[c][1], vmin=data[c][2], vmax=data[c][3], origin="lower")
            if c == 1 and quiver_stride > 0:
                norm = np.hypot(vx_grid, vy_grid)
                vx_dir = vx_grid / (norm + 1e-12)
                vy_dir = vy_grid / (norm + 1e-12)
                xs = np.arange(field_grid_shape[1])
                ys = np.arange(field_grid_shape[0])
                xs, ys = np.meshgrid(xs, ys)
                ax.quiver(
                    xs[::quiver_stride, ::quiver_stride],
                    ys[::quiver_stride, ::quiver_stride],
                    vx_dir[::quiver_stride, ::quiver_stride],
                    vy_dir[::quiver_stride, ::quiver_stride],
                    color="#f1fa8c",
                    alpha=0.7,
                    scale_units="xy",
                    scale=0.4,
                    width=0.006,
                    pivot="mid",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles[c], color="white", fontsize=11, pad=6)
            ax.text(0.02, 0.92, f"t = {t}", transform=ax.transAxes, color="#dfe6e9", fontsize=9, ha="left")
            for spine in ax.spines.values():
                spine.set_color("#5c5f78")
            row_axes.append((ax, img))
        axes.append(row_axes)

    labels = ["Vmem (mV)", "Field (a.u.)", "G_pol (nS)"]
    for c in range(3):
        imgs = [axes[r][c][1] for r in range(len(timepoints))]
        cbar = fig.colorbar(imgs[0], ax=[axes[r][c][0] for r in range(len(timepoints))], fraction=0.025, pad=0.02)
        cbar.ax.set_ylabel(labels[c], color="white", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    add_feedback_diagram(fig, gs[len(timepoints), :], facecolor="#0f0f23")
    fig.suptitle(
        "Stigmergic electric patterning: cells write and read the field",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(output, dpi=250)
    plt.close(fig)


def main():
    args = parse_args()
    params = load_parameters(args.params)
    num_iters = args.num_iters or params["simParameters"]["numSimIters"]
    clamp_end = params.get("clampParameters", {}).get("clampEndIter", 0)
    timepoints = choose_timepoints(num_iters, clamp_end)
    stig_model = run_stigmergic_sim(params, num_iters)
    plot_stigmergic(stig_model, timepoints, args.output, args.quiver_stride)
    print(f"Saved stigmergic pattern visualization to {args.output}")
    print(f"Timepoints shown: {timepoints} out of {num_iters} iterations.")


if __name__ == "__main__":
    main()
