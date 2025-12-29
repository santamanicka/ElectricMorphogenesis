#!/usr/bin/env python3
"""
Stigmergic Patterning - Essence Visualization

Creates an illustrative animation of the stigmergic feedback loop:
1) Vmem seeds create an extracellular field.
2) Cells sense the local field, which suppresses G_pol.
3) Lower G_pol depolarizes neighbors, reinforcing the field.
4) A stable electric motif emerges from indirect coupling (stigmergy).

This is a pedagogical demo (not a faithful replay of simulateTrainedModel.py).
It uses a lightweight dynamical system to highlight the core idea.

Usage:
    python visualize_stigmergic_essence.py              # on-screen animation
    python visualize_stigmergic_essence.py --save_gif   # save GIF (default name)
    python visualize_stigmergic_essence.py --frames 300 --grid 15
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def smooth_field(arr, kernel):
    """Lightweight 2D convolution with wraparound for a tiled tissue."""
    padded = np.pad(arr, 1, mode="wrap")
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            window = padded[i : i + 3, j : j + 3]
            out[i, j] = np.sum(window * kernel)
    return out


def make_kernel():
    # Simple isotropic kernel approximating short-range sensing (~1–2 cell radii)
    k = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
    return k / k.sum()


# ------------------------------------------------------------
# Core simulation (minimal toy model)
# ------------------------------------------------------------
def simulate(grid_size=13, frames=240, seed_steps=25, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    kernel = make_kernel()

    # State variables
    vmem = -0.18 + 0.01 * rng.standard_normal((grid_size, grid_size))  # volts (small range)
    g_pol = 1.0 + 0.05 * rng.standard_normal((grid_size, grid_size))   # normalized conductance

    # Pre-compute a static "clamp" pattern (mimics the boundary stimulation)
    clamp = np.zeros_like(vmem)
    clamp[[0, -1], :] = 0.6
    clamp[:, [0, -1]] = 0.6
    clamp[grid_size // 2, grid_size // 2] = 0.8  # central attractor

    history = []
    for t in range(frames):
        # Extracellular field built from current Vmem (magnitude only)
        field_raw = smooth_field(vmem, kernel)
        field_mag = np.abs(field_raw)

        # Optionally inject a boundary clamp early on to break symmetry
        if t < seed_steps:
            field_mag = field_mag + clamp

        # G_pol is suppressed by strong local field (negative gain)
        g_target = sigmoid(-(field_mag - 0.12) * 18)  # high field -> low G_pol
        g_pol = g_pol + 0.25 * (g_target - g_pol)

        # Vmem integrates depolarizing field drive and hyperpolarizing G_pol
        dep_drive = 0.75 * field_mag
        hyp_drive = 0.9 * g_pol
        leak = 0.1 * vmem
        vmem = vmem + 0.08 * (dep_drive - hyp_drive - leak)

        # Small noise to start domains
        vmem += 0.002 * rng.standard_normal(vmem.shape)

        history.append(
            {
                "vmem": vmem.copy(),
                "field": field_mag.copy(),
                "gpol": g_pol.copy(),
                "t": t,
            }
        )
    return history


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
def draw_feedback_ax(ax):
    ax.set_facecolor("#0f0f23")
    ax.axis("off")
    nodes = {
        "Vmem": (0.12, 0.5, "#2ecc71"),
        "Field": (0.42, 0.5, "#00a8e8"),
        "G_pol": (0.72, 0.5, "#f39c12"),
    }
    for label, (x, y, color) in nodes.items():
        circ = Circle((x, y), 0.08, color=color, alpha=0.9, ec="white", lw=2, transform=ax.transAxes)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=12, fontweight="bold", transform=ax.transAxes)

    arrows = [
        ("Vmem", "Field", "Charges write field"),
        ("Field", "G_pol", "Field sensed locally\nsuppresses G_pol"),
        ("G_pol", "Vmem", "Lower G_pol depolarizes\nneighbors → more field"),
    ]
    pos = {k: (v[0], v[1]) for k, v in nodes.items()}
    for src, dst, text in arrows:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        arrow = FancyArrowPatch(
            (x1 + 0.1, y1),
            (x2 - 0.1, y2),
            arrowstyle="->",
            mutation_scale=14,
            color="white",
            lw=2,
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)
        ax.text((x1 + x2) / 2, y1 + 0.17, text, ha="center", va="center", color="#dfe6e9", fontsize=9, transform=ax.transAxes)

    ax.text(
        0.5,
        0.16,
        "No coordinates, only local sensing → emergent motif",
        ha="center",
        va="center",
        color="white",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )


def animate(history, save_gif=False, filename="stigmergic_essence"):
    frames = len(history)
    vmax_vmem = max(np.max(h["vmem"]) for h in history)
    vmin_vmem = min(np.min(h["vmem"]) for h in history)
    vmax_field = max(np.max(h["field"]) for h in history)
    vmax_g = max(np.max(h["gpol"]) for h in history)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax in axes:
        ax.set_facecolor("#0f0f23")
        ax.set_xticks([])
        ax.set_yticks([])
    fb_ax = axes[2]
    draw_feedback_ax(fb_ax)

    im_v = axes[0].imshow(history[0]["vmem"], cmap="coolwarm", vmin=vmin_vmem, vmax=vmax_vmem)
    im_f = axes[1].imshow(history[0]["field"], cmap="magma", vmin=0, vmax=vmax_field)

    axes[0].set_title("Vmem (depolarized domains)", color="white", fontsize=11, pad=8)
    axes[1].set_title("Extracellular field |E|", color="white", fontsize=11, pad=8)

    cbar1 = fig.colorbar(im_v, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.ax.set_ylabel("a.u.", color="white")
    cbar1.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar1.ax.axes, "yticklabels"), color="white")

    cbar2 = fig.colorbar(im_f, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.ax.set_ylabel("|E|", color="white")
    cbar2.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar2.ax.axes, "yticklabels"), color="white")

    txt = axes[0].text(0.02, 0.95, "", color="white", fontsize=10, transform=axes[0].transAxes)

    def update(frame):
        snap = history[frame]
        im_v.set_data(snap["vmem"])
        im_f.set_data(snap["field"])
        txt.set_text(f"t = {snap['t']}")
        return [im_v, im_f, txt]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=60, blit=True)
    fig.suptitle("Stigmergy: cells write the field they later read", color="white", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.6)

    if save_gif:
        ani.save(f"{filename}.gif", writer="pillow", dpi=120)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Essence-level stigmergic pattern visualization.")
    parser.add_argument("--frames", type=int, default=240, help="Number of animation frames.")
    parser.add_argument("--grid", type=int, default=13, help="Grid size (cells per side).")
    parser.add_argument("--save_gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--output", type=str, default="stigmergic_essence", help="Output filename stem.")
    args = parser.parse_args()

    hist = simulate(grid_size=args.grid, frames=args.frames)
    animate(hist, save_gif=args.save_gif, filename=args.output)


if __name__ == "__main__":
    main()
