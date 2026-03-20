#!/usr/bin/env python3
"""
Topoiesis Fluid Flow Analogy Animation
=======================================

Animates the fluid flow analogy for the topoiesis model:
  Rain falling uniformly on a flat basin with open drains at the edges.

- Left panel:  Small basin (7x7) — water never reaches the rescue threshold
- Right panel: Large basin (15x15) — interior floods past threshold, edges drain

The steady-state water depth follows the same sin×sin eigenmode as the
stress field in the CEMA group rescue model.

Usage:
    python animate_topoiesis_fluid.py
    python animate_topoiesis_fluid.py --fps 20 --duration 12
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import argparse
import time


# ─── Physics ───────────────────────────────────────────────────────────────

def make_absorbing_laplacian(L):
    """Structural Laplacian for LxL grid with absorbing BCs."""
    N = L * L
    k_max = 4
    D_F, gamma = 0.5, 0.0001
    A = np.zeros((N, N))
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if i > 0:     A[idx, (i-1)*L + j] = 1
            if i < L-1:   A[idx, (i+1)*L + j] = 1
            if j > 0:     A[idx, i*L + (j-1)] = 1
            if j < L-1:   A[idx, i*L + (j+1)] = 1
    return D_F * (k_max * np.eye(N) - A) + gamma * np.eye(N)


def steady_state_field(L, S=1.0):
    """Steady-state field F = L_abs^{-1} * S."""
    return np.linalg.solve(make_absorbing_laplacian(L), S * np.ones(L*L)).reshape(L, L)


def simulate_diffusion(L, S=1.0, dt=0.02, num_steps=600):
    """Time-dependent diffusion with absorbing BCs. Returns (num_steps+1, L, L)."""
    D_F, gamma, k_max = 0.5, 0.0001, 4
    N = L * L
    A = np.zeros((N, N))
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if i > 0:     A[idx, (i-1)*L + j] = 1
            if i < L-1:   A[idx, (i+1)*L + j] = 1
            if j > 0:     A[idx, i*L + (j-1)] = 1
            if j < L-1:   A[idx, i*L + (j+1)] = 1

    F = np.zeros(N)
    S_vec = S * np.ones(N)
    frames = [F.reshape(L, L).copy()]
    for _ in range(num_steps):
        F = F + dt * (D_F * (A @ F - k_max * F) - gamma * F + S_vec)
        F = np.maximum(F, 0)
        frames.append(F.reshape(L, L).copy())
    return np.array(frames)


# ─── Constants ─────────────────────────────────────────────────────────────

C_FIT = 57.2

# Water colormap: transparent → light cyan → azure → deep blue
WATER_CMAP = LinearSegmentedColormap.from_list('water', [
    (0.88, 0.94, 1.0),   # very pale blue (basin floor visible)
    (0.55, 0.78, 0.95),  # sky blue
    (0.25, 0.55, 0.85),  # medium blue
    (0.10, 0.32, 0.68),  # deep blue
    (0.05, 0.18, 0.50),  # navy
], N=256)


# ─── Animation ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Topoiesis fluid flow animation')
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--output', type=str, default='data/topoiesis_fluid_analogy.gif')
    parser.add_argument('--dpi', type=int, default=100)
    args = parser.parse_args()

    num_frames = int(args.fps * args.duration)

    # ─── Simulate ────────────────────────────────────────────────────
    L_small, L_large = 7, 15
    num_sim_steps = 3000  # long enough for both basins to reach ~steady state

    print(f"Simulating {L_small}x{L_small} and {L_large}x{L_large} basins...")
    t0 = time.time()
    frames_s = simulate_diffusion(L_small, dt=0.05, num_steps=num_sim_steps)
    frames_l = simulate_diffusion(L_large, dt=0.05, num_steps=num_sim_steps)
    F_ss_s = steady_state_field(L_small)
    F_ss_l = steady_state_field(L_large)
    print(f"  Done in {time.time()-t0:.1f}s")

    sim_idx = np.linspace(0, num_sim_steps, num_frames, dtype=int)

    # Threshold: same absolute field value for both basins
    F_threshold = F_ss_l.max() * C_FIT / (L_large ** 2)
    z_max = F_ss_l.max() * 1.15

    rescue_s = np.mean(F_ss_s > F_threshold)
    rescue_l = np.mean(F_ss_l > F_threshold)

    print(f"  Small: peak={F_ss_s.max():.1f}, threshold={F_threshold:.1f}, rescue={rescue_s*100:.0f}%")
    print(f"  Large: peak={F_ss_l.max():.1f}, threshold={F_threshold:.1f}, rescue={rescue_l*100:.0f}%")

    # Meshgrids
    Xs, Ys = np.meshgrid(np.arange(1, L_small+1), np.arange(1, L_small+1), indexing='ij')
    Xl, Yl = np.meshgrid(np.arange(1, L_large+1), np.arange(1, L_large+1), indexing='ij')

    # Pre-generate rain
    np.random.seed(42)
    ndrops = 15
    total = ndrops * num_frames
    rain = {
        'xs': np.random.uniform(0.5, L_small+0.5, total),
        'ys': np.random.uniform(0.5, L_small+0.5, total),
        'zs': np.random.uniform(0.55*z_max, 1.05*z_max, total),
        'xl': np.random.uniform(0.5, L_large+0.5, total),
        'yl': np.random.uniform(0.5, L_large+0.5, total),
        'zl': np.random.uniform(0.55*z_max, 1.05*z_max, total),
    }

    # ─── Figure setup ────────────────────────────────────────────────
    # Use GridSpec with proportional widths so the small basin's entire
    # panel (axes frame included) is physically smaller than the large one.
    fig = plt.figure(figsize=(15, 7.5))
    fig.patch.set_facecolor('white')

    gs = GridSpec(1, 2, width_ratios=[L_small, L_large],
                  left=0.01, right=0.99, bottom=0.06, top=0.90, wspace=0.05)
    ax_s = fig.add_subplot(gs[0], projection='3d')
    ax_l = fig.add_subplot(gs[1], projection='3d')

    norm = Normalize(vmin=0, vmax=z_max)

    def style_3d(ax, L, title):
        # Each axis fits its own basin — the GridSpec ratio handles
        # making the left panel physically smaller.
        ax.set_xlim(0, L + 1)
        ax.set_ylim(0, L + 1)
        ax.set_zlim(0, z_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(title, fontsize=13, fontweight='bold', color='#1e293b', pad=8)
        ax.view_init(elev=30, azim=-55)
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#e2e8f0')
        ax.yaxis.pane.set_edgecolor('#e2e8f0')
        ax.zaxis.pane.set_edgecolor('#e2e8f0')
        ax.grid(False)

    def draw_basin_floor(ax, L):
        """Draw a sandy-colored basin floor at z=0."""
        x = np.array([0.5, L+0.5])
        y = np.array([0.5, L+0.5])
        Xf, Yf = np.meshgrid(x, y)
        Zf = np.zeros_like(Xf)
        ax.plot_surface(Xf, Yf, Zf, color='#f5f0e8', alpha=0.6,
                       edgecolor='#d4c9b5', linewidth=0.5)

    def draw_threshold_ring(ax, L, X, Y):
        """Draw the threshold as a red wireframe plane + contour on floor."""
        Z_t = np.full_like(X, F_threshold, dtype=float)
        ax.plot_wireframe(X.astype(float), Y.astype(float), Z_t,
                         color='#ef4444', linewidth=0.6, alpha=0.4,
                         rstride=max(1, L//5), cstride=max(1, L//5))

    def draw_drain_arrows(ax, L):
        """Small arrows at edges showing drainage outward."""
        color = '#94a3b8'
        pos = np.linspace(2, L-1, min(4, L-2))
        for p in pos:
            for (dx, dy, x0, y0) in [(0, -0.7, p, 0.6), (0, 0.7, p, L+0.4),
                                      (-0.7, 0, 0.6, p), (0.7, 0, L+0.4, p)]:
                ax.quiver(x0, y0, 0, dx, dy, 0,
                         color=color, alpha=0.4, arrow_length_ratio=0.5,
                         linewidth=0.7)

    def draw_floor_status(ax, L, X, Y, F_cur):
        """Colored dots on the floor: green = rescued, red = failed."""
        rescued = F_cur > F_threshold
        # Rescued cells
        if np.any(rescued):
            ax.scatter(X[rescued], Y[rescued],
                      np.zeros(np.sum(rescued)) + 0.3,
                      c='#22c55e', s=20, alpha=0.7, marker='s',
                      edgecolors='#16a34a', linewidth=0.3, zorder=5)
        # Failed cells
        if np.any(~rescued):
            ax.scatter(X[~rescued], Y[~rescued],
                      np.zeros(np.sum(~rescued)) + 0.3,
                      c='#fca5a5', s=14, alpha=0.5, marker='s',
                      edgecolors='#ef4444', linewidth=0.2, zorder=4)

    # ─── Frame update ────────────────────────────────────────────────
    def update(fi):
        ax_s.cla()
        ax_l.cla()

        si = sim_idx[fi]
        progress = si / num_sim_steps
        Fs = frames_s[si]
        Fl = frames_l[si]

        # ── Small basin ──────────────────────────────────────────
        style_3d(ax_s, L_small, f'{L_small}x{L_small} basin (below critical size)')
        draw_basin_floor(ax_s, L_small)
        draw_drain_arrows(ax_s, L_small)

        if Fs.max() > 0.5:
            ax_s.plot_surface(Xs.astype(float), Ys.astype(float), Fs,
                             cmap=WATER_CMAP, norm=norm,
                             edgecolor='#3b82f6', linewidth=0.3,
                             alpha=0.88, shade=True)
        draw_threshold_ring(ax_s, L_small, Xs, Ys)

        if progress > 0.4:
            draw_floor_status(ax_s, L_small, Xs, Ys, Fs)

        # Rain
        d0, d1 = fi * ndrops, (fi+1) * ndrops
        if progress < 0.92:
            ax_s.scatter(rain['xs'][d0:d1], rain['ys'][d0:d1], rain['zs'][d0:d1],
                        c='#93c5fd', s=10, alpha=0.6, marker='.')

        # Status
        peak_s = Fs.max()
        if peak_s < F_threshold:
            label_s = f'Peak {peak_s:.0f} < threshold {F_threshold:.0f}'
            col_s = '#dc2626'
        else:
            label_s = f'Peak {peak_s:.0f} above threshold!'
            col_s = '#16a34a'
        ax_s.text2D(0.5, 0.01, label_s, transform=ax_s.transAxes,
                   fontsize=9, ha='center', color=col_s, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col_s, alpha=0.85))

        # Threshold label
        ax_s.text(L_small+0.5, 1, F_threshold*1.05, 'threshold',
                 fontsize=7, color='#dc2626', style='italic', ha='center')

        # ── Large basin ──────────────────────────────────────────
        style_3d(ax_l, L_large, f'{L_large}x{L_large} basin (above critical size)')
        draw_basin_floor(ax_l, L_large)
        draw_drain_arrows(ax_l, L_large)

        if Fl.max() > 0.5:
            ax_l.plot_surface(Xl.astype(float), Yl.astype(float), Fl,
                             cmap=WATER_CMAP, norm=norm,
                             edgecolor='#3b82f6', linewidth=0.15,
                             alpha=0.88, shade=True)
        draw_threshold_ring(ax_l, L_large, Xl, Yl)

        if progress > 0.4:
            draw_floor_status(ax_l, L_large, Xl, Yl, Fl)

        # Rain
        if progress < 0.92:
            ax_l.scatter(rain['xl'][d0:d1], rain['yl'][d0:d1], rain['zl'][d0:d1],
                        c='#93c5fd', s=10, alpha=0.6, marker='.')

        # Status
        peak_l = Fl.max()
        frac = np.mean(Fl > F_threshold)
        if frac > 0:
            label_l = f'Peak {peak_l:.0f} | {frac*100:.0f}% above threshold'
            col_l = '#16a34a'
        else:
            label_l = f'Peak {peak_l:.0f} — filling...'
            col_l = '#d97706'
        ax_l.text2D(0.5, 0.01, label_l, transform=ax_l.transAxes,
                   fontsize=9, ha='center', color=col_l, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col_l, alpha=0.85))

        ax_l.text(L_large+0.5, 1, F_threshold*1.05, 'threshold',
                 fontsize=7, color='#dc2626', style='italic', ha='center')

        if (fi + 1) % 15 == 0:
            print(f'  Frame {fi+1}/{num_frames}')
        return []

    # ─── Title and subtitle ──────────────────────────────────────────
    fig.suptitle('Topoiesis: Rain on a Basin with Edge Drains',
                 fontsize=17, fontweight='bold', color='#0f172a', y=0.97)
    fig.text(0.5, 0.935,
             'Water depth = collective field strength  |  '
             'Red wireframe = rescue threshold  |  '
             'Green squares = rescued cells',
             ha='center', fontsize=9, color='#64748b', style='italic')

    # ─── Render ──────────────────────────────────────────────────────
    print(f"\nRendering {num_frames} frames...")
    t0 = time.time()
    anim = FuncAnimation(fig, update, frames=num_frames, blit=False)

    print(f"Saving GIF to {args.output}...")
    anim.save(args.output, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"  GIF saved ({time.time()-t0:.1f}s)")

    mp4_path = args.output.replace('.gif', '.mp4')
    try:
        print(f"Saving MP4 to {mp4_path}...")
        anim.save(mp4_path, writer=FFMpegWriter(fps=args.fps, bitrate=3000), dpi=args.dpi)
        print(f"  MP4 saved")
    except Exception as e:
        print(f"  MP4 skipped ({e})")

    plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()