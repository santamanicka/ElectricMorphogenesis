#!/usr/bin/env python3
"""
Schematic: Cross-Embryo Stress-Based Rescue Signaling Paradigm

Single-panel figure showing:
  - Within-embryo circular feedback (GRN -> Vmem -> field -> Ca2+ -> Stress -> GRN)
  - Between-embryo stress coupling (donor S(t) -> recipient GRN damping)
  - CEMA constraint: only stressed embryos participate (healthy = zero signal)

Usage:
    python visualize_rescue_schematic.py
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='data/rescue_signaling_schematic')
args = parser.parse_args()

rng = np.random.default_rng(42)

# ============================================================
# COLOR PALETTE
# ============================================================
C = {
    'vmem':    '#2E86AB',
    'ca':      '#F6AE2D',
    'stress':  '#E94F37',
    'grn':     '#2ECC71',
    'gj':      '#7B68EE',
    'field':   '#1ABC9C',
    'txt':     '#2C3E50',
    'healthy': '#27AE60',
    'damaged': '#C0392B',
    'rescued': '#2980B9',
    'cycle':   '#8E44AD',
    'couple':  '#E67E22',
}


# ============================================================
# SYNTHETIC DATA
# ============================================================
def make_vmem(kind, n=5):
    if kind == 'healthy':
        v = np.zeros((n, n))
        for cx, cy in [(1.2, 1.2), (3.2, 1.2), (2.0, 3.2)]:
            for i in range(n):
                for j in range(n):
                    v[i, j] += 0.85 * np.exp(-((i - cy)**2 + (j - cx)**2) / 1.0)
        return gaussian_filter(v + rng.normal(0, 0.04, (n, n)), 0.4)
    elif kind == 'damaged':
        return np.abs(rng.normal(0, 0.4, (n, n)))
    else:  # rescued
        h = make_vmem('healthy', n)
        d = make_vmem('damaged', n)
        return 0.70 * h + 0.30 * d


# ============================================================
# FIGURE — single full-page axes
# ============================================================
fig = plt.figure(figsize=(20, 14))
ax = fig.add_axes([0.02, 0.04, 0.96, 0.90])
ax.set_xlim(-2, 20)
ax.set_ylim(0, 13)
ax.axis('off')
ax.set_aspect('equal')
fig.patch.set_facecolor('white')

# ============================================================
# TITLE
# ============================================================
fig.text(0.5, 0.97 - 0.05,
         'Cross-Embryo Stress-Based Rescue Signaling Paradigm',
         fontsize=20, fontweight='bold', ha='center', va='top', color=C['txt'])
fig.text(0.5, 0.945 - 0.05,
         'Intra-embryo bioelectric feedback  |  Inter-embryo stress coupling  |  GRN damping rescue | GRN → Field → Stress negative feedback',
         fontsize=11, ha='center', va='top', color='#7f8c8d', style='italic')


# ============================================================
# HELPERS
# ============================================================
def rounded_box(x, y, w, h, color, fill_alpha=0.12,
                lw=2.2, label=None, label_fs=11, label_y=None):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0.08,rounding_size=0.25',
                         facecolor=color, alpha=fill_alpha,
                         edgecolor=color, linewidth=lw)
    ax.add_patch(box)
    if label:
        ly = label_y if label_y is not None else y + h + 0.15
        ax.text(x + w / 2, ly, label, fontsize=label_fs,
                fontweight='bold', ha='center', va='bottom', color=color)


def flow_arrow(x1, y1, x2, y2, color='#5D6D7E', lw=2.0, rad=0.0,
               label='', label_side='right', head=15, style='->',
               label_fs=8.5):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        connectionstyle=f'arc3,rad={rad}',
        mutation_scale=head, linewidth=lw, color=color)
    ax.add_patch(arrow)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ha = 'left' if label_side == 'right' else 'right'
        off = 0.18 if label_side == 'right' else -0.18
        ax.text(mx + off, my, label, fontsize=label_fs, color=color,
                ha=ha, va='center', style='italic')


def draw_tissue_grid(cx, cy, vmem, size=2.2, show_field=True):
    n = 5
    cell_r = size / n * 0.40
    spacing = size / (n - 1)
    x0, y0 = cx - size / 2, cy - size / 2
    vn = (vmem - vmem.min()) / (vmem.max() - vmem.min() + 1e-8)

    # Gap junction lines
    for i in range(n):
        for j in range(n):
            px = x0 + j * spacing
            py = y0 + (n - 1 - i) * spacing
            if j < n - 1:
                px2 = x0 + (j + 1) * spacing
                gw = 0.6 + 1.4 * 0.5 * (vn[i, j] + vn[i, j + 1])
                ax.plot([px, px2], [py, py], color=C['gj'], lw=gw, alpha=0.4, zorder=1)
            if i < n - 1:
                py_next = y0 + (n - 2 - i) * spacing
                gw = 0.6 + 1.4 * 0.5 * (vn[i, j] + vn[i + 1, j])
                ax.plot([px, px], [py, py_next], color=C['gj'], lw=gw, alpha=0.4, zorder=1)

    # Cells coloured by Vmem
    for i in range(n):
        for j in range(n):
            px = x0 + j * spacing
            py = y0 + (n - 1 - i) * spacing
            col = plt.cm.RdBu_r(0.2 + 0.6 * vn[i, j])
            c = Circle((px, py), cell_r, facecolor=col, edgecolor='#555',
                        lw=0.7, zorder=2)
            ax.add_patch(c)

    # Electric field arrows in cell-square centres (4x4 squares for a 5x5 grid)
    if show_field:
        gy, gx = np.gradient(vmem)
        ex, ey = -gx, gy
        scale = spacing * 0.32
        for i in range(n - 1):
            for j in range(n - 1):
                avg_ex = 0.25 * (ex[i, j] + ex[i, j+1] + ex[i+1, j] + ex[i+1, j+1])
                avg_ey = 0.25 * (ey[i, j] + ey[i, j+1] + ey[i+1, j] + ey[i+1, j+1])
                mag = np.sqrt(avg_ex**2 + avg_ey**2) + 1e-8
                avg_ex, avg_ey = avg_ex / mag, avg_ey / mag
                sx = x0 + (j + 0.5) * spacing
                sy = y0 + (n - 1 - i - 0.5) * spacing
                ax.annotate('', xy=(sx + avg_ex * scale, sy + avg_ey * scale),
                            xytext=(sx - avg_ex * scale * 0.7,
                                    sy - avg_ey * scale * 0.7),
                            arrowprops=dict(arrowstyle='->', color=C['field'],
                                            lw=1.3, mutation_scale=12),
                            zorder=3)


def draw_energy_well(cx, cy, w=1.4, h=0.7, stress_val=0.5):
    s = np.linspace(0, 1, 200)
    energy = 4 * (s - 0.15)**2 * (s - 0.85)**2 * 20 - 0.3 * s
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    xs = cx - w / 2 + s * w
    ys = cy - h / 2 + energy * h  # wells point downward (minima = low y)
    ax.plot(xs, ys, color=C['stress'], lw=2, zorder=5)
    ax.fill_between(xs, cy - h / 2 - 0.02, ys, alpha=0.10, color=C['stress'], zorder=4)
    # Place ball at position corresponding to stress_val
    bx = cx - w / 2 + stress_val * w
    bi = int(np.argmin(np.abs(xs - bx)))
    by = ys[bi] + 0.05
    ax.plot(bx, by, 'o', color=C['stress'], ms=6, zorder=6)
    ax.plot([cx, cx], [cy - h / 2, cy + h / 2], '--', color='#bbb', lw=0.8, zorder=4)
    ax.text(cx, cy + h / 2 + 0.08, 'K_S', fontsize=7, ha='center', color='#999')
    ax.text(cx - w * 0.35, cy - h / 2 - 0.3, 'native\npattern', fontsize=7,
            ha='center', color='#3498db', fontweight='bold')
    ax.text(cx + w * 0.35, cy - h / 2 - 0.3, 'stressed\npattern', fontsize=7,
            ha='center', color=C['stress'], fontweight='bold')


# ============================================================
# LAYOUT — spaced-out vertical positions for the cascade
# With 13 units of vertical space we can spread things out nicely.
# ============================================================
# Y positions (top to bottom):
GRN_Y   = 11.0;  GRN_H   = 0.9
TISSUE_Y = 8.0;  TISSUE_H = 2.2
VMEM_Y   = 5.5;  VMEM_H   = 1.1
CA_Y     = 3.5;  CA_H     = 0.8
STRESS_Y = 1.0;  STRESS_H = 1.7

# Embryo centres
DONOR_CX = 4.0
RECIP_CX = 14.0
COUPLE_CX = 9.5

# ============================================================
# BACKGROUND PANELS
# ============================================================
# CEMA: both embryos are stressed (low damping); healthy = zero signal.
rounded_box(1.5, 0.5, 6.0, 11.5, C['stress'], fill_alpha=0.05, lw=2.5,
            label='DONOR  EMBRYO  (stressed, teratogen = 0.5)',
            label_fs=12, label_y=12.15)
rounded_box(11.5, 0.5, 6.0, 11.5, C['rescued'], fill_alpha=0.05, lw=2.5,
            label='RECIPIENT  EMBRYO  (stressed, teratogen = 0.9)',
            label_fs=12, label_y=12.15)


# ============================================================
# DRAW ONE EMBRYO COLUMN
# ============================================================
def draw_embryo(cx, vmem_kind, side='left'):
    is_donor = (side == 'left')

    # ── GRN ──
    grn_w = 3.6
    rounded_box(cx - grn_w / 2, GRN_Y, grn_w, GRN_H,
                C['grn'], fill_alpha=0.15, lw=2)
    label_txt = 'GRN  (reduced weights)' if is_donor else 'GRN  (modulated weights)'
    ax.text(cx, GRN_Y + GRN_H / 2 + 0.05, label_txt,
            fontsize=10, ha='center', va='center',
            color=C['grn'], fontweight='bold')
    damp = 'd = 0.5  (stressed)' if is_donor else 'd = 0.1 + rescue'
    ax.text(cx, GRN_Y + 0.08, damp,
            fontsize=8, ha='center', va='bottom', color=C['grn'], style='italic')

    # ── Arrow: GRN → Tissue ──
    flow_arrow(cx, GRN_Y - 0.08, cx, TISSUE_Y + TISSUE_H + 0.08, C['grn'],
               label='G_dep modulation', label_side='right')

    # ── Tissue + field ──
    rounded_box(cx - 2.2, TISSUE_Y - 0.1, 4.4, TISSUE_H + 0.1,
                C['gj'], fill_alpha=0.08, lw=1.8)
    ax.text(cx, TISSUE_Y + TISSUE_H - 0.1, 'Ion Channels + Gap Junctions + Field',
            fontsize=9, ha='center', va='bottom', color=C['gj'], fontweight='bold')
    vmem = make_vmem(vmem_kind)
    draw_tissue_grid(cx, TISSUE_Y + TISSUE_H / 2 - 0.15, vmem, size=1.8)
    ax.text(cx + 1.0, TISSUE_Y - 0.35, 'Electric field (E = −∇Vmem)',
            fontsize=7.5, ha='center', va='bottom', color=C['field'], style='italic')

    # ── Arrow: Tissue → Vmem ──
    flow_arrow(cx, TISSUE_Y - 0.1, cx, VMEM_Y + VMEM_H + 0.1, C['vmem'],
               label='Vmem pattern', label_side='left')

    # ── Vmem heatmap ──
    rounded_box(cx - 1.8, VMEM_Y, 3.6, VMEM_H,
                C['vmem'], fill_alpha=0.10, lw=1.5)
    sq = 0.20;  gap = 0.025;  step = sq + gap
    total = 5 * sq + 4 * gap
    sx0 = cx - total / 2
    sy_top = VMEM_Y + VMEM_H / 2 + total / 2
    vn = (vmem - vmem.min()) / (vmem.max() - vmem.min() + 1e-8)
    for i in range(5):
        for j in range(5):
            col = plt.cm.RdBu_r(0.2 + 0.6 * vn[i, j])
            r = Rectangle((sx0 + j * step, sy_top - (i + 1) * step),
                           sq, sq, facecolor=col, edgecolor='white', lw=0.4, zorder=2)
            ax.add_patch(r)
    ax.text(cx - total / 2 - 0.2, VMEM_Y + VMEM_H / 2, 'Vmem',
            fontsize=9, ha='right', va='center', color=C['vmem'], fontweight='bold')

    # ── Arrow: Vmem → Ca²⁺ ──
    flow_arrow(cx, VMEM_Y - 0.08, cx, CA_Y + CA_H + 0.08, C['ca'],
               label='Voltage-gated Calcium Channel', label_side='right')

    # ── Ca²⁺ ──
    rounded_box(cx - 1.8, CA_Y, 3.6, CA_H,
                C['ca'], fill_alpha=0.15, lw=1.5)
    ax.text(cx, CA_Y + CA_H / 2, 'Ca²⁺  (intracellular)',
            fontsize=10, ha='center', va='center', color=C['ca'], fontweight='bold')

    # ── Arrow: Ca → Stress ──
    flow_arrow(cx, CA_Y - 0.08, cx, STRESS_Y + STRESS_H + 0.08, C['stress'],
               label='bistable switch', label_side='left')

    # ── Stress ──
    rounded_box(cx - 2.2, STRESS_Y, 4.4, STRESS_H,
                C['stress'], fill_alpha=0.10, lw=1.8)
    ax.text(cx - 1.0, STRESS_Y + STRESS_H - 0.10, 'Stress  S',
            fontsize=10, ha='center', va='top', color=C['stress'], fontweight='bold')
    mean_s = 0.45 if is_donor else 0.72
    draw_energy_well(cx + 0.7, STRESS_Y + STRESS_H / 2 + 0.05,
                     w=1.5, h=0.65, stress_val=mean_s)
    ax.text(cx - 1.5, STRESS_Y + 0.20,
            f'embryo_stress\n= mean(S) = {mean_s:.2f}',
            fontsize=8, ha='center', va='bottom', color=C['damaged'], fontweight='bold')

    # ── FEEDBACK ARC — bottom→top, bending OUTWARD ──
    # For bottom→top direction: rad > 0 → curves LEFT, rad < 0 → curves RIGHT
    arc_offset = -2.5 if is_donor else 2.5
    arc_x = cx + arc_offset * 0.75
    rad = -0.6 if is_donor else 0.6   # left panel: bend RIGHT (inward); right panel: bend LEFT (inward)

    arc = FancyArrowPatch(
        (arc_x, STRESS_Y + STRESS_H * 0.7),   # START = bottom (Stress)
        (arc_x, GRN_Y + GRN_H * 0.3),         # END   = top (GRN)
        arrowstyle='->', mutation_scale=18, linewidth=2.8,
        color=C['cycle'],
        connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(arc)

    # Feedback labels
    lbl_x = cx + arc_offset * 1.10
    ha = 'right' if is_donor else 'left'
    for lbl, ly in [('S → GRN',      GRN_Y + 0.2),
                     ('field → Vmem', TISSUE_Y + TISSUE_H / 2),
                     ('Ca²⁺ → S',    CA_Y + 0.2)]:
        ax.text(lbl_x + 0.02, ly, lbl, fontsize=7.5, color=C['cycle'],
                ha=ha, va='center', style='italic')
    ax.text(lbl_x, (GRN_Y + STRESS_Y) / 2 + 1.5, 'feedback\ncycle',
            fontsize=8.5, color=C['cycle'], ha=ha, va='center', fontweight='bold')


# ── Draw both embryos ──
draw_embryo(DONOR_CX, 'healthy', 'left')
draw_embryo(RECIP_CX, 'rescued', 'right')


# ============================================================
# CROSS-EMBRYO COUPLING (centre column)
# ============================================================
coup_w = 3.2
rounded_box(COUPLE_CX - coup_w / 2, 3.0, coup_w, 7.5, C['couple'], fill_alpha=0.06, lw=1.8,
            label='Cross-Embryo Coupling', label_fs=11, label_y=10.65)

# S_donor(t) time-series
ts_y, ts_h, ts_w = 8.8, 1.4, 2.4
ax.text(COUPLE_CX, ts_y + ts_h + 0.18, 'Stressed donor S(t)',
        fontsize=8.5, ha='center', va='bottom', color=C['couple'], fontweight='bold')
ax.plot([COUPLE_CX - ts_w / 2, COUPLE_CX + ts_w / 2], [ts_y, ts_y], '-', color='#aaa', lw=0.8)
ax.plot([COUPLE_CX - ts_w / 2, COUPLE_CX - ts_w / 2], [ts_y, ts_y + ts_h], '-', color='#aaa', lw=0.8)
t_pts = np.linspace(0, 1, 80)
s_vals = 0.70 / (1 + np.exp(-12 * (t_pts - 0.35)))
xs_ts = COUPLE_CX - ts_w / 2 + t_pts * ts_w
ys_ts = ts_y + s_vals / 0.80 * ts_h * 0.85
ax.plot(xs_ts, ys_ts, color=C['stress'], lw=2.2)
ax.text(COUPLE_CX - ts_w / 2 - 0.10, ts_y + ts_h / 2, 'S(t)',
        fontsize=8, color=C['stress'], ha='right', va='center', rotation=90)
ax.text(COUPLE_CX, ts_y - 0.15, 'time →', fontsize=7, ha='center', va='top', color='#999')

# Big coupling arrow
arrow_y = 7.0
flow_arrow(COUPLE_CX - 1.4, arrow_y, COUPLE_CX + 1.4, arrow_y, C['couple'], lw=3.5,
           style='fancy,head_length=0.5,head_width=0.35,tail_width=0.12',
           head=20)
ax.text(COUPLE_CX, arrow_y + 0.30, 'S_donor(t)',
        fontsize=10, ha='center', va='bottom', color=C['couple'], fontweight='bold')
ax.text(COUPLE_CX, arrow_y - 0.30, '→ Δ GRN damping',
        fontsize=9, ha='center', va='top', color=C['couple'])

# Formula box
form_y = 4.2
form_w = 2.6
rounded_box(COUPLE_CX - form_w / 2, form_y, form_w, 1.3, C['couple'], fill_alpha=0.08, lw=1.2)
ax.text(COUPLE_CX, form_y + 0.95, 'Effective damping:',
        fontsize=8, ha='center', va='center', color=C['couple'], fontweight='bold')
ax.text(COUPLE_CX, form_y + 0.45,
        'σ( logit(d_base)\n   + α · S_donor(t) )',
        fontsize=8.5, ha='center', va='center', color=C['couple'], fontfamily='monospace')

# Cross-embryo arrows: donor Stress → coupling, coupling → recipient GRN
ax.annotate('', xy=(COUPLE_CX - 1.4, form_y + 0.5 - 0.5), xytext=(DONOR_CX + 2.2, STRESS_Y + STRESS_H * 0.5),
            arrowprops=dict(arrowstyle='->', color=C['couple'], lw=2, mutation_scale=14,
                            connectionstyle='angle,angleA=0,angleB=-90,rad=5'))
ax.text((DONOR_CX + COUPLE_CX) / 2, 2.1, 'stress\nsignal', fontsize=8, color=C['couple'],
        ha='center', va='center', style='italic')

ax.annotate('', xy=(RECIP_CX - 1.8, GRN_Y + GRN_H * 0.4 - 0.4), xytext=(COUPLE_CX + 1.5, arrow_y),
            arrowprops=dict(arrowstyle='->', color=C['couple'], lw=2, mutation_scale=14,
                            connectionstyle='angle,angleA=0,angleB=-90,rad=5'))
ax.text((COUPLE_CX + RECIP_CX) / 2 + 0.08, 9.5, 'Δdamping\nrescue', fontsize=8, color=C['couple'],
        ha='center', va='center', style='italic')


# ============================================================
# LEGEND
# ============================================================
legend_elements = [
    mpatches.Patch(facecolor=C['grn'],    label='GRN (gene network)'),
    mpatches.Patch(facecolor=C['vmem'],   label='Vmem (bioelectric)'),
    mpatches.Patch(facecolor=C['field'],  label='Electric field'),
    mpatches.Patch(facecolor=C['ca'],     label='Ca²⁺ (intracellular)'),
    mpatches.Patch(facecolor=C['stress'], label='Stress S (bistable)'),
    mpatches.Patch(facecolor=C['gj'],     label='Gap junctions'),
    mpatches.Patch(facecolor=C['couple'], label='Cross-embryo coupling'),
    mpatches.Patch(facecolor=C['cycle'],  label='Intra-embryo feedback'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=8,
           fontsize=8.5, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.07),
           handlelength=1.2, handleheight=0.9)

# ============================================================
# SAVE
# ============================================================
plt.savefig(args.output + '.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(args.output + '.pdf', bbox_inches='tight', facecolor='white')
print(f'Saved: {args.output}.png  |  {args.output}.pdf')
plt.close()
