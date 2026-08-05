"""
Compare the intrinsic dimensionality of the G_pol coding space vs the
final Vmem patterning space across the random-clamp ensemble.

Metrics used:
  - Scree plots (variance explained per PC)
  - Participation ratio (PR) = (Σλ)² / Σλ²  — effective number of dimensions
  - Threshold dimensionality: #PCs needed for 80% / 90% cumulative variance
  - Boundary vs interior split for both variables

Output: data/dimensionality_comparison.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
import torch
from embryo import model
import utilities

parser = argparse.ArgumentParser()
parser.add_argument('--ensemble_prefix', type=str, default='data/ensemble')
parser.add_argument('--source_dat',      type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--output_prefix',   type=str, default='data/dimensionality_comparison')
parser.add_argument('--lattice_dims',    type=str, default='(11,11)')
args = parser.parse_args()

import ast
numRows, numCols = ast.literal_eval(args.lattice_dims)
numCells = numRows * numCols

# ── Load data ────────────────────────────────────────────────────────────────
gpol = np.load(f'{args.ensemble_prefix}_gpol_prepatterns.npy')   # (N, 121)
vmem = np.load(f'{args.ensemble_prefix}_vmem_final.npy')          # (N, 121)
N = gpol.shape[0]
print(f"Ensemble: N={N} samples, {numCells} cells")

# ── Get boundary / interior masks ────────────────────────────────────────────
torch.set_grad_enabled(False)
params_base = torch.load(args.source_dat, weights_only=False)
params_base['ATPParameters'] = None
params_base['latticePeriodicBoundaryGJ'] = False
iv = params_base['simParameters']['initialValues']
if 'ligandConc' not in iv:
    iv['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
utils_obj = utilities.utilities()
m_ref = model(params_base, 1)
tissueDomeIndices = utils_obj.computeDomeIndices(m_ref.electricNetwork, mode='tissue')
del m_ref
boundary_mask = np.zeros(numCells, dtype=bool)
boundary_mask[tissueDomeIndices] = True
interior_mask = ~boundary_mask
n_bnd, n_int = boundary_mask.sum(), interior_mask.sum()

# ── Dimensionality metrics ────────────────────────────────────────────────────
def participation_ratio(eigenvalues):
    ev = np.array(eigenvalues)
    return ev.sum() ** 2 / (ev ** 2).sum()

def threshold_dim(explained_ratio, threshold):
    cumvar = np.cumsum(explained_ratio)
    idx = np.searchsorted(cumvar, threshold)
    return int(idx + 1)

def run_pca_analysis(data, label, max_components=None):
    """Run PCA, return dict of metrics + fitted pca object."""
    n_samples, n_features = data.shape
    n_components = min(n_samples - 1, n_features) if max_components is None else max_components
    pca = PCA(n_components=n_components)
    pca.fit(data)
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    pr = participation_ratio(ev)
    d80 = threshold_dim(evr, 0.80)
    d90 = threshold_dim(evr, 0.90)
    print(f"\n{label}:")
    print(f"  Participation ratio:   {pr:.1f}  (effective dimensions)")
    print(f"  Dims for 80% variance: {d80}")
    print(f"  Dims for 90% variance: {d90}")
    print(f"  PC1 explains:          {evr[0]*100:.1f}%")
    print(f"  Top-3 cumulative:      {evr[:3].sum()*100:.1f}%")
    return dict(pca=pca, ev=ev, evr=evr, pr=pr, d80=d80, d90=d90, label=label)

# Full-grid analyses
res_gpol = run_pca_analysis(gpol, 'G_pol (coding space, all 121 cells)')
res_vmem = run_pca_analysis(vmem, 'Vmem  (patterning space, all 121 cells)')

# Boundary-only
res_gpol_bnd = run_pca_analysis(gpol[:, boundary_mask], f'G_pol boundary ({n_bnd} cells)')
res_vmem_bnd = run_pca_analysis(vmem[:, boundary_mask], f'Vmem  boundary ({n_bnd} cells)')

# Interior-only
res_gpol_int = run_pca_analysis(gpol[:, interior_mask], f'G_pol interior ({n_int} cells)')
res_vmem_int = run_pca_analysis(vmem[:, interior_mask], f'Vmem  interior ({n_int} cells)')

# ── Compression ratio ─────────────────────────────────────────────────────────
pr_ratio = res_vmem['pr'] / res_gpol['pr']
print(f"\nPR(Vmem) / PR(G_pol)  = {res_vmem['pr']:.1f} / {res_gpol['pr']:.1f} = {pr_ratio:.2f}")
if pr_ratio < 1:
    print("  → Tissue COMPRESSES: many G_pol addresses map to fewer distinct patterns")
elif pr_ratio > 1:
    print("  → Tissue EXPANDS: dynamics create more diversity than the G_pol seed")
else:
    print("  → Dimensionality preserved")

# ── Figure ────────────────────────────────────────────────────────────────────
n_show = 30   # show top-N PCs in scree plots
# PCA returns at most numSamples-1 components, so a small ensemble yields fewer than n_show.
# The boundary/interior panels below already clamp this; the full-tissue panel did not.
n_show = min(n_show, len(res_gpol['evr']), len(res_vmem['evr']))
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Row 0: Full scree comparison ──────────────────────────────────────────────
ax_scree = fig.add_subplot(gs[0, :2])
ks = np.arange(1, n_show + 1)
ax_scree.plot(ks, res_gpol['evr'][:n_show] * 100, 'o-', color='steelblue',
              linewidth=1.5, markersize=4, label=f"G_pol coding  (PR={res_gpol['pr']:.1f})")
ax_scree.plot(ks, res_vmem['evr'][:n_show] * 100, 's-', color='crimson',
              linewidth=1.5, markersize=4, label=f"Vmem patterning  (PR={res_vmem['pr']:.1f})")
ax_scree.set_xlabel('Principal Component', fontsize=10)
ax_scree.set_ylabel('Variance Explained (%)', fontsize=10)
ax_scree.set_title('Coding vs Patterning space — Scree comparison\n(full 121-cell grid)',
                   fontsize=10)
ax_scree.legend(fontsize=9)
ax_scree.axhline(100 / n_show, color='gray', linewidth=0.8, linestyle='--',
                 label='Uniform (flat scree)')

# ── Row 0: PR bar chart ───────────────────────────────────────────────────────
ax_pr = fig.add_subplot(gs[0, 2])
regions = ['Full\nG_pol', 'Full\nVmem', 'Bnd\nG_pol', 'Bnd\nVmem', 'Int\nG_pol', 'Int\nVmem']
prs     = [res_gpol['pr'], res_vmem['pr'],
           res_gpol_bnd['pr'], res_vmem_bnd['pr'],
           res_gpol_int['pr'], res_vmem_int['pr']]
colors  = ['steelblue', 'crimson', 'steelblue', 'crimson', 'steelblue', 'crimson']
bars = ax_pr.bar(regions, prs, color=colors, edgecolor='k', linewidth=0.6, alpha=0.85)
ax_pr.set_ylabel('Participation Ratio\n(effective dimensions)', fontsize=9)
ax_pr.set_title('Effective dimensionality\nby region', fontsize=10)
for bar, val in zip(bars, prs):
    ax_pr.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}',
               ha='center', fontsize=8)

# ── Row 1: Boundary screes ────────────────────────────────────────────────────
ax_bnd = fig.add_subplot(gs[1, :2])
n_bnd_show = min(n_bnd, n_show)
ks_bnd = np.arange(1, n_bnd_show + 1)
ax_bnd.plot(ks_bnd, res_gpol_bnd['evr'][:n_bnd_show] * 100, 'o-', color='steelblue',
            linewidth=1.5, markersize=4, label=f"G_pol boundary  (PR={res_gpol_bnd['pr']:.1f})")
ax_bnd.plot(ks_bnd, res_vmem_bnd['evr'][:n_bnd_show] * 100, 's-', color='crimson',
            linewidth=1.5, markersize=4, label=f"Vmem boundary  (PR={res_vmem_bnd['pr']:.1f})")
ax_bnd.set_xlabel('PC', fontsize=10)
ax_bnd.set_ylabel('Variance Explained (%)', fontsize=10)
ax_bnd.set_title(f'Boundary cells only ({n_bnd} cells)', fontsize=10)
ax_bnd.legend(fontsize=9)

ax_int = fig.add_subplot(gs[1, 2])
n_int_show = min(n_int, n_show)
ks_int = np.arange(1, n_int_show + 1)
ax_int.plot(ks_int, res_gpol_int['evr'][:n_int_show] * 100, 'o-', color='steelblue',
            linewidth=1.5, markersize=4, label=f"G_pol  (PR={res_gpol_int['pr']:.1f})")
ax_int.plot(ks_int, res_vmem_int['evr'][:n_int_show] * 100, 's-', color='crimson',
            linewidth=1.5, markersize=4, label=f"Vmem  (PR={res_vmem_int['pr']:.1f})")
ax_int.set_xlabel('PC', fontsize=10)
ax_int.set_ylabel('Variance Explained (%)', fontsize=10)
ax_int.set_title(f'Interior cells only ({n_int} cells)', fontsize=10)
ax_int.legend(fontsize=9)

# ── Row 2: Loading maps — Vmem PC1–PC4 ────────────────────────────────────────
gs2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2, :], wspace=0.3)
for col_idx in range(min(4, len(res_vmem['pca'].components_))):
    ax = fig.add_subplot(gs2[col_idx])
    load = res_vmem['pca'].components_[col_idx].reshape(numRows, numCols)
    vabs = np.abs(load).max()
    im = ax.imshow(load, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    ax.set_title(f'Vmem PC{col_idx+1} loading\n({res_vmem["evr"][col_idx]*100:.1f}%)',
                 fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    # Outline boundary
    for idx in tissueDomeIndices:
        r, c = divmod(idx, numCols)
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                   fill=False, edgecolor='lime', linewidth=0.7))
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Coding space (G_pol) vs Patterning space (Vmem) — Dimensionality comparison',
             fontsize=12, y=1.01)
plt.savefig(f'{args.output_prefix}.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: data/dimensionality_comparison.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'':30s} {'PR':>6} {'d80':>5} {'d90':>5} {'PC1%':>7}")
print('-' * 55)
for r in [res_gpol, res_vmem, res_gpol_bnd, res_vmem_bnd, res_gpol_int, res_vmem_int]:
    print(f"{r['label']:30s} {r['pr']:>6.1f} {r['d80']:>5d} {r['d90']:>5d} "
          f"{r['evr'][0]*100:>6.1f}%")
