"""
PCA analysis of G_pol pre-patterns from the random-clamp ensemble.

Loads ensemble_gpol_prepatterns.npy and optionally the 4 trained baseline
G_pol pre-patterns, then produces:

  data/pca_scree.png            — variance explained per PC
  data/pca_loadings.png         — PC1–PC4 loading maps on the 11×11 grid
                                  (boundary vs interior annotated)
  data/pca_scores.png           — scatter of ensemble in PC1–PC2 space,
                                  with baseline patterns overlaid as named points
  data/pca_boundary_vs_interior.png — separate PCA on boundary / interior cells
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
parser.add_argument('--n_components', type=int, default=10)
parser.add_argument('--lattice_dims', type=str, default=None,
                    help='optional; derived from --source_dat when omitted')
parser.add_argument('--source_dat', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--output_prefix', type=str, default='data/pca',
                    help='figures are written as <output_prefix>_<name>.png. Change it for a\n'
                         'different lattice size, otherwise the published 11x11 figures are\n'
                         'overwritten in place.')
args = parser.parse_args()

import ast
numRows, numCols = torch.load(args.source_dat, weights_only=False)['latticeDims']
if args.lattice_dims is not None:
    override = ast.literal_eval(args.lattice_dims)
    if tuple(override) != (numRows, numCols):
        raise SystemExit(f"--lattice_dims {override} contradicts {args.source_dat}, "
                         f"which is {(numRows, numCols)}")
numCells = numRows * numCols
n_components = args.n_components

# --- Load ensemble ---
gpol = np.load(f'{args.ensemble_prefix}_gpol_prepatterns.npy')   # (N, 121)
vmem = np.load(f'{args.ensemble_prefix}_vmem_final.npy')          # (N, 121)
N = gpol.shape[0]
print(f"Loaded ensemble: {N} samples, {numCells} cells")
print(f"G_pol range: [{gpol.min():.3e}, {gpol.max():.3e}]")

# --- Derive boundary vs interior cell masks ---
torch.set_grad_enabled(False)
params_base = torch.load(args.source_dat, weights_only=False)
params_base['ATPParameters'] = None
params_base['latticePeriodicBoundaryGJ'] = False
iv = params_base['simParameters']['initialValues']
if 'ligandConc' not in iv:
    iv['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)

utils_obj = utilities.utilities()
m_ref = model(params_base, 1)
circuit_ref = m_ref.electricNetwork
tissueDomeIndices = utils_obj.computeDomeIndices(circuit_ref, mode='tissue')  # boundary cells
boundary_mask = np.zeros(numCells, dtype=bool)
boundary_mask[tissueDomeIndices] = True
interior_mask = ~boundary_mask
print(f"Boundary cells: {boundary_mask.sum()}  Interior cells: {interior_mask.sum()}")
del m_ref, circuit_ref

# --- Load baseline G_pol pre-patterns ---
BASELINES = [
    ('Stigmergic',     'data/StigmergicModelParameters.dat'),
    ('ap_band',        'data/bestModelParameters_fieldVector_ap_band_1.dat'),
    ('stripes',        'data/bestModelParameters_fieldVector_stripes_1.dat'),
    ('triangular_wave','data/bestModelParameters_fieldVector_triangular_wave_1.dat'),
]

baseline_gpol = {}
for name, path in BASELINES:
    try:
        p = torch.load(path, weights_only=False)
        p['ATPParameters'] = None
        p['latticePeriodicBoundaryGJ'] = False
        iv_b = p['simParameters']['initialValues']
        if 'ligandConc' not in iv_b:
            iv_b['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
        m_b = model(p, 1)
        m_b.setExperimentalConditions((iv_b, 1))
        clamp = p['clampParameters']
        pre_idx = clamp['clampEndIter'] + 1
        m_b.simulate(externalInputs=p['simParameters']['externalInputs'],
                     clampParameters=clamp, perturbation=None,
                     numSimIters=pre_idx + 1)
        baseline_gpol[name] = m_b.electricNetwork.G_pol.detach().numpy().reshape(numCells)
        del m_b
        print(f"  Loaded baseline: {name}")
    except Exception as e:
        print(f"  Skipped {name}: {e}")

# ============================================================
# 1. Full PCA on all 121 cells
# ============================================================
n_components = min(n_components, N, numCells)
pca = PCA(n_components=n_components)
scores = pca.fit_transform(gpol)           # (N, n_components)
loadings = pca.components_                 # (n_components, 121)
explained = pca.explained_variance_ratio_

print(f"\nVariance explained by top {n_components} PCs:")
cumvar = 0
for k in range(n_components):
    cumvar += explained[k]
    print(f"  PC{k+1}: {explained[k]*100:.1f}%  (cumulative {cumvar*100:.1f}%)")

# Project baselines into PC space
baseline_scores = {}
for name, gp in baseline_gpol.items():
    baseline_scores[name] = pca.transform(gp.reshape(1, -1))[0]

# ============================================================
# 2. Scree plot
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, n_components + 1), explained * 100, color='steelblue', edgecolor='k', linewidth=0.5)
ax.plot(range(1, n_components + 1), np.cumsum(explained) * 100, 'o-', color='crimson', label='Cumulative')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('G_pol pre-pattern PCA — Scree')
ax.legend()
plt.tight_layout()
plt.savefig(f'{args.output_prefix}_scree.png', dpi=150)
plt.close()

# ============================================================
# 3. Loading maps for PC1–PC4
# ============================================================
n_pc_show = min(4, n_components)
fig, axes = plt.subplots(1, n_pc_show, figsize=(4 * n_pc_show, 4))
if n_pc_show == 1:
    axes = [axes]
for k in range(n_pc_show):
    lmap = loadings[k].reshape(numRows, numCols)
    vabs = np.abs(lmap).max()
    im = axes[k].imshow(lmap, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    axes[k].set_title(f'PC{k+1} loadings\n({explained[k]*100:.1f}% var)')
    plt.colorbar(im, ax=axes[k], fraction=0.046)
    # Annotate boundary ring
    for idx in tissueDomeIndices:
        r, c = divmod(idx, numCols)
        axes[k].add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                         fill=False, edgecolor='lime', linewidth=0.8))
plt.suptitle('G_pol pre-pattern PC loadings  (green = boundary cells)', fontsize=10)
plt.tight_layout()
plt.savefig(f'{args.output_prefix}_loadings.png', dpi=150)
plt.close()

# ============================================================
# 4. PC1 vs PC2 scatter with baselines overlaid
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(scores[:, 0], scores[:, 1], s=8, alpha=0.4, color='steelblue', label='Ensemble')
colors = ['crimson', 'darkorange', 'green', 'purple']
markers = ['*', 'D', 's', '^']
for (name, _), color, marker in zip(BASELINES, colors, markers):
    if name in baseline_scores:
        sc = baseline_scores[name]
        ax.scatter(sc[0], sc[1], s=180, color=color, marker=marker,
                   zorder=5, label=name, edgecolors='k', linewidths=0.8)
ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
ax.set_title('G_pol pre-pattern PC space\n(baselines overlaid on random-clamp ensemble)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{args.output_prefix}_scores.png', dpi=150)
plt.close()

# ============================================================
# 5. Separate PCA: boundary vs interior
# ============================================================
fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 3, figure=fig)

for panel_idx, (label, mask) in enumerate([('Boundary', boundary_mask),
                                            ('Interior', interior_mask)]):
    gpol_sub = gpol[:, mask]        # (N, n_sub)
    n_sub = gpol_sub.shape[1]
    n_pc_sub = min(5, N, n_sub)
    pca_sub = PCA(n_components=n_pc_sub)
    scores_sub = pca_sub.fit_transform(gpol_sub)
    expl_sub = pca_sub.explained_variance_ratio_

    ax = fig.add_subplot(gs[panel_idx])
    ax.bar(range(1, n_pc_sub + 1), expl_sub * 100, color='steelblue' if label == 'Boundary' else 'darkorange',
           edgecolor='k', linewidth=0.5)
    ax.plot(range(1, n_pc_sub + 1), np.cumsum(expl_sub) * 100, 'o-', color='crimson')
    ax.set_title(f'{label} cells ({n_sub} cells)\nPC1={expl_sub[0]*100:.1f}%, '
                 f'top-2={np.sum(expl_sub[:2])*100:.1f}%')
    ax.set_xlabel('PC')
    ax.set_ylabel('Var. explained (%)')
    ax.set_ylim(0, 105)

# Panel 3: boundary vs interior PC1 explained variance bar chart
ax3 = fig.add_subplot(gs[2])
regions = ['Full\n(121)', 'Boundary\n(%d)' % boundary_mask.sum(),
           'Interior\n(%d)' % interior_mask.sum()]
pc1_vars = []
for mask in [np.ones(numCells, dtype=bool), boundary_mask, interior_mask]:
    pca_tmp = PCA(n_components=1)
    pca_tmp.fit(gpol[:, mask])
    pc1_vars.append(pca_tmp.explained_variance_ratio_[0] * 100)
ax3.bar(regions, pc1_vars, color=['gray', 'steelblue', 'darkorange'], edgecolor='k')
ax3.set_ylabel('PC1 variance explained (%)')
ax3.set_title('PC1 concentration\nby region')
for j, v in enumerate(pc1_vars):
    ax3.text(j, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

plt.suptitle('Boundary vs Interior G_pol pre-pattern PCA', fontsize=11)
plt.tight_layout()
plt.savefig(f'{args.output_prefix}_boundary_vs_interior.png', dpi=150)
plt.close()

print("\nFigures saved:")
print(f"  {args.output_prefix}_scree.png")
print(f"  {args.output_prefix}_loadings.png")
print(f"  {args.output_prefix}_scores.png")
print(f"  {args.output_prefix}_boundary_vs_interior.png")
