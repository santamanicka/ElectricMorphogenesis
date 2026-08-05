"""
Characterize the spatial modes of the G_pol pre-pattern PCs by:

1. Unrolling the boundary ring into a 1D perimeter strip
2. Plotting PC loadings along the perimeter for PC1–PC6
3. Computing the DFT of each loading profile to identify dominant spatial frequencies
4. Annotating with developmental-axis labels (A-P, L-R, etc.)

Output: data/pca_mode_characterization.png
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
parser.add_argument('--n_components', type=int, default=8)
parser.add_argument('--source_dat', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--output_prefix', type=str, default='data/pca_mode')
parser.add_argument('--lattice_dims', type=str, default='(11,11)')
args = parser.parse_args()

import ast
import ast
numRows, numCols = ast.literal_eval(args.lattice_dims)
numCells = numRows * numCols

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
tissueDomeIndices = utils_obj.computeDomeIndices(circuit_ref, mode='tissue')  # boundary cell indices
del m_ref, circuit_ref

# Build a canonical clockwise perimeter ordering of boundary cells
# Top row: row=0, cols 0..10  (L→R)
# Right col: col=10, rows 1..10 (T→B, skip corner already counted)
# Bottom row: row=10, cols 9..0 (R→L, skip corner)
# Left col: col=0, rows 9..1 (B→T, skip both corners)
perimeter_cells = []
perimeter_labels = []

for c in range(numCols):                          # top row
    perimeter_cells.append(0 * numCols + c)
    perimeter_labels.append(f'T{c}')
for r in range(1, numRows):                       # right column
    perimeter_cells.append(r * numCols + (numCols - 1))
    perimeter_labels.append(f'R{r}')
for c in range(numCols - 2, -1, -1):             # bottom row
    perimeter_cells.append((numRows - 1) * numCols + c)
    perimeter_labels.append(f'B{c}')
for r in range(numRows - 2, 0, -1):              # left column
    perimeter_cells.append(r * numCols + 0)
    perimeter_labels.append(f'L{r}')

perimeter_cells = np.array(perimeter_cells)
P = len(perimeter_cells)  # should be 4*(11-1) = 40

# Verify all boundary cells are included
assert set(perimeter_cells) == set(tissueDomeIndices), \
    f"Perimeter ordering doesn't match dome indices: {set(perimeter_cells).symmetric_difference(set(tissueDomeIndices))}"

print(f"Perimeter length: {P} cells (expected {4*(numRows-1)})")

# Position along perimeter as fraction [0,1), labelled by axis segment
seg_names = ['Top (L→R)', 'Right (T→B)', 'Bottom (R→L)', 'Left (B→T)']
seg_positions = [0, numCols, numCols + numRows - 1, 2*numCols + numRows - 2]  # start indices of each segment

# --- Load ensemble and run PCA ---
gpol = np.load(f'{args.ensemble_prefix}_gpol_prepatterns.npy')  # (N, 121)
N = gpol.shape[0]
n_components = min(args.n_components, N, numCells)
pca = PCA(n_components=n_components)
pca.fit(gpol)
loadings = pca.components_           # (n_components, 121)
explained = pca.explained_variance_ratio_

# Extract boundary loadings in perimeter order
boundary_loadings = loadings[:, perimeter_cells]  # (n_components, P)

# ============================================================
# Figure 1: Perimeter loading profiles + DFT
# ============================================================
n_show = min(6, n_components)
fig = plt.figure(figsize=(16, 3.2 * n_show))
gs = gridspec.GridSpec(n_show, 3, figure=fig, width_ratios=[2, 1.2, 0.8])

theta = np.linspace(0, 2 * np.pi, P, endpoint=False)   # position in radians

for k in range(n_show):
    profile = boundary_loadings[k]           # (P,)

    # ---- Perimeter line plot ----
    ax_line = fig.add_subplot(gs[k, 0])
    ax_line.plot(range(P), profile, 'steelblue', linewidth=1.5)
    ax_line.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax_line.fill_between(range(P), profile, 0, where=(profile > 0),
                          color='crimson', alpha=0.25)
    ax_line.fill_between(range(P), profile, 0, where=(profile < 0),
                          color='steelblue', alpha=0.25)
    # Segment dividers and labels
    for seg_idx, (pos, sname) in enumerate(zip(seg_positions, seg_names)):
        ax_line.axvline(pos, color='gray', linewidth=0.8, linestyle=':')
        if k == 0:
            ax_line.text(pos + 0.5, ax_line.get_ylim()[1] if ax_line.get_ylim()[1] > 0 else 0.02,
                         sname, fontsize=6, rotation=45, va='bottom', ha='left')
    ax_line.set_xlim(0, P - 1)
    ax_line.set_xlabel('Perimeter position', fontsize=8)
    ax_line.set_ylabel('Loading', fontsize=8)
    ax_line.set_title(f'PC{k+1} ({explained[k]*100:.1f}%) — boundary loading profile', fontsize=9)
    ax_line.tick_params(labelsize=7)

    # ---- DFT of loading profile (spatial frequency content) ----
    fft_vals = np.fft.rfft(profile)
    freqs = np.fft.rfftfreq(P)
    power = np.abs(fft_vals) ** 2
    # Convert to wavenumber (number of full cycles around perimeter)
    wavenumbers = np.round(freqs * P).astype(int)

    ax_fft = fig.add_subplot(gs[k, 1])
    ax_fft.bar(wavenumbers[:len(power)], power, color='darkorange', edgecolor='k', linewidth=0.4)
    ax_fft.set_xlabel('Spatial wavenumber (cycles/perimeter)', fontsize=8)
    ax_fft.set_ylabel('Power', fontsize=8)
    ax_fft.set_title(f'PC{k+1} spatial frequency', fontsize=9)
    ax_fft.tick_params(labelsize=7)
    # Annotate dominant mode
    dom_k = wavenumbers[np.argmax(power[1:]) + 1]   # skip k=0 (mean)
    ax_fft.text(0.97, 0.9, f'dom. k={dom_k}', transform=ax_fft.transAxes,
                ha='right', fontsize=8, color='darkred')

    # ---- Grid heatmap (to cross-reference spatial location) ----
    lmap = np.zeros(numCells)
    lmap[perimeter_cells] = profile
    lmap = lmap.reshape(numRows, numCols)
    vabs = np.abs(profile).max()

    ax_grid = fig.add_subplot(gs[k, 2])
    ax_grid.imshow(lmap, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    ax_grid.set_title(f'PC{k+1} grid', fontsize=9)
    ax_grid.axis('off')

plt.suptitle('G_pol pre-pattern PC modes: boundary perimeter analysis', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f'{args.output_prefix}_characterization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: data/pca_mode_characterization.png")

# ============================================================
# Figure 2: Dominant wavenumber summary across all PCs
# ============================================================
fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

dom_wavenumbers = []
dom_powers_fraction = []
for k in range(n_components):
    profile = boundary_loadings[k]
    fft_vals = np.fft.rfft(profile)
    power = np.abs(fft_vals) ** 2
    dom_idx = np.argmax(power[1:]) + 1
    dom_wavenumbers.append(int(np.round(np.fft.rfftfreq(P)[dom_idx] * P)))
    dom_powers_fraction.append(power[dom_idx] / power[1:].sum())

axes[0].bar(range(1, n_components + 1), dom_wavenumbers, color='steelblue', edgecolor='k')
axes[0].set_xlabel('PC index')
axes[0].set_ylabel('Dominant spatial wavenumber')
axes[0].set_title('Dominant wavenumber per PC\n(cycles around perimeter)')
axes[0].set_xticks(range(1, n_components + 1))

axes[1].bar(range(1, n_components + 1), [f * 100 for f in dom_powers_fraction],
            color='darkorange', edgecolor='k')
axes[1].set_xlabel('PC index')
axes[1].set_ylabel('Dominant mode power (%)')
axes[1].set_title('Fraction of spectral power in dominant mode')
axes[1].set_xticks(range(1, n_components + 1))

# Axis labels for wavenumber values
wn_names = {0: 'DC', 1: 'dipole\n(A-P or L-R)', 2: 'quadrupole',
            3: 'k=3', 4: 'k=4', 5: 'k=5'}
ax2_twin = axes[0].twiny()
ax2_twin.set_xlim(axes[0].get_xlim())
ax2_twin.set_xticks([])

plt.tight_layout()
plt.savefig(f'{args.output_prefix}_summary.png', dpi=150)
plt.close()
print("Saved: data/pca_mode_summary.png")

# ============================================================
# Print summary table
# ============================================================
print(f"\n{'PC':>4} {'Var%':>6} {'Dom_k':>6} {'Dom_power%':>12} {'Axis label':>20}")
print('-' * 55)
axis_labels = {0: 'DC (global mean)', 1: 'dipole (A-P or L-R)', 2: 'quadrupole',
               3: 'tripole (k=3)', 4: 'k=4 mode', 5: 'k=5 mode'}
for k in range(n_components):
    label = axis_labels.get(dom_wavenumbers[k], f'k={dom_wavenumbers[k]}')
    print(f"{k+1:>4} {explained[k]*100:>6.1f} {dom_wavenumbers[k]:>6}   "
          f"{dom_powers_fraction[k]*100:>10.1f}%   {label:>20}")
