"""
PC perturbation experiment.

Starting from the face G_pol pre-pattern, perturb G_pol along each of the
top K PCs by ±1σ and ±2σ (σ = ensemble standard deviation along that PC),
then run free evolution and record the final Vmem pattern.

Each row of the output figure is one PC; columns are -2σ, -1σ, 0 (face), +1σ, +2σ.
A second figure shows the signed difference from the face for each perturbation.

Outputs:
  data/pc_perturbations_vmem.png   — raw Vmem (per-row colorscale)
  data/pc_perturbations_diff.png   — Δ Vmem relative to face (shared colorscale)
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--n_pcs',    type=int,   default=10)
parser.add_argument('--face_dat', type=str,   default='data/StigmergicModelParameters.dat')
parser.add_argument('--ensemble_prefix', type=str, default='data/ensemble')
parser.add_argument('--num_free_steps',  type=int, default=899)
args = parser.parse_args()

numRows, numCols = 11, 11
numCells = numRows * numCols
N_PCS = args.n_pcs
DELTAS = [-2, -1, 0, 1, 2]
zero_col = DELTAS.index(0)

# ── Fit PCA on ensemble ─────────────────────────────────────────────────────
gpol_ensemble = np.load(f'{args.ensemble_prefix}_gpol_prepatterns.npy')
pca = PCA(n_components=N_PCS)
scores = pca.fit_transform(gpol_ensemble)
sigma = scores.std(axis=0)          # per-PC ensemble σ
explained = pca.explained_variance_ratio_
print(f"PCA fitted on {len(gpol_ensemble)} samples")
print(f"PC σ values: {sigma}")

# ── Load face pre-pattern state ─────────────────────────────────────────────
def load_prepattern(path):
    p = torch.load(path, weights_only=False)
    p['ATPParameters'] = None
    p['latticePeriodicBoundaryGJ'] = False
    iv = p['simParameters']['initialValues']
    if 'ligandConc' not in iv:
        iv['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
    m = model(p, 1)
    m.setExperimentalConditions((iv, 1))
    clamp = p['clampParameters']
    pre_idx = clamp['clampEndIter'] + 1
    m.simulate(externalInputs=p['simParameters']['externalInputs'],
               clampParameters=clamp, perturbation=None,
               numSimIters=pre_idx + 1)
    gpol_pre  = m.electricNetwork.G_pol.detach().clone()   # (numCells,)
    vmem_pre  = m.timeseriesVmem[pre_idx].detach().clone() # (1, numCells, 1)
    G_ref     = float(m.electricNetwork.G_ref)
    return gpol_pre, vmem_pre, G_ref, p

print("Loading face pre-pattern...")
gpol_pre, vmem_pre, G_ref, params = load_prepattern(args.face_dat)
gpol_pre_np = gpol_pre.numpy().reshape(numCells)

min_gpol = 0.0
max_gpol = 2.0 * G_ref

print(f"G_ref = {G_ref:.3e},  G_pol range in pre-pattern: "
      f"[{gpol_pre_np.min():.3e}, {gpol_pre_np.max():.3e}]")

# ── Run free evolution from a given G_pol ────────────────────────────────────
def run_from_gpol(params, vmem_init, gpol_np, num_steps):
    """Start from (vmem_init, eV=0, gpol_np), no clamp, run num_steps."""
    p = params
    m = model(p, 1)
    m.setExperimentalConditions((p['simParameters']['initialValues'], 1))
    c = m.electricNetwork

    iv_ov = dict(p['simParameters']['initialValues'])
    iv_ov['Vmem'] = vmem_init.clone().double()
    iv_ov['eV']   = torch.zeros((1, c.numFieldGridPoints, 1), dtype=torch.float64)
    c.initVariables(iv_ov)
    c.G_pol = torch.DoubleTensor(gpol_np).reshape_as(c.G_pol)

    m.simulate(externalInputs=p['simParameters']['externalInputs'],
               clampParameters=None, perturbation=None,
               numSimIters=num_steps)
    return c.Vmem[0, :, 0].detach().numpy()

# ── Run all perturbations ─────────────────────────────────────────────────────
# results[k, j] = final Vmem for PC k+1, delta DELTAS[j]
results = np.zeros((N_PCS, len(DELTAS), numCells))

print(f"\nRunning {N_PCS} PCs × {len(DELTAS)} perturbation levels = "
      f"{N_PCS * len(DELTAS)} simulations ({args.num_free_steps} steps each)...")

# Run face baseline (delta=0) once; reuse for all rows
print(f"  Face baseline (delta=0)...")
face_vmem = run_from_gpol(params, vmem_pre, gpol_pre_np, args.num_free_steps)
for k in range(N_PCS):
    results[k, zero_col] = face_vmem

# Perturbations
for k in range(N_PCS):
    pc_vec = pca.components_[k]          # (numCells,)
    sig_k  = sigma[k]
    for j, delta in enumerate(DELTAS):
        if delta == 0:
            continue
        perturbed = gpol_pre_np + delta * sig_k * pc_vec
        perturbed = np.clip(perturbed, min_gpol, max_gpol)
        clip_frac = np.mean(
            (gpol_pre_np + delta * sig_k * pc_vec < min_gpol) |
            (gpol_pre_np + delta * sig_k * pc_vec > max_gpol)
        )
        vmem_final = run_from_gpol(params, vmem_pre, perturbed, args.num_free_steps)
        results[k, j] = vmem_final
        print(f"  PC{k+1:2d}  delta={delta:+d}σ  "
              f"clip_frac={clip_frac:.2f}  "
              f"Vmem range [{vmem_final.min()*1000:.1f}, {vmem_final.max()*1000:.1f}] mV")

# ── Figure 1: Raw Vmem, per-row colorscale ────────────────────────────────────
col_labels = [f'{d:+d}σ' if d != 0 else '0  (face)' for d in DELTAS]
fig1, axes1 = plt.subplots(N_PCS, len(DELTAS),
                            figsize=(len(DELTAS) * 2.0, N_PCS * 2.0))

for k in range(N_PCS):
    row_min = results[k].min()
    row_max = results[k].max()
    for j in range(len(DELTAS)):
        ax = axes1[k, j]
        vmap = results[k, j].reshape(numRows, numCols)
        ax.imshow(vmap, cmap='RdBu_r', vmin=row_min, vmax=row_max)
        ax.set_xticks([])
        ax.set_yticks([])
        # Highlight face column
        if DELTAS[j] == 0:
            for spine in ax.spines.values():
                spine.set_edgecolor('gold')
                spine.set_linewidth(2.5)
        if k == 0:
            ax.set_title(col_labels[j], fontsize=8)
    # Row label
    axes1[k, 0].set_ylabel(f'PC{k+1}\n({explained[k]*100:.1f}%)',
                            fontsize=7, rotation=0, labelpad=38, va='center')

plt.suptitle('PC perturbations from face pre-pattern — final Vmem\n'
             '(per-row colorscale; gold border = unperturbed face)', fontsize=10)
plt.tight_layout()
plt.savefig('data/pc_perturbations_vmem.png', dpi=150)
plt.close()
print("\nSaved: data/pc_perturbations_vmem.png")

# ── Figure 2: Δ Vmem relative to face, shared colorscale ─────────────────────
diffs = results - results[:, zero_col:zero_col+1, :]   # (N_PCS, n_deltas, numCells)
# Use only non-zero columns for colorscale
nonzero_diffs = np.concatenate([diffs[:, j] for j in range(len(DELTAS)) if DELTAS[j] != 0])
vabs = np.percentile(np.abs(nonzero_diffs), 98)

fig2, axes2 = plt.subplots(N_PCS, len(DELTAS),
                            figsize=(len(DELTAS) * 2.0, N_PCS * 2.0))

for k in range(N_PCS):
    for j in range(len(DELTAS)):
        ax = axes2[k, j]
        dmap = diffs[k, j].reshape(numRows, numCols)
        ax.imshow(dmap, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
        ax.set_xticks([])
        ax.set_yticks([])
        if DELTAS[j] == 0:
            for spine in ax.spines.values():
                spine.set_edgecolor('gold')
                spine.set_linewidth(2.5)
        if k == 0:
            ax.set_title(col_labels[j], fontsize=8)
    axes2[k, 0].set_ylabel(f'PC{k+1}\n({explained[k]*100:.1f}%)',
                            fontsize=7, rotation=0, labelpad=38, va='center')

plt.suptitle('PC perturbations — Δ Vmem relative to face\n'
             '(shared colorscale across all panels; red = more depolarised)', fontsize=10)
plt.tight_layout()
plt.savefig('data/pc_perturbations_diff.png', dpi=150)
plt.close()
print("Saved: data/pc_perturbations_diff.png")

# ── Print: which PCs produce the largest pattern change ─────────────────────
print(f"\nMax |ΔVmem| per PC at ±2σ (mV):")
for k in range(N_PCS):
    d_pos = np.abs(diffs[k, DELTAS.index(+2)]).max() * 1000
    d_neg = np.abs(diffs[k, DELTAS.index(-2)]).max() * 1000
    print(f"  PC{k+1:2d}: +2σ → {d_pos:.2f} mV,  -2σ → {d_neg:.2f} mV")
