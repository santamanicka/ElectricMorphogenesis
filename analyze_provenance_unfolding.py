"""
How does the provenance map itself unfold over time -- gracefully (smooth, roughly monotonic)
or chaotically (bursty, reversing) -- for the learned clamp versus individual random clamps.

Compares against INDIVIDUAL random seeds, not the ensemble average, since averaging over many
random draws would smooth out whatever per-seed chaos exists and bias the comparison.

Tracks, per facial-feature block, two things over the trajectory:
  1. boundary share -- an interpretable scalar, whether its climb is smooth/monotonic or wobbly
  2. "provenance velocity" -- the step-to-step L1 change in the full pooled provenance vector,
     the same kind of rate-of-change diagnostic used throughout this session for raw Vmem, now
     applied to the provenance map itself

  python analyze_provenance_unfolding.py
"""

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities
from embryo import model

torch.set_grad_enabled(False)
utils = utilities.utilities()
p = copy.deepcopy(torch.load('data/StigmergicModelParameters.dat', weights_only=False))
p['ATPParameters'] = None; p['latticePeriodicBoundaryGJ'] = False
inst = model(p, 1)
boundaryIndices = np.array([int(i) for i in utils.computeDomeIndices(inst.electricNetwork, mode='tissue')])
numCells = 121

blocks = {
    'eye':   np.array([24, 25, 35, 36, 29, 30, 40, 41]),
    'nose':  np.array([49, 60, 71]),
    'mouth': np.array([92, 93, 94]),
}

conditions = [('learned', 'data/provenanceTimecourse_learned.npz')] + \
            [(f'random seed {s}', f'data/provenanceTimecourse_seed{s}_random.npz') for s in range(1, 5)]

results = {}
for label, path in conditions:
    d = np.load(path, allow_pickle=True)
    steps = sorted(int(k.split('step')[1]) for k in d.keys() if k.startswith('pVmem_step'))
    perBlock = {}
    for name, idxs in blocks.items():
        boundaryShare, velocity = [], []
        prevRow = None
        for s in steps:
            full = np.asarray(d[f'pVmem_step{s}'])
            row = full[idxs].mean(axis=0)
            others = np.setdiff1d(np.arange(numCells), idxs)
            boundaryShare.append(row[np.intersect1d(others, boundaryIndices)].sum())
            if prevRow is not None:
                velocity.append(np.abs(row - prevRow).sum())
            prevRow = row
        perBlock[name] = dict(steps=np.array(steps), boundaryShare=np.array(boundaryShare),
                              velocity=np.array(velocity))
    results[label] = perBlock

# --- Figure: boundary share and velocity, per block, all conditions overlaid -------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
colors = {'learned': '#C44E52'}
randomColors = plt.cm.Blues(np.linspace(0.4, 0.9, 4))
for i, s in enumerate(range(1, 5)):
    colors[f'random seed {s}'] = randomColors[i]

for col, name in enumerate(blocks.keys()):
    axShare, axVel = axes[0, col], axes[1, col]
    for label, _ in conditions:
        r = results[label][name]
        lw = 2.2 if label == 'learned' else 1.2
        alpha = 1.0 if label == 'learned' else 0.85
        axShare.plot(r['steps'], r['boundaryShare'], color=colors[label], linewidth=lw, alpha=alpha,
                    label=label)
        axVel.plot(r['steps'][1:], r['velocity'], color=colors[label], linewidth=lw, alpha=alpha)
    axShare.set_title(f'{name}: boundary share', fontsize=10)
    axShare.set_xlabel('step'); axShare.set_ylabel('boundary share')
    axVel.set_title(f'{name}: provenance velocity', fontsize=10)
    axVel.set_xlabel('step'); axVel.set_ylabel('|ΔP| per 10 steps')
    axVel.set_yscale('log')
    if col == 0:
        axShare.legend(fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig('figures/provenanceUnfolding.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceUnfolding.png")

# --- Quantitative smoothness: total variation of the velocity curve, and reversal count --------
print(f"\n{'block':<8}{'condition':<16}{'total variation':>18}{'# reversals':>14}{'mean velocity':>16}")
for name in blocks.keys():
    for label, _ in conditions:
        vel = results[label][name]['velocity']
        share = results[label][name]['boundaryShare']
        diffs = np.diff(share)
        reversals = int((np.diff(np.sign(diffs)) != 0).sum())
        totalVariation = np.abs(np.diff(vel)).sum()
        print(f"{name:<8}{label:<16}{totalVariation:>18.4f}{reversals:>14}{vel.mean():>16.4f}")
