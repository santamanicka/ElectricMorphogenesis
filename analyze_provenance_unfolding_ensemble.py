"""
How does the provenance map unfold over time -- gracefully (smooth, roughly monotonic) or
chaotically (bursty, reversing) -- for the learned clamp versus random clamps, using the canonical
"learned vs 100-seed ensemble average" comparison basis (see measure_provenance_ensemble_timecourse.py).

Supersedes analyze_provenance_unfolding.py's individual-seed comparison for the main-text result;
the individual-seed version is kept separately as it documents a real methodological point (see
Appendix B): ensemble averages and individual seeds can disagree about "is there a difference from
learned," and small samples of individual seeds are not a reliable stand-in for the ensemble.

Tracks, per facial-feature block, two things over the trajectory:
  1. boundary share -- an interpretable scalar, whether its climb is smooth/monotonic or wobbly
  2. "provenance velocity" -- the step-to-step L1 change in the full pooled provenance vector

  python analyze_provenance_unfolding_ensemble.py
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

ensembleData = np.load('data/provenanceTimecourse_randomEnsemble100.npz', allow_pickle=True)
numSeeds = int(ensembleData['numSeeds'])
ensembleSteps = sorted(int(k.split('step')[1]) for k in ensembleData.keys()
                       if k.startswith('eye_pVmem_step'))

conditions = [('learned', 'data/provenanceTimecourse_learned.npz', None),
             (f'random ({numSeeds}-seed ensemble)', None, ensembleData)]

results = {}
for label, path, preloaded in conditions:
    perBlock = {}
    for name, idxs in blocks.items():
        boundaryShare, velocity = [], []
        prevRow = None
        if preloaded is None:
            d = np.load(path, allow_pickle=True)
            steps = sorted(int(k.split('step')[1]) for k in d.keys() if k.startswith('pVmem_step'))
        else:
            d = preloaded
            steps = ensembleSteps
        for s in steps:
            row = (np.asarray(d[f'pVmem_step{s}'])[idxs].mean(axis=0) if preloaded is None
                  else np.asarray(d[f'{name}_pVmem_step{s}']))
            others = np.setdiff1d(np.arange(numCells), idxs)
            boundaryShare.append(row[np.intersect1d(others, boundaryIndices)].sum())
            if prevRow is not None:
                velocity.append(np.abs(row - prevRow).sum())
            prevRow = row
        perBlock[name] = dict(steps=np.array(steps), boundaryShare=np.array(boundaryShare),
                              velocity=np.array(velocity))
    results[label] = perBlock

# --- Figure: boundary share and velocity, per block, learned vs ensemble ------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
colors = {'learned': '#C44E52', f'random ({numSeeds}-seed ensemble)': '#4C72B0'}

for col, name in enumerate(blocks.keys()):
    axShare, axVel = axes[0, col], axes[1, col]
    for label, _, _ in conditions:
        r = results[label][name]
        lw = 2.2 if label == 'learned' else 1.8
        axShare.plot(r['steps'], r['boundaryShare'], color=colors[label], linewidth=lw, label=label)
        axVel.plot(r['steps'][1:], r['velocity'], color=colors[label], linewidth=lw)
    axShare.set_title(f'{name}: boundary share', fontsize=10)
    axShare.set_xlabel('step'); axShare.set_ylabel('boundary share')
    axVel.set_title(f'{name}: provenance velocity', fontsize=10)
    axVel.set_xlabel('step'); axVel.set_ylabel('|ΔP| per 10 steps')
    axVel.set_yscale('log')
    if col == 0:
        axShare.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('figures/provenanceUnfoldingEnsemble.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceUnfoldingEnsemble.png")

# --- Quantitative smoothness: total variation of the velocity curve, and reversal count --------
print(f"\n{'block':<8}{'condition':<28}{'total variation':>18}{'# reversals':>14}{'mean velocity':>16}")
for name in blocks.keys():
    for label, _, _ in conditions:
        vel = results[label][name]['velocity']
        share = results[label][name]['boundaryShare']
        diffs = np.diff(share)
        reversals = int((np.diff(np.sign(diffs)) != 0).sum())
        totalVariation = np.abs(np.diff(vel)).sum()
        print(f"{name:<8}{label:<28}{totalVariation:>18.4f}{reversals:>14}{vel.mean():>16.4f}")

print(f"\n{'block':<8}{'condition':<28}{'final boundary share':>22}")
for name in blocks.keys():
    for label, _, _ in conditions:
        finalShare = results[label][name]['boundaryShare'][-1]
        print(f"{name:<8}{label:<28}{finalShare*100:>21.2f}%")
