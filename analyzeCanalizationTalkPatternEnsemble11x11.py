"""How many different patterns do the codes of a sweep actually reach? EXPLORATORY: nothing here was predicted or registered.

The 400-code sweep of each target (relayLoopSweepNets...json: 160 or 280 space-filling codes, 120 near the trained code, and for the stripe 120 inside the window it forms in)
is read at the target's own readout (stripe 504, face 2173). The face sweep stores no tissue, so its codes are replayed here; the stripe sweep stores the tissue. For
each target it reports, over the interior's 81 cells as dark / light (below -34.6 mV):

  distinctSets        how many different dark sets the codes reach
  effectiveNumber     exp(entropy) of the dark-set frequencies: how many equally common patterns would give the same spread
  codesPerEffective   codes divided by that number: how many codes land on one pattern, on average (the many-to-one ratio)
  commonestShare      the share of codes that give the single commonest set
  participationRatio  (sum of eigenvalues)^2 / sum of squared eigenvalues of the patterns' covariance: how many cell directions the patterns spread over
  pcsFor90            how many principal components hold 90% of the variance

for all codes and for the space-filling ones alone (the kinds are sampled differently between the targets, so the space-filling subset is the like-for-like one).

    python3 analyzeCanalizationTalkPatternEnsemble11x11.py

Writes data/canalizationTalkPatternEnsemble1888Hold301.json and data/canalizationTalkSweepPatterns1888Hold301.npz (never overwriting).
"""
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary
from canalizationTalkCommon import *

jsonPath, npzPath = 'data/canalizationTalkPatternEnsemble1888Hold301.json', 'data/canalizationTalkSweepPatterns1888Hold301.npz'
for path in (jsonPath, npzPath):
    if os.path.exists(path):
        raise SystemExit(f'{path} exists; not overwriting')

replayer = Replayer()
patterns, kinds = {}, {}
for key, suffix in (('stripe', '1888Hold301StripesInteriorMinus60Minus5'), ('face', '1888Hold301FaceMinus60Minus5')):
    sweep = json.load(open(f'data/relayLoopSweepNets{suffix}.json'))
    target = TARGETS[key]
    frames, kindList = [], []
    for code in sweep['codes'].values():
        if key == 'stripe':
            frames.append(np.array(code['scoredVmem'], float))
        else:
            coefficients = np.array(code['multipliers']) * target['coefficients']
            frames.append(replayer.readout(ringValuesOf(coefficients, 2.0), target['readIteration']))
        kindList.append(code['kind'])
    patterns[key], kinds[key] = np.array(frames), np.array(kindList)
    print(key, patterns[key].shape, flush=True)
np.savez_compressed(npzPath, stripe=patterns['stripe'], face=patterns['face'], stripeKinds=kinds['stripe'], faceKinds=kinds['face'])


def describe(vmem):
    dark = (vmem[:, INTERIOR] < THRESHOLD)
    sets, counts = np.unique(dark, axis=0, return_counts=True)
    frequencies = counts / counts.sum()
    entropy = float(-(frequencies * np.log(frequencies)).sum())
    eigenvalues = np.clip(np.linalg.eigvalsh(np.cov(dark.astype(float).T)), 0, None)[::-1]
    return dict(codes=int(len(dark)), distinctSets=int(len(sets)), effectiveNumber=float(np.exp(entropy)), codesPerEffective=float(len(dark) / np.exp(entropy)),
                commonestShare=float(counts.max() / counts.sum()), participationRatio=float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()),
                pcsFor90=int(np.searchsorted(np.cumsum(eigenvalues) / eigenvalues.sum(), 0.90) + 1))


result = dict(note='EXPLORATORY; nothing registered. Dark = interior Vmem below -34.6 mV at the target\'s readout (stripe 504, face 2173).')
for key in ('stripe', 'face'):
    spaceFilling = np.array([kind.endswith('Global') or kind == 'sweepGlobal' for kind in kinds[key]])
    result[key] = dict(all=describe(patterns[key]), spaceFilling=describe(patterns[key][spaceFilling]))
    print(key, json.dumps(result[key], indent=1), flush=True)
json.dump(result, open(jsonPath, 'w'), indent=1)
