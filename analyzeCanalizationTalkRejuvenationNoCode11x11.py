"""The rejuvenation search's baseline of doing nothing (EXPLORATORY; not registered): each aged tissue state of the search is left to run free for 3,000 iterations (no ring code held, no extra
update: the release dynamics the original run was already following), and the overlap of its dark interior cells with the target is followed in time.

Two numbers per state, both over the search's own scoring window (iterations 301 to 2,999 after the start of the run): the highest overlap at any iteration, and the overlap at the iteration
of lowest balanced-RMS score (the definition the search's captions use). For the three best-overlap cases of the search (those of figures/canalizationTalkRejuvenationBest.png), the same two
numbers for the searched code, held and released from the same state, so that the two are compared on one definition. The window starts at 301 for both, as in the search, even where the held
code is on for longer than that.

    python3 analyzeCanalizationTalkRejuvenationNoCode11x11.py

Writes data/canalizationTalkRejuvenationNoCode1888Hold301.json (never overwriting).
"""
import json
import os

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from boundaryHarmonicStep import ringHoldBatchReplayFromState
from canalizationTalkCommon import RING_ANGLES, ringValuesOf

outputPath = 'data/canalizationTalkRejuvenationNoCode1888Hold301.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
ITERATIONS, WINDOW_START = 3000, 301
TARGETS = {'Stripe': boundary.centreStripeCellIndices, 'FaceOrder3': boundary.featureCellIndices, 'FaceOrder4': boundary.featureCellIndices}
BEST = {'Stripe': dict(offset=100, suffix='', order=2, restart=1, hold=301), 'FaceOrder3': dict(offset=300, suffix='LongHold1000', order=3, restart=11, hold=1000),
        'FaceOrder4': dict(offset=300, suffix='', order=4, restart=0, hold=301)}


def follow(label, state, ringValues, hold):
    """(highest overlap in the window, its iteration, overlap at the lowest-score iteration, that score, that iteration) for the code(s) in `ringValues`, held `hold` iterations from the state."""
    cells = np.asarray(TARGETS[label])
    target = np.full(boundary.numCells, -5.0)
    target[cells] = -60.0
    targetTensor = torch.tensor(target)
    featureMask = np.isin(np.arange(boundary.numCells), cells)
    overlaps, scores = [], []

    def onIteration(iteration, vmem):
        pattern = vmem[0]
        squared = (pattern - targetTensor) ** 2
        scores.append(float((squared[featureMask].mean().sqrt() + squared[~featureMask].mean().sqrt()) / 2))
        overlaps.append(boundary.structuralIntersectionOverUnion(pattern.numpy(), cells))
    ringHoldBatchReplayFromState(ringValues, hold, ITERATIONS, onIteration, state['vmemVolts'], state['gpol'])
    overlaps, scores = np.array(overlaps), np.array(scores)
    window = np.arange(WINDOW_START, ITERATIONS)
    bestTime = int(window[np.argmax(overlaps[window])])
    bestScoreTime = int(window[np.argmin(scores[window])])
    return dict(highestOverlap=float(overlaps[bestTime]), highestOverlapIteration=bestTime, overlapAtLowestScore=float(overlaps[bestScoreTime]), lowestScore=float(scores[bestScoreTime]),
                lowestScoreIteration=bestScoreTime, highestOverlapAnyTime=float(overlaps.max()))


results = dict(note='EXPLORATORY; free run of every aged state of the rejuvenation search (no ring code, no extra update), and the best-overlap searched codes on one definition', noCode={}, searched={})
for label in TARGETS:
    results['noCode'][label] = {}
    for offset in (100, 300, 600, 3000):
        state = dict(np.load(f'data/agedTissueState1888Hold301{label}After{offset}.npz', allow_pickle=True))
        results['noCode'][label][str(offset)] = follow(label, state, np.zeros((1, len(RING_ANGLES))), 0)
    spec = BEST[label]
    state = dict(np.load(f'data/agedTissueState1888Hold301{label}After{spec["offset"]}.npz', allow_pickle=True))
    run = np.load(f'data/boundaryHarmonicRejuvenationTraining1888Hold301{label}After{spec["offset"]}{spec["suffix"]}/order{spec["order"]}_restart{spec["restart"]:02d}.npz', allow_pickle=True)
    ring = ringValuesOf(run['bestCoefficients'], float(state['ceiling']))[None]
    results['searched'][label] = dict(case=spec, **follow(label, state, ring, spec['hold']), storedBestOverlap=float(boundary.structuralIntersectionOverUnion(np.asarray(run['bestVmem'], float), TARGETS[label])))
json.dump(results, open(outputPath, 'w'), indent=1)

print(f'{"":11}{"offset":>7}{"no code: highest":>18}{"at iter":>9}{"at lowest score":>17}{"(mV)":>7}')
for label, byOffset in results['noCode'].items():
    for offset, r in byOffset.items():
        print(f'{label:11}{offset:>7}{r["highestOverlap"]:18.2f}{r["highestOverlapIteration"]:9d}{r["overlapAtLowestScore"]:17.2f}{r["lowestScore"]:7.1f}')
print('\nsearched best-overlap cases:')
for label, r in results['searched'].items():
    print(f'{label:11}+{r["case"]["offset"]:<5}highest {r["highestOverlap"]:.2f} (iteration {r["highestOverlapIteration"]}), at lowest score {r["overlapAtLowestScore"]:.2f} ({r["lowestScore"]:.1f} mV, iteration {r["lowestScoreIteration"]}), stored {r["storedBestOverlap"]:.2f}')
print('wrote', outputPath)
