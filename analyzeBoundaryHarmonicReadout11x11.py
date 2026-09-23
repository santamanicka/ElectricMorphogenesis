"""Is the code's mark on the pattern gone, or only spread out? (PolyPatterning_Sim.md, Section 12).

A10 asks whether any single spatial mode belongs to a single order, and finds none once the face has formed. That test
is about localisation: it can only see influence that sits in one mode. This script asks the looser question. Across
the same ensemble, boundaryCodeUtilities.crossValidatedReadout finds the combination of modes most predictable from
the coefficients — the best linear readout of the pattern the code has, in any basis, not only the one chosen — and
scores it on codes held out of the fit, so a spread-out mark cannot hide from it and an overfitted one cannot pass.

A slice's grid ties its extent to its spacing, so a coarse slice covers a wider region as well as sampling it less
finely. Within one slice the two can be separated at a fixed number of codes: thin the grid to keep the region and
lose the fine spacing, or take a central block to keep the spacing and lose the region.

Reads the ownership runs that stored mode amplitudes and writes
data/boundaryHarmonicReadout<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import glob
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--ownershipGlob', type=str, default='data/boundaryHarmonicModeOwnership1888Hold301FaceMinus60Minus5*.json')
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicReadout1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--numFolds', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')

result = dict(runs={})
for path in sorted(glob.glob(args.ownershipGlob)):
    data = json.load(open(path))
    if not data.get('amplitudes'):
        continue
    name = os.path.basename(path).split('Minus5')[-1].replace('.json', '')
    codes = np.array(data['codes'])
    varying = codes.std(0) > 1e-9
    predictors = (codes[:, varying] - codes[:, varying].mean(0)) / codes[:, varying].std(0)
    entry = dict(condition=data.get('condition'), sampling=data.get('sampling', 'random'), numCodes=len(codes),
                 sliceHalfWidth=data.get('sliceHalfWidth'), numPredictors=int(varying.sum()), moments={})
    for moment, stored in sorted(data['amplitudes'].items(), key=lambda item: int(item[0])):
        amplitudes = np.array(stored)
        if amplitudes[:, amplitudes.std(0) > 1e-9].shape[1] == 0:
            # every code ended at the same pattern, so there is no readout to find
            entry['moments'][moment] = dict(heldOutCorrelation=None, perFold=None, degenerate=True)
            print(f'{name}, iteration {moment}: every code gives the same pattern, so there is nothing to read out', flush=True)
            continue
        score, perFold, _ = boundary.crossValidatedReadout(predictors, amplitudes, seed=args.seed, numFolds=args.numFolds)
        entry['moments'][moment] = dict(heldOutCorrelation=round(score, 4), perFold=perFold,
                                        heldOutVarianceExplained=round(score ** 2, 4), degenerate=False)
        print(f'{name}, iteration {moment}: held-out readout {score:.3f} '
              f'(it accounts for {score ** 2 * 100:.0f}% of its own variance)', flush=True)

    if entry['sampling'] == 'slice' and data.get('amplitudes'):
        moment = max(data['amplitudes'], key=int)
        amplitudes = np.array(data['amplitudes'][moment])
        firstAxis = np.array(sorted(set(np.round(codes[:, 1], 8))))
        secondAxis = np.array(sorted(set(np.round(codes[:, 2], 8))))
        column = np.searchsorted(firstAxis, np.round(codes[:, 1], 8))
        row = np.searchsorted(secondAxis, np.round(codes[:, 2], 8))
        middle = (len(firstAxis) - 1) / 2
        comparison = {}
        for label, mask in (('wide region, coarse spacing', (column % 2 == 0) & (row % 2 == 0)),
                            ('narrow region, fine spacing', (np.abs(column - middle) < len(firstAxis) / 4)
                                                            & (np.abs(row - middle) < len(secondAxis) / 4))):
            selected = amplitudes[mask]
            if mask.sum() < 100 or selected[:, selected.std(0) > 1e-9].shape[1] == 0:
                continue
            positions = codes[mask][:, varying]
            positions = (positions - positions.mean(0)) / positions.std(0)
            score, _, _ = boundary.crossValidatedReadout(positions, selected, seed=args.seed, numFolds=args.numFolds)
            spacing = float(np.diff(sorted(set(np.round(codes[mask][:, 1], 8))))[0])
            comparison[label] = dict(numCodes=int(mask.sum()), spacing=round(spacing, 6),
                                     halfExtent=round(float((codes[mask][:, 1].max() - codes[mask][:, 1].min()) / 2), 5),
                                     heldOutCorrelation=round(score, 4))
            print(f'  {name}, iteration {moment}, {label}: {int(mask.sum())} codes, spacing {spacing:.5f}, '
                  f'readout {score:.2f}', flush=True)
        entry['extentVersusSpacing'] = dict(moment=moment, comparison=comparison)
    result['runs'][name] = entry

json.dump(result, open(args.outputPath, 'w'))
print('wrote', args.outputPath)
