"""Is the code's mark on the pattern gone, or only spread out? (PolyPatterning_Sim.md, Section 12).

A10 asks whether any single spatial mode belongs to a single order, and finds none once the face has formed. That test
is about localisation: it can only see influence that sits in one mode. This script asks the looser question. Across
the same ensemble, canonical correlation finds the combination of modes most predictable from a combination of
coefficients — the best linear readout of the pattern the code has, in any basis — and scores it on held-out codes, so
a spread-out mark cannot hide from it and an overfitted one cannot pass.

Reads the ownership runs that stored mode amplitudes and writes
data/boundaryHarmonicReadout<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.cross_decomposition import CCA

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
    components = min(4, predictors.shape[1])
    generator = np.random.default_rng(args.seed)
    folds = generator.permutation(len(codes)) % args.numFolds
    entry = dict(condition=data.get('condition'), sampling=data.get('sampling', 'random'), numCodes=len(codes),
                 sliceHalfWidth=data.get('sliceHalfWidth'), numPredictors=int(varying.sum()), moments={})
    for moment, stored in sorted(data['amplitudes'].items(), key=lambda item: int(item[0])):
        amplitudes = np.array(stored)
        amplitudes = amplitudes[:, amplitudes.std(0) > 1e-9]
        amplitudes = (amplitudes - amplitudes.mean(0)) / amplitudes.std(0)
        trained, heldOut = [], []
        for fold in range(args.numFolds):
            train, test = folds != fold, folds == fold
            model = CCA(n_components=components, max_iter=2000).fit(predictors[train], amplitudes[train])
            for selection, store in ((train, trained), (test, heldOut)):
                first, second = model.transform(predictors[selection], amplitudes[selection])
                store.append([abs(float(np.corrcoef(first[:, k], second[:, k])[0, 1])) for k in range(components)])
        entry['moments'][moment] = dict(trainCorrelations=np.round(np.mean(trained, 0), 4).tolist(),
                                        heldOutCorrelations=np.round(np.mean(heldOut, 0), 4).tolist(),
                                        heldOutVarianceExplained=round(float(np.mean(heldOut, 0)[0] ** 2), 4))
        print(f"{name}, iteration {moment}: held-out canonical correlations "
              f"{np.round(np.mean(heldOut, 0), 3).tolist()} (best readout explains "
              f"{entry['moments'][moment]['heldOutVarianceExplained'] * 100:.0f}% of its own variance)", flush=True)
    result['runs'][name] = entry

json.dump(result, open(args.outputPath, 'w'))
print('wrote', args.outputPath)
