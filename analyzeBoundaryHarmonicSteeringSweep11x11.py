"""Turn one order at a time and read the delta in the developmental program (PolyPatterning_Sim.md, Section 12).

The ensemble analysis asks which orders predict which parts of the program across many codes. This asks the
sharper question: starting from the trained code, is any single order a knob — does moving it alone move a part
of the program monotonically? A knob would show a Spearman correlation near +-1 across the sweep. Anything else
means the regression's answer comes from averaging over a neighbourhood, not from a control the code affords.

Writes data/boundaryHarmonicSteeringSweep<rest of the summary's name> (never overwriting).
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--recordPath', type=str, required=True, help="npz from --mode steering")
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
args = parser.parse_args()

outputPath = args.summaryPath.replace('boundaryHarmonicTrainingSummary', 'boundaryHarmonicSteeringSweep')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

data = np.load(args.recordPath)
packed, labels = data['packed'], data['steering']
interiorMean, featureMean, backgroundMean = data['interiorMean'], data['featureMean'], data['backgroundMean']
frames, stride = data['gpolFrames'], int(data['gpolStride'])
interior = np.array(boundary.interiorCellIndices)
featureCells = sorted(set(boundary.featureCellIndices.tolist()))
isFeature = np.isin(interior, featureCells)


def programOf(index):
    mean = interiorMean[index].astype(float)
    trough = 302 + int(mean[302:1300].argmin())
    peak = trough + int(mean[trough:].argmax())
    frame = frames[index, min(peak // stride, frames.shape[1] - 1)][interior]
    bits = np.unpackbits(packed[index, 301::5], axis=1)[:, :boundary.numCells].astype(bool)[:, interior]
    intersection = (bits & isFeature).sum(1)
    union = (bits | isFeature).sum(1)
    return dict(holdPeak=float(mean[:302].max()), troughTime=int(trough), troughValue=float(mean[trough]),
                peakTime=int(peak), peakValue=float(mean[peak]),
                selectivity=float(featureMean[index, peak] - backgroundMean[index, peak]),
                targeting=float(isFeature[np.argsort(frame)[::-1][:len(featureCells)]].sum()),
                quality=float(np.where(union > 0, intersection / np.maximum(union, 1), 0).max()))


variables = ['holdPeak', 'troughValue', 'peakValue', 'selectivity', 'targeting', 'quality']
baseline = programOf(0)
numOrders = int(labels[:, 0].max()) + 1
sweeps, monotonicity = {}, {}
for order in range(numOrders):
    rows = sorted([i for i in range(len(labels)) if int(labels[i, 0]) == order], key=lambda i: labels[i, 1])
    steps = [float(labels[i, 1]) for i in rows]
    values = {key: [programOf(i)[key] for i in rows] for key in variables}
    sweeps[str(order)] = dict(steps=steps, values={k: [round(v, 4) for v in values[k]] for k in variables})
    monotonicity[str(order)] = {}
    for key in variables:
        if len(rows) >= 4:
            rho, p = spearmanr(steps, values[key])
            monotonicity[str(order)][key] = dict(rho=round(float(rho), 3), p=round(float(p), 4),
                                                 monotone=bool(abs(rho) > 0.8 and p < 0.05))
        else:
            monotonicity[str(order)][key] = dict(rho=None, p=None, monotone=False)

knobs = [(order, key) for order in monotonicity for key in variables if monotonicity[order][key]['monotone']]
result = dict(baseline={k: round(v, 4) for k, v in baseline.items()}, variables=variables,
              sweeps=sweeps, monotonicity=monotonicity,
              knobs=[dict(order=int(o), variable=k) for o, k in knobs])
json.dump(result, open(outputPath, 'w'))

print('baseline (the trained code):', {k: round(v, 3) for k, v in baseline.items()}, flush=True)
print('\nmonotone single-order controls (|rho| > 0.8, p < 0.05):', flush=True)
for order, key in knobs:
    m = monotonicity[order][key]
    print(f"  a{order} -> {key}:  rho {m['rho']:+.2f}, p {m['p']:.3f}", flush=True)
print('\nphase-3 variables (selectivity, targeting, quality):', flush=True)
for key in ['selectivity', 'targeting', 'quality']:
    worst = max(monotonicity[o][key]['p'] for o in monotonicity)
    best = min(monotonicity[o][key]['p'] for o in monotonicity)
    print(f'  {key}: best p across the four orders {best:.3f}', flush=True)
print('\nwrote', outputPath, flush=True)
