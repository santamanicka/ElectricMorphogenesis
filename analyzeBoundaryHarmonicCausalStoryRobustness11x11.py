"""Is the face's skeleton robust and its completion fragile? Code jitter, every half-ring arc, random subsets, the release time.

EXPLORATORY. Criteria: data/boundaryHarmonicCausalStoryRobustnessPredictions1888Hold301FaceMinus60Minus5.json, committed
before this script existed. The same machinery as analyzeBoundaryHarmonicCausalStory11x11.py (the pure step with only part
of the ring held, or the whole ring with a jittered code, or the whole ring released early; the clamp's tissue-wide extra
update on every hold iteration in every run), taken to state 2174, the scored moment.

    python3 analyzeBoundaryHarmonicCausalStoryRobustness11x11.py
"""
import argparse
import json
import time

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from boundaryHarmonicStep import Step

parser = argparse.ArgumentParser()
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--predictionsPath', type=str, default='data/boundaryHarmonicCausalStoryRobustnessPredictions1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicCausalStoryRobustness1888Hold301FaceMinus60Minus5.json')
args = parser.parse_args()
torch.set_grad_enabled(False)
started = time.time()

summary = json.load(open(args.summaryPath))
hold = int(summary['hold'])
winner = summary['orders']['3']['best']
coefficients = np.asarray(np.load(
    f"{summary['trainingDirs'][winner['round']]}/order3_restart{winner['restart']:02d}.npz")['bestCoefficients'], float)
ringValues = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(len(coefficients)))) @ coefficients
step = Step(ringCode=ringValues)
n, Gref = step.numCells, step.Gref
LAST, PEAK = 2174, 1766
ring = [int(c) for c in boundary.boundaryRingCells]
fullMask = step.ringMask.clone()
fullCode = step.ringCode.clone()

featureCells = np.array(sorted(set(boundary.featureCellIndices.tolist())))
interior = np.array(boundary.interiorCellIndices)
backgroundCells = np.array([c for c in interior if c not in set(featureCells.tolist())])
leftEye, rightEye, nose, mouth = (list(part) for part in boundary.featureParts)
groups = {'eyes': np.array(leftEye + rightEye), 'nose': np.array(nose), 'mouth': np.array(mouth), 'background': backgroundCells}


def run(held, releaseAt=hold, code=None):
    """Absolute selectivity at state 1766, the interior-mean trough (recorded iteration), and the outcome at the scored moment."""
    mask = torch.zeros(n, dtype=torch.double)
    mask[list(held)] = 1.0
    step.ringMask = mask
    step.ringCode = fullCode if code is None else code
    V, G = step.initialVmem.clone(), step.initialGpol.clone()
    interiorMean, atPeak = [], None
    for m in range(LAST):
        V, G = step(V, G, m < releaseAt, m < hold)
        interiorMean.append(float(G[interior].mean()))
        if m + 1 == PEAK:
            atPeak = G.numpy().copy()
    step.ringMask, step.ringCode = fullMask, fullCode
    interiorMean = np.array([float(step.initialGpol[interior].mean())] + interiorMean)
    window = interiorMean[302:1301]
    vMilli = V.numpy() * 1000.0
    dark = vMilli < boundary.hyperpolarizedThresholdMilliVolts
    return dict(selectivity=float((atPeak[featureCells].mean() - atPeak[backgroundCells].mean()) / Gref),
                trough=302 + int(np.argmin(window)) - 1,
                dark={name: int(dark[cells].sum()) for name, cells in groups.items()},
                overlap=boundary.structuralIntersectionOverUnion(vMilli))


def say(*parts):
    print(f'[{time.time() - started:5.0f}s]', *parts, flush=True)


baseline = run([], releaseAt=0)
full = run(ring)
say(f"baseline selectivity {baseline['selectivity']:.5f}; full ring {full['selectivity']:.5f}, overlap {full['overlap']:.3f}")
gapFull = full['selectivity'] - baseline['selectivity']

# ------------------------------------------------------------------ release scan
releaseScan = {}
for releaseAt in list(range(5, 301, 5)) + [301]:
    releaseScan[releaseAt] = run(ring, releaseAt=releaseAt)
say('release scan done')

# ------------------------------------------------------------------ jitter
generator = np.random.default_rng(20250926)
jitter = {}
for sigma in (0.0, 0.01, 0.03, 0.10):
    draws = []
    for _ in range(1 if sigma == 0.0 else 100):
        factors = 1.0 + sigma * generator.standard_normal(len(ring)) if sigma else np.ones(len(ring))
        code = torch.zeros(n, dtype=torch.double)
        code[step.ring] = torch.as_tensor(np.asarray(ringValues) * factors, dtype=torch.double) * Gref
        draws.append(run(ring, code=code))
    jitter[str(sigma)] = draws
    say(f'jitter sigma {sigma}: median gap {np.median([d["selectivity"] - baseline["selectivity"] for d in draws]):+.4f}, '
        f'overlap >= 0.5 in {sum(d["overlap"] >= 0.5 for d in draws)} of {len(draws)}')

# ------------------------------------------------------------------ random subsets and every arc
subsets = []
for _ in range(200):
    held = [c for c in ring if generator.random() < 0.5]
    if held:
        subsets.append(dict(held=held, **run(held)))
say('random subsets done')
arcs = []
for offset in range(len(ring)):
    held = [ring[(offset + k) % len(ring)] for k in range(20)]
    arcs.append(dict(offset=offset, held=held, **run(held)))
say('arcs done')

# ------------------------------------------------------------------ verdicts
V1 = dict(release301=releaseScan[301]['selectivity'], zeroJitter=jitter['0.0'][0]['selectivity'], overlaps=[releaseScan[301]['overlap'], jitter['0.0'][0]['overlap']])
V1['holds'] = bool(abs(V1['release301'] - 0.40124) < 1e-4 and abs(V1['zeroJitter'] - 0.40124) < 1e-4 and all(abs(o - 0.9333) < 0.001 for o in V1['overlaps']))
relative = [abs(d['selectivity'] - full['selectivity']) / gapFull for d in jitter['0.03']]
R1 = dict(medianRelativeChange=float(np.median(relative)), holds=bool(np.median(relative) < 0.25))
kept = sum(d['overlap'] >= 0.5 for d in jitter['0.03'])
R2 = dict(keptOverlap=kept, of=100, holds=bool(kept < 50))
R3 = dict(largestArcOverlap=max(a['overlap'] for a in arcs), holds=bool(max(a['overlap'] for a in arcs) < 0.5))
pool = subsets + arcs
below = [r for r in pool if r['selectivity'] < 0.30]
R4 = dict(runsBelow=len(below), of=len(pool), largestOverlapBelow=max((r['overlap'] for r in below), default=None),
          holds=bool(all(r['overlap'] < 0.5 for r in below)))
ratios = {h: (r['selectivity'] - baseline['selectivity']) / gapFull for h, r in releaseScan.items() if h >= 100}
R5 = dict(minRatio=min(ratios.values()), maxRatio=max(ratios.values()), outside=[h for h, v in ratios.items() if not 0.9 <= v <= 1.1],
          holds=bool(all(0.9 <= v <= 1.1 for v in ratios.values())))
verdicts = dict(V1=V1, R1=R1, R2=R2, R3=R3, R4=R4, R5=R5)
for key, verdict in verdicts.items():
    say(f"{key}: {'holds' if verdict['holds'] else 'FAILS'}", {k: v for k, v in verdict.items() if k != 'holds'})

json.dump(dict(note='EXPLORATORY; criteria in the predictions file, registered before this script existed.', predictions=json.load(open(args.predictionsPath)),
               baseline=baseline, fullRing=full, releaseScan={str(k): v for k, v in releaseScan.items()}, jitter=jitter,
               randomSubsets=subsets, arcs=arcs, verdicts=verdicts), open(args.outputPath, 'w'))
say('wrote', args.outputPath)
