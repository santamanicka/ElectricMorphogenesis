"""The stripe at the face's ceiling: replay the codes of the even-only {0, 2, ..., 20} arm that form it, and read their conductance. EXPLORATORY.

Registered in data/boundaryHarmonicStripeLateRoutePredictions1888Hold301StripesInteriorMinus60Minus5.json (L1-L5), committed before this was run.
The training of the exploratory even-only arms at ceiling 1.3 (Amendment 6) found ten restarts of the arm of orders 0, 2, ..., 20 that end at overlap 1.0
at a best moment around 1740, with ring values no higher than 1.3. This replays them (every restart of the arm at overlap >= --overlapThreshold)
and, for comparison, the ceiling-2 order-2 stripe code, on model 1888 for --numIterations iterations in one batch (hold, then released), recording the interior mean
G_pol / G_ref and the structural overlap at every iteration.

Per code: the landmarks of the interior mean conductance (first peak inside the hold, trough = lowest between iterations 302 and 1300, second peak = highest after
the trough, as in analyzeBoundaryHarmonicStripeProgram11x11.py), the stripe cells dark at iteration 300, the best moment as in training (lowest balanced RMS from the
release to iteration 2999), the highest overlap, its longest unbroken run at 0.85 or more and the number of such runs, the ring's range, and the root-mean-square residual
of the ring values against their least-squares fit by orders {0, 2, 4, 6}. Predictions L1-L5 are evaluated as registered and printed.

    python3 analyzeBoundaryHarmonicStripeLateRoute11x11.py

Writes data/boundaryHarmonicStripeLateRoute1888Hold301StripesInteriorMinus60Minus5Ceiling1p3.json (never overwriting).
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301StripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--trainingDir', type=str, default=f'data/boundaryHarmonicTraining{SUFFIX}EvenOrdersPopulation64')
parser.add_argument('--arm', type=str, default='orders0-2-4-6-8-10-12-14-16-18-20')
parser.add_argument('--comparisonRunPath', type=str, default=f'data/boundaryHarmonicTraining{SUFFIX}Ceiling2/order2_restart06.npz')
parser.add_argument('--overlapThreshold', type=float, default=0.9)
parser.add_argument('--numIterations', type=int, default=20000)
parser.add_argument('--traceStride', type=int, default=10)
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicStripeLateRoute{SUFFIX}Ceiling1p3.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
startTime = time.time()
HOLD = 301
stripe = np.array(sorted(boundary.centreStripeCellIndices.tolist()))
interior = np.array(boundary.interiorCellIndices)
target = np.full(boundary.numCells, -5.0)
target[stripe] = -60.0

# ------------------------------------------------------------------------------------------------ the codes: every restart of the arm that forms the stripe, then the comparison
codes = []
for path in sorted(glob.glob(f'{args.trainingDir}/{args.arm}_restart*.npz')):
    run = np.load(path)
    overlap = float(boundary.structuralIntersectionOverUnion(run['bestVmem'].astype(float), boundary.centreStripeCellIndices))
    if overlap >= args.overlapThreshold:
        codes.append(dict(label=f"{args.arm} restart {int(run['restart'])}", kind='lateRoute', restart=int(run['restart']), startType=str(run['startType']),
                          trainedScore=float(run['bestScore']), trainedBestMoment=int(run['bestIteration']), orders=[int(o) for o in run['orders']],
                          coefficients=[float(c) for c in run['bestCoefficients']], ringValues=run['bestRingValues'].astype(float)))
comparison = np.load(args.comparisonRunPath)
codes.append(dict(label='order-2 stripe code (ceiling 2.0)', kind='comparison', restart=int(comparison['restart']), startType=str(comparison['startType']),
                  trainedScore=float(comparison['bestScore']), trainedBestMoment=int(comparison['bestIteration']), orders=[int(o) for o in comparison['orders']] if 'orders' in comparison.files else [0, 1, 2],
                  coefficients=[float(c) for c in comparison['bestCoefficients']], ringValues=comparison['bestRingValues'].astype(float)))
numLate = sum(code['kind'] == 'lateRoute' for code in codes)
print(f'{numLate} late-route codes and the comparison, {args.numIterations} iterations', flush=True)
ringValues = np.array([code['ringValues'] for code in codes])

# ------------------------------------------------------------------------------------------------ the replay
numCodes = len(codes)
darkAt300 = np.zeros(numCodes, dtype=int)
interiorMean = np.zeros((numCodes, args.numIterations))
overlapTrace = np.zeros((numCodes, args.numIterations))
score = np.full((numCodes, args.numIterations), np.inf)
featureMask = torch.zeros(boundary.numCells, dtype=torch.bool)
featureMask[torch.as_tensor(stripe)] = True
interiorMask = torch.zeros(boundary.numCells, dtype=torch.bool)
interiorMask[torch.as_tensor(interior)] = True
torchTarget = torch.tensor(target)


def onIteration(iteration, vmem, conductance):
    dark = (vmem < boundary.hyperpolarizedThresholdMilliVolts) & interiorMask[None]
    featureDark, stray = (dark & featureMask[None]).sum(1), (dark & ~featureMask[None]).sum(1)
    overlapTrace[:, iteration] = (featureDark.double() / (len(stripe) + stray).double()).numpy()
    interiorMean[:, iteration] = conductance[:, torch.as_tensor(interior)].mean(1).numpy()
    if iteration == 300:
        darkAt300[:] = featureDark.numpy()
    if HOLD <= iteration < 3000:
        squared = (vmem - torchTarget) ** 2
        score[:, iteration] = (squared[:, featureMask].mean(1).sqrt() * 0.5 + squared[:, ~featureMask].mean(1).sqrt() * 0.5).numpy()
    if iteration % 2000 == 0:
        print(f'[{time.time() - startTime:5.0f}s] iteration {iteration}', flush=True)


boundary.ringHoldBatchReplay(boundary.loadCheckpoint(1888), ringValues, HOLD, args.numIterations, onIteration, passConductance=True)

# ------------------------------------------------------------------------------------------------ the readings
angles = boundary.ringAngles(boundary.boundaryRingCells)
lowOrderBasis = np.cos(np.outer(angles, [0, 2, 4, 6]))
results = []
for k, code in enumerate(codes):
    mean, overlap = interiorMean[k], overlapTrace[k]
    firstPeak = int(mean[:HOLD].argmax())
    trough = HOLD + 1 + int(mean[HOLD + 1:1300].argmin())
    secondPeak = trough + int(mean[trough:].argmax())
    bestMoment = int(score[k].argmin())
    above = overlap >= 0.85
    runs, run, longest = 0, 0, 0
    for flag in above:
        run = run + 1 if flag else 0
        if flag and run == 1:
            runs += 1
        longest = max(longest, run)
    fit = lowOrderBasis @ np.linalg.lstsq(lowOrderBasis, code['ringValues'], rcond=None)[0]
    results.append(dict(label=code['label'], kind=code['kind'], restart=code['restart'], startType=code['startType'], orders=code['orders'], coefficients=code['coefficients'],
                        trainedScore=code['trainedScore'], trainedBestMoment=code['trainedBestMoment'], ringMin=float(code['ringValues'].min()), ringMax=float(code['ringValues'].max()),
                        firstPeak=firstPeak, trough=trough, secondPeak=secondPeak, bestMoment=bestMoment, bestScore=float(score[k].min()),
                        overlapAtBestMoment=float(overlap[bestMoment]), highestOverlap=float(overlap.max()), highestOverlapIteration=int(overlap.argmax()),
                        stripeCellsDarkAt300=int(darkAt300[k]), longestRunAtOrAbove0p85=int(longest), numberOfRuns=runs, iterationsAtOrAbove0p85=int(above.sum()),
                        lowOrderFitResidual=float(np.sqrt(((code['ringValues'] - fit) ** 2).mean())),
                        interiorMeanTrace=[round(float(v), 4) for v in mean[::args.traceStride]], overlapTrace=[round(float(v), 3) for v in overlap[::args.traceStride]]))

late = [r for r in results if r['kind'] == 'lateRoute']
count = lambda test: int(sum(bool(test(r)) for r in late))
verdicts = {
    'L1-readAfterTheTrough': dict(count=count(lambda r: r['bestMoment'] > r['trough']), of=len(late), holds=bool(count(lambda r: r['bestMoment'] > r['trough']) >= 8),
                                  bestMoments=[r['bestMoment'] for r in late], troughs=[r['trough'] for r in late]),
    'L2-notWrittenInTheHold': dict(count=count(lambda r: r['stripeCellsDarkAt300'] <= 13), of=len(late), holds=bool(count(lambda r: r['stripeCellsDarkAt300'] <= 13) >= 8),
                                   darkAt300=[r['stripeCellsDarkAt300'] for r in late]),
    'L3-onTheSecondRise': dict(count=count(lambda r: abs(r['bestMoment'] - r['secondPeak']) <= 300), of=len(late),
                               holds=bool(count(lambda r: abs(r['bestMoment'] - r['secondPeak']) <= 300) >= 8), secondPeaks=[r['secondPeak'] for r in late]),
    'L4-brief': dict(count=count(lambda r: r['longestRunAtOrAbove0p85'] < 200), of=len(late), holds=bool(count(lambda r: r['longestRunAtOrAbove0p85'] < 200) >= 8),
                     longestRuns=[r['longestRunAtOrAbove0p85'] for r in late]),
    'L5-notALowOrderCode': dict(count=count(lambda r: r['lowOrderFitResidual'] >= 0.05), of=len(late), holds=bool(count(lambda r: r['lowOrderFitResidual'] >= 0.05) >= 8),
                                residuals=[round(r['lowOrderFitResidual'], 4) for r in late]),
}
for key, verdict in verdicts.items():
    print(f"{key}: {'holds' if verdict['holds'] else 'FAILS'}", {k: v for k, v in verdict.items() if k != 'holds'}, flush=True)
comparisonResult = results[-1]
print('comparison:', {k: v for k, v in comparisonResult.items() if k not in ('interiorMeanTrace', 'overlapTrace', 'coefficients')}, flush=True)
json.dump(dict(note='EXPLORATORY: the ten ceiling-1.3 codes of the even-only {0, 2, ..., 20} arm that form the stripe, replayed with the order-2 ceiling-2 code for comparison; criteria registered in '
                    f'boundaryHarmonicStripeLateRoutePredictions{SUFFIX}.json before this was run.', numIterations=args.numIterations, traceStride=args.traceStride, hold=HOLD,
               trainingDir=args.trainingDir, arm=args.arm, codes=results, verdicts=verdicts), open(args.outputPath, 'w'), separators=(',', ':'))
print('wrote', args.outputPath, f'({time.time() - startTime:.0f}s)', flush=True)
