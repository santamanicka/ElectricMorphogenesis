"""Replay every 11x11 band-hold checkpoint (files 1600-1983) and score its face four ways, plus store its
late-window pattern (PolyPatterning_Sim.md, Section 12).

For each checkpoint, every iteration is scored and the single best moment per metric is kept:
  featureRMS_mV       voltage distance to target over the 14 eyes/nose/mouth cells (lower is better)
  balancedRMS_mV      0.5 * feature RMS + 0.5 * RMS over the other 107 cells (Sim.md 12.11; lower is better)
  structuralIoU       IoU of hyperpolarised interior cells with the feature cells (higher is better)
  partScore           coverage + separation - 2 * spurious (higher is better; see boundaryCodeUtilities)
The pattern file stores, per checkpoint, Vmem at the final iteration and the per-cell mean and standard
deviation over the last 1000 iterations.

--scanFrom start scans from iteration 0, as the published results were computed. That lets best moments
fall inside the hold, while the boundary is still clamped (51 of 384 for balancedRMS). --scanFrom release
scans only the iterations after the clamp is released.
"""
import argparse
import csv
import time

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--scanFrom', choices=['start', 'release'], default='start')
parser.add_argument('--fileNumbers', type=str, default='1600-1983', help='inclusive range, e.g. 1600-1983')
parser.add_argument('--metricsPath', type=str, default='data/bandHoldFaceMetrics11x11_all384.csv')
parser.add_argument('--patternsPath', type=str, default='data/bandHoldPatterns11x11_all384.npz')
parser.add_argument('--windowIterations', type=int, default=1000)
args = parser.parse_args()

firstFile, lastFile = (int(part) for part in args.fileNumbers.split('-'))
fileNumbers = list(range(firstFile, lastFile + 1))


def scoreCheckpoint(fileNumber):
    checkpoint = boundary.loadCheckpoint(fileNumber)
    metadata = boundary.checkpointMetadata(fileNumber, checkpoint)
    target = boundary.targetVmemMilliVolts(checkpoint)
    numIterations = checkpoint['simParameters']['numSimIters']
    firstScoredIteration = 0 if args.scanFrom == 'start' else metadata['holdIterations']
    best = dict(feature=(np.inf, -1), balanced=(np.inf, -1), structural=(-1.0, -1, None), part=(-np.inf, -1, None))
    window = []

    def onIteration(iteration, vmem):
        if iteration >= numIterations - args.windowIterations:
            window.append(vmem.copy())
        if iteration < firstScoredIteration:
            return
        feature = boundary.featureRootMeanSquareError(vmem, target)
        balanced = boundary.balancedRootMeanSquareError(vmem, target)
        structural = boundary.structuralIntersectionOverUnion(vmem)
        part = boundary.partSeparationScore(vmem)
        if feature < best['feature'][0]:
            best['feature'] = (feature, iteration)
        if balanced < best['balanced'][0]:
            best['balanced'] = (balanced, iteration)
        if structural > best['structural'][0]:
            best['structural'] = (structural, iteration, vmem.copy())
        if part[0] > best['part'][0]:
            best['part'] = (part[0], iteration, part)

    boundary.replay(checkpoint, checkpoint['clampParameters'], onIteration)
    window = np.stack(window)
    _, structuralComponents = boundary.interiorDarkComponents(best['structural'][2])
    _, coverage, separation, spurious = best['part'][2]
    metadata.update(trainedBestLoss=float(checkpoint['trainParameters']['bestLoss']),
                    featureRMS_mV=round(best['feature'][0], 4), featureBestIteration=best['feature'][1],
                    balancedRMS_mV=round(best['balanced'][0], 4), balancedBestIteration=best['balanced'][1],
                    structuralIoU=round(best['structural'][0], 4), structuralBestIteration=best['structural'][1],
                    structuralComponents=structuralComponents,
                    partScore=round(best['part'][0], 4), partBestIteration=best['part'][1],
                    partCoverage=round(coverage, 4), partSeparation=separation, partSpurious=round(spurious, 4))
    patterns = dict(final=window[-1].astype(np.float32), windowMean=window.mean(0).astype(np.float32),
                    windowStd=window.std(0).astype(np.float32))
    return metadata, patterns


rows, patternsByFile = [], {}
startTime = time.time()
for count, fileNumber in enumerate(fileNumbers, start=1):
    metadata, patternsByFile[fileNumber] = scoreCheckpoint(fileNumber)
    rows.append(metadata)
    if count % 48 == 0 or count == len(fileNumbers):
        print(f"  {count}/{len(fileNumbers)} checkpoints  {time.time() - startTime:.0f}s", flush=True)

with open(args.metricsPath, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
np.savez_compressed(args.patternsPath, fileNumbers=np.array(fileNumbers),
                    final=np.stack([patternsByFile[number]['final'] for number in fileNumbers]),
                    windowMean=np.stack([patternsByFile[number]['windowMean'] for number in fileNumbers]),
                    windowStd=np.stack([patternsByFile[number]['windowStd'] for number in fileNumbers]),
                    windowIterations=args.windowIterations, scanFrom=args.scanFrom)
print(f"wrote {args.metricsPath} and {args.patternsPath} (scanFrom={args.scanFrom}) in {time.time() - startTime:.0f}s")
