"""Interim full-tissue and facial-feature-only readout for the still-running tissue-band
depth-restricted static-hold experiment (tissueBandGpolTwoFoldSymmetry, screen4/weight1000, depths 3
and 7, 100- and 200-iteration holds, both loss methods). These checkpoints are each group's current
"best so far" while training continues -- not the final result -- so treat the numbers as provisional.

Full-tissue loss is read straight from each checkpoint's stored bestLoss (no replay needed). Facial-
feature-only RMS distance (eyes+nose+mouth, target-binary so correlation is undefined there -- same
reasoning as compareFacialFeatureScoreByStrategy.py) requires a fresh replay since no trajectory is
stored at this resolution; the replay also reproduces the checkpoint's own full-tissue correlation loss
over its readout window as a trust check against the stored bestLoss, same method as before.
"""
import argparse

import numpy as np
import torch

from embryo import model


def rowColumnBlock(circuitRows, circuitCols, rowFractions, columnFractions):
    firstRow, lastRow = (round(f * circuitRows) for f in rowFractions)
    firstCol, lastCol = (round(f * circuitCols) for f in columnFractions)
    return [r * circuitCols + c for r in range(firstRow, lastRow)
            for c in range(firstCol, lastCol)]


def faceFeatureIndices(circuitRows, circuitCols):
    leftEye = rowColumnBlock(circuitRows, circuitCols, (2/11, 4/11), (2/11, 4/11))
    rightEye = rowColumnBlock(circuitRows, circuitCols, (2/11, 4/11), (7/11, 9/11))
    nose = rowColumnBlock(circuitRows, circuitCols, (4/11, 7/11), (5/11, 6/11))
    mouth = rowColumnBlock(circuitRows, circuitCols, (8/11, 9/11), (4/11, 7/11))
    return leftEye + rightEye, nose, mouth


parser = argparse.ArgumentParser()
parser.add_argument('--minIter', type=int, default=500)
parser.add_argument('--stride', type=int, default=10)
args = parser.parse_args()

torch.set_grad_enabled(False)
rows, cols = 30, 30

referenceCheckpoint = torch.load('data/bestModelParameters_fieldVector_30x30_1804.dat', map_location='cpu', weights_only=False)
target = referenceCheckpoint['trainParameters']['targetVmem'].reshape(rows, cols).numpy() * 1000
eyeIdx, noseIdx, mouthIdx = faceFeatureIndices(rows, cols)
featureIdx = np.array(sorted(eyeIdx + noseIdx + mouthIdx))
targetFeature = target.reshape(-1)[featureIdx]


def featureScore(vmemFlatMV):
    v = vmemFlatMV[featureIdx]
    return float(np.sqrt(np.mean((v - targetFeature) ** 2)))


def fullTissueCorrelationLoss(tailVmemNative, targetNative):
    observed = tailVmemNative
    target_ = targetNative.reshape(-1)
    centredObserved = observed - observed.mean(axis=1, keepdims=True)
    centredTarget = target_ - target_.mean()
    covariance = (centredObserved * centredTarget).sum(axis=1)
    normalisation = (np.sqrt((centredObserved ** 2).sum(axis=1)) * np.sqrt((centredTarget ** 2).sum()))
    return float((1 - covariance / (normalisation + 1e-12)).mean())


def replayAndScoreFeatures(fileNumber, minIter, stride):
    p = torch.load(f'data/bestModelParameters_fieldVector_30x30_{fileNumber}.dat', map_location='cpu', weights_only=False)
    numSimIters = p['simParameters']['numSimIters']
    evalDuration = int(p['trainParameters']['evalDurationProp'] * numSimIters)
    targetNative = p['trainParameters']['targetVmem'].numpy()
    storedBestLoss = p['trainParameters']['bestLoss']
    lossMethod = p['trainParameters']['lossMethod']
    clampParameters = dict(p['clampParameters'])
    clampEndIter = int(clampParameters['clampEndIter'])
    p['latticePeriodicBoundaryGJ'] = False
    p['ATPParameters'] = None
    system = model(p, p['simParameters']['numSamples'])
    circuit = system.electricNetwork
    checkpoints = set(range(0, numSimIters, stride))
    checkpoints.add(numSimIters - 1)
    best = None
    tailVmemNative = []
    for it in range(numSimIters):
        cp = clampParameters if it <= clampEndIter else None
        system.simulate(clampParameters=cp, numSimIters=1, outerIter=it, fieldModulation=False)
        vmemNative = circuit.Vmem[0, :, 0].detach().numpy()
        if it >= numSimIters - evalDuration:
            tailVmemNative.append(vmemNative.copy())
        if it in checkpoints and it >= minIter:
            s = featureScore(vmemNative * 1000)
            if best is None or s < best[0]:
                best = (s, it)
    trust = None
    if lossMethod == 'correlation':
        replayedLoss = fullTissueCorrelationLoss(np.stack(tailVmemNative), targetNative)
        trust = (storedBestLoss, replayedLoss, abs(replayedLoss - storedBestLoss))
    return best, storedBestLoss, lossMethod, trust, clampEndIter, numSimIters


groups = {
    "Gpol-only S11/str0.25 corr": 1521,
    "Gpol-only S11/str0.25 glob": 1527,
    "GpolVmem  S11/str0.25 corr": 1534,
    "GpolVmem  S11/str0.25 glob": 1541,
}

print(f"{'group':<18}{'file':>6}{'stored bestLoss':>18}{'feature RMS (mV)':>20}{'at iter':>10}   trust check")
for label, fileNumber in groups.items():
    (score, iter_), storedLoss, lossMethod, trust, clampEndIter, numSimIters = replayAndScoreFeatures(fileNumber, args.minIter, args.stride)
    trustStr = f"stored={trust[0]:.4f} replayed={trust[1]:.4f} diff={trust[2]:.4f}" if trust else "(globalsum -- no correlation trust check)"
    print(f"{label:<18}{fileNumber:>6}{storedLoss:>18.4f}{score:>20.2f}{iter_:>10}   {trustStr}")
