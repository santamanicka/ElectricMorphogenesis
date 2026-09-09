"""Facial-feature-only scoring (Sim.md Sec 11.14's method) applied to the 11x11 band-hold sweep
(Sec 12.4): for each of the sixteen configurations (T in {100,300} x mechanism in {Gpol-only,
Gpol+Vmem} x depth in {D1,D2} x lossMethod in {correlation,globalsum}), replay its own best-of-twelve-
seeds checkpoint (the same winner plot11x11BandSweepEvolution.py already visualizes) over its entire
trajectory (every iteration, t=0 to numSimIters-1) and record the single best facial-feature score
reached anywhere in that trajectory, together with the timestep it occurred at -- not just the
checkpoint's own readout window.

Facial features = eyes + nose + mouth (faceFeatureIndices), excluding skin/boundary entirely -- a
stricter, more specific test than the full-tissue bulk loss in Sec 12.4, matching Sec 11.14's method at
30x30 exactly. faceFeatureIndices is lattice-size-generic (it *is* the native 11x11 definition,
verified against learnCellularFieldNetwork.py's own hardcoded blocks), so no rescaling is needed here.

Score is RMS distance (mV) to target (not correlation -- restricted to facial features alone the target
vector has zero variance, so Pearson correlation is undefined there, exactly as at 30x30).

Trust check: alongside each feature score, replay reproduces the checkpoint's own stored bestLoss using
its own lossMethod (correlation or globalsum) over the same tail evalDuration window -- if the two are
close, the replay is tracking the original trajectory faithfully enough for the facial-feature scoring
built on top of it to be trusted.
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model


def rowColumnBlock(circuitRows, circuitCols, rowFractions, columnFractions):
    """Reimplemented from learnCellularFieldNetwork.py rather than imported -- that file executes
    argument parsing and other top-level code on import, which is unsafe from another script."""
    firstRow, lastRow = (round(f * circuitRows) for f in rowFractions)
    firstCol, lastCol = (round(f * circuitCols) for f in columnFractions)
    return [r * circuitCols + c for r in range(firstRow, lastRow)
            for c in range(firstCol, lastCol)]


def faceFeatureIndices(circuitRows, circuitCols):
    """Eyes, nose and mouth of the smiley (matches learnCellularFieldNetwork.py's faceFeatureIndices
    exactly -- at 11x11 these fractions ARE the native hardcoded blocks, not a rescaling)."""
    leftEye = rowColumnBlock(circuitRows, circuitCols, (2/11, 4/11), (2/11, 4/11))
    rightEye = rowColumnBlock(circuitRows, circuitCols, (2/11, 4/11), (7/11, 9/11))
    nose = rowColumnBlock(circuitRows, circuitCols, (4/11, 7/11), (5/11, 6/11))
    mouth = rowColumnBlock(circuitRows, circuitCols, (8/11, 9/11), (4/11, 7/11))
    return leftEye + rightEye, nose, mouth


parser = argparse.ArgumentParser()
parser.add_argument('--minIter', type=int, default=0)
parser.add_argument('--stride', type=int, default=1)
args = parser.parse_args()

torch.set_grad_enabled(False)
rows, cols = 11, 11

eyeIdx, noseIdx, mouthIdx = faceFeatureIndices(rows, cols)
featureIdx = np.array(sorted(eyeIdx + noseIdx + mouthIdx))
print(f"facial-feature cells: {len(featureIdx)} (eyes {len(eyeIdx)}, nose {len(noseIdx)}, mouth {len(mouthIdx)}) "
      f"of {rows * cols} total")


def featureScore(vmemFlatMV, targetFeature):
    v = vmemFlatMV[featureIdx]
    return float(np.sqrt(np.mean((v - targetFeature) ** 2)))


def fullTissueLoss(tailVmemNative, targetNative, method):
    # Reimplements learnCellularFieldNetwork.py's computeLoss(method) for a single-sample, numpy
    # replay: tailVmemNative is (evalDuration, numCells), targetNative is (numCells,).
    if method == 'globalsum':
        return float(np.sqrt(np.sum((targetNative - tailVmemNative) ** 2)))
    elif method == 'correlation':
        centredObserved = tailVmemNative - tailVmemNative.mean(axis=1, keepdims=True)
        centredTarget = targetNative - targetNative.mean()
        covariance = (centredObserved * centredTarget).sum(axis=1)
        normalisation = (np.sqrt((centredObserved ** 2).sum(axis=1)) * np.sqrt((centredTarget ** 2).sum()))
        return float((1 - covariance / (normalisation + 1e-12)).mean())
    else:
        raise ValueError(f'unhandled lossMethod {method!r}')


def bestInGroup(nums):
    best = None
    for n in nums:
        f = f'data/bestModelParameters_fieldVector_11x11_{n}.dat'
        try:
            p = torch.load(f, map_location='cpu', weights_only=False)
        except FileNotFoundError:
            continue
        L = float(p['trainParameters']['bestLoss'])
        if best is None or L < best[0]:
            best = (L, n, p)
    return best


def replayAndScoreFeatures(p, minIter, stride):
    numSimIters = p['simParameters']['numSimIters']
    evalDuration = int(p['trainParameters']['evalDurationProp'] * numSimIters)
    lossMethod = p['trainParameters']['lossMethod']
    targetNative = p['trainParameters']['targetVmem'].numpy().reshape(-1)  # native units (volts)
    targetFeature = (targetNative * 1000)[featureIdx]  # mV, for feature scoring
    storedBestLoss = p['trainParameters']['bestLoss']
    clampParameters = dict(p['clampParameters'])
    clampEndIter = int(clampParameters['clampEndIter'])
    p = dict(p)
    p['latticePeriodicBoundaryGJ'] = False
    p['ATPParameters'] = None
    system = model(p, p['simParameters']['numSamples'])
    system.setExperimentalConditions((p['simParameters']['initialValues'], p['simParameters']['numSamples']))
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
            s = featureScore(vmemNative * 1000, targetFeature)
            if best is None or s < best[0]:
                best = (s, it)
    replayedBestLoss = fullTissueLoss(np.stack(tailVmemNative), targetNative, lossMethod)
    print(f"  [trust check, {lossMethod}] stored bestLoss={storedBestLoss:.4f}, "
          f"replayed loss over same readout window={replayedBestLoss:.4f}, "
          f"diff={abs(replayedBestLoss - storedBestLoss):.4f}")
    return best


fileNum = 1600
cases = []
for T in [100, 300]:
    for mech, mechLabel in [('Gpol', 'Gpol-only'), ('GpolVmem', 'Gpol+Vmem')]:
        for depth in [1, 2]:
            for loss, lossLabel in [('corr', 'correlation'), ('glob', 'globalsum')]:
                nums = list(range(fileNum, fileNum + 6)) + list(range(fileNum + 96, fileNum + 96 + 6))
                cases.append((f'{mechLabel}, D{depth}, T={T}, {lossLabel}', nums))
                fileNum += 6

results = {}
for label, nums in cases:
    L, n, p = bestInGroup(nums)
    print(f"{label} (winner: file {n}, bulk bestLoss={L:.4f}):")
    s, it = replayAndScoreFeatures(p, args.minIter, args.stride)
    results[label] = (s, it, n, L)
    print(f"  best facial-feature score={s:.2f} mV at iter {it}")

print()
print(f"Ranking (facial features only, best RMS distance to target over the entire trajectory, lower is better):")
for label, (s, it, n, L) in sorted(results.items(), key=lambda kv: kv[1][0]):
    print(f"  {s:6.2f} mV  {label:38s} (file {n}, best-face timestep={it}, bulk bestLoss={L:.3f})")

fig, ax = plt.subplots(figsize=(8, 6.5))
labelsSorted = [label for label, _ in sorted(results.items(), key=lambda kv: kv[1][0])]
values = [results[l][0] for l in labelsSorted]
colors = ['C0' if 'correlation' in l else 'C1' for l in labelsSorted]
ax.barh(labelsSorted[::-1], values[::-1], color=colors[::-1])
ax.set_xlabel('best facial-feature RMS distance to target, mV (lower better), over the entire trajectory')
ax.set_title('11x11 band-hold sweep: facial-feature-only ranking (blue=correlation loss, orange=globalsum)')
fig.tight_layout()
outPath = 'figures/facialFeatureScore11x11BandSweep.png'
fig.savefig(outPath, dpi=140, bbox_inches='tight')
print(f"\nwrote {outPath}")
