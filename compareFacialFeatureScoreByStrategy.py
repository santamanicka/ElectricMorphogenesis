"""For each of the four screen11/strength0.25 strategies (random clamp, transplant, trained
boundary clamp, trained single-shot), find the best facial-feature-only score reached at any point
after a given iteration, not just at the readout window each was originally scored on.

Facial features = eyes + nose + mouth (faceFeatureIndices), excluding skin/boundary entirely --
a stricter, more specific test than the bulk-only scoring of Sim.md Sec 11.11. All four are scored
against the same fixed idealised face target (identical across every training checkpoint regardless
of mechanism/screen, verified directly), not the transplant's own scaledTarget1000_mV reference, so
the comparison is apples-to-apples with what the trained mechanisms were actually optimised against.

Score is RMS distance (mV) to target, not correlation: the target is binary (eyes+nose+mouth all sit
at the same -60 mV feature value, background at -9.2 mV), so restricted to facial features alone the
target vector has zero variance and Pearson correlation is undefined there (confirmed empirically --
every trial returned NaN on a first attempt scored that way). Lower is better.

Random clamp and transplant read from already-saved trajectories (no re-simulation) -- there is no
replay-fidelity question for these two, since nothing is being re-simulated. Trained boundary and
trained single-shot have no stored trajectory, so this replays their trained clamp parameters fresh
out to their own numSimIters=1000 -- subject to the same chaotic-sensitivity caveat as
runExtendedHorizon.py/compareTrainedPrepatternCommitment.py. To check that fidelity rather than just
assert it, the replay also reproduces the exact full-tissue correlation loss the checkpoint was
originally trained/scored on (learnCellularFieldNetwork.py's computeLoss(method='correlation'),
averaged over the same tail evalDuration window) and reports it beside the checkpoint's stored
bestLoss -- if the two are close, the replay is tracking the original trajectory faithfully enough
through the readout window for the facial-feature scoring built on top of it to be trusted.
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model


def rowColumnBlock(circuitRows, circuitCols, rowFractions, columnFractions):
    """Cell indices of a rectangular block given as (start, end) fractions of the lattice.
    Reimplemented from learnCellularFieldNetwork.py rather than imported -- that file executes
    argument parsing and other top-level code on import, which is unsafe from another script."""
    firstRow, lastRow = (round(f * circuitRows) for f in rowFractions)
    firstCol, lastCol = (round(f * circuitCols) for f in columnFractions)
    return [r * circuitCols + c for r in range(firstRow, lastRow)
            for c in range(firstCol, lastCol)]


def faceFeatureIndices(circuitRows, circuitCols):
    """Eyes, nose and mouth of the smiley, as fractions of the lattice (matches
    learnCellularFieldNetwork.py's faceFeatureIndices exactly)."""
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
target = referenceCheckpoint['trainParameters']['targetVmem'].reshape(rows, cols).numpy() * 1000  # mV
eyeIdx, noseIdx, mouthIdx = faceFeatureIndices(rows, cols)
featureIdx = np.array(sorted(eyeIdx + noseIdx + mouthIdx))
targetFeature = target.reshape(-1)[featureIdx]
print(f"facial-feature cells: {len(featureIdx)} (eyes {len(eyeIdx)}, nose {len(noseIdx)}, mouth {len(mouthIdx)})")


def featureScore(vmemFlatMV):
    v = vmemFlatMV[featureIdx]
    return float(np.sqrt(np.mean((v - targetFeature) ** 2)))


results = {}

# -- Random clamp (8 trials, native fieldDomeTwoFoldSymmetry-derived, released at screen11/str0.25) --
d = torch.load('data/randomClampDerivedPrepattern_strength0.25.dat', map_location='cpu', weights_only=False)
bestRandom = None
for label, storedIters, V, _corr, _readRow in d['results']:
    if label == 'baseline':
        continue
    for it, row in zip(storedIters, V):
        if it < args.minIter:
            continue
        s = featureScore(row)
        if bestRandom is None or s < bestRandom[0]:
            bestRandom = (s, int(it), label)
results['random clamp (8 trials)'] = bestRandom
print(f"random clamp: best feature score={bestRandom[0]:.2f} mV at iter {bestRandom[1]} ({bestRandom[2]})")

# -- Transplant release (full 5000-iter sweep, screen10.909/strength0.25) --
rel = torch.load('data/releasedScaledFace_30x30_screen10.9091_strength0.25.dat', map_location='cpu', weights_only=False)
storedIters = rel['storedIters']
V = rel['Vmem_mV']
V = V.numpy() if hasattr(V, 'numpy') else V
bestTransplant = None
for it, row in zip(storedIters, V):
    if it < args.minIter:
        continue
    s = featureScore(row)
    if bestTransplant is None or s < bestTransplant[0]:
        bestTransplant = (s, int(it))
results['transplant (screen10.909/str0.25)'] = bestTransplant
print(f"transplant: best feature score={bestTransplant[0]:.2f} mV at iter {bestTransplant[1]}")


def fullTissueCorrelationLoss(tailVmemNative, targetNative):
    # Exact reimplementation of learnCellularFieldNetwork.py's computeLoss(method='correlation'):
    # Pearson correlation across all cells (not just facial features), per tail iteration, averaged
    # over the evalDuration readout window. This is what bestLoss actually is -- reproducing it from
    # a fresh replay is the trust check: if it lands near the stored bestLoss, the replay is tracking
    # the original trajectory closely enough (at least through the readout window) for the
    # facial-feature scoring built on top of it to be believed.
    observed = tailVmemNative  # (evalDuration, numCells)
    target = targetNative.reshape(-1)  # (numCells,)
    centredObserved = observed - observed.mean(axis=1, keepdims=True)
    centredTarget = target - target.mean()
    covariance = (centredObserved * centredTarget).sum(axis=1)
    normalisation = (np.sqrt((centredObserved ** 2).sum(axis=1)) * np.sqrt((centredTarget ** 2).sum()))
    return float((1 - covariance / (normalisation + 1e-12)).mean())


def replayAndScoreFeatures(fileNumber, minIter, stride):
    p = torch.load(f'data/bestModelParameters_fieldVector_30x30_{fileNumber}.dat', map_location='cpu', weights_only=False)
    numSimIters = p['simParameters']['numSimIters']
    evalDuration = int(p['trainParameters']['evalDurationProp'] * numSimIters)
    targetNative = p['trainParameters']['targetVmem'].numpy()  # native units (volts), matches training
    storedBestLoss = p['trainParameters']['bestLoss']
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
    replayedBestLoss = fullTissueCorrelationLoss(np.stack(tailVmemNative), targetNative)
    print(f"  [trust check] file {fileNumber}: stored bestLoss={storedBestLoss:.4f}, "
          f"replayed full-tissue loss over same readout window={replayedBestLoss:.4f}, "
          f"diff={abs(replayedBestLoss - storedBestLoss):.4f}")
    return best


bestBoundary = replayAndScoreFeatures(1823, args.minIter, args.stride)
results['trained boundary clamp (file 1823)'] = bestBoundary
print(f"trained boundary clamp: best feature score={bestBoundary[0]:.2f} mV at iter {bestBoundary[1]}")

bestSingleShot = replayAndScoreFeatures(1833, args.minIter, args.stride)
results['trained single-shot (file 1833)'] = bestSingleShot
print(f"trained single-shot: best feature score={bestSingleShot[0]:.2f} mV at iter {bestSingleShot[1]}")

print()
print(f"Ranking (facial features only, best RMS distance to target after iter {args.minIter}, lower is better):")
for label, (s, it, *_) in sorted(results.items(), key=lambda kv: kv[1][0]):
    print(f"  {s:6.2f} mV  {label}  (iter {it})")

fig, ax = plt.subplots(figsize=(7, 4))
labels = list(results.keys())
values = [results[l][0] for l in labels]
colors = ['C0', 'C1', 'C2', 'C3']
ax.barh(labels, values, color=colors)
ax.set_xlabel(f'best facial-feature RMS distance to target, mV (lower better), iter >= {args.minIter}')
fig.tight_layout()
fig.savefig('figures/facialFeatureScoreByStrategy.png', dpi=140, bbox_inches='tight')
print("\nwrote figures/facialFeatureScoreByStrategy.png")
