"""Comprehensive scoring of all 192 individual checkpoints of the 11x11 band-hold sweep (16
configurations x 12 seeds each), not just the 16 per-config winners -- so a later session can look at
the full seed-level distribution across every score this investigation has used, without re-running any
replays.

Everything below comes from a single forward-only replay per checkpoint (every iteration, t=0 to
numSimIters-1, its own stored clampParameters) -- no checkpoint is replayed twice:

- bulkBestLoss: the checkpoint's own stored training-time loss (correlation or globalsum, whichever it
  was trained on), reproduced here as a trust check (bulkCorrelationLoss/bulkGlobalsumLoss below, over
  the checkpoint's own tail evalDuration window) -- if the reproduction is close to bulkBestLoss, the
  replay is tracking the original run faithfully.
- bulkCorrelationLoss, bulkGlobalsumLoss: full-tissue loss under BOTH methods, always both computed
  (not just the one the checkpoint was trained on), so every row is comparable on either scale
  regardless of which loss method it happened to train with (Sim.md Sec 12.4's caveat is that the
  globalsum scale isn't comparable across itself and 30x30 -- it says nothing about comparing globalsum
  and correlation to each other at 11x11, which these two columns make possible directly).
- storedActualVmemFeatureRMS_mV: facial-feature RMS distance of the checkpoint's own saved best-face
  readout (actualVmem) -- zero-cost, no replay needed, this is what training itself already scored best.
- featureRMS_mV / featureBestTimestep: best facial-feature-only (eyes+nose+mouth) RMS distance reached
  anywhere in the entire replayed trajectory, and when (Sec 12.7's method, per-checkpoint here rather
  than per-config-winner).
- eyesRMS_mV / noseRMS_mV / mouthRMS_mV (at featureBestTimestep): the combined score's three components
  at that same shared best moment -- does the best combined face come from all three features doing
  well at once, or from one carrying the average?
- eyesBestRMS_mV/eyesBestTimestep, noseBestRMS_mV/noseBestTimestep, mouthBestRMS_mV/mouthBestTimestep:
  each individual feature's own best moment anywhere in the trajectory, independently of the other two
  and of the combined score's best moment.

Runtime: ~2s/checkpoint x 192 = ~6-7 minutes total (forward-only replay, cheap at 11x11).
"""
import argparse
import csv

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
parser.add_argument('--outPath', type=str, default='data/facialFeatureScore11x11_all192.csv')
args = parser.parse_args()

torch.set_grad_enabled(False)
rows, cols = 11, 11

eyeIdx, noseIdx, mouthIdx = faceFeatureIndices(rows, cols)
featureIdx = np.array(sorted(eyeIdx + noseIdx + mouthIdx))
eyeIdxArr, noseIdxArr, mouthIdxArr = np.array(eyeIdx), np.array(noseIdx), np.array(mouthIdx)
print(f"facial-feature cells: {len(featureIdx)} of {rows * cols} total "
      f"(eyes {len(eyeIdx)}, nose {len(noseIdx)}, mouth {len(mouthIdx)})")


def rmsAt(vmemFlatMV, idx, targetFullMV):
    v = vmemFlatMV[idx]
    return float(np.sqrt(np.mean((v - targetFullMV[idx]) ** 2)))


def fullTissueLoss(tailVmemNative, targetNative, method):
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


def replayAndScore(p):
    numSimIters = p['simParameters']['numSimIters']
    evalDuration = int(p['trainParameters']['evalDurationProp'] * numSimIters)
    lossMethod = p['trainParameters']['lossMethod']
    targetNative = p['trainParameters']['targetVmem'].numpy().reshape(-1)  # volts
    targetMV = targetNative * 1000
    storedBestLoss = p['trainParameters']['bestLoss']
    storedActualVmemMV = p['trainParameters']['actualVmem'].numpy().reshape(-1) * 1000
    storedActualFeatureRMS = rmsAt(storedActualVmemMV, featureIdx, targetMV)

    clampParameters = dict(p['clampParameters'])
    clampEndIter = int(clampParameters['clampEndIter'])
    p = dict(p)
    p['latticePeriodicBoundaryGJ'] = False
    p['ATPParameters'] = None
    system = model(p, p['simParameters']['numSamples'])
    system.setExperimentalConditions((p['simParameters']['initialValues'], p['simParameters']['numSamples']))
    circuit = system.electricNetwork

    bestCombined = None       # (score, iter) -- combined eyes+nose+mouth
    bestEyes = None
    bestNose = None
    bestMouth = None
    tailVmemNative = []
    for it in range(numSimIters):
        cp = clampParameters if it <= clampEndIter else None
        system.simulate(clampParameters=cp, numSimIters=1, outerIter=it, fieldModulation=False)
        vmemNative = circuit.Vmem[0, :, 0].detach().numpy()
        if it >= numSimIters - evalDuration:
            tailVmemNative.append(vmemNative.copy())
        vmemMV = vmemNative * 1000
        sCombined = rmsAt(vmemMV, featureIdx, targetMV)
        sEyes = rmsAt(vmemMV, eyeIdxArr, targetMV)
        sNose = rmsAt(vmemMV, noseIdxArr, targetMV)
        sMouth = rmsAt(vmemMV, mouthIdxArr, targetMV)
        if bestCombined is None or sCombined < bestCombined[0]:
            bestCombined = (sCombined, it, sEyes, sNose, sMouth)
        if bestEyes is None or sEyes < bestEyes[0]:
            bestEyes = (sEyes, it)
        if bestNose is None or sNose < bestNose[0]:
            bestNose = (sNose, it)
        if bestMouth is None or sMouth < bestMouth[0]:
            bestMouth = (sMouth, it)

    tailVmemNative = np.stack(tailVmemNative)
    bulkCorrelationLoss = fullTissueLoss(tailVmemNative, targetNative, 'correlation')
    bulkGlobalsumLoss = fullTissueLoss(tailVmemNative, targetNative, 'globalsum')
    replayedOwnMethodLoss = bulkCorrelationLoss if lossMethod == 'correlation' else bulkGlobalsumLoss
    trustCheckDiff = abs(replayedOwnMethodLoss - storedBestLoss)

    return {
        'bulkBestLoss': storedBestLoss,
        'bulkCorrelationLoss': bulkCorrelationLoss,
        'bulkGlobalsumLoss': bulkGlobalsumLoss,
        'trustCheckDiff': trustCheckDiff,
        'storedActualVmemFeatureRMS_mV': storedActualFeatureRMS,
        'featureRMS_mV': bestCombined[0],
        'featureBestTimestep': bestCombined[1],
        'eyesRMS_mV': bestCombined[2],
        'noseRMS_mV': bestCombined[3],
        'mouthRMS_mV': bestCombined[4],
        'eyesBestRMS_mV': bestEyes[0],
        'eyesBestTimestep': bestEyes[1],
        'noseBestRMS_mV': bestNose[0],
        'noseBestTimestep': bestNose[1],
        'mouthBestRMS_mV': bestMouth[0],
        'mouthBestTimestep': bestMouth[1],
    }


def buildRowsMeta():
    # Same 16-configuration x 12-seed (2 batches of 6) construction as
    # plot11x11BandSweepEvolution.py / compareFacialFeatureScore11x11.py, but expanded to one row per
    # individual file rather than one row per config-winner.
    fileNum = 1600
    rows_meta = []
    for T in [100, 300]:
        for mech, mechLabel in [('Gpol', 'Gpol-only'), ('GpolVmem', 'Gpol+Vmem')]:
            for depth in [1, 2]:
                for loss, lossLabel in [('corr', 'correlation'), ('glob', 'globalsum')]:
                    for batch, offset in [(1, 0), (2, 96)]:
                        for seedInBatch in range(6):
                            n = fileNum + offset + seedInBatch
                            rows_meta.append({
                                'fileNumber': n, 'mechanism': mechLabel, 'depth': depth, 'T': T,
                                'lossMethod': lossLabel, 'batch': batch, 'seedInBatch': seedInBatch,
                            })
                    fileNum += 6
    return rows_meta


FIELDNAMES = ['fileNumber', 'mechanism', 'depth', 'T', 'lossMethod', 'batch', 'seedInBatch',
              'bulkBestLoss', 'bulkCorrelationLoss', 'bulkGlobalsumLoss', 'trustCheckDiff',
              'storedActualVmemFeatureRMS_mV', 'featureRMS_mV', 'featureBestTimestep',
              'eyesRMS_mV', 'noseRMS_mV', 'mouthRMS_mV',
              'eyesBestRMS_mV', 'eyesBestTimestep', 'noseBestRMS_mV', 'noseBestTimestep',
              'mouthBestRMS_mV', 'mouthBestTimestep']


def main():
    rows_meta = buildRowsMeta()
    print(f"scoring {len(rows_meta)} checkpoints...")

    results = []
    with open(args.outPath, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for i, meta in enumerate(rows_meta):
            n = meta['fileNumber']
            f = f'data/bestModelParameters_fieldVector_11x11_{n}.dat'
            try:
                p = torch.load(f, map_location='cpu', weights_only=False)
            except FileNotFoundError:
                print(f"  [{i+1}/{len(rows_meta)}] file {n}: MISSING, skipping")
                continue
            scores = replayAndScore(p)
            row = dict(meta)
            row.update(scores)
            results.append(row)
            writer.writerow(row)
            fh.flush()
            print(f"  [{i+1}/{len(rows_meta)}] file {n} ({meta['mechanism']}, D{meta['depth']}, T={meta['T']}, "
                  f"{meta['lossMethod']}, seed {meta['seedInBatch']} of batch {meta['batch']}): "
                  f"featureRMS={scores['featureRMS_mV']:.2f} mV @ t={scores['featureBestTimestep']}, "
                  f"trust diff={scores['trustCheckDiff']:.4f}")

    print(f"\nwrote {args.outPath} ({len(results)} rows)")
    best = min(results, key=lambda r: r['featureRMS_mV'])
    print(f"\noverall best facial-feature score: file {best['fileNumber']} "
          f"({best['mechanism']}, D{best['depth']}, T={best['T']}, {best['lossMethod']}), "
          f"{best['featureRMS_mV']:.2f} mV @ t={best['featureBestTimestep']}")


if __name__ == '__main__':
    main()
