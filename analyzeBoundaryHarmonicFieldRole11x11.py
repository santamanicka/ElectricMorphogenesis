"""What the extracellular field's feedback does to the face (PolyPatterning_Sim.md, Section 12).

The trained code is replayed under four regimes: the field's transduction on or off, and the ring released after the
usual hold or held for the whole run. Without the field, G_pol is no longer driven after the clamp, so the tissue
merely relaxes; with it, the tissue keeps evolving. Each run reports the balanced RMS at its own best moment, the
moment it falls, and how much of the face appears.

Writes data/boundaryHarmonicFieldRole<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import copy
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--trainedRunPath', type=str, default='data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz')
args = parser.parse_args()

run = dict(np.load(args.trainedRunPath))
outputPath = f"data/boundaryHarmonicFieldRole{int(run['referenceCheckpoint'])}Hold{int(run['holdIterations'])}{run['targetName']}.json"
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
hold, numIterations = int(run['holdIterations']), int(run['numIterations'])
code, target = run['bestCoefficients'], np.asarray(run['target']).ravel()
featureMask = np.isin(np.arange(boundary.numCells), boundary.featureCellIndices)
angles = boundary.ringAngles(boundary.boundaryRingCells)
ringValues = np.clip(np.cos(np.outer(angles, np.arange(len(code)))) @ code, 0, 2)[None, :]

result = dict(code=code.tolist(), regimes={})
for fieldEnabled in (True, False):
    for released in (True, False):
        reference = boundary.loadCheckpoint(int(run['referenceCheckpoint']))
        if not fieldEnabled:
            reference = copy.deepcopy(reference)
            reference['fieldParameters'] = dict(reference['fieldParameters'])
            reference['fieldParameters']['fieldEnabled'] = False
        best = dict(score=np.inf, iteration=0, vmem=None)
        scoreFrom = hold if released else 0

        def onIteration(iteration, vmem, best=best, scoreFrom=scoreFrom):
            if iteration < scoreFrom:
                return
            values = vmem.numpy()[0]
            squared = (values - target) ** 2
            score = 0.5 * np.sqrt(squared[featureMask].mean()) + 0.5 * np.sqrt(squared[~featureMask].mean())
            if score < best['score']:
                best.update(score=score, iteration=iteration, vmem=values)

        boundary.ringHoldBatchReplay(reference, ringValues, hold if released else numIterations, numIterations, onIteration)
        dark = best['vmem'] < boundary.hyperpolarizedThresholdMilliVolts
        name = ('field on' if fieldEnabled else 'field off') + (', released' if released else ', held throughout')
        result['regimes'][name] = dict(score=round(float(best['score']), 3), iteration=int(best['iteration']),
                                       featureCoverage=round(float(dark[boundary.featureCellIndices].mean()), 4),
                                       spuriousDark=int(dark[boundary.nonFeatureInteriorCellIndices].sum()),
                                       vmem=np.round(best['vmem'], 1).tolist())
        entry = result['regimes'][name]
        print(f"{name}: {entry['score']:.2f} mV at iteration {entry['iteration']}, "
              f"{entry['featureCoverage'] * 100:.0f}% of the face's cells dark, {entry['spuriousDark']} stray dark cells", flush=True)

json.dump(result, open(outputPath, 'w'))
print('wrote', outputPath)
