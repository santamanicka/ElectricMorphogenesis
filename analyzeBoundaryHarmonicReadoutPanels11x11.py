"""The single readout of A13, made visible code by code (PolyPatterning_Sim.md, Section 12).

A13 reports one number: the best linear readout of the pattern that the code has, 0.49 on held-out codes at the moment
the face is scored. This script produces what that number looks like. The same scattered ensemble is replayed, the
canonical pair is fitted on the interior's spatial Fourier modes exactly as in A13, and three things are written for
every code: the part of the ring code the readout responds to, the pattern that readout predicts, and the pattern the
tissue actually reaches.

The readout is one number per code, u, a weighted sum of the four coefficients. The ring panel shows the code's
component along that direction alone, so it varies with u and nothing else. The predicted pattern is each cell's best
linear prediction from u. The difference between that and the actual pattern is what the 0.49 leaves out.

Writes data/boundaryHarmonicReadoutPanels<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np
import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--trainedRunPath', type=str, default='data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz')
parser.add_argument('--numCodes', type=int, default=1024)
parser.add_argument('--numFolds', type=int, default=5)
parser.add_argument('--seed', type=int, default=17)
args = parser.parse_args()

run = dict(np.load(args.trainedRunPath))
outputPath = f"data/boundaryHarmonicReadoutPanels{int(run['referenceCheckpoint'])}Hold{int(run['holdIterations'])}{run['targetName']}.json"
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
reference = boundary.loadCheckpoint(int(run['referenceCheckpoint']))
hold, numIterations = int(run['holdIterations']), int(run['numIterations'])
trainedCode, trainedMoment = run['bestCoefficients'], int(run['bestIteration'])
target = np.asarray(run['target']).ravel()
featureMask = np.isin(np.arange(boundary.numCells), boundary.featureCellIndices)
angles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(angles, np.arange(len(trainedCode))))

# the same ensemble as the ownership runs: same seed, same rejection sampling
generator = np.random.default_rng(args.seed)
codes = []
while len(codes) < args.numCodes:
    candidate = np.concatenate([generator.uniform(0.3, 1.3, 1), generator.uniform(-0.6, 0.6, len(trainedCode) - 1)])
    ringValues = basis @ candidate
    if ringValues.min() >= 0.02 and ringValues.max() <= 1.98:
        codes.append(candidate)
codes = np.array(codes)
print(f'{len(codes)} codes; replaying to iteration {trainedMoment}', flush=True)

patterns = {}
boundary.ringHoldBatchReplay(reference, codes @ basis.T, hold, numIterations,
                             lambda iteration, vmem: patterns.update({iteration: vmem.numpy().copy()})
                             if iteration == trainedMoment else None)
vmem = patterns[trainedMoment]
squared = (vmem - target) ** 2
scores = 0.5 * np.sqrt(squared[:, featureMask].mean(1)) + 0.5 * np.sqrt(squared[:, ~featureMask].mean(1))

# ------------------------------------------------------------------- the readout, on A13's interior modes
size = boundary.latticeRows - 2
interior = boundary.interiorCellIndices
modes = np.column_stack([np.outer(np.cos(np.pi * p * (np.arange(size) + 0.5) / size),
                                  np.cos(np.pi * q * (np.arange(size) + 0.5) / size)).ravel()
                         for p in range(size) for q in range(size)])
modes = modes / np.linalg.norm(modes, axis=0)
amplitudes = np.array([modes.T @ (row[interior] - row[interior].mean()) for row in vmem])
amplitudes = amplitudes[:, amplitudes.std(0) > 1e-9]

codeMean, codeDeviation = codes.mean(0), codes.std(0)
standardCodes = (codes - codeMean) / codeDeviation
heldOutCorrelation, perFold, direction = boundary.crossValidatedReadout(standardCodes, amplitudes, seed=1)

# v: where each pattern sits along the readout's direction in mode space
kept = amplitudes[:, amplitudes.std(0) > 1e-9]
patternReadout = ((kept - kept.mean(0)) / kept.std(0)) @ direction
patternReadout = (patternReadout - patternReadout.mean()) / patternReadout.std()

# u: what the code predicts of v, which is the readout proper
design = np.column_stack([np.ones(len(codes)), standardCodes])
fit = np.linalg.lstsq(design, patternReadout, rcond=None)[0]
readout = design @ fit
if np.corrcoef(readout, patternReadout)[0, 1] < 0:                # sign is arbitrary; point them the same way
    readout, patternReadout, fit = -readout, -patternReadout, -fit
readoutSpread = readout.std()
readout = (readout - readout.mean()) / readoutSpread
inSampleCorrelation = float(np.corrcoef(readout, patternReadout)[0, 1])
readoutWeights = fit[1:] / readoutSpread

# the ring code the readout responds to: the code's component along the readout's own direction
codeDirection = readoutWeights / codeDeviation
alongReadout = codeDirection / (codeDirection @ codeDirection)
ringMean, ringDirection = basis @ codeMean, basis @ alongReadout * readout.std()

# the pattern that readout predicts: every cell's best linear prediction from u alone
patternMean = vmem.mean(0)
patternDirection = ((vmem - patternMean) * readout[:, None]).mean(0) / readout.var()

print(f'held-out readout {heldOutCorrelation:.3f}, in sample {inSampleCorrelation:.3f}; the readout explains '
      f'{np.mean([(np.corrcoef(vmem[:, cell], readout)[0, 1] ** 2) for cell in range(boundary.numCells)]) * 100:.0f}% '
      f'of the average cell\'s variance', flush=True)
print('readout weights ' + ', '.join(f'a{index} {value:+.3f}' for index, value in enumerate(readoutWeights)), flush=True)
result = dict(trainedMoment=trainedMoment, readoutWeights=np.round(readoutWeights, 4).tolist(), numCodes=len(codes), heldOutCorrelation=round(float(np.mean(heldOut)), 4),
              codeMean=np.round(codeMean, 5).tolist(), ringMean=np.round(ringMean, 4).tolist(),
              ringDirection=np.round(ringDirection, 4).tolist(), patternMean=np.round(patternMean, 2).tolist(),
              patternDirection=np.round(patternDirection, 3).tolist(), ceiling=float(np.max(basis @ codeMean)),
              codes=[dict(coefficients=np.round(codes[index], 4).tolist(), readout=round(float(readout[index]), 3),
                          patternReadout=round(float(patternReadout[index]), 3), score=round(float(scores[index]), 2),
                          vmem=np.round(vmem[index], 1).tolist())
                     for index in range(len(codes))])
json.dump(result, open(outputPath, 'w'))
print('wrote', outputPath, f'({os.path.getsize(outputPath) / 1e6:.1f} MB)')
