"""Build the "Boundary Dial" report page (PolyPatterning_Sim.md, Section 12).

Part 1 (trained seeds, Gpol-only D1): the dial as mean ring conductance, the interior pattern at low / middle /
high dial, the change it drives and its pattern-PC2 mode, and the trained codes' leading mode. Part 2 (reference
checkpoint 1888): the dial across its physical range, from a fine dial-only sweep with the checkpoint's own
100-iteration hold, compared with the same sweep under 301- and 500-iteration holds, with the late readout fixed at
iterations 2000-2999 or aligned to release, once simulated. Part 3 (reference checkpoint 1888): first-order gradients
within each single-cell regime and rotation grids, included once they have been simulated and analysed. Part 4: circular harmonics of clamp and pattern (dial sweeps, gradient pairs, trained
seeds), included once analyzeBoundaryPatternHarmonics11x11.py has run.

Inputs: data/bandHoldFaceMetrics11x11_all384.csv and data/bandHoldPatterns11x11_all384.npz
(scoreBandHoldFaceMetrics11x11.py); data/boundaryDialSweep1888.npz, data/boundaryDialSweep1888Hold301.npz,
data/boundaryDialSweep1888Hold500.npz, the Hold<n>Aligned sweeps and data/boundaryRegimePairs1888.npz
(simulateBoundaryDialLandscape11x11.py); data/boundaryDialLandscapeSummary1888.json
(analyzeBoundaryDialLandscape11x11.py); data/boundaryPatternHarmonicsSummary1888.json
(analyzeBoundaryPatternHarmonics11x11.py); data/boundaryHoldInfluence1888.json (analyzeBoundaryHoldInfluence11x11.py,
the clamp-free reference); and the checkpoints. The page's JSON keys are the ones
figures/boundaryDialTemplate.html reads.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--metricsPath', type=str, default='data/bandHoldFaceMetrics11x11_all384.csv')
parser.add_argument('--patternsPath', type=str, default='data/bandHoldPatterns11x11_all384.npz')
parser.add_argument('--sweepPath', type=str, default='data/boundaryDialSweep1888.npz')
parser.add_argument('--longHoldSweepPaths', type=str,
                    default='data/boundaryDialSweep1888Hold301.npz,data/boundaryDialSweep1888Hold500.npz')
parser.add_argument('--alignedSweepPaths', type=str,
                    default='data/boundaryDialSweep1888Hold301Aligned.npz,data/boundaryDialSweep1888Hold500Aligned.npz')
parser.add_argument('--regimePairsPath', type=str, default='data/boundaryRegimePairs1888.npz')
parser.add_argument('--landscapeSummaryPath', type=str, default='data/boundaryDialLandscapeSummary1888.json')
parser.add_argument('--patternHarmonicsPath', type=str, default='data/boundaryPatternHarmonicsSummary1888.json')
parser.add_argument('--holdInfluencePath', type=str, default='data/boundaryHoldInfluence1888.json')
parser.add_argument('--templatePath', type=str, default='figures/boundaryDialTemplate.html')
parser.add_argument('--outputPath', type=str, default='figures/boundaryDial.html')
parser.add_argument('--dataOutputPrefix', type=str, default=None, help='also write the Part 1 JSON payload to <prefix>part1.json')
args = parser.parse_args()

# ============================================================================= Part 1: trained seeds
metrics = pd.read_csv(args.metricsPath).sort_values('fileNumber').reset_index(drop=True)
seeds = metrics[(metrics.mechanism == 'Gpol-only') & (metrics.depth == 1)].reset_index(drop=True)
patternFile = np.load(args.patternsPath)
patternRow = {int(number): index for index, number in enumerate(patternFile['fileNumbers'])}
codeRecords = boundary.loadBandHoldCodes(seeds.fileNumber.tolist())
ringCells = boundary.boundaryRingCells
ringCodes = np.stack([codeRecords[number]['field'][ringCells] for number in seeds.fileNumber])
patterns = np.stack([patternFile['windowMean'][patternRow[number]] for number in seeds.fileNumber])
dial = ringCodes.mean(1)

codePca = PCA().fit(ringCodes - ringCodes.mean(0))
codeLoading = codePca.components_[0]
codeScore = codePca.transform(ringCodes - ringCodes.mean(0))[:, 0]
if spearmanr(codeScore, dial).statistic < 0:
    codeLoading, codeScore = -codeLoading, -codeScore
positions = np.arange(len(ringCells))
harmonicPower = np.array([np.abs((codeLoading * np.exp(-2j * np.pi * order * positions / len(ringCells))).sum())
                          for order in range(21)]) ** 2

centred = patterns - patterns.mean(1, keepdims=True)
patternPca = PCA(5).fit(StandardScaler().fit_transform(centred))
patternScores = patternPca.transform(StandardScaler().fit_transform(centred))
modeLoading, modeScore = patternPca.components_[1], patternScores[:, 1]
if spearmanr(dial, modeScore).statistic < 0:
    modeLoading, modeScore = -modeLoading, -modeScore
configurations = seeds.holdIterations.astype(str) + '/' + seeds.lossMethod
controlled = spearmanr(boundary.rankWithinGroups(dial, configurations), boundary.rankWithinGroups(modeScore, configurations))
order = np.argsort(dial)
lowest, middle, highest = order[:24], order[36:60], order[-24:]
changeMap = centred[highest].mean(0) - centred[lowest].mean(0)
hyperpolarizingLobe, depolarizingLobe = np.argsort(modeLoading)[:15], np.argsort(-modeLoading)[:15]
lobes = {name: (float(centred[indices][:, hyperpolarizingLobe].mean()), float(centred[indices][:, depolarizingLobe].mean()))
         for name, indices in (('low', lowest), ('high', highest))}

part1 = dict(ringCells=ringCells.tolist(), codeLoading=[round(float(value), 4) for value in codeLoading],
             codeVar=float(codePca.explained_variance_ratio_[0]), codeRhoDose=float(spearmanr(codeScore, dial).statistic),
             order0Share=float(harmonicPower[0] / harmonicPower.sum()),
             harmonicShare=[round(float(power / harmonicPower.sum()), 4) for power in harmonicPower[:9]],
             lowPat=[round(float(value), 2) for value in patterns[lowest].mean(0)],
             midPat=[round(float(value), 2) for value in patterns[middle].mean(0)],
             highPat=[round(float(value), 2) for value in patterns[highest].mean(0)],
             doseTerciles=[float(dial[lowest].mean()), float(dial[middle].mean()), float(dial[highest].mean())],
             relDiff=[round(float(value), 2) for value in changeMap], pc2=[round(float(value), 3) for value in modeLoading / np.abs(modeLoading).max()],
             pc2Var=float(patternPca.explained_variance_ratio_[1]), diffVsPc2=float(np.corrcoef(changeMap, modeLoading)[0, 1]),
             rho=float(spearmanr(dial, modeScore).statistic), rhoCtrl=float(controlled.statistic),
             lobe=lobes, featureCells=boundary.featureCellIndices.tolist(),
             negLobe=hyperpolarizingLobe.tolist(), posLobe=depolarizingLobe.tolist(),
             pts=[dict(f=int(seeds.fileNumber[index]), d=round(float(dial[index]), 4), y=round(float(modeScore[index]), 3),
                       L=str(seeds.lossMethod[index]), T=int(seeds.holdIterations[index])) for index in range(len(seeds))])

# ================================================================ Part 2: the dial across its physical range
lowerThreshold, upperThreshold = boundary.singleCellBistableRange
saddle = boundary.singleCellSaddleMilliVolts
interior = boundary.interiorCellIndices
filmDials = [0.0, 0.3, 0.4, 0.6, 0.95, 1.1, 1.15, 1.22, 1.35, 1.48, 1.65, 2.0]


def sweepReadouts(sweep):
    """Per-dial readouts of one dial sweep, as the page's JSON."""
    sweepDials, latePatterns = sweep['dialLevel'], sweep['windowMeanVmem']
    lateConductance = sweep['windowMeanGpol'][:, interior]
    faceScores = [boundary.partSeparationScore(pattern) for pattern in latePatterns]
    filmIndices = [int(np.argmin(np.abs(sweepDials - value))) for value in filmDials]
    return dict(dials=sweepDials.round(3).tolist(), holdIterations=int(sweep['holdIterations']),
                steps=np.sqrt(((latePatterns[1:] - latePatterns[:-1]) ** 2).mean(1)).round(3).tolist(),
                ringHyperpolarisedAtHold=(sweep['endOfHoldVmem'][:, ringCells] < saddle).mean(1).round(3).tolist(),
                interiorHyperpolarisedAtHold=(sweep['endOfHoldVmem'][:, interior] < saddle).mean(1).round(3).tolist(),
                interiorHyperpolarisedLate=(latePatterns[:, interior] < saddle).mean(1).round(3).tolist(),
                interiorInWindowLate=((lateConductance >= lowerThreshold) & (lateConductance <= upperThreshold)).mean(1).round(3).tolist(),
                interiorConductanceLate=lateConductance.mean(1).round(3).tolist(),
                contrast=latePatterns[:, interior].std(1).round(3).tolist(),
                churn=sweep['windowStdVmem'][:, interior].mean(1).round(3).tolist(),
                separation=[int(score[2]) for score in faceScores], coverage=[round(float(score[1]), 3) for score in faceScores],
                partScore=[round(float(score[0]), 3) for score in faceScores],
                film=[dict(dial=float(sweepDials[index]), pattern=latePatterns[index].round(2).tolist(),
                           separation=int(faceScores[index][2])) for index in filmIndices],
                window=[lowerThreshold, upperThreshold], trainedDial=float(dial[seeds.fileNumber == 1888][0]) if (seeds.fileNumber == 1888).any() else None)


sweep = np.load(args.sweepPath)
sweepDials, latePatterns = sweep['dialLevel'], sweep['windowMeanVmem']
sweepPayload = sweepReadouts(sweep)
longHoldPayloads = {}
compareIndices = [int(np.argmin(np.abs(sweepDials - value))) for value in (0.0, 0.6, 1.15, 1.3, 1.47, 1.51, 1.75)]
for path in args.longHoldSweepPaths.split(','):
    if not os.path.exists(path):
        continue
    longHold = np.load(path)
    payload = sweepReadouts(longHold)
    payload['holdDifference'] = np.sqrt(((longHold['windowMeanVmem'] - latePatterns) ** 2).mean(1)).round(3).tolist()
    payload['compareFilm'] = [dict(dial=float(sweepDials[index]), shortLate=latePatterns[index].round(2).tolist(),
                                   shortSeparation=sweepPayload['separation'][index],
                                   longEndOfHold=longHold['endOfHoldVmem'][index].round(2).tolist(),
                                   longLate=longHold['windowMeanVmem'][index].round(2).tolist(),
                                   longSeparation=payload['separation'][index]) for index in compareIndices]
    longHoldPayloads[str(int(longHold['holdIterations']))] = payload

# the longer holds again, with the late window aligned to release rather than fixed at iterations 2000-2999
alignedPayloads = {}
for path in args.alignedSweepPaths.split(','):
    if not os.path.exists(path):
        continue
    aligned = np.load(path)
    hold = str(int(aligned['holdIterations']))
    readouts = sweepReadouts(aligned)
    payload = {key: readouts[key] for key in ('steps', 'separation', 'coverage', 'partScore', 'contrast', 'churn')}
    payload['windowStart'] = int(aligned['windowStart'])
    payload['holdDifference'] = np.sqrt(((aligned['windowMeanVmem'] - latePatterns) ** 2).mean(1)).round(3).tolist()
    fixedPath = path.replace('Aligned', '')
    if os.path.exists(fixedPath):
        payload['timingDifference'] = np.sqrt(((aligned['windowMeanVmem'] - np.load(fixedPath)['windowMeanVmem']) ** 2).mean(1)).round(3).tolist()
    payload['film'] = [aligned['windowMeanVmem'][index].round(2).tolist() for index in compareIndices]
    payload['filmSeparation'] = [readouts['separation'][index] for index in compareIndices]
    alignedPayloads[hold] = payload

# ======================================================== Part 3: first-order gradients, regime by regime
regimesPayload = None
if os.path.exists(args.regimePairsPath) and os.path.exists(args.landscapeSummaryPath):
    runs = np.load(args.regimePairsPath)
    landscape = json.load(open(args.landscapeSummaryPath))
    if landscape['regimes']:
        twinPatterns = latePatterns[runs['pairSweepIndex']]
        sizes = np.sqrt(((runs['pairWindowMeanVmem'] - twinPatterns) ** 2).mean(1))
        sweepPayload['gridDials'] = sorted(set(float(value) for value in runs['gridDial']))
        regimesPayload = dict(points=[dict(regime=str(runs['pairRegime'][index]), dial=round(float(runs['pairDialLevel'][index]), 3),
                                           strength=round(float(runs['pairGradientStrength'][index]), 4), size=round(float(sizes[index]), 3))
                                      for index in range(len(sizes))],
                              regimes=landscape['regimes'], rotationGrids=landscape['rotationGrids'])

part1Json = json.dumps(part1, separators=(',', ':'))
sweepJson, regimesJson = json.dumps(sweepPayload, separators=(',', ':')), json.dumps(regimesPayload, separators=(',', ':'))
longHoldJson = json.dumps(longHoldPayloads or None, separators=(',', ':'))
alignedJson = json.dumps(alignedPayloads or None, separators=(',', ':'))
harmonicsJson = open(args.patternHarmonicsPath).read() if os.path.exists(args.patternHarmonicsPath) else 'null'
influenceJson = open(args.holdInfluencePath).read() if os.path.exists(args.holdInfluencePath) else 'null'
if args.dataOutputPrefix:
    open(args.dataOutputPrefix + 'part1.json', 'w').write(part1Json)
page = (open(args.templatePath).read().replace('__DATA__', part1Json).replace('__SWEEP__', sweepJson)
        .replace('__REGIMES__', regimesJson).replace('__LONGHOLDS__', longHoldJson)
        .replace('__ALIGNED__', alignedJson)
        .replace('__HARMONICS__', harmonicsJson).replace('__FREE__', influenceJson))
open(args.outputPath, 'w').write(page)
print(f"Part 1: rho {part1['rho']:+.3f} (controlled {part1['rhoCtrl']:+.3f}), PC2 {part1['pc2Var'] * 100:.1f}% of variance, "
      f"code PC1 {part1['codeVar'] * 100:.1f}% ({part1['order0Share'] * 100:.0f}% order 0)")
print(f"Part 2: {len(sweepDials)} dial levels, longer holds {', '.join(longHoldPayloads) or 'not yet available'} (release-aligned: {', '.join(alignedPayloads) or 'none'}); Part 3: {'included' if regimesPayload else 'not yet available'}")
print(f"Part 4: {'included' if harmonicsJson != 'null' else 'not yet available'}; clamp-free reference: {'included' if influenceJson != 'null' else 'not yet available'}")
print(f"wrote {args.outputPath}")
