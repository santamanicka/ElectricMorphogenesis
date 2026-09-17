"""Build the "Boundary Dial" report page (PolyPatterning_Sim.md, Section 12).

Part 1 (trained seeds, Gpol-only D1): the dial as mean ring conductance, the interior pattern at low / middle /
high dial, the change it drives and its pattern-PC2 mode, and the trained codes' leading mode. Part 2 (paired
simulations on the reference checkpoint): the dial's mode against the first-order gradient's steerable modes,
a rotation filmstrip, symmetric share, shell reach, dimensionality, controllable modes and gating.

Inputs: data/bandHoldFaceMetrics11x11_all384.csv and data/bandHoldPatterns11x11_all384.npz
(scoreBandHoldFaceMetrics11x11.py), data/boundaryFirstOrderPairs1888.npz (simulateBoundaryHarmonics11x11.py) and
the checkpoints. The page's JSON keys are the ones figures/boundaryDialTemplate.html reads.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--metricsPath', type=str, default='data/bandHoldFaceMetrics11x11_all384.csv')
parser.add_argument('--patternsPath', type=str, default='data/bandHoldPatterns11x11_all384.npz')
parser.add_argument('--pairsPath', type=str, default='data/boundaryFirstOrderPairs1888.npz')
parser.add_argument('--templatePath', type=str, default='figures/boundaryDialTemplate.html')
parser.add_argument('--outputPath', type=str, default='figures/boundaryDial.html')
parser.add_argument('--dataOutputPrefix', type=str, default=None, help='also write the two JSON payloads to <prefix>part1.json, part2.json')
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

# ============================================================================ Part 2: paired simulations
runs = np.load(args.pairsPath)
dialLevel, gradientStrength, gradientDirection = runs['pairDialLevel'], runs['pairGradientStrength'], runs['pairGradientDirection']
twins = runs['pairTwinPatterns']
differentials = runs['pairGradientPatterns'] - twins
twinsByDial = twins[np.argsort(dialLevel)]
dialMode = twinsByDial[-15:].mean(0) - twinsByDial[:15].mean(0)
centredDialMode = dialMode - dialMode.mean()

gridStrength, gridDirection = runs['gridGradientStrength'], runs['gridGradientDirection']
gridChanges = runs['gridPatterns'] - runs['gridBaselinePattern']
steering = {}
for strength in sorted(set(gridStrength)):
    members = gridStrength == strength
    sortOrder = np.argsort(gridDirection[members])
    changes, directions = gridChanges[members][sortOrder], gridDirection[members][sortOrder]
    independent = changes.mean(0)
    topToBottom = (changes * np.cos(directions)[:, None]).sum(0) * 2 / 8
    leftToRight = (changes * np.sin(directions)[:, None]).sum(0) * 2 / 8
    energy = (changes ** 2).sum()
    steering[strength] = dict(W=independent, U=topToBottom, V=leftToRight, X=changes, phi=directions,
                              eW=(independent ** 2).sum() * 8 / energy,
                              eUV=((topToBottom ** 2).sum() + (leftToRight ** 2).sum()) * 4 / energy,
                              rW=float(np.corrcoef(independent - independent.mean(), centredDialMode)[0, 1]))

features = np.column_stack([dialLevel, np.log(gradientStrength), gradientStrength * np.cos(gradientDirection),
                            gradientStrength * np.sin(gradientDirection), np.cos(gradientDirection), np.sin(gradientDirection),
                            dialLevel * gradientStrength])
modePca = PCA(8).fit(differentials)
modeScores = modePca.transform(differentials)
estimator = make_pipeline(StandardScaler(), GridSearchCV(KernelRidge(kernel='rbf'), {'alpha': [1e-3, 1e-2, 1e-1], 'gamma': [0.05, 0.2, 0.5]}, cv=3))
modeR2 = []
for mode in range(8):
    predictions = cross_val_predict(estimator, features, modeScores[:, mode], cv=KFold(6, shuffle=True, random_state=0))
    modeR2.append(float(1 - ((modeScores[:, mode] - predictions) ** 2).sum() / ((modeScores[:, mode] - modeScores[:, mode].mean()) ** 2).sum()))
tracking = [tuple(float(spearmanr(modeScores[:, mode], values).statistic) for values in
                  (np.cos(gradientDirection), np.sin(gradientDirection), dialLevel, np.log(gradientStrength))) for mode in range(8)]
moved = np.sqrt((differentials ** 2).mean(1)) > 1.0
symmetricShares = np.array([boundary.symmetricShare(change) for change in differentials[moved]])
dialOnlyVariance = PCA().fit(twins - twins.mean(0)).explained_variance_ratio_
filmStrength = 0.1

part2 = dict(dialChange=dialMode.round(2).tolist(), evDial=PCA(8).fit(twins - twins.mean(0)).explained_variance_ratio_.round(4).tolist(),
             dialVsTrained=float(np.corrcoef(centredDialMode, np.array(part1['relDiff']))[0, 1]),
             steerG=filmStrength, W=steering[filmStrength]['W'].round(2).tolist(), U=steering[filmStrength]['U'].round(2).tolist(),
             V=steering[filmStrength]['V'].round(2).tolist(), eW=float(steering[filmStrength]['eW']), eUV=float(steering[filmStrength]['eUV']),
             film=[dict(phi=float(direction), d=change.round(2).tolist())
                   for direction, change in zip(steering[filmStrength]['phi'], steering[filmStrength]['X'])],
             steerByG=[dict(G=float(strength), eW=float(entry['eW']), eUV=float(entry['eUV']), rW=entry['rW']) for strength, entry in sorted(steering.items())],
             dModes=[dict(load=(modePca.components_[mode] / np.abs(modePca.components_[mode]).max()).round(3).tolist(),
                          var=float(modePca.explained_variance_ratio_[mode]), r2=modeR2[mode], sym=boundary.symmetricShare(modePca.components_[mode]),
                          cos=tracking[mode][0], sin=tracking[mode][1], dc=tracking[mode][2], logG=tracking[mode][3]) for mode in range(8)],
             symDeltaMedian=float(np.median(symmetricShares)),
             symDeltaIQR=[float(np.percentile(symmetricShares, 25)), float(np.percentile(symmetricShares, 75))], nBig=int(moved.sum()),
             dialShell=boundary.shellRootMeanSquare(dialMode),
             gradShell=np.median([boundary.shellRootMeanSquare(change) for change in differentials[gradientStrength >= 0.2]], 0).tolist(),
             nStrong=int((gradientStrength >= 0.2).sum()),
             prDialAll=boundary.participationRatio(dialOnlyVariance),
             prDeltaAll=boundary.participationRatio(PCA().fit(differentials).explained_variance_ratio_),
             dialRange=float(np.sqrt((dialMode ** 2).mean())),
             gate=[dict(G=float(gradientStrength[index]), dc=float(dialLevel[index]), m=float(np.sqrt((differentials[index] ** 2).mean())))
                   for index in range(len(dialLevel))])

part1Json, part2Json = json.dumps(part1, separators=(',', ':')), json.dumps(part2, separators=(',', ':'))
if args.dataOutputPrefix:
    open(args.dataOutputPrefix + 'part1.json', 'w').write(part1Json)
    open(args.dataOutputPrefix + 'part2.json', 'w').write(part2Json)
page = open(args.templatePath).read().replace('__DATA__', part1Json).replace('__DATA1__', part2Json)
open(args.outputPath, 'w').write(page)
print(f"Part 1: rho {part1['rho']:+.3f} (controlled {part1['rhoCtrl']:+.3f}), PC2 {part1['pc2Var'] * 100:.1f}% of variance, "
      f"code PC1 {part1['codeVar'] * 100:.1f}% ({part1['order0Share'] * 100:.0f}% order 0)")
print(f"Part 2: dial dimensions {part2['prDialAll']:.2f}, gradient {part2['prDeltaAll']:.2f}; "
      f"match to Part 1 r {part2['dialVsTrained']:+.2f}; controlled modes {[mode + 1 for mode in range(8) if modeR2[mode] > 0.2]}")
print(f"wrote {args.outputPath}")
