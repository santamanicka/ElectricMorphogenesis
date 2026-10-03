"""Score the two-bump screen of simulateBoundaryHarmonicDoubleStripesBumps11x11.py against its registration
(data/boundaryHarmonicDoubleStripesBumpPredictions1888Hold301DoubleStripesInteriorMinus60Minus5.json and its Amendment 1):
the primary readings B1p, B2p, B3p on the pure two-bump profiles (M = K = S = B), the secondary readings B1, B2, B3 on the registered grid, and the
controls C1, C2. Every criterion is reported with the numbers behind it, held or failed.

    python3 analyzeBoundaryHarmonicDoubleStripesBumps11x11.py

Writes data/boundaryHarmonicDoubleStripesBumpScoring<checkpoint>Hold<hold><targetName>.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np

SUFFIX = '1888Hold301DoubleStripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--screenPath', type=str, default=f'data/boundaryHarmonicDoubleStripesBumps{SUFFIX}.json')
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicDoubleStripesBumpScoring{SUFFIX}.json')
parser.add_argument('--overlapThreshold', type=float, default=0.9)
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
screen = json.load(open(args.screenPath))
registration = json.load(open(f'data/boundaryHarmonicDoubleStripesBumpPredictions{SUFFIX}.json'))
threshold = args.overlapThreshold
one = screen['stage1']
overlap, score = np.array(one['overlapAtBest']), np.array(one['score'])
families = [set(f) for f in one['families']]
T, M, K, S = (np.array(one[key]) for key in 'TMKS')
criterionOf = lambda name: next(p['criterion'] for p in registration['predictions'] if p['name'].startswith(name))
bases = {0.8, 1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.40, 1.43, 1.44, 1.46, 1.50}


def ranked(indices):
    return sorted(indices, key=lambda i: (-overlap[i], score[i]))


def describe(i):
    return dict(index=int(i), T=T[i], M=M[i], K=K[i], S=S[i], families=sorted(families[i]), overlapAtBest=float(overlap[i]), score=float(score[i]), bestIteration=one['bestIteration'][i],
                strayDarkCells=one['strayAtBest'][i], flankCellsDark=one['featureDarkAtBest'][i], centreCellsDarkAtBest=one['centreDarkAtBest'][i],
                highestOverlapAtAnyIteration=one['maxOverlap'][i], ringMin=float(min(one['ringValues'][i])), ringMax=float(max(one['ringValues'][i])))


pure = [i for i in range(len(families)) if 'pure' in families[i]]
wide = [i for i in range(len(families)) if 'wide' in families[i]]
uniform = [i for i in range(len(families)) if 'uniform' in families[i]]
two = screen['stage2']
projectionsOf = lambda profile: [j for j in range(len(two['profile'])) if two['profile'][j] == profile]


def lowestOrder(profile, kind, limit=None):
    """The lowest maxOrder of a projection of this profile with overlap >= threshold, or None."""
    reached = sorted(two['maxOrder'][j] for j in projectionsOf(profile) if two['kind'][j] == kind and two['overlapAtBest'][j] >= threshold)
    return reached[0] if reached else None


def projectionTable(profile):
    return [dict(kind=two['kind'][j], maxOrder=two['maxOrder'][j], overlapAtBest=two['overlapAtBest'][j], score=two['score'][j], fitMaxError=two['fitMaxError'][j],
                 sideRipple=two['sideRipple'][j], ringMin=two['ringMin'][j], ringMax=two['ringMax'][j], coefficients=two['coefficients'][j]) for j in projectionsOf(profile)]


verdicts = {}
bestPure = ranked(pure)[0]
verdicts['B1p-twoBumpsFormTheFlanks'] = dict(
    criterion=criterionOf('B1'), reading='primary: the pure two-bump profiles (M = K = S = B < T), Amendment 1', profilesTried=len(pure),
    highestOverlapAtBestMoment=float(overlap[bestPure]), profilesAtOrAboveThreshold=int(sum(overlap[i] >= threshold for i in pure)),
    topProfiles=[describe(i) for i in ranked(pure)[:8]], holds=bool(overlap[bestPure] >= threshold))
plateau = [i for i in wide if M[i] == T[i] and K[i] == S[i] and S[i] in bases]
bestUniform, bestPlateau = ranked(uniform)[0], (ranked(plateau)[0] if plateau else None)
rival = max(overlap[bestUniform], overlap[bestPlateau] if bestPlateau is not None else 0.0)
verdicts['B2p-theDipMatters'] = dict(
    criterion='the pure two-bump profile with the highest overlap beats the best uniform ring and the best plateau profile by at least 0.10 in overlap at the best moment (Amendment 1)',
    bestPure=describe(bestPure), bestUniformRing=describe(bestUniform), bestPlateauProfile=describe(bestPlateau) if bestPlateau is not None else None,
    margin=float(overlap[bestPure] - rival), holds=bool(overlap[bestPure] - rival >= 0.10))
if overlap[bestPure] >= threshold:
    even, contiguous = lowestOrder(bestPure, 'evenOnly'), lowestOrder(bestPure, 'contiguous')
    verdicts['B3p-aLowOrderCosineCodeReproducesIt'] = dict(
        criterion=criterionOf('B3'), reading='primary: the best pure two-bump profile', profile=describe(bestPure), lowestEvenOnlyOrderThatForms=even, lowestContiguousOrderThatForms=contiguous,
        projections=projectionTable(bestPure), holds=bool(even is not None and even <= 10))
else:
    verdicts['B3p-aLowOrderCosineCodeReproducesIt'] = dict(criterion=criterionOf('B3'), reading='primary: not scored, no pure profile forms (Amendment 1)', holds=None,
                                                          projectionsOfTheBestPureProfileForReference=projectionTable(bestPure))

# secondary readings on the registered grid (including high sides)
dips = [i for i in wide if M[i] < T[i]]
bestOverall = ranked(range(len(families)))[0]
verdicts['B1-twoBumpsFormTheFlanks (registered grid, secondary)'] = dict(
    criterion=criterionOf('B1'), profilesWithADip=len(dips), highestOverlapAtBestMoment=float(overlap[ranked(dips)[0]]), best=describe(ranked(dips)[0]),
    profilesAtOrAboveThreshold=int(sum(overlap[i] >= threshold for i in dips)), holds=bool(overlap[ranked(dips)[0]] >= threshold))
verdicts['B2-theDipMatters (registered grid, secondary)'] = dict(
    criterion=criterionOf('B2'), bestOverall=describe(bestOverall), holds=bool(M[bestOverall] < T[bestOverall]))
bestOverallLowest = lowestOrder(bestOverall, 'evenOnly') if bestOverall in set(screen['chosenProfiles']['pure'] + screen['chosenProfiles']['wider']) else None
verdicts['B3-aLowOrderCosineCodeReproducesIt (registered grid, secondary)'] = dict(
    criterion=criterionOf('B3'), profile=describe(bestOverall), lowestEvenOnlyOrderThatForms=bestOverallLowest, highestOverlap=float(overlap[bestOverall]),
    holds=(bool(bestOverallLowest is not None and bestOverallLowest <= 10) if overlap[bestOverall] >= threshold else None))
verdicts['C1-uniformRingsDoNotForm'] = dict(criterion=registration['controls']['C1-uniformRingsDoNotForm'], uniformRings=[describe(i) for i in uniform],
                                            holds=bool(all(overlap[i] < threshold for i in uniform)))
nearest = [i for i in wide if T[i] == 1.48 and M[i] == 1.48 and K[i] == 1.48 and S[i] in (1.2, 1.3)]
verdicts['C2-stripeShapedRingsAreNotTheFlanks'] = dict(criterion=registration['controls']['C2-stripeShapedRingsAreNotTheFlanks'],
                                                       note='the grid has no side level 1.25; the neighbours 1.2 and 1.3 are shown', profiles=[describe(i) for i in nearest], descriptive=True)

for key, verdict in verdicts.items():
    status = 'descriptive' if verdict.get('descriptive') else ('holds' if verdict['holds'] else ('not scored' if verdict['holds'] is None else 'FAILS'))
    print(f'{key}: {status}', {k: v for k, v in verdict.items() if k in ('profilesTried', 'highestOverlapAtBestMoment', 'profilesAtOrAboveThreshold', 'margin', 'lowestEvenOnlyOrderThatForms', 'lowestContiguousOrderThatForms')}, flush=True)
json.dump(dict(note='Scoring of the two-bump screen against its registration and Amendment 1.', screen=args.screenPath, overlapThreshold=threshold, verdicts=verdicts,
               numProfiles=len(families), numProjections=len(two['profile'])), open(args.outputPath, 'w'), separators=(',', ':'))
print('wrote', args.outputPath)
