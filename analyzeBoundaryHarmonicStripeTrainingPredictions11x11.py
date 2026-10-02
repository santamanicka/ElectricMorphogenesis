"""Score the stripe training against its registration: the gate (G1, G2), the predictions S1-S8 and the controls C1-C3 of
data/boundaryHarmonicTrainingPredictions1888Hold301StripesInteriorMinus60Minus5.json, as amended by Amendments 1-6 (and the note to 6): X1-X3 of Amendment 6 are scored on the exploratory even-only rung 2'.

Every criterion is evaluated as registered and reported with its number, passed or failed; nothing is folded in. Where an amendment
fixes how a criterion is read (Amendment 1: S2 is recorded as failed if the stripe needs an order above 6 or the ceiling of 2.0), that
reading is applied and the plain facts are given beside it.

Inputs: every restart of every stripe training folder (read from the .npz files: score, best moment, overlap, code), the summaries made
by analyzeBoundaryHarmonicTraining11x11.py (one per ceiling: the ceiling-1.3 pilot and the ceiling-2 pilot, each with its random codes, long
run and even-only arm), and the replays of analyzeBoundaryHarmonicFieldRole11x11.py.

    python3 analyzeBoundaryHarmonicStripeTrainingPredictions11x11.py --pilotSummaryPath ... --ceiling2SummaryPath ... --replayPath ...

Writes data/boundaryHarmonicTrainingPredictionScoring1888Hold301StripesInteriorMinus60Minus5.json (never overwriting).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301StripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--pilotSummaryPath', type=str, default=f'data/boundaryHarmonicTrainingSummary{SUFFIX}Ceiling1p3Combined.json')
parser.add_argument('--ceiling2SummaryPath', type=str, default=f'data/boundaryHarmonicTrainingSummary{SUFFIX}Ceiling2Combined.json')
parser.add_argument('--replayPath', type=str, default=f'data/boundaryHarmonicFieldRole{SUFFIX}.json')
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicTrainingPredictionScoring{SUFFIX}.json')
parser.add_argument('--overlapThreshold', type=float, default=0.9)
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
registration = json.load(open(f'data/boundaryHarmonicTrainingPredictions{SUFFIX}.json'))
threshold = args.overlapThreshold
stem = f'data/boundaryHarmonicTraining{SUFFIX}'

# ------------------------------------------------------------------------------------------------ every restart of every folder
FOLDERS = {'': ('1', 1.3, 16), 'Population64': ('2', 1.3, 64), 'Ceiling2': ('3 pilot', 2.0, 16), 'Ceiling2Population64': ('3b full run', 2.0, 64),
           'Ceiling2HigherOrdersPopulation16': ("4' (exploratory, cancelled)", 2.0, 16), 'Ceiling2EvenOrdersPopulation32': ("4'' (exploratory, cancelled)", 2.0, 32),
           'EvenOrdersPopulation64': ("2' (exploratory)", 1.3, 64)}
restarts = []
for suffix, (rung, ceiling, population) in FOLDERS.items():
    for path in sorted(glob.glob(f'{stem}{suffix}/*_restart*.npz')):
        run = np.load(path)
        orders = [int(o) for o in run['orders']] if 'orders' in run else list(range(int(run['maxOrder']) + 1))
        restarts.append(dict(rung=rung, folder=os.path.basename(os.path.dirname(path)), arm=re.sub(r'_restart.*', '', os.path.basename(path)), orders=orders,
                             contiguous=orders == list(range(orders[-1] + 1)), restart=int(run['restart']), ceiling=float(run['ceiling']), population=int(run['populationSize']),
                             score=float(run['bestScore']), iteration=int(run['bestIteration']), startType=str(run['startType']), coefficients=[float(c) for c in run['bestCoefficients']],
                             overlap=float(boundary.structuralIntersectionOverUnion(run['bestVmem'].astype(float), boundary.centreStripeCellIndices)),
                             numEvaluations=int(run['numEvaluations'])))
print(f'{len(restarts)} restarts read from {len({r["folder"] for r in restarts})} folders', flush=True)
exploratoryRung = "2' (exploratory)"
exploratory = [r for r in restarts if r['rung'] == exploratoryRung]
restarts = [r for r in restarts if r['rung'] != exploratoryRung]       # S1-S8, G1, G2 and C1-C3 are read on the rungs they were registered for
armsOf = lambda rung, arm=None: [r for r in restarts if r['rung'] == rung and (arm is None or r['arm'] == arm)]
best = lambda items: min(items, key=lambda r: r['score']) if items else None

verdicts = {}
# ---------------------------------------------------------------------------------------------------------------------- S1, G1
formed = [r for r in restarts if r['overlap'] >= threshold]
gateCode = best(formed)
verdicts['S1-stripesAreReachable'] = dict(criterion=next(p['criterion'] for p in registration['predictions'] if p['name'].startswith('S1')),
                                          restartsAtOrAboveThreshold=len(formed), firstRungs=sorted({r['rung'] for r in formed}),
                                          bestCode=dict(rung=gateCode['rung'], arm=gateCode['arm'], restart=gateCode['restart'], score=gateCode['score'], overlap=gateCode['overlap'],
                                                        coefficients=gateCode['coefficients'], iteration=gateCode['iteration']) if gateCode else None,
                                          holds=bool(formed))

# ---------------------------------------------------------------------------------------------------------------------- S2
def smallestContiguousSize(rungs):
    sizes = sorted({r['orders'][-1] for r in restarts if r['rung'] in rungs and r['contiguous'] and r['overlap'] >= threshold})
    return sizes[0] if sizes else None


smallest13, smallest2 = smallestContiguousSize({'1', '2'}), smallestContiguousSize({'3 pilot', '3b full run'})
verdicts['S2-fewerOrdersThanTheFace'] = dict(
    criterion=next(p['criterion'] for p in registration['predictions'] if p['name'].startswith('S2')),
    smallestContiguousSizeAtCeiling1p3=smallest13, smallestContiguousSizeAtCeiling2p0=smallest2,
    reading='Amendment 1: a stripe found only at ceiling 2.0 is reported as such and S2 is then recorded as failed',
    holds=bool(smallest13 is not None and smallest13 <= 2))

# ---------------------------------------------------------------------------------------------------------------------- S3
def bestScore(rung, arm):
    return min((r['score'] for r in armsOf(rung, arm)), default=None)


s3 = {}
for label, rung, contiguousArm, subsetArm in (('pilotAtCeiling1p3', '1', 'order2', 'orders0-2'), ('pilotAtCeiling2p0', '3 pilot', 'order2', 'orders0-2'),
                                               ('fullRunAtCeiling2p0', '3b full run', 'order4', 'orders0-2-4'), ('fullRunAtCeiling1p3', '2', 'order4', 'orders0-2-4')):
    a, b = bestScore(rung, subsetArm), bestScore(rung, contiguousArm)
    s3[label] = dict(evenOnlyBest=a, contiguousBest=b, difference=(a - b) if a is not None and b is not None else None,
                     holds=bool(a is not None and b is not None and a <= b + 1.0) if a is not None and b is not None else None)
verdicts['S3-oddOrdersAddNothing'] = dict(criterion=next(p['criterion'] for p in registration['predictions'] if p['name'].startswith('S3')), arms=s3,
                                          holds=bool(all(v['holds'] for v in s3.values() if v['holds'] is not None)) if any(v['holds'] is not None for v in s3.values()) else None)

# ---------------------------------------------------------------------------------------------------------------------- S4
sizesAtThreshold = {}
for r in formed:
    if r['contiguous'] and (r['orders'][-1] not in sizesAtThreshold or r['score'] < sizesAtThreshold[r['orders'][-1]]['score']):
        sizesAtThreshold[r['orders'][-1]] = r
oddCoefficients = {str(size): {f'a{n}': c for n, c in zip(r['orders'], r['coefficients']) if n % 2 == 1} for size, r in sizesAtThreshold.items()}
verdicts['S4-bestCodesAreTopBottomSymmetric'] = dict(criterion=next(p['criterion'] for p in registration['predictions'] if p['name'].startswith('S4')),
                                                     oddCoefficientsOfTheBestCodeAtEachSize=oddCoefficients,
                                                     holds=bool(all(abs(c) <= 0.10 for coefficients in oddCoefficients.values() for c in coefficients.values())) if oddCoefficients else None)

# ---------------------------------------------------------------------------------------------------------------------- S5
s5 = {}
for label, rung in (('pilotAtCeiling1p3', '1'), ('pilotAtCeiling2p0', '3 pilot')):
    arm = armsOf(rung, 'order3')
    s5[label] = dict(restarts=len(arm), atOrAboveThreshold=sum(1 for r in arm if r['overlap'] >= threshold), holds=bool(sum(1 for r in arm if r['overlap'] >= threshold) >= 10))
verdicts['S5-moreRestartsSucceed'] = dict(criterion=next(p['criterion'] for p in registration['predictions'] if p['name'].startswith('S5')), arms=s5,
                                          holds=bool(s5['pilotAtCeiling2p0']['holds']))

# ---------------------------------------------------------------------------------------------------------------------- S6, G2, C1-C3 (from the ceiling-2 summary)
if os.path.exists(args.ceiling2SummaryPath) and gateCode:
    summary2 = json.load(open(args.ceiling2SummaryPath))
    size = str(gateCode['orders'][-1])
    entry = summary2['orders'][size]
    longRun = entry['longRun']
    verdicts['S6-theStripePersists'] = dict(criterion=next(p['criterion'] for p in registration['predictions'] if p['name'].startswith('S6')),
                                            longestRunOfIterationsAtOrAboveThreshold=longRun.get('faceShapeLongestRun'), totalIterations=longRun['faceShapeIterations'],
                                            visits=longRun['faceShapeVisits'], span=longRun['faceShapeSpan'], thefacesLongestWas=119,
                                            holds=bool(longRun.get('faceShapeLongestRun') is not None and longRun['faceShapeLongestRun'] >= 150))
    randomBest = entry['randomCodes']['best']
    verdicts['G2-wellBelowRandomCodes'] = dict(criterion=registration['gate']['criterion'].split('AND G2:')[1].split('If G1')[0].strip(),
                                               codeScore=gateCode['score'], bestRandomCodeScore=randomBest, margin=randomBest - gateCode['score'],
                                               holds=bool(randomBest - gateCode['score'] >= 5.0))
    verdicts['C1-randomCodesDoNotMakeStripes'] = dict(criterion=registration['controls']['C1-randomCodesDoNotMakeStripes'],
                                                      maxOverlapOfTheRandomCodesBySize={k: v['randomCodes'].get('maxStructuralIoU') for k, v in summary2['orders'].items()},
                                                      holds=bool(all((v['randomCodes'].get('maxStructuralIoU') or 0) < threshold for v in summary2['orders'].values())))
    verdicts['C2-noClampBaseline'] = dict(criterion=registration['controls']['C2-noClampBaseline'], score=summary2['free']['score'], iteration=summary2['free']['iteration'],
                                          overlap=summary2['free']['parts']['structuralIoU'], descriptive=True)
    verdicts['C3-storedSweepBaseline'] = dict(criterion=registration['controls']['C3-storedSweepBaseline'], score=summary2['bestSweepCode']['rerunScore'],
                                              overlap=summary2['bestSweepCode']['parts']['structuralIoU'], descriptive=True)
# ---------------------------------------------------------------------------------------------------------------------- S7, S8
replays = json.load(open(args.replayPath))['regimes']
verdicts['S7-theFieldIsRequired'] = dict(criterion=next(p['criterion'] for p in registration['predictions'] if p['name'].startswith('S7')),
                                         mostInteriorCellsDarkAtOnceWithTheFieldOff=replays['field off, released']['maxDarkInteriorCells'],
                                         holds=bool(replays['field off, released']['maxDarkInteriorCells'] == 0))
verdicts['S8-theReleaseIsRequired'] = dict(criterion=next(p['criterion'] for p in registration['predictions'] if p['name'].startswith('S8')),
                                           highestOverlapWithTheRingHeldThroughout=replays['field on, held throughout']['maxStructuralOverlap'],
                                           iteration=replays['field on, held throughout']['iteration'],
                                           holds=bool(replays['field on, held throughout']['maxStructuralOverlap'] < threshold))

# ---------------------------------------------------------------------------------------------------------------------- X1-X3 (Amendment 6, exploratory rung 2')
amendment6 = json.load(open(f'data/boundaryHarmonicTrainingPredictionsAmendment6_{SUFFIX}.json'))
criterionOf = lambda name: next(p['criterion'] for p in amendment6['predictions'] if p['name'].startswith(name))
armsOfExploratory = sorted({r['arm'] for r in exploratory})
maxOverlapByArm = {arm: max((r['overlap'] for r in exploratory if r['arm'] == arm), default=None) for arm in armsOfExploratory}
atCeiling1p3 = [r for r in restarts if r['ceiling'] < 1.5] + exploratory
verdicts['X1-theFacesCeilingDoesNotMakeTheStripe'] = dict(
    criterion=criterionOf('X1'), restartsByArm={arm: len([r for r in exploratory if r['arm'] == arm]) for arm in armsOfExploratory}, maxOverlapByArm=maxOverlapByArm,
    restartsAtOrAboveThreshold=[dict(arm=r['arm'], restart=r['restart'], startType=r['startType'], overlap=r['overlap'], score=r['score'], iteration=r['iteration'],
                                     coefficients=r['coefficients']) for r in exploratory if r['overlap'] >= threshold],
    highestOverlapOfAnyRestartAtCeiling1p3=max((r['overlap'] for r in atCeiling1p3), default=None), holds=bool(exploratory and not any(r['overlap'] >= threshold for r in exploratory)))
fourSix = [r for r in exploratory if r['arm'] == 'orders0-2-4-6']
nearFour = [r for r in fourSix if r['overlap'] >= 0.8]
verdicts['X2-theNearMissIsAFamily'] = dict(
    criterion=criterionOf('X2'), restartsOfTheArm=len(fourSix), atOrAbove0p8=len(nearFour),
    byStartType={kind: dict(restarts=len([r for r in fourSix if r['startType'] == kind]), atOrAbove0p8=len([r for r in nearFour if r['startType'] == kind]))
                 for kind in sorted({r['startType'] for r in fourSix})},
    note='Amendment 6 note: the library contains the near miss itself, so the random-start split is the independent reading',
    holds=bool(len(nearFour) >= 3) if fourSix else None)
near = [r for r in exploratory if r['overlap'] >= 0.8]
late = [r for r in near if r['iteration'] > 1130]
verdicts['X3-theNearMissIsReadLate'] = dict(
    criterion=criterionOf('X3'), restartsAtOrAbove0p8=len(near), bestMomentAfter1130=len(late), bestMoments=sorted(r['iteration'] for r in near),
    holds=(bool(len(late) >= 0.75 * len(near)) if len(near) >= 4 else None))

for key, verdict in verdicts.items():
    status = 'descriptive' if verdict.get('descriptive') else ('holds' if verdict['holds'] else ('not yet scorable' if verdict['holds'] is None else 'FAILS'))
    print(f'{key}: {status}', {k: v for k, v in verdict.items() if k not in ('criterion', 'holds', 'descriptive')}, flush=True)

codesSimulated = {r['rung']: 0 for r in restarts}
for r in restarts:
    codesSimulated[r['rung']] += r['numEvaluations']
result = dict(note='Each criterion as registered (Amendments 1-6 applied where they say how to read it); the facts beside each are the numbers behind the verdict.',
              registration=f'data/boundaryHarmonicTrainingPredictions{SUFFIX}.json', amendments=[f'data/boundaryHarmonicTrainingPredictionsAmendment{k}{SUFFIX}.json'
                                                                                              if k == 1 else f'data/boundaryHarmonicTrainingPredictionsAmendment{k}_{SUFFIX}.json'
                                                                                              for k in range(1, 7)] + [f'data/boundaryHarmonicTrainingPredictionsAmendment6Note_{SUFFIX}.json'],
              overlapThreshold=threshold, verdicts=verdicts, codesSimulatedByRung=codesSimulated, totalCodesSimulated=int(sum(codesSimulated.values())),
              restartsByRung={rung: len([r for r in restarts if r['rung'] == rung]) for rung in codesSimulated},
              gateCode=verdicts['S1-stripesAreReachable']['bestCode'])
result['restartsByRung'][exploratoryRung] = len(exploratory)
result['codesSimulatedByRung'][exploratoryRung] = int(sum(r['numEvaluations'] for r in exploratory))
result['totalCodesSimulated'] += result['codesSimulatedByRung'][exploratoryRung]
json.dump(result, open(args.outputPath, 'w'), separators=(',', ':'))
print('wrote', args.outputPath, flush=True)
