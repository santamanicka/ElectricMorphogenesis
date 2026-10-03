"""Score the double-stripe training against its registration: the gate (G1, G2), the predictions D1-D8 and the controls C1-C3 of
data/boundaryHarmonicTrainingPredictions1888Hold301DoubleStripesInteriorMinus60Minus5.json, read as its note (Amendment 1) says.
A per-target counterpart of analyzeBoundaryHarmonicStripeTrainingPredictions11x11.py. Every criterion is evaluated as registered and
reported with the numbers behind it, held or failed; nothing is folded in.

Inputs: every restart file of the three arms (A: ceiling 1.3 pilot, B: ceiling 2.0 pilot, C: the even-only chain at 1.3; 1a and 1b: rung 1 of the ladder, Amendment 2), read for score,
best moment, code and the overlap of the saved best pattern with the 54 flank cells; the summaries of
analyzeBoundaryHarmonicTraining11x11.py (the contiguous sizes of arm A, tag Pilot, and of arm B, tag Ceiling2Pilot, and one per even-only
set that has a formed restart, run with --orders and tag EvenSet<orders joined by ->); and the replays of
analyzeBoundaryHarmonicFieldRole11x11.py on the primary code.

    python3 analyzeBoundaryHarmonicDoubleStripesTrainingPredictions11x11.py --selectPrimary      # only names the primary code; writes nothing
    python3 analyzeBoundaryHarmonicDoubleStripesTrainingPredictions11x11.py                      # the scoring; run once, never overwrites

Writes data/boundaryHarmonicTrainingPredictionScoring1888Hold301DoubleStripesInteriorMinus60Minus5.json.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301DoubleStripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--replayPath', type=str, default=f'data/boundaryHarmonicFieldRole{SUFFIX}.json')
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicTrainingPredictionScoring{SUFFIX}.json')
parser.add_argument('--overlapThreshold', type=float, default=0.9)
parser.add_argument('--selectPrimary', action='store_true', help='print the primary code and the best code of each route, write nothing')
args = parser.parse_args()
if not args.selectPrimary and os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
registration = json.load(open(f'data/boundaryHarmonicTrainingPredictions{SUFFIX}.json'))
threshold = args.overlapThreshold
stem = f'data/boundaryHarmonicTraining{SUFFIX}'
criterionOf = lambda prefix: next(p['criterion'] for p in registration['predictions'] if p['name'].startswith(prefix))

# ------------------------------------------------------------------------------------------------ every restart of every arm
ARMS = {'': ('A', 1.3, 16), 'Ceiling2': ('B', 2.0, 16), 'EvenOrdersPopulation64': ('C', 1.3, 64),
        'Population64': ('1a', 1.3, 64), 'Ceiling2Population64': ('1b', 2.0, 64)}      # 1a and 1b: rung 1 of the ladder (Amendment 2)
restarts = []
for suffix, (arm, ceiling, population) in ARMS.items():
    for path in sorted(glob.glob(f'{stem}{suffix}/*_restart*.npz')):
        run = np.load(path)
        orders = [int(o) for o in run['orders']] if 'orders' in run else list(range(int(run['maxOrder']) + 1))
        restarts.append(dict(arm=arm, folder=os.path.basename(os.path.dirname(path)), path=path, label=re.sub(r'_restart.*', '', os.path.basename(path)), orders=orders,
                             contiguous=orders == list(range(orders[-1] + 1)), restart=int(run['restart']), ceiling=float(run['ceiling']),
                             population=int(run['populationSize']), score=float(run['bestScore']), iteration=int(run['bestIteration']), startType=str(run['startType']),
                             coefficients=[float(c) for c in run['bestCoefficients']],
                             overlap=float(boundary.structuralIntersectionOverUnion(run['bestVmem'].astype(float), boundary.flankCellIndices)),
                             numEvaluations=int(run['numEvaluations'])))
print(f'{len(restarts)} restarts read from {len({r["folder"] for r in restarts})} folders', flush=True)
ARMS_ORDER = ['A', 'B', 'C', '1a', '1b']
armsOf = lambda arm, label=None: [r for r in restarts if r['arm'] == arm and (label is None or r['label'] == label)]
best = lambda items: min(items, key=lambda r: r['score']) if items else None
highestOverlap = {f"{arm}: {label}": round(max(r['overlap'] for r in restarts if r['arm'] == arm and r['label'] == label), 4)
                  for arm in ARMS_ORDER for label in sorted({r['label'] for r in restarts if r['arm'] == arm})}
formed = [r for r in restarts if r['overlap'] >= threshold]                       # G1 and G2: any registered arm, ladder rungs included
formedABC = [r for r in formed if r['arm'] in ('A', 'B', 'C')]                    # D1 is registered on arms A, B and C

# ------------------------------------------------------------------------------------------------ the controls of each code space (the summaries)
summaries = []
for path in sorted(glob.glob(f'data/boundaryHarmonicTrainingSummary{SUFFIX}*.json')):
    if path.endswith('Pilot.json') or path.endswith('Ceiling2Pilot.json') or path.endswith('Combined.json') or 'EvenSet' in path:
        summaries.append(dict(path=path, content=json.load(open(path))))
summaries.sort(key=lambda summary: not summary['path'].endswith('Combined.json'))          # Combined first: it holds rung 1's sizes and the pilot's


def controlsOf(restart):
    """The summary entry (random codes, long run, no-clamp, stored sweep) of the code space a restart belongs to, or None."""
    for summary in summaries:
        content = summary['content']
        if abs(content['ceiling'] - restart['ceiling']) > 1e-9:
            continue
        if restart['contiguous'] and 'orderSet' not in content and str(restart['orders'][-1]) in content['orders']:
            return summary, content['orders'][str(restart['orders'][-1])]
        if not restart['contiguous'] and content.get('orderSet') == restart['orders']:
            return summary, content['orders'][str(restart['orders'][-1])]
    return None, None


def gateReading(restart):
    summary, entry = controlsOf(restart)
    if entry is None:
        return dict(margin=None, bestRandomCodeScore=None, holds=None, summary=None)
    randomBest = entry['randomCodes']['best']
    return dict(margin=randomBest - restart['score'], bestRandomCodeScore=randomBest, holds=bool(randomBest - restart['score'] >= 5.0), summary=summary['path'])


for restart in formed:
    restart['gate2'] = gateReading(restart)
qualifying = [r for r in formed if r['gate2']['holds']]
primary = best(qualifying)
routeOf = lambda r: f"ceiling {r['ceiling']:g}, {'contiguous' if r['contiguous'] else 'even-only'}"
bestPerRoute = {}
for r in qualifying:
    if routeOf(r) not in bestPerRoute or r['score'] < bestPerRoute[routeOf(r)]['score']:
        bestPerRoute[routeOf(r)] = r
describe = lambda r: dict(arm=r['arm'], label=r['label'], restart=r['restart'], path=r['path'], ceiling=r['ceiling'], score=r['score'], overlap=r['overlap'],
                          iteration=r['iteration'], startType=r['startType'], coefficients=r['coefficients'], orders=r['orders'])
if args.selectPrimary:
    print(f'{len(formed)} formed restarts (overlap >= {threshold}); {len(qualifying)} also meet G2 against a computed control')
    print('formed, G2 not yet scorable (no summary of its code space):', sorted({(r['arm'], r['label']) for r in formed if r['gate2']['holds'] is None}))
    print('primary code:', describe(primary) if primary else None)
    for route, r in bestPerRoute.items():
        print('best of route', route, '->', r['path'], f"score {r['score']:.3f}, overlap {r['overlap']:.3f}, best moment {r['iteration']}")
    raise SystemExit(0)

verdicts = {}
# ---------------------------------------------------------------------------------------------------------------------- D1, G1, G2
verdicts['D1-doubleStripesAreReachable'] = dict(
    criterion=criterionOf('D1'), restartsAtOrAboveThreshold=len(formedABC), byArm={arm: len([r for r in formedABC if r['arm'] == arm]) for arm in 'ABC'},
    byRoute={route: len([r for r in formedABC if routeOf(r) == route]) for route in sorted({routeOf(r) for r in formedABC})},
    bestFormedRestart=describe(best(formedABC)) if formedABC else None, holds=bool(formedABC))
verdicts['G1-someRestartIsFormed'] = dict(
    criterion=registration['gate']['criterion'].split('AND G2:')[0].strip(),
    restartsAtOrAboveThreshold=len(formed), byArm={arm: len([r for r in formed if r['arm'] == arm]) for arm in ARMS_ORDER},
    highestOverlapByArmAndCodeSize=highestOverlap, holds=bool(formed))
verdicts['G2-wellBelowRandomCodes'] = dict(
    criterion=registration['gate']['criterion'].split('AND G2:')[1].split('Every arm')[0].strip(),
    formedRestartsMeetingG2=len(qualifying), formedRestartsNotScorable=len([r for r in formed if r['gate2']['holds'] is None]),
    primaryCode=dict(**describe(primary), gate2=primary['gate2']) if primary else None,
    bestCodePerRoute={route: dict(**describe(r), gate2=r['gate2']) for route, r in bestPerRoute.items()},
    holds=bool(qualifying) if formed else None)

# ---------------------------------------------------------------------------------------------------------------------- D2
armAContiguous = [r for r in armsOf('A') if r['contiguous'] and r['orders'][-1] <= 3]
sizesAtThreshold = sorted({r['orders'][-1] for r in armAContiguous if r['overlap'] >= threshold})
verdicts['D2-flanksNeedNoMoreThanTheFace'] = dict(
    criterion=criterionOf('D2'), smallestContiguousSizeAtCeiling1p3=sizesAtThreshold[0] if sizesAtThreshold else None,
    bestOverlapByContiguousSizeAtCeiling1p3={str(size): max(r['overlap'] for r in armAContiguous if r['orders'][-1] == size) for size in sorted({r['orders'][-1] for r in armAContiguous})},
    holds=bool(sizesAtThreshold))

# ---------------------------------------------------------------------------------------------------------------------- D3
d3 = {}
for label, arm in (('pilotAtCeiling1p3', 'A'), ('pilotAtCeiling2p0', 'B')):
    a, b = best(armsOf(arm, 'orders0-2')), best(armsOf(arm, 'order2'))
    d3[label] = dict(evenOnlyBest=a['score'] if a else None, contiguousBest=b['score'] if b else None,
                     difference=(a['score'] - b['score']) if a and b else None, holds=bool(a['score'] <= b['score'] + 1.0) if a and b else None)
verdicts['D3-oddOrdersAddNothing'] = dict(criterion=criterionOf('D3'), arms=d3,
                                          holds=bool(all(v['holds'] for v in d3.values())) if all(v['holds'] is not None for v in d3.values()) else None)

# ---------------------------------------------------------------------------------------------------------------------- D4
bestAtSize = {}
for r in formed:
    key = (r['ceiling'], r['orders'][-1])
    if r['contiguous'] and (key not in bestAtSize or r['score'] < bestAtSize[key]['score']):
        bestAtSize[key] = r
oddCoefficients = {f'ceiling {c:g}, order {n}': {f'a{k}': v for k, v in zip(r['orders'], r['coefficients']) if k % 2 == 1} for (c, n), r in sorted(bestAtSize.items())}
verdicts['D4-bestCodesAreTopBottomSymmetric'] = dict(
    criterion=criterionOf('D4'), oddCoefficientsOfTheBestContiguousCodeAtEachSize=oddCoefficients,
    note='the odd coefficients of an even-only code are zero by construction and are not scored (Amendment 1)',
    holds=bool(all(abs(c) <= 0.10 for coefficients in oddCoefficients.values() for c in coefficients.values())) if oddCoefficients else None)

# ---------------------------------------------------------------------------------------------------------------------- D5
d5 = {}
for label, arm in (('pilotAtCeiling1p3', 'A'), ('pilotAtCeiling2p0', 'B')):
    items = armsOf(arm, 'order3')
    d5[label] = dict(restarts=len(items), atOrAboveThreshold=sum(1 for r in items if r['overlap'] >= threshold),
                     holds=bool(sum(1 for r in items if r['overlap'] >= threshold) >= 10))
verdicts['D5-moreRestartsSucceed'] = dict(criterion=criterionOf('D5'), arms=d5, holds=bool(d5['pilotAtCeiling1p3']['holds']))

# ---------------------------------------------------------------------------------------------------------------------- D6 (the primary code's 20,000-iteration run)
if primary:
    _, entry = controlsOf(primary)
    longRun = entry['longRun']
    verdicts['D6-thePatternPersistsLongerThanTheStripe'] = dict(
        criterion=criterionOf('D6'), primaryCode=describe(primary), longestRunOfIterationsAtOrAboveThreshold=longRun.get('faceShapeLongestRun'),
        totalIterations=longRun['faceShapeIterations'], visits=longRun['faceShapeVisits'], span=longRun['faceShapeSpan'],
        theSingleStripesLongestWas=34, theFacesLongestWas=119, theThresholdOfS6Was=150,
        holds=bool(longRun.get('faceShapeLongestRun') is not None and longRun['faceShapeLongestRun'] > 34))
else:
    verdicts['D6-thePatternPersistsLongerThanTheStripe'] = dict(criterion=criterionOf('D6'), note='not scored: no primary code (no restart is formed and meets G2)', holds=None)
# ---------------------------------------------------------------------------------------------------------------------- C1-C3 (from the summaries; they need no primary code)
if summaries:
    first = summaries[-1]['content'] if not any(s_['path'].endswith('Pilot.json') for s_ in summaries) else next(s_ for s_ in summaries if s_['path'].endswith('Pilot.json'))['content']
    verdicts['C1-randomCodesDoNotMakeDoubleStripes'] = dict(
        criterion=registration['controls']['C1-randomCodesDoNotMakeDoubleStripes'],
        maxOverlapOfTheRandomCodesByCodeSpace={f"{s_['path'].split(SUFFIX)[-1]} size {k}": v['randomCodes'].get('maxStructuralIoU') for s_ in summaries for k, v in s_['content']['orders'].items()},
        holds=bool(all((v['randomCodes'].get('maxStructuralIoU') or 0) < threshold for s_ in summaries for v in s_['content']['orders'].values())))
    verdicts['C2-noClampBaseline'] = dict(criterion=registration['controls']['C2-noClampBaseline'], score=first['free']['score'], iteration=first['free']['iteration'],
                                          overlap=first['free']['parts']['structuralIoU'], descriptive=True)
    if 'bestSweepCode' in first:
        verdicts['C3-storedSweepBaseline'] = dict(criterion=registration['controls']['C3-storedSweepBaseline'], score=first['bestSweepCode']['rerunScore'],
                                                  overlap=first['bestSweepCode']['parts']['structuralIoU'], descriptive=True)
# ---------------------------------------------------------------------------------------------------------------------- D7, D8 (replays of the primary code)
if primary and os.path.exists(args.replayPath):
    replays = json.load(open(args.replayPath))['regimes']
    verdicts['D7-theFieldIsRequired'] = dict(
        criterion=criterionOf('D7'), mostInteriorCellsDarkAtOnceWithTheFieldOff=replays['field off, released']['maxDarkInteriorCells'],
        holds=bool(replays['field off, released']['maxDarkInteriorCells'] == 0))
    verdicts['D8-theReleaseIsNotRequired'] = dict(
        criterion=criterionOf('D8'), highestOverlapWithTheRingHeldThroughout=replays['field on, held throughout']['maxStructuralOverlap'],
        iteration=replays['field on, held throughout']['iteration'], holds=bool(replays['field on, held throughout']['maxStructuralOverlap'] >= threshold))

for key in ('D7-theFieldIsRequired', 'D8-theReleaseIsNotRequired'):
    verdicts.setdefault(key, dict(criterion=criterionOf(key[:2]), note='not scored: needs the replays of a primary code (no restart is formed and meets G2)', holds=None))
for key, verdict in verdicts.items():
    status = 'descriptive' if verdict.get('descriptive') else ('holds' if verdict['holds'] else ('not yet scorable' if verdict['holds'] is None else 'FAILS'))
    print(f'{key}: {status}', {k: v for k, v in verdict.items() if k not in ('criterion', 'holds', 'descriptive', 'note')}, flush=True)

codesSimulated = {}
for r in restarts:
    codesSimulated[r['arm']] = codesSimulated.get(r['arm'], 0) + r['numEvaluations']
result = dict(note='Each criterion as registered and as the note (Amendment 1) reads it; the facts beside each are the numbers behind the verdict.',
              registration=f'data/boundaryHarmonicTrainingPredictions{SUFFIX}.json',
              amendments=[f'data/boundaryHarmonicTrainingPredictionsAmendment1_{SUFFIX}.json'], overlapThreshold=threshold, verdicts=verdicts,
              codesSimulatedByArm=codesSimulated, totalCodesSimulated=int(sum(codesSimulated.values())),
              restartsByArm={arm: len(armsOf(arm)) for arm in ARMS_ORDER}, primaryCode=describe(primary) if primary else None,
              bestCodePerRoute={route: describe(r) for route, r in bestPerRoute.items()})
json.dump(result, open(args.outputPath, 'w'), separators=(',', ':'))
print('wrote', args.outputPath, flush=True)
