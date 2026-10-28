"""Collects everything the order-4 report draws, from the committed order-3 and order-4 result files, into one JSON that
buildBoundaryHarmonicOrder4Report11x11.py turns into the page. Nothing is recomputed from simulations except the criteria of the four
registrations whose per-criterion outcomes were never stored (Layers P1-P6, ModeOwnership P1-P5; Knockout and FieldStrength outcomes are the
recorded ones, see `recordedOutcomes`). EXPLORATORY: this is bookkeeping for the report, not a registered analysis.

Writes data/order4ReportData1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 collectOrder4ReportData11x11.py
"""
import json
import os

import numpy as np

SUFFIX = '1888Hold301FaceMinus60Minus5'
outputPath = f'data/order4ReportData{SUFFIX}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
ORDERS = {3: '', 4: 'Order4'}
load = lambda name, order: json.load(open(f'data/{name}{SUFFIX}{ORDERS[order]}.json'))
report = {}

# ------------------------------------------------------------------------------------------------ patterns and conductance curves
patterns = {}
for order in ORDERS:
    p = json.load(open(f'data/reportPatterns1888Hold301FaceOrder{order}.json'))
    curve = p['conductanceEvery5']
    iterations, interior = np.array(curve['iterations']), np.array(curve['interiorMean'])
    release = p['hold']
    window = (iterations > release) & (iterations <= 1400)
    trough = int(iterations[window][np.argmin(interior[window])])
    afterTrough = (iterations > trough) & (iterations <= 2600)
    peak = int(iterations[afterTrough][np.argmax(interior[afterTrough])])
    afterPeak = (iterations > peak) & (iterations <= 2999)
    laterFall = int(iterations[afterPeak][np.argmin(interior[afterPeak])])
    atHold = float(interior[iterations == 300][0])
    value = lambda it: float(interior[iterations == it][0])
    shapeInOrder = bool(value(trough) < atHold and value(peak) > value(trough) and value(laterFall) < value(peak))      # collapse after the release, trough, second peak, later fall
    collapse = shapeInOrder
    patterns[str(order)] = dict(p, curveLandmarks=dict(trough=trough, secondPeak=peak, laterFall=laterFall, collapseAfterRelease=collapse))
report['patterns'] = patterns

# ------------------------------------------------------------------------------------------------ the criterion-by-criterion table
def claims(registration):
    path = f'data/boundaryHarmonic{registration}Predictions{SUFFIX}.json'
    if not os.path.exists(path):
        return {}
    predictions = json.load(open(path))['predictions']
    items = predictions.items() if isinstance(predictions, dict) else [(x.get('name', ''), x) for x in predictions]
    found = {}
    for name, entry in items:
        identifier = name.replace('-', ' ').split()[0] if name else ''
        text = entry.get('claim') if isinstance(entry, dict) else str(entry)
        found[identifier] = text or (entry.get('criterion') if isinstance(entry, dict) else '')
    return found


def holds(value):
    return value.get('holds') if isinstance(value, dict) else value


rows = []                                                                                  # (registration, id, claim, holds at order 3, holds at order 4)
fromVerdicts = [('Relay', 'Relay: where the effect travels'), ('RingOnlyRelay', 'Ring-only relay'), ('CoarseGrain', 'Coarse-graining the relay'),
                ('WallCounterfactual', 'Wall counterfactuals'), ('CausalStory', 'Causal story'), ('CausalStoryRobustness', 'Causal story, robustness'),
                ('SetPoint', 'Set point'), ('AggregateNucleation', 'Aggregate nucleation')]
for key, title in fromVerdicts:
    verdicts = {o: load(f'boundaryHarmonic{key}', o)['verdicts'] for o in ORDERS}
    claimsOf = claims(key)
    for identifier in verdicts[3]:
        if identifier.startswith('V'):
            continue
        rows.append(dict(registration=title, id=identifier, claim=claimsOf.get(identifier, ''), order3=holds(verdicts[3][identifier]), order4=holds(verdicts[4][identifier])))

# Layers P1-P6 and ModeOwnership P1-P5, computed here from the stored matrices (the original scoring was from printouts)
layers, layerValues = {}, {}
for order in ORDERS:
    d = load('boundaryHarmonicLayers', order)
    response, knockout = d['response'], d['knockoutResponse']
    moment, step = str(d['trainedMoment']), '0.005'
    diagonal = lambda it: response[str(it)][step]['diagonalShare']
    mean = lambda it, st: response[str(it)][st]['meanDiagonalShare']
    seqAmplitude = [mean(300, str(s)) for s in d['steps']]
    seqTime = [mean(it, step) for it in (1, 8, 64, 300)]
    firstBelow = next((it for it in d['probeIterations'] if str(it) in knockout and knockout[str(it)]['meanDiagonalShare'] < 0.5), None)
    residual = lambda it: float(np.mean(response[str(it)][step]['residualShare']))
    layers[order] = dict(P1=all(diagonal(it)[k] > 0.5 for it in (1, 8) for k in range(4)), P2=all(a > b for a, b in zip(seqAmplitude, seqAmplitude[1:])),
                         P3=all(a > b for a, b in zip(seqTime, seqTime[1:])), P4=knockout[moment]['meanDiagonalShare'] < 0.5,
                         P5=firstBelow is not None and firstBelow <= d['featureOnsetIteration'], P6=residual(int(moment)) > residual(1))
    layerValues[order] = dict(P1=[round(x, 3) for x in diagonal(1)], P2=[round(x, 3) for x in seqAmplitude], P3=[round(x, 3) for x in seqTime],
                              P4=round(knockout[moment]['meanDiagonalShare'], 3), P5=[firstBelow, d['featureOnsetIteration']], P6=[round(residual(1), 3), round(residual(int(moment)), 3)])
claimsOf = claims('Layer')
for identifier in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6'):
    rows.append(dict(registration='Layers', id=identifier, claim=claimsOf.get(identifier, ''), order3=layers[3][identifier], order4=layers[4][identifier]))

ownership = {}
for order in ORDERS:
    base, held, off = (json.load(open(f'data/boundaryHarmonicModeOwnership{SUFFIX}{ORDERS[order]}{condition}.json')) for condition in ('Baseline', 'HeldThroughout', 'FieldOff'))
    moment = str(base['trainedMoment'])
    share = lambda d, it: d['moments'][str(it)]['ownedVarianceShare']
    firstBelow = lambda d: next((it for it in sorted(map(int, d['moments'])) if d['moments'][str(it)]['ownedVarianceShare'] < 0.1), None)
    fieldOffFirst, baselineFirst = firstBelow(off), firstBelow(base)
    ownership[order] = dict(P1=share(base, 300) >= 0.3 and share(base, moment) < 0.1, P2=share(held, moment) >= 0.3,
                            P3=share(off, moment) > share(base, moment) and (fieldOffFirst is None or fieldOffFirst > baselineFirst),
                            P4=False, P5=base['moments']['300']['ownedTargetShare'] < 0.5,
                            curve=dict(iterations=sorted(map(int, base['moments'])), baseline=[share(base, i) for i in sorted(map(int, base['moments']))],
                                       fieldOff=[share(off, i) for i in sorted(map(int, off['moments']))], heldThroughout=[share(held, i) for i in sorted(map(int, held['moments']))]),
                            values=dict(P1=[share(base, 300), share(base, moment)], P2=share(held, moment), P3=[share(off, moment), share(base, moment)], P5=base['moments']['300']['ownedTargetShare']))
orders0to6 = json.load(open(f'data/boundaryHarmonicModeOwnership{SUFFIX}Orders0to6.json'))['moments']['300']
assert sorted(set(range(7)) - set(orders0to6['ownersPresent'])) == [0, 2], 'P4 changed'
claimsOf = claims('ModeOwnership')
for identifier in ('P1', 'P2', 'P3', 'P4', 'P5'):
    rows.append(dict(registration='Mode ownership', id=identifier, claim=claimsOf.get(identifier, ''), order3=ownership[3][identifier], order4=ownership[4][identifier]))

recordedOutcomes = dict(
    Knockout=dict(note='all four criteria failed at all three moments at both orders (recorded when scored; the original criteria are in boundaryHarmonicKnockoutPredictions)', ids=['a1', 'a2', 'a3', 'combinations']),
    FieldStrength=dict(note='Q1 and Q2 fail at both orders, Q3 holds at both (recorded when scored)', outcomes={'Q1': False, 'Q2': False, 'Q3': True}))
claimsOf = claims('Knockout')
for identifier in recordedOutcomes['Knockout']['ids']:
    rows.append(dict(registration='Knockout', id=identifier, claim=claimsOf.get(identifier, ''), order3=False, order4=False, recorded=True))
claimsOf = claims('FieldStrength')
for identifier, outcome in recordedOutcomes['FieldStrength']['outcomes'].items():
    rows.append(dict(registration='Field strength', id=identifier, claim=claimsOf.get(identifier, ''), order3=outcome, order4=outcome, recorded=True))

confirmation = {o: json.load(open(f'data/relayLoopModulesConfirmation{SUFFIX}{ORDERS[o]}.json')) for o in ORDERS}
claimsOf = claims('RelayModulesConfirmation')
for criterion3, criterion4 in zip(confirmation[3]['criteria'], confirmation[4]['criteria']):
    assert criterion3['name'] == criterion4['name']
    rows.append(dict(registration='Steering confirmation', id=criterion3['name'].split('-')[0], claim=criterion3['claim'], order3=criterion3['passed'], order4=criterion4['passed']))
for row in rows:
    row['agree'] = row['order3'] == row['order4']
report['criteria'] = rows
report['agreement'] = dict(agree=sum(r['agree'] for r in rows), total=len(rows))
report['layers'] = dict(values=layerValues)
report['ownership'] = ownership
report['confirmation'] = {str(o): dict(cells=confirmation[o]['cellDescription']['cells'], criteria=confirmation[o]['criteria']) for o in ORDERS}

# ------------------------------------------------------------------------------------------------ second finding: the fifth term
knockouts = {}
for order in ORDERS:
    d = load('boundaryHarmonicKnockout', order)
    knockouts[str(order)] = dict(trainedScore=d['trained']['ownBest']['score'], trainedMoment=d['trainedMoment'],
                                 single=[dict(order=e['orders'][0], ringChangeRMS=e['ringChangeRMS'], scoreAtTrainedMoment=e['knockout']['trainedMoment']['score'],
                                              vmem=e['knockout']['trainedMoment']['vmem'], controlScoreMedian=float(np.median(e['controlScores'])) if e['controlScores'] else None)
                                         for e in d['knockouts'] if len(e['orders']) == 1])
report['knockouts'] = knockouts
report['facesInSweep'] = {str(o): {k: v for k, v in load('relayLoopOrderEdgeMap', o)['outcomes']['facesInSweep'].items() if k != 'multipliersOfFaces'} for o in ORDERS}
report['jitter'] = {str(o): load('boundaryHarmonicCodeJitter', o)['summary'] for o in ORDERS}

# ------------------------------------------------------------------------------------------------ appendix data
wall = {}
for order in ORDERS:
    results = load('boundaryHarmonicCausalStory', order)['results']
    wall[str(order)] = dict(whole=results['fullRing']['selectivityDifference'], upper=results['upperWall']['selectivityDifference'], lower=results['lowerWall']['selectivityDifference'],
                            overlap=dict(whole=results['fullRing']['overlapAtScored'], upper=results['upperWall']['overlapAtScored'], lower=results['lowerWall']['overlapAtScored']),
                            dark=dict(upper=results['upperWall']['darkAtScored'], lower=results['lowerWall']['darkAtScored'], baseline=results['baseline']['darkAtScored']),
                            release={k[len('fullRingReleasedAt'):]: v['selectivityDifference'] for k, v in results.items() if k.startswith('fullRingReleasedAt')})
report['wall'] = wall
dial = {}
for order in ORDERS:
    d = json.load(open(f'data/boundaryHarmonicRingCodePatterns{SUFFIX}{ORDERS[order]}.json'))
    wide = d['families']['wideSlider']
    pick = [i for i, o in enumerate(wide['order']) if o == 0]
    dial[str(order)] = dict(setting=[wide['coefficients'][i][0] for i in pick], overlap=[wide['overlapAtRead'][i] for i in pick])
report['dial'] = dial
ensemble = {str(o): json.load(open(f'data/canalizationTalkPatternEnsemble1888Hold301{"Order4" if o == 4 else ""}.json'))['face'] for o in ORDERS}
report['compression'] = {o: dict(spaceFilling=e['spaceFilling'], all=e['all']) for o, e in ensemble.items()}
visits = {str(o): json.load(open(f'data/canalizationTalkFamilyVisits1888Hold301{"Order4" if o == 4 else ""}.json')) for o in ORDERS}
report['imprint'] = {o: dict(edges=v['timeCourse']['edges'], trained=v['timeCourse']['face']['faceClass']['trained'], copies=v['timeCourse']['face']['faceClass']['copiesMean'],
                             controls=v['timeCourse']['face']['faceClass']['controlsMean'], controlsMax=v['timeCourse']['face']['faceClass']['controlsMax'],
                             inSample=v['family']['faceTrained']['face_inSample']) for o, v in visits.items()}
edgeMaps = {str(o): {m['edge']: m for m in load('relayLoopOrderEdgeMap', o)['orderMap']} for o in ORDERS}
regionNames = ['top', 'upper sides', 'lower sides', 'bottom']
regionEffects = []
for edge, regions in (('write ringBottom->bgBL', (3, 2)), ('write ringLeft->bgBL', (3, 2)), ('flood ringTop->bgTL', (0,))):
    for r in regions:
        a, b = edgeMaps['3'][edge], edgeMaps['4'][edge]
        regionEffects.append(dict(edge=edge, region=regionNames[r], order3=dict(effect=a['regionLogitPerSd'][r], stability=a['regionSignStability'][r]),
                                  order4=dict(effect=b['regionLogitPerSd'][r], stability=b['regionSignStability'][r])))
report['regionEffects'] = regionEffects
report['learnableEdges'] = {str(o): [load('relayLoopOrderEdgeMap', o)['learnableEdges'], len(load('relayLoopOrderEdgeMap', o)['edgeScores'])] for o in ORDERS}
report['scores'] = dict(canalization=json.load(open(f'data/boundaryHarmonicOrder4CanalizationMemoryScores{SUFFIX}.json')), steering=json.load(open(f'data/boundaryHarmonicOrder4SteeringScores{SUFFIX}.json')))
json.dump(report, open(outputPath, 'w'), separators=(',', ':'))
print(f'wrote {outputPath}: {report["agreement"]["agree"]} of {report["agreement"]["total"]} criteria agree; landmarks',
      {o: patterns[o]['curveLandmarks'] for o in patterns})
