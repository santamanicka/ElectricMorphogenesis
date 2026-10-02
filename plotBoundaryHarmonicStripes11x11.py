"""Build the stripes report ("The Stripes", PolyPatterning pattern 6, the French flag) from its template and the analysis JSON.

The report follows the face's three (training, latching switch, relay loop) as one page: patterns first, the main findings, a
storyline, appendices, and a Methods section that defines every term. It is rebuilt from the committed data/ files:

    python3 plotBoundaryHarmonicStripes11x11.py [--overwrite]

Inputs. --summaryPath: the training summary (analyzeBoundaryHarmonicTraining11x11.py) the patterns and scores are read from.
The ladder figure reads every stripe training folder directly (every restart's score, overlap at its best moment, code and
start), so each rung that has been run appears whether or not a summary was made of it. A section appears only when its data
file exists, as with the switch-rule builder's include flags.

The Relay Loop sections (the relay network with its slider and grid, the steering lab) and the pattern sliders are spliced in from the fragments
figures/boundaryHarmonicStripesRelayLoop.{css,html,js} and figures/boundaryHarmonicStripesPatternSlider.{html,js}, each when its data exists:
data/relayLoopStripesPageData<suffix>.json (assembleRelayLoopStripesData11x11.py) with the newest data/relayLoopSteeringData<suffix>Codes<n>.json
(assembleRelayLoopStripesSteeringData11x11.py) for the loop, and data/boundaryHarmonicRingCodePatterns<suffix>.json
(simulateBoundaryHarmonicRingCodePatterns11x11.py) for the pattern sliders.

Refuses to overwrite an existing page unless --overwrite is given.
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
parser.add_argument('--summaryPath', type=str, default=f'data/boundaryHarmonicTrainingSummary{SUFFIX}Ceiling2Combined.json',
                    help='the summary the patterns and scores are read from: the ceiling-2 runs (pilot and full run), where the two-harmonic stripe is found')
parser.add_argument('--faceCeilingSummaryPath', type=str, default=f'data/boundaryHarmonicTrainingSummary{SUFFIX}Ceiling1p3Combined.json',
                    help='the same at the face\'s ceiling of 1.3, drawn as a second row of the gallery for contrast')
parser.add_argument('--templatePath', type=str, default='figures/boundaryHarmonicStripesTemplate.html')
parser.add_argument('--outputPath', type=str, default='figures/boundaryHarmonicStripes.html')
parser.add_argument('--relayMoviePath', type=str, default=f'data/boundaryHarmonicRelayVariantMovie_trained{SUFFIX}.json',
                    help='buildBoundaryHarmonicRelayVariantMovie11x11.py\'s output for the stripe code (the relay section appears when it exists)')
parser.add_argument('--loopPath', type=str, default=f'data/relayLoopStripesPageData{SUFFIX}.json', help='assembleRelayLoopStripesData11x11.py\'s output (the Relay Loop sections appear when it and a steering file exist)')
parser.add_argument('--steeringPath', type=str, default=None, help='assembleRelayLoopStripesSteeringData11x11.py\'s output; default is the newest data/relayLoopSteeringData<suffix>Codes<n>.json')
parser.add_argument('--orderEdgeMapPath', type=str, default=f'data/relayLoopStripesOrderEdgeMap{SUFFIX}.json')
parser.add_argument('--patternsPath', type=str, default=f'data/boundaryHarmonicRingCodePatterns{SUFFIX}.json', help='simulateBoundaryHarmonicRingCodePatterns11x11.py\'s output (the pattern sliders appear when it exists)')
parser.add_argument('--facePatternsPath', type=str, default='data/boundaryHarmonicRingCodePatterns1888Hold301FaceMinus60Minus5.json', help='the same on the face code, for comparison')
parser.add_argument('--smoothnessPath', type=str, default='data/boundaryHarmonicRingCodePatternSmoothness1888Hold301StripesAndFace.json', help='analyzeBoundaryHarmonicRingCodePatternSmoothness11x11.py\'s output (the comparison table appears with the pattern sliders when it exists)')
parser.add_argument('--restartRecordsPath', type=str, default=f'data/boundaryHarmonicTrainingRestartRecords{SUFFIX}.json', help='extractBoundaryHarmonicTrainingRestartRecords11x11.py\'s output: the ladder is read from it for any folder whose raw checkpoints are absent')
parser.add_argument('--useRestartRecords', action='store_true', help='read the ladder from the restart records even where the raw checkpoints exist (to check the two agree)')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
if os.path.exists(args.outputPath) and not args.overwrite:
    raise SystemExit(f'{args.outputPath} exists; pass --overwrite to rebuild it')

rounded = lambda values, digits=3: np.round(np.asarray(values, dtype=float), digits).tolist()

# ------------------------------------------------------------------------------------------------ the training summary
summary = json.load(open(args.summaryPath))
data = dict(ceiling=summary['ceiling'], hold=summary['hold'], numIterations=summary['numIterations'], target=summary['target'],
            targetCells=summary['featureCells'], free=summary['free'], bestSweepCode=summary.get('bestSweepCode'),
            orders={}, trainingDirs=summary['trainingDirs'])
for order, entry in summary['orders'].items():
    longRun = entry['longRun']
    data['orders'][order] = dict(
        best=entry['best'], random=dict(median=entry['randomCodes']['median'], best=entry['randomCodes']['best'],
                                        maxOverlap=entry['randomCodes'].get('maxStructuralIoU')),
        neighboursMedian=entry['neighbours']['median'],
        longRun=dict(traceStride=longRun['traceStride'], overlapTrace=longRun['overlapTrace'], bestIteration=longRun['bestIteration'],
                     shapeIterations=longRun['faceShapeIterations'], longestRun=longRun.get('faceShapeLongestRun'),
                     snapshots=longRun['snapshots']))
data['subsetArms'] = summary.get('subsetArms', {})
faceCeiling = json.load(open(args.faceCeilingSummaryPath)) if os.path.exists(args.faceCeilingSummaryPath) else None
if faceCeiling:
    data['ordersAtFaceCeiling'] = dict(ceiling=faceCeiling['ceiling'], orders={order: dict(best=entry['best']) for order, entry in faceCeiling['orders'].items()})

# ------------------------------------------------------------------------------------ every restart of every training folder
RUNGS = [   # folder suffix after the target name -> (rung, label)
    ('', ('1', 'rung 1 · pilot · ceiling 1.3 · population 16')),
    ('Population64', ('2', 'rung 2 · ceiling 1.3 · population 64')),
    ('EvenOrdersPopulation64', ('2′', 'rung 2′ (exploratory) · even orders · ceiling 1.3 · population 64')),
    ('Ceiling2', ('3', 'rung 3 · pilot · ceiling 2.0 · population 16')),
    ('Ceiling2Population64', ('3b', 'rung 3b · ceiling 2.0 · population 64')),
    ('Ceiling2HigherOrdersPopulation16', ('4′', 'rung 4′ (exploratory) · ceiling 2.0 · population 16')),
    ('Ceiling2EvenOrdersPopulation32', ('4″', 'rung 4″ (exploratory) · even orders · ceiling 2.0 · population 32')),
    ('Ceiling2HigherOrders', ('4', 'rung 4 · ceiling 2.0 · population 64')),
]
stem = f'data/boundaryHarmonicTraining{SUFFIX}'
restartRecords = json.load(open(args.restartRecordsPath))['folders'] if os.path.exists(args.restartRecordsPath) else {}


def restartsOf(suffix):
    """Every restart of one rung's folder: from the raw checkpoints when they exist (and --useRestartRecords is not given), else from the scalar records."""
    paths = [] if args.useRestartRecords else sorted(glob.glob(f'{stem}{suffix}/*_restart*.npz'))
    if paths:
        for path in paths:
            run = np.load(path)
            yield dict(arm=re.sub(r'_restart.*', '', os.path.basename(path)), orders=[int(o) for o in run['orders']] if 'orders' in run else list(range(int(run['maxOrder']) + 1)),
                       populationSize=int(run['populationSize']), ceiling=float(run['ceiling']), restart=int(run['restart']), startType=str(run['startType']),
                       bestScore=float(run['bestScore']), bestIteration=int(run['bestIteration']), bestCoefficients=np.asarray(run['bestCoefficients']), bestVmem=run['bestVmem'])
    else:
        for record in restartRecords.get(suffix, []):
            yield dict(record, bestCoefficients=np.asarray(record['bestCoefficients']), bestVmem=np.asarray(record['bestVmem'], dtype=np.float32))


ladder = []
for suffix, (rung, label) in RUNGS:
    folder = f'{stem}{suffix}'
    arms = {}
    for run in restartsOf(suffix):
        arm, orders = run['arm'], run['orders']
        overlap = boundary.structuralIntersectionOverUnion(run['bestVmem'].astype(float), boundary.centreStripeCellIndices)
        arms.setdefault(arm, dict(arm=arm, orders=orders, population=run['populationSize'], ceiling=run['ceiling'], restarts=[]))
        arms[arm]['restarts'].append(dict(restart=run['restart'], start=run['startType'], score=round(run['bestScore'], 3),
                                          overlap=round(overlap, 4), iteration=run['bestIteration'],
                                          coefficients=rounded(run['bestCoefficients'], 3)))
        # the arm's showcase restart: the highest overlap, then the lowest score; its pattern at its best moment and its ring code
        showcase = arms[arm].get('showcase')
        if showcase is None or (overlap, -run['bestScore']) > (showcase['overlap'], -showcase['score']):
            ring = np.clip(np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), orders)) @ run['bestCoefficients'], 0, run['ceiling'])
            arms[arm]['showcase'] = dict(overlap=overlap, score=run['bestScore'], restart=run['restart'],
                                         vmem=np.round(run['bestVmem'], 0).astype(int).tolist(), ringValues=rounded(ring, 2))
    if arms:
        ladder.append(dict(rung=rung, label=label, folder=os.path.basename(folder), arms=sorted(arms.values(), key=lambda a: (a['orders'][-1], len(a['orders'])))))
data['ladder'] = ladder

# ------------------------------------------------------------------------- sections that appear when their data exist
def loadIfExists(path):
    return json.load(open(path)) if os.path.exists(path) else None


replays = loadIfExists(f'data/boundaryHarmonicFieldRole{SUFFIX}.json')
if replays:
    data['replays'] = dict(code=replays['code'], regimes={name: dict(score=r['score'], iteration=r['iteration'], vmem=r['vmem'], featureCoverage=r['featureCoverage'],
                                                                      spuriousDark=r['spuriousDark'], maxDarkInteriorCells=r.get('maxDarkInteriorCells'),
                                                                      maxStructuralOverlap=r.get('maxStructuralOverlap')) for name, r in replays['regimes'].items()})

segments = loadIfExists(f'data/boundaryHarmonicStripeRingSegments{SUFFIX}Order2Restart06Ceiling2WithPatterns.json')
if segments:
    data['segments'] = dict(
        coefficients=segments['coefficients'], ringValues=rounded(segments['ringValues'], 3), scoredIteration=segments['scoredIteration'], ring=segments['ring'], sets=segments['sets'],
        outcomes={k: v for k, v in segments['outcomes'].items() if k in segments['sets'] or k in ('fullRing', 'topAndBottomExcluded', 'leftAndRightExcluded')},
        baseline=segments['baseline'], patterns=segments['patterns'], verdicts=segments['verdicts'],
        controls={k: dict(size=v['size'], median=v['median'], percentile95=v['percentile95'], setDifference=v['setDifference'], setExceedsShare=v['setExceedsShare'])
                  for k, v in segments['randomSubsetControls'].items()},
        halfArcs=[dict(offset=h['offset'], held=round(h['held'], 4), complement=round(h['complement'], 4)) for h in segments['halfArcs']],
        singleCells={k: round(v, 4) for k, v in segments['singleCells'].items()})


def landscapeOf(path):
    """One (a0, a2) or (T, S) map: the grid, the overlap at each code's best moment and its balanced score, flat, rows over the first axis."""
    landscape = loadIfExists(path)
    if not landscape:
        return None
    return dict(levels=bool(landscape.get('gridIsLevels')), first=landscape['a0'], second=landscape['a2'], ceiling=landscape['ceiling'],
                overlap=[round(v, 2) for v in landscape['overlapAtBest']], maxOverlap=[round(v, 2) for v in landscape['maxOverlap']],
                score=[round(v, 1) for v in landscape['score']], bestIteration=landscape['bestIteration'], ringMax=[round(v, 2) for v in landscape['ringMax']])


relayMovie = loadIfExists(args.relayMoviePath)
if relayMovie:
    data['relay'] = dict(movie=relayMovie['movie'], resolutions=relayMovie['resolutions'], difference=relayMovie['difference'],
                         ringCells=[int(c) for c in boundary.boundaryRingCells])

program = loadIfExists(f'data/boundaryHarmonicStripeProgram{SUFFIX}Ceiling2Pilot.json')
if program:
    data['program'] = {key: program[key] for key in ('hold', 'bestMoment', 'numIterations', 'landmarks', 'stripeFormedIterations', 'times', 'interiorMean', 'stripeMean', 'flankMean',
                                                    'darkCount', 'stripeDarkCount', 'overlap', 'snapshots', 'verdicts')}
relayScore = loadIfExists(f'data/boundaryHarmonicStripeRelay{SUFFIX}.json')
if relayScore:
    data['relayScore'] = {key: relayScore[key] for key in ('pairs', 'field', 'contact', 'phases', 'primaryIteration', 'gap', 'nodeOrder')}
    data['relayScore']['verdicts'] = {k: {kk: vv for kk, vv in v.items()} for k, v in relayScore['verdicts'].items()}

endsConfirmation = loadIfExists(f'data/boundaryHarmonicStripeEndsConfirmationScoring{SUFFIX}.json')

# ---- the second route: the ceiling-1.3 codes of the even-only arm of every order up to 20 that form the stripe (exploratory), against the order-2 code
lateRoute = loadIfExists(f'data/boundaryHarmonicStripeLateRoute{SUFFIX}Ceiling1p3.json')
if lateRoute:
    lateCodes = [c for c in lateRoute['codes'] if c['kind'] == 'lateRoute']
    comparison = next(c for c in lateRoute['codes'] if c['kind'] == 'comparison')
    bestLate = min(lateCodes, key=lambda c: c['bestScore'])
    until = 3000 // lateRoute['traceStride']
    trace = lambda c: dict(label=c['label'], interiorMean=c['interiorMeanTrace'][:until], overlap=c['overlapTrace'][:until], bestMoment=c['bestMoment'], trough=c['trough'])
    data['lateRoute'] = dict(traceStride=lateRoute['traceStride'], hold=lateRoute['hold'], numIterations=lateRoute['numIterations'], verdicts=lateRoute['verdicts'],
                             codes=[{k: v for k, v in c.items() if k not in ('interiorMeanTrace', 'overlapTrace')} for c in lateRoute['codes']],
                             traces=dict(late=trace(bestLate), early=trace(comparison)))

def trainingEvidence(v, shortKeys):
    """One readable line per training prediction, from the numbers the scoring wrote."""
    get = lambda key: v[shortKeys[key]]
    s1, s2, s3, s4, s5, s6, s7, s8 = (get(k) for k in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'))
    code = s1['bestCode']
    oddText = '; '.join('order %s: %s' % (size, ', '.join('%s = %.3f' % (name, value) for name, value in coefficients.items())) for size, coefficients in s4['oddCoefficientsOfTheBestCodeAtEachSize'].items())
    return dict(
        S1='%d restarts at overlap 0.9 or more, first in rung %s; best code: %s restart %d, %.2f mV, overlap %.3f, best moment %d' % (
            s1['restartsAtOrAboveThreshold'], s1['firstRungs'][0], code['arm'], code['restart'], code['score'], code['overlap'], code['iteration']),
        S2='smallest contiguous size at overlap 0.9 or more: none at ceiling 1.3, %s at ceiling 2.0; the registered reading records a stripe found only at 2.0 as a failure' % s2['smallestContiguousSizeAtCeiling2p0'],
        S3='even-only best against contiguous best, mV: ' + '; '.join('%s %.2f against %.2f' % (name, arm['evenOnlyBest'], arm['contiguousBest']) for name, arm in s3['arms'].items()),
        S4='odd coefficients of the best formed code of each size: ' + oddText,
        S5='order-3 restarts at overlap 0.9 or more: ' + '; '.join('%s %d of %d' % (name, arm['atOrAboveThreshold'], arm['restarts']) for name, arm in s5['arms'].items()),
        S6='longest run at overlap 0.9 or more: %d iterations in %d visit (%d to %d); the face’s longest was %d' % (s6['longestRunOfIterationsAtOrAboveThreshold'], s6['visits'], s6['span'][0], s6['span'][1], s6['thefacesLongestWas']),
        S7='most interior cells dark at once with the field off: %d' % s7['mostInteriorCellsDarkAtOnceWithTheFieldOff'],
        S8='highest overlap with the ring held throughout: %.2f, at iteration %d' % (s8['highestOverlapWithTheRingHeldThroughout'], s8['iteration']))


# ---- the registered predictions with their verdicts, verbatim, from the files each analysis wrote
def verdictRows():
    rows = []
    def add(group, registration, verdicts, evidence):
        for prediction in registration['predictions']:
            key = prediction['name'].split('-')[0]
            if key not in verdicts and prediction['name'] in verdicts:
                verdicts = {**verdicts, key: verdicts[prediction['name']]}
            if key in verdicts:
                rows.append(dict(group=group, name=prediction['name'], claim=prediction['claim'], criterion=prediction['criterion'], basis=prediction.get('basis', ''),
                                 verdict='holds' if verdicts[key]['holds'] else ('not applicable' if verdicts[key]['holds'] is None else 'fails'),
                                 evidence=evidence.get(key, '')))
    mechanism = json.load(open(f'data/boundaryHarmonicStripeMechanismPredictions{SUFFIX}.json'))
    if program:
        v = program['verdicts']
        add('The program', mechanism, v, dict(
            P1='stripe code %.4f; best code of orders 0, 1, 3: %s' % (v['P1']['stripeCode'], ', '.join('%.4f' % v['P1']['accuracyByOrder'][o] for o in ('0', '1', '3'))),
            P2='best moment %d, trough %d (first peak %d, second peak %d)' % (v['P2']['bestMoment'], v['P2']['trough'], v['P2']['firstPeak'], v['P2']['secondPeak']),
            P3='selectivity %.3f G_ref' % v['P3']['selectivity'],
            P4='%d cells dark at 300 and light at 504, %d of them flank cells' % (v['P4']['darkAtHoldEndLightAtBest'], v['P4']['ofWhichFlank']),
            P5='%d of %d stripe cells dark at iteration 300' % (v['P5']['stripeCellsDarkAtHoldEnd'], v['P5']['of'])))
    if relayScore:
        v = relayScore['verdicts']
        add('The relay', mechanism, v, dict(
            V1='closure %.1e, flux conserved to %.1e' % (v['V1']['closureWorst'], max(v['V1']['conservation'].values())),
            V3='left-right asymmetry %.1e of the largest transfer' % v['V3']['relativeAsymmetry'],
            R1='the ring’s held conductance is %.1f%% of the %.3f injected; the clamp’s voltage update the rest' % (100 * v['R1']['share'], v['R1']['total']),
            R2='the field carries %.1f%% of the gross transfer' % (100 * v['R2']['fieldShare']),
            R3='largest field transfer: flood %s → %s (%.3f), clear %s → %s (%.3f)' % (*v['R3']['largestFieldTransfer']['flood']['pair'], v['R3']['largestFieldTransfer']['flood']['value'],
                                                                                     *v['R3']['largestFieldTransfer']['clear']['pair'], v['R3']['largestFieldTransfer']['clear']['value']),
            R4='ring left %.4f against ring top %.4f: ratio %.3f' % (v['R4']['ringLeft'], v['R4']['ringTop'], v['R4']['ratio'])))
    if segments:
        v = segments['verdicts']
        add('The ring segments', mechanism, v, dict(
            V2='replay score %.4f mV at iteration %d (stored %.4f, %d)' % (v['V2']['replayScore'], v['V2']['replayIteration'], v['V2']['storedScore'], v['V2']['storedIteration']),
            W1='the six stripe-facing cells give %.0f%% of the whole ring’s selectivity difference' % (100 * v['W1']['stripeFacingShareOfFull']),
            W2='top and bottom rows %.0f%%, left and right columns %.0f%%' % (100 * v['W2']['topAndBottomShare'], 100 * v['W2']['leftAndRightShare']),
            W3='six cells %.3f against the 95th percentile %.3f; top and bottom rows %.3f against %.3f' % (v['W3']['stripeFacing']['difference'], v['W3']['stripeFacing']['percentile95'],
                                                                                                         v['W3']['topAndBottom']['difference'], v['W3']['topAndBottom']['percentile95']),
            W4='overlap: top row %.2f, bottom row %.2f, both %.2f' % (v['W4']['topRowOverlap'], v['W4']['bottomRowOverlap'], v['W4']['topAndBottomOverlap']),
            W5='the halves sum to %.3f against %.3f: %+.0f%%' % (v['W5']['sumOfHalves'], v['W5']['full'], 100 * v['W5']['relativeError'])))
    if endsConfirmation:
        v = endsConfirmation['verdicts']
        ends = json.load(open(f'data/boundaryHarmonicStripeEndsPredictions{SUFFIX}.json'))
        rate = lambda x: '%d/%d = %.3f [%.3f, %.3f]' % (x['count'], x['of'], x['rate'], *x['interval95'])
        add('The two ends', ends, v, dict(
            E1='upper half formed, T_top inside W %s; outside %s' % (rate(v['E1-theTopWritesTheUpperHalf']['insideWindow']), rate(v['E1-theTopWritesTheUpperHalf']['outsideWindow'])),
            E2='lower half formed, T_bottom inside W %s; outside %s' % (rate(v['E2-theBottomWritesTheLowerHalf']['insideWindow']), rate(v['E2-theBottomWritesTheLowerHalf']['outsideWindow'])),
            E3='top only: upper %s, lower %s; bottom only: lower %s, upper %s' % (rate(v['E3-theEndsAreWrittenWithoutEachOther']['topOnlyUpperFormed']), rate(v['E3-theEndsAreWrittenWithoutEachOther']['topOnlyLowerFormed']),
                                                                                  rate(v['E3-theEndsAreWrittenWithoutEachOther']['bottomOnlyLowerFormed']), rate(v['E3-theEndsAreWrittenWithoutEachOther']['bottomOnlyUpperFormed'])),
            E4='stripe formed: both %s; the other three strata %s' % (rate(v['E4-theStripeNeedsBothEnds']['bothStratum']), rate(v['E4-theStripeNeedsBothEnds']['otherStrata'])),
            E5='both levels above 1.4975: %s' % rate(v['E5-theWindowIsThin']['neitherWithBothLevelsAbove1p4975'])))
    if lateRoute:
        v = lateRoute['verdicts']
        span = lambda values: '%d' % min(values) if min(values) == max(values) else '%d to %d' % (min(values), max(values))
        add('The second route, at the face’s ceiling (exploratory)', json.load(open(f'data/boundaryHarmonicStripeLateRoutePredictions{SUFFIX}.json')),
            {k: v[k] for k in v}, dict(
            L1='best moments %s, troughs %s: after the trough in %d of %d' % (span(v['L1-readAfterTheTrough']['bestMoments']), span(v['L1-readAfterTheTrough']['troughs']),
                                                                              v['L1-readAfterTheTrough']['count'], v['L1-readAfterTheTrough']['of']),
            L2='stripe cells dark at iteration 300: %s of 27; at most 13 in %d of %d codes' % (span(v['L2-notWrittenInTheHold']['darkAt300']), v['L2-notWrittenInTheHold']['count'], v['L2-notWrittenInTheHold']['of']),
            L3='second peaks at %s; within 300 iterations of the best moment in %d of %d' % (span(v['L3-onTheSecondRise']['secondPeaks']), v['L3-onTheSecondRise']['count'], v['L3-onTheSecondRise']['of']),
            L4='longest run at overlap 0.85 or more: %s iterations' % span(v['L4-brief']['longestRuns']),
            L5='residual of the {0, 2, 4, 6} fit: %.2f to %.2f G_ref' % (min(v['L5-notALowOrderCode']['residuals']), max(v['L5-notALowOrderCode']['residuals']))))
    trainingScoring = loadIfExists(f'data/boundaryHarmonicTrainingPredictionScoring{SUFFIX}.json')
    if trainingScoring:
        training = json.load(open(f'data/boundaryHarmonicTrainingPredictions{SUFFIX}.json'))
        v = trainingScoring['verdicts']
        shortKeys = {k.split('-')[0]: k for k in v}
        add('The training', training, {k: v[full] for k, full in shortKeys.items() if k.startswith('S')},
            trainingEvidence(v, shortKeys))
        if 'X1' in shortKeys:
            amendment6 = json.load(open(f'data/boundaryHarmonicTrainingPredictionsAmendment6_{SUFFIX}.json'))
            x1, x2, x3 = (v[shortKeys[k]] for k in ('X1', 'X2', 'X3'))
            add('The face’s ceiling, even-only arms (exploratory)', amendment6, {k: v[full] for k, full in shortKeys.items() if k.startswith('X')}, dict(
                X1='highest overlap by arm: %s; %d restarts at 0.9 or more' % (', '.join('%s %.3f' % ('even 0–' + arm.split('-')[-1] if arm != 'orders0-2-4-6' else 'even 0–6', value) for arm, value in x1['maxOverlapByArm'].items()),
                                                                                  len(x1['restartsAtOrAboveThreshold'])),
                X2='%d of %d restarts at 0.8 or more (library starts %d of %d, random starts %d of %d)' % (x2['atOrAbove0p8'], x2['restartsOfTheArm'], x2['byStartType']['library']['atOrAbove0p8'],
                                                                                                      x2['byStartType']['library']['restarts'], x2['byStartType']['random']['atOrAbove0p8'], x2['byStartType']['random']['restarts']),
                X3='%d of %d restarts at 0.8 or more have their best moment after 1130 (earliest %d)' % (x3['bestMomentAfter1130'], x3['restartsAtOrAbove0p8'], min(x3['bestMoments']))))
    return rows


data['predictions'] = verdictRows()

edgeScan = loadIfExists(f'data/boundaryHarmonicStripeEdgeScan{SUFFIX}Order2Restart06Ceiling2'.replace('.', 'p') + '.json')
if edgeScan:
    data['edgeScan'] = dict(delta=edgeScan['delta'], epsilon=edgeScan['epsilon'], overlap=[round(v, 2) for v in edgeScan['overlapAtBest']], maxOverlap=[round(v, 2) for v in edgeScan['maxOverlap']],
                            score=[round(v, 1) for v in edgeScan['score']], crossingDelta=edgeScan['crossingDelta'], endCellValues=[round(v, 4) for v in edgeScan['endCellValues']])

# ---- the stripe's two ends: the exploratory scan, four illustrative relays, and the registered confirmatory draw
import relayLoopNets
endsScan = loadIfExists(f'data/boundaryHarmonicStripeEndsScan{SUFFIX}Side1p2127.json')
endsConfirmation = loadIfExists(f'data/boundaryHarmonicStripeEndsConfirmationScoring{SUFFIX}.json')
if endsScan:
    layout = relayLoopNets.layoutFor('stripesInterior')
    pairIndex = {tuple(pair): k for k, pair in enumerate(layout.PAIRS)}
    variants = []
    for key in ('endsBoth', 'endsTopOnly', 'endsBottomOnly', 'endsNeither'):
        record = loadIfExists(f'data/relayLoopStripeEnds/{key}.json')
        if record:
            clearField = record['field'][1]
            variants.append(dict(key=key, levels=record['multipliers'], scoredVmem=record['scoredVmem'], upperDark=record['upperDark'], lowerDark=record['lowerDark'], gap=record['gap'],
                                 overlap=record['faceOverlap'], topPush=clearField[pairIndex[('ringTop', 'stripeUpper')]], bottomPush=clearField[pairIndex[('ringBottom', 'stripeLower')]]))
    exceptions = []
    drawPath = f'data/boundaryHarmonicStripeEndsConfirmation{SUFFIX}.json'
    if endsConfirmation and os.path.exists(drawPath):      # codes of the registered draw that formed the stripe with an end outside the window
        draw = json.load(open(drawPath))
        for k, label in enumerate(draw['labels']):
            if label not in ('both', 'stripeCode') and draw['overlapAtBest'][k] >= 0.9:
                exceptions.append(dict(stratum=label, top=draw['topLevel'][k], bottom=draw['bottomLevel'][k], overlap=draw['overlapAtBest'][k]))
    data['ends'] = dict(levels=endsScan['levels'], side=endsScan['sideLevel'], upperDark=endsScan['upperDarkAtBest'], lowerDark=endsScan['lowerDarkAtBest'], overlap=[round(v, 2) for v in endsScan['overlapAtBest']],
                        variants=variants, confirmation=endsConfirmation, exceptions=exceptions)

landscapeStem = f'data/boundaryHarmonicStripeLandscape{SUFFIX}Order2Ceiling2'
data['landscapes'] = {key: landscapeOf(f'{landscapeStem}{suffix}.json') for key, suffix in
                      (('coarse', ''), ('zoomPerfect', 'ZoomAroundA1p35A2p137'), ('zoomRidge', 'ZoomAroundA0p949A2p558'), ('levels', 'LevelsTopBottomVersusSides'))}
data['landscapes'] = {k: v for k, v in data['landscapes'].items() if v}
familiesPath = f'data/boundaryHarmonicStripeLevelsFamilies{SUFFIX}Order2Ceiling2.json'
if os.path.exists(familiesPath) and 'levels' in data['landscapes']:
    data['landscapes']['levels']['families'] = json.load(open(familiesPath))['codes']

# ------------------------------------------------------------------------ the relay loop sections and the pattern sliders (each only when its data exists)
steeringPaths = [args.steeringPath] if args.steeringPath else glob.glob(f'data/relayLoopSteeringData{SUFFIX}Codes*.json')
if os.path.exists(args.loopPath) and steeringPaths:
    data['relayLoop'] = json.load(open(args.loopPath))
    data['steering'] = json.load(open(max(steeringPaths, key=lambda path: int(path.rsplit('Codes', 1)[1][:-5]))))
if 'relayLoop' in data and os.path.exists(args.orderEdgeMapPath):
    data['orderEdgeMap'] = json.load(open(args.orderEdgeMapPath))
if os.path.exists(args.patternsPath):
    patterns = json.load(open(args.patternsPath))
    for family in patterns['families'].values():
        family.pop('scoreAtRead', None)                                    # not drawn
    ringAngles = boundary.ringAngles(boundary.boundaryRingCells)
    patterns['ringBasis'] = np.round(np.cos(np.outer(ringAngles, np.arange(len(patterns['trainedCoefficients'])))), 5).tolist()
    patterns['featureCells'] = summary['featureCells']
    data['patterns'] = patterns
    if os.path.exists(args.facePatternsPath):                                                          # the same sliders on the face code, for comparison
        facePatterns = json.load(open(args.facePatternsPath))
        for family in facePatterns['families'].values():
            family.pop('scoreAtRead', None)
        facePatterns['ringBasis'] = np.round(np.cos(np.outer(ringAngles, np.arange(len(facePatterns['trainedCoefficients'])))), 5).tolist()
        data['patternsFace'] = facePatterns
    if os.path.exists(args.smoothnessPath):
        data['smoothness'] = json.load(open(args.smoothnessPath))


def fragment(name):
    return open(f'figures/boundaryHarmonicStripes{name}', encoding='utf-8').read()


# ---------------------------------------------------------------------------------------------------- the page
page = open(args.templatePath, encoding='utf-8').read()
assert page.count('__DATA__') == 1
for placeholder, name, wanted in (('/*__LOOP_CSS__*/', 'RelayLoop.css', 'relayLoop' in data or 'patterns' in data), ('<!--__LOOP_HTML__-->', 'RelayLoop.html', 'relayLoop' in data),
                                  ('/*__LOOP_JS__*/', 'RelayLoop.js', 'relayLoop' in data), ('<!--__PATTERN_HTML__-->', 'PatternSlider.html', 'patterns' in data),
                                  ('/*__PATTERN_JS__*/', 'PatternSlider.js', 'patterns' in data)):
    assert page.count(placeholder) == 1, placeholder
    page = page.replace(placeholder, fragment(name) if wanted else '')
page = page.replace('__DATA__', json.dumps(data, separators=(',', ':')))
open(args.outputPath, 'w', encoding='utf-8').write(page)
print(f'relay loop sections: {"yes" if "relayLoop" in data else "no data yet"}; pattern sliders: {"yes" if "patterns" in data else "no data yet"}')
print(f'wrote {args.outputPath} ({len(page) / 1e6:.2f} MB): {len(data["orders"])} code sizes, '
      f'{sum(len(a["restarts"]) for r in ladder for a in r["arms"])} restarts in {len(ladder)} rungs')
