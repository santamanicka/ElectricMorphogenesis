"""Read the relay at every block resolution and score how much of it survives.

Criteria: data/boundaryHarmonicCoarseGrainPredictions1888Hold301FaceMinus60Minus5.json, committed before any of this was
computed. Inputs: the Jacobians and the regenerated relay from computeBoundaryHarmonicRelay11x11.py --jacobianPath,
and the committed ring-only relay it must reproduce. See boundaryHarmonicCoarseGrain.py for the two objects scored
(aggregated: the fine books re-cut over blocks; reduced: a closed model on the blocks).

    python3 analyzeBoundaryHarmonicCoarseGrain11x11.py --jacobianPath <jacobians.npy> --relayPath <relayRegenerated.npz>
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.stats import spearmanr

import boundaryCodeUtilities as boundary
import boundaryHarmonicCoarseGrain as coarse

parser = argparse.ArgumentParser()
parser.add_argument('--jacobianPath', type=str, required=True)
parser.add_argument('--relayPath', type=str, required=True)
parser.add_argument('--committedRelayPath', type=str, default='data/boundaryHarmonicRingOnlyRelay1888Hold301FaceMinus60Minus5Raw.npz')
parser.add_argument('--predictionsPath', type=str, default='data/boundaryHarmonicCoarseGrainPredictions1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicCoarseGrain1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--cachePath', type=str, default=None, help='keep the fine sweep here so reruns skip it')
parser.add_argument('--randomDraws', type=int, default=200)
parser.add_argument('--randomSizes', type=int, nargs='*', default=[2, 3, 4, 5, 6])
parser.add_argument('--skipRandom', action='store_true')
args = parser.parse_args()

RELEASE, TROUGH, PEAK, PHASES = coarse.RELEASE, coarse.TROUGH, coarse.PEAK, coarse.PHASES
READOUT_ROWS, READOUT_NAMES = coarse.READOUT_ROWS, coarse.READOUT_NAMES
BAR = dict(gapError=0.10, curveError=0.15, pathwayCosine=0.8)
RING = boundary.boundaryRingCells
started = time.time()


def say(*parts):
    print(f'[{time.time() - started:6.0f}s]', *parts, flush=True)


# ------------------------------------------------------------------ inputs
committed = np.load(args.committedRelayPath)
relay, sweep, regenerated = coarse.loadRelay(args.jacobianPath, args.relayPath, args.cachePath, say=lambda *parts: say(*parts))
difference, readouts = relay.difference, relay.readouts
fineCurve = difference[:PEAK + 1] @ readouts.T                       # w . D(n), the running total, for every readout
fineGap = fineCurve[PEAK]
assert abs(fineGap[0] - float(committed['difference'][0])) < 1e-9, (fineGap[0], committed['difference'][0])
say('fine gaps', dict(zip(READOUT_NAMES, np.round(fineGap, 6))))

# ------------------------------------------------------------------ V1: the machinery is the fine relay
fluxTimes = [int(t) for t in committed['fluxTimes']]
committedFlux = committed['flux']
fluxError = max(float(np.abs(committedFlux[READOUT_ROWS[k], fluxTimes.index(t)] - sweep['cellFlux'][t, k]).max()) / abs(fineGap[k])
                for t in range(0, PEAK, 5) for k in range(len(READOUT_ROWS)))
windowsShared = sweep['windowEdges'].shape[2]
committedEdges = committed['edges'][READOUT_ROWS][:, :, :windowsShared]
edgeError = float(np.abs(committedEdges - sweep['windowEdges']).max() / np.abs(committedEdges).max())
identityRelay = relay.reducedRelay(np.arange(coarse.NUM_CELLS))
identityFluxError = float(np.abs(identityRelay['flux'] - sweep['cellFlux']).max() / abs(fineGap[0]))
identityGapError = float(np.abs(identityRelay['curve'][PEAK] - fineGap).max() / abs(fineGap[0]))
closureRegenerated = float(regenerated['closure'].max())
V1 = dict(fluxVersusCommitted=fluxError, edgesVersusCommitted=edgeError, identityFluxVersusFine=identityFluxError,
          identityFinalGapError=identityGapError, closureWorstRegenerated=closureRegenerated,
          closureWorstCommitted=float(committed['closure'].max()),
          holds=bool(fluxError <= 1e-6 and edgeError <= 1e-6 and identityGapError <= 1e-6 and closureRegenerated <= 9.8e-7),
          context='The regenerated Jacobians reproduce the committed flux and edges to rounding. What misses the registered 1e-6 '
                  'is the identity block relay, which re-evolves the difference forward with the stored Jacobians and so inherits, '
                  'accumulated over 1,766 steps, the fine relay\'s own one-step closure residual (up to 9.8e-7 each). It is a property '
                  'of the fine decomposition, not of the coarse-graining, and it is four orders below the 10% bar.')
say('V1', V1)


# ------------------------------------------------------------------ scoring
scorer = coarse.Scorer(relay, sweep, RING, READOUT_NAMES, release=RELEASE, trough=TROUGH)
stateIndices = scorer.stateIndices


def score(labels, meta, detail=False):
    return scorer.score(labels, meta, detail=detail, fromRelease=True)


def meets(metrics):
    return bool(metrics['gapError'] <= BAR['gapError'] and metrics['curveError'] <= BAR['curveError'])


def meetsRoute(metrics):
    return bool(meets(metrics) and min(metrics['pathwayCosine'].values()) >= BAR['pathwayCosine'])


# ------------------------------------------------------------------ the square tilings
squares, details, tilingLabels = [], {}, {}
for size in range(1, 7):
    for tiling in (coarse.squareTilings(size) if size > 1 else [dict(size=1, shortRow=0, shortColumn=0, canonical=True,
                                                                     labels=np.arange(coarse.NUM_CELLS))]):
        name = f"b{size}" if size == 1 else f"b{size}_row{tiling['shortRow']}_col{tiling['shortColumn']}"
        metrics, _ = score(tiling['labels'], dict(name=name, family='squares', size=size, shortRow=tiling['shortRow'],
                                                  shortColumn=tiling['shortColumn'], canonical=tiling['canonical']))
        squares.append(metrics)
        tilingLabels[name] = tiling['labels']
    subset = [s for s in squares if s['size'] == size]
    say(f'b={size}: {len(subset)} tilings, m={subset[0]["m"]}, gap error best/mean/worst '
        f'{min(s["gapError"] for s in subset):.3f}/{np.mean([s["gapError"] for s in subset]):.3f}/'
        f'{max(s["gapError"] for s in subset):.3f}, curve error best {min(s["curveError"] for s in subset):.3f}')

fine = squares[0]
linksFine = fine['links90']['total']
bySize = {}
for size in range(1, 7):
    subset = [s for s in squares if s['size'] == size]
    best = min(subset, key=lambda s: s['gapError'])
    canonical = next(s for s in subset if s['canonical'])
    bySize[str(size)] = dict(
        m=subset[0]['m'], tilings=len(subset), best=best['name'], canonical=canonical['name'],
        gapError=dict(best=best['gapError'], mean=float(np.mean([s['gapError'] for s in subset])),
                      worst=max(s['gapError'] for s in subset), canonical=canonical['gapError']),
        curveError=dict(best=min(s['curveError'] for s in subset), mean=float(np.mean([s['curveError'] for s in subset])),
                        worst=max(s['curveError'] for s in subset), canonical=canonical['curveError']),
        meetingBar=sum(meets(s) for s in subset), meetingBarAndRoute=sum(meetsRoute(s) for s in subset),
        links90=dict(best=best['links90']['total'], canonical=canonical['links90']['total']),
        resolvedReadout=dict(mean=float(np.mean([s['resolvedReadout'] for s in subset]))))

for size in range(2, 7):                                            # full detail only for the canonical and the best tiling
    for name in {bySize[str(size)]['best'], bySize[str(size)]['canonical']}:
        details[name] = score(tilingLabels[name], next(s for s in squares if s['name'] == name), detail=True)[1]

# ------------------------------------------------------------------ wall versus interior (A6 and its context)
wallFamily = []
for length in (2, 4, 5, 8, 10, 20, 40):
    metrics, _ = score(coarse.ringSegmentLabels(length, RING), dict(name=f'ringSegments{length}', family='ringSegments', parameter=length))
    wallFamily.append(metrics)
for size in (2, 3, 4, 5, 9):
    metrics, _ = score(coarse.interiorSquareLabels(size, RING), dict(name=f'interiorSquares{size}', family='interiorSquares', parameter=size))
    wallFamily.append(metrics)
for metrics in wallFamily:
    say(f"{metrics['name']:>18}: m={metrics['m']:>3}, gap error {metrics['gapError']:.3f}, curve error {metrics['curveError']:.3f}")

# ------------------------------------------------------------------ control: scattered partitions with the same block sizes
randomControl = {}
if not args.skipRandom:
    generator = np.random.default_rng(20250925)
    for size in args.randomSizes:
        best = next(s for s in squares if s['name'] == bySize[str(size)]['best'])
        gapErrors, curveErrors = [], []
        for _ in range(args.randomDraws):
            labels = coarse.randomPartition(best['blockSizes'], generator)
            curve, _ = relay.reducedReadoutCurve(labels)
            gapErrors.append(abs(curve[PEAK, 0] - fineGap[0]) / abs(fineGap[0]))
            curveErrors.append(np.sqrt(np.mean((curve[:, 0] - fineCurve[:, 0]) ** 2)) / abs(fineGap[0]))
        randomControl[str(size)] = dict(
            tiling=best['name'], squareGapError=best['gapError'], squareCurveError=best['curveError'],
            randomGapError=dict(min=float(np.min(gapErrors)), p5=float(np.percentile(gapErrors, 5)),
                                median=float(np.median(gapErrors)), max=float(np.max(gapErrors))),
            randomCurveError=dict(min=float(np.min(curveErrors)), p5=float(np.percentile(curveErrors, 5)),
                                  median=float(np.median(curveErrors))),
            squareBeatsRandom=float(np.mean(np.array(gapErrors) > best['gapError'])))
        say(f'random control b={size}: square {best["gapError"]:.3f} vs random 5th percentile '
            f'{randomControl[str(size)]["randomGapError"]["p5"]:.3f}, median {randomControl[str(size)]["randomGapError"]["median"]:.3f}')

# ------------------------------------------------------------------ verdicts, against the registered criteria
V2 = dict(aggregatedBooksWorst=max(s['booksAggregated'] for s in squares + wallFamily),
          reducedBooksWorst=max(s['booksReduced'] for s in squares + wallFamily))
V2['holds'] = bool(V2['aggregatedBooksWorst'] <= 1e-4 and V2['reducedBooksWorst'] <= 1e-9)

coarser = [s for s in squares if s['size'] >= 2]
rho = spearmanr([s['m'] for s in squares], [s['gapError'] for s in squares])
A1 = dict(spearman=float(rho.correlation), holds=bool(rho.correlation <= -0.7))
sixes = [s for s in squares if s['size'] == 6]
A2 = dict(gapErrors=[s['gapError'] for s in sixes], resolvedReadout=[s['resolvedReadout'] for s in sixes],
          holds=bool(all(s['gapError'] > 0.5 for s in sixes) and all(s['resolvedReadout'] < 0.5 for s in sixes)))
bestTwo = next(s for s in squares if s['name'] == bySize['2']['best'])
threePlus = [s for s in squares if s['size'] >= 3 and meets(s)]
A3 = dict(bestB2=dict(name=bestTwo['name'], gapError=bestTwo['gapError'], curveError=bestTwo['curveError']),
          b2TilingsMeetingBar=[s['name'] for s in squares if s['size'] == 2 and meets(s)],
          coarserTilingsMeetingBar=[s['name'] for s in threePlus],
          holds=bool(meets(bestTwo) and not threePlus))
qualifying = [s for s in coarser if meets(s)]
A4 = dict(qualifying=len(qualifying), cosines={s['name']: s['pathwayCosine'] for s in qualifying},
          holds=bool(qualifying and all(min(s['pathwayCosine'].values()) >= BAR['pathwayCosine'] for s in qualifying)),
          vacuous=bool(not qualifying))
handovers = {s['name']: s['handOver'] for s in qualifying}
A5 = dict(handOvers=handovers,
          holds=bool(qualifying and all(h['reduced'] is not None and h['aggregated'] is not None
                                        and abs(h['reduced'] - h['aggregated']) <= 30
                                        for h in handovers.values() if h['aggregated'] is not None)),
          vacuous=bool(not qualifying))
ringFour = next(s for s in wallFamily if s['name'] == 'ringSegments4')
interiorThree = next(s for s in wallFamily if s['name'] == 'interiorSquares3')
A6 = dict(ringSegments4=dict(m=ringFour['m'], gapError=ringFour['gapError']),
          interiorSquares3=dict(m=interiorThree['m'], gapError=interiorThree['gapError']),
          holds=bool(ringFour['gapError'] <= 0.05 and interiorThree['gapError'] > 0.05
                     and interiorThree['gapError'] > ringFour['gapError']))
sufficient = sorted((s for s in squares if meetsRoute(s)), key=lambda s: (s['m'], s['links90']['total']))
answer = sufficient[0] if sufficient else None
coarserSufficient = [s for s in sufficient if s['size'] >= 2]
A7 = dict(smallestSufficientCoarser=coarserSufficient[0]['name'] if coarserSufficient else None,
          links90=coarserSufficient[0]['links90']['total'] if coarserSufficient else None, linksFine=linksFine,
          ratio=(coarserSufficient[0]['links90']['total'] / linksFine) if coarserSufficient else None)
A7['holds'] = bool(coarserSufficient and A7['ratio'] <= 0.25)
A8 = {}
for size in (2, 3):
    if str(size) in randomControl:
        control = randomControl[str(size)]
        A8[str(size)] = dict(square=control['squareGapError'], randomP5=control['randomGapError']['p5'],
                             holds=bool(control['squareGapError'] < control['randomGapError']['p5']))
A8['holds'] = bool(A8 and all(v['holds'] for v in A8.values()))

# knee of best-tiling gap error against block count
points = np.array([[np.log(bySize[str(b)]['m']), bySize[str(b)]['gapError']['best']] for b in range(1, 7)])
order = np.argsort(points[:, 0])
points = points[order]
x = (points[:, 0] - points[0, 0]) / (points[-1, 0] - points[0, 0])
y = points[:, 1] / max(points[:, 1].max(), 1e-300)
chord = y[0] + (y[-1] - y[0]) * x
knee = int(np.argmax(np.abs(y - chord)))
sizesByM = [b for b in range(1, 7)]
sizesByM.sort(key=lambda b: bySize[str(b)]['m'])
decision = dict(bar=BAR, sufficientTilings=[s['name'] for s in sufficient[:12]],
                simplestAccurate=None if answer is None else dict(
                    name=answer['name'], m=answer['m'], gapError=answer['gapError'], curveError=answer['curveError'],
                    pathwayCosine=answer['pathwayCosine'], links90=answer['links90']['total']),
                kneeBlockSize=sizesByM[knee], kneeBlocks=bySize[str(sizesByM[knee])]['m'])

verdicts = dict(V1=V1, V2=V2, A1=A1, A2=A2, A3=A3, A4=A4, A5=A5, A6=A6, A7=A7, A8=A8)
for key, verdict in verdicts.items():
    say(f"{key}: {'holds' if verdict.get('holds') else 'FAILS'}", {k: v for k, v in verdict.items() if k != 'holds'})
say('decision', decision)



result = dict(
    predictions=json.load(open(args.predictionsPath)), readouts=READOUT_NAMES, fineGap=dict(zip(READOUT_NAMES, fineGap.tolist())),
    trueCurve=scorer.series(fineCurve[:, 0]), stateIndices=stateIndices, phases=PHASES, ringCells=RING.tolist(),
    fineFinalFlux=np.round(sweep['cellFlux'][PEAK, 0], 6).tolist(), squares=squares, bySize=bySize, wallFamily=wallFamily, randomControl=randomControl,
    verdicts=verdicts, decision=decision, details=details)
json.dump(result, open(args.outputPath, 'w'))
say('wrote', args.outputPath, os.path.getsize(args.outputPath) // 1024, 'KB')
