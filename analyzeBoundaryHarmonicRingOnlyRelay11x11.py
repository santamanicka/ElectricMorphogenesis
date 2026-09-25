"""Score the ring-only relay (baseline extraUpdateOnly) against its registration.

Criteria: data/boundaryHarmonicRingOnlyRelayPredictions1888Hold301FaceMinus60Minus5.json, committed before this
run. Input: the npz from computeBoundaryHarmonicRelay11x11.py --baseline extraUpdateOnly.

    python3 analyzeBoundaryHarmonicRingOnlyRelay11x11.py --relayPath <relayRingOnly.npz>
"""
import argparse
import json

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--relayPath', type=str, required=True)
parser.add_argument('--noClampRelayPath', type=str, default=None,
                    help='the free-baseline relay npz, for the above-and-beyond comparison')
parser.add_argument('--predictionsPath', type=str, default='data/boundaryHarmonicRingOnlyRelayPredictions1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicRingOnlyRelay1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--topEdges', type=int, default=60)
args = parser.parse_args()

r = np.load(args.relayPath)
assert str(r['baseline']) == 'extraUpdateOnly', r['baseline']
names = [str(x) for x in r['readoutNames']]
flux, times = r['flux'], [int(t) for t in r['fluxTimes']]
edges, gross, window = r['edges'], r['grossEdges'], int(r['edgeWindow'])
source, difference = r['sourceFlux'], r['difference']
D = r['D']                                                          # (states, 2n), n=121
hold = int(r['hold'])
n = 121
ring = np.array(boundary.boundaryRingCells)
interior = np.array(boundary.interiorCellIndices)
groups = dict(nose=[49, 60, 71], eyes=[24, 25, 29, 30, 35, 36, 40, 41], mouth=[92, 93, 94])
primary = names.index('selectivity')
RELEASE, TROUGH, PEAK = 302, 586, 1766                              # state indices of recorded 301, 585, 1765

shell = np.zeros(n, dtype=int)
for s in range(6):
    shell[boundary.shellCells(s)] = s


def at(state):
    return times.index(state - state % 5)


def cancellation(k, state):
    f = flux[k, at(state)]
    return float(np.abs(f).sum() / abs(f.sum()))


# ------------------------------------------------------------------ validity
closureWorst = float(r['closure'].max())
conservation = {name: float(c) for name, c in zip(names, r['conservation'])}
V1 = dict(closureWorst=closureWorst, conservation=conservation,
          holds=bool(closureWorst <= 1e-6 and max(conservation.values()) <= 1e-3))
V2 = dict(index={str(s - 1): round(cancellation(primary, s), 3) for s in (RELEASE, TROUGH, PEAK)})
V2['holds'] = bool(all(v <= 5 for v in V2['index'].values()))

# ------------------------------------------------------------------ R1: does isolating the ring leave little else
sp = source[primary]
injected = sp.sum()
ringG = sp[1, ring].sum()
R1 = dict(total=float(injected), ringConductance=float(ringG), voltageSideEffect=float(sp[0].sum()),
          otherConductance=float(sp[1].sum() - ringG), share=float(ringG / injected))
R1['holds'] = bool(R1['share'] >= 0.95)


def shares(k, state):
    f = np.abs(flux[k, at(state)])
    return float(f[ring].sum() / f.sum()), float(f[interior].sum() / f.sum())


ringAtRelease, _ = shares(primary, RELEASE)
R2 = dict(ringShare=round(ringAtRelease, 3), holds=bool(ringAtRelease <= 0.10))
handover = next((t for t in times if RELEASE <= t <= PEAK
                 and np.abs(flux[primary, at(t)])[interior].sum() > np.abs(flux[primary, at(t)])[ring].sum()), None)
R3 = dict(handoverRecorded=(handover - 1) if handover is not None else None,
          holds=bool(handover is not None and (handover - RELEASE) <= 20))


def halfTime(cells):
    series = np.array([np.abs(flux[primary, at(t)])[cells].sum() for t in times if RELEASE <= t <= PEAK])
    stamps = [t for t in times if RELEASE <= t <= PEAK]
    target = 0.5 * series[-1]
    return next(s for s, v in zip(stamps, series) if v >= target) - 1


R4 = dict(noseHalf=halfTime(groups['nose']), eyesHalf=halfTime(groups['eyes']), mouthHalf=halfTime(groups['mouth']))
R4['holds'] = bool(R4['noseHalf'] >= R4['eyesHalf'])

first, last = RELEASE // window, (PEAK - 1) // window
fieldGross, contactGross = gross[primary, 0, first:last + 1].sum(), gross[primary, 1, first:last + 1].sum()
R5 = dict(fieldShare=float(fieldGross / (fieldGross + contactGross)),
          windowsFrom=first * window - 1, windowsTo=(last + 1) * window - 1)
R5['holds'] = bool(R5['fieldShare'] >= 0.5)

# ------------------------------------------------------------------ B1, B2: shell-depth diagnostics
third = (TROUGH - RELEASE) // 3


def depthCurve(weight):
    """weight: (states, n) array of |value| on the G_pol component. Returns depth(t) for t in [RELEASE, PEAK)."""
    span = weight[RELEASE:PEAK]
    denom = span.sum(1)
    return (span * shell).sum(1) / np.where(denom > 0, denom, 1.0)


rawDepth = depthCurve(np.abs(D[RELEASE:PEAK, n:]))
B1 = dict(firstThird=round(float(rawDepth[:third].mean()), 3), finalThird=round(float(rawDepth[-third:].mean()), 3))
B1['holds'] = bool(B1['firstThird'] <= 1.5 and B1['finalThird'] > 3.0)

fluxWeight = np.zeros((PEAK - RELEASE, n))
for i, t in enumerate(range(RELEASE, PEAK)):
    fluxWeight[i] = np.abs(flux[primary, at(t)])
fluxDepth = depthCurve(fluxWeight)
B2 = dict(firstThird=round(float(fluxDepth[:third].mean()), 3), finalThird=round(float(fluxDepth[-third:].mean()), 3))
B2['holds'] = bool(B2['firstThird'] <= 1.5 and B2['finalThird'] > 3.0)

# the full curves, release to the face readout, for the report's diagnostic figure (no registered threshold)
FACE = 2174
rawDepthFull = depthCurve(np.abs(D[RELEASE:FACE, n:])) if D.shape[0] >= FACE else rawDepth
depthTimes = [t - 1 for t in range(RELEASE, RELEASE + len(rawDepthFull))]
fluxDepthFull = []
fluxTimesInRange = [t for t in times if RELEASE <= t < min(FACE, times[-1] + 1)]
for t in fluxTimesInRange:
    f = np.abs(flux[primary, at(t)])
    total = f.sum()
    fluxDepthFull.append(float((f * shell).sum() / total) if total > 0 else fluxDepthFull[-1])

# ------------------------------------------------------------------ what the page draws (same as the first relay)
snapshotStates = [RELEASE, TROUGH, 1001, 1401, PEAK]
maps = {name: [[round(float(v), 6) for v in flux[k, at(s)]] for s in snapshotStates] for k, name in enumerate(names)}
phases = dict(flood=(0, RELEASE), clear=(RELEASE, TROUGH), write=(TROUGH, PEAK))


def network(k, lo, hi):
    w0, w1 = lo // window, (hi - 1) // window
    total = edges[k, :, w0:w1 + 1].sum(1)
    both = total.sum(0)
    net = both - both.T
    out = []
    for i, j in zip(*np.unravel_index(np.argsort(-np.abs(net), axis=None), net.shape)):
        if len(out) >= args.topEdges:
            break
        if net[i, j] <= 0:
            continue
        fieldPart = total[0, i, j] - total[0, j, i]
        out.append(dict(to=int(i), frm=int(j), flux=round(float(net[i, j]), 6),
                        field=round(float(fieldPart / net[i, j]), 3)))
    return out


networks = {name: {phase: network(k, lo, hi) for phase, (lo, hi) in phases.items()}
            for k, name in enumerate(names) if name != 'face'}
networks['face'] = {phase: network(names.index('face'), lo, hi)
                    for phase, (lo, hi) in dict(phases, latch=(PEAK, 2174)).items()}


def intoTarget(k, lo, hi, targets):
    """Every edge landing on a target cell, from the FULL matrix (not the top-60 overall ranking, which the
    nose's own three cells cannot compete in against the 67-cell background term their readout subtracts)."""
    w0, w1 = lo // window, (hi - 1) // window
    total = edges[k, :, w0:w1 + 1].sum(1)
    both = total.sum(0)
    net = both - both.T
    out = []
    for target in targets:
        for j in np.argsort(-np.abs(net[target])):
            if j == target or net[target, j] <= 0:
                continue
            fieldPart = total[0, target, j] - total[0, j, target]
            out.append(dict(to=int(target), frm=int(j), flux=round(float(net[target, j]), 6),
                            field=round(float(fieldPart / net[target, j]), 3)))
    return sorted(out, key=lambda e: -e['flux'])


networks['intoNose'] = {phase: intoTarget(primary, lo, hi, groups['nose']) for phase, (lo, hi) in phases.items()}


def grossIntoNose(edgeArray, window_, lo, hi):
    """Total positive net flux landing on the nose cells, both channels -- the traffic actually carried, not a
    top-N cutoff, so it is comparable across two different edge arrays with different node."""
    w0, w1 = lo // window_, (hi - 1) // window_
    total = edgeArray[primary, :, w0:w1 + 1].sum(1).sum(0)
    net = total - total.T
    rows = net[groups['nose']]
    return float(rows[rows > 0].sum())


aboveAndBeyond = None
if args.noClampRelayPath:
    free = np.load(args.noClampRelayPath)
    assert str(free['readoutNames'][primary]) == 'selectivity'
    freeWindow = int(free['edgeWindow'])
    aboveAndBeyond = {}
    for label, (lo, hi) in dict(phases, releaseToPeak=(RELEASE, PEAK)).items():
        ringOnlyGross = grossIntoNose(edges, window, lo, hi)
        freeGross = grossIntoNose(free['edges'], freeWindow, lo, hi)
        aboveAndBeyond[label] = dict(ringOnly=round(ringOnlyGross, 4), noClamp=round(freeGross, 4),
                                     ratio=round(ringOnlyGross / freeGross, 3) if freeGross > 0 else None)
    freeNames = [str(x) for x in free['readoutNames']]
    aboveAndBeyond['noClampNoseReadoutDifference'] = round(float(free['difference'][freeNames.index('nose')]), 4)
    print('above-and-beyond, gross flux into the nose, ring-only vs no-clamp:', aboveAndBeyond, flush=True)

GREF = 1e-9   # the model's G_ref; the assertion below fails if the arrays disagree with it
featureCells = np.array(sorted(set(boundary.featureCellIndices.tolist())))
backgroundCells = np.array([c for c in interior if c not in set(featureCells.tolist())])


def conditionStats(state):
    """The three numbers that separate the conditions. Recorded iteration t is state t + 1."""
    G = state[:, n:].astype(float) / GREF
    interiorMean = G[1:, interior].mean(1)                              # indexed by recorded iteration
    trough = 302 + int(interiorMean[302:1300].argmin())
    selectivity = float(G[1766, featureCells].mean() - G[1766, backgroundCells].mean())
    return dict(interiorMeanAtRelease=round(float(interiorMean[hold]), 3), troughAt=trough,
                selectivityAt1765=round(selectivity, 3))


conditions = None
if args.noClampRelayPath:
    trainedStats = conditionStats(r['trainedState'])
    assert trainedStats == conditionStats(free['trainedState']), 'the two relays disagree on the trained run'
    conditions = dict(noClamp=conditionStats(free['baselineState']),
                      relayBaseline=conditionStats(r['baselineState']), trained=trainedStats)
    check = trainedStats['selectivityAt1765'] - conditions['relayBaseline']['selectivityAt1765']
    assert abs(check - difference[primary]) < 2e-3, (check, difference[primary])
    print('conditions:', conditions, flush=True)


def participationRatio(values):
    """(sum|x|)^2 / sum(x^2): the number of equally-weighted entries that would give the same ratio -- how many
    of the possible directed pairs the relay is effectively spread over, not a top-N cutoff."""
    flat = np.abs(values).ravel()
    total, sumSquares = flat.sum(), (flat ** 2).sum()
    return float(total ** 2 / sumSquares) if sumSquares > 0 else 0.0


reduction = {}
for label, (lo, hi) in dict(phases, releaseToPeak=(RELEASE, PEAK)).items():
    w0, w1 = lo // window, (hi - 1) // window
    total = edges[primary, :, w0:w1 + 1].sum(1).sum(0)             # (121, 121), both channels combined
    reduction[label] = dict(
        wholeNetwork=dict(participationRatio=round(participationRatio(total), 1), possible=121 * 120),
        intoNose=dict(participationRatio=round(participationRatio(total[groups['nose']]), 1), possible=3 * 120))

result = dict(
    baseline='extraUpdateOnly',
    predictions=json.load(open(args.predictionsPath)),
    readouts=names, difference={n_: float(d) for n_, d in zip(names, difference)},
    verdicts=dict(V1=V1, V2=V2, R1=R1, R2=R2, R3=R3, R4=R4, R5=R5, B1=B1, B2=B2),
    snapshots=[s - 1 for s in snapshotStates], maps=maps,
    ringSource=[round(float(v), 6) for v in sp[1, ring]], ringCells=ring.tolist(),
    networks=networks, reduction=reduction, aboveAndBeyond=aboveAndBeyond, conditions=conditions,
    groupSeries={g: [round(float(np.abs(flux[primary, at(t)])[c].sum()), 6) for t in times if RELEASE <= t <= PEAK]
                for g, c in dict(groups, ring=list(ring)).items()},
    seriesTimes=[t - 1 for t in times if RELEASE <= t <= PEAK],
    depthDiagnostic=dict(rawTimes=depthTimes, rawDepth=[round(float(x), 4) for x in rawDepthFull],
                         fluxTimes=[t - 1 for t in fluxTimesInRange], fluxDepth=[round(x, 4) for x in fluxDepthFull]))
json.dump(result, open(args.outputPath, 'w'))
for key, verdict in result['verdicts'].items():
    print(f"{key}: {'holds' if verdict['holds'] else 'FAILS'}  " +
          str({k: v for k, v in verdict.items() if k != 'holds'}), flush=True)
print('difference', result['difference'])
print('wrote', args.outputPath)
