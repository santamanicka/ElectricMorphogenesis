"""Score the relay decomposition against its registration and write what the report draws.

Criteria: data/boundaryHarmonicRelayPredictions1888Hold301FaceMinus60Minus5.json, committed before any of this
was computed. Input: the npz from computeBoundaryHarmonicRelay11x11.py. State index n = recorded iteration + 1.

    python3 analyzeBoundaryHarmonicRelay11x11.py --relayPath <relay.npz>
"""
import argparse
import json

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--relayPath', type=str, required=True)
parser.add_argument('--predictionsPath', type=str, default='data/boundaryHarmonicRelayPredictions1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicRelay1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--topEdges', type=int, default=60)
args = parser.parse_args()

r = np.load(args.relayPath)
names = [str(x) for x in r['readoutNames']]
flux, times = r['flux'], [int(t) for t in r['fluxTimes']]
edges, gross, window = r['edges'], r['grossEdges'], int(r['edgeWindow'])
source, difference = r['sourceFlux'], r['difference']
hold = int(r['hold'])
ring = np.array(boundary.boundaryRingCells)
interior = np.array(boundary.interiorCellIndices)
groups = dict(nose=[49, 60, 71], eyes=[24, 25, 29, 30, 35, 36, 40, 41], mouth=[92, 93, 94])
features = sorted(set(boundary.featureCellIndices.tolist()))
primary = names.index('selectivity')
RELEASE, TROUGH, PEAK = 302, 586, 1766                              # state indices of recorded 301, 585, 1765


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

# ------------------------------------------------------------------ R1: where the difference comes from
sp = source[primary]                                                # [V side effect, G injection] per cell
injected = sp.sum()
ringG = sp[1, ring].sum()
R1 = dict(total=float(injected), ringConductance=float(ringG), voltageSideEffect=float(sp[0].sum()),
          otherConductance=float(sp[1].sum() - ringG), share=float(ringG / injected))
R1['holds'] = bool(R1['share'] >= 0.9)

# ------------------------------------------------------------------ R2, R3: the boundary hands over
def shares(k, state):
    f = np.abs(flux[k, at(state)])
    return float(f[ring].sum() / f.sum()), float(f[interior].sum() / f.sum())


ringAtRelease, _ = shares(primary, RELEASE)
R2 = dict(ringShare=round(ringAtRelease, 3), holds=bool(ringAtRelease >= 0.5))
handover = next((t for t in times if RELEASE <= t <= PEAK
                 and np.abs(flux[primary, at(t)])[interior].sum() > np.abs(flux[primary, at(t)])[ring].sum()), None)
R3 = dict(handoverRecorded=(handover - 1) if handover is not None else None,
          holds=bool(handover is not None and TROUGH < handover < PEAK))

# ------------------------------------------------------------------ R4: the blind spot fills last
def halfTime(cells):
    series = np.array([np.abs(flux[primary, at(t)])[cells].sum() for t in times if RELEASE <= t <= PEAK])
    stamps = [t for t in times if RELEASE <= t <= PEAK]
    target = 0.5 * series[-1]
    return next(s for s, v in zip(stamps, series) if v >= target) - 1


R4 = dict(noseHalf=halfTime(groups['nose']), eyesHalf=halfTime(groups['eyes']), mouthHalf=halfTime(groups['mouth']))
R4['holds'] = bool(R4['noseHalf'] > R4['eyesHalf'])

# ------------------------------------------------------------------ R5: which channel carries it
first, last = RELEASE // window, (PEAK - 1) // window
fieldGross, contactGross = gross[primary, 0, first:last + 1].sum(), gross[primary, 1, first:last + 1].sum()
R5 = dict(fieldShare=float(fieldGross / (fieldGross + contactGross)),
          windowsFrom=first * window - 1, windowsTo=(last + 1) * window - 1)
R5['holds'] = bool(R5['fieldShare'] >= 0.5)

# ------------------------------------------------------------------ what the page draws
snapshotStates = [RELEASE, TROUGH, 1001, 1401, PEAK]
maps = {name: [[round(float(v), 6) for v in flux[k, at(s)]] for s in snapshotStates] for k, name in enumerate(names)}
phases = dict(flood=(0, RELEASE), clear=(RELEASE, TROUGH), write=(TROUGH, PEAK))


def network(k, lo, hi):
    """Net signed flux carried from cell j to cell i over a span, both channels, the strongest edges."""
    w0, w1 = lo // window, (hi - 1) // window
    total = edges[k, :, w0:w1 + 1].sum(1)                            # (channel, i, j)
    both = total.sum(0)
    net = both - both.T
    out = []
    for i, j in zip(*np.unravel_index(np.argsort(-np.abs(net), axis=None), net.shape)):
        if len(out) >= args.topEdges:
            break
        if net[i, j] <= 0:                                           # keep the direction the flux actually goes
            continue
        fieldPart = total[0, i, j] - total[0, j, i]
        out.append(dict(to=int(i), frm=int(j), flux=round(float(net[i, j]), 6),
                        field=round(float(fieldPart / net[i, j]), 3)))
    return out


networks = {name: {phase: network(k, lo, hi) for phase, (lo, hi) in phases.items()}
            for k, name in enumerate(names) if name != 'face'}
networks['face'] = {phase: network(names.index('face'), lo, hi)
                    for phase, (lo, hi) in dict(phases, latch=(PEAK, 2174)).items()}
transfer = r['transfer']
transferWindows = [tuple(int(x) for x in w) for w in r['transferWindows']]
whole = transfer[transferWindows.index((RELEASE, PEAK)), primary]   # from where at release to where at the peak
intoFeatures = whole[features].sum(0)                               # by source cell at release
result = dict(
    predictions=json.load(open(args.predictionsPath)),
    readouts=names, difference={n: float(d) for n, d in zip(names, difference)},
    verdicts=dict(V1=V1, V2=V2, R1=R1, R2=R2, R3=R3, R4=R4, R5=R5),
    snapshots=[s - 1 for s in snapshotStates], maps=maps,
    ringSource=[round(float(v), 6) for v in sp[1, ring]], ringCells=ring.tolist(),
    voltageSource=[round(float(v), 6) for v in sp[0]],
    groupSeries={g: [round(float(np.abs(flux[primary, at(t)])[c].sum()), 6) for t in times if RELEASE <= t <= PEAK]
                 for g, c in dict(groups, ring=list(ring)).items()},
    seriesTimes=[t - 1 for t in times if RELEASE <= t <= PEAK],
    networks=networks,
    releaseToPeak=dict(bySource=[round(float(v), 6) for v in intoFeatures],
                       ringShare=float(intoFeatures[ring].sum() / intoFeatures.sum())),
    tangentGrowth=dict(free=float(r['tangentNorm'][0, RELEASE] / r['tangentNorm'][0, PEAK]),
                       trained=float(r['tangentNorm'][1, RELEASE] / r['tangentNorm'][1, PEAK])),
    splitSteps=int((r['pieceCount'] > 1).sum()))
json.dump(result, open(args.outputPath, 'w'))
for key, verdict in result['verdicts'].items():
    print(f"{key}: {'holds' if verdict['holds'] else 'FAILS'}  " +
          str({k: v for k, v in verdict.items() if k != 'holds'}), flush=True)
print('difference', result['difference'])
print('tangent-linear adjoint growth release<-peak:', result['tangentGrowth'])
print('wrote', args.outputPath)
