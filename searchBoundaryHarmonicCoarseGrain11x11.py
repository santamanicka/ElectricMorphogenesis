"""EXPLORATORY, not registered: the smallest accurate closed block model a greedy search can find.

Squares and Ward clusters both give closed-model accuracy that is erratic in the number of blocks (a pass at 49 blocks,
failures at 56, 64 and 81), so a single partition per size says little about the smallest size that can work. This
searches for it. Start from the lattice itself (every cell its own block). At each step merge the two neighbouring
blocks whose merger costs the closed model least, judged by the final-gap error plus the running-total error, and record
the partition at every block count and how it scores (measures as in the registered analysis, plus the nose, eyes and
mouth readouts the search may or may not have been judging).

--objective selectivity judges mergers on the selectivity readout alone. It finds closed models that reproduce it down to
about ten blocks and is kept as the demonstration that this is a fit to the objective: the route and the other readouts
are wrong there. --objective allReadouts (default) judges selectivity, nose, eyes and mouth equally.

A merger of blocks a and b changes the block model only by averaging their two entries after each step, so all the
candidate mergers at one level are run together, in the lifted space (every block's cells sharing its value), as columns
of one matrix. Nothing here feeds a registered verdict.

    python3 searchBoundaryHarmonicCoarseGrain11x11.py --jacobianPath <jacobians.npy> --relayPath <relayRegenerated.npz>
"""
import argparse
import json
import os
import time

import numpy as np

import boundaryCodeUtilities as boundary
import boundaryHarmonicCoarseGrain as coarse

parser = argparse.ArgumentParser()
parser.add_argument('--jacobianPath', type=str, required=True)
parser.add_argument('--relayPath', type=str, required=True)
parser.add_argument('--cachePath', type=str, default=None)
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicCoarseGrainSearch1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--stopAt', type=int, default=8, help='stop merging at this many blocks')
parser.add_argument('--scoreBelow', type=int, default=100, help='score every level with this many blocks or fewer')
parser.add_argument('--objective', type=str, default='allReadouts', choices=('selectivity', 'allReadouts'))
args = parser.parse_args()

RELEASE, PEAK = coarse.RELEASE, coarse.PEAK
BAR = dict(gapError=0.10, curveError=0.15, pathwayCosine=0.8)
FEATURE_BAR = 0.25                                                  # exploratory: each of nose, eyes, mouth within 25%
n = coarse.NUM_CELLS
started = time.time()


def say(*parts):
    print(f'[{time.time() - started:6.0f}s]', *parts, flush=True)


relay, sweep, regenerated = coarse.loadRelay(args.jacobianPath, args.relayPath, args.cachePath, say=say)
scorer = coarse.Scorer(relay, sweep, boundary.boundaryRingCells, coarse.READOUT_NAMES)
hold = relay.hold
fineGaps, trueCurves = scorer.fineGap, scorer.trueCurve             # every readout: (K,) and (states, K)
judged = [0] if args.objective == 'selectivity' else list(range(len(fineGaps)))
steps = PEAK
rows, columns = np.divmod(np.arange(n), 11)


def adjacentBlocks(labels):
    """Pairs of blocks that touch (some cells are neighbours on the lattice)."""
    pairs = set()
    for i in range(n):
        for j in range(n):
            if abs(rows[i] - rows[j]) + abs(columns[i] - columns[j]) == 1 and labels[i] != labels[j]:
                pairs.add((int(min(labels[i], labels[j])), int(max(labels[i], labels[j]))))
    return sorted(pairs)


def levelState(labels):
    """Total block Jacobians (steps, 2m, 2m), block sources, block readouts and sizes for a partition."""
    operators = relay.blockOperators(labels)
    return dict(m=int(labels.max()) + 1, total=relay.project(labels, roles=False), source=operators['blockSource'],
                readouts=operators['readouts'], sizes=np.bincount(labels).astype(float))


def evaluateMergers(state, pairs, returnCurves=False):
    """Closed-model readout curves for merging each pair, all at once. Columns of Y are the candidates' states lifted
    to the current blocks; after each step the two merged blocks' entries (Vmem and G_pol) are replaced by their
    size-weighted average. Returns each merger's final-gap error and running-total error, each against its own fine
    gap and averaged over the judged readouts."""
    m, total, blockSource, readouts, sizes = state['m'], state['total'], state['source'], state['readouts'], state['sizes']
    count = len(pairs)
    first = np.array([a for a, _ in pairs])
    second = np.array([b for _, b in pairs])
    weightFirst = sizes[first] / (sizes[first] + sizes[second])
    weightSecond = 1.0 - weightFirst
    columnsIndex = np.arange(count)
    Y = np.zeros((2 * m, count))
    curves = np.zeros((steps + 1, len(fineGaps), count))
    for step in range(steps):
        Z = total[step] @ Y
        if step < hold:
            Z += blockSource[step][:, None]
        for offset in (0, m):
            merged = weightFirst * Z[first + offset, columnsIndex] + weightSecond * Z[second + offset, columnsIndex]
            Z[first + offset, columnsIndex] = merged
            Z[second + offset, columnsIndex] = merged
        Y = Z
        curves[step + 1] = readouts @ Y
    gapErrors = np.abs(curves[PEAK] - fineGaps[:, None]) / np.abs(fineGaps)[:, None]                      # (readouts, count)
    curveErrors = np.sqrt(np.mean((curves - trueCurves[:, :, None]) ** 2, axis=0)) / np.abs(fineGaps)[:, None]
    result = (gapErrors[judged].mean(0), curveErrors[judged].mean(0))
    return result + (curves[:, 0, :],) if returnCurves else result


def merge(labels, first, second):
    labels = labels.copy()
    labels[labels == second] = first
    return coarse.compress(labels)


# the lifted-space merger evaluation must equal the standard projection of the merged partition
labels = np.arange(n)
check = levelState(labels)
checkPairs = adjacentBlocks(labels)[::47][:4]
_, _, checkCurves = evaluateMergers(check, checkPairs, returnCurves=True)
for column, (first, second) in enumerate(checkPairs):
    standard, _ = relay.reducedReadoutCurve(merge(labels, first, second))
    agreement = float(np.abs(standard[:, 0] - checkCurves[:, column]).max() / abs(fineGaps[0]))
    say(f'merger of cells {first},{second}: lifted-space curve vs standard projection {agreement:.2e}')
    assert agreement < 1e-9

path = []
while True:
    state = levelState(labels)
    m = state['m']
    pairs = adjacentBlocks(labels)
    gapErrors, curveErrors = evaluateMergers(state, pairs)
    best = int(np.argmin(gapErrors + curveErrors))
    del state                                                       # the scorer projects again; keep the peak memory down
    if m > args.scoreBelow:                                         # near the lattice every level is accurate; skip the heavy scoring
        say(f"m={m:>3}: next merger would give {gapErrors[best]:.3f}/{curveErrors[best]:.3f}")
        labels = merge(labels, *pairs[best])
        continue
    metrics, _ = scorer.score(labels, dict(name=f'search_{m}', family='search'))
    path.append(dict(m=m, labels=labels.tolist(), gapError=metrics['gapError'], curveError=metrics['curveError'],
                     fromRelease=metrics['fromRelease'], featureGapError=metrics['featureGapError'],
                     pathwayCosine=metrics['pathwayCosine'], netRetained=metrics['netRetained']['both'],
                     retainedShare=metrics['retainedShare'], links90=metrics['links90']['total'],
                     nextMerger=dict(pair=list(pairs[best]), gapError=float(gapErrors[best]), curveError=float(curveErrors[best]))))
    say(f"m={m:>3}: gap {metrics['gapError']:.3f}, curve {metrics['curveError']:.3f}, from release "
        f"{metrics['fromRelease']['gapError']:.3f}/{metrics['fromRelease']['curveError']:.3f}, nose/eyes/mouth "
        + '/'.join(f'{metrics["featureGapError"][k]:.2f}' for k in ('nose', 'eyes', 'mouth'))
        + f", cosines {metrics['pathwayCosine']['clear']:.2f}/{metrics['pathwayCosine']['write']:.2f}, "
        f"net retained {metrics['netRetained']['both']:.2f}")
    if m <= args.stopAt:
        break
    labels = merge(labels, *pairs[best])


def gapAndCurve(record):
    return bool(record['gapError'] <= BAR['gapError'] and record['curveError'] <= BAR['curveError'])


def registered(record):
    return bool(gapAndCurve(record) and min(record['pathwayCosine'].values()) >= BAR['pathwayCosine'])


def strict(record):
    return bool(registered(record) and all(record['featureGapError'][k] <= FEATURE_BAR for k in ('nose', 'eyes', 'mouth')))


def smallest(test):
    ok = [r['m'] for r in path if test(r)]
    return min(ok) if ok else None


def holdsFromHere(test):
    """Smallest block count from which every larger level on the path passes."""
    best = None
    for record in sorted(path, key=lambda r: -r['m']):
        if not test(record):
            break
        best = record['m']
    return best


summary = dict(
    objective=args.objective, featureBar=FEATURE_BAR,
    gapAndCurveOnly=dict(smallest=smallest(gapAndCurve), holdsFromHere=holdsFromHere(gapAndCurve)),
    registeredBar=dict(smallest=smallest(registered), holdsFromHere=holdsFromHere(registered)),
    strictBar=dict(smallest=smallest(strict), holdsFromHere=holdsFromHere(strict)),
    fromRelease=dict(smallest=smallest(lambda r: r['fromRelease']['gapError'] <= BAR['gapError']
                                       and r['fromRelease']['curveError'] <= BAR['curveError'])))
say('summary', summary)
keep = {summary['registeredBar']['smallest'], summary['strictBar']['smallest'], 81, 64, 49, 36, 25, 16, 10} - {None}
details = {f"search_{r['m']}": scorer.score(np.array(r['labels']), dict(name=f"search_{r['m']}", family='search'), detail=True)[1]
           for r in path if r['m'] in keep}
json.dump(dict(note='EXPLORATORY greedy backward merging of neighbouring blocks; not registered.', bar=BAR, path=path, summary=summary,
               details=details, stateIndices=scorer.stateIndices, trueCurve=scorer.series(trueCurves[:, 0]),
               ringCells=boundary.boundaryRingCells.tolist()), open(args.outputPath, 'w'))
say('wrote', args.outputPath, os.path.getsize(args.outputPath) // 1024, 'KB')
