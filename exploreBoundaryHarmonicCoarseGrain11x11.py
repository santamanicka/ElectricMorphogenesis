"""EXPLORATORY, not registered: what does work, once square blocks have been shown not to?

The registered coarse-graining (analyzeBoundaryHarmonicCoarseGrain11x11.py) found no square tiling coarser than the
lattice itself whose closed block model reproduces the relay. Added afterwards, at the user's request ('squares first,
non-square partitions as a labelled extra'), and never folded into the registered verdicts:

  named       partitions made by hand from the lattice's geometry (shells, feature groups);
  clusters    contiguous Ward clustering of the cells by what the fine relay says they do (their difference over time,
              and the share of the gap they hold over time), at many block counts;
  pod         a closed model on the leading patterns of the difference (proper orthogonal decomposition), the best
              lattice-free basis for holding the state, at many dimensions;
  pruning     the fine relay with all but the strongest cell-to-cell links removed.

Every closed model is scored as in the registered analysis, whole run and started from the released state.

    python3 exploreBoundaryHarmonicCoarseGrain11x11.py --jacobianPath <jacobians.npy> --relayPath <relayRegenerated.npz>
"""
import argparse
import json
import os
import time

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering

import boundaryCodeUtilities as boundary
import boundaryHarmonicCoarseGrain as coarse

parser = argparse.ArgumentParser()
parser.add_argument('--jacobianPath', type=str, required=True)
parser.add_argument('--relayPath', type=str, required=True)
parser.add_argument('--cachePath', type=str, default=None)
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicCoarseGrainExploratory1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--clusterSizes', type=int, nargs='*', default=[4, 6, 8, 10, 12, 16, 20, 25, 30, 36, 40, 44, 49, 56, 64, 81])
parser.add_argument('--podSizes', type=int, nargs='*', default=[2, 4, 6, 8, 10, 12, 16, 20, 30, 40, 60, 100])
parser.add_argument('--linkCounts', type=int, nargs='*', default=[100, 200, 400, 800, 1200, 1600, 2400, 3200, 4800, 7200, 14520])
args = parser.parse_args()

RELEASE, PEAK, PHASES = coarse.RELEASE, coarse.PEAK, coarse.PHASES
BAR = dict(gapError=0.10, curveError=0.15, pathwayCosine=0.8)
RING = boundary.boundaryRingCells
n = coarse.NUM_CELLS
started = time.time()


def say(*parts):
    print(f'[{time.time() - started:6.0f}s]', *parts, flush=True)


relay, sweep, regenerated = coarse.loadRelay(args.jacobianPath, args.relayPath, args.cachePath, say=say)
scorer = coarse.Scorer(relay, sweep, RING, coarse.READOUT_NAMES)
difference, source, readouts = relay.difference, relay.source, relay.readouts
gap, trueCurve = scorer.fineGap[0], scorer.trueCurve[:, 0]
details = {}


def passes(metrics):
    return bool(metrics['gapError'] <= BAR['gapError'] and metrics['curveError'] <= BAR['curveError']
                and min(metrics['pathwayCosine'].values()) >= BAR['pathwayCosine'])


def passesFromRelease(metrics):
    return bool(metrics['fromRelease']['gapError'] <= BAR['gapError'] and metrics['fromRelease']['curveError'] <= BAR['curveError'])


# ------------------------------------------------------------------ named partitions
shellOf = np.zeros(n, dtype=int)
for shell in range(6):
    shellOf[boundary.shellCells(shell)] = shell
leftEye, rightEye, nose, mouth = (np.array(part) for part in boundary.featureParts)
featureGroup = np.zeros(n, dtype=int)                               # 0 background interior, 1 ring, then the four features
featureGroup[RING] = 1
for code, cells in enumerate((leftEye, rightEye, nose, mouth), start=2):
    featureGroup[cells] = code
isFeature = np.isin(np.arange(n), boundary.featureCellIndices)
rows, columns = np.divmod(np.arange(n), 11)
ringSide = np.where(rows == 0, 1, 0) + np.where(rows == 10, 2, 0) + np.where(columns == 0, 4, 0) + np.where(columns == 10, 8, 0)
named = dict(
    shells6=shellOf,
    ringAndInterior2=(shellOf == 0).astype(int),
    ringSidesCornersAndInterior9=np.where(shellOf == 0, 20 + ringSide, 0),
    featureGroups6=featureGroup,
    featureGroupsAndShells9=np.where(isFeature | (shellOf == 0), featureGroup, 10 + np.minimum(shellOf, 4)))
namedMetrics = []
for name, labels in named.items():
    metrics, detail = scorer.score(coarse.compress(np.asarray(labels)), dict(name=name, family='named'), detail=True)
    namedMetrics.append(metrics)
    details[name] = detail
    say(f"{name:>30}: m={metrics['m']:>3}, gap error {metrics['gapError']:.3f}, curve error {metrics['curveError']:.3f}, "
        f"from release {metrics['fromRelease']['gapError']:.3f}/{metrics['fromRelease']['curveError']:.3f}, "
        f"floors state {metrics['stateFloorError']:.3f} source {metrics['sourceFloorError']:.3f}")

# ------------------------------------------------------------------ contiguous Ward clustering of the cells by behaviour
neighbours = sparse.lil_matrix((n, n))
for i in range(n):
    for j in range(n):
        if abs(rows[i] - rows[j]) + abs(columns[i] - columns[j]) == 1:
            neighbours[i, j] = 1
sample = np.arange(0, PEAK + 1, 5)
states = np.concatenate([difference[sample, :n] / np.sqrt((difference[sample, :n] ** 2).mean()),
                         difference[sample, n:] / np.sqrt((difference[sample, n:] ** 2).mean())], axis=0).T
shares = sweep['cellFlux'][sample, 0, :].T
shares = shares / np.sqrt((shares ** 2).mean())
featureSets = dict(state=states, both=np.concatenate([states, shares], axis=1))
clusterMetrics = []
for kind, features in featureSets.items():
    for m in args.clusterSizes:
        labels = AgglomerativeClustering(n_clusters=m, connectivity=neighbours.tocsr(), linkage='ward').fit_predict(features)
        metrics, _ = scorer.score(labels, dict(name=f'cluster_{kind}_{m}', family='clusters', features=kind))
        metrics['labels'] = labels.tolist()
        clusterMetrics.append(metrics)
        say(f"cluster on {kind:>5}, m={m:>3}: gap error {metrics['gapError']:.3f}, curve error {metrics['curveError']:.3f}, "
            f"cosines {metrics['pathwayCosine']['clear']:.2f}/{metrics['pathwayCosine']['write']:.2f}, "
            f"from release {metrics['fromRelease']['gapError']:.3f}/{metrics['fromRelease']['curveError']:.3f}, "
            f"floors state {metrics['stateFloorError']:.3f} source {metrics['sourceFloorError']:.3f}, "
            f"net retained {metrics['netRetained']['both']:.2f}, {'PASSES' if passes(metrics) else 'fails'}")

# ------------------------------------------------------------------ lattice-free benchmarks
steps = PEAK
total = np.stack([relay.total(s) for s in range(steps)])
hold = relay.hold
w = readouts[0]
scaleV = np.sqrt((difference[:PEAK + 1, :n] ** 2).mean())
scaleG = np.sqrt((difference[:PEAK + 1, n:] ** 2).mean())
scale = np.concatenate([np.full(n, 1 / scaleV), np.full(n, 1 / scaleG)])         # voltage and conductance made comparable
snapshots = difference[:PEAK + 1] * scale
_, singularValues, rightVectors = np.linalg.svd(snapshots, full_matrices=False)
energy = np.cumsum(singularValues ** 2) / np.sum(singularValues ** 2)
basisAll = rightVectors.T


pod = []
for r in args.podSizes:
    basis = basisAll[:, :r]
    record = dict(dimension=r, snapshotEnergy=float(energy[r - 1]))
    for label, start in (('whole', 0), ('fromRelease', RELEASE)):
        y = basis.T @ snapshots[start] if start else np.zeros(r)
        curve = np.zeros(PEAK + 1)
        curve[start] = (basis @ y) @ (w / scale)
        for s in range(start, steps):
            scaledJacobian = (scale[:, None] * total[s]) / scale[None, :]
            y = basis.T @ (scaledJacobian @ (basis @ y))
            if s < hold:
                y = y + basis.T @ (scale * source[s])
            curve[s + 1] = (basis @ y) @ (w / scale)
        record[label] = dict(gapError=float(abs(curve[PEAK] - gap) / abs(gap)),
                             curveError=float(np.sqrt(np.mean((curve[start:] - trueCurve[start:]) ** 2)) / abs(gap)))
    pod.append(record)
    say(f"POD r={r:>3} (snapshot energy {energy[r - 1]:.5f}): whole {record['whole']['gapError']:.3f}/{record['whole']['curveError']:.3f}, "
        f"from release {record['fromRelease']['gapError']:.3f}/{record['fromRelease']['curveError']:.3f}")

# links ranked by the size of the fine relay's transfers over the whole run
importance = np.abs(sweep['windowEdges'][0]).sum((0, 1))
np.fill_diagonal(importance, -1.0)
order = np.dstack(np.unravel_index(np.argsort(-importance, axis=None), importance.shape))[0]
pruning = []
for count in args.linkCounts:
    keep = np.eye(n, dtype=bool)
    for i, j in order[:count]:
        keep[i, j] = True
    mask = np.kron(np.ones((2, 2)), keep.astype(float))
    record = dict(links=count, fractionOfPairs=count / (n * (n - 1)))
    for label, start in (('whole', 0), ('fromRelease', RELEASE)):
        y = difference[start].copy() if start else np.zeros(2 * n)
        curve = np.zeros(PEAK + 1)
        curve[start] = w @ y
        for s in range(start, steps):
            y = (total[s] * mask) @ y
            if s < hold:
                y = y + source[s]
            curve[s + 1] = w @ y
        record[label] = dict(gapError=float(abs(curve[PEAK] - gap) / abs(gap)),
                             curveError=float(np.sqrt(np.mean((curve[start:] - trueCurve[start:]) ** 2)) / abs(gap)))
    pruning.append(record)
    say(f"links kept {count:>6} ({record['fractionOfPairs'] * 100:5.1f}%): whole {record['whole']['gapError']:.3f}/{record['whole']['curveError']:.3f}, "
        f"from release {record['fromRelease']['gapError']:.3f}/{record['fromRelease']['curveError']:.3f}")

# ------------------------------------------------------------------ summary
def smallestPassing(records, test):
    ok = [r for r in records if test(r)]
    return min(ok, key=lambda r: r['m'])['m'] if ok else None


def holdsFrom(records, test):
    """Smallest block count from which every larger tested block count passes."""
    ordered = sorted(records, key=lambda r: -r['m'])
    best = None
    for record in ordered:
        if not test(record):
            break
        best = record['m']
    return best


summary = {}
for kind in featureSets:
    subset = [m for m in clusterMetrics if m['features'] == kind]
    summary[kind] = dict(smallestPassingWholeRun=smallestPassing(subset, passes),
                         passingFromHere=holdsFrom(subset, passes),
                         smallestPassingFromRelease=smallestPassing(subset, passesFromRelease),
                         passingFromReleaseFromHere=holdsFrom(subset, passesFromRelease))
summary['pod'] = dict(
    smallestDimensionWithinBarWholeRun=next((r['dimension'] for r in pod if r['whole']['gapError'] <= 0.1 and r['whole']['curveError'] <= 0.15), None),
    smallestDimensionWithinBarFromRelease=next((r['dimension'] for r in pod if r['fromRelease']['gapError'] <= 0.1 and r['fromRelease']['curveError'] <= 0.15), None))
summary['pruning'] = dict(
    fewestLinksWithinBarWholeRun=next((r['links'] for r in pruning if r['whole']['gapError'] <= 0.1 and r['whole']['curveError'] <= 0.15), None),
    fewestLinksWithinBarFromRelease=next((r['links'] for r in pruning if r['fromRelease']['gapError'] <= 0.1 and r['fromRelease']['curveError'] <= 0.15), None))
say('summary', summary)

result = dict(
    note='EXPLORATORY. Added after the registered square tilings had been scored, at the user\'s request; none of it feeds a registered verdict.',
    bar=BAR, named=namedMetrics, clusters=clusterMetrics, pod=pod, pruning=pruning, summary=summary,
    podSnapshotEnergy={str(r): float(energy[r - 1]) for r in args.podSizes}, details=details,
    stateIndices=scorer.stateIndices, trueCurve=scorer.series(trueCurve), ringCells=RING.tolist())
json.dump(result, open(args.outputPath, 'w'))
say('wrote', args.outputPath, os.path.getsize(args.outputPath) // 1024, 'KB')
