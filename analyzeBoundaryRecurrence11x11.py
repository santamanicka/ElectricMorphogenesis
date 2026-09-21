"""Free or new: does a clamped run make patterns the free tissue never makes? (PolyPatterning_Sim.md, Section 12)

Reads the trajectories of simulateBoundaryRecurrence11x11.py and compares single snapshots, never averages, except in
the last check, which speaks to the averaged readouts of the earlier reports.

  libraries       the free run (every iteration of --freeIterations) and the pooled free runs from random symmetric
                  starting states (every iteration of each).
  calibration     how far one iteration moves the free pattern, and how closely the free run comes back to its own
                  earlier patterns at least --exclusionIterations later. The replay tolerance is the --tolerancePercentile
                  percentile of the shifted control's distances to the free run up to --controlIterations, where the
                  control is known to follow the free run's own path: a snapshot that close to some free snapshot counts
                  as a replay of it.
  nearest free    for every snapshot of each test run after release, the RMS distance (all 121 cells) to the nearest
                  free-run snapshot at any time, and which time that is; and to the nearest snapshot of the pooled free
                  runs (the free run and every random-start run but the test run itself, so a random-start run is judged
                  against the rest as a held-out free run). Test runs: the dial runs, the shifted control and each
                  random-start run. A snapshot that is not symmetric under the square's 8 symmetries is matched in all 8
                  of its images, since a lattice turned or mirrored is the same pattern.
  motifs          each snapshot's interior motif (cells below the single-cell saddle, -29.3 mV) against the motifs the free
                  run shows at single moments, and against those of the pooled free runs (the test run itself left out).
  cross-recurrence distances between each run's first --crossIterations iterations and the free run's first
                  2 x --crossIterations, every --crossStride iterations, for the figure.
  averages        each dial run's mean over iterations 2000-2999 against the free run's mean over every 1000-iteration
                  window, to see whether the averaged patterns of the earlier reports are the free run's averages at
                  another phase; against it, how close each free window's average comes to that of another free window
                  that does not overlap it, and the random-start runs' readout averages.
  closest match   for each dial run, its snapshot closest to any free snapshot before --frameIterations, with that
                  free snapshot, for the figure.
  traces          each run's departure from the square's symmetries, and the free run's distance to its own past,
                  every 100 iterations.
  frames          snapshots every --frameStride iterations up to --frameIterations for the animation, each with its
                  nearest-free distance and matched free time.

Writes data/boundaryRecurrenceSummary<checkpoint>Hold<hold>.json for plotBoundaryRecurrence11x11.py.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--holdIterations', type=int, default=301)
parser.add_argument('--exclusionIterations', type=int, default=300)
parser.add_argument('--settleIterations', type=int, default=5000, help='free-run iterations treated as its transient in the calibration')
parser.add_argument('--tolerancePercentile', type=float, default=95)
parser.add_argument('--controlIterations', type=int, default=10000)
parser.add_argument('--periods', type=str, default='301,2000,3000,5000,10000,20000')
parser.add_argument('--queryStride', type=int, default=1, help='snapshots of the dial runs and the control tested')
parser.add_argument('--randomQueryStride', type=int, default=10, help='snapshots of the random-start runs tested')
parser.add_argument('--crossIterations', type=int, default=5000)
parser.add_argument('--crossStride', type=int, default=20)
parser.add_argument('--frameIterations', type=int, default=5000)
parser.add_argument('--frameStride', type=int, default=50)
parser.add_argument('--windowStart', type=int, default=2000)
parser.add_argument('--windowIterations', type=int, default=1000)
args = parser.parse_args()

startTime = time.time()
size = boundary.latticeRows
interior = boundary.interiorCellIndices
directory = f'data/boundaryRecurrence{args.referenceCheckpoint}Hold{args.holdIterations}'
runs = {os.path.basename(path)[:-4]: np.load(path) for path in sorted(glob.glob(f'{directory}/*.npz'))}
free = runs['free']['vmem'].astype(np.float32)
randomNames = sorted((name for name in runs if name.startswith('freeRandom')), key=lambda name: int(name[10:]))
dialNames = sorted((name for name in runs if name.startswith('dial')), key=lambda name: float(name[4:]))
testNames = dialNames + ['freeShifted'] + randomNames
centre = np.float32(free.mean())


def log(message):
    print(f"[{time.time() - startTime:5.0f}s] {message}", flush=True)


class Library:
    """Snapshots to search, with the squared norms the nearest-neighbour search needs."""
    def __init__(self, snapshots, owner, times):
        self.snapshots = np.ascontiguousarray(snapshots - centre, dtype=np.float32)
        self.norms = (self.snapshots.astype(np.float64) ** 2).sum(1).astype(np.float32)
        self.owner, self.times = np.asarray(owner), np.asarray(times)

    def nearest(self, queries, queryTimes=None, queryOwner=None, exclusion=0):
        """RMS distance from each query to its nearest library snapshot, and that snapshot's index. Library snapshots
        of the same owner within `exclusion` iterations of the query are skipped; exclusion < 0 skips the owner."""
        queries = np.ascontiguousarray(queries - centre, dtype=np.float32)
        chunk = max(64, int(1.5e8 // len(self.times)))
        best, where = np.empty(len(queries)), np.empty(len(queries), dtype=int)
        for start in range(0, len(queries), chunk):
            block = queries[start:start + chunk]
            squared = (block.astype(np.float64) ** 2).sum(1).astype(np.float32)[:, None] + self.norms[None, :] - 2 * block @ self.snapshots.T
            if queryOwner is not None and (exclusion > 0 or exclusion < 0):
                same = self.owner[None, :] == queryOwner
                if exclusion > 0:
                    same = same & (np.abs(self.times[None, :] - queryTimes[start:start + chunk, None]) < exclusion)
                squared[np.broadcast_to(same, squared.shape)] = np.inf
            index = squared.argmin(1)
            exact = ((block.astype(np.float64) - self.snapshots[index].astype(np.float64)) ** 2).mean(1)
            best[start:start + chunk], where[start:start + chunk] = np.sqrt(exact), index
        return best, where


def images(snapshots):
    square = snapshots.reshape(-1, size, size)
    turned = [np.rot90(square, turns, axes=(1, 2)) for turns in range(4)]
    return [each.reshape(len(snapshots), -1) for each in turned + [each[:, :, ::-1] for each in turned]]


def nearestAnyImage(library, queries, queryTimes=None, queryOwner=None, exclusion=0):
    """Library.nearest, taking each query that is not symmetric in whichever of its 8 images lies nearest."""
    best, where = library.nearest(queries, queryTimes, queryOwner, exclusion)
    asymmetric = np.where(symmetryDepartures(queries) > 0.01)[0]
    if len(asymmetric):
        for image in images(queries[asymmetric])[1:]:
            distance, index = library.nearest(image, None if queryTimes is None else queryTimes[asymmetric], queryOwner, exclusion)
            better = distance < best[asymmetric]
            best[asymmetric[better]], where[asymmetric[better]] = distance[better], index[better]
    return best, where


def symmetryDepartures(snapshots):
    square = snapshots.reshape(-1, size, size)
    return np.maximum(np.abs(square - np.rot90(square, 1, axes=(1, 2))).max((1, 2)), np.abs(square - square[:, :, ::-1]).max((1, 2)))


def symmetryDeparture(course):
    square = course.reshape(-1, size, size)
    return float(max(np.abs(square - np.rot90(square, 1, axes=(1, 2))).max(), np.abs(square - square[:, :, ::-1]).max()))


def motifKeys(course):
    return [row.tobytes() for row in np.packbits(course[:, interior] < boundary.singleCellSaddleMilliVolts, axis=1)]


summary = dict(referenceCheckpoint=args.referenceCheckpoint, holdIterations=args.holdIterations, freeIterations=len(free),
               runIterations=int(len(runs[dialNames[0]]['vmem'])), exclusionIterations=args.exclusionIterations,
               settleIterations=args.settleIterations, tolerancePercentile=args.tolerancePercentile, runs={})

# ------------------------------------------------------------------------------------------ checks and calibration
symmetry = {name: symmetryDeparture(run['vmem']) for name, run in runs.items()}
log("largest departure from the square's symmetries: " + ", ".join(f"{name} {value:.1e}" for name, value in symmetry.items()))
sweep = np.load(f'data/boundaryDialSweep{args.referenceCheckpoint}Hold{args.holdIterations}.npz')
reproduction = {}
for name in dialNames:
    level = float(name[4:])
    windowMean = runs[name]['vmem'][args.windowStart:args.windowStart + args.windowIterations].astype(float).mean(0)
    reproduction[name] = float(np.abs(windowMean - sweep['windowMeanVmem'][int(round(level * 100))]).max())
log("dial runs against the dial sweep's readout (largest |difference|, mV): " + ", ".join(f"{name} {value:.1e}" for name, value in reproduction.items()))
summary['checks'] = dict(symmetryDeparture=symmetry, sweepReproduction=reproduction)

freeLibrary = Library(free, np.zeros(len(free), dtype=int), np.arange(len(free)))
perIteration = np.sqrt(((free[1:].astype(float) - free[:-1]) ** 2).mean(1))
settledTimes = np.arange(args.settleIterations, len(free), 10)
selfDistance, selfWhere = freeLibrary.nearest(free[settledTimes], settledTimes, 0, args.exclusionIterations)
returnLag = np.abs(freeLibrary.times[selfWhere] - settledTimes)
controlTimes = np.arange(args.holdIterations, args.controlIterations)
controlDistance, _ = freeLibrary.nearest(runs['freeShifted']['vmem'][controlTimes].astype(np.float32))
threshold = float(np.percentile(controlDistance, args.tolerancePercentile))
sameTime = np.sqrt(((runs['freeShifted']['vmem'].astype(np.float64) - free[:len(runs['freeShifted']['vmem'])]) ** 2).mean(1))
summary['calibration'] = dict(stepMedian=float(np.median(perIteration)), stepMax=float(perIteration.max()),
                              selfQuantiles={str(q): float(np.percentile(selfDistance, q)) for q in (5, 25, 50, 75, 95, 99, 100)},
                              returnLagMedian=float(np.median(returnLag)), tolerance=threshold,
                              controlMedian=float(np.median(controlDistance)),
                              controlSameTime=dict(times=list(range(0, len(sameTime), 100)), distance=np.round(sameTime[::100], 4).tolist()))
log(f"free run: one iteration moves the pattern {np.median(perIteration):.3f} mV (median), {perIteration.max():.2f} mV at most; "
    f"its nearest return {args.exclusionIterations}+ iterations away is {np.median(selfDistance):.3f} mV (median); "
    f"shifted control within {np.median(controlDistance):.3f} mV (median) of the free run up to iteration {args.controlIterations}, "
    f"replay tolerance ({args.tolerancePercentile:g}th percentile) {threshold:.3f} mV")
freeMotifs = set(motifKeys(free))
settledFreeMotifs = set(motifKeys(free[args.settleIterations:]))
log(f"free run: {len(freeMotifs)} distinct single-moment motifs ({len(settledFreeMotifs)} after iteration {args.settleIterations})")
summary['calibration']['freeMotifs'] = len(freeMotifs)

randomCourses = {name: runs[name]['vmem'].astype(np.float32) for name in randomNames}
pooledLibrary = Library(np.concatenate([free] + [randomCourses[name] for name in randomNames]),
                        np.concatenate([np.full(len(free), -1)] + [np.full(len(randomCourses[name]), k) for k, name in enumerate(randomNames)]),
                        np.concatenate([np.arange(len(free))] + [np.arange(len(randomCourses[name])) for name in randomNames]))
runMotifs = {name: set(motifKeys(randomCourses[name])) for name in randomNames}
log(f"pooled free library: {len(pooledLibrary.times)} snapshots from the free run and {len(randomNames)} random-start runs, "
    f"{len(freeMotifs.union(*runMotifs.values()))} distinct motifs")

# ------------------------------------------------------------------------------------------ each test run
periods = [int(value) for value in args.periods.split(',')]
for name in testNames:
    course = runs[name]['vmem'].astype(np.float32)
    release = max(int(runs[name]['holdIterations']), args.holdIterations)
    stride = args.randomQueryStride if name.startswith('freeRandom') else args.queryStride
    times = np.arange(release, len(course), stride)
    toFree, freeIndex = nearestAnyImage(freeLibrary, course[times])
    owner = randomNames.index(name) if name in randomNames else None
    pooledTimes = times[::max(1, 10 // stride)]
    toPooled, _ = nearestAnyImage(pooledLibrary, course[pooledTimes], pooledTimes, owner, exclusion=-1 if owner is not None else 0)
    pooledMotifs = freeMotifs.union(*(motifs for other, motifs in runMotifs.items() if other != name))
    keys = motifKeys(course[times])
    motifInFree = np.array([key in freeMotifs for key in keys])
    motifInPooled = np.array([key in pooledMotifs for key in keys])
    replay = toFree <= threshold
    byPeriod = []
    for low, high in zip(periods[:-1], periods[1:]):
        members, pooledMembers = (times >= low) & (times < high), (pooledTimes >= low) & (pooledTimes < high)
        if members.any():
            byPeriod.append(dict(low=low, high=high, toFreeMedian=float(np.median(toFree[members])), toFreeLow=float(np.percentile(toFree[members], 5)),
                                 replayShare=float(replay[members].mean()), toPooledMedian=float(np.median(toPooled[pooledMembers])),
                                 motifInFree=float(motifInFree[members].mean()), motifInPooled=float(motifInPooled[members].mean()),
                                 newMotifs=len({key for key, member in zip(keys, members) if member} - pooledMotifs)))
    entry = dict(release=release, stride=stride, iterations=len(course), byPeriod=byPeriod, closestToFree=float(toFree.min()),
                 replayShare=float(replay.mean()), newMotifs=len(set(keys) - freeMotifs), newMotifsPooled=len(set(keys) - pooledMotifs),
                 symmetryDeparture=symmetry[name])
    if not name.startswith('freeRandom'):
        coarse = slice(None, None, max(1, 10 // stride))
        entry['trace'] = dict(times=times[coarse].tolist(), toFree=np.round(toFree[coarse], 3).tolist(), toPooled=np.round(toPooled, 3).tolist(),
                              matchedFreeTime=freeLibrary.times[freeIndex][coarse].tolist(), motifInFree=motifInFree[coarse].astype(int).tolist(),
                              motifInPooled=motifInPooled[coarse].astype(int).tolist())
    summary['runs'][name] = entry
    log(f"{name}: closest to any free snapshot {entry['closestToFree']:.3f} mV; replays {100 * entry['replayShare']:.1f}% of the time; " + "; ".join(
        f"{p['low']}-{p['high']}: free {p['toFreeMedian']:.2f}, pooled {p['toPooledMedian']:.2f} mV, motif in free {100 * p['motifInFree']:.0f}% / pooled {100 * p['motifInPooled']:.0f}%"
        for p in byPeriod))

closest = {}
for name in dialNames:
    course = runs[name]['vmem'][args.holdIterations:args.frameIterations + 1].astype(np.float32)
    distance, where = freeLibrary.nearest(course)
    best = int(distance.argmin())
    closest[name] = dict(time=args.holdIterations + best, distance=float(distance[best]), freeTime=int(freeLibrary.times[where[best]]),
                         vmem=np.round(course[best], 1).tolist(), freeVmem=np.round(free[freeLibrary.times[where[best]]], 1).tolist())
    log(f"{name}: closest to the free run before iteration {args.frameIterations} at iteration {closest[name]['time']} "
        f"({closest[name]['distance']:.2f} mV from the free run at iteration {closest[name]['freeTime']})")
summary['closest'] = closest
traceTimes = np.arange(0, len(free), 100)
pastDistance = []
for traceTime in traceTimes:
    if traceTime < args.exclusionIterations + 100:
        pastDistance.append(None)
        continue
    pastEnd = traceTime - args.exclusionIterations
    past = Library(free[:pastEnd], np.zeros(pastEnd, dtype=int), np.arange(pastEnd))
    pastDistance.append(round(float(past.nearest(free[traceTime:traceTime + 1])[0][0]), 3))
summary['traces'] = dict(times=traceTimes.tolist(), freeToOwnPast=pastDistance,
                         symmetry={name: np.round(symmetryDepartures(run['vmem'][::100].astype(np.float64)), 4).tolist() for name, run in runs.items()})

# ------------------------------------------------------------------------------------------ cross-recurrence maps
crossTimes = np.arange(0, args.crossIterations + 1, args.crossStride)
freeTimes = np.arange(0, 2 * args.crossIterations + 1, args.crossStride)
freeBlock = free[freeTimes].astype(np.float64)
summary['cross'] = dict(runTimes=crossTimes.tolist(), freeTimes=freeTimes.tolist(), maps={})
for name in ['free'] + dialNames + ['freeShifted']:
    block = runs[name]['vmem'][crossTimes].astype(np.float64)
    distance = np.sqrt(((block[:, None, :] - freeBlock[None, :, :]) ** 2).mean(-1))
    summary['cross']['maps'][name] = np.clip(np.round(distance * 10), 0, 250).astype(int).tolist()   # tenths of a mV

# ------------------------------------------------------------------------------------------ averaged readouts
cumulative = np.concatenate([np.zeros((1, free.shape[1])), np.cumsum(free.astype(np.float64), 0)])
starts = np.arange(0, len(free) - args.windowIterations + 1)
freeWindows = (cumulative[starts + args.windowIterations] - cumulative[starts]) / args.windowIterations
sameWindowFree = freeWindows[args.windowStart]
averages = {}
for name in dialNames:
    windowMean = runs[name]['vmem'][args.windowStart:args.windowStart + args.windowIterations].astype(np.float64).mean(0)
    distance = np.sqrt(((freeWindows - windowMean) ** 2).mean(1))
    randomWindows = []
    for randomName in randomNames:
        randomCumulative = np.concatenate([np.zeros((1, size * size)), np.cumsum(randomCourses[randomName].astype(np.float64), 0)])
        randomStarts = np.arange(0, len(randomCourses[randomName]) - args.windowIterations + 1, 10)
        randomMeans = (randomCumulative[randomStarts + args.windowIterations] - randomCumulative[randomStarts]) / args.windowIterations
        randomWindows.append(np.sqrt(((randomMeans - windowMean) ** 2).mean(1)).min())
    averages[name] = dict(sameWindow=float(np.sqrt(((sameWindowFree - windowMean) ** 2).mean())), bestWindow=float(distance.min()),
                          bestStart=int(starts[distance.argmin()]), bestRandomWindow=float(min(randomWindows)),
                          profile=np.round(distance[::50], 3).tolist())
    log(f"{name}: readout average vs free average over the same window {averages[name]['sameWindow']:.2f} mV; "
        f"vs the closest free window {averages[name]['bestWindow']:.2f} mV (starting at {averages[name]['bestStart']}); "
        f"vs the closest random-start window {averages[name]['bestRandomWindow']:.2f} mV")
for name in randomNames:
    windowMean = randomCourses[name][args.windowStart:args.windowStart + args.windowIterations].astype(np.float64).mean(0)
    distance = np.sqrt(((freeWindows - windowMean) ** 2).mean(1))
    averages[name] = dict(sameWindow=float(np.sqrt(((sameWindowFree - windowMean) ** 2).mean())), bestWindow=float(distance.min()),
                          bestStart=int(starts[distance.argmin()]))
log("random-start runs' readout averages vs the closest free window: " + ", ".join(f"{averages[name]['bestWindow']:.2f}" for name in randomNames))
windowStarts = np.arange(0, len(free) - args.windowIterations + 1, 10)
windowMeans = freeWindows[windowStarts]
windowNorms = (windowMeans ** 2).sum(1)
windowDistance = np.sqrt(np.maximum(windowNorms[:, None] + windowNorms[None] - 2 * windowMeans @ windowMeans.T, 0) / (size * size))
windowDistance[np.abs(windowStarts[:, None] - windowStarts[None]) < args.windowIterations] = np.inf
freeWindowNearest = windowDistance.min(1)
log(f"each free window's average against the closest non-overlapping free window: median {np.median(freeWindowNearest):.2f} mV, "
    f"5th percentile {np.percentile(freeWindowNearest, 5):.2f}; for the window starting at {args.windowStart}: {freeWindowNearest[args.windowStart // 10]:.2f}")
freeSpread = np.sqrt(((freeWindows[args.settleIterations::50][:, None] - freeWindows[args.settleIterations::50][None]) ** 2).mean(-1))
summary['averages'] = dict(runs=averages, profileStride=50, windowIterations=args.windowIterations,
                           freeWindowNearest=dict(median=float(np.median(freeWindowNearest)), low=float(np.percentile(freeWindowNearest, 5)),
                                                  high=float(np.percentile(freeWindowNearest, 95)), atReadout=float(freeWindowNearest[args.windowStart // 10]),
                                                  values=np.round(np.sort(freeWindowNearest), 2).tolist()),
                           freeWindowSpreadMax=float(freeSpread.max()), freeWindowSpreadMedian=float(np.median(freeSpread[np.triu_indices(len(freeSpread), 1)])))
log(f"free run's own 1000-iteration averages differ by {summary['averages']['freeWindowSpreadMedian']:.2f} mV (median), "
    f"{summary['averages']['freeWindowSpreadMax']:.2f} mV at most, across its settled windows")

# ------------------------------------------------------------------------------------------ frames and galleries
frameTimes = np.arange(0, args.frameIterations + 1, args.frameStride)
frames = {}
for name in ['free'] + dialNames:
    snapshots = runs[name]['vmem'][frameTimes].astype(np.float32)
    exclusion = args.exclusionIterations if name == 'free' else 0
    distance, where = freeLibrary.nearest(snapshots, frameTimes, 0 if name == 'free' else None, exclusion)
    frames[name] = dict(vmem=np.round(snapshots, 1).tolist(), toFree=np.round(distance, 3).tolist(), matchedFreeTime=freeLibrary.times[where].tolist(),
                        motifInFree=[int(key in freeMotifs) for key in motifKeys(snapshots)])
summary['frames'] = dict(times=frameTimes.tolist(), runs=frames, holdIterations=args.holdIterations,
                         window=[args.windowStart, args.windowStart + args.windowIterations - 1])
summary['randomStarts'] = [dict(name=name, startVmem=np.round(runs[name]['startVmem'], 1).tolist(),
                                readoutVmem=np.round(randomCourses[name][args.windowStart + args.windowIterations // 2], 1).tolist(),
                                byPeriod=summary['runs'][name]['byPeriod'])
                           for name in randomNames]

outputPath = f'data/boundaryRecurrenceSummary{args.referenceCheckpoint}Hold{args.holdIterations}.json'
json.dump(summary, open(outputPath, 'w'), separators=(',', ':'))
log(f"wrote {outputPath}")
