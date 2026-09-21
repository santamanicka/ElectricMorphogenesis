"""Free or new, for the tilted dial (PolyPatterning_Sim.md, Section 12): are the tilt codes' patterns the free tissue's
or the dial alone's own patterns at another time?

Reads the tilted run set of simulateBoundaryRecurrence11x11.py (--runSet tilted: the typical code of each Tilted Dial
gallery motif, the dial alone at each of their DCs, and free runs from random starting states with only the symmetry a
tilt code keeps) and the dial run set (the free run, the dial runs and the random-start runs with all 8 symmetries).
Single snapshots throughout; the replay tolerance is the dial-set analysis's (analyzeBoundaryRecurrence11x11.py).

  free            each tilt run's distance to the nearest free-run snapshot at any time, after release.
  dial alone      its distance to the nearest snapshot of any dial-alone run (every DC run, any time), and to its own
                  twin's at the same DC.
  pooled free     its distance to the nearest snapshot of any free run (the free run and every random-start run of any
                  symmetry), against held-out random-start runs of the tilt code's own symmetry as the null.
  motifs          for each gallery motif, which the Tilted Dial report read from the 2000-2999 average: whether any single
                  snapshot of the dial-alone runs, or of the free runs, shows it; and each tilt run's readout snapshots'
                  motifs against those of the dial-alone runs and the free runs.
Snapshots that are not symmetric are matched in all 8 of their images.

Writes data/boundaryTiltRecurrenceSummary<checkpoint>Hold<hold>.json for plotBoundaryRecurrence11x11.py.
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
parser.add_argument('--windowStart', type=int, default=2000)
parser.add_argument('--windowIterations', type=int, default=1000)
parser.add_argument('--pooledStride', type=int, default=10)
parser.add_argument('--exampleIteration', type=int, default=2500)
args = parser.parse_args()

startTime = time.time()
size = boundary.latticeRows
interior = boundary.interiorCellIndices
base = f'data/boundaryRecurrence{args.referenceCheckpoint}Hold{args.holdIterations}'
load = lambda directory: {os.path.basename(path)[:-4]: np.load(path) for path in sorted(glob.glob(f'{directory}/*.npz'))}
dialSet, tiltSet = load(base), load(base + 'Tilted')
dialSummary = json.load(open(f'data/boundaryRecurrenceSummary{args.referenceCheckpoint}Hold{args.holdIterations}.json'))
tolerance = dialSummary['calibration']['tolerance']
free = dialSet['free']['vmem'].astype(np.float32)
centre = np.float32(free.mean())
tiltNames = [name for name in tiltSet if name.startswith('tilt')]
dialRuns = {name: run for name, run in list(dialSet.items()) + list(tiltSet.items()) if name.startswith('dial')}
classOf = {0.0: 'Mirror', 45.0: 'Diagonal', 22.5: 'Asymmetric'}
classRuns = {symmetry: sorted((name for name in tiltSet if name.startswith(f'free{symmetry}')), key=lambda name: int(name[4 + len(symmetry):]))
             for symmetry in ('Mirror', 'Diagonal', 'Asymmetric')}
squareRuns = [name for name in dialSet if name.startswith('freeRandom')]


def log(message):
    print(f"[{time.time() - startTime:5.0f}s] {message}", flush=True)


class Library:
    def __init__(self, snapshots, owner):
        self.snapshots = np.ascontiguousarray(snapshots - centre, dtype=np.float32)
        self.norms = (self.snapshots.astype(np.float64) ** 2).sum(1).astype(np.float32)
        self.owner = np.asarray(owner)

    def nearest(self, queries, skipOwner=None):
        """RMS distance from each query (in whichever of its 8 images lies nearest) to its nearest library snapshot."""
        best, where = np.full(len(queries), np.inf), np.zeros(len(queries), dtype=int)
        square = queries.reshape(-1, size, size)
        turned = [np.rot90(square, turns, axes=(1, 2)) for turns in range(4)]
        imageSet = [each.reshape(len(queries), -1) for each in turned + [each[:, :, ::-1] for each in turned]]
        asymmetric = np.maximum(np.abs(square - turned[1]).max((1, 2)), np.abs(square - square[:, :, ::-1]).max((1, 2))) > 0.01
        chunk = max(64, int(1.5e8 // len(self.owner)))
        for k, image in enumerate(imageSet):
            members = np.arange(len(queries)) if k == 0 else np.where(asymmetric)[0]
            block = np.ascontiguousarray(image[members] - centre, dtype=np.float32)
            for start in range(0, len(block), chunk):
                part = block[start:start + chunk]
                squared = (part.astype(np.float64) ** 2).sum(1).astype(np.float32)[:, None] + self.norms[None] - 2 * part @ self.snapshots.T
                if skipOwner is not None:
                    squared[:, self.owner == skipOwner] = np.inf
                index = squared.argmin(1)
                exact = np.sqrt(((part.astype(np.float64) - self.snapshots[index]) ** 2).mean(1))
                rows = members[start:start + chunk]
                better = exact < best[rows]
                best[rows[better]], where[rows[better]] = exact[better], index[better]
        return best, where


def motifKey(snapshot):
    return np.packbits(np.asarray(snapshot)[interior] < boundary.singleCellSaddleMilliVolts).tobytes()


def canonicalMotifs(course):
    """Every motif in a course, in all 8 images, so a motif counts as shown whichever way round it appears."""
    keys = set()
    dark = (course.reshape(-1, size, size) < boundary.singleCellSaddleMilliVolts)[:, 1:-1, 1:-1]
    for turns in range(4):
        for flip in (False, True):
            image = np.rot90(dark, turns, axes=(1, 2))
            image = image[:, :, ::-1] if flip else image
            keys.update(row.tobytes() for row in np.packbits(image.reshape(len(dark), -1), axis=1))
    return keys


def interiorKey(snapshot):
    return np.packbits((np.asarray(snapshot).reshape(size, size) < boundary.singleCellSaddleMilliVolts)[1:-1, 1:-1].reshape(-1)).tobytes()


freeLibrary = Library(free, np.zeros(len(free), dtype=int))
dialNames = sorted(dialRuns, key=lambda name: float(name[4:]))
dialLibrary = Library(np.concatenate([dialRuns[name]['vmem'][args.holdIterations:] for name in dialNames]),
                      np.concatenate([[k] * (len(dialRuns[name]['vmem']) - args.holdIterations) for k, name in enumerate(dialNames)]))
poolNames = ['free'] + squareRuns + [name for runs in classRuns.values() for name in runs]
poolCourses = {name: (dialSet.get(name) or tiltSet.get(name))['vmem'] for name in poolNames}
pooledLibrary = Library(np.concatenate([poolCourses[name] for name in poolNames]),
                        np.concatenate([[k] * len(poolCourses[name]) for k, name in enumerate(poolNames)]))
log(f"libraries: free {len(free)}, dial alone {len(dialLibrary.owner)} snapshots from {len(dialNames)} runs, pooled free {len(pooledLibrary.owner)} from {len(poolNames)} runs")
dialMotifs = set().union(*(canonicalMotifs(dialRuns[name]['vmem'][args.holdIterations:]) for name in dialNames))
freeMotifs = {name: canonicalMotifs(course) for name, course in poolCourses.items()}
allFreeMotifs = set().union(*freeMotifs.values())
log(f"distinct single-moment motifs: dial alone {len(dialMotifs)}, free runs {len(allFreeMotifs)}")

gradientSummary = json.load(open(f'data/boundaryGradientLandscapeSummary{args.referenceCheckpoint}Hold{args.holdIterations}.json'))
galleryMotif = {}
for entry in gradientSummary['motifs']['gallery']:
    e = entry['example']
    galleryMotif[f"tilt{e['direction']:g}_{e['dial']:g}_{e['gradient']:g}"] = (entry, interiorKey(e['pattern']))

window = slice(args.windowStart, args.windowStart + args.windowIterations)
rows = []
for name in tiltNames:
    run = tiltSet[name]
    course = run['vmem'].astype(np.float32)
    after = course[args.holdIterations:]
    phi, dial, gradient = float(run['gradientDirection']), float(run['dialLevel']), float(run['gradientStrength'])
    symmetry = classOf[phi]
    toFree, freeIndex = freeLibrary.nearest(after)
    toDial, dialIndex = dialLibrary.nearest(after)
    twin = dialNames.index(f'dial{dial:g}')
    toTwin, _ = Library(dialRuns[f'dial{dial:g}']['vmem'][args.holdIterations:], np.zeros(len(after), dtype=int)).nearest(after)
    readout = course[window]
    toPooled, pooledIndex = pooledLibrary.nearest(readout[::args.pooledStride])
    readoutMotifs = [key for key in (interiorKey(snapshot) for snapshot in readout)]
    averagedEntry, averagedKey = galleryMotif[name]
    example = course[args.exampleIteration]
    exampleDial, exampleDialIndex = dialLibrary.nearest(example[None])
    examplePooled, examplePooledIndex = pooledLibrary.nearest(example[None])
    rows.append(dict(
        name=name, direction=phi, dial=dial, gradient=gradient, symmetry=symmetry, galleryRuns=averagedEntry['runs'],
        isDialMotif=averagedEntry['isDialMotif'],
        freeClosest=float(toFree.min()), freeReplayShare=float((toFree <= tolerance).mean()),
        dialClosest=float(toDial.min()), dialReplayShare=float((toDial <= tolerance).mean()), twinClosest=float(toTwin.min()),
        readoutToFree=float(np.median(toFree[args.windowStart - args.holdIterations:args.windowStart - args.holdIterations + args.windowIterations])),
        readoutToDial=float(np.median(toDial[args.windowStart - args.holdIterations:args.windowStart - args.holdIterations + args.windowIterations])),
        readoutToPooled=float(np.median(toPooled)),
        readoutMotifInDial=float(np.mean([key in dialMotifs for key in readoutMotifs])),
        readoutMotifInFree=float(np.mean([key in allFreeMotifs for key in readoutMotifs])),
        averagedMotifInDial=averagedKey in dialMotifs, averagedMotifInFree=averagedKey in allFreeMotifs,
        averagedMotifInOwnRun=averagedKey in canonicalMotifs(course[args.holdIterations:]),
        example=dict(iteration=args.exampleIteration, vmem=np.round(example, 1).tolist(),
                     dial=np.round(dialLibrary.snapshots[exampleDialIndex[0]] + centre, 1).tolist(), dialDistance=float(exampleDial[0]),
                     dialRun=dialNames[dialLibrary.owner[exampleDialIndex[0]]],
                     pooled=np.round(pooledLibrary.snapshots[examplePooledIndex[0]] + centre, 1).tolist(), pooledDistance=float(examplePooled[0]),
                     pooledRun=poolNames[pooledLibrary.owner[examplePooledIndex[0]]])))
    r = rows[-1]
    log(f"{name} ({symmetry}): closest to the free run {r['freeClosest']:.2f} mV ({100 * r['freeReplayShare']:.1f}% replay); to the dial alone "
        f"{r['dialClosest']:.2f} mV ({100 * r['dialReplayShare']:.1f}%), own twin {r['twinClosest']:.2f}; readout: free {r['readoutToFree']:.2f}, "
        f"dial {r['readoutToDial']:.2f}, pooled {r['readoutToPooled']:.2f} mV; readout motifs seen in the dial alone {100 * r['readoutMotifInDial']:.0f}%, "
        f"free runs {100 * r['readoutMotifInFree']:.0f}%; averaged gallery motif shown at a moment by the dial alone {r['averagedMotifInDial']}, "
        f"free runs {r['averagedMotifInFree']}, its own run {r['averagedMotifInOwnRun']}")

# null: held-out random-start runs of each symmetry against the rest of the pool, in the readout window
null = {}
for symmetry, names in classRuns.items():
    values, motifShares = [], []
    for name in names:
        readout = poolCourses[name][window].astype(np.float32)
        values.append(float(np.median(pooledLibrary.nearest(readout[::args.pooledStride], skipOwner=poolNames.index(name))[0])))
        others = set().union(*(motifs for other, motifs in freeMotifs.items() if other != name))
        motifShares.append(float(np.mean([interiorKey(snapshot) in others for snapshot in readout])))
    null[symmetry] = dict(readoutToPooled=values, readoutMotifInFree=motifShares)
    log(f"null, {symmetry} random starts held out: readout median distance to the pooled free runs {np.median(values):.2f} mV "
        f"(range {min(values):.2f}-{max(values):.2f}); readout motifs seen in the other free runs {100 * np.median(motifShares):.0f}% (median)")

summary = dict(tolerance=tolerance, rows=rows, null=null, dialNames=dialNames, dialMotifs=len(dialMotifs), freeMotifs=len(allFreeMotifs),
               poolRuns=len(poolNames), window=[args.windowStart, args.windowStart + args.windowIterations - 1])
outputPath = f'data/boundaryTiltRecurrenceSummary{args.referenceCheckpoint}Hold{args.holdIterations}.json'
json.dump(summary, open(outputPath, 'w'), separators=(',', ':'))
log(f"wrote {outputPath}")
