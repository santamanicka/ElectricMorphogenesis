"""Free or new, for the tilted dial, from tilted starts (PolyPatterning_Sim.md, Section 12): can the free tissue reach the
tilt codes' patterns from a starting state that already carries a tilt, and does the clamp's direct imprint on the ring
matter once the hold ends?

Reads the tiltedStarts run set of simulateBoundaryRecurrence11x11.py and compares it with the tilt runs of the tilted
run set. Single snapshots unless stated; the replay tolerance is the dial-set analysis's (analyzeBoundaryRecurrence11x11.py).

  tilted starts   free runs from states with a tilt in G_pol and Vmem, of each tilt code's symmetry. For each tilt run:
                  its distance to the nearest tilted-start snapshot at any time after release; the median over its readout
                  window (2000-2999); whether its readout snapshots' motifs, and its gallery motif (read from the window
                  average), appear at any single moment of a tilted-start run or in any tilted-start run's 1000-iteration
                  window average (every --averageStride iterations). The null holds each tilted-start run out in turn.
  release         each tilt code's state at release with the ring's G_pol (RingGpol), or its G_pol and Vmem (RingAll),
                  put back to the free run's. Their course is compared with the tilt run's at the same time since release:
                  the same-time distance, and the readout window average's motif against the gallery motif. Nudge, the
                  control, keeps the whole state but raises every cell's Vmem by 1 mV. --releaseOnly redoes just this part.
Snapshots that are not symmetric are matched in all 8 of their images.

Writes data/boundaryTiltStartRecurrenceSummary<checkpoint>Hold<hold>.json for plotBoundaryRecurrence11x11.py.
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
parser.add_argument('--readoutStride', type=int, default=10)
parser.add_argument('--averageStride', type=int, default=100)
parser.add_argument('--traceStride', type=int, default=10)
parser.add_argument('--exampleIteration', type=int, default=2500)
parser.add_argument('--releaseOnly', action='store_true', help='recompute only the release comparisons, into the existing summary')
args = parser.parse_args()

startTime = time.time()
size = boundary.latticeRows
hold = args.holdIterations
base = f'data/boundaryRecurrence{args.referenceCheckpoint}Hold{hold}'
load = lambda directory: {os.path.basename(path)[:-4]: np.load(path) for path in sorted(glob.glob(f'{directory}/*.npz'))}
dialSet, tiltSet, startSet = load(base), load(base + 'Tilted'), load(base + 'TiltedStarts')
tolerance = json.load(open(f'data/boundaryRecurrenceSummary{args.referenceCheckpoint}Hold{hold}.json'))['calibration']['tolerance']
centre = np.float32(dialSet['free']['vmem'].mean())
tiltNames = [name for name in tiltSet if name.startswith('tilt')]
classOf = {0.0: 'Mirror', 45.0: 'Diagonal', 22.5: 'Asymmetric'}
startNames = {phi: sorted((name for name in startSet if name.startswith(f'freeTilt{phi:g}_')), key=lambda name: int(name.split('_')[1]))
              for phi in classOf}
allStartNames = [name for names in startNames.values() for name in names]
oldPoolNames = ['free'] + [name for name in dialSet if name.startswith('freeRandom')] + [name for name in tiltSet if name.startswith('free')]
window = slice(args.windowStart, args.windowStart + args.windowIterations)
# a free run's iteration t matches a tilt run's iteration t + hold: both start from a state at release
releaseWindow = slice(args.windowStart - hold, args.windowStart - hold + args.windowIterations)


def log(message):
    print(f"[{time.time() - startTime:5.0f}s] {message}", flush=True)


images = boundary.squareImages
Library = lambda snapshots, owner: boundary.SnapshotLibrary(snapshots, owner, centre)


def darkInterior(course):
    return (np.asarray(course).reshape(-1, size, size) < boundary.singleCellSaddleMilliVolts)[:, 1:-1, 1:-1]


def canonicalMotifs(course):
    """Every motif in a course, in all 8 images, so a motif counts as shown whichever way round it appears."""
    dark = darkInterior(course)
    return {row.tobytes() for image in images(dark) for row in np.packbits(image.reshape(len(dark), -1), axis=1)}


def interiorKey(snapshot):
    return np.packbits(darkInterior(snapshot).reshape(-1)).tobytes()


def windowAverages(course):
    """The 1000-iteration window averages of a course, one starting every --averageStride iterations."""
    running = np.concatenate([np.zeros((1, course.shape[1])), np.cumsum(course, 0, dtype=np.float64)])
    starts = np.arange(0, len(course) - args.windowIterations + 1, args.averageStride)
    return (running[starts + args.windowIterations] - running[starts]) / args.windowIterations


def pooledMotifs(names, source, skip=0):
    moments, averages = {}, {}
    for name in names:
        course = source[name]['vmem'][skip:].astype(np.float32)
        moments[name], averages[name] = canonicalMotifs(course), canonicalMotifs(windowAverages(course))
    return moments, averages


startCourses = {name: startSet[name]['vmem'].astype(np.float32) for name in allStartNames}
startLibrary = Library(np.concatenate([startCourses[name] for name in allStartNames]),
                       np.concatenate([[k] * len(startCourses[name]) for k, name in enumerate(allStartNames)]))
startMoments, startAverages = pooledMotifs(allStartNames, startSet)
oldSource = {**dialSet, **{name: run for name, run in tiltSet.items() if name.startswith('free')}}
oldMoments, oldAverages = pooledMotifs(oldPoolNames, oldSource)
anyStartMoment, anyStartAverage = set().union(*startMoments.values()), set().union(*startAverages.values())
anyOldAverage = set().union(*oldAverages.values())
log(f"tilted-start library: {len(startLibrary.owner)} snapshots from {len(allStartNames)} runs; distinct motifs at a moment {len(anyStartMoment)}, "
    f"in window averages {len(anyStartAverage)}")

gradientSummary = json.load(open(f'data/boundaryGradientLandscapeSummary{args.referenceCheckpoint}Hold{hold}.json'))
galleryMotif = {}
for entry in gradientSummary['motifs']['gallery']:
    e = entry['example']
    galleryMotif[f"tilt{e['direction']:g}_{e['dial']:g}_{e['gradient']:g}"] = interiorKey(e['pattern'])


def motifDifference(average, key):
    """Interior cells whose dark / light state differs from the motif `key`, in whichever of the 8 images differs least."""
    target = np.unpackbits(np.frombuffer(key, dtype=np.uint8))[:81].reshape(9, 9).astype(bool)
    return int(min((image != target).sum() for image in images(darkInterior(average)[0])))


releaseVariants = ('RingGpol', 'RingAll', 'Nudge')


def releaseComparisons(name, course):
    """Each release variant of a tilt run against the tilt run itself, from the moment of release."""
    after, readout, key = course[hold:], course[window], galleryMotif[name]
    releases = {}
    for variant in releaseVariants:
        other = startSet[f'release{variant}_{name[4:]}']['vmem'].astype(np.float32)
        length = min(len(other), len(after))
        sameTime = np.sqrt(((other[:length].astype(np.float64) - after[:length]) ** 2).mean(1))
        average = other[releaseWindow].mean(0)
        releases[variant] = dict(
            sameTime=np.round(sameTime[::args.traceStride], 3).tolist(),
            firstApart=int(np.argmax(sameTime > tolerance)) if (sameTime > tolerance).any() else None,
            rejoined=float((sameTime[releaseWindow] <= tolerance).mean()),
            readoutAverageDistance=float(np.sqrt(((average - readout.mean(0)) ** 2).mean())),
            motifDifference=motifDifference(average, key),
            closestToReadout=float(Library(other, np.zeros(len(other), dtype=int)).nearest(readout[::args.readoutStride])[0].min()),
            example=np.round(other[args.exampleIteration - hold], 1).tolist(), average=np.round(average, 1).tolist())
    releases['ownMotifDifference'] = motifDifference(readout.mean(0), key)
    return releases


outputPath = f'data/boundaryTiltStartRecurrenceSummary{args.referenceCheckpoint}Hold{hold}.json'
if args.releaseOnly:
    summary = json.load(open(outputPath))
    for row in summary['rows']:
        row['release'] = releaseComparisons(row['name'], tiltSet[row['name']]['vmem'].astype(np.float32))
        log(f"{row['name']}: tilt's own readout motif off the gallery motif by {row['release']['ownMotifDifference']} cells; " + '; '.join(
            f"{variant}: apart after {row['release'][variant]['firstApart']}, readout average {row['release'][variant]['readoutAverageDistance']:.2f} mV off, "
            f"motif off by {row['release'][variant]['motifDifference']} cells" for variant in releaseVariants))
    json.dump(summary, open(outputPath, 'w'), separators=(',', ':'))
    log(f"updated {outputPath}")
    raise SystemExit


rows = []
for name in tiltNames:
    run = tiltSet[name]
    course = run['vmem'].astype(np.float32)
    after = course[hold:]
    phi, dial, gradient = float(run['gradientDirection']), float(run['dialLevel']), float(run['gradientStrength'])
    toStart, _ = startLibrary.nearest(after)
    readout = course[window]
    toStartReadout, startIndex = startLibrary.nearest(readout[::args.readoutStride])
    matched = [allStartNames.index(other) for other in startNames[phi]]
    matchedShare = float(np.mean(np.isin(startLibrary.owner[startIndex], matched)))
    readoutKeys = [interiorKey(snapshot) for snapshot in readout]
    key = galleryMotif[name]
    example = course[args.exampleIteration]
    exampleDistance, exampleIndex = startLibrary.nearest(example[None])
    releases = releaseComparisons(name, course)
    rows.append(dict(
        name=name, direction=phi, dial=dial, gradient=gradient, symmetry=classOf[phi],
        startClosest=float(toStart.min()), startReplayShare=float((toStart <= tolerance).mean()),
        readoutToStart=float(np.median(toStartReadout)), readoutNearestMatchedSymmetry=matchedShare,
        readoutMotifInStart=float(np.mean([each in anyStartMoment for each in readoutKeys])),
        averagedMotifInStartMoment=key in anyStartMoment, averagedMotifInStartAverage=key in anyStartAverage,
        averagedMotifInOldAverage=key in anyOldAverage, readoutAverage=np.round(readout.mean(0), 1).tolist(),
        example=dict(iteration=args.exampleIteration, vmem=np.round(example, 1).tolist(),
                     start=np.round(startLibrary.snapshot(exampleIndex[0]), 1).tolist(), startDistance=float(exampleDistance[0]),
                     startRun=allStartNames[startLibrary.owner[exampleIndex[0]]]),
        release=releases))
    r = rows[-1]
    log(f"{name}: tilted starts closest {r['startClosest']:.2f} mV ({100 * r['startReplayShare']:.1f}% within tolerance), readout median "
        f"{r['readoutToStart']:.2f} mV ({100 * matchedShare:.0f}% nearest in its own symmetry); readout motifs seen {100 * r['readoutMotifInStart']:.0f}%; "
        f"gallery motif at a moment {r['averagedMotifInStartMoment']}, in an average {r['averagedMotifInStartAverage']} (old pool {r['averagedMotifInOldAverage']}); "
        + '; '.join(f"{variant}: apart after {releases[variant]['firstApart']}, readout average {releases[variant]['readoutAverageDistance']:.2f} mV off, "
                    f"motif off by {releases[variant]['motifDifference']} cells" for variant in releaseVariants))

null = {}
for phi, names in startNames.items():
    values, motifShares = [], []
    for name in names:
        readout = startCourses[name][window]
        values.append(float(np.median(startLibrary.nearest(readout[::args.readoutStride], skipOwner=allStartNames.index(name))[0])))
        others = set().union(*(motifs for other, motifs in startMoments.items() if other != name))
        motifShares.append(float(np.mean([interiorKey(snapshot) in others for snapshot in readout])))
    null[classOf[phi]] = dict(readoutToStart=values, readoutMotifInStart=motifShares)
    log(f"null, {classOf[phi]} tilted starts held out: readout median {np.median(values):.2f} mV (range {min(values):.2f}-{max(values):.2f}); "
        f"readout motifs seen in the other tilted starts {100 * np.median(motifShares):.0f}% (median)")

starts = {name: dict(direction=float(startSet[name]['gradientDirection']), dial=float(startSet[name]['dialLevel']),
                     gradient=float(startSet[name]['gradientStrength']), vmemLevel=float(startSet[name]['vmemLevel']),
                     vmemTilt=float(startSet[name]['vmemTilt']), startVmem=np.round(startSet[name]['startVmem'], 1).tolist(),
                     startGpol=np.round(startSet[name]['startGpol'], 3).tolist(), readoutAverage=np.round(startCourses[name][window].mean(0), 1).tolist())
          for name in allStartNames}
summary = dict(tolerance=tolerance, rows=rows, null=null, starts=starts, startMotifs=len(anyStartMoment), startAverageMotifs=len(anyStartAverage),
               window=[args.windowStart, args.windowStart + args.windowIterations - 1], traceStride=args.traceStride)
json.dump(summary, open(outputPath, 'w'), separators=(',', ':'))
log(f"wrote {outputPath}")
