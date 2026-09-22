"""Free runs, the dial alone and tilts at 0 degrees, three ways (PolyPatterning_Sim.md, Section 12): how close each
group's patterns come to the other two's, and which motifs each group shows.

Groups (runs of simulateBoundaryRecurrence11x11.py):
  tilt   the tilt codes at phi = 0 (tilted run set).
  dial   the dial alone at every DC (dial and tilted run sets).
  free   free runs whose starts keep the left-right mirror a tilt at 0 degrees keeps, in three groups by their start:
         freeSquare, all 8 symmetries of the square (the free run and the freeRandom starts); freeMirror, random with the
         mirror only (the freeMirror starts); freeTilt, a tilt at 0 degrees (freeTilt0, tiltedStarts run set). The motif
         overlaps pool them as one group, free.
Every run's snapshots are taken after the hold ends (iteration --holdIterations on) and up to --runIterations.

  distances  for each run of group A, the median over its readout window (every --readoutStride iterations) of each
             snapshot's distance to the nearest snapshot of group B at any time, leaving the run itself out.
  shares     for each run of group A, the share of its readout snapshots whose motif some other run of group B shows at
             any moment.
  motifs     the distinct motifs (interior cells below the single-cell saddle, counted once over rotations and
             reflections) each group shows at any single moment, and their overlaps; and for each tilt run, the share of
             its readout snapshots whose motif the dial alone, the free runs, both or neither show.
  examples   each tilt code at --exampleIteration beside the nearest snapshot of the dial alone and of each free group.
Snapshots that are not symmetric are matched in all 8 of their images.

Writes data/boundaryTiltThreeWaySummary<checkpoint>Hold<hold>.json for plotBoundaryRecurrence11x11.py.
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
parser.add_argument('--runIterations', type=int, default=20000)
parser.add_argument('--windowStart', type=int, default=2000)
parser.add_argument('--windowIterations', type=int, default=1000)
parser.add_argument('--readoutStride', type=int, default=10)
parser.add_argument('--exampleIteration', type=int, default=2500)
args = parser.parse_args()

startTime = time.time()
hold = args.holdIterations
base = f'data/boundaryRecurrence{args.referenceCheckpoint}Hold{hold}'
runs = {}
for directory in (base, base + 'Tilted', base + 'TiltedStarts'):
    for path in sorted(glob.glob(f'{directory}/*.npz')):
        runs[os.path.basename(path)[:-4]] = path
subgroups = dict(
    tilt=[name for name in runs if name.startswith('tilt0_')],
    dial=sorted((name for name in runs if name.startswith('dial')), key=lambda name: float(name[4:])),
    freeStandard=['free'],
    freeRandom=[name for name in runs if name.startswith('freeRandom')],
    freeMirror=[name for name in runs if name.startswith('freeMirror')],
    freeTilt=[name for name in runs if name.startswith('freeTilt0_')])
groups = dict(tilt=subgroups['tilt'], dial=subgroups['dial'], freeSquare=subgroups['freeStandard'] + subgroups['freeRandom'],
              freeMirror=subgroups['freeMirror'], freeTilt=subgroups['freeTilt'])
courses = {name: np.load(runs[name])['vmem'][hold:args.runIterations].astype(np.float32) for names in groups.values() for name in names}
centre = np.mean([course.mean() for course in courses.values()])
window = slice(args.windowStart - hold, args.windowStart - hold + args.windowIterations)


def log(message):
    print(f"[{time.time() - startTime:5.0f}s] {message}", flush=True)


libraries = {group: boundary.SnapshotLibrary(np.concatenate([courses[name] for name in names]),
                                             np.concatenate([[k] * len(courses[name]) for k, name in enumerate(names)]), centre)
             for group, names in groups.items()}
log('runs: ' + ', '.join(f'{group} {len(names)}' for group, names in subgroups.items()))

distances = {}
for fromGroup, names in groups.items():
    for toGroup, names2 in groups.items():
        values = {}
        for name in names:
            readout = courses[name][window][::args.readoutStride]
            skip = names2.index(name) if name in names2 else None
            values[name] = float(np.median(libraries[toGroup].nearest(readout, skipOwner=skip)[0]))
        distances[f'{fromGroup}>{toGroup}'] = values
        log(f"{fromGroup} -> {toGroup}: median over runs {np.median(list(values.values())):.2f} mV "
            f"(range {min(values.values()):.2f}-{max(values.values()):.2f})")

runMotifs = {name: set(boundary.interiorMotifs(course, canonical=True)) for name, course in courses.items()}
motifSets = {group: set().union(*(runMotifs[name] for name in names)) for group, names in groups.items()}
motifSets['free'] = motifSets['freeSquare'] | motifSets['freeMirror'] | motifSets['freeTilt']
motifShares = {}
for fromGroup, names in groups.items():
    for toGroup, names2 in groups.items():
        values = {}
        for name in names:
            others = set().union(*(runMotifs[other] for other in names2 if other != name))
            values[name] = float(np.mean([key in others for key in boundary.interiorMotifs(courses[name][window], canonical=True)]))
        motifShares[f'{fromGroup}>{toGroup}'] = values
        log(f"{fromGroup} -> {toGroup}: readout snapshots whose motif the other runs show, median over runs {100 * np.median(list(values.values())):.0f}% "
            f"(range {100 * min(values.values()):.0f}-{100 * max(values.values()):.0f}%)")
tilt, dial, free = (motifSets[group] for group in ('tilt', 'dial', 'free'))
venn = dict(tiltOnly=len(tilt - dial - free), dialOnly=len(dial - tilt - free), freeOnly=len(free - tilt - dial),
            tiltDial=len(tilt & dial - free), tiltFree=len(tilt & free - dial), dialFree=len(dial & free - tilt), all=len(tilt & dial & free))
perRun = {group: float(np.mean([len(runMotifs[name]) for name in names])) for group, names in groups.items()}
log(f"distinct motifs: tilt {len(tilt)}, dial {len(dial)}, free {len(free)} (square {len(motifSets['freeSquare'])}, mirror {len(motifSets['freeMirror'])}, tilted {len(motifSets['freeTilt'])}); regions {venn}; per run {perRun}")

rows = []
for name in groups['tilt']:
    course = courses[name]
    keys = boundary.interiorMotifs(course[window], canonical=True)
    inDial, inFree = np.array([key in dial for key in keys]), np.array([key in free for key in keys])
    example = course[args.exampleIteration - hold]
    row = dict(name=name, dial=float(np.load(runs[name])['dialLevel']), gradient=float(np.load(runs[name])['gradientStrength']),
               readoutMotifs=dict(both=float((inDial & inFree).mean()), dialOnly=float((inDial & ~inFree).mean()),
                                  freeOnly=float((~inDial & inFree).mean()), neither=float((~inDial & ~inFree).mean())),
               example=dict(iteration=args.exampleIteration, vmem=np.round(example, 1).tolist()))
    for group in ('dial', 'freeSquare', 'freeMirror', 'freeTilt'):
        distance, index = libraries[group].nearest(example[None])
        owner = groups[group][libraries[group].owner[index[0]]]
        row['example'][group] = dict(vmem=np.round(libraries[group].snapshot(index[0]), 1).tolist(), distance=float(distance[0]), run=owner,
                                     iteration=int(hold + index[0] - np.searchsorted(libraries[group].owner, libraries[group].owner[index[0]])))
    rows.append(row)
    log(f"{name}: readout motifs in both {100 * row['readoutMotifs']['both']:.0f}%, dial only {100 * row['readoutMotifs']['dialOnly']:.0f}%, "
        f"free only {100 * row['readoutMotifs']['freeOnly']:.0f}%, neither {100 * row['readoutMotifs']['neither']:.0f}%; example: dial "
        f"{row['example']['dial']['distance']:.2f} mV ({row['example']['dial']['run']}), free square {row['example']['freeSquare']['distance']:.2f} mV "
        f"({row['example']['freeSquare']['run']}), free mirror {row['example']['freeMirror']['distance']:.2f} mV ({row['example']['freeMirror']['run']}), "
        f"free tilted {row['example']['freeTilt']['distance']:.2f} mV ({row['example']['freeTilt']['run']})")

summary = dict(groups=groups, subgroups=subgroups, distances=distances, motifShares=motifShares, motifCounts={group: len(keys) for group, keys in motifSets.items()},
               venn=venn, motifsPerRun=perRun, rows=rows, window=[args.windowStart, args.windowStart + args.windowIterations - 1],
               snapshotRange=[hold, args.runIterations - 1])
outputPath = f'data/boundaryTiltThreeWaySummary{args.referenceCheckpoint}Hold{hold}.json'
json.dump(summary, open(outputPath, 'w'), separators=(',', ':'))
log(f"wrote {outputPath}")
