"""Does a code keep its tissue in a family of patterns long after the hold? EXPLORATORY: nothing here is registered. The measures and the controls below were written down before the first run.

The question (a "priming" or "constraint" sense of canalization): the stripe code seems to keep generating stripe-like patterns and the face code face-like ones, visiting them every now and then. Both trained
codes were selected on the first 3,000 iterations after release (the training score stops at 2,999), so everything after 3,000 is held out. Three things make a fair test:

  64-bit    the model runs in float64 (boundaryCodeUtilities.fullDoublePrecision); in 32-bit the face code's tissue turns lopsided from rounding alone (analyzeBoundaryHarmonicMirrorSymmetry11x11.py).
  symmetry  a code's mirror symmetries are kept by the dynamics for ever (the lattice is symmetric), so "the code's trajectory is symmetric" is no finding. Controls therefore have the same symmetry as the code they
            are compared with: stripe-class controls have a tilt of exactly 0 (both mirrors), face-class controls are left-right symmetric (orders 0-3), both drawn from the space-filling kinds of the 400-code sweeps.
  windows   in-sample = iterations 301-2,999; held out = 3,000-9,999 and 10,000-19,999.

Codes (each run for --horizon iterations, interior dark set stored every --stride iterations): the two trained codes; --controls symmetry-matched random codes per class; --copies copies of each trained code with 1% multiplicative
noise on every coefficient (symmetry kept); the tissue with no ring held.

Measures, all computed from the interior's 81 dark / light cells (below -34.6 mV):
  exact       the plain IoU with the target (stripe: the 27-cell centre stripe; face: the 14 feature cells), counted as a visit at 0.5 or more: the narrow reading of "visits the pattern".
  family      the broad reading, fixed before the first run (the stripe family to include two, three or more stripes, the face family to include deformed faces):
              stripe family: at least 6 dark cells, and at least 80% of them lie in "bars", the 4-connected dark components that are at least 3 cells tall, at least twice as tall as wide, and at least 80% filled.
              The number of bars (one, two, three, four or more) is recorded.
              face family: the IoU with some deformed face is at least 0.8, a deformed face being any union of at least two of the four face blocks (left eye, right eye, nose, mouth), each block moved independently
              by up to one cell in either direction (so the features sit slightly differently from one another).
              Reported per window and per code: share of frames, episodes (runs of such frames, gaps of up to 4 frames joined) per 1,000 iterations, and mean dwell. The trained code's share is compared with the matched
              controls' (how many controls reach it) and with the other family (a double dissociation).
  realm       one-nearest-neighbour identification of the code from a single frame (Hamming distance, random tie-breaking), library = frames of the in-sample window, test = frames of a later window, among the trained code and its
              matched controls (chance = 1 / number of classes); and the share of the noisy copies' frames identified as their parent. This says whether each code has a recognisable territory that persists and survives small changes.
  late realm  (added after the first results, so exploratory in a second sense) the same identification with the library taken from held-out window A and the test frames from held-out window B: does each code keep
              a stationary territory of its own after the selected window, even if it is not the early one?
  time course the family share (and the number of distinct patterns, and the density of dark cells) in bins of 250 iterations up to 6,000 and of 2,000 after, for the trained code, the mean of its noisy copies and the matched controls.
  memory      the normalised Hamming distance between a noisy copy and its parent through time, against that between the parent and an unrelated control: how long the exact state is remembered, as opposed to the territory.

    python3 analyzeCanalizationTalkFamilyVisits11x11.py

Writes data/canalizationTalkFamilyTrajectories1888Hold301.npz (packed dark sets), data/canalizationTalkFamilyFlags1888Hold301.npz (per frame: number of bars, best deformed-face IoU, exact IoUs) and
data/canalizationTalkFamilyVisits1888Hold301.json (never overwriting); with the trajectory npz present, only the analysis is redone into a new --outputPath.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

import boundaryCodeUtilities as boundary
from canalizationTalkCommon import *
import canalizationTalkCommon as common

parser = argparse.ArgumentParser()
parser.add_argument('--horizon', type=int, default=20000)
parser.add_argument('--stride', type=int, default=5)
parser.add_argument('--controls', type=int, default=24)
parser.add_argument('--copies', type=int, default=12)
parser.add_argument('--noise', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=20261004)
parser.add_argument('--trajectoryPath', type=str, default='data/canalizationTalkFamilyTrajectories1888Hold301.npz')
parser.add_argument('--outputPath', type=str, default='data/canalizationTalkFamilyVisits1888Hold301.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
rng = np.random.default_rng(args.seed)
interior = np.array(INTERIOR)
started = time.time()


def say(*message):
    print(f'[{time.time() - started:5.0f}s]', *message, flush=True)


# ------------------------------------------------------------------------------------------------------------------ the runs
def buildCodes():
    codes = []
    for key in ('stripe', 'face'):
        codes.append(dict(name=f'{key}Trained', group=f'{key}Trained', coefficients=TARGETS[key]['coefficients'].copy(), held=True))
    stripeSweep = json.load(open('data/relayLoopSweepNets1888Hold301StripesInteriorMinus60Minus5.json'))['codes']
    faceSweep = json.load(open('data/relayLoopSweepNets1888Hold301FaceMinus60Minus5.json'))['codes']
    stripePool = [np.array([c['multipliers'][0], 0.0, c['multipliers'][2]]) for c in stripeSweep.values() if c['kind'] == 'sweepGlobal']
    facePool = [np.array(c['multipliers']) * TARGETS['face']['coefficients'] for c in faceSweep.values() if c['kind'] == 'sweepGlobal']
    for group, pool in (('stripeControl', stripePool), ('faceControl', facePool)):
        for number, index in enumerate(rng.choice(len(pool), args.controls, replace=False)):
            codes.append(dict(name=f'{group}{number:02d}', group=group, coefficients=pool[index], held=True))
    for key in ('stripe', 'face'):
        for number in range(args.copies):
            noisy = TARGETS[key]['coefficients'] * (1 + args.noise * rng.standard_normal(len(TARGETS[key]['coefficients'])))
            codes.append(dict(name=f'{key}Copy{number:02d}', group=f'{key}Copy', coefficients=noisy, held=True))
    codes.append(dict(name='noRing', group='noRing', coefficients=TARGETS['face']['coefficients'] * 0, held=False))
    return codes


if os.path.exists(args.trajectoryPath):
    store = np.load(args.trajectoryPath, allow_pickle=True)
    names, groups = list(store['names']), list(store['groups'])
    dark = np.unpackbits(store['packed'], axis=2)[:, :, :81].astype(bool)                       # runs x frames x 81
    lopsided = store['lopsided']
    say('trajectories read:', dark.shape)
else:
    codes = buildCodes()
    names, groups, packed, lopsided = [], [], [], []
    with boundary.fullDoublePrecision(True):
        replayer = common.Replayer()
        for number, code in enumerate(codes):
            ceiling = 2.0
            ring = ringValuesOf(code['coefficients'], ceiling)
            frames, _ = replayer.run(ring, args.horizon, heldCells=None if code['held'] else np.array([], dtype=int))
            square = frames.reshape(-1, LATTICE, LATTICE)
            lopsided.append([float(np.abs(square - square[:, :, ::-1]).max()), float(np.abs(square - square[:, ::-1, :]).max())])
            bits = frames[::args.stride][:, interior] < THRESHOLD
            packed.append(np.packbits(bits, axis=1))
            names.append(code['name'])
            groups.append(code['group'])
            if number % 8 == 0:
                say(f'{number + 1}/{len(codes)} runs')
    packed, lopsided = np.array(packed), np.array(lopsided)
    np.savez_compressed(args.trajectoryPath, names=np.array(names), groups=np.array(groups), packed=packed, lopsided=lopsided, stride=args.stride, horizon=args.horizon)
    dark = np.unpackbits(packed, axis=2)[:, :, :81].astype(bool)
    say('trajectories written:', dark.shape)

# ------------------------------------------------------------------------------------------------------------------ families
from scipy.ndimage import label

frames = dark.shape[1]
stride = args.stride
iteration = np.arange(frames) * stride
windows = {'inSample': (301, 3000), 'heldOutA': (3000, 10000), 'heldOutB': (10000, 20000)}
interiorMask = lambda cells: np.isin(interior, cells).reshape(9, 9)
exactTargets = {'stripe': np.isin(interior, STRIPE_CELLS), 'face': np.isin(interior, FACE_CELLS)}
blocks = [interiorMask(boundary.featureParts[0]), interiorMask(boundary.featureParts[1]), interiorMask(boundary.featureParts[2]), interiorMask(boundary.featureParts[3])]


def shifted(mask, dr, dc):
    out = np.zeros_like(mask)
    rows, columns = np.where(mask)
    rows, columns = rows + dr, columns + dc
    keep = (rows >= 0) & (rows < 9) & (columns >= 0) & (columns < 9)
    out[rows[keep], columns[keep]] = True
    return out


deformedFaces = []
for subset in itertools.chain.from_iterable(itertools.combinations(range(4), r) for r in (2, 3, 4)):
    for moves in itertools.product(itertools.product((-1, 0, 1), repeat=2), repeat=len(subset)):
        deformedFaces.append(np.logical_or.reduce([shifted(blocks[b], *move) for b, move in zip(subset, moves)]).reshape(-1))
deformedFaces = np.unique(np.array(deformedFaces), axis=0).astype(np.float32)


def barCount(bits):
    """(number of bars, share of dark cells in bars) for one 81-cell pattern; a bar is a 4-connected dark component at least 3 tall, twice as tall as wide, 80% filled."""
    square = bits.reshape(9, 9)
    labelled, count = label(square)
    bars, inBars = 0, 0
    for number in range(1, count + 1):
        rows, columns = np.where(labelled == number)
        height, width = rows.max() - rows.min() + 1, columns.max() - columns.min() + 1
        if height >= 3 and height >= 2 * width and len(rows) >= 0.8 * height * width:
            bars += 1
            inBars += len(rows)
    return bars, inBars / max(bits.sum(), 1)


def frameFlags(runBits):
    """Per frame: the number of bars if the frame is in the stripe family (else 0), the best deformed-face IoU, and the exact IoUs with the two targets."""
    flat = runBits.reshape(-1, 81)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    bars = np.zeros(len(unique), int)
    for k, bits in enumerate(unique):
        if bits.sum() >= 6:
            count, share = barCount(bits)
            bars[k] = count if (count >= 1 and share >= 0.8) else 0
    unionSize = unique.sum(1).astype(np.float32)
    faceBest = np.zeros(len(unique), np.float32)
    for start in range(0, len(unique), 4000):
        part = unique[start:start + 4000].astype(np.float32)
        intersection = part @ deformedFaces.T
        union = part.sum(1)[:, None] + deformedFaces.sum(1)[None, :] - intersection
        faceBest[start:start + 4000] = (intersection / np.maximum(union, 1)).max(1)
    exact = {}
    for key, target in exactTargets.items():
        intersection = unique.astype(np.float32) @ target.astype(np.float32)
        exact[key] = (intersection / np.maximum(unionSize + target.sum() - intersection, 1))
    return bars[inverse], faceBest[inverse], {key: value[inverse] for key, value in exact.items()}


def episodes(flag, gap=4):
    """Runs of true frames, runs separated by up to `gap` frames joined; returns (starts, lengths) in frames."""
    indices = np.where(flag)[0]
    if len(indices) == 0:
        return np.array([], int), np.array([], int)
    breaks = np.where(np.diff(indices) > gap + 1)[0]
    starts = np.concatenate([[indices[0]], indices[breaks + 1]])
    ends = np.concatenate([indices[breaks], [indices[-1]]])
    return starts, ends - starts + 1


results = dict(note='EXPLORATORY; measures and controls written before the first run. See the module docstring.', horizon=args.horizon, stride=stride, windows=windows,
               lopsidedMillivolts={n: dict(leftRight=float(a), topBottom=float(b)) for n, (a, b) in zip(names, lopsided) if n.endswith('Trained') or n == 'noRing'},
               deformedFaceTemplates=int(len(deformedFaces)))
index = {name: k for k, name in enumerate(names)}
flags = {}                                                                                  # per run: bars, face IoU, exact IoUs
familyRows = {}
for k, name in enumerate(names):
    bars, faceBest, exact = frameFlags(dark[k])
    flags[name] = (bars, faceBest, exact)
    row = {}
    for family, flag in (('stripe', bars > 0), ('face', faceBest >= 0.8), ('exactStripe', exact['stripe'] >= 0.5), ('exactFace', exact['face'] >= 0.5)):
        for window, (low, high) in windows.items():
            mask = (iteration >= low) & (iteration < high)
            starts, lengths = episodes(flag & mask)
            span = (min(high, args.horizon) - low) / 1000
            row[f'{family}_{window}'] = dict(share=float((flag & mask).sum() / mask.sum()), episodesPer1000=float(len(starts) / span), meanDwell=float(lengths.mean() * stride) if len(lengths) else 0.0)
    for window, (low, high) in windows.items():                                              # how many stripes, when a frame is in the stripe family
        mask = (iteration >= low) & (iteration < high) & (bars > 0)
        row[f'barCounts_{window}'] = {label_: int((bars[mask] == n).sum() if n < 4 else (bars[mask] >= 4).sum()) for label_, n in (('one', 1), ('two', 2), ('three', 3), ('fourOrMore', 4))}
    familyRows[name] = row
results['family'] = familyRows


def rankAmongControls(own, group, family, window):
    values = np.array([familyRows[n][f'{family}_{window}']['share'] for n, g in zip(names, groups) if g == group])
    ownValue = familyRows[own][f'{family}_{window}']['share']
    return dict(trained=ownValue, controlsMedian=float(np.median(values)), controlsMax=float(values.max()), controlsAtLeastAsHigh=int((values >= ownValue).sum()), controls=len(values))


results['familyComparison'] = {
    f'{key}Code_{family}_{window}': rankAmongControls(f'{key}Trained', f'{key}Control', family, window)
    for key in ('stripe', 'face') for family in ('stripe', 'face', 'exactStripe', 'exactFace') for window in windows}

# ------------------------------------------------------------------------------------------------------------------ realm: identify the code from one frame
def identify(classNames, testNames, window, perClass=300, libraryPerClass=600, seedOffset=0, libraryWindow='inSample'):
    generator = np.random.default_rng(args.seed + seedOffset)
    inLow, inHigh = windows[libraryWindow]
    low, high = windows[window]
    libraryFrames = np.where((iteration >= inLow) & (iteration < inHigh))[0]
    testFrames = np.where((iteration >= low) & (iteration < high))[0]
    library, labels = [], []
    for label, name in enumerate(classNames):
        chosen = generator.choice(libraryFrames, min(libraryPerClass, len(libraryFrames)), replace=False)
        library.append(dark[index[name]][chosen])
        labels += [label] * len(chosen)
    library, labels = np.concatenate(library).astype(np.float32), np.array(labels)
    outcome = []
    for name in testNames:
        chosen = generator.choice(testFrames, min(perClass, len(testFrames)), replace=False)
        queries = dark[index[name]][chosen].astype(np.float32)
        distance = queries.sum(1)[:, None] + library.sum(1)[None, :] - 2 * queries @ library.T + 1e-3 * generator.random((len(queries), len(library)))
        outcome.append(labels[distance.argmin(1)])
    return outcome


realm = {}
for key in ('stripe', 'face'):
    classNames = [f'{key}Trained'] + [n for n, g in zip(names, groups) if g == f'{key}Control']
    for window in windows:
        assigned = identify(classNames, classNames, window)
        accuracy = float(np.mean([np.mean(a == k) for k, a in enumerate(assigned)]))
        trainedRecall = float(np.mean(assigned[0] == 0))
        parents = [n for n, g in zip(names, groups) if g == f'{key}Copy']
        copyAssigned = identify(classNames, parents, window, seedOffset=1)
        realm[f'{key}_{window}'] = dict(classes=len(classNames), chance=1 / len(classNames), accuracyAllClasses=accuracy, trainedCodeRecall=trainedRecall,
                                        copiesAssignedToParent=float(np.mean([np.mean(a == 0) for a in copyAssigned])), copiesAssignedToAnyControl=float(np.mean([np.mean(a != 0) for a in copyAssigned])))
results['realm'] = realm

# ------------------------------------------------------------------------------------------------------------------ memory: how long a noisy copy stays near its parent
memory = {}
bins = [(0, 301), (301, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 20000)]
for key in ('stripe', 'face'):
    parent = dark[index[f'{key}Trained']]
    copies = [dark[index[n]] for n, g in zip(names, groups) if g == f'{key}Copy']
    controls = [dark[index[n]] for n, g in zip(names, groups) if g == f'{key}Control'][:12]
    row = {}
    for low, high in bins:
        mask = (iteration >= low) & (iteration < min(high, args.horizon))
        near = np.mean([np.mean(parent[mask] != c[mask]) for c in copies])
        far = np.mean([np.mean(parent[mask] != c[mask]) for c in controls])
        row[f'{low}-{high}'] = dict(copyToParent=float(near), controlToParent=float(far))
    memory[key] = row
results['memory'] = memory
# ------------------------------------------------------------------------------------------------------------------ late realm and time course (added after the first results)
lateRealm = {}
for key in ('stripe', 'face'):
    classNames = [f'{key}Trained'] + [n for n, g in zip(names, groups) if g == f'{key}Control']
    assigned = identify(classNames, classNames, 'heldOutB', libraryWindow='heldOutA')
    parents = [n for n, g in zip(names, groups) if g == f'{key}Copy']
    copyAssigned = identify(classNames, parents, 'heldOutB', seedOffset=1, libraryWindow='heldOutA')
    lateRealm[key] = dict(classes=len(classNames), chance=1 / len(classNames), accuracyAllClasses=float(np.mean([np.mean(a == k) for k, a in enumerate(assigned)])), trainedCodeRecall=float(np.mean(assigned[0] == 0)),
                          copiesAssignedToParent=float(np.mean([np.mean(a == 0) for a in copyAssigned])))
results['lateRealm'] = lateRealm

edges = np.concatenate([np.arange(0, 6000, 250), np.arange(6000, args.horizon + 1, 2000)])
timeCourse = dict(edges=edges.tolist())
for family, pick in (('stripe', lambda n: flags[n][0] > 0), ('face', lambda n: flags[n][1] >= 0.8), ('exactStripe', lambda n: flags[n][2]['stripe'] >= 0.5), ('exactFace', lambda n: flags[n][2]['face'] >= 0.5)):
    series = {}
    for name in names:
        flag = pick(name)
        series[name] = [float(flag[(iteration >= lo) & (iteration < hi)].mean()) for lo, hi in zip(edges[:-1], edges[1:])]
    timeCourse[family] = {}
    for key in ('stripe', 'face'):
        controls = np.array([series[n] for n, g in zip(names, groups) if g == f'{key}Control'])
        copies = np.array([series[n] for n, g in zip(names, groups) if g == f'{key}Copy'])
        timeCourse[family][f'{key}Class'] = dict(trained=series[f'{key}Trained'], copiesMean=copies.mean(0).tolist(), controlsMean=controls.mean(0).tolist(), controlsMax=controls.max(0).tolist())
density = {}
for name in names:
    count = dark[index[name]].sum(1)
    density[name] = [float(count[(iteration >= lo) & (iteration < hi)].mean()) for lo, hi in ((301, 3000), (3000, 10000), (10000, 20000))]
timeCourse['darkCellsPerWindow'] = {n: density[n] for n in names if n.endswith('Trained') or n == 'noRing'}
timeCourse['darkCellsCopiesMean'] = {key: np.mean([density[n] for n, g in zip(names, groups) if g == f'{key}Copy'], axis=0).tolist() for key in ('stripe', 'face')}
timeCourse['darkCellsControlsMean'] = {key: np.mean([density[n] for n, g in zip(names, groups) if g == f'{key}Control'], axis=0).tolist() for key in ('stripe', 'face')}
timeCourse['darkCellsControlsSd'] = {key: np.std([density[n] for n, g in zip(names, groups) if g == f'{key}Control'], axis=0).tolist() for key in ('stripe', 'face')}
results['timeCourse'] = timeCourse
flagsPath = 'data/canalizationTalkFamilyFlags1888Hold301.npz'
if not os.path.exists(flagsPath):
    np.savez_compressed(flagsPath, names=np.array(names), groups=np.array(groups), stride=stride,
                        bars=np.array([flags[n][0] for n in names], dtype=np.int8), faceBest=np.array([flags[n][1] for n in names], dtype=np.float16),
                        exactStripe=np.array([flags[n][2]['stripe'] for n in names], dtype=np.float16), exactFace=np.array([flags[n][2]['face'] for n in names], dtype=np.float16))
json.dump(results, open(args.outputPath, 'w'), indent=1)
say('wrote', args.outputPath)
