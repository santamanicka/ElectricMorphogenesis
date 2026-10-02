"""The relay movie and its custom-layout re-cut for a steered or knocked-out ring code, not the trained one.

A trimmed sibling of analyzeBoundaryHarmonicRingOnlyRelay11x11.py's movie section and
analyzeBoundaryHarmonicMovieResolutions11x11.py, merged: same exact bookkeeping (cell-level frames capped to the
movie's top-45 field / top-30 gap-junction transfers, the square tilings and the named custom layout re-cut from
the full untruncated block nets), read from computeBoundaryHarmonicRelay11x11.py --ringCodeVariantsPath's output.
Left out on purpose, because neither means anything for a code nobody has registered predictions about: the
verdicts (R1-R5, B1-B2), the cross-check against a prior committed movie, and everything the switch-rule report
draws that is not the movie (networks, transfers, snapshots, depth diagnostic).

    python3 buildBoundaryHarmonicRelayVariantMovie11x11.py --relayPath <variantRaw.npz> --variantKey steerOrder0
                                                            --outputPath data/boundaryHarmonicRelayVariantMovie_steerOrder0....json
"""
import argparse
import json

import numpy as np

import boundaryCodeUtilities as boundary
import boundaryHarmonicCoarseGrain as coarse

parser = argparse.ArgumentParser()
parser.add_argument('--relayPath', type=str, required=True)
parser.add_argument('--variantsPath', type=str, default='data/boundaryHarmonicRingCodeVariants1888Hold301FaceMinus60Minus5.json',
                    help="the JSON listing the variant; the string 'none' names a code that is in no list, e.g. the trained code")
parser.add_argument('--variantKey', type=str, required=True)
parser.add_argument('--outputPath', type=str, required=True)
args = parser.parse_args()

relay = np.load(args.relayPath)
assert str(relay['baseline']) == 'extraUpdateOnly', relay['baseline']
names = [str(x) for x in relay['readoutNames']]
primary = names.index('selectivity')
flux, times = relay['flux'], [int(t) for t in relay['fluxTimes']]
edges, window = relay['edges'], int(relay['edgeWindow'])
D = relay['D']
hold = int(relay['hold'])
finalGap = float(relay['difference'][primary])
variant = (dict(key=args.variantKey, kind='trained') if args.variantsPath == 'none'
           else next(v for v in json.load(open(args.variantsPath))['variants'] if v['key'] == args.variantKey))
target = str(relay['targetName']) if 'targetName' in relay.files else 'face'      # a raw file with no tag is the face's

n = coarse.NUM_CELLS
GREF = 1e-9
PEAK = int(relay['primaryIteration']) + 1 if 'primaryIteration' in relay.files else coarse.PEAK   # state index of the write peak (1766 for the face)
TROUGH = int(relay['troughIteration']) + 1 if 'troughIteration' in relay.files else coarse.TROUGH
READOUT_STEP = PEAK - 1
lastWindow = READOUT_STEP // window
ring = np.array(boundary.boundaryRingCells)
interior = np.array(boundary.interiorCellIndices)
featureCells = (np.array(sorted(relay['featureCells'].tolist())) if 'featureCells' in relay.files
                else np.array(sorted(set(boundary.featureCellIndices.tolist()))))
backgroundCells = np.array([c for c in interior if c not in set(featureCells.tolist())])
FIELD_EDGES, CONTACT_EDGES = 45, 30
TOLERANCE = 1e-9


def at(state):
    return times.index(state - state % 5)


def cellWindow(w):
    startState, endState = window * w, min(window * (w + 1), READOUT_STEP)
    total = edges[primary, :, w]
    nets = [total[channel] - total[channel].T for channel in (0, 1)]
    both = total.sum(0)
    netIn = (both - both.T).sum(1)
    if w == lastWindow:
        endStock = np.zeros(n)
        endStock[featureCells] = D[PEAK, n + featureCells] / (len(featureCells) * GREF)
        endStock[backgroundCells] = -D[PEAK, n + backgroundCells] / (len(backgroundCells) * GREF)
    else:
        endStock = flux[primary, at(endState)]
    startStock = flux[primary, at(startState)]
    return nets, startStock, endStock, (endStock - startStock) - netIn


def strongest(net, cap, floor=0.0):
    listed = []
    for flat in np.argsort(-net, axis=None)[:60]:
        i, j = np.unravel_index(flat, net.shape)
        if net[i, j] <= floor or len(listed) >= cap:
            break
        listed.append([int(i), int(j), round(float(net[i, j]), 6)])
    return listed


def oneOutgoingPerBlock(net, floor=0.0):
    listed = []
    for sender in range(net.shape[1]):
        receiver = int(np.argmax(net[:, sender]))
        if net[receiver, sender] > floor:
            listed.append([receiver, sender, round(float(net[receiver, sender]), 6)])
    listed.sort(key=lambda edge: -edge[2])
    return listed


cells = [cellWindow(w) for w in range(lastWindow + 1)]
startStockCells = flux[primary, at(0)]
grossCells = [sum(float(net[net > 0].sum()) for net in nets) for nets, _, _, _ in cells]
duringHold = [window * w < hold for w in range(lastWindow + 1)]

strayInjection = max(float(np.abs(injection[np.setdiff1d(np.arange(n), ring)]).max()) for _, _, _, injection in cells)
lateInjection = max(float(np.abs(injection).max()) for w, (_, _, _, injection) in enumerate(cells) if not duringHold[w])
print(f'{args.variantKey}: injection outside the ring, worst window: {strayInjection:.2e}; '
      f'after the hold, worst cell: {lateInjection:.2e} (of a gap {finalGap:.3f})', flush=True)


def recut(labels, oneOutgoing=False, member=None):
    if member is None:
        member = coarse.indicator(labels)
    assert np.allclose(member.sum(1), 1.0) and (member.sum(0) > 0).all()
    worst = dict(inflow=0.0, stockSum=0.0, movement=0.0, offRingInjection=0.0, lateInjection=0.0)
    hasRing = (member[ring].sum(0) > 0)
    injectedTotal, frames = 0.0, []
    for w, (nets, before, after, injection) in enumerate(cells):
        blockNets = [member.T @ net @ member for net in nets]
        for net in blockNets:
            np.fill_diagonal(net, 0.0)
        frame = {}
        for channel, name, cap in ((0, 'field', FIELD_EDGES), (1, 'contact', CONTACT_EDGES)):
            net = blockNets[channel]
            frame[name] = oneOutgoingPerBlock(net, floor=5e-7) if oneOutgoing else strongest(net, cap, floor=5e-7)
            frame['gross' + name.capitalize()] = round(float(net[net > 0].sum()), 5)
            worst['movement'] = max(worst['movement'], float(net[net > 0].sum() - nets[channel][nets[channel] > 0].sum()))
        blockBefore, blockAfter, blockInjection = member.T @ before, member.T @ after, member.T @ injection
        inflow = (blockNets[0] + blockNets[1]).sum(1)
        worst['inflow'] = max(worst['inflow'], float(np.abs(blockAfter - blockBefore - inflow - blockInjection).max()))
        worst['stockSum'] = max(worst['stockSum'], abs(float(blockAfter.sum() - after.sum())))
        worst['offRingInjection'] = max(worst['offRingInjection'], float(np.abs(blockInjection[~hasRing]).max()) if (~hasRing).any() else 0.0)
        if not duringHold[w]:
            worst['lateInjection'] = max(worst['lateInjection'], float(np.abs(blockInjection).max()))
        injectedTotal += float(blockInjection.sum())
        frame['stock'] = [round(float(v), 5) for v in blockAfter]
        frame['injected'] = [round(float(v), 5) for v in blockInjection]
        frames.append(frame)
    worst['injectedMinusGap'] = injectedTotal - finalGap
    assert worst['inflow'] < TOLERANCE and worst['stockSum'] < TOLERANCE and worst['movement'] < TOLERANCE, (args.variantKey, worst)
    assert abs(worst['injectedMinusGap']) < 1e-4 * abs(finalGap), (args.variantKey, worst)
    return frames, [round(float(v), 5) for v in member.T @ startStockCells], worst


def visibleShare(frames):
    visible = [f['grossField'] + f['grossContact'] for f in frames]
    return dict(overall=float(sum(visible) / sum(grossCells)), byWindow=[round(float(v / g), 4) for v, g in zip(visible, grossCells)])


resolutions, everyWorst = [], {}
for size in range(2, 7):
    tiling = next(t for t in coarse.squareTilings(size) if t['canonical'])
    labels = tiling['labels']
    frames, start, worst = recut(labels)
    rowBands, columnBands = coarse.bandLengths(size, tiling['shortRow']), coarse.bandLengths(size, tiling['shortColumn'])
    share = visibleShare(frames)
    key = f'squares{size}'
    everyWorst[key] = worst
    resolutions.append(dict(key=key, size=size, blocks=int(labels.max()) + 1, rowBands=rowBands, columnBands=columnBands,
                            labels=[int(x) for x in labels], startStock=start, frames=frames, visible=share))

if target == 'face':
    customWeight, customLabels, customNames = coarse.namedRegionLabels(ring, boundary.featureParts)
    mirrorNames = ('background top-left', 'background top-right', 'background bottom-left', 'background bottom-right')
elif target in ('stripesInterior', 'doubleStripesInterior'):       # the same ten blocks; only the cells that are the target differ
    customWeight, customLabels, customNames = coarse.stripeRegionLabels(ring, boundary.stripeParts)
    mirrorNames = ('left flank upper', 'right flank upper', 'left flank lower', 'right flank lower')
else:
    raise ValueError(f'unknown target {target!r}')
splitCells = [dict(cell=int(cell), left=int(blocks[0]), right=int(blocks[1]))
              for cell in range(n) for blocks in [np.flatnonzero(customWeight[cell])] if len(blocks) == 2]
frames, start, worst = recut(customLabels, oneOutgoing=True, member=customWeight)
share = visibleShare(frames)
everyWorst['custom'] = worst
for frame in frames:
    for name in ('field', 'contact'):
        senders = [edge[1] for edge in frame[name]]
        assert len(senders) == len(set(senders)), (args.variantKey, name, frame[name])
tlIndex, trIndex, blIndex, brIndex = (customNames.index(name) for name in mirrorNames)
mirrorGap = dict(
    topLeftVsRight=[round(f['stock'][tlIndex] - f['stock'][trIndex], 6) for f in frames],
    bottomLeftVsRight=[round(f['stock'][blIndex] - f['stock'][brIndex], 6) for f in frames])
resolutions.append(dict(key='custom', size=None, blocks=len(customNames), names=customNames, rowBands=None, columnBands=None,
                        labels=[int(x) for x in customLabels], startStock=start, frames=frames, visible=share, mirrorGap=mirrorGap,
                        splitCells=splitCells))
print(f'{args.variantKey}: custom layout built, movement between blocks {share["overall"]:.1%} of movement between cells', flush=True)

# ------------------------------------------------------------------ the movie itself: same shape as the trained code's
deviation = D[:, n:].astype(float) / GREF


def readoutParts(state):
    return dict(face=round(float(deviation[state, featureCells].mean()), 5),
                background=round(float(-deviation[state, backgroundCells].mean()), 5))


movieFrames = []
for w in range(lastWindow + 1):
    nets, startStock, endStock, injection = cells[w]
    total = edges[primary, :, w]
    frame = dict(start=window * w, end=min(window * w + window - 1, READOUT_STEP))
    for channel, name in ((0, 'field'), (1, 'contact')):
        net = nets[channel]
        frame[name] = strongest(net, FIELD_EDGES if channel == 0 else CONTACT_EDGES, floor=0.0)
        frame['gross' + name.capitalize()] = round(float(net[net > 0].sum()), 5)
    groups3 = dict(face=featureCells, background=backgroundCells, ring=ring)
    both = total.sum(0)
    netIn = (both - both.T).sum(1)
    frame['netIn'] = {g: round(float(netIn[idx].sum()), 5) for g, idx in groups3.items()}
    frame['injected'] = {g: round(float(injection[idx].sum()), 5) for g, idx in groups3.items()}
    frame['stock'] = [round(float(v), 5) for v in endStock]
    frame['readout'] = readoutParts(PEAK if w == lastWindow else min(window * (w + 1), READOUT_STEP))
    movieFrames.append(frame)
movie = dict(frames=movieFrames, startStock=[round(float(v), 5) for v in startStockCells],
             startReadout=readoutParts(0), startIteration=0, holdEndIteration=hold - 1,
             troughIteration=TROUGH - 1, readoutIteration=READOUT_STEP)
print(f'{args.variantKey}: movie built, {len(movieFrames)} frames, iterations {movieFrames[0]["start"]}-{movieFrames[-1]["end"]}, '
      f'selectivity gap {finalGap:+.4f}', flush=True)

json.dump(dict(variant=variant, baseline='extraUpdateOnly', difference={nm: float(d) for nm, d in zip(names, relay['difference'])},
               movie=movie, resolutions=resolutions), open(args.outputPath, 'w'))
print('wrote', args.outputPath, flush=True)
