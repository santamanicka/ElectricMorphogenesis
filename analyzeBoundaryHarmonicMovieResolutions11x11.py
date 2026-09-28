"""The relay movie at coarser resolutions: the exact books, summed over blocks of cells.

Not a test and not a simulation. Every frame of the relay movie (analyzeBoundaryHarmonicRingOnlyRelay11x11.py) is a
set of per-cell held shares and per-cell-pair net transfers, all read off the raw decomposition of the ring-only relay
(data/boundaryHarmonicRingOnlyRelay...Raw.npz, committed). Summing them over the cells of a block gives the block's held
share, and summing the net transfers over every pair of cells one in each of two blocks gives the net transfer between
the blocks. That re-cut of the books is exact: a block's change over a window is still what arrived from other blocks,
less what left, plus what the clamp injected, and the shares still add to the final selectivity gap. What it hides is
whatever moved between two cells of the same block, which cancels out of the sum.

This is the "aggregated relay" of the coarse-graining analysis (analyzeBoundaryHarmonicCoarseGrain11x11.py). It is NOT
the closed block model, which averages Vmem and G_pol over each block and runs the difference dynamics forward; that
model fails at every square size and is not drawn here.

Resolutions: the canonical tiling (cut from the top-left corner, the short band last) of the 11 x 11 lattice by squares
of 2, 3, 4, 5 and 6 cells. Each carries the block index of every cell, the start stock, and per window the block stocks,
the block injections, the strongest 45 field and 30 gap-junction net transfers as the movie draws them, and the gross
movement (sum of the positive net transfers) between blocks, to set against the same at cell resolution.

One more resolution, "custom", cuts the books differently: eleven named blocks (the ring's four sides, the background's
four quadrants, the eyes, the nose and the mouth; boundaryHarmonicCoarseGrain.namedRegionLabels), and instead of the
movie's top-45/top-30 cutoff, each block keeps at most one outgoing edge per channel -- its own single strongest field
transfer and single strongest gap-junction transfer -- however many blocks send transfers into it. This prunes which of
the exact transfers are drawn, not what they are: the block shares, injections and balance are exact regardless. The
four background quadrants are cut by the lattice's own middle row and column, so a quadrant and its mirror across the
vertical midline hold close to the same share throughout (mirrorGap below quotes how close).

Checks that stop the script if they fail: the cell-level construction reproduces the committed movie frame for frame;
every block's change over every window equals its net inflow plus its injection; block shares add to the cell shares in
every window; the injections add to the final gap and enter only blocks holding ring cells, and only during the hold;
movement between blocks never exceeds movement between cells.

    python3 analyzeBoundaryHarmonicMovieResolutions11x11.py
"""
import argparse
import json

import numpy as np

import boundaryCodeUtilities as boundary
import boundaryHarmonicCoarseGrain as coarse

parser = argparse.ArgumentParser()
parser.add_argument('--relayPath', type=str, default='data/boundaryHarmonicRingOnlyRelay1888Hold301FaceMinus60Minus5Raw.npz')
parser.add_argument('--movieJsonPath', type=str, default='data/boundaryHarmonicRingOnlyRelay1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicMovieResolutions1888Hold301FaceMinus60Minus5.json')
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
committed = json.load(open(args.movieJsonPath))['movie']

n = coarse.NUM_CELLS
GREF = 1e-9
PEAK = coarse.PEAK
READOUT_STEP = PEAK - 1
lastWindow = READOUT_STEP // window
ring = np.array(boundary.boundaryRingCells)
interior = np.array(boundary.interiorCellIndices)
featureCells = np.array(sorted(set(boundary.featureCellIndices.tolist())))
backgroundCells = np.array([c for c in interior if c not in set(featureCells.tolist())])
FIELD_EDGES, CONTACT_EDGES = 45, 30                                  # the movie's cap per window and channel
TOLERANCE = 1e-9                                                     # for the identities that hold by construction


def at(state):
    return times.index(state - state % 5)


def cellWindow(w):
    """What the movie's window w holds per cell (and cell pair): the net transfers of each channel, the held share at
    the window's start and end, and what the clamp injected into each cell (change minus net inflow, as the movie does)."""
    startState, endState = window * w, min(window * (w + 1), READOUT_STEP)
    total = edges[primary, :, w]
    nets = [total[channel] - total[channel].T for channel in (0, 1)]
    both = total.sum(0)
    netIn = (both - both.T).sum(1)
    if w == lastWindow:                                              # the readout's own state, as in the movie
        endStock = np.zeros(n)
        endStock[featureCells] = D[PEAK, n + featureCells] / (len(featureCells) * GREF)
        endStock[backgroundCells] = -D[PEAK, n + backgroundCells] / (len(backgroundCells) * GREF)
    else:
        endStock = flux[primary, at(endState)]
    startStock = flux[primary, at(startState)]
    return nets, startStock, endStock, (endStock - startStock) - netIn


def strongest(net, cap, floor=0.0):
    """The movie's rule: the strongest positive net transfers, as [to, from, size] (to receives from from). `floor`
    drops transfers too small to survive the movie's six-decimal rounding when asked (the cell movie lists them as 0.0)."""
    listed = []
    for flat in np.argsort(-net, axis=None)[:60]:
        i, j = np.unravel_index(flat, net.shape)
        if net[i, j] <= floor or len(listed) >= cap:
            break
        listed.append([int(i), int(j), round(float(net[i, j]), 6)])
    return listed


def oneOutgoingPerBlock(net, floor=0.0):
    """The custom layout's rule: each block (a column of `net`) keeps only its own largest positive net transfer,
    however many blocks send it a transfer in return -- out-degree at most one, in-degree unrestricted. Same
    [to, from, size] shape as strongest(), so the movie draws it without any change on its side."""
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
duringHold = [window * w < hold for w in range(lastWindow + 1)]      # windows in which the clamp is still writing

# ------------------------------------------------------------------ the cell-level construction against the committed movie
identity = np.arange(n)
committedWorst = dict(stock=0.0, edges=0, gross=0.0, listLengths=0)
for w, frame in enumerate(committed['frames']):
    nets, _, after, _ = cells[w]
    committedWorst['stock'] = max(committedWorst['stock'], float(np.abs(np.array(frame['stock']) - after).max()))
    for channel, name, cap in ((0, 'field', FIELD_EDGES), (1, 'contact', CONTACT_EDGES)):
        rebuilt = strongest(nets[channel], cap)
        committedWorst['edges'] += int(rebuilt != frame[name])
        committedWorst['listLengths'] += len(rebuilt)
        committedWorst['gross'] = max(committedWorst['gross'], abs(round(float(nets[channel][nets[channel] > 0].sum()), 5)
                                                                    - frame['gross' + name.capitalize()]))
assert committedWorst['stock'] < 6e-6 and committedWorst['edges'] == 0 and committedWorst['gross'] < 1e-9, committedWorst
print('cell construction against the committed movie:', committedWorst)

strayInjection = max(float(np.abs(injection[np.setdiff1d(np.arange(n), ring)]).max()) for _, _, _, injection in cells)
lateInjection = max(float(np.abs(injection).max()) for w, (_, _, _, injection) in enumerate(cells) if not duringHold[w])
print(f'injection outside the ring, worst window: {strayInjection:.2e}; after the hold, worst cell: {lateInjection:.2e} (of a gap {finalGap:.3f})')


def recut(labels, oneOutgoing=False):
    """The movie's frames with the cells summed over the blocks of `labels`, and how well the exactness checks hold.
    With oneOutgoing, each block's displayed edges are pruned to its single strongest transfer of each channel
    (oneOutgoingPerBlock) instead of the movie's top-45/top-30 cutoff (strongest); either way the block-level nets
    computed here are the full, untruncated ones, so the balance and injection identities below do not depend on
    which edges are kept for display."""
    member = coarse.indicator(labels)                                # (cells, blocks)
    assert (member.sum(1) == 1).all() and (member.sum(0) > 0).all()
    worst = dict(inflow=0.0, stockSum=0.0, movement=0.0, offRingInjection=0.0, lateInjection=0.0)
    hasRing = (member[ring].sum(0) > 0)
    injectedTotal, frames = 0.0, []
    for w, (nets, before, after, injection) in enumerate(cells):
        blockNets = [member.T @ net @ member for net in nets]
        for net in blockNets:
            np.fill_diagonal(net, 0.0)                               # what moves inside a block cancels exactly; drop the rounding noise
        frame = {}
        for channel, name, cap in ((0, 'field', FIELD_EDGES), (1, 'contact', CONTACT_EDGES)):
            net = blockNets[channel]
            frame[name] = oneOutgoingPerBlock(net, floor=5e-7) if oneOutgoing else strongest(net, cap, floor=5e-7)
            frame['gross' + name.capitalize()] = round(float(net[net > 0].sum()), 5)
            worst['movement'] = max(worst['movement'], float(net[net > 0].sum() - nets[channel][nets[channel] > 0].sum()))
        blockBefore, blockAfter, blockInjection = member.T @ before, member.T @ after, member.T @ injection
        inflow = (blockNets[0] + blockNets[1]).sum(1)                # what arrives from other blocks, less what leaves
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
    assert worst['inflow'] < TOLERANCE and worst['stockSum'] < TOLERANCE and worst['movement'] < TOLERANCE, worst
    assert abs(worst['injectedMinusGap']) < 1e-4 * abs(finalGap), worst
    return frames, [round(float(v), 5) for v in member.T @ startStockCells], worst


def visibleShare(frames):
    """Movement between blocks over movement between cells, over the whole movie and window by window."""
    visible = [f['grossField'] + f['grossContact'] for f in frames]
    return dict(overall=float(sum(visible) / sum(grossCells)), byWindow=[round(float(v / g), 4) for v, g in zip(visible, grossCells)])


resolutions, everyWorst = [], {}
for size in range(2, 7):
    tiling = next(t for t in coarse.squareTilings(size) if t['canonical'])
    labels = tiling['labels']
    frames, start, worst = recut(labels)
    rowBands, columnBands = coarse.bandLengths(size, tiling['shortRow']), coarse.bandLengths(size, tiling['shortColumn'])
    share = visibleShare(frames)
    assert max(v for v in share['byWindow']) <= 1 + 1e-9
    key = f'squares{size}'
    everyWorst[key] = worst
    resolutions.append(dict(key=key, size=size, blocks=int(labels.max()) + 1, rowBands=rowBands, columnBands=columnBands,
                            labels=[int(x) for x in labels], startStock=start, frames=frames, visible=share))
    print(f'{size} x {size}: {len(rowBands)} x {len(columnBands)} = {labels.max() + 1} blocks, bands {rowBands}; '
          f'movement between blocks is {share["overall"]:.1%} of movement between cells overall; worst identities '
          + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()), flush=True)

# ------------------------------------------------------------------ the custom layout: named blocks, one outgoing edge each
customLabels, customNames = coarse.namedRegionLabels(ring, boundary.featureParts)
assert len(customNames) == 11 and sorted(np.bincount(customLabels).tolist()) == sorted([11, 11, 9, 9, 8, 3, 3, 18, 17, 16, 16])
frames, start, worst = recut(customLabels, oneOutgoing=True)
share = visibleShare(frames)
assert max(v for v in share['byWindow']) <= 1 + 1e-9
everyWorst['custom'] = worst
for frame in frames:                                                 # the constraint itself: at most one outgoing edge per block, per channel
    for name in ('field', 'contact'):
        senders = [edge[1] for edge in frame[name]]
        assert len(senders) == len(set(senders)), (name, frame[name])

# the four background quadrants are cut through the lattice's own middle row and column (namedRegionLabels), so a left
# quadrant and its mirror on the right should hold close to the same share throughout, up to the one background cell
# on the vertical midline that alternation could not split evenly between them (none, for the bottom pair; one cell's
# worth, for the top pair, since five such cells sit above the nose, an odd number). Not a hypothesis test: a diagnostic
# of how close the construction comes, kept in the output for the report to quote.
tlIndex, trIndex, blIndex, brIndex = (customNames.index(name) for name in
                                       ('background top-left', 'background top-right', 'background bottom-left', 'background bottom-right'))
mirrorGap = dict(
    topLeftVsRight=[round(f['stock'][tlIndex] - f['stock'][trIndex], 6) for f in frames],
    bottomLeftVsRight=[round(f['stock'][blIndex] - f['stock'][brIndex], 6) for f in frames])
worstMirrorGap = {k: max(abs(v) for v in vs) for k, vs in mirrorGap.items()}
resolutions.append(dict(key='custom', size=None, blocks=len(customNames), names=customNames, rowBands=None, columnBands=None,
                        labels=[int(x) for x in customLabels], startStock=start, frames=frames, visible=share, mirrorGap=mirrorGap))
print(f'custom: {len(customNames)} named blocks {customNames}; movement between blocks is {share["overall"]:.1%} of movement '
      f'between cells overall (each block keeps only its single strongest transfer of each channel); worst identities '
      + ', '.join(f'{k} {v:.1e}' for k, v in worst.items())
      + f'; largest gap between mirrored quadrants, over the whole movie: top {worstMirrorGap["topLeftVsRight"]:.4f}, '
      f'bottom {worstMirrorGap["bottomLeftVsRight"]:.4f} (final gap {finalGap:.3f})', flush=True)

json.dump(dict(
    note='The relay movie with the cells of each canonical square tiling summed into blocks, plus one named-block layout '
         '(namedRegionLabels): the exact books re-cut, not a coarse simulation. Movement inside a block is hidden, and for '
         'the named layout so is every outgoing edge but a block\'s single strongest one per channel; everything else is exact.',
    source=args.relayPath, fineFrames=len(cells), finalGap=finalGap, cellGross=[round(g, 5) for g in grossCells],
    checks=dict(committedMovie=committedWorst, injectionOutsideRing=strayInjection, injectionAfterHold=lateInjection,
                byResolution=everyWorst),
    resolutions=resolutions), open(args.outputPath, 'w'))
print('wrote', args.outputPath)
