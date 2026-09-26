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


def recut(labels):
    """The movie's frames with the cells summed over the blocks of `labels`, and how well the exactness checks hold."""
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
            frame[name] = strongest(net, cap, floor=5e-7)
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

json.dump(dict(
    note='The relay movie with the cells of each canonical square tiling summed into blocks: the exact books re-cut, not a '
         'coarse simulation. Movement inside a block is hidden; everything else is exact.',
    source=args.relayPath, fineFrames=len(cells), finalGap=finalGap, cellGross=[round(g, 5) for g in grossCells],
    checks=dict(committedMovie=committedWorst, injectionOutsideRing=strayInjection, injectionAfterHold=lateInjection,
                byResolution=everyWorst),
    resolutions=resolutions), open(args.outputPath, 'w'))
print('wrote', args.outputPath)
