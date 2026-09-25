"""Read the relay at block resolution: group the 121 cells into blocks and re-cut or reduce the fine relay.

Built on the fine relay's per-step Jacobian (computeBoundaryHarmonicRelay11x11.py --jacobianPath). With x = (Vmem,
G_pol) per cell, D(n) the trained-minus-baseline difference and Jbar(n) the averaged step Jacobian, the fine relay is

    D(n+1) = Jbar(n) D(n) + src(n),   abar(T) = w,   abar(n) = abar(n+1) Jbar(n),
    flux_k(n) = abar(n) . D(n) restricted to cell k,      e(i<-j)(n) = abar_i(n+1) Jbar_ij(n) D_j(n).

Two objects at block resolution, for any grouping of cells into blocks:

aggregated   the same books summed over each block: block flux is the sum of its cells' fluxes, a block-to-block
             edge is the sum of the cell-to-cell edges between them. Exact, so it always sums to the fine gap;
             it only hides what happens inside a block.
reduced      a closed model on the blocks. P averages Vmem and G_pol over each block and Q copies a block value to
             its cells (P Q = I). The block difference is evolved by y(n+1) = (P Jbar(n) Q) y(n) + P src(n), read
             out with w Q, and its adjoint, flux and edges are computed from P Jbar Q exactly as the fine relay
             computes them from Jbar. Nothing is fitted; the one approximation is that a block's cells share a value.

The channel split is the fine relay's: field (start-of-step Vmem entering through the field) and contact (through
the gap junctions and the cell's own membrane, plus the conductance path), off-diagonal only.
"""
import numpy as np

ROWS = COLS = 11
NUM_CELLS = ROWS * COLS
CHANNELS = ('field', 'contact')


# ------------------------------------------------------------------ groupings of cells
def bandLengths(size, shortPosition):
    """Lengths of the bands that cut 11 rows (or columns) into bands of `size`; the one shorter band, if 11 is not a
    multiple of size, sits at index shortPosition."""
    count, remainder = divmod(ROWS, size)
    lengths = [size] * count
    if remainder:
        lengths.insert(shortPosition, remainder)
    return lengths


def labelsFromBands(rowLengths, columnLengths):
    """Block index of every cell (row-major over the block grid) for the given row and column bands."""
    rowBand = np.repeat(np.arange(len(rowLengths)), rowLengths)
    columnBand = np.repeat(np.arange(len(columnLengths)), columnLengths)
    return (rowBand[:, None] * len(columnLengths) + columnBand[None, :]).ravel()


def squareTilings(size):
    """Every way of tiling the lattice with size x size squares, the short band placed anywhere. Each is a dict with
    the block index of every cell ('labels') and where the short band sits; 'canonical' is the tiling cut from the
    top-left corner, short band last."""
    count, remainder = divmod(ROWS, size)
    positions = range(count + 1) if remainder else [0]
    tilings = []
    for shortRow in positions:
        for shortColumn in positions:
            tilings.append(dict(size=size, shortRow=shortRow, shortColumn=shortColumn,
                                canonical=bool(not remainder or (shortRow == count and shortColumn == count)),
                                labels=labelsFromBands(bandLengths(size, shortRow), bandLengths(size, shortColumn))))
    return tilings


def blockSizes(labels):
    return np.bincount(labels)


def indicator(labels):
    """(cells, blocks) matrix, 1 where a cell belongs to a block: Q, and the aggregation sum."""
    blocks = np.zeros((len(labels), labels.max() + 1))
    blocks[np.arange(len(labels)), labels] = 1.0
    return blocks


def averaging(labels):
    """(blocks, cells) matrix P that averages a per-cell vector over each block."""
    member = indicator(labels).T
    return member / member.sum(1, keepdims=True)


def compress(labels):
    """Renumber block indices to 0..m-1 in order of first appearance."""
    _, first = np.unique(labels, return_index=True)
    order = {int(labels[i]): k for k, i in enumerate(sorted(first))}
    return np.array([order[int(x)] for x in labels])


def ringSegmentLabels(length, ringCells):
    """The ring cells grouped into consecutive runs of `length` (ringCells is in order round the ring); every other
    cell is a block of its own."""
    labels = np.arange(NUM_CELLS)
    for run, start in enumerate(range(0, len(ringCells), length)):
        labels[np.asarray(ringCells)[start:start + length]] = NUM_CELLS + run
    return compress(labels)


def interiorSquareLabels(size, ringCells, shortPosition=None):
    """Every ring cell a block of its own; the 9 x 9 interior tiled by size x size squares. 9 is a multiple of 3 and
    9; otherwise the short band goes at shortPosition (default: last)."""
    interior = ROWS - 2
    count, remainder = divmod(interior, size)
    bands = [size] * count
    if remainder:
        bands.insert(count if shortPosition is None else shortPosition, remainder)
    inner = labelsFromBands(bands, bands).reshape(interior, interior)
    labels = np.arange(NUM_CELLS).reshape(ROWS, COLS)
    labels[1:-1, 1:-1] = NUM_CELLS + inner
    return compress(labels.ravel())


def randomPartition(sizes, generator):
    """Cells scattered at random into blocks of the given sizes."""
    order = generator.permutation(int(np.sum(sizes)))
    labels = np.empty(len(order), dtype=int)
    start = 0
    for block, size in enumerate(sizes):
        labels[order[start:start + size]] = block
        start += size
    return labels


# ------------------------------------------------------------------ the fine relay's inputs
class FineRelay:
    """The fine relay's per-step operators, difference and source, and the readouts to decompose.

    jacobians: (steps, 3, 2n, n) array or memmap, the role blocks (field; gap + own membrane; conductance), each
    mapping the start-of-step Vmem (first two) or G_pol (last) to the next (Vmem, G_pol).
    difference: (states, 2n), D(n) exactly. source: (hold, 2n). readouts: (K, 2n) rows, all read at state
    readoutState. phases: {name: (firstStep, lastStepExclusive)} over which edges are summed."""

    def __init__(self, jacobians, difference, source, readouts, readoutState, phases, windowLength=50):
        self.jacobians = jacobians
        self.difference = np.asarray(difference, dtype=float)
        self.source = np.asarray(source, dtype=float)
        self.readouts = np.asarray(readouts, dtype=float)
        self.readoutState = readoutState
        self.phases = phases
        self.windowLength = windowLength
        self.numCells = NUM_CELLS
        self.hold = len(self.source)
        self.numReadouts = len(self.readouts)

    def total(self, step):
        """The (2n, 2n) step Jacobian at `step`: [field + contact | conductance] columns."""
        blocks = np.asarray(self.jacobians[step])
        return np.concatenate([blocks[0] + blocks[1], blocks[2]], axis=1)

    # -------------------------------------------------------------- the fine sweep, once
    def sweep(self):
        """Adjoint at every state, per-cell flux at every state, and the cell-to-cell edges summed over each phase
        and each 50-iteration window, by channel. All from the stored Jacobians."""
        n, T, K = self.numCells, self.readoutState, self.numReadouts
        adjoint = np.zeros((T + 1, K, 2 * n))
        adjoint[T] = self.readouts
        phaseEdges = {name: np.zeros((K, 2, n, n)) for name in self.phases}
        windowEdges = np.zeros((K, 2, T // self.windowLength + 1, n, n))
        offDiagonal = 1.0 - np.eye(n)
        for step in range(T - 1, -1, -1):
            blocks = np.asarray(self.jacobians[step])
            after = adjoint[step + 1]
            aV, aG = after[:, :n], after[:, n:]
            d = self.difference[step]
            for channel, parts in ((0, ((0, d[:n]),)), (1, ((1, d[:n]), (2, d[n:])))):
                total = np.zeros((K, n, n))
                for role, source in parts:
                    block = blocks[role]
                    total += (aV[:, :, None] * block[None, :n] + aG[:, :, None] * block[None, n:]) \
                        * source[None, None, :] * offDiagonal
                windowEdges[:, channel, step // self.windowLength] += total
                for name, (first, last) in self.phases.items():
                    if first <= step < last:
                        phaseEdges[name][:, channel] += total
            adjoint[step] = after @ np.concatenate([blocks[0] + blocks[1], blocks[2]], axis=1)
        cellFlux = np.einsum('tkc,tc->tkc', adjoint, self.difference[:T + 1])
        cellFlux = cellFlux[:, :, :n] + cellFlux[:, :, n:]
        injection = np.zeros((K, n))
        for step in range(self.hold):
            after = adjoint[step + 1]
            injection += after[:, :n] * self.source[step, :n] + after[:, n:] * self.source[step, n:]
        return dict(adjoint=adjoint, cellFlux=cellFlux, phaseEdges=phaseEdges, windowEdges=windowEdges,
                    injection=injection)

    # -------------------------------------------------------------- aggregated, from the fine sweep
    @staticmethod
    def aggregate(sweep, labels):
        member = indicator(labels)
        m = member.shape[1]
        edges = {name: np.einsum('cb,kxcd,da->kxba', member, edge, member) for name, edge in sweep['phaseEdges'].items()}
        for name in edges:
            for k in range(edges[name].shape[0]):
                for channel in range(2):
                    np.fill_diagonal(edges[name][k, channel], 0.0)
        return dict(flux=sweep['cellFlux'] @ member, edges=edges, injection=sweep['injection'] @ member)

    # -------------------------------------------------------------- reduced, on the blocks
    def project(self, labels, roles=True):
        """P Jbar Q for every step: (steps, 3, 2, m, m) role blocks, indexed [role, output Vmem/G_pol, I, J], or the
        total (steps, 2m, 2m) when roles is False."""
        averagingMatrix, lift = averaging(labels), indicator(labels)
        n, m = self.numCells, lift.shape[1]
        steps = self.readoutState
        out = np.zeros((steps, 3, 2, m, m)) if roles else np.zeros((steps, 2 * m, 2 * m))
        for step in range(steps):
            blocks = np.asarray(self.jacobians[step])
            if roles:
                stacked = blocks.reshape(3, 2, n, n)
                out[step] = averagingMatrix @ (stacked @ lift)
            else:
                left = (blocks[0] + blocks[1]).reshape(2, n, n)
                right = blocks[2].reshape(2, n, n)
                a = averagingMatrix @ (left @ lift)
                b = averagingMatrix @ (right @ lift)
                out[step, :m, :m], out[step, m:, :m] = a[0], a[1]
                out[step, :m, m:], out[step, m:, m:] = b[0], b[1]
        return out

    @staticmethod
    def totalOf(roles, m):
        """(steps, 2m, 2m) total from the role blocks."""
        steps = roles.shape[0]
        total = np.zeros((steps, 2 * m, 2 * m))
        left = roles[:, 0] + roles[:, 1]
        total[:, :m, :m], total[:, m:, :m] = left[:, 0], left[:, 1]
        total[:, :m, m:], total[:, m:, m:] = roles[:, 2, 0], roles[:, 2, 1]
        return total

    def blockOperators(self, labels):
        """The pieces the reduced relay needs: P, Q, block-averaged source and difference, block readouts."""
        averagingMatrix, lift = averaging(labels), indicator(labels)
        n, m = self.numCells, lift.shape[1]

        def average(vector):                                        # (..., 2n) -> (..., 2m)
            return np.concatenate([vector[..., :n] @ averagingMatrix.T, vector[..., n:] @ averagingMatrix.T], axis=-1)

        readouts = np.concatenate([self.readouts[:, :n] @ lift, self.readouts[:, n:] @ lift], axis=1)
        return dict(m=m, average=average, blockSource=average(self.source), trueState=average(self.difference),
                    readouts=readouts)

    def reducedTrajectory(self, total, operators, start=0):
        """y(n) for n = start..readoutState, from the total block Jacobians. From start = 0 the block model is driven
        by the block-averaged source through the hold; from a later start it begins at the block average of the true
        difference at that state (states before start are left at zero) and needs no source at all once the hold is
        over, which separates coarse-graining the relay from coarse-graining what writes into it."""
        m, steps = operators['m'], total.shape[0]
        y = np.zeros((steps + 1, 2 * m))
        if start:
            y[start] = operators['trueState'][start]
        for step in range(start, steps):
            y[step + 1] = total[step] @ y[step]
            if step < self.hold:
                y[step + 1] += operators['blockSource'][step]
        return y

    def reducedReadoutCurve(self, labels, start=0):
        """Only what the accuracy scores need: the block model's readout w_c . y(n), for every readout. Cheaper than
        the whole relay, so it can be run on many groupings."""
        operators = self.blockOperators(labels)
        total = self.project(labels, roles=False)
        y = self.reducedTrajectory(total, operators, start)
        return y @ operators['readouts'].T, y

    def reducedRelay(self, labels, roles=None, start=0):
        """The reduced relay: trajectory, adjoint, per-block flux, injection, and edges summed over each phase, by
        channel, for every readout. With start > 0 only the steps from there on are carried (see reducedTrajectory)."""
        operators = self.blockOperators(labels)
        m, T, K = operators['m'], self.readoutState, self.numReadouts
        roles = self.project(labels) if roles is None else roles
        total = self.totalOf(roles, m)
        y = self.reducedTrajectory(total, operators, start)
        adjoint = np.zeros((T + 1, K, 2 * m))
        adjoint[T] = operators['readouts']
        phaseEdges = {name: np.zeros((K, 2, m, m)) for name in self.phases}
        offDiagonal = 1.0 - np.eye(m)
        injection = np.zeros((K, m))
        for step in range(T - 1, start - 1, -1):
            after = adjoint[step + 1]
            aV, aG = after[:, :m], after[:, m:]
            state = y[step]
            for channel, parts in ((0, ((0, state[:m]),)), (1, ((1, state[:m]), (2, state[m:])))):
                edge = np.zeros((K, m, m))
                for role, source in parts:
                    edge += (aV[:, :, None] * roles[step, role, 0][None] + aG[:, :, None] * roles[step, role, 1][None]) \
                        * source[None, None, :] * offDiagonal
                for name, (first, last) in self.phases.items():
                    if first <= step < last:
                        phaseEdges[name][:, channel] += edge
            if step < self.hold:
                injection += aV * operators['blockSource'][step, :m] + aG * operators['blockSource'][step, m:]
            adjoint[step] = after @ total[step]
        share = adjoint * y[:, None, :]
        flux = share[:, :, :m] + share[:, :, m:]
        return dict(y=y, adjoint=adjoint, flux=flux, edges=phaseEdges, injection=injection,
                    curve=y @ operators['readouts'].T, trueState=operators['trueState'], operators=operators)


# ------------------------------------------------------------------ what is scored
def netTransfer(edges):
    """Signed net transfer between blocks, both channels: N[I, J] = flow from J into I minus flow from I into J."""
    both = edges.sum(0)
    return both - both.T


def pathwayCosine(reducedEdges, aggregatedEdges):
    """Cosine of the two net-transfer patterns, over unordered block pairs."""
    a, b = netTransfer(reducedEdges), netTransfer(aggregatedEdges)
    upper = np.triu_indices(a.shape[0], 1)
    va, vb = a[upper], b[upper]
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / norm) if norm > 0 else float('nan')


def linksCarrying(edges, fraction=0.9):
    """Fewest directed block pairs whose positive net transfer adds up to `fraction` of the total positive net."""
    net = netTransfer(edges)
    positive = np.sort(net[net > 0])[::-1]
    if not len(positive):
        return 0
    return int(np.searchsorted(np.cumsum(positive), fraction * positive.sum()) + 1)


def resolvedReadout(labels, readoutRow):
    """Share of the readout's structure that survives blocking: sum over blocks of |summed weight| over sum of
    |weight| (1 when no block mixes weights of opposite sign)."""
    n = len(labels)
    weights = readoutRow[n:] if len(readoutRow) == 2 * n else readoutRow
    blockWeight = np.bincount(labels, weights=weights)
    return float(np.abs(blockWeight).sum() / np.abs(weights).sum())


def handOver(flux, labels, ringCells, firstState):
    """First state at or after firstState where the summed |flux| of blocks with no ring cell exceeds that of blocks
    containing one; None if there is no block without a ring cell or it never happens."""
    touchesRing = np.zeros(labels.max() + 1, dtype=bool)
    touchesRing[np.unique(labels[np.asarray(ringCells)])] = True
    if touchesRing.all():
        return None
    ring, inside = np.abs(flux[:, touchesRing]).sum(1), np.abs(flux[:, ~touchesRing]).sum(1)
    later = np.where((inside > ring) & (np.arange(len(ring)) >= firstState))[0]
    return int(later[0]) if len(later) else None


# ------------------------------------------------------------------ scoring a grouping against the fine relay
def lifted(vector, labels):
    """Q P applied to a (..., 2n) state: each block's cells replaced by the block average, Vmem and G_pol separately."""
    n = NUM_CELLS
    averagingMatrix, lift = averaging(labels), indicator(labels)
    return np.concatenate([(vector[..., :n] @ averagingMatrix.T) @ lift.T, (vector[..., n:] @ averagingMatrix.T) @ lift.T], axis=-1)


class Scorer:
    """Scores any grouping of the cells against the fine relay, for the registered measures (see the predictions file)
    and a few floors that need no dynamics: what blocking alone does to the final state and to the ring's source."""

    def __init__(self, relay, sweep, ringCells, readoutNames, release=302, trough=586, sampleEvery=5):
        self.relay, self.sweep, self.ringCells, self.readoutNames = relay, sweep, np.asarray(ringCells), readoutNames
        self.release, self.trough, self.peak = release, trough, relay.readoutState
        self.trueCurve = relay.difference[:self.peak + 1] @ relay.readouts.T           # w . D(n), every readout
        self.fineGap = self.trueCurve[self.peak]
        self.stateIndices = list(range(0, self.peak, sampleEvery)) + [self.peak]
        gross = sum(np.abs(sweep['phaseEdges'][phase][0]).sum(0) for phase in ('clear', 'write'))
        self.grossFine = gross
        net = {phase: sweep['phaseEdges'][phase][0].sum(0) for phase in ('clear', 'write')}
        self.netFine = {phase: np.abs(np.triu(net[phase] - net[phase].T, 1)).sum() for phase in net}

    def series(self, x):
        """Every fifth state and the last, rounded, for the JSON."""
        return np.round(np.asarray(x)[self.stateIndices], 6).tolist()

    def floors(self, labels):
        """Errors that blocking alone causes, before any dynamics: the final readout of the block-averaged true final
        state, and the gap the fine adjoint assigns to the block-averaged source."""
        relay, gap = self.relay, self.fineGap[0]
        stateFloor = abs(relay.readouts[0] @ lifted(relay.difference[self.peak], labels) - gap) / abs(gap)
        blocked = lifted(relay.source, labels)
        sourceGap = float(np.einsum('sc,sc->', self.sweep['adjoint'][1:relay.hold + 1, 0], blocked))
        return dict(stateFloorError=float(stateFloor), sourceFloorError=float(abs(sourceGap - gap) / abs(gap)))

    def score(self, labels, meta, detail=False, fromRelease=True):
        relay, sweep = self.relay, self.sweep
        aggregated = FineRelay.aggregate(sweep, labels)
        reduced = relay.reducedRelay(labels)
        m = int(labels.max()) + 1
        curve, gap = reduced['curve'], self.fineGap
        after = np.arange(self.release, self.peak + 1)
        fluxReduced, fluxAggregated = reduced['flux'][:, 0], aggregated['flux'][:, 0]
        cosines = {phase: pathwayCosine(reduced['edges'][phase][0], aggregated['edges'][phase][0]) for phase in ('clear', 'write')}
        links = {phase: linksCarrying(reduced['edges'][phase][0]) for phase in ('clear', 'write')}
        linksAggregated = {phase: linksCarrying(aggregated['edges'][phase][0]) for phase in ('clear', 'write')}
        cellAbs = {state: float(np.abs(sweep['cellFlux'][state, 0]).sum()) for state in (self.release, self.trough, self.peak)}
        crossing = labels[:, None] != labels[None, :]
        netRetained = {}
        for phase in ('clear', 'write'):
            blockNet = aggregated['edges'][phase][0].sum(0)
            netRetained[phase] = float(np.abs(np.triu(blockNet - blockNet.T, 1)).sum() / self.netFine[phase])
        metrics = dict(
            meta, m=m, blockSizes=np.bincount(labels).tolist(),
            finalGap=dict(zip(self.readoutNames, np.round(curve[self.peak], 6).tolist())),
            gapError=float(abs(curve[self.peak, 0] - gap[0]) / abs(gap[0])),
            featureGapError={name: float(abs(curve[self.peak, k] - gap[k]) / abs(gap[k])) for k, name in enumerate(self.readoutNames)},
            curveError=float(np.sqrt(np.mean((curve[:, 0] - self.trueCurve[:, 0]) ** 2)) / abs(gap[0])),
            stateErrorFinal=float(np.linalg.norm(reduced['y'][self.peak] - reduced['trueState'][self.peak])
                                  / max(np.linalg.norm(reduced['trueState'][self.peak]), 1e-300)),
            shareError=float(np.mean(np.abs(fluxReduced[after] - fluxAggregated[after]).sum(1) / np.abs(fluxAggregated[after]).sum(1))),
            pathwayCosine=cosines,
            handOver=dict(reduced=handOver(fluxReduced, labels, self.ringCells, self.release),
                          aggregated=handOver(fluxAggregated, labels, self.ringCells, self.release)),
            links90=dict(links, total=sum(links.values())), links90Aggregated=dict(linksAggregated, total=sum(linksAggregated.values())),
            resolvedReadout=resolvedReadout(labels, relay.readouts[0]),
            crossingShare=float((self.grossFine * crossing).sum() / self.grossFine.sum()),
            netRetained=dict(netRetained, both=float((netRetained['clear'] * self.netFine['clear'] + netRetained['write'] * self.netFine['write'])
                                                      / (self.netFine['clear'] + self.netFine['write']))),
            retainedShare={str(state): float(np.abs(fluxAggregated[state]).sum() / cellAbs[state]) for state in cellAbs},
            booksAggregated=float(np.abs(fluxAggregated[after].sum(1) - gap[0]).max() / abs(gap[0])),
            booksReduced=float(np.abs(fluxReduced[after].sum(1) - curve[self.peak, 0]).max() / max(abs(curve[self.peak, 0]), 1e-300)),
            **self.floors(labels))
        if fromRelease:
            released, _ = relay.reducedReadoutCurve(labels, start=self.release)
            metrics['fromRelease'] = dict(
                gapError=float(abs(released[self.peak, 0] - gap[0]) / abs(gap[0])),
                curveError=float(np.sqrt(np.mean((released[self.release:, 0] - self.trueCurve[self.release:, 0]) ** 2)) / abs(gap[0])))
        if not detail:
            return metrics, None
        both = lambda edges: [np.round(edges[k], 6).tolist() for k in range(2)]
        details = dict(
            labels=labels.tolist(), curve=self.series(curve[:, 0]),
            flux=self.series(fluxReduced), fluxAggregated=self.series(fluxAggregated),
            injection=np.round(reduced['injection'][0], 6).tolist(),
            injectionAggregated=np.round(aggregated['injection'][0], 6).tolist(),
            edges={phase: both(reduced['edges'][phase][0]) for phase in relay.phases},
            edgesAggregated={phase: both(aggregated['edges'][phase][0]) for phase in relay.phases})
        return metrics, details


# ------------------------------------------------------------------ loading the fine relay
RELEASE, TROUGH, PEAK = 302, 586, 1766                              # state indices of recorded 301, 585, 1765
PHASES = dict(flood=(0, RELEASE), clear=(RELEASE, TROUGH), write=(TROUGH, PEAK))
READOUT_ROWS = [0, 2, 3, 4]                                         # selectivity, nose, eyes, mouth among the fine relay's five
READOUT_NAMES = ['selectivity', 'nose', 'eyes', 'mouth']


def loadRelay(jacobianPath, relayPath, cachePath=None, say=print):
    """The fine relay's inputs and its sweep, from the regenerated relay npz and the Jacobians. The sweep (adjoint,
    per-cell flux, phase and window edges) is kept at cachePath if given, since it is the same for every grouping."""
    regenerated = np.load(relayPath)
    relay = FineRelay(np.load(jacobianPath), regenerated['differenceExact'], regenerated['src'],
                      regenerated['readoutWeights'][READOUT_ROWS], PEAK, PHASES)
    import os
    if cachePath and os.path.exists(cachePath):
        cached = np.load(cachePath)
        sweep = dict(adjoint=cached['adjoint'], cellFlux=cached['cellFlux'], injection=cached['injection'],
                     windowEdges=cached['windowEdges'], phaseEdges={name: cached['edges_' + name] for name in PHASES})
        say('fine sweep loaded from cache')
    else:
        sweep = relay.sweep()
        say('fine sweep done')
        if cachePath:
            np.savez(cachePath, adjoint=sweep['adjoint'], cellFlux=sweep['cellFlux'], injection=sweep['injection'],
                     windowEdges=sweep['windowEdges'], **{'edges_' + name: sweep['phaseEdges'][name] for name in PHASES})
    return relay, sweep, regenerated
