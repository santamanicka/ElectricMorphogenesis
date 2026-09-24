"""Decompose the trained-minus-unclamped difference exactly, into the network that carries it.

Registered first: data/boundaryHarmonicRelayPredictions1888Hold301FaceMinus60Minus5.json.

With x = (Vmem, G_pol) and D(n) = x_trained(n) - x_free(n), one iteration of the model gives

    D(n+1) = Jbar(n) D(n) + src(n)            exactly,

where Jbar(n) is the step Jacobian averaged over the straight segment from x_free(n) to x_trained(n) (Gauss-Legendre,
refined until the identity closes) and src(n) is what the clamp does to the FREE state at that step -- the ring's
G_pol set to the code, and the extra Vmem update the clamp runs on every cell -- nonzero only during the hold.
Nothing is linearised: this decomposes the actual finite difference, so the chaos that makes infinitesimal linear
response useless over two thousand iterations does not enter.

For a readout w at state index T, the adjoint abar(T) = w, abar(n) = abar(n+1) Jbar(n) gives the flux
phi_k(n) = abar(n) . D(n) restricted to cell k: how much of the final difference cell k's deviation is carrying at
that moment. After the hold, sum_k phi_k(n) = w . D(T) at every n. Edges e(i <- j) = abar_i(n+1) Jbar_ij(n) D_j(n)
split exactly into a field channel and a contact channel, because the start-of-step Vmem enters the field and the
gap-junction current as separate arguments of the step.

State index n counts iterations completed, so recorded iteration t is n = t + 1; the clamp acts on steps n < hold.

    python3 computeBoundaryHarmonicRelay11x11.py --outputPath <relay.npz>
"""
import argparse
import json
import time

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from boundaryHarmonicStep import Step

parser = argparse.ArgumentParser()
parser.add_argument('--outputPath', type=str, required=True)
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--nodes', type=int, default=16, help='Gauss-Legendre nodes along each segment')
parser.add_argument('--tolerance', type=float, default=1e-6, help='one-step closure required, relative')
parser.add_argument('--fluxStride', type=int, default=5)
parser.add_argument('--edgeWindow', type=int, default=50)
args = parser.parse_args()
torch.set_grad_enabled(False)

summary = json.load(open(args.summaryPath))
hold = int(summary['hold'])
winner = summary['orders']['3']['best']
coefficients = np.asarray(np.load(
    f"{summary['trainingDirs'][winner['round']]}/order3_restart{winner['restart']:02d}.npz")['bestCoefficients'], float)
ringValues = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(len(coefficients)))) @ coefficients
step = Step(ringCode=ringValues)
n = step.numCells
Gref = step.Gref

READOUT_PRIMARY, READOUT_FACE = 1765 + 1, 2173 + 1          # state indices
LAST = READOUT_FACE
interior = np.array(boundary.interiorCellIndices)
features = np.array(sorted(set(boundary.featureCellIndices.tolist())))
background = np.array([c for c in interior if c not in set(features)])
groups = dict(nose=[49, 60, 71], eyes=[24, 25, 29, 30, 35, 36, 40, 41], mouth=[92, 93, 94])

# ------------------------------------------------------------------ readouts, as rows over the 2n-dim state
readoutNames = ['selectivity', 'face', 'nose', 'eyes', 'mouth']
readoutTime = dict(selectivity=READOUT_PRIMARY, face=READOUT_FACE, nose=READOUT_PRIMARY, eyes=READOUT_PRIMARY,
                   mouth=READOUT_PRIMARY)
W = torch.zeros(len(readoutNames), 2 * n, dtype=torch.double)
W[0, n + features] = 1.0 / (len(features) * Gref)                 # selectivity, in units of G_ref
W[0, n + background] = -1.0 / (len(background) * Gref)
W[1, features] = 1000.0 / len(features)                            # face contrast, in mV
W[1, background] = -1000.0 / len(background)
for row, name in enumerate(['nose', 'eyes', 'mouth'], start=2):
    W[row, n + np.array(groups[name])] = 1.0 / (len(groups[name]) * Gref)
    W[row, n + background] = -1.0 / (len(background) * Gref)

# ------------------------------------------------------------------ the two trajectories, in float64
v0, g0 = step.initialVmem.clone(), step.initialGpol.clone()
states = {}
for name in ('trained', 'free'):
    V = torch.zeros(LAST + 1, n, dtype=torch.double)
    G = torch.zeros(LAST + 1, n, dtype=torch.double)
    V[0], G[0] = v0, g0
    for m in range(LAST):
        V[m + 1], G[m + 1] = step(V[m], G[m], clamped=(name == 'trained' and m < hold))
    states[name] = torch.cat([V, G], dim=1)                        # (LAST+1, 2n)
X0, X1 = states['free'], states['trained']
D = X1 - X0
for name, X in (('trained', X1), ('free', X0)):
    meanG = (X[:, n + interior].mean(1) / Gref).numpy()
    trough = 302 + int(meanG[302:1300].argmin())
    print(f'{name:>8}: first peak {meanG[:hold + 1].max():.3f}, trough at recorded {trough - 1}, '
          f'second peak at recorded {trough + int(meanG[trough:].argmax()) - 1}', flush=True)
difference = {name: float(W[k] @ D[readoutTime[name]]) for k, name in enumerate(readoutNames)}
print('differences to decompose:', {k: round(v, 4) for k, v in difference.items()}, flush=True)

# the clamp's own contribution at each hold step, measured on the free state
src = torch.zeros(hold, 2 * n, dtype=torch.double)
for m in range(hold):
    clampedV, clampedG = step(X0[m, :n], X0[m, n:], clamped=True)
    freeV, freeG = step(X0[m, :n], X0[m, n:], clamped=False)
    src[m] = torch.cat([clampedV - freeV, clampedG - freeG])


def preClip(points):
    """G_pol before the model clips it to [min, max], at a batch of states: the only non-smooth step in the map."""
    v, g = points[:, :n], points[:, n:]
    sigma = torch.sigmoid(step.gain * step.fieldRead(v) + step.bias)
    drive = 10.0 * (-g + (2.0 * sigma - 1.0) * step.weight) / step.tau
    return g + step.dt * drive * Gref


def kinks(m, gridPoints):
    """Where along the segment any cell's G_pol crosses a clip bound. The Jacobian jumps there, so quadrature that
    straddles one converges only like 1/N; splitting at each crossing restores exponential convergence."""
    lam = torch.linspace(0.0, 1.0, gridPoints, dtype=torch.double)
    gp = preClip(X0[m].unsqueeze(0) + lam.unsqueeze(1) * D[m].unsqueeze(0))
    brackets = []
    for bound in (step.minG, step.maxG):
        above = gp > bound
        k, cell = torch.where(above[:-1] != above[1:])
        for a, c in zip(k.tolist(), cell.tolist()):
            brackets.append((float(lam[a]), float(lam[a + 1]), c, bound, bool(above[a, c])))
    if not brackets:
        return []
    lo = torch.tensor([b[0] for b in brackets], dtype=torch.double)
    hi = torch.tensor([b[1] for b in brackets], dtype=torch.double)
    cells = torch.tensor([b[2] for b in brackets])
    bounds = torch.tensor([b[3] for b in brackets], dtype=torch.double)
    loAbove = torch.tensor([b[4] for b in brackets])
    for _ in range(55):                                              # bisection, all brackets at once
        mid = 0.5 * (lo + hi)
        gMid = preClip(X0[m].unsqueeze(0) + mid.unsqueeze(1) * D[m].unsqueeze(0))[torch.arange(len(mid)), cells]
        sameAsLo = (gMid > bounds) == loAbove
        lo = torch.where(sameAsLo, mid, lo)
        hi = torch.where(sameAsLo, hi, mid)
    return sorted(set(round(float(x), 15) for x in 0.5 * (lo + hi)))


def fieldTurns(m, sharpest=0.05):
    """The other sharp feature. The step reads the field through E/|E|, and E is exactly linear along the segment,
    so where a grid point's field vector passes close to zero its direction swings within a lambda-width of
    |E|min / |dE/dlambda|. Break the segment around each such passage, graded by its width."""
    E0 = torch.einsum('agc,c->ag', step.L, X0[m, :n])
    dE = torch.einsum('agc,c->ag', step.L, D[m, :n])
    speed = torch.sqrt((dE ** 2).sum(0)).clamp_min(1e-300)
    lamStar = -(E0 * dE).sum(0) / speed ** 2
    width = torch.sqrt(((E0 + lamStar * dE) ** 2).sum(0)) / speed
    breaks = []
    for g in torch.where((lamStar > 0) & (lamStar < 1) & (width < sharpest))[0].tolist():
        centre, w = float(lamStar[g]), float(width[g])
        breaks += [min(1.0, max(0.0, centre + k * w)) for k in (-20, -5, -1, 0, 1, 5, 20)]
    return breaks


def secant(m, nodes, gridPoints=129):
    """Jbar(m) as role blocks, by Gauss-Legendre on each smooth piece of the segment, plus the endpoint Jacobians
    for the tangent-linear diagnostic. Returns the number of pieces too."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    breaks = sorted(set([0.0] + kinks(m, gridPoints) + fieldTurns(m) + [1.0]))
    lam, weights = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b - a <= 0:
            continue
        lam.extend(a + (b - a) * (x + 1) / 2)
        weights.extend((b - a) * w / 2)
    lam = torch.tensor(np.concatenate([lam, [0.0, 1.0]]), dtype=torch.double)
    weights = torch.tensor(weights, dtype=torch.double)
    count = len(weights)
    points = X0[m].unsqueeze(0) + lam.unsqueeze(1) * D[m].unsqueeze(0)
    averaged, ends = None, {}
    for start in range(0, len(lam), 64):
        chunk = step.analyticJacobians(points[start:start + 64, :n], points[start:start + 64, n:], clamped=(m < hold))
        take = slice(0, max(0, min(64, count - start)))
        part = {role: torch.einsum('q,qij->ij', weights[start:start + 64], block[take]) for role, block in chunk.items()}
        averaged = part if averaged is None else {role: averaged[role] + part[role] for role in part}
        for role, block in chunk.items():
            tail = block[max(0, count - start):]
            if len(tail):
                ends[role] = tail if role not in ends else torch.cat([ends[role], tail])
    return averaged, ends, len(breaks) - 1


def closureOf(J, m):
    """One-step identity residual, voltage and conductance each against their own scale."""
    predicted = J @ D[m] + (src[m] if m < hold else 0.0)
    worst = 0.0
    for part in (slice(0, n), slice(n, 2 * n)):
        scale = torch.maximum(D[m + 1, part].abs().max(), predicted[part].abs().max()).clamp_min(1e-300)
        worst = max(worst, float((D[m + 1, part] - predicted[part]).abs().max() / scale))
    return worst


def assemble(blocks):
    return torch.cat([blocks['field'] + blocks['gap'] + blocks['self_'], blocks['g']], dim=1)


# ------------------------------------------------------------------ the backward sweep
windows = [(302, 586), (586, 1001), (1001, 1501), (1501, 1766), (302, 1766)]
active, transfer = {}, {}
abar = torch.zeros(len(readoutNames), 2 * n, dtype=torch.double)
tangent = torch.zeros(2, 2 * n, dtype=torch.double)             # selectivity through J(free) and J(trained)
fluxTimes = list(range(0, LAST + 1, args.fluxStride))
flux = np.zeros((len(readoutNames), len(fluxTimes), n))
sourceFlux = np.zeros((len(readoutNames), 2, n))                   # per readout: [V side effect, G injection] per cell
numEdgeWindows = LAST // args.edgeWindow + 1
edges = np.zeros((len(readoutNames), 2, numEdgeWindows, n, n))     # [field, contact]
grossEdges = np.zeros((len(readoutNames), 2, numEdgeWindows))
closure = np.zeros(LAST)
pieceCount = np.zeros(LAST, dtype=int)
refined, tangentNorm = [], np.zeros((2, LAST + 1))
offDiagonal = torch.ones(n, n, dtype=torch.double) - torch.eye(n, dtype=torch.double)
started = time.time()

for m in range(LAST, -1, -1):
    for k, name in enumerate(readoutNames):
        if m == readoutTime[name]:
            abar[k] += W[k]
    if m == READOUT_PRIMARY:
        tangent += W[0]
    tangentNorm[:, m] = tangent.norm(dim=1).numpy()
    if m % args.fluxStride == 0:
        cellFlux = (abar * D[m]).reshape(len(readoutNames), 2, n).sum(1)
        flux[:, m // args.fluxStride] = cellFlux.numpy()
    for (t1, t2) in windows:
        if m == t2:
            active[(t1, t2)] = dict(P=torch.eye(2 * n, dtype=torch.double), a=abar.clone())
        if m == t1 and (t1, t2) in active:
            entry = active.pop((t1, t2))
            # T[r, i, j] = sum_c sum_c' a_r(t2)[(i,c)] P[(i,c),(j,c')] D(t1)[(j,c')]
            weighted = entry['a'].unsqueeze(-1) * entry['P'].unsqueeze(0) * D[t1].unsqueeze(0).unsqueeze(0)
            transfer[(t1, t2)] = weighted.reshape(len(readoutNames), 2, n, 2, n).sum((1, 3)).numpy()
    if m == 0:
        break

    s = m - 1                                                      # the step from state s to state m
    blocks, ends, pieces = secant(s, args.nodes)
    J = assemble(blocks)
    closure[s] = closureOf(J, s)
    pieceCount[s] = pieces
    nodes, grid = args.nodes, 129
    while closure[s] > args.tolerance and nodes <= 64:
        nodes, grid = nodes * 2, grid * 4 - 3                     # finer kink search, more nodes per piece
        blocks, ends, pieces = secant(s, nodes, grid)
        J = assemble(blocks)
        closure[s] = closureOf(J, s)
        pieceCount[s] = pieces
    if nodes != args.nodes:
        refined.append((s, nodes, closure[s]))

    if s < hold:                                                   # what the clamp injected at this step
        injected = (abar * src[s]).reshape(len(readoutNames), 2, n)
        sourceFlux += injected.numpy()

    # edges into cell i from cell j over this step, split by the channel the start-of-step Vmem entered through
    window = s // args.edgeWindow
    aV, aG = abar[:, :n], abar[:, n:]
    for channel, parts in ((0, ('field',)), (1, ('gap', 'self_', 'g'))):
        total = torch.zeros(len(readoutNames), n, n, dtype=torch.double)
        for role in parts:
            block = blocks[role]
            source = D[s, n:] if role == 'g' else D[s, :n]
            weighted = (aV.unsqueeze(-1) * block[:n].unsqueeze(0) + aG.unsqueeze(-1) * block[n:].unsqueeze(0))
            total += weighted * source.unsqueeze(0).unsqueeze(0) * offDiagonal
        edges[:, channel, window] += total.numpy()
        grossEdges[:, channel, window] += total.abs().sum((1, 2)).numpy()

    for entry in active.values():
        entry['P'] = entry['P'] @ J
    abar = abar @ J
    tangent = torch.stack([tangent[0] @ assemble({r: b[0] for r, b in ends.items()}),
                           tangent[1] @ assemble({r: b[1] for r, b in ends.items()})])
    if m % 250 == 0:
        print(f'  state {m}: closure so far {closure[s:].max():.1e}, {len(refined)} steps refined, '
              f'{int((pieceCount[s:] > 1).sum())} split at a clip, '
              f'{time.time() - started:.0f}s', flush=True)

# after the hold the flux must sum to the final difference at every moment
conservation = {}
for k, name in enumerate(readoutNames):
    total = difference[name]
    times = [t for t in fluxTimes if hold + 1 <= t <= readoutTime[name]]
    sums = np.array([flux[k, fluxTimes.index(t)].sum() for t in times])
    conservation[name] = float(np.abs(sums - total).max() / abs(total))
    injectedTotal = sourceFlux[k].sum() + flux[k, 0].sum()
    print(f'{name:>12}: difference {total:+.4f}; flux conserved to {conservation[name]:.1e} after the hold; '
          f'sources account for {injectedTotal:+.4f}', flush=True)
print(f'one-step closure: worst {closure.max():.1e}; {len(refined)} steps needed more nodes', flush=True)

np.savez_compressed(
    args.outputPath, readoutNames=np.array(readoutNames), readoutTimes=np.array([readoutTime[r] for r in readoutNames]),
    difference=np.array([difference[r] for r in readoutNames]), fluxTimes=np.array(fluxTimes), flux=flux,
    sourceFlux=sourceFlux, edges=edges, grossEdges=grossEdges, edgeWindow=args.edgeWindow, closure=closure,
    refined=np.array(refined) if refined else np.zeros((0, 3)), conservation=np.array([conservation[r] for r in readoutNames]),
    transferWindows=np.array(windows), transfer=np.stack([transfer[w] for w in windows]),
    tangentNorm=tangentNorm, pieceCount=pieceCount, hold=hold, D=D.numpy().astype(np.float32),
    trainedState=X1.numpy().astype(np.float32), freeState=X0.numpy().astype(np.float32))
print('wrote', args.outputPath, flush=True)
