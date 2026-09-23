"""Effective fixed points of each cell's voltage equation with its neighbours held at their current values.

Quasi-static reduction: Vmem relaxes fast compared with G_pol and with the neighbours, so at each moment a cell
sits at a stable root of its own dV/dt, and can only move to the other branch when the branch it is on vanishes.
"""
import sys
import numpy as np

Gref, Gdep = 1e-9, 1.5e-9
Epol, Edep = -0.055, -0.005
Vth, VT, Z = -0.027, 0.027, 3.0
G0, V0 = 5e-11, 0.012
ROWS = COLS = 11
numCells = ROWS * COLS

neighbourIndex = np.zeros((numCells, 4), dtype=int)
neighbourMask = np.zeros((numCells, 4), dtype=bool)
for cell in range(numCells):
    r, c = divmod(cell, COLS)
    for k, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        rr, cc = r + dr, c + dc
        if 0 <= rr < ROWS and 0 <= cc < COLS:
            neighbourIndex[cell, k] = rr * COLS + cc
            neighbourMask[cell, k] = True


def currentOnGrid(grid, gpolSiemens, vmemVolts):
    """f(V) on a voltage grid. gpolSiemens, vmemVolts are (chunk, cells) -> (chunk, cells, nV)."""
    V = grid[None, None, :]
    sMinus = 1.0 / (1.0 + np.exp(Z * (V - Vth) / VT))
    total = -gpolSiemens[..., None] * (V - Epol) * sMinus - Gdep * (V - Edep) * (1.0 - sMinus)
    for k in range(4):
        Vj = vmemVolts[:, neighbourIndex[:, k]][..., None]
        contribution = (2.0 * G0 / (1.0 + np.cosh((V - Vj) / V0))) * (Vj - V)
        total += np.where(neighbourMask[None, :, k, None], contribution, 0.0)
    return total


def branchesOf(grid, gpolRatio, vmemMilliVolts, chunk=25):
    """Per moment and cell: number of stable roots, the low and high roots (mV), and which one the cell is on."""
    T = len(gpolRatio)
    mid = 0.5 * (grid[:-1] + grid[1:]) * 1000.0
    nStable = np.zeros((T, numCells), dtype=np.int8)
    low = np.full((T, numCells), np.nan, dtype=np.float32)
    high = np.full((T, numCells), np.nan, dtype=np.float32)
    onDark = np.zeros((T, numCells), dtype=bool)
    for start in range(0, T, chunk):
        stop = min(start + chunk, T)
        v = vmemMilliVolts[start:stop].astype(np.float64)
        f = currentOnGrid(grid, gpolRatio[start:stop].astype(np.float64) * Gref, v / 1000.0)
        cross = (f[..., :-1] > 0) & (f[..., 1:] <= 0)
        any_ = cross.any(-1)
        nStable[start:stop] = cross.sum(-1)
        first = cross.argmax(-1)
        last = cross.shape[-1] - 1 - cross[..., ::-1].argmax(-1)
        lo = np.where(any_, mid[first], np.nan)
        hi = np.where(any_, mid[last], np.nan)
        low[start:stop], high[start:stop] = lo, hi
        onDark[start:stop] = np.where(any_, np.abs(v - lo) <= np.abs(v - hi), v < -34.6)
    return nStable, low, high, onDark


if __name__ == '__main__':
    data = np.load(sys.argv[1])
    vmem, gpol = data['vmem'], data['gpol']
    grid = np.linspace(-0.070, 0.005, 1201)
    out = {}
    for o in range(vmem.shape[0]):
        n, lo, hi, dark = branchesOf(grid, gpol[o], vmem[o])
        out[f'nStable{o}'], out[f'low{o}'], out[f'high{o}'], out[f'onDark{o}'] = n, lo, hi, dark
        print('order index %d done' % o, flush=True)
    np.savez_compressed(sys.argv[2], orders=data['orders'], hold=data['hold'],
                        bestIterations=data['bestIterations'], **out)
    print('wrote', sys.argv[2], flush=True)
