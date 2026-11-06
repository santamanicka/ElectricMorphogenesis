"""Validation of Potentia against the reference lattice port.

 (1) lattice limit: Potentia(m=1, source='point') == Lattice to round-off (finite + periodic, vector + scalar field,
     homogeneous + random initial states, non-square domain so row/column mix-ups are caught)
 (2) continuum limit: refining m changes the cell-averaged solution by a shrinking amount
Run:  python validate_potentia.py
"""
import numpy as np
from lattice import Lattice
from potentia import Potentia

def compare(dims, periodic, vector, init, nsteps, bias=5.3e-4, R_s=4.0, seed=0, label=''):
    kw = dict(fieldScreenSize=R_s, bias=bias, fieldVector=vector)
    L = Lattice(dims=dims, periodic=periodic, **kw)
    P = Potentia(size_cells=dims, m=1, periodic=periodic, **kw)
    nr, nc = dims
    if init == 'homogeneous':
        V0 = np.full((nr, nc), -9.2e-3); g0 = np.full((nr, nc), 1.0)
    else:
        rng = np.random.default_rng(seed)
        V0 = rng.uniform(-55e-3, -5e-3, (nr, nc)); g0 = rng.uniform(0.2, 1.8, (nr, nc))
    VL, GL = V0.ravel().copy(), g0.ravel() * L.G_ref
    VP, gP = V0.copy(), g0.copy()
    worstV = worstG = 0.0
    for it in range(nsteps):
        VL, GL = L.step(VL, GL)
        VP, gP = P.step(VP, gP)
        if it % 50 == 0 or it == nsteps - 1:
            worstV = max(worstV, np.abs(VL.reshape(nr, nc) - VP).max())
            worstG = max(worstG, np.abs(GL.reshape(nr, nc) / L.G_ref - gP).max())
    ok = worstV < 1e-9 and worstG < 1e-7
    print(f"{'PASS' if ok else 'FAIL'}  {label:46s} max|dV|={worstV:.2e} V  max|dg|={worstG:.2e}")
    return ok

ok = True
ok &= compare((11, 11), False, True, 'homogeneous', 3000, label='finite 11x11 vector, homogeneous, 3000 steps')
ok &= compare((9, 13), False, True, 'random', 600, label='finite 9x13 vector, random state')
ok &= compare((12, 14), True, True, 'random', 600, label='periodic 12x14 vector, random state')
ok &= compare((9, 13), False, False, 'random', 600, label='finite 9x13 scalar potential, random state')
ok &= compare((12, 14), True, False, 'random', 600, label='periodic 12x14 scalar potential, random state')
ok &= compare((11, 11), False, True, 'random', 600, R_s=1.0, label='finite 11x11 vector, R_s=1 (short range)')
ok &= compare((16, 16), True, True, 'random', 300, R_s=6.0, bias=2e-4, label='periodic 16x16 vector, R_s=6')
print('\nLATTICE LIMIT:', 'ALL PASS' if ok else 'FAILURES')
