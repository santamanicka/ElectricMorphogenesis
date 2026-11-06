"""Continuum limit: refine the pixel grid (m pixels per cell) at fixed physical parameters.
Reports the RMS difference of the cell-averaged Vmem against the finest grid, vs physical time."""
import sys, time
import numpy as np
from potentia import Potentia

size = (11, 11)
times = [2.0, 5.0, 10.0, 30.0]
ms = [1, 2, 3, 4, 6, 8]
kw = dict(fieldScreenSize=4.0, bias=5.3e-4)
sol = {}
for m in ms:
    P = Potentia(size_cells=size, m=m, **kw)
    V, g = P.initial()
    out = {}
    t0 = time.time(); t = 0.0
    for T in times:
        n = int(round((T - t) / P.dt)); 
        V, g, _ = P.run(n, V, g); t += n * P.dt
        out[T] = P.cell_average(V)
    sol[m] = out
    print(f'm={m}: dt={P.dt:.5f}  {time.time()-t0:5.1f}s', flush=True)
ref = sol[ms[-1]]
print('\nRMS difference of cell-averaged Vmem vs m=%d (mV)' % ms[-1])
print('   m   ' + ''.join(f't={T:>5.0f}s  ' for T in times))
for m in ms[:-1]:
    print(f'  {m:2d}   ' + ''.join(f'{np.sqrt(((sol[m][T]-ref[T])**2).mean())*1e3:8.3f}  ' for T in times))
print('\nnumber of hyperpolarised cells (V < -30 mV)')
for m in ms:
    print(f'  m={m}: ' + '  '.join(f't={T:.0f}s:{int((sol[m][T] < -30e-3).sum()):3d}' for T in times))
np.save('results_convergence_cellV.npy', {m: {T: sol[m][T] for T in times} for m in ms}, allow_pickle=True)
