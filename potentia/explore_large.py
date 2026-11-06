import numpy as np, matplotlib, time; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from potentia import Potentia

def d4_asym(V):
    """relative departure from D4 symmetry (square domain)"""
    sym = (V + V[::-1] + V[:, ::-1] + V[::-1, ::-1] + V.T + V.T[::-1] + V.T[:, ::-1] + V.T[::-1, ::-1]) / 8
    return np.abs(V - sym).max() * 1e3  # mV

N, m = 31, 2
P = Potentia(size_cells=(N, N), m=m, fieldScreenSize=4.0, bias=5.3e-4)
V, g = P.initial()
snap_t = [3, 6, 10, 20, 40, 80, 120, 160]
t0 = time.time(); t = 0; frames = {}; asym = []
for T in range(1, 161):
    n = int(round(1.0 / P.dt))
    for _ in range(n):
        V, g = P.step(V, g)
    asym.append((T, d4_asym(V)))
    if T in snap_t: frames[T] = (V.copy(), g.copy())
print('runtime', round(time.time() - t0, 1), 's;  D4 asymmetry (mV) at t=', [(T, round(a, 6)) for T, a in asym if T in (3, 10, 20, 40, 80, 120, 160)])
fig, ax = plt.subplots(2, len(snap_t), figsize=(2.3 * len(snap_t), 5))
for j, T in enumerate(snap_t):
    Vf, gf = frames[T]
    ax[0, j].imshow(Vf * 1e3, vmin=-55, vmax=-5, cmap='viridis'); ax[0, j].set_title(f't={T}s'); ax[0, j].axis('off')
    ax[1, j].imshow(gf, vmin=0, vmax=2, cmap='magma'); ax[1, j].axis('off')
ax[0, 0].text(-0.1, 0.5, 'Vmem', transform=ax[0, 0].transAxes, rotation=90, va='center', ha='right')
ax[1, 0].text(-0.1, 0.5, 'g', transform=ax[1, 0].transAxes, rotation=90, va='center', ha='right')
plt.tight_layout(); plt.savefig('results/explore_large31.png', dpi=100)
