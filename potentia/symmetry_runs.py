"""How long does a homogeneous start keep the symmetry of the shape + grid, and what happens once it is broken?   python symmetry_runs.py

For disc, ring (D4 on a square grid: 90 degree rotations and axis flips) and a flat-top hexagon (D2: flips about the N-S and E-W axes) the model and the start are
both invariant under the group, so the evolution stays in the symmetric subspace until round-off (or a deliberate perturbation) is amplified by the chaotic dynamics.
Symmetry error = rms(V - g V) / rms(V - mean V) over the tissue, the maximum over the group elements g (0 = exactly symmetric, ~1 = unrelated).
"""
import os
import numpy as np
from potentia import Potentia
from shapes import shape_mask
from sweeps import STIG

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
box, Np = (30, 30), (60, 60)
shapes = {'disc': (shape_mask('disc', Np), 'D4'), 'ring': (shape_mask('annulus', Np, 0.5), 'D4'), 'hexagon (flat top)': (shape_mask('hexagon', Np), 'D2')}
lines = []


def log(s=''):
    print(s, flush=True); lines.append(s)


def sym_error(V, mk, group):
    ops = [lambda a: a[::-1, :], lambda a: a[:, ::-1]]
    if group == 'D4':
        ops += [np.rot90, lambda a: a.T]
    ref = np.sqrt(np.mean((V[mk] - V[mk].mean()) ** 2))
    return max(np.sqrt(np.mean((V[mk] - op(V)[mk]) ** 2)) for op in ops) / max(ref, 1e-30)


T = 600.0
times = [10, 50, 100, 200, 300, 400, 500, 600]
log(f'Symmetry error of Vmem versus time, 30 x 30 cells, m = 2, Stigmergic parameters (homogeneous start; "perturbed" adds N(0, sigma) to G_pol with sigma = 1e-6)')
log(f'    {"case":<34}' + ''.join(f'{t:>10d} s' for t in times))
final = {}
for name, (mk, group) in shapes.items():
    for tag, noise in (('exact', 0.0), ('perturbed 1e-6', 1e-6)):
        P = Potentia(size_cells=box, m=2, mask=mk, **STIG)
        g0 = 1.0 + noise * np.random.default_rng(0).standard_normal(P.Np)
        V, g = P.initial_from_g(g0, 'dep')
        rec = {}
        for it in range(int(round(T / P.dt)) + 1):
            t = it * P.dt
            if it % 1000 == 0 and int(round(t)) in times:
                rec[int(round(t))] = sym_error(V, mk, group)
            if it < int(round(T / P.dt)):
                V, g = P.step(V, g)
        final[(name, tag)] = (V * 1e3, mk)
        log(f'    {name + ", " + tag + " (" + group + ")":<34}' + ''.join(f'{rec[t]:>12.1e}' for t in times))
log()
log('    exact: the symmetry error stays at round-off level (1e-16 .. 1e-10) until chaos amplifies it; perturbed: the symmetry is broken from the start and the error grows to order 1.')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.4))
    for c, name in enumerate(shapes):
        for r, tag in enumerate(('exact', 'perturbed 1e-6')):
            V, mk = final[(name, tag)]
            ax = axs[r, c]; ax.imshow(np.where(mk, V, np.nan), cmap='viridis', vmin=-55, vmax=-5, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'{name}, {tag}, t = 600 s', fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_symmetry_runs.png'), dpi=110)
except Exception as e:
    log(f'(figure not written: {e})')
with open(os.path.join(OUT, 'symmetry_runs.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
