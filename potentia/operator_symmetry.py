"""Which parts of Potentia carry the square symmetry, and which are rotation invariant?   python operator_symmetry.py   (writes results/operator_symmetry.txt)

Each term of the update is applied to a random state and to the state transformed by an element of D4 (rotations by 90, 180, 270 degrees; flips about the two axes and the two
diagonals); the table shows the largest difference between op(g . x) and g . op(x), relative to the size of the output (0 = the term commutes with the transformation).
"""
import os
import numpy as np
from potentia import Potentia
from shapes import shape_mask
from sweeps import STIG

lines = []
ops = {'rot90': np.rot90, 'rot180': lambda a: np.rot90(a, 2), 'rot270': lambda a: np.rot90(a, 3), 'flip N-S': lambda a: a[::-1, :], 'flip E-W': lambda a: a[:, ::-1],
       'flip diagonal': lambda a: a.T, 'flip anti-diagonal': lambda a: np.rot90(a, 2).T}
rng = np.random.default_rng(0)


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))


for title, mask in (('square tissue, 20 x 20 cells (rectangle)', None), ('disc (a D4-symmetric shape)', shape_mask('disc', (40, 40)))):
    P = Potentia(size_cells=(20, 20), m=2, mask=mask, **STIG)
    mf = np.ones(P.Np) if P.mask is None else P.maskf
    V = rng.uniform(-55e-3, -5e-3, P.Np) * mf; g = rng.uniform(0.2, 1.8, P.Np) * mf
    terms = {'gap-junction term (gated, per axis)': lambda V, g: P.gj(V), 'ion-channel current (local)': lambda V, g: P.ion(V, g), 'perceived field (Coulomb sum + 4 corner probes)': lambda V, g: P.perceived_field(V),
             'one full step, V': lambda V, g: P.step(V, g)[0], 'one full step, G_pol': lambda V, g: P.step(V, g)[1]}
    lines.append(f'{title}: max |op(g.x) - g.op(x)| / max|op(x)|')
    lines.append(f'    {"term":<50}' + ''.join(f'{k:>20}' for k in ops))
    for name, f in terms.items():
        base = f(V, g)
        lines.append(f'    {name:<50}' + ''.join(f'{rel(f(op(V), op(g)), op(base)):>20.1e}' for op in ops.values()))
    lines.append('')
lines.append('Not invariant under continuous rotations (output of a radially symmetric input, variation around a circle; results/anisotropy_study.txt, m = 2 / 4 / 8):')
lines.append('    gated gap-junction term 0.47 / 0.50 / 0.51 (4th harmonic 0.12-0.13);  same stencil without the gate 0.0026 / 0.0006 / 0.0002;  perceived field of a smooth bump 0.0007 / 0.0015 / 0.0014')
print('\n'.join(lines))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'operator_symmetry.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
