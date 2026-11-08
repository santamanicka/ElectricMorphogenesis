"""Which rotations leave a pattern unchanged?   python rotation_invariance.py   (writes results/rotation_invariance.txt)

A pattern that "follows the symmetry of the shape" would be invariant under every rotation that maps the shape onto itself (a disc: all angles; a hexagon: multiples of 60 degrees).
Potentia's equations are only invariant under the 90-degree rotations and axis flips of the square grid, so only the rotations that are in both groups can survive.
Measure: rotate the Gaussian-smoothed Vmem (sigma = 2 px) about the centre by an angle (cubic interpolation) and compute rms(V - rotated V) / spread over a central disc of radius 20 px.
The error of a rotation by an angle that is exactly a symmetry is the interpolation floor (read it off the 180-degree and 90-degree rows for the shapes that have them).
Controls: a smooth isotropic field (concentric rings, invariant under every rotation) and a smoothed random field (invariant under none).
"""
import os
import numpy as np
from scipy import ndimage
from potentia import Potentia
from shapes import shape_mask
from sweeps import STIG

lines = []
Np = (60, 60)
yy, xx = np.mgrid[:60, :60] + 0.5 - 30
disk = np.hypot(xx, yy) <= 20.0


def rot_err(V, angle):
    S = ndimage.gaussian_filter(V, 2.0, mode='nearest')
    Rv = ndimage.rotate(S, angle, reshape=False, order=3, mode='nearest')
    return float(np.sqrt(np.mean((S[disk] - Rv[disk]) ** 2)) / np.sqrt(np.mean((S[disk] - S[disk].mean()) ** 2)))


angles = [30, 45, 60, 90, 120, 180]
fields = {}
for name, shp in (('disc', 'disc'), ('ring', 'annulus'), ('hexagon (flat top)', 'hexagon')):
    P = Potentia(size_cells=(30, 30), m=2, mask=shape_mask(shp, Np, 0.5 if shp == 'annulus' else None), **STIG)
    V, g = P.initial_from_g(1.0, 'dep')
    for _ in range(10000):
        V, g = P.step(V, g)
    fields[name + ', homogeneous start, t = 100 s'] = V * 1e3
rr = np.hypot(xx, yy)
fields['control: concentric smooth rings (all rotations)'] = -30 + 20 * np.sin(2 * np.pi * rr / 10.0)
fields['control: smoothed random field (no rotation)'] = ndimage.gaussian_filter(np.random.default_rng(0).standard_normal((60, 60)), 2.5) * 20 - 30
lines.append('Rotation error of the central region (0 = invariant under that rotation; the interpolation floor is read off a rotation that is an exact symmetry)')
lines.append(f'    {"field":<58}' + ''.join(f'{a:>8d} deg' for a in angles))
for name, V in fields.items():
    lines.append(f'    {name:<58}' + ''.join(f'{rot_err(V, a):>12.3f}' for a in angles))
lines.append('')
lines.append('Exact symmetry of each pattern: disc and ring are invariant under 90-degree rotations and axis flips (D4); the hexagon only under the two flips (and so under 180 degrees).')
print('\n'.join(lines))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'rotation_invariance.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
