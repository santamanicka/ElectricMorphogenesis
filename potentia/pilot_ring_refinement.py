"""Resolution-convergence check for the one steady-ring candidate of pilot_localized_seeds.py.

Candidate: Stigmergic parameters with R_s = 2 (bias x1.0), hyperpolarised background (g = 2),
central depolarised G_pol = 0 disc of radius 3 cells, 30 x 30 tissue.  A real soliton should
(a) look the same at m = 1, 2, 4 and (b) not change when the tissue edge is removed.
Reported per snapshot: area (in cell^2) of cells with Vmem > -30 mV within 10 cells of the centre,
and the active area outside that radius (edge-driven activity, finite tissue only).

    python3 pilot_ring_refinement.py            # writes results/pilot_ring_refinement.txt
"""
import os
import sys

import numpy as np

from potentia import Potentia
from sweeps import STIG

N = 30
SNAPS = (15, 30, 60, 100, 150)
HERE = os.path.dirname(os.path.abspath(__file__))


def ring(m, periodic, T=150, seed_cells=3.0, Rs=2.0):
    P = Potentia(size_cells=(N, N), m=m, periodic=periodic, **{**STIG, "fieldScreenSize": Rs})
    yy, xx = np.mgrid[:P.Np[0], :P.Np[1]]
    c = (P.Np[0] - 1) / 2
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    g = np.full(P.Np, 2.0)
    g[r2 <= (seed_cells * m) ** 2] = 0.0
    V, g = P.initial_from_g(g, "dep")
    nrec = int(round(1 / P.dt))
    nsteps = int(round(T / P.dt))
    rec = {}
    for it in range(nsteps + 1):
        t = it / nrec
        if it % nrec == 0 and int(round(t)) in SNAPS:
            act = V * 1e3 > -30
            inner = act & (r2 <= (10 * m) ** 2)
            outer = act & (r2 > (10 * m) ** 2)
            rec[int(round(t))] = (inner.sum() / m ** 2, outer.sum() / m ** 2)
        if it < nsteps:
            V, g = P.step(V, g)
    return rec


def main():
    lines = ["Steady-ring candidate (bias x1.0, R_s=2, seed radius 3 cells), 30x30 tissue",
             "inner = active area (cell^2) within 10 cells of centre; outer = active area beyond (edge-driven)", ""]
    for periodic in (True, False):
        for m in (1, 2, 4):
            rec = ring(m, periodic)
            tag = "periodic (diagnostic only)" if periodic else "finite"
            lines.append(f"{tag:26s} m={m}: " + "  ".join(
                f"t={t}s inner {a:5.1f} outer {o:5.1f}" for t, (a, o) in rec.items()))
            print(lines[-1], flush=True)
    out = os.path.join(HERE, "results", "pilot_ring_refinement.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
