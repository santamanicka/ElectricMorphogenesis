"""Does the Stigmergic operating point never settle / never repeat?  Lattice (m=1) vs Potentia (m=2,4,8), finite 11x11, homogeneous start."""
import numpy as np, time
from sweeps import run_point
T = 600.0
print(f'11x11 finite tissue, homogeneous start, T={T:.0f}s, Stigmergic field parameters')
print(f'{"model":14s} {"regime":34s} {"act_last":>8s} {"trend":>6s} {"novelty":>8s} {"nov_p10":>8s} {"f_peak":>7s} {"peak_rat":>8s} {"runtime":>7s}')
rows = []
for model, m in (('lattice', 1), ('potentia', 2), ('potentia', 4), ('potentia', 8)):
    r = run_point(dict(size=(11, 11), m=m, T=T, rec_dt=0.5, model=model))
    d = r['desc']
    print(f'{model+" m="+str(m):14s} {r["regime"]:34s} {d["act_last"]:8.2f} {d["act_trend"]:6.2f} {d["novelty"]:8.2f} {d["novelty_p10"]:8.2f} {d["f_peak"]:7.3f} {d["peak_rat"]:8.2f} {r["runtime"]:6.0f}s', flush=True)
