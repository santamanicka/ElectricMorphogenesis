"""Anatomy search with the simulator's position-dependent field range (localrange.py), planarian only.

The range at a pixel is `range` % of the LOCAL width of the tissue (not of the body width), in one of three modes (thickness, smooth, gaussian);
the Gaussian mode has a smoothing length `sig` (a share of the largest thickness) and an edge shoulder `edge` (in ranges; 0 = none).
All runs: the simulator's planarian outline (shapes_simulator.py), straightened with the drawing's profile (pinch), uniform start, 4.4 x as long as wide.
Same scoring as anatomy_feature_search.py; results in anatomy/local/results.jsonl (kept apart from the uniform-range sweeps).

    python anatomy_local_search.py stage1      # mode x percentage at 32 cells wide, with uniform-range baselines, T = 150 s
    python anatomy_local_search.py stage2 <mode> <pct>   # smoothing length x edge shoulder (Gaussian), round a chosen percentage
    python anatomy_local_search.py stage3 <json config>  # one configuration at other sizes, T = 240 s
"""
import os, sys, json, itertools
import anatomy_feature_search as S

HERE = os.path.dirname(os.path.abspath(__file__))
S.D = os.path.join(HERE, 'anatomy', 'local')
S.RES = os.path.join(S.D, 'results.jsonl')
BASE = dict(shape='planarian', aspect=4.4, sim=1, pinch=1)
T1 = 150.0

if __name__ == '__main__':
    stage = sys.argv[1]
    cfg = []
    if stage == 'stage1':
        for f in (12.5, 15, 17.5, 20, 22.5, 25, 30):
            cfg.append(dict(BASE, rows=32, range=f, T=T1))
        for mode in ('gaussian', 'smooth', 'thickness'):
            for p in (10, 15, 20, 25, 30, 40, 50):
                cfg.append(dict(BASE, rows=32, range=p, T=T1, rmap=mode))
    elif stage == 'stage2':
        mode, p = sys.argv[2], float(sys.argv[3])
        for sig, edge in itertools.product((0.05, 0.10, 0.20, 0.30), (0.0, 0.5, 1.0, 2.0)):
            cfg.append(dict(BASE, rows=32, range=p, T=T1, rmap='gaussian', sig=sig, edge=edge))
    elif stage == 'stage3':
        base = json.loads(sys.argv[2])
        for r in (24, 40, 48):
            cfg.append(dict(BASE, **base, rows=r, T=240.0))
    S.sweep(cfg)
