import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from potentia import Potentia
T = 120.0
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
data = {}
for m in (1, 2, 4):
    P = Potentia(size_cells=(11, 11), m=m, fieldScreenSize=4.0, bias=5.3e-4)
    V, g = P.initial()
    n = int(round(T / P.dt)); rec = 10
    ts, fh, mv, mg = [], [], [], []
    for it in range(n):
        if it % rec == 0:
            c = P.cell_average(V); ts.append(it * P.dt)
            fh.append((c < -30e-3).mean()); mv.append(c.mean() * 1e3); mg.append(P.cell_average(g).mean())
        V, g = P.step(V, g)
    data[m] = (np.array(ts), np.array(fh), np.array(mv), np.array(mg))
    axes[0].plot(ts, fh, label=f'm={m}'); axes[1].plot(ts, mv, label=f'm={m}'); axes[2].plot(ts, mg, label=f'm={m}')
axes[0].set_ylabel('fraction cells < -30 mV'); axes[1].set_ylabel('mean Vmem (mV)'); axes[2].set_ylabel('mean g = G_pol/G_ref')
axes[2].set_xlabel('time (s)'); axes[0].legend(); axes[0].set_title('11x11 tissue, homogeneous start, Stigmergic field parameters')
plt.tight_layout(); plt.savefig('results/explore_timeseries.png', dpi=110)
for m,(ts,fh,mv,mg) in data.items():
    late = ts > 40
    print(f'm={m}: late-time (t>40s) mean fraction hyperpol {fh[late].mean():.3f}  std {fh[late].std():.3f}  | mean Vmem {mv[late].mean():.2f} mV std {mv[late].std():.2f}')
