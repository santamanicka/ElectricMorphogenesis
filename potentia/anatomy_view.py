"""Contact sheet of G_pol snapshots of one run.   python anatomy_view.py <tag> [out.png]"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anatomy')
def sheet(tagname, out=None, times=(10, 20, 30, 45, 60, 80, 100, 130, 160, 200, 250, 300), key='G'):
    d = np.load(os.path.join(D, 'runs', tagname + '.npz'))
    G, mask, t = d[key], d['mask'], d['t']
    fig, ax = plt.subplots(len(times) // 2, 2, figsize=(14, 1.6 * len(times) // 2 * 14 / 14 * (mask.shape[0] / mask.shape[1]) * 4.4 + 1))
    for a, tt in zip(ax.T.ravel(), times):
        k = int(np.argmin(abs(t - tt)))
        a.imshow(np.where(mask, G[k], np.nan), cmap='viridis', vmin=0, vmax=2 if key == 'G' else None, interpolation='nearest')
        a.set_title(f't={t[k]:g}s', fontsize=7, pad=1); a.axis('off')
    fig.suptitle(tagname, fontsize=8); fig.tight_layout()
    out = out or os.path.join(D, 'sheet_' + tagname + '.png'); fig.savefig(out, dpi=90); plt.close(fig); return out
if __name__ == '__main__':
    print(sheet(*sys.argv[1:]))
