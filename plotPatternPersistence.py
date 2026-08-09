"""Show that the trained pattern is a transient rather than an attractor.

Both models are scored on iterations 900-1000, and both pass closest to their target there. The clamp
is released at iteration 100, so what the loss measures is a pattern still decaying from the clamp
rather than one the tissue sustains: run either model on to iteration 3000 and the pattern it settles
into is a different one. This is a property of the published 11x11 model too, not an artefact of the
30x30 runs, which is why both scales are drawn.
"""
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

panels = [('data/snapshots_ref11_3000.npy', 'data/StigmergicModelParameters.dat', '11x11 reference', 11),
          ('data/snapshots_30x30_3000.npy', 'data/bestModelParameters_fieldVector_24.dat', '30x30 best', 30)]
times = [1000, 1500, 3000]
fig, axes = plt.subplots(2, 1+len(times), figsize=(3.0*(1+len(times)), 6.4))
for row, (snapfile, parfile, label, n) in enumerate(panels):
    V = np.load(snapfile)
    tgt = torch.load(parfile, weights_only=False)['trainParameters']['targetVmem'].detach().numpy().reshape(n, n)*1000
    vmin, vmax = min(tgt.min(), V.min()), max(tgt.max(), V.max())
    axes[row][0].imshow(tgt, cmap='gray', vmin=vmin, vmax=vmax)
    axes[row][0].set_title(f'{label}\ntarget', fontsize=9)
    for col, t in enumerate(times):
        img = V[min(t//10, len(V)-1)].reshape(n, n)
        ax = axes[row][col+1]
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        rms = np.sqrt(((img - tgt)**2).mean())
        scored = ' (scored here)' if t == 1000 else ''
        ax.set_title(f'iter {t}{scored}\nRMS to target {rms:.1f} mV', fontsize=9)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('The trained pattern does not persist: both models drift away after the scored window', fontsize=11)
fig.tight_layout()
fig.savefig('figures/patternPersistence.png', dpi=150)
print('  wrote figures/patternPersistence.png')
