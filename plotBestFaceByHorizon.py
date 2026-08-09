"""Draw the best face at each training horizon beside the target.

The two horizons are shown together because their losses are on the same scale -- same formula, same
100 iteration readout, same target -- but they score the tissue at different points in its evolution,
so the pictures are what settle whether the shorter horizon was reading an unfinished pattern or a
different one. The 11x11 reference is included as the only case known to resolve the features.
"""
import glob, os, re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def bestAt(horizon, pattern):
    """Best face model at this horizon.

    The reachability control writes bestModelParameters_fieldVector_ensemble_*, whose targets are
    ensemble patterns on an unrelated loss scale, so a bare glob picks one of those as 'best' and
    reports a loss of 0.01 against the face's 7.6. The name is filtered and the target is then
    checked against the face itself, since a name is a convention and the target is the fact.
    """
    best = None
    # the training target is the interior features plus the dome border, so it is identified by its
    # own cell count taken from a known face run rather than rebuilt from faceFeatureIndices, which
    # returns the interior features alone
    faceCells = int((torch.load('data/bestModelParameters_fieldVector_24.dat', map_location='cpu',
                                weights_only=False)['trainParameters']['targetVmem'].detach().numpy()
                     < -0.03).sum())
    for f in glob.glob(pattern):
        tail = os.path.basename(f).split('fieldVector_', 1)[1].rsplit('.dat', 1)[0]
        if not re.fullmatch(r'(\d+x\d+_)?\d+', tail):
            continue
        p = torch.load(f, map_location='cpu', weights_only=False)
        if tuple(p.get('latticeDims') or ()) != (30, 30):
            continue
        if int(p['simParameters']['numSimIters']) != horizon:
            continue
        if int((p['trainParameters']['targetVmem'].detach().numpy() < -0.03).sum()) != faceCells:
            continue
        L = float(p['trainParameters']['bestLoss'])
        if best is None or L < best[1]:
            best = (f, L)
    return best

panels = []
ref = 'data/StigmergicModelParameters.dat'
p = torch.load(ref, map_location='cpu', weights_only=False)
panels.append((ref, '11x11 reference', 11, int(p['simParameters']['numSimIters']),
               float(p['trainParameters']['bestLoss'])))
for horizon in (1000, 2500):
    b = bestAt(horizon, 'data/bestModelParameters_fieldVector_*.dat')
    if b is None:
        continue
    panels.append((b[0], f'30x30 best', 30, horizon, b[1]))

fig, axes = plt.subplots(2, len(panels), figsize=(3.4*len(panels), 7.0))
for col, (path, label, n, horizon, loss) in enumerate(panels):
    p = torch.load(path, map_location='cpu', weights_only=False)
    tp = p['trainParameters']
    tgt = tp['targetVmem'].detach().numpy().reshape(n, n)*1000
    act = tp['actualVmem'].detach().numpy().reshape(n, n)*1000
    vmin, vmax = min(tgt.min(), act.min()), max(tgt.max(), act.max())
    axes[0][col].imshow(tgt, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0][col].set_title(f'{label}\ntarget', fontsize=10)
    axes[1][col].imshow(act, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1][col].set_title(f'{horizon} iter horizon\nloss {loss:.4g}', fontsize=10)
    print(f"  {label:18s} horizon {horizon:5d}  loss {loss:8.4f}  {os.path.basename(path)}")
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Best face at each horizon (30x30 runs still in progress)', fontsize=12)
fig.tight_layout()
fig.savefig('figures/bestFaceByHorizon.png', dpi=150)
print('  wrote figures/bestFaceByHorizon.png')
