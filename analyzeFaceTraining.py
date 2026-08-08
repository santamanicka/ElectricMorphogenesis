"""Collect face training runs and show the best pattern against the target.

Runs are judged as a batch, never individually: training escapes its flat initial state in only a
minority of random restarts, so the interesting quantity is the best of many and the escape rate,
not any one trajectory.
"""
import argparse, glob, os, re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--latticeRows', type=int, default=30)
parser.add_argument('--latticeCols', type=int, default=30)
parser.add_argument('--pattern', type=str, default='data/bestModelParameters_fieldVector_*.dat')
parser.add_argument('--output', type=str, default='figures/faceTraining.png')
args = parser.parse_args()

runs = []
for f in sorted(glob.glob(args.pattern)):
    try:
        p = torch.load(f, map_location='cpu', weights_only=False)
    except Exception:
        continue
    # Only face runs. Other targets write the same prefix with their name inserted --
    # ..._fieldVector_ensemble_0.dat is a reachability run whose loss is on an unrelated scale --
    # and averaging those in would silently corrupt every summary below.
    tail = os.path.basename(f).split('fieldVector_', 1)[1].rsplit('.dat', 1)[0]
    if not re.fullmatch(r'(\d+x\d+_)?\d+', tail):
        continue
    dims = p.get('latticeDims')
    if tuple(dims or ()) != (args.latticeRows, args.latticeCols):
        continue   # a run at another lattice size, which older files can silently contain
    tp = p.get('trainParameters', {})
    if 'bestLoss' not in tp:
        continue
    runs.append((float(tp['bestLoss']), f, tp))

if not runs:
    raise SystemExit(f"no {args.latticeRows}x{args.latticeCols} runs matched {args.pattern}")
runs.sort(key=lambda r: r[0])

losses = np.array([r[0] for r in runs])
# Most restarts stall on a plateau and land within a hair of each other, so the median marks that
# plateau. An escape is a run clearly below it rather than one merely better than the worst.
plateau = float(np.median(losses))
escaped = int((losses < plateau * 0.99).sum())
print(f"  {len(runs)} runs at {args.latticeRows}x{args.latticeCols}")
print(f"  best {losses.min():.6g}, plateau (median) {plateau:.6g}, worst {losses.max():.6g}")
print(f"  {escaped} of {len(runs)} escaped the plateau (>1% below it)")
for L, f, _ in runs[:5]:
    print(f"    {L:.6g}  {os.path.basename(f)}")

bestLoss, bestFile, bestTrain = runs[0]
rows, cols = args.latticeRows, args.latticeCols
target = bestTrain['targetVmem'].detach().numpy().reshape(rows, cols) * 1000.0
actual = bestTrain['actualVmem'].detach().numpy().reshape(rows, cols) * 1000.0

os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
lim = max(abs(target).max(), abs(actual).max())
for ax, img, title in [(axes[0], target, 'target'),
                       (axes[1], actual, f'best of {len(runs)} (loss {bestLoss:.4g})')]:
    m = ax.imshow(img, cmap='viridis', vmin=-lim, vmax=0)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(m, ax=ax, fraction=0.046, label='Vmem (mV)')
axes[2].hist(losses, bins=min(20, len(losses)), color='0.4')
axes[2].axvline(bestLoss, color='crimson', lw=2, label=f'best {bestLoss:.4g}')
axes[2].axvline(plateau, color='steelblue', lw=1.5, ls='--', label=f'plateau {plateau:.4g}')
axes[2].set_xlabel('best loss per restart'); axes[2].set_ylabel('restarts')
axes[2].set_title('restarts are judged as a batch'); axes[2].legend()
fig.suptitle(f'Face training at {rows}x{cols}: {escaped}/{len(runs)} restarts escaped the plateau')
fig.tight_layout()
fig.savefig(args.output, dpi=150)
print(f"  wrote {args.output}")
