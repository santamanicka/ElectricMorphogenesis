"""Replay an already-trained checkpoint's clamp forcing under a longer simulation horizon than it
was trained at, to test whether the target is reachable later in the free evolution even though
training's own readout window (numSimIters, fixed at training time) never found it there.

Loss is recomputed with the checkpoint's own lossMethod, using the training formula's readout
window (readoutIters, an absolute iteration count, not a proportion -- see runTargetTraining.sh's
note on why proportions silently change protocol at a different horizon) ending at every stride
point, so the reported number at iteration numSimIters(original) matches the checkpoint's own
bestLoss exactly as a sanity check.
"""
import argparse

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--fileNumber', type=int, default=1804)
parser.add_argument('--numSimIters', type=int, default=5000)
parser.add_argument('--readoutIters', type=int, default=100)
parser.add_argument('--stride', type=int, default=250)
args = parser.parse_args()

torch.set_grad_enabled(False)

sourcePath = f'data/bestModelParameters_fieldVector_30x30_{args.fileNumber}.dat'
p = torch.load(sourcePath, map_location='cpu', weights_only=False)
rows, cols = p['latticeDims']
numSamples = p['simParameters']['numSamples']
originalNumSimIters = p['simParameters']['numSimIters']
lossMethod = p['trainParameters']['lossMethod']
reportedBestLoss = float(p['trainParameters']['bestLoss'])
clampParameters = dict(p['clampParameters'])
p['latticePeriodicBoundaryGJ'] = False
p['ATPParameters'] = None

system = model(p, numSamples)
circuit = system.electricNetwork
system.simulate(clampParameters=clampParameters, numSimIters=args.numSimIters, storeVariables=['Vmem'])
vmem = system.timeseriesVmem  # (numSimIters,numSamples,numCells,1)
target = p['trainParameters']['targetVmem']  # (numSamples,numCells,1)


def windowedLoss(endIter):
    window = vmem[endIter - args.readoutIters:endIter]
    if lossMethod == 'correlation':
        centredObserved = window - window.mean(dim=2, keepdim=True)
        centredTarget = target - target.mean(dim=1, keepdim=True)
        covariance = (centredObserved * centredTarget).sum(dim=2)
        normalisation = (centredObserved.pow(2).sum(dim=2).sqrt() * centredTarget.pow(2).sum(dim=1).sqrt())
        return (1 - (covariance / (normalisation + 1e-12))).mean().item()
    else:  # globalsum
        return ((target - window) ** 2).sum().sqrt().item()


checkpoints = list(range(args.readoutIters, args.numSimIters + 1, args.stride))
if checkpoints[-1] != args.numSimIters:
    checkpoints.append(args.numSimIters)

print(f"checkpoint {args.fileNumber}: lossMethod={lossMethod}, trained at numSimIters={originalNumSimIters}, "
      f"reported bestLoss={reportedBestLoss:.4f}")
print(f"replayed loss at iteration {originalNumSimIters} (sanity check, should be close to reported): "
      f"{windowedLoss(originalNumSimIters):.4f}")
print()
losses = {}
for it in checkpoints:
    L = windowedLoss(it)
    losses[it] = L
    flag = '  <-- best so far' if L == min(losses.values()) else ''
    print(f"  iter {it:5d}: loss {L:.4f}{flag}")

bestIter = min(losses, key=losses.get)
print(f"\nbest loss over extended horizon: {losses[bestIter]:.4f} at iteration {bestIter} "
      f"(vs reported {reportedBestLoss:.4f} at {originalNumSimIters})")

# Evolution snapshots
snapIters = list(range(0, args.numSimIters, args.stride))
if snapIters[-1] != args.numSimIters - 1:
    snapIters.append(args.numSimIters - 1)
numCols = len(snapIters)
fig = plt.figure(figsize=(2.0 * numCols, 6.5))
gs = fig.add_gridspec(2, numCols, height_ratios=[2, 1], hspace=0.35)
for col, it in enumerate(snapIters):
    frame = vmem[it, 0, :, 0].reshape(rows, cols).numpy() * 1000
    axImg = fig.add_subplot(gs[0, col])
    axImg.imshow(frame, cmap='gray')
    axImg.set_title(f'iter {it}', fontsize=9)
    axImg.set_xticks([]); axImg.set_yticks([])
axIterations = list(losses.keys())
axLossValues = [losses[it] for it in axIterations]
axLoss = fig.add_subplot(gs[1, :])
axLoss.plot(axIterations, axLossValues, marker='o')
axLoss.axvline(originalNumSimIters, color='gray', linestyle='--', label=f'trained horizon ({originalNumSimIters})')
axLoss.axhline(reportedBestLoss, color='red', linestyle=':', label=f'reported bestLoss ({reportedBestLoss:.3f})')
axLoss.set_xlabel('iteration (end of readout window)')
axLoss.set_ylabel(f'{lossMethod} loss')
axLoss.legend(fontsize=8)
fig.suptitle(f'File {args.fileNumber}: pattern evolution and windowed loss out to {args.numSimIters} iterations', fontsize=12, y=1.02)
outputPath = f'figures/extendedHorizon_{args.fileNumber}.png'
fig.savefig(outputPath, dpi=140, bbox_inches='tight')
print(f"\nwrote {outputPath}")
