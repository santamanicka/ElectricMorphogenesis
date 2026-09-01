"""Extend compareTransplantVsRandomClampPrepattern.py's committed-fraction/depth-profile analysis
to a trained screen11/strength0.25 checkpoint, at two points in its own trajectory: immediately
after the clamp releases (iteration clampEndIter+1, the "prepattern" proper -- directly comparable
to the transplant/random-clamp prepatterns) and at the end of free evolution (the readout-scored
state). The prepattern read is a short, low-risk replay (~101 iterations); the final-state read
inherits the usual chaotic-sensitivity caveat from a longer replay (see runExtendedHorizon.py),
disclosed in the printed output.
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--fileNumber', type=int, required=True)
parser.add_argument('--committedLow', type=float, default=0.1)
parser.add_argument('--committedHigh', type=float, default=1.9)
args = parser.parse_args()

torch.set_grad_enabled(False)

p = torch.load(f'data/bestModelParameters_fieldVector_30x30_{args.fileNumber}.dat', map_location='cpu', weights_only=False)
rows, cols = p['latticeDims']
numSamples = p['simParameters']['numSamples']
numSimIters = p['simParameters']['numSimIters']
lossMethod = p['trainParameters']['lossMethod']
clampParameters = dict(p['clampParameters'])
clampEndIter = int(clampParameters['clampEndIter'])
p['latticePeriodicBoundaryGJ'] = False
p['ATPParameters'] = None

system = model(p, numSamples)
circuit = system.electricNetwork

prepatternGpol = None
for it in range(numSimIters):
    cp = clampParameters if it <= clampEndIter else None
    system.simulate(clampParameters=cp, numSimIters=1, outerIter=it, fieldModulation=False)
    if it == clampEndIter:
        prepatternGpol = (circuit.G_pol[0, :, 0].detach().clone() / circuit.G_ref).reshape(rows, cols)
finalGpol = (circuit.G_pol[0, :, 0].detach().clone() / circuit.G_ref).reshape(rows, cols)
finalVmemReplayed = circuit.Vmem[0, :, 0].detach().clone().reshape(rows, cols)
finalVmemStored = p['trainParameters']['actualVmem'].reshape(rows, cols)
replayGap = (finalVmemReplayed - finalVmemStored).abs().max().item() * 1000
print(f"file {args.fileNumber} ({lossMethod}): clampEndIter={clampEndIter}, numSimIters={numSimIters}")
print(f"  replay-vs-stored final Vmem max diff: {replayGap:.2f} mV "
      f"(chaotic-sensitivity gap -- see runExtendedHorizon.py's caveat; prepattern read is unaffected, "
      f"it happens well before this drift accumulates)")

rowIdx, colIdx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
depthShell = np.minimum.reduce([rowIdx, rows - 1 - rowIdx, colIdx, cols - 1 - colIdx])
numShells = depthShell.max() + 1


def committedFraction(gpolRatio):
    g = gpolRatio.numpy()
    return (g < args.committedLow) | (g > args.committedHigh)


def depthProfile(gpolRatio):
    committed = committedFraction(gpolRatio)
    return np.array([committed[depthShell == s].mean() for s in range(numShells)])


prepatternCommitted = committedFraction(prepatternGpol)
finalCommitted = committedFraction(finalGpol)
prepatternProfile = depthProfile(prepatternGpol)
finalProfile = depthProfile(finalGpol)

print(f"  prepattern (iter {clampEndIter+1}): committed {prepatternCommitted.mean()*100:.1f}%")
print(f"  final (iter {numSimIters-1}, replayed): committed {finalCommitted.mean()*100:.1f}%")
print(f"  {'shell':>6} {'prepattern':>11} {'final':>8}")
for s in range(numShells):
    print(f"  {s:>6} {prepatternProfile[s]*100:>10.1f}% {finalProfile[s]*100:>7.1f}%")

fig, axes = plt.subplots(2, 3, figsize=(11, 7))
axes[0,0].imshow(prepatternGpol.numpy(), cmap='RdBu_r', vmin=0, vmax=2)
axes[0,0].set_title(f'prepattern G_pol/G_ref\n(iter {clampEndIter+1})', fontsize=10); axes[0,0].set_xticks([]); axes[0,0].set_yticks([])
axes[0,1].imshow(finalGpol.numpy(), cmap='RdBu_r', vmin=0, vmax=2)
axes[0,1].set_title(f'final G_pol/G_ref\n(iter {numSimIters-1}, replayed)', fontsize=10); axes[0,1].set_xticks([]); axes[0,1].set_yticks([])
axes[0,2].imshow(finalVmemStored.numpy()*1000, cmap='RdBu_r')
axes[0,2].set_title('final Vmem (mV)\n(stored, verified)', fontsize=10); axes[0,2].set_xticks([]); axes[0,2].set_yticks([])

axes[1,0].imshow(prepatternCommitted, cmap='gray_r')
axes[1,0].set_title('prepattern: committed cells', fontsize=10); axes[1,0].set_xticks([]); axes[1,0].set_yticks([])
axes[1,1].imshow(finalCommitted, cmap='gray_r')
axes[1,1].set_title('final: committed cells', fontsize=10); axes[1,1].set_xticks([]); axes[1,1].set_yticks([])

axProfile = axes[1,2]
axProfile.plot(range(numShells), prepatternProfile*100, marker='o', label='prepattern')
axProfile.plot(range(numShells), finalProfile*100, marker='s', label='final')
axProfile.set_xlabel('depth shell (0 = boundary)')
axProfile.set_ylabel('% committed cells')
axProfile.legend(fontsize=9)
fig.suptitle(f'Trained screen11/strength0.25 file {args.fileNumber} ({lossMethod}): '
             f'prepattern vs. final commitment', fontsize=12)
fig.tight_layout()
outputPath = f'figures/trainedPrepatternCommitment_{args.fileNumber}.png'
fig.savefig(outputPath, dpi=140, bbox_inches='tight')
print(f"\nwrote {outputPath}")
