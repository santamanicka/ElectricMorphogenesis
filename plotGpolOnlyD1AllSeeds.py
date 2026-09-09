"""Visualize every individual trained checkpoint in the Gpol-only/D1 class of the 11x11 band-hold
sweep -- not just the four per-sub-config winners plot11x11BandSweepEvolution.py shows, but all 48
(T in {100,300} x lossMethod in {correlation,globalsum} x 12 seeds each).

For each checkpoint: target, pattern evolution (replayed from scratch via its own stored
clampParameters, every 300 iterations from t=1 to t=2701), and its stored best face (actualVmem). Row
labels carry the file number, bulk bestLoss, and (from the already-computed
data/facialFeatureScore11x11_all192.csv) the facial-feature-only RMS score and its timestep, so this
figure doubles as a visual cross-check of that CSV. File 1751 -- Sec 12.8's best face in the entire
192-checkpoint sweep -- is outlined in red if present in this class (it is: Gpol-only, D1, T=300,
globalsum).
"""
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

torch.set_grad_enabled(False)

# The four Gpol-only/D1 sub-configs of the sixteen-configuration sweep, by their base file number
# (batch 1 = base..base+5, batch 2 = base+96..base+101) -- same construction as
# plot11x11BandSweepEvolution.py / scoreAllFacialFeature11x11.py.
subConfigs = [
    (100, 'correlation', 1600),
    (100, 'globalsum', 1606),
    (300, 'correlation', 1648),
    (300, 'globalsum', 1654),
]

cases = []
for T, lossLabel, base in subConfigs:
    for batch, offset in [(1, 0), (2, 96)]:
        for seedInBatch in range(6):
            n = base + offset + seedInBatch
            cases.append((n, T, lossLabel, batch, seedInBatch))

print(f"{len(cases)} checkpoints in the Gpol-only/D1 class")

# Pull the already-computed facial-feature scores for these files rather than replaying twice.
featureScores = {}
with open('data/facialFeatureScore11x11_all192.csv') as fh:
    for row in csv.DictReader(fh):
        featureScores[int(row['fileNumber'])] = (float(row['featureRMS_mV']), int(row['featureBestTimestep']))


def replayEvolution(p, numSimIters, snapIters):
    p = dict(p)
    p['latticePeriodicBoundaryGJ'] = False
    p['ATPParameters'] = None
    system = model(p, p['simParameters']['numSamples'])
    system.setExperimentalConditions((p['simParameters']['initialValues'], p['simParameters']['numSamples']))
    circuit = system.electricNetwork
    clampParameters = dict(p['clampParameters'])
    clampEndIter = int(clampParameters['clampEndIter'])
    wanted = set(snapIters)
    snaps = {}
    for it in range(numSimIters):
        cp = clampParameters if it <= clampEndIter else None
        system.simulate(clampParameters=cp, numSimIters=1, outerIter=it, fieldModulation=False)
        if it in wanted:
            snaps[it] = circuit.Vmem[0, :, 0].detach().numpy().copy() * 1000
    return snaps


stride = 300
snapIters = list(range(1, 3000, stride))  # 1, 301, ..., 2701
numCols = 1 + len(snapIters) + 1  # target + evolution + best face
rows, cols = 11, 11

fig, axes = plt.subplots(len(cases), numCols, figsize=(1.1 * numCols, 1.2 * len(cases)))

for row, (n, T, lossLabel, batch, seedInBatch) in enumerate(cases):
    f = f'data/bestModelParameters_fieldVector_11x11_{n}.dat'
    p = torch.load(f, map_location='cpu', weights_only=False)
    target = p['trainParameters']['targetVmem'].reshape(rows, cols).numpy() * 1000
    actual = p['trainParameters']['actualVmem'].reshape(rows, cols).numpy() * 1000
    bulkLoss = float(p['trainParameters']['bestLoss'])
    numSimIters = p['simParameters']['numSimIters']
    featureRMS, featureIter = featureScores.get(n, (float('nan'), -1))

    label = (f'T={T} {lossLabel[:4]}\nseed{seedInBatch} b{batch}\nfile {n}\n'
             f'bulk={bulkLoss:.2f}\nfeat={featureRMS:.1f}mV')
    highlight = (n == 1751)

    axTarget = axes[row, 0]
    axTarget.imshow(target, cmap='gray')
    axTarget.set_ylabel(label, fontsize=6, rotation=0, ha='right', va='center')
    if row == 0:
        axTarget.set_title('target', fontsize=8)
    axTarget.set_xticks([]); axTarget.set_yticks([])

    snaps = replayEvolution(p, numSimIters, snapIters)
    for col, it in enumerate(snapIters):
        ax = axes[row, col + 1]
        if it in snaps:
            ax.imshow(snaps[it].reshape(rows, cols), cmap='gray')
        else:
            ax.axis('off')
        if row == 0:
            ax.set_title(f't={it}', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    axFinal = axes[row, -1]
    axFinal.imshow(actual, cmap='gray')
    axFinal.set_title('best face' if row == 0 else '', fontsize=7)
    axFinal.set_xticks([]); axFinal.set_yticks([])

    if highlight:
        for ax in axes[row, :]:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2.5)
            ax.set_visible(True)

    print(f"  [{row+1}/{len(cases)}] file {n}: bulkLoss={bulkLoss:.3f}, featureRMS={featureRMS:.2f}mV @ t={featureIter}")

fig.suptitle('11x11 Gpol-only, D1: all 48 trained seeds (T in {100,300} x loss in {correlation,globalsum}), '
             'target -> evolution (t=1..2701, every 300) -> best face. Red outline = file 1751 (Sec 12.8 best face).',
             fontsize=11)
fig.tight_layout()
outPath = 'figures/gpolOnlyD1AllSeeds11x11.png'
fig.savefig(outPath, dpi=110, bbox_inches='tight')
print(f'\nwrote {outPath}')
