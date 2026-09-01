"""Perturb the scaled (11x11->30x30) prepattern with 2-fold (left-right mirror) symmetric noise, and
check whether the t~680 face-suggestive echo (seen in the unperturbed screen=10.909,
strength=0.3667 release) is a robust feature of a neighborhood of states, or a fragile coincidence
of that one exact interpolated point.

Symmetry convention matches the rest of this codebase's 'twofold' clamps (see
utilities.computeSymmetricalIndices, used identically in map_basin_structure.py): reflection is
about the vertical mid-column, i.e. column c's mirror partner is column (numCols-1-c). Noise is
drawn independently per left-half cell and copied onto its mirror partner, so base + noise stays
exactly 2-fold symmetric (the base scaled prepattern already is, to float precision).
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['perturb', 'random', 'coarse'], default='perturb',
                     help="'perturb': add symmetric noise to the actual scaled prepattern. "
                          "'random': ignore the learned prepattern's spatial content entirely and draw "
                          "an i.i.d.-per-cell field matching only its mean/std, then symmetrize. "
                          "'coarse': keep the real prepattern's coarse spatial envelope (boundary-vs-"
                          "interior trend, gross shape) but replace its fine-detail residual with "
                          "symmetric random noise matched to the residual's own std -- tests whether "
                          "the echo needs the specific feature placement, or just the right gross shape.")
parser.add_argument('--coarseBlockSize', type=int, default=5,
                     help="coarse mode only: block size for the average-pool/upsample that defines "
                          "'overall structure' (30 must be divisible by it; 5 -> 6x6 coarse grid)")
parser.add_argument('--fieldScreenSize', type=float, default=10.909090909090908)
parser.add_argument('--fieldStrength', type=float, default=0.3667)
parser.add_argument('--numSimIters', type=int, default=1000)
parser.add_argument('--numTrials', type=int, default=8)
parser.add_argument('--vmemNoiseStd', type=float, default=0.0005, help='volts (0.5 mV); perturb mode only')
parser.add_argument('--gpolNoiseStd', type=float, default=0.1, help='ratio to G_ref; perturb mode only')
parser.add_argument('--readIter', type=int, default=680)
parser.add_argument('--seed0', type=int, default=1000)
args = parser.parse_args()

PREPATTERN_FILE = './data/scaledFacePrepattern_11to30.dat'
BASE_PARAMETER_FILE = './data/StigmergicModelParameters.dat'
STORE_STRIDE = 10
tag = f'{args.mode}_strength{args.fieldStrength:g}'
FIGURE = f'./figures/perturbedPrepatternSymmetric_{tag}.png'
OUTPUT_DATA = f'./data/perturbedPrepatternSymmetric_{tag}.dat'

prepattern = torch.load(PREPATTERN_FILE, weights_only=False)
rows, cols = prepattern['targetRows'], prepattern['targetCols']
numCells = rows * cols
baseVmem = prepattern['scaledVmem'].reshape(rows, cols).clone()
baseGpolRatio = prepattern['scaledGpolRatio'].reshape(rows, cols).clone()
scaledTarget1000 = prepattern['scaledTarget1000'].reshape(rows, cols).numpy()

# left-right mirror index map, matching utilities.computeSymmetricalIndices(symmetry='twofold')
mirrorCol = cols - 1 - np.arange(cols)
leftCols = np.arange(cols // 2)  # cols=30 -> 0..14, mirror partners 29..15, exact bijection

base = torch.load(BASE_PARAMETER_FILE, weights_only=False)
fieldParameters = dict(prepattern['fieldParameters'])
fieldParameters['fieldScreenSize'] = args.fieldScreenSize
fieldParameters['fieldStrength'] = args.fieldStrength
parameters = {
    'latticeDims': (rows, cols),
    'GJParameters': dict(base['GJParameters']),
    'fieldParameters': fieldParameters,
    'ligandParameters': dict(base['ligandParameters']),
    'GRNParameters': dict(base['GRNParameters']),
    'latticePeriodicBoundaryGJ': False,
    'ATPParameters': None,
}
numSamples = 1
initialValuesTemplate = {
    'Vmem': torch.zeros((numSamples, numCells, 1), dtype=torch.float64),
    'eV': torch.zeros((numSamples, (rows + 1) * (cols + 1), 1), dtype=torch.float64),
    'ligandConc': torch.zeros((numSamples, numCells, 1), dtype=torch.float64),
    'G_pol': {'cells': [[[0]]] * numSamples, 'values': [torch.DoubleTensor([1.0])] * numSamples},
    'G_dep': {'cells': [], 'values': torch.DoubleTensor([])},
}


def symmetricField(std, seed, generatorOffset, mean=0.0):
    g = torch.Generator().manual_seed(seed * 7919 + generatorOffset)
    field = torch.full((rows, cols), mean, dtype=torch.float64)
    if std == 0:
        return field
    leftValues = torch.randn(rows, len(leftCols), generator=g, dtype=torch.float64) * std + mean
    field[:, leftCols] = leftValues
    field[:, mirrorCol[leftCols]] = leftValues
    return field


symmetricNoise = symmetricField  # mean=0 default; perturb mode adds this to the base field
vmemMean, vmemStd = baseVmem.mean().item(), baseVmem.std().item()
gpolMean, gpolStd = baseGpolRatio.mean().item(), baseGpolRatio.std().item()


def coarseEnvelope(field2d, blockSize):
    t = field2d.reshape(1, 1, rows, cols)
    pooled = F.avg_pool2d(t, kernel_size=blockSize)
    up = F.interpolate(pooled, size=(rows, cols), mode='bilinear', align_corners=False)
    return up.reshape(rows, cols)


if args.mode == 'coarse':
    coarseVmem = coarseEnvelope(baseVmem, args.coarseBlockSize)
    coarseGpol = coarseEnvelope(baseGpolRatio, args.coarseBlockSize)
    fineResidualVmemStd = (baseVmem - coarseVmem).std().item()
    fineResidualGpolStd = (baseGpolRatio - coarseGpol).std().item()
    asym = (coarseVmem - coarseVmem.flip(1)).abs().max().item()
    print(f"  coarse envelope: blockSize={args.coarseBlockSize} ({rows//args.coarseBlockSize}x"
          f"{cols//args.coarseBlockSize} grid)  fine-residual std: Vmem={fineResidualVmemStd*1000:.4f}mV "
          f"Gpol={fineResidualGpolStd:.4f}  (coarse-envelope symmetry check: max asym {asym*1000:.2e}mV)")


def release(vmem, gpolRatio):
    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((initialValuesTemplate, numSamples))
    circuit = instance.electricNetwork
    circuit.Vmem = vmem.reshape(1, numCells, 1).clone()
    circuit.G_pol = (gpolRatio.reshape(1, numCells, 1).clone() * circuit.G_ref)
    instance.simulate(clampParameters=None, fieldModulation=True, numSimIters=args.numSimIters,
                       storeVariables=['Vmem'], storeStride=STORE_STRIDE)
    storedIters = np.array(instance.storedIters)
    V = np.stack([v[0, :, 0].detach().numpy() for v in instance.timeseriesVmem]) * 1000
    targetFlat = scaledTarget1000.reshape(-1) * 1000
    correlation = np.array([np.corrcoef(frame, targetFlat)[0, 1] for frame in V])
    return storedIters, V, correlation


trials = [('baseline', baseVmem, baseGpolRatio)]
for i in range(args.numTrials):
    seed = args.seed0 + i
    if args.mode == 'perturb':
        vmemNoise = symmetricNoise(args.vmemNoiseStd, seed, 1)
        gpolNoise = symmetricNoise(args.gpolNoiseStd, seed, 2)
        trialVmem = baseVmem + vmemNoise
        trialGpol = torch.clip(baseGpolRatio + gpolNoise, 0.0, 2.0)
    elif args.mode == 'random':  # ignore the learned spatial content, match only mean/std, i.i.d. per cell
        trialVmem = symmetricField(vmemStd, seed, 1, mean=vmemMean)
        trialGpol = torch.clip(symmetricField(gpolStd, seed, 2, mean=gpolMean), 0.0, 2.0)
    else:  # coarse: keep the real coarse envelope, randomize only the fine-detail residual
        vmemResidual = symmetricField(fineResidualVmemStd, seed, 1, mean=0.0)
        gpolResidual = symmetricField(fineResidualGpolStd, seed, 2, mean=0.0)
        trialVmem = coarseVmem + vmemResidual
        trialGpol = torch.clip(coarseGpol + gpolResidual, 0.0, 2.0)
    trials.append((f'seed{seed}', trialVmem, trialGpol))

results = []
if args.mode == 'perturb':
    print(f"  mode=perturb  screen={args.fieldScreenSize:g}  strength={args.fieldStrength:g}  "
          f"vmemNoiseStd={args.vmemNoiseStd*1000:g}mV  gpolNoiseStd={args.gpolNoiseStd:g}  "
          f"readIter={args.readIter}")
elif args.mode == 'random':
    print(f"  mode=random (stats-matched, structure ignored)  screen={args.fieldScreenSize:g}  "
          f"strength={args.fieldStrength:g}  Vmem~N({vmemMean*1000:.3f},{vmemStd*1000:.3f})mV  "
          f"Gpol~N({gpolMean:.3f},{gpolStd:.3f})  readIter={args.readIter}  "
          f"('baseline' below is still the real learned-scaled prepattern, for reference)")
else:
    print(f"  mode=coarse (real envelope, random fine detail)  screen={args.fieldScreenSize:g}  "
          f"strength={args.fieldStrength:g}  readIter={args.readIter}  "
          f"('baseline' below is still the real learned-scaled prepattern, for reference)")
for label, vmem, gpol in trials:
    storedIters, V, correlation = release(vmem, gpol)
    readRow = int(np.searchsorted(storedIters, args.readIter))
    readRow = min(readRow, len(storedIters) - 1)
    print(f"  {label:10s}  r@iter{args.readIter}={correlation[readRow]:+.3f}  "
          f"peak r={correlation.max():+.3f} at iter {storedIters[np.argmax(correlation)]}")
    prepatternMv = vmem.reshape(rows, cols).numpy() * 1000
    results.append((label, storedIters, V, correlation, readRow, prepatternMv))

torch.save({'results': [(label, si, V, corr, rr, pp) for label, si, V, corr, rr, pp in results],
            'scaledTarget1000_mV': scaledTarget1000 * 1000, 'rows': rows, 'cols': cols,
            'readIter': args.readIter}, OUTPUT_DATA)
print(f'  wrote {OUTPUT_DATA}')

# ── figure ───────────────────────────────────────────────────────────────
n = len(results)
fig = plt.figure(figsize=(2.1 * (n + 1), 7.6))
grid = gridspec.GridSpec(3, n + 1, height_ratios=[1, 1, 1.2], hspace=0.55, wspace=0.12)

axTarget = fig.add_subplot(grid[0, 0])
axTarget.imshow(scaledTarget1000, cmap='gray')
axTarget.set_title('target', fontsize=8)
axTarget.set_xticks([]); axTarget.set_yticks([])
fig.add_subplot(grid[1, 0]).axis('off')

for col, (label, storedIters, V, correlation, readRow, prepatternMv) in enumerate(results, start=1):
    axPre = fig.add_subplot(grid[0, col])
    axPre.imshow(prepatternMv, cmap='gray', vmin=prepatternMv.min(), vmax=prepatternMv.max())
    axPre.set_title(f'{label}\nprepattern (t=0)', fontsize=7)
    axPre.set_xticks([]); axPre.set_yticks([])

    frame = V[readRow].reshape(rows, cols)
    axis = fig.add_subplot(grid[1, col])
    axis.imshow(frame, cmap='gray', vmin=frame.min(), vmax=frame.max())
    axis.set_title(f'iter {storedIters[readRow]}, r={correlation[readRow]:.2f}', fontsize=7)
    axis.set_xticks([]); axis.set_yticks([])

axCorr = fig.add_subplot(grid[2, :])
for label, storedIters, V, correlation, readRow, prepatternMv in results:
    style = dict(linewidth=2.2, color='black') if label == 'baseline' else dict(linewidth=1.0, alpha=0.8)
    axCorr.plot(storedIters, correlation, label=label, **style)
axCorr.axvline(args.readIter, color='gray', linewidth=0.7, linestyle='--')
axCorr.axhline(0, color='gray', linewidth=0.5)
axCorr.set_xlabel('iteration'); axCorr.set_ylabel('correlation to\n11x11 iter-1000 target')
axCorr.legend(fontsize=6, ncol=5, loc='upper right')
if args.mode == 'perturb':
    axCorr.set_title(f'2-fold-symmetric perturbations of the prepattern (Vmem std={args.vmemNoiseStd*1000:g}mV, '
                      f'Gpol std={args.gpolNoiseStd:g})', fontsize=9)
    fig.suptitle(f'Does the t~{args.readIter} face echo survive symmetric perturbation of the initial condition?',
                 fontsize=11)
elif args.mode == 'random':
    axCorr.set_title('Fully random 2-fold-symmetric fields, matched only to the prepattern\'s mean/std '
                      '(spatial structure discarded); "baseline" = real learned-scaled prepattern', fontsize=9)
    fig.suptitle(f'Does the t~{args.readIter} face echo need the learned prepattern\'s structure, '
                 f'or just its amplitude + symmetry?', fontsize=11)
else:
    axCorr.set_title(f'Real coarse envelope (blockSize={args.coarseBlockSize}) + random symmetric fine-detail '
                      f'residual; "baseline" = real learned-scaled prepattern', fontsize=9)
    fig.suptitle(f'Does the t~{args.readIter} face echo need the specific feature placement, '
                 f'or just the gross coarse shape?', fontsize=11)
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'  wrote {FIGURE}')
