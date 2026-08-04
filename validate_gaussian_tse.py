"""
Validate the Gaussian (spectral) TSE estimator against the discrete one on the 11x11
lattice, before adopting it for larger grids.

The discrete TSE estimator undersamples subsets larger than ~log(T)/log(numBins) cells,
and the resulting bias grows with numCells -- so it cannot be carried to a 30x30 lattice
unchanged. The Gaussian estimator has no such limit (the covariance is estimated once and
every subset entropy is a log-determinant), but it discards everything above second order.
The open question is whether the field-action-range optimum survives that reduction.

This script replicates the 'fixBiasSweepWeightScreenGJ' sweep conditions from
analyzeCellularFieldNetwork.py -- randomized initial G_pol, no clamp, free evolution --
over a subsampled parameter grid, and computes both estimators plus cheap spectral
summaries on identical trajectories.

Verdict criterion: if both estimators place the marginal optimum at the same field
screen size, the Gaussian estimator captures the effect and can be used at 30x30.

  python validate_gaussian_tse.py --numSamples 6 --numSimIters 5000
"""

import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities
from cellularFieldNetwork import cellularFieldNetwork

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--latticeDims',     type=str,   default='(11,11)')
parser.add_argument('--fieldScreenSizes', type=str,  default='[1,2,4,6,10,15,20]')
parser.add_argument('--sourceDat',       type=str,   default='data/StigmergicModelParameters.dat',
                    help='take GJ/field parameters from this trained model; only screen size is swept')
parser.add_argument('--numSamples',      type=int,   default=8)
parser.add_argument('--numSimIters',     type=int,   default=5000)
parser.add_argument('--burnInFrac',      type=float, default=0.5,
                    help='fraction of the run discarded so the relaxation transient does not '
                         'dominate the covariance')
parser.add_argument('--numScales',       type=int,   default=20)
parser.add_argument('--numSubsets',      type=int,   default=25)
parser.add_argument('--maxScale',        type=int,   default=None,
                    help='largest subset size for the Gaussian estimator. Pin this to the same '
                         'value across lattice sizes to make grids directly comparable; the '
                         'default (numCells-1) makes TSE scale with numCells and so is only '
                         'comparable within one lattice size.')
parser.add_argument('--skipDiscrete',    action='store_true', help='Gaussian + spectral only')
parser.add_argument('--seed',            type=int,   default=0)
parser.add_argument('--outputPrefix',    type=str,   default='data/gaussianTSEValidation')
args = parser.parse_args()

latticeDims = eval(args.latticeDims)
fieldScreenSizes = eval(args.fieldScreenSizes)
numCells = int(np.prod(latticeDims))
burnIn = int(args.burnInFrac * args.numSimIters)
utils = utilities.utilities()
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)

# Field transduction only produces spatial structure in a narrow regime: when
# |gain * eV| is comparable to the bias. At the argparse defaults of
# analyzeCellularFieldNetwork.py (gain=1.0, bias=0.03) the sigmoid is pinned at the bias
# and std(G_pol) is identically zero, so screen size cannot matter. The trained model's
# own field parameters are the known-good regime, so they are taken from file rather
# than reconstructed.
sourceParameters = torch.load(args.sourceDat, weights_only=False)
GJParameters = sourceParameters['GJParameters']
sourceFieldParameters = sourceParameters['fieldParameters']
print(f"field parameters from {args.sourceDat}: "
      f"gain={sourceFieldParameters['fieldTransductionGain']} "
      f"bias={sourceFieldParameters['fieldTransductionBias'].item():.4g} "
      f"weight={sourceFieldParameters['fieldTransductionWeight'].item():.4g} "
      f"strength={sourceFieldParameters['fieldStrength']} "
      f"| GJStrength={GJParameters['GJStrength']} "
      f"| trained screen size={sourceFieldParameters['fieldScreenSize']}")


def buildParameters(fieldScreenSize):
    fieldParameters = dict(sourceFieldParameters)
    fieldParameters['fieldScreenSize'] = int(fieldScreenSize)
    return {
        'GJParameters': GJParameters,
        'fieldParameters': fieldParameters,
        'ligandParameters': None,
        'GRNParameters': None,
        'ATPParameters': None,
    }


def defineInitialValues(circuit, numSamples):
    """Randomized G_pol per sample; sample 0 stays homogeneous.

    Follows defineInitialValues() in analyzeCellularFieldNetwork.py: uniform initial Vmem,
    zero field, and G_pol drawn from [0,2) in G_ref units so the ensemble spans both
    unistable and bistable cells.
    """
    allCells = list(range(circuit.numCells))
    initVmem = torch.full((numSamples, circuit.numCells, 1), -9.2e-3, dtype=torch.float64)
    values = [[torch.rand(circuit.numCells, dtype=torch.float64) * 2] for _ in range(numSamples)]
    values[0] = [torch.ones(circuit.numCells, dtype=torch.float64)]
    return {
        'Vmem': initVmem,
        'eV': torch.zeros((numSamples, circuit.numFieldGridPoints, 1), dtype=torch.float64),
        'ligandConc': torch.zeros((numSamples, circuit.numCells, 1), dtype=torch.float64),
        'G_pol': {'cells': [[allCells]] * numSamples, 'values': values},
        'G_dep': {'cells': [], 'values': torch.DoubleTensor([])},
    }


records = []
print(f"lattice {latticeDims}  |  {len(fieldScreenSizes)} screen sizes x {args.numSamples} samples")
print(f"numSimIters={args.numSimIters}  burnIn={burnIn}  scales={args.numScales}  "
      f"subsets={args.numSubsets}  discrete={'off' if args.skipDiscrete else 'on'}\n")

tStart = time.time()
for n, screenSize in enumerate(fieldScreenSizes):
    circuit = cellularFieldNetwork(latticeDims=latticeDims, latticePeriodicBoundary=False,
                                   parameters=buildParameters(screenSize),
                                   numSamples=args.numSamples)
    initialValues = defineInitialValues(circuit, args.numSamples)
    circuit.initVariables(initialValues)
    circuit.initParameters(initialValues)
    circuit.simulate(externalInputs={'gene': None}, numSimIters=args.numSimIters, saveData=True)

    # spatial spread of G_pol confirms the field actually differentiated cells;
    # if this is ~0 the regime is inert and screen size cannot matter
    gpolSpread = (circuit.G_pol[:, :, 0].std(dim=1) / circuit.G_ref).detach().numpy()

    for sample in range(args.numSamples):
        Vmem = circuit.timeseriesVmem[burnIn:, sample, :, 0].detach().numpy()
        gaussianTSE = utils.computeTSEComplexityGaussian(
            Vmem, numScales=args.numScales, numSubsets=args.numSubsets, rng=rng,
            maxScale=args.maxScale)
        spectral = utils.computeSpectralMetrics(Vmem)
        discreteTSE = np.nan
        if not args.skipDiscrete:
            discreteTSE = utils.computeTSEComplexityDiscrete(
                Vmem, numScales=args.numScales, numSubsets=args.numSubsets, rng=rng)
        records.append({
            'fieldScreenSize': int(screenSize), 'sample': sample,
            'gpolSpread': float(gpolSpread[sample]),
            'discreteTSE': float(discreteTSE), 'gaussianTSE': float(gaussianTSE),
            **{k: float(v) for k, v in spectral.items()},
        })

    recent = records[-args.numSamples:]
    elapsed = time.time() - tStart
    print(f"[{n+1:>2}/{len(fieldScreenSizes)}] screen={screenSize:>2} "
          f"| std(Gpol)/Gref={np.mean([r['gpolSpread'] for r in recent]):>7.4f} "
          f"| discrete={np.nanmean([r['discreteTSE'] for r in recent]):>9.3f} "
          f"gaussian={np.mean([r['gaussianTSE'] for r in recent]):>9.3f} "
          f"| {elapsed:.0f}s elapsed, ~{elapsed/(n+1)*(len(fieldScreenSizes)-n-1):.0f}s left")
    del circuit

# ── Marginal curves over field screen size ───────────────────────────────────────
metrics = ['discreteTSE', 'gaussianTSE', 'participationRatio', 'totalCorrelation',
           'spectralEntropy', 'gpolSpread']
marginal = {m: [] for m in metrics}
stderr = {m: [] for m in metrics}
for screenSize in fieldScreenSizes:
    subset = [r for r in records if r['fieldScreenSize'] == screenSize]
    for m in metrics:
        values = np.array([r[m] for r in subset])
        values = values[np.isfinite(values)]
        marginal[m].append(values.mean() if len(values) else np.nan)
        stderr[m].append(values.std() / np.sqrt(len(values)) if len(values) else np.nan)

print(f"\n{'screen':>7}" + ''.join(f"{m:>20}" for m in metrics))
print('-' * (7 + 20 * len(metrics)))
for i, screenSize in enumerate(fieldScreenSizes):
    print(f"{screenSize:>7}" + ''.join(f"{marginal[m][i]:>20.3f}" for m in metrics))

print(f'\nOptimum location over field screen size '
      f'(all other parameters fixed at the {args.sourceDat} values):')
for m in metrics:
    values = np.array(marginal[m])
    if not np.isfinite(values).any():
        continue
    best = int(np.nanargmax(values))
    interior = 0 < best < len(values) - 1
    print(f"  {m:>20}: max at screenSize={fieldScreenSizes[best]:<3} "
          f"{'INTERIOR PEAK' if interior else '(endpoint - monotone)'}")

finite = [(r['discreteTSE'], r['gaussianTSE']) for r in records
          if np.isfinite(r['discreteTSE']) and np.isfinite(r['gaussianTSE'])]
if finite:
    d, g = np.array(finite).T
    print(f"\nPer-run agreement over {len(finite)} runs:")
    print(f"  Pearson  r = {np.corrcoef(d, g)[0,1]:.3f}")
    order_d, order_g = np.argsort(np.argsort(d)), np.argsort(np.argsort(g))
    print(f"  Spearman r = {np.corrcoef(order_d, order_g)[0,1]:.3f}")

torch.save({'records': records, 'marginal': marginal, 'stderr': stderr,
            'fieldScreenSizes': fieldScreenSizes, 'args': vars(args)},
           f'{args.outputPrefix}.dat')

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
ax = axes[0]
for m, colour in [('discreteTSE', 'crimson'), ('gaussianTSE', 'steelblue')]:
    values, err = np.array(marginal[m]), np.array(stderr[m])
    if not np.isfinite(values).any():
        continue
    normalised = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values) + 1e-30)
    scale = (np.nanmax(values) - np.nanmin(values) + 1e-30)
    ax.errorbar(fieldScreenSizes, normalised, yerr=err / scale, marker='o',
                color=colour, capsize=3, label=m)
ax.set_xlabel('Field screen size (action range)')
ax.set_ylabel('Normalised complexity')
ax.set_title('Discrete vs Gaussian TSE\n(min-max normalised)')
ax.set_xticks(fieldScreenSizes)
ax.legend(fontsize=9)

ax = axes[1]
if finite:
    ax.scatter(d, g, s=18, alpha=0.6, color='darkslateblue')
    ax.set_xlabel('Discrete TSE')
    ax.set_ylabel('Gaussian TSE')
    ax.set_title(f'Per-run agreement (r = {np.corrcoef(d, g)[0,1]:.3f})')
else:
    ax.text(0.5, 0.5, 'discrete estimator skipped', ha='center', transform=ax.transAxes)
    ax.set_axis_off()

ax = axes[2]
for m, colour in [('participationRatio', 'seagreen'), ('spectralEntropy', 'darkorange')]:
    values = np.array(marginal[m])
    normalised = (values - values.min()) / (values.max() - values.min() + 1e-30)
    ax.plot(fieldScreenSizes, normalised, marker='s', color=colour, label=m)
ax.set_xlabel('Field screen size (action range)')
ax.set_ylabel('Normalised value')
ax.set_title('Cheap spectral summaries')
ax.set_xticks(fieldScreenSizes)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{args.outputPrefix}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {args.outputPrefix}.dat, {args.outputPrefix}.png")
