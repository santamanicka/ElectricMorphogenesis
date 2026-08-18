"""
Decompose free-running (or clamped) Vmem dynamics into spatial basis patterns.

Runs a tissue for --numSimIters, takes the last --decompositionWindow iterations as a
time series of spatial snapshots, and fits a basis to it (PCA or DMD) to answer what
spatial patterns are present in the dynamics and how each one behaves over time -- growing,
decaying, oscillating, or static.

PCA modes are ranked by variance explained and carry no notion of time: a mode's "weight"
is just its projection at each snapshot. DMD modes are ranked by amplitude and each carries
an eigenvalue that fixes its growth/decay rate and oscillation frequency, which is what
"what's driving the dynamics" actually asks for -- see the growth-rate annotation on each
mode in the basis figure.

  python decompose_pattern_dynamics.py --sourceDat data/StigmergicModelParameters.dat \\
      --clampMode none --numSimIters 2000 --decompositionWindow 500 --decompositionMethod dmd
"""

import argparse
import ast
import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat', type=str, default='data/StigmergicModelParameters.dat',
                    help='tissue parameter file; also the source of the learned clamp when '
                         '--clampMode learned')
parser.add_argument('--latticeDim', type=int, default=None,
                    help='square lattice side length; default uses --sourceDat\'s own '
                         'latticeDims. Ignored (must be unset) under --clampMode learned, '
                         'since the trained clamp is sized to the file\'s native lattice')
parser.add_argument('--fieldScreenSize', type=float, default=None,
                    help='override the field action range; default keeps --sourceDat\'s value. '
                         'Ignored under --clampMode learned')
parser.add_argument('--clampMode', type=str, default='random', choices=['none', 'learned', 'random'],
                    help='none: free evolution. learned: the clamp trained into --sourceDat. '
                         'random: a fresh random two-fold symmetric boundary field clamp')
parser.add_argument('--clampIters', type=int, default=100,
                    help='clamp duration for --clampMode random; ignored otherwise, since '
                         'the learned clamp carries its own duration and none has no clamp')
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--numSimIters', type=int, default=2000)
parser.add_argument('--decompositionWindow', type=int, default=500,
                    help='number of final iterations to decompose')
parser.add_argument('--decompositionMethod', type=str, default='pca', choices=['pca', 'dmd'])
parser.add_argument('--numModes', type=int, default=8)
parser.add_argument('--numFramesShown', type=int, default=6,
                    help='frames within the decomposition window shown in the '
                         'composed-vs-observed comparison')
parser.add_argument('--vLimit', type=float, default=None,
                    help='half-width of the Vmem colour scale in mV about V_th; default '
                         'auto-scales to the displayed frames')
parser.add_argument('--loadTrajectory', type=str, default=None,
                    help='an .npz of pre-computed Vmem trajectories to decompose instead of '
                         'running a fresh simulation, e.g. data/unclampedEvolution.npz as '
                         'written by visualize_unclamped_evolution.py (keys like '
                         '"screen5_released", each (numSimIters, numCells) in volts). Makes '
                         '--sourceDat, --clampMode, --clampIters and --seed irrelevant to the '
                         'simulation, since none is run; --latticeDim still applies, to reshape '
                         'the loaded cells (default: inferred as a square)')
parser.add_argument('--trajectoryKey', type=str, default=None,
                    help='key within --loadTrajectory to decompose; required when '
                         '--loadTrajectory is set. Run with an invalid key to list the file\'s '
                         'keys')
parser.add_argument('--outputPrefix', type=str, default='data/patternDynamics')
args = parser.parse_args()

if args.loadTrajectory is not None and args.trajectoryKey is None:
    raise SystemExit('--trajectoryKey is required with --loadTrajectory')
if args.loadTrajectory is None and args.clampMode == 'learned' and (
        args.latticeDim is not None or args.fieldScreenSize is not None):
    raise SystemExit('--clampMode learned uses --sourceDat as-is; --latticeDim and '
                      '--fieldScreenSize must be left unset')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
utils = utilities.utilities()

V_TH_MV = -27.0   # cellularFieldNetwork.py's V_th, mV; fixed across all parameter files


# --- Build the tissue -------------------------------------------------------------------

def buildLearnedParameters():
    """Load --sourceDat unmodified, so the trained clamp and its native init are kept exact."""
    parameters = copy.deepcopy(torch.load(args.sourceDat, weights_only=False))
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    initialValues = parameters['simParameters']['initialValues']
    if 'ligandConc' not in initialValues:
        numCells = parameters['latticeDims'][0] * parameters['latticeDims'][1]
        initialValues['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
    if parameters['clampParameters'] is None:
        raise SystemExit(f'{args.sourceDat} carries no learned clamp; use --clampMode random or none')
    return parameters


def buildFreshParameters(latticeDim, numFieldGridPoints=None):
    """Scalar tissue parameters carried over onto a lattice of the requested size.

    Mirrors visualize_30x30_tail.py: everything defining the tissue (channel conductances,
    gap junction strength, field transduction) is a scalar and transfers unchanged; only
    what is sized by the lattice (init arrays, field grid, clamp) is rebuilt.
    """
    numCells = latticeDim * latticeDim
    parameters = copy.deepcopy(torch.load(args.sourceDat, weights_only=False))
    parameters['latticeDims'] = (latticeDim, latticeDim)
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    if args.fieldScreenSize is not None:
        parameters['fieldParameters']['fieldScreenSize'] = args.fieldScreenSize
    if numFieldGridPoints is not None:
        initialValues = parameters['simParameters']['initialValues']
        initialValues['Vmem'] = torch.full((1, numCells, 1), -9.2e-3, dtype=torch.float64)
        initialValues['eV'] = torch.zeros((1, numFieldGridPoints, 1), dtype=torch.float64)
        initialValues['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
        initialValues['G_pol'] = {'cells': [[list(range(numCells))]],
                                  'values': [[torch.ones(numCells, dtype=torch.float64)]]}
        initialValues['G_dep'] = {'cells': [], 'values': torch.DoubleTensor([])}
    return parameters


if args.loadTrajectory is not None:
    with np.load(args.loadTrajectory) as stored:
        if args.trajectoryKey not in stored:
            raise SystemExit(f'"{args.trajectoryKey}" not in {args.loadTrajectory}; '
                             f'available keys: {list(stored.keys())}')
        Vmem = stored[args.trajectoryKey] * 1000.0   # stored in volts; (numSimIters, numCells)
    numCells = Vmem.shape[1]
    side = int(round(numCells ** 0.5))
    numRows = numCols = args.latticeDim if args.latticeDim is not None else side
    if numRows * numCols != numCells:
        raise SystemExit(f'{args.loadTrajectory}["{args.trajectoryKey}"] has {numCells} cells, '
                         f'not a {numRows}x{numCols} square; pass --latticeDim explicitly')
    clampParameters = None
    print(f"Loaded {args.loadTrajectory}[\"{args.trajectoryKey}\"]: "
         f"{numRows}x{numCols} lattice, {Vmem.shape[0]} iterations")
else:
    if args.clampMode == 'learned':
        parameters = buildLearnedParameters()
        modelInstance = model(parameters, 1)
        modelInstance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
        clampParameters = parameters['clampParameters']
    else:
        latticeDim = args.latticeDim if args.latticeDim is not None else torch.load(
            args.sourceDat, weights_only=False)['latticeDims'][0]
        # Two passes: the first only to learn the field grid size, which sets the eV init shape.
        modelInstance = model(buildFreshParameters(latticeDim), 1)
        parameters = buildFreshParameters(latticeDim, modelInstance.electricNetwork.numFieldGridPoints)
        modelInstance = model(parameters, 1)
        modelInstance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))

        clampParameters = None
        if args.clampMode == 'random':
            circuit = modelInstance.electricNetwork
            leftHalfIndices = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
            mirroredIndices = utils.computeSymmetricalIndices(circuit, leftHalfIndices, mode='field',
                                                               symmetry='twofold')
            allIndices = np.concatenate((leftHalfIndices, mirroredIndices))
            _, uniqueIdx = np.unique(allIndices, return_index=True)
            clampPointIndices = allIndices[uniqueIdx]

            numHalf = len(leftHalfIndices)
            timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
            frequencies = torch.rand(numHalf, dtype=torch.double) * 900.0 + 100.0
            phases = torch.rand(numHalf, dtype=torch.double) * 2 * torch.pi
            amplitudes = torch.rand(numHalf, dtype=torch.double) * 2.0 - 1.0
            clampValues = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
                           * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
            clampParameters = {'clampMode': 'fieldDomeTwoFoldSymmetry',
                               'clampIndices': (np.zeros(len(clampPointIndices), dtype=int), clampPointIndices),
                               'clampValues': clampValues,
                               'clampStartIter': 0,
                               'clampEndIter': args.clampIters}

    numRows, numCols = parameters['latticeDims']
    numCells = numRows * numCols
    print(f"{numRows}x{numCols} lattice, clampMode={args.clampMode}, {args.numSimIters} iterations")
    modelInstance.simulate(externalInputs=parameters['simParameters']['externalInputs'],
                           clampParameters=clampParameters, perturbation=None,
                           numSimIters=args.numSimIters, storeVariables=('Vmem', 'Gpol'))

    Vmem = modelInstance.timeseriesVmem[:, 0, :, 0].detach().numpy() * 1000.0   # (numSimIters, numCells) mV

changeRate = np.abs(np.diff(Vmem, axis=0)).mean(axis=1)
numSimIters = Vmem.shape[0]


# --- Decompose ---------------------------------------------------------------------------

def decomposePCA(snapshots, numModes):
    from sklearn.decomposition import PCA
    mean = snapshots.mean(axis=0)
    centred = snapshots - mean
    limit = min(centred.shape[0] - 1, centred.shape[1])
    n = min(numModes, limit)
    pca = PCA(n_components=n).fit(centred)
    spatialModes = pca.components_                       # (n, numCells)
    weights = pca.transform(centred)                      # (T, n)
    reconstruction = mean + weights @ spatialModes
    info = {'label': [f'{v*100:.1f}% var' for v in pca.explained_variance_ratio_]}
    return spatialModes, weights, reconstruction, info


def decomposeDMD(snapshots, numModes):
    """Exact DMD (Tu et al. 2014). dt = 1 iteration; growth/frequency are per-iteration."""
    X = snapshots.T                                       # (numCells, T)
    X1, X2 = X[:, :-1], X[:, 1:]
    T = X.shape[1]
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    # Directions past the energy elbow are noise floor, not signal, and fitting Atilde against
    # them gives spurious eigenvalues with |mu| > 1 that a t=0..T extrapolation blows up by many
    # orders of magnitude (verified: a mode on a 0.19-singular-value direction, 4 orders below
    # the leading one, produced |mu|=1.15 and a 233 mV reconstruction RMS on a 9.6 mV signal).
    # Truncating at 99.9% cumulative energy keeps the modes the window's snapshots actually
    # support and drops the rest.
    energy = np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-300)
    energyRank = int(np.searchsorted(energy, 0.999) + 1)
    rMax = max(min(numModes, len(S), energyRank), 1)

    # Energy alone is not enough: an eigenvalue that clears the cut can still have |mu| barely
    # above 1 and blow up over a long window regardless (verified: |mu|=1.017 on a fully
    # energy-supported mode gave 1.017^889 = 3.3e6 and a 22.8 V RMS reconstruction on a 9 mV
    # signal, over an 890-step window). A lower rank is a more damped, conservative fit, so back
    # off from rMax until the extrapolated reconstruction stays within a sane physical bound.
    bound = 20 * np.abs(snapshots).max()
    for r in range(rMax, 0, -1):
        Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r].conj().T
        Atilde = Ur.conj().T @ X2 @ Vr @ np.diag(1.0 / Sr)
        eigvals, W = np.linalg.eig(Atilde)
        Phi = X2 @ Vr @ np.diag(1.0 / Sr) @ W             # DMD modes, complex (numCells, r)
        b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]
        dynamics = eigvals[:, None] ** np.arange(T)[None, :]  # (r, T)
        # Real input data gives Atilde a real characteristic polynomial, so complex eigenvalues
        # arrive in conjugate pairs with identical |b| and Re(Phi); the actual (real) contribution
        # of such a pair is 2*Re(b * mu^t), split across both entries. Summing the full complex
        # set here (before the final .real) gets that right for the reconstruction, but displaying
        # both entries separately -- as an earlier version of this function did -- draws two
        # identical panels and two overlapping lines for what is physically one oscillating mode.
        reconstruction = (Phi @ (b[:, None] * dynamics)).real.T
        if np.abs(reconstruction).max() <= bound or r == 1:
            break
    if r < numModes:
        print(f"  DMD: using {r} of the requested {numModes} modes (window energy and/or "
              f"extrapolation stability over {T} steps limits it); returning {r}")

    order = np.argsort(-np.abs(b))
    Phi, eigvals, b = Phi[:, order], eigvals[order], b[order]
    isReal = np.abs(eigvals.imag) < 1e-9 * np.maximum(np.abs(eigvals.real), 1.0)
    keep = isReal | (eigvals.imag > 0)
    multiplier = np.where(isReal[keep], 1.0, 2.0)
    Phi, eigvals, b = Phi[:, keep], eigvals[keep], b[keep]
    numKept = keep.sum()

    weightsComplex = multiplier[None, :] * (b[None, :] * eigvals[None, :] ** np.arange(T)[:, None])
    omega = np.log(eigvals)
    growthPctPerIter = (np.abs(eigvals) - 1) * 100
    freqPerIter = omega.imag / (2 * np.pi)
    labels = []
    for k in range(numKept):
        if abs(freqPerIter[k]) > 1e-6:
            labels.append(f'{growthPctPerIter[k]:+.1f}%/iter, period {1.0/abs(freqPerIter[k]):.0f} iter')
        else:
            labels.append(f'{growthPctPerIter[k]:+.1f}%/iter, non-oscillatory')
    info = {'label': labels}
    return Phi.real.T, weightsComplex.real, reconstruction, info


decomposers = {'pca': decomposePCA, 'dmd': decomposeDMD}
window = Vmem[-args.decompositionWindow:]
windowIters = np.arange(numSimIters - window.shape[0], numSimIters)
spatialModes, weights, reconstruction, info = decomposers[args.decompositionMethod](window, args.numModes)
numModes = spatialModes.shape[0]

residual = window - reconstruction
rmsError = np.sqrt((residual ** 2).mean())
varExplained = 1 - residual.var() / max(window.var(), 1e-300)
print(f"{args.decompositionMethod.upper()} with {numModes} modes over the last {window.shape[0]} iterations:")
print(f"  reconstruction RMS error: {rmsError:.3f} mV, variance explained: {varExplained*100:.1f}%")


# --- Figure 1: basis patterns + weights over time + change-rate context ------------------

vCentre = V_TH_MV
numCols_ = 4
numRowsFig = int(np.ceil(numModes / numCols_))
fig = plt.figure(figsize=(3.2 * numCols_, 2.6 * numRowsFig + 6.5))
gs = fig.add_gridspec(numRowsFig + 2, numCols_, height_ratios=[2.0] * numRowsFig + [2.2, 1.3], hspace=0.5)

for k in range(numModes):
    ax = fig.add_subplot(gs[k // numCols_, k % numCols_])
    mode = spatialModes[k].reshape(numRows, numCols)
    vabs = np.abs(mode).max()
    im = ax.imshow(mode, cmap='RdBu_r', vmin=-vabs, vmax=vabs, interpolation='nearest')
    ax.set_title(f'Mode {k+1}\n{info["label"][k]}', fontsize=8.5, color='0.25')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

axWeights = fig.add_subplot(gs[numRowsFig, :])
for k in range(numModes):
    axWeights.plot(windowIters, weights[:, k], linewidth=1.2, label=f'mode {k+1}')
axWeights.set_xlabel('iteration', fontsize=9, color='0.25')
axWeights.set_ylabel('mode weight (mV)', fontsize=9, color='0.25')
axWeights.legend(fontsize=7, ncol=min(numModes, 8), loc='upper right')
axWeights.tick_params(labelsize=8, colors='0.35')
axWeights.grid(True, alpha=0.18, linewidth=0.6)
for spine in ['top', 'right']:
    axWeights.spines[spine].set_visible(False)

axRate = fig.add_subplot(gs[numRowsFig + 1, :])
axRate.plot(np.arange(1, numSimIters), changeRate, linewidth=1.2, color='#3b6ea5')
axRate.axvspan(windowIters[0], windowIters[-1], color='#3b6ea5', alpha=0.10, linewidth=0)
if clampParameters is not None:
    axRate.axvline(clampParameters['clampEndIter'], color='0.55', linewidth=1.0, linestyle='--')
axRate.set_yscale('log')
axRate.set_xlabel('iteration', fontsize=9, color='0.25')
axRate.set_ylabel('mean |ΔVmem|/step (mV)', fontsize=9, color='0.25')
axRate.tick_params(labelsize=8, colors='0.35')
axRate.grid(True, alpha=0.18, linewidth=0.6)
for spine in ['top', 'right']:
    axRate.spines[spine].set_visible(False)

fig.suptitle(f'{args.decompositionMethod.upper()} basis, {numRows}x{numCols} lattice, '
            f'clampMode={args.clampMode}, window=last {window.shape[0]} of {numSimIters} iters',
            fontsize=11, color='0.2', y=0.995)
plt.savefig(f'{args.outputPrefix}_basis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {args.outputPrefix}_basis.png")


# --- Figure 2: composed vs observed --------------------------------------------------------

frameIdx = np.linspace(0, window.shape[0] - 1, min(args.numFramesShown, window.shape[0])).astype(int)
frameIdx = np.unique(frameIdx)
observedFrames = window[frameIdx]
composedFrames = reconstruction[frameIdx]
residualFrames = residual[frameIdx]

vLimit = args.vLimit if args.vLimit is not None else np.abs(observedFrames - vCentre).max()
vMin, vMax = vCentre - vLimit, vCentre + vLimit
residualLimit = np.abs(residualFrames).max()

numFrames = len(frameIdx)
fig2, axes = plt.subplots(3, numFrames, figsize=(2.1 * numFrames + 1.2, 6.6))
if numFrames == 1:
    axes = axes.reshape(3, 1)
rowLabels = ['observed', f'composed ({numModes} modes)', 'residual']
for col, idx in enumerate(frameIdx):
    im0 = axes[0, col].imshow(observedFrames[col].reshape(numRows, numCols), cmap='RdBu_r',
                              vmin=vMin, vmax=vMax, interpolation='nearest')
    axes[0, col].set_title(f't = {windowIters[idx]}', fontsize=9, color='0.25')
    im1 = axes[1, col].imshow(composedFrames[col].reshape(numRows, numCols), cmap='RdBu_r',
                              vmin=vMin, vmax=vMax, interpolation='nearest')
    im2 = axes[2, col].imshow(residualFrames[col].reshape(numRows, numCols), cmap='PuOr_r',
                              vmin=-residualLimit, vmax=residualLimit, interpolation='nearest')
    for row in range(3):
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

for row, label in enumerate(rowLabels):
    axes[row, 0].set_ylabel(label, fontsize=9, color='0.25')

fig2.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01, label='Vmem (mV)')
fig2.colorbar(im2, ax=axes[2, :].tolist(), fraction=0.02, pad=0.01, label='residual (mV)')
fig2.suptitle(f'Composed vs observed, {args.decompositionMethod.upper()} '
             f'(RMS error {rmsError:.3f} mV, {varExplained*100:.1f}% variance explained)',
             fontsize=11, color='0.2', y=0.99)
plt.savefig(f'{args.outputPrefix}_reconstruction.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {args.outputPrefix}_reconstruction.png")

np.savez(f'{args.outputPrefix}_decomposition.npz', spatialModes=spatialModes, weights=weights,
        reconstruction=reconstruction, observed=window, windowIters=windowIters,
        latticeDims=(numRows, numCols), decompositionMethod=args.decompositionMethod,
        rmsError=rmsError, varExplained=varExplained)
print(f"Saved {args.outputPrefix}_decomposition.npz")
