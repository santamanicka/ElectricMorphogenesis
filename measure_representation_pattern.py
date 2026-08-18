"""
Measure how one cell's state comes to influence every other cell's state, as a function of time.

Formalisation: perturb one cell's G_pol (the slow variable; Vmem perturbations are absorbed by
fast membrane relaxation and leave no lasting trace) by a small step at a reference state, and
track how the resulting deviation from the unperturbed trajectory spreads across the tissue at a
sequence of later horizons. This is the same finite-difference response-operator method as
measure_response_operator.py -- average the +/- step responses (cancels first-order asymmetry)
and check that doubling the step roughly doubles the response, since measure_amplification.py
already showed this linear regime does not hold indefinitely: separation from a single-cell
perturbation grows exponentially and then saturates against G_pol's [0, 2*G_ref] clip.

At the shortest horizon this is necessarily local (only gap-junction/field neighbours respond).
As the horizon grows the responding set spreads outward, and at long enough horizons every cell
has some response -- but past the horizon where doubling stops giving 2x, the response magnitude
reflects saturation rather than a meaningful per-unit-input sensitivity, and is reported as such.

  python measure_representation_pattern.py --condition uniform
  python measure_representation_pattern.py --condition random --clampSeed 7
  python measure_representation_pattern.py --condition learned
"""

import argparse
import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--fieldScreenSize', type=float, default=None)
parser.add_argument('--condition', type=str, default='uniform', choices=['uniform', 'random', 'learned'],
                    help='reference state the probe cells are perturbed from: uniform (t=0, flat '
                         'tissue), random (a random two-fold symmetric clamp\'s pre-pattern state '
                         'at t=clampIters+1), learned (the trained clamp\'s pre-pattern state)')
parser.add_argument('--clampIters', type=int, default=100)
parser.add_argument('--clampSeed', type=int, default=7, help='for --condition random')
parser.add_argument('--stepFraction', type=float, default=0.05, help='G_pol step, as a fraction of G_ref')
parser.add_argument('--horizons', type=str, default='1,10,50,150,300')
parser.add_argument('--outputPrefix', type=str, default='data/representationPattern')
args = parser.parse_args()

horizons = [int(h) for h in args.horizons.split(',')]
maxHorizon = max(horizons)
utils = utilities.utilities()


def buildParameters():
    parameters = copy.deepcopy(torch.load(args.sourceDat, weights_only=False))
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    if args.fieldScreenSize is not None:
        parameters['fieldParameters']['fieldScreenSize'] = args.fieldScreenSize
    numCells = parameters['latticeDims'][0] * parameters['latticeDims'][1]
    values = parameters['simParameters']['initialValues']
    if 'ligandConc' not in values:
        values['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
    return parameters


numRows, numCols = torch.load(args.sourceDat, weights_only=False)['latticeDims']
reference = model(buildParameters(), 1)
circuit = reference.electricNetwork
numCells, G_ref = circuit.numCells, circuit.G_ref
boundaryIndices = np.array(utils.computeDomeIndices(circuit, mode='tissue'))
del reference

# Representative probe cells spanning the lattice's distinct symmetry orbits under D4: two mirror
# corners (to directly check equivariance), an edge-midpoint boundary cell, a near-boundary
# interior cell, and the centre.
probeCells = {
    'corner_TL': 0,
    'corner_TR': numCols - 1,
    'edgeMid_top': numCols // 2,
    'nearBoundary': numCols + numCols // 2,
    'centre': (numCols // 2) * numCols + numCols // 2,
}
print(f"{numRows}x{numCols} lattice, condition={args.condition}, probe cells: {probeCells}")


def clampedToPrePattern(seed):
    leftHalf = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
    mirrored = utils.computeSymmetricalIndices(circuit, leftHalf, mode='field', symmetry='twofold')
    allIdx = np.concatenate((leftHalf, mirrored))
    _, uniqueIdx = np.unique(allIdx, return_index=True)
    clampIndices = (np.zeros(len(allIdx[uniqueIdx]), dtype=int), allIdx[uniqueIdx])
    numHalf = len(leftHalf)
    timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
    generator = torch.Generator().manual_seed(seed)
    frequencies = torch.rand(numHalf, generator=generator, dtype=torch.double) * 900 + 100
    phases = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 * torch.pi
    amplitudes = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 - 1
    clampValues = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
                   * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
    instance = model(buildParameters(), 1)
    instance.simulate(numSimIters=args.clampIters + 2, fieldModulation=True, perturbation=None,
                      clampParameters={'clampMode': 'fieldDomeTwoFoldSymmetry',
                                       'clampIndices': clampIndices, 'clampValues': clampValues,
                                       'clampStartIter': 0, 'clampEndIter': args.clampIters},
                      storeVariables=('Vmem', 'Gpol'))
    preIdx = args.clampIters + 1
    vmemPre = instance.timeseriesVmem[preIdx].clone()
    gpolPre = instance.timeseriesGpol[preIdx].clone()
    del instance
    return vmemPre, gpolPre


def uniformReference():
    """The flat, fully symmetric tissue: every cell at the same Vmem and G_ref conductance."""
    vmem = torch.full((1, numCells, 1), -9.2e-3, dtype=torch.float64)
    gpol = torch.full((1, numCells, 1), G_ref, dtype=torch.float64)
    return vmem, gpol


def learnedToPrePattern():
    parameters = torch.load(args.sourceDat, weights_only=False)
    if parameters['clampParameters'] is None:
        raise SystemExit(f'{args.sourceDat} carries no learned clamp; use --condition random or uniform')
    parameters = copy.deepcopy(parameters)
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    instance = model(parameters, 1)
    instance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    instance.simulate(externalInputs=parameters['simParameters']['externalInputs'],
                      clampParameters=parameters['clampParameters'], perturbation=None,
                      numSimIters=args.clampIters + 2, storeVariables=('Vmem', 'Gpol'))
    preIdx = args.clampIters + 1
    vmemPre = instance.timeseriesVmem[preIdx].clone()
    gpolPre = instance.timeseriesGpol[preIdx].clone()
    del instance
    return vmemPre, gpolPre


if args.condition == 'uniform':
    vmemRef, gpolRef = uniformReference()
elif args.condition == 'random':
    vmemRef, gpolRef = clampedToPrePattern(args.clampSeed)
else:
    vmemRef, gpolRef = learnedToPrePattern()


def freeEvolveFrom(vmemStart, gpolStart, numIters):
    parameters = buildParameters()
    instance = model(parameters, 1)
    instance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    net = instance.electricNetwork
    initialValues = dict(parameters['simParameters']['initialValues'])
    initialValues['Vmem'] = vmemStart.clone().double()
    initialValues['eV'] = torch.zeros_like(net.eV)
    net.initVariables(initialValues)
    net.G_pol = gpolStart.clone().double()
    instance.simulate(externalInputs=parameters['simParameters']['externalInputs'], clampParameters=None,
                      perturbation=None, numSimIters=numIters, storeVariables=('Vmem',))
    trace = instance.timeseriesVmem[:, 0, :, 0].detach().numpy() * 1000.0   # (numIters, numCells) mV
    del instance
    return trace


print("Running base (unperturbed) trajectory...")
baseTrace = freeEvolveFrom(vmemRef, gpolRef, maxHorizon + 1)
step = args.stepFraction * G_ref


def perturbedTrace(cellIdx, magnitude):
    perturbed = gpolRef.clone()
    perturbed[0, cellIdx, 0] += magnitude
    perturbed = torch.clip(perturbed, 0.0, 2.0 * G_ref)
    delivered = (perturbed - gpolRef)[0, cellIdx, 0].item()
    return delivered, freeEvolveFrom(vmemRef, perturbed, maxHorizon + 1)


results = {}
for name, cellIdx in probeCells.items():
    plusDelivered, plusTrace = perturbedTrace(cellIdx, +step)
    minusDelivered, minusTrace = perturbedTrace(cellIdx, -step)
    scale = (plusDelivered - minusDelivered) / G_ref   # response per G_ref of conductance change,
                                                        # not per physical unit (G_ref itself is 1e-9)
    responses = {}
    asymmetries = {}
    for T in horizons:
        plusResp = plusTrace[T] - baseTrace[T]
        minusResp = minusTrace[T] - baseTrace[T]
        denom = np.linalg.norm(plusResp) + np.linalg.norm(minusResp)
        asymmetries[T] = float(np.linalg.norm(plusResp + minusResp) / denom) if denom > 0 else np.nan
        responses[T] = (plusResp - minusResp) / scale if scale != 0 else plusResp * 0
    # Doubling check at the longest horizon: does 2x step give ~2x response?
    doubleDelivered, doubleTrace = perturbedTrace(cellIdx, +2 * step)
    doubleResp = doubleTrace[maxHorizon] - baseTrace[maxHorizon]
    expected = 2 * np.linalg.norm(plusTrace[maxHorizon] - baseTrace[maxHorizon])
    linearity = float(np.linalg.norm(doubleResp) / expected) if expected > 0 else np.nan
    results[name] = dict(cellIdx=cellIdx, responses=responses, asymmetries=asymmetries, linearity=linearity)
    reachStr = "  ".join(f"T={T}: |R|={np.linalg.norm(responses[T]):7.3f}mV asym={asymmetries[T]:.2f}"
                         for T in horizons)
    print(f"{name:<14} (cell {cellIdx:>3})  {reachStr}  doubling={linearity:.2f}x of 2x")


# --- Figure: response field per probe cell, across horizons -------------------------------------

fig, axes = plt.subplots(len(probeCells), len(horizons), figsize=(2.2 * len(horizons), 2.2 * len(probeCells)))
for row, (name, res) in enumerate(results.items()):
    vabs = max(np.abs(res['responses'][T]).max() for T in horizons) or 1e-9
    for col, T in enumerate(horizons):
        ax = axes[row, col] if len(probeCells) > 1 else axes[col]
        grid = res['responses'][T].reshape(numRows, numCols)
        im = ax.imshow(grid, cmap='RdBu_r', vmin=-vabs, vmax=vabs, interpolation='nearest')
        cellIdx = res['cellIdx']
        ax.plot(cellIdx % numCols, cellIdx // numCols, 'k+', markersize=8, markeredgewidth=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f'T={T}', fontsize=9)
        if col == 0:
            ax.set_ylabel(name, fontsize=8)
fig.suptitle(f'Representation pattern per probe cell (+), condition={args.condition}, mV per G_ref of step',
            fontsize=11, y=1.0)
plt.tight_layout()
plt.savefig(f'{args.outputPrefix}_{args.condition}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {args.outputPrefix}_{args.condition}.png")

np.savez(f'{args.outputPrefix}_{args.condition}.npz',
        **{f'{name}_T{T}': res['responses'][T] for name, res in results.items() for T in horizons},
        **{f'{name}_asym_T{T}': res['asymmetries'][T] for name, res in results.items() for T in horizons},
        **{f'{name}_linearity': res['linearity'] for name, res in results.items()},
        probeCells=probeCells, horizons=horizons, latticeDims=(numRows, numCols))
print(f"Saved {args.outputPrefix}_{args.condition}.npz")
