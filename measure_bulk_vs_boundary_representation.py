"""
For every cell's state at the readout, how much comes from bulk (other interior) cells versus
from the boundary?

Builds the full response matrix M[source, target] -- perturb every one of the 121 cells' G_pol at
the pre-pattern step (t = clampIters+1) and record every cell's Vmem deviation at the readout,
using the same +/- symmetrized, linearity-checked protocol as measure_response_operator.py. For
each target cell this gives a column of who moved it and by how much; splitting that column by
whether the source sits on the boundary or in the bulk gives a per-cell verdict; the full matrix
also answers a finer question later -- which specific cell represents a given target most --
without rerunning anything.

  python measure_bulk_vs_boundary_representation.py --condition learned
"""

import argparse
import copy

import numpy as np
import torch

import utilities
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--fieldScreenSize', type=float, default=None)
parser.add_argument('--condition', type=str, default='learned', choices=['learned', 'random'])
parser.add_argument('--clampSeed', type=int, default=7, help='for --condition random')
parser.add_argument('--clampIters', type=int, default=100)
parser.add_argument('--readoutIter', type=int, default=1000, help='iteration (from t=0) to read out at')
parser.add_argument('--readoutIters', type=int, default=200, help='average the last N steps into the readout')
parser.add_argument('--stepFraction', type=float, default=0.05)
parser.add_argument('--outputPrefix', type=str, default='data/bulkVsBoundary')
args = parser.parse_args()

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
boundaryMask = np.zeros(numCells, bool); boundaryMask[boundaryIndices] = True
interiorIndices = np.arange(numCells)[~boundaryMask]
del reference
print(f"{numRows}x{numCols} lattice, {len(boundaryIndices)} boundary / {len(interiorIndices)} interior cells")


def clampParametersFor():
    if args.condition == 'learned':
        parameters = torch.load(args.sourceDat, weights_only=False)
        if parameters['clampParameters'] is None:
            raise SystemExit(f'{args.sourceDat} carries no learned clamp')
        return parameters['clampParameters']
    leftHalf = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
    mirrored = utils.computeSymmetricalIndices(circuit, leftHalf, mode='field', symmetry='twofold')
    allIdx = np.concatenate((leftHalf, mirrored))
    _, uniqueIdx = np.unique(allIdx, return_index=True)
    clampIndices = (np.zeros(len(allIdx[uniqueIdx]), dtype=int), allIdx[uniqueIdx])
    numHalf = len(leftHalf)
    timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
    generator = torch.Generator().manual_seed(args.clampSeed)
    frequencies = torch.rand(numHalf, generator=generator, dtype=torch.double) * 900 + 100
    phases = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 * torch.pi
    amplitudes = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 - 1
    clampValues = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
                   * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
    return {'clampMode': 'fieldDomeTwoFoldSymmetry', 'clampIndices': clampIndices,
           'clampValues': clampValues, 'clampStartIter': 0, 'clampEndIter': args.clampIters}


def clampedToPrePattern():
    parameters = buildParameters()
    instance = model(parameters, 1)
    instance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    instance.simulate(externalInputs=parameters['simParameters']['externalInputs'],
                      clampParameters=clampParametersFor(), perturbation=None,
                      numSimIters=args.clampIters + 2, storeVariables=('Vmem', 'Gpol'))
    preIdx = args.clampIters + 1
    vmemPre = instance.timeseriesVmem[preIdx].clone()
    gpolPre = instance.timeseriesGpol[preIdx].clone()
    del instance
    return vmemPre, gpolPre


vmemPre, gpolPre = clampedToPrePattern()
step = args.stepFraction * G_ref
freeSteps = args.readoutIter - (args.clampIters + 1)


def readoutFrom(vmemStart, gpolStart):
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
                      perturbation=None, numSimIters=freeSteps, storeVariables=('Vmem',))
    preceding = instance.timeseriesVmem[-(args.readoutIters - 1):, 0, :, 0].detach().numpy()
    final = net.Vmem[0, :, 0].detach().numpy()
    del instance
    return (preceding.sum(axis=0) + final) / args.readoutIters * 1000.0   # mV, (numCells,)


baseReadout = readoutFrom(vmemPre, gpolPre)


def perturbedReadout(cellIdx, magnitude):
    perturbed = gpolPre.clone()
    perturbed[0, cellIdx, 0] += magnitude
    perturbed = torch.clip(perturbed, 0.0, 2.0 * G_ref)
    delivered = (perturbed - gpolPre)[0, cellIdx, 0].item()
    return delivered, readoutFrom(vmemPre, perturbed)


M = np.zeros((numCells, numCells))          # M[source, target], mV per G_ref
asymmetry = np.zeros(numCells)
for source in range(numCells):
    plusDelivered, plusReadout = perturbedReadout(source, +step)
    minusDelivered, minusReadout = perturbedReadout(source, -step)
    plusResp = plusReadout - baseReadout
    minusResp = minusReadout - baseReadout
    denom = np.linalg.norm(plusResp) + np.linalg.norm(minusResp)
    asymmetry[source] = np.linalg.norm(plusResp + minusResp) / denom if denom > 0 else np.nan
    scale = (plusDelivered - minusDelivered) / G_ref
    M[source] = (plusResp - minusResp) / scale if scale != 0 else plusResp * 0
    if source % 20 == 0:
        print(f"  source {source:>3}/{numCells}: asym={asymmetry[source]:.2f}")

np.savez(f'{args.outputPrefix}_{args.condition}.npz', M=M, asymmetry=asymmetry,
        boundaryIndices=boundaryIndices, interiorIndices=interiorIndices,
        baseReadout=baseReadout, latticeDims=(numRows, numCols), readoutIter=args.readoutIter)
print(f"Saved {args.outputPrefix}_{args.condition}.npz")
print(f"mean asymmetry: {np.nanmean(asymmetry):.3f}, "
     f"reliable (asym<0.3): {(asymmetry < 0.3).sum()}/{numCells}")
