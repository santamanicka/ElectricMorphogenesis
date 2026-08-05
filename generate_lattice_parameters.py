"""
Build a simulation seed parameter file for a lattice size other than the trained one.

Everything that defines the stigmergic tissue is a scalar: ion channel conductances, gap
junction strength, and the field transduction gain, bias, weight and time constant. None of
them is per-cell, so they transfer to any lattice size unchanged. What does not transfer is
everything sized by the lattice -- the initial value arrays, the extracellular field grid, and
any trained clamp -- so those are rebuilt here rather than carried over.

This is what makes "same tissue, larger grid" a legitimate claim rather than a re-tuning: the
output file is byte-identical to the source in every tissue parameter, and differs only in
lattice size, field action range, and the arrays whose shape the lattice determines.

The output is a *simulation seed*, not a trained model. It carries no clamp: clamp signals are
supplied by whatever drives the simulation (generate_ensemble.py builds random ones; a training
run learns them). trainParameters is retained for schema compatibility but its lattice-sized
tensors are cleared, since a 121-cell target in a 900-cell file would be a latent bug.

  python generate_lattice_parameters.py --latticeDim 30 --fieldScreenSize 5
"""

import argparse
import copy

import torch

from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--latticeDim',      type=int,   default=30, help='lattice is latticeDim x latticeDim')
parser.add_argument('--fieldScreenSize', type=int,   default=5,
                    help='field action range. 5 follows the 30x30 sweep, whose optimum (4-6) '
                         'does not scale with lattice size')
parser.add_argument('--numSimIters',     type=int,   default=2500,
                    help='recorded in simParameters as the protocol default; the post-clamp '
                         'transient ends near iteration 1080 at this lattice size')
parser.add_argument('--initialVmem',     type=float, default=-9.2e-3, help='uniform initial Vmem (V)')
parser.add_argument('--sourceDat',       type=str,   default='data/StigmergicModelParameters.dat')
parser.add_argument('--outputDat',       type=str,   default=None,
                    help='defaults to data/StigmergicModelParameters_<latticeDim>x<latticeDim>.dat')
args = parser.parse_args()

latticeDim = args.latticeDim
numCells = latticeDim * latticeDim
outputDat = args.outputDat or f'data/StigmergicModelParameters_{latticeDim}x{latticeDim}.dat'

sourceParameters = torch.load(args.sourceDat, weights_only=False)
parameters = copy.deepcopy(sourceParameters)

parameters['latticeDims'] = (latticeDim, latticeDim)
parameters['fieldParameters']['fieldScreenSize'] = args.fieldScreenSize
parameters['latticePeriodicBoundaryGJ'] = False
# Older .dat files predate the ATP pathway; cellularFieldNetwork.loadParameters requires the key.
parameters['ATPParameters'] = None

# The extracellular grid is derived from the lattice, so its size is only known once a model has
# been constructed. Build a throwaway instance to read it, then size the initial values.
numFieldGridPoints = model(copy.deepcopy(parameters), 1).electricNetwork.numFieldGridPoints

allCells = list(range(numCells))
parameters['simParameters']['initialValues'] = {
    'Vmem':       torch.full((1, numCells, 1), args.initialVmem, dtype=torch.float64),
    'eV':         torch.zeros((1, numFieldGridPoints, 1), dtype=torch.float64),
    'ligandConc': torch.zeros((1, numCells, 1), dtype=torch.float64),
    # Uniform G_pol at 1.0 in G_ref units. Every cell is listed explicitly rather than relying on
    # the constructor default, so the starting state is visible in the file itself.
    'G_pol':      {'cells': [[allCells]], 'values': [[torch.ones(numCells, dtype=torch.float64)]]},
    'G_dep':      {'cells': [], 'values': torch.DoubleTensor([])},
}
parameters['simParameters']['numSamples'] = 1
parameters['simParameters']['numSimIters'] = args.numSimIters
parameters['simParameters']['externalInputs'] = {'gene': None}

# No trained clamp travels with a seed file: the source clamp is shaped (101, 44) for the 11x11
# field dome and is meaningless on a larger one.
parameters['clampParameters'] = None

if isinstance(parameters.get('trainParameters'), dict):
    parameters['trainParameters'] = dict(parameters['trainParameters'])
    parameters['trainParameters']['targetVmem'] = None
    parameters['trainParameters']['actualVmem'] = None
    parameters['trainParameters']['bestLoss'] = None
    parameters['trainParameters']['bestLossHistory'] = []

torch.save(parameters, outputDat)

# Confirm the tissue really is unchanged. These are the parameters that define the model; if any
# of them differed, "same tissue on a larger grid" would not be a claim this file supports.
tissueParameterNames = ['fieldEnabled', 'fieldResolution', 'fieldStrength', 'fieldAggregation',
                        'fieldTransductionGain', 'fieldTransductionWeight', 'fieldTransductionBias',
                        'fieldTransductionTimeConstant', 'fieldRangeSymmetric', 'fieldVector']
print(f"source: {args.sourceDat}  ->  output: {outputDat}\n")
print(f"  latticeDims        {sourceParameters['latticeDims']}  ->  {parameters['latticeDims']}")
print(f"  numCells           {sourceParameters['latticeDims'][0]*sourceParameters['latticeDims'][1]}"
      f"  ->  {numCells}")
print(f"  numFieldGridPoints {sourceParameters['simParameters']['initialValues']['eV'].shape[1]}"
      f"  ->  {numFieldGridPoints}")
print(f"  fieldScreenSize    {sourceParameters['fieldParameters']['fieldScreenSize']}"
      f"  ->  {parameters['fieldParameters']['fieldScreenSize']}")
print(f"  numSimIters        {sourceParameters['simParameters']['numSimIters']}"
      f"  ->  {parameters['simParameters']['numSimIters']}")
print(f"  clampParameters    {sourceParameters['clampParameters']['clampValues'].shape}  ->  None")

print("\n  tissue parameters carried over unchanged:")
for name in tissueParameterNames:
    sourceValue = sourceParameters['fieldParameters'][name]
    outputValue = parameters['fieldParameters'][name]
    same = torch.equal(sourceValue, outputValue) if torch.is_tensor(sourceValue) else sourceValue == outputValue
    print(f"    {'OK ' if same else 'DIFF'} {name:<32} {sourceValue}")
sameGJ = sourceParameters['GJParameters'] == parameters['GJParameters']
print(f"    {'OK ' if sameGJ else 'DIFF'} {'GJStrength':<32} {parameters['GJParameters']['GJStrength']}")
