"""
How sensitive is the free evolution to the pre-pattern G_pol, and where does that sensitivity live?

Section 8.3 established that G_pol at t = clampEndIter + 1 is the entire state the free evolution
reads. This script perturbs exactly that state and measures how far the resulting pattern moves,
which is the test of whether the tissue holds a basin around a written code or amplifies any
deviation from it.

Perturbing the initial Vmem instead does NOT answer this: the clamp overwrites it, leaving a
difference at the pre-pattern step some three thousand times smaller than the difference between
two clamps. That measures how robustly the clamp writes, not how sensitively the tissue reads.

Three placements, selected with --placement:
  full      random direction over all cells -- dominated by the interior simply by cell count
  boundary  the 116 boundary cells only, where the clamp writes
  interior  an equally sized random subset of interior cells, matched per-cell amplitude, which
            removes the cell-count confound in the boundary/interior comparison

Amplitudes are quoted as fractions of the per-cell RMS at which two different clamps differ on
the boundary, so 1.0 is "as wrong as using a different clamp entirely".

  python sweep_prepattern_sensitivity.py --placement boundary interior
"""

import argparse
import copy

import numpy as np
import torch

import utilities
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat',    type=str,   default='data/StigmergicModelParameters_30x30.dat')
parser.add_argument('--placement',    type=str,   nargs='+', default=['full', 'boundary', 'interior'],
                    choices=['full', 'boundary', 'interior'])
parser.add_argument('--fractions',    type=str,   default='[0.001,0.01,0.1,0.3,1.0]',
                    help='perturbation sizes as fractions of the between-clamp boundary per-cell RMS')
parser.add_argument('--numInteriorSubsets', type=int, default=2,
                    help='independent random interior subsets, to gauge subset-to-subset scatter')
parser.add_argument('--clampSeed',    type=int,   default=7)
parser.add_argument('--otherClampSeed', type=int, default=99, help='sets the between-clamp reference scale')
parser.add_argument('--clampIters',   type=int,   default=100)
parser.add_argument('--numSimIters',  type=int,   default=2500)
parser.add_argument('--readoutIters', type=int,   default=200)
parser.add_argument('--seed',         type=int,   default=11)
parser.add_argument('--outputPrefix', type=str,   default='data/prepatternSensitivity')
args = parser.parse_args()

fractions = eval(args.fractions)
prePatternIter = args.clampIters + 1
utils = utilities.utilities()


def buildParameters():
    parameters = copy.deepcopy(torch.load(args.sourceDat, weights_only=False))
    # The original trained .dat files predate the ATP pathway and the periodic-boundary flag.
    if 'ATPParameters' not in parameters:
        parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    return parameters


def buildClamp(circuit, clampSeed):
    torch.manual_seed(clampSeed)
    leftHalf = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
    mirrored = utils.computeSymmetricalIndices(circuit, leftHalf, mode='field', symmetry='twofold')
    allIndices = np.concatenate((leftHalf, mirrored))
    _, uniqueIdx = np.unique(allIndices, return_index=True)
    points = allIndices[uniqueIdx]
    timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
    frequencies = torch.rand(len(leftHalf), dtype=torch.double) * 900.0 + 100.0
    phases = torch.rand(len(leftHalf), dtype=torch.double) * 2 * torch.pi
    amplitudes = torch.rand(len(leftHalf), dtype=torch.double) * 2.0 - 1.0
    values = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
              * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
    return {'clampMode': 'fieldDomeTwoFoldSymmetry',
            'clampIndices': (np.zeros(len(points), dtype=int), points),
            'clampValues': values, 'clampStartIter': 0, 'clampEndIter': args.clampIters}


def clampedToPrePattern(clampSeed):
    """Run through the clamp; return the state the free evolution would inherit."""
    parameters = buildParameters()
    modelInstance = model(parameters, 1)
    modelInstance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    modelInstance.simulate(externalInputs=parameters['simParameters']['externalInputs'],
                           clampParameters=buildClamp(modelInstance.electricNetwork, clampSeed),
                           perturbation=None, numSimIters=prePatternIter + 1,
                           storeVariables=('Vmem', 'Gpol'))
    return (modelInstance.timeseriesVmem[prePatternIter].clone(),
            modelInstance.timeseriesGpol[prePatternIter].clone(),
            modelInstance.electricNetwork)


def freeEvolveFrom(vmemPre, gpolPre):
    """Free evolution from a pre-pattern state; returns the time-averaged Vmem readout in mV.

    Matches condition C of Section 8.3 (eV zeroed), which reproduces the full simulation exactly.
    """
    parameters = buildParameters()
    modelInstance = model(parameters, 1)
    modelInstance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    circuit = modelInstance.electricNetwork
    initialValues = dict(parameters['simParameters']['initialValues'])
    initialValues['Vmem'] = vmemPre.clone().double()
    initialValues['eV'] = torch.zeros_like(circuit.eV)
    circuit.initVariables(initialValues)
    circuit.G_pol = gpolPre.clone().double()
    modelInstance.simulate(externalInputs=parameters['simParameters']['externalInputs'],
                           clampParameters=None, perturbation=None,
                           numSimIters=args.numSimIters - prePatternIter,
                           storeVariables=('Vmem', 'Gpol'))
    preceding = modelInstance.timeseriesVmem[-(args.readoutIters - 1):, 0, :, 0].detach().numpy()
    final = circuit.Vmem[0, :, 0].detach().numpy()
    return (preceding.sum(axis=0) + final) / args.readoutIters * 1000.0


vmemReference, gpolReference, circuit = clampedToPrePattern(args.clampSeed)
vmemOther, gpolOther, _ = clampedToPrePattern(args.otherClampSeed)
G_ref = circuit.G_ref
numCells = circuit.numCells

boundaryIndices = np.array(utils.computeDomeIndices(circuit, mode='tissue'))
interiorIndices = np.setdiff1d(np.arange(numCells), boundaryIndices)
numPerturbed = len(boundaryIndices)

difference = ((gpolOther - gpolReference) / G_ref)[0, :, 0].numpy()
boundaryPerCellRMS = float(np.sqrt((difference[boundaryIndices] ** 2).mean()))
interiorPerCellRMS = float(np.sqrt((difference[interiorIndices] ** 2).mean()))
boundaryShare = (difference[boundaryIndices] ** 2).sum() / (difference ** 2).sum()

print(f"between-clamp G_pol difference (seeds {args.clampSeed} vs {args.otherClampSeed}):")
print(f"  boundary {len(boundaryIndices):>4} cells ({100*len(boundaryIndices)/numCells:>2.0f}% of tissue) "
      f"| per-cell RMS {boundaryPerCellRMS:.4f} | {100*boundaryShare:.1f}% of the squared difference")
print(f"  interior {len(interiorIndices):>4} cells ({100*len(interiorIndices)/numCells:>2.0f}% of tissue) "
      f"| per-cell RMS {interiorPerCellRMS:.4f} | {100*(1-boundaryShare):.1f}%")
print(f"\nperturbation amplitudes are fractions of the boundary per-cell RMS ({boundaryPerCellRMS:.4f})")
print(f"'full' perturbs all {numCells} cells; 'boundary' and 'interior' perturb {numPerturbed} each\n")

referencePattern = freeEvolveFrom(vmemReference, gpolReference)
otherPattern = freeEvolveFrom(vmemOther, gpolOther)
betweenClampCorrelation = np.corrcoef(referencePattern, otherPattern)[0, 1]

rng = np.random.default_rng(args.seed)
placements = []
for name in args.placement:
    if name == 'full':
        placements.append(('full', np.arange(numCells)))
    elif name == 'boundary':
        placements.append(('boundary', boundaryIndices))
    else:
        for subset in range(args.numInteriorSubsets):
            placements.append((f'interior #{subset+1}',
                               rng.choice(interiorIndices, numPerturbed, replace=False)))

records = []
generator = torch.Generator().manual_seed(args.seed)
print(f"{'placement':<14} {'fraction':>9} {'per-cell RMS achieved':>22} {'RMS dVmem':>11} {'corr':>7}")
print('-' * 68)
for placementName, indices in placements:
    for fraction in fractions:
        perturbed = gpolReference.clone()
        noise = torch.randn((len(indices),), generator=generator, dtype=torch.float64)
        noise = noise / noise.pow(2).mean().sqrt()               # unit per-cell RMS
        perturbed[0, indices, 0] += noise * fraction * boundaryPerCellRMS * G_ref
        # The conductance ODE clips G_pol to this range, so a perturbation that would leave it is
        # truncated. The achieved amplitude is reported rather than the nominal one.
        perturbed = torch.clip(perturbed, 0.0, 2.0 * G_ref)
        achieved = ((perturbed - gpolReference)[0, indices, 0] / G_ref).pow(2).mean().sqrt().item()
        pattern = freeEvolveFrom(vmemReference, perturbed)
        rmsDifference = float(np.sqrt(((referencePattern - pattern) ** 2).mean()))
        correlation = float(np.corrcoef(referencePattern, pattern)[0, 1])
        records.append({'placement': placementName, 'fraction': fraction,
                        'achievedPerCellRMS': achieved, 'rmsDifference': rmsDifference,
                        'correlation': correlation})
        print(f"{placementName:<14} {fraction:>9g} {achieved:>22.4f} "
              f"{rmsDifference:>11.2f} {correlation:>7.3f}")

print(f"\nfor reference, a completely different clamp gives correlation {betweenClampCorrelation:.3f}")
torch.save({'records': records, 'boundaryPerCellRMS': boundaryPerCellRMS,
            'interiorPerCellRMS': interiorPerCellRMS, 'boundaryShare': float(boundaryShare),
            'betweenClampCorrelation': float(betweenClampCorrelation), 'args': vars(args)},
           f'{args.outputPrefix}.dat')
print(f"Saved {args.outputPrefix}.dat")
