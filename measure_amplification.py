"""
How fast does the tissue amplify a boundary perturbation, and does that explain CT?

The response-operator measurement found no linear regime at the readout horizon: doubling a step
did not double the response, and opposite steps sometimes produced identical ones. A dose-response
sweep across horizons showed why. The gain from a fixed boundary step grows roughly a hundredfold
between 200 and 900 free steps while the amplitude over which linearity holds shrinks by the same
order, which is exponential amplification of the perturbation rather than a failure of the
measurement.

That reframes what CT measures. The boundary code still determines the final pattern -- the
dynamics are deterministic and Section 8.3 showed the pre-pattern G_pol is sufficient -- but two
codes that differ slightly diverge at an exponential rate, so by the readout their patterns are
related only through whatever component of the difference has not yet been amplified into
saturation. CT is the size of that surviving component. A tissue can therefore be fully determined
by its boundary and still be uncontrollable in any useful sense.

This measures the amplification rate directly, by evolving a base state and a minimally perturbed
one side by side and tracking their separation. If the rate rises with field action range, it
explains the collapse in CT mechanistically rather than by correlation: the reaches at which the
boundary loses control are the reaches at which the tissue most violently amplifies the difference
between one boundary code and another.
"""

import argparse, ast, gc
import numpy as np
import torch
import utilities
from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat',   type=str, default='data/StigmergicModelParameters_30x30.dat')
parser.add_argument('--screenSizes', type=str, default='[2,3,4,6,11,15]')
parser.add_argument('--numSimIters', type=int, default=2500)
parser.add_argument('--clampIters',  type=int, default=100)
parser.add_argument('--clampSeed',   type=int, default=7)
parser.add_argument('--stepFraction',type=float, default=1e-5,
                    help='per-cell G_pol step; must be small enough to stay linear throughout')
parser.add_argument('--outputPrefix',type=str, default='data/amplification')
args = parser.parse_args()
screenSizes = ast.literal_eval(args.screenSizes)
torch.set_grad_enabled(False)
prePatternIter = args.clampIters + 1


def buildParameters(screenSize):
    parameters = torch.load(args.sourceDat, weights_only=False)
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['fieldParameters']['fieldScreenSize'] = screenSize
    cells = parameters['latticeDims'][0] * parameters['latticeDims'][1]
    values = parameters['simParameters']['initialValues']
    if 'ligandConc' not in values:
        values['ligandConc'] = torch.zeros((1, cells, 1), dtype=torch.float64)
    return parameters


utils = utilities.utilities()
reference = model(buildParameters(screenSizes[0]), 1)
circuit = reference.electricNetwork
numCells, G_ref = circuit.numCells, circuit.G_ref
boundaryIndices = np.array(utils.computeDomeIndices(circuit, mode='tissue'))
boundaryMask = np.zeros(numCells, bool); boundaryMask[boundaryIndices] = True
interiorMask = ~boundaryMask
leftHalf = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
mirrored = utils.computeSymmetricalIndices(circuit, leftHalf, mode='field', symmetry='twofold')
allIndices = np.concatenate((leftHalf, mirrored))
_, uniqueIdx = np.unique(allIndices, return_index=True)
clampIndices = (np.zeros(len(allIndices[uniqueIdx]), dtype=int), allIndices[uniqueIdx])
numHalf = len(leftHalf)
timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
del reference; gc.collect()


def trajectory(screenSize, gpolOverride=None):
    """Clamp, then free-evolve, returning the interior Vmem trace in mV."""
    parameters = buildParameters(screenSize)
    generator = torch.Generator().manual_seed(args.clampSeed)
    frequencies = torch.rand(numHalf, generator=generator, dtype=torch.double) * 900 + 100
    phases = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 * torch.pi
    amplitudes = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 - 1
    clampValues = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
                   * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
    instance = model(parameters, 1)
    instance.simulate(numSimIters=prePatternIter + 1, fieldModulation=True, perturbation=None,
                      clampParameters={'clampMode': 'fieldDomeTwoFoldSymmetry',
                                       'clampIndices': clampIndices, 'clampValues': clampValues,
                                       'clampStartIter': 0, 'clampEndIter': args.clampIters},
                      storeVariables=('Vmem', 'Gpol'))
    vmemPre = instance.timeseriesVmem[prePatternIter].clone()
    gpolPre = instance.timeseriesGpol[prePatternIter].clone()
    del instance; gc.collect()
    if gpolOverride is not None:
        gpolPre = gpolOverride(gpolPre)
    parameters = buildParameters(screenSize)
    instance = model(parameters, 1)
    instance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    net = instance.electricNetwork
    initialValues = dict(parameters['simParameters']['initialValues'])
    initialValues['Vmem'] = vmemPre.clone().double()
    initialValues['eV'] = torch.zeros_like(net.eV)
    net.initVariables(initialValues)
    net.G_pol = gpolPre.clone().double()
    instance.simulate(externalInputs=parameters['simParameters']['externalInputs'],
                      clampParameters=None, perturbation=None,
                      numSimIters=args.numSimIters - prePatternIter, storeVariables=('Vmem',))
    trace = np.stack([v[0, :, 0].detach().numpy() for v in instance.timeseriesVmem]) * 1000.0
    del instance; gc.collect()
    return trace[:, interiorMask]


def perturbOneCell(gpol):
    perturbed = gpol.clone()
    perturbed[0, boundaryIndices[1], 0] += args.stepFraction * G_ref
    return torch.clip(perturbed, 0.0, 2.0 * G_ref)


records = {}
print(f"{numCells} cells | step {args.stepFraction:g} G_ref on one boundary cell | "
      f"{args.numSimIters - prePatternIter} free steps")
print(f"\n{'screen':>7}{'separation at':>16}{'':>10}{'':>10}{'':>10}{'growth rate':>14}{'doublings':>11}")
print(f"{'':>7}{'100':>10}{'400':>10}{'900':>10}{'end':>10}{'per 100 steps':>14}{'total':>11}")
for screenSize in screenSizes:
    base = trajectory(screenSize)
    perturbed = trajectory(screenSize, perturbOneCell)
    separation = np.linalg.norm(perturbed - base, axis=1)
    steps = np.arange(len(separation))
    # Fit the exponent over the window where separation is growing but has not saturated.
    usable = (separation > separation[separation > 0].min() * 3) & (separation < 0.5 * separation.max())
    rate = np.polyfit(steps[usable], np.log(separation[usable]), 1)[0] if usable.sum() > 20 else np.nan
    sample = lambda k: separation[min(k, len(separation) - 1)]
    doublings = np.log2(separation[-1] / separation[separation > 0][0]) if separation[-1] > 0 else np.nan
    records[screenSize] = dict(separation=separation, rate=float(rate), doublings=float(doublings))
    print(f"{screenSize:>7}{sample(100):>10.2e}{sample(400):>10.2e}{sample(900):>10.2e}"
          f"{separation[-1]:>10.2e}{rate*100:>14.4f}{doublings:>11.1f}")
np.savez(f'{args.outputPrefix}.npz', records=records, screenSizes=screenSizes,
         stepFraction=args.stepFraction)
print(f"\nSaved {args.outputPrefix}.npz")
