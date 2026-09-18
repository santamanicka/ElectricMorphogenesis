"""Compare the 11x11 native g-trend against 30x30 across the weight axis, to find where (if
anywhere) 30x30 qualitatively matches.

Both lattices share identical transduction parameters (weight 1000, gain -1, bias 0.0005,
timeConstant 10) -- 30x30 inherits them unchanged from the 11x11-trained tissue, so 11x11's native
trend is precisely the 30x30-weight-1000 case, not some separate baseline. The earlier weight sweep
jumped from 1000 (7 of 14 shells fully committed) to 300 (0 of 14 committed) with nothing in
between, so the transition zone that might resemble 11x11's shape -- one committed shell followed by
a smooth graded decay to the centre -- has not been examined.

"Committed" here means |g| > 0.8, i.e. within 20% of a clip bound; this is a looser, sign-based
version of the near-0/near-2 threshold used earlier, chosen so it can be applied uniformly across
both lattice sizes and the full weight range without re-deriving separate thresholds each time.
"""
import numpy as np
import torch

from embryo import model

torch.set_grad_enabled(False)


def depthShell(rows, cols):
    r, c = np.indices((rows, cols))
    return np.minimum(np.minimum(r, rows - 1 - r), np.minimum(c, cols - 1 - c)).reshape(-1)


def prepattern(parameterfile, weight=None):
    parameters = torch.load(parameterfile, weights_only=False)
    if weight is not None:
        parameters['fieldParameters'] = dict(parameters['fieldParameters'])
        parameters['fieldParameters']['fieldTransductionWeight'] = torch.DoubleTensor([weight])
    rows, cols = parameters['latticeDims']
    numCells = rows * cols
    numSamples = parameters['simParameters']['numSamples']
    initialValues = parameters['simParameters']['initialValues']
    if 'ligandConc' not in initialValues:
        initialValues['ligandConc'] = torch.zeros((numSamples, numCells, 1), dtype=torch.float64)
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['ATPParameters'] = None
    clampParameters = dict(parameters['clampParameters'])
    clampEnd = int(clampParameters['clampEndIter'])

    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((initialValues, numSamples))
    circuit = instance.electricNetwork
    instance.simulate(clampParameters=clampParameters, fieldModulation=True,
                       numSimIters=clampEnd + 1, storeVariables=['Vmem'])
    gpol = (circuit.G_pol.detach().clone().reshape(-1) / circuit.G_ref).numpy()
    return gpol, rows, cols


def depthCurve(gpol, rows, cols):
    depth = depthShell(rows, cols)
    maxDepth = depth.max()
    curve = []
    for d in range(1, maxDepth + 1):
        g = gpol[depth == d] - 1.0
        curve.append(g.mean())
    return np.array(curve)


eleven = depthCurve(*prepattern('./data/StigmergicModelParameters.dat'))
print("  11x11 native (weight 1000, its own trained clamp): depth 1 -> centre")
print("   " + "  ".join(f"{v:+.3f}" for v in eleven))
committed11 = int((np.abs(eleven) > 0.8).sum())
print(f"  committed shells (|g|>0.8): {committed11} of {len(eleven)}\n")

path30 = './data/bestModelParameters_fieldVector_30x30_616.dat'
weights = [1000, 900, 800, 700, 600, 500, 400, 300]

print("  30x30, same trained clamp, across weight -- depth 1 -> centre (14 shells)")
for weight in weights:
    curve = depthCurve(*prepattern(path30, weight))
    committed = int((np.abs(curve) > 0.8).sum())
    print(f"  weight {weight:5.0f}  committed {committed:2d}/14   "
          + "  ".join(f"{v:+.3f}" for v in curve))
