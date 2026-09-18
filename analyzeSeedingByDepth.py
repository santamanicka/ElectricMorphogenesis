"""Bulk pre-patterning by depth shell, at both lattice sizes.

The aggregate interior numbers (§ interior seeding) average over the whole bulk and could hide
structure that only shows up depth-resolved -- particularly whether the un-saturated, still-
differentiating region is a thin band near the boundary (consistent with §7's controlled band that
does not scale with tissue size) rather than spread evenly through the interior.

Depth 0 is the boundary ring itself (excluded from "interior" everywhere else in this analysis);
depth 1 is the next ring in, and so on to the centre.
"""
import numpy as np
import torch

from embryo import model

torch.set_grad_enabled(False)


def depthShell(rows, cols):
    r, c = np.indices((rows, cols))
    depth = np.minimum(np.minimum(r, rows-1-r), np.minimum(c, cols-1-c))
    return depth.reshape(-1)


def prepattern(parameterfile):
    parameters = torch.load(parameterfile, weights_only=False)
    rows, cols = parameters['latticeDims']
    numCells = rows*cols
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
                      numSimIters=clampEnd+1, storeVariables=['Vmem'])
    gpol = (circuit.G_pol.detach().clone().reshape(-1)/circuit.G_ref).numpy()
    return gpol, rows, cols


for label, path in (('11x11', './data/StigmergicModelParameters.dat'),
                    ('30x30', './data/bestModelParameters_fieldVector_30x30_616.dat')):
    gpol, rows, cols = prepattern(path)
    depth = depthShell(rows, cols)
    maxDepth = depth.max()
    print(f"\n  ===== {label}: {maxDepth+1} depth shells =====")
    print(f"  {'depth':>6s} {'cells':>7s} {'near-0':>8s} {'near-2':>8s} {'mid-range':>10s} {'mean':>8s} {'std':>8s}")
    for d in range(maxDepth+1):
        g = gpol[depth == d]
        near0 = 100*(g < 0.2).mean()
        near2 = 100*(g > 1.8).mean()
        mid = 100 - near0 - near2
        print(f"  {d:6d} {len(g):7d} {near0:7.1f}% {near2:7.1f}% {mid:9.1f}% {g.mean():8.3f} {g.std():8.3f}")
