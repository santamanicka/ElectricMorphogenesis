"""Does scaling the clamp amplitude let the interior reach BOTH poles, or just reach the same one faster?

G_pol's driving term is (2*sigmoid(gain*eV + bias) - 1) * fieldTransductionWeight, and
fieldTransductionWeight = 1000 -- large enough that G_pol saturates to a clip bound almost
immediately whenever gain*eV + bias is not extremely close to zero. So amplitude does not gate
WHETHER a cell bifurcates; it can only change WHICH side it lands on, and only if it is large enough
to flip the sign of gain*eV + bias at that cell.

Two independent random clamps drove every interior cell to the identical pole (G_pol/G_ref = 0.000,
RMS difference ~0) at the default amplitude, which is consistent with bias dominating eV at that
depth for every interior cell regardless of clamp content. If that is the mechanism, then scaling
amplitude should eventually let some interior cells reach the OTHER pole (G_pol/G_ref near 2), and
two different clamps should start disagreeing about which cells do. If it is not the mechanism --
if interior eV never approaches bias in magnitude no matter how hard the boundary is driven, because
distance and harmonic filtering attenuate it faster than amplitude can compensate -- then no
amplitude will produce the missing pole and the sweep will show the same all-0.000 collapse at every
scale tested.
"""
import numpy as np
import torch

from embryo import model

torch.set_grad_enabled(False)


def interiorMask(rows, cols):
    mask = np.ones((rows, cols), bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    return mask.reshape(-1)


def prepattern(parameterfile, amplitude, useTrainedClamp=True, seed=0):
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
    if not useTrainedClamp:
        generator = torch.Generator().manual_seed(seed)
        clampParameters['clampValues'] = torch.randn(clampParameters['clampValues'].shape,
                                                      generator=generator,
                                                      dtype=clampParameters['clampValues'].dtype)
    clampParameters['clampValues'] = clampParameters['clampValues']*amplitude
    clampEnd = int(clampParameters['clampEndIter'])

    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((initialValues, numSamples))
    circuit = instance.electricNetwork
    instance.simulate(clampParameters=clampParameters, fieldModulation=True,
                      numSimIters=clampEnd+1, storeVariables=['Vmem'])
    gpol = (circuit.G_pol.detach().clone().reshape(-1)/circuit.G_ref).numpy()
    ev = circuit.eV.detach().clone().reshape(-1).numpy()
    return gpol, ev, rows, cols


amplitudes = [1, 3, 10, 30, 100, 300, 1000]
models = [('11x11', './data/StigmergicModelParameters.dat'),
          ('30x30', './data/bestModelParameters_fieldVector_30x30_616.dat')]

for label, path in models:
    print(f"\n  ===== {label}, trained clamp =====")
    print(f"  {'amplitude':>10s} {'interior |eV| mean':>19s} {'near-0':>8s} {'near-2':>8s} {'mid':>8s}")
    for amplitude in amplitudes:
        gpol, ev, rows, cols = prepattern(path, amplitude, useTrainedClamp=True)
        inte = interiorMask(rows, cols)
        gi = gpol[inte]
        near0 = 100*(gi < 0.2).mean()
        near2 = 100*(gi > 1.8).mean()
        mid = 100 - near0 - near2
        # eV is on the field grid, not the cell grid, but its overall scale still tracks amplitude
        print(f"  {amplitude:10d} {np.abs(ev).mean():19.6f} {near0:7.1f}% {near2:7.1f}% {mid:7.1f}%")

    print(f"\n  ===== {label}, two independent random clamps: do they ever disagree? =====")
    print(f"  {'amplitude':>10s} {'clampA near-2':>14s} {'clampB near-2':>14s} {'interior RMS diff':>18s}")
    for amplitude in amplitudes:
        ga, _, rows, cols = prepattern(path, amplitude, useTrainedClamp=False, seed=1)
        gb, _, _, _ = prepattern(path, amplitude, useTrainedClamp=False, seed=2)
        inte = interiorMask(rows, cols)
        ai, bi = ga[inte], gb[inte]
        rms = np.sqrt(((ai - bi)**2).mean())
        print(f"  {amplitude:10d} {100*(ai>1.8).mean():13.1f}% {100*(bi>1.8).mean():13.1f}% {rms:18.5f}")
