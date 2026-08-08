"""
Does CT survive intervention, or is it correlation the tissue never agreed to?

CT is fitted observationally: a ridge regression from the boundary code to interior pattern modes,
scored by cross-validated variance explained. That fit makes predictions about what should happen
if the boundary code were *moved*, and nothing in the metric has ever tested them. This does.

For interior mode m the regression supplies a coefficient vector beta_m, the boundary-code
direction it claims drives that mode. The experiment steps the pre-pattern G_pol along that
direction, restarts the tissue from the perturbed state (Section 8.3: pre-pattern G_pol is the
entire state free evolution needs), and asks three questions of the outcome.

  gain         does mode m move by beta_m . delta_g, the predicted amount?
  specificity  does mode m move ALONE, or does the push swing other modes with it?
  range        over what amplitude does the linear prediction survive?

The second question is the one with teeth. CT sums R^2 across modes as though each were
separately settable. If a push meant for one mode swings three others, the number of independently
controllable directions is smaller than CT reports, and CT overcounts. A random-direction null of
matched amplitude is run alongside: if fitted directions do not beat random, the map has no causal
content at all and CT is measuring ensemble structure rather than control.

Two caveats are built into the measurement rather than argued away. The conductance ODE clips
G_pol to [0, 2*G_ref], so the delivered perturbation is measured and reported rather than assumed
from the nominal amplitude. And the intervention is free rather than reachable: it writes a
boundary code directly, which no clamp is guaranteed to be able to produce. That measures whether
the dynamics obey the fit, not whether the actuator can ask them to -- a distinction worth keeping
separate, and the gap between the two is a further experiment.
"""

import argparse, gc
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import utilities
from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat',    type=str,   default='data/StigmergicModelParameters_30x30.dat')
parser.add_argument('--sweepDir',     type=str,   default='data/fieldRangeSweepDense')
parser.add_argument('--fieldScreenSize', type=float, default=4)
parser.add_argument('--numModes',     type=int,   default=5)
parser.add_argument('--fractions',    type=str,   default='[0.05,0.1,0.2,0.4]')
parser.add_argument('--numRandom',    type=int,   default=4)
parser.add_argument('--numSimIters',  type=int,   default=2500)
parser.add_argument('--clampIters',   type=int,   default=100)
parser.add_argument('--readoutIters', type=int,   default=200)
parser.add_argument('--clampSeed',    type=int,   default=7)
parser.add_argument('--otherClampSeed', type=int, default=99)
parser.add_argument('--ridgeAlphas',  type=str,   default='logspace(-6,6,25)')
parser.add_argument('--outputPrefix', type=str,   default='data/ctCausal')
args = parser.parse_args()
fractions = eval(args.fractions)
torch.set_grad_enabled(False)
prePatternIter = args.clampIters + 1


def screenDirName(value):
    return (f'screen{int(value):02d}' if float(value).is_integer()
            else f'screen{int(value):02d}p{round((float(value) % 1) * 10):.0f}')


def buildParameters():
    parameters = torch.load(args.sourceDat, weights_only=False)
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['fieldParameters']['fieldScreenSize'] = args.fieldScreenSize
    cells = parameters['latticeDims'][0] * parameters['latticeDims'][1]
    values = parameters['simParameters']['initialValues']
    if 'ligandConc' not in values:
        values['ligandConc'] = torch.zeros((1, cells, 1), dtype=torch.float64)
    return parameters


utils = utilities.utilities()
reference = model(buildParameters(), 1)
circuit = reference.electricNetwork
numCells = circuit.numCells
G_ref = circuit.G_ref
boundaryIndices = np.array(utils.computeDomeIndices(circuit, mode='tissue'))
boundaryMask = np.zeros(numCells, bool); boundaryMask[boundaryIndices] = True
interiorMask = ~boundaryMask
leftHalf = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
mirrored = utils.computeSymmetricalIndices(circuit, leftHalf, mode='field', symmetry='twofold')
allIndices = np.concatenate((leftHalf, mirrored))
_, uniqueIdx = np.unique(allIndices, return_index=True)
clampPointIndices = allIndices[uniqueIdx]
clampIndices = (np.zeros(len(clampPointIndices), dtype=int), clampPointIndices)
numHalf = len(leftHalf)
timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
del reference; gc.collect()


def clampedToPrePattern(seed):
    """Run one clamp to the pre-pattern step; return (Vmem, G_pol) there."""
    generator = torch.Generator().manual_seed(seed)
    frequencies = torch.rand(numHalf, generator=generator, dtype=torch.double) * 900 + 100
    phases = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 * torch.pi
    amplitudes = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 - 1
    clampValues = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
                   * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
    instance = model(buildParameters(), 1)
    instance.simulate(numSimIters=prePatternIter + 1, fieldModulation=True, perturbation=None,
                      clampParameters={'clampMode': 'fieldDomeTwoFoldSymmetry',
                                       'clampIndices': clampIndices, 'clampValues': clampValues,
                                       'clampStartIter': 0, 'clampEndIter': args.clampIters},
                      storeVariables=('Vmem', 'Gpol'))
    vmem = instance.timeseriesVmem[prePatternIter].clone()
    gpol = instance.timeseriesGpol[prePatternIter].clone()
    del instance; gc.collect()
    return vmem, gpol


def freeEvolveFrom(vmemPre, gpolPre):
    """Free evolution from a pre-pattern state; time-averaged interior readout in mV."""
    parameters = buildParameters()
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
    preceding = instance.timeseriesVmem[-(args.readoutIters - 1):, 0, :, 0].detach().numpy()
    final = net.Vmem[0, :, 0].detach().numpy()
    del instance; gc.collect()
    return (preceding.sum(axis=0) + final) / args.readoutIters * 1000.0


# ── The observational model whose predictions are on trial ───────────────────
prefix = f'{args.sweepDir}/{screenDirName(args.fieldScreenSize)}'
ensembleCode = np.load(f'{prefix}_gpol_prepatterns.npy')[:, boundaryMask]
ensembleVmem = np.load(f'{prefix}_vmem_final.npy')[:, interiorMask] * 1000.0
ensembleVmem = ensembleVmem - ensembleVmem.mean(axis=1, keepdims=True)   # regional centring
patternMean = ensembleVmem.mean(axis=0)
pca = PCA(n_components=args.numModes).fit(ensembleVmem - patternMean)
scores = pca.transform(ensembleVmem - patternMean)

codeMean = ensembleCode.mean(axis=0)
codeScale = ensembleCode.std(axis=0)
codeScale[codeScale < 1e-30] = 1e-30            # cells that never vary contribute nothing
# The ridge must be fitted on standardised features. G_pol is of order 1e-9, so X'X is of order
# 1e-18 and every alpha in any sane search range swamps it, shrinking all coefficients to exactly
# zero -- which a first run of this script did, silently, reporting predicted responses of 0.00
# and gains of 1e18. Coefficients are transformed back to raw G_pol units afterwards, since the
# intervention has to be expressed as a voltage-conductance step, not a z-score.
ridge = RidgeCV(alphas=eval(f'np.{args.ridgeAlphas}'), alpha_per_target=True)
ridge.fit((ensembleCode - codeMean) / codeScale, scores)
coefficients = ridge.coef_ / codeScale          # (numModes, numBoundaryCells), raw G_pol units

def projectPattern(pattern):
    centred = pattern - pattern.mean()
    return pca.transform((centred - patternMean).reshape(1, -1))[0]

print(f"lattice {circuit.numCells} cells | fieldScreenSize {args.fieldScreenSize} | "
      f"{args.numModes} modes | ensemble N={len(ensembleVmem)}")
print(f"mode amplitudes (score SD): " + ', '.join(f'{s:.1f}' for s in scores.std(axis=0)))

vmemBase, gpolBase = clampedToPrePattern(args.clampSeed)
_, gpolOther = clampedToPrePattern(args.otherClampSeed)
betweenClampRMS = ((gpolOther - gpolBase)[0, boundaryIndices, 0] / G_ref).pow(2).mean().sqrt().item()
basePattern = freeEvolveFrom(vmemBase, gpolBase)
baseScores = projectPattern(basePattern[interiorMask])
print(f"between-clamp boundary G_pol per-cell RMS: {betweenClampRMS:.4f} (units of G_ref)\n")


def intervene(direction, fraction):
    """Step boundary G_pol along `direction`; return achieved step and observed score changes."""
    step = torch.zeros_like(gpolBase)
    unit = direction / np.linalg.norm(direction)
    step[0, boundaryIndices, 0] = torch.tensor(unit * fraction * betweenClampRMS * G_ref
                                               * np.sqrt(len(boundaryIndices)))
    perturbed = torch.clip(gpolBase + step, 0.0, 2.0 * G_ref)
    achieved = (perturbed - gpolBase)[0, boundaryIndices, 0].numpy()      # what was really delivered
    pattern = freeEvolveFrom(vmemBase, perturbed)
    return achieved, projectPattern(pattern[interiorMask]) - baseScores


records = []
print(f"{'target':>8}{'fraction':>9}{'delivered':>11}{'predicted':>11}{'observed':>10}"
      f"{'gain':>8}{'on-target':>11}")
for modeIndex in range(args.numModes):
    for fraction in fractions:
        achieved, delta = intervene(coefficients[modeIndex], fraction)
        predicted = float(achieved @ coefficients[modeIndex])
        observed = float(delta[modeIndex])
        gain = observed / predicted if predicted != 0 else np.nan
        onTarget = delta[modeIndex] ** 2 / (delta ** 2).sum() if (delta ** 2).sum() > 0 else np.nan
        clippedFraction = float(np.mean(np.abs(achieved) < 1e-15))
        records.append(dict(target=f'mode{modeIndex+1}', fraction=fraction, predicted=predicted,
                            observed=observed, gain=gain, onTarget=onTarget,
                            deltaAll=delta.copy(), clipped=clippedFraction,
                            responseNorm=float(np.linalg.norm(delta))))
        print(f"{f'mode {modeIndex+1}':>8}{fraction:>9g}"
              f"{np.abs(achieved).mean()/G_ref:>11.4f}{predicted:>11.2f}{observed:>10.2f}"
              f"{gain:>8.2f}{onTarget:>11.1%}")

print(f"\n{'random':>8}{'fraction':>9}{'delivered':>11}{'response norm':>15}")
rng = np.random.default_rng(0)
for trial in range(args.numRandom):
    fraction = fractions[len(fractions) // 2]
    achieved, delta = intervene(rng.standard_normal(len(boundaryIndices)), fraction)
    records.append(dict(target=f'random{trial+1}', fraction=fraction, predicted=np.nan,
                        observed=np.nan, gain=np.nan, onTarget=np.nan, deltaAll=delta.copy(),
                        clipped=np.nan, responseNorm=float(np.linalg.norm(delta))))
    print(f"{f'random {trial+1}':>8}{fraction:>9g}{np.abs(achieved).mean()/G_ref:>11.4f}"
          f"{np.linalg.norm(delta):>15.2f}")

fittedNorms = [r['responseNorm'] for r in records
               if r['target'].startswith('mode') and r['fraction'] == fractions[len(fractions)//2]]
randomNorms = [r['responseNorm'] for r in records if r['target'].startswith('random')]
print(f"\nat fraction {fractions[len(fractions)//2]}: fitted directions move the interior by "
      f"{np.mean(fittedNorms):.2f} on average, random directions by {np.mean(randomNorms):.2f} "
      f"({np.mean(fittedNorms)/max(np.mean(randomNorms), 1e-12):.2f}x)")
np.savez(f'{args.outputPrefix}_screen{args.fieldScreenSize}.npz', records=records,
         coefficients=coefficients, modeAmplitudes=scores.std(axis=0),
         betweenClampRMS=betweenClampRMS)
print(f"Saved {args.outputPrefix}_screen{args.fieldScreenSize}.npz")
