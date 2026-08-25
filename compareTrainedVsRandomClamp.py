"""For each boundary-clamp screen size, compare the trained best face against a baseline of 10
untrained (randomly parameterized) clamps of the same family, to see how much the learned
clampFrequencies/clampPhases actually buy over chance.

Each random draw uses the same clampMode/clampIndices/clampDurationProp as the trained checkpoint
(same boundary points, same 100-iteration forcing window, same field screen/weight) -- only the
per-point frequency, phase and amplitude are redrawn from the same distributions
learnCellularFieldNetwork.py uses at its own random initialization (frequency ~ U(100,1000), phase
~ U(0,2pi), amplitude ~ U(-1,1)), independently per clamp point. The trained clamp additionally
ties each point to its mirror partner (fieldDomeTwoFoldSymmetry); this control does not enforce
that tie, which is a modest simplification worth keeping in mind when reading the comparison.

Because the random draw doesn't depend on lossMethod, the same 10 simulations are scored under both
the correlation and globalsum formulas, so only one random-clamp run is needed per screen.
"""
import argparse

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--numDraws', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.set_grad_enabled(False)
torch.manual_seed(args.seed)

# (screenLabel, weight, corrFile, globFile) -- the best checkpoint per screen from the boundary-clamp
# comparison plots. globFile is None where no globalsum run exists yet (screen10).
boundaryScreens = [
    ('screen2', 700, 1604, 1706),
    ('screen3', 700, 1804, 1902),
    ('screen8', 700, 1403, 1502),
    ('screen4', 1000, 904, 511),
    ('screen10', 1000, 954, None),
    ('screen24', 1000, 1001, 1106),
]


def loadCheckpoint(fileNumber):
    p = torch.load(f'data/bestModelParameters_fieldVector_30x30_{fileNumber}.dat', map_location='cpu', weights_only=False)
    p['latticePeriodicBoundaryGJ'] = False
    p['ATPParameters'] = None
    return p


def correlationLoss(observedWindow, target):
    centredObserved = observedWindow - observedWindow.mean(dim=2, keepdim=True)
    centredTarget = target - target.mean(dim=1, keepdim=True)
    covariance = (centredObserved * centredTarget).sum(dim=2)
    normalisation = (centredObserved.pow(2).sum(dim=2).sqrt() * centredTarget.pow(2).sum(dim=1).sqrt())
    return (1 - (covariance / (normalisation + 1e-12))).mean().item()


def globalsumLoss(observedWindow, target):
    return ((target - observedWindow) ** 2).sum().sqrt().item()


def runRandomDraw(templateParams, clampParameters, numSimIters, evalDuration, minFreq, maxFreq, minAmp, maxAmp):
    sampleIndices, clampPointIndices = clampParameters['clampIndices']
    numClampIters = int(clampParameters['clampEndIter']) - int(clampParameters['clampStartIter']) + 1
    numPoints = len(clampPointIndices)
    freq = torch.rand(numPoints, dtype=torch.double) * (maxFreq - minFreq) + minFreq
    phase = torch.rand(numPoints, dtype=torch.double) * 2 * torch.pi
    amp = torch.rand(numPoints, dtype=torch.double) * (maxAmp - minAmp) + minAmp
    timeIndices = torch.linspace(0, 0.5, numClampIters, dtype=torch.double).view(-1, 1)
    randomClampValues = torch.cos(timeIndices * freq + phase) * amp

    randomClampParameters = dict(clampParameters)
    randomClampParameters['clampValues'] = randomClampValues

    system = model(templateParams, templateParams['simParameters']['numSamples'])
    system.simulate(clampParameters=randomClampParameters, numSimIters=numSimIters, storeVariables=['Vmem'])
    window = system.timeseriesVmem[-evalDuration:]
    return window  # (evalDuration,numSamples,numCells,1), full readout window -- do not average yet,
    # the loss formulas below need every frame, matching computeLoss() in learnCellularFieldNetwork.py


results = []
for screenLabel, weight, corrFile, globFile in boundaryScreens:
    print(f"=== {screenLabel} (weight {weight}) ===", flush=True)
    corrP = loadCheckpoint(corrFile)
    rows, cols = corrP['latticeDims']
    numSimIters = corrP['simParameters']['numSimIters']
    evalDurationProp = corrP['trainParameters']['evalDurationProp']
    evalDuration = int(evalDurationProp * numSimIters)
    clampParameters = dict(corrP['clampParameters'])
    target = corrP['trainParameters']['targetVmem']
    corrBestLoss = float(corrP['trainParameters']['bestLoss'])
    corrActual = corrP['trainParameters']['actualVmem']

    globP = loadCheckpoint(globFile) if globFile is not None else None
    globBestLoss = float(globP['trainParameters']['bestLoss']) if globP is not None else None
    globActual = globP['trainParameters']['actualVmem'] if globP is not None else None

    averagedFrames = []
    drawCorrLosses = []
    drawGlobLosses = []
    for draw in range(args.numDraws):
        window = runRandomDraw(corrP, clampParameters, numSimIters, evalDuration,
                                minFreq=100.0, maxFreq=1000.0, minAmp=-1.0, maxAmp=1.0)
        averagedFrames.append(window.mean(dim=0))  # single frame, for the visual comparison only
        drawCorrLosses.append(correlationLoss(window, target))
        drawGlobLosses.append(globalsumLoss(window, target))
        print(f"  draw {draw}: corrLoss={drawCorrLosses[-1]:.4f}  globLoss={drawGlobLosses[-1]:.4f}", flush=True)

    randomAveragePattern = torch.stack(averagedFrames, dim=0).mean(dim=0)  # (numSamples,numCells,1)
    corrLossMean = sum(drawCorrLosses) / len(drawCorrLosses)
    corrLossStd = (sum((x - corrLossMean) ** 2 for x in drawCorrLosses) / len(drawCorrLosses)) ** 0.5
    globLossMean = sum(drawGlobLosses) / len(drawGlobLosses)
    globLossStd = (sum((x - globLossMean) ** 2 for x in drawGlobLosses) / len(drawGlobLosses)) ** 0.5

    print(f"  trained corr loss: {corrBestLoss:.4f}  |  random corr loss: {corrLossMean:.4f} +/- {corrLossStd:.4f}")
    if globBestLoss is not None:
        print(f"  trained glob loss: {globBestLoss:.4f}  |  random glob loss: {globLossMean:.4f} +/- {globLossStd:.4f}")
    print(flush=True)

    results.append(dict(screenLabel=screenLabel, weight=weight, rows=rows, cols=cols, target=target,
                         corrActual=corrActual, corrBestLoss=corrBestLoss,
                         globActual=globActual, globBestLoss=globBestLoss,
                         randomAveragePattern=randomAveragePattern,
                         corrLossMean=corrLossMean, corrLossStd=corrLossStd,
                         globLossMean=globLossMean, globLossStd=globLossStd))

fig, axes = plt.subplots(len(results), 4, figsize=(12.5, 3.2 * len(results)))
for row, r in enumerate(results):
    rows_, cols_ = r['rows'], r['cols']
    axTarget = axes[row, 0]
    axTarget.imshow(r['target'].reshape(rows_, cols_).numpy() * 1000, cmap='gray')
    axTarget.set_ylabel(f"{r['screenLabel']}\n(w{r['weight']})", fontsize=10)
    axTarget.set_title('target' if row == 0 else '', fontsize=10)
    axTarget.set_xticks([]); axTarget.set_yticks([])

    axCorr = axes[row, 1]
    axCorr.imshow(r['corrActual'].reshape(rows_, cols_).numpy() * 1000, cmap='gray')
    axCorr.set_title(('trained (correlation)\n' if row == 0 else '') + f"loss {r['corrBestLoss']:.3f}", fontsize=9)
    axCorr.set_xticks([]); axCorr.set_yticks([])

    axGlob = axes[row, 2]
    if r['globActual'] is not None:
        axGlob.imshow(r['globActual'].reshape(rows_, cols_).numpy() * 1000, cmap='gray')
        axGlob.set_title(('trained (globalsum)\n' if row == 0 else '') + f"loss {r['globBestLoss']:.3f}", fontsize=9)
    else:
        axGlob.set_title('no checkpoint', fontsize=9)
    axGlob.set_xticks([]); axGlob.set_yticks([])

    axRand = axes[row, 3]
    axRand.imshow(r['randomAveragePattern'].reshape(rows_, cols_).numpy() * 1000, cmap='gray')
    axRand.set_title((f'{args.numDraws}-random average\n' if row == 0 else '')
                      + f"corr {r['corrLossMean']:.2f}+/-{r['corrLossStd']:.2f}\n"
                      + f"glob {r['globLossMean']:.1f}+/-{r['globLossStd']:.1f}", fontsize=8)
    axRand.set_xticks([]); axRand.set_yticks([])

fig.suptitle(f'Trained boundary-clamp faces vs. {args.numDraws}-random-clamp average, by screen size', fontsize=13)
fig.tight_layout()
outputPath = 'figures/trainedVsRandomClampByScreen.png'
fig.savefig(outputPath, dpi=140, bbox_inches='tight')
print(f"wrote {outputPath}")
