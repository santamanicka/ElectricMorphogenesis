"""PolyPatterning_Sim.md Sec 11.11: depth 3 vs. depth 7 for the band-limited static hold
(tissueBandGpolTwoFoldSymmetry, screen4/weight1000, 100-iteration hold). Fixed, final figure --
not regenerated as later experiments progress, unlike plotTissueBandInterim.py which this was
forked from. Same layout/convention as plotBestFaceAllClasses.py: rows are classes, columns are
target / clamp iter 1 / prepattern / correlation-loss best / globalsum-loss best. target/actual are
read straight from each checkpoint's own stored targetVmem/actualVmem -- nothing re-simulated there.
The prepattern and clamp-iter-1 columns are not stored anywhere, so they require a short replay of
just the clamp phase -- cheap and low chaos-risk, same reasoning as
compareTrainedPrepatternCommitment.py's own prepattern read.
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

torch.set_grad_enabled(False)


def replayPrepattern(fileNumber):
    p = torch.load(f'data/bestModelParameters_fieldVector_30x30_{fileNumber}.dat', map_location='cpu', weights_only=False)
    clampParameters = dict(p['clampParameters'])
    clampStartIter = int(clampParameters['clampStartIter'])
    clampEndIter = int(clampParameters['clampEndIter'])
    p['latticePeriodicBoundaryGJ'] = False
    p['ATPParameters'] = None
    system = model(p, p['simParameters']['numSamples'])
    circuit = system.electricNetwork
    clampStartVmem = None
    for it in range(clampEndIter + 2):
        cp = clampParameters if it <= clampEndIter else None
        system.simulate(clampParameters=cp, numSimIters=1, outerIter=it, fieldModulation=False)
        if it == clampStartIter:
            # state right after the first clamped step -- one simulate() call in, not the raw
            # pre-clamp initial condition
            clampStartVmem = circuit.Vmem[0, :, 0].detach().numpy().copy() * 1000
    prepatternVmem = circuit.Vmem[0, :, 0].detach().numpy() * 1000
    return clampStartVmem, prepatternVmem, clampStartIter, clampEndIter

classes = [
    ('depth 3, 100-iter hold', [1407,1408,1409,1410,1411,1412], [1413,1414,1415,1416,1417,1418]),
    ('depth 7, 100-iter hold', [1419,1420,1421,1422,1423,1424], [1425,1426,1427,1428,1429,1430]),
]


def bestInGroup(nums):
    best = None
    for n in nums:
        f = f"data/bestModelParameters_fieldVector_30x30_{n}.dat"
        try:
            p = torch.load(f, map_location='cpu', weights_only=False)
        except Exception:
            continue
        L = float(p['trainParameters']['bestLoss'])
        if best is None or L < best[0]:
            best = (L, n, p)
    return best


rows, cols = 30, 30
# Deliberately auto-scaled per panel (imshow's default), not anchored to the target's fixed -60/-9.2
# mV range -- that anchoring is more physically honest for judging closeness to target, but it washes
# out real internal structure in low-amplitude panels (iter 1 renders as blank white under it, even
# though it has its own -- much smaller -- variation worth seeing). Per-panel scaling means panels are
# not directly comparable to each other in absolute terms; read amplitude from the printed numbers
# in the conversation, not from how dark two different panels look.
fig, axes = plt.subplots(len(classes), 5, figsize=(15.5, 3.2 * len(classes)))
for row, (label, corrNums, globNums) in enumerate(classes):
    c = bestInGroup(corrNums)
    g = bestInGroup(globNums)

    axTarget = axes[row, 0]
    if c is not None:
        target = c[2]['trainParameters']['targetVmem'].reshape(rows, cols).numpy() * 1000
        axTarget.imshow(target, cmap='gray')
    axTarget.set_ylabel(label, fontsize=11)
    axTarget.set_title('target' if row == 0 else '', fontsize=10)
    axTarget.set_xticks([]); axTarget.set_yticks([])

    axStart = axes[row, 1]
    axPre = axes[row, 2]
    if c is not None:
        n = c[1]
        clampStartVmem, prepattern, clampStartIter, clampEndIter = replayPrepattern(n)

        axStart.imshow(clampStartVmem.reshape(rows, cols), cmap='gray')
        startTitle = f'clamp iter {clampStartIter+1}\nfile {n} (corr checkpoint)' if row == 0 else f'file {n}, iter {clampStartIter+1}'
        axStart.set_title(startTitle, fontsize=9)

        axPre.imshow(prepattern.reshape(rows, cols), cmap='gray')
        preTitle = f'prepattern (iter {clampEndIter+1})\nfile {n} (corr checkpoint)' if row == 0 else f'file {n}, iter {clampEndIter+1}'
        axPre.set_title(preTitle, fontsize=9)
    else:
        axStart.set_title('no checkpoint', fontsize=9)
        axPre.set_title('no checkpoint', fontsize=9)
    axStart.set_xticks([]); axStart.set_yticks([])
    axPre.set_xticks([]); axPre.set_yticks([])

    axCorr = axes[row, 3]
    if c is not None:
        L, n, p = c
        actual = p['trainParameters']['actualVmem'].reshape(rows, cols).numpy() * 1000
        axCorr.imshow(actual, cmap='gray')
        axCorr.set_title(f'correlation (final)\nfile {n}, loss {L:.3f}' if row == 0 else f'file {n}, loss {L:.3f}', fontsize=9)
    else:
        axCorr.set_title('no checkpoint', fontsize=9)
    axCorr.set_xticks([]); axCorr.set_yticks([])

    axGlob = axes[row, 4]
    if g is not None:
        L, n, p = g
        actual = p['trainParameters']['actualVmem'].reshape(rows, cols).numpy() * 1000
        axGlob.imshow(actual, cmap='gray')
        axGlob.set_title(f'globalsum (final)\nfile {n}, loss {L:.3f}' if row == 0 else f'file {n}, loss {L:.3f}', fontsize=9)
    else:
        axGlob.set_title('no checkpoint', fontsize=9)
    axGlob.set_xticks([]); axGlob.set_yticks([])

fig.suptitle('Tissue-band static hold, depth 3 vs. depth 7 (30x30, screen4/weight1000, 100-iter hold)', fontsize=12)
fig.tight_layout()
fig.savefig('figures/tissueBandDepthComparison_screen4weight1000.png', dpi=140, bbox_inches='tight')
print('wrote figures/tissueBandDepthComparison_screen4weight1000.png')
