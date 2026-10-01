"""EXPLORATORY: the readable version of the order -> edge map. Nothing here was predicted or registered beforehand.

analyzeRelayLoopOrderEdgeMap11x11.py learns, for 25 signed edges, how the ring's four region levels (top, upper sides, lower sides,
bottom) switch each edge in or out of the net's top three. This script asks whether that table has a story:
  1. how many independent patterns the 25 x 4 map contains (SVD);
  2. whether the edges fall into a few modules that the same regions switch together, defined from the map's clearest edges:
       lower channel   the trained lower circulation: in write, ring bottom -> bg bottom-left or ring left -> bg bottom-left
       flood push      in flood, ring top -> bg top-left
       reversed lower  its mirror image: in write bg bottom-left -> ring bottom, or in clear ring bottom -> mouth, or in flood
                       ring bottom -> bg bottom-left
  3. which ring levels each module needs, from the 400 sweep codes (and where the trained code sits);
  4. whether the modules say anything about the outcome: face overlap >= 0.3 and >= 0.5, and the conductance gap, by module;
  5. how a +0.2 step in each order moves the ring in the plane of its two switches (the order -> region step is exact, from
     the clipped-cosine basis), which is what makes order 0 the master dial.
A "module present" is an edge being in the code's top three of its phase, the same view as the Relay Loop page.

Writes data/relayLoopOrderEdgeStory1888Hold301FaceMinus60Minus5.json and figures/relayLoopOrderEdgeStory.png (never overwriting).

    python3 analyzeRelayLoopOrderEdgeStory11x11.py
"""
import json
import os

import numpy as np
from scipy import stats

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301FaceMinus60Minus5'
outputPath = f'data/relayLoopOrderEdgeStory{SUFFIX}.json'
figurePath = 'figures/relayLoopOrderEdgeStory.png'
for path in (outputPath, figurePath):
    if os.path.exists(path):
        raise SystemExit(f'{path} exists; not overwriting')

edgeMap = json.load(open(f'data/relayLoopOrderEdgeMap{SUFFIX}.json'))
sweep = json.load(open(f'data/relayLoopSweepNets{SUFFIX}.json'))
page = json.load(open(f'data/relayLoopFullNets{SUFFIX}.json'))
trainedCoefficients = np.asarray(json.load(open(f'data/boundaryHarmonicRingCodeFiveLevel{SUFFIX}.json'))['trainedCoefficients'])
basis = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(4)))
foldedAngle = np.abs(np.angle(np.exp(1j * boundary.ringAngles(boundary.boundaryRingCells))))
bins = np.digitize(foldedAngle, [np.pi / 4, np.pi / 2, 3 * np.pi / 4])
REGIONS = ['top', 'upper sides', 'lower sides', 'bottom']
pairs, phaseNames = [tuple(p) for p in sweep['pairs']], sweep['phases']
keys = sorted(sweep['codes'])
transfer = np.array([sweep['codes'][k]['field'] for k in keys])
face = np.array([sweep['codes'][k]['faceOverlap'] for k in keys])
gap = np.array([sweep['codes'][k]['gap'] for k in keys])
multipliers = np.array([sweep['codes'][k]['multipliers'] for k in keys])


def regionLevels(multiplierRows):
    values = np.clip(basis @ (np.atleast_2d(multiplierRows) * trainedCoefficients).T, 0.0, 2.0).T
    return np.array([[row[bins == b].mean() for b in range(4)] for row in values])


levels = regionLevels(multipliers)
trainedLevels = regionLevels(np.ones(4))[0]


def topEdges(code):
    chosen = set()
    for phase in range(3):
        for index in np.argsort(-np.abs(code[phase]))[:3]:
            a, b = pairs[index]
            chosen.add((phaseNames[phase], a, b) if code[phase, index] > 0 else (phaseNames[phase], b, a))
    return chosen


tops = [topEdges(code) for code in transfer]
anyOf = lambda *edges: np.array([any(e in t for e in edges) for t in tops])
lowerChannel = anyOf(('write', 'ringBottom', 'bgBL'), ('write', 'ringLeft', 'bgBL'))
floodPush = anyOf(('flood', 'ringTop', 'bgTL'))
reversedLower = anyOf(('write', 'bgBL', 'ringBottom'), ('clear', 'ringBottom', 'mouth'), ('flood', 'ringBottom', 'bgBL'))
result = dict(status='EXPLORATORY: no predictions or decision criteria were registered; see the module docstring', sweepCodes=len(keys))

# ---------------------------------------------------------------- 1. how many patterns
regionMap = np.array([m['regionLogitPerSd'] for m in edgeMap['orderMap']])
_, singular, rows = np.linalg.svd(regionMap, full_matrices=False)
share = singular ** 2 / np.sum(singular ** 2)
result['svd'] = dict(edges=len(regionMap), varianceShare=share.round(3).tolist(),
                     patterns={f'pattern{k + 1}': dict(zip(REGIONS, rows[k].round(2).tolist())) for k in range(2)})

# ---------------------------------------------------------------- 3. the ring levels each module needs
wilson = lambda hits, total: tuple(float(x) for x in ((lambda p, z: ((p + z * z / (2 * total) - z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / (1 + z * z / total),
                                                                    (p + z * z / (2 * total) + z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / (1 + z * z / total)))(hits / total, 1.96)))
describe = lambda mask: dict(codes=int(mask.sum()), levelMean=dict(zip(REGIONS, levels[mask].mean(0).round(2).tolist())),
                             levelSd=dict(zip(REGIONS, levels[mask].std(0).round(2).tolist())))
result['ringLevels'] = dict(trained=dict(zip(REGIONS, trainedLevels.round(2).tolist())), lowerChannelPresent=describe(lowerChannel), lowerChannelAbsent=describe(~lowerChannel),
                            floodPushPresent=describe(floodPush), floodPushAbsent=describe(~floodPush), reversedLowerPresent=describe(reversedLower))

# ---------------------------------------------------------------- 4. the modules and the outcome
cells = {}
for lower in (False, True):
    for push in (False, True):
        mask = (lowerChannel == lower) & (floodPush == push)
        cells[f"lower={lower}, push={push}"] = dict(codes=int(mask.sum()), faceAtLeast0_3=float((face[mask] >= 0.3).mean()), faceAtLeast0_3Interval=wilson(int((face[mask] >= 0.3).sum()), int(mask.sum())),
                                                    faceAtLeast0_5=int((face[mask] >= 0.5).sum()), meanGap=float(gap[mask].mean()))
both = lowerChannel & floodPush
table = [[int(((face >= 0.3) & both).sum()), int(((face < 0.3) & both).sum())], [int(((face >= 0.3) & ~both).sum()), int(((face < 0.3) & ~both).sum())]]
result['modulesAndOutcome'] = dict(cells=cells, bothVersusRestFisherP=float(stats.fisher_exact(table)[1]), bothVersusRestTable=table,
                                   facesAtLeast0_5=[dict(key=keys[i], multipliers=multipliers[i].round(2).tolist(), faceOverlap=float(face[i]), lowerChannel=bool(lowerChannel[i]),
                                                         floodPush=bool(floodPush[i]), reversedLower=bool(reversedLower[i]), ringLevels=dict(zip(REGIONS, levels[i].round(2).tolist())))
                                                    for i in np.where(face >= 0.5)[0]],
                                   reversedLowerAmongFaces=int(reversedLower[face >= 0.3].sum()), facesAtLeast0_3=int((face >= 0.3).sum()))

# ---------------------------------------------------------------- 5. what a step in each order does to the ring
step = 0.2
orderSteps = []
for order in range(4):
    moved = np.ones(4)
    moved[order] += step
    orderSteps.append(dict(order=order, plusStepInMultiplier=step, change=dict(zip(REGIONS, (regionLevels(moved)[0] - trainedLevels).round(3).tolist()))))
result['orderSteps'] = orderSteps

# the same table on the space-filling codes alone, so the near-trained cloud (which has both modules and most of the faces) cannot be what drives it
isGlobal = np.array([k.startswith('sweepGlobal') for k in keys])
globalCells = {}
for lower in (False, True):
    for push in (False, True):
        mask = isGlobal & (lowerChannel == lower) & (floodPush == push)
        globalCells[f"lower={lower}, push={push}"] = dict(codes=int(mask.sum()), faceAtLeast0_3=float((face[mask] >= 0.3).mean()), meanGap=float(gap[mask].mean()))
globalTable = [[int(((face >= 0.3) & both & isGlobal).sum()), int(((face < 0.3) & both & isGlobal).sum())], [int(((face >= 0.3) & ~both & isGlobal).sum()), int(((face < 0.3) & ~both & isGlobal).sum())]]
result['modulesAndOutcome']['globalCodesOnly'] = dict(cells=globalCells, bothVersusRestFisherP=float(stats.fisher_exact(globalTable)[1]), bothVersusRestTable=globalTable)

json.dump(result, open(outputPath, 'w'), indent=1)
print(f'wrote {outputPath}')

# ---------------------------------------------------------------- the figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1.25, 0.9, 1.0]))
# A: the two switches
axis = axes[0]
bottom, lowerSides = levels[:, 3], levels[:, 2]
axis.scatter(bottom[~lowerChannel], lowerSides[~lowerChannel], s=14, c='#B8C0C8', label='lower channel absent')
axis.scatter(bottom[lowerChannel], lowerSides[lowerChannel], s=14, c='#33548F', label='lower channel present (trained direction)')
axis.scatter(bottom[reversedLower], lowerSides[reversedLower], s=34, facecolors='none', edgecolors='#B84052', linewidths=1.0, label='reversed lower channel present')
near = (face >= 0.3) & (face < 0.5)
axis.scatter(bottom[near], lowerSides[near], s=60, marker='D', facecolors='none', edgecolors='#2E7D46', linewidths=1.2, label='face overlap 0.3-0.5')
axis.scatter(bottom[face >= 0.5], lowerSides[face >= 0.5], s=170, marker='*', c='#2E7D46', edgecolors='k', linewidths=0.5, label='face (overlap >= 0.5)')
axis.scatter([trainedLevels[3]], [trainedLevels[2]], s=90, marker='X', c='k', label='trained code', zorder=5)
for order, color in enumerate(['#6E7C88', '#B4531F', '#7E3FA0', '#2A8F8F']):
    change = orderSteps[order]['change']
    axis.annotate('', xy=(trainedLevels[3] + 3 * change['bottom'], trainedLevels[2] + 3 * change['lower sides']), xytext=(trainedLevels[3], trainedLevels[2]),
                  arrowprops=dict(arrowstyle='-|>', color=color, lw=2.2), zorder=6)
    axis.text(trainedLevels[3] + 3.4 * change['bottom'], trainedLevels[2] + 3.4 * change['lower sides'], f'order {order}', color=color, fontsize=9, fontweight='bold', zorder=6)
axis.set_xlabel('ring bottom level (mean G_pol / G_ref)'); axis.set_ylabel('ring lower-sides level')
axis.set_title('A. Two ring levels switch the lower channel\n(arrows: a +0.2 step in each order, drawn 3x; 400 sweep codes)', fontsize=10)
axis.legend(fontsize=7, loc='upper left')
# B: the modules and the face
axis = axes[1]
labels = ['neither', 'flood push\nonly', 'lower channel\nonly', 'both']
order_ = [(False, False), (False, True), (True, False), (True, True)]
rates = [cells[f'lower={l}, push={p}']['faceAtLeast0_3'] for l, p in order_]
counts = [cells[f'lower={l}, push={p}']['codes'] for l, p in order_]
errors = np.array([[r - cells[f'lower={l}, push={p}']['faceAtLeast0_3Interval'][0], cells[f'lower={l}, push={p}']['faceAtLeast0_3Interval'][1] - r] for r, (l, p) in zip(rates, order_)]).T
axis.bar(range(4), rates, color=['#B8C0C8', '#6FC28D', '#7E9BD6', '#2E7D46'], yerr=errors, capsize=3)
axis.set_xticks(range(4), [f'{label}\nn={count}' for label, count in zip(labels, counts)], fontsize=8); axis.set_ylabel('share of codes with face overlap >= 0.3 (95% interval)')
axis.set_title('B. The face needs both modules\n(all 400 codes: p = %.0e; the 280 space-filling codes alone: p = %.0e)' % (result['modulesAndOutcome']['bothVersusRestFisherP'], result['modulesAndOutcome']['globalCodesOnly']['bothVersusRestFisherP']), fontsize=9)
# C: what each module needs
axis = axes[2]
x = np.arange(4); width = 0.2
series = [('trained', trainedLevels, 'k'), ('lower channel present', levels[lowerChannel].mean(0), '#33548F'), ('flood push present', levels[floodPush].mean(0), '#2E7D46'), ('reversed lower present', levels[reversedLower].mean(0), '#B84052')]
for k, (name, values, color) in enumerate(series):
    axis.bar(x + (k - 1.5) * width, values, width, label=name, color=color, alpha=0.85)
axis.set_xticks(x, REGIONS, fontsize=9); axis.set_ylabel('ring level (mean G_pol / G_ref)'); axis.legend(fontsize=8)
axis.set_title('C. The ring levels each module sits at', fontsize=10)
fig.tight_layout(); fig.savefig(figurePath, dpi=150)
print(f'wrote {figurePath}')
