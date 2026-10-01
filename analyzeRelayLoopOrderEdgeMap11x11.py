"""EXPLORATORY: learn which edges of the causal net each region of the ring switches on, and from that an order -> edge map.

Nothing here was predicted or registered beforehand. Trained on the sweep (buildBoundaryHarmonicRingCodeSweep11x11.py: 280
space-filling and 120 near-trained codes, every ring value inside [0, 2] G_pol / G_ref), checked three ways and always against
a shuffled-label null:
  random folds   10-fold stratified (flattering: neighbouring codes land in both halves)
  blocked folds  the codes are grouped into 8 blocks of multiplier space and a whole block is held out (does it extrapolate?)
  the page       fit on all sweep codes, scored on the 126 codes of the Relay Loop page (a lattice that includes clipped,
                 knocked-out and two-order codes the sweep never contains)

Target: whether a signed edge (phase, sender -> receiver) is among the three biggest transfers of its phase, the page's own top-3
view. Input: the ring's value in four regions (mean over the cells at folded angles 0-45, 45-90, 90-135 and 135-180 degrees from
the top: top, upper sides, lower sides, bottom), which is how the orders act on the tissue; the four multipliers and six regions
are compared. A random forest gives the predictive scores; an L2 logistic regression on standardised regions gives the signed
map, with bootstrap sign stability. The order map composes the region map with the orders' loadings on the regions (the
clipped-cosine basis), per +0.1 of an order's multiplier.

Writes data/relayLoopOrderEdgeMap1888Hold301FaceMinus60Minus5.json and figures/relayLoopOrderEdgeMap.png (never overwriting).

    python3 analyzeRelayLoopOrderEdgeMap11x11.py
"""
import json
import os
import warnings

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import boundaryCodeUtilities as boundary

warnings.filterwarnings('ignore')
SUFFIX = '1888Hold301FaceMinus60Minus5'
outputPath = f'data/relayLoopOrderEdgeMap{SUFFIX}.json'
figurePath = 'figures/relayLoopOrderEdgeMap.png'
for path in (outputPath, figurePath):
    if os.path.exists(path):
        raise SystemExit(f'{path} exists; not overwriting')

TREES, SHUFFLES, BOOTSTRAPS, MIN_PREVALENCE = 200, 10, 200, 0.08
page = json.load(open(f'data/relayLoopFullNets{SUFFIX}.json'))
sweep = json.load(open(f'data/relayLoopSweepNets{SUFFIX}.json'))
assert page['pairs'] == sweep['pairs'] and page['phases'] == sweep['phases']
pairs, phaseNames = [tuple(p) for p in sweep['pairs']], sweep['phases']
trainedCoefficients = np.asarray(json.load(open(f'data/boundaryHarmonicRingCodeFiveLevel{SUFFIX}.json'))['trainedCoefficients'])
basis = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(4)))
foldedAngle = np.abs(np.angle(np.exp(1j * boundary.ringAngles(boundary.boundaryRingCells))))


def regionBins(count):
    return np.digitize(foldedAngle, np.linspace(0, np.pi, count + 1)[1:-1])


def load(nets):
    keys = sorted(nets['codes'])
    multipliers = np.array([nets['codes'][k]['multipliers'] for k in keys])
    transfer = np.array([nets['codes'][k]['field'] for k in keys])
    return keys, multipliers, transfer, np.array([nets['codes'][k]['gap'] for k in keys]), np.array([nets['codes'][k]['faceOverlap'] for k in keys])


def regionMeans(multipliers, count):
    values = np.clip(basis @ (multipliers * trainedCoefficients).T, 0.0, 2.0).T                      # code, ring cell
    bins = regionBins(count)
    return np.array([[row[bins == b].mean() for b in range(count)] for row in values])


def topEdges(transfer):
    """Per code, the set of signed top-3-per-phase edges (phase, sender, receiver)."""
    sets = []
    for code in transfer:
        chosen = set()
        for phase in range(3):
            for index in np.argsort(-np.abs(code[phase]))[:3]:
                a, b = pairs[index]
                chosen.add((phaseNames[phase], a, b) if code[phase, index] > 0 else (phaseNames[phase], b, a))
        sets.append(chosen)
    return sets


sweepKeys, sweepMultipliers, sweepTransfer, sweepGap, sweepFace = load(sweep)
pageKeys, pageMultipliers, pageTransfer, pageGap, pageFace = load(page)
sweepTop, pageTop = topEdges(sweepTransfer), topEdges(pageTransfer)
trained = pageKeys.index('trained')
inputs = {'ring-region means (4)': (regionMeans(sweepMultipliers, 4), regionMeans(pageMultipliers, 4)),
          'multipliers (4)': (sweepMultipliers, pageMultipliers),
          'ring-region means (6)': (regionMeans(sweepMultipliers, 6), regionMeans(pageMultipliers, 6))}
blocks = KMeans(8, n_init=10, random_state=0).fit_predict(sweepMultipliers)
randomFolds, blockedFolds = StratifiedKFold(10, shuffle=True, random_state=0), GroupKFold(8)
WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
forest = lambda: RandomForestClassifier(TREES, min_samples_leaf=3, random_state=0, n_jobs=WORKERS)


def blockedAuc(label, features, groups=blocks):
    predicted = cross_val_predict(forest(), features, label, cv=blockedFolds, groups=groups, method='predict_proba')[:, 1]
    return roc_auc_score(label, predicted)


# ---------------------------------------------------------------- which edges are worth modelling
counts = {}
for chosen in sweepTop:
    for edge in chosen:
        counts[edge] = counts.get(edge, 0) + 1
edges = sorted(e for e, c in counts.items() if MIN_PREVALENCE <= c / len(sweepTop) <= 1 - MIN_PREVALENCE)
print(f'{len(sweepKeys)} sweep codes, {len(pageKeys)} page codes; {len(edges)} signed edges in {MIN_PREVALENCE:.0%}..{1 - MIN_PREVALENCE:.0%} of sweep top-3 sets')
generator = np.random.default_rng(0)
regionsSweep, regionsPage = inputs['ring-region means (4)']
scaler = StandardScaler().fit(regionsSweep)
loadings = np.array([[basis[regionBins(4) == r, o].mean() for o in range(4)] for r in range(4)])         # region, order
scoresByEdge = []
for edge in edges:
    label = np.array([edge in s for s in sweepTop]).astype(int)
    pageLabel = np.array([edge in s for s in pageTop]).astype(int)
    row = dict(edge=' '.join([edge[0], f'{edge[1]}->{edge[2]}']), prevalence=float(label.mean()), pagePrevalence=float(pageLabel.mean()))
    for name, (training, external) in inputs.items():
        row[f'blockedAuc {name}'] = float(blockedAuc(label, training))
        row[f'randomAuc {name}'] = float(roc_auc_score(label, cross_val_predict(forest(), training, label, cv=randomFolds, method='predict_proba')[:, 1]))
        if 5 <= pageLabel.sum() <= len(pageLabel) - 5:
            row[f'pageAuc {name}'] = float(roc_auc_score(pageLabel, forest().fit(training, label).predict_proba(external)[:, 1]))
    nulls = [blockedAuc(generator.permutation(label), regionsSweep) for _ in range(SHUFFLES)]
    row['shuffledNullMean'], row['shuffledNullMax'] = float(np.mean(nulls)), float(np.max(nulls))
    scoresByEdge.append(row)
    print(f"  {row['edge']:34s} prevalence {row['prevalence']:.2f}  blocked AUC regions {row['blockedAuc ring-region means (4)']:.2f} | multipliers "
          f"{row['blockedAuc multipliers (4)']:.2f} | 6 regions {row['blockedAuc ring-region means (6)']:.2f} ; page {row.get('pageAuc ring-region means (4)', float('nan')):.2f} ; null max {row['shuffledNullMax']:.2f}", flush=True)

# ---------------------------------------------------------------- the signed map for the edges that are learnable
learnable = [r for r in scoresByEdge if r['blockedAuc ring-region means (4)'] >= 0.70 and r['blockedAuc ring-region means (4)'] > r['shuffledNullMax'] + 0.05]
print(f'{len(learnable)} of {len(scoresByEdge)} edges are learnable by blocked AUC >= 0.70 and 0.05 above the shuffled maximum')
standard = scaler.transform(regionsSweep)
spread = scaler.scale_
orderMap = []
for row in learnable:
    edge = next(e for e in edges if ' '.join([e[0], f'{e[1]}->{e[2]}']) == row['edge'])
    label = np.array([edge in s for s in sweepTop]).astype(int)
    fit = LogisticRegression(C=0.5, max_iter=1000).fit(standard, label)
    regionEffect = fit.coef_[0]                                                                   # logit per +1 standard deviation of each region
    perOrder = lambda coef: ((coef / spread)[:, None] * loadings * trainedCoefficients[None, :] * 0.1).sum(0)   # logit per +0.1 of each order's multiplier
    draws = []
    for _ in range(BOOTSTRAPS):
        pick = generator.integers(0, len(label), len(label))
        if label[pick].min() == label[pick].max():
            continue
        draws.append(LogisticRegression(C=0.5, max_iter=1000).fit(standard[pick], label[pick]).coef_[0])
    draws = np.array(draws)
    orderDraws = np.array([perOrder(d) for d in draws])
    orderMap.append(dict(edge=row['edge'], blockedAuc=row['blockedAuc ring-region means (4)'], pageAuc=row.get('pageAuc ring-region means (4)'),
                         logisticAuc=float(roc_auc_score(label, fit.predict_proba(standard)[:, 1])),
                         regionLogitPerSd=regionEffect.round(3).tolist(), regionSignStability=[float((np.sign(draws[:, r]) == np.sign(regionEffect[r])).mean()) for r in range(4)],
                         orderLogitPerTenthMultiplier=perOrder(regionEffect).round(3).tolist(),
                         orderSignStability=[float((np.sign(orderDraws[:, o]) == np.sign(perOrder(regionEffect)[o])).mean()) for o in range(4)]))

# ---------------------------------------------------------------- how local is the trained net? and what the outcomes depend on
localMask = np.array([k.startswith('sweepLocal') for k in sweepKeys])
radius = np.array([sweep['codes'][k].get('radius', np.nan) for k in sweepKeys])
trainedTop = pageTop[trained]
retention = np.array([len(s & trainedTop) / len(s | trainedTop) for s in sweepTop])
local = [dict(radius=float(r), codes=int((radius == r).sum()), meanTopThreeRetention=float(retention[radius == r].mean()),
              meanGap=float(sweepGap[radius == r].mean()), faceAtLeast0_5=int((sweepFace[radius == r] >= 0.5).sum()), faceAtLeast0_3=int((sweepFace[radius == r] >= 0.3).sum()))
         for r in sorted(set(radius[localMask]))]
outcomes = {}
for name, target in (('gap', sweepGap), ('faceOverlap', sweepFace)):
    predicted = cross_val_predict(RandomForestRegressor(TREES, min_samples_leaf=3, random_state=0, n_jobs=WORKERS), regionsSweep, target, cv=blockedFolds, groups=blocks)
    outcomes[name] = dict(blockedRSquared=float(1 - ((target - predicted) ** 2).sum() / ((target - target.mean()) ** 2).sum()),
                          globalCodesMean=float(target[~localMask].mean()))
faces = sweepFace >= 0.5
outcomes['facesInSweep'] = dict(atLeast0_5=int(faces.sum()), atLeast0_3=int((sweepFace >= 0.3).sum()), codes=len(sweepFace),
                                multipliersOfFaces=sweepMultipliers[faces].round(3).tolist())
conductance = dict(lowest=min(c['gpolMin'] for c in sweep['codes'].values()), highest=max(c['gpolMax'] for c in sweep['codes'].values()))

result = dict(status='EXPLORATORY: no predictions or decision criteria were registered; see the module docstring', sweepCodes=len(sweepKeys), pageCodes=len(pageKeys),
              edgeScores=scoresByEdge, learnableEdges=len(learnable), orderMap=orderMap, local=local, outcomes=outcomes, gpolRangeInSweep=conductance,
              regionLoadingsOfOrders=dict(regions=['top', 'upper sides', 'lower sides', 'bottom'], loadings=loadings.round(3).tolist()))
json.dump(result, open(outputPath, 'w'), indent=1)
print(f'wrote {outputPath}')

# ---------------------------------------------------------------- the map as a figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if orderMap:
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(orderMap) + 1.8), sharey=True)
    for axis, (title, key, stability, labels) in zip(axes, (('ring region', 'regionLogitPerSd', 'regionSignStability', ['top', 'upper sides', 'lower sides', 'bottom']),
                                                           ('order (per +0.1 of its multiplier)', 'orderLogitPerTenthMultiplier', 'orderSignStability', ['order 0', 'order 1', 'order 2', 'order 3']))):
        values = np.array([m[key] for m in orderMap]); sure = np.array([m[stability] for m in orderMap]) >= 0.9
        limit = np.abs(values).max()
        axis.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                axis.text(j, i, f'{values[i, j]:+.2f}' + ('' if sure[i, j] else '?'), ha='center', va='center', fontsize=8)
        axis.set_xticks(range(len(labels)), labels, fontsize=9); axis.set_title(f'effect on the edge being in the top 3, by {title}\n(logit; ? = sign not stable in 90% of bootstraps)', fontsize=9)
    axes[0].set_yticks(range(len(orderMap)), [f"{m['edge']}  (AUC {m['blockedAuc']:.2f})" for m in orderMap], fontsize=8)
    fig.tight_layout(); fig.savefig(figurePath, dpi=150)
    print(f'wrote {figurePath}')
