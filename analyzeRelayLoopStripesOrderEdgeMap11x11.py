"""EXPLORATORY: learn which edges of the stripe's causal net each region of the ring switches on, and from that an order -> edge map. The stripes'
counterpart of analyzeRelayLoopOrderEdgeMap11x11.py (which see for the method); nothing here was predicted or registered beforehand.

Trained on the sweep (buildBoundaryHarmonicRingCodeStripeSweep11x11.py: 160 space-filling codes over the coefficient box, 120 near the stripe code and
120 spread over the window the stripe forms in, every ring value inside [0, 2] G_pol / G_ref; merged by mergeRelayLoopSweep11x11.py --target
stripesInterior), checked three ways and always against a shuffled-label null: random folds (flattering), blocked folds (a whole block of coefficient
space held out: does it extrapolate?) and the page (fit on the sweep, scored on the page's own codes: the slider and grid lattices, the knockouts and the
curated codes).

Target: whether a signed edge (phase, sender -> receiver) is among the three biggest field transfers of its phase, the page's top-3 view. Input: the
ring's value in four regions (mean over the cells at folded angles 0-45, 45-90, 90-135 and 135-180 degrees from the top: top, upper sides, lower
sides, bottom), which is how the orders act on the tissue; the three coefficients and six regions are compared. A random forest gives the predictive
scores; an L2 logistic regression on standardised regions gives the signed map, with bootstrap sign stability. The order map composes the region map
with the orders' loadings on the regions (the clipped-cosine basis), per +0.1 of an order's coefficient.

Also written, because it bears on whether the net changes smoothly: how much of the stripe code's top-three set survives at each distance from it
(the sweep's local cloud), and how well the region levels predict the outcomes (the stripe's overlap at 504 and the selectivity gap).

Writes data/relayLoopStripesOrderEdgeMap<suffix>.json (never overwriting). The report draws it; there is no separate figure.

    python3 analyzeRelayLoopStripesOrderEdgeMap11x11.py
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
SUFFIX = '1888Hold301StripesInteriorMinus60Minus5'
outputPath = f'data/relayLoopStripesOrderEdgeMap{SUFFIX}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

TREES, SHUFFLES, BOOTSTRAPS, MIN_PREVALENCE = 200, 10, 200, 0.08
placement = json.load(open(f'data/boundaryHarmonicRingCodeStripeLevels{SUFFIX}.json'))
sweep = json.load(open(f'data/relayLoopSweepNets{SUFFIX}.json'))
page = dict(pairs=sweep['pairs'], phases=sweep['phases'], codes={})
for variant in placement['variants']:                              # the page's own codes, from their records
    record = json.load(open(f'data/relayLoopStripes/{variant["key"]}.json'))
    page['codes'][variant['key']] = {f: record[f] for f in ('multipliers', 'field', 'gap', 'faceOverlap', 'upperDark', 'lowerDark')}
pairs, phaseNames = [tuple(p) for p in sweep['pairs']], sweep['phases'][:2]       # flood and clear: the stripe has no write phase
trainedCoefficients = np.asarray(placement['trainedCoefficients'])
NUM_ORDERS = len(trainedCoefficients)
basis = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(NUM_ORDERS)))
foldedAngle = np.abs(np.angle(np.exp(1j * boundary.ringAngles(boundary.boundaryRingCells))))


def regionBins(count):
    return np.digitize(foldedAngle, np.linspace(0, np.pi, count + 1)[1:-1])


def load(nets):
    keys = sorted(nets['codes'])
    multipliers = np.array([nets['codes'][k]['multipliers'] for k in keys])             # the coefficients themselves, for the stripes
    transfer = np.array([nets['codes'][k]['field'] for k in keys])
    return keys, multipliers, transfer, np.array([nets['codes'][k]['gap'] for k in keys]), np.array([nets['codes'][k]['faceOverlap'] for k in keys])


def regionMeans(multipliers, count):
    values = np.clip(basis @ multipliers.T, 0.0, 2.0).T                                              # code, ring cell
    bins = regionBins(count)
    return np.array([[row[bins == b].mean() for b in range(count)] for row in values])


def topEdges(transfer):
    """Per code, the set of signed top-3-per-phase edges (phase, sender, receiver)."""
    sets = []
    for code in transfer:
        chosen = set()
        for phase in range(len(phaseNames)):
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
          'coefficients (3)': (sweepMultipliers, pageMultipliers),
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
loadings = np.array([[basis[regionBins(4) == r, o].mean() for o in range(NUM_ORDERS)] for r in range(4)])   # region, order
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
    print(f"  {row['edge']:34s} prevalence {row['prevalence']:.2f}  blocked AUC regions {row['blockedAuc ring-region means (4)']:.2f} | coefficients "
          f"{row['blockedAuc coefficients (3)']:.2f} | 6 regions {row['blockedAuc ring-region means (6)']:.2f} ; page {row.get('pageAuc ring-region means (4)', float('nan')):.2f} ; null max {row['shuffledNullMax']:.2f}", flush=True)

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
    perOrder = lambda coef: ((coef / spread)[:, None] * loadings * 0.1).sum(0)                     # logit per +0.1 of each order's coefficient
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
                         orderLogitPerTenthCoefficient=perOrder(regionEffect).round(3).tolist(),
                         orderSignStability=[float((np.sign(orderDraws[:, o]) == np.sign(perOrder(regionEffect)[o])).mean()) for o in range(NUM_ORDERS)]))

# ---------------------------------------------------------------- how local is the trained net? and what the outcomes depend on
localMask = np.array([k.startswith('sweepLocal') for k in sweepKeys])
radius = np.array([sweep['codes'][k].get('radius', np.nan) for k in sweepKeys])
trainedTop = pageTop[trained]
retention = np.array([len(s & trainedTop) / len(s | trainedTop) for s in sweepTop])
local = [dict(radius=float(r), codes=int((radius == r).sum()), meanTopThreeRetention=float(retention[radius == r].mean()),
              meanGap=float(sweepGap[radius == r].mean()), stripeAtLeast0_9=int((sweepFace[radius == r] >= 0.9).sum()), stripeAtLeast0_5=int((sweepFace[radius == r] >= 0.5).sum()))
         for r in sorted(set(radius[localMask]))]
outcomes = {}
for name, target in (('gap', sweepGap), ('stripeOverlap', sweepFace)):
    predicted = cross_val_predict(RandomForestRegressor(TREES, min_samples_leaf=3, random_state=0, n_jobs=WORKERS), regionsSweep, target, cv=blockedFolds, groups=blocks)
    outcomes[name] = dict(blockedRSquared=float(1 - ((target - predicted) ** 2).sum() / ((target - target.mean()) ** 2).sum()),
                          globalCodesMean=float(target[~localMask].mean()))
formed = sweepFace >= 0.9
outcomes['stripesInSweep'] = dict(atLeast0_9=int(formed.sum()), atLeast0_5=int((sweepFace >= 0.5).sum()), codes=len(sweepFace),
                                  coefficientsOfStripes=sweepMultipliers[formed].round(3).tolist())
conductance = dict(lowest=min(c['gpolMin'] for c in sweep['codes'].values()), highest=max(c['gpolMax'] for c in sweep['codes'].values()))

result = dict(status='EXPLORATORY: no predictions or decision criteria were registered; see the module docstring', sweepCodes=len(sweepKeys), pageCodes=len(pageKeys),
              edgeScores=scoresByEdge, learnableEdges=len(learnable), orderMap=orderMap, local=local, outcomes=outcomes, gpolRangeInSweep=conductance,
              regionLoadingsOfOrders=dict(regions=['top', 'upper sides', 'lower sides', 'bottom'], loadings=loadings.round(3).tolist()), orders=NUM_ORDERS)
json.dump(result, open(outputPath, 'w'), indent=1)
print(f'wrote {outputPath}')

