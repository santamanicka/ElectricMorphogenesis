"""Scores the second rejuvenation pass against the predictions registered in data/canalizationTalkRejuvenationExtensionPredictions1888Hold301.json (committed before any of these runs).

Reads the per-variant score files written by analyzeCanalizationTalkRejuvenation11x11.py:
    A  --offset 100 | 600 | 3000                                   (all three targets)
    B  --directorySuffix Population64 --labels FaceOrder4
    C  --holdIterations 1000 --directorySuffix LongHold1000
    D  --directorySuffix Order4Family --stripeOrder 4 --labels Stripe
and the first pass (offset 300), then evaluates V2 and S1 to S9 and prints one table.

    python3 summarizeCanalizationTalkRejuvenationExtension11x11.py

Writes data/canalizationTalkRejuvenationExtensionScores1888Hold301.json (never overwriting).
"""
import json
import os

outputPath = 'data/canalizationTalkRejuvenationExtensionScores1888Hold301.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
ALL = 'StripeFaceOrder3FaceOrder4'
paths = {'first pass +300': 'data/canalizationTalkRejuvenationScores1888Hold301.json',
         'A +100': f'data/canalizationTalkRejuvenationScores1888Hold301After100{ALL}.json',
         'A +600': f'data/canalizationTalkRejuvenationScores1888Hold301After600{ALL}.json',
         'A +3000': f'data/canalizationTalkRejuvenationScores1888Hold301After3000{ALL}.json',
         'B population 64': 'data/canalizationTalkRejuvenationScores1888Hold301After300Population64FaceOrder4.json',
         'C hold 1000': f'data/canalizationTalkRejuvenationScores1888Hold301After300LongHold1000{ALL}.json',
         'D stripe orders 0-4': 'data/canalizationTalkRejuvenationScores1888Hold301After300Order4FamilyStripe.json'}
runs = {name: json.load(open(path)) for name, path in paths.items()}
first = runs['first pass +300']['targets']


def highestOverlap(entry):
    return max(r['overlap'] for r in entry['restarts'])


rows = []
for name, run in runs.items():
    for label, t in run['targets'].items():
        rows.append(dict(variant=name, target=label, restarts=t['restartsFinished'], bestScore=t['bestAgedScore'], overlapAtBestScore=t['bestAgedOverlap'], highestOverlap=highestOverlap(t),
                         original=t['originalOnAgedState'], randomBest=t['random64']['bestScore'], G1=t['G1'], G2=t['G2'], restartsReaching0p9=t['restartsReachingOverlap0p9']))

P, observed = {}, {}
# V2: padded family reproduces the first pass's original code on the same state; V1 for every aged state of A (stored first-training score reproduced from the uniform start)
d = runs['D stripe orders 0-4']['targets']['Stripe']['originalOnAgedState']
f = first['Stripe']['originalOnAgedState']
v2a = abs(d['score'] - f['score']) <= 0.01 and d['bestIteration'] == f['bestIteration']
v1 = {name: {k: t['V1']['holds'] for k, t in run['targets'].items()} for name, run in runs.items() if name != 'C hold 1000'}
V2 = dict(paddedOriginalReproducesFirstPass=bool(v2a), padded=d, firstPass=f, V1=v1, holds=bool(v2a and all(all(v.values()) for v in v1.values())))
a = {off: runs[f'A +{off}']['targets'] for off in (100, 600, 3000)}
targets = ('Stripe', 'FaceOrder3', 'FaceOrder4')
P['S1'] = dict(holds=all(a[3000][k]['bestAgedScore'] - a[100][k]['bestAgedScore'] >= 3.0 for k in targets),
               observed={k: dict(at100=a[100][k]['bestAgedScore'], at300=first[k]['bestAgedScore'], at600=a[600][k]['bestAgedScore'], at3000=a[3000][k]['bestAgedScore']) for k in targets})
P['S2'] = dict(holds=not any(a[o][k]['G1'] for o in (600, 3000) for k in targets), highestOverlap={o: {k: highestOverlap(a[o][k]) for k in targets} for o in (600, 3000)})
P['S3'] = dict(holds=any(a[100][k]['G1'] for k in targets), G1={k: a[100][k]['G1'] for k in targets}, highestOverlap={k: highestOverlap(a[100][k]) for k in targets})
b = runs['B population 64']['targets']['FaceOrder4']
P['S4'] = dict(holds=not b['G1'], highestOverlap=highestOverlap(b))
P['S5'] = dict(holds=bool(b['bestAgedScore'] <= first['FaceOrder4']['bestAgedScore'] - 1.0 or highestOverlap(b) > highestOverlap(first['FaceOrder4'])),
               observed=dict(bestScore=b['bestAgedScore'], firstPassBestScore=first['FaceOrder4']['bestAgedScore'], highestOverlap=highestOverlap(b), firstPassHighestOverlap=highestOverlap(first['FaceOrder4'])))
c = runs['C hold 1000']['targets']
P['S6'] = dict(holds=not any(c[k]['G1'] for k in targets), highestOverlap={k: highestOverlap(c[k]) for k in targets})
P['S7'] = dict(holds=all(c[k]['bestAgedScore'] >= first[k]['bestAgedScore'] - 2.0 for k in targets), observed={k: dict(hold1000=c[k]['bestAgedScore'], hold301=first[k]['bestAgedScore']) for k in targets})
dd = runs['D stripe orders 0-4']['targets']['Stripe']
P['S8'] = dict(holds=not dd['G1'], highestOverlap=highestOverlap(dd))
P['S9'] = dict(holds=bool(dd['bestAgedScore'] <= first['Stripe']['bestAgedScore'] - 1.0), observed=dict(orders0to4=dd['bestAgedScore'], orders0to2=first['Stripe']['bestAgedScore']))
results = dict(note='scored against data/canalizationTalkRejuvenationExtensionPredictions1888Hold301.json', V2=V2, predictions=P, summary={k: v['holds'] for k, v in P.items()}, rows=rows)
json.dump(results, open(outputPath, 'w'), indent=1)

print('V2 holds:', V2['holds'], '(padded original reproduces first pass:', v2a, ')')
print(f'\n{"variant":22}{"target":12}{"n":>3}{"best mV":>9}{"overlap":>9}{"highest":>9}{"original":>10}{"(overlap)":>10}{"random":>8}{">=0.9":>6}  G1  G2')
for r in rows:
    print(f'{r["variant"]:22}{r["target"]:12}{r["restarts"]:3d}{r["bestScore"]:9.2f}{r["overlapAtBestScore"]:9.2f}{r["highestOverlap"]:9.2f}{r["original"]["score"]:10.2f}{r["original"]["overlap"]:10.2f}'
          f'{r["randomBest"]:8.2f}{r["restartsReaching0p9"]:6d}  {r["G1"]!s:5}{r["G2"]}')
print('\npredictions:', results['summary'])
print('wrote', outputPath)
