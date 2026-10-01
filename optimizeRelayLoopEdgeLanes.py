"""Chooses each edge's curve in the Relay Loop page, once, for every ring code.

An edge's identity, for comparing ring codes by eye, is its unordered node pair and its phase. This gives every such
(pair, phase) one fixed lane -- how far and to which side it bows, in the pair's canonical direction (node order in
the page) -- so the same pair in the same phase is the same curve in every panel, whichever code, lens or direction.
The lanes are picked by coordinate descent to keep edges apart in every view the page can show: the top-3-per-phase
edges of each of the trained, curated, slider and grid codes, and the trained code's eleven tracked pairs read off each
of them, mirror twins included, drawn at the thumbnail stroke widths (the strictest case; the tracked lens' log-scale widths, the top-3 lens' linear ones). A view is penalised where
the middle 60% of two edges run within their stroke widths of each other, or where an edge passes over a node it does
not end on. Candidates are a fixed ladder of lanes; a small cost keeps lanes near the phase default so the result
stays a readable rule ("clear bows one way, write the other, flood between") with exceptions only where the
geometry forces them.

Writes data/relayLoopEdgeLanes.json (never overwriting); buildRelayLoopArtifact.py splices it into the page.

    python3 optimizeRelayLoopEdgeLanes.py
"""
import functools
import json
import math
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

outputPath = 'data/relayLoopEdgeLanes.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

# ---- the page's own geometry (figures/relayLoopTemplate.html) ----
M, CM = 80, 42
P = lambda cx, cy: (M + cx * CM, M + cy * CM)
NODES = dict(ringTop=P(5.5, 0.5), ringBottom=P(5.5, 10.5), ringLeft=P(0.5, 5.5), ringRight=P(10.5, 5.5), eyes=P(5.5, 3.0),
             nose=P(5.5, 5.5), mouth=P(5.5, 8.5), bgTL=P(1.5, 1.5), bgTR=P(9.5, 1.5), bgBL=P(1.5, 9.5), bgBR=P(9.5, 9.5))
NODE_ORDER = list(NODES)
MIRROR = dict(ringLeft='ringRight', ringRight='ringLeft', bgTL='bgTR', bgTR='bgTL', bgBL='bgBR', bgBR='bgBL')
PHASES = ['flood', 'clear', 'write']
DEFAULT_LANE = dict(flood=-0.9, clear=0.9, write=-2.2)
LADDER = [-2.2, -1.6, -1.0, -0.6, 0.0, 0.6, 1.0, 1.6, 2.2]
THUMBNAIL_WIDTH = 2.3                       # the grid thumbnails draw strokes this much thicker than the main diagram, in the same units
template = open('figures/relayLoopTemplate.html').read()
trainedEdges = [(m.group(1), m.group(2), m.group(3), float(m.group(4))) for m in
                re.finditer(r"\{from:'(\w+)', to:'(\w+)', phase:'(\w+)', value:([\d.]+)", template)]
assert len(trainedEdges) == 11, len(trainedEdges)
TRAINED_MAX = max(v for *_, v in trainedEdges)
SCALE_CAP = 3
TRACKED_FLOOR = TRAINED_MAX * 0.01          # the tracked lens draws width on a log scale from here to the cap (the page's normOf)


def widthOf(value, lens):
    if lens == 'tracked':
        low, high = math.log(TRACKED_FLOOR), math.log(TRAINED_MAX * SCALE_CAP)
        norm = min(1.0, max(0.0, (math.log(max(value, 1e-12)) - low) / (high - low)))
    else:
        norm = min(1.0, value / (TRAINED_MAX * SCALE_CAP))
    return THUMBNAIL_WIDTH * (1.6 + 8.4 * norm)
pairKey = lambda a, b: '|'.join(sorted((a, b), key=NODE_ORDER.index))


@functools.lru_cache(maxsize=None)
def arcPoints(sender, receiver, lane):
    """25 points along the middle 60% of the page's quadratic arc, and its end points (computePath)."""
    ax, ay = NODES[sender]
    bx, by = NODES[receiver]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    bow = lane * min(40.0, 14.0 + length * 0.12)
    sx, sy, ex, ey = ax + ux * 21, ay + uy * 21, bx - ux * 26, by - uy * 26
    mx, my = (sx + ex) / 2 + (-uy) * bow, (sy + ey) / 2 + ux * bow
    t = np.linspace(0.2, 0.8, 25)[:, None]
    points = (1 - t) ** 2 * np.array([sx, sy]) + 2 * (1 - t) * t * np.array([mx, my]) + t ** 2 * np.array([ex, ey])
    return points, (sx, sy), (ex, ey)


# ---- every view's edges: (sender, receiver, phase, value) on the canonical side ----
with tempfile.TemporaryDirectory() as directory:
    subprocess.run([sys.executable, 'assembleRelayLoopFiveLevelData.py', '--outputDirectory', directory], check=True,
                   stdout=subprocess.DEVNULL)
    load = lambda name: json.load(open(f'{directory}/{name}.json'))
    slider, grid, tracked, variants, trainedTop3 = (load(n) for n in ('sliderData', 'gridData', 'trackedData', 'variantData', 'trainedTop3Phases'))
phasesOfKey = {'trained': trainedTop3}
phasesOfKey.update({k: v['phases'] for k, v in variants.items()})
for s in slider.values():
    phasesOfKey.update({p['key']: p['phases'] for p in s['points']})
for g in grid.values():
    phasesOfKey.update({c['key']: c['phases'] for row in g['cells'] for c in row})
views = []
for key, phases in phasesOfKey.items():
    views.append(('top3', [(e['from'], e['to'], p, e['value']) for p, es in phases.items() for e in es]))
    if key == 'trained':
        views.append(('tracked', list(trainedEdges)))
    else:
        views.append(('tracked', [((b, a) if tracked[key][i]['reversed'] else (a, b)) + (phase, tracked[key][i]['value'])
                                  for i, (a, b, phase, _) in enumerate(trainedEdges)]))
variables = sorted({(pairKey(a, b), p) for _, view in views for a, b, p, _ in view})
print(f'{len(views)} views, {len(variables)} (pair, phase) lanes to choose')
viewsOf = {var: [i for i, (_, view) in enumerate(views) if any((pairKey(a, b), p) == var for a, b, p, _ in view)] for var in variables}


def viewScore(index, lanes):
    arcs = []
    lens, edges = views[index]
    for a, b, phase, value in edges:
        lane = lanes[(pairKey(a, b), phase)] * (1 if NODE_ORDER.index(a) < NODE_ORDER.index(b) else -1)
        for sender, receiver, signed in [(a, b, lane)] + ([(MIRROR.get(a, a), MIRROR.get(b, b), -lane)] if (a in MIRROR or b in MIRROR) else []):
            points, start, end = arcPoints(sender, receiver, signed)
            arcs.append((points, start, end, widthOf(value, lens)))
    score = 0.0
    for i in range(len(arcs)):
        for j in range(i + 1, len(arcs)):
            gap = (arcs[i][3] + arcs[j][3]) / 2 + 1.5
            distance = np.linalg.norm(arcs[i][0][:, None, :] - arcs[j][0][None, :, :], axis=2)
            fraction = max((distance.min(1) < gap).mean(), (distance.min(0) < gap).mean())
            score += max(0.0, fraction - 0.1) + (1.0 if fraction > 0.25 else 0.0)
    for points, start, end, _ in arcs:
        for node in NODES.values():
            if math.hypot(node[0] - start[0], node[1] - start[1]) < 30 or math.hypot(node[0] - end[0], node[1] - end[1]) < 30:
                continue
            if np.linalg.norm(points - np.array(node), axis=1).min() < 17:
                score += 3.0
    return score


lanes = {var: DEFAULT_LANE[var[1]] for var in variables}
cache = [viewScore(i, lanes) for i in range(len(views))]
regularise = lambda var, lane: 0.05 * abs(lane - DEFAULT_LANE[var[1]])
total = lambda: sum(cache) + sum(regularise(v, l) for v, l in lanes.items())
print(f'default lanes: score {total():.1f}, {sum(s >= 1 for s in cache)} views with a clash')
for sweep in range(8):
    moved = 0
    for var in variables:
        start = lanes[var]
        best, bestLane, bestScores = None, start, None
        for lane in LADDER:
            lanes[var] = lane
            scores = {i: viewScore(i, lanes) for i in viewsOf[var]}
            cost = sum(scores.values()) + regularise(var, lane)
            if best is None or cost < best - 1e-9:
                best, bestLane, bestScores = cost, lane, scores
        lanes[var] = bestLane
        for i, score in bestScores.items():
            cache[i] = score
        moved += bestLane != start
    print(f'sweep {sweep + 1}: score {total():.1f}, {sum(s >= 1 for s in cache)} views with a clash, '
          f'{sum(lanes[v] != DEFAULT_LANE[v[1]] for v in variables)} lanes off the phase default, {moved} moved', flush=True)
    if not moved:
        break

table = {}
for (pair, phase), lane in sorted(lanes.items()):
    table.setdefault(pair, {})[phase] = lane
json.dump(dict(default=DEFAULT_LANE, lanes=table), open(outputPath, 'w'), indent=1)
print(f'wrote {outputPath}: {len(table)} pairs')
