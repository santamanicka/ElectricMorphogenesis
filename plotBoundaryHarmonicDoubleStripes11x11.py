"""Builds the double-stripe report page, figures/boundaryHarmonicDoubleStripes.html, from its template and the data files.

    python3 plotBoundaryHarmonicDoubleStripes11x11.py [--overwrite]

Edit figures/boundaryHarmonicDoubleStripesTemplate.html, not the output. The template's {{PLACEHOLDERS}} are filled from
  - boundaryCodeUtilities (the target's cells, drawn as 11 x 11 grids),
  - data/boundaryHarmonicDoubleStripesRestartTable<suffix>.json (every saved restart's best moment, written by
    tabulateBoundaryHarmonicDoubleStripesRestarts11x11.py): the pattern galleries, the stage tables and the overlap chart,
  - data/boundaryHarmonicTrainingSummary<suffix>*.json (the random-code controls, the long replay of the best code),
  - data/boundaryHarmonicDoubleStripesBumpPatterns<suffix>.json (the two-bump screen's best maps),
  - data/boundaryHarmonicTrainingPredictions<suffix>.json (the registered question and predictions, verbatim),
  - data/boundaryHarmonicDoubleStripesTrainingJobs<suffix>.json (the first submitted training stages).
Sections of prose (the findings, the storyline) are written into the template by hand. Nothing is simulated or rescored here.
Like every script here it refuses to overwrite its output unless --overwrite is given.
"""
import argparse
import collections
import datetime
import html
import json
import os
import random

import numpy as np

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301DoubleStripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--templatePath', type=str, default='figures/boundaryHarmonicDoubleStripesTemplate.html')
parser.add_argument('--outputPath', type=str, default='figures/boundaryHarmonicDoubleStripes.html')
parser.add_argument('--registrationPath', type=str, default=f'data/boundaryHarmonicTrainingPredictions{SUFFIX}.json')
parser.add_argument('--jobsPath', type=str, default=f'data/boundaryHarmonicDoubleStripesTrainingJobs{SUFFIX}.json')
parser.add_argument('--restartTablePath', type=str, default=f'data/boundaryHarmonicDoubleStripesRestartTable{SUFFIX}.json')
parser.add_argument('--bumpPatternsPath', type=str, default=f'data/boundaryHarmonicDoubleStripesBumpPatterns{SUFFIX}.json')
parser.add_argument('--replaySummaryPath', type=str, default=f'data/boundaryHarmonicTrainingSummary{SUFFIX}Ceiling2HigherOrdersPopulation16.json')
parser.add_argument('--controlSummaryPaths', type=str, nargs='+',
                    default=[f'data/boundaryHarmonicTrainingSummary{SUFFIX}Ceiling2Combined.json',
                             f'data/boundaryHarmonicTrainingSummary{SUFFIX}Ceiling1p3Combined.json',
                             f'data/boundaryHarmonicTrainingSummary{SUFFIX}Ceiling2HigherOrdersPopulation16.json',
                             f'data/boundaryHarmonicTrainingSummary{SUFFIX}Ceiling2EvenSet0-2-4.json',
                             f'data/boundaryHarmonicTrainingSummary{SUFFIX}Ceiling2EvenSet0-2-4-6.json'])
parser.add_argument('--numGalleryPatterns', type=int, default=24)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
if os.path.exists(args.outputPath) and not args.overwrite:
    raise SystemExit(f'{args.outputPath} exists; pass --overwrite to rebuild it')

registration = json.load(open(args.registrationPath))
jobs = json.load(open(args.jobsPath))
table = json.load(open(args.restartTablePath))
restarts, stages = table['restarts'], table['stages']
escape = html.escape
THRESHOLD = table['threshold']
FORMED = 0.9
FLOOD_STRAY = 10   # a pattern with this many dark centre-stripe cells is the whole interior dark, not a stripe pair
for row in restarts:
    row['flooded'] = row['strayDark'] >= FLOOD_STRAY
stageTitle = {stage['key']: stage['title'] for stage in stages}
stageCeiling = {stage['key']: next(r['ceiling'] for r in restarts if r['stage'] == stage['key']) for stage in stages}

# ---------------------------------------------------------------------------------------------- drawing helpers
CELL, GAP = 20, 2
PITCH = CELL + GAP
SIDE = boundary.latticeRows * PITCH - GAP


def cellFill(voltage):
    """Light at -5 mV to dark at -60 mV, mixed from the page's own cell colours so both themes work."""
    fraction = min(max((-5.0 - voltage) / 55.0, 0.0), 1.0)
    return f'color-mix(in srgb, var(--cell-dark) {round(fraction * 100)}%, var(--cell-light))'


def flankOutlines():
    """Dashed boxes around the two flank stripes (columns 1-3 and 7-9, rows 1-9)."""
    return ''.join(f'<rect class="flank-outline" x="{column * PITCH - 1}" y="{PITCH - 1}" width="{3 * PITCH}" height="{9 * PITCH}" rx="3"/>'
                   for column in (1, 7))


def mapSvg(voltages, label, outline=True):
    rects = []
    for cell in range(boundary.numCells):
        row, column = divmod(cell, boundary.latticeCols)
        rects.append(f'<rect x="{column * PITCH}" y="{row * PITCH}" width="{CELL}" height="{CELL}" rx="2" style="fill:{cellFill(voltages[cell])}"/>')
    return f'<svg viewBox="0 0 {SIDE} {SIDE}" role="img" aria-label="{escape(label)}">' + ''.join(rects) + (flankOutlines() if outline else '') + '</svg>'


def orderText(label):
    """'order18' -> 'orders 0-18'; 'orders0-2-4' -> 'even orders {0, 2, 4}'."""
    if label.startswith('orders'):
        numbers = label[len('orders'):].split('-')
        return 'even orders {' + (', '.join(numbers) if len(numbers) <= 5 else f'{numbers[0]}, {numbers[1]}, ..., {numbers[-1]}') + '}'
    number = int(label[len('order'):])
    return 'order 0' if number == 0 else f'orders 0-{number}'


def tile(voltages, label, lines, outline=True):
    caption = ''.join(f'<span class="{cls}">{text}</span>' for cls, text in lines)
    return f'<figure class="tile">{mapSvg(voltages, label, outline)}<figcaption>{caption}</figcaption></figure>'


def restartTile(row, extra=''):
    label = f"{stageTitle[row['stage']]}, {orderText(row['orderLabel'])}, overlap {row['overlap']:.2f}"
    lines = [('lead', f"overlap {row['overlap']:.2f}"), ('', f"{row['flankDark']} of 54 flank cells, {row['strayDark']} stray"),
             ('small', f"{stageTitle[row['stage']]}, {orderText(row['orderLabel'])}"), ('small', f"ceiling {row['ceiling']:.1f}, score {row['score']:.1f}")]
    if extra:
        lines.append(('small', extra))
    return tile(row['vmem'], label, lines)


def darkSignature(row):
    return tuple(i for i, v in enumerate(row['vmem']) if v < THRESHOLD)


def betterFirst(row):
    return (-row['overlap'], row['score'])


# ---------------------------------------------------------------------------------------------- colour tokens for charts
# Two series: ceiling 1.3 and ceiling 2.0. Chart text and grid use the page's own tokens.
class Scale:
    def __init__(self, low, high, start, end):
        self.low, self.high, self.start, self.end = low, high, start, end

    def __call__(self, value):
        return self.start + (value - self.low) / (self.high - self.low) * (self.end - self.start)


def seriesClass(ceiling):
    return 'dot-low' if ceiling < 1.5 else 'dot-high'


# ---------------------------------------------------------------------------------------------- the pattern galleries
targetFigures = ''.join(
    f'<figure>{mapSvg(voltages, label, outline)}<figcaption>{escape(label)}: {count} dark cells of the 81 interior cells.</figcaption></figure>'
    for label, cells, outline, count in (('Single stripe (the previous target)', boundary.centreStripeCellIndices, False, len(boundary.centreStripeCellIndices)),
                                         ('Double stripe (this target)', boundary.flankCellIndices, True, len(boundary.flankCellIndices)))
    for voltages in [[-60.0 if cell in set(int(c) for c in cells) else -5.0 for cell in range(boundary.numCells)]])

notFlooded = [row for row in restarts if not row['flooded']]
stageBestTiles = ''
for stage in stages:
    rows = [row for row in notFlooded if row['stage'] == stage['key']]
    stageBestTiles += restartTile(min(rows, key=betterFirst))

groups = collections.defaultdict(list)
for row in notFlooded:
    groups[darkSignature(row)].append(row)
ranked = sorted(groups.values(), key=lambda rows: betterFirst(min(rows, key=betterFirst)))
galleryTiles = ''
for rows in ranked[:args.numGalleryPatterns]:
    best = min(rows, key=betterFirst)
    galleryTiles += restartTile(best, f'also reached by {len(rows) - 1} more' if len(rows) > 1 else '')
numDistinct = len(groups)

floods = [row for row in restarts if row['flooded']]
floodTile = restartTile(min(floods, key=betterFirst), f'reached by {len(floods)} restarts of arm B') if floods else ''

# ---------------------------------------------------------------------------------------------- overlap strip chart (finding 1)
width, height, left, right, top, bottom = 760, 330, 52, 16, 20, 62
xScale = Scale(0, len(stages), left, width - right)
yScale = Scale(0.0, 1.0, height - bottom, top)
parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Double-stripe overlap of every restart, by stage, against the formation line at 0.9">']
for tick in (0, 0.25, 0.5, 0.75, 1.0):
    parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{yScale(tick):.1f}" y2="{yScale(tick):.1f}"/>'
                 f'<text class="tick" x="{left - 8}" y="{yScale(tick) + 4:.1f}" text-anchor="end">{tick:g}</text>')
parts.append(f'<line class="gate" x1="{left}" x2="{width - right}" y1="{yScale(FORMED):.1f}" y2="{yScale(FORMED):.1f}"/>'
             f'<text class="note" x="{width - right}" y="{yScale(FORMED) - 6:.1f}" text-anchor="end">formed: overlap 0.9 or more</text>')
parts.append(f'<text class="axis-title" transform="translate(13 {(top + height - bottom) / 2:.0f}) rotate(-90)" text-anchor="middle">overlap with the 54 flank cells</text>')
for index, stage in enumerate(stages):
    rows = [row for row in restarts if row['stage'] == stage['key']]
    centre = xScale(index + 0.5)
    generator = random.Random(index)
    for row in sorted(rows, key=lambda r: r['restart']):
        x, y = centre + (generator.random() - 0.5) * 0.7 * (xScale(1) - xScale(0)), yScale(row['overlap'])
        name = f"{stage['title']}, {orderText(row['orderLabel'])}, restart {row['restart']}: overlap {row['overlap']:.2f}, score {row['score']:.2f}"
        if row['flooded']:
            parts.append(f'<circle class="flood {seriesClass(row["ceiling"])}-ring" cx="{x:.1f}" cy="{y:.1f}" r="4.2"><title>{escape(name)}; the whole interior is dark</title></circle>')
        else:
            parts.append(f'<circle class="{seriesClass(row["ceiling"])}" cx="{x:.1f}" cy="{y:.1f}" r="2.7"><title>{escape(name)}</title></circle>')
    median = float(np.median([row['overlap'] for row in rows]))
    parts.append(f'<line class="median" x1="{centre - 20:.1f}" x2="{centre + 20:.1f}" y1="{yScale(median):.1f}" y2="{yScale(median):.1f}"/>')
    parts.append(f'<text class="tick" x="{centre:.1f}" y="{height - bottom + 18}" text-anchor="middle">{escape(stage["title"])}</text>'
                 f'<text class="tick muted" x="{centre:.1f}" y="{height - bottom + 32}" text-anchor="middle">ceiling {stageCeiling[stage["key"]]:g}</text>'
                 f'<text class="tick muted" x="{centre:.1f}" y="{height - bottom + 46}" text-anchor="middle">{len(rows)} restarts</text>')
overallBest = max(notFlooded, key=lambda r: r['overlap'])
bestIndex = [s['key'] for s in stages].index(overallBest['stage'])
parts.append(f'<text class="note" x="{xScale(bestIndex + 0.5) + 26:.1f}" y="{yScale(overallBest["overlap"]) - 8:.1f}">best: {overallBest["overlap"]:.2f}</text>')
floodStage = [s['key'] for s in stages].index(floods[0]['stage']) if floods else 0
if floods:
    parts.append(f'<text class="note" x="{xScale(floodStage + 0.5) + 26:.1f}" y="{yScale(floods[0]["overlap"]) - 7:.1f}">8 floods</text>')
parts.append('</svg>')
overlapChart = ''.join(parts)

# ---------------------------------------------------------------------------------------------- stage tables
def statsRow(rows):
    overlaps = np.array([row['overlap'] for row in rows])
    clean = np.array([row['overlap'] for row in rows if not row['flooded']])
    return dict(count=len(rows), codes=sum(row['numEvaluations'] for row in rows), best=float(clean.max()), median=float(np.median(overlaps)),
                aboveHalf=int(sum(1 for row in rows if row['overlap'] >= 0.5 and not row['flooded'])), flooded=int(sum(row['flooded'] for row in rows)),
                bestScore=float(min(row['score'] for row in rows)), formed=int((overlaps >= FORMED).sum()))


def stageRows():
    out = ''
    for stage in stages:
        rows = [row for row in restarts if row['stage'] == stage['key']]
        s = statsRow(rows)
        out += (f"<tr><td>{escape(stage['title'])}</td><td>{escape(stage['route'])}</td><td class=\"num\">{rows[0]['ceiling']:g}</td><td class=\"num\">{rows[0]['populationSize']}</td>"
                f"<td class=\"num\">{s['count']}</td><td class=\"num\">{s['codes']:,}</td><td class=\"num\">{s['best']:.2f}</td><td class=\"num\">{s['median']:.2f}</td>"
                f"<td class=\"num\">{s['aboveHalf']}</td><td class=\"num\">{s['bestScore']:.2f}</td><td class=\"num\">{s['formed']}</td></tr>")
    total = statsRow(restarts)
    out += (f"<tr class=\"total\"><td>All</td><td></td><td class=\"num\"></td><td class=\"num\"></td><td class=\"num\">{total['count']}</td><td class=\"num\">{total['codes']:,}</td>"
            f"<td class=\"num\">{total['best']:.2f}</td><td class=\"num\">{total['median']:.2f}</td><td class=\"num\">{total['aboveHalf']}</td><td class=\"num\">{total['bestScore']:.2f}</td><td class=\"num\">{total['formed']}</td></tr>")
    return out


stageTable = ('<div class="scroll"><table><thead><tr><th>Stage</th><th>Route</th><th class="num">Ceiling</th><th class="num">Population</th><th class="num">Restarts</th>'
              '<th class="num">Codes simulated</th><th class="num">Best overlap</th><th class="num">Median overlap</th><th class="num">At least 0.5</th>'
              '<th class="num">Best score</th><th class="num">Formed</th></tr></thead>'
              f'<tbody>{stageRows()}</tbody></table></div>')

detailRows = ''
for stage in stages:
    labels = sorted({row['orderLabel'] for row in restarts if row['stage'] == stage['key']},
                    key=lambda text: (text.startswith('orders'), len(text), int(text.replace('orders', '').replace('order', '').split('-')[0] or 0) if not text.startswith('orders') else 0, text))
    for label in labels:
        rows = [row for row in restarts if row['stage'] == stage['key'] and row['orderLabel'] == label]
        s = statsRow(rows)
        detailRows += (f"<tr><td>{escape(stage['title'])}</td><td>{escape(orderText(label))}</td><td class=\"num\">{s['count']}</td><td class=\"num\">{s['best']:.2f}</td>"
                       f"<td class=\"num\">{s['median']:.2f}</td><td class=\"num\">{s['aboveHalf']}</td><td class=\"num\">{s['bestScore']:.2f}</td></tr>")
detailTable = ('<div class="scroll"><table><thead><tr><th>Stage</th><th>Code</th><th class="num">Restarts</th><th class="num">Best overlap</th><th class="num">Median overlap</th>'
               f'<th class="num">At least 0.5</th><th class="num">Best score</th></tr></thead><tbody>{detailRows}</tbody></table></div>')

# ---------------------------------------------------------------------------------------------- reach map (finding 2)
good = [row for row in notFlooded if row['overlap'] >= 0.5]
share = np.zeros(boundary.numCells)
for row in good:
    share += np.array(row['vmem']) < THRESHOLD
share /= len(good)
cell, pitch, margin = 30, 32, 16
mapSide = 9 * pitch - 2
parts = [f'<svg viewBox="0 0 {mapSide + margin} {mapSide + margin}" role="img" aria-label="Share of the {len(good)} good restarts in which each interior cell is dark">']
for index in range(9):
    parts.append(f'<text class="tick muted" x="{margin + index * pitch + cell / 2:.0f}" y="10" text-anchor="middle">{index + 1}</text>'
                 f'<text class="tick muted" x="8" y="{margin + index * pitch + cell / 2 + 4:.0f}" text-anchor="middle">{index + 1}</text>')
for row in range(1, 10):
    for column in range(1, 10):
        value = share[row * boundary.latticeCols + column]
        parts.append(f'<rect x="{margin + (column - 1) * pitch}" y="{margin + (row - 1) * pitch}" width="{cell}" height="{cell}" rx="2" '
                     f'style="fill:color-mix(in srgb, var(--cell-dark) {round(value * 100)}%, var(--cell-light))"/>'
                     f'<text class="{"on-dark" if value > 0.5 else "on-light"}" x="{margin + (column - 1) * pitch + cell / 2}" y="{margin + (row - 1) * pitch + cell / 2 + 4}" text-anchor="middle">{round(value * 100)}</text>')
for column in (0, 6):
    parts.append(f'<rect class="flank-outline" x="{margin + column * pitch - 1}" y="{margin - 1}" width="{3 * pitch}" height="{9 * pitch}" rx="3"/>')
parts.append('</svg>')
reachMap = ''.join(parts)
columnShare = [float(np.mean([share[row * boundary.latticeCols + column] for row in range(1, 10)])) for column in range(1, 10)]

# ---------------------------------------------------------------------------------------------- the best code's replay (finding 2, appendix)
replay = json.load(open(args.replaySummaryPath))['orders']['18']
longRun, bestEntry = replay['longRun'], replay['best']
overlapTrace = np.array(longRun['overlapTrace'])
START, STRIDE = 301, longRun['traceStride']
snapshots = longRun['snapshots']
replayTiles = ''
for letter, snapshot in zip('abcdefg', snapshots):
    voltages = snapshot['vmem']
    dark = {i for i, v in enumerate(voltages) if v < THRESHOLD}
    flankDark = len(dark & set(int(c) for c in boundary.flankCellIndices))
    stray = len((dark & set(int(c) for c in boundary.interiorCellIndices)) - set(int(c) for c in boundary.flankCellIndices))
    offset = snapshot['offset']
    when = 'best moment' if offset == 0 else f"{abs(offset)} iterations {'before' if offset < 0 else 'after'}"
    replayTiles += tile(voltages, f"Best code of rung 3a, iteration {snapshot['iteration']}",
                        [('lead', f'{letter}. iteration {snapshot["iteration"]}'), ('', f'{flankDark} of 54 flank cells, {stray} stray'), ('small', f'{when}, score {snapshot["score"]:.1f}')])

width, height, left, right, top, bottom = 760, 250, 52, 16, 16, 40
xScale, yScale = Scale(START, 3000, left, width - right), Scale(0.0, 1.0, height - bottom, top)
parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Overlap of the best rung-3a code over the scored window">']
for tick in (0, 0.5, 1.0):
    parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{yScale(tick):.1f}" y2="{yScale(tick):.1f}"/><text class="tick" x="{left - 8}" y="{yScale(tick) + 4:.1f}" text-anchor="end">{tick:g}</text>')
for tick in (500, 1000, 1500, 2000, 2500, 3000):
    parts.append(f'<text class="tick" x="{xScale(tick):.1f}" y="{height - bottom + 16}" text-anchor="middle">{tick}</text>')
parts.append(f'<text class="axis-title" x="{(left + width - right) / 2:.0f}" y="{height - 6}" text-anchor="middle">iteration (the ring is released at 301)</text>')
parts.append(f'<text class="axis-title" transform="translate(13 {(top + height - bottom) / 2:.0f}) rotate(-90)" text-anchor="middle">overlap</text>')
parts.append(f'<line class="gate" x1="{left}" x2="{width - right}" y1="{yScale(FORMED):.1f}" y2="{yScale(FORMED):.1f}"/><text class="note" x="{width - right}" y="{yScale(FORMED) - 6:.1f}" text-anchor="end">formed: 0.9</text>')
points = ' '.join(f'{xScale(START + STRIDE * index):.1f},{yScale(value):.1f}' for index, value in enumerate(overlapTrace) if START + STRIDE * index <= 3000)
parts.append(f'<polyline class="line" points="{points}"/>')
for letter, snapshot in zip('abcdefg', snapshots):
    if snapshot['iteration'] <= 3000:
        index = (snapshot['iteration'] - START) // STRIDE
        parts.append(f'<circle class="mark" cx="{xScale(snapshot["iteration"]):.1f}" cy="{yScale(overlapTrace[index]):.1f}" r="3.4"><title>{letter}: iteration {snapshot["iteration"]}, overlap {overlapTrace[index]:.2f}</title></circle>')
parts.append('</svg>')
replayChart = ''.join(parts)
laterMax = float(overlapTrace[(3000 - START) // STRIDE + 1:].max())
windowShare = float((overlapTrace[:(3000 - START) // STRIDE + 1] >= 0.5).mean())

# ---------------------------------------------------------------------------------------------- the best code (appendix)
ringValues = bestEntry['ringValues']
ringCells = [int(c) for c in boundary.boundaryRingCells]
cell, pitch = 22, 24
parts = ['<svg viewBox="0 0 %d %d" role="img" aria-label="The held ring values of the best code">' % (11 * pitch - 2, 11 * pitch - 2)]
for ringIndex, cellIndex in enumerate(ringCells):
    row, column = divmod(cellIndex, boundary.latticeCols)
    value = ringValues[ringIndex]
    parts.append(f'<rect x="{column * pitch}" y="{row * pitch}" width="{cell}" height="{cell}" rx="2" style="fill:color-mix(in srgb, var(--cell-dark) {round(value / 2.0 * 100)}%, var(--cell-light))"><title>ring cell {ringIndex}: {value:.2f}</title></rect>'
                 f'<text class="{"on-dark" if value > 1.0 else "on-light"} tiny" x="{column * pitch + cell / 2}" y="{row * pitch + cell / 2 + 3}" text-anchor="middle">{value:.1f}</text>')
parts.append('</svg>')
ringMap = ''.join(parts)

coefficients = bestEntry['coefficients']
width, left, right, top, bottom = 560, 56, 20, 14, 30
height = top + 20 * len(coefficients) + bottom
valueScale = Scale(-0.5, 1.25, left, width - right)
rowsBottom = top + 20 * len(coefficients)
parts = [f'<svg class="chart narrow" viewBox="0 0 {width} {height}" role="img" aria-label="The cosine coefficients of the best code">']
for tick in (-0.5, 0, 0.5, 1.0):
    parts.append(f'<line class="grid" x1="{valueScale(tick):.1f}" x2="{valueScale(tick):.1f}" y1="{top}" y2="{rowsBottom}"/><text class="tick" x="{valueScale(tick):.1f}" y="{rowsBottom + 16}" text-anchor="middle">{tick:g}</text>')
for order, value in enumerate(coefficients):
    y = top + order * 20
    x0, x1 = sorted((valueScale(0), valueScale(value)))
    parts.append(f'<text class="tick" x="{left - 8}" y="{y + 13}" text-anchor="end">a{order}</text><rect class="bar" x="{x0:.1f}" y="{y + 3}" width="{max(x1 - x0, 1):.1f}" height="12" rx="2"><title>a{order} = {value:.3f}</title></rect>')
parts.append('</svg>')
coefficientChart = ''.join(parts)

# ---------------------------------------------------------------------------------------------- controls (appendix)
def randomControls(path):
    summary = json.load(open(path))
    return {int(size): dict(best=v['randomCodes']['best'], median=v['randomCodes']['median'], maxOverlap=v['randomCodes']['maxStructuralIoU'])
            for size, v in summary['orders'].items()}, summary['ceiling']


random2 = {}
for path in args.controlSummaryPaths:
    controls, ceiling = randomControls(path)
    if 'EvenSet' in path or ceiling < 1.5:
        continue
    random2.update(controls)
sizes = sorted(random2)
trainedBest = {size: min(row['score'] for row in restarts if row['ceiling'] == 2.0 and row['orderLabel'] == f'order{size}') for size in sizes}
width, height, left, right, top, bottom = 760, 300, 52, 16, 16, 44
xScale, yScale = Scale(0, len(sizes), left, width - right), Scale(17.0, 28.0, height - bottom, top)
parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Best trained score against the 64 random codes of each size, ceiling 2.0">']
for tick in (18, 20, 22, 24, 26, 28):
    parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{yScale(tick):.1f}" y2="{yScale(tick):.1f}"/><text class="tick" x="{left - 8}" y="{yScale(tick) + 4:.1f}" text-anchor="end">{tick}</text>')
parts.append(f'<line class="gate" x1="{left}" x2="{width - right}" y1="{yScale(20.0):.1f}" y2="{yScale(20.0):.1f}"/><text class="note" x="{width - right}" y="{yScale(20.0) - 6:.1f}" text-anchor="end">20.0: the whole interior dark</text>')
parts.append(f'<text class="axis-title" transform="translate(13 {(top + height - bottom) / 2:.0f}) rotate(-90)" text-anchor="middle">best score (lower is better)</text>')
parts.append(f'<text class="axis-title" x="{(left + width - right) / 2:.0f}" y="{height - 4}" text-anchor="middle">code size: orders 0 to N</text>')
for index, size in enumerate(sizes):
    x = xScale(index + 0.5)
    parts.append(f'<text class="tick" x="{x:.1f}" y="{height - bottom + 16}" text-anchor="middle">{size}</text>')
    parts.append(f'<line class="median" x1="{x - 9:.1f}" x2="{x + 9:.1f}" y1="{yScale(random2[size]["median"]):.1f}" y2="{yScale(random2[size]["median"]):.1f}"><title>N={size}: median of 64 random codes {random2[size]["median"]:.2f}</title></line>')
    parts.append(f'<circle class="random" cx="{x:.1f}" cy="{yScale(random2[size]["best"]):.1f}" r="4"><title>N={size}: best of 64 random codes {random2[size]["best"]:.2f}</title></circle>')
    parts.append(f'<circle class="dot-high" cx="{x:.1f}" cy="{yScale(trainedBest[size]):.1f}" r="4.4"><title>N={size}: best trained {trainedBest[size]:.2f}</title></circle>')
parts.append('</svg>')
controlsChart = ''.join(parts)
controlRows = ''.join(
    f"<tr><td>{size}</td><td class=\"num\">{trainedBest[size]:.2f}</td><td class=\"num\">{random2[size]['best']:.2f}</td><td class=\"num\">{random2[size]['median']:.2f}</td><td class=\"num\">{random2[size]['maxOverlap']:.2f}</td></tr>"
    for size in sizes)
controlsTable = ('<div class="scroll"><table><thead><tr><th>Orders 0 to N</th><th class="num">Best trained score</th><th class="num">Best random score</th><th class="num">Median random score</th>'
                 f'<th class="num">Highest random overlap</th></tr></thead><tbody>{controlRows}</tbody></table></div>')
maxRandomOverlap = max(max(controls['maxOverlap'] for controls in randomControls(path)[0].values()) for path in args.controlSummaryPaths)

# ---------------------------------------------------------------------------------------------- the two-bump screen (appendix)
bumps = json.load(open(args.bumpPatternsPath))['profiles']
bumpTiles = ''
for name, profile in bumps.items():
    levels = profile['levels']
    voltages = profile['maps'][str(profile['bestIteration'])]
    levelText = ', '.join(f'{key} {value:g}' for key, value in levels.items())
    dark = {i for i, v in enumerate(voltages) if v < THRESHOLD}
    flankDark = len(dark & set(int(c) for c in boundary.flankCellIndices))
    bumpTiles += tile(voltages, name, [('lead', f'overlap {profile["overlapAtBest"]:.2f}'), ('', f'{flankDark} of 54 flank cells'), ('small', name), ('small', f'{levelText}'), ('small', f'score {profile["score"]:.1f}, iteration {profile["bestIteration"]}')])

# ---------------------------------------------------------------------------------------------- registered predictions, with what can be read now
armA = [row for row in restarts if row['stage'] == 'A']
orderThreeA = [row for row in armA if row['orderLabel'] == 'order3']
statuses = {
    'D1': ('fails', 'fail', f"None of {len(restarts)} restarts reaches 0.9. The highest is {max(row['overlap'] for row in notFlooded):.2f} (the 8 floods are 0.67)."),
    'D2': ('fails', 'fail', f"Arm A, orders 0-3 at ceiling 1.3: the highest overlap is {max(row['overlap'] for row in armA if row['orderLabel'] in ('order0', 'order1', 'order2', 'order3')):.2f}."),
    'D3': ('awaits scoring', '', 'It compares the best scores of two code sets. The registered scoring script has not been run.'),
    'D4': ('cannot be tested', 'na', 'It reads the code of a size that reaches 0.9. None does.'),
    'D5': ('fails', 'fail', f"Arm A, order 3: {sum(row['overlap'] >= FORMED for row in orderThreeA)} of {len(orderThreeA)} restarts reach 0.9; the highest is {max(row['overlap'] for row in orderThreeA):.2f}."),
    'D6': ('cannot be tested', 'na', 'It replays the primary code. There is none.'),
    'D7': ('cannot be tested', 'na', 'It replays the primary code. There is none.'),
    'D8': ('cannot be tested', 'na', 'It replays the primary code. There is none.'),
}


def predictionCard(prediction):
    key = prediction['name'].split('-')[0]
    label, cls, reading = statuses[key]
    return (f"<article class=\"prediction\"><h3>{escape(prediction['name'])} <span class=\"chip {cls}\">{label}</span></h3><dl>"
            f"<dt>Claim</dt><dd>{escape(prediction['claim'])}</dd><dt>Criterion</dt><dd>{escape(prediction['criterion'])}</dd>"
            f"<dt>Basis</dt><dd>{escape(prediction['basis'])}</dd><dt>Reading now</dt><dd>{escape(reading)}</dd></dl></article>")


predictionCards = ''.join(predictionCard(p) for p in registration['predictions'])

jobRows = ''.join(
    f"<tr><td>{stage['arm']}</td><td>{escape(stage['label'])}</td><td class=\"mono\">{stage['job']}</td><td class=\"num\">{stage['tasks']}</td>"
    f"<td class=\"num\">{stage['ceiling']}</td><td class=\"num\">{stage['population']} x {stage['generations']}</td><td class=\"mono\">{escape(stage['extra']) or '-'}</td></tr>"
    for stage in jobs['stages'])
jobsTable = ('<div class="scroll"><table><thead><tr><th>Arm</th><th>Stage</th><th>Job</th><th class="num">Restarts</th><th class="num">Ceiling</th>'
             '<th class="num">Population x generations</th><th>Extra arguments</th></tr></thead>'
             f'<tbody>{jobRows}</tbody></table></div>')

# ---------------------------------------------------------------------------------------------- fill the template
page = open(args.templatePath).read()
for placeholder, value in (('{{BUILT}}', datetime.datetime.now().strftime('%Y-%m-%d %H:%M')), ('{{TARGET_FIGURES}}', targetFigures),
                           ('{{STAGE_BEST_TILES}}', stageBestTiles), ('{{GALLERY_TILES}}', galleryTiles), ('{{FLOOD_TILE}}', floodTile),
                           ('{{OVERLAP_CHART}}', overlapChart), ('{{STAGE_TABLE}}', stageTable), ('{{DETAIL_TABLE}}', detailTable),
                           ('{{REACH_MAP}}', reachMap), ('{{REPLAY_TILES}}', replayTiles), ('{{REPLAY_CHART}}', replayChart),
                           ('{{RING_MAP}}', ringMap), ('{{COEFFICIENT_CHART}}', coefficientChart), ('{{CONTROLS_CHART}}', controlsChart),
                           ('{{CONTROLS_TABLE}}', controlsTable), ('{{BUMP_TILES}}', bumpTiles), ('{{PREDICTION_CARDS}}', predictionCards),
                           ('{{REGISTRATION_QUESTION}}', escape(registration['question'])), ('{{JOBS_TABLE}}', jobsTable)):
    page = page.replace(placeholder, value)
assert '{{' not in page, 'an unfilled placeholder is left in the page'
open(args.outputPath, 'w').write(page)

total = statsRow(restarts)
print(f'wrote {args.outputPath}')
print(f"restarts {total['count']}, codes simulated {total['codes']:,}, formed {total['formed']}, best overlap {total['best']:.3f} (not counting {len(floods)} floods), best score {total['bestScore']:.2f}")
print(f'distinct dark-cell patterns among the {len(notFlooded)} non-flooded restarts: {numDistinct}; gallery shows {args.numGalleryPatterns}')
print(f'good restarts (overlap >= 0.5, not flooded): {len(good)}; mean share dark by column 1..9: {[round(v, 2) for v in columnShare]}; centre cell {share[60]:.2f}')
print(f'best-code replay: later max overlap {laterMax:.2f}, share >= 0.5 in the scored window {windowShare:.3f}; highest random-code overlap in any summary {maxRandomOverlap:.3f}')
