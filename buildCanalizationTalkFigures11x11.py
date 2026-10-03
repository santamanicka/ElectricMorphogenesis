"""Slide-ready figures for the talk "Can spatial bulk pattern development be canalized from the boundary?". EXPLORATORY: nothing here was predicted or registered.

Reads the data files of the Training the Ring, Latching Switch, Relay's Loop, Stripes and Double Stripes reports and replays a few ring codes on the reference
tissue (checkpoint 1888, hold 301) with the pure step; nothing is trained. The stripe stands for the simple target and the face for the complex one.

    python3 buildCanalizationTalkFigures11x11.py                      # every figure
    python3 buildCanalizationTalkFigures11x11.py --stages codes,time  # some

Writes presentation/backup_quantitative/<number>_<name>.png (never overwriting unless --overwrite) and, for the numbers they print, data/canalizationTalkNumbers.json.
"""
import argparse
import json
import os

import numpy as np
from matplotlib.patches import FancyArrowPatch

import boundaryCodeUtilities as boundary
from canalizationTalkCommon import *

parser = argparse.ArgumentParser()
parser.add_argument('--stages', type=str, default='codes,time,stripeRing,faceRing,slidingPatterns,slidingAllOrders,slidingNets,channel,routes')
parser.add_argument('--outputDirectory', type=str, default='presentation/backup_quantitative')
parser.add_argument('--cachePath', type=str, default='data/canalizationTalkReplays1888Hold301.npz')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
stages = args.stages.split(',')
replayer = Replayer()
out = lambda name: f'{args.outputDirectory}/{name}'


def cached(name, build):
    """Replays are cheap but not free: keep them in one npz so a redraw does not simulate again."""
    store = dict(np.load(args.cachePath, allow_pickle=True)) if os.path.exists(args.cachePath) else {}
    if name not in store:
        store[name] = build()
        np.savez_compressed(args.cachePath, **store)
    return store[name]


def trainedCourse(key):
    target = TARGETS[key]
    return cached(f'course_{key}', lambda: replayer.run(ringValuesOf(target['coefficients'], target['ceiling']), 3000)[0].astype(np.float32))


def codeText(key):
    names = ['a₀', 'a₁', 'a₂', 'a₃']
    return '   '.join(f'{names[i]} = {value:+.3f}' if i else f'{names[i]} = {value:.3f}' for i, value in enumerate(TARGETS[key]['coefficients']))


# =========================================================================================================== 01 targets, codes, patterns
if 'codes' in stages:
    figure = newFigure(13.33, 7.0)
    grid = figure.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], hspace=0.30, wspace=0.10)
    for row, key in enumerate(('stripe', 'face')):
        target = TARGETS[key]
        course = trainedCourse(key)
        targetImage = np.full(LATTICE * LATTICE, -5.0)
        targetImage[target['cells']] = -60.0
        ringValues = ringValuesOf(target['coefficients'], target['ceiling'])
        axes = [figure.add_subplot(grid[row, column]) for column in range(4)]
        drawTissue(axes[0], targetImage)
        drawRingCode(axes[1], ringValues, vmax=1.5)
        drawTissue(axes[2], course[target['readIteration']], outlineCells=target['cells'])
        if row == 0:
            for axis, title in zip(axes[:3], ('target', 'ring code, held 301 iterations', 'tissue after release')):
                axis.set_title(title, fontsize=14, color=INK_2, pad=8)
        axes[0].set_ylabel(target['label'], fontsize=22, color=target['colour'], fontweight='bold', rotation=90, labelpad=12)
        axes[2].set_xlabel(f'iteration {target["readIteration"]}', fontsize=12, color=INK_3)
        axes[3].axis('off')
        coefficientCount = len(target['coefficients'])
        randomBest = {'stripe': '18.1', 'face': '20.6\u201322.4'}[key]
        overlapName = {'stripe': 'stripe: 27 of 27 cells dark, 0 strays', 'face': 'features: 14 of 14 cells dark, 1 stray'}[key]
        axes[3].text(0.02, 1.0, f'{coefficientCount} numbers', fontsize=26, fontweight='bold', color=target['colour'], va='top', transform=axes[3].transAxes)
        axes[3].text(0.02, 0.72, codeText(key).replace('   ', '\n'), fontsize=13, color=INK, va='top', family='monospace', linespacing=1.5, transform=axes[3].transAxes)
        axes[3].text(0.02, 0.0, f'score {target["score"]:.2f} mV\nrandom codes: best {randomBest} mV\n{overlapName}', fontsize=11.5, color=INK_2, va='bottom', linespacing=1.5, transform=axes[3].transAxes)
    figure.suptitle('A few numbers on the boundary are enough to steer the bulk to a target', fontsize=19, fontweight='bold', color=INK, y=0.99)
    figure.text(0.5, 0.045, 'ochre dot: ring cell held above 1.439, the upper edge of the isolated cell\u2019s bistable window (the stripe code has ten; the face code none)', ha='center', fontsize=11.5, color=INK_3)
    save(figure, out('01_twoTargetsOneHandle.png'), args.overwrite)

# =========================================================================================================== 02 time course
if 'time' in stages:
    snapshots = [100, 300, 504, 1000, 1850, 2173]
    figure = newFigure(13.33, 5.6)
    grid = figure.add_gridspec(2, len(snapshots), hspace=0.18, wspace=0.08)
    for row, key in enumerate(('stripe', 'face')):
        target = TARGETS[key]
        course = trainedCourse(key)
        for column, iteration in enumerate(snapshots):
            axis = figure.add_subplot(grid[row, column])
            isReadout = iteration == target['readIteration'] or (key == 'face' and iteration == 2173)
            drawTissue(axis, course[iteration], outlineCells=target['cells'] if iteration >= 300 else None, frameColour=target['colour'] if isReadout else None)
            if row == 0:
                axis.set_title(f'iteration {iteration}' + ('\nring held' if iteration < HOLD else '\nreleased'), fontsize=13, color=INK_2)
            if column == 0:
                axis.set_ylabel(target['label'], fontsize=20, color=target['colour'], fontweight='bold', labelpad=10)
            inTarget, strays = darkCounts(course[iteration], target['cells'])
            axis.set_xlabel(f'{inTarget}/{len(target["cells"])} dark, {strays} stray', fontsize=11, color=INK_3)
    figure.suptitle('The stripe is written while the ring is held; the face appears about 1,900 iterations after the code lets go', fontsize=17, fontweight='bold', y=1.03)
    save(figure, out('02_whenThePatternIsWritten.png'), args.overwrite)

# =========================================================================================================== helpers for the ring-code panels
def drawRingProfile(axis, key, showEdge=False, annotate=True):
    """The held value of each ring cell against its angle from straight up (the code a_0 + a_1 cos + ...), the cells above the bistable edge ringed."""
    target = TARGETS[key]
    values = ringValuesOf(target['coefficients'], target['ceiling'])
    order = np.argsort(RING_ANGLES)
    degrees = np.degrees(RING_ANGLES[order])
    axis.plot(degrees, values[order], '-', color=target['colour'], lw=2.4, zorder=2)
    axis.scatter(degrees, values[order], s=34, color=target['colour'], zorder=3)
    if showEdge:
        axis.axhline(BISTABLE_EDGE, color=INK_3, ls=(0, (4, 3)), lw=1.4)
        axis.text(-90, BISTABLE_EDGE - 0.012, 'bistable window\u2019s\nupper edge, 1.439', fontsize=10, color=INK_3, va='top', ha='center', linespacing=1.1)
        above = values[order] > BISTABLE_EDGE
        axis.scatter(degrees[above], values[order][above], s=95, facecolors='none', edgecolors=OCHRE, linewidths=2.0, zorder=4)
    axis.set_xlim(-185, 185)
    axis.set_xticks([-180, -90, 0, 90, 180])
    axis.set_xticklabels(['bottom', 'left', 'top', 'right', 'bottom'])
    axis.set_ylabel('held G_pol / G_ref')
    return values


# =========================================================================================================== 03 stripe: ring bumps write the two ends
if 'stripeRing' in stages:
    segments = json.load(open('data/boundaryHarmonicStripeRingSegments1888Hold301StripesInteriorMinus60Minus5Order2Restart06Ceiling2WithPatterns.json'))
    verdicts = segments['verdicts']
    ends = json.load(open('data/boundaryHarmonicStripeEndsScan1888Hold301StripesInteriorMinus60Minus5Side1p2127.json'))
    confirmation = json.load(open('data/boundaryHarmonicStripeEndsConfirmationScoring1888Hold301StripesInteriorMinus60Minus5.json'))['verdicts']
    figure = newFigure(13.33, 7.5)
    outer = figure.add_gridspec(2, 2, width_ratios=[1, 1.55], height_ratios=[1, 1.1], hspace=0.38, wspace=0.12, left=0.06, right=0.985, top=0.90, bottom=0.08)
    axis = figure.add_subplot(outer[0, 0])
    drawRingProfile(axis, 'stripe', showEdge=True)
    axis.set_ylim(1.15, 1.56)
    axis.set_title('the code: two bumps, top and bottom', fontsize=13.5, color=INK_2, loc='left')
    panels = [('noRing', 'no ring', None), ('stripeFacing', '6 end cells', verdicts['W1']['stripeFacingShareOfFull']),
              ('topAndBottom', 'top + bottom', verdicts['W2']['topAndBottomShare']), ('leftAndRight', 'both sides', verdicts['W2']['leftAndRightShare']),
              ('fullRing', 'whole ring', 1.0)]
    inner = outer[0, 1].subgridspec(1, len(panels), wspace=0.12)
    for column, (name, label, share) in enumerate(panels):
        axis = figure.add_subplot(inner[0, column])
        outcome = segments['outcomes'][name] if name in segments['outcomes'] else segments['baseline']
        drawTissue(axis, segments['patterns'][name], outlineCells=STRIPE_CELLS, lineWidth=1.5)
        axis.set_title(label, fontsize=11, color=INK_2, pad=4)
        shareText = '' if share is None else f'{100 * share:.0f}% of effect\n'
        axis.set_xlabel(f'{shareText}{outcome["stripeDark"]}/27 dark', fontsize=10.5, color=INK_3)
    figure.text(0.50, 0.905, 'which part of the ring writes it?  only the listed ring cells are held', fontsize=13.5, color=INK_2, ha='left')
    # the two ends are written separately: the (top level, bottom level) plane
    top = np.array(ends['a0']) + np.array(ends['a1']) + np.array(ends['a2'])
    bottom = np.array(ends['a0']) - np.array(ends['a1']) + np.array(ends['a2'])
    upperFormed = np.array(ends['upperDarkAtBest']) >= 11
    lowerFormed = np.array(ends['lowerDarkAtBest']) >= 11
    axis = figure.add_subplot(outer[1, 0])
    topLevels, bottomLevels = np.unique(np.round(top, 4)), np.unique(np.round(bottom, 4))
    classMap = np.zeros((len(bottomLevels), len(topLevels)))
    for topValue, bottomValue, upper, lower in zip(np.round(top, 4), np.round(bottom, 4), upperFormed, lowerFormed):
        classMap[np.searchsorted(bottomLevels, bottomValue), np.searchsorted(topLevels, topValue)] = 1 + (upper and not lower) * 1 + (lower and not upper) * 2 + (upper and lower) * 3 - 1 * (not upper and not lower) * 0
    classMap[classMap == 1] = 0                                                       # 0 neither, 2 upper only, 3 lower only, 4 both
    from matplotlib.colors import ListedColormap
    colours = ['#E4E8EC', None, TEAL, OCHRE, '#2B2B2B']
    axis.pcolormesh(topLevels, bottomLevels, np.ma.masked_where(classMap == 1, classMap), cmap=ListedColormap([colours[0], colours[0], colours[2], colours[3], colours[4]]), vmin=0, vmax=4, shading='nearest')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='s', ls='', color=colour, label=label, markersize=9) for colour, label in ((TEAL, 'upper half only'), (OCHRE, 'lower half only'), ('#2B2B2B', 'whole stripe'), ('#E4E8EC', 'neither'))]
    axis.set_xlabel('ring level at the top  (G_pol / G_ref)')
    axis.set_ylabel('ring level at the bottom')
    for level in (1.485, 1.495):
        axis.axvline(level, color=INK_3, ls=':', lw=1.0)
        axis.axhline(level, color=INK_3, ls=':', lw=1.0)
    axis.legend(handles=handles, fontsize=10, frameon=False, loc='upper center', bbox_to_anchor=(0.45, -0.22), ncol=4, handletextpad=0.2, columnspacing=1.0)
    axis.set_title('the two ends are separate switches', fontsize=13.5, color=INK_2, loc='left')
    textAxis = figure.add_subplot(outer[1, 1])
    textAxis.axis('off')
    e1, e2, e3, e4 = (confirmation[k] for k in ('E1-theTopWritesTheUpperHalf', 'E2-theBottomWritesTheLowerHalf', 'E3-theEndsAreWrittenWithoutEachOther', 'E4-theStripeNeedsBothEnds'))
    lines = [('Pre-registered test on 2,000 new codes', True),
             (f'\u2022 upper half forms {100 * e1["insideWindow"]["rate"]:.0f}% when the top level is in the window, {e1["outsideWindow"]["count"]}/{e1["outsideWindow"]["of"]} outside', False),
             (f'\u2022 lower half forms {100 * e2["insideWindow"]["rate"]:.0f}% when the bottom level is in the window, {e2["outsideWindow"]["count"]}/{e2["outsideWindow"]["of"]} outside', False),
             (f'\u2022 the top alone never writes the lower half ({e3["topOnlyLowerFormed"]["count"]}/250),\n   nor the bottom alone the upper ({e3["bottomOnlyUpperFormed"]["count"]}/250)', False),
             (f'\u2022 whole stripe: {100 * e4["bothStratum"]["rate"]:.0f}% with both ends in the window, {100 * e4["otherStrata"]["rate"]:.1f}% otherwise ({e4["otherStrata"]["count"]}/{e4["otherStrata"]["of"]})', False),
             ('\u2022 the window is thin: 1.485\u20131.495, just above the 1.439 edge', False)]
    y = 0.98
    for text, bold in lines:
        textAxis.text(0.02, y, text, fontsize=13 if bold else 12, fontweight='bold' if bold else 'normal', color=INK if bold else INK_2, va='top', transform=textAxis.transAxes, linespacing=1.4)
        y -= 0.12 if '\n' not in text else 0.19
    figure.suptitle('Stripe: bumps at the top and bottom of the ring write the two ends of the stripe', fontsize=18, fontweight='bold', y=0.985)
    save(figure, out('03_stripe_ringBumpsWriteTheEnds.png'), args.overwrite)

# =========================================================================================================== 04 face: no part alone, two modules
if 'faceRing' in stages:
    wall = json.load(open('data/boundaryHarmonicWallCounterfactual1888Hold301FaceMinus60Minus5.json'))
    modules = json.load(open('data/relayLoopModulesConfirmation1888Hold301FaceMinus60Minus5.json'))
    target = TARGETS['face']
    faceRing = ringValuesOf(target['coefficients'], target['ceiling'])
    wallPatterns = cached('wallPatterns_face', lambda: np.array([
        replayer.run(faceRing, target['readIteration'] + 1, heldCells=np.array(cells, dtype=int), keepEvery=target['readIteration'] + 1)[0][-1]
        for cells in ([], wall['upperWall']['cells'], wall['lowerWall']['cells'], list(RING_CELLS))]))
    figure = newFigure(13.33, 7.5)
    outer = figure.add_gridspec(2, 2, width_ratios=[1, 1.55], height_ratios=[1, 1.1], hspace=0.38, wspace=0.12, left=0.06, right=0.985, top=0.90, bottom=0.08)
    axis = figure.add_subplot(outer[0, 0])
    drawRingProfile(axis, 'face')
    for low, high, name in ((0, 45, 'top'), (45, 90, 'upper\nsides'), (90, 135, 'lower\nsides'), (135, 180, 'bottom')):
        for sign in (-1, 1):
            axis.axvspan(sign * low, sign * high, color='#F1F3F5' if name in ('upper\nsides', 'bottom') else 'white', zorder=0)
    axis.set_ylim(0.0, 1.55)
    for centre, name in ((22.5, 'top'), (67.5, 'upper\nsides'), (112.5, 'lower\nsides'), (157.5, 'bottom')):
        axis.text(centre, 1.52, name, ha='center', va='top', fontsize=9, color=INK_3, linespacing=1.1)
    axis.set_title('the code: smooth, with a deep trough on the lower sides', fontsize=13, color=INK_2, loc='left')
    labels = [('no ring', None), ('upper wall only', wall['upperWall']['alone'] / wall['fullRing']), ('lower wall only', wall['lowerWall']['alone'] / wall['fullRing']), ('whole ring', 1.0)]
    inner = outer[0, 1].subgridspec(1, 4, wspace=0.12)
    for column, (label, share) in enumerate(labels):
        axis = figure.add_subplot(inner[0, column])
        pattern = wallPatterns[column]
        drawTissue(axis, pattern, outlineCells=FACE_CELLS, lineWidth=1.5)
        inTarget, strays = darkCounts(pattern, FACE_CELLS)
        axis.set_title(label, fontsize=11, color=INK_2, pad=4)
        shareText = '' if share is None else f'{100 * share:.0f}% of the lead\n'
        axis.set_xlabel(f'{shareText}{inTarget}/14 dark, {strays} stray', fontsize=10.5, color=INK_3)
    figure.text(0.50, 0.905, 'which part of the ring writes it?  only the listed ring cells are held', fontsize=13.5, color=INK_2, ha='left')
    cells = modules['cellDescription']['cells']
    order = [('both', 'both\nmodules'), ('pushOnly', 'flood push\nonly'), ('lowerOnly', 'lower channel\nonly'), ('neither', 'neither')]
    axis = figure.add_subplot(outer[1, 0])
    rates = [cells[key]['faceLike'] / cells[key]['codes'] for key, _ in order]
    bars = axis.bar(range(4), [100 * r for r in rates], color=[OCHRE, '#D9A27F', '#D9A27F', '#E3C9B6'], width=0.62)
    for position, key in enumerate(key for key, _ in order):
        axis.text(position, 100 * rates[position] + 1.5, f'{cells[key]["faceLike"]}/{cells[key]["codes"]}', ha='center', fontsize=11, color=INK_2)
    axis.set_xticks(range(4))
    axis.set_xticklabels([label for _, label in order], fontsize=10.5)
    axis.set_ylabel('codes with a face-like pattern (%)')
    axis.set_ylim(0, 75)
    axis.set_title('two ring-level modules, jointly associated with the face', fontsize=12.5, color=INK_2, loc='left')
    textAxis = figure.add_subplot(outer[1, 1])
    textAxis.axis('off')
    criteria = {c['name']: c for c in modules['criteria']}
    c3 = criteria['C3-bothModulesGoWithTheFace']
    lines = [('Where the face\u2019s code differs from the stripe\u2019s', True, 1),
             (f'\u2022 no wall half makes a face; each writes about half of the lead,\n   and they add ({wall["upperWall"]["alone"]:.2f} + {wall["lowerWall"]["alone"]:.2f} vs {wall["fullRing"]:.2f})', False, 2),
             ('\u2022 reading a role into each order fails: 0 of 16 pre-registered\n   knockout predictions held', False, 2),
             (f'\u2022 pre-registered test, 320 new codes: face-like {100 * c3["rateBoth"]:.0f}% with both\n   modules, {100 * c3["rateOtherThreePooled"]:.0f}% otherwise (Fisher p = {c3["fisherP"]:.0e}) \u2014 yet only 1 of 320\n   reached a full face: the modules go with face-like patterns', False, 3),
             ('\u2022 when the face appears, no order owns any part of it\n   (share of the pattern owned by a single order: 0.06)', False, 2)]
    y = 0.98
    for text, bold, lineCount in lines:
        textAxis.text(0.02, y, text, fontsize=13 if bold else 11.8, fontweight='bold' if bold else 'normal', color=INK if bold else INK_2, va='top', transform=textAxis.transAxes, linespacing=1.4)
        y -= 0.062 * lineCount + 0.045
    figure.suptitle('Face: no part of the ring writes it, and no order owns a part of it', fontsize=18, fontweight='bold', y=0.985)
    save(figure, out('04_face_noPartAlone_twoModules.png'), args.overwrite)

# =========================================================================================================== 05 sliding one order at a time
TILE_OFFSETS = [-0.08, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08]
ORDER_NAMES = {0: 'a\u2080  dial', 1: 'a\u2081  tilt', 2: 'a\u2082  oval', 3: 'a\u2083'}


def slidingPatterns(key, order, step):
    """The tissue at the target's readout when one order of the trained code is moved by -0.08 ... +0.08 (the others at their trained values)."""
    target = TARGETS[key]
    offsets = np.round(np.arange(-0.08, 0.08 + step / 2, step), 6)

    def build():
        frames = []
        for offset in offsets:
            coefficients = target['coefficients'].copy()
            coefficients[order] += offset
            frames.append(replayer.readout(ringValuesOf(coefficients, 2.0), target['readIteration']))
        return np.array(frames, dtype=np.float32)
    return offsets, cached(f'slide_{key}_order{order}_step{step}', build)


def overlapOf(key, vmem):
    return boundary.structuralIntersectionOverUnion(vmem, STRIPE_CELLS if key == 'stripe' else None)


if 'slidingPatterns' in stages:
    figure = newFigure(13.33, 8.2)
    outer = figure.add_gridspec(2, 1, hspace=0.30, left=0.07, right=0.985, top=0.92, bottom=0.06)
    for block, key in enumerate(('stripe', 'face')):
        target = TARGETS[key]
        offsets, patterns = slidingPatterns(key, 0, 0.002)
        inner = outer[block].subgridspec(2, len(TILE_OFFSETS), height_ratios=[1.0, 0.85], hspace=0.40, wspace=0.10)
        for column, offset in enumerate(TILE_OFFSETS):
            axis = figure.add_subplot(inner[0, column])
            pattern = patterns[int(np.argmin(np.abs(offsets - offset)))]
            drawTissue(axis, pattern, outlineCells=target['cells'], lineWidth=1.4, frameColour=target['colour'] if offset == 0 else None)
            inTarget, strays = darkCounts(pattern, target['cells'])
            axis.set_title(f'{offset:+.2f}' if offset else 'trained', fontsize=11.5, color=target['colour'] if offset == 0 else INK_2, fontweight='bold' if offset == 0 else 'normal', pad=3)
            axis.set_xlabel(f'{inTarget}/{len(target["cells"])} dark, {strays} stray', fontsize=9.5, color=INK_3)
            if column == 0:
                axis.set_ylabel(target['label'], fontsize=20, color=target['colour'], fontweight='bold', labelpad=8)
        axis = figure.add_subplot(inner[1, :])
        dark = np.array([darkCounts(p, target['cells']) for p in patterns])
        axis.plot(offsets, dark[:, 0] / len(target['cells']), color=target['colour'], lw=2.4, label=f'share of the {len(target["cells"])} target cells that are dark')
        axis.plot(offsets, dark[:, 1] / (len(INTERIOR) - len(target['cells'])), color=INK_3, lw=1.6, ls=(0, (4, 2)), label=f'share of the {len(INTERIOR) - len(target["cells"])} other interior cells that are dark')
        axis.axvline(0, color=INK_3, lw=0.8)
        axis.set_xlim(-0.08, 0.08)
        axis.set_ylim(-0.03, 1.08)
        axis.set_xlabel('change in a\u2080, the dial  (G_pol / G_ref)' if block == 1 else '', fontsize=12)
        axis.set_yticks([0, 0.5, 1])
        axis.legend(fontsize=10, frameon=False, loc='upper right', ncol=2, bbox_to_anchor=(1.0, 1.18))
    figure.suptitle('Sliding the dial (order 0) away from the trained code, with every other order held', fontsize=17, fontweight='bold', y=0.985)
    save(figure, out('05_slidingTheDial.png'), args.overwrite)

if 'slidingAllOrders' in stages:
    rows = [('stripe', order) for order in range(3)] + [('face', order) for order in range(4)]
    figure = newFigure(13.33, 10.5)
    grid = figure.add_gridspec(len(rows), len(TILE_OFFSETS), hspace=0.62, wspace=0.12, left=0.10, right=0.975, top=0.93, bottom=0.03)
    for row, (key, order) in enumerate(rows):
        target = TARGETS[key]
        offsets, patterns = slidingPatterns(key, order, 0.002 if order == 0 else 0.005)
        for column, offset in enumerate(TILE_OFFSETS):
            axis = figure.add_subplot(grid[row, column])
            pattern = patterns[int(np.argmin(np.abs(offsets - offset)))]
            drawTissue(axis, pattern, outlineCells=target['cells'], lineWidth=1.2, frameColour=target['colour'] if offset == 0 else None)
            inTarget, strays = darkCounts(pattern, target['cells'])
            if row == 0:
                axis.set_title(f'{offset:+.2f}' if offset else 'trained', fontsize=11.5, color=INK_2, pad=4)
            axis.set_xlabel(f'{inTarget}/{len(target["cells"])} dark, {strays} stray', fontsize=8.0, color=INK_3, labelpad=2)
            if column == 0:
                axis.set_ylabel(f'{target["label"]}\n{ORDER_NAMES[order]}', fontsize=11.5, color=target['colour'], fontweight='bold', labelpad=6)
    figure.suptitle('Every order slid in turn (change in the coefficient, G_pol / G_ref); the others held at their trained values', fontsize=15, fontweight='bold', y=0.98)
    save(figure, out('05b_slidingEveryOrder.png'), args.overwrite)

if 'slidingAllOrders' in stages:
    statistics = {}
    for key, order in rows:
        step = 0.002 if order == 0 else 0.005
        offsets, patterns = slidingPatterns(key, order, step)
        overlaps = np.array([overlapOf(key, pattern) for pattern in patterns])
        statistics[f'{key}_order{order}'] = dict(step=step, stops=int(len(offsets)), formedAtLeast0p9=int((overlaps >= 0.9).sum()), atLeast0p5=int((overlaps >= 0.5).sum()),
                                                shareAtLeast0p5=float((overlaps >= 0.5).mean()), widthOfFormedWindow=float(((overlaps >= 0.9).sum()) * step))
    numbersPath = 'data/canalizationTalkNumbers.json'
    existing = json.load(open(numbersPath)) if os.path.exists(numbersPath) else {}
    existing['sliding'] = dict(note='each order slid alone by -0.08 ... +0.08 around the trained code, the tissue read at the target\'s readout (stripe 504, face 2173)', orders=statistics)
    json.dump(existing, open(numbersPath, 'w'), indent=1)

# =========================================================================================================== 06 the relay nets as one order slides
NODE_POSITIONS = {
    'face': dict(ringTop=(5, 0), ringBottom=(5, 10), ringLeft=(0, 5), ringRight=(10, 5), bgTL=(2.9, 2.9), bgTR=(7.1, 2.9), bgBL=(2.7, 7.5), bgBR=(7.3, 7.5),
                 eyes=(5, 2.5), nose=(5, 5), mouth=(5, 8)),
    'stripe': dict(ringTop=(5, 0), ringBottom=(5, 10), ringLeft=(0, 5), ringRight=(10, 5), stripeUpper=(5, 3), stripeLower=(5, 7), flankLeftUpper=(2, 3),
                   flankLeftLower=(2, 7), flankRightUpper=(8, 3), flankRightLower=(8, 7))}
MIRROR = dict(ringLeft='ringRight', bgTL='bgTR', bgBL='bgBR', flankLeftUpper='flankRightUpper', flankLeftLower='flankRightLower')
MIRROR.update({value: key for key, value in list(MIRROR.items())})


def nodeColour(key, name):
    if name.startswith('ring'):
        return DIAL
    if name in ('eyes', 'nose', 'mouth', 'stripeUpper', 'stripeLower'):
        return TARGETS[key]['colour']
    return '#9AA5AE'


def topEdges(key, record, phaseIndex, pairs, count=3):
    """The `count` biggest signed field transfers of a phase, as (sender, receiver, value), each also drawn on the mirror side."""
    if key == 'stripe':
        edges = [(e['from'], e['to'], e['value']) for e in record['phases'][['flood', 'clear'][phaseIndex]]]
    else:
        values = np.array(record['field'][phaseIndex])
        edges = []
        for index in np.argsort(-np.abs(values))[:count]:
            sender, receiver = pairs[index] if values[index] > 0 else pairs[index][::-1]
            edges.append((sender, receiver, abs(float(values[index]))))
    both = list(edges)
    for sender, receiver, value in edges:
        twin = (MIRROR.get(sender, sender), MIRROR.get(receiver, receiver), value)
        if twin[:2] != (sender, receiver):
            both.append(twin)
    return both


NODE_LABELS = dict(ringTop='ring top', ringBottom='ring bottom', ringLeft='ring left', ringRight='ring right', stripeUpper='stripe upper', stripeLower='stripe lower',
                   flankLeftUpper='flank', flankLeftLower='flank', eyes='eyes', nose='nose', mouth='mouth', bgTL='background', bgBL='background')
LABEL_OFFSETS = dict(ringTop=(0.45, -0.1, 'left'), ringBottom=(0.45, 0.25, 'left'), ringLeft=(0.0, -0.75, 'center'), ringRight=(0.0, -0.75, 'center'),
                     stripeUpper=(0.45, 0.65, 'left'), stripeLower=(0.45, 0.65, 'left'), flankLeftUpper=(0.0, -0.75, 'center'), flankLeftLower=(0.0, 0.95, 'center'),
                     eyes=(0.45, 0.0, 'left'), nose=(0.45, 0.0, 'left'), mouth=(0.45, 0.0, 'left'), bgTL=(0.0, -0.75, 'center'), bgBL=(0.0, 0.95, 'center'))


def drawNet(axis, key, edges, scale, labels=False, backdrop=None):
    positions = NODE_POSITIONS[key]
    if backdrop is not None:
        axis.imshow(np.asarray(backdrop).reshape(LATTICE, LATTICE), cmap=VMEM_MAP, vmin=VMEM_LOW, vmax=VMEM_HIGH, alpha=0.40, interpolation='nearest')
    axis.set_xlim(-2.0, 12.0)
    axis.set_ylim(11.0, -1.0)
    axis.set_aspect('equal')
    axis.axis('off')
    for name, (x, y) in positions.items():
        axis.plot(x, y, 'o', ms=9 if name.startswith('ring') else 10, color=nodeColour(key, name), mec='white', mew=1.2, zorder=3)
        if labels and name in LABEL_OFFSETS:
            dx, dy, alignment = LABEL_OFFSETS[name]
            axis.text(x + dx, y + dy, NODE_LABELS[name], fontsize=7.5, color=INK_2, ha=alignment, va='center', zorder=5)
    for sender, receiver, value in sorted(edges, key=lambda e: e[2]):
        (x0, y0), (x1, y1) = positions[sender], positions[receiver]
        width = float(np.clip(0.8 + 6.0 * value / scale, 0.8, 7.0))
        axis.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f'-|>,head_length={0.42 + 0.03 * width:.2f},head_width={0.16 + 0.02 * width:.2f}', mutation_scale=11, lw=width,
                                       color=INK, alpha=0.80, shrinkA=6, shrinkB=7, connectionstyle='arc3,rad=0.10', zorder=2))


if 'slidingNets' in stages:
    faceNets = json.load(open('data/relayLoopFullNets1888Hold301FaceMinus60Minus5.json'))
    stripeNets = json.load(open('data/relayLoopStripesPageData1888Hold301StripesInteriorMinus60Minus5.json'))
    stripePoints = {round(point['offset'], 3): point for point in stripeNets['sliderData']['0']['points']}
    stripeStops = [-0.15, -0.04, 0.0, 0.04, 0.15]
    faceStops = [('five_o0_m050', 0.5), ('five_o0_m075', 0.75), ('trained', 1.0), ('five_o0_m150', 1.5), ('five_o0_m200', 2.0)]
    clearIndex = 1
    stripeTrainedEdges = topEdges('stripe', stripePoints[0.0], clearIndex, None)
    faceTrainedEdges = topEdges('face', faceNets['codes']['trained'], clearIndex, faceNets['pairs'])
    scales = {'stripe': max(e[2] for e in stripeTrainedEdges), 'face': max(e[2] for e in faceTrainedEdges)}
    figure = newFigure(13.33, 7.0)
    grid = figure.add_gridspec(2, 5, hspace=0.30, wspace=0.04, left=0.06, right=0.99, top=0.86, bottom=0.04)
    for column in range(5):
        for row, key in enumerate(('stripe', 'face')):
            target = TARGETS[key]
            axis = figure.add_subplot(grid[row, column])
            if key == 'stripe':
                offset = stripeStops[column]
                record = stripePoints[round(offset, 3)]
                coefficients = target['coefficients'].copy()
                coefficients[0] += offset
                edges = topEdges('stripe', record, clearIndex, None)
                title = f'a\u2080 = {coefficients[0]:.2f}' + ('  (trained)' if offset == 0 else f'  ({offset:+.2f})')
            else:
                code, multiplier = faceStops[column]
                record = faceNets['codes'][code]
                coefficients = target['coefficients'].copy()
                coefficients[0] *= multiplier
                edges = topEdges('face', record, clearIndex, faceNets['pairs'])
                title = f'a\u2080 = {coefficients[0]:.2f}' + ('  (trained)' if multiplier == 1 else f'  (\u00d7{multiplier:g})')
            vmem = cached(f'netBackdrop_{key}_{column}', lambda: replayer.readout(ringValuesOf(coefficients, 2.0), target['readIteration']).astype(np.float32))
            drawNet(axis, key, edges, scales[key], labels=(column == 0), backdrop=vmem)
            axis.set_title(title, fontsize=12, color=target['colour'] if title.endswith('(trained)') else INK_2, fontweight='bold' if title.endswith('(trained)') else 'normal', pad=2)
            inTarget, strays = darkCounts(vmem, target['cells'])
            axis.text(5, 11.6, f'pattern: {inTarget}/{len(target["cells"])} dark, {strays} stray', ha='center', va='top', fontsize=9.5, color=INK_3)
            if column == 0:
                axis.text(-2.6, 5, target['label'], rotation=90, ha='center', va='center', fontsize=20, color=target['colour'], fontweight='bold')
    figure.suptitle('The causal net (clear phase, three biggest field transfers) as the dial slides', fontsize=17, fontweight='bold', y=0.975)
    figure.text(0.5, 0.915, 'arrow width: size of the transfer, on the trained code\u2019s scale \u00b7 purple: ring blocks \u00b7 colour: the target\u2019s own blocks \u00b7 grey: background / flanks \u00b7 tissue shaded by the pattern at the readout',
                ha='center', fontsize=10.5, color=INK_3)
    save(figure, out('06_slidingRelayNets.png'), args.overwrite)

# =========================================================================================================== 07 how wide is the channel
REGION_INDEX = np.minimum((np.degrees(np.abs(RING_ANGLES)) // 45).astype(int), 3)          # top, upper sides, lower sides, bottom (folded angle)
DISTANCE_BINS = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 2.0]
DISTANCE_LABELS = ['<0.01', '0.01\u2013\n0.02', '0.02\u2013\n0.05', '0.05\u2013\n0.1', '0.1\u2013\n0.2', '0.2\u2013\n0.4', '>0.4']


def regionLevels(coefficients):
    values = ringValuesOf(coefficients, 2.0)
    return np.array([values[REGION_INDEX == region].mean() for region in range(4)])


def topThreeSet(vector, pairs):
    return {tuple(pairs[i]) if vector[i] > 0 else tuple(pairs[i][::-1]) for i in np.argsort(-np.abs(vector))[:3]}


def sweepStatistics(key):
    """For each of the 400 codes of the target's sweep: its distance from the trained code in ring-region space (rms of the four region levels,
    G_pol / G_ref), its overlap with the target and how much of the trained code's top-three field transfers (flood, clear) it keeps."""
    suffix = {'face': '1888Hold301FaceMinus60Minus5', 'stripe': '1888Hold301StripesInteriorMinus60Minus5'}[key]
    sweep = json.load(open(f'data/relayLoopSweepNets{suffix}.json'))
    pairs = [tuple(pair) for pair in sweep['pairs']]
    trained = TARGETS[key]['coefficients']
    if key == 'face':
        trainedField = np.array(json.load(open('data/relayLoopFullNets1888Hold301FaceMinus60Minus5.json'))['codes']['trained']['field'])
        trainedSets = [topThreeSet(trainedField[phase], pairs) for phase in range(2)]
    else:
        phases = json.load(open('data/relayLoopStripesPageData1888Hold301StripesInteriorMinus60Minus5.json'))['trainedTop3Phases']
        trainedSets = [{(edge['from'], edge['to']) for edge in phases[phase]} for phase in ('flood', 'clear')]
    distances, overlaps, retentions = [], [], []
    for code in sweep['codes'].values():
        coefficients = np.array(code['multipliers']) * (trained if key == 'face' else 1.0)
        distances.append(float(np.sqrt(np.mean((regionLevels(coefficients) - regionLevels(trained)) ** 2))))
        overlaps.append(code['faceOverlap'])
        field = np.array(code['field'])
        retentions.append(np.mean([len(topThreeSet(field[phase], pairs) & trainedSets[phase]) / 3 for phase in range(2)]))
    return np.array(distances), np.array(overlaps), np.array(retentions)


if 'channel' in stages:
    numbers = {}
    figure = newFigure(13.33, 7.6)
    grid = figure.add_gridspec(2, 2, hspace=0.68, wspace=0.30, left=0.08, right=0.985, top=0.84, bottom=0.11)
    axisPattern, axisNet = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
    baselines = {'stripe': 9 / 27, 'face': 0.1875}
    for key in ('stripe', 'face'):
        colour = TARGETS[key]['colour']
        distances, overlaps, retentions = sweepStatistics(key)
        positions, medianOverlap, meanRetention, counts = [], [], [], []
        for index, (low, high) in enumerate(zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:])):
            inBin = (distances >= low) & (distances < high)
            if inBin.sum():
                positions.append(index)
                medianOverlap.append(float(np.median(overlaps[inBin])))
                meanRetention.append(float(retentions[inBin].mean()))
                counts.append(int(inBin.sum()))
        axisPattern.plot(positions, medianOverlap, 'o-', color=colour, lw=2.4, ms=7, label=TARGETS[key]['label'])
        axisPattern.axhline(baselines[key], color=colour, ls=':', lw=1.3)
        axisNet.plot(positions, meanRetention, 'o-', color=colour, lw=2.4, ms=7, label=TARGETS[key]['label'])
        numbers[key] = dict(distanceBins=DISTANCE_BINS, codesPerBin=counts, medianOverlap=medianOverlap, meanTopThreeRetention=meanRetention, noRingOverlap=baselines[key])
    for axis in (axisPattern, axisNet):
        axis.set_xticks(range(len(DISTANCE_LABELS)))
        axis.set_xticklabels(DISTANCE_LABELS, fontsize=9.5)
        axis.set_xlabel('distance from the trained code (G_pol / G_ref)', fontsize=11)
    axisPattern.set_ylabel('median overlap with the target')
    axisPattern.set_title('the pattern: a thin slice', fontsize=13.5, color=INK_2, loc='left')
    axisPattern.text(6.1, baselines['stripe'] + 0.015, 'no ring', fontsize=9, color=TEAL, ha='right')
    axisPattern.text(6.1, baselines['face'] - 0.075, 'no ring', fontsize=9, color=OCHRE, ha='right')
    axisPattern.set_ylim(0, 1.0)
    axisNet.set_ylabel('top-3 transfers kept')
    axisNet.set_title('the causal net: persists much further', fontsize=13.5, color=INK_2, loc='left')
    axisNet.set_ylim(0, 1.0)
    axisNet.legend(frameon=False, fontsize=11)
    # (c) cell-wise jitter of the held code, the face robustness protocol
    axisJitter = figure.add_subplot(grid[1, 0])
    sigmas = ['0.01', '0.03', '0.1']
    width = 0.19
    for offset, (key, threshold, alpha) in zip((-1.5, -0.5, 0.5, 1.5), (('stripe', 'atLeast0p5', 1.0), ('face', 'atLeast0p5', 1.0), ('stripe', 'atLeast0p9', 0.45), ('face', 'atLeast0p9', 0.45))):
        jitter = json.load(open(f'data/boundaryHarmonicCodeJitter1888Hold301{"FaceMinus60Minus5" if key == "face" else "StripesInteriorMinus60Minus5"}.json'))['summary']
        values = [jitter[sigma][threshold] for sigma in sigmas]
        axisJitter.bar(np.arange(3) + offset * width, values, width * 0.92, color=TARGETS[key]['colour'], alpha=alpha,
                       label=f'{TARGETS[key]["label"]}, overlap \u2265 {"0.5" if threshold == "atLeast0p5" else "0.9"}')
        for position, value in enumerate(values):
            axisJitter.text(position + offset * width, value + 1.5, str(value), ha='center', fontsize=8.5, color=INK_2)
        numbers.setdefault('jitter', {})[f'{key}_{threshold}'] = dict(zip(sigmas, values))
    axisJitter.set_xticks(range(3))
    axisJitter.set_xticklabels(['1%', '3%', '10%'])
    axisJitter.set_xlabel('independent noise on each of the 40 held ring values')
    axisJitter.set_ylabel('noisy codes that still\nmake it (of 100)')
    axisJitter.set_title('the pattern under noise', fontsize=13.5, color=INK_2, loc='left')
    axisJitter.legend(frameon=False, fontsize=9, loc='upper right')
    axisJitter.set_ylim(0, 112)
    # (d) how well the ring's region levels predict the net and the outcome on held-out codes
    axisPredict = figure.add_subplot(grid[1, 1])
    maps = {key: json.load(open(path)) for key, path in (('stripe', 'data/relayLoopStripesOrderEdgeMap1888Hold301StripesInteriorMinus60Minus5.json'), ('face', 'data/relayLoopOrderEdgeMap1888Hold301FaceMinus60Minus5.json'))}
    metrics = [('edges:\nunseen region', lambda m: float(np.median([e['blockedAuc ring-region means (4)'] for e in m['edgeScores']]))),
               ('edges:\npage codes', lambda m: float(np.median([e['pageAuc ring-region means (4)'] for e in m['edgeScores'] if e.get('pageAuc ring-region means (4)') is not None]))),
               ('gap:\nR\u00b2', lambda m: m['outcomes']['gap']['blockedRSquared']),
               ('overlap:\nR\u00b2', lambda m: m['outcomes'].get('stripeOverlap', m['outcomes'].get('faceOverlap'))['blockedRSquared'])]
    for offset, key in zip((-0.5, 0.5), ('stripe', 'face')):
        values = [metric(maps[key]) for _, metric in metrics]
        axisPredict.bar(np.arange(len(metrics)) + offset * 0.38, values, 0.34, color=TARGETS[key]['colour'], label=TARGETS[key]['label'])
        for position, value in enumerate(values):
            axisPredict.text(position + offset * 0.38, value + 0.015, f'{value:.2f}', ha='center', fontsize=9, color=INK_2)
        numbers.setdefault('predictability', {})[key] = dict(zip(['edgesBlockedAuc', 'edgesPageAuc', 'gapRSquared', 'overlapRSquared'], values))
    axisPredict.set_xticks(range(len(metrics)))
    axisPredict.set_xticklabels([name for name, _ in metrics], fontsize=9.5)
    axisPredict.set_ylim(0, 1.05)
    axisPredict.set_ylabel('AUC or R\u00b2, held out')
    axisPredict.set_title('what the ring\u2019s four region levels predict', fontsize=13.5, color=INK_2, loc='left')
    figure.suptitle('How wide is the channel?', fontsize=19, fontweight='bold', y=0.985)
    figure.text(0.5, 0.925, 'The outcome needs a thin slice of code space; the causal net behind it is far more forgiving.   Exploratory: 400-code sweeps of each target, sampled differently.', ha='center', fontsize=11.5, color=INK_2)
    save(figure, out('07_howWideIsTheChannel.png'), args.overwrite)
    numbersPath = 'data/canalizationTalkNumbers.json'
    existing = json.load(open(numbersPath)) if os.path.exists(numbersPath) else {}
    existing['channel'] = numbers
    json.dump(existing, open(numbersPath, 'w'), indent=1)

# =========================================================================================================== 08 the stripe's two routes against the face's code (backup)
if 'routes' in stages:
    records = json.load(open('data/boundaryHarmonicTrainingRestartRecords1888Hold301StripesInteriorMinus60Minus5.json'))['folders']
    stripeSet = set(STRIPE_CELLS.tolist())

    def stripeOverlap(vmem):
        dark = {int(c) for c in INTERIOR if vmem[c] < THRESHOLD}
        return len(dark & stripeSet) / len(dark | stripeSet)
    formedLate = [r for r in records['EvenOrdersPopulation64'] if len(r['orders']) == 11 and stripeOverlap(r['bestVmem']) >= 0.9]
    figure = newFigure(13.33, 4.9)
    grid = figure.add_gridspec(1, 3, wspace=0.14, left=0.06, right=0.99, top=0.76, bottom=0.14)
    order = np.argsort(RING_ANGLES)
    degrees = np.degrees(RING_ANGLES[order])
    axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
    drawRingProfile(axes[0], 'stripe', showEdge=True)
    axes[0].set_title('stripe, route 1\nceiling 2.0, orders 0\u20132', fontsize=12.5, color=INK_2, loc='left')
    for record in formedLate:
        values = np.clip(np.cos(np.outer(RING_ANGLES, record['orders'])) @ np.array(record['bestCoefficients']), 0, 1.3)
        axes[1].plot(degrees, values[order], color=TEAL, alpha=0.45, lw=1.3)
    axes[1].set_title(f'stripe, route 2 ({len(formedLate)} codes that form it)\nceiling 1.3, even orders 0\u201320', fontsize=12.5, color=INK_2, loc='left')
    axes[1].axhline(BISTABLE_EDGE, color=INK_3, ls=(0, (4, 3)), lw=1.2)
    drawRingProfile(axes[2], 'face')
    axes[2].set_title('face\nceiling 1.3, orders 0\u20133', fontsize=12.5, color=INK_2, loc='left')
    for axis in axes[1:]:
        axis.set_ylabel('')
        axis.set_xlim(-185, 185)
        axis.set_xticks([-180, -90, 0, 90, 180])
        axis.set_xticklabels(['bot.', 'left', 'top', 'right', 'bot.'])
    axes[1].set_xlabel('')
    for axis in axes:
        axis.set_ylim(0, 2.05 if axis is axes[0] else 1.65)
    axes[0].set_ylim(0, 1.65)
    figure.suptitle('The stripe\u2019s readable code needs the ring to cross the bistable edge; under the face\u2019s limits it is a spiky high-order code', fontsize=15, fontweight='bold', y=0.97)
    save(figure, out('08_stripeTwoRoutes_ringCodes.png'), args.overwrite)

print('stages done:', ', '.join(stages))
