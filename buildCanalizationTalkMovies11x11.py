"""Movies for the talk "Can spatial bulk pattern development be canalized from the boundary?". EXPLORATORY: nothing here was predicted or registered.

  codeToPattern  the stripe code and the face code, each held on the ring for 301 iterations and then released, the tissue every 20 iterations
  slidingDial    the dial (order 0) of each trained code slid from -0.08 to +0.08 in steps of 0.002, the tissue read at the target's own readout
                 (iteration 504 for the stripe, 2173 for the face); the other orders are held at their trained values

Reads the replays cached by buildCanalizationTalkFigures11x11.py (run its stages codes and slidingPatterns first), draws the frames with matplotlib and encodes
them with ffmpeg (H.264, yuv420p, so PowerPoint plays them). The default ffmpeg is the one on the path; the module build on the cluster has no H.264 encoder, so
pass --ffmpeg with a build that has libx264.

    python3 buildCanalizationTalkMovies11x11.py --ffmpeg /path/to/ffmpeg

Writes presentation/backup_quantitative/movies/<name>.mp4 (never overwriting unless --overwrite).
"""
import argparse
import os
import shutil
import subprocess

import numpy as np

from canalizationTalkCommon import *

parser = argparse.ArgumentParser()
parser.add_argument('--movies', type=str, default='codeToPattern,slidingDial')
parser.add_argument('--outputDirectory', type=str, default='presentation/backup_quantitative/movies')
parser.add_argument('--cachePath', type=str, default='data/canalizationTalkReplays1888Hold301.npz')
parser.add_argument('--ffmpeg', type=str, default='ffmpeg')
parser.add_argument('--framesDirectory', type=str, default='canalizationTalkFrames')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
cache = dict(np.load(args.cachePath, allow_pickle=True))
WIDTH, HEIGHT, DPI = 12.8, 7.2, 100


def newTissueAxis(figure, rectangle, target, vmem):
    axis = figure.add_axes(rectangle)
    image = axis.imshow(np.asarray(vmem).reshape(LATTICE, LATTICE), cmap=VMEM_MAP, vmin=VMEM_LOW, vmax=VMEM_HIGH, interpolation='nearest')
    for edge in np.arange(-0.5, LATTICE, 1.0):
        axis.axhline(edge, color='white', lw=0.8)
        axis.axvline(edge, color='white', lw=0.8)
    axis.add_patch(Rectangle((0.5, 0.5), LATTICE - 2, LATTICE - 2, fill=False, ec=INK_3, lw=0.9))
    outlineCellSet(axis, target['cells'], OCHRE, 2.2)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0)
    return axis, image


def highlight(axis, colour, on):
    for spine in axis.spines.values():
        spine.set_visible(on)
        spine.set_edgecolor(colour)
        spine.set_linewidth(4.0 if on else 0)


def encode(frameCount, name, framesPerSecond):
    os.makedirs(args.outputDirectory, exist_ok=True)
    path = f'{args.outputDirectory}/{name}.mp4'
    if os.path.exists(path) and not args.overwrite:
        print(f'exists, kept: {path}')
        return
    command = [args.ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(framesPerSecond), '-i', f'{args.framesDirectory}/frame%04d.png', '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', '-crf', '18', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-movflags', '+faststart', path]
    subprocess.run(command, check=True)
    print(f'wrote {path} ({frameCount} frames, {frameCount / framesPerSecond:.1f} s)')


def freshFrames():
    shutil.rmtree(args.framesDirectory, ignore_errors=True)
    os.makedirs(args.framesDirectory)


# =========================================================================================================== from ring code to pattern
if 'codeToPattern' in args.movies:
    courses = {key: cache[f'course_{key}'] for key in ('stripe', 'face')}
    iterations = sorted(set(range(0, 3000, 20)) | {300, 504, 2173})
    plan = []
    for iteration in iterations:
        dwell = 30 if iteration in (504, 2173) else 1
        plan.extend([iteration] * dwell)
    freshFrames()
    figure = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    figure.text(0.5, 0.945, 'A boundary code is held on the ring, then released', ha='center', fontsize=24, fontweight='bold', color=INK)
    status = figure.text(0.5, 0.885, '', ha='center', fontsize=17, color=INK_2)
    panels = {}
    for key, left in (('stripe', 0.07), ('face', 0.53)):
        target = TARGETS[key]
        axis, image = newTissueAxis(figure, [left, 0.20, 0.40, 0.62], target, courses[key][0])
        axis.set_title(f'{target["label"]}: {len(target["coefficients"])} numbers', fontsize=20, fontweight='bold', color=target['colour'], pad=8)
        caption = figure.text(left + 0.20, 0.145, '', ha='center', fontsize=13.5, color=INK_2)
        panels[key] = (axis, image, caption)
    barAxis = figure.add_axes([0.07, 0.045, 0.86, 0.035])
    barAxis.set_xlim(0, 3000)
    barAxis.set_ylim(0, 1)
    barAxis.axvspan(0, HOLD, color=DIAL, alpha=0.85)
    barAxis.axvspan(HOLD, 3000, color='#E4E8EC')
    barAxis.text(HOLD / 2, 0.5, 'held', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    for iteration, label, colour in ((504, 'stripe 504', TEAL), (2173, 'face 2173', OCHRE)):
        barAxis.axvline(iteration, color=colour, lw=2)
        barAxis.text(iteration + 25, 0.5, label, color=colour, va='center', fontsize=12, fontweight='bold')
    barAxis.set_yticks([])
    barAxis.set_xlabel('iteration', fontsize=12)
    for spine in barAxis.spines.values():
        spine.set_visible(False)
    marker = barAxis.axvline(0, color=INK, lw=3)
    for frame, iteration in enumerate(plan):
        status.set_text('ring held: the code writes into the tissue' if iteration < HOLD else 'ring released: the tissue is on its own')
        status.set_color(DIAL if iteration < HOLD else INK_2)
        marker.set_xdata([iteration, iteration])
        for key, (axis, image, caption) in panels.items():
            target = TARGETS[key]
            vmem = courses[key][iteration]
            image.set_data(vmem.reshape(LATTICE, LATTICE))
            inTarget, strays = darkCounts(vmem, target['cells'])
            isReadout = iteration == target['readIteration']
            highlight(axis, target['colour'], isReadout)
            caption.set_text(f'iteration {iteration}:  {inTarget}/{len(target["cells"])} target cells dark, {strays} stray' + ('   ← formed' if isReadout else ''))
            caption.set_color(target['colour'] if isReadout else INK_2)
            caption.set_fontweight('bold' if isReadout else 'normal')
        figure.savefig(f'{args.framesDirectory}/frame{frame:04d}.png', dpi=DPI)
    plt.close(figure)
    encode(len(plan), 'movie_01_codeToPattern', 15)

# =========================================================================================================== sliding the dial
if 'slidingDial' in args.movies:
    offsets = np.round(np.arange(-0.08, 0.08 + 0.001, 0.002), 6)
    patterns = {key: cache[f'slide_{key}_order0_step0.002'] for key in ('stripe', 'face')}
    plan = []
    for index in range(len(offsets)):
        plan.extend([index] * (14 if abs(offsets[index]) < 1e-9 else 1))
    freshFrames()
    figure = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    figure.text(0.5, 0.945, 'Sliding the dial (order 0) away from the trained code', ha='center', fontsize=24, fontweight='bold', color=INK)
    offsetText = figure.text(0.5, 0.885, '', ha='center', fontsize=19, color=INK_2)
    panels = {}
    degrees = np.degrees(RING_ANGLES)
    sortOrder = np.argsort(degrees)
    for key, left in (('stripe', 0.07), ('face', 0.53)):
        target = TARGETS[key]
        axis, image = newTissueAxis(figure, [left, 0.34, 0.40, 0.50], target, patterns[key][0])
        axis.set_title(f'{target["label"]}', fontsize=20, fontweight='bold', color=target['colour'], pad=8)
        caption = figure.text(left + 0.20, 0.285, '', ha='center', fontsize=15, color=INK_2)
        profileAxis = figure.add_axes([left + 0.03, 0.10, 0.34, 0.15])
        trainedProfile = ringValuesOf(target['coefficients'], 2.0)
        profileAxis.plot(degrees[sortOrder], trainedProfile[sortOrder], color='#C9D1D8', lw=2)
        line, = profileAxis.plot(degrees[sortOrder], trainedProfile[sortOrder], color=target['colour'], lw=2.4)
        profileAxis.set_xlim(-185, 185)
        profileAxis.set_ylim(0, 2.0)
        profileAxis.set_xticks([-180, 0, 180])
        profileAxis.set_xticklabels(['bottom', 'top', 'bottom'], fontsize=10)
        profileAxis.tick_params(labelsize=10)
        profileAxis.set_ylabel('held G_pol / G_ref', fontsize=10)
        panels[key] = (axis, image, caption, line)
    for frame, index in enumerate(plan):
        offset = offsets[index]
        offsetText.set_text('trained code' if abs(offset) < 1e-9 else f'a₀ {offset:+.3f} from the trained value')
        offsetText.set_fontweight('bold' if abs(offset) < 1e-9 else 'normal')
        for key, (axis, image, caption, line) in panels.items():
            target = TARGETS[key]
            vmem = patterns[key][index]
            image.set_data(vmem.reshape(LATTICE, LATTICE))
            coefficients = target['coefficients'].copy()
            coefficients[0] += offset
            line.set_ydata(ringValuesOf(coefficients, 2.0)[sortOrder])
            inTarget, strays = darkCounts(vmem, target['cells'])
            highlight(axis, target['colour'], abs(offset) < 1e-9)
            caption.set_text(f'{inTarget}/{len(target["cells"])} target cells dark, {strays} stray')
        figure.savefig(f'{args.framesDirectory}/frame{frame:04d}.png', dpi=DPI)
    plt.close(figure)
    encode(len(plan), 'movie_02_slidingTheDial', 10)

shutil.rmtree(args.framesDirectory, ignore_errors=True)
