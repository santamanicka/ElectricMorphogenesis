"""Movies of the best-overlap cases of the rejuvenation search (EXPLORATORY; the cases are those shown in figures/canalizationTalkRejuvenationBest.png): how the tissue evolves from the
start, through the original code's pattern and its fading, to the searched code held on the aged tissue and its best moment.

Each movie replays, in 64-bit with the pure step, (1) the target's trained code held for 301 iterations and released, up to the aged state of the search (best moment + offset, as in
makeAgedTissueState11x11.py), then (2) the searched code of the best-overlap restart held for its hold length from that state and released, for 3,000 iterations (the search's own run).
The clock counts iterations from the uniform start of the original run: the new hold begins at the age of the aged state, and the new best moment is at age + the stored best iteration.
Left: membrane voltage (white depolarised, dark hyperpolarised; the target outlined); middle: polarising conductance G_pol / G_ref of every cell (the ring cells show the code while it is
held; the interior shows the tissue's slow memory); bottom: the overlap of the dark interior cells with the target (the search's measure) with the phases shaded.

    python3 buildCanalizationTalkRejuvenationMovies11x11.py --ffmpeg <ffmpeg with libx264>

Writes figures/movies/rejuvenationBestOverlap<Case>.mp4 (never overwriting unless --overwrite).
"""
import argparse
import os
import subprocess

import numpy as np
import torch

import boundaryCodeUtilities as boundary
import canalizationTalkCommon as common
from canalizationTalkCommon import *

parser = argparse.ArgumentParser()
parser.add_argument('--cases', type=str, default='stripe,faceOrder3,faceOrder4')
parser.add_argument('--ffmpeg', type=str, default='ffmpeg')
parser.add_argument('--outputDirectory', type=str, default='figures/movies')
parser.add_argument('--stride', type=int, default=10, help='iterations between frames')
parser.add_argument('--framesPerSecond', type=int, default=30)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

DATA = 'data/boundaryHarmonicRejuvenationTraining1888Hold301'
CASES = {
    'stripe': dict(label='Stripe', title='stripe', offset=100, suffix='', restart=1, order=2, hold=301, cells=boundary.centreStripeCellIndices, colour=TEAL, name='Stripe'),
    'faceOrder3': dict(label='FaceOrder3', title='face, order 3', offset=300, suffix='LongHold1000', restart=11, order=3, hold=1000, cells=boundary.featureCellIndices, colour=OCHRE, name='FaceOrder3'),
    'faceOrder4': dict(label='FaceOrder4', title='face, order 4', offset=300, suffix='', restart=0, order=4, hold=301, cells=boundary.featureCellIndices, colour=OCHRE, name='FaceOrder4'),
}
WIDTH, HEIGHT, DPI = 12.8, 7.2, 100
ITERATIONS = 3000                                                                  # the search's run after the start of the new hold


def replay(case):
    """(voltage mV, G_pol / G_ref) after every iteration of the original run to the aged state, then of the new hold and release; the codes; the stored run."""
    state = np.load(f'data/agedTissueState1888Hold301{case["label"]}After{case["offset"]}.npz', allow_pickle=True)
    run = np.load(f'{DATA}{case["label"]}After{case["offset"]}{case["suffix"]}/order{case["order"]}_restart{case["restart"]:02d}.npz', allow_pickle=True)
    ceiling, age = float(state['ceiling']), int(state['age'])
    with boundary.fullDoublePrecision(True):
        step = common.Replayer().step
        step.ringMask = torch.zeros(step.numCells, dtype=torch.double)
        step.ringMask[torch.as_tensor(RING_CELLS, dtype=torch.long)] = 1.0

        def setCode(coefficients):
            code = torch.zeros(step.numCells, dtype=torch.double)
            code[step.ring] = torch.as_tensor(ringValuesOf(coefficients, ceiling), dtype=torch.double) * step.Gref
            step.ringCode = code

        vmem, gpol = step.initialVmem.clone(), step.initialGpol.clone()
        voltage, conductance = [], []
        setCode(state['coefficients'])
        for iteration in range(age):
            vmem, gpol = step(vmem, gpol, iteration < HOLD, iteration < HOLD)
            voltage.append(vmem.numpy() * 1000.0)
            conductance.append(gpol.numpy() / step.Gref)
        assert np.allclose(vmem.numpy(), state['vmemVolts'], atol=1e-9) and np.allclose(gpol.numpy(), state['gpol'], atol=1e-9), 'the replay does not reach the stored aged state'
        setCode(run['bestCoefficients'])
        for iteration in range(ITERATIONS):
            vmem, gpol = step(vmem, gpol, iteration < case['hold'], iteration < case['hold'])
            voltage.append(vmem.numpy() * 1000.0)
            conductance.append(gpol.numpy() / step.Gref)
    best = age + int(run['bestIteration'])
    # the replay must reproduce the pattern the search stored at its best moment (float32 in the file)
    assert np.allclose(voltage[best], np.asarray(run['bestVmem'], float), atol=2e-3), 'the replay does not reproduce the stored best pattern'
    return np.array(voltage), np.array(conductance), age, int(state['bestMoment']), best, run


def encode(frames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    command = [args.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{WIDTH * DPI:.0f}x{HEIGHT * DPI:.0f}', '-r', str(args.framesPerSecond), '-i', '-',
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', path]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


for key in args.cases.split(','):
    case = CASES[key]
    path = f'{args.outputDirectory}/rejuvenationBestOverlap{case["name"]}.mp4'
    if os.path.exists(path) and not args.overwrite:
        print(f'exists, kept: {path}')
        continue
    voltage, conductance, age, originalBest, newBest, run = replay(case)
    cells = np.asarray(case['cells'])
    total = len(voltage)
    overlap = np.array([boundary.structuralIntersectionOverUnion(v, cells) for v in voltage])
    newHoldEnd = age + case['hold']
    # frames: every `stride` iterations, with pauses at the original best moment, the start of the new hold and the new best moment
    dwell = {originalBest: 45, age: 30, newBest: 90}
    plan = []
    for clock in sorted(set(range(0, total, args.stride)) | set(dwell) | {total - 1}):
        plan.extend([clock] * dwell.get(clock, 1))
    plan.extend([total - 1] * 30)
    print(f'{key}: {total} iterations, aged state at {age}, original best moment {originalBest}, new best moment {newBest} (overlap {overlap[newBest]:.2f}); {len(plan)} frames')

    figure = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    figure.text(0.5, 0.945, f'{case["title"]}: a code searched on the aged tissue', ha='center', fontsize=23, fontweight='bold', color=case['colour'])
    status = figure.text(0.5, 0.885, '', ha='center', fontsize=17, fontweight='bold')
    clockText = figure.text(0.5, 0.848, '', ha='center', fontsize=14, color=INK_2)

    def panel(rectangle, colourMap, low, high, title):
        axis = figure.add_axes(rectangle)
        image = axis.imshow(np.zeros((LATTICE, LATTICE)), cmap=colourMap, vmin=low, vmax=high, interpolation='nearest')
        for edge in np.arange(-0.5, LATTICE, 1.0):
            axis.axhline(edge, color='white', lw=0.8)
            axis.axvline(edge, color='white', lw=0.8)
        axis.add_patch(Rectangle((0.5, 0.5), LATTICE - 2, LATTICE - 2, fill=False, ec=INK_3, lw=0.9))
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(title, fontsize=15, color=INK_2, pad=6)
        return axis, image

    voltageAxis, voltageImage = panel([0.07, 0.30, 0.40, 0.47], VMEM_MAP, VMEM_LOW, VMEM_HIGH, 'membrane voltage (target outlined)')
    outlineCellSet(voltageAxis, cells, case['colour'], 2.2)
    conductanceAxis, conductanceImage = panel([0.53, 0.30, 0.40, 0.47], CODE_MAP, 0.0, 2.0, 'polarising conductance G_pol / G_ref')

    trace = figure.add_axes([0.07, 0.075, 0.86, 0.175])
    trace.set_xlim(0, total)
    trace.set_ylim(0, 1.32)
    bands = [(0, HOLD, DIAL, 'original code held'), (HOLD, age, '#E4E8EC', 'released: forms, fades, ages'), (age, newHoldEnd, DIAL, 'new code held'), (newHoldEnd, total, '#E4E8EC', 'released')]
    for left, right, colour, text in bands:
        trace.axvspan(left, right, color=colour, alpha=0.85 if colour == DIAL else 1.0, lw=0)
        if colour == DIAL:
            trace.text((left + right) / 2, 1.30, 'held', ha='center', va='top', fontsize=10.5, color='white', fontweight='bold')
    trace.axvline(originalBest, color=case['colour'], lw=1.6, ls=(0, (4, 3)))
    trace.text(originalBest - 0.006 * total, 1.12, 'original best', fontsize=9.5, color=case['colour'], va='center', ha='right')
    trace.axvline(newBest, color=GREEN, lw=1.6, ls=(0, (4, 3)))
    trace.text(newBest + 0.006 * total, 1.12, 'new best', fontsize=9.5, color=GREEN, va='center', ha='left')
    trace.plot(np.arange(total), overlap, color=INK_3, lw=1.0, alpha=0.35)
    revealed, = trace.plot([], [], color=INK, lw=2.2)
    cursor = trace.axvline(0, color=INK, lw=2.4)
    trace.set_ylabel('overlap with\nthe target', fontsize=11)
    trace.set_xlabel('iterations since the start of the original run', fontsize=11)
    trace.set_yticks([0, 0.5, 1.0])
    for spine in ('top', 'right'):
        trace.spines[spine].set_visible(False)

    process = encode(None, path)
    for clock in plan:
        voltageImage.set_data(voltage[clock].reshape(LATTICE, LATTICE))
        conductanceImage.set_data(conductance[clock].reshape(LATTICE, LATTICE))
        if clock < HOLD:
            text, colour = 'original code held on the ring', DIAL
        elif clock < age:
            text, colour = 'original code released: the tissue is on its own', INK_2
        elif clock < newHoldEnd:
            text, colour = 'searched code held on the ring of the aged tissue', DIAL
        else:
            text, colour = 'searched code released: the tissue is on its own', INK_2
        status.set_text(text)
        status.set_color(colour)
        note = f'iteration {clock}'
        if clock >= age:
            note += f'   (new hold began at {age}; {clock - age} since)'
        if clock == originalBest:
            note += f'   original best moment, overlap {overlap[clock]:.2f}'
        if clock == age:
            note += f'   aged tissue, overlap {overlap[clock]:.2f}'
        if clock == newBest:
            note += f'   new best moment, overlap {overlap[clock]:.2f}'
        clockText.set_text(note)
        revealed.set_data(np.arange(clock + 1), overlap[:clock + 1])
        cursor.set_xdata([clock, clock])
        figure.canvas.draw()
        process.stdin.write(np.asarray(figure.canvas.buffer_rgba()).tobytes())
    process.stdin.close()
    process.wait()
    plt.close(figure)
    print(f'wrote {path} ({len(plan)} frames, {len(plan) / args.framesPerSecond:.1f} s)')
