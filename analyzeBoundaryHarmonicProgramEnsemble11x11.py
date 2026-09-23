"""Does the reduced program hold across an ensemble of ring codes, and what do the orders steer?
(PolyPatterning_Sim.md, Section 12).

Every code is replayed and reduced to the same small program: the conductance's peak during the hold, its trough
after release, its second peak, how far that second rise separates the face's cells from the rest (selectivity),
whether the cells it lifts highest are the right ones (targeting), and how the darkening events divide into
nucleators and recruits. The script then asks which of those the code's orders steer, and how well the program
predicts the face the run actually reaches.

Face quality is scored at each code's own best moment, not at a fixed iteration: codes reach their second peak
anywhere from iteration 900 to 2,900, and scoring them all at the trained code's moment hides most of the signal.

Writes data/boundaryHarmonicProgramEnsemble<rest of the summary's name> (never overwriting).
"""
import argparse
import itertools
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--recordPath', type=str, required=True, help='npz from the ensemble replay')
parser.add_argument('--slicePath', type=str, default='', help='optional second npz, an ensemble around the trained code')
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--iouStride', type=int, default=5)
args = parser.parse_args()

outputPath = args.summaryPath.replace('boundaryHarmonicTrainingSummary', 'boundaryHarmonicProgramEnsemble')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

interior = np.array(boundary.interiorCellIndices)
featureCells = sorted(set(boundary.featureCellIndices.tolist()))
isFeature = np.isin(interior, featureCells)
targetInterior = isFeature.copy()

neighbourIndex = np.zeros((boundary.numCells, 4), dtype=int)
neighbourMask = np.zeros((boundary.numCells, 4), dtype=bool)
for cell in range(boundary.numCells):
    row, column = divmod(cell, boundary.latticeCols)
    for slot, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        r, c = row + dr, column + dc
        if 0 <= r < boundary.latticeRows and 0 <= c < boundary.latticeCols:
            neighbourIndex[cell, slot], neighbourMask[cell, slot] = r * boundary.latticeCols + c, True

programKeys = ['holdPeak', 'troughTime', 'troughValue', 'peakTime', 'peakValue',
               'selectivity', 'targeting', 'nucleators', 'recruits', 'firstNucleation', 'events']


def reduceEnsemble(path):
    data = np.load(path)
    packed, codes = data['packed'], data['codes']
    interiorMean, featureMean, backgroundMean = data['interiorMean'], data['featureMean'], data['backgroundMean']
    conductanceFrames, stride = data['gpolFrames'], int(data['gpolStride'])
    count = len(codes)
    program = {key: np.zeros(count) for key in programKeys}
    quality = np.zeros(count)
    qualityMoment = np.zeros(count, dtype=int)
    for index in range(count):
        mean = interiorMean[index].astype(float)
        trough = 302 + int(mean[302:1300].argmin())
        peak = trough + int(mean[trough:].argmax())
        bits = np.unpackbits(packed[index], axis=1)[:, :boundary.numCells].astype(bool)

        nucleators = recruits = 0
        firstNucleation = np.nan
        for time, cell in np.argwhere(bits[1201:] != bits[1200:-1]):
            cell = int(cell)
            if not bits[1201 + time, cell] or cell not in set(interior.tolist()):
                continue
            darkAround = int(np.where(neighbourMask[cell], bits[1200 + time, neighbourIndex[cell]], False).sum())
            if darkAround == 0:
                nucleators += 1
                firstNucleation = 1201 + time if np.isnan(firstNucleation) else firstNucleation
            else:
                recruits += 1

        frame = conductanceFrames[index, min(peak // stride, conductanceFrames.shape[1] - 1)][interior]
        program['holdPeak'][index] = mean[:302].max()
        program['troughTime'][index] = trough
        program['troughValue'][index] = mean[trough]
        program['peakTime'][index] = peak
        program['peakValue'][index] = mean[peak]
        program['selectivity'][index] = featureMean[index, peak] - backgroundMean[index, peak]
        program['targeting'][index] = isFeature[np.argsort(frame)[::-1][:len(featureCells)]].sum()
        program['nucleators'][index] = nucleators
        program['recruits'][index] = recruits
        program['firstNucleation'][index] = firstNucleation if not np.isnan(firstNucleation) else 3000
        program['events'][index] = nucleators + recruits

        sampled = bits[301::args.iouStride][:, interior]
        intersection = (sampled & targetInterior).sum(1)
        union = (sampled | targetInterior).sum(1)
        overlap = np.where(union > 0, intersection / np.maximum(union, 1), 0.0)
        bestIndex = int(overlap.argmax())
        quality[index], qualityMoment[index] = overlap[bestIndex], 301 + bestIndex * args.iouStride
        if index % 250 == 0:
            print(f'  {os.path.basename(path)}: code {index}', flush=True)
    return codes, program, quality, qualityMoment


def steering(codes, program, quality):
    standardised = (codes - codes.mean(0)) / codes.std(0)
    numOrders = codes.shape[1]

    def expand(matrix):
        return np.column_stack([matrix] + [matrix[:, a] * matrix[:, b]
                                           for a, b in itertools.combinations_with_replacement(range(matrix.shape[1]), 2)])

    fold = np.random.default_rng(1).permutation(len(codes)) % 5
    shuffled = np.random.default_rng(7).permutation(len(codes))

    def crossValidated(design, values):
        if values.std() < 1e-9:
            return float('nan')
        design = np.column_stack([np.ones(len(design)), design])
        predicted = np.zeros(len(values))
        for f in range(5):
            train = fold != f
            predicted[~train] = design[~train] @ np.linalg.lstsq(design[train], values[train], rcond=None)[0]
        return 1 - ((values - predicted) ** 2).sum() / ((values - values.mean()) ** 2).sum()

    out = {}
    for name, values in list(program.items()) + [('quality', quality)]:
        full = crossValidated(expand(standardised), values)
        out[name] = dict(
            r2=round(float(full), 4),
            shuffled=round(float(crossValidated(expand(standardised[shuffled]), values)), 4),
            dropLoss=[round(float(full - crossValidated(expand(standardised[:, [j for j in range(numOrders) if j != k]]), values)), 4)
                      for k in range(numOrders)],
            alone=[round(float(crossValidated(expand(standardised[:, [k]]), values)), 4) for k in range(numOrders)],
            correlationWithQuality=round(float(np.corrcoef(values, quality)[0, 1]), 4))
    return out


result = dict(featureCells=featureCells, interiorCells=interior.tolist(), programKeys=programKeys, ensembles={})
for label, path in [('scattered', args.recordPath)] + ([('slice', args.slicePath)] if args.slicePath else []):
    codes, program, quality, moment = reduceEnsemble(path)
    result['ensembles'][label] = dict(
        numCodes=len(codes), codes=[[round(float(v), 4) for v in row] for row in codes],
        program={k: [round(float(v), 4) for v in program[k]] for k in programKeys},
        quality=[round(float(v), 4) for v in quality], qualityMoment=[int(v) for v in moment],
        steering=steering(codes, program, quality),
        summary={k: dict(median=round(float(np.median(program[k])), 3), sd=round(float(program[k].std()), 3),
                         low=round(float(np.quantile(program[k], .05)), 3), high=round(float(np.quantile(program[k], .95)), 3))
                 for k in programKeys})
    print(f"  {label}: {len(codes)} codes, best quality {quality.max():.3f}, median {np.median(quality):.3f}", flush=True)

json.dump(result, open(outputPath, 'w'))
print('wrote', outputPath, os.path.getsize(outputPath) // 1024, 'KiB', flush=True)
