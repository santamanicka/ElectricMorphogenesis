"""Build the "Free or New?" report page (PolyPatterning_Sim.md, Section 12): whether the boundary dial's patterns are the
free tissue's own patterns at another time, checked snapshot by snapshot.

Inputs: data/boundaryRecurrenceSummary1888Hold301.json (analyzeBoundaryRecurrence11x11.py, from the trajectories of
simulateBoundaryRecurrence11x11.py) and data/boundaryTiltRecurrenceSummary1888Hold301.json (analyzeBoundaryTiltRecurrence11x11.py,
the tilted dial) and data/boundaryTiltThreeWaySummary1888Hold301.json (analyzeBoundaryTiltThreeWay11x11.py, three ways at
0 degrees) and data/boundaryTiltStartRecurrenceSummary1888Hold301.json (analyzeBoundaryTiltStartRecurrence11x11.py,
tilted starts and release states). The page's JSON keys are the ones figures/boundaryRecurrenceTemplate.html reads.
"""
import argparse
import json

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--summaryPath', type=str, default='data/boundaryRecurrenceSummary1888Hold301.json')
parser.add_argument('--tiltSummaryPath', type=str, default='data/boundaryTiltRecurrenceSummary1888Hold301.json')
parser.add_argument('--threeWaySummaryPath', type=str, default='data/boundaryTiltThreeWaySummary1888Hold301.json')
parser.add_argument('--tiltStartSummaryPath', type=str, default='data/boundaryTiltStartRecurrenceSummary1888Hold301.json')
parser.add_argument('--templatePath', type=str, default='figures/boundaryRecurrenceTemplate.html')
parser.add_argument('--outputPath', type=str, default='figures/boundaryRecurrence.html')
args = parser.parse_args()

summary = json.load(open(args.summaryPath))
# the tilted starts and release states, without the per-iteration traces and starting states the page does not draw
tiltStart = json.load(open(args.tiltStartSummaryPath))
tiltStart.pop('starts')
for row in tiltStart['rows']:
    for variant in ('RingGpol', 'RingAll', 'Nudge'):
        row['release'][variant].pop('sameTime')
# the control's distance from the free run at the same iteration, per 100 iterations: median and largest
directory = f"data/boundaryRecurrence{summary['referenceCheckpoint']}Hold{summary['holdIterations']}"
free, control = (np.load(f'{directory}/{name}.npz')['vmem'].astype(np.float64) for name in ('free', 'freeShifted'))
sameTime = np.sqrt(((control - free[:len(control)]) ** 2).mean(1)).reshape(-1, 100)
summary['calibration']['controlSameTime'] = dict(times=list(range(0, len(control), 100)), median=np.round(np.median(sameTime, 1), 4).tolist(),
                                                 largest=np.round(sameTime.max(1), 4).tolist())
dialNames = sorted((name for name in summary['runs'] if name.startswith('dial')), key=lambda name: float(name[4:]))
randomNames = sorted((name for name in summary['runs'] if name.startswith('freeRandom')), key=lambda name: int(name[10:]))
payload = dict(
    hold=summary['holdIterations'], freeIterations=summary['freeIterations'], runIterations=summary['runIterations'],
    exclusion=summary['exclusionIterations'], tolerancePercentile=summary['tolerancePercentile'], dialNames=dialNames, randomNames=randomNames,
    calibration=summary['calibration'], checks=summary['checks'], frames=summary['frames'], closest=summary['closest'],
    cross=summary['cross'], averages=summary['averages'], traces=summary['traces'], randomStarts=summary['randomStarts'],
    runs={name: {key: value for key, value in entry.items() if key != 'lateLags'} for name, entry in summary['runs'].items()},
    tilt=json.load(open(args.tiltSummaryPath)),
    tiltStart=tiltStart,
    threeWay={key: value for key, value in json.load(open(args.threeWaySummaryPath)).items() if key not in ('groups', 'subgroups')})
page = open(args.templatePath).read().replace('__DATA__', json.dumps(payload, separators=(',', ':')))
open(args.outputPath, 'w').write(page)
print(f"wrote {args.outputPath} ({len(page) / 1e6:.2f} MB)")
