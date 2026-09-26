"""Build the switch-rule page from its template and the analysis JSON."""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', type=str, default='data/boundaryHarmonicSwitchRule1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--ensemblePath', type=str, default='data/boundaryHarmonicProgramEnsemble1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--templatePath', type=str, default='figures/boundaryHarmonicSwitchRuleTemplate.html')
parser.add_argument('--outputPath', type=str, default='figures/boundaryHarmonicSwitchRule.html')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--includeRelay', action='store_true',
                    help='draw the relay section; off until its results are written up')
parser.add_argument('--includeCoarse', action='store_true',
                    help='draw the coarse-graining section; off until its results are written up')
parser.add_argument('--includeStory', action='store_true',
                    help='draw the causal-story section and its additions to the phase cards; off until its results are written up')
args = parser.parse_args()

if os.path.exists(args.outputPath) and not args.overwrite:
    raise SystemExit(f'{args.outputPath} exists; pass --overwrite to rebuild it')
data = json.load(open(args.dataPath))
data['ensemble'] = json.load(open(args.ensemblePath)) if os.path.exists(args.ensemblePath) else None
interventionPath = 'data/boundaryHarmonicLatchIntervention1888Hold301FaceMinus60Minus5.json'
data['intervention'] = json.load(open(interventionPath)) if os.path.exists(interventionPath) else None
clampPath = 'data/boundaryHarmonicClampContribution1888Hold301FaceMinus60Minus5.json'
data['clamp'] = json.load(open(clampPath)) if os.path.exists(clampPath) else None
steeringPath = 'data/boundaryHarmonicSteeringSweep1888Hold301FaceMinus60Minus5.json'
data['steering'] = json.load(open(steeringPath)) if os.path.exists(steeringPath) else None
recruitmentPath = 'data/boundaryHarmonicRecruitmentNecessity1888Hold301FaceMinus60Minus5.json'
data['recruitment'] = json.load(open(recruitmentPath)) if os.path.exists(recruitmentPath) else None
aggregatePath = 'data/boundaryHarmonicAggregateNucleation1888Hold301FaceMinus60Minus5.json'
data['aggregate'] = json.load(open(aggregatePath)) if os.path.exists(aggregatePath) else None
couplingPath = 'data/boundaryHarmonicFieldCoupling1888Hold301FaceMinus60Minus5.json'
data['coupling'] = json.load(open(couplingPath)) if os.path.exists(couplingPath) else None
setPointPath = 'data/boundaryHarmonicSetPoint1888Hold301FaceMinus60Minus5.json'
data['setPoint'] = json.load(open(setPointPath)) if os.path.exists(setPointPath) else None
relayPath = 'data/boundaryHarmonicRingOnlyRelay1888Hold301FaceMinus60Minus5.json'
data['relay'] = json.load(open(relayPath)) if (args.includeRelay and os.path.exists(relayPath)) else None
resolutionsPath = 'data/boundaryHarmonicMovieResolutions1888Hold301FaceMinus60Minus5.json'
if data['relay'] and data['relay'].get('movie') and os.path.exists(resolutionsPath):
    movieResolutions = json.load(open(resolutionsPath))                 # the movie's frames re-cut over blocks of cells
    data['relay']['movie']['resolutions'] = movieResolutions['resolutions']


def coarseReportData():
    """What the coarse-graining section draws, from the committed analysis JSONs, without their per-partition matrices."""
    name = 'data/boundaryHarmonic{}1888Hold301FaceMinus60Minus5.json'
    registered = json.load(open(name.format('CoarseGrain')))
    exploratory = json.load(open(name.format('CoarseGrainExploratory')))
    searchAll = json.load(open(name.format('CoarseGrainSearchAllReadouts')))
    searchSelectivity = json.load(open(name.format('CoarseGrainSearchSelectivity')))
    wiring = json.load(open(name.format('WallWiring')))
    counterfactual = json.load(open(name.format('WallCounterfactual')))
    def flat(entry):
        return dict(name=entry['name'], m=entry['m'], gapError=entry['gapError'], curveError=entry['curveError'],
                    netRetained=entry['netRetained']['both'], stateFloor=entry['stateFloorError'], sourceFloor=entry['sourceFloorError'],
                    fromRelease=entry['fromRelease'], cosine=entry['pathwayCosine'], featureGapError=entry['featureGapError'],
                    resolvedReadout=entry['resolvedReadout'], links90=entry['links90']['total'])
    squares = [dict(flat(entry), size=entry['size'], canonical=entry['canonical']) for entry in registered['squares']]
    patterns = []
    for size in range(1, 7):
        info = registered['bySize'][str(size)]
        if size == 1:
            patterns.append(dict(size=1, m=121, tilings=1, labels=list(range(121)), exact=registered['fineFinalFlux'],
                                 closed=registered['fineFinalFlux'], gap=registered['fineGap']['selectivity']))
            continue
        detail = registered['details'][info['best']]
        best = next(entry for entry in registered['squares'] if entry['name'] == info['best'])
        patterns.append(dict(size=size, m=info['m'], tilings=info['tilings'], name=info['best'], labels=detail['labels'],
                             exact=detail['fluxAggregated'][-1], closed=detail['flux'][-1], gap=best['finalGap']['selectivity']))
    searchBest = searchAll['summary']['strictBar']['smallest'] or searchAll['summary']['registeredBar']['smallest']
    if searchBest:
        detail = searchAll['details'][f'search_{searchBest}']
        record = next(r for r in searchAll['path'] if r['m'] == searchBest)
        patterns.append(dict(size='search', m=searchBest, tilings=0, labels=detail['labels'], exact=detail['fluxAggregated'][-1],
                             closed=detail['flux'][-1], gap=record['gapError'] and detail['curve'][-1]))

    def searchPath(result):
        return dict(summary=result['summary'], path=[dict(m=r['m'], gapError=r['gapError'], curveError=r['curveError'],
                                                        fromRelease=r['fromRelease'], featureGapError=r['featureGapError'],
                                                        cosine=r['pathwayCosine'], netRetained=r['netRetained']) for r in result['path']])
    keep = ('display',)
    return dict(
        fineGap=registered['fineGap']['selectivity'], ringCells=registered['ringCells'], predictions=registered['predictions'],
        verdicts=registered['verdicts'], decision=registered['decision'], bySize=registered['bySize'],
        randomControl=registered['randomControl'], squares=squares, patterns=patterns,
        wallFamily=[flat(entry) for entry in registered['wallFamily']],
        named=[flat(entry) for entry in exploratory['named']],
        clusters=[dict(flat(entry), features=entry['features']) for entry in exploratory['clusters']],
        pod=exploratory['pod'], pruning=exploratory['pruning'], exploratorySummary=exploratory['summary'],
        searchAll=searchPath(searchAll), searchSelectivity=searchPath(searchSelectivity),
        wiring=dict(ringCells=wiring['ringCells'], contribution=wiring['contribution'], injectionConductance=wiring['injectionConductance'],
                    injectionVoltage=wiring['injectionVoltage'], dent=wiring['dent'], retainedByCount=wiring['retainedByCount'],
                    singularValues=wiring['singularValues'], columnTotals=wiring['columnTotals'], gap=wiring['gap'],
                    segments=wiring['display']['8']['segments'], segmentShares=wiring['display']['8']['contribution'],
                    segmentWiring=wiring['display']['8']['wiring'], groupNames=wiring['groupNames'][1:],
                    twoSegments=dict(segments=wiring['display']['2']['segments'], shares=wiring['display']['2']['contribution'],
                                     wiring=wiring['display']['2']['wiring'])),
        counterfactual={k: counterfactual[k] for k in ('predictions', 'baselineSelectivity', 'segments', 'exactShare', 'heldAlone',
                                                       'heldEverythingElse', 'fullRing', 'upperWall', 'lowerWall', 'singleCells',
                                                       'halfSplits', 'sampledStates', 'timeCourses', 'groupOutcomes', 'verdicts')})


def storyReportData():
    """What the causal-story section draws: the four main runs' time courses (every 5 states), each run's outcome, the release scan,
    every perturbed run's second-peak lead and finished pattern, and both registrations with their verdicts."""
    name = 'data/boundaryHarmonic{}1888Hold301FaceMinus60Minus5.json'
    causal = json.load(open(name.format('CausalStory')))
    robust = json.load(open(name.format('CausalStoryRobustness')))
    states = causal['sampledStates']
    thin = lambda values: list(values)
    courses = {}
    for run, course in causal['courses'].items():
        courses[run] = dict(dark={group: thin(values) for group, values in course['dark'].items()},
                            level={group: thin(course['groupG'][group]) for group in ('eyes', 'background')})
        if course['parts']:
            courses[run]['parts'] = {group: thin(values) for group, values in course['parts'].items()}
    kinds = ['code noise', 'release time', 'random half', 'arc of twenty']
    points = ([[0, round(r['selectivity'], 4), round(r['overlap'], 4)] for sigma, draws in robust['jitter'].items() if float(sigma) > 0 for r in draws]
              + [[1, round(r['selectivity'], 4), round(r['overlap'], 4)] for r in robust['releaseScan'].values()]
              + [[2, round(r['selectivity'], 4), round(r['overlap'], 4)] for r in robust['randomSubsets']]
              + [[3, round(r['selectivity'], 4), round(r['overlap'], 4)] for r in robust['arcs']])
    jitter = {sigma: dict(selectivity=[round(r['selectivity'], 4) for r in draws], overlap=[round(r['overlap'], 4) for r in draws],
                          trough=[r['trough'] for r in draws]) for sigma, draws in robust['jitter'].items()}
    outcome = lambda r: dict(selectivity=r['selectivity'], trough=r['trough'], overlap=r['overlap'], dark=r['dark'])
    firstPeak = {}
    for run, course in causal['courses'].items():
        upToHold = [(value, state) for value, state in zip(course['interiorG'], states) if state <= 400]
        value, state = max(upToHold)
        firstPeak[run] = dict(state=state, value=value)
    return dict(
        states=thin(states), courses=courses, results=causal['results'], firstPeak=firstPeak,
        releaseScan=[dict(release=int(k), **outcome(v)) for k, v in sorted(robust['releaseScan'].items(), key=lambda kv: int(kv[0]))],
        baseline=outcome(robust['baseline']), fullRing=outcome(robust['fullRing']), kinds=kinds, points=points, jitter=jitter,
        subsetTroughs=[r['trough'] for r in robust['randomSubsets']], arcTroughs=[r['trough'] for r in robust['arcs']],
        causalPredictions=causal['predictions'], causalVerdicts=causal['verdicts'],
        robustPredictions=robust['predictions'], robustVerdicts=robust['verdicts'])


coarsePath = 'data/boundaryHarmonicCoarseGrainSearchAllReadouts1888Hold301FaceMinus60Minus5.json'
storyPath = 'data/boundaryHarmonicCausalStoryRobustness1888Hold301FaceMinus60Minus5.json'
data['story'] = storyReportData() if (args.includeStory and os.path.exists(storyPath)) else None
data['coarse'] = coarseReportData() if (args.includeCoarse and os.path.exists(coarsePath)) else None
page = open(args.templatePath).read().replace('__DATA__', json.dumps(data, separators=(',', ':')))
if not data['ensemble']:
    import re
    page = re.sub(r'<!--ENSEMBLE-->.*?<!--/ENSEMBLE-->', '', page, flags=re.S)
if not data['steering']:
    import re
    page = re.sub(r'<!--STEER-->.*?<!--/STEER-->', '', page, flags=re.S)
if not data['aggregate']:
    import re
    page = re.sub(r'<!--AGGREGATE-->.*?<!--/AGGREGATE-->', '', page, flags=re.S)
if not data['setPoint']:
    import re
    page = re.sub(r'<!--SETPOINT-->.*?<!--/SETPOINT-->', '', page, flags=re.S)
if not data['relay']:
    import re
    page = re.sub(r'<!--RELAY-->.*?<!--/RELAY-->', '', page, flags=re.S)
if not data['coarse']:
    import re
    page = re.sub(r'<!--COARSE-->.*?<!--/COARSE-->', '', page, flags=re.S)
if not data['story']:
    import re
    page = re.sub(r'<!--STORY-->.*?<!--/STORY-->', '', page, flags=re.S)
if not data['clamp']:
    import re
    page = re.sub(r'<!--CLAMP-->.*?<!--/CLAMP-->', '', page, flags=re.S)
open(args.outputPath, 'w').write(page)
print(f'wrote {args.outputPath} ({len(page) / 1e6:.2f} MB)')
