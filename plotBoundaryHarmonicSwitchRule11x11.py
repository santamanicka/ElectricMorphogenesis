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
args = parser.parse_args()

if os.path.exists(args.outputPath) and not args.overwrite:
    raise SystemExit(f'{args.outputPath} exists; pass --overwrite to rebuild it')
data = json.load(open(args.dataPath))
data['ensemble'] = json.load(open(args.ensemblePath)) if os.path.exists(args.ensemblePath) else None
interventionPath = 'data/boundaryHarmonicLatchIntervention1888Hold301FaceMinus60Minus5.json'
data['intervention'] = json.load(open(interventionPath)) if os.path.exists(interventionPath) else None
page = open(args.templatePath).read().replace('__DATA__', json.dumps(data, separators=(',', ':')))
if not data['ensemble']:
    import re
    page = re.sub(r'<!--ENSEMBLE-->.*?<!--/ENSEMBLE-->', '', page, flags=re.S)
open(args.outputPath, 'w').write(page)
print(f'wrote {args.outputPath} ({len(page) / 1e6:.2f} MB)')
