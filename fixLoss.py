# convert .sum().sqrt() loss to .mean().sqrt()

import torch
import numpy as np
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--fileRange', type=str, default='(0,0)')

args = parser.parse_args()
fileRange = ast.literal_eval(args.fileRange)
fileRange = list(range(fileRange[0],fileRange[1]))

for fileNumber in fileRange:
	fname = './data/bestModelParameters_' + str(fileNumber) + '.dat'
	data = torch.load(fname)
	evalDurationProp = data['trainParameters']['evalDurationProp']
	numSimIters = data['simParameters']['numSimIters']
	numSimEvalIters = evalDurationProp * numSimIters
	correctedBestLossHistory = np.array(data['trainParameters']['bestLossHistory'])
	for row in range(len(correctedBestLossHistory)):
		_, bestLoss = correctedBestLossHistory[row]
		correctedBestLoss = np.sqrt(((bestLoss**2)/numSimEvalIters)).tolist()
		correctedBestLossHistory[row][1] = correctedBestLoss
	data['trainParameters']['bestLossHistory'] = correctedBestLossHistory
	bestLoss = np.array(data['trainParameters']['bestLoss'])
	correctedBestLoss = np.sqrt(((bestLoss**2)/numSimEvalIters)).tolist()
	data['trainParameters']['bestLoss'] = correctedBestLoss
	torch.save(data,fname)
