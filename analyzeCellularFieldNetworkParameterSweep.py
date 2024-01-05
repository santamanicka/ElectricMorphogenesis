import torch
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

data = torch.load('./data/parameterSweep.dat')

def computeEntropy(vmem):  # vmem should be a 1D tensor
	counts = torch.unique(vmem.round(decimals=2),return_counts=True)[1]
	probabilities = counts/counts.sum()
	H = entropy(probabilities)
	return (H)

clampModes = ['field','tissue']
fieldResolutions = torch.arange(1,11)

allClampModes, allClampProps, allFieldResolutions, allEntropies = [], [], [], []
for index in data:
	record = data[index]
	recClampMode = record['clampMode']
	recFieldResolution = record['fieldResolution']
	recClampProportion = record['clampedCellsPropNorm']
	recVmem = record['Vmem'].flatten()
	H = computeEntropy(recVmem)
	allClampModes.append(recClampMode)
	allClampProps.append(recClampProportion)
	allEntropies.append(H)
	allFieldResolutions.append(recFieldResolution)

allClampProps = np.array(allClampProps)
allEntropies = np.array(allEntropies)

data = dict()
paramCombination = 0
for clampMode in clampModes:
	clampIdx = (allClampModes == clampMode)
	fieldResolutions = allFieldResolutions[clampIdx]
	for fieldResolution in fieldResolutions:
		print(paramCombination)
		data[paramCombination] = dict()
		fieldResIdx = (allFieldResolutions == fieldResolution)
		uprops = np.unique(allClampProps[clampIdx & fieldResIdx])
		complexity = [allEntropies[allClampProps == uprops[i]].mean() for i in range(len(uprops))]
		data[paramCombination]['clampMode'] = clampMode
		data[paramCombination]['fieldResolution'] = fieldResolution
		data[paramCombination]['uprops'] = uprops
		data[paramCombination]['complexity'] = complexity
		paramCombination += 1

torch.save(data,'./data/parameterSweepAnalysis.dat')