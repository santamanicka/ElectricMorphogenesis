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

clampMode = 'tissue'
fieldResolution = 1

allClampProps, allEntropies = [], []
for index in data:
	record = data[index]
	recClampMode = record['clampMode']
	if recClampMode == clampMode:
		if (recClampMode == 'field' and record['fieldResolution'] == fieldResolution) or (recClampMode == 'tissue'):
			clampProportion = record['clampedCellsProp']
			vmem = record['Vmem'].flatten()
			H = computeEntropy(vmem)
			allClampProps.append(clampProportion)
			allEntropies.append(H)

uprops = np.unique(np.array(allClampProps))

complexity = [np.array(allEntropies)[allClampProps == uprops[i]].mean() for i in range(len(uprops))]

plt.plot(uprops,complexity)
plt.show()

# plt.plot(allClampProps,allEntropies)
# plt.show()
