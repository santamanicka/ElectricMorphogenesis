import torch
from scipy.stats import entropy
import matplotlib.pyplot as plt

data = torch.load('./data/parameterSweep.dat')

def computeEntropy(vmem):  # vmem should be a 1D tensor
	counts = torch.unique(vmem.round(decimals=2),return_counts=True)[1]
	probabilities = counts/counts.sum()
	H = entropy(probabilities)
	return (H)

clampMode = 'field'

allClampProps, allEntropies = [], []
for record in data:
	if record['clampMode'] == clampMode:
		clampProportion = record['clampedCellsProp']
		vmem = record['vmem'].flatten()
		H = computeEntropy(vmem)
		allClampProps.append(clampProportion)
		allEntropies.append(H)

plt.plot(allClampProps,allEntropies)
plt.show()
