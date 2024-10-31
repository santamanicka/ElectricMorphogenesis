import numpy as np
import torch
import dit

def computeTSEComplexity(data):
	numCells = data.shape[1]
	cellIndicesAll = np.array(range(numCells))
	totalSubScalesComplexity = 0
	scales = np.linspace(2,numCells-1,50,dtype=np.int16)
	for scale in scales:
		totalComplexityScale = 0
		for subsetsample in range(100):
			cellIndicesSubset = np.random.choice(cellIndicesAll,scale,replace=False)
			states = data[:,cellIndicesSubset]
			uniquestates, countsstates = np.unique(states,axis=0,return_counts=True)
			probsstates = countsstates / sum(countsstates)
			statestr = [''.join(str(bit) for bit in state) for state in uniquestates]
			distrdict = dict(zip(statestr,probsstates))
			distr = dit.Distribution(distrdict)
			entropy = dit.multivariate.entropy(distr)
			totalComplexityScale += entropy
		totalComplexityScale /= 100
		totalSubScalesComplexity += totalComplexityScale
	states = data[:,cellIndicesAll]
	uniquestates, countsstates = np.unique(states,axis=0,return_counts=True)
	probsstates = countsstates / sum(countsstates)
	statestr = [''.join(str(bit) for bit in state) for state in uniquestates]
	distrdict = dict(zip(statestr,probsstates))
	distr = dit.Distribution(distrdict)
	fullScaleComplexity = dit.multivariate.entropy(distr)
	TSEComplexity = totalSubScalesComplexity - (np.sum(scales) * fullScaleComplexity / numCells)
	return (TSEComplexity)

Sfx = 'FixedNone_FieldVector_'
fileRange = range(1,626)
allsavedata = {}
for fileNumber in fileRange:
	filename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + '.dat'
	data = torch.load(filename)
	_, VmemToVmem = data['characteristics']['Sensitivity']['Derivatives']
	VmemToVmem = VmemToVmem.abs().clone()
	nzidx = np.array([VmemToVmem[i].any().item() for i in range(VmemToVmem.shape[0])])
	if nzidx.any():
		VmemToVmem = VmemToVmem[nzidx]  # shape = (numTimePoints,numSources,numTargets)
	mx = VmemToVmem.amax(1, keepdim=True)  # max per time per target variable
	VmemToVmem = VmemToVmem / mx
	VmemToVmem[torch.isnan(VmemToVmem)] = 0.0
	VmemToVmem = VmemToVmem.sum(2)  # shape = (numTimePoints,numSources)
	centers = VmemToVmem.median(dim=0).values  # time wise medians
	# binarize data
	VmemToVmem[VmemToVmem < centers] = 0
	VmemToVmem[VmemToVmem >= centers] = 1  # shape = (numTimePoints,numSources)
	VmemToVmem = VmemToVmem.detach().numpy()
	TSEComplexity = computeTSEComplexity(VmemToVmem)
	allsavedata['filenumber'] = fileNumber
	savedata = {}
	savedata['GJStrength'] = data['GJParameters']['GJStrength']
	savedata['fieldScreenSize'] = data['fieldParameters']['fieldScreenSize']
	savedata['fieldTransductionWeight'] = data['fieldParameters']['fieldTransductionWeight']
	savedata['fieldTransductionBias'] = data['fieldParameters']['fieldTransductionBias']
	savedata['TSEComplexity'] = TSEComplexity
	allsavedata[fileNumber] = savedata
savefilename = './data/modelCharacteristics_' + Sfx + 'SensitivityTSEComplexity.dat'
torch.save(allsavedata, savefilename)
