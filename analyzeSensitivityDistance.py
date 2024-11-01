import numpy as np
import torch
import utilities

def computeCausalDistance(data):
    xc, yc = utils.computeCellularCoordinates(latticeDims,cell_radius)
    cellularCoordinates = (xc.reshape(1,-1),yc.reshape(1,-1))
    dists = utils.computePairwiseDistances(cellularCoordinates,cellularCoordinates).numpy()  # shape = (1,numCells,numCells)
    distsTargets = dists[:,:,targetVariables]
    causalDistanceMatrix = data * distsTargets  # shape = (numTimePoints,numSources,numTargets)
    # causalDistance = causalDistanceMatrix.mean()
    causalDistanceVariance = causalDistanceMatrix.var(1).mean()  # variance per target variable averaged over time
    return causalDistanceVariance

cell_radius = 5.0e-6
targetVariables = [0,5,60]  # representative points
utils = utilities.utilities()
Sfx = 'FixedNone_FieldVector_'
fileRange = range(1,626)
allsavedata = {}
for fileNumber in fileRange:
    print("Filenumber = ",fileNumber)
    filename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + '.dat'
    data = torch.load(filename)
    latticeDims = data['latticeDims']
    numCells = np.prod(latticeDims)
    _, VmemToVmem = data['characteristics']['Sensitivity']['Derivatives']
    VmemToVmem = VmemToVmem.abs().clone()
    nzidx = np.array([VmemToVmem[i].any().item() for i in range(VmemToVmem.shape[0])])
    if nzidx.any():
        VmemToVmem = VmemToVmem[nzidx]  # shape = (numTimePoints,numSources,numTargets)
    mx = VmemToVmem.amax(1,keepdim=True)  # max per time per target variable
    VmemToVmem = VmemToVmem / mx
    VmemToVmem[torch.isnan(VmemToVmem)] = 0.0  # shape = (numTimePoints,numSources,numTargets)
    VmemToVmem = VmemToVmem.detach().numpy()
    causalDistance = computeCausalDistance(VmemToVmem)
    allsavedata['filenumber'] = fileNumber
    savedata = {}
    savedata['GJStrength'] = data['GJParameters']['GJStrength']
    savedata['fieldScreenSize'] = data['fieldParameters']['fieldScreenSize']
    savedata['fieldTransductionWeight'] = data['fieldParameters']['fieldTransductionWeight']
    savedata['fieldTransductionBias'] = data['fieldParameters']['fieldTransductionBias']
    savedata['CausalDistance'] = causalDistance
    allsavedata[fileNumber] = savedata
savefilename = './data/modelCharacteristics_' + Sfx + 'SensitivityDistance.dat'
torch.save(allsavedata, savefilename)
