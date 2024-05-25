from cellularFieldNetwork import cellularFieldNetwork
import numpy as np
import torch

parameterfilename = './data/bestModelParameters_0.dat'
parameters = torch.load(parameterfilename)

latticeDims = parameters['latticeDims']
fieldParameters = parameters['fieldParameters']
GRNParameters = parameters['GRNParameters']
numSamples = parameters['simParameters']['numSamples']
initialValues = parameters['simParameters']['initialValues']
clampParameters = parameters['clampParameters']
externalInputs = parameters['simParameters']['externalInputs']
numSimIters = parameters['simParameters']['numSimIters']
evalDurationProp = parameters['trainParameters']['evalDurationProp']
targetVmem = parameters['trainParameters']['targetVmem']

modelparameters = dict()
modelparameters['fieldParameters'] = fieldParameters
modelparameters['GRNParameters'] = GRNParameters
circuit = cellularFieldNetwork(latticeDims=latticeDims,parameters=modelparameters,numSamples=numSamples)
circuit.initVariables(initialValues)
circuit.initParameters(initialValues)
circuit.G_0 = 0.0
circuit.simulate(externalInputs=externalInputs,clampParameters=clampParameters,perturbationParameters=None,
				 numSimIters=numSimIters,stochasticIonChannels=False,setGradient=False,retainGradients=False,saveData=True)
evalDuration = int(evalDurationProp*numSimIters)
loss = ((targetVmem - circuit.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
np.set_printoptions(precision=2,suppress=True)
print("loss = ",loss)
