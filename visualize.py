from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import seaborn as sns
import numpy as np
import torch
import pandas as pd

fieldVector = True
visualize = 'Sensitivity'  # 'Sensitivity', 'PositionalInformation'
variables = 'VmemToVmem'  # 'eVToVmem' or 'VmemToVmem'; applicable for visualize = 'Sensitivity'
filenum, version, region, threshold, time = '1294', '', 'mouth', 0.995, 40
indices = []
save = True

if fieldVector:
	Sfx = 'FieldVector_'
else:
	Sfx = ''

def plotPositionalInformationEnsemble():
	GJStrength, fieldScreenSize, fieldTransductionWeight, fieldTransductionBias, positionalInfoHomo, positionalInfoHetero = [], [], [], [], [], []
	for fileNumber in range(1,626):
		filename = './data/modelCharacteristics_FixedNone_' + Sfx + str(fileNumber) + '.dat'
		data = torch.load(filename)
		latticeDims = data['latticeDims']
		numcells = np.prod(latticeDims)
		GJStrength.append(data['GJParameters']['GJStrength'].round(decimals=2))
		fieldScreenSize.append(data['fieldParameters']['fieldScreenSize'])
		fieldTransductionWeight.append(data['fieldParameters']['fieldTransductionWeight'][0].item())
		fieldTransductionBias.append(data['fieldParameters']['fieldTransductionBias'][0].item())  # don't round since precision upto 4 decimals exist
		numones = np.amax(data['characteristics']['CellularFrequency'][0][0].reshape(1,-1),axis=0,initial=1)
		numzeros = np.amax((data['simParameters']['numSimIters']-numones).reshape(1,-1),axis=0,initial=1)
		numones1to0 = data['characteristics']['CellularFrequency'][1][0]
		numones0to1 = data['characteristics']['CellularFrequency'][2][0]
		cellfreqs = ((numones0to1 / numones) + (numones1to0 / numzeros)) / 2
		positionalInfoHomo.append(cellfreqs)
		allCellfreqs = np.zeros(numcells)
		for s in range(1,101):
			numones = np.amax(data['characteristics']['CellularFrequency'][0][s].reshape(1,-1),axis=0,initial=1)
			numzeros = np.amax((data['simParameters']['numSimIters']-numones).reshape(1,-1),axis=0,initial=1)
			numones1to0 = data['characteristics']['CellularFrequency'][1][s]
			numones0to1 = data['characteristics']['CellularFrequency'][2][s]
			cellfreqs = ((numones0to1/numones)+(numones1to0/numzeros))/2
			allCellfreqs += cellfreqs
		meanCellfreqs = allCellfreqs/100
		positionalInfoHetero.append(meanCellfreqs)
	df = pd.DataFrame({'GJStrength':GJStrength,'fieldRange':fieldScreenSize,'fieldTransductionWeight':fieldTransductionWeight,
					   'fieldTransductionBias':fieldTransductionBias,'positionalInfoHomo':positionalInfoHomo,'positionalInfoHetero':positionalInfoHetero})
	axes = plt.subplots(5,5,figsize=(15,15), sharey=True)
	Weights = df['fieldTransductionWeight'].unique()
	Biases =  df['fieldTransductionBias'].unique()
	for i in range(len(Biases)):
		for j in range(len(Weights)):
			W, B = Weights[j], Biases[i]
			dfsubset = df[(df['fieldTransductionWeight']==W) & (df['fieldTransductionBias']==B) & (df['fieldRange']==4)]
			posinfohomo = dfsubset.agg({'positionalInfoHomo':'mean'})['positionalInfoHomo'].reshape(*latticeDims)
			posinfohetero = dfsubset.agg({'positionalInfoHetero':'mean'})['positionalInfoHetero'].reshape(*latticeDims)
			posinfodiff = np.abs(posinfohetero - posinfohomo)
			sns.heatmap(posinfodiff,ax=axes[1][i, j])
	plt.show()

def normalize(data, threshold=0.0):
	mx = data.amax(1, keepdim=True)  # max per time per target variable
	data = data / mx
	data[torch.isnan(data)] = 0.0
	ntimes, nsources, ntargets = data.shape
	dataTr = data.clone().transpose(1, 2)
	zidx = dataTr.argsort(2, descending=False)
	cutoff = int(threshold * nsources)
	zidx = zidx[:, :, 0:cutoff].flatten()
	xidx = torch.repeat_interleave(torch.arange(ntimes), cutoff * ntargets)
	yidx = torch.arange(ntargets).repeat_interleave(cutoff).tile(ntimes)
	dataTr[xidx, yidx, zidx] = 0.0
	data = dataTr.view(ntimes, ntargets, nsources).transpose(1, 2)
	return data

def animateSensitvity(filenum, version, variables, region, indices, threshold=0.0, plot=True, returnData=False):
	SensitivityData = torch.load(filename)
	# 	SensitivityData = torch.load('./data/modelCharacteristics_FixedBias_'  + filenum + version + '.dat')
	if SensitivityData['fieldParameters']['fieldEnabled']:
		S = SensitivityData['characteristics']['Sensitivity']
		if isinstance(S, dict):
			eVToVmem, VmemToVmem = S['Derivatives']
		else:
			eVToVmem, VmemToVmem = S
	elif SensitivityData['ligandParameters']['ligandEnabled']:
		ligandToVmem, VmemToVmem = SensitivityData['characteristics']['Sensitivity']
	circuitRows, circuitCols = latticeDims = SensitivityData['latticeDims']
	eVDims = (circuitRows + 1, circuitCols + 1)
	if region == 'full':
		data = eval(variables).abs().sum(2)
		Sfx = '_full'
	elif region == 'fulltimecellnorm':
		data = eval(variables).abs()
		data = normalize(data, threshold)
		data = data.sum(2)
		Sfx = '_fulltimecellnorm' + '_thresh' + str(threshold)
	elif region == 'eye1':
		data = eval(variables)[:,:,[14,15,20,21]].abs()  # eye1 representative
		data = normalize(data, threshold)
		data = data.sum(2)
		Sfx = '_eye1' + '_thresh' + str(threshold)
	elif region == 'nose':
		data = eval(variables)[:,:,[29,35,41]].abs().sum(2)  # nose representative
		Sfx = '_nose'
	elif region == 'mouth':
		data = eval(variables)[:,:,[52,53]].abs().sum(2)  # mouth representative
		Sfx = '_mouth'
	elif region == 'skin':
		# data = eval(variables)[:,:,[0,5,30,60,65]].abs().sum(2)  # skin representatives
		data = eval(variables)[:,:,[0,1,2,3,4,5,6,12,18,24,30,36,42,48,54,60,61,62,63,64,65]].abs().sum(2)  # skin representatives
		Sfx = '_skin'
	elif region == 'representative':
		data = eval(variables)[:,:,[0,1,2]].abs().sum(2)  # representative
		Sfx = '_representative'
	elif region == 'default':
		data = eval(variables).abs().sum(2)  # default
		Sfx = '_default'
	elif region == 'select':
		data = eval(variables)[:,:,indices].abs().sum(2)  # default
		Sfx = '_select' + str(indices)
	if variables == 'eVToVmem':
		dims = eVDims
	elif variables == 'VmemToVmem':
		dims = latticeDims
	elif variables == 'ligandToVmem':
		dims = latticeDims
	data = data.reshape(-1, *dims)
	nzidx = [data[i].any() for i in range(data.shape[0])]
	data = data[nzidx]
	if plot:
		fig, ax = plt.subplots()
		mn, mx = data[0].min(), data[0].max()
		heatmap = ax.pcolormesh(data[0], cmap='seismic', vmin=mn, vmax=mx)
		def animate(t):
			# Update the heatmap
			d = np.flipud(data[t])
			mn, mx = d.min(), d.max()
			heatmap.set_array(d)
			heatmap.set_clim(vmin=mn, vmax=mx)
			# Return the updated heatmap
			return heatmap,
		ani = FuncAnimation(fig, animate, frames=np.arange(0, data.shape[0]), interval=0.1, blit=True)
		Writer = animation.writers['ffmpeg']
		mywriter = Writer(fps=1, metadata=dict(artist='Me'))
		if fieldVector:
			fSfx = 'FieldVector_'
		else:
			fSfx = ''
		ani.save('./data/Smiley_' + variables + 'Sensitivity_' + fSfx + 'Set' + filenum + version + Sfx + '.mp4', writer=mywriter)
	if returnData:
		return data

def plotTissuePatterned(save=True):
	yside, xside = 11, 11
	numcells = yside * xside
	xOffset, yOffset = 10, 10
	x = np.tile(range(0,xside*xOffset,xOffset),yside)
	y = np.repeat(range(0,-yside*yOffset,-yOffset),xside)
	xypts = np.column_stack((x,y))
	fig, ax = plt.subplots(1)
	eye1, eye2, nose, mouth = [24,25,35,36], [29,30,40,41], [49,60,71], [92,93,94]
	topskin = np.arange(xside)
	bottomskin = topskin + ((yside-1)*xside)
	rightskin = np.arange(xside-1,yside*xside,yside)
	leftskin = rightskin - xside + 1
	skin = np.unique(np.concatenate([topskin,bottomskin,rightskin,leftskin]))
	patterncells = np.concatenate([eye1, eye2, nose, mouth, skin])
	othercells = np.setdiff1d(range(numcells),patterncells)
	ax.scatter(xypts[patterncells,0], xypts[patterncells,1], s=200, c='red', alpha=0.4)  # points
	ax.scatter(xypts[othercells,0], xypts[othercells,1], s=200, c='grey', alpha=0.4)  # points
	ax.set_ylim(-yside*yOffset+yOffset-yOffset,yOffset)
	ax.set_xlim(-xOffset,xside*xOffset-xOffset+xOffset)
	ax.axis('off')
	if save:
		savefname = './data/TissuePatterened.pdf'
		plt.savefig(savefname,bbox_inches="tight")
	else:
		plt.show()

def computeSensivityNetwork(filename, variables, threshold=0.0):
	SensitivityData = torch.load(filename)
	if SensitivityData['fieldParameters']['fieldEnabled']:
		S = SensitivityData['characteristics']['Sensitivity']
		if isinstance(S,dict):
			eVToVmem, VmemToVmem = S['Derivatives']
		else:
			eVToVmem, VmemToVmem = S
	elif SensitivityData['ligandParameters']['ligandEnabled']:
			ligandToVmem, VmemToVmem = SensitivityData['characteristics']['Sensitivity']
	circuitRows, circuitCols = latticeDims = SensitivityData['latticeDims']
	eVDims = (circuitRows+1, circuitCols+1)
	networkTimeseries = eval(variables)  # signed; shape = (numTimePoints,nsources,ntargets)
	mx = networkTimeseries.abs().amax(dim=1,keepdim=True)  # abs max per time per target variable
	# mx = networkTimeseries.abs().amax(dim=(1,2),keepdim=True)  # abs max per time
	networkTimeseries = networkTimeseries/mx  # signed; shape = (numTimePoints,nsources,ntargets)
	networkTimeseries[torch.isnan(networkTimeseries)] = 0.0
	ntimes, nsources, ntargets = networkTimeseries.shape
	networkTimeseriesTr = networkTimeseries.clone().transpose(1,2)  # shape = (numTimePoints,ntargets,nsources)
	zidx = networkTimeseriesTr.abs().argsort(2,descending=False)  # selects the strongest positives and negatives but keeps their signs
	trim = int(threshold * nsources)  # threshold of 0.995 leaves one source; 0.99 leaves two sources of the 121 sources
	zidx = zidx[:,:,0:trim].flatten()
	xidx = torch.repeat_interleave(torch.arange(ntimes),trim*ntargets)
	yidx = torch.arange(ntargets).repeat_interleave(trim).tile(ntimes)
	networkTimeseriesTr[xidx,yidx,zidx] = 0.0
	networkTimeseries = networkTimeseriesTr.view(ntimes,ntargets,nsources).transpose(1,2)
	return networkTimeseries

def plotSensitivityNetwork(filename, variables, threshold, time, save):
	networkTimeseries = computeSensivityNetwork(filename, variables, threshold)  # shape = (numTimePoints,nsources,ntargets)
	nzidx = [networkTimeseries[i].any() for i in range(networkTimeseries.shape[0])]
	data = networkTimeseries[nzidx]  # shape = (numTimePoints,nsources,ntargets)
	WeightMatrix = data[time]
	nsourcecols = int(np.sqrt(WeightMatrix.shape[0]))
	# ntargets = WeightMatrix.shape[1]
	yside, xside = nsourcecols, nsourcecols
	numcells = yside * xside
	AdjMatrix = WeightMatrix.clone()
	AdjMatrix[AdjMatrix != 0] = 1
	EdgeList = np.array(np.nonzero(AdjMatrix))
	## In case ntargets < numCells, then correct the indices
	# ntargetcols = int(ntargets / nsourcecols)
	# correctidx = np.concatenate([np.arange(ntargetcols)+(i*nsourcecols) for i in range(nsourcecols)])
	# replidx = EdgeList[:,1][np.array([np.where(EdgeList[:,1]==i)[0] for i in range(ntargets)]).flatten()]
	# for target in np.arange(ntargets-1,-1,-1):
	# 	EdgeList[:,1][EdgeList[:,1] == target] = correctidx[target]
	xOffset, yOffset = 10, 10
	x = np.tile(range(0,xside*xOffset,xOffset),yside)
	y = np.repeat(range(0,-yside*yOffset,-yOffset),xside)
	xypts = np.column_stack((x,y))
	edges = xypts[EdgeList]
	SignedWeights = np.array(torch.masked_select(WeightMatrix,torch.tensor(AdjMatrix,dtype=torch.bool)))
	SignedWeights = SignedWeights.reshape(-1,1,1)
	SignedWeights = np.repeat(SignedWeights,2,1)
	AbsWeights = SignedWeights.copy().__abs__()
	edges = np.concatenate((edges,AbsWeights,SignedWeights),2)
	# plot cells and mark the face features
	fig, ax = plt.subplots(1)
	eye1, eye2, nose, mouth = [24,25,35,36], [29,30,40,41], [49,60,71], [92,93,94]
	topskin = np.arange(xside)
	bottomskin = topskin + ((yside-1)*xside)
	rightskin = np.arange(xside-1,yside*xside,yside)
	leftskin = rightskin - xside + 1
	skin = np.unique(np.concatenate([topskin,bottomskin,rightskin,leftskin]))
	patterncells = np.concatenate([eye1, eye2, nose, mouth, skin])
	othercells = np.setdiff1d(range(numcells),patterncells)
	ax.scatter(xypts[patterncells,0], xypts[patterncells,1], s=200, c='red', alpha=0.4)  # points
	ax.scatter(xypts[othercells,0], xypts[othercells,1], s=200, c='grey', alpha=0.4)  # points
	style = "Simple, tail_width=0.5, head_width=4, head_length=8"
	kw = dict(arrowstyle=style, alpha=0.8)
	# plot edges
	# the direction p2->p1 ensures that the rows of the WeightMatrix are interpreted as incoming connections
	cmap = plt.get_cmap('PuOr')
	norm = plt.Normalize(vmin=-1, vmax=1)
	Edges = [patches.FancyArrowPatch(p1[0:2],p2[0:2],connectionstyle="arc3,rad=0.75",lw=p1[2],color=cmap(norm(p1[3])),**kw) for p1,p2 in edges]
	for edge in Edges:
		ax.add_patch(edge)
	selfloopnodes = EdgeList[EdgeList[:,0]==EdgeList[:,1],0]
	for node in selfloopnodes:
		ax.plot(xypts[node,0],xypts[node,1],marker=r'$\circlearrowleft$',ms=15,mew=0.01,color=cmap(norm(edges[node,0,3])))
	ax.set_ylim(-yside*yOffset+yOffset-yOffset,yOffset)
	ax.set_xlim(-xOffset,xside*xOffset-xOffset+xOffset)
	# ax.set_aspect('equal')
	ax.axis('off')
	if save:
		savefname = './data/SensitivityNetwork_Signed_' + Sfx + filenum + '_threshold' + str(threshold) + '_timestep' + str(time) + '.pdf'
		plt.savefig(savefname,bbox_inches="tight")
	else:
		plt.show()

if visualize == 'Sensitivity':
	filename = './data/modelCharacteristics_Sensitivity_' + Sfx + filenum + version + '.dat'
	# animateSensitvity(filenum, version, variables, region, indices, threshold, plot=True, returnData=False)
	for time in [10,20,30,40,50,80]:
		plotSensitivityNetwork(filename, variables, threshold, time, save)
elif visualize == 'PositionalInformation':
	plotPositionalInformationEnsemble()
