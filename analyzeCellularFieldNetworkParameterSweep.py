import torch
import numpy as np
from scipy.stats import entropy
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.graph_objs import Layout
from scipy.ndimage import gaussian_filter

clampMode = 'fieldDome'
clampModeFileSuffix = {'field':'Field','tissue':'Tissue','fieldDome':'FieldDome'}
PlotTitles = {'field':'Bulk field','tissue':'Bulk Vmem','fieldDome':'Boundary field'}

plot = True

parameterSweepPlotFileName = 'parameterSweepPlot' + clampModeFileSuffix[clampMode] + '.dat'
parameterSweepPlotFileName = './data/' + parameterSweepPlotFileName
parameterSweepAnalysisFileName = 'parameterSweepAnalysis' + clampModeFileSuffix[clampMode] + '.dat'
parameterSweepAnalysisFileName = './data/' + parameterSweepAnalysisFileName

if os.path.isfile(parameterSweepPlotFileName):
	computePlotData = False
	computeAnalysisData = False
elif os.path.isfile(parameterSweepAnalysisFileName):
	computePlotData = True
	computeAnalysisData = False
else:
	computeAnalysisData = True
	computePlotData = True

numParameterValues = 10
VmemBins = np.arange(-0.0, -0.1, -0.04)  # bin size of -40mV to bin observed vmem values

def computeEntropy(vmem):  # vmem should be a 1D tensor
	# counts = torch.unique(vmem.round(decimals=2),return_counts=True)[1]
	labels = np.digitize(vmem,VmemBins)
	counts = np.unique(labels,return_counts=True)[1]
	probabilities = counts/counts.sum()
	H = entropy(probabilities)
	return (H)

print("Clamp mode = ",clampMode)

if computeAnalysisData:
	Sfx = clampModeFileSuffix[clampMode]
	data = torch.load('./data/parameterSweep'+Sfx+'.dat')
	allFieldResolutions, allClampVoltages, allClampDurProps, \
		allClampCellsProps, allSampleIndices, allEntropies = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
	for index in data:
		record = data[index]
		fieldResolution = record['fieldResolution'].item()
		clampVoltage = record['clampVoltage'].item()
		clampDurationProp = record['clampDurationProp'].item()
		clampedCellsProp = record['clampedCellsProp'].item()
		vmemSamples = record['Vmem']
		numSamples = vmemSamples.shape[0]
		for s in range(numSamples):
			vmem = vmemSamples[s].flatten()
			H = computeEntropy(vmem)
			allSampleIndices = np.concatenate((allSampleIndices,np.array([s])))
			allEntropies = np.concatenate((allEntropies,np.array([H])))
		# recTimeSeriesVmem = record['timeseriesVmem']
		# numTimeIndices = len(recTimeSeriesVmem)
		# for t in range(numTimeIndices):
		# 	recVmem = recTimeSeriesVmem[t].flatten()
		# 	H = computeEntropy(recVmem)
		# 	allTimeIndices = np.concatenate((allTimeIndices,np.array([t+1])))
		# 	allEntropies = np.concatenate((allEntropies,np.array([H])))
		allFieldResolutions = np.concatenate((allFieldResolutions,np.repeat(fieldResolution,numSamples)))
		allClampVoltages = np.concatenate((allClampVoltages,np.repeat(clampVoltage,numSamples)))
		allClampDurProps = np.concatenate((allClampDurProps,np.repeat(clampDurationProp,numSamples)))
		allClampCellsProps = np.concatenate((allClampCellsProps,np.repeat(clampedCellsProp,numSamples)))

	df = pd.DataFrame({'fieldResolution':allFieldResolutions,'clampVoltage':allClampVoltages,
					   'clampDuration':allClampDurProps,'clampedCells':allClampCellsProps,
					   'sampleIndex':allSampleIndices,'complexity':allEntropies})
					   # 'timeIndex':allTimeIndices,'complexity':allEntropies})

	torch.save(df,parameterSweepAnalysisFileName)
	print("Analysis file generated!")
else:
	if computePlotData:
		df = torch.load(parameterSweepAnalysisFileName)
		fieldResolutions = df['fieldResolution'].unique()
		clampVoltages = df['clampVoltage'].unique()
		clampDurationProps = df['clampDuration'].unique()
		df['clampedCells'] = df['clampedCells'].round(2)
		# dfsub = df[(df['clampVoltage'] == clampVoltages[5]) & (df['clampDuration'] == clampDurationProps[5])] \
		# 			[['fieldResolution','clampedCells','timeIndex', 'complexity']]
		# dfsubPlot = df.groupby(['fieldResolution','clampedCells'],as_index=False).mean()[['fieldResolution','clampedCells','complexity']]
		dfsubPlot = df.groupby(['fieldResolution','clampVoltage'],as_index=False).mean()[['fieldResolution','clampVoltage','complexity']]
		parameterSweepPlotFileName = 'parameterSweepPlot' + clampModeFileSuffix[clampMode] + '.dat'
		parameterSweepPlotFileName = './data/' + parameterSweepPlotFileName
		torch.save(dfsubPlot,parameterSweepPlotFileName)
		print("Plot file generated!")
	else:
		dfsubPlot = torch.load(parameterSweepPlotFileName)
	if plot:
		# x, y = np.meshgrid(dfsubPlot['fieldResolution'].unique(),dfsubPlot['clampedCells'].unique())
		x, y = np.meshgrid(dfsubPlot['fieldResolution'].unique(),dfsubPlot['clampVoltage'].unique())
		z = np.array(dfsubPlot['complexity']).reshape(*x.shape,order='F')
		zg = gaussian_filter(z, [1,0.1])
		plotData = go.Surface(z=zg, x=x, y=y)
		layout = Layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
								   xaxis_title=dict(text='Field resolution',font=dict(size=20)),
								   # yaxis_title=dict(text='Clamp proportion',font=dict(size=20)),
								   yaxis_title=dict(text='Clamp voltage', font=dict(size=20)),
								   zaxis_title=dict(text='Pattern entropy',font=dict(size=20))))
		fig = go.Figure(data=plotData,layout=layout)
		fig.update_layout(title=dict(text=PlotTitles[clampMode]),title_x=0.5,title_y=0.75,title_font_size=20)
		fig.update_traces(colorbar=dict(x=0.7,len=0.5))
		fig.show()