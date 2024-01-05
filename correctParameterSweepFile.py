import torch

data = torch.load('./data/parameterSweep.dat')

circuitRows,circuitCols = 4,6

numCells = circuitRows * circuitCols
numParameterValues = 20
fieldResolutions = torch.arange(1,11)
clampModes = ['field','tissue']
clampVoltages = torch.linspace(-0.01,-0.2,numParameterValues)
clampDurationProps = torch.linspace(0.1,0.9,numParameterValues)
clampedCellsProps = torch.linspace(0.05,0.95,numParameterValues)

def computeNumExtracellularGridPoints(numRows,numCols,fieldResolution):
	r,c,s = numRows,numCols,fieldResolution
	numExtracellularGridPoints = s*(r+c+(2*r*c)) - (r*c) + 1
	return numExtracellularGridPoints

paramCombination = 0
for clampMode in clampModes:
	for fieldResolution in fieldResolutions:
		for clampVoltage in clampVoltages:
			for clampDurationProp in clampDurationProps:
				for clampedCellsProp in clampedCellsProps:
					print(paramCombination)
					record = data[paramCombination]
					numExtracellularGridPoints = computeNumExtracellularGridPoints(circuitRows,circuitCols,fieldResolution)
					if clampMode == 'field':
						clampProportionNorm = (clampedCellsProp * numExtracellularGridPoints) / (numExtracellularGridPoints + numCells)
					elif clampMode == 'tissue':
						clampProportionNorm = (clampedCellsProp * numCells) / (numExtracellularGridPoints + numCells)
						record['fieldResolution'] = fieldResolution
					record['clampedCellsPropNorm'] = clampProportionNorm
					record['tissueDimensions'] = (circuitRows,circuitCols)
					data[paramCombination] = record
					paramCombination += 1

torch.save(data,'./data/parameterSweep.dat')