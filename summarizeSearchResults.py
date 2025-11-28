import torch
import numpy as np
from itertools import chain
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--fileRange', type=str, default='(1,101)')
parser.add_argument('--facialGRN', action='store_true')
parser.add_argument('--fieldVector', action='store_true')
parser.add_argument('--ligandEnabled', action='store_true')
parser.add_argument('--GRNEnabled', action='store_true')
parser.add_argument('--top', type=int, default=5)

args = parser.parse_args()
fileRange = ast.literal_eval(args.fileRange)
facialGRN = args.facialGRN
fieldVector = args.fieldVector
ligandEnabled = args.ligandEnabled
GRNEnabled = args.GRNEnabled
fileNumbers = list(range(fileRange[0],fileRange[1]))
top = args.top

# fileNumbers = list(range(1401,1501))
# fileNumbers1 = list(range(1501,1601))
# fileNumbers.extend(fileNumbers1)

allfilenums, allerrors, allweights, allbiases = [], [], [], []
for fileNumber in fileNumbers:
    try:
        if facialGRN:
            bestModel = torch.load('./data/bestLearnedFacialParams_' + str(fileNumber) + '.dat')
        elif fieldVector:
            if ligandEnabled:
                bestModel = torch.load('./data/bestModelParameters_fieldVector_Ligand_' + str(fileNumber) + '.dat')
                if GRNEnabled:
                    bestModel = torch.load('./data/bestModelParameters_fieldVector_Ligand_GRN_' + str(fileNumber) + '.dat')
            elif GRNEnabled:
                bestModel = torch.load('./data/bestModelParameters_fieldVector_GRN_' + str(fileNumber) + '.dat')
            else:
                bestModel = torch.load('./data/bestModelParameters_fieldVector_' + str(fileNumber) + '.dat')
        else:
            bestModel = torch.load('./data/bestModelParameters_' + str(fileNumber) + '.dat')
    except:
        continue
    else:
        if facialGRN:
            performance = bestModel['loss']
            allerrors.append(performance)
            allfilenums.append(fileNumber)
        else:
            performance = bestModel['trainParameters']['bestLoss']
            weight = bestModel['fieldParameters']['fieldTransductionWeight']
            bias = bestModel['fieldParameters']['fieldTransductionBias']
            allfilenums.append(fileNumber)
            allerrors.append(performance)
            allweights.append(weight)
            allbiases.append(bias)

if facialGRN:
    allfilenums = np.array(allfilenums).reshape(-1, 1)
    allerrors = np.array(allerrors).reshape(-1, 1)
    alldata = np.concatenate((allfilenums,allerrors), 1)
    print(*alldata[alldata[:, 1].argsort()][0:top], sep='\n')  # '*' prefix helps print every line separately
else:
    allfilenums = np.array(allfilenums).reshape(-1,1)
    allerrors = np.array(allerrors).reshape(-1,1)
    allweights = np.array(allweights).reshape(-1,1)
    allbiases = np.array(allbiases).reshape(-1,1)
    # alldata = np.concatenate((allfilenums,allerrors,allweights,allbiases),1)
    alldata = np.concatenate((allfilenums,allerrors),1)
    print(*alldata[alldata[:,1].argsort()][0:top],sep='\n')  # '*' prefix helps print every line separately