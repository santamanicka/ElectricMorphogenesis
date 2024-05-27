import torch
import numpy as np
from itertools import chain
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--fileRange', type=str, default='(1,101)')

args = parser.parse_args()
fileRange = ast.literal_eval(args.fileRange)
fileNumers = list(range(fileRange[0],fileRange[1]))

# fileNumers = chain(range(401,500))

allfilenums, allerrors = [], []
for fileNumber in fileNumers:
    try:
        bestModel = torch.load('./data/bestModelParameters_' + str(fileNumber) + '.dat')
    except:
        continue
    else:
        performance = bestModel['trainParameters']['bestLoss']
        allfilenums.append(fileNumber)
        allerrors.append(performance)

allfilenums = np.array(allfilenums).reshape(-1,1)
allerrors = np.array(allerrors).reshape(-1,1)
alldata = np.concatenate((allfilenums,allerrors),1)
print(*alldata[alldata[:,1].argsort()],sep='\n')  # '*' prefix helps print every line separately