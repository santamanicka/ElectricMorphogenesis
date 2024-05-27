import torch
import numpy as np
from itertools import chain

fileRange = chain(range(401,500))

allfilenums, allerrors = [], []
for fileNumber in fileRange:
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