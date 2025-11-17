import torch
import numpy as np

dim = 15
d = torch.load('./data/embryoNetSavedSims_EmbryoStigmergic_ATP0_' + str(dim) + ',' + str(dim) + '.dat',weights_only=False)

def defineFace():
    skinIndices = np.concatenate((np.arange(0,11),np.array([[i,i+10] for i in range(11,110,11)]).flatten(),
                                  np.arange(110,121)))
    eyeIndices = np.array([24,25,35,36,29,30,40,41])  # left and right eyes
    noseIndices = np.array([49,60,71])
    mouthIndices = np.array([92,93,94])
    face = np.zeros(121)
    face[skinIndices] = 1  # Skin
    face[eyeIndices] = 1  # Eyes 1 and 2
    face[noseIndices] = 1  # Nose
    face[mouthIndices] = 1  # Mouth
    return face

face = defineFace()
samples = d.keys()
numMismatchesFull = []
for sample in samples:
    embryoIndices = d[sample].keys()
    for embryo in embryoIndices:
        pattern = d[sample][embryo]
        pattern[pattern > -0.03] = 0
        pattern[pattern <= -0.03] = 1
        mismatch = np.abs(pattern - face).sum().item()
        numMismatchesFull.append(mismatch)

numMismatchesFull = np.array(numMismatchesFull)