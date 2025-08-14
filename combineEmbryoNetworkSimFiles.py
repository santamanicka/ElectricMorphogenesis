import torch
import argparse
import ast
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=str, default='(10,10)')
parser.add_argument('--modelNameEmbryo', type=str, default='False')
parser.add_argument('--modelNameATP', type=str, default='False')
parser.add_argument('--samples', type=str, default='(1,100)')
parser.add_argument('--delete', type=str, default='False')

args = parser.parse_args()
dims = ast.literal_eval(args.dims)
modelNameEmbryo = args.modelNameEmbryo
modelNameATP = args.modelNameATP
samples = ast.literal_eval(args.samples)
delete = ast.literal_eval(args.delete)

fulldata = dict()
for sample in range(samples[0],samples[1]+1):
    Sfx = '_Embryo' + modelNameEmbryo + '_ATP' + modelNameATP + '_'
    samplefilename = './data/embryoNetSavedSims_' + Sfx + ','.join(map(str,dims)) + '_sample' + str(sample) + '.dat'
    try:
        data = torch.load(samplefilename)
    except:
        continue
    else:
        fulldata[sample] = data['Vmem']
        if delete:
            os.remove(samplefilename)

Sfx = 'Embryo' + modelNameEmbryo + '_ATP' + modelNameATP + '_'
fullfilename = './data/embryoNetSavedSims_' + Sfx + ','.join(map(str,dims)) + '.dat'
torch.save(fulldata,fullfilename)