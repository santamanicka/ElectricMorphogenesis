import torch
import argparse
import ast
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=str, default='(10,10)')
parser.add_argument('--samples', type=str, default='(1,100)')
parser.add_argument('--delete', type=str, default='False')

args = parser.parse_args()
dims = ast.literal_eval(args.dims)
samples = ast.literal_eval(args.samples)
delete = ast.literal_eval(args.delete)

fulldata = dict()
for sample in range(samples[0],samples[1]+1):
    samplefilename = './data/embryoNetSavedSims_' + ','.join(map(str,dims)) + '_sample' + str(sample) + '.dat'
    data = torch.load(samplefilename)
    fulldata[sample] = data
    if delete:
        os.remove(samplefilename)

fullfilename = './data/embryoNetSavedSims_' + ','.join(map(str,dims)) + '.dat'
torch.save(fulldata,fullfilename)