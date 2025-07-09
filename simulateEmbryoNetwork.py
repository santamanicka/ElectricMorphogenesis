import torch
from embryoNetwork import embryoNetwork
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=str, default='(5,5)')
parser.add_argument('--teratogenExposure', type=str, default='True')
parser.add_argument('--boundaryAssistance', type=str, default='True')
parser.add_argument('--modelNumEmbryo', type=int, default=0)
parser.add_argument('--modelNumATP', type=int, default=0)
parser.add_argument('--nsamples', type=int, default=1)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--sampleNumber', type=int, default=0)
parser.add_argument('--save', type=str, default='True')

args = parser.parse_args()

parameters = dict()
parameters['dims'] = ast.literal_eval(args.dims)
parameters['teratogenExposure'] = ast.literal_eval(args.teratogenExposure)
parameters['boundaryAssistance'] = ast.literal_eval(args.boundaryAssistance)
parameters['modelNumEmbryo'] = args.modelNumEmbryo
parameters['modelNumATP'] = args.modelNumATP
nsamples = args.nsamples
niters = args.niters
sampleNumber = args.sampleNumber
save = ast.literal_eval(args.save)

embryoNet = embryoNetwork(parameters=parameters,nsamples=nsamples,niters=niters)
if save:
    embryoNet.savedSims = dict()
embryoNet.simulate(save=save)
if save:
    filename = './data/embryoNetSavedSims_' + ','.join(map(str,parameters['dims'])) + '_sample' + str(sampleNumber) + '.dat'
    torch.save(embryoNet.savedSims,filename)
