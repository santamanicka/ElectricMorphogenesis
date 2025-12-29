import torch
from embryoNetwork import embryoNetwork
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=str, default='(5,5)')
parser.add_argument('--teratogenExposure', type=str, default='True')
parser.add_argument('--boundaryAssistance', type=str, default='True')
parser.add_argument('--fieldModulation', type=str, default='False')
parser.add_argument('--GRNEnabled', type=str, default='False')
parser.add_argument('--LigandEnabled', type=str, default='False')
parser.add_argument('--modelNameEmbryo', type=str, default='0')
parser.add_argument('--modelNameATP', type=str, default='0')
parser.add_argument('--nsamples', type=int, default=1)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--sampleNumber', type=int, default=0)
parser.add_argument('--save', type=str, default='True')
parser.add_argument('--use_parallel', type=str, default='True', help='Use parallel processing for embryo simulations')
parser.add_argument('--force_cpu', type=str, default='False', help='Force CPU mode (useful for avoiding dtype issues on MPS)')

args = parser.parse_args()

parameters = dict()
parameters['dims'] = ast.literal_eval(args.dims)
parameters['teratogenExposure'] = ast.literal_eval(args.teratogenExposure)
parameters['boundaryAssistance'] = ast.literal_eval(args.boundaryAssistance)
parameters['fieldModulation'] = ast.literal_eval(args.fieldModulation)
parameters['GRNEnabled'] = ast.literal_eval(args.GRNEnabled)
parameters['LigandEnabled'] = ast.literal_eval(args.LigandEnabled)
parameters['modelNameEmbryo'] = args.modelNameEmbryo
parameters['modelNameATP'] = args.modelNameATP
nsamples = args.nsamples
niters = args.niters
sampleNumber = args.sampleNumber
save = ast.literal_eval(args.save)
use_parallel = ast.literal_eval(args.use_parallel)
force_cpu = ast.literal_eval(args.force_cpu)

embryoNet = embryoNetwork(parameters=parameters,nsamples=nsamples,niters=niters,use_parallel=use_parallel,force_cpu=force_cpu)
# Note: savedSims is now automatically initialized in simulate() if save=True
embryoNet.simulate(save=save)
if save:
    Sfx = '_Embryo' + parameters['modelNameEmbryo'] + '_ATP' + parameters['modelNameATP'] + '_'
    filename = './data/embryoNetSavedSims_' + Sfx + ','.join(map(str,parameters['dims'])) + '_sample' + str(sampleNumber) + '.dat'
    torch.save(embryoNet.savedSims,filename)

#python simulateEmbryoNetwork.py --dims '(10,10)' --teratogenExposure True --boundaryAssistance False --fieldModulation True --GRNEnabled False --LigandEnabled False --modelNameEmbryo Stigmergic --modelNameATP 262 --nsamples 1 --niters 1000 --sampleNumber 0 --save True --use_parallel True