import torch
from embryoNetwork import embryoNetwork
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=str, default='(5,5)')
parser.add_argument('--teratogenExposure', type=str, default='True')
parser.add_argument('--modelNumEmbryo', type=int, default=0)
parser.add_argument('--modelNumATP', type=int, default=0)
parser.add_argument('--save', type=str, default='True')

args = parser.parse_args()

parameters = dict()
parameters['dims'] = ast.literal_eval(args.dims)
parameters['teratogenExposure'] = ast.literal_eval(args.teratogenExposure)
parameters['modelNumEmbryo'] = args.modelNumEmbryo
parameters['modelNumATP'] = args.modelNumATP
save = ast.literal_eval(args.save)

embryoNet = embryoNetwork(parameters=parameters,nsamples=1,niters=1000)
if save:
	embryoNet.savedSims = dict()
embryoNet.simulate(save=save)
if save:
	torch.save(embryoNet.savedSims,'./data/embryoNetSavedSims_' + ','.join(map(str,parameters['dims'])) + '.dat')