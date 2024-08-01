# Plots various heatmaps that were reported in 'Patterning in a bioelectric field network - summary of research so far V3.docx'

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

analysisMode = "fixBiasSweepWeightScreenGJ"  # fixWeightBiasSweepScreenGJ, fixBiasSweepWeightScreenGJ
fileNumberVersion = 0
if analysisMode == 'fixScreenGJSweepWeightBias':
    Sfx = 'FixedScreenSizeGJ_'
elif analysisMode == 'fixWeightBiasSweepScreenGJ':
    Sfx = 'FixedWeightBias_'
elif analysisMode == 'fixBiasSweepWeightScreenGJ':
    Sfx = 'FixedBias_'
elif analysisMode == 'sensitivity':
    Sfx = 'Sensitivity_'
elif analysisMode == 'robustness':
    Sfx = 'Robustness_'
if fileNumberVersion > 0:
    fileVersionSfx = '_V' + str(fileNumberVersion)
else:
    fileVersionSfx = ''

if analysisMode == "fixBiasSweepWeightScreenGJ":
    fileRange = range(1,501)
    GJStrength, fieldScreenSize, fieldTransductionWeight, Robustness = [], [], [], []
    for fileNumber in fileRange:
        filename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + fileVersionSfx + '.dat'
        data = torch.load(filename)
        GJStrength.append(data['GJParameters']['GJStrength'].round(decimals=2))
        fieldScreenSize.append(data['fieldParameters']['fieldScreenSize'])
        fieldTransductionWeight.append(data['fieldParameters']['fieldTransductionWeight'].round(decimals=2))
        # Robustness.append(data['characteristics']['Persistence'].mean().item())
        Robustness.append(data['characteristics']['Correlation'].mean().item())
    df = pd.DataFrame({'GJStrength':GJStrength,'fieldScreenSize':fieldScreenSize,'fieldTransductionWeight':fieldTransductionWeight,'Robustness':Robustness})
    heatmap = df.pivot_table(index='GJStrength',columns='fieldScreenSize',values='Robustness')
    heatmap_smooth = gaussian_filter(heatmap, sigma=1)
    # heatmap_smooth = heatmap
    fig, ax = plt.subplots()
    map = sns.heatmap(heatmap_smooth,cmap='seismic')
    # plt.show()
    plt.savefig('./data/modelCharacteristics_FixedBias_Correlation.png',bbox_inches="tight")

if analysisMode == "fixWeightBiasSweepScreenGJ":
    fileRange = range(1,301)
    GJStrength, fieldScreenSize, TotalCorr, Entropy = [], [], [], []
    for fileNumber in fileRange:
        filename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + fileVersionSfx + '.dat'
        data = torch.load(filename)
        GJStrength.append(data['GJParameters']['GJStrength'].round(decimals=2))
        fieldScreenSize.append(data['fieldParameters']['fieldScreenSize'])
        TotalCorr.append(np.array(data['characteristics']['Information'][0]).mean().item())
        Entropy.append(np.array(data['characteristics']['Information'][1]).mean().item())
    df = pd.DataFrame({'GJStrength':GJStrength,'fieldScreenSize':fieldScreenSize,'TotalCorrelation':TotalCorr,'Entropy':Entropy})
    heatmap = df.pivot_table(index='GJStrength',columns='fieldScreenSize',values='Entropy')
    # heatmap_smooth = gaussian_filter(heatmap, sigma=1)
    heatmap_smooth = heatmap
    fig, ax = plt.subplots()
    map = sns.heatmap(heatmap_smooth,cmap='seismic')
    plt.show()

# d10 = torch.load('./data/VmemEVPCAMeasures_50K_10x10.dat')
# df10 = pd.DataFrame(d10)
# df10['diff'] = df10[[6,7,8]].sum(1) - df10[[10,11,12]].sum(1)
# df10Plot = df10[[0,1,'diff']]
# heat10 = df10Plot.pivot_table(index=0,columns=1,values='diff')
# heat10_smooth = gaussian_filter(heat10, sigma=1)
# mx = np.abs(heat10_smooth).max()
# xticklabels = df10Plot[1].unique()
# xticklabels = (xticklabels*100/xticklabels.max()).round(0).astype(int)
# xticklabels = [str(i)+'%' for i in xticklabels]
# yticklabels = df10Plot[0].unique().round(2)
# fig, ax = plt.subplots()
# map = sns.heatmap(heat10_smooth,cmap="vlag",xticklabels=xticklabels,yticklabels=yticklabels)
# map.set_xticklabels(ax.get_xticklabels(), rotation=45)
# map.set_xlabel('Field reach', fontsize=12)
# map.set_ylabel('Gap junction strength', fontsize=12)
# map.set_title('eV-Vmem PCA diff for 10x10 tissue',fontsize=14)
# map.figure.tight_layout()
# plt.show()
#
# d15 = torch.load('./data/VmemEVPCAMeasures_50K_15x15.dat')
# df15 = pd.DataFrame(d15)
# df15['diff'] = df15[[6,7,8]].sum(1) - df15[[10,11,12]].sum(1)
# df15Plot = df15[[0,1,'diff']]
# heat15 = df15Plot.pivot_table(index=0,columns=1,values='diff')
# heat15_smooth = gaussian_filter(heat15, sigma=1)
# mx = np.abs(heat15_smooth).max()
# xticklabels = df15Plot[1].unique()
# xticklabels = (xticklabels*100/xticklabels.max()).round(0).astype(int)
# xticklabels = [str(i)+'%' for i in xticklabels]
# yticklabels = df15Plot[0].unique().round(2)
# fig, ax = plt.subplots()
# map = sns.heatmap(heat15_smooth,cmap="vlag",xticklabels=xticklabels,yticklabels=yticklabels)
# map.set_xticklabels(ax.get_xticklabels(), rotation=45)
# map.set_xlabel('Field reach', fontsize=12)
# map.set_ylabel('Gap junction strength', fontsize=12)
# map.set_title('eV-Vmem PCA diff for 15x15 tissue',fontsize=14)
# map.figure.tight_layout()
# plt.show()
#
# d10 = torch.load('./data/VmemInformationMeasures_50K_10x10.dat')
# df10 = pd.DataFrame(d10)
# # df10['total'] = df10[[7,9]].sum(1)
# df10['total'] = ((df10[7]/df10[10])+(df10[9]/df10[12]))/2
# df10['total'] = df10['total'].fillna(0)
# df10Plot = df10[[0,1,'total']]
# heat10 = df10Plot.pivot_table(index=0,columns=1,values='total')
# heat10_smooth = gaussian_filter(heat10, sigma=1)
# mx = np.abs(heat10_smooth).max()
# xticklabels = df10Plot[1].unique()
# xticklabels = (xticklabels*100/xticklabels.max()).round(0).astype(int)
# xticklabels = [str(i)+'%' for i in xticklabels]
# yticklabels = df10Plot[0].unique().round(2)
# fig, ax = plt.subplots()
# map = sns.heatmap(heat10_smooth,cmap="vlag",xticklabels=xticklabels,yticklabels=yticklabels)
# map.set_xticklabels(ax.get_xticklabels(), rotation=45)
# map.set_xlabel('Field reach', fontsize=12)
# map.set_ylabel('Gap junction strength', fontsize=12)
# map.set_title('Normalized bulk+boundary entropy of 10x10 tissue',fontsize=14)
# map.figure.tight_layout()
# plt.show()
#
# d15 = torch.load('./data/VmemInformationMeasures_50K_15x15.dat')
# df15 = pd.DataFrame(d15)
# df15['total'] = ((df15[7]/df15[10])+(df15[9]/df15[12]))/2
# df15['total'] = df15['total'].fillna(0)
# df15Plot = df15[[0,1,'total']]
# heat15 = df15Plot.pivot_table(index=0,columns=1,values='total')
# heat15_smooth = gaussian_filter(heat15, sigma=1)
# mx = np.abs(heat15_smooth).max()
# xticklabels = df15Plot[1].unique()
# xticklabels = (xticklabels*100/xticklabels.max()).round(0).astype(int)
# xticklabels = [str(i)+'%' for i in xticklabels]
# yticklabels = df15Plot[0].unique().round(2)
# fig, ax = plt.subplots()
# map = sns.heatmap(heat15_smooth,cmap="vlag",xticklabels=xticklabels,yticklabels=yticklabels)
# map.set_xticklabels(ax.get_xticklabels(), rotation=45)
# map.set_xlabel('Field reach', fontsize=12)
# map.set_ylabel('Gap junction strength', fontsize=12)
# map.set_title('Normalized bulk+boundary entropy of 15x15 tissue',fontsize=14)
# map.figure.tight_layout()
# plt.show()
#
# d10 = torch.load('./data/VmemInformationMeasures_50K_10x10.dat')
# df10 = pd.DataFrame(d10)
# df10['total'] = df10[[5]].sum(1)
# df10Plot = df10[[0,1,'total']]
# heat10 = df10Plot.pivot_table(index=0,columns=1,values='total')
# heat10_smooth = gaussian_filter(heat10, sigma=1)
# mx = np.abs(heat10_smooth).max()
# xticklabels = df10Plot[1].unique()
# xticklabels = (xticklabels*100/xticklabels.max()).round(0).astype(int)
# xticklabels = [str(i)+'%' for i in xticklabels]
# yticklabels = df10Plot[0].unique().round(2)
# fig, ax = plt.subplots()
# map = sns.heatmap(heat10_smooth,cmap="vlag",xticklabels=xticklabels,yticklabels=yticklabels)
# map.set_xticklabels(ax.get_xticklabels(), rotation=45)
# map.set_xlabel('Field reach', fontsize=12)
# map.set_ylabel('Gap junction strength', fontsize=12)
# map.set_title('Bulk-boundary integration of 10x10 tissue',fontsize=14)
# map.figure.tight_layout()
# plt.show()
#
# d15 = torch.load('./data/VmemInformationMeasures_50K_15x15.dat')
# df15 = pd.DataFrame(d15)
# df15['total'] = df15[[5]].sum(1)
# df15Plot = df15[[0,1,'total']]
# heat15 = df15Plot.pivot_table(index=0,columns=1,values='total')
# heat15_smooth = gaussian_filter(heat15, sigma=1)
# mx = np.abs(heat15_smooth).max()
# xticklabels = df15Plot[1].unique()
# xticklabels = (xticklabels*150/xticklabels.max()).round(0).astype(int)
# xticklabels = [str(i)+'%' for i in xticklabels]
# yticklabels = df15Plot[0].unique().round(2)
# fig, ax = plt.subplots()
# map = sns.heatmap(heat15_smooth,cmap="vlag",xticklabels=xticklabels,yticklabels=yticklabels)
# map.set_xticklabels(ax.get_xticklabels(), rotation=45)
# map.set_xlabel('Field reach', fontsize=12)
# map.set_ylabel('Gap junction strength', fontsize=12)
# map.set_title('Bulk-boundary integration of 15x15 tissue',fontsize=14)
# map.figure.tight_layout()
# plt.show()
#
# d10g = torch.load('./data/VmemEVCorrelation_global_50K_10x10.dat')
# d10l = torch.load('./data/VmemEVCorrelation_local_50K_10x10.dat')
# df10 = pd.DataFrame(d10g)
# df10l = pd.DataFrame(d10l)
# df10['total'] = df10[2]*(df10[2] - df10l[2])
# df10Plot = df10[[0,1,'total']]
# heat10 = df10Plot.pivot_table(index=0,columns=1,values='total')
# heat10_smooth = gaussian_filter(heat10, sigma=1)
# mx = np.abs(heat10_smooth).max()
# xticklabels = df10Plot[1].unique()
# xticklabels = (xticklabels*100/xticklabels.max()).round(0).astype(int)
# xticklabels = [str(i)+'%' for i in xticklabels]
# yticklabels = df10Plot[0].unique().round(2)
# fig, ax = plt.subplots()
# map = sns.heatmap(heat10_smooth,cmap="vlag",xticklabels=xticklabels,yticklabels=yticklabels)
# map.set_xticklabels(ax.get_xticklabels(), rotation=45)
# map.set_xlabel('Field reach', fontsize=12)
# map.set_ylabel('Gap junction strength', fontsize=12)
# map.set_title('Field-voltage global-local correlation diff for 10x10 tissue',fontsize=14)
# map.figure.tight_layout()
# plt.show()
#
# d15g = torch.load('./data/VmemEVCorrelation_global_50K_15x15.dat')
# d15l = torch.load('./data/VmemEVCorrelation_local_50K_15x15.dat')
# df15 = pd.DataFrame(d15g)
# df15l = pd.DataFrame(d15l)
# df15['total'] = df15[2]*(df15[2] - df15l[2])
# df15Plot = df15[[0,1,'total']]
# heat15 = df15Plot.pivot_table(index=0,columns=1,values='total')
# heat15_smooth = gaussian_filter(heat15, sigma=1)
# mx = np.abs(heat15_smooth).max()
# xticklabels = df15Plot[1].unique()
# xticklabels = (xticklabels*100/xticklabels.max()).round(0).astype(int)
# xticklabels = [str(i)+'%' for i in xticklabels]
# yticklabels = df15Plot[0].unique().round(2)
# fig, ax = plt.subplots()
# map = sns.heatmap(heat15_smooth,cmap="vlag",xticklabels=xticklabels,yticklabels=yticklabels)
# map.set_xticklabels(ax.get_xticklabels(), rotation=45)
# map.set_xlabel('Field reach', fontsize=12)
# map.set_ylabel('Gap junction strength', fontsize=12)
# map.set_title('Field-voltage global-local correlation diff for 15x15 tissue',fontsize=14)
# map.figure.tight_layout()
# plt.show()
#
# d10 = torch.load('./data/VmemEVSynchronicity_intra50K_10x10.dat')
# df10 = pd.DataFrame(d10)
# df10['total'] = df10[5] - df10[4]
# df10Plot = df10[[0,1,'total']]
# heat10 = df10Plot.pivot_table(index=0,columns=1,values='total')
# heat10_smooth = gaussian_filter(heat10, sigma=1)
# mx = np.abs(heat10_smooth).max()
# xticklabels = df10Plot[1].unique()
# xticklabels = (xticklabels*100/xticklabels.max()).round(0).astype(int)
# xticklabels = [str(i)+'%' for i in xticklabels]
# yticklabels = df10Plot[0].unique().round(2)
# fig, ax = plt.subplots()
# map = sns.heatmap(heat10_smooth,cmap="vlag",xticklabels=xticklabels,yticklabels=yticklabels)
# map.set_xticklabels(ax.get_xticklabels(), rotation=45)
# map.set_xlabel('Field reach', fontsize=12)
# map.set_ylabel('Gap junction strength', fontsize=12)
# map.set_title('Field-voltage synchronicity diff for 10x10 tissue',fontsize=14)
# map.figure.tight_layout()
# plt.show()
#
# d15 = torch.load('./data/VmemEVSynchronicity_intra50K_15x15.dat')
# df15 = pd.DataFrame(d15)
# df15['total'] = df15[5] - df15[4]
# df15Plot = df15[[0,1,'total']]
# heat15 = df15Plot.pivot_table(index=0,columns=1,values='total')
# heat15_smooth = gaussian_filter(heat15, sigma=1)
# mx = np.abs(heat15_smooth).max()
# xticklabels = df15Plot[1].unique()
# xticklabels = (xticklabels*150/xticklabels.max()).round(0).astype(int)
# xticklabels = [str(i)+'%' for i in xticklabels]
# yticklabels = df15Plot[0].unique().round(2)
# fig, ax = plt.subplots()
# map = sns.heatmap(heat15_smooth,cmap="vlag",xticklabels=xticklabels,yticklabels=yticklabels)
# map.set_xticklabels(ax.get_xticklabels(), rotation=45)
# map.set_xlabel('Field reach', fontsize=12)
# map.set_ylabel('Gap junction strength', fontsize=12)
# map.set_title('Field-voltage synchronicity diff for 15x15 tissue',fontsize=14)
# map.figure.tight_layout()
# plt.show()
#
