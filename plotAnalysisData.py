# Plots various heatmaps that were reported in 'Patterning in a bioelectric field network - summary of research so far V3.docx'
import utilities
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import argparse
import ast

# analysisMode = "fixBiasSweepWeightScreenGJ"  # fixWeightBiasSweepScreenGJ, fixBiasSweepWeightScreenGJ
# characteristicNames = "['TotalCorrelation','Entropy','evVmemDimensionDiff']"

parser = argparse.ArgumentParser()
parser.add_argument('--analysisMode', type=str, default='fixBiasSweepWeightScreenGJ')
parser.add_argument('--characteristicNames', type=str, default='["None"]')
parser.add_argument('--sample', type=str, default='All')

args = parser.parse_args()
analysisMode = args.analysisMode
characteristicNames = ast.literal_eval(args.characteristicNames)
sample = args.sample

fileNumberVersion = 0
if analysisMode == 'fixScreenGJSweepWeightBias':
    Sfx = 'FixedScreenSizeGJ_'
elif analysisMode == 'fixWeightBiasSweepScreenGJ':
    Sfx = 'FixedWeightBias_'
elif analysisMode == 'fixBiasSweepWeightScreenGJ':
    if ('Sensitivity' in characteristicNames) and ('Hessian' in characteristicNames):
        Sfx = 'FixedBias_'
    elif 'Sensitivity' in characteristicNames:
        Sfx = 'FixedBias_Sensitivity_'
    elif 'Hessian' in characteristicNames:
        Sfx = 'FixedBias_Hessian_'
    elif ('Covariance' in characteristicNames) or ('CovarianceNeuralComplexity' in characteristicNames):
        Sfx = 'FixedBias_Covariance_'
    else:
        Sfx = 'FixedBias_'
elif analysisMode == 'fixBiasSweepWeightLigandGJ':
    Sfx = 'FixedBias_Ligand_'
elif analysisMode == 'patternability':
    Sfx = 'FixedBias_Patternability_'
else:
    Sfx = ''
if fileNumberVersion > 0:
    fileVersionSfx = '_V' + str(fileNumberVersion)
else:
    fileVersionSfx = ''

def plotCharacteristic(df,characteristic=None):
    df['fieldTransductionWeight'] = [df['fieldTransductionWeight'][i].item() for i in range(len(df['fieldTransductionWeight']))]
    if characteristic == 'TSEComplexity':
        # dfComplex = df.melt(id_vars=['GJStrength', 'fieldRange', 'fieldTransductionWeight'],value_vars=['TSEComplexityHomo','TSEComplexityHetero'],var_name='Sample', value_name='TSEComplexity')
        # dfComplex['Sample'] = dfComplex['Sample'].replace({'TSEComplexityHomo':'Homogenous','TSEComplexityHetero':'Heterogenous'})
        fig, ax1 = plt.subplots()
        # sns.lineplot(data=dfComplex,x='fieldRange',y='TSEComplexity',hue='Sample',errorbar='ci')
        sns.lineplot(data=df,x='fieldRange',y='TSEComplexityHomo',errorbar='ci',color='blue',ax=ax1)
        ax2 = ax1.twinx()
        sns.lineplot(data=df,x='fieldRange',y='TSEComplexityHetero',errorbar='ci',color='red',ax=ax2)
        fieldRangeValues = df['fieldRange'].unique()
        plt.xticks(fieldRangeValues,fieldRangeValues)
        ax1.set_xlabel('Field Range',fontsize=16)
        ax1.set_ylabel('Complexity',fontsize=16)
        blue_line = mlines.Line2D([],[],color='blue',linestyle='-',label='Homogenous')
        red_line = mlines.Line2D([],[],color='red',linestyle='-',label='Heterogenous')
        ax1.legend(handles=[blue_line,red_line],loc='upper left',fontsize=12)
        # ax1.legend(title='Initial condition',title_fontsize=12,fontsize=12,bbox_to_anchor=(1.0,1.0))
        # ax.annotate("Optimal",xy=(4.0,-0.05),xytext=(4.5,0.2),arrowprops=dict(facecolor='black',arrowstyle='->',connectionstyle="arc3,rad=-0.2"),fontsize=12)
        plt.tight_layout()
        plt.savefig('./data/modelCharacteristics_FixedBias_' + characteristic + '.png',bbox_inches="tight")
    if characteristic == 'Dimensionality':
        # dfComprDiff = df.melt(id_vars=['GJStrength', 'fieldRange', 'fieldTransductionWeight'],value_vars=['evAggVmemDimensionDiffHomo', 'evAggVmemDimensionDiffHetero'],var_name='Sample', value_name='CompressionDiff')
        # dfComprDiff['Sample'] = dfComprDiff['Sample'].replace({'evAggVmemDimensionDiffHomo':'Homogenous', 'evAggVmemDimensionDiffHetero':'Heterogenous'})
        fig, ax1 = plt.subplots()
        # sns.lineplot(data=dfComprDiff,x='fieldRange',y='CompressionDiff',hue='Sample',errorbar='ci')
        sns.lineplot(data=df,x='fieldRange',y='evAggVmemDimensionDiffHomo',errorbar='ci',color='blue',ax=ax1)
        ax2 = ax1.twinx()
        sns.lineplot(data=df,x='fieldRange',y='evAggVmemDimensionDiffHetero',errorbar='ci',color='red',ax=ax2)
        fieldRangeValues = df['fieldRange'].unique()
        plt.xticks(fieldRangeValues,fieldRangeValues)
        ax1.set_xlabel('Field Range',fontsize=16)
        ax1.set_ylabel('Compression Difference',fontsize=16)
        blue_line = mlines.Line2D([],[],color='blue',linestyle='-',label='Homogenous')
        red_line = mlines.Line2D([],[],color='red',linestyle='-',label='Heterogenous')
        ax1.legend(handles=[blue_line,red_line],loc='upper left',fontsize=12)
        # ax1.legend(title='Initial condition',title_fontsize=12,fontsize=12,bbox_to_anchor=(1.0,1.0))
        # ax1.annotate("Optimal",xy=(4.0,-0.003),xytext=(4.5,0.005),arrowprops=dict(color=sns.color_palette()[0],arrowstyle='->',connectionstyle="arc3,rad=-0.2"),fontsize=12)
        # ax.annotate("Optimal",xy=(10.0,-0.003),xytext=(10.5,0.005),arrowprops=dict(color=sns.color_palette()[1],arrowstyle='->',connectionstyle="arc3,rad=-0.2"),fontsize=12)
        plt.tight_layout()
        plt.savefig('./data/modelCharacteristics_FixedBias_' + characteristic + '.png',bbox_inches="tight")
    if characteristic == 'PositionalInformation':
        # dfPosInfo = df.melt(id_vars=['GJStrength', 'fieldRange', 'fieldTransductionWeight'],value_vars=['PositionalInformationHomo','PositionalInformationHetero'],var_name='Sample', value_name='PositionalInformation')
        # dfPosInfo['Sample'] = dfPosInfo['Sample'].replace({'PositionalInformationHomo':'Homogenous','PositionalInformationHetero':'Heterogenous'})
        fig, ax1 = plt.subplots()
        xvar = 'fieldRange'
        # sns.lineplot(data=dfPosInfo,x=xvar,y='PositionalInformation',hue='Sample',errorbar='ci')
        sns.lineplot(data=df,x=xvar,y='PositionalInformationHomo',color='blue',errorbar='ci',ax=ax1)
        ax2 = ax1.twinx()
        sns.lineplot(data=df,x=xvar,y='PositionalInformationHetero',color='red',errorbar='ci',ax=ax2)
        xvals = df[xvar].unique()
        plt.xticks(xvals,xvals)
        ax1.set_xlabel(xvar,fontsize=16)
        ax1.set_ylabel('Positional Information',fontsize=16)
        blue_line = mlines.Line2D([],[],color='blue',linestyle='-',label='Homogenous')
        red_line = mlines.Line2D([],[],color='red',linestyle='-',label='Heterogenous')
        ax1.legend(handles=[blue_line,red_line],loc='upper left',fontsize=12)
        # ax1.legend(title='Initial condition',title_fontsize=12,fontsize=12,bbox_to_anchor=(1.0,1.0))
        # ax.annotate("Optimal",xy=(4.0,-0.05),xytext=(4.5,0.2),arrowprops=dict(facecolor='black',arrowstyle='->',connectionstyle="arc3,rad=-0.2"),fontsize=12)
        plt.tight_layout()
        plt.savefig('./data/modelCharacteristics_FixedBias_' + characteristic + '.png',bbox_inches="tight")
    if characteristic == 'Entropy':
        # dfEntr = df.melt(id_vars=['GJStrength', 'fieldRange', 'fieldTransductionWeight'],value_vars=['EntropyHomo','EntropyHetero'],var_name='Sample', value_name='Entropy')
        # dfEntr['Sample'] = dfEntr['Sample'].replace({'EntropyHomo':'Homogenous','EntropyHetero':'Heterogenous'})
        fig, ax1 = plt.subplots()
        sns.lineplot(data=df,x='fieldRange',y='EntropyHomo',color='blue',errorbar='ci',ax=ax1)
        ax2 = ax1.twinx()
        sns.lineplot(data=df,x='fieldRange',y='EntropyHetero',color='red',errorbar='ci',ax=ax2)
        fieldRangeValues = df['fieldRange'].unique()
        plt.xticks(fieldRangeValues,fieldRangeValues)
        ax1.set_xlabel('Field Range',fontsize=16)
        ax1.set_ylabel('Entropy',fontsize=16)
        ax1.legend(title='Initial condition',title_fontsize=12,fontsize=12,bbox_to_anchor=(1.0,1.0))
        # ax.annotate("Optimal",xy=(4.0,-0.05),xytext=(4.5,0.2),arrowprops=dict(facecolor='black',arrowstyle='->',connectionstyle="arc3,rad=-0.2"),fontsize=12)
        plt.tight_layout()
        plt.savefig('./data/modelCharacteristics_FixedBias_' + characteristic + '.png',bbox_inches="tight")
    if characteristic == 'Correlation':
        dfCorrSample = df.melt(id_vars=['GJStrength', 'fieldRange', 'fieldTransductionWeight'],value_vars=['CorrelationHomo','CorrelationHetero','TotalCorrelationHomo','TotalCorrelationHetero'],var_name='Sample', value_name='Correlation')
        dfCorrSample['Sample'] = dfCorrSample['Sample'].replace({'CorrelationHomo':'Homogenous', 'TotalCorrelationHomo':'Homogenous','CorrelationHetero':'Heterogenous','TotalCorrelationHetero':'Heterogenous'})
        dfCorrMeasure = df.melt(id_vars=['GJStrength', 'fieldRange', 'fieldTransductionWeight'],value_vars=['CorrelationHomo','CorrelationHetero','TotalCorrelationHomo','TotalCorrelationHetero'],var_name='Measure', value_name='Correlation')
        dfCorrMeasure['Measure'] = dfCorrMeasure['Measure'].replace({'CorrelationHomo':'Pairwise', 'TotalCorrelationHomo':'Total','CorrelationHetero':'Pairwise','TotalCorrelationHetero':'Total'})
        dfCorr = dfCorrSample.merge(dfCorrMeasure)
        # fig, ax = plt.subplots()
        g = sns.FacetGrid(dfCorr, col="Measure",hue='Sample',sharey=False)
        g.map(sns.lineplot,'fieldRange','Correlation',errorbar='ci')
        g.set_axis_labels("Field Range", "Correlation",fontsize=12)
        fieldRangeValues = df['fieldRange'].unique()
        g.set(xticks=fieldRangeValues)
        g.add_legend(title="Initial condition",title_fontsize=12,fontsize=12)
        sns.move_legend(g, "upper right", bbox_to_anchor=(1.0,0.8))
        g.axes.ravel()[0].set_title('Pairwise Correlation',fontsize=12)
        g.axes.ravel()[0].annotate("Optimal",xy=(4.0,0.52),xytext=(5.8,0.57),arrowprops=dict(facecolor='black',arrowstyle='->',connectionstyle="arc3,rad=-0.2"),fontsize=10)
        g.axes.ravel()[1].set_title('Total Correlation',fontsize=12)
        g.axes.ravel()[1].annotate("Optimal",xy=(4.0,-0.05),xytext=(4.5,0.15),arrowprops=dict(facecolor='black',arrowstyle='->',connectionstyle="arc3,rad=-0.2"),fontsize=10)
        plt.tight_layout()  # g.tight_layout() works but messes the layout
        plt.savefig('./data/modelCharacteristics_FixedBias_' + characteristic + '.png',bbox_inches="tight")
    if characteristic == 'JacobianAndHessian':
        fig, ax1 = plt.subplots()
        sns.lineplot(data=df,x='fieldRange',y='Jacobian',errorbar='ci',ax=ax1,color='black',linestyle='-')
        ax2 = ax1.twinx()
        sns.lineplot(data=df,x='fieldRange',y='Hessian',errorbar='ci',ax=ax2,color='black',linestyle='--')
        ax1.set_xlabel('Field Range',fontsize=16)
        ax1.set_ylabel('Jacobian Magnitude',fontsize=16)
        ax2.set_ylabel('Hessian Magnitude',fontsize=16)
        fieldRangeValues = df['fieldRange'].unique()
        plt.xticks(fieldRangeValues,fieldRangeValues)
        blue_line = mlines.Line2D([],[],color='black',linestyle='-',label='Jacobian')
        red_line = mlines.Line2D([],[],color='black',linestyle='--',label='Hessian')
        ax1.legend(handles=[blue_line,red_line],loc='upper right',fontsize=12)
        # ax1.annotate("Optimal",xy=(4.0,0.000053),xytext=(4.5,0.00008),arrowprops=dict(facecolor='black',arrowstyle='->',connectionstyle="arc3,rad=-0.2"),fontsize=12)
        # ax1.set_ylim(0.00005,0.0004)
        plt.tight_layout()
        plt.savefig('./data/modelCharacteristics_FixedBias_' + characteristic + '.png',bbox_inches="tight")

if analysisMode == 'patternability':
    fileRange = range(1,501)
    GJStrength, fieldScreenSize, fieldTransductionWeight, PatternabilityMean, PatternabilityMin = [], [], [], [], []
    read = True
    for fileNumber in fileRange:
        read = True
        while read:
            try:
                filename = './data/ModelCharacteristics_' + Sfx + str(fileNumber) + fileVersionSfx + '.dat'
                data = torch.load(filename)
            except:
                read = False
            else:
                read = False
                GJStrength.append(data[1]['GJParameters']['GJStrength'].round(decimals=2))
                fieldScreenSize.append(data[1]['fieldParameters']['fieldScreenSize'])
                fieldTransductionWeight.append(data[1]['fieldParameters']['fieldTransductionWeight'].round(decimals=2).item())
                maxSamples = np.array(list(data.keys())).max()
                meanPatternability = 2.5-np.array([data[sample]['trainParameters']['bestLoss'] for sample in range(1,maxSamples)]).mean().round(2)  # inverse of distance
                minPatternability = 2.5-np.array([data[sample]['trainParameters']['bestLoss'] for sample in range(1,maxSamples)]).min().round(2)
                PatternabilityMean.append(meanPatternability)
                PatternabilityMin.append(minPatternability)
    df = pd.DataFrame({'GJStrength':GJStrength,'fieldRange':fieldScreenSize,'fieldTransductionWeight':fieldTransductionWeight,
                       'PatternabilityMean':PatternabilityMean,'PatternabilityMin':PatternabilityMin})
    for characteristic in ['PatternabilityMean','PatternabilityMin']:
        heatmap = df.pivot_table(index='GJStrength',columns='fieldRange',values=characteristic)
        # heatmap_smooth = gaussian_filter(heatmap, sigma=1)
        heatmap_smooth = heatmap
        fig, ax = plt.subplots()
        map = sns.heatmap(heatmap_smooth,cmap='seismic')
        # plt.show()
        plt.savefig('./data/modelCharacteristics_FixedBias_' + characteristic + '_.png',bbox_inches="tight")

if analysisMode == "fixBiasSweepWeightScreenGJ":
    fileRange = range(1,501)
    if ('Sensitivity' in characteristicNames) and ('Hessian' in characteristicNames):
        (GJStrength, fieldScreenSize, fieldTransductionWeight, Sensitivity, Hessian) = [], [], [], [], [],
        for fileNumber in fileRange:
            filename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + fileVersionSfx + '.dat'
            data = torch.load(filename)
            GJStrength.append(data['GJParameters']['GJStrength'].round(decimals=2))
            fieldScreenSize.append(data['fieldParameters']['fieldScreenSize'])
            fieldTransductionWeight.append(data['fieldParameters']['fieldTransductionWeight'].round(decimals=2))
            _, VmemToVmem = data['characteristics']['Sensitivity']['Derivatives']
            VmemToVmem = VmemToVmem.abs().clone()
            nzidx = np.array([VmemToVmem[i].any().item() for i in range(VmemToVmem.shape[0])])
            if nzidx.any():
                VmemToVmem = VmemToVmem[nzidx]
            SensitivityTimeSeries = np.array([(VmemToVmem[t]).mean().item() for t in range(VmemToVmem.shape[0])])
            Sensitivity.append(np.abs(SensitivityTimeSeries.mean()))
            evToVmemToVmem = data['characteristics']['Hessian']['Derivatives']
            evToVmemToVmem = evToVmemToVmem.abs().clone()
            nzidx = np.array([evToVmemToVmem[i].any().item() for i in range(evToVmemToVmem.shape[0])])
            if nzidx.any():
                evToVmemToVmem = evToVmemToVmem[nzidx]
            HessianTimeSeries = np.array([(evToVmemToVmem[t]).mean().item() for t in range(evToVmemToVmem.shape[0])])
            Hessian.append(np.abs(HessianTimeSeries.mean()))
        df = pd.DataFrame({'GJStrength':GJStrength,'fieldRange':fieldScreenSize,'fieldTransductionWeight':fieldTransductionWeight,
                           'Jacobian':Sensitivity,'Hessian':Hessian})
        plotCharacteristic(df,'JacobianAndHessian')
    elif 'Sensitivity' in characteristicNames:
        (GJStrength, fieldScreenSize, fieldTransductionWeight, CausalDistance, CausalDistanceDerivative,
         Sensitivity, SensitivityDerivative, SelfOtherTradeoff) = [], [], [], [], [], [], [], []
        for fileNumber in fileRange:
            filename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + fileVersionSfx + '.dat'
            data = torch.load(filename)
            GJStrength.append(data['GJParameters']['GJStrength'].round(decimals=2))
            fieldScreenSize.append(data['fieldParameters']['fieldScreenSize'])
            fieldTransductionWeight.append(data['fieldParameters']['fieldTransductionWeight'].round(decimals=2))
            eVToVmem, VmemToVmem = data['characteristics']['Sensitivity']
            VmemToVmem = VmemToVmem.abs()
            nzidx = np.array([VmemToVmem[i].any().item() for i in range(VmemToVmem.shape[0])])
            if nzidx.any():
                VmemToVmem = VmemToVmem[nzidx]
                weights = VmemToVmem.clone()
                # weights /= weights.max()
            else:
                weights = VmemToVmem.clone()
            utils = utilities.utilities()
            numRows, numCols = 11, 11
            xc, yc = torch.repeat_interleave(torch.arange(numRows),numCols).view(1,-1), torch.tile(torch.arange(numCols),(numRows,)).view(1,-1)
            distances = utils.computePairwiseDistances((xc,yc),(xc,yc))
            CausalDistanceTimeSeries = np.array([(weights[t,:,:]*distances[0,:,[0,5,60]]).mean().item() for t in range(weights.shape[0])])
            CausalDistanceTimeSeries = CausalDistanceTimeSeries/CausalDistanceTimeSeries.max()  # normalization
            CausalDistance.append(CausalDistanceTimeSeries.mean())
            CausalDistanceDerivative.append(np.abs(CausalDistanceTimeSeries[1:]-CausalDistanceTimeSeries[0:-1]).mean())
            SensitivityTimeSeries = np.array([(VmemToVmem[t]).mean().item() for t in range(VmemToVmem.shape[0])])
            # SensitivityTimeSeries = SensitivityTimeSeries / SensitivityTimeSeries.max()  # normalization
            Sensitivity.append(np.abs(SensitivityTimeSeries.mean()))
            SensitivityDerivative.append(np.abs(SensitivityTimeSeries[1:]-SensitivityTimeSeries[0:-1]).mean())
            selfSensitivity = np.array([VmemToVmem[t,[0,5,60],[0,1,2]] for t in range(VmemToVmem.shape[0])]).reshape(-1,3)
            otherCells = np.setdiff1d(range(VmemToVmem.shape[1]),[0,5,60]).tolist()
            otherSensitivity = np.array([VmemToVmem[t,otherCells,cell].mean().item() for t in range(VmemToVmem.shape[0])
                                         for cell in range(VmemToVmem.shape[2])]).reshape(-1,3)
            selfOtherDiff = (selfSensitivity - otherSensitivity).sum()
            SelfOtherTradeoff.append(selfOtherDiff)
        df = pd.DataFrame({'GJStrength':GJStrength,'fieldScreenSize':fieldScreenSize,'fieldTransductionWeight':fieldTransductionWeight,
                           'CausalDistance':CausalDistance,'CausalDistanceDerivative':CausalDistanceDerivative,
                           'Sensitivity':Sensitivity,'SensitivityDerivative':SensitivityDerivative,'SelfOtherTradeoff':SelfOtherTradeoff})
        torch.save(df,'./data/SensitivityJacobianDataframe.dat')
        # for characteristic in ['CausalDistance','CausalDistanceDerivative','Sensitivity','SensitivityDerivative','SelfOtherTradeoff']:
        #     heatmap = df.pivot_table(index='GJStrength',columns='fieldScreenSize',values=characteristic)
        #     # heatmap_smooth = gaussian_filter(heatmap, sigma=1)
        #     heatmap_smooth = heatmap
        #     fig, ax = plt.subplots()
        #     map = sns.heatmap(heatmap_smooth,cmap='seismic')
        #     # plt.show()
        #     plt.savefig('./data/modelCharacteristics_FixedBias_' + characteristic + '.png',bbox_inches="tight")
    elif 'Hessian' in characteristicNames:
        (GJStrength, fieldScreenSize, fieldTransductionWeight, Hessian, HessianDerivative, HessianCausalDistance,
         HessianCausalDistanceDerivative, SelfOtherTradeoff) = [], [], [], [], [], [], [], []
        for fileNumber in fileRange:
            filename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + fileVersionSfx + '.dat'
            data = torch.load(filename)
            GJStrength.append(data['GJParameters']['GJStrength'].round(decimals=2))
            fieldScreenSize.append(data['fieldParameters']['fieldScreenSize'])
            fieldTransductionWeight.append(data['fieldParameters']['fieldTransductionWeight'].round(decimals=2))
            eVToVmemToVmem = data['characteristics']['Hessian']['Derivatives'].abs()
            nzidx = np.array([eVToVmemToVmem[i].any().item() for i in range(eVToVmemToVmem.shape[0])])
            # for t in range(weights.shape[0]):
            #     weights[t] /= weights[t].max()
            if nzidx.any():
                eVToVmemToVmem = eVToVmemToVmem[nzidx]
                weights = eVToVmemToVmem.clone()
                # weights /= weights.max()
            else:
                weights = eVToVmemToVmem.clone()
            utils = utilities.utilities()
            numRows, numCols = 11, 11
            xc, yc = torch.repeat_interleave(torch.arange(numRows),numCols).view(1,-1), torch.tile(torch.arange(numCols),(numRows,)).view(1,-1)
            distances = utils.computePairwiseDistances((xc,yc),(xc,yc))
            numExtracellularGridPoints = (numRows+1)*(numCols+1)
            HessianCausalDistanceTimeSeries = np.array([np.array([(weights[t,ec,:,:]*distances[0,:,[0,5,60]]).mean().item() for ec in range(numExtracellularGridPoints)]).mean()
                                                        for t in range(eVToVmemToVmem.shape[0])])
            HessianCausalDistance.append(HessianCausalDistanceTimeSeries.mean())
            HessianCausalDistanceDerivative.append(np.abs(HessianCausalDistanceTimeSeries[1:]-HessianCausalDistanceTimeSeries[0:-1]).mean())
            HessianTimeSeries = np.array([(eVToVmemToVmem[t]).mean().item() for t in range(eVToVmemToVmem.shape[0])])
            # HessianTimeSeries = HessianTimeSeries / HessianTimeSeries.max()  # normalization
            Hessian.append(np.abs(HessianTimeSeries.mean()))
            HessianDerivative.append(np.abs(HessianTimeSeries[1:]-HessianTimeSeries[0:-1]).mean())
            selfSensitivity = np.array([eVToVmemToVmem[t,:,[0,5,60],[0,1,2]].sum(0) for t in range(eVToVmemToVmem.shape[0])]).reshape(-1,3)
            otherCells = np.setdiff1d(range(eVToVmemToVmem.shape[2]),[0,5,60]).tolist()
            otherSensitivity = np.array([eVToVmemToVmem[t,:,otherCells,cell].sum(0).mean().item() for t in range(eVToVmemToVmem.shape[0])
                                         for cell in range(eVToVmemToVmem.shape[3])]).reshape(-1,3)
            selfOtherDiff = (selfSensitivity - otherSensitivity).sum()
            SelfOtherTradeoff.append(selfOtherDiff)
        df = pd.DataFrame({'GJStrength':GJStrength,'fieldScreenSize':fieldScreenSize,'fieldTransductionWeight':fieldTransductionWeight,
                           'Hessian':Hessian,'HessianDerivative':HessianDerivative,
                           'HessianCausalDistance':HessianCausalDistance,'HessianCausalDistanceDerivative':HessianCausalDistanceDerivative,
                           'SelfOtherTradeoff':SelfOtherTradeoff})
        for characteristic in ['Hessian','HessianDerivative','HessianCausalDistance','HessianCausalDistanceDerivative','SelfOtherTradeoff']:
            heatmap = df.pivot_table(index='GJStrength',columns='fieldScreenSize',values=characteristic)
            # heatmap_smooth = gaussian_filter(heatmap, sigma=1)
            heatmap_smooth = heatmap
            fig, ax = plt.subplots()
            map = sns.heatmap(heatmap_smooth,cmap='seismic')
            # plt.show()
            plt.savefig('./data/modelCharacteristics_FixedBias_' + characteristic + '.png',bbox_inches="tight")
    elif 'Covariance' in characteristicNames:
        numRows, numCols = 11, 11
        numCells = numRows * numCols
        cellIndices = range(numCells)
        GJStrength, fieldScreenSize, fieldTransductionWeight, SelfOtherTradeoff = [], [], [], []
        for fileNumber in fileRange:
            filename = './data/modelCharacteristics_' + Sfx + str(float(fileNumber)) + fileVersionSfx + '.dat'
            data = torch.load(filename)
            GJStrength.append(data['GJParameters']['GJStrength'].round(decimals=2))
            fieldScreenSize.append(data['fieldParameters']['fieldScreenSize'])
            fieldTransductionWeight.append(data['fieldParameters']['fieldTransductionWeight'].round(decimals=2))
            CovarianceMatrices = data['characteristics']['Covariance']
            selfCovariance = np.array([CovarianceMatrices[t,cellIndices,cellIndices] for t in range(CovarianceMatrices.shape[0])]).reshape(-1,numCells)
            otherCovariance = np.array([CovarianceMatrices[t,np.setdiff1d(cellIndices,cell),cell].mean().item() for t in range(CovarianceMatrices.shape[0])
                                         for cell in range(CovarianceMatrices.shape[2])]).reshape(-1,numCells)
            selfOtherDiff = (selfCovariance - otherCovariance).sum()
            SelfOtherTradeoff.append(selfOtherDiff)
        df = pd.DataFrame({'GJStrength':GJStrength,'fieldScreenSize':fieldScreenSize,'fieldTransductionWeight':fieldTransductionWeight,
                           'SelfOtherTradeoff':SelfOtherTradeoff})
        heatmap = df.pivot_table(index='GJStrength',columns='fieldScreenSize',values='SelfOtherTradeoff')
        # heatmap_smooth = gaussian_filter(heatmap, sigma=1)
        heatmap_smooth = heatmap
        fig, ax = plt.subplots()
        map = sns.heatmap(heatmap_smooth,cmap='seismic')
        # plt.show()
        plt.savefig('./data/modelCharacteristics_FixedBias_' + 'SelfOtherTradeoff' + '.png',bbox_inches="tight")
    elif 'CovarianceNeuralComplexity' in characteristicNames:
        numRows, numCols = 11, 11
        numCells = numRows * numCols
        cellIndices = range(numCells)
        GJStrength, fieldScreenSize, fieldTransductionWeight, CovarianceNeuralComplexityHomo, CovarianceNeuralComplexityHetero = [], [], [], [], []
        for fileNumber in fileRange:
            filename = './data/modelCharacteristics_' + Sfx + str(float(fileNumber)) + fileVersionSfx + '.dat'
            data = torch.load(filename)
            GJStrength.append(data['GJParameters']['GJStrength'].round(decimals=2))
            fieldScreenSize.append(data['fieldParameters']['fieldScreenSize'])
            fieldTransductionWeight.append(data['fieldParameters']['fieldTransductionWeight'].round(decimals=2))
            CovarianceMatrices = data['characteristics']['Covariance']
            scales = np.linspace(2,numCells-1,50,dtype=np.int16)
            totalSubScalesDeterminantHomo, totalSubScalesDeterminantHetero = 0, 0
            for scale in scales:
                totalDeterminantScaleHomo, totalDeterminantScaleHetero = 0, 0
                for subsetsample in range(100):
                    cellIndicesSubset = np.random.choice(numCells,scale,replace=False)
                    r, c = np.repeat(cellIndicesSubset,scale), np.tile(cellIndicesSubset,scale)
                    CovarianceMatrixScaleHomo = CovarianceMatrices[0,r,c].reshape(scale,scale)  # homogenous sample
                    totalDeterminantScaleHomo += np.linalg.det(CovarianceMatrixScaleHomo).__abs__()
                    CovarianceMatrixScaleHetero = CovarianceMatrices[:,r,c].reshape(-1,scale,scale)  # homogenous sample
                    totalDeterminantScaleHetero += np.linalg.det(CovarianceMatrixScaleHetero).__abs__().mean()
                totalDeterminantScaleHomo /= 100
                totalSubScalesDeterminantHomo += totalDeterminantScaleHomo
                totalDeterminantScaleHetero /= 100
                totalSubScalesDeterminantHetero += totalDeterminantScaleHetero
            fullScaleDeterminantHomo = np.linalg.det(CovarianceMatrices[0]).__abs__()  # homogenous sample
            complexityHomo = totalSubScalesDeterminantHomo - (np.sum(scales)*fullScaleDeterminantHomo/numCells)
            fullScaleDeterminantHetero = np.linalg.det(CovarianceMatrices[1:]).__abs__().mean() # heterogenous sample mean
            complexityHetero = totalSubScalesDeterminantHetero - (np.sum(scales)*fullScaleDeterminantHetero/numCells)
            CovarianceNeuralComplexityHomo.append(complexityHomo)
            CovarianceNeuralComplexityHetero.append(complexityHetero)
        df = pd.DataFrame({'GJStrength':GJStrength,'fieldScreenSize':fieldScreenSize,'fieldTransductionWeight':fieldTransductionWeight,
                           'CovarianceNeuralComplexityHomo':CovarianceNeuralComplexityHomo,'CovarianceNeuralComplexityHetero':CovarianceNeuralComplexityHetero})
        torch.save(df,'./data/CovarianceNeuralComplexityDataframe.dat')
        # heatmap = df.pivot_table(index='GJStrength',columns='fieldScreenSize',values='CovarianceNeuralComplexity')
        # # heatmap_smooth = gaussian_filter(heatmap, sigma=1)
        # heatmap_smooth = heatmap
        # fig, ax = plt.subplots()
        # map = sns.heatmap(heatmap_smooth,cmap='seismic')
        # # plt.show()
        # plt.savefig('./data/modelCharacteristics_FixedBias_' + 'CovarianceNeuralComplexity' + '.png',bbox_inches="tight")
    else:
        GJStrength, fieldScreenSize, fieldTransductionWeight = [], [], []
        Robustness = []
        if sample == 'Segregated':
            (CorrelationHomo, TotalCorrHomo, EntropyHomo, evDimensionHomo, evAggDimensionHomo, vmemDimensionHomo,
            evAggVmemDimensionDiffHomo, evVmemDimensionDiffHomo, evAggVmemDimensionRatioHomo, TSEComplexityHomo,
             PositionalInformationHomo, CellfreqsHomo) = [], [], [], [], [], [], [], [], [], [], [], []
            (CorrelationHetero, TotalCorrHetero, EntropyHetero, evDimensionHetero, evAggDimensionHetero, vmemDimensionHetero,
            evAggVmemDimensionDiffHetero, evVmemDimensionDiffHetero, evAggVmemDimensionRatioHetero, TSEComplexityHetero,
             PositionalInformationHetero, CellfreqsHetero) = [], [], [], [], [], [], [], [], [], [], [], []
        else:
            (Correlation, TotalCorr, Entropy, evDimension, evAggDimension, vmemDimension, eVAggVmemDimemsion, evAggVmemDimensionDiff,
            evVmemDimensionDiff, evAggVmemDimensionRatio, eVAggVmemDimensionMI) = [], [], [], [], [], [], [], [], [], [], []
        for fileNumber in fileRange:
            filename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + fileVersionSfx + '.dat'
            data = torch.load(filename)
            GJStrength.append(data['GJParameters']['GJStrength'].round(decimals=2))
            fieldScreenSize.append(data['fieldParameters']['fieldScreenSize'])
            fieldTransductionWeight.append(data['fieldParameters']['fieldTransductionWeight'].round(decimals=2))
            if sample == 'All':
                Correlation.append(data['characteristics']['Correlation'].mean().item())
                TotalCorr.append(np.array(data['characteristics']['Information'][0]).mean().item())
                Entropy.append(np.array(data['characteristics']['Information'][1]).mean().item())
                Robustness.append(0.1-data['characteristics']['Robustness'].mean().item())  # inverse of distance
                evDim, evAggDim, vmemDim, eVAggVmemDim, _ = data['characteristics']['Dimensionality']
                evDim, evAggDim, vmemDim, eVAggVmemDim = np.array(evDim), np.array(evAggDim), np.array(vmemDim), np.array(eVAggVmemDim)
                evDimension.append(evDim[:,[0,1,2]].sum(1).mean())
                evAggDimension.append(evAggDim[:,[0,1,2]].sum(1).mean())
                vmemDimension.append(vmemDim[:,[0,1,2]].sum(1).mean())
                eVAggVmemDimemsion.append(eVAggVmemDim[:,[0,1,2]].sum(1).mean())
                evAggVmemDimensionDiff.append((evAggDim[:,[0,1,2]].sum(1) - vmemDim[:,[0,1,2]].sum(1)).mean())
                eVAggVmemDimensionMI.append((2 - evAggDim[:,[0,1,2]].sum(1) - vmemDim[:,[0,1,2]].sum(1) + eVAggVmemDim[:,[0,1,2]].sum(1)).mean())
                evVmemDimensionDiff.append((evDim[:,[0,1,2]].sum(1) - vmemDim[:,[0,1,2]].sum(1)).mean())
                evAggVmemDimensionRatio.append((evAggDim[:,[0,1,2]].sum(1) / vmemDim[:,[0,1,2]].sum(1)).mean())
            elif sample == 'Homogenous':
                Correlation.append(data['characteristics']['Correlation'][0].item())
                TotalCorr.append(np.array(data['characteristics']['Information'][0])[0].item())
                Entropy.append(np.array(data['characteristics']['Information'][1])[0].item())
                Robustness.append(0.1-data['characteristics']['Robustness'].mean().item())  # inverse of distance
                evDim, evAggDim, vmemDim, eVAggVmemDim = data['characteristics']['Dimensionality']
                evDim, evAggDim, vmemDim, eVAggVmemDim = np.array(evDim), np.array(evAggDim), np.array(vmemDim), np.array(eVAggVmemDim)
                evDimension.append(evDim[0,[0,1,2]].sum().mean())
                evAggDimension.append(evAggDim[0,[0,1,2]].sum().mean())
                vmemDimension.append(vmemDim[0,[0,1,2]].sum().mean())
                eVAggVmemDimemsion.append(eVAggVmemDim[0,[0,1,2]].sum().mean())
                evAggVmemDimensionDiff.append((evAggDim[0,[0,1,2]].sum() - vmemDim[0,[0,1,2]].sum()).mean())
                eVAggVmemDimensionMI.append((2 - evAggDim[0,[0,1,2]].sum() - vmemDim[0,[0,1,2]].sum() + eVAggVmemDim[0,[0,1,2]].sum()).mean())
                evVmemDimensionDiff.append((evDim[0,[0,1,2]].sum() - vmemDim[0,[0,1,2]].sum()).mean())
                evAggVmemDimensionRatio.append((evAggDim[0,[0,1,2]].sum() / vmemDim[0,[0,1,2]].sum()).mean())
            elif sample == 'Segregated':
                evDim, evAggDim, vmemDim, _ = data['characteristics']['Dimensionality']
                evDim, evAggDim, vmemDim = np.array(evDim), np.array(evAggDim), np.array(vmemDim)
                Robustness.append(0.1 - data['characteristics']['Robustness'].mean().item())  # inverse of distance
                CorrelationHomo.append(data['characteristics']['Correlation'][0].item())
                TotalCorrHomo.append(np.array(data['characteristics']['Information'][0])[0].item())
                EntropyHomo.append(np.array(data['characteristics']['Information'][1])[0].item())
                TSEComplexityHomo.append(np.array(data['characteristics']['TSEComplexity'])[0].item())
                numones = np.amax(data['characteristics']['CellularFrequency'][0][0].reshape(1,-1),axis=0,initial=1)
                numzeros = np.amax((data['simParameters']['numSimIters']-numones).reshape(1,-1),axis=0,initial=1)
                numones1to0 = data['characteristics']['CellularFrequency'][1][0]
                numones0to1 = data['characteristics']['CellularFrequency'][2][0]
                cellfreqs = ((numones0to1/numones)+(numones1to0/numzeros))/2
                numuniquecellfres = len(np.unique(cellfreqs))
                numcells = np.prod(data['latticeDims'])
                PositionalInformationHomo.append(numuniquecellfres/numcells)
                CellfreqsHomo.append(cellfreqs)
                evDimensionHomo.append(evDim[0,[0,1,2]].sum().mean())
                evAggDimensionHomo.append(evAggDim[0,[0,1,2]].sum().mean())
                vmemDimensionHomo.append(vmemDim[0,[0,1,2]].sum().mean())
                evAggVmemDimensionDiffHomo.append((evAggDim[0,[0,1,2]].sum() - vmemDim[0,[0,1,2]].sum()).mean())
                evVmemDimensionDiffHomo.append((evDim[0,[0,1,2]].sum() - vmemDim[0,[0,1,2]].sum()).mean())
                evAggVmemDimensionRatioHomo.append((evAggDim[0,[0,1,2]].sum() / vmemDim[0,[0,1,2]].sum()).mean())
                CorrelationHetero.append(data['characteristics']['Correlation'][1:].mean().item())
                TotalCorrHetero.append(np.array(data['characteristics']['Information'][0])[1:].mean().item())
                EntropyHetero.append(np.array(data['characteristics']['Information'][1])[1:].mean().item())
                TSEComplexityHetero.append(np.array(data['characteristics']['TSEComplexity'])[1:].mean().item())
                allPositionalInformationHetero = []
                allCellfreqs = np.zeros(numcells)
                for s in range(1,101):
                    numones = np.amax(data['characteristics']['CellularFrequency'][0][0].reshape(1,-1),axis=0,initial=1)
                    numzeros = np.amax((data['simParameters']['numSimIters']-numones).reshape(1,-1),axis=0,initial=1)
                    numones1to0 = data['characteristics']['CellularFrequency'][1][s]
                    numones0to1 = data['characteristics']['CellularFrequency'][2][s]
                    cellfreqs = ((numones0to1/numones)+(numones1to0/numzeros))/2
                    numuniquecellfreqs = len(np.unique(cellfreqs))
                    allPositionalInformationHetero.append(numuniquecellfreqs/numcells)
                    allCellfreqs += cellfreqs
                PositionalInformationHetero.append(np.array(allPositionalInformationHetero).mean().item())
                allCellfreqs /= 100
                CellfreqsHetero.append(allCellfreqs)
                evDimensionHetero.append(evDim[1:,[0,1,2]].sum(1).mean())
                evAggDimensionHetero.append(evAggDim[1:,[0,1,2]].sum(1).mean())
                vmemDimensionHetero.append(vmemDim[1:,[0,1,2]].sum(1).mean())
                evAggVmemDimensionDiffHetero.append((evAggDim[1:,[0,1,2]].sum(1) - vmemDim[1:,[0,1,2]].sum(1)).mean())
                evVmemDimensionDiffHetero.append((evDim[1:,[0,1,2]].sum(1) - vmemDim[1:,[0,1,2]].sum(1)).mean())
                evAggVmemDimensionRatioHetero.append((evAggDim[1:,[0,1,2]].sum(1) / vmemDim[1:,[0,1,2]].sum(1)).mean())
        if sample == 'Segregated':
            df = pd.DataFrame({'GJStrength':GJStrength,'fieldRange':fieldScreenSize,'fieldTransductionWeight':fieldTransductionWeight,'Robustness':Robustness,
                           'CorrelationHomo':CorrelationHomo,'TotalCorrelationHomo':TotalCorrHomo,'EntropyHomo':EntropyHomo,
                           'evDimensionHomo':evDimensionHomo,'evAggDimensionHomo':evAggDimensionHomo,'vmemDimensionHomo':vmemDimensionHomo,
                           'evAggVmemDimensionDiffHomo':evAggVmemDimensionDiffHomo,'evVmemDimensionDiffHomo':evVmemDimensionDiffHomo,
                           'evAggVmemDimensionRatioHomo':evAggVmemDimensionRatioHomo,'TSEComplexityHomo':TSEComplexityHomo,
                           'PositionalInformationHomo':PositionalInformationHomo,'CellfreqsHomo':CellfreqsHomo,
                           'CorrelationHetero': CorrelationHetero, 'TotalCorrelationHetero': TotalCorrHetero,'EntropyHetero': EntropyHetero,
                           'evDimensionHetero': evDimensionHetero, 'evAggDimensionHetero': evAggDimensionHetero,'vmemDimensionHetero': vmemDimensionHetero,
                           'evAggVmemDimensionDiffHetero': evAggVmemDimensionDiffHetero,'evVmemDimensionDiffHetero': evVmemDimensionDiffHetero,
                           'evAggVmemDimensionRatioHetero': evAggVmemDimensionRatioHetero,'TSEComplexityHetero':TSEComplexityHetero,
                           'PositionalInformationHetero':PositionalInformationHetero,'CellfreqsHetero':CellfreqsHetero})
        else:
            df = pd.DataFrame({'GJStrength':GJStrength,'fieldRange':fieldScreenSize,'fieldTransductionWeight':fieldTransductionWeight,
                           'Correlation':Correlation,'TotalCorrelation':TotalCorr,'Entropy':Entropy,
                           'evDimension':evDimension,'evAggDimension':evAggDimension,'vmemDimension':vmemDimension,'eVAggVmemDimemsion':eVAggVmemDimemsion,
                           'evAggVmemDimensionDiff':evAggVmemDimensionDiff,'evVmemDimensionDiff':evVmemDimensionDiff,
                           'evAggVmemDimensionRatio':evAggVmemDimensionRatio,'eVAggVmemDimensionMI':eVAggVmemDimensionMI,
                           'Robustness':Robustness})
        for characteristic in characteristicNames:
            plotCharacteristic(df,characteristic)

if analysisMode == "fixBiasSweepWeightLigandGJ":
    fileRange = range(1,501)
    if 'Hessian' in characteristicNames:
        GJStrength, vmemToLigandCurrentStrength, ligandGatingWeight, Sensitivity, SensitivityDerivative, Hessian = [], [], [], [], [], []
        for fileNumber in fileRange:
            filename = './data/modelCharacteristics_' + Sfx + str(float(fileNumber)) + fileVersionSfx + '.dat'
            data = torch.load(filename)
            GJStrength.append(data['GJParameters']['GJStrength'].round(decimals=2))
            vmemToLigandCurrentStrength.append(data['ligandParameters']['vmemToLigandCurrentStrength'])
            ligandGatingWeight.append(data['ligandParameters']['ligandGatingWeight'].round(decimals=2))
            VmemToVmem, ligandToVmemToVmem = data['characteristics']['Hessian']['Derivatives']
            nzidx = np.array([VmemToVmem[i].any().item() for i in range(VmemToVmem.shape[0])])
            if nzidx.any():
                VmemToVmem = VmemToVmem[nzidx]
                weights = VmemToVmem.clone()
                # weights /= weights.max()
            else:
                weights = VmemToVmem.clone()
            SensitivityTimeSeries = np.array([(VmemToVmem[t]).mean().item() for t in range(VmemToVmem.shape[0])])
            # SensitivityTimeSeries = SensitivityTimeSeries / SensitivityTimeSeries.max()  # normalization
            Sensitivity.append(np.abs(SensitivityTimeSeries.mean()))
            SensitivityDerivative.append(np.abs(SensitivityTimeSeries[1:]-SensitivityTimeSeries[0:-1]).mean())
        df = pd.DataFrame({'GJStrength':GJStrength,'vmemToLigandCurrentStrength':vmemToLigandCurrentStrength,'ligandGatingWeight':ligandGatingWeight,
                           'Sensitivity':Sensitivity,'SensitivityDerivative':SensitivityDerivative})
        for characteristic in ['Sensitivity','SensitivityDerivative']:
            heatmap = df.pivot_table(index='GJStrength',columns='vmemToLigandCurrentStrength',values=characteristic)
            # heatmap_smooth = gaussian_filter(heatmap, sigma=1)
            heatmap_smooth = heatmap
            fig, ax = plt.subplots()
            map = sns.heatmap(heatmap_smooth,cmap='seismic')
            # plt.show()
            plt.savefig('./data/modelCharacteristics_FixedBias_Ligand_' + characteristic + '.png',bbox_inches="tight")

if analysisMode == "fixWeightBiasSweepScreenGJ":
    fileRange = range(1,301)
    GJStrength, fieldScreenSize, TotalCorr, Entropy = [], [], [], []
    evDimension, evAggDimension, vmemDimension ,evVmemDimensionDiff = [], [], [], []
    for fileNumber in fileRange:
        filename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + fileVersionSfx + '.dat'
        data = torch.load(filename)
        GJStrength.append(data['GJParameters']['GJStrength'].round(decimals=2))
        fieldScreenSize.append(data['fieldParameters']['fieldScreenSize'])
        TotalCorr.append(np.array(data['characteristics']['Information'][0]).mean().item())
        Entropy.append(np.array(data['characteristics']['Information'][1]).mean().item())
        evDim, evAggDim, vmemDim = data['characteristics']['Dimensionality']
        evDim, evAggDim, vmemDim = np.array(evDim), np.array(evAggDim), np.array(vmemDim)
        evDimension.append(evDim[:,[0,1,2]].sum(1).mean())
        evAggDimension.append(evAggDim[:,[0,1,2]].sum(1).mean())
        vmemDimension.append(vmemDim[:,[0,1,2]].sum(1).mean())
        evVmemDimensionDiff.append((evAggDim[:,[0,1,2]].sum(1) - vmemDim[:,[0,1,2]].sum(1)).mean())
    # df = pd.DataFrame({'GJStrength':GJStrength,'fieldScreenSize':fieldScreenSize,'TotalCorrelation':TotalCorr,'Entropy':Entropy})
    df = pd.DataFrame({'GJStrength':GJStrength,'fieldScreenSize':fieldScreenSize,
                       'evDimension':evDimension,'evAggDimension':evAggDimension,'vmemDimension':vmemDimension,
                       'evVmemDimensionDiff':evVmemDimensionDiff})
    heatmap = df.pivot_table(index='GJStrength',columns='fieldScreenSize',values='evVmemDimensionDiff')
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
