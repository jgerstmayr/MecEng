#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  Post Processing for FlexuAI results files (.csv)
# Author:   Sebastian Weyrer
# Date:     2026-06-09
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / 'src'))
from ExuAI.utilities import AddSpacesToCamelCase
from collections import defaultdict
from matplotlib.patches import Patch
from generatePaperPlotsAndTables import WriteGroupedSuccessRateTable, FormatPercent, SpecialFormatRate # sits here since if-clause needed in generatePaperPlotsAndTables
from generateModelComparisonTable import FormatRate

#++++++++++++++++++++++++++++++++++++++++++++++++++
# read input dataPlot
# reads the merged results of the flexible MBS runs (Tables 4, 5 and Figure 11), i.e., the folder
# logsFlexuAIPipeFMBS_v4 extracted from the zipped results into <repo>/results
scriptDir = Path(__file__).resolve().parent
resultsDir = scriptDir.parents[1] / 'results' / 'logsFlexuAIPipeFMBS_v4'
csvPathLLM = resultsDir / 'overallResults.csv'
csvPathFMBS = resultsDir / 'overallResultsSuccessRates.csv'
dfLLM = pd.read_csv(csvPathLLM, delimiter=';', header=2, decimal=',')
dfLLM = dfLLM.set_index('Case Name') # get names of LLMs and shorten them
dfLLM['flexuAIFMBSStartingRate'] = dfLLM['flexuAIFMBSGenerated'] / dfLLM['nTasks']
dfLLM.index = dfLLM.index.str.split('-OL').str[0]
dfLLM['eval time'] = pd.to_timedelta(dfLLM['eval time']).dt.total_seconds() / 60
namesOfLLMs = dfLLM['LLM full name'].to_list()
dfFMBS = pd.read_csv(csvPathFMBS, delimiter=';', header=[3, 4], decimal=',', nrows=len(dfLLM.index.to_list()))
outputDir = scriptDir.parent / 'paperOutput'
outputDir.mkdir(exist_ok=True)
proprietaryModelNames = ['Claude-Opus-4.8', 'GPT-5.2', 'Claude-Opus-4.8-Thk1', 'GPT-5.2-Thk1'] # add if available

#%% LLM plot
# categories that we want to plot
categoriesToPlot = ['choose items parts', 'relative difference volume',
                    'relative eigenfrequencies norm difference',
                    'overall success parts', 'flexuAIFMBSStartingRate', 'choose items', 'solver success',
                    'topology match', 'solution match', 'graph content match',
                    'overall success']
shortNamesInPlot = ['part comp.', 'mesh', 'eigenfreq.',
                    'parts overall', 'MBS possible', 'comp.', 'solver', 'topol.',
                    'numsol', 'graph', 'overall']
categoriesForInfoTable = ['LLM dur. (min)', 'flexuAIPartsEvalTime (min)', 'flexuAIFMBSEvalTime (min)', 'nTokGen (k)', 'ctx size used (P95)', 'batch']
shortNamesInInfoTable =['LLM dur.', 'parts eval. dur.', 'MBS eval. dur.', 'gen. tokens (k)', 'ctx size used (P95)', 'batch size']
excludeCategoriesForTable = [2, 6, 7, 8]
categoriesForTable = []
shortNamesInTable = []
for i, category in enumerate(categoriesToPlot):
    if i not in excludeCategoriesForTable:
        categoriesForTable += [category]
        shortNamesInTable += [shortNamesInPlot[i]]
    else:
        pass
if len(categoriesToPlot) != len(shortNamesInPlot):
    print('WARNING: Specify same amount of short names as categories!')
dataPlot = dfLLM[categoriesToPlot].apply(pd.to_numeric, errors='coerce').to_numpy()
dataTable = dfLLM[categoriesForTable].apply(pd.to_numeric, errors='coerce').to_numpy()
dataInfoTable = dfLLM[categoriesForInfoTable].apply(pd.to_numeric, errors='coerce').to_numpy()

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot
figBarChart, axBarChart = plt.subplots(1, 1, figsize=[7, 4])
x = np.arange(len(categoriesToPlot))
width = 0.15
multiplier = 0
for i, measurement in enumerate(dataPlot):
    offset = width * multiplier
    rects = axBarChart.bar(x+offset, measurement*100, width, label=namesOfLLMs[i])
    # axBarChart.bar_label(rects, padding=3)
    multiplier += 1
axBarChart.set_xticks(x + width * (len(dataPlot) - 1) / 2)
axBarChart.set_xticklabels(shortNamesInPlot, rotation=45, ha='right')
axBarChart.set_axisbelow(True)
axBarChart.grid()
axBarChart.legend()
split = 3.5 + width * (len(dataPlot)-1) / 2
axBarChart.axvline(x=split, color='black', linewidth=1, linestyle='--')
axBarChart.text(split - 0.3, 0.95, 'flexible part', transform=axBarChart.get_xaxis_transform(),
                ha='right', va='top', fontsize=11)
axBarChart.text(split + 0.3, 0.95, 'flexible MBS', transform=axBarChart.get_xaxis_transform(),
                ha='left', va='top', fontsize=11)
axBarChart.set_ylabel('success rate in %')
plt.tight_layout()
figBarChart.savefig(str(outputDir) + '/flexuAIEvaluationResults.pdf')
figBarChart.savefig(str(outputDir) + '/flexuAIEvaluationResults.png', dpi=600)

#%% fMBS difficulty plot
fMBSFMBS = dfFMBS['fMBS'].apply(pd.to_numeric, errors='coerce')
orderedResultsListFMBS = []
for fMBSName in fMBSFMBS:
    fMBSMean = np.mean(fMBSFMBS[fMBSName].to_numpy())
    orderedResultsListFMBS += [(fMBSName, fMBSMean)]
orderedResultsListFMBS = sorted(orderedResultsListFMBS, key=lambda x: x[1], reverse=True)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# define short names for the fMBS
fMBSNames = ['cantilever beam',
             'flexible pendulum',
             'wind turbine',
             'grinding bench',
             'rotor in bearing block',
             'flexible table with load',
             'rotor in bearing block with pedestal',
             'rotor in two bearing blocks with pedestal',
             'door hinge',
             'mounting bracket free falling',
             'flexible slider crank simple',
             'flexible slider crank advanced']
shortFMBSNames = ['cantilever',
                  'flex. pendulum',
                  'wind turbine',
                  'grinding bench',
                  'rotor bearing simpl.', 'table', 'rotor bearing adv.',
                  'rotor 2 bearings', 'door hinge', 'mounting bracket', 'slider crank simpl.',
                  'slider crank adv.']
if len(fMBSNames) != len(shortFMBSNames):
    print('WARNING: Specify same amount of short names as names!')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot
names = [shortFMBSNames[fMBSNames.index(AddSpacesToCamelCase(r[0]))] for r in orderedResultsListFMBS]
values = [1-r[1] for r in orderedResultsListFMBS]
figFMBS, axFMBS = plt.subplots(1, 1, figsize=[7, 4])
colors = ['red' if v == 1.0 else 'steelblue' for v in values]
axFMBS.bar(names, values, width=0.5, color=colors)
axFMBS.set_xticks(np.arange(len(names)))
axFMBS.set_xticklabels(names, rotation=45, ha='right')
axFMBS.set_ylabel('relative difficulty of flexible MBS')
axFMBS.set_axisbelow(True)
legendElements = [Patch(facecolor='steelblue', label='diff. < 1'),
                   Patch(facecolor='red', label='diff. = 1')]
axFMBS.legend(handles=legendElements)
axFMBS.grid()
plt.tight_layout()
figFMBS.savefig(str(outputDir) + '/flexuAIFMBSDifficulty.pdf')
figFMBS.savefig(str(outputDir) + '/flexuAIFMBSDifficulty.png', dpi=600)

#%% part difficulty plot
# Get all column positions belonging to Parts groups (including unnamed)
topHeaders = dfFMBS.columns.get_level_values(0)
partMask = []
current = False
for h in topHeaders:
    if h == 'Parts':
        current = True
    elif h == 'fMBS':
        current = False
    partMask.append(current)
fMBSPart = dfFMBS.iloc[:, partMask].apply(pd.to_numeric, errors='coerce')
fMBSPart.columns = fMBSPart.columns.get_level_values(1)
def GetOrderedPartSuccess(dfSubset):
    orderedList = []
    for partName in dfSubset:
        partMean = np.mean(dfSubset[partName].to_numpy())
        orderedList += [(partName.split(' of ')[0], partMean)]
    orderedList = sorted(orderedList, key=lambda x: x[1], reverse=True)
    # check for duplicates (can happen since various fMBS use the same parts)
    grouped = defaultdict(list)
    for name, value in orderedList:
        grouped[name].append(value)
    orderedList = sorted([(name, np.mean(vals)) for name, vals in grouped.items()],
                         key=lambda x: x[1], reverse=True)
    return orderedList
modelNames = dfFMBS.iloc[:, 0]
isProprietary = modelNames.isin(proprietaryModelNames)
orderedResultsListPartAPI = GetOrderedPartSuccess(fMBSPart[isProprietary.values])
orderedResultsListPartOpenWeight = GetOrderedPartSuccess(fMBSPart[~isProprietary.values])

#++++++++++++++++++++++++++++++++++++++++++++++++++
# define short names for the parts
partNames = ['flexible beam drilled part',
             'flexible beam part',
             'flexible beam fillets part',
             'wind turbine rotor part',
             'slider part simple',
             'unbalanced rotor part',
             'grinding bench housing part',
             'bearing block part',
             'grinding bench spindle with disks part',
             'rod part advanced',
             'wind turbine tower part',
             'flexible beam hollow part',
             'flexible table part',
             'hinge leaf gap side part',
             'hinge leaf pin side part',
             'crank part simple',
             'balanced rotor part',
             'bearing block with pedestal part',
             'mounting bracket part',
             'rod part simple',
             'crank part advanced',
             'slider part advanced']
shortPartNames = ['beam drilled (57)',
                  'beam (54)',
                  'beam fillets (55)',
                  'turbine rotor (60)',
                  'slider simpl. (69)',
                  'unbal. rotor (52)',
                  'grind. housing (61)',
                  'bearing block (51)',
                  'grind. spindle (62)',
                  'rod adv. (71)',
                  'turbine tower (59)',
                  'beam hollow (56)',
                  'flex. table (58)',
                  'hinge gap (63)',
                  'hinge pin (64)',
                  'crank simpl. (67)',
                  'bal. rotor (53)',
                  'bearing w. ped. (66)',
                  'bracket (65)',
                  'rod simpl. (68)',
                  'crank adv. (70)',
                  'slider adv. (72)']
if len(partNames) != len(shortPartNames):
    print('WARNING: Specify same amount of short names as names!')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot
openWeightDict = dict(orderedResultsListPartOpenWeight)
proprietaryDict = dict(orderedResultsListPartAPI)
# order is determined by open-weight results (already sorted descending)
names = [shortPartNames[partNames.index(AddSpacesToCamelCase(r[0]))] for r in orderedResultsListPartOpenWeight]
print('Plotting', len(names), 'part(s).')
partKeysOrdered = [r[0] for r in orderedResultsListPartOpenWeight]
valuesOpenWeight = [1 - openWeightDict[k] for k in partKeysOrdered]
valuesProprietary = [1 - proprietaryDict.get(k, np.nan) for k in partKeysOrdered]
edgesOpenWeight = ['black' if v == 1.0 else 'none' for v in valuesOpenWeight]
edgesProprietary = ['black' if v == 1.0 else 'none' for v in valuesProprietary]
x = np.arange(len(names))
width = 0.35
figPart, axPart = plt.subplots(1, 1, figsize=[9, 4.5])
axPart.bar(x - width/2, valuesOpenWeight, width=width, color='steelblue',
           edgecolor=edgesOpenWeight, linewidth=2, label='open-weight')
axPart.bar(x + width/2, valuesProprietary, width=width, color='darkorange',
           edgecolor=edgesProprietary, linewidth=2, label='proprietary')
axPart.set_xticks(x)
axPart.set_xticklabels(names, rotation=45, ha='right')
axPart.set_ylabel('relative difficulty of flexible part')
axPart.set_axisbelow(True)
legendElementsCombined = [
    Patch(facecolor='steelblue', label='open-weight'),
    Patch(facecolor='darkorange', label='proprietary'),
    # Patch(facecolor='none', edgecolor='black', linewidth=2, label='unsolved'),
]
axPart.legend(handles=legendElementsCombined)
axPart.grid()
plt.tight_layout()
figPart.savefig(str(outputDir) + '/flexuAIPartDifficulty.pdf')
figPart.savefig(str(outputDir) + '/flexuAIPartDifficulty.png', dpi=600)

#%% write overall results LaTeX table
outputDir.mkdir(parents=True, exist_ok=True)
tableOutputPath = outputDir / 'tableOverallResultsFlexuAI.tex'
allRows = list(zip(namesOfLLMs, dataTable.tolist()))
proprietaryRows = [r for r in allRows if r[0] in proprietaryModelNames]
openWeightRows = [r for r in allRows if r[0] not in proprietaryModelNames]
rowGroups = []
def OverallSuccess(row):
    overallColumnIndex = shortNamesInTable.index('overall')  # adjust string to match your actual header label
    _, values = row
    return values[overallColumnIndex]
if proprietaryRows:
    proprietaryRows.sort(key=OverallSuccess, reverse=True)
    rowGroups.append({'rows': proprietaryRows, 'meanLabel': 'mean (proprietary)'})
if openWeightRows:
    openWeightRows.sort(key=OverallSuccess, reverse=True)
    rowGroups.append({'rows': openWeightRows, 'meanLabel': f'mean (open-weight, n={len(openWeightRows)})'})
WriteGroupedSuccessRateTable(rowGroups=rowGroups,
                             columnHeaders=shortNamesInTable,
                             outputPath=tableOutputPath,
                             formatter=SpecialFormatRate)

#%% special information table about LLM statistics
outputDir.mkdir(parents=True, exist_ok=True)
tableOutputPath = outputDir / 'infoTableFlexuAI.tex'
allRows = list(zip(namesOfLLMs, dataInfoTable.tolist()))
proprietaryRows = [r for r in allRows if r[0] in proprietaryModelNames]
openWeightRows = [r for r in allRows if r[0] not in proprietaryModelNames]
rowGroups = []
# overall success (not part of this table's columns) is only used for sorting
overallSuccessVals = dfLLM['overall success'].apply(pd.to_numeric, errors='coerce').to_numpy()
overallSuccessDict = dict(zip(namesOfLLMs, overallSuccessVals))
def OverallSuccessSortKey(row):
    name, _ = row
    return overallSuccessDict[name]
if proprietaryRows:
    proprietaryRows.sort(key=OverallSuccessSortKey, reverse=True)
    rowGroups.append({'rows': proprietaryRows, 'meanLabel': 'mean (proprietary)'})
if openWeightRows:
    openWeightRows.sort(key=OverallSuccessSortKey, reverse=True)
    rowGroups.append({'rows': openWeightRows, 'meanLabel': f'mean (open-weight, n={len(openWeightRows)})'})
def SpecialFormatter(value, name):
    if name == 'parts overall':
        return FormatPercent(value)
    elif name == 'ctx size used (P95)':
        if value == 0:
            return '-'
        else:
            return str(int(value))
    elif name == 'batch size':
        if value == 1:
            return '-'
        else:
            return str(int(value))
    elif name == 'gen. tokens (k)':
        return str(int(value))
    else:
        return str(round(value, 1))
WriteGroupedSuccessRateTable(rowGroups=rowGroups,
                             columnHeaders=shortNamesInInfoTable,
                             outputPath=tableOutputPath,
                             formatter=SpecialFormatter,
                             spanningHeader='LLM statistics')

#%% success rate drop plot: choose items parts vs. downstream metrics
baseCategory = 'choose items parts'
materialCategory = 'material match'
interfaceCategory = 'interface name match'
massCategory = 'relative mass difference'
eigenfreqCategory = 'relative eigenfrequencies norm difference'
volumeCategory = 'relative difference volume'
baseVals = pd.to_numeric(dfLLM[baseCategory], errors='coerce').to_numpy()
materialVals = pd.to_numeric(dfLLM[materialCategory], errors='coerce').to_numpy()
interfaceVals = pd.to_numeric(dfLLM[interfaceCategory], errors='coerce').to_numpy()
massVals = pd.to_numeric(dfLLM[massCategory], errors='coerce').to_numpy()
eigenfreqVals = pd.to_numeric(dfLLM[eigenfreqCategory], errors='coerce').to_numpy()
volumeVals = pd.to_numeric(dfLLM[volumeCategory], errors='coerce').to_numpy()
dropMaterial = (baseVals - materialVals) * 100          # material match vs. choose items parts
dropInterface = (baseVals - interfaceVals) * 100        # interface match vs. choose items parts
dropMass = (baseVals - massVals) * 100                  # mass diff. vs. choose items parts
dropEigenfreq = (baseVals - eigenfreqVals) * 100        # eigenfreq. diff. vs. choose items parts
dropVolume = (baseVals - volumeVals) * 100              # volume diff. vs. choose items parts
dropData = np.array([dropMaterial, dropInterface, dropMass, dropEigenfreq, dropVolume])  # shape: (5, nModels)
dropShortNames = ['material match', 'interface match', 'mass diff.', 'eigenfreq. diff.', 'volume diff.']
#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot
figDrop, axDrop = plt.subplots(1, 1, figsize=[8, 4.5])
x = np.arange(len(namesOfLLMs))
width = 0.16
multiplier = 0
for i, dropVals in enumerate(dropData):
    offset = width * multiplier
    axDrop.bar(x + offset, dropVals, width, label=dropShortNames[i])
    multiplier += 1
axDrop.set_xticks(x + width * (len(dropData) - 1) / 2)
axDrop.set_xticklabels(namesOfLLMs, rotation=45, ha='right')
axDrop.set_axisbelow(True)
axDrop.grid()
axDrop.legend(loc='upper right')
axDrop.axhline(y=0, color='black', linewidth=0.8)
axDrop.set_ylabel('success rate drop vs. choose components rate in %')
plt.tight_layout()
figDrop.savefig(str(outputDir) + '/flexuAISuccessRateDrops.pdf')
figDrop.savefig(str(outputDir) + '/flexuAISuccessRateDrops.png', dpi=600)
