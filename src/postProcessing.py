#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: postProcessing.py
#
# Details:  This script handles post-processing of LLM test results.
#           It merges results from multiple test runs and generates reports.
#
# Authors:  Tobias Möltner, Johannes Gerstmayr
# Date:     2026-04-08
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import csv
import json
import datetime
import traceback

from ExuAI.utilities import ReadJSONToDict, WriteDictToJSON, JoinPath, NumToCSV, IsScalar
from ExuAI.myLLM import ConvertFullName2modelKeyName, llmModelsDict
from ExuAI.modelDescriptions import simpleAndMediumMBSDescriptions, geometryDescriptions
from ExuAI.logger import Logger

# layout of the two side-by-side paper plots (temperature correlation and repeated runs):
# figure width ~3 in is shown at ~2.5 in in the paper, giving ~9-10 pt text;
# identical figure size and axes box so both subfigures are aligned
pairedPlotLayout = {'figsize': (3.2, 3.6), 'left': 0.20, 'right': 0.97, 'top': 0.97, 'bottom': 0.40,
                    'labelFontSize': 12, 'tickFontSize': 11, 'legendFontSize': 8.5, 'annotationFontSize': 7}

def MergeTestModelResults(dirTestModels, outputJSONpath, logger:Logger, folderPrefix="log_"):
    """
    merge results into .json file and into .csv file

    Args:
        dirTestModels (TYPE): DESCRIPTION.
        outputFileName (TYPE, optional): DESCRIPTION. Defaults to "allLLMresults.json".

    Returns:
        all LLM results as dict, same as saved to json

    """
    
    import numpy as np
    
    log_dirs = sorted([d for d in os.listdir(dirTestModels) if d.startswith(folderPrefix) and os.path.isdir(JoinPath(dirTestModels, d))], key=str.lower)
    
    # Read 'results.json' from each valid subdirectory
    allLLMresults = {} #written to json file
    resultsData = {} #contains dict of all results
    miscData = {'csvKeys':[]} #store keys to be added into csv file

    allLLMresults['resultsData'] = resultsData
    allLLMresults['misc'] = miscData
    
    for log_dir in log_dirs:
        if log_dir.startswith('_') or log_dir in ('logsTrash', 'results'):
            continue
        shortName = log_dir[len(folderPrefix):]
        json_path = JoinPath(dirTestModels, log_dir, "results.json")
        
        if os.path.exists(json_path):  # Check if 'results.json' exists
            resultsData[shortName] = ReadJSONToDict(json_path)  # Read JSON data into dict
        else:
            logger.LogDebug(f"Warning: 'results.json' not found in {log_dir}", printToConsole=True)
            #just continue
            
    
    if resultsData == {}:
        errMsg = "No valid data found -> check directory / results.json must be available in log_* dirs"
        logger.LogDebug(errMsg)
        return allLLMresults
        
    else:
        #%%++++++++++++++++++++++++++++
        #extract data from each results json and put into common data file
        #overall data            
        miscData['merge date'] = datetime.date.today().strftime("%Y-%m-%d")
        miscData['merge time']= datetime.datetime.now().strftime("%H:%M:%S")
        nRound = 4

        

        for key, value in resultsData.items(): #key=case name (QwenCoder2.5-14B-OL-B16-T0.0)
            try:
                #extract case data
                llmModelName = value['llmModelName']
                llmData = llmModelsDict[llmModelName]
                llmModelShortName = llmData['shortName']

                llmDataDate = value.get('executionDateStr','N/A')
                llmDataTime = value.get('executionTimeStr','N/A')
                batchSize = value.get('llmBatchSize','N/A')
                GPUused = value.get('llmGPUused','N/A')
                
                nTotalMainTasks = value.get('nTotalMainTasks', 1) #for overall calculations
                llmConfig = value.get('llmConfig',{})
                llmDurationGlobal = llmConfig.get('durationGlobal',0)

                agentConfig = value.get('agentConfig',{})
                llmGenerateDefaultArgs = agentConfig.get('llmGenerateDefaultArgs',{})
                temperature = llmGenerateDefaultArgs.get('temperature', -1)
                if temperature is None: temperature = -1

                #++++++++++++++++++++++++++++++++++++++++++++++
                #write results data per case (e.g. QwenCoder2.5-14B-OL-B16-T0.0))
                caseData = {}
                resultsData[key] = caseData
                
                caseData['eval date'] = llmDataDate
                caseData['eval time'] = llmDataTime
                caseData['nTasks'] = NumToCSV(nTotalMainTasks)
                caseData['LLM dur. (min)'] = NumToCSV(llmDurationGlobal/60,2)
                nTokens = llmConfig.get('numberOfTokensGlobal', 0)
                nThkTokens = llmConfig.get('numberOfThinkingTokensGlobal', 0)
                caseData['nTokGen (k)'] = int(nTokens/1024)
                caseData['nThkTok (k)'] = int(nThkTokens/1024)
                tokensPerSecond = llmConfig.get('tokensPerSecondGlobal')
                if tokensPerSecond:
                    caseData['tokens / s'] = NumToCSV(tokensPerSecond, 2)
                else:
                    runTimeForTps = value.get('runTime', 0) or llmDurationGlobal
                    caseData['tokens / s'] = NumToCSV(
                        (nTokens + nThkTokens) / max(0.01, runTimeForTps), 2)
                caseData['maxCtx (k)'] = NumToCSV(llmConfig.get('contextWindowMaxSize','N/A'))

                if 'rateExecutabilityMainTasks' in value:
                    isFlexuAI = value.get('agentConfig').get('isFlexuAI')
                    flexuAItaskType = value.get('agentConfig').get('flexuAITaskType')
                    #flexuAI fMBS
                    if isFlexuAI and flexuAItaskType == 'fMBS':
                        caseData['choose items parts'] = NumToCSV(value.get('rateChooseItemsMatchPreliminaryTasks', 0),nRound)
                        caseData['material match'] = NumToCSV(value.get('rateMaterialMatchPreliminaryTasks', 0),nRound)
                        caseData['relative mass difference'] = NumToCSV(value.get('rateRelativeMassDifferencePreliminaryTasks', 0),nRound)
                        caseData['relative eigenfrequencies norm difference'] = NumToCSV(value.get('rateRelativeEigenfrequenciesNormDifferencePreliminaryTasks', 0),nRound)
                        caseData['relative difference volume'] = NumToCSV(value.get('rateRelativeDifferenceVolumePreliminaryTasks', 0),nRound)
                        caseData['interface name match'] = NumToCSV(value.get('rateInterfaceNameMatchPreliminaryTasks', 0),nRound)
                        caseData['overall success parts'] = NumToCSV(value.get('rateSuccessPreliminaryTasks', 0),nRound)
                    #flexuAI part
                    if isFlexuAI and flexuAItaskType == 'part':
                        caseData['choose items parts'] = NumToCSV(value.get('rateChooseItemsMatchMainTasks', 0),nRound)
                        caseData['material match'] = NumToCSV(value.get('rateMaterialMatchMainTasks', 0),nRound)
                        caseData['relative mass difference'] = NumToCSV(value.get('rateRelativeMassDifferenceMainTasks', 0),nRound)
                        caseData['relative eigenfrequencies norm difference'] = NumToCSV(value.get('rateRelativeEigenfrequenciesNormDifferenceMainTasks', 0),nRound)
                        caseData['relative difference volume'] = NumToCSV(value.get('rateRelativeDifferenceVolumeMainTasks', 0),nRound)
                        caseData['interface name match'] = NumToCSV(value.get('rateInterfaceNameMatchMainTasks', 0),nRound)
                        caseData['overall success'] = NumToCSV(value.get('rateSuccessMainTasks', 0),nRound)
                    else:
                        caseData['choose items'] = NumToCSV(value.get('rateChooseItemsMatchMainTasks', 0),nRound)
                        caseData['reached assemble'] = NumToCSV(value.get('rateReachedMbsAssembleMainTasks', 0),nRound)
                        caseData['finished assemble'] = NumToCSV(value.get('rateFinishedMbsAssembleMainTasks', 0),nRound)
                        caseData['solver init'] = NumToCSV(value.get('rateSolverInitSuccessMainTasks', 0),nRound)
                        caseData['solver success'] = NumToCSV(value.get('rateSolverSuccessMainTasks', 0),nRound)
                        caseData['topology match'] = NumToCSV(value.get('rateGraphMatchMainTasks', 0),nRound)
                        caseData['solution match'] = NumToCSV(value.get('rateSolutionFileMatchMainTasks', 0),nRound)
                        caseData['graph content match'] = NumToCSV(value.get('rateGraphMatchContentMainTasks', 0),nRound)
                        caseData['overall success'] = NumToCSV(value.get('rateSuccessMainTasks', 0),nRound)
                        caseData['kin./marker nERR'] = NumToCSV(value.get('errorCategoryCounter',{}).get("Kinematics / Marker parameters", 0))
                        caseData['RB ref.conf. nERR'] = NumToCSV(value.get('errorCategoryCounter',{}).get("Rigid Body Reference Config.", 0))
                        caseData['vel. nERR'] = NumToCSV(value.get('errorCategoryCounter',{}).get("Initial Velocities", 0))
                        caseData['inertia/mass nERRs'] = NumToCSV(value.get('errorCategoryCounter',{}).get("Inertia and Mass Parameters", 0))
                        caseData['gravity/load nERR'] = NumToCSV(value.get('errorCategoryCounter',{}).get("Gravity and Load Parameters", 0))
                        caseData['connect./constr. nERR'] = NumToCSV(value.get('errorCategoryCounter',{}).get("Connector / Constraint parameters", 0))
                        caseData['other nERR'] = NumToCSV(value.get('errorCategoryCounter',{}).get("Other", 0))

                        caseData['graphs timeout'] = NumToCSV(value.get('rateGraphsIsomorphicNoTimeoutMainTasks', 0),nRound)
                        caseData['solver Sample timeout'] = NumToCSV(value.get('rateSolverNoTimeoutSampleMainTasks', 0),nRound)
                        caseData['solver LLM timeout'] = NumToCSV(value.get('rateSolverNoTimeoutLLMMainTasks', 0),nRound)
                        caseData['ctx window exceeded'] = NumToCSV(1 - value.get('rateLlmNoContextWindowExceededMainTasks', 1),nRound)
                        caseData['ctx exceeded tok. (max)'] = NumToCSV(value.get('contextWindowExceededTokensMax', 0))

                caseData['temperature'] = NumToCSV(temperature,2)
                caseData['default temperature'] = NumToCSV(llmData.get('defaultTemperature', -1),2)
                caseData['thinking'] = str(llmConfig.get('enable_thinking', 'N/A'))
                caseData['batch'] = NumToCSV(batchSize)
                caseData['LLM date'] = llmData.get('releaseDate', 'N/A')
                caseData['LLM ctxSize (k)'] = NumToCSV(llmData.get('ctxSize','N/A'))
                
                contextSizeUsedList = value.get('contextSizeUsedList', [])
                promptTokensList = value.get('numberOfPromptTokensList', [])
                caseData['ctx size used (P95)'] = NumToCSV(np.percentile(contextSizeUsedList, 95) if contextSizeUsedList else 0,1)
                caseData['ctx size used (max)'] = NumToCSV(max(contextSizeUsedList) if contextSizeUsedList else 0)
                caseData['prompt tokens (mean)'] = NumToCSV(sum(promptTokensList)/len(promptTokensList) if promptTokensList else 0,1)
                caseData['api cost (EUR)'] = NumToCSV(llmConfig.get('apiCostGlobalEUR', 0),4)
                caseData['api budget rem. (EUR)'] = NumToCSV(llmConfig.get('apiBudgetRemainingEUR', 0),4)
                
                if isFlexuAI:
                    agentConfig = value.get('agentConfig', {})
                    caseData['isFlexuAI'] = isFlexuAI
                    caseData['flexuAITaskType'] = flexuAItaskType
                    caseData['flexuAIFMBSEvalTime (min)'] = NumToCSV(agentConfig.get('flexuAIFMBSEvalTime', 0)/60, 2)
                    caseData['flexuAIPartsEvalTime (min)'] = NumToCSV(agentConfig.get('flexuAIPartsEvalTime', 0)/60, 2)
                    caseData['flexuAIFMBSGenerated'] = agentConfig.get('flexuAIFMBSGenerated', 0)
                    caseData['flexuAIPartsGenerated'] = agentConfig.get('flexuAIPartsGenerated', 0)
                

                caseData['LLM full name'] = llmModelName
                caseData['LLM short name'] = llmModelShortName
                caseData['VRAM (GB)'] = NumToCSV(llmData['VRAM'],1)
                caseData['GPU'] = GPUused
                caseData['pipeline'] = llmData['type']

                #++++++++++++++++++++++++++++++++++++++++++++++
                #end of csv-data:
                miscData['csvKeys'] = list(caseData) #now contains keys

                #++++++++++++++++++++++++++++++++++++++++++++++
                #not in csv:
                caseData['runtime (min)'] = NumToCSV(value.get('runTime', 1)/60,2)
                caseData['quant.'] = NumToCSV(value.get('llmModelQuantization', 'N/A'))
                caseData['#params (B)'] = NumToCSV(llmData['nParam'])
                caseData['nTokRem (k)'] = int(value.get('numberOfRemovedTokensGlobal', 0)/1024)
                caseData['modelSuccessRates'] = value.get('modelSuccessRate',{})
                caseData['LLM paper name'] = ConvertFullName2modelKeyName(llmModelName)
                    
            except KeyboardInterrupt:
                print("Keyboard interrupt requested...exiting")
                raise
            except:
                exceptionText = traceback.format_exc()
                logger.LogDebug('ERROR in Merge: could not process data from llm test '+key+'\ntraceback='+exceptionText,
                                printToConsole=True)

        #write json file
        WriteDictToJSON(allLLMresults, outputJSONpath)
        return allLLMresults


def ExportMergedResultsToCSV(mergedJsonFilePath, outputCSVpath, logger:Logger):
    """
    Export merged results from JSON to CSV format using the csvKeys from misc data.
    
    Reads the merged JSON file created by MergeTestModelResults and extracts 
    the keys specified in misc['csvKeys'], writing them to a CSV file.
    
    Parameters:
    -----------
    mergedJsonFilePath : str
        Path to the merged JSON file (from MergeTestModelResults)
    outputCSVPath : str
        Path for output CSV file.
    
    Returns:
    --------
    None
    """
    if not os.path.exists(mergedJsonFilePath):
        print(f"ERROR: ExportMergedResultsToCSV: JSON file not found: {mergedJsonFilePath}")
        return
    
    # Load the merged results
    allLLMresults = ReadJSONToDict(mergedJsonFilePath)
    
    if 'resultsData' not in allLLMresults or 'misc' not in allLLMresults:
        logger.LogDebug("ERROR: ExportMergedResultsToCSV: Invalid merged JSON structure. Expected 'resultsData' and 'misc' keys.",
                        printToConsole=True)
        return
    
    resultsData = allLLMresults['resultsData']
    miscData = allLLMresults['misc']
    csvKeys = miscData.get('csvKeys', [])
    mergeDate = miscData.get('merge date', 'N/A')
    mergeTime = miscData.get('merge time', 'N/A')
    
    if not csvKeys:
        logger.LogDebug("WARNING: ExportMergedResultsToCSV: No csvKeys found in misc data",printToConsole=True)
        return
    
    if not resultsData:
        logger.LogDebug("WARNING: ExportMergedResultsToCSV: No results data found",printToConsole=True)
        return
        
    # Prepare data for CSV
    rows = []
    
    # Add header information: merge date and time
    rows.append(['Merge Date', mergeDate])
    rows.append(['Merge Time', mergeTime])
    rows.append([])  # Empty row for spacing
    
    # Header row: Add 'Case Name' as first column, then csvKeys
    header = ['Case Name'] + csvKeys
    rows.append(header)
    
    # Data rows
    for caseName, caseData in resultsData.items():
        row = [caseName]
        for key in csvKeys:
            value = caseData.get(key, 'N/A')
            # Convert complex types to string for CSV
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            row.append(str(value)) #note that numbers are already strings with "," as comma
        rows.append(row)
    
    # Write to CSV
    with open(outputCSVpath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(rows)
    
    logger.LogDebug(f'CSV report saved to: {outputCSVpath}',printToConsole=True)



def ExportModelSuccessRatesToCSV(mergedJsonFilePath, outputCSVpath, logger:Logger):

    if not os.path.exists(mergedJsonFilePath):
        print(f"ERROR: ExportModelSuccessRatesToCSV: JSON file not found: {mergedJsonFilePath}")
        return
    
    # Load the merged results
    allLLMresults = ReadJSONToDict(mergedJsonFilePath)
    
    if 'resultsData' not in allLLMresults or 'misc' not in allLLMresults:
        logger.LogDebug("ERROR: ExportModelSuccessRatesToCSV: Invalid merged JSON structure. Expected 'resultsData' and 'misc' keys.",
                        printToConsole=True)
        return
    
    resultsData = allLLMresults['resultsData']

    if not resultsData:
        logger.LogDebug("WARNING: No results data found", printToConsole=True)
        return

    miscData = allLLMresults['misc']
    mergeDate = miscData.get('merge date', 'N/A')
    mergeTime = miscData.get('merge time', 'N/A')
    # Model columns are the models actually present in the merged results (union of
    # modelSuccessRates keys across all cases), NOT the full simpleAndMediumMBSDescriptions
    # registry: only models that were actually evaluated appear (no never-run N/A columns).
    # Order: registry order first, then models that are not in the registry.
    presentModels = []
    presentModelsSet = set()
    for _caseData in resultsData.values():
        for _modelName in (_caseData.get('modelSuccessRates') or {}).keys():
            if _modelName not in presentModelsSet:
                presentModelsSet.add(_modelName)
                presentModels.append(_modelName)
    registryOrder = list(simpleAndMediumMBSDescriptions)
    allModelNamesList = [m for m in registryOrder if m in presentModelsSet] \
                        + [m for m in presentModels if m not in registryOrder]

    if len(allModelNamesList) == 0:
        logger.LogDebug("WARNING: ExportModelSuccessRatesToCSV: No model names",
                        printToConsole=True)
        return

    # Prepare data for CSV
    rows = []
    
    # Add header information: merge date and time
    rows.append(['Merge Date', mergeDate])
    rows.append(['Merge Time', mergeTime])

    # check if we are in flexuAI pipeline
    isFlexuAI = False
    caseName, caseData = next(iter(resultsData.items())) # first occurance decides whether we have flexuAI
    if 'isFlexuAI' in caseData:
        isFlexuAI = True
        flexuAITaskType = caseData['flexuAITaskType']
    
    # Header row: Add 'Case Name' as first column and eval date, as this may be important
    header = ['Case Name', 'eval date'] + allModelNamesList
    # full 'mbs' metric set
    successTypes = ['overallSuccess', 'chooseItemsMatch', 'executableLLM',
                    'reachedMbsAssemble', 'finishedMbsAssemble', 'solverInitSuccessful',
                    'solverSuccess', 'solutionFileDiff', 'graphNodesCountMatch',
                    'graphMatch', 'graphMatchContent', 'graphsIsomorphicNoTimeout',
                    'solverNoTimeoutSample', 'solverNoTimeoutLLM', 'llmNoContextWindowExceeded']
    if isFlexuAI:
        header = ['Case Name', 'eval date'] # initialize header
        partSuccessTypes = ['overallSuccess', 'chooseItemsMatch', 'materialMatch',
                            'relativeMassDifference', 'relativeEigenfrequenciesNormDifference',
                            'relativeDifferenceVolume', 'interfaceNameMatch']
        if flexuAITaskType == 'fMBS':
            # find out which fMBS have which preliminary results
            caseName, caseData = next(iter(resultsData.items())) # first run determines the headers
            row = [caseName] + [caseData['eval date']]
            successTypes = list(dict.fromkeys(successTypes + partSuccessTypes))
            # metrics that only apply to parts (not to fMBS assembly)
            partOnlyMetrics = {'materialMatch', 'relativeMassDifference',
                               'relativeEigenfrequenciesNormDifference',
                               'relativeDifferenceVolume', 'interfaceNameMatch'}
            # metrics that only apply to fMBS assembly (not to individual parts)
            fMBSOnlyMetrics = {'solutionFileDiff', 'graphMatch', 'graphMatchContent', 'reachedMbsAssemble'}
            # shared metrics (e.g. overallSuccess, chooseItemsMatch) use combined header
            if 'modelSuccessRates' in caseData:
                modelSuccessRates = caseData['modelSuccessRates']
                fMBSNameList = []
                groupHeader = ['', '']       # placeholders for Case Name, eval date columns
                combinedHeader = ['Case Name', 'eval date']
                partsOnlyHeader = ['Case Name', 'eval date']
                fMBSOnlyHeader = ['Case Name', 'eval date']
                for fMBSName in modelSuccessRates:
                    fMBSNameList += [fMBSName]
                    nParts = 0
                    if 'preliminaryTasks' in modelSuccessRates[fMBSName]:
                        for partName in modelSuccessRates[fMBSName]['preliminaryTasks']:
                            combinedHeader += [partName + ' of ' + fMBSName]
                            partsOnlyHeader += [partName + ' of ' + fMBSName]
                            nParts += 1
                    if nParts > 0:
                        groupHeader += ['Parts'] + [''] * (nParts - 1)
                    groupHeader += ['fMBS']
                    combinedHeader += [fMBSName]
                    fMBSOnlyHeader += [fMBSName]
        elif flexuAITaskType == 'part':
            partNamesList = list(geometryDescriptions)
            successTypes = partSuccessTypes
            header = ['Case Name', 'eval date'] + partNamesList
            allModelNamesList = partNamesList # overwrite
    for successType in successTypes:
        rows.append([])  # Empty row for spacing
        rows.append([successType+':'])
        if isFlexuAI and flexuAITaskType == 'fMBS':
            isPartOnly = successType in partOnlyMetrics
            isFMBSOnly = successType in fMBSOnlyMetrics
            if isPartOnly:
                rows.append(partsOnlyHeader)
            elif isFMBSOnly:
                rows.append(fMBSOnlyHeader)
            else: # shared metric
                rows.append(groupHeader)
                rows.append(combinedHeader)
            for caseName, caseData in resultsData.items():
                row = [caseName] + [caseData['eval date']]
                if 'modelSuccessRates' in caseData:
                    modelSuccessRates = caseData['modelSuccessRates']
                    try:
                        for fMBSName in fMBSNameList:
                            if not isFMBSOnly: # emit part columns for part-only and shared metrics
                                for partName in modelSuccessRates[fMBSName]['preliminaryTasks']:
                                    ratesDict = modelSuccessRates[fMBSName]['preliminaryTasks'][partName]
                                    value = ratesDict.get(successType, 'N/A')
                                    if IsScalar(value):
                                        value = NumToCSV(value,4) #gives e.g. 75.12%
                                    else:
                                        value = str(value)
                                    row.append(value)
                            if not isPartOnly: # emit fMBS column for fMBS-only and shared metrics
                                ratesDict = modelSuccessRates[fMBSName]
                                value = ratesDict.get(successType, 'N/A')
                                if IsScalar(value):
                                    value = NumToCSV(value,4) #gives e.g. 75.12%
                                else:
                                    value = str(value)
                                row.append(value)
                        rows.append(row)
                    except KeyboardInterrupt:
                        print("Keyboard interrupt requested...exiting")
                        raise
                    except Exception as e:
                        logger.LogDebug(f'Export modelSuccessRates raise exception={e}')
                        continue
        else:
            rows.append(header)
            # Data rows
            for caseName, caseData in resultsData.items():
                row = [caseName] + [caseData['eval date']]
                if 'modelSuccessRates' in caseData:
                    modelSuccessRates = caseData['modelSuccessRates']
                    try:
                        for key in allModelNamesList:
                            ratesDict = modelSuccessRates.get(key, {})
                            value = ratesDict.get(successType, 'N/A')
                            if IsScalar(value):
                                value = NumToCSV(value,4) #gives e.g. 75.12%
                            else:
                                value = str(value)
                            row.append(value)
                        rows.append(row)
                    except KeyboardInterrupt:
                        print("Keyboard interrupt requested...exiting")
                        raise
                    except Exception as e:
                        logger.LogDebug(f'Export modelSuccessRates raise exception={e}')
                        continue
    
    # Write to CSV
    with open(outputCSVpath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(rows)
    
    logger.LogDebug(f'ModelSuccessRates saved to: {outputCSVpath}',printToConsole=True)



def PlotPerformanceVsReleaseDate(mergedJsonFilePath, outputPlotPath, logger:Logger,
                                  metricKey='overall success'):
    """
    Scatter plot:
    X: LLM release date
    Y: a single metric (e.g. overall success)
    Bubble size: #params (B)
    Color: coder LLM vs. general-purpose LLM
    """

    try:
        import matplotlib.pyplot as plt
        import datetime
        import math
        from collections import defaultdict
        from matplotlib.lines import Line2D
        import matplotlib.patheffects as pe

        if not os.path.exists(mergedJsonFilePath):
            logger.LogDebug(f"ERROR: JSON file not found: {mergedJsonFilePath}", printToConsole=True)
            return

        data = ReadJSONToDict(mergedJsonFilePath)['resultsData']

        # bubble AREA is directly proportional to #params (B)
        # areaPerParam is a fixed points^2-per-billion-parameters scale factor
        areaPerParam = 12.0  # points^2 per billion parameters
        areaMinPoint = 50.0  # minimum area so that very small / zero-param entries stay visible

        x_dates = []
        y_vals = []
        sizes = []
        paramsList = []
        colors = []
        edgeColors = []
        isApiModel = []

        # API models have no publicly known parameter count (#params (B) <= 0) and are
        # drawn as grey circles instead of bubbles scaled by parameter size.
        apiMarkerSize = 90.0

        for caseName, caseData in data.items():
            try:
                release_date = caseData.get('LLM date', 'N/A')
                metricVal = float(str(caseData.get(metricKey, 0)).replace(',', '.'))
                params = float(str(caseData.get('#params (B)', 1)).replace(',', '.'))

                if release_date == 'N/A':
                    continue

                date_obj = datetime.datetime.strptime(release_date, "%Y-%m")
                isApi = params <= 0

                x_dates.append(date_obj)
                y_vals.append(metricVal)
                paramsList.append(params)
                isApiModel.append(isApi)
                sizes.append(apiMarkerSize if isApi else areaMinPoint + areaPerParam * max(0.0, params))
                is_coder_model = 'coder' in caseName.lower()
                colors.append('lightgreen' if isApi else ('tab:orange' if is_coder_model else 'tab:blue'))
                edgeColors.append(('tab:orange' if is_coder_model else 'tab:blue') if isApi
                                   else ('#FFD6A8' if is_coder_model else '#A9D6F5'))

            except KeyboardInterrupt:
                print("Keyboard interrupt requested...exiting")
                raise
            except:
                continue

        
        plt.figure(figsize=(5.0, 4.8))
        plt.scatter(x_dates, y_vals, s=sizes, c=colors,
                edgecolors=edgeColors, linewidths=1.5)

        # fix y-axis to [0,1] metric range with margin so bubbles at the extremes stay fully visible
        yMargin = 0.08
        plt.ylim(-yMargin, 1.0 + yMargin)

        legend_handles = [Line2D([0], [0], marker='o', color='w', label='coder LLMs',
                                markerfacecolor='tab:orange', markersize=8),
                            Line2D([0], [0], marker='o', color='w', label='general-purpose LLMs',
                                markerfacecolor='tab:blue', markersize=8),
                            Line2D([0], [0], marker='o', color='w', label='API models (#params unknown)',
                                markerfacecolor='lightgreen', markeredgecolor='tab:blue', markeredgewidth=1.5, markersize=8)]
        plt.legend(handles=legend_handles)

        # add text inside bubbles; distribute labels if multiple entries share same center
        pointGroups = defaultdict(list)
        for idx, (x, y) in enumerate(zip(x_dates, y_vals)):
            pointGroups[(x, round(y, 6))].append(idx)

        for _, idxList in pointGroups.items():
            nGroup = len(idxList)
            for iLocal, idx in enumerate(idxList):
                if isApiModel[idx]: #no known parameter count, hollow circle stays unlabeled
                    continue
                x = x_dates[idx]
                y = y_vals[idx]
                p = paramsList[idx]

                dx = 0.0
                dy = 0.0
                if nGroup > 1:
                    # place overlapping labels on a small ring around the bubble center
                    angle = 2.0 * math.pi * iLocal / nGroup
                    radiusPts = 7.0
                    dx = radiusPts * math.cos(angle)
                    dy = radiusPts * math.sin(angle)

                label = plt.annotate(
                    f"{p:.0f}B",  # show integer params (e.g. 7, 14, 70)
                    (x, y),
                    textcoords='offset points',
                    xytext=(dx, dy),
                    ha='center',
                    va='center',
                    fontsize=11,
                    color='white',
                    fontweight='bold'
                )
                label.set_path_effects([
                    pe.Stroke(linewidth=2.2, foreground='black', alpha=0.45),
                    pe.Normal()
                ])

        plt.xlabel("release date", fontsize=11)
        plt.ylabel("success rate", fontsize=11)

        plt.xticks(rotation=45)
        plt.tight_layout()

        # ensure directory exists
        os.makedirs(os.path.dirname(outputPlotPath), exist_ok=True)

        plt.savefig(outputPlotPath)
        logger.LogDebug(f"Plot saved to: {outputPlotPath}", printToConsole=True)

    except KeyboardInterrupt:
        print("Keyboard interrupt requested...exiting")
        raise
    except:
        logger.LogDebug(f"Could not complete PlotPerformanceVsReleaseDate for folder {mergedJsonFilePath}")
        

def PlotParameterVariationBars(mergedJsonFilePath, outputPlotPathPVCI, outputPlotPathPVMBS, logger:Logger):
    """
    Two grouped bar plots for parameter variation runs.
    PVCI plot: Y = 'choose items', bars per run index (0=standard, 1,2,... = PVCI runs).
    PVMBS plot: Y = 'overall success', bars per run index (0=standard, 1,2,... = PVMBS runs).
    Groups on X-axis: one group per base LLM model name.
    Standard runs (no suffix) appear as run index 0 in both plots.
    """
    try:
        import re
        import numpy as np
        import matplotlib.pyplot as plt

        if not os.path.exists(mergedJsonFilePath):
            logger.LogDebug(f"ERROR: JSON file not found: {mergedJsonFilePath}", printToConsole=True)
            return

        data = ReadJSONToDict(mergedJsonFilePath)['resultsData']

        pvPattern = re.compile(r'^(.+)-(PVCI|PVMBS)(\d+)$')

        pvciData = {}           # baseName -> {runIndex: float}  (choose items)
        pvciSuccessData = {}    # baseName -> {runIndex: float}  (overall success for PVCI runs)
        pvmbsData = {}          # baseName -> {runIndex: float}  (overall success)
        displayNames = {}       # baseName -> paper name for X-axis labels

        def ParseVal(rawValue):
            try:
                return float(str(rawValue).replace(',', '.'))
            except (ValueError, TypeError):
                return None

        for caseName, caseData in data.items():
            m = pvPattern.match(caseName)
            if m:
                baseName = m.group(1)
                pvType = m.group(2)
                runIdx = int(m.group(3))
            else:
                baseName = caseName
                pvType = None
                runIdx = 0

            if 'LLM paper name' in caseData:
                llmPaperName = caseData['LLM paper name']
            else:
                llmFullNameFallback = caseData.get('LLM full name', baseName)
                llmPaperName = ConvertFullName2modelKeyName(llmFullNameFallback)
                logger.LogText(f"WARNING: 'LLM paper name' not found for case '{caseName}', using fallback '{llmPaperName}'", printToConsole=True)
            displayNames.setdefault(baseName, llmPaperName)

            if pvType in (None, 'PVCI'):
                val = ParseVal(caseData.get('choose items'))
                if val is not None:
                    pvciData.setdefault(baseName, {})[runIdx] = val
                valSuccess = ParseVal(caseData.get('overall success'))
                if valSuccess is not None:
                    pvciSuccessData.setdefault(baseName, {})[runIdx] = valSuccess

            if pvType in (None, 'PVMBS'):
                val = ParseVal(caseData.get('overall success'))
                if val is not None:
                    pvmbsData.setdefault(baseName, {})[runIdx] = val

        def MakeBarPlot(groupedData, metricName, pvLabel, outputPath):
            if not groupedData:
                logger.LogDebug(f"No data for {pvLabel} plot — skipping", printToConsole=True)
                return

            allRunIndices = sorted(set(idx for runs in groupedData.values() for idx in runs))
            baseNames = list(groupedData.keys())
            tickLabels = [displayNames.get(base, base) for base in baseNames]
            nGroups = len(baseNames)
            nBars = len(allRunIndices)

            x = np.arange(nGroups)
            width = 0.8 / max(nBars, 1)

            # smaller figure so that, included at width=\linewidth, text appears larger in the paper;
            # same figsize as the overlay plot so both render at the same height in the paper
            fig, ax = plt.subplots(figsize=(max(5.5, nGroups * 1.1), 3.3))
            labelFontSize = 9
            tickFontSize = 9
            legendFontSize = 9
            ax.set_axisbelow(True)

            for i, runIdx in enumerate(allRunIndices):
                values = [groupedData[base].get(runIdx, 0.0) for base in baseNames]
                shortLabel = pvLabel[2:] if pvLabel.startswith('PV') else pvLabel
                label = f"{shortLabel}0 (standard)" if runIdx == 0 else f"{shortLabel}{runIdx}"
                offset = (i - nBars / 2.0 + 0.5) * width
                ax.bar(x + offset, values, width, label=label)

            ax.set_xlabel("LLM model", fontsize=labelFontSize)
            ax.set_ylabel(metricName, fontsize=labelFontSize)
            ax.set_xticks(x)
            ax.set_xticklabels(tickLabels, rotation=45, ha='right', fontsize=tickFontSize)
            ax.tick_params(axis='y', labelsize=tickFontSize)
            ax.set_ylim(0.0, 1.05)
            ax.legend(fontsize=legendFontSize, loc='lower right')
            ax.grid(True, axis='y')
            fig.tight_layout()

            os.makedirs(os.path.dirname(outputPath), exist_ok=True)
            fig.savefig(outputPath, dpi=300)
            plt.close(fig)
            logger.LogDebug(f"Plot saved to: {outputPath}", printToConsole=True)

        MakeBarPlot(pvmbsData, 'overall success', 'PVMBS', outputPlotPathPVMBS)

        # test plot: overlay both metrics in one plot with slight x-offset
        def MakeOverlayBarPlot(groupedDataA, metricNameA, groupedDataB, metricNameB, pvLabel, outputPath):
            if not groupedDataA or not groupedDataB:
                return
            allRunIndices = sorted(set(idx for runs in groupedDataA.values() for idx in runs))
            baseNames = list(groupedDataA.keys())
            tickLabels = [displayNames.get(base, base) for base in baseNames]
            nGroups = len(baseNames)
            nBars = len(allRunIndices)
            x = np.arange(nGroups)
            width = 0.8 / max(nBars, 1)
            nudge = width * 0.05
            shortLabel = pvLabel[2:] if pvLabel.startswith('PV') else pvLabel
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            hatches = ['/', '\\', '/', '\\', '/', '\\', '/']
            hatchLineWidth = 1.2
            plt.rcParams['hatch.linewidth'] = hatchLineWidth

            # smaller figure so that, included at width=\linewidth, text appears larger in the paper
            fig, ax = plt.subplots(figsize=(max(5.5, nGroups * 1.1), 3.3))
            labelFontSize = 9
            tickFontSize = 9
            legendFontSize = 9
            barHandles = []
            for i, runIdx in enumerate(allRunIndices):
                label = f"{shortLabel}0 (standard)" if runIdx == 0 else f"{shortLabel}{runIdx}"
                offset = (i - nBars / 2.0 + 0.5) * width
                color = colors[i % len(colors)]
                valsA = [groupedDataA[base].get(runIdx, 0.0) for base in baseNames]
                b = ax.bar(x + offset - nudge, valsA, width, color=color, alpha=0.95, zorder=2)
                barHandles.append((b[0], label))
            for i, runIdx in enumerate(allRunIndices):
                offset = (i - nBars / 2.0 + 0.5) * width
                color = colors[i % len(colors)]
                hatch = hatches[i % len(hatches)]
                valsB = [groupedDataB[base].get(runIdx, 0.0) for base in baseNames]
                ax.bar(x + offset + nudge, valsB, width, color=color, alpha=0.7, hatch=hatch, edgecolor='black', linewidth=hatchLineWidth, zorder=3)

            hatchPatch = plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.45, hatch='//')
            solidPatch = plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.95)
            legendHandles = [h for h, _ in barHandles] + [solidPatch, hatchPatch]
            legendLabels = [l for _, l in barHandles] + [metricNameA, metricNameB]

            ax.set_xlabel("LLM model", fontsize=labelFontSize)
            ax.set_ylabel(metricNameA, fontsize=labelFontSize)
            ax2 = ax.twinx()
            ax2.set_ylabel(metricNameB, fontsize=labelFontSize, labelpad=10)
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks([])
            ax.set_xticks(x)
            ax.set_xticklabels(tickLabels, rotation=45, ha='right', fontsize=tickFontSize)
            ax.tick_params(axis='y', labelsize=tickFontSize)
            ax.set_ylim(0.0, 1.05)
            ax.legend(legendHandles, legendLabels, loc='lower right', fontsize=legendFontSize, ncol=2)
            ax.grid(True, axis='y')
            fig.tight_layout()
            overlayPath = outputPath.replace('ChooseItems.png', 'ChooseItemsOverlay.png').replace('ChooseItems.pdf', 'ChooseItemsOverlay.pdf')
            os.makedirs(os.path.dirname(overlayPath), exist_ok=True)
            fig.savefig(overlayPath, dpi=300)
            plt.close(fig)
            logger.LogDebug(f"Overlay plot saved to: {overlayPath}", printToConsole=True)

        MakeOverlayBarPlot(pvciData, 'choose components', pvciSuccessData, 'overall success', 'PVCI',
                           outputPlotPathPVCI)

    except KeyboardInterrupt:
        print("Keyboard interrupt requested...exiting")
        raise
    except:
        logger.LogDebug(f"Could not complete PlotParameterVariationBars: {traceback.format_exc()}", printToConsole=True)


def PlotMetricCorrelationPerLLM(mergedJsonFilePath, keyX, keyY, outputPlotPath, logger:Logger):
    """
    Plot correlation between two metrics per model.
    Includes:
    - scatter points per run
    - mean + std per keyX
    - regression line per model
    - correlation coefficient output
    """

    try:
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        if not os.path.exists(mergedJsonFilePath):
            logger.LogDebug(f"ERROR: JSON file not found: {mergedJsonFilePath}")
            return
    
        data = ReadJSONToDict(mergedJsonFilePath)['resultsData']

        def IsThinkingEnabled(thinkingValue):
            # thinking is "on" unless it is False, 'low', or unknown ('N/A')
            thinkingStr = str(thinkingValue).strip().lower()
            return thinkingStr not in ('false', 'low', 'n/a')

        # group by (LLM, thinking enabled) so that thinking and non-thinking runs of the
        # same model are kept as separate series instead of being merged into one group
        grouped = defaultdict(list)

        for caseName, caseData in data.items():
            model = caseData.get('LLM paper name', 'N/A')
            isThinkCase = IsThinkingEnabled(caseData.get('thinking', 'N/A'))
            grouped[(model, isThinkCase)].append(caseData)

        L = pairedPlotLayout
        fig = plt.figure(figsize=L['figsize'])
        hasPlottedData = False
        # iterate over LLMs (and thinking/non-thinking variants)
        for (model, isThinkModel), cases in grouped.items():
            x_vals = []
            y_vals = []

            # collect raw values; skip sentinel/invalid values (-1 = unknown, >1 = out of valid rate range)
            for caseData in cases:
                try:
                    x = float(str(caseData.get(keyX, 0)).replace(',', '.'))
                    y = float(str(caseData.get(keyY, 0)).replace(',', '.'))

                    if x == -1 or y == -1 or x > 1 or y > 1:
                        continue

                    x_vals.append(x)
                    y_vals.append(y)
                except Exception:
                    continue

            if len(x_vals) < 3:
                logger.LogDebug(f"Skipping {model}: not enough data")
                continue

            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)
            hasConstantX = np.all(x_arr == x_arr[0])
            hasConstantY = np.all(y_arr == y_arr[0])

            if hasConstantX:
                logger.LogDebug(f"Skipping {model}: all {keyX} values are identical ({x_arr[0]}); correlation undefined")
                continue

            legendLabel = f"{model}" + (" (think)" if isThinkModel else "")

            # determine which points sit at this model's default temperature
            # drawn as hollow circles with
            defaultTempIdx = set()
            if keyX == 'temperature':
                try:
                    defaultTempVals = [float(str(c.get('default temperature', -1)).replace(',', '.')) for c in cases]
                    defaultTemp = defaultTempVals[0] if defaultTempVals else -1
                    if defaultTemp >= 0:
                        defaultTempIdx = {i for i, xv in enumerate(x_vals) if xv == defaultTemp}
                except Exception:
                    pass

            normalIdx = [i for i in range(len(x_vals)) if i not in defaultTempIdx]

            # plot raw data for metrics (excluding default-temperature points, drawn separately below)
            scatterHandle = plt.scatter([x_vals[i] for i in normalIdx], [y_vals[i] for i in normalIdx],
                                         alpha=0.8, label=legendLabel)
            hasPlottedData = True
            modelColor = scatterHandle.get_facecolor()[0] if normalIdx else None

            if defaultTempIdx:
                xHighlight = [x_vals[i] for i in defaultTempIdx]
                yHighlight = [y_vals[i] for i in defaultTempIdx]
                if modelColor is None:
                    # no normal points were plotted for this model; plot the highlight to get its color, then re-plot in matching color
                    placeholder = plt.scatter(xHighlight, yHighlight, alpha=0.0)
                    modelColor = placeholder.get_facecolor()[0]
                markerSize = 70
                plt.scatter(xHighlight, yHighlight, marker='o', s=markerSize,
                            facecolors='none', edgecolors=modelColor, linewidths=1.5, zorder=5)
                plt.scatter(xHighlight, yHighlight, marker='+', s=markerSize,
                            color=modelColor, linewidths=1.5, zorder=5)

            # compute correlation
            if hasConstantY:
                logger.LogDebug(f"{model}: correlation({keyX}, {keyY}) = undefined (constant {keyY})")
            else:
                corr = np.corrcoef(x_arr, y_arr)[0, 1]
                logger.LogDebug(f"{model}: correlation({keyX}, {keyY}) = {corr:.3f}")

            try:
                RankWarning = np.exceptions.RankWarning
            except:
                RankWarning = np.RankWarning
            # plot regression line (only meaningful when x varies)
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('error', RankWarning)
                    coef = np.polyfit(x_arr, y_arr, 1)
                x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
                y_line = np.polyval(coef, x_line)
                plt.plot(x_line, y_line, linestyle='--', alpha=0.7, color=modelColor)
            except (RankWarning, Exception):
                pass

        if not hasPlottedData:
            plt.close()
            logger.LogDebug("No correlation-plot generated: not enough data for any model.")
            return

        plt.xlabel(keyX, fontsize=L['labelFontSize'])
        plt.ylabel(keyY, fontsize=L['labelFontSize'])
        plt.xticks(fontsize=L['tickFontSize'])
        plt.yticks(fontsize=L['tickFontSize'])

        # legend below the axes (fixed axes box, aligned with repeated-runs plot)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=((L['left']+L['right'])/2, 0.0), ncol=2,
                   fontsize=L['legendFontSize'], frameon=False, handletextpad=0.2, columnspacing=0.6, handlelength=1.2)
        plt.grid(True)
        fig.subplots_adjust(left=L['left'], right=L['right'], top=L['top'], bottom=L['bottom'])
    
        os.makedirs(os.path.dirname(outputPlotPath), exist_ok=True)
        plt.savefig(outputPlotPath, dpi=300)
    
        logger.LogDebug(f"Plot saved to: {outputPlotPath}", printToConsole=True)
    except KeyboardInterrupt:
        print("Keyboard interrupt requested...exiting")
        raise
    except:
        logger.LogDebug(f"Could not complete PlotPerformanceVsReleaseDate for folder {mergedJsonFilePath}")
    

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# STATISTICS OF REPEATED RUNS (revision: statistical basis of success rates)
# repetitions of identical settings (LLM, temperature, task set) are compared via their overall success rate

def _ReadRunSummary(resultsJsonPath, cache, logger:Logger):
    """
    Read overall success and settings of one run (results.json); returns None if the run is incomplete.
    'cache' (path -> summary) avoids re-reading the large results.json files if size and modification time did not change.
    """
    absPath = os.path.abspath(resultsJsonPath)
    if not os.path.isfile(absPath):
        return None
    fileStat = os.stat(absPath)
    stamp = [fileStat.st_size, int(fileStat.st_mtime)]
    if absPath in cache and cache[absPath].get('stamp') == stamp:
        return cache[absPath]
    try:
        data = ReadJSONToDict(absPath)
    except Exception:
        logger.LogDebug(f"WARNING: could not read {absPath} (incomplete run?) - skipped", printToConsole=True)
        return None
    llmName = data.get('llmModelName', data.get('llmConfig', {}).get('modelName', 'N/A'))
    summary = {'stamp': stamp,
               'llmPaperName': ConvertFullName2modelKeyName(llmName),
               'temperature': data.get('agentConfig', {}).get('llmGenerateDefaultArgs', {}).get('temperature', None),
               'batchSize': data.get('llmBatchSize', None),
               'gpu': data.get('llmGPUused', 'N/A'),
               'nTasks': data.get('numberOfTasksWithVariationLoaded', None),
               'rate': data.get('rateSuccessMainTasks', None)}
    cache[absPath] = summary
    return summary

def PlotRepeatedRunStatistics(resultsRootDir, outputPlotPath, outputCSVpath, logger:Logger,
                              runFolderPattern='logsTM50T*_deterministicCheck*',
                              referenceDirs=[], cachePath=None, referenceDirsPerLLM={}):
    """
    Plot overall success of repeated runs with identical settings.
    Every folder matching runFolderPattern in resultsRootDir contains log_<config> subfolders;
    runs with the same <config> name are repetitions; incomplete runs (no results.json) are skipped.
    The plot shows each repetition, mean +- standard deviation (also written as numbers: mean in %, SD in pp),
    and the reference run reported in the paper (first match in referenceDirs, same LLM and temperature).
    referenceDirsPerLLM (LLM paper name -> list of dirs) overrides referenceDirs for single LLMs.
    Statistics are also written to a CSV file.
    """
    try:
        import glob
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        from matplotlib.lines import Line2D

        if cachePath is None:
            cachePath = JoinPath(resultsRootDir, 'repeatedRunsCache.json')
        cache = ReadJSONToDict(cachePath) if os.path.isfile(cachePath) else {}

        # collect repetitions per configuration (log folder name without 'log_')
        runsPerConfig = defaultdict(list)
        for logFolder in sorted(glob.glob(os.path.join(resultsRootDir, runFolderPattern, 'log_*'))):
            summary = _ReadRunSummary(os.path.join(logFolder, 'results.json'), cache, logger)
            if summary is None:
                logger.LogDebug(f"skipping {logFolder}: no (complete) results.json", printToConsole=True)
                continue
            runsPerConfig[os.path.basename(logFolder)[len('log_'):]].append(summary)

        def FindReference(configName, temperature, llmPaperName):
            # reference: run reported in the paper with same LLM and temperature (batch size may differ)
            llmShortName = configName.split('-T')[0].split('-B')[0]
            for refDir in referenceDirsPerLLM.get(llmPaperName, referenceDirs):
                for logFolder in sorted(glob.glob(os.path.join(refDir, 'log_' + llmShortName + '*' + f"-T{temperature}"))):
                    if 'Thk' in os.path.basename(logFolder):
                        continue
                    summary = _ReadRunSummary(os.path.join(logFolder, 'results.json'), cache, logger)
                    if summary is not None:
                        return summary
            return None

        stats = []
        for configName, runs in runsPerConfig.items():
            if len(runs) < 2:
                logger.LogDebug(f"skipping {configName}: only {len(runs)} run(s)", printToConsole=True)
                continue
            rates = np.array([run['rate'] for run in runs], dtype=float)
            meanRate = rates.mean()
            reference = FindReference(configName, runs[0]['temperature'], runs[0]['llmPaperName'])
            stats.append({'config': configName, 'llm': runs[0]['llmPaperName'], 'temperature': runs[0]['temperature'],
                          'batchSize': runs[0]['batchSize'], 'gpu': runs[0]['gpu'], 'nRuns': len(runs),
                          'mean': meanRate, 'sd': rates.std(ddof=1), 'min': rates.min(), 'max': rates.max(),
                          'referenceRate': reference['rate'] if reference else None,
                          'referenceBatchSize': reference['batchSize'] if reference else None,
                          'referenceGPU': reference['gpu'] if reference else None,
                          'rates': rates})
        WriteDictToJSON(cache, cachePath)
        if not stats:
            logger.LogDebug("PlotRepeatedRunStatistics: no repeated runs found", printToConsole=True)
            return

        stats.sort(key=lambda s: (s['llm'], float(s['temperature'])))
        llmNames = []
        for s in stats:
            if s['llm'] not in llmNames:
                llmNames.append(s['llm'])
        temperatures = sorted(set(float(s['temperature']) for s in stats))

        # CSV
        csvKeys = ['config', 'llm', 'temperature', 'batchSize', 'gpu', 'nRuns', 'mean', 'sd', 'min', 'max',
                   'referenceRate', 'referenceBatchSize', 'referenceGPU']
        with open(outputCSVpath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(csvKeys + ['rates of runs'])
            for s in stats:
                writer.writerow([NumToCSV(round(float(s[k]), 5)) if isinstance(s[k], (float, np.floating)) else s[k] for k in csvKeys]
                                + [' '.join(f"{r:.3f}" for r in s['rates'])])
        logger.LogDebug(f"CSV saved to: {outputCSVpath}", printToConsole=True)

        # plot (same figure size as temperature correlation plot, as both are shown side by side in the paper)
        tempColors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
        colorOfTemp = {t: tempColors[i % len(tempColors)] for i, t in enumerate(temperatures)}
        L = pairedPlotLayout
        fig, ax = plt.subplots(figsize=L['figsize'])

        groupWidth = 0.8
        slotWidth = groupWidth/len(temperatures)
        rng = np.random.default_rng(0) # fixed jitter
        for s in stats:
            x0 = llmNames.index(s['llm']) - groupWidth/2 + (temperatures.index(float(s['temperature'])) + 0.5)*slotWidth
            jitter = rng.uniform(-0.18, 0.18, len(s['rates']))*slotWidth
            ax.scatter(x0 + jitter, s['rates'], s=14, color=colorOfTemp[float(s['temperature'])], alpha=0.75, linewidths=0, zorder=3)
            ax.errorbar(x0, s['mean'], yerr=s['sd'], fmt='_', color='black', markersize=9,
                        capsize=2.5, elinewidth=1.1, markeredgewidth=1.3, zorder=4)
            if s['referenceRate'] is not None:
                ax.scatter([x0 + 0.33*slotWidth], [s['referenceRate']], marker='D', s=22,
                           facecolors='none', edgecolors='black', linewidths=1.1, zorder=5)
            # mean (%) and SD (pp) written below the lowest point of each group
            yLow = min([s['min']] + ([s['referenceRate']] if s['referenceRate'] is not None else []))
            ax.text(x0, yLow - 0.008, f"{100*s['mean']:.1f}\n$\\pm${100*s['sd']:.2f}", ha='center', va='top',
                    fontsize=L['annotationFontSize'], linespacing=1.0, zorder=6,
                    bbox=dict(boxstyle='square,pad=0.05', facecolor='white', edgecolor='none', alpha=0.8))

        ax.set_xticks(range(len(llmNames)))
        ax.set_xticklabels(llmNames, fontsize=L['tickFontSize'], rotation=20, ha='right', rotation_mode='anchor')
        ax.set_xlim(-0.5, len(llmNames) - 0.5)
        # vertical lines separating the LLMs
        for xSep in range(1, len(llmNames)):
            ax.axvline(xSep - 0.5, color='0.45', linewidth=0.9, zorder=1)
        ax.set_ylabel('overall success', fontsize=L['labelFontSize'])
        ax.tick_params(axis='y', labelsize=L['tickFontSize'])
        # major grid every 0.1 (labelled), additional lines every 0.05 in between
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.grid(True, axis='y', which='major')
        ax.grid(True, axis='y', which='minor', color='0.8', linewidth=0.7, linestyle='--')
        ax.tick_params(axis='y', which='minor', length=2)
        ax.set_axisbelow(True)
        allY = [v for s in stats for v in [s['min'], s['max']]] + \
               [s['referenceRate'] for s in stats if s['referenceRate'] is not None]
        yRange = max(allY) - min(allY)
        # extra space below for the mean/SD numbers
        ax.set_ylim(min(allY) - 0.17*yRange, max(allY) + 0.05*yRange)
        legendHandles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colorOfTemp[t], markersize=7,
                                label=f'runs, temp. {t:g}') for t in temperatures]
        legendHandles += [Line2D([0], [0], marker='_', color='black', markersize=12, markeredgewidth=1.6, label='mean $\\pm$ SD'),
                          Line2D([0], [0], marker='D', color='w', markerfacecolor='none', markeredgecolor='black',
                                 markersize=6, label='reference run')]
        # legend below the axes (fixed axes box, aligned with temperature correlation plot)
        fig.legend(handles=legendHandles, loc='lower center', bbox_to_anchor=((L['left']+L['right'])/2, 0.0), ncol=2,
                   fontsize=L['legendFontSize'], frameon=False, handletextpad=0.2, columnspacing=0.6, handlelength=1.2)
        fig.subplots_adjust(left=L['left'], right=L['right'], top=L['top'], bottom=L['bottom'])
        os.makedirs(os.path.dirname(os.path.abspath(outputPlotPath)), exist_ok=True)
        fig.savefig(outputPlotPath, dpi=300)
        plt.close(fig)
        logger.LogDebug(f"Plot saved to: {outputPlotPath}", printToConsole=True)

        for s in stats:
            refStr = '--' if s['referenceRate'] is None else f"{100*s['referenceRate']:.1f}%"
            logger.LogDebug(f"{s['config']}: runs={s['nRuns']}, mean={100*s['mean']:.2f}%, SD={100*s['sd']:.2f}pp, "
                            f"range={100*s['min']:.1f}-{100*s['max']:.1f}%, reference={refStr}", printToConsole=True)

    except KeyboardInterrupt:
        print("Keyboard interrupt requested...exiting")
        raise
    except:
        logger.LogDebug(f"Could not complete PlotRepeatedRunStatistics: {traceback.format_exc()}", printToConsole=True)

# merge results of a results folder and create CSV files and plots; go into /src folder and call, e.g.:
#   python postProcessing.py ../results/logsTempVar_merge_v1 -temperature
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-process LLM test results: merge JSON, export CSVs, and optionally generate plots.")
    parser.add_argument("directory",
                        help="Directory containing log_* subdirectories with results.json files")
    parser.add_argument("-time", action="store_true", default=False,
                        help="Plot LLM overall success vs. release date (scatter plot)")
    parser.add_argument("-temperature", action="store_true", default=False,
                        help="Plot correlation between temperature and overall success per LLM")
    parser.add_argument("-pv", action="store_true", default=False,
                        help="Plot parameter variation bar charts (PVCI: choose items, PVMBS: overall success)")
    parser.add_argument("-repeat", action="store_true", default=False,
                        help="Plot statistics of repeated runs (directory = results folder containing the repeated-run folders)")
    args = parser.parse_args()

    resultsDir = args.directory
    logDir = JoinPath(resultsDir, "logsTrash")
    jsonOutputFilePath = JoinPath(resultsDir, "allLLMresults.json")
    overallResultsCSVoutputFilePath = JoinPath(resultsDir, "overallResults.csv")
    overallResultsSuccessRatesCSVoutputFilePath = JoinPath(resultsDir, "overallResultsSuccessRates.csv")

    overallSuccessVsReleaseDatePlotPath = JoinPath(resultsDir, "overallSuccessVsReleaseDatePlot.pdf")
    chooseItemsVsReleaseDatePlotPath = JoinPath(resultsDir, "chooseItemsVsReleaseDatePlot.pdf")
    correlationPlotPath = JoinPath(resultsDir, "temperatureCorrelation.pdf")

    pvciBarPlotPath = JoinPath(resultsDir, "paramVariationChooseItems.pdf")
    pvmbsBarPlotPath = JoinPath(resultsDir, "paramVariationOverallSuccess.pdf")

    logger = Logger(logDir=logDir)

    # statistics of repeated runs (revision: statistical basis of success rates)
    # call from /src folder:  python postProcessing.py ../results -repeat
    if args.repeat:
        PlotRepeatedRunStatistics(resultsRootDir=resultsDir,
                                  outputPlotPath=JoinPath(resultsDir, "repeatedRunsStatistics.pdf"),
                                  outputCSVpath=JoinPath(resultsDir, "repeatedRunsStatistics.csv"),
                                  logger=logger,
                                  runFolderPattern='logsRepeatedRuns/logsTemp*', # repetitions on RTX 5090
                                  # reference runs reported in the paper: main table (T=0) and temperature study (T=1)
                                  referenceDirs=[JoinPath(resultsDir, "logsTM50_temp0_merge_v2"),
                                                 JoinPath(resultsDir, "logsTempVar_merge_v1")],
                                  # gemma4:26b: Table 1 reports the temperature-study run (same settings as the repetitions)
                                  referenceDirsPerLLM={'gemma4:26b': [JoinPath(resultsDir, "logsTempVar_merge_v1")]})
        if not (args.time or args.temperature or args.pv):
            sys.exit(0) # only repeated-run statistics requested; resultsDir is not a single run folder to be merged
    mergedResults = MergeTestModelResults(resultsDir, jsonOutputFilePath, logger)

    # Export overall results per LLM to CSV using the csvKeys
    ExportMergedResultsToCSV(jsonOutputFilePath, overallResultsCSVoutputFilePath, logger)

    # Export model success rates per LLM to CSV using the modelSuccessRates dict in caseData
    ExportModelSuccessRatesToCSV(jsonOutputFilePath, overallResultsSuccessRatesCSVoutputFilePath, logger)

    # plot LLM performance (overall success) in MBS generation progress over time (release dates)
    if args.time:
        PlotPerformanceVsReleaseDate(jsonOutputFilePath, overallSuccessVsReleaseDatePlotPath, logger, metricKey='overall success')
        # plot LLM performance (choose items) in MBS generation progress over time (release dates)
        PlotPerformanceVsReleaseDate(jsonOutputFilePath, chooseItemsVsReleaseDatePlotPath, logger, metricKey='choose items')

    # plot correlation between two metrics in generation results over different LLMs (e.g. temperature and overall success)
    if args.temperature:
        PlotMetricCorrelationPerLLM(mergedJsonFilePath=jsonOutputFilePath,
                              keyX='temperature', keyY='overall success', # change keys to analyze other metric-correlations
                              outputPlotPath=correlationPlotPath,
                              logger=logger)

    # plot parameter variation bar charts: PVCI runs (choose items) and PVMBS runs (overall success)
    if args.pv:
        PlotParameterVariationBars(mergedJsonFilePath=jsonOutputFilePath,
                                   outputPlotPathPVCI=pvciBarPlotPath,
                                   outputPlotPathPVMBS=pvmbsBarPlotPath,
                                   logger=logger)
