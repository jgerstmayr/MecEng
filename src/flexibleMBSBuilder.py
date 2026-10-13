#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Details:  This is the flexible MultiBody Systems (fMBSs) builder; it is the
#           main component of the 'flexuAI' project; it builds upon the
#           ModelBuilder; a femInterface is referred to as 'part' (as it
#           can be built into a machine) as if it were a flexible body; the
#           flexible MBS is the 'fMBS'. Main changes compared to the
#           ModelBuilder are:
#           -> pre and post subtasks are devided depending on the taskType
#           -> special exuAI_in and exuAI_out structure is established
#           -> the global dict is structure deeper for preliminary tasks.
#
# Author:   Sebastian Weyrer
# Date:     2026-04-27
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%% necessary imports and class function / class-methods definitions
import os
import re
import time
import copy
import numpy as np
import exudyn as exu
from exudyn.FEM import *
from ExuAI.logger import Logger
from ExuAI.templatesAndLists import simpleMBSChooseItemsPromptTemplate, simpleMBSGenerateCodePromptTemplate, partsGenerateCodePromptTemplate, partsChooseItemsPromptTemplate
from ExuAI.utilities import JoinPath, CheckLLMoutput, PreProcessLLMcode, String2PythonComment, EvaluateNumerical
from ExuAI.flexUtilities import RelativeMassDifference, RelativeEigenfrequenciesNormDifference, MeshVolumeDifference, AddSpecialCodeForPartsCreation, CheckIfSafetyCheckIsPresent, MeshDictLog
from modelBuilder import MakeTaskDataDict, GenerateMBSTask, ChooseItemsTask, WriteFile, CompareLLMAndSampleMBSNonNumerical, globalEvaluationMetrics
scriptDir = os.path.dirname(os.path.abspath(__file__)) # get directory of this file

# derived class from the standard ChooseItemsTask
# the derivation of the class was necessary because
# -> prompts (in PreSubtask) and
# -> continuation (in PreSubtask)
# have to be different. PostSubtask is NOT modified (PostSubtask with itemsCheck is the same as in 'normal' modelBuilder (and only applies for fMBS)!)
class ChooseItemsTaskForFlexuAI(ChooseItemsTask):
    def GetTaskName(self):
        return 'chooseItems'

    # taskData is globalData['storedDataPerTask']['taskNameWithID']
    # subTaskData is taskData['generateCode'], where 'generateCode' is evaluationTaskName
    # currentTaskNameWithID is obtained via dict key and not as value of a key
    def PreSubtask(self, subTaskData, taskData, currentMBSmodelNameID, currentMBSmodelName):

        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # decide whether we have a part or a fMBS
        if 'taskType' in taskData:
            if taskData['taskType'] == 'fMBS':
                # decide whether we should skip this fMBS or whether we can continue
                if 'skipDueToPartsError' in taskData:
                    if taskData['skipDueToPartsError'] == False:
                        prompt = self.agent.GetItemSelectionPrompt(taskData['taskDescription'], helperFileName=self.helperFileName,
                                                                    promptTemplate=simpleMBSChooseItemsPromptTemplate)
                    elif taskData['skipDueToPartsError'] == True:
                        self.agent.logger.LogText('Skipping fMBS "' + currentMBSmodelNameID + '" due to part(s) not being made correctly.',
                                                    printToConsole=True, separator=True, quickinfo=True)
                        return False
                else:
                    self.agent.logger.LogDebug('Key "skipDueToPartsError" is not available! Check code.',
                                            printToConsole=True, separator=True)
                    return False
            elif taskData['taskType'] == 'part':
                if not self.skipBatchedInference:
                    prompt = self.agent.GetItemSelectionPrompt(taskData['taskDescription'], helperFileName=self.helperFileName,
                                                                promptTemplate=partsChooseItemsPromptTemplate)
                elif self.skipBatchedInference:
                    prompt = ''
            else:
                self.agent.logger.LogDebug('Pre-ChooseItemsTaskForFlexuAI: Specified task type is not available!',
                                            printToConsole=True, separator=True)
                return False
        else:
            self.agent.logger.LogDebug('Pre-ChooseItemsTaskForFlexuAI: Task type is not available in model descriptions!',
                                            printToConsole=True, separator=True)
            return False
        subTaskData['prompt'] = prompt
        return True #success of subtask

# derived class from the standard GenerateMBSTask
# the derivation of the class was necessary because
# -> prompts (in PreSubtask) and
# -> evaluation (in PostSubtask)
# have to be different.
class GenerateCodeTaskForFlexuAI(GenerateMBSTask):
    def GetTaskName(self):
        return 'generateCode'

    # taskData is globalData['storedDataPerTask']['taskNameWithID']
    # subTaskData is taskData['generateCode']
    # currentTaskNameWithID is obtained via dict key and not as value of a key
    def PreSubtask(self, subTaskData, taskData, currentMBSmodelNameID, currentMBSmodelName):
        helperTags = taskData[ChooseItemsTaskForFlexuAI.GetTaskName(None)]['helperTags']
        mbsModelDescription=taskData['taskDescription']
        subTaskData['chooseItemsMatch'] = taskData['chooseItems']['chooseItemsMatch'] # for parts and fMBS

        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # decide whether we have a part or a fMBS
        if 'taskType' in taskData:
            if taskData['taskType'] == 'fMBS':
                [prompt] = self.agent.GetCodeGenerationPrompt(helperTags, mbsModelDescription, helperFileName=self.helperFileName,
                                                              promptTemplate=simpleMBSGenerateCodePromptTemplate)
            elif taskData['taskType'] == 'part':
                [prompt] = self.agent.GetCodeGenerationPrompt(helperTags, mbsModelDescription, helperFileName=self.helperFileName,
                                                              promptTemplate=partsGenerateCodePromptTemplate)
            else:
                self.agent.logger.LogDebug('Pre-GenerateCodeTaskForFlexuAI: Specified task type is not available!',
                                            printToConsole=True, separator=True)
                return False
        else:
            self.agent.logger.LogDebug('Pre-GenerateCodeTaskForFlexuAI: Task type is not available in model descriptions!',
                                            printToConsole=True, separator=True)
            return False
        subTaskData['prompt'] = prompt
        return True
    
    # taskData is globalData['storedDataPerTask']['taskNameWithID']
    # subTaskData is taskData['generateCode'], where 'generateCode' is evaluationTaskName
    # currentTaskNameWithID is obtained via dict key and not as value of a key
    def PostSubtask(self, subTaskData, taskData, currentTaskNameWithID, currentTaskName):
        try:
            # get the response of the LLM and clean it up such that it is a code
            # Set context window exceeded flag and token count from inference (stored in BatchedInference)
            contextWindowExceeded = subTaskData.get('contextWindowExceeded', False)
            contextWindowExceededTokens = subTaskData.get('contextWindowExceededTokens', 0)
            subTaskData['llmNoContextWindowExceeded'] = not contextWindowExceeded
            subTaskData['contextWindowExceededTokens'] = contextWindowExceededTokens
            if contextWindowExceeded:
                raise RuntimeError(
                    f"Context window exceeded by {contextWindowExceededTokens} tokens for task '{currentTaskNameWithID}'")
            response = CheckLLMoutput(self.agent.logger, subTaskData['response'], checkUndesiredKeywords=True, checkHarmfulCommands=True)
            self.agent.logger.LogDebug(f"GenerateCodeTask: PostSubtask: CheckLLMoutput for {currentTaskNameWithID}")
            LLMCode = PreProcessLLMcode(self.agent.logger, response, addTimeout=False, addFileWrite=False)
            self.agent.logger.LogDebug(f"GenerateCodeTask: PostSubtask: PreProcessLLMcode for {currentTaskNameWithID}")
            LLMCode = LLMCode.split('\nmbs.Assemble()')[0]+'\n\n' # remove mbs.Assemble from LLM code
            self.agent.logger.LogDebug('cleaned-up code of model:\n' + LLMCode)
            self.agent.totalNumberOfCodesGenerated += 1

            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # decide where to write solutionFiles (not for parts) and set directory accordingly
            subTaskData['solutionNameLLM'] = JoinPath(self.globalData['solutionDir'], currentTaskNameWithID + 'LLM.txt')
            subTaskData['solutionNameSample'] = JoinPath(self.globalData['solutionDir'], currentTaskName + 'Sample.txt') # will change over time
            sampleFileDict = self.agent.GetSampleFileDict(filePath=JoinPath(scriptDir, 'flexu/sampleFiles'),
                                                           mbsModelName=currentTaskName,
                                                           sampleDict=taskData['dataOfTask'],
                                                           chosenParametersDict=taskData['chosenParameters']) 
            solverCode = sampleFileDict['solverCode']
            sampleCode = sampleFileDict['executableCode']
            subTaskData['sampleCode'] = sampleFileDict['executableCodeShort']

            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # harmonize code
            LLMCodeHarmonized = LLMCode + solverCode

            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # decide whether we have a part or a fMBS
            # self.globalData['taskSolutionLLMDir'] here the LLM generated code etc. should be saved (part (.py / .npz) and fMBS (.py))
            # self.globalData['taskSolutionSampleDir'] here the sample based code should be saved (part (.py) and fMBS (.py))
            # for fMBS, a subfolder is made (and in this subfolder the parts are saved)
            # if only parts are created, no subfolder is made and .py and .npz lies in this directory
            # solution folder stays empty if only parts are made (per definition)
            startTime = time.time() # get new start time
            if 'taskType' in taskData:
                if taskData['taskType'] == 'fMBS':
                    self.agent.flexuAIFMBSGenerated += 1
                    self.agent.logger.LogText('Evaluate fMBS ' + currentTaskNameWithID, 
                                               printToConsole=True, separator=True, quickinfo=True)
                    # place assemble flag for special metrics
                    LLMCodeHarmonized = LLMCodeHarmonized.replace('mbs.Assemble()',
                                                                  f'mbs.sys["{self.agent.preAssembleFlag}"]=True\n'+
                                                                  'mbs.Assemble()\n'+
                                                                  f'mbs.sys["{self.agent.postAssembleFlag}"]=True')
                    fMBSModelDescriptionPython = '# ** given model description: **\n'
                    fMBSModelDescriptionPython += String2PythonComment(taskData['taskDescription'])
                    saveDirectoryOfLLMfMBS = self.globalData['taskSolutionLLMDir'] + '/' + currentTaskNameWithID + '/' # here current name and in part it is superior name then
                    saveDirectoryOfSamplefMBS = self.globalData['taskSolutionSampleDir'] + '/' + currentTaskNameWithID + '/'
                    WriteFile(currentTaskNameWithID + '.py',
                            fMBSModelDescriptionPython + LLMCodeHarmonized,
                            path=saveDirectoryOfLLMfMBS, flush=True)
                    WriteFile(currentTaskName + '_sample.py',
                            sampleCode,
                            path=saveDirectoryOfSamplefMBS, flush=True)
                    self.agent.logger.LogText('Saving code for fMBS "' + currentTaskNameWithID + '" at ' + '(LLM) "' + saveDirectoryOfLLMfMBS + '" and ' + '(Sample) "' + saveDirectoryOfSamplefMBS + '".', 
                                               printToConsole=False, separator=True, quickinfo=True)
                    
                    #++++++++++++++++++++++++++++++++++++++++++++++++++
                    # run the LLM generated code and the sample file code
                    # by making a string replacement we safely ensure that the fMBS uses exactly that part that is intended
                    pathOfSavedPart = self.globalData['taskSolutionLLMDir'] + '/' + currentTaskNameWithID + '/'
                    LLMCodeHarmonized = re.sub(r"LoadFromFile\(\s*(['\"])([^'\"]+\.npz)\1\s*\)", rf"LoadFromFile(\1{pathOfSavedPart}\2\1)", LLMCodeHarmonized)
                    LLMCodeExecutionDict = self.agent.EvaluateMBScode(LLMCodeHarmonized, currentTaskNameWithID,
                                                                    solutionFileName=subTaskData['solutionNameLLM'], 
                                                                    mode='LLM based', printErrorToConsole=False)
                    subTaskData['executableLLM'] = LLMCodeExecutionDict['executable']
                    localVariablesLLM = LLMCodeExecutionDict['localVariables']
                    solverSuccessLLM = LLMCodeExecutionDict['solverSuccess']
                    subTaskData['solverInitSuccessful'] = LLMCodeExecutionDict['solverInitSuccessful']
                    subTaskData['solverSuccess'] = LLMCodeExecutionDict['solverSuccess'] 
                    subTaskData['reachedMbsAssemble'] = LLMCodeExecutionDict['reachedMbsAssemble']
                    subTaskData['finishedMbsAssemble'] = LLMCodeExecutionDict['finishedMbsAssemble']
                    subTaskData['solverNoTimeoutLLM'] = LLMCodeExecutionDict['solverNoTimeout']
                    # do this such that the correct part is loaded
                    sampleCode = re.sub(r"LoadFromFile\(\s*(re\.sub\([^)]+\)\s*\+\s*['\"]\.npz['\"])\s*\)", rf"LoadFromFile('{pathOfSavedPart}' + \1)", sampleCode)
                    sampleCodeExecutionDict = self.agent.EvaluateMBScode(sampleCode, currentTaskNameWithID,
                                                                        solutionFileName=subTaskData['solutionNameSample'], 
                                                                        mode='sample file based', printErrorToConsole=False)
                    subTaskData['executableSample'] = sampleCodeExecutionDict['executable']
                    localVariablesSample = sampleCodeExecutionDict['localVariables']
                    solverSuccessSample = sampleCodeExecutionDict['solverSuccess']
                    subTaskData['solverNoTimeoutSample'] = sampleCodeExecutionDict['solverNoTimeout']

                    #++++++++++++++++++++++++++++++++++++++++++++++++++
                    # do (non-) numerical evaluation of mbs
                    CompareLLMAndSampleMBSNonNumerical(self.agent, subTaskData, taskData, currentTaskNameWithID,
                                                       localVariablesLLM=localVariablesLLM, localVariablesSample=localVariablesSample,
                                                       taskName=self.taskName)
                    if (subTaskData['executableSample'] and subTaskData['executableLLM'] and solverSuccessSample and solverSuccessLLM):
                        subTaskData['solutionFileDiff'] = EvaluateNumerical(fileNameReference=subTaskData['solutionNameSample'],
                                                                            fileNameEvaluate=subTaskData['solutionNameLLM'],
                                                                            localVariablesReference=localVariablesSample,
                                                                            localVariablesEvaluate=localVariablesLLM,
                                                                            logger=self.agent.logger)
                    self.agent.flexuAIFMBSEvalTime += time.time() - startTime
                elif taskData['taskType'] == 'part':
                    self.agent.flexuAIPartsGenerated += 1
                    self.agent.logger.LogText('Evaluate part "' + currentTaskNameWithID + '"; (preliminary) task (with ID) "' + taskData['currentTaskNameWithID'] + '" in global data.', 
                                               printToConsole=True, separator=True, quickinfo=True)
                    partModelDescriptionPython = '# ** given model description: **\n'
                    partModelDescriptionPython += String2PythonComment(taskData['taskDescription'])
                    if 'superiorTaskNameWithID' not in taskData:
                        saveDirectoryOfLLMPart = self.globalData['taskSolutionLLMDir'] + '/'
                        saveDirectoryOfSamplePart = self.globalData['taskSolutionSampleDir'] + '/'
                        saveNameOfCodeForPart = currentTaskNameWithID
                    else:
                        # here we have to access a special variable
                        saveDirectoryOfLLMPart = self.globalData['taskSolutionLLMDir'] + '/' + taskData['superiorTaskNameWithID'] + '/'
                        saveDirectoryOfSamplePart = self.globalData['taskSolutionSampleDir'] + '/' + taskData['superiorTaskNameWithID'] + '/'
                        saveNameOfCodeForPart = taskData['partName'] # it should just be a clear name in this case
                    LLMCodeHarmonized = AddSpecialCodeForPartsCreation(LLMCodeHarmonized)
                    sampleCode = AddSpecialCodeForPartsCreation(sampleCode)
                    WriteFile(saveNameOfCodeForPart + '.py',
                            partModelDescriptionPython + LLMCodeHarmonized,
                            path=saveDirectoryOfLLMPart, flush=True)
                    WriteFile(saveNameOfCodeForPart + '_sample.py',
                              sampleCode,
                              path=saveDirectoryOfSamplePart, flush=True)
                    self.agent.logger.LogText('Saving code for part "' + saveNameOfCodeForPart + '" at ' + '(LLM) "' + saveDirectoryOfLLMPart + '" and ' + '(Sample) "' + saveDirectoryOfSamplePart + '".', 
                                               printToConsole=False, separator=True, quickinfo=True)
                    
                    #++++++++++++++++++++++++++++++++++++++++++++++++++
                    # general initializations
                    materialLLMPresentForCheck = False
                    materialSamplePresentForCheck = False
                    inDict = exu.sys['exuAI_in']
                    # dict's build in inDict.copy() makes a shallow copy (so only non-nested objects are copied which is ok here)
                    # here, we use copy.copy(inDict) which also makes a shallow copy
                    subTaskData['exuAI_in'] = copy.copy(inDict)

                    #++++++++++++++++++++++++++++++++++++++++++++++++++
                    # run the LLM generated code and the sample file code
                    exu.sys['exuAI_out'] = {} # INITIALIZE AS EMPTY DICT
                    # if safety check is not present, just skip everything on LLM-side
                    if not CheckIfSafetyCheckIsPresent(LLMCodeHarmonized):
                        self.agent.logger.LogText('LLM code does not include mesh node safety check.', 
                                                printToConsole=False, separator=True, quickinfo=True)
                    else:
                        try:
                            LLMCodeExecutionDict = self.agent.EvaluateMBScode(LLMCodeHarmonized, currentTaskName,
                                                                            solutionFileName=subTaskData['solutionNameLLM'], # not written / not used
                                                                            mode='LLM based', isPart=True,
                                                                            printErrorToConsole=False,
                                                                            )
                            outDict = exu.sys['exuAI_out'] # get out dict again
                            if LLMCodeExecutionDict['solverSuccess']: # only then we have the outDict
                                MeshDictLog('LLM code', self.agent, inDict=inDict, outDict=outDict)
                            subTaskData['executableLLM'] = LLMCodeExecutionDict['executable']
                            if subTaskData['executableLLM']:
                                LLMPart = outDict['femObject'] # get the LLM part out of the dict (we always have this entry in the outDict as we add it)
                            if 'materials' in outDict:
                                LLMMaterials = outDict['materials']
                                materialLLMPresentForCheck = True
                        except:
                            raise ValueError('Post-GenerateCodeTaskForFlexuAI: EvaluateMBScode failed for part')
                    
                    #++++++++++++++++++++++++++++++++++++++++++++++++++
                    # run the sample file code
                    exu.sys['exuAI_out'] = {} # INITIALIZE AS EMPTY DICT
                    if not CheckIfSafetyCheckIsPresent(sampleCode):
                        self.agent.logger.LogError('Sample code does not include mesh node safety check.', 
                                                    printToConsole=True, separator=True, quickinfo=True)
                        raise Exception # this should NEVER happen!
                    else:
                        sampleCodeExecutionDict = self.agent.EvaluateMBScode(sampleCode, currentTaskName,
                                                                            solutionFileName=subTaskData['solutionNameSample'], # not written / not used
                                                                            mode='sample file based', isPart=True)
                        outDict = exu.sys['exuAI_out'] # get out dict again
                        if sampleCodeExecutionDict['solverSuccess']: # only then we have the outDict
                            MeshDictLog('Sample code', self.agent, inDict=inDict, outDict=outDict)
                        subTaskData['executableSample'] = sampleCodeExecutionDict['executable']
                        if subTaskData['executableSample']:
                            samplePart = outDict['femObject']
                        if 'materials' in outDict:
                            sampleMaterials = outDict['materials']
                            materialSamplePresentForCheck = True
                    
                    #++++++++++++++++++++++++++++++++++++++++++++++++++
                    # evaluate part
                    # for this first check, no executability is neccessary!
                    if materialLLMPresentForCheck and materialSamplePresentForCheck:
                        LLMMaterial = LLMMaterials[next(iter(LLMMaterials))]
                        sampleMaterial = sampleMaterials[next(iter(sampleMaterials))]
                        subTaskData['materialMatch'] = (LLMMaterial['youngsModulus'] == sampleMaterial['youngsModulus'] and
                                                        LLMMaterial['poissonsRatio'] == sampleMaterial['poissonsRatio'] and
                                                        LLMMaterial['density'] == sampleMaterial['density'])
                    # for the following checks, executability is necessary
                    if subTaskData['executableSample'] and subTaskData['executableLLM']:
                        # check if samplePart is a valid FEM object and do general checks for sample part
                        if not isinstance(samplePart, FEMinterface):
                            self.agent.logger.LogError('Sample part is NOT a valid FEM object!', 
                                                        printToConsole=True, separator=True, quickinfo=True)
                            raise Exception # this should NEVER happen!
                        if samplePart.NumberOfNodes() == 0:
                            self.agent.logger.LogError('Sample part does not hold any nodes!', 
                                                        printToConsole=True, separator=True, quickinfo=True)
                            raise Exception # this should NEVER happen!
                        if len(samplePart.GetEigenFrequenciesHz()) == 0:
                            self.agent.logger.LogError('Sample part does not hold eigenfrequencies!', 
                                                        printToConsole=True, separator=True, quickinfo=True)
                            raise Exception # this should NEVER happen!
                        if len(samplePart.surface) == 0:
                            self.agent.logger.LogError('Sample part has no surfaces!', 
                                                        printToConsole=True, separator=True, quickinfo=True)
                            raise Exception # this should NEVER happen!
                        # check if LLM Part is a valid fem object - do not raise exception in this case, just do not check
                        if isinstance(LLMPart, FEMinterface):
                            # no nodes means something went really wrong, do not compare further
                            if LLMPart.NumberOfNodes() > 0:
                                subTaskData['relativeMassDifference'] = RelativeMassDifference(LLMFemObject=LLMPart, sampleFemObject=samplePart)
                                if len(LLMPart.GetEigenFrequenciesHz()) == len(samplePart.GetEigenFrequenciesHz()):
                                    # if not nEigenmodesToCompare are present, just compare those that we have ...
                                    subTaskData['relativeEigenfrequenciesNormDifference'] = RelativeEigenfrequenciesNormDifference(LLMFemObject=LLMPart,
                                                                                                                                    sampleFemObject=samplePart,
                                                                                                                                    nEigenModesToCompare=inDict['nEigenModesToCompare'])
                                
                                if len(LLMPart.surface) > 0:
                                    LLMTriangles = np.asarray(LLMPart.surface[0]['Trigs'], dtype=int)
                                    LLMPoints = np.asarray(LLMPart.GetNodePositionsAsArray(), dtype=np.float64)
                                    sampleTriangles = np.asarray(samplePart.surface[0]['Trigs'], dtype=int)
                                    samplePoints = np.asarray(samplePart.GetNodePositionsAsArray(), dtype=np.float64)
                                    meshVolumeDifferenceDict = MeshVolumeDifference(triangles1=sampleTriangles,
                                                                                    points1=samplePoints,
                                                                                    triangles2=LLMTriangles,
                                                                                    points2=LLMPoints)    
                                    subTaskData['relativeDifferenceVolume'] = meshVolumeDifferenceDict['relativeDiff']
                                    subTaskData['referenceVolume'] = meshVolumeDifferenceDict['volume1']
                                # also check name of interfaces by making sorted list of strings (this can also be empty)
                                llmInterfaceNames = sorted(nodeSet['Name'] for nodeSet in LLMPart.GetDictionary()['nodeSets'])
                                sampleInterfaceNames = sorted(nodeSet['Name'] for nodeSet in samplePart.GetDictionary()['nodeSets'])
                                if llmInterfaceNames == sampleInterfaceNames:
                                    subTaskData['interfaceNameMatch'] = True

                    #++++++++++++++++++++++++++++++++++++++++++++++++++
                    # and save the part (if executable)
                    if subTaskData['executableLLM']:
                        partNameGivenByLLM = LLMPart.metaData['name']
                        # add description that lead to the femObject such that you can access it if needed
                        LLMPart.metaData['description'] = taskData['taskDescription']
                        # since the LLM must load this again, do not ID string here; otherwise add special string here
                        if 'superiorTaskNameWithID' not in taskData:
                            LLMPart.SaveToFile(saveDirectoryOfLLMPart + currentTaskNameWithID + '.npz')
                        else:
                            # here the name is important and a subfolder is already made so no issues; here it is very important that name assigned by LLM is used
                            LLMPart.SaveToFile(saveDirectoryOfLLMPart + partNameGivenByLLM + '.npz')
                        self.agent.logger.LogText('Saving part "' + currentTaskNameWithID + '" at "' + saveDirectoryOfLLMPart + '".', 
                                            printToConsole=False, separator=True, quickinfo=True)
                    self.agent.flexuAIPartsEvalTime += time.time() - startTime
                else:
                    self.agent.logger.LogDebug('Post-GenerateCodeTaskForFlexuAI: Specified task type is not available!',
                                                printToConsole=True, separator=True)
                    raise ValueError('Post-GenerateCodeTaskForFlexuAI: Specified task type is not available!')
            else:
                self.agent.logger.LogDebug('Post-GenerateCodeTaskForFlexuAI: Task type is not available in model descriptions!',
                                            printToConsole=True, separator=True)
                raise ValueError('Post-GenerateCodeTaskForFlexuAI:  Task type is not available in model descriptions!')
            # make this entry if we do not have it (preliminary tasks) or just write accordingly
            self.globalData['nExecutabilitySamples'] += int(subTaskData['executableSample']) # just add if executable
            self.globalData['nExecutabilityMainTasks'] += int(subTaskData['executableLLM']) # just add if executable
            self.globalData['nCompletedSamples'] += 1
            return True # success of subtask
        except RuntimeError as e:
            self.agent.logger.LogWarning(f'GenerateCodeTaskForFlexuAI: context window exceeded for "{currentTaskNameWithID}": {e}')
        except:
            self.agent.logger.LogError(f'Postprocessing failed: Model "{currentTaskNameWithID}", task: {self.taskName}')
            raise Exception

# wrapper function that calls chooseItems and createCode automatically
# agent is the agent that is used to process the tasks
# helperFileName references to the helperFile that is used to process the tasks
# taskName is the name of the task that is used to reference to the task
def FlexuAIChooseItemsAndGenerateCode(agent, globalData, helperFileName='exudyn', taskName='createFMBS',
                                specificHelperTagsList=[]):
    chooseItemsTask = ChooseItemsTaskForFlexuAI(globalData, agent.maxTokensChooseItems,
                                                helperFileName=helperFileName, agent=agent, 
                                                descriptiveName=taskName+'ChooseItems',
                                                specificHelperTagsList=specificHelperTagsList)
    currentIDs = chooseItemsTask.Run()
    generateCodeTask = GenerateCodeTaskForFlexuAI(globalData=globalData, helperFileName=helperFileName,
                                                  maxTokens=agent.maxTokensGenerateModel, agent=agent,
                                                  descriptiveName=taskName+'CreateCode')
    currentIDs = generateCodeTask.Run(currentIDs)
    return globalData
        
# callback function for InitializeTaskProcessing that adds the part creation task to the taskData
# partsLoader is the loader for the parts
# evaluationNameForPartsCreation is the evaluation name that is used
# with verbose, some additional info will be given in the console
def InitializationCallbackForPartsCreation(preliminaryTaskData, taskData, partLoader=None,
                                           evaluationTaskNameForPartCreation='generateCode', 
                                           logger=None,
                                           printToConsole=False):
    # extract, which parts are needed for the fMBS task
    fMBSParameters = taskData['chosenParameters']
    fMBSData = taskData['dataOfTask']
    partsDict = {}
    logger.LogText('Inherit data for "' + taskData['currentTaskNameWithID'] + '"', 
                    printToConsole=printToConsole, separator=True, quickinfo=True)
    for fMBSParameterKey, fMBSParameterValue in fMBSParameters.items():
        logger.LogText(f'Parameter {fMBSParameterKey:<30} {fMBSParameterValue}', 
                    printToConsole=printToConsole, separator=False, quickinfo=True)
        if fMBSParameterKey.startswith('part'):
            partsDict[fMBSParameterKey] = fMBSParameterValue.strip('{}')
    nParts = len(partsDict)
    logger.LogText(str(nParts) + ' part(s) needed', 
                    printToConsole=printToConsole, separator=True, quickinfo=True)
        
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # look, if there is meta information for the fMBS available
    if 'metaInformation' in fMBSData:
        fMBSMetaInformation = fMBSData['metaInformation']
        for partIndex in fMBSMetaInformation:
            if 'masterParameters' in fMBSData['metaInformation'][partIndex]:
                logger.LogText('Master parameters are found for "' + partsDict[partIndex] + '".', 
                    printToConsole=printToConsole, separator=True, quickinfo=True)
            else:
                logger.LogText('No master parameters are found for "' + partsDict[partIndex] + '".', 
                    printToConsole=printToConsole, separator=True, quickinfo=True)
    else:
        fMBSMetaInformation = {}
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # make an entry in the subtaskData dict for each part that should be created
    for partKey, partName in partsDict.items():
        logger.LogText('Part: ' + str(partName), 
                    printToConsole=printToConsole, separator=True, quickinfo=True)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # check whether for the corresponding part, master parameters are available
        masterParametersDict = {}
        if fMBSMetaInformation != {}:
            partMetaInformation = fMBSMetaInformation[partKey]
            if 'masterParameters' in partMetaInformation:
                for masterParameterKey in partMetaInformation['masterParameters']:
                    masterParametersDict[masterParameterKey] = fMBSParameters[masterParameterKey] # take the master parameters from the fMBS
        for masterParameterKey, masterParameterValue in masterParametersDict.items():
            logger.LogText(f'Master parameter {masterParameterKey:<30} {masterParameterValue}', 
                printToConsole=printToConsole, separator=True, quickinfo=True)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # save the information on this SINGLE part, use master parameters when loading the part
        partCreationData = {}
        preliminaryTaskData[partName] = partCreationData
        MakeTaskDataDict(partCreationData, taskLoader=partLoader, currentTaskName=partName,
                         currentTaskNameWithID=partName,
                         randomID=0, randomIDStr='', taskCount=0, spaceVariationID=0,
                         evaluationTaskName=evaluationTaskNameForPartCreation, randomizeParameters=True, # non-master parameters are randomized
                         masterParameters=masterParametersDict)
        partDescription = partCreationData['taskDescription']

        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # now also replace the part names in the fMBS description with the part description
        # with this, the fMBS description is finished
        fMBSDescription = taskData['taskDescription']
        taskData['taskDescription'] = fMBSDescription.replace('{' + partName + '}', partDescription)

def RestructureGlobalDataAndGetGlobalDataForPartCreation(globalData, logger:Logger):
    globalDataForPartCreation = {}
    partData = {}
    globalDataForPartCreation['storedDataPerTask'] = partData
    for taskNameWithID in globalData['storedDataPerTask']:
        if 'preliminaryTask' in globalData['storedDataPerTask'][taskNameWithID]:
            for partName in globalData['storedDataPerTask'][taskNameWithID]['preliminaryTask']:
                partNameWithID = partName
                counter = 0
                baseNameOfPart = partNameWithID
                while partNameWithID in partData:
                    partNameWithID = f'{baseNameOfPart}{counter}'
                    counter += 1
                # here we build the reference to the global data
                partData[partNameWithID] = globalData['storedDataPerTask'][taskNameWithID]['preliminaryTask'][partName]
                partData[partNameWithID]['superiorTaskNameWithID'] = taskNameWithID # here we say what the superior task is named
                partData[partNameWithID]['partName'] = partName # just the clear part name
    if globalDataForPartCreation['storedDataPerTask']:
            # add some specific info here such that it is also available in the reconstructured dict
            # these will be thrown away since they are not necessary, but for logging could be useful
            globalDataForPartCreation['solutionDir'] = globalData['solutionDir'] # will not be used anyway, but take global one
            globalDataForPartCreation['taskSolutionLLMDir'] = globalData['taskSolutionLLMDir'] # take global one
            globalDataForPartCreation['taskSolutionSampleDir'] = globalData['taskSolutionSampleDir'] # take global one
            globalDataForPartCreation['nExecutabilityMainTasks'] = 0 # do not reference from global data
            globalDataForPartCreation['nExecutabilitySamples'] = 0 # do not reference from global data
            globalDataForPartCreation['nCompletedSamples'] = 0 # do not reference from global data
            return globalDataForPartCreation
    else:
        return {}

def SkipFMBSDueToPartsNotBeingCreatedCorrectly(agent, logger, globalData={}, evaluationTaskNameForPartCreation=''):
    for taskNameWithID in globalData['storedDataPerTask']:
                globalData['storedDataPerTask'][taskNameWithID]['skipDueToPartsError'] = False # per default set False
                for partName in globalData['storedDataPerTask'][taskNameWithID]['preliminaryTask']:
                    overallSuccess = True
                    for metric in globalEvaluationMetrics['part']:
                        metricName = metric[0]
                        metricSuccessCondition = metric[1]['successIf']
                        result = globalData['storedDataPerTask'][taskNameWithID]['preliminaryTask'][partName][evaluationTaskNameForPartCreation][metricName]
                        if metricSuccessCondition == 'belowThreshold':
                            if result < agent.diffSolutionTolerance and result >= 0:
                                pass
                            else:
                                overallSuccess = False
                        if metricSuccessCondition == 'True':
                            if result == True:
                                pass
                            else:
                                overallSuccess = False
                        if isinstance(metricSuccessCondition, float):
                            if result < metricSuccessCondition and result >= 0:
                                pass
                            else:
                                overallSuccess = False
                    if not overallSuccess:
                        # skipDueToPartsError = True will be treated accordingly by the according FlexuAI functions now
                        globalData['storedDataPerTask'][taskNameWithID]['skipDueToPartsError'] = True
                        logger.LogText('Skipping fMBS "' + taskNameWithID + '" due to part "' + partName + '" (and maybe others) being not made correctly.', printToConsole=False, separator=True, quickinfo=True)
                        break
    return globalData