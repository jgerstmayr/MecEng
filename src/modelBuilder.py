#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: modelBuilder.py
#
# Details:  Task processing (item selection, code generation, evaluation) for MBS, flexible MBS
#           and part creation; called from modelBuilderDriver.py, not intended to be run directly.
#
# Authors:  Tobias Möltner, Johannes Gerstmayr, Sebastian Weyrer (ed.)
# Date:     2025-11-19
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%% import modules
import os
import numpy as np
import copy
import time
import datetime
import traceback
import exudyn as exu
from ExuAI.agent import LabAgent
from ExuAI.compareGraphs import GetMBSDifferences, GetCategorizedGraphErrorsAndWrongParametersList, errorCategories
from ExuAI.utilities import WriteFile, GetScalarClassObjectsDict, String2PythonComment, JoinPath,\
        DeleteFileIfExisting, EnsureEmptyDir, EvaluateNumerical, DynamicRound, \
        PreProcessLLMcode, CheckLLMoutput, ExtractElementsInResponse
scriptDir = os.path.dirname(os.path.abspath(__file__))
helperFilesDict = {'exudyn': JoinPath(scriptDir, 'ExuAI', 'exudynHelper.py'),
                   'flExudyn': JoinPath(scriptDir, 'flexu', 'flExudynHelper.py'),
                   'NetgenOCC': JoinPath(scriptDir, 'flexu', 'NetgenOCCHelper.py')}
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#%% define evaluation metrics that will be used throughout the whole script (GLOBAL)
# belowThreshold uses agent.diffSolutionTolerance; if number is specified, it uses this number
mbsEvaluationMetricsList = [['chooseItemsMatch', {'default': -1, 'successIf': 'True'}],
                            ['executableLLM', {'default': False, 'successIf': 'True'}],
                            ['reachedMbsAssemble', {'default': -1, 'successIf': 'True'}],
                            ['finishedMbsAssemble', {'default': -1, 'successIf': 'True'}],
                            ['solverInitSuccessful', {'default': -1, 'successIf': 'True'}],
                            ['solverSuccess', {'default': -1, 'successIf': 'True'}],
                            ['solutionFileDiff', {'default': -3, 'successIf': 'belowThreshold'}],
                            ['graphNodesCountMatch', {'default': -1, 'successIf': 'True'}],
                            ['graphMatch', {'default': -1, 'successIf': 'True'}],
                            ['graphMatchContent', {'default': -1, 'successIf': 'True'}],
                            ['graphsIsomorphicNoTimeout', {'default': True, 'successIf': 'True'}],
                            ['solverNoTimeoutSample', {'default': True, 'successIf': 'True'}],
                            ['solverNoTimeoutLLM', {'default': True, 'successIf': 'True'}],
                            ['llmNoContextWindowExceeded', {'default': True, 'successIf': 'True'}],
                            ]

globalEvaluationMetrics = {'mbs': mbsEvaluationMetricsList,
                            'fMBS': mbsEvaluationMetricsList,
                            'part': [['chooseItemsMatch', {'default': -1, 'successIf': 'True'}],
                                     ['executableLLM', {'default': False, 'successIf': 'True'}],
                                     ['materialMatch', {'default': False, 'successIf': 'True'}],
                                     ['relativeMassDifference', {'default': -3, 'successIf': 1e-2}], # rel. diff 1%
                                     ['relativeEigenfrequenciesNormDifference', {'default': -3, 'successIf': 5e-2}], # rel. diff 5%
                                     ['relativeDifferenceVolume', {'default': -3, 'successIf': 1e-2}], # rel. diff 1%
                                     ['interfaceNameMatch', {'default': False, 'successIf': 'True'}]
                                     ]}

#%%
#a generic processor for subtasks; operates on single subtasks (task=subtask0+subtask1+...)
#automated batched processing for LLMs in MainTask
#taskNam: name that is used in subdict (e.g. chooseItems)
#globalData: the global dict with 'storedDataPerTask' containing the data per task
#in PreSubtask, the LLM-prompt must be stored in taskData[self.taskName]['prompt']
#in PostSubtask, the LLM-response can be read from taskData[self.taskName]['response']
#   NOTE: the response may be None in case it failed!
class TaskProcessor:
    def __init__(self, globalData, maxTokens, helperFileName='exudyn',
                 agent:LabAgent=None, descriptiveName = '', skipBatchedInference=False):
        self.taskName = self.GetTaskName()
        self.descriptiveName = descriptiveName if descriptiveName != '' else self.taskName
        self.globalData = globalData
        self.agent = agent
        self.storedDataPerTask = globalData['storedDataPerTask']
        self.maxTokens = maxTokens
        self.helperFileName = helperFileName
        self.skipBatchedInference = skipBatchedInference

    #OVERRIDE:
    def GetTaskName(self):
        raise ValueError('TaskProcessor: '+self.taskName+': PreSubtask has to be overwritten in derived class!') 
        return 'none'

    #run all subtasks with pre- and postprocessing as well as (batched) inference
    #tasksIDList can be used to run only a subset of tasks (if other failed already)
    #returns current taskID list with successful IDs
    def Run(self, tasksIDList = None):
        self.agent.logger.LogText(f"TaskProcessor: START task: {self.taskName}", separator=True, printToConsole=True)
        if tasksIDList is None: #initialize if not provided (usually first Subtask)
            self.currentTasksIDList = []
            for currentMBSmodelNameID, taskData in self.storedDataPerTask.items():
                self.currentTasksIDList.append(currentMBSmodelNameID)
            
        else:
            self.currentTasksIDList = tasksIDList

        self.agent.logger.LogText(f"TaskProcessor: Start PreProcessing: {self.taskName}", printToConsole=True)
        self.PreProcessing()
        
        self.agent.logger.LogText(f"TaskProcessor: Start BatchedInference: {self.taskName}", printToConsole=True)
        if not self.skipBatchedInference:
            self.BatchedInference()

        self.agent.logger.LogText(f"TaskProcessor: Start PostProcessing: {self.taskName}", printToConsole=True)

        self.PostProcessing()

        self.agent.logger.LogText(f"TaskProcessor: END task: {self.taskName}", printToConsole=True)
        self.agent.logger.LogText("========================================",printToConsole=True,timestamp=False)
        
        return self.currentTasksIDList
    
    #OVERRIDE! a single pre-subtask (per taskData); must define LLM-prompt in subTaskData 'prompt'
    def PreSubtask(self, subTaskData, taskData, currentMBSmodelNameID, currentMBSmodelName):
        raise ValueError('TaskProcessor: '+self.taskName+': PreSubtask has to be overwritten in derived class!')
        return False #false indicates that subtask failed!

    #run preprocessing for all subtasks
    def PreProcessing(self):
        nextTaskList = []
        for ID, currentMBSmodelNameID in enumerate(self.currentTasksIDList):
            taskData = self.storedDataPerTask[currentMBSmodelNameID]
            try:
                currentMBSmodelName = taskData['currentTaskName']
                if self.taskName not in taskData:
                    taskData[self.taskName] = {} #each subtask data gets new sub-dict with specific subtask info, if it does not exist already
                taskData[self.taskName]['success'] = False
                if self.PreSubtask(taskData[self.taskName], taskData, currentMBSmodelNameID, currentMBSmodelName):
                    nextTaskList.append(currentMBSmodelNameID)
            except KeyboardInterrupt:
                print("Keyboard interrupt requested...exiting"); raise
            except:
                exceptionText = traceback.format_exc()
                self.agent.logger.LogError(f'TaskProcessor.PreProcessing failed: model "{currentMBSmodelNameID}", task: {self.taskName}; exception: {exceptionText}')
        self.currentTasksIDList = nextTaskList

    #OVERRIDE! a single post-subtask (per taskData); usually processing LLM-response; 
    def PostSubtask(self, subTaskData, taskData, currentMBSmodelNameID, currentMBSmodelName):
        raise ValueError('TaskProcessor: '+self.taskName+': PostSubtask has to be overwritten in derived class!')
        return False #false indicates that subtask failed!

    #run preprocessing for all subtasks
    def PostProcessing(self):
        nextTaskList = []
        for ID, currentMBSmodelNameID in enumerate(self.currentTasksIDList):
            taskData = self.storedDataPerTask[currentMBSmodelNameID]
            try:
                currentMBSmodelName = taskData['currentTaskName']
                if self.PostSubtask(taskData[self.taskName], taskData, currentMBSmodelNameID, currentMBSmodelName):
                    nextTaskList.append(currentMBSmodelNameID)
                    taskData[self.taskName]['success'] = True
            except KeyboardInterrupt:
                print("Keyboard interrupt requested...exiting"); raise
            except:
                exceptionText = traceback.format_exc()
                self.agent.logger.LogError(f'TaskProcessor.PostProcessing failed: model "{currentMBSmodelNameID}", task: {self.taskName}; exception: {exceptionText}')
        self.currentTasksIDList = nextTaskList
    
    #run batched inference for all LLM-Subtasks
    def BatchedInference(self):
        #create list of prompts and task names for batched inference:
        self.currentPromptList = []
        self.currentTaskNamesList = []
        usePromptLists = False #set if lists show up
        subListIDs = [] #per task, sublist of ID of prompts/responses

        #for each active task, create prompt and task name    
        responseID = 0
        for ID, currentMBSmodelNameID in enumerate(self.currentTasksIDList):
            taskData = self.storedDataPerTask[currentMBSmodelNameID]
            taskPrompt = taskData[self.taskName]['prompt']
            if isinstance(taskPrompt, str):
                self.currentPromptList.append(taskPrompt)
                self.currentTaskNamesList.append(f'{self.taskName} for {currentMBSmodelNameID}')
                subListIDs.append([responseID])
                responseID += 1
            else:
                nPrompts = len(taskPrompt)
                localIDs = []
                for i in range(nPrompts):
                    self.currentPromptList.append(taskPrompt[i])
                    self.currentTaskNamesList.append(f'{self.taskName} for {currentMBSmodelNameID}-sub{i}')
                    localIDs.append(responseID)
                    responseID += 1
                subListIDs.append(localIDs)
                usePromptLists = True

        tStartBatchedItems = time.time()
        self.agent.logger.LogText('Start generation for task "'+self.descriptiveName+'"')
        self.agent.logger.LogText('number of batches for '+self.taskName+f'-prompt: {len(self.currentPromptList)}',
                                  printToConsole=True, timestamp=False)
        responsesList = self.agent.llmModel.Generate(prompt=self.currentPromptList, systemPrompt=None, 
                                                max_tokens=self.maxTokens,
                                                taskNamesList=self.currentTaskNamesList,
                                                **self.agent.llmGenerateDefaultArgs)
        tBatchedItems = time.time()-tStartBatchedItems
        self.agent.logger.LogText(f'TaskProcessor: End BatchedInference: {self.taskName}: needed {round(tBatchedItems,2)} seconds',
                                  printToConsole=True, timestamp=False)
        self.globalData[self.taskName+'Duration'] = tBatchedItems

        contextWindowExceededList = getattr(self.agent.llmModel, 'contextWindowExceededList', [])
        contextWindowExceededTokensList = getattr(self.agent.llmModel, 'contextWindowExceededTokensList', [])
        contextSizeUsedList = getattr(self.agent.llmModel, 'contextSizeUsedList', [])
        numberOfPromptTokensList = getattr(self.agent.llmModel, 'numberOfPromptTokensList', [])

        for ID, currentMBSmodelNameID in enumerate(self.currentTasksIDList):
            taskData = self.storedDataPerTask[currentMBSmodelNameID]
            if not usePromptLists:
                taskData[self.taskName]['response'] = responsesList[ID]
                taskData[self.taskName]['contextWindowExceeded'] = contextWindowExceededList[ID] if ID < len(contextWindowExceededList) else False
                taskData[self.taskName]['contextWindowExceededTokens'] = contextWindowExceededTokensList[ID] if ID < len(contextWindowExceededTokensList) else 0
                taskData[self.taskName]['contextSizeUsed'] = contextSizeUsedList[ID] if ID < len(contextSizeUsedList) else 0
                taskData[self.taskName]['numberOfPromptTokens'] = numberOfPromptTokensList[ID] if ID < len(numberOfPromptTokensList) else 0
            else:
                localResponses = []
                localCWE = []
                localCWEtokens = []
                localContextSizeUsed = []
                localPromptTokens = []
                for subID in subListIDs[ID]:
                    localResponses.append(responsesList[subID])
                    localCWE.append(contextWindowExceededList[subID] if subID < len(contextWindowExceededList) else False)
                    localCWEtokens.append(contextWindowExceededTokensList[subID] if subID < len(contextWindowExceededTokensList) else 0)
                    localContextSizeUsed.append(contextSizeUsedList[subID] if subID < len(contextSizeUsedList) else 0)
                    localPromptTokens.append(numberOfPromptTokensList[subID] if subID < len(numberOfPromptTokensList) else 0)

                taskData[self.taskName]['response'] = localResponses #list
                taskData[self.taskName]['contextWindowExceeded'] = any(localCWE)
                taskData[self.taskName]['contextWindowExceededTokens'] = max(localCWEtokens) if localCWEtokens else 0
                taskData[self.taskName]['contextSizeUsed'] = max(localContextSizeUsed) if localContextSizeUsed else 0
                taskData[self.taskName]['numberOfPromptTokens'] = max(localPromptTokens) if localPromptTokens else 0
                

        self.globalData['numberOfRemovedTokensGlobal'] = self.agent.llmModel.numberOfRemovedTokensGlobal
        self.globalData['numberOfTokensGlobal'] = self.agent.llmModel.numberOfTokensGlobal
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Item selection task
class ChooseItemsTask(TaskProcessor):
    def __init__(self, globalData, maxTokes, helperFileName='',
                     agent:LabAgent=None, descriptiveName='', skipBatchedInference=False,
                     specificHelperTagsList=[], promptVariation=0, listVariation=0):
        # enable passing of specific helper tags to circumvent choosing if i.e. only one is available
        super().__init__(globalData, maxTokes, helperFileName=helperFileName,
                         agent=agent, descriptiveName=descriptiveName, skipBatchedInference=skipBatchedInference)
        self.specificHelperTagsList = specificHelperTagsList
        self.promptVariation = promptVariation
        self.listVariation = listVariation
        if self.specificHelperTagsList != []:
            self.skipBatchedInference = True
                
    def GetTaskName(self):
        return 'chooseItems'

    #create single prompt in subTaskData
    def PreSubtask(self, subTaskData, taskData, currentMBSmodelNameID, currentMBSmodelName):

        #create prompt
        promptTemplate = None #default
        if self.promptVariation > 0:
            from ExuAI.templatesAndLists import simpleMBSChooseItemsPromptVariations
            promptTemplate = simpleMBSChooseItemsPromptVariations[self.promptVariation - 1]
        subTaskData['prompt'] = self.agent.GetItemSelectionPrompt(taskData['taskDescription'], 
                                                                  helperFileName=self.helperFileName,
                                                                  promptTemplate=promptTemplate,
                                                                  listVariation=self.listVariation)
        return True #success of subtask

    #check if there are helperTags that haven't been found by LLM
    def CheckHelperTags(self, subTaskData, taskData, currentMBSmodelNameID, currentMBSmodelName, 
                        parsedHelperDict, extractedHelperTags):
        sampleFileDict = self.agent.GetSampleFileDict(filePath=GetSampleFileDir(taskData),
                                                 mbsModelName=currentMBSmodelName,
                                                 sampleDict=taskData['dataOfTask'], 
                                                 chosenParametersDict=taskData['chosenParameters']) 
        modelCode = sampleFileDict['modelCode']
        subTaskData['chooseItemsMatch'] = False
        subTaskData['chooseItemsSurplusTags'] = []
        subTaskData['chooseItemsMissingTags'] = []
        self.chooseItemsMatchFailedTotal += 1 #failed per default

        try:
            sampleTags = []
            for key, value in parsedHelperDict.items():
                if 'USES' in value and value['USES'].strip() != '':
                    keywordsUsed = value['USES'].split(',')
                    if all(keyword.strip() in modelCode for keyword in keywordsUsed):
                        sampleTags.append(key)
            # self.agent.logger.LogDebug(f"ERROR: ChooseItemsTask: {currentMBSmodelNameID}: sample tags = {sampleTags}",
            #                            printToConsole=True)
            #if overlapping tags exist (where the first one has a subset of "USES" of the second one), delete the simpler (FIRST) one:
            # overlapping tags only change the sample tags (if the sample already demands the harder tag, don require the easier one)
            # sample tags are the strictly REQUIRED tags and there can be more chosen by the LLM which is not an error
            # removing the easier tag means that we do not punish the LLM if the easier tag is missing BUT the harder MUST be there!
            overlappingTags = [
                ['force for point mass','force with time-dependent user function'],
                ['force for point mass','force for body'],
                ['force for body','force with time-dependent user function'],
                ['torque','torque with user function'],
                ['rigid body cylinder','rolling disc with penalty-based joint'],
                ['rigid body cylinder','ideal rolling disc joint'],
                ]
            equivalentTagsForFlexuAIList = [['box', 'extrusion'],
                                            ['cylinder', 'extrusion']]
            
            for oTag in overlappingTags:
                if oTag[0] in sampleTags and oTag[1] in sampleTags:
                    sampleTags.remove(oTag[0])
            missingTags = [] # initialize empty
            if not set(sampleTags).issubset(set(extractedHelperTags)):
                missingTags = list(set(sampleTags) - set(extractedHelperTags))
                # since there are some queivalent tags, check if there is a tag chosen that is equivalent to a missing one
                # e.g. cylinder and extrusion
                if taskData['taskType'] == 'part':
                    tagsWithoutEquivalentInExtractedTags = []
                    for missingTag in missingTags:
                        resolved = False
                        for equivalentGroup in equivalentTagsForFlexuAIList:
                            if missingTag in equivalentGroup: # check if the missing tag is in an equivalent group
                                for equivalentTag in equivalentGroup:
                                    if equivalentTag in extractedHelperTags: # we have an equivalent tag in the selected helper tags ...
                                        resolved = True
                                        self.agent.logger.LogDebug(f"NOTE: ChooseItemsTask: {currentMBSmodelNameID}: Removed {missingTag} from missing items because {equivalentTag} is present!", printToConsole=True)
                                        break
                            if resolved: # now also break this for loop and check for the next missing tag if there is an equivalent
                                break
                        if not resolved: # if we could not resolve so far, it is really missing
                            tagsWithoutEquivalentInExtractedTags.append(missingTag)
                    missingTags = tagsWithoutEquivalentInExtractedTags
            if missingTags: # tags are missing and list is NOT empty (check it AFTER possibly searching for equivalent tags)
                self.agent.logger.LogDebug(f"ERROR: ChooseItemsTask: {currentMBSmodelNameID}: ExtractElementsInResponse: MISSING tags = {missingTags}",
                                           printToConsole=True)
                subTaskData['chooseItemsMissingTags'] = missingTags
            else:
                subTaskData['chooseItemsMatch'] = True
                self.chooseItemsMatchFailedTotal -= 1 #set back failed
    
                            
            #also check if there are additional tags that are not needed:
            surplusTags = list(set(extractedHelperTags) - set(sampleTags))
            if len(surplusTags):
                self.agent.logger.LogDebug(f"ERROR: ChooseItemsTask: {currentMBSmodelNameID}: ExtractElementsInResponse: SURPLUS tags = {surplusTags}",
                                           printToConsole=False)
                subTaskData['chooseItemsSurplusTags'] = surplusTags
        except Exception:
            self.agent.logger.LogError(f'GenerateMBSTask CheckHelperTags failed: model "{currentMBSmodelNameID}", task: {self.taskName}')
            

    #process LLM-response and create tags and success;
    #if successful, set subTaskData['success'] = True
    def PostSubtask(self, subTaskData, taskData, currentMBSmodelNameID, currentMBSmodelName):
        if not self.specificHelperTagsList:
            response = CheckLLMoutput(self.agent.logger, subTaskData['response'], checkUndesiredKeywords=True, checkHarmfulCommands=True)
            
            failed = response is None
            
            if not failed: #response could be None or extraction could fail (list=None)
                helperTagsDict = ExtractElementsInResponse(response, self.agent.parsedHelperDict[self.helperFileName])
                helperTags = helperTagsDict['list']
                failed = helperTags is None
                if helperTags is None:
                    self.agent.logger.LogDebug(f"ERROR: ChooseItemsTask: {currentMBSmodelNameID}: ExtractElementsInResponse: no list found! message={helperTagsDict['message']}",
                                          printToConsole=True)
                else:
                    subTaskData['helperTags'] = helperTags
                    self.agent.logger.LogDebug(f'ChooseItemsTask: {currentMBSmodelNameID}: extracted tags='+str(helperTags))
                    self.CheckHelperTags(subTaskData=subTaskData, taskData=taskData, 
                                         currentMBSmodelNameID=currentMBSmodelNameID, currentMBSmodelName=currentMBSmodelName, 
                                         parsedHelperDict=self.agent.parsedHelperDict[self.helperFileName], 
                                         extractedHelperTags=helperTags)

                    return True #success of subtask
        else:
            self.agent.logger.LogDebug(f'ChooseItemsTask: {currentMBSmodelNameID}: Specific helper tag(s) is/are given; will use it/them.',
                                       printToConsole=False)
            self.agent.logger.LogDebug(f'ChooseItemsTask: {currentMBSmodelNameID}: extracted tags='+str(self.specificHelperTagsList)) 
            subTaskData['helperTags'] = self.specificHelperTagsList
            return True
        return False #failed

    #add printing overall results of ChooseItemsTask
    def PostProcessing(self):
            if not self.specificHelperTagsList:
                self.chooseItemsMatchFailedTotal = 0 #just for printing overview at end of ChooseItemsTask
            #execute default functionality:
            super().PostProcessing()
            if not self.specificHelperTagsList:
                self.agent.logger.LogDebug(f'ChooseItemsTask SUMMARY: total number of FAILED chooseItems = {self.chooseItemsMatchFailedTotal}',
                                        printToConsole=True)
            
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% MBS generation task
class GenerateMBSTask(TaskProcessor):
    def __init__(self, globalData, maxTokens, helperFileName='',
                     agent:LabAgent=None, descriptiveName='', skipBatchedInference=False,
                     promptVariation=0):
        super().__init__(globalData, maxTokens, helperFileName=helperFileName,
                         agent=agent, descriptiveName=descriptiveName, skipBatchedInference=skipBatchedInference)
        self.promptVariation = promptVariation

    def GetTaskName(self):
        return 'generateMBS'
    #create single prompt in subTaskData
    def PreSubtask(self, subTaskData, taskData, currentMBSmodelNameID, currentMBSmodelName):
        helperTags = taskData[ChooseItemsTask.GetTaskName(None)]['helperTags']
        
        #map chooseItemsMatch from chooseItems to generateMBS
        subTaskData['chooseItemsMatch'] = taskData['chooseItems']['chooseItemsMatch']

        # Generate Exudyn model
        mbsModelDescription=taskData['taskDescription']

        # construct prompt including code, LLM selected items with according standard parameters
        promptTemplate = None #default
        if self.promptVariation > 0:
            from ExuAI.templatesAndLists import simpleMBSGenerateCodePromptVariations
            promptTemplate = simpleMBSGenerateCodePromptVariations[self.promptVariation - 1]
        [prompt] = self.agent.GetCodeGenerationPrompt(helperTags, mbsModelDescription, helperFileName=self.helperFileName,
                                                      promptTemplate=promptTemplate)
        subTaskData['prompt'] = prompt
        return True #success of subtask

    #process LLM-response; create Exudyn code and run code
    #if successful, set subTaskData['success'] = True
    def PostSubtask(self, subTaskData, taskData, currentMBSmodelNameID, currentMBSmodelName):
        try: #we like to print in any case
            self.agent.logger.LogDebug(f"GenerateMBSTask: PostSubtask: CheckLLMoutput for {currentMBSmodelNameID}")
            # Set context window exceeded flag and token count from inference (stored in BatchedInference)
            contextWindowExceeded = subTaskData.get('contextWindowExceeded', False)
            contextWindowExceededTokens = subTaskData.get('contextWindowExceededTokens', 0)
            subTaskData['llmNoContextWindowExceeded'] = not contextWindowExceeded
            subTaskData['contextWindowExceededTokens'] = contextWindowExceededTokens
            subTaskData['contextSizeUsed'] = subTaskData.get('contextSizeUsed', 0)
            subTaskData['numberOfPromptTokens'] = subTaskData.get('numberOfPromptTokens', 0)
            if contextWindowExceeded:
                raise RuntimeError(
                    f"Context window exceeded by {contextWindowExceededTokens} tokens for task '{currentMBSmodelNameID}'")
            response = CheckLLMoutput(self.agent.logger, subTaskData['response'], checkUndesiredKeywords=True, 
                                      checkHarmfulCommands=True)
            
            self.agent.logger.LogDebug(f"GenerateMBSTask: PostSubtask: PreProcessLLMcode for {currentMBSmodelNameID}")
            #clean up code, as it will be used in the eval prompt:
            exudynCode = PreProcessLLMcode(self.agent.logger, response, addTimeout = True, addFileWrite = False,
                                           solverTimeoutSeconds=self.agent.solverTimeoutSeconds)
            
            if exudynCode is None:
                self.agent.logger.LogDebug(f"GenerateMBSTask: PostSubtask: CODE IS NONE for {currentMBSmodelNameID}")
                return False
            else:
            
                self.agent.logger.LogDebug('cleaned-up code of model:\n'+exudynCode) #check if cleanup worked
                self.agent.totalNumberOfCodesGenerated += 1
        
                #++++++++++++++++++++
                subTaskData['solutionNameLLM'] = JoinPath(self.globalData['solutionDir'],currentMBSmodelNameID + 'LLM.txt')
                subTaskData['solutionNameSample'] = JoinPath(self.globalData['solutionDir'],currentMBSmodelName + 'Sample.txt') #will change over time
                
        
                sampleFileDict = self.agent.GetSampleFileDict(filePath=JoinPath(scriptDir, "sampleFiles"),
                                                         mbsModelName=currentMBSmodelName,
                                                         sampleDict=taskData['dataOfTask'], 
                                                         chosenParametersDict=taskData['chosenParameters']) 
                solverCode = sampleFileDict["solverCode"]
        
                sampleCode = sampleFileDict['executableCode']
                subTaskData['sampleCode'] = sampleFileDict['executableCodeShort']
                
                #replace solver part in LLM-generated model:
                exudynCodeClean = exudynCode.split('\nmbs.Assemble()')[0]+'\n\n'
                exudynCodeClean += solverCode
                exudynCodeClean = exudynCodeClean.replace('mbs.Assemble()',
                                                          f'mbs.sys["{self.agent.preAssembleFlag}"]=True\n'+
                                                          'mbs.Assemble()\n'+
                                                          f'mbs.sys["{self.agent.postAssembleFlag}"]=True')
                
                mbsModelDescriptionPython = '# ** given model description: **\n'
                mbsModelDescriptionPython += String2PythonComment(taskData['taskDescription'])
                
                WriteFile(currentMBSmodelName+taskData['randomIDstr']+'.py', 
                          mbsModelDescriptionPython+exudynCodeClean, #add description to simplify manual error checking
                          path=self.globalData['taskSolutionLLMDir'], flush=True)
                
                # also save the sample file for comparison
                WriteFile(currentMBSmodelName+taskData['randomIDstr']+'_sample.py', 
                          sampleCode, #no description in sample code added
                          path=self.globalData['taskSolutionSampleDir'], flush=True)
                
                subTaskData['exudynCodeClean'] = exudynCodeClean
                
                self.agent.logger.LogDebug(f"GenerateMBSTask: PostSubtask: EvaluateMBScode (sample) for {currentMBSmodelNameID}")
                sampleCodeDict = self.agent.EvaluateMBScode(
                                                        sampleCode, 
                                                        currentMBSmodelNameID,
                                                        solutionFileName=subTaskData['solutionNameSample'], 
                                                        mode='Sample file based',
                                                        printErrorToConsole=False)
                subTaskData['executableSample'] = sampleCodeDict['executable']
                localVariablesSample = sampleCodeDict['localVariables']
                solverSuccessSample = sampleCodeDict['solverSuccess']
                subTaskData['solverNoTimeoutSample'] = sampleCodeDict['solverNoTimeout']
                
                self.agent.logger.LogDebug(f"GenerateMBSTask: PostSubtask: EvaluateMBScode for {currentMBSmodelNameID}")
                llmCodeDict = self.agent.EvaluateMBScode(
                                                    exudynCodeClean,
                                                    currentMBSmodelNameID,
                                                    solutionFileName=subTaskData['solutionNameLLM'], 
                                                    mode='LLM generated',
                                                    printErrorToConsole=False)
                subTaskData['executableLLM'] = llmCodeDict['executable']
                subTaskData['solverInitSuccessful'] = llmCodeDict['solverInitSuccessful']
                subTaskData['solverSuccess'] = llmCodeDict['solverSuccess'] 
                subTaskData['reachedMbsAssemble'] = llmCodeDict['reachedMbsAssemble']
                subTaskData['finishedMbsAssemble'] = llmCodeDict['finishedMbsAssemble']
                subTaskData['solverNoTimeoutLLM'] = llmCodeDict['solverNoTimeout']
                
                self.globalData['nCompletedSamples'] += 1
                localVariablesLLM = llmCodeDict['localVariables']
                solverSuccessLLM = llmCodeDict['solverSuccess']
                # even before doing numerical evaluation, we can check if mbs match
                CompareLLMAndSampleMBSNonNumerical(self.agent, subTaskData, taskData, currentMBSmodelNameID,
                                                   localVariablesLLM=localVariablesLLM, localVariablesSample=localVariablesSample,
                                                   taskName=self.taskName)
                # check if simulation worked - not necessary if the two following succeds
                if (subTaskData['executableSample'] and subTaskData['executableLLM'] and solverSuccessSample and solverSuccessLLM):
                    # simulation worked, compare with reference coordinatesSolution.txt
        
                    subTaskData['solutionFileDiff'] = EvaluateNumerical(fileNameReference=subTaskData['solutionNameSample'],
                                            fileNameEvaluate=subTaskData['solutionNameLLM'],
                                            localVariablesReference=localVariablesSample,
                                            localVariablesEvaluate=localVariablesLLM,
                                            logger=self.agent.logger)
    
                #update globalData and print
                                
                self.globalData['nExecutabilitySamples'] += int(subTaskData['executableSample'])
                self.globalData['nExecutabilityMainTasks'] += int(subTaskData['executableLLM'])
    
                self.agent.logger.LogDebug(f"GenerateMBSTask: finished PostSubtask for {currentMBSmodelNameID}")
                
                return True #success of subtask

        except RuntimeError as e:
            self.agent.logger.LogWarning(f'GenerateMBSTask: context window exceeded for model "{currentMBSmodelNameID}": {e}')
        except SystemExit:
            self.agent.logger.LogError(f'GenerateMBSTask Postprocessing failed (SystemExit): model "{currentMBSmodelNameID}", task: {self.taskName}')
            #do we like to see that in console?
            self.agent.logger.LogText(' - model '+currentMBSmodelNameID+':'+
                           ' - executable='+ str(subTaskData['executableLLM'])+
                           ', diff='+ str(DynamicRound(subTaskData['solutionFileDiff'],6)), 
                           printToConsole=True, separator=True, quickinfo=True)
        except Exception:
            self.agent.logger.LogError(f'GenerateMBSTask Postprocessing failed: model "{currentMBSmodelNameID}", task: {self.taskName}')
            #do we like to see that in console?
            self.agent.logger.LogText(' - model '+currentMBSmodelNameID+':'+
                           ' - executable='+ str(subTaskData['executableLLM'])+
                           ', diff='+ str(DynamicRound(subTaskData['solutionFileDiff'],6)), 
                           printToConsole=True, separator=True, quickinfo=True)
            raise

        return False #success of subtask (anyway not reached!)

# helper function that compares two mbs (available in local variables)
# this helper function is used for 'normal' pipeline and FlexuAI
def CompareLLMAndSampleMBSNonNumerical(agent, subTaskData, taskData, currentMBSModelNameID, localVariablesSample=None, localVariablesLLM=None, taskName=''):
    if 'mbs' in localVariablesSample and 'mbs' in localVariablesLLM:
        mbsLLM = localVariablesLLM['mbs']
        mbsSample = localVariablesSample['mbs']
        if isinstance(mbsLLM, exu.MainSystem):
            try:
                if mbsLLM.systemIsConsistent:
                    skipMeshDependentQuantitiesInFFRFObjects = False # which is default
                    if 'taskType' in taskData:
                        if taskData['taskType'] == 'fMBS':
                            skipMeshDependentQuantitiesInFFRFObjects = True
                    mbsDiffDict = GetMBSDifferences(mbsReference=mbsSample, mbsCompare=mbsLLM,
                                                    logger=agent.logger, skipMeshDependentQuantitiesInFFRFObjects=skipMeshDependentQuantitiesInFFRFObjects,
                                                    model=currentMBSModelNameID)
                    subTaskData['graphMatch'] = mbsDiffDict['graphsMatch']
                    subTaskData['graphDiffDict'] = mbsDiffDict
                    subTaskData['graphNumberOfNodes'] = mbsDiffDict.get('numberOfGraphNodes','N/A')
                    subTaskData['graphNumberOfEdges'] = mbsDiffDict.get('numberOfGraphEdges','N/A')
                    subTaskData['graphNodesCountMatch'] = mbsDiffDict.get('graphNodesCountMatch','N/A')
                    subTaskData['graphsIsomorphicNoTimeout'] = mbsDiffDict.get('graphsIsomorphicNoTimeout','N/A')
                    if mbsDiffDict['graphsMatch']:
                        subTaskData['graphMatchContent'] = mbsDiffDict['graphsMatchContent']
                        subTaskData['graphsReachedMaxIterationsToFindMatch'] = mbsDiffDict['reachedMaxIterations']
                        subTaskData['graphsAnnotationIterations'] = mbsDiffDict['annotationIterations']
                    agent.logger.LogDict(mbsDiffDict, header='MBS Difference Dict')
                else:
                    agent.logger.LogError(f'GenerateMBSTask Postprocessing: model "{currentMBSModelNameID}", task: {taskName}: mbs is not consistent (graph cannot be compared)!', 
                                          printToConsole=False)

            except Exception as e:
                exceptionText = traceback.format_exc()
                agent.logger.LogError(f'GenerateMBSTask GetMBSDifferences failed: model "{currentMBSModelNameID}", task: {taskName}:\n    {e}')
                agent.logger.LogError(f'exception traceback:\n{exceptionText}')
        else:
            agent.logger.LogError('GenerateMBSTask Postprocessing: mbs is no exu.MainSystem!',printToConsole=True)
        
# standardized task layout since each task type has same layout
# this function AUTOMATICALLY adds metric depending on the task type
# this is a helper function that assembles a dict for ONE task
# the helper function thus ensures that the dict always has the same structure
# taskLoader is the loader used to load parameterized models
# currentTaskName is the name of the current task
# currentTaskNameWithID is the name of the task with a variation ID
# randomID is the integer value of the randomization
# randomIDStr is a string for the randomization
# taskCount is a counter to enumerate all the tasks
# spaceVariationID is an ID for the spaceVariation
# evaluationTaskName gives the evaluation a standard name
def MakeTaskDataDict(taskData, taskLoader=None, currentTaskName='', currentTaskNameWithID='',
                     randomID=0, randomIDStr='', taskCount=0, spaceVariationID=0,
                     evaluationTaskName='', randomizeParameters=True, masterParameters={}):
    # general info about the task to be done
    taskData['currentTaskName'] = currentTaskName
    taskData['currentTaskNameWithID'] = currentTaskNameWithID
    taskData['randomID'] = randomID
    taskData['randomIDstr'] = randomIDStr
    taskData['taskCount'] = taskCount
    taskData['spaceVariationID'] = spaceVariationID
    # special info about the task to be done, including description and taskType
    modelDict = taskLoader.GetModelDescriptionAndDict(currentTaskName, 
                                                       randomizeParameters=randomizeParameters,
                                                       spaceVariationID=spaceVariationID,
                                                       masterParameters=masterParameters)
    taskData['taskDescription'] = copy.copy(modelDict['modelDescription'])
    taskData['dataOfTask'] = copy.copy(modelDict['modelData'])
    taskData['chosenParameters'] = copy.copy(modelDict['chosenParameters'])
    if 'taskType' in modelDict['modelData']:
        taskData['taskType'] = copy.copy(modelDict['modelData']['taskType'])
    else:
        taskData['taskType'] = 'mbs' # which is the default setting
    # depending on the task type, set metrics
    taskData[evaluationTaskName] = {}
    for metric in globalEvaluationMetrics[taskData['taskType']]:
        metricName = metric[0]
        metricDefaultValue = metric[1]['default']
        taskData[evaluationTaskName][metricName] = metricDefaultValue
    # some metrics are ALWAYS written in the evaluation stage
    taskData[evaluationTaskName]['success'] = False # general success flag of processing
    taskData[evaluationTaskName]['exuAI_in'] = {} # a special dict that contains information passed to the LLM generated code for global setting
    taskData[evaluationTaskName]['exuAI_out'] = {} # a special dict that contains (selected) information taken out of the LLM generated code
    taskData[evaluationTaskName]['executableSample'] = False # default, if it fails earlier (always tracked)

# doSpaceVariation: if non-zero, it variates spaces, using increasing ID for variation
# CallbackForPreliminaryTask a callback here can simply add subtasks to a task, i.e. creating parts for MBSs
# this callback must receive as first positional argument a dictionary, in which the data for the preliminary task will be stored
# this callback must receive as second positional argument the taskData dictionary, in which data of the superior task can be accessed (if there are dependencies)
def InitializeTaskProcessing(agent, mbsModelLoader, numberOfRandomizations=0, doSpaceVariation=0,
                             evaluationTaskName='evaluation', CallbackForPreliminaryTask=None,
                             taskSolutionLLMDir='mbsModelsLLM',
                             taskSolutionSampleDir='mbsModelsSample' #
                             ):
    logger = agent.logger
    logDir = logger.logDir
    solutionDir = JoinPath(logDir, 'solution')
    os.makedirs(solutionDir, exist_ok=True)

    #additional directory for model comparison
    os.makedirs(logDir, exist_ok=True)
    taskSolutionLLMDir = JoinPath(logDir, taskSolutionLLMDir)
    taskSolutionSampleDir = JoinPath(logDir, taskSolutionSampleDir) #

    EnsureEmptyDir(taskSolutionLLMDir) # creates empty Dir or deletes content if existend
    EnsureEmptyDir(taskSolutionSampleDir) # creates empty Dir or deletes content if existend
    EnsureEmptyDir(solutionDir) # creates empty Dir or deletes content if existend

    randomizeParameters = (numberOfRandomizations>0) and not doSpaceVariation

    taskNamesLoaded = mbsModelLoader.ListOfModels()
    numberOfTasksWithoutVariationLoaded = mbsModelLoader.NumberOfModels()

    logger.LogText('InitializeTaskProcessing: using LLM model: '+agent.llmModel.modelName, 
                   printToConsole=True, separator=True, quickinfo=True)
    logger.LogText('Creating simulation code for mbs models: '+ str(taskNamesLoaded), 
                   printToConsole=True, separator=True, quickinfo=True)
    logger.LogText('InitializeTaskProcessing: using Exudyn version: '+exu.config.Version(), 
                   printToConsole=True, separator=True, quickinfo=True)
    ollamaVersion = 'N/A'
    if agent.llmModel.backend == 'ollama':
        try:
            import subprocess
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=5)
            ollamaVersion = result.stdout.strip() or result.stderr.strip()
        except Exception as e:
            ollamaVersion = f'unavailable ({e})'
        logger.LogText('InitializeTaskProcessing: using Ollama version: '+ollamaVersion,
                       printToConsole=True, separator=True, quickinfo=True)

    if doSpaceVariation:
        logger.LogText('Do space variation with factor: '+str(doSpaceVariation),
                       printToConsole=True, separator=True, quickinfo=True)


    DeleteFileIfExisting(logDir, "coordinatesSolution.txt") #this is unused, but could show up in case something goes wrong

    storedDataPerTask = {}
    logger.LogDebug('Initial GPU info:'+str(agent.llmModel.startInfoGPU)+'\n')

    #store important data for reproducibility and postprocessing (or continuation)!
    # in global data (except storedDataPerTask) NO special task infos should be stored!!
    globalData = {
         #general info:
         'llmModelName':agent.llmModel.modelName,
         'llmModelQuantization':agent.llmModel.GetQuantization(),
         'llmBatchSize':agent.llmModel.batchSize,
         'llmGPUused':agent.llmModel.startInfoGPU['GPUmodel'],
         'gpuStartInfo':agent.llmModel.startInfoGPU,
         'exudynVersion':exu.config.Version(),
         'ollamaVersion':ollamaVersion,
         'taskNamesLoaded':taskNamesLoaded, 
         'doSpaceVariation':doSpaceVariation,
         'numberOfRandomizations':numberOfRandomizations,
         'maxDifficultyLevel': mbsModelLoader.maxDifficultyLevel,
         'numberOfTasksWithoutVariationLoaded':numberOfTasksWithoutVariationLoaded,
         'numberOfTasksWithVariationLoaded':numberOfTasksWithoutVariationLoaded*max(1,numberOfRandomizations), #including randomizations, all model creation tasks
         'diffSolutionTolerance':agent.diffSolutionTolerance,
         'logDir':logDir,
         'solutionDir':solutionDir,
         'taskSolutionLLMDir':taskSolutionLLMDir,
         'taskSolutionSampleDir':taskSolutionSampleDir,
         'tStartAllTasks':time.time(),
         'dateTime': datetime.datetime.now().isoformat(sep=' ', timespec='milliseconds'),
         'executionDateStr': datetime.date.today().strftime("%Y-%m-%d"),
         'executionTimeStr': datetime.datetime.now().strftime("%H:%M:%S"),
         #main data storage:
         'storedDataPerTask': storedDataPerTask,
         #data that will be filled or added later
         'nExecutabilityMainTasks':0,
         'nExecutabilitySamples':0, #just for debugging, if implemented models are not running
         'nCompletedSamples':0,  #to see what is max nExecutabilitySamples
         }
    storedDataPerTask = globalData['storedDataPerTask']
    #initialize task data
    # here, task specific information is stored
    for mbsModelCnt, currentMBSmodelName in enumerate(taskNamesLoaded):
        for randomID in range(max(1,numberOfRandomizations) ):
            randomIDstr = ''
            if numberOfRandomizations > 0:
                randomIDstr = str(randomID)
            currentMBSmodelNameID = currentMBSmodelName + randomIDstr            
            spaceVariationID=randomID*doSpaceVariation

            #++++++++++++++++++++++++++++++++++++++++++++++++
            # initialize single task data
            taskData = {}
            storedDataPerTask[currentMBSmodelNameID] = taskData
            MakeTaskDataDict(taskData, taskLoader=mbsModelLoader, currentTaskName=currentMBSmodelName,
                             currentTaskNameWithID=currentMBSmodelNameID,
                             randomID=randomID, randomIDStr=randomIDstr, taskCount=mbsModelCnt,
                             evaluationTaskName=evaluationTaskName,
                             spaceVariationID=spaceVariationID, randomizeParameters=randomizeParameters)
            # this callback adds an entry per task containing information of a task that must be done before the main task
            if CallbackForPreliminaryTask != None:
                preliminaryTaskData = {}
                taskData['preliminaryTask'] = preliminaryTaskData
                CallbackForPreliminaryTask(preliminaryTaskData, taskData)

    RemoveTasksWithoutSampleFiles(globalData, logger)
    return globalData

#get directory of sample files (reference solutions) depending on task type
def GetSampleFileDir(taskData):
    if taskData.get('taskType', 'mbs') in ('fMBS', 'part'):
        return JoinPath(scriptDir, 'flexu', 'sampleFiles')
    return JoinPath(scriptDir, 'sampleFiles')

#remove tasks for which the sample file (reference solution) of the task or of one of its
#preliminary tasks (parts) is not available; this is done after all tasks have been initialized,
#such that the randomized parameters of the remaining tasks are not altered
def RemoveTasksWithoutSampleFiles(globalData, logger):
    storedDataPerTask = globalData['storedDataPerTask']
    removedTasks = []
    for taskNameWithID, taskData in list(storedDataPerTask.items()):
        requiredTasks = [taskData] + list(taskData.get('preliminaryTask', {}).values())
        for requiredTask in requiredTasks:
            fileName = requiredTask['dataOfTask'].get('sampleFileName', '')
            if not os.path.isfile(JoinPath(GetSampleFileDir(requiredTask), fileName)):
                removedTasks.append(taskNameWithID)
                del storedDataPerTask[taskNameWithID]
                break
    if len(removedTasks):
        taskNamesLoaded = [name for name in globalData['taskNamesLoaded']
                           if any(taskData['currentTaskName'] == name for taskData in storedDataPerTask.values())]
        globalData['taskNamesLoaded'] = taskNamesLoaded
        globalData['numberOfTasksWithoutVariationLoaded'] = len(taskNamesLoaded)
        globalData['numberOfTasksWithVariationLoaded'] = len(storedDataPerTask)
        globalData['tasksRemovedMissingSampleFile'] = removedTasks
        logger.LogText(f'{len(removedTasks)} tasks removed, as their sample files are not available: '+str(removedTasks),
                       printToConsole=True, separator=True, quickinfo=True)
        logger.LogText(f'Remaining models ({len(taskNamesLoaded)}): '+str(taskNamesLoaded),
                       printToConsole=True, separator=True, quickinfo=True)

def FinalizeTaskProcessingNEW(agent, globalData, storeGeneratedTextsAndResults=True, evaluationTaskName=''):
    logger=agent.logger
    logger.LogDebug('Start FinalizeTaskProcessing')
    resultsDict = {}
    storedDataPerTask = globalData['storedDataPerTask']

    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # restructute PER-TASK results into structured records and INITIALIZE main results dict
    taskRecords = [] # a list that contains infos for each task and points to the resultsDict
    resultsDict = {} # a list where for each variation the results are stored (and if preliminary tasks are there, also for them)
    # currentMBSmodelNameID is set automatically when initializing
    for modelNameWithID in storedDataPerTask:
        taskData = storedDataPerTask[modelNameWithID]
        name = taskData['currentTaskName']
        randomID = taskData['randomID']
        taskType = taskData['taskType']
        evaluationData = taskData[evaluationTaskName]
        # FLAT main results dict
        resultsDict[modelNameWithID] = {metricName: evaluationData.get(metricName, defaultValue) for metricName, defaultValue in globalEvaluationMetrics[taskType]}
        preliminaryTaskDataList = taskData.get('preliminaryTask', []) # initialize empty in case no preliminary tasks
        if preliminaryTaskDataList:
            preliminaryTaskRecords = []
            preliminaryResultsDict = {}
            resultsDict[modelNameWithID]['preliminaryTasks'] = preliminaryResultsDict
            for preliminaryTaskName, preliminaryTaskData in preliminaryTaskDataList.items():
                preliminaryModelNameWithID = preliminaryTaskData['currentTaskNameWithID']
                preliminaryName = preliminaryTaskData['currentTaskName']
                preliminaryRandomID = preliminaryTaskData['randomID']
                preliminaryTaskType = preliminaryTaskData['taskType']
                preliminaryEvaluationData = preliminaryTaskData[evaluationTaskName]
                # FLAT preliminary results dict
                preliminaryResultsDict[preliminaryModelNameWithID] = {metricName: preliminaryEvaluationData.get(metricName, defaultValue) for metricName, defaultValue in globalEvaluationMetrics[preliminaryTaskType]}
                # record task info
                preliminaryTaskRecords.append({'name': preliminaryName, 'id': preliminaryRandomID, 'taskType': preliminaryTaskType, 'taskData': preliminaryTaskData, 'resultsDict': preliminaryResultsDict[preliminaryModelNameWithID]})
        else:
            preliminaryTaskRecords = []
        taskRecords.append({'name': name, 'id': randomID, 'taskType': taskType, 'taskData': taskData, 'preliminaryTaskRecords': preliminaryTaskRecords, 'resultsDict': resultsDict[modelNameWithID]})
    logger.LogDict(resultsDict, header='FINAL TEST RESULTS', quickinfo=True) # now log ALL results
    globalData['resultsDict'] = resultsDict

    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # general MAIN task executability check and write special info into globalData
    totalRunTime = time.time() - globalData['tStartAllTasks']
    globalData['runTime'] = totalRunTime
    globalData['llmConfig'] = GetScalarClassObjectsDict(agent.llmModel)
    globalData['agentConfig'] = GetScalarClassObjectsDict(agent)
    globalData['agentConfig']['llmGenerateDefaultArgs'] = agent.llmGenerateDefaultArgs
    gpuInfo = agent.llmModel.currentInfoGPU
    gpuInfo['peakGPUmemory'] = agent.llmModel.peakGPUmemory
    globalData['gpuEndInfo'] = gpuInfo
    if not storeGeneratedTextsAndResults:
        del globalData['storedDataPerTask']

    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # per model, initialize success dict (now it is accumulated for each model)
    # for a model with preliminary tasks, we will also accumulated them
    def InitializeSuccessDict(taskType='', successDict={}):
        for metric in globalEvaluationMetrics[taskType]:
            metricName = metric[0]
            successDict['numberOfTasks'] = 0
            successDict['overallSuccess'] = 0 # add this here since it gives us if any metric failed or not
            successDict[metricName] = 0 # since we will count now, we initialize with zero here
    modelSuccessCount = {} # for every name, one entry is made (this counts the successes)
    for taskRecord in taskRecords:
        name = taskRecord['name']
        if name not in modelSuccessCount: # if it is not in modelSuccessCount
            taskType = taskRecord['taskType']
            modelSuccessCount[name] = {}
            InitializeSuccessDict(taskType=taskType, successDict=modelSuccessCount[name])
            if taskRecord['preliminaryTaskRecords']:
                modelSuccessCount[name]['preliminaryTasks'] = {} # also add dict where we can put preliminary tasks at
        # do this also if we already have the superior task (since preliminary task names may be different!!)
        # if preliminary tasks exist, set up a sub-dict for each one
        if taskRecord['preliminaryTaskRecords']:
            for preliminaryTaskRecord in taskRecord['preliminaryTaskRecords']:
                preliminaryName = preliminaryTaskRecord['name']
                # if two same preliminary tasks are done per model, they are in the same success dict (e.g. 2 pendulums for one fMBS)
                if preliminaryName not in modelSuccessCount[name]['preliminaryTasks']:
                    preliminaryTaskType = preliminaryTaskRecord['taskType']
                    modelSuccessCount[name]['preliminaryTasks'][preliminaryName] = {}
                    InitializeSuccessDict(taskType=preliminaryTaskType, successDict=modelSuccessCount[name]['preliminaryTasks'][preliminaryName])
    # now we have for each name the statistics initialized and also for each preliminary task name per model

    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # now we go through the resultsDict
    def IncoporateResultsDictIntoModelSuccessDict(name, taskType='', resultsDict={}, modelSuccessCount={}):
        overallSuccess = True
        for metric in globalEvaluationMetrics[taskType]:
            metricName = metric[0]
            metricSuccessCondition = metric[1]['successIf']
            if metricSuccessCondition == 'belowThreshold':
                if resultsDict[metricName] < agent.diffSolutionTolerance and resultsDict[metricName] >= 0:
                    modelSuccessCount[name][metricName] += 1
                else:
                    overallSuccess = False
            elif metricSuccessCondition == 'True':
                if resultsDict[metricName] == True:
                    modelSuccessCount[name][metricName] += 1
                else:
                    overallSuccess = False
            elif metricSuccessCondition == 'False':
                if resultsDict[metricName] == False:
                    modelSuccessCount[name][metricName] += 1
                else:
                    overallSuccess = False
            elif isinstance(metricSuccessCondition, float):
                if resultsDict[metricName] < metricSuccessCondition and resultsDict[metricName] >= 0:
                    modelSuccessCount[name][metricName] += 1
                else:
                    overallSuccess = False
            else:
                raise ValueError(f'IncoporateResultsDictIntoModelSuccessDict: unknown metricSuccessCondition "{metricSuccessCondition}"')
        modelSuccessCount[name]['numberOfTasks'] += 1 # also ALWAYS count number of tasks
        # count overall success
        if overallSuccess:
            modelSuccessCount[name]['overallSuccess'] += 1
    for record in taskRecords:
        name = record['name']
        taskType = record['taskType']
        resultsDict = record['resultsDict']
        IncoporateResultsDictIntoModelSuccessDict(name, taskType=taskType, resultsDict=resultsDict, modelSuccessCount=modelSuccessCount)
        if record['preliminaryTaskRecords']:
            for preliminaryRecord in record['preliminaryTaskRecords']:
                preliminaryName = preliminaryRecord['name']
                preliminaryTaskType = preliminaryRecord['taskType']
                preliminaryResultsDict = preliminaryRecord['resultsDict']
                # modelsuccess[name] already exists here
                IncoporateResultsDictIntoModelSuccessDict(preliminaryName, taskType=preliminaryTaskType,
                                                          resultsDict=preliminaryResultsDict, modelSuccessCount=modelSuccessCount[name]['preliminaryTasks'])

    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # compute scores and save them in global data
    globalData['modelSuccessCount'] = modelSuccessCount # success per model (count) incl. preliminary tasks

    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # get the metrics that shall be incorporated in overall metrics
    overallMetrics = {'chooseItemsMatch':'ChooseItemsMatch',
                      'graphMatch':'GraphMatch',
                      'graphMatchContent':'GraphMatchContent',
                      'solutionFileDiff':'SolutionFileMatch',
                      'solverInitSuccessful':'SolverInitSuccess',
                      'solverSuccess':'SolverSuccess', # this corresponds to executability metric!
                      'reachedMbsAssemble':'ReachedMbsAssemble',
                      'finishedMbsAssemble':'FinishedMbsAssemble',
                      'graphsIsomorphicNoTimeout':'GraphsIsomorphicNoTimeout',
                      'solverNoTimeoutSample':'SolverNoTimeoutSample',
                      'solverNoTimeoutLLM':'SolverNoTimeoutLLM',
                      'llmNoContextWindowExceeded':'LlmNoContextWindowExceeded'}
    if agent.isFlexuAI:
        # we always have these metrics available in flexuAI mode
        partMetrics = {'chooseItemsMatch':'ChooseItemsMatch',
                       'materialMatch':'MaterialMatch',
                       'relativeMassDifference':'RelativeMassDifference',
                       'relativeEigenfrequenciesNormDifference':'RelativeEigenfrequenciesNormDifference',
                       'relativeDifferenceVolume':'RelativeDifferenceVolume',
                       'interfaceNameMatch':'InterfaceNameMatch'}
        if agent.flexuAITaskType == 'part':
            overallMetrics = partMetrics # overwrite overallMetrics in this case
        elif agent.flexuAITaskType == 'fMBS':
            preliminaryOverallMetrics = partMetrics
    skipPrintMetrics = ['solverInitSuccessful','finishedMbsAssemble','solverSuccess','solverInitSuccessful',
                        'graphsIsomorphicNoTimeout','solverNoTimeoutSample','solverNoTimeoutLLM',
                        'llmNoContextWindowExceeded',
                        'graphNodesCountMatch','graphMatch','executableLLM']
    for key, label in overallMetrics.items():
        globalData['n'+label+'MainTasks'] = 0 # initialize the SPECIFIC global logs
        # we only have preliminary tasks when we have are in fMBS mode
    if agent.isFlexuAI and agent.flexuAITaskType == 'fMBS':
        for key, label in preliminaryOverallMetrics.items():
            globalData['n'+label+'PreliminaryTasks'] = 0 # initialize the SPECIFIC global logs

    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # compute success rates and write overall metrics in globalDict if needed
    modelSuccessRate = {}
    # these are metrics that are always globally written
    nChooseItemsMatchMainTasks = 0
    nExecutabilityMainTasks = 0
    nSuccessMainTasks = 0
    nTotalMainTasks = 0
    nSolverTimeoutSample = 0
    nSolverTimeoutLLM = 0
    nContextWindowExceeded = 0
    # these are only there if there are really preliminary tasks
    if agent.isFlexuAI and agent.flexuAITaskType == 'fMBS':
        nTotalPreliminaryTasks = 0
        nSuccessPreliminaryTasks = 0
        nExecutabilityPreliminaryTasks = 0
    for name in modelSuccessCount:
        modelSuccessRate[name] = {}
        numberOfTasks = modelSuccessCount[name]['numberOfTasks']
        for metric in modelSuccessCount[name]:
            if metric != 'numberOfTasks' and metric != 'preliminaryTasks':
                modelSuccessRate[name][metric] = round(modelSuccessCount[name][metric] / numberOfTasks, 4)
            for key, label in overallMetrics.items():
                if metric == key:
                    globalData['n'+label+'MainTasks'] += modelSuccessCount[name][metric]
        if 'preliminaryTasks' in modelSuccessCount[name]:
            modelSuccessRate[name]['preliminaryTasks'] = {}
            for preliminaryTaskName in modelSuccessCount[name]['preliminaryTasks']:
                modelSuccessRate[name]['preliminaryTasks'][preliminaryTaskName] = {}
                numberOfPreliminaryTasks = modelSuccessCount[name]['preliminaryTasks'][preliminaryTaskName]['numberOfTasks']
                for metric in modelSuccessCount[name]['preliminaryTasks'][preliminaryTaskName]:
                    if metric != 'numberOfTasks':
                        modelSuccessRate[name]['preliminaryTasks'][preliminaryTaskName][metric] = round(modelSuccessCount[name]['preliminaryTasks'][preliminaryTaskName][metric] / numberOfPreliminaryTasks, 4)
                    for key, label in preliminaryOverallMetrics.items():
                        if metric == key:
                            globalData['n'+label+'PreliminaryTasks'] += modelSuccessCount[name]['preliminaryTasks'][preliminaryTaskName][metric]
                nExecutabilityPreliminaryTasks += modelSuccessCount[name]['preliminaryTasks'][preliminaryTaskName]['executableLLM']
                nTotalPreliminaryTasks += modelSuccessCount[name]['preliminaryTasks'][preliminaryTaskName]['numberOfTasks']
                nSuccessPreliminaryTasks += modelSuccessCount[name]['preliminaryTasks'][preliminaryTaskName]['overallSuccess']
        # log OVERALL quantities
        # log overall results for non-preliminary tasks
        nChooseItemsMatchMainTasks += modelSuccessCount[name]['chooseItemsMatch']
        nExecutabilityMainTasks += modelSuccessCount[name]['executableLLM']
        nSuccessMainTasks += modelSuccessCount[name]['overallSuccess']
        nTotalMainTasks += numberOfTasks
        if 'solverNoTimeoutSample' in modelSuccessCount[name]:
            nSolverTimeoutSample += 1-int(modelSuccessCount[name]['solverNoTimeoutSample'])
        if 'solverNoTimeoutLLM' in modelSuccessCount[name]:
            nSolverTimeoutLLM += 1-int(modelSuccessCount[name]['solverNoTimeoutLLM'])
        if 'llmNoContextWindowExceeded' in modelSuccessCount[name]:
            nContextWindowExceeded += 1-int(modelSuccessCount[name]['llmNoContextWindowExceeded'])

    # Compute context window exceeded token statistics from individual task records
    cweTokensList = []
    for record in taskRecords:
        tokens = record['taskData'].get(evaluationTaskName, {}).get('contextWindowExceededTokens', 0)
        if tokens > 0:
            cweTokensList.append(tokens)
    contextWindowExceededTokensMax = max(cweTokensList) if cweTokensList else 0
    contextWindowExceededTokensMean = round(sum(cweTokensList) / len(cweTokensList), 1) if cweTokensList else 0
    globalData['contextWindowExceededTokensMax'] = contextWindowExceededTokensMax
    globalData['contextWindowExceededTokensMean'] = contextWindowExceededTokensMean

    # store raw per-task LLM inference metrics for rate/mean/max computations in postProcessing.py
    globalData['contextSizeUsedList'] = [record['taskData'].get(evaluationTaskName, {}).get('contextSizeUsed', 0) for record in taskRecords]
    globalData['numberOfPromptTokensList'] = [record['taskData'].get(evaluationTaskName, {}).get('numberOfPromptTokens', 0) for record in taskRecords]

    nTotalMainTasksDiv = max(1, nTotalMainTasks)
    globalData['modelSuccessRate'] = modelSuccessRate # success per model (rate) incl. preliminary tasks
    # write special metrics hard coded
    globalData['nTotalMainTasks'] = nTotalMainTasks
    globalData['nExecutabilityMainTasks'] = nExecutabilityMainTasks
    globalData['rateExecutabilityMainTasks'] = nExecutabilityMainTasks/nTotalMainTasksDiv
    globalData['nSuccessMainTasks'] = nSuccessMainTasks
    globalData['rateSuccessMainTasks'] = nSuccessMainTasks/nTotalMainTasksDiv
    if agent.isFlexuAI and agent.flexuAITaskType == 'fMBS':
        nTotalPreliminaryTasksDiv = max(1, nTotalPreliminaryTasks)
        globalData['nTotalPreliminaryTasks'] = nTotalPreliminaryTasks
        globalData['nExecutabilityPreliminaryTasks'] = nExecutabilityPreliminaryTasks
        globalData['rateExecutabilityPreliminaryTasks'] = nExecutabilityPreliminaryTasks / nTotalPreliminaryTasksDiv
        globalData['nSuccessPreliminaryTasks'] = nSuccessPreliminaryTasks
        globalData['rateSuccessPreliminaryTasks'] = nSuccessPreliminaryTasks/nTotalPreliminaryTasksDiv
        
    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # print nice summary
    def PrintModelSuccessRate(modelSuccessRate):
        skipKeys = ['numberOfTasks', 'preliminaryTasks']
        skipPrint = skipKeys+skipPrintMetrics
        for name, successRate in modelSuccessRate.items():
            metricStrings = [f'{metricName} = {val*100}%' for metricName, val in successRate.items() if metricName not in skipKeys]
            s = 'SUMMARY ' + name + ': ' + ', '.join(metricStrings)
            logger.LogDebug(s, printToConsole=False)
            metricStringsPrint = [f'{metricName} = {val*100}%' for metricName, val in successRate.items() if metricName not in skipPrint]
            sPrint = 'SUMMARY ' + name + ': ' + ', '.join(metricStringsPrint)
            print(sPrint)
            if 'preliminaryTasks' in successRate:
                for prelimName, prelimRate in successRate['preliminaryTasks'].items():
                    metricStrings = [f'{metricName} = {val*100}%' for metricName, val in prelimRate.items() if metricName not in skipKeys]
                    logger.LogDebug('\t -> prelim ' + prelimName + ': ' + ', '.join(metricStrings), printToConsole=True)
                    
    logger.LogDebug('\n==========================', printToConsole=True)
    PrintModelSuccessRate(modelSuccessRate)
    logger.LogDebug('\n==========================', printToConsole=True)
    logger.LogText('numberOfTokensGlobal: ' + str(agent.llmModel.numberOfTokensGlobal) + ', ' +
                   'numberOfThinkingTokensGlobal: ' + str(agent.llmModel.numberOfThinkingTokensGlobal) + ', ' +
                   'numberOfPromptTokensGlobal: ' + str(agent.llmModel.numberOfPromptTokensGlobal) + ', ' +
                   'numberOfRemovedTokensGlobal: ' + str(agent.llmModel.numberOfRemovedTokensGlobal) + ', ' +
                   'tokensPerSecondGlobal: ' + str(agent.llmModel.tokensPerSecondGlobal) + '\n',
                    printToConsole=True, separator=True, quickinfo=True)
    if getattr(agent.llmModel, 'backend', '') == 'academic_ai':
        logger.LogText(
            'apiCostGlobalEUR: ' + str(round(agent.llmModel.apiCostGlobalEUR, 6)) + ', ' +
            'apiBudgetRemainingEUR: ' + str(round(agent.llmModel.apiBudgetRemainingEUR, 6)) + ', ' +
            'apiInputTokensGlobal: ' + str(agent.llmModel.apiInputTokensGlobal) + ', ' +
            'apiOutputTokensGlobal: ' + str(agent.llmModel.apiOutputTokensGlobal),
            printToConsole=True,
            separator=False,
            quickinfo=True,
        )
    logger.LogText('Final GPU info:' + str(gpuInfo),separator=True, printToConsole=True)
    logger.LogDebug('\n==========================', printToConsole=True)
    maxLen = max(len('n' + label + 'MainTasks') for label in overallMetrics.values())
    if agent.isFlexuAI and agent.flexuAITaskType == 'fMBS':
        maxLenPreliminary = max(len('n' + label + 'PreliminaryTasks') for label in preliminaryOverallMetrics.values())
        maxLen = max([maxLen, maxLenPreliminary])
    logger.LogDebug('Total run time' + ' ' * (maxLen - len('Total run time')) + f": {round(totalRunTime, 3)} seconds", printToConsole=True)
    logger.LogText('nExecutabilityMainTasks' + ' ' * (maxLen - len('nExecutabilityMainTasks')) + f": {globalData['nExecutabilityMainTasks']} of {globalData['numberOfTasksWithVariationLoaded']}", printToConsole=True)
    logger.LogText('nSuccessMainTasks' + ' ' * (maxLen - len('nSuccessMainTasks')) + f": {globalData['nSuccessMainTasks']} of {globalData['numberOfTasksWithVariationLoaded']}", printToConsole=True)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # write special counts
    maxLen = max(len('n' + label + 'MainTasks') for label in overallMetrics.values())
    if agent.isFlexuAI and agent.flexuAITaskType == 'fMBS':
        maxLenPreliminary = max(len('n' + label + 'PreliminaryTasks') for label in preliminaryOverallMetrics.values())
        maxLen = max([maxLen, maxLenPreliminary])
    for key, label in overallMetrics.items():
        if key in []:
            continue
        overallName = 'n'+label+'MainTasks'
        spacesLeft = ' '*(maxLen-min(maxLen, len(overallName)))
        logger.LogText(overallName+spacesLeft+f": {globalData[overallName]} of {globalData['numberOfTasksWithVariationLoaded']}", printToConsole=True)
    if agent.isFlexuAI and agent.flexuAITaskType == 'fMBS':
        logger.LogText('==========================', printToConsole=True)
        logger.LogText('nExecutabilityPreliminaryTasks' + ' ' * (maxLen - len('nExecutabilityPreliminaryTasks')) + f": {globalData['nExecutabilityPreliminaryTasks']} of {globalData['nTotalPreliminaryTasks']}", printToConsole=True)
        logger.LogText('nSuccessPreliminaryTasks' + ' ' * (maxLen - len('nSuccessPreliminaryTasks')) + f": {globalData['nSuccessPreliminaryTasks']} of {globalData['nTotalPreliminaryTasks']}", printToConsole=True)
        for key, label in preliminaryOverallMetrics.items():
            overallName = 'n'+label+'PreliminaryTasks'
            spacesLeft = ' '*(maxLen-min(maxLen, len(overallName)))
            logger.LogText(overallName+spacesLeft+f": {globalData[overallName]} of {nTotalPreliminaryTasks}", printToConsole=True)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # warn metrics
    warnMetricsList = ['nGraphsIsomorphicNoTimeoutMainTasks','nSolverNoTimeoutSampleMainTasks','nSolverNoTimeoutLLMMainTasks','nLlmNoContextWindowExceededMainTasks']
    for warnMetric in warnMetricsList:
        if warnMetric in globalData:
            if globalData[warnMetric] < nTotalMainTasks:
                logger.LogText(f"*** WARNING: {warnMetric} = {globalData[warnMetric]} / {nTotalMainTasks}", printToConsole=True)
        else:
            logger.LogText(f'*** WARNING: metric {warnMetric} not in globalData', printToConsole=True)
    if nSolverTimeoutLLM > 0:
        logger.LogDebug(f"********************\nWARNING: solver timeout happened in {nSolverTimeoutLLM} LLM MBS models!\n********************\n", printToConsole=True)
    if nSolverTimeoutSample > 0:
        logger.LogDebug(f"********************\nWARNING: solver timeout happened in {nSolverTimeoutSample} sample MBS models!\n********************\n", printToConsole=True)
    if nContextWindowExceeded > 0:
        logger.LogDebug(f"********************\nWARNING: LLM context window exceeded in {nContextWindowExceeded} tasks!"
                        f" Mean exceeded by {contextWindowExceededTokensMean} tokens,"
                        f" max exceeded by {contextWindowExceededTokensMax} tokens."
                        f"\nConsider increasing contextWindowMaxSize.\n********************\n", printToConsole=True)

    logger.LogDebug('\n==========================', printToConsole=True)
    logger.LogDebug(f"OVERALL chooseItemsMatch of main tasks: {round(nChooseItemsMatchMainTasks/nTotalMainTasksDiv*100, 4)}%", printToConsole=True)
    logger.LogDebug(f"OVERALL executability of main tasks   : {round(nExecutabilityMainTasks/nTotalMainTasksDiv*100, 4)}%", printToConsole=True)
    logger.LogDebug(f"OVERALL success of main tasks         : {round(nSuccessMainTasks/nTotalMainTasksDiv*100, 4)}%", printToConsole=True)
    # write and store special rates
    for key, label in overallMetrics.items():
        rate = round(globalData['n'+label+'MainTasks']/max(1,nTotalMainTasks), 4)
        globalData['rate'+label+'MainTasks'] = rate
        spacesLeft = ' '*(20-min(20,len(label)))
        logger.LogDebug(f"OVERALL {label}{spacesLeft}: {rate*100}%", printToConsole=False)
    if agent.isFlexuAI and agent.flexuAITaskType == 'fMBS':
        for key, label in preliminaryOverallMetrics.items():
            overallName = 'n'+label+'PreliminaryTasks'
            rate = round(globalData[overallName]/max(1,nTotalPreliminaryTasks), 4)
            globalData['rate'+label+'PreliminaryTasks'] = rate
            spacesLeft = ' '*(20-min(20,len(label)))
            logger.LogDebug(f"OVERALL {label}{spacesLeft}: {rate*100}%", printToConsole=False)
    logger.LogDebug("Success w.r.t. executable tasks:",printToConsole=False, separator=True)
    logger.LogDebug(f"RELATIVE success of main tasks      : {round(nSuccessMainTasks/max(1,nExecutabilityMainTasks)*100, 4)}%", printToConsole=False)
    #write and store special rates; relative to executable tasks
    for key, label in overallMetrics.items():
        overallName = 'n'+label+'MainTasks'
        rate = round(globalData[overallName]/max(1,nExecutabilityMainTasks), 4)
        globalData['relative'+label+'MainTasks'] = rate
        spacesLeft = ' '*(20-min(20,len(label)))
        logger.LogDebug(f"RELATIVE {label}{spacesLeft}: {rate*100}%", printToConsole=False)
    if globalData['nExecutabilitySamples'] != globalData['nCompletedSamples']:
        logger.LogError(f"Only {globalData['nExecutabilitySamples']} samples out of {globalData['nCompletedSamples']} evaluated samples worked! CHECK implementation!")
    return globalData

def ExtractMBSGraphContentErrors(agent, globalData, evaluationTaskName=''):
    errorCounter = {}  # dict that maps category of error to paremeters (count) and list of differences
    for currentTaskNameWithID, subTaskData in globalData['storedDataPerTask'].items():
        if 'graphDiffDict' in subTaskData[evaluationTaskName]:
            graphContentErrorList = GetCategorizedGraphErrorsAndWrongParametersList(subTaskData[evaluationTaskName]['graphDiffDict'])
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++
            # extract errors
            seen = set() # to prevent counting an error twice PER CASE
            for graphError in graphContentErrorList:
                name = graphError['name']
                parameter = graphError['parameter']
                errorCounter.setdefault(name, {}) # makes new empty dict if not existing
                # does parameter already exist or not? This is handled here
                errorCounter[name].setdefault(parameter, {'count': 0, 'diffs': []})
                if (name, parameter) not in seen:
                    errorCounter[name][parameter]['count'] += 1
                if graphError['parameterAbsDiff'] is not None:
                    errorCounter[name][parameter]['diffs'].append(graphError['parameterAbsDiff'])
                seen.add((name, parameter)) # add the combo for the task and prevent counting again
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # summary print
    logger = agent.logger
    logger.LogDebug('\n==========================', printToConsole=True)
    logger.LogDebug(f"\n{'error name':<20} {'parameter':<20} {'unique occurances':>4} {'sum of abs diff':>30}", printToConsole=True)
    for name, parameters in errorCounter.items():
        for parameter, data in parameters.items():
            # also compute the sum of the absolute differences (maybe that is VERY small such that it is numerical...)
            try:
                sumDiff = round(np.sum(data['diffs']), 4) if data['diffs'] else 'n/a'
            except:
                sumDiff = 'n/a' #if not computable
            logger.LogDebug(f'{name:<20} {parameter:<20} {data["count"]:>4} {sumDiff:>30}', printToConsole=True)
    globalData['errorCounter'] = errorCounter
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # also get error categories out of the extracted data
    def GetErrorCategory(errorName):
        for errorCategory in errorCategories:
            if errorName in errorCategory[1]:
                return errorCategory[0]
        return 'Other'
    errorCategoryCounter = {}
    for errorCategory in errorCategories:
        errorCategoryCounter.setdefault(errorCategory[0], 0)
    errorCategoryCounter.setdefault('Other', 0) # always add OTHER
    for name, parameters in errorCounter.items():
        for parameter, data in parameters.items():
            errorCategory = GetErrorCategory(name)
            errorCategoryCounter[errorCategory] += data['count']
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++
    # summary print
    logger.LogDebug('\n==========================', printToConsole=True)
    logger.LogDebug(f"\n{'error category':<50} {'unique occurances':>4}", printToConsole=True)
    for category, count in errorCategoryCounter.items():
            logger.LogDebug(f'{category:<50}{count:>4}', printToConsole=True)
    globalData['errorCategoryCounter'] = errorCategoryCounter
    return globalData
