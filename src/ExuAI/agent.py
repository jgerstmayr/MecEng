#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: agent.py
#
# Details:  This is the main file (main class) for LLM agents.
#
# Authors:  Johannes Gerstmayr, Tobias Möltner, Sebastian Weyrer (ed.)
# Date:     2025-10-09
# Notes:    For part generation, a special safe process is included that writes
#           results into a temp file to protect parent process!
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

try:
    from .myLLM import MyLLM
    from .logger import Logger, STYLE
    from .readHelperFile import ParseHelper, GetRequiredTags, CreateSampleCode
    from .utilities import PreProcessLLMcode, JoinPath,\
            DeleteFileIfExisting, ReadFile
    from .templatesAndLists import simpleMBSChooseItemsPromptTemplate, simpleMBSGenerateCodePromptTemplate
except:
    from myLLM import MyLLM
    from logger import Logger, STYLE
    from readHelperFile import ParseHelper, GetRequiredTags, CreateSampleCode
    from utilities import PreProcessLLMcode, JoinPath,\
            DeleteFileIfExisting, ReadFile
    from templatesAndLists import simpleMBSChooseItemsPromptTemplate, simpleMBSGenerateCodePromptTemplate
import os
import sys
import time
import json
import numpy as np
import traceback
import platform
import numpy.random as random # set (global) random seed for reproducibility

import exudyn as exu
random.seed(42)

# for safe Python execution (catched netgen seg-faults)
def _run_safe_python_process(code, resultsFile, exuIn, exuOut):
    """Worker function that runs in a totally separate process."""
    try:
        exu.sys['exuAI_in'] = exuIn # now we have the inDict avauilable in this non-memory shared process!
        exu.sys['exuAI_out'] = exuOut
        local_env = {}
        exec(code, local_env, local_env) 
        result = {"success": True} # no python exception, success True, and nice exit code
    except Exception as e:
        result = {"success": False, "error": str(e)} # python exception, success False, and nice exit code
    finally: # if the kernel dies, we never land here, resultsFile stays empty!
        try:
            with open(resultsFile, 'w') as f:
                json.dump(result, f)
        except Exception:
            pass
        os._exit(0) # added this to shutdown child process (which sometimes does not happen)


# class to represent agent
# initialize agent with two main functions
# - choose items (only used if the chosen helper file has more than one item)
# - generate EXUDYN model
# to choose items, always an MBS model description is given
# the agent always generates an EXUDYN model, such that the functions are deliberately named ChooseItemsForExudynModel and GenerateExudynModel
class LabAgent:
    def __init__(self, llmModel:MyLLM, logger:Logger, helperFilesDict=dict(),
                 agentOutputDataDir='agentOutputData/', maxDifficultyLevel=10000):
        self.llmModel = llmModel
        self.logger = logger
        self.maxDifficultyLevel = maxDifficultyLevel
        if len(agentOutputDataDir) > 0 and agentOutputDataDir[-1] != '/':
            agentOutputDataDir += '/' # add this in case it is missing
        self.useExudynTimeout = True # might come in handy if code does not terminate (adding exudyn lines required or alternative)
        # inference parameter (passed as argument to myLLM.Generate()); exclude user input in llm-output and cleanup. show number of generated tokens and tokens per second
        self.maxTokensChooseItems = 512
        self.maxTokensGenerateModel = 1024 * 2 # 1024*3 in paper!!
        self.solverTimeoutSeconds = 120 #reduce for faster tests
        self.solverAddTimeout = False

        self.isFlexuAI = False #only True if flexuAI is active
        self.flexuAITaskType = 'fMBS' #switch between part generation ('part') only or part+fMBS ('fMBS')
        self.flexuAIFMBSEvalTime = 0
        self.flexuAIPartsEvalTime = 0
        self.flexuAIFMBSGenerated = 0
        self.flexuAIPartsGenerated = 0
        self.llmGenerateDefaultArgs = {}
        self.llmGenerateDefaultArgs['returnFullText'] = False
        self.llmGenerateDefaultArgs['cleanUpTokenizationSpaces'] = True
        self.llmGenerateDefaultArgs['showTiming'] = False
        self.llmGenerateDefaultArgs['doSample'] = False
        self.llmGenerateDefaultArgs['numBeams'] = 1
        # debug information
        self.logger.LogDebug('Initialize Agent ...')
        self.logger.LogDebug('Number of helper files to be used: ' + str(len(helperFilesDict)))
        pythonVersion = str(sys.version_info.major) + '.' + str(sys.version_info.minor) + '.' + str(sys.version_info.micro)
        self.logger.LogText('architecture        = ' + platform.architecture()[0], separator=True, timestamp=False)
        self.logger.LogText('processor           = ' + platform.processor(), timestamp=False)
        self.logger.LogText('platform            = ' + sys.platform, timestamp=False)
        self.logger.LogText('Python version      = ' + pythonVersion, timestamp=False)
        self.logger.LogText('NumPy version       = ' + np.__version__, timestamp=False)
        # parse the given helper file to dict and define system tags
        excludedTags = ['simulationPure', 'visualization', 'not used']
        self.parsedHelperDict = {}
        for key in helperFilesDict:
            self.parsedHelperDict[key] = ParseHelper(helperFilesDict[key], self.maxDifficultyLevel, excludeTags=excludedTags)
        self.listSystemTags = ['general import', 'assemble'] # used but no choice for agent and always included in generated codes
        self.listPromptTags = ['choose items', 'generate model'] # prompts that are available in an helper
        # some global counters
        self.totalNumberOfCodesGenerated = 0
        self.diffSolutionTolerance = 1e-5 #for small numerical deviations due to object order!

        self.preAssembleFlag = 'EvaluteMBScode.preAssemble' #flag to determine if mbs.Assemble() has been reached
        self.postAssembleFlag = 'EvaluteMBScode.postAssemble'
        
        self.samplesErrors = [] #add sample file errors here to output them at the end!!!
        
        #a fictive model which replaces LLM output for tests (runWithoutLLM=True):
        self.minimalMBSmodel="""
import exudyn as exu
SC = exu.SystemContainer()
mbs = SC.AddSystem()
oMass = mbs.CreateMassPoint(referencePosition = [0,0,0],physicsMass = 10,gravity = [0,0,-10])
mbs.Assemble()
mbs.SolveDynamic()
"""
    
    # updateDictsList: a list that contains dictionaries, where each of the dictionaries holds a list under 'tags'
    # to which the update should be applied and a 'text' (as string) with which the tag should be updated (added)
    # you may also give your update a 'title' and a boolean under the key 'apply' to enable/disable
    # e.g: updateDictsList = [{'title':'markerInCreateFunctions',
    #                          'helperFileName':'exudyn',
    #                          'tags':['CreateRevoluteJoint', 'CreateDistanceConstraint'] # ALL tags that see the same changes
    #                          '[tagField]':'Markers can also be ... .', [tagField] is 'CODE', 'NOTES', 'INFO', 'REQUIRES' ...
    #                          'mode':{'append','prepend','replace'}, #hier sollten mehrere Moden möglich sein
    # ...]
    def UpdateHelpers(self, updateDictsList=[], verbose=True):
        availableUpdateModes = ['replace', 'append', 'prepend']
        # go through all update dicts
        for updateDict in updateDictsList:
            titleOfUpdateRule = updateDict['title']
            errorString = 'ERROR: UpdateHelpers with update rule "' + titleOfUpdateRule + '": '
            warningString = 'WARNING: UpdateHelpers with update rule "' + titleOfUpdateRule + '": '
            infoString = 'UpdateHelpers with update rule "' + titleOfUpdateRule + '": '
            # name of helper that is updated now
            helperFileThatShouldBeUpdated = updateDict['helperFileName']
            # get the key, info, and mode that the update is done with
            for key, data in updateDict.items():
                if key.startswith('[') and key.endswith(']'):
                    entryToBeModified = key[1:-1] # string
                    # if this entry does not exist, it will be created automatically
                    updateInfo = data
                    break
            mode = updateDict['mode']
            # check whether mode is available
            tryUpdate = True
            if mode not in availableUpdateModes:
                self.logger.LogDebug(errorString + 'The specified mode ' + mode + ' to update helpers is not available! Skipping update rule entirely.',
                                     printToConsole=True)
                tryUpdate = False
            # check whether tags are given as list format
            tagsThatShouldBeUpdated = updateDict['tags']
            if type(tagsThatShouldBeUpdated) is not list:
                self.logger.LogDebug(errorString + 'Tags are not specified as strings in list format. Skipping update rule entirely.',
                                     printToConsole=True)
                tryUpdate = False
            if tryUpdate:
                for tagThatShoulBeUpdated in tagsThatShouldBeUpdated: # updateDict['tags'] is list
                    if tagThatShoulBeUpdated in self.parsedHelperDict[helperFileThatShouldBeUpdated]:
                        if entryToBeModified not in self.parsedHelperDict[helperFileThatShouldBeUpdated][tagThatShoulBeUpdated]:
                            self.logger.LogText(warningString + 'Creating field "' + entryToBeModified + '" in helper "' + helperFileThatShouldBeUpdated + '" at tag "' + tagThatShoulBeUpdated + '" since field you specified is not present in helper file for this tag.', 
                                                printToConsole=verbose, separator=True, quickinfo=True)
                        # apply the change in both cases: if tag is available or if tag is created
                        if mode == availableUpdateModes[0]: # 'replace'
                            self.parsedHelperDict[helperFileThatShouldBeUpdated][tagThatShoulBeUpdated][entryToBeModified] = updateInfo
                        elif mode == availableUpdateModes[1]: # 'append'
                            self.parsedHelperDict[helperFileThatShouldBeUpdated][tagThatShoulBeUpdated][entryToBeModified] += updateInfo
                        elif mode == availableUpdateModes[2]: # 'prepend'
                            self.parsedHelperDict[helperFileThatShouldBeUpdated][tagThatShoulBeUpdated][entryToBeModified] = updateInfo + self.parsedHelperDict[helperFileThatShouldBeUpdated][tagThatShoulBeUpdated][entryToBeModified] # reassigns info
                        self.logger.LogText(infoString +  'Updating helper "' + helperFileThatShouldBeUpdated + '" at tag "' + tagThatShoulBeUpdated + '" and field "' + entryToBeModified + '" with mode "' + mode + '" by:\n"' + updateInfo + '"', 
                                            printToConsole=verbose, separator=True, quickinfo=True)
                    else:
                        self.logger.LogDebug(errorString + 'Tag "' + tagThatShoulBeUpdated + '" is not available in helper "' + helperFileThatShouldBeUpdated + '". Skipping update rule for that specific tag.',
                                             printToConsole=True)


    # choose items of a list of available items in the given parsed helper dict that are necessary for building the simulation model
    # here the mbs model description is used
    # max tokens may be set different to make LLM more precise (if smaller max tokens are used)
    def GetItemSelectionPrompt(self, mbsModelDescriptions, helperFileName='',
                               promptTemplate=None, listVariation=0):
        import json

        # format a list of (key, description) tuples into different string representations
        # listVariation: 0=plain (default), 1=markdown table, 2=JSON, 3=numbered+backticks, 4=XML-style
        def FormatItemsList(itemsModel, listVariation=0):
            if listVariation == 0:
                # plain: "key: description"
                return "\n".join(f"{key}: {desc}" for key, desc in itemsModel)
            elif listVariation == 1:
                # markdown table
                lines = ["| Element | Description |", "| --- | --- |"]
                for key, desc in itemsModel:
                    lines.append(f"| {key} | {desc} |")
                return "\n".join(lines)
            elif listVariation == 2:
                # JSON array
                items = [{"element": key, "description": desc} for key, desc in itemsModel]
                return json.dumps(items, indent=2)
            elif listVariation == 3:
                # numbered list with backtick-quoted tags
                lines = []
                for i, (key, desc) in enumerate(itemsModel, 1):
                    lines.append(f"{i}. `{key}`: {desc}")
                return "\n".join(lines)
            elif listVariation == 4:
                # XML-style
                lines = []
                for key, desc in itemsModel:
                    lines.append(f'<item element="{key}">{desc}</item>')
                return "\n".join(lines)
            else:
                raise ValueError(f"Unknown listVariation: {listVariation}. Must be 0..4.")
        items = self.parsedHelperDict[helperFileName] # these must not be EXUDYN items; depends on the helper file used


        # get all items and corresponding descriptions (#INFO) for setting up exudyn model
        itemsModel = []
        keysModel = []
        for key, value in items.items():
            # do not make system tags and prompt tags available
            if key not in self.listSystemTags and key not in self.listPromptTags:
                keysModel.append(key)
                itemsModel.append((key, value['INFO']))

        # format items list based on listVariation
        itemsString = FormatItemsList(itemsModel, listVariation)

        if promptTemplate == None:
            promptTemplate = simpleMBSChooseItemsPromptTemplate
        else:
            promptTemplate = promptTemplate
        try:
            # make the prompt
            prompt = promptTemplate.format(modelDescription=mbsModelDescriptions,
                                           items=itemsString)
        except:
            self.logger.LogDebug('ERROR: could not format prompt; check if the prompt template holds the keys "modelDescription" and "items"')

        return prompt


    # generate an EXUDYN model with the selected items
    # here the mbs model description is used
    # max tokens may be set different to make LLM more precise (if smaller max tokens are used)
    def GetCodeGenerationPrompt(self, helperTags, mbsModelDescription, helperFileName='',
                                promptTemplate=None):

        for tag in helperTags:
            # look for additionally required tags in the chosen tags for dependencies
            requiredTags = GetRequiredTags(self.parsedHelperDict[helperFileName], tag)
            if len(requiredTags) != 0:
                newTags = [tag for tag in requiredTags if tag not in helperTags]
                # and extedn the helper tags if necessary
                helperTags.extend(newTags)
        # prompt template for code generation
        if promptTemplate == None:
            promptTemplate = simpleMBSGenerateCodePromptTemplate
        else:
            promptTemplate = promptTemplate

        # get the code of the tags the LLM has chosen
        tagSyntax = CreateSampleCode(self.parsedHelperDict[helperFileName], helperTags)
        # make the general structure of the code using 'general import' and 'assemble'
        # here the so called system tags are used --> general import ensures that exudyn is loaded as exu
        generalStructure = CreateSampleCode(self.parsedHelperDict[helperFileName], self.listSystemTags)
        # replace 'add items here' with the code of the tags to get a sample code
        sampleCode = generalStructure.replace('# add items here', tagSyntax)
        # make the prompt
        try:
            prompt = promptTemplate.format(generalStructure=sampleCode, 
                                           modelDescription=mbsModelDescription)
        except:
            self.logger.LogDebug('ERROR: could not format prompt; check if the prompt template holds the keys "generalStructure" and "modelDescription"')
        
        return [prompt] # list of one prompt (prompt lists are processed in modelBuilder.py)
    
   
    #read sample file, use sample dict to replace parameters from chosenParameters
    #chosenParameters={'mass':10,'stiffness':5000}
    def GetSampleFileDict(self, filePath, mbsModelName, sampleDict, chosenParametersDict=None):
        # sample files:

        parametersStart = '#%%parametersstart%%'
        sampleCodeStart = '#%%samplecodestart%%'
        sampleCodeStop = '#%%samplecodestop%%'
        modelCodeStop = '#%%modelStop%%'
        sampleCodeAssemble = 'mbs.Assemble()'
        
        if 'sampleFileName' not in sampleDict:
            errMsg = f'GetSampleFileDict: model {mbsModelName} has no sampleFileName'
            self.samplesErrors.append(errMsg)
            raise ValueError(errMsg)

        fileName = sampleDict['sampleFileName']

        try:
            code = ReadFile(filePath, fileName)
        except KeyboardInterrupt:
            print("Keyboard interrupt requested...exiting")
            raise
        except:
            errMsg = f'GetSampleFileDict with path {filePath}, file {fileName} failed (model={mbsModelName})'
            self.samplesErrors.append(errMsg)
            raise ValueError(errMsg)
            
        if sampleCodeStart not in code:
            errMsg = f'GetSampleFileDict: {sampleCodeStart} tag not found (model={mbsModelName})'
            self.samplesErrors.append(errMsg)
            raise ValueError(errMsg)
    
        preCode = ''
        mainCode = code
        if parametersStart in code:
            preCode = code.split(parametersStart)[0]
            mainCode = code.split(parametersStart)[1]
        # process code
        for text in ['exu.StartRenderer()',
                     'mbs.WaitForUserToContinue()',
                     'SC.WaitForRenderEngineStopFlag()',
                     'exu.StopRenderer()',
                     'StopRenderer',
                     'SC.renderer.Start()',
                     'SC.renderer.DoIdleTasks()',
                     'SC.renderer.DoIdleTasks()',
                     'SC.renderer.Stop()',
                     'mbs.SolutionViewer()',
                     ]:
            mainCode = mainCode.replace(text, 'pass') #replace with pass, e.g. if code appears in if-clause
    
        for _ in range(4):
            mainCode = mainCode.replace('\npass\n','\n')
        
        if modelCodeStop in code:
            modelCode = mainCode.split(sampleCodeStart)[1].split(modelCodeStop)[0]
            solverCode = mainCode.split(modelCodeStop)[1].split(sampleCodeStop)[0]
            paramsCode = mainCode.split(sampleCodeStart)[0]
        else:
            modelCode = mainCode.split(sampleCodeStart)[1].split(sampleCodeAssemble)[0]
            solverCode = sampleCodeAssemble + mainCode.split(sampleCodeAssemble)[1].split(sampleCodeStop)[0]
            paramsCode = mainCode.split(sampleCodeStart)[0]
        if 'parameters' in sampleDict:
            params = sampleDict['parameters']
        else:
            params = sampleDict  # assume sampleDict is already a dict of parameters
        # check if parameters are included
        for key, value in params.items():
            if not key.startswith('_') and key not in paramsCode:
                parameterMissing = True
                if 'metaInformation' in sampleDict:
                    # the only unused parameters allowed are master parameters
                    if any(key in part["masterParameters"] for part in sampleDict['metaInformation'].values()):
                        parameterMissing = False
                if parameterMissing:
                    errMsg = f'GetSampleFileDict: parameter "{key}" missing in sample file "{fileName}"'
                    self.samplesErrors.append(errMsg)
                    raise ValueError(errMsg)
                
        paramsText = ''
        if chosenParametersDict != None: #if parameters are available, use them
            for key, value in chosenParametersDict.items():
                valueStr = str(value)
                if isinstance(value,str): 
                    valueStr = '"' + valueStr + '"'
                paramsText += key + ' = ' + valueStr  + '\n'
        else:
            for key, value in sampleDict.items():
                valueStr = str(value['default'])
                if isinstance(value['default'],str):
                    valueStr = '"' + valueStr + '"'
                paramsText += key + ' = ' + valueStr + '\n'
        paramsText += '\n'

        params_Pre = ''

        for line in preCode.split('\n'): 
            if len(line.strip()) > 0 and not ('#' in line.strip()[0]): # add non-comments as these are default values, not passed through 
                params_Pre += line + '\n'

        executableCodeShort = params_Pre + paramsText + modelCode + solverCode #without comments at beginning
        executableCode = preCode + paramsText + modelCode + solverCode 
        return {'modelCode':modelCode,
                'solverCode':solverCode,
                'executableCode':executableCode,
                'paramsCode': params_Pre + paramsText,
                'executableCodeShort':executableCodeShort,
                'fileContent':code,
                }

    
    #evaluate either sample or LLM generated code; mode is used to printout infos
    #use local env to pass the evaluation local variables that are already available
    def EvaluateMBScode(self, code, currentMBSmodelNameID, solutionFileName, mode='', 
                        isAnalysis=False, printErrorToConsole=True, isPart=False):
        logger = self.logger
        logCodeAndOutput = False if 'sample' in mode.lower() else True

        codeExecutionSeparateProcess = True if isPart else False
    
        DeleteFileIfExisting(logger.logDir, solutionFileName)
        executionDict = self.ExecuteCodeAndGetLocalVariables(code, mode+", general exudyn model",
                                                             replaceCoordinatesSolutionFile=solutionFileName,
                                                             logCodeAndOutput=logCodeAndOutput,
                                                             printErrorToConsole=printErrorToConsole,
                                                             showTiming=True,
                                                             currentMBSmodelNameID=currentMBSmodelNameID,
                                                             useCodeExecutionTimeout=self.useExudynTimeout,
                                                             codeExecutionSeparateProcess=codeExecutionSeparateProcess)
        executable = executionDict['executable']
        localVariables = executionDict['localVariables']
        
        #log this always to find failed samples easier
        exStr = '' if executable else 'NOT '
        logger.LogText(mode+f' code is {exStr}executable for {currentMBSmodelNameID}; executable={str(executable)}')
            
        solveDynamicSuccessful = None
        solveStaticSuccessful = None
        solvedGeneral = None
        solverInitSuccessful = None
        reachedMbsAssemble = None
        finishedMbsAssemble = None
        mbsInLocalVariables = None
        solverNoTimeout = True #False if timeout; shall only pop up in case of timeout; if solver fails, no problem

        
        if isPart:
            # local variables and executable info will be returned
            # check if we could solve general (which means executable in this case)
            logger.LogText('EvaluateMBSCode: Special evaluation is performed for parts.')
            import exudyn as exu
            if executable and exu.sys['exuAI_out']: # check if out dict is not empty
                solvedGeneral = True
            else:
                solvedGeneral = False
        else:
            output = None
            try:
                if 'mbs' not in localVariables:
                    logger.LogError(f'EvaluateMBScode: model={currentMBSmodelNameID}, {mode} code: "mbs" is not in localVariables')
                    mbsInLocalVariables = False
                else:
                    mbsSysDict = localVariables['mbs'].sys
                    mbsInLocalVariables = True
                    reachedMbsAssemble = mbsSysDict.get(self.preAssembleFlag, False)
                    finishedMbsAssemble = mbsSysDict.get(self.postAssembleFlag, False)
                    
                    if not isAnalysis:
                        if 'dynamicSolver' in mbsSysDict:
                            solverInitSuccessful = mbsSysDict["dynamicSolver"].output.initializationSuccessful
                            solveDynamicSuccessful = True if solverInitSuccessful and mbsSysDict["dynamicSolver"].output.finishedSuccessfully else False

                            try:
                                solverNoTimeout = not (self.solverAddTimeout and 
                                                       mbsSysDict["dynamicSolver"].output.simulationTimeout)
                            except Exception:
                                pass #for older Exudyn versions, output.simulationTimeout does not exist
                                logger.LogDebug('output.simulationTimeout')

                            
                            logger.LogDebug(f'solverNoTimeout={solverNoTimeout},solverInitSuccessful={solverInitSuccessful}, finishedMbsAssemble={finishedMbsAssemble},self.solverAddTimeout={self.solverAddTimeout}, timer.total={mbsSysDict["dynamicSolver"].timer.total}, self.solverTimeoutSeconds={self.solverTimeoutSeconds}')

                        elif 'staticSolver' in mbsSysDict:
                            solverInitSuccessful = mbsSysDict["staticSolver"].output.initializationSuccessful
                            solveStaticSuccessful = True if solverInitSuccessful and mbsSysDict["staticSolver"].output.finishedSuccessfully else False

                            try:
                                solverNoTimeout = not (self.solverAddTimeout and 
                                                       mbsSysDict["staticSolver"].output.simulationTimeout)
                            except Exception:
                                pass #for older Exudyn versions, output.simulationTimeout does not exist

                        elif reachedMbsAssemble: #otherwise we will get notification that reachedMbsAssemble=False
                            logger.LogError(f'EvaluateMBScode: model={currentMBSmodelNameID}, {mode} code: mbs.sys does not contain dynamicSolver/staticSolver (probably did not solve)', printToConsole=False)
                    else:
                        solvedGeneral = False
            
                        if 'output' in localVariables:
                            output = localVariables['output']
                            if type(output)==list or type(output)==dict: #we cannot evaluate more for now!
                                solvedGeneral = True
                                logger.LogDebug(f'Received output, model={currentMBSmodelNameID}, {mode} code:\noutput="{output}"')
                            else:
                                solvedGeneral = False
                        else:
                            logger.LogError(f'EvaluateMBScode: model={currentMBSmodelNameID}, {mode} code: local variables do not contain "output"', printToConsole=False)
            except KeyboardInterrupt:
                print("Keyboard interrupt requested...exiting")
                raise
            except:
                errorMessage = traceback.format_exc()
                logger.LogError(f'EvaluateMBScode: model={currentMBSmodelNameID}, {mode} code: dynamicSolver/staticSolver/output: failed to retrieve solution information (probably no mbs has been created!)')
                logger.LogDebug(f'traceback=\n{errorMessage}')
        solverSuccess = solveDynamicSuccessful == True or solveStaticSuccessful == True or solvedGeneral == True
        # special settings depending on task type
        if not isPart:
            if not solverSuccess:
                if solveDynamicSuccessful == False:
                    logger.LogError(f'EvaluateMBScode: model={currentMBSmodelNameID}, {mode} code: dynamicSolver not successful!', printToConsole=False)
                elif solveStaticSuccessful == False:
                    logger.LogError(f'EvaluateMBScode: model={currentMBSmodelNameID}, {mode} code: staticSolver not successful!', printToConsole=False)
                elif solvedGeneral == False:
                    logger.LogError(f'EvaluateMBScode: model={currentMBSmodelNameID}, {mode} code: evaluation method not successful! output:"{output}"')
                if solverInitSuccessful == False:
                    logger.LogError(f'EvaluateMBScode: model={currentMBSmodelNameID}, {mode} code: solver did not converge!', printToConsole=False)
            if not reachedMbsAssemble and (self.preAssembleFlag in code) and mbsInLocalVariables==True:
                logger.LogError(f'EvaluateMBScode: model={currentMBSmodelNameID}, {mode} code: mbs.Assemble() has not been reached!', 
                                printToConsole=False)
        else:
            # here we can add more info when it is a part
            pass
        return dict(executable=executable, 
                    localVariables=localVariables, 
                    solverSuccess=solverSuccess, 
                    solverInitSuccessful=solverInitSuccessful, 
                    reachedMbsAssemble=reachedMbsAssemble, 
                    finishedMbsAssemble=finishedMbsAssemble,
                    solverNoTimeout=solverNoTimeout,
                    )

    # Executes provided code in a separate process to prevent infinite loops.
    # replaceCoordinatesSolutionFile: provide string to replace the output filename
    def ExecuteCodeAndGetLocalVariables(self, code, codeType, showTiming=False, timeout=60,
                                        replaceCoordinatesSolutionFile = None,
                                        logCodeAndOutput = True,
                                        printErrorToConsole = True,
                                        currentMBSmodelNameID = '',
                                        useCodeExecutionTimeout=False,
                                        codeExecutionSeparateProcess=False):
        #++++++++++++++++++
        exudynLogOutput = ''
        exudynLogFileName = JoinPath(self.logger.logDir,'exudynTemp.log')
        with open(exudynLogFileName, 'w', encoding='utf-8') as file:
            file.write("")  # Overwrite with an empty file
        
        code = PreProcessLLMcode(self.logger, code, 
                                 replaceCoordinatesSolutionFile=replaceCoordinatesSolutionFile,
                                 addTimeout = self.solverAddTimeout, 
                                 solverTimeoutSeconds=self.solverTimeoutSeconds) 
        
        if logCodeAndOutput:
            self.logger.LogText("Execution code type: " + str(codeType) + '\n'+
                                "Exudyn code:\n" +
                                code, style=STYLE.python)
    
        executable = False
        errorMessage = ''
        returnDict = {} #in case of failure
        # only for parts
        if codeExecutionSeparateProcess:
            self.logger.LogDebug("Execute code in separate Process (timeout; catches seg-faults)")
            import multiprocessing
            import tempfile
            import json
            import os
            # make a temporary file to which we write the results dict
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                resultsFile = f.name
                self.logger.LogDebug(f"NOTE: File-Info for safe code execution: {resultsFile}.")
            exuIn = exu.sys['exuAI_in']
            exuOut = {} # exu.sys['exuAI_out']
            if sys.platform == 'linux':
                ctx = multiprocessing.get_context("spawn")
                p = ctx.Process(target=_run_safe_python_process,
                                args=(code, resultsFile, exuIn, exuOut)) # exuOut NEVER used since this is just to check if it runs ...
            else:
                p = multiprocessing.Process(target=_run_safe_python_process, 
                                            args=(code, resultsFile, exuIn, exuOut)) # exuOut NEVER used since this is just to check if it runs ...
            try:
                p.start()
                p.join(timeout) # Wait for it to finish
                failed = False
                if p.is_alive():
                    self.logger.LogDebug(f"Task timed out for model={currentMBSmodelNameID}! Terminating...", printToConsole=True)
                    p.terminate() 
                    p.join()
                    failed = True
                    returnDict['error'] = "Task timed out!"
                # seg fault (_run_safe_python_process handles exceptions such that a normal python exception delivers exitCode==0)
                elif p.exitcode != 0:
                    try:
                        # is the results file there and is it non-empty?
                        if os.path.exists(resultsFile) and os.path.getsize(resultsFile) > 0:
                            # should not happen!
                            with open(resultsFile, 'r') as f:
                                result = json.load(f)
                        else:
                            result = None # results file empty
                    except:
                        result = None # results file not there / corrupt
                    if result is None:
                        self.logger.LogDebug(f"Python KERNEL CRASHED when exucuting code for model={currentMBSmodelNameID}. Exit code: {p.exitcode}", printToConsole=True)
                        returnDict['error'] = f"KERNEL CRASHED! Exit code: {p.exitcode}"
                    else: # should not happen!
                        self.logger.LogDebug(f"Failure in safe code execution; seg fault detected but results file written...", printToConsole=True)
                        self.logger.LogDebug(f"Code execution gives result={result.get('success',-2)}")
                    failed = True
                else:
                    #regular exception (we do NOT set failed in order to run execution and get output)
                    # for special exceptions we have to stop (see specialExceptionsThatHarmParentList)
                    result = {"success": -3} #special value to distinguish; should not happen!
                    try:
                        with open(resultsFile, 'r') as f:
                            result = json.load(f) # now we have result
                    except Exception: # then we had a seg-fault - should not happen
                        self.logger.LogDebug("Results file corrupted, do not continue.")
                        failed = True
                    if result.get("success", -2) != True:
                        exceptionMessage = result.get("error", None)
                        self.logger.LogDebug(f"Exception when executing code for model={currentMBSmodelNameID}, result={result.get('success',-2)}! Exit code: {p.exitcode}, Error message: {exceptionMessage}", printToConsole=False)
                        returnDict['error'] = "Exception when executing model!"
                        specialExceptionsThatHarmParentList = ['Meshing failed!']
                        # special exceptions where also failed is set to protect parent
                        if exceptionMessage in specialExceptionsThatHarmParentList:
                            failed = True
                            self.logger.LogDebug(f"Due to error message '{exceptionMessage}', skipping part for meshing to protect parent!", printToConsole=True)
                    #else: all fine!
                if failed:
                    self.logger.LogDebug("No main evaluation to protect parent!", printToConsole=True)
                    return dict(executable=executable, localVariables=returnDict) #did not work
            # finally delete the temp results file
            finally:
                try:
                    os.remove(resultsFile)
                except OSError:
                    pass

        if useCodeExecutionTimeout:
            self.logger.LogDebug("Execute code with separate thread (timeout possible)")
            #run code with timeout using threading (avoids Windows multiprocessing spawn issues)
            import threading
            localEnv = {}

            def _run(code_str, env, results):
                try:
                    exec(code_str, env, env)
                except Exception:
                    results['error'] = traceback.format_exc()
                results.update(env) # not only update in try-clause!

            ts = time.time()
            thread = threading.Thread(target=_run,
                                      args=(code, localEnv, returnDict),
                                      daemon=True)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                # timeout: thread keeps running as daemon but we move on
                returnDict = {}
                errorMessage = 'Execution killed due to timeout '+str(timeout)+' seconds'
                self.logger.LogError(f"Execution timeout after {timeout}s (model={currentMBSmodelNameID}).")
            else:
                if 'error' in returnDict:
                    errorMessage = str(returnDict['error'])
                else:
                    executable = True

            ts = time.time() - ts
            #++++++++++++++++++
        else:
            self.logger.LogDebug("Execute code without timeout / safe Python process")
            localEnv = {}
            ts = time.time()
    
            try:
                exec(code, localEnv, localEnv) #still has exu.sys available!         
                executable = True
            except Exception as e:
                errorMessage = traceback.format_exc()
                errorMessage = f"{str(e)}\n{errorMessage}"
                
            ts = time.time() - ts
            
            returnDict = localEnv
    
        try:
            with open(exudynLogFileName, 'r') as file:
                exudynLogOutput = file.read()
    
            if logCodeAndOutput: #and exudynLogOutput!='':
                self.logger.LogText('Exudyn code log:\n'+exudynLogOutput, style=STYLE.python)
        except KeyboardInterrupt:
            print("Keyboard interrupt requested...exiting")
            raise
        except:
            pass #no reporting
            
        if showTiming and logCodeAndOutput:
            self.logger.LogText("Exudyn code execution time: "+str( round(ts, 6)))
        
        if not executable:
            self.logger.LogError(f'Execute code error message (model={currentMBSmodelNameID}):\n  '+errorMessage, 
                                 printToConsole=printErrorToConsole)
            if not logCodeAndOutput:
                self.logger.LogText("Execution code type: " + str(codeType) + '\n'+
                                    "Exudyn code (not executable):\n" +
                                    code, style=STYLE.python)
    
        return dict(executable=executable, localVariables=returnDict)
