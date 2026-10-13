# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: modelBuilderDriver.py
#
# Details:  Command line driver for the model creation tests (rigid MBS and flexible MBS / FlexuAI);
#           type python modelBuilderDriver.py -h to see help.
#           Runs the tests for a single LLM, or merges existing results (-m).
#
# Authors:  Johannes Gerstmayr, Sebastian Weyrer (FlexuAI-Pipeline)
# Date:     2025-03-15
# Notes:    Ensure that the LLMs are available through Ollama (or locally / via the Hugging Face hub for other backends).
#           For Ollama batching (OLLAMA_NUM_PARALLEL), the driver restarts the Ollama server; on Windows, close the
#           Ollama tray app first (taskkill /f /im ollama.exe).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%% import necessary libraries and modules
import argparse
import sys
import os
import ast
import platform
import traceback

from ExuAI.myLLM import llmModelsDict
from ExuAI.utilities import JoinPath, LogOverallCounter2File
from ExuAI.agent import LabAgent
from ExuAI.logger import Logger
from ExuAI.myLLM import MyLLM
from ExuAI.modelLoader import ModelLoader
from ExuAI.modelDescriptions import flexibleMBSDescriptions, geometryDescriptions, simpleAndMediumMBSDescriptions,\
    AddTaskTypeToDescription
from modelBuilder import ChooseItemsTask, GenerateMBSTask, InitializeTaskProcessing,\
    FinalizeTaskProcessingNEW, helperFilesDict, ExtractMBSGraphContentErrors
from ExuAI.utilities import WriteDictToJSON, PrintHeadlineToConsole
from flexibleMBSBuilder import InitializationCallbackForPartsCreation,\
    RestructureGlobalDataAndGetGlobalDataForPartCreation, FlexuAIChooseItemsAndGenerateCode, SkipFMBSDueToPartsNotBeingCreatedCorrectly
from postProcessing import MergeTestModelResults, ExportMergedResultsToCSV, ExportModelSuccessRatesToCSV, PlotMetricCorrelationPerLLM, PlotPerformanceVsReleaseDate 

if __name__ == "__main__":
    #put everything in here for multithreading subprocess (safe execution)
    #%% Create the argument parser
    description = "Call test models with LLM model name and other parameters. Available LLM models are:\n"
    sep=''
    for key, value in llmModelsDict.items():
        description += sep+value['shortName']+'\n'
        sep = ', '
    parser = argparse.ArgumentParser(description=description)
    # define arguments
    parser.add_argument("-n", "--llmName", type=str, default="Gemma4-31B-OL", help="LLM Model Name")
    parser.add_argument("-r", "--randomizations", type=int, default=1, help="Number of randomizations / space variations")
    parser.add_argument("-d", "--difficulty", type=int, default=44, help="Max difficulty level of tasks (44 = all rigid MBS tasks, 50 = all flexible MBS tasks)")
    parser.add_argument("-q", "--quantization", type=int, default=16, help="HF model quantization (ignored for other LLMs; 4,8 or 16=default)")
    parser.add_argument("-b", "--batchSize", type=int, default=0, help="batch size for HF and ollama models (ignored for other LLMs)")
    parser.add_argument("-g", "--device", type=str, default=None,help="GPU device number (e.g. 0 or 0,1 or [0,1])")
    parser.add_argument("-s", "--doSpaceVariation",  action='store_true', help="perform space variations instead of randomizations")
    parser.add_argument("-t", "--thinkingMode",  type=int, default=0, help="set thinking mode in models (default...0=low/off, 1=medium, 2=high)") # differentiating between low/medium/high thinking modes is currently only implemented for Ollama-GPT-OSS, for other models it is just on/off
    parser.add_argument("-temp", "--temperature",  type=float, default=-1, help="temperature for LLM; temperature < 0 means default value (None)")
    parser.add_argument("--apiAuthMethod", type=str, default="x-client-secret", help="Academic AI auth: bearer | x-client-id | x-client-secret")
    parser.add_argument("--apiBaseURL", type=str, default="https://ca-backend-uibkai-prod.wittystone-43265e4a.swedencentral.azurecontainerapps.io", help="Academic AI base URL")
    parser.add_argument("--apiTimeout", type=int, default=120, help="Academic AI timeout in seconds")
    parser.add_argument("--apiBudgetEUR", type=float, default=20.0, help="Budget used for Academic AI cost tracking")
    parser.add_argument("-m", "--merge", type=str, default="", help="folder name to merge all results.json")
    parser.add_argument("-f", "--destinationFolder", type=str, default="logsTM", help="destination folder for results")
    parser.add_argument("-PVmbs", "--promptVariationMBS", type=int, default=0, help="prompt template variation for generateMBS task(default=0)")
    parser.add_argument("-PVci", "--promptVariationCI", type=int, default=0, help="prompt template variation for chooseItems task (default=0)")
    parser.add_argument("-lV", "--listVariation", type=int, default=0, help="list format variation for chooseItems element list (0=plain, 1=markdown table, 2=JSON, 3=numbered+backticks, 4=XML-style)")
    parser.add_argument("-skipMBS", "--skipGenerateMBS", action='store_true', help="skip MBS code generation")
    parser.add_argument("-models", '--specificModelNames', type=str, nargs='+', default=None, help='Specify specific models to be build; leave None if all models should be build.')
    # special FlexuAI settings (including settings for generated mesh to speed up evaluation)
    parser.add_argument("-flexuAI", "--flexuAI", action='store_true', help="activate the flexuAI pipeline to either generate machine parts or full flexible mbs with machine parts")
    parser.add_argument("-flexuAITaskType", "--flexuAITaskType", type=str, default='fMBS', help='Select task type (either "fMBS" for flexible MultiBody System or "part" for single machine parts).')
    parser.add_argument("-mH", "--maxHFactor", type=float, default=1, help='Factor that is multiplied with a maxH recommendation; if the maxH value set by the LLM is below the multiplied maxH recommendation, the multiplied recommendation is used.')
    parser.add_argument("-mO", '--meshOrder', type=int, default=2, help='Order of the mesh elements that are used both, in the LLM code and the sample files.')
    parser.add_argument("-mG", "--meshGrading", type=float, default=0.9, help='Grading factor used when meshing the geometries (larger gives coarser meshes).')
    parser.add_argument("-mC", "--meshCurvatureSafety", type=float, default=0.7, help='Curvature safety used when meshing the geometries (smaller gives coarser meshes).')
    # Parse arguments
    args = parser.parse_args()
    
    #%% from the parsed arguments, assign values
    llmShortName = args.llmName
    numberOfRandomizations = args.randomizations
    maxDifficultyLevel = args.difficulty
    quantizationHF = args.quantization
    batchSize = args.batchSize
    device = args.device
    thinkingMode = args.thinkingMode
    temperature = args.temperature if args.temperature >= 0 else None
    apiAuthMethod = args.apiAuthMethod
    apiBaseURL = args.apiBaseURL
    apiTimeout = args.apiTimeout
    apiBudgetEUR = args.apiBudgetEUR
    doSpaceVariation = args.doSpaceVariation * 7 #7 used to also include some endline variations ...
    promptVariationMBS = args.promptVariationMBS
    promptVariationCI = args.promptVariationCI
    listVariation = args.listVariation
    mergeDir = args.merge
    destinationFolder = args.destinationFolder
    skipGenerateMBS = args.skipGenerateMBS
    specificModelNames = args.specificModelNames
    # special flexuAI variables (all starting with flexuAI)
    flexuAI = args.flexuAI # this is default False
    flexuAITaskType = args.flexuAITaskType
    flexuAIMaxHFactor = args.maxHFactor
    flexuAIMeshOrder = args.meshOrder
    flexuAIMeshGrading = args.meshGrading
    flexuAIMeshCurvatureSafety = args.meshCurvatureSafety
    doMerge = mergeDir != ''

    # check for required packages
    try:
        import importlib
        for packageName in ['exudyn', 'numpy', 'scipy', 'networkx', 'matplotlib', 'pymeshlab'] + (['ngsolve'] if flexuAI else []):
            importlib.import_module(packageName)
    except ImportError as e:
        print(f'Missing required package: {e}. Install via: pip install -r requirements.txt')
        sys.exit(1)
    

    if flexuAI:
        if promptVariationCI != 0:
            print('WARNING: Prompt variation is not implemented for FlexuAI!')
            sys.exit()
    
    destinationFolder=JoinPath('../results/',destinationFolder)
    print(f'destinationFolder={str(destinationFolder)}')

    
    isLinux = platform.system() == 'Linux'
    isWindows = platform.system() == 'Windows'
    
    # get LLM model name
    llmModelName = ''
    for key, value in llmModelsDict.items():
        if llmShortName == value['shortName']:
            llmModelName = key
    
    if llmModelName == '':
        print('testModelCreation: NO VALID LLM MODEL SHORT NAME FOUND!\ncall with -h to get all available names')
        sys.exit()
    
    # choose backend based on model type
    modelType = llmModelsDict[llmModelName].get('type', '')
    if modelType == 'API':
        if llmShortName.startswith('Claude-'):
            backend = 'anthropic_api'
        else:
            backend = 'academic_ai'
    elif modelType == 'vllm':
        backend = 'vllm'
    elif llmModelName.endswith('.gguf'):
        backend = 'gpt4all'
    elif 'OL' in llmShortName:
        backend = 'ollama'
    else:
        backend = 'hftransformers'
    
    enableThinking = True if thinkingMode > 0 else False
    thinkingTokenFactor = 1.3334+3.6666*enableThinking
    if backend in ('ollama', 'vllm') and 'gpt-oss' in llmModelName.lower():
        if thinkingMode == 0:
            enableThinking = 'low'
        elif thinkingMode == 1:
            enableThinking = 'medium'
            thinkingTokenFactor = 3 #special for gpt-oss; does only little thinking, so this should be ok
        elif thinkingMode == 2:
            enableThinking = 'high'
        else:
            print('Invalid thinking mode for gpt-oss. Use 0 for low, 1 for medium, or 2 for high.')
            sys.exit()
                
    #determine quantization and quantization string (needed only for HF-backend)
    modelQuantization = {4:'4bit', 8:'8bit'}.get(quantizationHF, None)
    if backend != 'hftransformers': # gguf, ollama or API backend
        quantString = ''
        modelQuantization = None
    else: # HF backend
        quantString = '-Q'+str(quantizationHF)
    
    shortName = llmModelsDict[llmModelName]['shortName']
    
    logDir=JoinPath(destinationFolder,'log_'+shortName+quantString)
    
    if batchSize>1:
        logDir += '-B'+str(batchSize)
    
    if thinkingMode != 0:
        logDir += '-Thk'+str(thinkingMode)
    if temperature is not None:
        logDir += '-T'+str(temperature)
    if promptVariationMBS != 0:
        logDir += '-PVMBS'+str(promptVariationMBS)
    if promptVariationCI != 0:
        logDir += '-PVCI'+str(promptVariationCI)
    if listVariation != 0:
        logDir += '-lV'+str(listVariation)
    
    logger = Logger(logDir=logDir)
    logger.limitConsoleText = 5000
    
    logger.LogText(
                   "***==========================***\n"
                   "***   Starting modelBuilder  ***\n"
                   "***==========================***\n",
                   printToConsole=True)
    if flexuAI:
        logger.LogText(
                   "***==========================***\n"
                   "***      Starting flexuAI    ***\n"
                   "***==========================***\n",
                   printToConsole=True)

    #extend context size with token factor in case we have thinking models:
    if ('qwen3.5' in llmShortName.lower() or
        'qwen3.6' in llmShortName.lower() or
        'gemma4' in llmShortName.lower() or
        'gpt-oss' in llmShortName.lower() or
        'qwencoder3.5' in llmShortName.lower() or
        'deepseek-r1' in llmShortName.lower() or
        'ministral' in llmShortName.lower() or # ministral chooses many items, thus it requires more context size
        'dolphincoder' in llmShortName.lower() or
        'glm-4.7' in llmShortName.lower() or
        'nemotron-cascade-2' in llmShortName.lower() or
        'qwen2' in llmShortName.lower() or
        'claude' in llmShortName.lower() or # claude API models generate verbose code and hit the token limit
        flexuAI):
        thinkingTokenFactor = max(thinkingTokenFactor, 2) #increase due to lot of thinking during coding - runs out of tokens!
        logger.LogDebug(f"Adapted thinkingTokenFactor to {thinkingTokenFactor} due to expected thinking and/or FlexuAI",
                        printToConsole=True)

                       
    hostIsHopper = False
    if isLinux:
        hostIsHopper = True if 'hopper' in os.uname().nodename else False
        
    if hostIsHopper:
        logger.LogText('Running on Hopper...')
    
    #%%++++++++++++++++++++++++++++++++++++++++++++++++++++
    if doMerge:
        mergedResultsFilePath = JoinPath(mergeDir, 'allLLMresults.json')
        resultsCSVfilePath = JoinPath(mergeDir, 'overallResults.csv')
        mergeLogger = Logger(logDir=JoinPath(mergeDir, 'logsTrash'))
        MergeTestModelResults(dirTestModels=mergeDir, outputJSONpath=mergedResultsFilePath, logger=mergeLogger)
        ExportMergedResultsToCSV(mergedJsonFilePath=mergedResultsFilePath, outputCSVpath=resultsCSVfilePath, logger=mergeLogger)
        ExportModelSuccessRatesToCSV(mergedJsonFilePath=mergedResultsFilePath,
                                     outputCSVpath=JoinPath(mergeDir, 'overallResultsSuccessRates.csv'), logger=mergeLogger)

        # plot LLM performance (overall success) in MBS generation progress over time (release dates)
        PlotPerformanceVsReleaseDate(mergedResultsFilePath, JoinPath(mergeDir, "overallSuccessVsReleaseDatePlot.pdf"),
                                     logger=mergeLogger, metricKey='overall success')
        PlotPerformanceVsReleaseDate(mergedResultsFilePath, JoinPath(mergeDir, "chooseItemsVsReleaseDatePlot.pdf"),
                                     logger=mergeLogger, metricKey='choose items')
        # plot correlation between two metrics in generation results over different LLMs (e.g. temperature and overall success)
        PlotMetricCorrelationPerLLM(mergedJsonFilePath=mergedResultsFilePath,
                                    keyX='temperature', keyY='overall success',
                                    outputPlotPath=JoinPath(mergeDir, "correlation.pdf"), logger=mergeLogger)
        sys.exit()
        
    # define path to stored LLMs for local loading (HOPPER specific setting)
    pathToStoredLLMs = "/opt/llm_models/" if hostIsHopper else '~/.cache/huggingface/hub'
    
                
    # device handling
    if device is not None:
        try:
            # case: "[0,1]"
            if device.startswith('['):
                device = ast.literal_eval(device)
            # case: "0,1"
            elif ',' in device:
                device = [int(x) for x in device.split(',')]
            # case: "0"
            else:
                device =int(device)
        except Exception:
            raise ValueError(f"Invalid device format: {device}")
    # for windows and all other linux machines
    if not hostIsHopper and backend not in ('academic_ai', 'anthropic_api'):
        device=0 # default to device 0 for local machines, adjust if needed

    if backend == 'ollama':
        import subprocess
        import time
    
        #restart server with batch size
        def restart_ollama_with_batch(batch_size):
            global device
            if isLinux:
                import socket

                # 1. Kill any ollama serve processes owned by THIS user so no prior
                #    keep_alive instance holds the GPU we want.
                kill_result = subprocess.run(
                    ["pkill", "-u", str(os.getuid()), "-f", "ollama serve"],
                    capture_output=True
                )
                if kill_result.returncode == 0:
                    logger.LogText("Ollama: killed previous ollama serve process(es) owned by this user.",
                                   printToConsole=True)
                time.sleep(2)  # give the killed process time to release the GPU

                # find a free port (avoid 11434 which may be used by system service)
                def find_free_port(start=11435, end=11550):
                    for port in range(start, end):
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            if s.connect_ex(('127.0.0.1', port)) != 0:
                                return port
                    raise RuntimeError("No free port found for Ollama")
    
                port = find_free_port()
    
                # Set environment variables for new instance
                new_env = os.environ.copy()
                new_env["OLLAMA_NUM_PARALLEL"] = str(batch_size)
                if hostIsHopper:
                    new_env["OLLAMA_MODELS"] = "/opt/llm_models/ollama" # directory for shared access on pulled ollama models
                new_env["OLLAMA_NOPRELOAD"] = "1"
                new_env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
                # logging only ERROR-messages of ollama server
                new_env["OLLAMA_DEBUG"] = "ERROR"      # HTTP/API logs
                new_env["OLLAMA_LOG_LEVEL"] = "error"  # runner / ggml logs

                # 2. Restrict Ollama to the requested GPU(s) via CUDA_VISIBLE_DEVICES.
                #    None means no restriction — Ollama allocates GPUs automatically.
                if device is not None:
                    deviceString = ','.join(str(d) for d in device) if isinstance(device, list) else str(device)
                    new_env["CUDA_VISIBLE_DEVICES"] = deviceString
                    os.environ["CUDA_VISIBLE_DEVICES"] = deviceString
                else:
                    device = 'auto'
                logger.LogText(
                    f"Starting private Ollama server on port {port} "
                    f"with OLLAMA_NUM_PARALLEL={batch_size}, CUDA_VISIBLE_DEVICES={device}",
                    printToConsole=True
                )
    
                # IMPORTANT: also set it for THIS process so MyLLM connects correctly
                os.environ["OLLAMA_HOST"] = f"http://127.0.0.1:{port}"
    
                subprocess.Popen(
                        ["ollama", "serve"],
                        env=new_env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True
                    )
    
                time.sleep(5)
    
            else:  # Windows 
                logger.LogText("Ollama: restart server with given batch size",printToConsole=True)
    
                # 1. Kill any existing Ollama processes
                os.system("taskkill /f /im ollama.exe >nul 2>&1")
                time.sleep(2)
    
                # 2. Set the environment variable for the new process
                new_env = os.environ.copy()
                new_env["OLLAMA_NUM_PARALLEL"] = str(batch_size)
                new_env["OLLAMA_NOPRELOAD"] = "1"
                
                # 3. Start the server in the background
                # Note: This assumes ollama is in your PATH
                subprocess.Popen(["ollama", "serve"], env=new_env, 
                                creationflags=subprocess.CREATE_NEW_CONSOLE
                                )
    
                logger.LogText(f"Ollama: restarted with batch size={batch_size}",printToConsole=True)
                time.sleep(5) # Give it a few seconds to boot up
        
        if batchSize > 0:
            restart_ollama_with_batch(max(1,batchSize))
    

    logger.LogText(f"LLM Model Name: {shortName}",printToConsole=True)
    logger.LogText(f"Number of Randomizations: {numberOfRandomizations}",printToConsole=True)
    logger.LogText(f"Max Difficulty Level: {maxDifficultyLevel}",printToConsole=True)
    if doSpaceVariation: 
        logger.LogText(f"Do space variations: {doSpaceVariation}",printToConsole=True)
    logger.LogText(f"logging to {logDir}",printToConsole=True)
    
    if backend in ('academic_ai', 'anthropic_api'):
        contextWindowMaxSize = 0 # API call does not support limit of context window
    else: # all other backends can limit context window
        contextWindowMaxSize = int((2048+1024)*thinkingTokenFactor)


    logger.LogText(f"contextWindowMaxSize: {contextWindowMaxSize}",printToConsole=True)
    logger.LogText(f"thinkingMode: {thinkingMode}",printToConsole=True)
    logger.LogText(f"thinkingTokenFactor: {thinkingTokenFactor}",printToConsole=True)
    logger.LogText(f"batchSize: {batchSize}",printToConsole=True)
    logger.LogText(f"backend: {backend}",printToConsole=True)
    logger.LogText(f"device: {device}",printToConsole=True)
    logger.LogText(f"modelQuantization: {modelQuantization}",printToConsole=True)
    if backend == 'academic_ai':
        logger.LogText(f"apiAuthMethod: {apiAuthMethod}", printToConsole=True)
        logger.LogText(f"apiBaseURL: {apiBaseURL}", printToConsole=True)
        logger.LogText(f"apiTimeout: {apiTimeout}", printToConsole=True)
        logger.LogText(f"apiBudgetEUR: {apiBudgetEUR}", printToConsole=True)
    elif backend == 'anthropic_api':
        logger.LogText(f"apiBudgetEUR: {apiBudgetEUR}", printToConsole=True)
    logger.LogText(f"promptVariationCI: {promptVariationCI}",printToConsole=True)
    logger.LogText(f"promptVariationMBS: {promptVariationMBS}",printToConsole=True)
    logger.LogText(f"listVariation: {listVariation}",printToConsole=True)
    
    llmModel = MyLLM(modelName=llmModelName,
                     logger=logger,
                     contextWindowMaxSize = contextWindowMaxSize,
                     batchSize=max(1,batchSize),
                     backend=backend,
                     device=device,
                     modelQuantization=modelQuantization,
                     logPrompt=True,
                     logThinking=True,
                     pathToStoredLLMs=pathToStoredLLMs,
                     enable_thinking=enableThinking,
                     apiAuthMethod=apiAuthMethod,
                     apiClientID=None, # credentials for Academic AI are read from environment variables (see Readme)
                     apiClientSecret=None,
                     apiBaseURL=apiBaseURL,
                     apiTimeout=apiTimeout,
                     apiBudgetEUR=apiBudgetEUR if backend in ('academic_ai', 'anthropic_api') else None)
    
    if flexuAI:
        logger.LogText('FlexuAI: Selected task type: ' + flexuAITaskType, printToConsole=True, separator=True, quickinfo=True)
        # add task type also to descriptions
        AddTaskTypeToDescription(geometryDescriptions=geometryDescriptions,
                                 flexibleMBSDescriptions=flexibleMBSDescriptions)
        if flexuAITaskType == 'fMBS':
            fMBSModelLoader = ModelLoader()
            fMBSModelLoader.LoadStandardModelDescriptions(modelDescriptionsDict=flexibleMBSDescriptions,
                                                          specificModels=specificModelNames,
                                                          maxDifficultyLevel=maxDifficultyLevel,
                                                          onlyWithSampleFiles=True,
                                                          onlyParameterizedModels=True)
            partModelLoader = ModelLoader()
            partModelLoader.LoadStandardModelDescriptions(modelDescriptionsDict=geometryDescriptions,
                                                          maxDifficultyLevel=args.difficulty,
                                                          specificModels=None, # must be none in this case
                                                          onlyWithSampleFiles=True,
                                                          onlyParameterizedModels=True)
            mainModelLoader = fMBSModelLoader # the global loader is the fMBSModelLoader here
            # define the callback to extract parts out of fMBS descriptions
            evaluationTaskNameForPartCreation = 'generateCode' # this must be the name of the task in which evaluation is done for the part creation
            # this function takes care that no 'randomization' is done within the parts
            InitializationCallback = lambda *args: InitializationCallbackForPartsCreation(*args, partLoader=partModelLoader,
                                                                                          evaluationTaskNameForPartCreation=evaluationTaskNameForPartCreation,
                                                                                          logger=logger)
            taskSolutionLLMDir = 'fMBSLLM' # directory where the LLM generated things will be saved (per fMBS a folder is created)
            taskSolutionSampleDir = 'fMBSSample' # directory where the sample files will be saved; this is for comparison and evaluation
            helperFileNameForMainInference = 'flExudyn' # for the superior task
            helperFileNameForPreliminaryInference = 'NetgenOCC'
            taskNameForMainInference = 'createFMBS'
            taskNameForPreliminaryInference = 'createPart'
            specificHelperTagsListForMainInference = [] # here we can set specific helper tags for main inference
        elif flexuAITaskType == 'part':
            partModelLoader = ModelLoader()
            partModelLoader.LoadStandardModelDescriptions(modelDescriptionsDict=geometryDescriptions,
                                                          maxDifficultyLevel=maxDifficultyLevel,
                                                          specificModels=specificModelNames,
                                                          onlyWithSampleFiles=True,
                                                          onlyParameterizedModels=True)
            mainModelLoader = partModelLoader # the global loader is the partModelLoader here

            #++++++++++++++++++++++++++++++++++++++++++++++++++
            InitializationCallback = None
            taskSolutionLLMDir = 'partLLM' # directory where the LLM generated things will be saved
            taskSolutionSampleDir = 'partSample' # directory where the sample files will be saved; this is for comparison and evaluation
            helperFileNameForMainInference = 'NetgenOCC' # for the superior task
            helperFileNameForPreliminaryInference = None # no preliminary inference here
            taskNameForMainInference = 'createPart'
            specificHelperTagsListForMainInference = [] # here we can set specific helper tags for main inference
        else:
            logger.LogDebug('Specified task type is not available!', printToConsole=True, separator=True)
            sys.exit()
        evaluationTaskNameForMainTask = 'generateCode' # in FlexuAI pipeline, we do not differentiate between mbs and parts
    else:
        mbsModelLoader = ModelLoader()
        # level 1 and level 2 rigid MBS tasks; models are filtered by maxDifficultyLevel (-d 44 used in the paper: all models)
        mbsModelLoader.LoadStandardModelDescriptions(modelDescriptionsDict=simpleAndMediumMBSDescriptions,
                                                     specificModels=specificModelNames,
                                                     maxDifficultyLevel=maxDifficultyLevel,
                                                     onlyWithSampleFiles=True,
                                                     onlyParameterizedModels=True)
        mainModelLoader = mbsModelLoader
        evaluationTaskNameForMainTask = 'generateMBS'
        InitializationCallback = None
        taskSolutionLLMDir = 'mbsModelsLLM'
        taskSolutionSampleDir = 'mbsModelsSample'
    logger.LogText(f"number of MBS models: {mainModelLoader.NumberOfModels()}", printToConsole=True)
    agent = LabAgent(llmModel=llmModel, 
                     logger=logger,
                     agentOutputDataDir=logDir, 
                     helperFilesDict=helperFilesDict,
                     maxDifficultyLevel=maxDifficultyLevel) # here we also pass the difficulty level to look for corresponding tags
    agent.useExudynTimeout = True # general timeout
    agent.solverTimeoutSeconds = 120 # default=120; change if timeout happen often in test; increase to avoid timeouts for complex models (300 seconds)
    doSpaceVariation = 0
    if temperature is not None:
        agent.llmGenerateDefaultArgs['temperature'] = temperature
    if thinkingTokenFactor > 1:
        agent.maxTokensGenerateModel = int(agent.maxTokensGenerateModel*thinkingTokenFactor)
        if 'qwen3.6' in llmShortName.lower() and thinkingMode > 0: #sometimes stops generation ...
            agent.maxTokensGenerateModel = int(1.3334*agent.maxTokensGenerateModel)
        if 'claude' in llmShortName.lower(): # claude sometimes still exceeds max_tokens during model generation
            agent.maxTokensGenerateModel = int(1.3334*agent.maxTokensGenerateModel)

        agent.maxTokensChooseItems = agent.maxTokensGenerateModel #just use the same, as thinking can be long during item selection
        logger.LogText(f'Increased maxTokens:\n  maxTokensChooseItems={agent.maxTokensChooseItems}\n  maxTokensGenerateModel={agent.maxTokensGenerateModel}', printToConsole=True)


    if flexuAI:
        agent.isFlexuAI = True
        agent.flexuAITaskType = flexuAITaskType

    
    #%% initialize global data (since FlexuAI 'lives' in the normal pipeline, this is all the same for both pipelines)
    globalData = InitializeTaskProcessing(agent,
                                          mbsModelLoader=mainModelLoader,
                                          numberOfRandomizations=numberOfRandomizations,
                                          doSpaceVariation=doSpaceVariation, #instead of model randomizations
                                          evaluationTaskName=evaluationTaskNameForMainTask,
                                          taskSolutionLLMDir=taskSolutionLLMDir,
                                          taskSolutionSampleDir=taskSolutionSampleDir,
                                          CallbackForPreliminaryTask=InitializationCallback)

    #%% MAIN TASKS (with a few differences between 'normal' and FlexuAI pipeline)
    if not flexuAI:
        chooseItemsTask = ChooseItemsTask(globalData, agent.maxTokensChooseItems,
                                            helperFileName='exudyn', agent=agent, 
                                            descriptiveName = 'choose items for MBS',
                                            promptVariation=promptVariationCI,
                                            listVariation=listVariation)
        currentIDs = chooseItemsTask.Run()
        if not skipGenerateMBS: # full pipeline process (choosing items and generate MBS code)
            generateMBStask = GenerateMBSTask(globalData, agent.maxTokensGenerateModel, 
                                                helperFileName='exudyn', agent=agent, 
                                                descriptiveName = 'generate MBS code',
                                                promptVariation=promptVariationMBS)
            currentIDs = generateMBStask.Run(currentIDs) #run task with given (successful IDs)
        else: 
            # copy chooseItemsMatch into the generateMBS evaluation dict (which is pre-initialized with defaults)
            for taskData in globalData['storedDataPerTask'].values():
                if 'chooseItems' in taskData and 'chooseItemsMatch' in taskData['chooseItems']:
                    taskData['generateMBS']['chooseItemsMatch'] = taskData['chooseItems']['chooseItemsMatch']
    else: # FlexuAI specific settings
        # add special ExuAI-in and ExuAI-out dict that is needed for flexuAI pipeline
        import exudyn as exu
        exu.sys['exuAI_in'] = {}
        exu.sys['exuAI_in']['maxHFactor'] = flexuAIMaxHFactor
        exu.sys['exuAI_in']['meshOrder'] = flexuAIMeshOrder
        exu.sys['exuAI_in']['nEigenModes'] = 6 # how many modes should we compute for reduction?
        exu.sys['exuAI_in']['nEigenModesToCompare'] = 3 # how many modes should we compare? (must be smaller/equal to/than nEigenModes)
        exu.sys['exuAI_in']['grading'] = flexuAIMeshGrading
        exu.sys['exuAI_in']['curvatureSafety'] = flexuAIMeshCurvatureSafety
        exu.sys['exuAI_in']['maxNumberOfFemNodes'] = 4e4
        logger.LogDict(exu.sys['exuAI_in'], header='ExuAI Input Dict')

        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # RESTRUCTURE, and PROCESS PRELIMINARY tasks
        globalDataForPartCreation = RestructureGlobalDataAndGetGlobalDataForPartCreation(globalData, logger=logger)
        PrintHeadlineToConsole(text='Starting part creation')
        logger.LogText('STARTING PART CREATION', printToConsole=False, separator=True, quickinfo=True)
        if globalDataForPartCreation:
            # system tags are added automatically in agent.GetCodeGenerationPrompt
            FlexuAIChooseItemsAndGenerateCode(agent, globalDataForPartCreation,
                                              helperFileName=helperFileNameForPreliminaryInference,
                                              taskName=taskNameForPreliminaryInference,
                                              specificHelperTagsList=[]) # no specific helper tags for part creation
            logger.LogText('Preliminary tasks finished with\n'+
                            'number of exec. samples: ' + str(globalDataForPartCreation['nExecutabilitySamples']) + '\n' +
                            'number of samples: ' + str(globalDataForPartCreation['nCompletedSamples']) + '\n'+
                            'number of exec. LLM codes: ' + str(globalDataForPartCreation['nExecutabilityMainTasks']) + '.',
                            printToConsole=True, separator=True)
            PrintHeadlineToConsole(text='Starting fMBS creation')
            logger.LogText('STARTING FMBS CREATION', printToConsole=False, separator=True, quickinfo=True)
            globalData = SkipFMBSDueToPartsNotBeingCreatedCorrectly(agent, logger, globalData=globalData, evaluationTaskNameForPartCreation=evaluationTaskNameForPartCreation)

        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # PROCESS MAIN TASKS
        globalData =  FlexuAIChooseItemsAndGenerateCode(agent, globalData, helperFileName=helperFileNameForMainInference,
                                                        taskName=taskNameForMainInference,
                                                        specificHelperTagsList=specificHelperTagsListForMainInference)

    #%% post processing (same for 'normal' pipeline and FlexuAI)
    if flexuAI and flexuAITaskType == 'part' or skipGenerateMBS:
        pass
    else:
        globalData = ExtractMBSGraphContentErrors(agent, globalData, evaluationTaskName=evaluationTaskNameForMainTask)
    globalData = FinalizeTaskProcessingNEW(agent, globalData, evaluationTaskName=evaluationTaskNameForMainTask)
    # write results, create summary
    LogOverallCounter2File(filePath='_globalTokenCount.txt', deltaCounter=llmModel.numberOfTokensGlobal+llmModel.numberOfThinkingTokensGlobal)
    LogOverallCounter2File(filePath='_globalPromptTokenCount.txt', deltaCounter=llmModel.numberOfPromptTokensGlobal)
    if llmModel.backend in ('academic_ai', 'anthropic_api'):
        backendLabel = 'Academic AI' if llmModel.backend == 'academic_ai' else 'Anthropic API'
        logger.LogText(
            f"{backendLabel} total cost: {llmModel.apiCostGlobalEUR:.6f} EUR | "
            f"remaining budget: {llmModel.apiBudgetRemainingEUR:.6f} EUR",
            printToConsole=True,
            separator=True,
            quickinfo=True,
        )
    WriteDictToJSON(globalData, JoinPath(logDir,'results.json'))
    llmModel.FreeMemory()
    mergedResultsFilePath = JoinPath(destinationFolder, 'allLLMresults.json')
    resultsCSVfilePath = JoinPath(destinationFolder, 'overallResults.csv')
    modelSuccessRatesCSVfilePath = JoinPath(destinationFolder, 'overallResultsSuccessRates.csv')
    overallSuccessVsReleaseDatePlotPath = JoinPath(destinationFolder, "overallSuccessVsReleaseDatePlot.pdf")
    chooseItemsVsReleaseDatePlotPath = JoinPath(destinationFolder, "chooseItemsVsReleaseDatePlot.pdf")
    correlationPlotPath = JoinPath(destinationFolder, "correlation.pdf")
    try:
        mergedResults = MergeTestModelResults(dirTestModels=destinationFolder,
                                              outputJSONpath=mergedResultsFilePath,
                                              logger=logger)
        # Export overall results to CSV using the csvKeys
        ExportMergedResultsToCSV(mergedJsonFilePath=mergedResultsFilePath, outputCSVpath=resultsCSVfilePath,
                                 logger=logger)
        # export model success rates
        ExportModelSuccessRatesToCSV(mergedJsonFilePath=mergedResultsFilePath, outputCSVpath=modelSuccessRatesCSVfilePath,
                                     logger=logger)
        # plot LLM performance (overall success) in MBS generation progress over time (release dates)
        PlotPerformanceVsReleaseDate(mergedResultsFilePath, overallSuccessVsReleaseDatePlotPath,
                                     logger=logger, metricKey='overall success')
        # plot LLM performance (choose items) in MBS generation progress over time (release dates)
        PlotPerformanceVsReleaseDate(mergedResultsFilePath, chooseItemsVsReleaseDatePlotPath,
                                     logger=logger, metricKey='choose items')
        # plot correlation between two metrics in generation results over different LLMs (e.g. temperature and overall success)
        PlotMetricCorrelationPerLLM(mergedJsonFilePath=mergedResultsFilePath, 
                                keyX='temperature', keyY='overall success', # change keys to analyze other metric-correlations
                                outputPlotPath=correlationPlotPath,
                                logger=logger)
    except KeyboardInterrupt:
        print("Keyboard interrupt requested...exiting")
        raise
    except:
        exceptionText = traceback.format_exc()
        logger.LogDebug('ERROR in Merging and plots:\ntraceback='+exceptionText,
                        printToConsole=True)

    if len(agent.samplesErrors):
        logger.LogText('SAMPLE DICTS / FILES raised the following ERRORS:',printToConsole=True)
        for errMsg in agent.samplesErrors:
            logger.LogText('  '+errMsg, printToConsole=True, timestamp=False)
        logger.LogText(f'There were {len(agent.samplesErrors)} ERRORS in SAMPLE DICTS / FILES',printToConsole=True)
        
