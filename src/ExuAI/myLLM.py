# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: myLLM.py
#
# Details:  The MyLLM class is intended for interacting with large language models (LLMs) of different kinds with one interface.
#           It includes functionalities for loading LLMs, to generate text, and to log at different levels;
#           Supported backends: Ollama, Anthropic API, Academic AI (OpenAI-compatible API), HF Transformers,
#           vLLM, GPT4All and llama.cpp.
#           It also includes procedures to extract special tokens (thinking, etc.); and logs token counts and performance.
#           The llmModelsDict includes the list of models (with short names) that can be used in the tests.
#
# Authors:  Tobias Möltner, Johannes Gerstmayr 
# Date:     2024-09-18
#
# License:  BSD-3 license
#
# Installation:
#           Ollama backend: pip install ollama; Anthropic backend: pip install anthropic;
#           other backends: transformers / torch, vllm, gpt4all[cuda], llama-cpp-python
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++
import sys
import os
import time
import zlib

try: 
    from .logger import Logger
    from .utilities import JoinPath, SecondsRemaining, Seconds2HHMMSS
except: 
    from logger import Logger
    from utilities import JoinPath, SecondsRemaining, Seconds2HHMMSS
    
isLinux = 'linux' in sys.platform #win32 is windows, darwin is MacOS!

#%%+++++++++++++++++++++++++
# Registry of LLMs: key = model name used by the backend (Ollama tag, HF repo, gguf file name, API name)
#   type: backend type ('ollama', 'huggingface', 'gguf', 'vllm', 'API')
#   VRAM: approx. memory in GB (for Ollama: model size); nParam: number of parameters in billion;
#   Q: quantization (bits); ctxSize: max. context size in k tokens; shortName: name used on the command line
#   and in result folders (must be a valid file name); releaseDate: release / upload date (YYYY-MM);
#   defaultTemperature: default sampling temperature of the model (-1 = unknown)
# The list contains all LLMs evaluated in the paper and one example for each further backend;
# further models can be added in the same way.
llmModelsDict = {}

llmAPIModelsDict = {
    # apiSupportsTemperature: newer Claude models removed temperature/top_p/top_k and reject
    #   them with HTTP 400; set to True only for models that still accept sampling parameters.
    # apiSupportsVision: model accepts image input (base64 image blocks).
    # Price estimations from https://platform.claude.com/docs/de/about-claude/pricing (17.06.2026)
    'Claude-Opus-4.8': {
        'VRAM':-1, 'nParam':-1, 'Q':-1, 'ctxSize':-1,
        'shortName':'Claude-Opus-4.8', 'quality':0,
        'apiModel':'claude-opus-4-8',
        'priceInputEURper1M':4.50, 'priceOutputEURper1M':22.00,
        'releaseDate':'2026-05',
        'apiSupportsTemperature':False, 'apiSupportsVision':True,
    },
    # Prices copied from provided Academic AI pricing table (EUR per 1M tokens) (11.06.2026)
    'GPT-5.2': {
        'VRAM':-1, 'nParam':-1, 'Q':-1, 'ctxSize':-1,
        'shortName':'GPT-5.2', 'quality':0,
        'apiModel':'gpt-5.2',
        'priceInputEURper1M':1.75, 'priceOutputEURper1M':14.00,
        'releaseDate':'2025-12', 
    },
}

for key, value in llmAPIModelsDict.items():
    value['type'] = 'API'
    value.setdefault('defaultTemperature', 1.0) #default sampling temperature used for this API model
    llmModelsDict[key] = value

llmModelsDict.update({
    #+++++++++++++++++++++++++ Ollama models (used in the paper) +++++++++++++++++++++++++
    # VRAM estimation through ollama model size in GB
    'phi4:latest': {'type':'ollama', 'VRAM':8.18, 'nParam':14.7, 'Q':4, 'ctxSize':16, 'shortName':'Phi4-OL', 'releaseDate':'2024-12', 'quality':64},
    'phi3:14b': {'type':'ollama', 'VRAM':7.9, 'nParam':14, 'Q':4, 'ctxSize':128, 'shortName':'Phi3-OL', 'releaseDate':'2024-04', 'quality':0},
    'qwen:32b': {'type':'ollama', 'VRAM':18.0, 'nParam':32, 'Q':4, 'ctxSize':32, 'shortName':'Qwen-32B-OL', 'releaseDate':'2023-09', 'quality':0},
    'qwen:72b': {'type':'ollama', 'VRAM':41.0, 'nParam':72, 'Q':4, 'ctxSize':32, 'shortName':'Qwen-72B-OL', 'releaseDate':'2023-09', 'quality':0},
    'codeqwen:7b': {'type':'ollama', 'VRAM':4.2, 'nParam':7.25, 'Q':4, 'ctxSize':64, 'shortName':'CodeQwen-7B-OL', 'releaseDate':'2024-04', 'quality':0},
    'qwen2:7b': {'type':'ollama', 'VRAM':4.4, 'nParam':7, 'Q':4, 'ctxSize':32, 'shortName':'Qwen2-7B-OL', 'releaseDate':'2024-07', 'quality':0},
    'qwen2:72b': {'type':'ollama', 'VRAM':41.0, 'nParam':72, 'Q':4, 'ctxSize':32, 'shortName':'Qwen2-72B-OL', 'releaseDate':'2024-07', 'quality':0},
    'qwen2.5-coder:32b': {'type':'ollama', 'VRAM':20, 'nParam':32, 'Q':4, 'ctxSize':32, 'shortName':'QwenCoder2.5-32B-OL', 'releaseDate':'2024-09', 'quality':0},
    'qwen3-coder:30b': {'type':'ollama', 'VRAM':19.0, 'nParam':30, 'Q':4, 'ctxSize':256, 'shortName':'QwenCoder3-30B-OL', 'releaseDate':'2025-07', 'quality':0, 'defaultTemperature':0.7},
    'mdq100/qwen3.5-coder:35b': {'type':'ollama', 'VRAM':24.0, 'nParam':35, 'Q':4, 'ctxSize':128, 'shortName':'QwenCoder3.5-35B-OL', 'releaseDate':'2026-02', 'quality':0, 'defaultTemperature':0.6},
    'qwen3.5:4b': {'type':'ollama', 'VRAM':3.4, 'nParam':4.7, 'Q':4, 'ctxSize':256, 'shortName':'Qwen3.5-4B-OL', 'releaseDate':'2026-02', 'quality':0, 'defaultTemperature':1},
    'qwen3.5:9b': {'type':'ollama', 'VRAM':6.6, 'nParam':9.65, 'Q':4, 'ctxSize':256, 'shortName':'Qwen3.5-9B-OL', 'releaseDate':'2026-02', 'quality':0, 'defaultTemperature':1},
    'qwen3.5:27b': {'type':'ollama', 'VRAM':17.0, 'nParam':28, 'Q':4, 'ctxSize':256, 'shortName':'Qwen3.5-27B-OL', 'releaseDate':'2026-02', 'quality':0, 'defaultTemperature':1},
    'qwen3.5:35b': {'type':'ollama', 'VRAM':24.0, 'nParam':35, 'Q':4, 'ctxSize':128, 'shortName':'Qwen3.5-35B-OL', 'releaseDate':'2026-02', 'quality':0, 'defaultTemperature':1},
    'qwen3.5:122b': {'type':'ollama', 'VRAM':81.0, 'nParam':122, 'Q':4, 'ctxSize':256, 'shortName':'Qwen3.5-122B-OL', 'releaseDate':'2026-02', 'quality':0, 'defaultTemperature':1},
    'qwen3.6:27b': {'type':'ollama', 'VRAM':17.0, 'nParam':28, 'Q':4, 'ctxSize':256, 'shortName':'Qwen3.6-27B-OL', 'releaseDate':'2026-04', 'quality':0, 'defaultTemperature':1},
    'qwen3.6:35b': {'type':'ollama', 'VRAM':24.0, 'nParam':35, 'Q':4, 'ctxSize':256, 'shortName':'Qwen3.6-35B-OL', 'releaseDate':'2026-04', 'quality':0, 'defaultTemperature':1},
    'gemma4:12b': {'type':'ollama', 'VRAM':7.4, 'nParam':11.9, 'Q':4, 'ctxSize':256, 'shortName':'Gemma4-12B-OL', 'releaseDate':'2026-04', 'quality':0, 'defaultTemperature':1},
    'gemma4:26b': {'type':'ollama', 'VRAM':18, 'nParam':25.8, 'Q':4, 'ctxSize':256, 'shortName':'Gemma4-26B-OL', 'releaseDate':'2026-04', 'quality':0, 'defaultTemperature':1},
    'gemma4:31b': {'type':'ollama', 'VRAM':20, 'nParam':31.3, 'Q':4, 'ctxSize':256, 'shortName':'Gemma4-31B-OL', 'releaseDate':'2026-04', 'quality':0, 'defaultTemperature':1},
    'codegemma:7b': {'type':'ollama', 'VRAM':5, 'nParam':7, 'Q':4, 'ctxSize':8, 'shortName':'codegemma-7B-OL', 'releaseDate':'2024-08', 'quality':0},
    'gpt-oss:20b': {'type':'ollama', 'VRAM':14, 'nParam':20, 'Q':4, 'ctxSize':128, 'shortName':'gpt-oss-20B-OL', 'releaseDate':'2025-08', 'quality':0, 'defaultTemperature':1},
    'gpt-oss:120b': {'type':'ollama', 'VRAM':65, 'nParam':120, 'Q':4, 'ctxSize':128, 'shortName':'gpt-oss-120B-OL', 'releaseDate':'2025-08', 'quality':0, 'defaultTemperature':1},
    'llama3.1:8b': {'type':'ollama', 'VRAM':4.9, 'nParam':8, 'Q':4, 'ctxSize':128, 'shortName':'Llama3.1-8B-OL', 'releaseDate':'', 'quality':0},
    'llama3.1:70b': {'type':'ollama', 'VRAM':43, 'nParam':70.6, 'Q':4, 'ctxSize':128, 'shortName':'Llama3.1-70B-OL', 'releaseDate':'2024-07', 'quality':0},
    'llama3.3:70b': {'type':'ollama', 'VRAM':43, 'nParam':70.6, 'Q':4, 'ctxSize':128, 'shortName':'Llama3.3-70B-OL', 'releaseDate':'2024-12', 'quality':0},
    'deepseek-r1:32b': {'type':'ollama', 'VRAM':20, 'nParam':32, 'Q':4, 'ctxSize':128, 'shortName':'deepseek-r1-32B-OL', 'releaseDate':'', 'quality':0},
    'deepseek-r1:70b': {'type':'ollama', 'VRAM':43, 'nParam':70, 'Q':4, 'ctxSize':128, 'shortName':'deepseek-r1-70B-OL', 'releaseDate':'', 'quality':0},
    'deepseek-coder-v2:latest': {'type':'ollama', 'VRAM':8.9, 'nParam':16, 'Q':4, 'ctxSize':160, 'shortName':'DeepSeekCoder-16B-OL', 'releaseDate':'', 'quality':0},
    'dolphincoder:15b': {'type':'ollama', 'VRAM':9.1, 'nParam':15, 'Q':4, 'ctxSize':16, 'shortName':'dolphincoder-15B-OL', 'releaseDate':'2024-05', 'quality':0},
    'ministral-3:14b': {'type':'ollama', 'VRAM':9.1, 'nParam':14, 'Q':4, 'ctxSize':256, 'shortName':'ministral-3-14B-OL', 'releaseDate':'', 'quality':0, 'defaultTemperature':0.15},
    'nemotron-cascade-2:30b': {'type':'ollama', 'VRAM':24, 'nParam':30, 'Q':4, 'ctxSize':256, 'shortName':'nemotron-cascade-2-30B-OL', 'releaseDate':'', 'quality':0, 'defaultTemperature':1},
    'nemotron3:33b': {'type':'ollama', 'VRAM':28, 'nParam':33, 'Q':4, 'ctxSize':128, 'shortName':'nemotron3-33B-OL', 'releaseDate':'2025-10', 'quality':0},
    'glm-4.7-flash:q4_K_M': {'type':'ollama', 'VRAM':19, 'nParam':30, 'Q':4, 'ctxSize':198, 'shortName':'glm-4.7-flash-30B-OL', 'releaseDate':'', 'quality':0, 'defaultTemperature':1},

    #+++++++++++++++++++++++++ examples for further backends (not used in the paper) +++++++++++++++++++++++++
    'qwen2.5-coder-32b-instruct-q4_0.gguf': {'type':'gguf', 'VRAM':17.75, 'nParam':32, 'Q':4, 'ctxSize':128, 'shortName':'QwenCoder2.5-32B-Q4', 'releaseDate':'2024-09', 'quality':64},
    'Qwen/Qwen2.5-Coder-32B-Instruct': {'type':'huggingface', 'VRAM':79.68, 'nParam':32, 'Q':16, 'ctxSize':32, 'shortName':'QwenCoder2.5-32B-HF', 'releaseDate':'2024-09', 'quality':0},
    # vLLM: keys carry a 'vllm:' prefix; 'hfRepo' is what vLLM actually loads
    'vllm:openai/gpt-oss-20b': {'type':'vllm', 'hfRepo':'openai/gpt-oss-20b', 'VRAM':14, 'nParam':20, 'Q':4, 'ctxSize':128, 'shortName':'gpt-oss-20B-VLLM', 'releaseDate':'2025-08', 'quality':0, 'defaultTemperature':1},
    })

# defaultTemperature = -1 marks models where the default temperature is not known
for keyModel, valueModel in llmModelsDict.items():
    valueModel.setdefault('defaultTemperature', -1)

#! @brief Strip the organization/family prefix from a llmModelsDict key, keeping the
#         model-specific remainder verbatim, e.g. 'Qwen/Qwen2.5-Coder-32B-Instruct' -> 'Qwen2.5-Coder-32B-Instruct'
#! @param fullName full model name (key in llmModelsDict)
#! @return model-specific name without organization prefix
def ConvertFullName2modelKeyName(fullName):
    return fullName.split('/')[-1]


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
typicalLLMstopTags= ['<|im_end|>', '<|endoftext|>']
#callback for GPT4All
localGenerateTokenCount = 0
localGenerateTokenCountStr = ''
def TokenCallback(token_id, token_string):
    global localGenerateTokenCount, localGenerateTokenCountStr, typicalLLMstopTags
    localGenerateTokenCount += 1
    localGenerateTokenCountStr += token_string

    for stop_string in typicalLLMstopTags + ['</score>']: #stop when score received ...
        if stop_string in localGenerateTokenCountStr[-50:]:
            return False

    return True

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class MyLLM:
    def __init__(self, modelName, pathToStoredLLMs='~/.cache/huggingface/hub',
                 modelQuantization=None, device=None, nThreads=None, logger=None, 
                 logPrompt = False, 
                 logThinking = False,
                 contextWindowMaxSize = 2048, #for gpt4all only #for Exudyn model generation >2048
                 useTokenCallback = True, #to count tokens and to catch endstrings; needed for some models!
                 backend = None, # choose llm inference backend
                 quantization_config = None,
                 batchSize = 1, #set batch size (hf transformers only)
                 enable_thinking = False,
                 apiAuthMethod = 'bearer',
                 apiToken = None,
                 apiClientID = None,
                 apiClientSecret = None,
                 apiBaseURL = 'https://ca-backend-uibkai-prod.wittystone-43265e4a.swedencentral.azurecontainerapps.io',
                 apiTimeout = 30,
                 apiBudgetEUR = None,
                 ):
        """
        Initialize the LLM model with corresponding promptTemplate. Uses either GPU(s) or CPU.

        Args:
            modelName (str): Name or path of the pre-trained model.
            nThreads (int): number of threads to be used in cpu mode.
            logger: either none or a exuai.logger class Logger
            device (Union[int, list[int]]): None (select automatically), -1 (cpu), integer (GPU device) or list of GPU devices
            verbose (bool): enable logging info 
            debug (bool): enable logging on debug level
            modelQuantization (str): Hugging Face model quantization (4bit, 8bit, fp16, bf16).
            backend (str): backend for inference, either 'gpt4all', 'hftransformers', 'llamacpp' or 'ollama'
            quantization_config (dict): optional quantization configuration for HuggingFace transformers backend BitsAndBytesConfig
        """
        # moved this output here since everytime a spawn process re-loads myLLM this appeared in the console
        if isLinux: 
            print('running MyLLM on linux')
        # reset global llm-parameter (these parameters track token-specific information of a LLM during runtime between initializations)
        self.numberOfTokensGlobal = 0
        self.numberOfThinkingTokensGlobal = 0
        self.numberOfPromptTokensGlobal = 0 #=input tokens
        
        self.numberOfTokensLocal = 0 
        self.numberOfThinkingTokensLocal = 0
        self.numberOfPromptTokensLocal = 0
        
        self.numberOfRemovedTokensGlobal = 0
        self.tokensPerSecondGlobal = 0 #includes thinking (for ollama)
        self.durationGlobal = 0
        self.numberOfGenerateCalls = 0
        self.apiCostLocalEUR = 0.0
        self.apiCostGlobalEUR = 0.0
        self.apiBudgetEUR = apiBudgetEUR
        self.apiBudgetRemainingEUR = -1.0 if apiBudgetEUR is None else float(apiBudgetEUR)
        self.apiInputTokensGlobal = 0
        self.apiOutputTokensGlobal = 0
        self.apiInputTokensLocal = 0
        self.apiOutputTokensLocal = 0
        self.apiSupportsTemperature = True #set per model for API backends; newer Claude models reject temperature
        self.contextWindowExceededLocal = False #True if context window was exceeded in last single inference
        self.contextWindowExceededList = [] #list of booleans, one per prompt, set after Generate()
        self.contextWindowExceededTokensLocal = 0 #tokens by which context window was exceeded in last inference (0 if not exceeded)
        self.contextSizeUsedLocal = 0 #prompt + generated tokens of last inference (compared against contextWindowMaxSize)
        self.contextWindowExceededTokensList = [] #list of ints per prompt: tokens exceeded (0 if not exceeded), set after Generate()
        self.contextSizeUsedList = [] #list of ints per prompt: prompt + generated tokens, set after Generate()
        self.numberOfPromptTokensList = [] #list of ints per prompt: prompt tokens, set after Generate()
        self.logPrompt = logPrompt
        self.logThinking = logThinking
        self.contextWindowMaxSize = contextWindowMaxSize
        self.useTokenCallback = useTokenCallback
        self.pathToStoredLLMs = pathToStoredLLMs # default path 
        self.batchSize = batchSize
        self.modelQuantizationHF = modelQuantization
        self.modelName = modelName
        self.enable_thinking = enable_thinking
        self.apiAuthMethod = apiAuthMethod
        self.apiBaseURL = apiBaseURL.rstrip('/')
        self.apiTimeout = apiTimeout
        self.apiToken = apiToken if apiToken is not None else os.getenv('BEARER_TOKEN')
        self.apiClientID = apiClientID if apiClientID is not None else os.getenv('CLIENT_ID')
        self.apiClientSecret = apiClientSecret if apiClientSecret is not None else os.getenv('CLIENT_SECRET')
        self.apiModelName = modelName
        self.apiPriceInputEURper1M = 0.0
        self.apiPriceOutputEURper1M = 0.0

        #options to track memory of specific gpu (adjust gpuMemTrackerDeviceIndex)
        self.peakGPUmemory = 0  #for all inferences; updated in every call to TorchGPUmemory()
        self.gpuMemTrackerDeviceIndex = 0 #adapt to your needs!
        self.TorchGPUmemoryReset()
        self.startInfoGPU = self.TorchGPUmemory() #info retrieved at start (HF)
        self.currentInfoGPU = self.startInfoGPU   #info after last batched inference (HF)
        
        # Check if this is a vision model (on the HF repo name: 'vllm:' registry keys contain 'vl')
        repoNameLower = llmModelsDict.get(modelName, {}).get('hfRepo', modelName).lower()
        self.isVisionModel = any(indicator in repoNameLower for indicator in [
            'vl', 'vision', 'llava', 'qwen2.5-vl', 'pixtral', 'gemma-3', 'gemma-4', 'qwen3.5', 'qwen3.6', 'ministral'])
    
        # Expand user path for Windows compatibility
        expandedPath = os.path.expanduser(self.pathToStoredLLMs)

        if (modelName.startswith('phi-4')
            or modelName.startswith('Viper-Coder')
            or modelName.startswith('FluentlyLM')
            or 'Qwen2.5' in modelName
            ):
            self.useTokenCallback = True #for this model, it is needed!

        
        self.logger = logger
        if self.logger is None:
            self.logger = Logger('LLMlog')
        
        try:
            from gpt4all import GPT4All
        except:
            self.logger.LogDebug('MyLLM Warning: did not find gpt4all Python package')

        deviceName = self.startInfoGPU['GPUmodel']
            
        # Set up backend
        self.possibleBackends = ['gpt4all', 'hftransformers', 'llamacpp', 'ollama', 'academic_ai', 'anthropic_api',
                                 'vllm']
        self.logger.LogDebug("Found model file at: " + expandedPath, printToConsole=True)
        self.logger.LogDebug(f"Initializing MyLLM with model name: {modelName}, device: {device}, GPU: {deviceName}, number of threads: {nThreads}", printToConsole=True)
        if backend.lower() not in self.possibleBackends:
            self.logger.LogText(f"LLM: Invalid backend provided! Possible backends: {self.possibleBackends}")
            raise ValueError(f"LLM: Invalid backend provided! Possible backends: {self.possibleBackends}")
        else:
            self.backend = backend.lower()
            self.logger.LogText(f"LLM: Used backend: {self.backend}")

        if self.useTokenCallback:
            self.logger.LogText("LLM: Using TokenCallback for gguf models")
        
        templatePromptList = ["prompt"]
        templateSystemPrompt = "systemPrompt"
        templateImgPathList = ["imagePath/"]
        promptTemplate = self.FormatChatPrompt(templatePromptList, 
                                               templateSystemPrompt, 
                                               templateImgPathList if self.isVisionModel else [])
        if isinstance(promptTemplate, list):
            promptTemplate = str(promptTemplate[0])
            if promptTemplate == 'prompt':
                promptTemplate = 'None'
        self.logger.LogDebug("LLM: Used prompt template: " + promptTemplate, printToConsole=True) # only one prompt used here (promptTemplate is a list)
        ## CPU core-handling
        #  siehe exudyn RL bsp core/threads abfragen (>nthreads)
        if nThreads is None:
            #use number of cores, which always works well
            self.nThreads = max(1,int(os.cpu_count()/2)) 
        else:
            self.nThreads = nThreads
            
        ## device handling
        numGPUs = 0
        self.deviceMap = None
        
        
        if self.backend == 'gpt4all':  
            if device is None:
                self.device = None # choose automatically (first trying to select GPU)
                if isLinux: os.environ["CUDA_VISIBLE_DEVICES"] = str(0) # default set visible device 0 
                self.logger.LogDebug("GPT4All: Selecting CPU / GPU automatically")
            elif device == -1  or device=='cpu':
                self.device = 'cpu'  # choose CPU
                self.logger.LogDebug(f"GPT4All: Using CPU with {self.nThreads} threads")
            elif isinstance(device, int) or device=='cuda' or isinstance(device, list):
                # always choose gpu 0 if arg 'cuda' is provided (no multiGPU implemented in GPT4All (yet))
                if isinstance(device, list) or (isinstance(device, int) and device > 0): 
                    device = 0 
                    self.logger.LogWarning('GPT4All does not support device>0 or list, using GPU 0')
                if device=='cuda': 
                    device = 0 
                if isLinux: os.environ["CUDA_VISIBLE_DEVICES"] = str(device) 
                self.device = 'cuda'
                self.logger.LogDebug("GPT4All: Using GPU with cuda backend")
            else:
                raise ValueError('MyLLM: illegal device: '+str(device))
        elif self.backend == 'hftransformers':
            if device is None: # automatically choose GPU and CPU (if necessary)
                self.device = None
                self.deviceMap = "auto"
                self.logger.LogDebug("Selecting torch cuda device automatically and CPU (if needed)")

                # Log used GPUs
                # uses ALL CUDA-devices!!! 
                try:
                    import torch
                    numGPUs = torch.cuda.device_count() # this command sets all CUDA devices visible
                except ImportError as e:
                    self.logger.LogDebug(f'MyLLM Warning: did not find torch Python package: {e}')
                if numGPUs > 0:
                    # List of visible devices
                    visibleDevices = ",".join(str(i) for i in range(numGPUs))

                    # Set the detected GPUs as visible
                    os.environ["CUDA_VISIBLE_DEVICES"] = visibleDevices
                    for i in range(numGPUs):
                        self.logger.LogText(f"Used GPU {i}: {torch.cuda.get_device_name(i)}")
                else:
                    self.logger.LogText("Attempting auto-selection of CUDA device, but none found.")
            elif device == -1 or device == 'cpu':
                self.device = -1
                self.logger.LogText(f"Using CPU with {self.nThreads} threads")
                self.deviceMap = None
            elif isinstance(device, int): # single specific gpu provided
                self.device = device
                self.deviceMap = None

                os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
                try:
                    import torch
                    self.logger.LogText(f"Selected GPU {device}: {torch.cuda.get_device_name(device)}")
                except ImportError as e:
                    raise ImportError(f'MyLLM Error: Missing required packages: {e}')
                    
            elif isinstance(device, list): # list of specific gpus provided
                self.device = None
                self.deviceMap = "auto"

                visibleDevices = ",".join(map(str, device))
                os.environ["CUDA_VISIBLE_DEVICES"] = visibleDevices
                numGPUs = len(device)

                try:
                    import torch
                    for i in range(numGPUs):
                        self.logger.LogText(f"Used GPU {i-1}: {torch.cuda.get_device_name(i-1)}")
                except ImportError as e:
                    raise ValueError(f'MyLLM Error: device specified as list, but torch not found: {e}')
                
            else:
                raise ValueError('MyLLM: illegal device: '+str(device)+' Provide single integer for single GPU or None for multi GPU usage')

        elif self.backend == 'ollama':
            # GPU selection for Ollama is handled by the driver (modelBuilderDriver.py) via
            # CUDA_VISIBLE_DEVICES set in the environment of the spawned 'ollama serve' process.
            # MyLLM does not touch CUDA_VISIBLE_DEVICES here so Ollama can allocate GPUs freely.
            self.device = None
            self.deviceMap = None

        elif self.backend == 'vllm':
            # vLLM owns device placement itself (it pre-allocates a fixed fraction of VRAM for
            # weights plus its paged KV cache), so MyLLM must not set a device or device_map.
            # A device list selects the GPUs and shards the model across them with tensor
            # parallelism (e.g. [0,1] on the two NVLink-coupled H100s); this frees VRAM for KV
            # cache and thus raises the number of sequences vLLM can decode concurrently.
            self.device = None
            self.deviceMap = None
            #! a single int device is deliberately left alone (as before), so an externally set
            #! CUDA_VISIBLE_DEVICES keeps selecting the GPU (sketch2sim passes device=0)
            self.vllmTensorParallelSize = 1
            if isinstance(device, list):
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device))
                self.vllmTensorParallelSize = len(device)

        elif self.backend == 'academic_ai':
            self.device = None
            self.deviceMap = None

            if self.modelName in llmModelsDict:
                modelInfo = llmModelsDict[self.modelName]
                self.apiModelName = modelInfo.get('apiModel', modelInfo.get('shortName', self.modelName))
                self.apiPriceInputEURper1M = float(modelInfo.get('priceInputEURper1M', 0.0) or 0.0)
                self.apiPriceOutputEURper1M = float(modelInfo.get('priceOutputEURper1M', 0.0) or 0.0)

            if self.apiAuthMethod not in ['bearer', 'x-client-id', 'x-client-secret']:
                raise ValueError("MyLLM: apiAuthMethod must be one of ['bearer', 'x-client-id', 'x-client-secret']")

            if not self.apiClientID:
                raise ValueError('MyLLM: apiClientID/CLIENT_ID is required for academic_ai backend')
            if self.apiAuthMethod == 'bearer' and not self.apiToken:
                raise ValueError('MyLLM: apiToken/BEARER_TOKEN is required for bearer auth')
            if self.apiAuthMethod == 'x-client-secret' and not self.apiClientSecret:
                raise ValueError('MyLLM: apiClientSecret/CLIENT_SECRET is required for x-client-secret auth')

        elif self.backend == 'anthropic_api':
            self.device = None
            self.deviceMap = None

            if self.modelName in llmModelsDict:
                modelInfo = llmModelsDict[self.modelName]
                self.apiModelName = modelInfo.get('apiModel', modelInfo.get('shortName', self.modelName))
                self.apiPriceInputEURper1M = float(modelInfo.get('priceInputEURper1M', 0.0) or 0.0)
                self.apiPriceOutputEURper1M = float(modelInfo.get('priceOutputEURper1M', 0.0) or 0.0)
                # newer Claude models removed temperature/top_p/top_k; see llmAPIModelsDict
                self.apiSupportsTemperature = bool(modelInfo.get('apiSupportsTemperature', False))

            anthropicApiKey = self.apiToken if self.apiToken is not None else os.getenv('ANTHROPIC_API_KEY')
            if not anthropicApiKey:
                raise ValueError('MyLLM: ANTHROPIC_API_KEY environment variable or apiToken is required for anthropic_api backend')
            self._anthropicApiKey = anthropicApiKey

            try:
                import anthropic as _anthropic_pkg
                self._anthropicClient = _anthropic_pkg.Anthropic(api_key=self._anthropicApiKey)
            except ImportError:
                raise ImportError("MyLLM: 'anthropic' package not found. Install with: pip install anthropic")

            if self.apiPriceInputEURper1M <= 0 or self.apiPriceOutputEURper1M <= 0:
                self.logger.LogWarning(
                    'Anthropic API pricing is 0.0 EUR/1M tokens. Update llmAPIModelsDict prices for accurate cost tracking.'
                )
            if self.apiBudgetEUR is not None:
                self.logger.LogText(f"Anthropic API budget set to {self.apiBudgetEUR} EUR")
            self.logger.LogText(f"Anthropic API backend selected. Model: {self.apiModelName}")

        elif self.backend == 'llamacpp':
            try:
                import torch
                from llama_cpp import Llama
            except ImportError as e:
                self.logger.LogError(f'MyLLM Error: Missing required packages: {e}')
                raise ImportError(f'MyLLM Error: Missing required packages: {e}')
                
            # Store device configuration for model initialization
            self.n_gpu_layers = -1  # Default: try to use all GPU layers
            self.main_gpu = 0       # Default main GPU
            
            if device is None:
                # Auto-select all available GPUs
                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
                    self.logger.LogDebug(f"llama-cpp: Auto-selecting all {num_gpus} visible GPUs")
                else:
                    self.n_gpu_layers = 0  # Force CPU if no GPUs available
                    self.logger.LogDebug("llama-cpp: No GPUs found, using CPU")
                    
            elif device == -1 or device == 'cpu':
                self.n_gpu_layers = 0  # Force CPU
                os.environ["LLAMA_CPP_FORCE_CPU"] = "1"
                self.logger.LogDebug("llama-cpp: Using CPU (LLAMA_CPP_FORCE_CPU=1)")
                
            elif isinstance(device, int):
                if device >= 0 and device < torch.cuda.device_count():
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
                    self.main_gpu = 0  # After setting CUDA_VISIBLE_DEVICES, main GPU is always 0
                    self.logger.LogDebug(f"llama-cpp: Using GPU {device}")
                else:
                    self.logger.LogWarning(f"GPU {device} not available, falling back to CPU")
                    self.n_gpu_layers = 0
                    
            elif isinstance(device, list):
                available_gpus = list(range(torch.cuda.device_count()))
                valid_devices = [d for d in device if d in available_gpus]
                
                if valid_devices:
                    visibleDevices = ",".join(map(str, valid_devices))
                    os.environ["CUDA_VISIBLE_DEVICES"] = visibleDevices
                    self.main_gpu = 0  # First GPU in the list becomes main GPU
                    self.logger.LogDebug(f"llama-cpp: Using GPUs {visibleDevices}")
                else:
                    self.logger.LogWarning("No valid GPUs in list, falling back to CPU")
                    self.n_gpu_layers = 0
            else:
                raise ValueError(f"MyLLM: illegal device for llama-cpp: {str(device)}")


        ## initialize models
        if self.backend == 'gpt4all':
            # GPT4All does not support multi GPU (yet), seperately handled here to use only one GPU            
            from gpt4all import GPT4All

            # Only uses nthreads if device is CPU
            modelLoaded = False
            
            modelPath = JoinPath(expandedPath, 'gguf_models', modelName)
            
            allowDownload = not os.path.isfile(modelPath)
            
            # Try to initialize on GPU first if no device specified
            if self.device is None or self.device == 'cuda':
                try:
                    gpt4allDevice = 'cuda'
                    self.model = GPT4All(
                        model_name=modelPath,
                        n_threads=self.nThreads,  # Threads are only for CPU
                        device=gpt4allDevice,
                        allow_download=allowDownload,
                        verbose=False,
                        n_ctx = self.contextWindowMaxSize,
                    )
                    modelLoaded = True
                    self.logger.LogDebug("GPT4All: Successfully loaded model on GPU")
                    
                except Exception as e:
                    self.logger.LogWarning(f"GPT4All: GPU loading failed, falling back to CPU. Error: {str(e)}")

            # If GPU loading failed or CPU was explicitly set
            if not modelLoaded:
                self.model = GPT4All(
                    model_name=modelPath,
                    n_threads=self.nThreads,
                    device='cpu',
                    allow_download=allowDownload,
                    verbose=False,
                    n_ctx = self.contextWindowMaxSize, #Johannes; changed from 2048, as it gave errors
                )
                self.device = 'cpu'
                self.logger.LogDebug("GPT4All: Using CPU instead of GPU")

            self.logger.LogText(f"Using GPT4All model: {modelName} on {self.device}, n_ctx={self.contextWindowMaxSize}", printToConsole=True)
        elif self.backend == 'hftransformers':
            # Load model config 
            try:
                from transformers import AutoConfig, pipeline, BitsAndBytesConfig, AutoTokenizer, AutoProcessor  # Add AutoProcessor
                import torch
            except:
                self.logger.LogDebug('MyLLM Warning: did not find transformers Python package')
            
            allowDownload = not os.path.isfile(JoinPath(expandedPath, modelName))    
            
            self.modelConfig = AutoConfig.from_pretrained(modelName, trust_remote_code=True)
            
            if self.isVisionModel:
                self.logger.LogText(f"Detected vision model: {modelName}")
                try:
                    # in __init__, HF transformers backend, vision branch
                    from transformers import AutoProcessor
                    modelNameLower = modelName.lower()
                    if "qwen3.5" in modelNameLower or "qwen3.6" in modelNameLower:
                        from transformers import Qwen3_5ForConditionalGeneration as VisionModel
                    elif "qwen3" in modelNameLower and "a3b" in modelNameLower and "vl" in modelNameLower:
                        from transformers import Qwen3VLMoeForConditionalGeneration as VisionModel
                    elif "qwen3" in modelNameLower and "vl" in modelNameLower:
                        from transformers import Qwen3VLForConditionalGeneration as VisionModel
                    elif "qwen2.5" in modelNameLower and "vl" in modelNameLower:
                        from transformers import Qwen2_5_VLForConditionalGeneration as VisionModel
                    elif "qwen2" in modelNameLower and "vl" in modelNameLower:
                        from transformers import Qwen2VLForConditionalGeneration as VisionModel
                    elif "ministral" in modelNameLower or "mistral3" in modelNameLower:
                        from transformers import Mistral3ForConditionalGeneration as VisionModel
                    elif "gemma-4" in modelNameLower:
                        from transformers import Gemma4ForConditionalGeneration as VisionModel
                    else:
                        from transformers import AutoModelForCausalLM as VisionModel


                    # Configure processor for optimized image processing
                    min_pixels = 256 * 28 * 28  # ~200K pixels minimum
                    max_pixels = 1280 * 28 * 28  # ~1M pixels maximum

                    try:
                        if "gemma-4" in modelNameLower:
                            # Gemma4Processor takes a discrete soft-token budget (70/140/280/560/1120)
                            # instead of min/max pixel bounds; use the documented default (280) rather
                            # than a larger tier - avoids untested code paths in this very new processor
                            self.processor = AutoProcessor.from_pretrained(
                                modelName,
                                trust_remote_code=True,
                                max_soft_tokens=280
                            )
                            self.logger.LogText("Vision processor configured with max_soft_tokens=280")
                        else:
                            self.processor = AutoProcessor.from_pretrained(
                                modelName,
                                trust_remote_code=True,
                                min_pixels=min_pixels,
                                max_pixels=max_pixels
                            )
                            self.logger.LogText(f"Vision processor configured with min_pixels={min_pixels}, max_pixels={max_pixels}")
                    except Exception as e:
                        # Fallback without optimization if processor doesn't support these parameters
                        self.processor = AutoProcessor.from_pretrained(modelName, trust_remote_code=True)
                        self.logger.LogWarning(f"Could not configure processor optimization: {e}")



                    modelKwargs = {}
                    # accelerate's per-tensor GPU dispatch during from_pretrained (device_map=
                    # "auto" or an explicit device index) segfaults (STATUS_ACCESS_VIOLATION in
                    # torch_cpu.dll) while materializing safetensors slices for some vision
                    # models - reproduced 2026-07-20 on venvP312 (torch 2.12.0.dev20260408+cu128,
                    # transformers 5.7.0) for Gemma4, and again 2026-08-21 for Qwen3-VL-32B: the
                    # crash is isolated to the device_map code path in
                    # transformers.core_model_loading (PySafeSlice.__getitem__ during dispatch);
                    # device_map=None loads and runs fine. Load to CPU, then move explicitly.
                    needsCpuThenMoveWorkaround = ("gemma-4" in modelNameLower
                                                  or "qwen3-vl-32b" in modelNameLower)
                    if self.device != -1:
                        import torch

                        # ADD QUANTIZATION SUPPORT FOR VISION MODELS
                        if quantization_config is not None:
                            modelKwargs["quantization_config"] = quantization_config
                            self.logger.LogText("Vision model: Using provided quantization config.")
                            # GetQuantization() reads modelQuantizationHF, not quantization_config;
                            # without this, a quantized vision model is misreported via the static
                            # llmModelsDict Q fallback (e.g. logged as 16-bit while running 4-bit).
                            if getattr(quantization_config, "load_in_4bit", False):
                                self.modelQuantizationHF = "4bit"
                            elif getattr(quantization_config, "load_in_8bit", False):
                                self.modelQuantizationHF = "8bit"
                        elif modelQuantization == "4bit":
                            modelKwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                # bfloat16, NOT float16: float16 tops out at 65504, and Qwen-VL
                                # activations overflow it -> Inf -> NaN logits -> argmax falls
                                # through to token id 0, which decodes to "!" in the Qwen
                                # tokenizer. That is the origin of the "!!!!" runaway output
                                # (2026-08-28); it was never a repetition/sampling problem.
                                # bfloat16 has float32's exponent range, so it cannot overflow
                                # this way. Matches the non-quantized default (dtype=bfloat16)
                                # below and the text-model path further down.
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                            self.logger.LogText("Vision model: Using 4-bit quantization.")
                        elif modelQuantization == "8bit":
                            modelKwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_8bit=True,
                            )
                            self.logger.LogText("Vision model: Using 8-bit quantization.")
                        else:
                            modelKwargs["dtype"] = torch.bfloat16  # Default

                        if not needsCpuThenMoveWorkaround:
                            modelKwargs["device_map"] = self.deviceMap
                        # affected models: device_map is deliberately omitted here; see workaround below.

                    if needsCpuThenMoveWorkaround:
                        self.model = VisionModel.from_pretrained(
                            modelName, trust_remote_code=True, **modelKwargs
                        )
                        if self.device != -1:
                            targetDevice = f"cuda:{self.device}" if isinstance(self.device, int) else "cuda"
                            self.model = self.model.to(targetDevice)
                            self.logger.LogText(f"device_map CPU-load workaround: moved model to {targetDevice} after CPU load")
                        self.model = self.model.eval()
                    else:
                        self.model = VisionModel.from_pretrained(
                            modelName,
                            trust_remote_code=True,
                            **modelKwargs
                        ).eval()

                    self._clear_hf_max_length_conflict()
                    self.pipe = None
                    self.logger.LogText(f"Successfully loaded vision model: {modelName}")
                except Exception as e:
                    self.logger.LogError(f"Failed to load vision model {modelName}: {e}")
                    raise
            
            else:
    ##########################################################################################
                tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True) # introduced for deepseek-coder-v2-instruct-lite (outcomment if resulting in error for other models!!!)
    ##########################################################################################            
                self.logger.LogDebug(f"Loaded HF-config for {modelName}: {self.modelConfig}")
                
                modelKwargs = {}
                
                # Quantization setup
                if self.device != -1:
                    if quantization_config is not None:
                        # Use provided quantization config
                        modelKwargs["quantization_config"] = quantization_config
                        self.logger.LogText("Using provided quantization config.")
                    elif modelQuantization is None:
                        modelKwargs['dtype'] = "auto"  # Default to auto
                        self.logger.LogText("No custom quantization provided, attempting auto-quantization ...")
                    elif modelQuantization == "4bit":
                        modelKwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16, #speed up computations
                            bnb_4bit_use_double_quant=False,
        
                        )
                        self.logger.LogText("Loaded model with 4-bit quantization.")
                    elif modelQuantization == "8bit":
                        modelKwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                        self.logger.LogText("Loaded model with 8-bit quantization.")
                    else:
                        # Use specified dtype for other quantizations (e.g., fp16, bf16)
                        modelKwargs['dtype'] = modelQuantization
                        self.logger.LogText(f"Custom quantization {modelQuantization} provided.")
                # Load the pipeline
                try:
                    task = None
                    if 'causal' in self.modelConfig.architectures[0].lower():
                        task = "text-generation"
                    elif 'gemma3' in self.modelConfig.architectures[0].lower():
                        task = 'image-text-to-text' if self.isVisionModel else 'text-generation'
                    # add other tasks here
                    else:
                        task = 'conversation' #default

                    self.pipe = pipeline(
                        task=task,
                        model=modelName,
                        tokenizer=tokenizer,
                        model_kwargs=modelKwargs,
                        device_map=self.deviceMap,  # Important for optimized GPU loading
                    )
                    self._clear_hf_max_length_conflict()
                    self.logger.LogText(f"Initialized HuggingFace model: {modelName}", printToConsole=True)
                except Exception as e:
                    self.logger.LogError(f"HF transformers: loading GPU failed - {str(e)}")
                    raise

        # Replace the llamacpp model initialization around line 600:
        elif self.backend == 'llamacpp':
            try:
                from llama_cpp import Llama
            except:
                self.logger.LogError('MyLLM Error: did not find llama_cpp Python package')
                raise ImportError('MyLLM Error: did not find llama_cpp Python package')
            
            modelPath = JoinPath(expandedPath, 'gguf_models', modelName)
            
            try:
                self.model = Llama(
                    model_path=modelPath,
                    n_gpu_layers=-1,              # Force all layers to GPU
                    main_gpu=0,                   # Explicitly set main GPU
                    n_threads=self.nThreads,      # CPU threads
                    n_ctx=self.contextWindowMaxSize, # Context window
                    verbose=False,
                )
                self.logger.LogText(f"llamacpp: Successfully loaded {modelName} with GPU", printToConsole=True)
            except Exception as e:
                self.logger.LogError(f"llamacpp: Failed to load model: {e}")
                # Fallback to CPU
                self.model = Llama(
                    model_path=modelPath,
                    n_gpu_layers=0,  # Force CPU
                    n_threads=self.nThreads,
                    n_ctx=self.contextWindowMaxSize,
                    verbose=False,
                )
                self.logger.LogText(f"llamacpp: Fallback to CPU loading successful")


        elif self.backend == 'vllm':
            # vLLM in-process (NOT the HTTP server): the pipeline runs in the same interpreter,
            # so there is no serialization hop. Measured 2026-08-29 against the hftransformers
            # path on identical work (32 sketches, 1856-token cap, Qwen3-VL-32B): 2.85 s/sample
            # vs 29.18 s/sample, i.e. ~10x. The win is scheduling, not kernels -- hftransformers
            # pads every sequence in a batch to the longest and cannot start a new sequence until
            # the whole batch finishes, while vLLM uses continuous batching over a paged KV cache.
            # Linux/WSL only: the Windows vLLM wheel ships no compiled CUDA kernels.
            try:
                from vllm import LLM as VllmLLM
            except ImportError:
                self.logger.LogError('MyLLM Error: did not find vllm Python package '
                                     '(Linux/WSL only; the Windows wheel has no CUDA kernels)')
                raise
            from transformers import AutoProcessor

            self.modelName = modelName
            #! 'vllm:'-prefixed registry keys map to the HF repo that is actually loaded
            vllmRepo = llmModelsDict.get(modelName, {}).get('hfRepo', modelName)
            #! vLLM pre-allocates this fraction of VRAM up front and holds it for the process
            #! lifetime, so two vLLM models cannot be resident at once -- the caller must
            #! FreeMemory() one before constructing the next (e.g. vision -> coder at Stage 2).
            gpuMemoryUtilization = getattr(self, 'vllmGpuMemoryUtilization', 0.90)
            #! vLLM reserves KV cache for the FULL max_model_len up front, unlike
            #! hftransformers which allocates as needed. A model advertising a huge window
            #! therefore fails to start rather than merely using more memory: Gemma4-31B
            #! declares 262144 tokens, demanding 45.8 GiB of KV cache on top of 62 GB of
            #! weights ("To serve at least one request with the model's max seq len..."").
            #! The largest context this pipeline has ever used is ~5.3k tokens (beam models,
            #! code generation), so cap the window rather than inherit the model's maximum.
            vllmMaxModelLen = 8192
            maxModelLen = min(self.contextWindowMaxSize or vllmMaxModelLen, vllmMaxModelLen)
            #! Hybrid attention/SSM models (Qwen3.6) hold one Mamba state cache block per
            #! decode sequence, and vLLM refuses CUDA graph capture when max_num_seqs exceeds
            #! the number of blocks that fit: Qwen3.6-27B offers 578 against vLLM's default of
            #! 1024 ("max_num_seqs (1024) exceeds available Mamba cache blocks (578)",
            #! 2026-08-29). Pure-attention models never hit this, but the lower cap costs them
            #! nothing: 256 still covers the largest batch this pipeline submits
            #! (--vision_batch_size 200).
            vllmMaxNumSeqs = int(os.environ.get('EXUAI_VLLM_MAX_NUM_SEQS', 256))
            tensorParallelSize = getattr(self, 'vllmTensorParallelSize', 1)

            #! Opt-in workaround for the NaN forward pass that makes Qwen3-VL and Qwen3.5
            #! emit token id 0 ('!') to the token cap and never stop. Upstream traces it to a
            #! numerically unstable graph produced by torch.compile for particular input
            #! shapes -- explicitly NOT a precision problem, as it occurs in float16 and
            #! bfloat16 alike (vllm-project/vllm-ascend#3934). Once one request produces NaN
            #! the KV cache is polluted, so every request sharing that prefix block emits the
            #! same garbage, which is why whole batches fail together with identical timings.
            #! enforce_eager ALONE does not disable torch.compile; the compilation mode must
            #! also be NONE (discuss.vllm.ai/t/true-eager-backend/2396).
            #!
            #! Kept OFF by default: it changes the numerical path, and comparing runs across
            #! different paths would invalidate an ablation. Enable per run with
            #! EXUAI_VLLM_DISABLE_COMPILE=1 to test the workaround or to rerun a model that
            #! keeps failing. Costs throughput, so switch it on only where it is needed.
            disableTorchCompile = os.environ.get('EXUAI_VLLM_DISABLE_COMPILE', '') == '1'
            compileKwargs = {}
            if disableTorchCompile:
                from vllm.config import CompilationConfig
                compileKwargs = {
                    'enforce_eager': True,
                    'compilation_config': CompilationConfig(mode=0),
                }
                self.logger.LogText(
                    'vLLM: torch.compile and CUDA graphs DISABLED for %s '
                    '(EXUAI_VLLM_DISABLE_COMPILE=1); slower, but avoids the Qwen3-VL NaN '
                    'graph-compilation fault.' % modelName, printToConsole=True)
            # Mistral vision models must be served in Mistral's OWN format. Loaded from the
            # HuggingFace-format checkpoint they start and generate fluently but never receive
            # the image: asked to describe a red circle beside a blue square, Ministral-3-8B
            # answered "I cannot see or analyze images directly", and on sketches it replied
            # "please provide the image" (verified 2026-08-29 -- the failure is silent, the
            # model simply behaves as text-only). Setting all three of tokenizer_mode /
            # config_format / load_format to 'mistral' fixes it, and requires the repo's
            # params.json, tekken.json and consolidated.safetensors to be present in the
            # cache alongside (or instead of) the HF shards.
            isMistralFormatModel = 'ministral' in modelName.lower() or 'mistral' in modelName.lower()
            mistralFormatKwargs = ({'tokenizer_mode': 'mistral',
                                    'config_format': 'mistral',
                                    'load_format': 'mistral'}
                                   if (isMistralFormatModel and self.isVisionModel) else {})
            #! consumed by _generate_vllm to route through llm.chat() instead of pre-templating
            self._vllmUsesMistralFormat = bool(mistralFormatKwargs)
            if mistralFormatKwargs:
                self.logger.LogText(f"vLLM: using Mistral native format for {modelName} "
                                    f"(HF format does not deliver images to this family)",
                                    printToConsole=True)
            #! vLLM workers are spawned processes and inherit sys.argv. The CuTe/cutlass DSL that
            #! JIT-compiles the Hopper flash-attention kernels runs its own argparse on sys.argv
            #! (option '-diagnostic'), so modelBuilderDriver's '-d 44' was taken as an abbreviation
            #! of it and killed the worker during warm-up (Gemma4/Qwen3.5/3.6, 2026-09-29). The
            #! caller has parsed its arguments already, so the workers get only the script name.
            #! Only done when such an abbreviation is present: spawned workers re-import the main
            #! script, and drivers without a __main__ guard (testDriverSketch2sim.py) must keep
            #! seeing their real arguments there.
            savedArgv = sys.argv
            if any(arg.startswith('-') and not arg.startswith('--') and len(arg) > 1
                   and '-diagnostic'.startswith(arg) for arg in sys.argv[1:]):
                sys.argv = sys.argv[:1]
            try:
                self.model = VllmLLM(
                    model=vllmRepo,
                    dtype='bfloat16',
                    gpu_memory_utilization=gpuMemoryUtilization,
                    max_model_len=maxModelLen,
                    max_num_seqs=vllmMaxNumSeqs,
                    tensor_parallel_size=tensorParallelSize,
                    limit_mm_per_prompt={'image': 1} if self.isVisionModel else None,
                    trust_remote_code=True,
                    **mistralFormatKwargs,
                    **compileKwargs,
                )
            finally:
                sys.argv = savedArgv
            #! used only to apply each model's chat template; keeps this backend model-agnostic
            #! instead of hard-coding Qwen/Gemma/Mistral prompt markers
            self.processor = AutoProcessor.from_pretrained(vllmRepo, trust_remote_code=True)
            self.pipe = None
            self.logger.LogText(f"Initialized vLLM model: {vllmRepo} "
                                f"(gpu_memory_utilization={gpuMemoryUtilization}, "
                                f"tensor_parallel_size={tensorParallelSize}, "
                                f"max_num_seqs={vllmMaxNumSeqs}, max_model_len={maxModelLen})",
                                printToConsole=True)
            #! catch an engine that came up corrupted before it burns a whole run
            self._RunVllmCanary()

        elif self.backend == 'ollama':
            try:
                import ollama
            except ImportError:
                self.logger.LogError('MyLLM Error: did not find ollama Python package')
                raise ImportError('MyLLM Error: did not find ollama Python package')
            
            self.logger.LogText(f"Ollama backend selected. Model: {modelName}")
            self.modelName = modelName
            
            # Check if model is available locally
            try:
                list_response = ollama.list()
                available_models = getattr(list_response, 'models', list_response)
                model_names = []
                for model in available_models:
                    if isinstance(model, dict):
                        model_names.append(model.get('name') or model.get('model'))
                    else:
                        model_names.append(getattr(model, 'model', None) or getattr(model, 'name', None))
                model_names = [name for name in model_names if name is not None]

                if modelName not in model_names:
                    self.logger.LogWarning(f"Model {modelName} not found locally. Available models: {model_names}")
                    self.logger.LogText(f"Attempting to pull model {modelName}...", printToConsole=True)
                    try:
                        ollama.pull(modelName)
                        self.logger.LogText(f"Successfully pulled model {modelName}", printToConsole=True)
                    except Exception as e:
                        self.logger.LogError(f"Failed to pull model {modelName}: {e}")
                        raise RuntimeError(f"Model {modelName} not available and pull failed: {e}")
                else:
                    self.logger.LogText(f"Model {modelName} found locally")
                    
            except Exception as e:
                self.logger.LogWarning(f"Could not verify model availability: {e}")
                self.logger.LogText("Proceeding with model loading...")
            
            # Store Ollama-specific configuration
            self.ollama_options = {}
            
            # Configure based on device parameter (optional)
            if device == -1 or device == 'cpu':
                self.logger.LogText("Ollama: Configured for CPU usage")
                # Note: Ollama handles GPU/CPU automatically, but you could set specific options here
            elif device is not None:
                self.logger.LogText(f"Ollama: Will use available GPU resources (device config: {device})")
            
            self.logger.LogText(f"Ollama initialization complete for model: {modelName}")

        elif self.backend == 'academic_ai':
            self.logger.LogText(f"Academic AI backend selected. API model: {self.apiModelName}")
            self.logger.LogText(f"Academic AI auth method: {self.apiAuthMethod}")
            if self.apiPriceInputEURper1M <= 0 or self.apiPriceOutputEURper1M <= 0:
                self.logger.LogWarning(
                    'Academic AI pricing is 0.0 EUR/1M tokens. Update llmAPIModelsDict prices for accurate cost tracking.'
                )
            if self.apiBudgetEUR is not None:
                self.logger.LogText(f"Academic AI budget set to {self.apiBudgetEUR} EUR")
        elif self.backend == 'anthropic_api':
            pass  # already logged in init block above
    

    # Format chat-style prompts based on the model family and used backend.
    def FormatChatPrompt(self, promptList=[], systemPrompt=None, imagePathList=[]):
        """Format one or multiple prompts based on model architecture conventions and used backend."""
        formattedList = []
        
        # hugging face transformers pipelines expect messages for chat models
        if self.backend in ['hftransformers', 'ollama', 'academic_ai', 'anthropic_api']:
            for i, prompt in enumerate(promptList):
                messages = []
                if systemPrompt:
                    messages.append({
                        "role": "system",
                        "content": systemPrompt
                    })
                if self.isVisionModel:
                    if not imagePathList:
                        pass #avoid regular warnings in debug file
                    else:   
                        messages.append({
                            "type": "image", "image": imagePathList[i]
                            })
                messages.append({
                    "role": "user",
                    "content": prompt
                })
                formattedList.append(messages)

            return formattedList
        
        # all other backends except gpt4all expect formatted strings
        if self.backend != 'gpt4all':
            nameLower = self.modelName.lower()
            
            for prompt in promptList:
                if any(name in nameLower for name in ['llama', 'alpaca', 'mistral']):
                    parts = []
                    if systemPrompt:
                        parts.append(f"<|system|>\n{systemPrompt}\n")
                    parts.append(f"<|user|>\n{prompt}\n<|assistant|>\n")
                    formatted = "\n".join(parts)
                elif "gemma" in nameLower:
                    parts = []
                    if systemPrompt:
                        parts.append(f"<start_of_turn>system\n{systemPrompt}<end_of_turn>")
                    parts.append(f"<start_of_turn>user\n{prompt}<end_of_turn>")
                    parts.append(f"<start_of_turn>model\n")
                    formatted = "\n".join(parts)
                elif "qwen" in nameLower:
                    parts = []
                    if systemPrompt:
                        parts.append(f"[SYSTEM]\n{systemPrompt}\n")
                    parts.append(f"[USER]\n{prompt}\n[ASSISTANT]\n")
                    formatted = "\n".join(parts)
                elif any(name in nameLower for name in ['deepseek', 'default']): #OLD: phi
                    parts = []
                    if systemPrompt:
                        parts.append(f"<|system|>{systemPrompt}")
                    parts.append(f"<|user|>{prompt}<|assistant|>")
                    formatted = "\n".join(parts)
                elif any(name in nameLower for name in ['phi']):
                    parts = []
                    
                    if systemPrompt:
                        parts.append(f"<|im_start|>system<|im_sep|>\n{systemPrompt}<|im_end|>\n")
                        
                    parts.append(f"im_start|>user<|im_sep|>\n{prompt}\n<|im_start|>assistant<|im_sep|>\n")
                    formatted = "".join(parts)
                else:
                    formatted = f"{systemPrompt}\n\n{prompt}" if systemPrompt else prompt
        
                formattedList.append(formatted)
        else:
            # GPT4All backend uses raw prompts
            formattedList = promptList
        # Return a list if multiple prompts were passed, otherwise a single string
        return formattedList
    

    def _clear_hf_max_length_conflict(self):
        """@brief Unset max_length on HF generation configs so only max_new_tokens applies.
           Qwen/Coder models often ship with a small default max_length (e.g. 20) which
           conflicts with max_new_tokens passed at generate time."""
        targets = []
        pipe = getattr(self, 'pipe', None)
        if pipe is not None:
            targets.append(pipe)
            if getattr(pipe, 'model', None) is not None:
                targets.append(pipe.model)
        model = getattr(self, 'model', None)
        if model is not None:
            targets.append(model)
        seen = set()
        for obj in targets:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            gen_cfg = getattr(obj, 'generation_config', None)
            if gen_cfg is not None and getattr(gen_cfg, 'max_length', None) is not None:
                gen_cfg.max_length = None

    
    def _hfTokenizer(self):
        """Return the HF tokenizer for the loaded text pipe or vision processor."""
        pipe = getattr(self, 'pipe', None)
        if pipe is not None and getattr(pipe, 'tokenizer', None) is not None:
            return pipe.tokenizer
        proc = getattr(self, 'processor', None)
        if proc is not None and getattr(proc, 'tokenizer', None) is not None:
            return proc.tokenizer
        return None

    def _hfTokenCount(self, text):
        """Token count via HF tokenizer when available, else char/4 estimate."""
        if text is None:
            return 0
        tokenizer = self._hfTokenizer()
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(str(text), add_special_tokens=False))
            except Exception:
                pass
        return max(0, len(str(text)) // 4)

    def _BuildNonFiniteLogitsGuard(self):
        """@brief LogitsProcessorList that REPORTS (never silently repairs) non-finite logits.

           Non-finite (NaN/Inf) logits make argmax fall through to token id 0, which is what
           actually produced the "!!!!" runaway output diagnosed 2026-08-28: `!` is token id 0
           in the Qwen tokenizer. The trigger there was bnb_4bit_compute_dtype=torch.float16
           overflowing float16's 65504 ceiling on Qwen-VL activations; that specific cause is
           fixed (bfloat16 everywhere), but this guard makes any future recurrence loud instead
           of silent.

           It matters that this only reports. The failure was found only by luck, because
           Qwen's token 0 renders as a visible "!". For Gemma, token id 0 is <pad>, which
           batch_decode(skip_special_tokens=True) strips -- so the identical numerical failure
           would surface as quietly truncated output that looks like a normal short answer, and
           would silently corrupt results rather than announce itself.

           transformers ships InfNanRemoveLogitsProcessor, but that is deliberately NOT used
           here: it overwrites NaN/Inf so generation "succeeds", which for a research pipeline
           converts a detectable hard failure into plausible-looking wrong data. Its own
           documentation notes there should be no correct reason to enable it. Scores are
           returned unmodified, so healthy generation is bit-identical to running without it.
           @return transformers.LogitsProcessorList with a single reporting processor
        """
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        logger = self.logger

        class _NonFiniteLogitsGuard(LogitsProcessor):
            def __init__(self):
                self.nBadSteps = 0

            def __call__(self, input_ids, scores):
                #! one reduction per decode step; negligible next to the forward pass itself
                if not torch.isfinite(scores).all():
                    self.nBadSteps += 1
                    if self.nBadSteps == 1:  # report once per generate() call, not per step
                        nBadRows = int((~torch.isfinite(scores)).any(dim=-1).sum().item())
                        logger.LogText(
                            f"LLM ERROR: non-finite (NaN/Inf) logits at generated token "
                            f"{input_ids.shape[1]}, affecting {nBadRows}/{scores.shape[0]} "
                            f"sequence(s). Output from this batch is NOT trustworthy: argmax "
                            f"degenerates to token id 0 (\"!\" for Qwen, <pad> -- invisible "
                            f"after decoding -- for Gemma). Most likely a numeric-precision "
                            f"regression; check bnb_4bit_compute_dtype is bfloat16, not "
                            f"float16.", printToConsole=True)
                return scores

        return LogitsProcessorList([_NonFiniteLogitsGuard()])

    @staticmethod
    def _BuildDegeneracyStoppingCriteriaList(window=16, threshold=0.5, minNewTokens=12):
        """@brief StoppingCriteriaList that halts an individual sequence mid-batch as soon as
           one token dominates its own recent trailing window -- a fast, direct detector for
           greedy-decoding repetition traps (temperature=0.0 has no mechanism to escape one once
           started), found 2026-08-28: a short "chooseItems" call generated 1250 tokens of "!!!!"
           before returning. no_repeat_ngram_size / repetition_penalty reduce severity but were
           shown to be gameable (the model evaded no_repeat_ngram_size by inserting an
           incrementing junk character each cycle, keeping every n-gram technically unique while
           staying just as degenerate) or merely delay the onset rather than prevent it -- and
           neither one stops generation EARLY, so a degenerate sequence still runs to
           max_new_tokens before the caller can detect and react to it (unacceptable for the
           10000-token code-generation step: ~55 minutes worst case at the ~3 tok/s observed
           here). This checks every step, not after the fact, and (unlike a blanket max_tokens
           cap) does not limit legitimate long output at all -- it only fires when one token
           actually dominates recent output, which real generated text/code does not do.
           @param window trailing tokens examined per step
           @param threshold fraction of the window one token must occupy to count as degenerate
           @param minNewTokens tokens generated before evaluation begins, to avoid firing on
                  short legitimate repeats (e.g. a few blank lines, "mbs.mbs.")
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _DegeneracyStoppingCriteria(StoppingCriteria):
            def __init__(self):
                self.startLength = None  # captured lazily on first call: input_ids.shape[1]
                                         # at that point == prompt length + 1 already-generated
                                         # token, so this needs no prior knowledge of prompt
                                         # length and works identically for both the vision
                                         # (model.generate) and text (pipeline) call sites.

            def __call__(self, input_ids, scores, **kwargs):
                if self.startLength is None:
                    self.startLength = input_ids.shape[1] - 1
                genLen = input_ids.shape[1] - self.startLength
                batchSize = input_ids.shape[0]
                if genLen < minNewTokens:
                    return torch.zeros(batchSize, dtype=torch.bool, device=input_ids.device)
                windowStart = max(self.startLength, input_ids.shape[1] - window)
                recent = input_ids[:, windowStart:]
                # Fully vectorised on purpose: this runs on EVERY decode step, so the obvious
                # per-sequence loop (torch.unique + counts.max().item() per row) cost one
                # GPU->CPU synchronisation per sequence per step -- 32 pipeline stalls per step
                # at batch 32, tens of thousands over a batch. Comparing the window against
                # itself gives each token's occurrence count in one kernel with no sync at all:
                # the window is tiny (16), so the (batch, 16, 16) intermediate is negligible.
                windowLength = recent.shape[1]
                occurrenceCounts = (recent.unsqueeze(2) == recent.unsqueeze(1)).sum(dim=2)
                maxOccurrences = occurrenceCounts.max(dim=1).values
                return maxOccurrences >= threshold * windowLength

        return StoppingCriteriaList([_DegeneracyStoppingCriteria()])

    # vision path for huggingface transformers models, takes image-paths, alongside prompts as inputs, batching enabled
    def _generate_vision_model(
        self,
        prompt,
        systemPrompt=None,
        imagePath=None,
        max_tokens=512,
        temperature=0.,
        **kwargs
    ):
        """
        Supports:
        - Single sample, single or multiple images: prompt=str, imagePath=str|List[str]
        - Batched samples: prompt=List[str], imagePath=List[str|List[str]]
            (imagePath[i] is str or list of str for sample i)
        """
        from qwen_vl_utils import process_vision_info
        import torch

        nSamples = len(prompt) if isinstance(prompt, (list, tuple)) else 1
        # reset the peak counter HERE (per chunk), not just once at the top of _generate_hf --
        # since _generate_hf now loops calling this once per chunk, a reset only at its top would
        # make maxReservedMB/maxAllocatedMB reflect the peak since the FIRST chunk, not this one.
        self.TorchGPUmemoryReset()
        gpuInfoStart = self.TorchGPUmemory()
        self.logger.LogDebug(
            f"LLM: vision batch START | samples={nSamples} | "
            f"reserved={gpuInfoStart['reservedMB']/1024:.2f}GB "
            f"allocated={gpuInfoStart['allocatedMB']/1024:.2f}GB "
            f"free={gpuInfoStart['freeMB']/1024:.2f}GB",
            printToConsole=True)

        # qwen_vl_utils.process_vision_info applies Qwen-specific smart-resize (28px
        # patch grid, Qwen min/max pixel budget) before the image ever reaches the
        # model's own processor. For Gemma4 (48px patch grid, max_soft_tokens budget)
        # that double-preprocessing was traced to a ~400GB one_hot allocation request
        # in Gemma4VisionPatchEmbedder for some image batches. Gemma4 loads images
        # natively instead, letting Gemma4Processor do its own fetch+resize.
        isGemma4 = "gemma-4" in self.modelName.lower()

        def _load_image_native(p):
            # load_image() doesn't understand the file:// scheme _norm_image_item
            # adds for qwen_vl_utils; strip it back to a plain local path/URL first.
            from transformers.image_utils import load_image
            if p.startswith("file://"):
                p = p[len("file://"):]
            return load_image(p)

        def _norm_image_item(p):
            # Convert local paths to file:// URIs as Qwen expects
            if not (p.startswith("http") or p.startswith("file://") or p.startswith("data:image")):
                return f"file://{p}"
            return p

        # ---- Case A: batched prompts ----
        if isinstance(prompt, (list, tuple)):
            B = len(prompt)

            # imagePath normalization: per-sample -> list[str] or []
            if imagePath is None:
                imagePath = [[] for _ in range(B)]
            elif not isinstance(imagePath, (list, tuple)):
                # one image for all samples
                imagePath = [[_norm_image_item(imagePath)] for _ in range(B)]
            else:
                # make sure length matches batch and every item becomes a list
                if len(imagePath) != B:
                    raise ValueError(f"imagePath length {len(imagePath)} must match batch size {B}")
                norm = []
                for item in imagePath:
                    if item is None:
                        norm.append([])
                    elif isinstance(item, (list, tuple)):
                        norm.append([_norm_image_item(x) for x in item])
                    else:
                        norm.append([_norm_image_item(item)])
                imagePath = norm  # List[List[str]]

            msgs_list = []
            for i in range(B):
                msg = []
                if systemPrompt:
                    msg.append({"role": "system", "content": systemPrompt})

                uc = [{"type": "text", "text": prompt[i]}]
                # attach images for this sample
                for pth in imagePath[i]:
                    uc.append({"type": "image", "image": pth})

                msg.append({"role": "user", "content": uc})
                msgs_list.append(msg)

            # Build per-sample chat strings
            _chat_tmpl_kwargs = (
                {"enable_thinking": self.enable_thinking}
                if ("qwen3.5" in self.modelName.lower() or "qwen3.6" in self.modelName.lower())
                else {}
            )
            texts = [
                self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **_chat_tmpl_kwargs)
                for m in msgs_list
            ]

            # Extract vision payloads per sample
            image_inputs_all = []
            video_inputs_all = []
            if isGemma4:
                # imagePath is already the normalized List[List[str]] built above;
                # load each path natively (see _load_image_native), bypassing Qwen's
                # smart-resize entirely. This project doesn't feed Gemma4 video content.
                for pathsForSample in imagePath:
                    image_inputs_all.append([_load_image_native(p) for p in pathsForSample])
                video_inputs_all = [[] for _ in range(B)]
            else:
                for m in msgs_list:
                    imgs_i, vids_i = process_vision_info(m)
                    # Normalize to lists and handle None/empty cases
                    if imgs_i is None:
                        image_inputs_all.append([])
                    elif isinstance(imgs_i, list):
                        image_inputs_all.append(imgs_i)
                    else:
                        image_inputs_all.append([imgs_i])

                    if vids_i is None:
                        video_inputs_all.append([])
                    elif isinstance(vids_i, list):
                        video_inputs_all.append(vids_i)
                    else:
                        video_inputs_all.append([vids_i])

            # Final safety: lengths must match
            assert len(texts) == len(image_inputs_all) == len(video_inputs_all), (
                f"len(texts)={len(texts)}, len(images)={len(image_inputs_all)}, len(videos)={len(video_inputs_all)}"
            )

            # Handle case where all image lists are empty
            all_images_empty = all(len(img_list) == 0 for img_list in image_inputs_all)
            all_videos_empty = all(len(vid_list) == 0 for vid_list in video_inputs_all)
            
            try:
                inputs = self.processor(
                    text=texts,
                    images=None if all_images_empty else image_inputs_all,
                    videos=None if all_videos_empty else video_inputs_all,
                    padding=True,
                    return_tensors="pt",
                )
            except Exception as e:
                # Fallback: try without images/videos if they cause issues
                self.logger.LogWarning(f"Failed to process with images/videos, falling back to text-only: {e}")
                inputs = self.processor(
                    text=texts,
                    padding=True,
                    return_tensors="pt",
                )
        # ---- Case B: single prompt ----
        else:
            msgs = []
            if systemPrompt:
                msgs.append({"role": "system", "content": systemPrompt})

            uc = [{"type": "text", "text": prompt}]
            rawImagePaths = []
            if imagePath:
                rawImagePaths = [str(p) for p in imagePath] if isinstance(imagePath, (list, tuple)) else [str(imagePath)]
                for pth in rawImagePaths:
                    uc.append({"type": "image", "image": _norm_image_item(pth)})
            msgs.append({"role": "user", "content": uc})

            _chat_tmpl_kwargs = (
                {"enable_thinking": self.enable_thinking}
                if ("qwen3.5" in self.modelName.lower() or "qwen3.6" in self.modelName.lower())
                else {}
            )
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **_chat_tmpl_kwargs)
            if isGemma4:
                image_inputs = [_load_image_native(p) for p in rawImagePaths] if rawImagePaths else None
                video_inputs = None
            else:
                image_inputs, video_inputs = process_vision_info(msgs)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        # Move to model device if possible
        try:
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        gen_kwargs = dict(
            max_new_tokens=max_tokens or 512,
            # temperature=0.0 everywhere means pure greedy decoding, which has no mechanism to
            # escape a repetition loop once one starts (found 2026-08-28: a "chooseItems" call
            # that should generate ~10-20 tokens instead generated 1250 tokens of literal "!!!!").
            # Tried and REJECTED (2026-08-28), both confirmed to actively cause degeneracy rather
            # than merely fail to prevent it, for structured-output tasks (JSON/tag-list
            # generation) where the correct answer legitimately echoes field names / schema text
            # that already appears in the prompt:
            #   - no_repeat_ngram_size: blocks n-grams across the FULL context (prompt +
            #     generation combined), not just newly-generated tokens. A legitimate JSON
            #     response naturally repeats field names given in the prompt's schema (e.g.
            #     "gravityDescription", "direction") -- blocking that forced the model to
            #     switch to Chinese tokens for the same field names to satisfy the constraint
            #     ("modelDescription" -> "模型描述" etc.), confirmed via a live reproduction.
            #   - repetition_penalty: penalizes every token in the full context the same way,
            #     independent of what surrounds it -- same class of problem.
            # stopping_criteria is the one mitigation that's actually safe here: it only reacts
            # to genuine degeneracy (one token dominating a trailing window), so it never
            # interferes with normal, varied structured output the way the above two do.
            #   - contrastive search (top_k=4, penalty_alpha=0.6, HF's official
            #     GenerationMode.CONTRASTIVE_SEARCH): tried and REJECTED (2026-08-28) for the
            #     SAME reason, reached by a different route. It scores each candidate by
            #     trading off model confidence against how similar the resulting hidden state
            #     would be to recent context. Structured numeric output is legitimately
            #     self-similar, so this misfires on it: live-reproduced on all 5
            #     doublePendulumRigidBodies variants, where writing the coordinate triple
            #     [0.0, 0.0, 0.0] had its third component penalized for repeating the first
            #     two and replaced with a CJK token -> "position": [0.0, 0.0, 极], which makes
            #     the IR unparseable and kills everything downstream.
            # The general lesson: ANY repetition-suppressing decoding modification -- hard
            # n-gram blocking, count-based penalties, or representation-level penalties --
            # corrupts structured output, because JSON/code output MUST repeat (coordinate
            # components, field names, indentation). It is a category mismatch, not a tuning
            # problem, so lowering penalty_alpha would only change the corruption rate.
            # stopping_criteria is the one mitigation that's actually safe here: it only reacts
            # to genuine degeneracy (one token dominating a trailing window), so it never
            # interferes with normal, varied structured output the way the above three do.
            stopping_criteria=self._BuildDegeneracyStoppingCriteriaList(),
            # reports NaN/Inf logits without altering healthy generation; see its docstring
            logits_processor=self._BuildNonFiniteLogitsGuard(),
        )
        self._clear_hf_max_length_conflict()

        if temperature is not None: # otherwise use default args from LLM
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = temperature > 0

        # Important: do NOT slice generated ids by input length in batched mode
        with torch.no_grad():
            ids = self.model.generate(**inputs, **gen_kwargs)

        gpuInfoAfterGenerate = self.TorchGPUmemory()
        self.logger.LogDebug(
            f"LLM: vision batch AFTER generate() (pre-cleanup) | samples={nSamples} | "
            f"reserved={gpuInfoAfterGenerate['reservedMB']/1024:.2f}GB "
            f"allocated={gpuInfoAfterGenerate['allocatedMB']/1024:.2f}GB "
            f"free={gpuInfoAfterGenerate['freeMB']/1024:.2f}GB | "
            # PEAK reached at ANY point during generate() (since the reset at batch START),
            # not just this one instant right as it returns -- this is the number that matches
            # what Task Manager shows continuously, since generate() memory fluctuates a lot
            # step-to-step during autoregressive decoding.
            f"PEAK_reserved={gpuInfoAfterGenerate['maxReservedMB']/1024:.2f}GB "
            f"PEAK_allocated={gpuInfoAfterGenerate['maxAllocatedMB']/1024:.2f}GB",
            printToConsole=True)

        # Decode all sequences
        out = self.processor.batch_decode(ids, skip_special_tokens=True)
        # Per-sample prompt/context token counts for ctxStats / postProcessing.
        try:
            tokenizer = getattr(self.processor, 'tokenizer', None)
            pad_id = getattr(tokenizer, 'pad_token_id', None)
            if pad_id is None and tokenizer is not None:
                pad_id = tokenizer.eos_token_id
            prompt_ids = inputs.get('input_ids')
            if prompt_ids is not None and pad_id is not None:
                prompt_lens = (prompt_ids != pad_id).sum(dim=1).tolist()
            else:
                prompt_lens = [0] * len(out)
            if pad_id is not None:
                total_lens = (ids != pad_id).sum(dim=1).tolist()
            else:
                total_lens = [ids.shape[-1]] * len(out)
            self._hfVisionContextStats = [
                {'promptTokens': int(p), 'contextSizeUsed': int(max(t, p))}
                for p, t in zip(prompt_lens, total_lens)
            ]
        except Exception:
            self._hfVisionContextStats = []

        result = [o.strip() for o in out]

        # Unlike _generate_hf's text-only batched path, this function had no cleanup between
        # calls: varying image/sketch shapes across chunks fragment PyTorch's CUDA caching
        # allocator instead of reusing cached blocks, so reserved VRAM ratchets upward call
        # after call until it approaches the card's ceiling (observed: ~16.5GB at load ->
        # 61.4GB -> slow climb to 95+/95.5GB over one batch=6 run). Mirror the text path's
        # (imperfect but non-zero) cleanup here.
        try:
            del inputs, ids
            import gc
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        gpuInfoEnd = self.TorchGPUmemory()
        self.logger.LogDebug(
            f"LLM: vision batch END (post-cleanup) | samples={nSamples} | "
            f"reserved={gpuInfoEnd['reservedMB']/1024:.2f}GB "
            f"allocated={gpuInfoEnd['allocatedMB']/1024:.2f}GB "
            f"free={gpuInfoEnd['freeMB']/1024:.2f}GB | "
            f"delta_reserved_since_start={(gpuInfoEnd['reservedMB']-gpuInfoStart['reservedMB'])/1024:+.2f}GB | "
            # this chunk's true peak, for the whole call incl. cleanup (cleanup itself can't
            # raise the peak, so this is effectively the peak during generate())
            f"PEAK_reserved_this_chunk={gpuInfoEnd['maxReservedMB']/1024:.2f}GB "
            f"PEAK_allocated_this_chunk={gpuInfoEnd['maxAllocatedMB']/1024:.2f}GB",
            printToConsole=True)

        # Return str for single-sample, List[str] for batch
        return result

    
    
    # huggingface's transformers backend generation (text and multimodal)         
    def _generate_hf(self, prompts, systemPrompt=None, imagePath=None,
                     max_tokens=None, temperature=None, top_p=None, doSample=False, 
                     numBeams=1, returnFullText=True, cleanUpTokenizationSpaces=False, 
                     taskNamesList=None,
                ):
        import time
        #import torch
        self.TorchGPUmemoryReset() #reset max values (but global peak stays!)
        self.currentInfoGPU = self.TorchGPUmemory() #get current values
        
        if self.isVisionModel:
            # BUG (found 2026-08-28): this used to call _generate_vision_model() once with the
            # ENTIRE prompts list, completely ignoring self.batchSize -- e.g. a 46-sample "generate
            # Exudyn code" task ran as ONE batch of 46 regardless of --vision_batch_size /
            # --code_batch_size. That silently defeated every batch-size cap tested (20, 6, 2 all
            # produced the identical ~95GB VRAM ceiling) since the configured size was never used
            # to chunk anything. Chunk here exactly like the text-only path below does.
            isPromptList = isinstance(prompts, (list, tuple))
            promptList = list(prompts) if isPromptList else [prompts]
            nPrompts = len(promptList)
            batchSize = self.batchSize if self.batchSize else nPrompts

            # normalize imagePath to a per-sample list so it can be chunked in lockstep with prompts
            if imagePath is None:
                imagePathList = [None] * nPrompts
            elif isinstance(imagePath, (list, tuple)) and len(imagePath) == nPrompts:
                imagePathList = list(imagePath)
            else:
                imagePathList = [imagePath] * nPrompts  # same image(s) applied to every sample

            nBatches = -(-nPrompts // batchSize)  # ceil division
            if nBatches > 1:
                self.logger.LogDebug(
                    f"LLM: Generate adjust for vision batching with batchsize {batchSize} for {nPrompts} prompts",
                    printToConsole=False)

            allOutputs = []
            allVisionStats = []
            for b in range(nBatches):
                batchStart = b * batchSize
                batchEnd = min((b + 1) * batchSize, nPrompts)
                if taskNamesList is not None and nBatches > 1:
                    self.logger.LogDebug(
                        f"LLM: Start batched vision tasks: {taskNamesList[batchStart:batchEnd]}",
                        separator=True)

                batchPrompts = promptList[batchStart:batchEnd] if isPromptList else promptList[0]
                batchImagePaths = imagePathList[batchStart:batchEnd] if isPromptList else imagePathList[0]

                batchOutputs = self._generate_vision_model(
                    batchPrompts,
                    systemPrompt=systemPrompt,
                    imagePath=batchImagePaths,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                allOutputs.extend(batchOutputs if isinstance(batchOutputs, list) else [batchOutputs])
                allVisionStats.extend(getattr(self, '_hfVisionContextStats', None) or [])

            self._hfVisionContextStats = allVisionStats  # full-length, chunk order == prompt order
            return allOutputs  # always a list, matching _generate_vision_model's own contract

        if isinstance(prompts, str): # prompt is single string
            prompts = [prompts]
            
        if self.pipe is None:
            raise RuntimeError(
                f"MyLLM: HF text pipeline not initialized for {self.modelName!r} "
                f"(model load may have failed silently)")

        nPrompts = len(prompts)
        nBatches = -(-len(prompts) // self.batchSize) # integer arithmetic (no floats) "rounds upward"
        

        formattedPrompts = self.FormatChatPrompt(prompts, systemPrompt)
        
        allOutputs = []
        
        t0Total = time.time()
        pad_token_id_OLD = self.pipe.tokenizer.pad_token_id
        padding_side_OLD = self.pipe.tokenizer.padding_side
        isBatch = nPrompts > 1
        if isBatch:
            self.logger.LogText(f"LLM: Generate adjust padding for batching with batchsize {self.batchSize} for {nPrompts} prompts",
                                printToConsole=False)
            self.logger.LogDebug('GPU Memory info, start of batch: '+str(self.currentInfoGPU),
                                 printToConsole=False)

            self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id
            self.pipe.tokenizer.padding_side = 'left'
            
        for b in range(nBatches): 
            batchStart = b * self.batchSize
            batchEnd = min((b + 1) * self.batchSize, nPrompts)

            if taskNamesList is not None:
                self.logger.LogDebug("LLM: Start "+"batched "*isBatch+f"tasks: {taskNamesList[batchStart:batchEnd]}", separator=True)

            batchFormatted = formattedPrompts[batchStart:batchEnd] if nBatches > 1 else formattedPrompts
            
            t0Batch = time.time()

            effective_max_tokens = max_tokens if max_tokens is not None else 512
            self._clear_hf_max_length_conflict()
            pipe_kwargs = dict(
                max_new_tokens=effective_max_tokens,
                batch_size=self.batchSize,
                num_beams=numBeams,
                return_full_text=returnFullText,
                clean_up_tokenization_spaces=cleanUpTokenizationSpaces,
                # see _generate_vision_model's matching comment for the full story:
                # no_repeat_ngram_size and repetition_penalty were both tried and REJECTED
                # (2026-08-28) -- confirmed to actively cause degeneracy for structured-output
                # tasks whose correct answer legitimately echoes prompt content (schema field
                # names, tag names), rather than merely failing to prevent it. Live-reproduced:
                # blocking a field name already in the prompt pushed the model to substitute
                # Chinese tokens for the same field ("modelDescription" -> "模型描述") to satisfy
                # the constraint. stopping_criteria only reacts to genuine degeneracy and never
                # distorts normal output, so it is the only one of the three kept.
                # contrastive search (top_k + penalty_alpha) was tried and REJECTED here too,
                # for the same reason -- it penalizes candidates whose hidden state resembles
                # recent context, and structured output is legitimately self-similar, so it
                # corrupted a coordinate triple into "[0.0, 0.0, 极]". See
                # _generate_vision_model's comment for the full reasoning.
                stopping_criteria=self._BuildDegeneracyStoppingCriteriaList(),
                # reports NaN/Inf logits without altering healthy generation; see its docstring
                logits_processor=self._BuildNonFiniteLogitsGuard(),
            )

            if temperature is not None:
                pipe_kwargs["temperature"] = temperature
                pipe_kwargs["do_sample"] = temperature > 0 # 
            if top_p is not None: # will only be used if do_sample=True (internally)
                pipe_kwargs["top_p"] = top_p

            batchResults = self.pipe(batchFormatted, **pipe_kwargs)

            batchDur = max(0.0001, time.time() - t0Batch)

            totalOutputLength = 0
            # collecting outputs and logging
            for i, out in enumerate(batchResults, start=batchStart):
                output = out[0]["generated_text"] if isinstance(out, list) else [out["generated_text"]] # ensure it is always a list
                promptText = str(formattedPrompts[i])

                totalOutputLength += len(output)
                self.durationLocal = batchDur / max(1,(batchEnd-batchStart))
                self.numberOfPromptTokensLocal = self._hfTokenCount(promptText)
                fullTokens = self._hfTokenCount(str(output))
                if fullTokens > self.numberOfPromptTokensLocal:
                    self.numberOfTokensLocal = max(1, fullTokens - self.numberOfPromptTokensLocal)
                    self.contextSizeUsedLocal = fullTokens
                else:
                    self.numberOfTokensLocal = max(1, len(str(output)) // 4)
                    self.contextSizeUsedLocal = self.numberOfPromptTokensLocal + self.numberOfTokensLocal
                
                self.tokensPerSecondLocal = self.numberOfTokensLocal/max(0.0001,self.durationLocal)
                
                taskName = '' if taskNamesList is None else taskNamesList[i]
                
                self.logger.LogLLMresponse(self, taskName,
                                           formattedPrompts[i], 
                                           output, 
                                           keyValues = {"maxTokens": max_tokens},
                                           logPrompt = self.logPrompt)
                
                self._update_generation_metrics(self.durationLocal) #updates global counters with local duration and self.numberOfTokensLocal
                
                allOutputs.append(output)

            if isBatch:
                [sRemaining, timeToGo] = SecondsRemaining(nPrompts, batchEnd, time.time() - t0Total)
                ttgString = ''
                if sRemaining > 0:
                    ttgString = ' | to go='+timeToGo
                
                gpuInfoString = ''
                
                self.currentInfoGPU = self.TorchGPUmemory()
                if self.currentInfoGPU["totalMB"] != -1:
                    vramUsed = round(self.currentInfoGPU["reservedMB"]/1024,2)
                    gpuInfoString = ' | VRAM:'+str(vramUsed)+'GB'
                self.logger.LogDebug(f"LLM: Batch Summary: {batchEnd-batchStart} batches of ({batchEnd}/{nPrompts}) | {totalOutputLength//4} tokens | {batchDur:.2f}s | {round((totalOutputLength//4)/batchDur, 2)} tok/s"+ttgString+gpuInfoString,
                                     printToConsole=True)

                self.logger.LogDebug('GPU Memory info, end of batch: '+str(self.currentInfoGPU),
                                     printToConsole=False)
                    
                

            if isBatch: #try to clean up a little bit after batched inference
                try:
                    self.logger.LogDebug("LLM: Batched inference: try empty cache",
                                         printToConsole=False)
                    
                    del batchResults, batchFormatted #to clean up memory (does not help)
                    import gc, torch
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass
            
        self.pipe.tokenizer.pad_token_id = pad_token_id_OLD
        self.pipe.tokenizer.padding_side = padding_side_OLD
        return allOutputs
    
    # gpt4all transformers backend generation (text only)
    # NOTE: streaming only supported here; n_batch for performance tuning (on token basis) 
    #       not for "parallel" infernce like HF-transformers     
    def _generate_gpt4all(self, chatPrompt, systemPrompt=None,
                          max_tokens=None, temperature=None, top_p=None, top_k=None, 
                          repeat_penalty=None, doSample=False, streaming=False,
                          n_batch=None,
                          ):
        import traceback
        global localGenerateTokenCount, localGenerateTokenCountStr, typicalLLMstopTags

        opts = {}
        if max_tokens is not None:
            opts["max_tokens"] = max_tokens
        if n_batch is not None:
            opts["n_batch"] = n_batch

        # Default inference args
        if temperature is not None:
            opts["temp"] = temperature
       

        if top_p is not None:
            opts["top_p"] = top_p

        if top_k is not None:
            opts["top_k"] = top_k

        if repeat_penalty is not None:
            opts["repeat_penalty"] = repeat_penalty

        opts["streaming"] = streaming
        if self.useTokenCallback:
            opts["callback"] = TokenCallback

        output = ""
        try:
            with self.model.chat_session(system_prompt=systemPrompt):
                localGenerateTokenCount = 0
                localGenerateTokenCountStr = ""

                if not streaming:
                    output = self.model.generate(chatPrompt, **opts)
                else:
                    for token in self.model.generate(chatPrompt, **opts):
                        output += str(token)

                self.numberOfTokensLocal = localGenerateTokenCount or max(1, len(output) // 4) #TOBIAS: make nicer!

                # Stop tag cleanup
                for stopTag in typicalLLMstopTags:
                    if stopTag in output:
                        output = output.split(stopTag)[0]

        except Exception:
            self.logger.LogError(traceback.format_exc())

        return output.strip()

    def _generate_llamacpp(self, chatPrompt, systemPrompt=None, imagePath=None,
                           max_tokens=None, temperature=None, top_p=None,
                           ):
        import os

        opts = {}
        if max_tokens is not None:
            opts["max_tokens"] = max_tokens
        if temperature is not None:
            opts["temperature"] = temperature
        if top_p is not None:
            opts["top_p"] = top_p

        try:
            if imagePath and os.path.exists(str(imagePath)):
                response = self.model(chatPrompt, image=str(imagePath), **opts)
                self.logger.LogDebug(f"[llama.cpp] multimodal inference with {imagePath}")
            else:
                response = self.model(chatPrompt, **opts)
        except Exception as e:
            self.logger.LogWarning(f"[llama.cpp] multimodal generation failed: {e}")
            response = self.model(chatPrompt, **opts)

        if isinstance(response, dict) and "choices" in response:
            output = response["choices"][0]["text"]
        elif isinstance(response, str):
            output = response
        else:
            output = str(response)

        self.numberOfTokensLocal = max(1, len(output) // 4)

        # Stop tag cleanup
        for stopTag in typicalLLMstopTags:
            if stopTag in output:
                output = output.split(stopTag)[0]

        return output.strip()

    def _generate_academic_ai(self, prompt, systemPrompt=None,
                              max_tokens=None, temperature=None, top_p=None,
                              streaming=False):
        import requests
        import time

        # Accept either chat messages or plain text prompt.
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict) and 'role' in prompt[0]:
            messages = prompt
        else:
            messages = []
            if systemPrompt:
                messages.append({'role': 'system', 'content': systemPrompt})
            messages.append({'role': 'user', 'content': str(prompt)})

        headers = {'Content-Type': 'application/json'}
        if self.apiAuthMethod == 'bearer':
            headers['Authorization'] = f'Bearer {self.apiToken}'
        elif self.apiAuthMethod == 'x-client-id':
            headers['X-Client-ID'] = self.apiClientID
        elif self.apiAuthMethod == 'x-client-secret':
            headers['X-Client-ID'] = self.apiClientID
            headers['X-Client-Secret'] = self.apiClientSecret

        # Keep payload close to the provided reference client and retry with model-name fallbacks.
        url = f"{self.apiBaseURL}/api/v1/llm/chat"
        modelCandidates = [self.apiModelName, self.modelName]
        if self.modelName in llmModelsDict:
            modelCandidates.append(llmModelsDict[self.modelName].get('shortName', self.modelName))
        # Preserve order and uniqueness.
        modelCandidates = list(dict.fromkeys([m for m in modelCandidates if isinstance(m, str) and m.strip()]))

        response = None
        lastErrorText = ''
        for modelCandidate in modelCandidates:
            payload = {
                'model': modelCandidate,
                'messages': messages,
            }
            # Academic AI endpoint currently validates strictly and may reject
            # optional generation args (e.g., max_tokens) as extra inputs.
            # Keep payload minimal and endpoint-compatible.

            # Use separate connect/read timeout and retry transient read timeouts.
            connectTimeout = 10
            readTimeout = max(30, int(self.apiTimeout))
            reqTimeout = (connectTimeout, readTimeout)
            lastRequestException = None
            for attempt in range(3):
                try:
                    response = requests.post(url, 
                                             headers=headers, 
                                             json=payload, 
                                             timeout=reqTimeout,
                                             allow_redirects=False #  
                                            )
                    lastRequestException = None
                    break
                except requests.exceptions.ReadTimeout as e:
                    lastRequestException = e
                    waitSeconds = 2 ** attempt
                    self.logger.LogWarning(
                        f"Academic AI timeout with model='{modelCandidate}', attempt={attempt+1}/3, "
                        f"readTimeout={readTimeout}s; retry in {waitSeconds}s"
                    )
                    time.sleep(waitSeconds)
                except requests.exceptions.RequestException as e:
                    lastRequestException = e
                    break

            if lastRequestException is not None and response is None:
                raise RuntimeError(
                    f"Academic AI transport request failed for model='{modelCandidate}' after retries: {lastRequestException}"
                )

            if 300 <= response.status_code < 400:
                self.logger.LogError(f"Unexpected redirect to {response.headers.get('Location')}")

            if response.status_code < 400:
                self.apiModelName = modelCandidate
                break

            errSnippet = response.text.strip().replace('\n', ' ')
            lastErrorText = errSnippet[:800]
            self.logger.LogWarning(
                f"Academic AI request failed with model='{modelCandidate}', status={response.status_code}, body={lastErrorText}"
            )

            # For auth-related errors, do not retry model aliases.
            if response.status_code in [401, 403]:
                response.raise_for_status()

        if response is None:
            raise RuntimeError('Academic AI request failed: no response object created')

        if response.status_code >= 400:
            statusCode = response.status_code
            bodyText = '' if response.text is None else response.text.strip().replace('\n', ' ')
            bodyText = bodyText[:1200]
            raise RuntimeError(
                f"Academic AI request failed with status={statusCode}; "
                f"modelsTried={modelCandidates}; authMethod={self.apiAuthMethod}; "
                f"responseBody={bodyText}"
            )

        try:
            data = response.json()
        except Exception as e:
            raise ValueError(f"Academic AI response is not valid JSON: {e}; body={response.text[:800]}")

        # Academic AI shape (example): {"data": {"content": "...", "usage": {...}}}
        payloadData = data.get('data', data)
        content = payloadData.get('content', None)

        # Fallbacks for OpenAI-style payloads.
        if content is None and isinstance(payloadData.get('choices'), list) and payloadData['choices']:
            choice0 = payloadData['choices'][0]
            if isinstance(choice0, dict):
                msg = choice0.get('message', {})
                content = msg.get('content', choice0.get('text', ''))

        output = '' if content is None else str(content)

        usage = payloadData.get('usage', {}) if isinstance(payloadData, dict) else {}
        prompt_tokens = usage.get('promptTokens', usage.get('prompt_tokens', 0))
        completion_tokens = usage.get('completionTokens', usage.get('completion_tokens', 0))

        try:
            prompt_tokens = int(prompt_tokens)
        except Exception:
            prompt_tokens = 0
        try:
            completion_tokens = int(completion_tokens)
        except Exception:
            completion_tokens = 0

        if prompt_tokens <= 0:
            prompt_tokens = max(1, len(str(messages)) // 4)
        if completion_tokens <= 0:
            completion_tokens = max(1, len(output) // 4)

        self.apiInputTokensLocal = prompt_tokens
        self.apiOutputTokensLocal = completion_tokens
        self.numberOfPromptTokensLocal = prompt_tokens
        self.numberOfTokensLocal = completion_tokens
        self.numberOfThinkingTokensLocal = 0

        # Cost tracking in EUR based on model pricing (per 1M input/output tokens).
        localCost = (
            prompt_tokens * self.apiPriceInputEURper1M +
            completion_tokens * self.apiPriceOutputEURper1M
        ) / 1e6
        self.apiCostLocalEUR = float(localCost)
        self.apiCostGlobalEUR += self.apiCostLocalEUR
        self.apiInputTokensGlobal += prompt_tokens
        self.apiOutputTokensGlobal += completion_tokens

        if self.apiBudgetEUR is not None:
            self.apiBudgetRemainingEUR = float(self.apiBudgetEUR) - self.apiCostGlobalEUR
            if self.apiBudgetRemainingEUR < 0:
                self.logger.LogWarning(
                    f"Academic AI budget exceeded by {-self.apiBudgetRemainingEUR:.4f} EUR"
                )

        self.contextWindowExceededLocal = False
        self.contextWindowExceededTokensLocal = 0
        self.contextSizeUsedLocal = 0

        return output.strip()

    #! @brief map of file extensions to the media types accepted by the Anthropic API
    anthropicImageMediaTypes = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }

    def _BuildAnthropicImageBlocks(self, imagePath):
        """
        @brief Convert image file path(s) into base64 image content blocks for the Anthropic API.
        @param imagePath single path, list of paths, or nested list (one entry per sample)
        @return list of image content blocks; empty list if no readable image was given
        """
        import base64

        if not imagePath:
            return []

        # flatten one nesting level, so that both List[str] and List[List[str]] are accepted
        if isinstance(imagePath, (list, tuple)):
            flatPaths = []
            for item in imagePath:
                if isinstance(item, (list, tuple)):
                    flatPaths.extend(item)
                else:
                    flatPaths.append(item)
        else:
            flatPaths = [imagePath]

        imageBlocks = []
        for singlePath in flatPaths:
            pathString = str(singlePath)
            if not os.path.exists(pathString):
                self.logger.LogWarning(f"Anthropic API: image not found, skipped: {pathString}")
                continue

            fileExtension = os.path.splitext(pathString)[1].lower()
            mediaType = self.anthropicImageMediaTypes.get(fileExtension)
            if mediaType is None:
                self.logger.LogWarning(
                    f"Anthropic API: unsupported image type {fileExtension!r}, skipped: {pathString}")
                continue

            with open(pathString, 'rb') as imageFile:
                encodedImage = base64.standard_b64encode(imageFile.read()).decode('utf-8')

            imageBlocks.append({
                'type': 'image',
                'source': {'type': 'base64', 'media_type': mediaType, 'data': encodedImage},
            })

        return imageBlocks

    def _generate_anthropic_api(self, prompt, systemPrompt=None, imagePath=None,
                                max_tokens=None, temperature=None):
        import time
        import anthropic

        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict) and 'role' in prompt[0]:
            messages = [m for m in prompt if m.get('role') != 'system']
            if not systemPrompt:
                for m in prompt:
                    if m.get('role') == 'system':
                        systemPrompt = m.get('content', '')
                        break
        else:
            messages = [{'role': 'user', 'content': str(prompt)}]

        # attach images to the first user message; images are placed before the text,
        # which is the ordering recommended for the Anthropic API
        imageBlocks = self._BuildAnthropicImageBlocks(imagePath)
        if imageBlocks:
            userMessageIndex = next((i for i, m in enumerate(messages)
                                     if m.get('role') == 'user'), None)
            if userMessageIndex is None:
                raise ValueError('MyLLM: images were provided but no user message exists to attach them to')

            userMessage = dict(messages[userMessageIndex])
            existingContent = userMessage.get('content', '')
            if isinstance(existingContent, str):
                textBlocks = [{'type': 'text', 'text': existingContent}] if existingContent else []
            else:
                textBlocks = list(existingContent)

            userMessage['content'] = imageBlocks + textBlocks
            messages = list(messages)
            messages[userMessageIndex] = userMessage
            self.logger.LogDebug(f"[Anthropic API] Using {len(imageBlocks)} image(s)")

        effectiveMaxTokens = max_tokens if max_tokens is not None else 16000

        kwargs = dict(
            model=self.apiModelName,
            max_tokens=effectiveMaxTokens,
            messages=messages,
        )
        if systemPrompt:
            kwargs['system'] = systemPrompt

        if self.enable_thinking:
            # Anthropic API rejects a custom temperature while extended thinking is enabled (must stay at its default of 1).
            kwargs['thinking'] = {"type": "adaptive"}
        elif temperature is not None and self.apiSupportsTemperature:
            kwargs['temperature'] = temperature
        elif temperature is not None:
            # newer Claude models removed the sampling parameters; sending them returns HTTP 400.
            # T. Moeltner, 2026-08-21
            self.logger.LogDebug(
                f"Anthropic API: model {self.apiModelName} does not accept temperature; "
                f"requested value {temperature} is ignored")

        output = ''
        inputTokens = 0
        outputTokens = 0
        thinkingText = ''

        #! Retry transient server-side failures. A benchmark run is hundreds of sequential
        #! calls, so an 'overloaded_error' or a dropped connection somewhere in the middle is
        #! the expected failure mode, not an exceptional one -- and without a retry it discards
        #! the whole run together with everything already paid for (a 115-model text-only run
        #! died after EUR 2.59 on 2026-09-01). Rate limits and overload are retried with
        #! exponential backoff; genuine client errors (401, 400, ...) are raised immediately,
        #! since repeating them only wastes time and money.
        maxAttempts = 4
        for attempt in range(1, maxAttempts + 1):
            try:
                with self._anthropicClient.messages.stream(**kwargs) as stream:
                    final = stream.get_final_message()

                output = ''
                thinkingText = ''
                for block in final.content:
                    if block.type == 'thinking':
                        thinkingText += block.thinking
                    elif block.type == 'text':
                        output += block.text

                inputTokens = final.usage.input_tokens
                outputTokens = final.usage.output_tokens
                break

            except (anthropic.RateLimitError, anthropic.APIConnectionError,
                    anthropic.InternalServerError) as e:
                if attempt == maxAttempts:
                    self.logger.LogError(
                        f"Anthropic API still failing after {maxAttempts} attempts: {e}")
                    raise
                delaySeconds = 5 * (2 ** (attempt - 1))     #! 5s, 10s, 20s
                self.logger.LogText(
                    f"Anthropic API transient failure ({type(e).__name__}), attempt "
                    f"{attempt}/{maxAttempts}; retrying in {delaySeconds}s",
                    printToConsole=True)
                time.sleep(delaySeconds)
            except anthropic.APIStatusError as e:
                #! 429 and 5xx are transient; 4xx client errors are not worth repeating
                transient = e.status_code == 429 or e.status_code >= 500 or (
                    'overloaded' in str(getattr(e, 'message', '')).lower())
                if not transient or attempt == maxAttempts:
                    self.logger.LogError(
                        f"Anthropic API error (status={e.status_code}): {e.message}")
                    raise
                delaySeconds = 5 * (2 ** (attempt - 1))
                self.logger.LogText(
                    f"Anthropic API transient error (status={e.status_code}), attempt "
                    f"{attempt}/{maxAttempts}; retrying in {delaySeconds}s",
                    printToConsole=True)
                time.sleep(delaySeconds)
            except Exception as e:
                self.logger.LogError(f"Anthropic API unexpected error: {e}")
                raise

        self.storedThinkingMessage = thinkingText if thinkingText else None
        self.apiInputTokensLocal = inputTokens
        self.apiOutputTokensLocal = outputTokens
        self.numberOfPromptTokensLocal = inputTokens
        self.numberOfTokensLocal = outputTokens
        self.numberOfThinkingTokensLocal = max(1, len(thinkingText) // 4) if thinkingText else 0

        localCost = (
            inputTokens * self.apiPriceInputEURper1M +
            outputTokens * self.apiPriceOutputEURper1M
        ) / 1e6
        self.apiCostLocalEUR = float(localCost)
        self.apiCostGlobalEUR += self.apiCostLocalEUR
        self.apiInputTokensGlobal += inputTokens
        self.apiOutputTokensGlobal += outputTokens

        if self.apiBudgetEUR is not None:
            self.apiBudgetRemainingEUR = float(self.apiBudgetEUR) - self.apiCostGlobalEUR
            if self.apiBudgetRemainingEUR < 0:
                self.logger.LogWarning(
                    f"Anthropic API budget exceeded by {-self.apiBudgetRemainingEUR:.4f} EUR"
                )

        self.contextWindowExceededLocal = False
        self.contextWindowExceededTokensLocal = 0
        self.contextSizeUsedLocal = 0

        return output.strip()

    def _generate_ollama(self, prompt, systemPrompt=None, imagePath=None,
                         max_tokens=None, temperature=None, top_p=None,
                         streaming=False,
                         ):
        import ollama
        import os

        opts = {}
        opts["num_ctx"] = self.contextWindowMaxSize

        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        if temperature is not None:
            opts["temperature"] = temperature
        if top_p is not None:
            opts["top_p"] = top_p

        # check if prompt is already formatted as OpenAI-style messages (list of dicts with "role" key)
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict) and "role" in prompt[0]:
            # prompt is already formatted as messages from FormatChatPrompt
            messages = prompt
            # if imagePath is provided and we have a user message, add images to it
            if imagePath:
                if isinstance(imagePath, (list, tuple)):
                    images = [str(img) for img in imagePath if os.path.exists(str(img))]
                else:
                    images = [str(imagePath)] if os.path.exists(str(imagePath)) else []
                if images:
                    # find the user message and add images to it
                    for msg in messages:
                        if msg.get("role") == "user":
                            msg["images"] = images
                            self.logger.LogDebug(f"[Ollama] Using {len(images)} image(s)")
                            break
        else:
            # prompt is a raw string - build messages manually for backward compatibility
            # normalize single/list prompt
            if isinstance(prompt, (list, tuple)):
                if len(prompt) > 1:
                    self.logger.LogWarning(f"Ollama received {len(prompt)} prompts, using first only")
                prompt_text = prompt[0]
            else:
                prompt_text = prompt

            # build message
            message = {"role": "user", "content": prompt_text}
            if imagePath:
                if isinstance(imagePath, (list, tuple)):
                    images = [str(img) for img in imagePath if os.path.exists(str(img))]
                else:
                    images = [str(imagePath)] if os.path.exists(str(imagePath)) else []
                if images:
                    message["images"] = images
                    self.logger.LogDebug(f"[Ollama] Using {len(images)} image(s)")

            messages = []
            if systemPrompt:
                messages.append({"role": "system", "content": systemPrompt})

            messages.append(message)
            
        try:
            response = ollama.chat(model=self.modelName, 
                                   messages=messages, 
                                   options=opts, 
                                   stream=streaming, 
                                   think=self.enable_thinking,
                                   keep_alive="1m") #this is really needed; with 0, the model is always unloaded!!!
        except ollama.ResponseError as e:
            if "not found" in str(e):
                self.logger.LogWarning(f"Model {self.modelName} not found during generation, pulling...")
                ollama.pull(self.modelName)
                response = ollama.chat(model=self.modelName, messages=messages, 
                                       options=opts, 
                                       stream=streaming,
                                       think=self.enable_thinking,
                                       keep_alive="1m")
            else:
                raise

        output = response["message"]["content"] if "message" in response else str(response)

        if self.logThinking and "message" in response and "thinking" in response["message"]:
            self.storedThinkingMessage = response['message']['thinking']
        
        self.numberOfThinkingTokensLocal = 0 if self.storedThinkingMessage is None else len(self.storedThinkingMessage) // 4 

        # Stop tag cleanup
        for stopTag in typicalLLMstopTags:
            if stopTag in output:
                output = output.split(stopTag)[0]

        # Token/time metrics
        self.numberOfTokensLocal = max(1, len(output) // 4)

        # check if context window was exceeded (prompt + generated tokens >= context window size)
        # NOTE: use attribute access with 'or' fallback because ollama's Pydantic .get() returns None
        # (not the default) when the field is defined on the model but has value None
        try:
            if response.prompt_eval_count is None or response.eval_count is None: # sanity check for response fields
                raise AttributeError(
                    f"prompt_eval_count={response.prompt_eval_count!r}, eval_count={response.eval_count!r}"
                )
            totalTokens = response.prompt_eval_count + response.eval_count
            self.numberOfPromptTokensLocal = response.prompt_eval_count
            self.contextSizeUsedLocal = totalTokens
            #This condition is only equal, because of the mechanism it works ...:
            self.contextWindowExceededLocal = totalTokens >= self.contextWindowMaxSize
            self.contextWindowExceededTokensLocal = max(0, totalTokens - self.contextWindowMaxSize)
            if self.contextWindowExceededLocal:
                self.logger.LogWarning(f"LLM: context window exceeded by {self.contextWindowExceededTokensLocal} tokens:"
                                       f" num_ctx={self.contextWindowMaxSize}, prompt={response.prompt_eval_count},"
                                       f" response={response.eval_count}, total={totalTokens}")

        except Exception as e:
            self.contextWindowExceededLocal = False
            self.contextWindowExceededTokensLocal = 0
            self.logger.LogWarning(
                f"_generate_ollama: context-window check skipped, token flags unavailable "
                f"({type(e).__name__}: {e})"
            )

        return output.strip()

    
    def _generate_ollama_batched(self, prompts, systemPrompt=None, imagePath=None,
                         max_tokens=None, temperature=None, top_p=None, 
                         taskNamesList=None, opts=None):

        import asyncio
        from ollama import AsyncClient
        
        if imagePath is not None:
            raise ValueError('MyLLM._generate_ollama_batched(...) not adjusted for images')
        
        isBatch = True
        # 1. Standardize input
        if isinstance(prompts, str):
            isBatch = False
            prompts = [prompts]
        
        nPrompts = len(prompts)
        allOutputs = [None] * nPrompts # Maintain order
        self.t0Total = time.time()
        self.globalStartTokens = self.numberOfTokensGlobal
        self.globalStartThinkingTokens = self.numberOfThinkingTokensGlobal
        self.completedCount = 0
        
        # 2. Define the async worker
        async def run_ollama_batch():
            client = AsyncClient()
            # Use a semaphore to respect your self.batchSize (Ollama's OLLAMA_NUM_PARALLEL)
            semaphore = asyncio.Semaphore(self.batchSize)
    
            async def call_chat(index, prompt):
                async with semaphore:
                    # Prepare messages
                    output = ''
                    contextWindowExceeded = False
                    contextWindowExceededTokens = 0
                    contextSizeUsed = 0
                    promptTokens = 0
                    try:
                        messages = []
                        if systemPrompt:
                            messages.append({'role': 'system', 'content': systemPrompt})
                        
                        user_msg = {'role': 'user', 'content': prompt}
                        messages.append(user_msg)
        
                        # Merge options
                        # Note: max_tokens in HF is num_predict in Ollama
                        ollama_opts = opts or {}
                        ollama_opts.update({
                            "num_ctx": self.contextWindowMaxSize,
                            "num_predict": max_tokens,
                        })

                        if temperature is not None:
                            ollama_opts["temperature"] = temperature

                        if top_p is not None:
                            ollama_opts["top_p"] = top_p
        
                        t0_req = time.time()
                        response = await client.chat(
                            model=self.modelName,
                            messages=messages,
                            options=ollama_opts,
                            think=self.enable_thinking,
                            keep_alive="1m"
                        )
                        dur_req = time.time() - t0_req
                        
                        output = response['message']['content']
                        thinkingMessage = None
                        if self.logThinking and "message" in response and "thinking" in response["message"]:
                            thinkingMessage = response['message']['thinking']
                        
                        # Metrics logic to match your original HF script
                        task_name = taskNamesList[index] if taskNamesList else f"Task {index}"

                        self.numberOfThinkingTokensLocal = 0 if thinkingMessage is None else len(thinkingMessage) // 4 
                        generatedTokens = response.get('eval_count', len(output) // 4 + self.numberOfThinkingTokensLocal)
                        #convert to int, if other type (None) ...
                        generatedTokens = int(generatedTokens) if str(generatedTokens).isdigit() else 0
                        
                        self.numberOfTokensLocal = max(0,generatedTokens - self.numberOfThinkingTokensLocal) #this shall only be the non-thinking tokens
                        promptEvalCount = response.get('prompt_eval_count', None)
                        self.numberOfPromptTokensLocal = int(promptEvalCount) if promptEvalCount is not None else len(prompt) // 4
                        self.contextSizeUsedLocal = self.numberOfPromptTokensLocal + generatedTokens
    
                        self.durationLocal = dur_req
    
                        self.tokensPerSecondLocal = (self.numberOfTokensLocal + self.numberOfThinkingTokensLocal )/max(0.0001,self.durationLocal)
                        
                        # Log using your existing logger
                        self.logger.LogLLMresponse(self, task=task_name, prompt=prompt, 
                                                   response=output,
                                                   keyValues={"maxTokens": max_tokens},
                                                   logPrompt = self.logPrompt,
                                                   thinkingMessage=thinkingMessage,
                                                   )
    
                        #%% +++++++++++++++++++++++++++++++
                        globalDuration = time.time() - self.t0Total
                        [sRemaining, timeToGo] = SecondsRemaining(nPrompts, index, globalDuration)
                        # self.logger.LogDebug(f"LLM: time to go {timeToGo}", 
                        #                      printToConsole=True)
        
                        ttgString = ''
                        if sRemaining > 0:
                            ttgString = ' | to go='+timeToGo
        
                        showTiming = True
                        self.completedCount += 1
                        nTok = self.numberOfTokensGlobal - self.globalStartTokens 
                        nTokThinking = self.numberOfThinkingTokensGlobal - self.globalStartThinkingTokens
    
                        #this is not compatible with async computation: self._update_generation_metrics(self.durationLocal) #updates global counters with local duration and self.numberOfTokensLocal
                        self.numberOfTokensGlobal += self.numberOfTokensLocal
                        self.numberOfThinkingTokensGlobal += self.numberOfThinkingTokensLocal
                        self.numberOfPromptTokensGlobal += self.numberOfPromptTokensLocal

                        # check if context window was exceeded
                        contextWindowExceededTokens = 0
                        try:
                            if response.prompt_eval_count is None or response.eval_count is None:
                                raise AttributeError(
                                    f"prompt_eval_count={response.prompt_eval_count!r}, eval_count={response.eval_count!r}"
                                )
                            totalTokens = response.prompt_eval_count + response.eval_count
                            self.contextSizeUsedLocal = totalTokens
                            contextSizeUsed = totalTokens
                            contextWindowExceeded = totalTokens >= self.contextWindowMaxSize
                            contextWindowExceededTokens = max(0, totalTokens - self.contextWindowMaxSize)
                            if contextWindowExceeded:
                                self.logger.LogWarning(f"LLM: context window exceeded by {contextWindowExceededTokens} tokens:"
                                                       f" num_ctx={self.contextWindowMaxSize}, prompt={response.prompt_eval_count},"
                                                       f" response={response.eval_count}, total={totalTokens}")

                        except Exception as e:
                            contextWindowExceeded = False
                            contextWindowExceededTokens = 0
                            self.logger.LogWarning(
                                f"_generate_ollama_batched: context-window check skipped, token flags unavailable "
                                f"({type(e).__name__}: {e})"
                            )
                        
                        printUpdate = 20
                        if self.completedCount % printUpdate == 0 and self.completedCount > 1 and showTiming:
                            tokensPerSecondBatch = (nTok+nTokThinking)/max(0.01,globalDuration)
                            strThinking = "" if nTokThinking==0 else f"{nTokThinking} thkTks | "
                            self.logger.LogDebug(f"LLM: OL Status Summary: {self.completedCount}/{nPrompts} prompts | "+
                                                 f"{nTok} tokens | "+
                                                 strThinking+
                                                 f"time={Seconds2HHMMSS(globalDuration)} | {round(tokensPerSecondBatch,1)} tok/s"+ttgString,
                                                 printToConsole=True)
                            
                        #%% +++++++++++++++++++++++++++++++
                    except KeyboardInterrupt:
                        print("Keyboard interrupt requested...exiting"); raise
                    except Exception as e:
                        self.logger.LogWarning(f"Ollama chat raise exception: {e}")

                    
                    promptTokens = self.numberOfPromptTokensLocal
                    return index, output, contextWindowExceeded, contextWindowExceededTokens, contextSizeUsed, promptTokens
    
            # Create tasks for ALL prompts
            tasks = [call_chat(i, p) for i, p in enumerate(prompts)]
            return await asyncio.gather(*tasks)


        if taskNamesList is not None:
            self.logger.LogDebug("LLM: Ollama Start "+"batched "*isBatch+f"tasks: {taskNamesList}", separator=True)
    
        # 3. Execute the async loop from your sync function
        # On Windows, sometimes loop management is tricky; this is the safest way:
        loop = asyncio.get_event_loop()
        batch_results = loop.run_until_complete(run_ollama_batch())
    
        # 4. Final Processing & Metrics
        allContextWindowExceeded = [False] * nPrompts
        allContextWindowExceededTokens = [0] * nPrompts
        allContextSizeUsed = [0] * nPrompts
        allPromptTokens = [0] * nPrompts
        for index, content, cwExceeded, cwExceededTokens, contextSizeUsed, promptTokens in batch_results:
            allOutputs[index] = content
            allContextWindowExceeded[index] = cwExceeded
            allContextWindowExceededTokens[index] = cwExceededTokens
            allContextSizeUsed[index] = contextSizeUsed
            allPromptTokens[index] = promptTokens
        self.contextWindowExceededList = allContextWindowExceeded
        self.contextWindowExceededTokensList = allContextWindowExceededTokens
        self.contextSizeUsedList = allContextSizeUsed
        self.numberOfPromptTokensList = allPromptTokens
    
        totalDur = time.time() - self.t0Total
        self.logger.LogDebug(f"LLM: Ollama Parallel Summary: {nPrompts} prompts | {totalDur:.2f}s | {nPrompts/totalDur:.2f} req/s")
        self.durationGlobal += totalDur

        self.tokensPerSecondGlobal = round((self.numberOfTokensGlobal + self.numberOfThinkingTokensGlobal) / max(0.0001, self.durationGlobal), 3)
        self.tokensPerSecondLocal = self.tokensPerSecondGlobal #no other metric available here ...
    
        return allOutputs
    
    def _RunVllmCanary(self):
        """@brief Verify a freshly loaded vLLM engine actually produces valid tokens.
           @return None; raises RuntimeError when the engine comes up corrupted

           A NaN forward pass makes every sequence emit token id 0 regardless of the prompt,
           so a single trivial generation is enough to detect it. Running this at load time
           separates two cases that are otherwise confounded, because the first real call is
           a 115-prompt batch: weights that came up corrupted (canary fails) versus
           corruption arising during generation (canary passes, batch still fails). It also
           turns a 15-minute wasted run into a one-second abort, letting the sweep retry in a
           fresh process immediately. Infrastructure problems in the canary itself are
           logged and ignored -- only an actual NaN result is fatal.
        """
        try:
            from vllm import SamplingParams
            canaryOutputs = self.model.generate(
                ['The capital of France is'],
                SamplingParams(temperature=0.0, max_tokens=8))
        except Exception as exception:
            self.logger.LogText('vLLM: canary could not run (%s); continuing without it'
                                % exception, printToConsole=True)
            return

        if self._FindNaNCorruptedOutputs(canaryOutputs, minimumTokens=4):
            text = (canaryOutputs[0].outputs[0].text or '')[:40]
            message = ('vLLM canary FAILED for %s: the freshly loaded engine emits token id '
                       '0 for a trivial prompt (%r), so the model is corrupted before any '
                       'real work. Aborting so the run can be retried in a fresh process.'
                       % (getattr(self, 'modelName', 'unknown'), text))
            self.logger.LogText('LLM ERROR: ' + message, printToConsole=True)
            raise RuntimeError(message)
        self.logger.LogText('vLLM: canary passed (engine produces valid tokens)',
                            printToConsole=True)

    def _FindNaNCorruptedOutputs(self, vllmOutputs, minimumTokens=16, zeroShareThreshold=0.8):
        """@brief Indices of sequences whose forward pass produced NaN logits.
           @param vllmOutputs list of vLLM RequestOutput objects
           @param minimumTokens ignore very short completions, too little evidence to judge
           @param zeroShareThreshold share of token id 0 above which a sequence counts as NaN
           @return list of indices into vllmOutputs

           argmax over an all-NaN logits vector returns index 0, and token id 0 decodes as an
           exclamation mark in the Qwen vocabulary, so a corrupted forward pass emits token 0
           until it hits the cap and never produces EOS. This is NOT the repeated-text
           degeneracy checked further down: it strikes an arbitrary SUBSET of a batch (191 of
           1035 sequences on 2026-08-30), so any test on the batch fraction misses it. Every
           sequence is therefore checked individually.
        """
        nanIndices = []
        for index, vllmOutput in enumerate(vllmOutputs):
            tokenIds = list(vllmOutput.outputs[0].token_ids or [])
            if len(tokenIds) < minimumTokens:
                continue
            zeroShare = sum(1 for tokenId in tokenIds if tokenId == 0) / len(tokenIds)
            if zeroShare > zeroShareThreshold:
                nanIndices.append(index)
        return nanIndices

    def _WriteNaNDiagnostics(self, vllmOutputs, nanIndices, maxTokens, contextLabel, attempt):
        """@brief Dump a corrupted batch so the failure can be reproduced faithfully later.
           @param vllmOutputs list of vLLM RequestOutput objects
           @param nanIndices indices identified by _FindNaNCorruptedOutputs
           @param maxTokens generation cap in force for this call
           @param contextLabel which vLLM entry point was used, generate or chat
           @param attempt 1 for the first occurrence, 2 for the retry

           The event is rare -- 1660 synthetic generations at production settings failed to
           reproduce it (2026-08-31) -- so the only realistic way to diagnose it is to capture
           the real batch at the moment it happens.
        """
        import json
        record = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'modelName': getattr(self, 'modelName', 'unknown'),
            'entryPoint': contextLabel,
            'attempt': attempt,
            'maxTokens': maxTokens,
            'batchSize': len(vllmOutputs),
            'corruptedCount': len(nanIndices),
            'sequences': [],
        }
        for index in nanIndices[:20]:
            completion = vllmOutputs[index].outputs[0]
            tokenIds = list(completion.token_ids or [])
            promptValue = getattr(vllmOutputs[index], 'prompt', None)
            record['sequences'].append({
                'index': index,
                'promptTokens': len(vllmOutputs[index].prompt_token_ids or []),
                'generatedTokens': len(tokenIds),
                'finishReason': completion.finish_reason,
                'zeroShare': round(sum(1 for t in tokenIds if t == 0) / max(1, len(tokenIds)), 4),
                'textHead': (completion.text or '')[:80],
                'promptHead': promptValue[:300] if isinstance(promptValue, str) else '',
            })
        try:
            logDir = getattr(self.logger, 'logDir', '.')
            fileName = os.path.join(
                logDir, 'nanCorruption_%s.json' % time.strftime('%Y%m%d_%H%M%S'))
            with open(fileName, 'w', encoding='utf-8') as fileHandle:
                fileHandle.write(json.dumps(record, indent=2, ensure_ascii=False))
            self.logger.LogText('LLM: NaN diagnostics written to ' + fileName,
                                printToConsole=True)
        except Exception as exception:
            self.logger.LogText('LLM: could not write NaN diagnostics: %s' % exception,
                                printToConsole=True)

    def _GuardAgainstNaNOutputs(self, vllmOutputs, regenerate, maxTokens, contextLabel):
        """@brief Detect a NaN-corrupted batch, retry it once, abort if it stays corrupted.
           @param vllmOutputs list of vLLM RequestOutput objects from the first attempt
           @param regenerate callable re-running the identical generation call
           @param maxTokens generation cap in force for this call
           @param contextLabel which vLLM entry point was used, generate or chat
           @return tuple of (outputs to use, extra seconds spent on the retry)

           A silently corrupted batch is the dangerous case: it still writes a plausible
           results.json, which is how a zero-percent row twice came close to reaching the
           tables. Retrying once also separates a transient fault from a reproducible one,
           and both answers are worth recording.
        """
        nanIndices = self._FindNaNCorruptedOutputs(vllmOutputs)
        if not nanIndices:
            return vllmOutputs, 0.0

        self.logger.LogText(
            'LLM WARNING: NaN forward pass -- %d/%d sequences are almost entirely token id 0. '
            'Retrying the batch once.' % (len(nanIndices), len(vllmOutputs)),
            printToConsole=True)
        self._WriteNaNDiagnostics(vllmOutputs, nanIndices, maxTokens, contextLabel, attempt=1)

        retryStart = time.time()
        retryOutputs = regenerate()
        retryDuration = time.time() - retryStart

        retryNanIndices = self._FindNaNCorruptedOutputs(retryOutputs)
        if retryNanIndices:
            self._WriteNaNDiagnostics(retryOutputs, retryNanIndices, maxTokens,
                                      contextLabel, attempt=2)
            message = ('vLLM NaN forward pass persists after one retry: %d/%d sequences emit '
                       'token id 0. The run is aborted so it cannot write corrupted results.'
                       % (len(retryNanIndices), len(retryOutputs)))
            self.logger.LogText('LLM ERROR: ' + message, printToConsole=True)
            raise RuntimeError(message)

        self.logger.LogText(
            'LLM: retry after NaN corruption came back clean (%d sequences affected on the '
            'first attempt) -- the fault is transient, not input-dependent.' % len(nanIndices),
            printToConsole=True)
        return retryOutputs, retryDuration

    def _generate_vllm(self, prompts, systemPrompt=None, imagePath=None,
                       max_tokens=None, temperature=None):
        """@brief Generate with the in-process vLLM engine.

           Submits EVERY prompt in one llm.generate() call on purpose. vLLM schedules them
           itself with continuous batching, admitting new sequences as others finish, so
           chunking the list here (as the hftransformers path must) would throw away the
           scheduling advantage that makes this backend ~10x faster.
           @param prompts str or list of prompt strings
           @param systemPrompt optional system prompt applied to every request
           @param imagePath None, one path, a flat list (one per prompt), or a per-prompt list
           @return list of generated strings, in the same order as prompts
        """
        from vllm import SamplingParams
        from PIL import Image

        promptList = list(prompts) if isinstance(prompts, (list, tuple)) else [prompts]
        nPrompts = len(promptList)
        isGptOss = 'gpt-oss' in self.modelName.lower()

        #! normalise imagePath into exactly one path (or None) per prompt; callers pass any of
        #! None / 'p.png' / ['a.png','b.png'] / [['a.png'], ['b.png']]
        def _SingleImagePath(entry):
            while isinstance(entry, (list, tuple)):
                entry = entry[0] if entry else None
            return entry

        if imagePath is None:
            imagePathPerPrompt = [None] * nPrompts
        elif isinstance(imagePath, (list, tuple)) and len(imagePath) == nPrompts:
            imagePathPerPrompt = [_SingleImagePath(entry) for entry in imagePath]
        else:
            imagePathPerPrompt = [_SingleImagePath(imagePath)] * nPrompts

        #! mistral-format models take the raw conversation via llm.chat() further below, so
        #! skip the pre-templating entirely rather than build prompts (and open every image)
        #! that would then be discarded
        requests = []
        for promptIndex, promptText in ([] if getattr(self, '_vllmUsesMistralFormat', False)
                                        else list(enumerate(promptList))):
            singleImagePath = imagePathPerPrompt[promptIndex]
            messages = []
            if systemPrompt:
                messages.append({'role': 'system', 'content': str(systemPrompt)})
            if singleImagePath:
                #! multimodal templates expect a list of typed content blocks
                userContent = [{'type': 'image'}, {'type': 'text', 'text': str(promptText)}]
            else:
                #! text-only chat templates (e.g. Qwen2.5-Coder) concatenate message['content']
                #! directly and raise "can only concatenate str (not list) to str" if handed the
                #! block-list form, so plain text must stay a plain string here
                userContent = str(promptText)
            messages.append({'role': 'user', 'content': userContent})

            #! Qwen3.5/3.6 gate their reasoning mode through a chat-template variable, and the
            #! template defaults it to ON. The hftransformers path passes enable_thinking here
            #! (see the _chat_tmpl_kwargs block in the batched vision branch); this path did not,
            #! so every Qwen3.5/3.6 run under vLLM produced a <think> block regardless of the
            #! --enable_thinking flag, which defaults to False. Measured cost on Qwen3.6-27B:
            #! 1456-2532 output tokens per sample against 442 for a non-thinking model, ~21 s
            #! per sample, and results that would not be comparable with the published runs --
            #! those went through the HF path with thinking off. Found 2026-08-29.
            chatTemplateKwargs = (
                {'enable_thinking': self.enable_thinking}
                if ('qwen3.5' in self.modelName.lower() or 'qwen3.6' in self.modelName.lower())
                else {}
            )
            #! gpt-oss always reasons; its template takes the effort level instead. Mirrors the
            #! ollama path, where modelBuilderDriver maps thinkingMode 0/1/2 to low/medium/high.
            if isGptOss:
                chatTemplateKwargs = {'reasoning_effort': self.enable_thinking
                                      if isinstance(self.enable_thinking, str) else 'low'}
            templatedPrompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chatTemplateKwargs)

            request = {'prompt': templatedPrompt}
            if singleImagePath:
                request['multi_modal_data'] = {'image': Image.open(singleImagePath).convert('RGB')}
            requests.append(request)

        samplingParams = SamplingParams(
            temperature=float(temperature) if temperature is not None else 0.0,
            max_tokens=max_tokens or 512,
            #! gpt-oss: keep the harmony channel markers so the final answer can be split off
            skip_special_tokens=not isGptOss,
        )

        # Mistral-format models are served with vLLM's own mistral tokenizer, which cannot
        # consume the pre-templated prompt string built above (it raises "Failed to apply
        # PixtralProcessor on data={'text': '[IMG]'}"). Hand the raw conversation to
        # llm.chat() instead and let vLLM template and bind the image itself; images travel
        # as base64 data URLs because chat() does not take PIL objects. Verified 2026-08-29:
        # this is the path on which Ministral-3-8B first described a test image correctly.
        if getattr(self, '_vllmUsesMistralFormat', False):
            import base64
            conversations = []
            for promptIndex, promptText in enumerate(promptList):
                singleImagePath = imagePathPerPrompt[promptIndex]
                messages = []
                if systemPrompt:
                    messages.append({'role': 'system', 'content': str(systemPrompt)})
                if singleImagePath:
                    with open(singleImagePath, 'rb') as imageFile:
                        encodedImage = base64.b64encode(imageFile.read()).decode('ascii')
                    userContent = [
                        {'type': 'image_url',
                         'image_url': {'url': f'data:image/png;base64,{encodedImage}'}},
                        {'type': 'text', 'text': str(promptText)}]
                else:
                    userContent = str(promptText)
                messages.append({'role': 'user', 'content': userContent})
                conversations.append(messages)

            gpuInfoStart = self.TorchGPUmemory()
            self.logger.LogDebug(
                f"LLM: vLLM chat START (mistral format) | prompts={len(conversations)} | "
                f"reserved={gpuInfoStart['reservedMB']/1024:.2f}GB", printToConsole=True)
            startTimeLocal = time.time()
            vllmOutputs = self.model.chat(conversations, samplingParams)
            durationLocal = max(0.0001, time.time() - startTimeLocal)
            vllmOutputs, retryDuration = self._GuardAgainstNaNOutputs(
                vllmOutputs, lambda: self.model.chat(conversations, samplingParams),
                max_tokens, 'chat')
            durationLocal += retryDuration

            outputs = []
            self._vllmContextStats = []
            for vllmOutput in vllmOutputs:
                completion = vllmOutput.outputs[0]
                outputs.append(completion.text)
                promptTokens = len(vllmOutput.prompt_token_ids or [])
                generatedTokens = len(completion.token_ids or [])
                self._vllmContextStats.append({
                    'promptTokens': promptTokens,
                    'generatedTokens': generatedTokens,
                    'contextSizeUsed': promptTokens + generatedTokens,
                })
            totalGenerated = sum(s['generatedTokens'] for s in self._vllmContextStats)
            self.logger.LogDebug(
                f"LLM: vLLM chat END | prompts={len(conversations)} | {durationLocal:.2f}s | "
                f"{totalGenerated} tokens | {totalGenerated/durationLocal:.1f} tok/s aggregate",
                printToConsole=True)
            return outputs

        gpuInfoStart = self.TorchGPUmemory()
        self.logger.LogDebug(
            f"LLM: vLLM generate START | prompts={nPrompts} | "
            f"reserved={gpuInfoStart['reservedMB']/1024:.2f}GB", printToConsole=True)

        startTimeLocal = time.time()
        vllmOutputs = self.model.generate(requests, samplingParams)
        durationLocal = max(0.0001, time.time() - startTimeLocal)
        vllmOutputs, retryDuration = self._GuardAgainstNaNOutputs(
            vllmOutputs, lambda: self.model.generate(requests, samplingParams),
            max_tokens, 'generate')
        durationLocal += retryDuration

        #! llm.generate() returns results in submission order, so index i still maps to prompt i
        outputs = []
        self._vllmContextStats = []
        for vllmOutput in vllmOutputs:
            completion = vllmOutput.outputs[0]
            text = completion.text
            if isGptOss:
                #! ollama returns only the 'final' channel as content (reasoning goes to
                #! message.thinking), so do the same; a sequence cut off while still in the
                #! analysis channel has no answer, exactly as under ollama
                finalMarker = '<|channel|>final<|message|>'
                text = text.split(finalMarker)[-1] if finalMarker in text else ''
                for endTag in ('<|return|>', '<|end|>'):
                    text = text.replace(endTag, '')
            outputs.append(text)
            promptTokens = len(vllmOutput.prompt_token_ids or [])
            generatedTokens = len(completion.token_ids or [])
            self._vllmContextStats.append({
                'promptTokens': promptTokens,
                'generatedTokens': generatedTokens,
                'contextSizeUsed': promptTokens + generatedTokens,
            })

        totalGenerated = sum(stat['generatedTokens'] for stat in self._vllmContextStats)

        #! Runaway-generation guard. The real failure is degenerate output: every prompt runs to
        #! the cap emitting a single repeated token (id 0, which decodes as '!'), seen on
        #! Qwen3-VL-8B, Qwen3-VL-32B and Qwen3.5-27B. Hitting the cap ALONE is not that: the
        #! element-selection task caps at 150 tokens, and a verbose model that explains itself
        #! before answering hits it legitimately -- Qwen3.5-27B did so on 919/1035 prompts of a
        #! perfectly good image_only run, which an earlier version of this check aborted by
        #! mistake (2026-08-31). So require BOTH: nearly every prompt capped, AND the text itself
        #! degenerate, i.e. one character making up almost the whole output.
        cappedCount = sum(1 for vllmOutput in vllmOutputs
                          if (vllmOutput.outputs[0].finish_reason or '') == 'length')
        if nPrompts >= 20 and cappedCount >= 0.8 * nPrompts:
            sampleSize = min(50, len(vllmOutputs))
            degenerateCount = 0
            for vllmOutput in vllmOutputs[:sampleSize]:
                text = (vllmOutput.outputs[0].text or '').strip()
                if len(text) < 20:
                    continue
                #! Two shapes of degeneracy, both seen: a single repeated character ('!!!!',
                #! 2026-08-29) and a repeated multi-character pattern ('BOUNDifoBOUNDifo...',
                #! 2026-08-31, produced by the CODER). A character-frequency test catches only
                #! the first -- the second spreads over several characters and slips through.
                #! Compressed size catches both: degenerate text compresses to almost nothing,
                #! while real prose, JSON and code do not fall below ~0.2 of their length.
                mostCommonShare = max(text.count(character) for character in set(text)) / len(text)
                compressedShare = len(zlib.compress(text.encode('utf-8', 'ignore'))) / len(text)
                if mostCommonShare > 0.9 or compressedShare < 0.12:
                    degenerateCount += 1
            if degenerateCount >= 0.8 * sampleSize:
                message = (f"vLLM runaway generation: {cappedCount}/{nPrompts} prompts hit the "
                           f"{max_tokens or 512}-token cap AND {degenerateCount}/{sampleSize} "
                           f"sampled outputs are a single repeated character. The run is aborted "
                           f"so it can be retried rather than producing unusable output.")
                self.logger.LogText('LLM ERROR: ' + message, printToConsole=True)
                raise RuntimeError(message)
            self.logger.LogText(
                f"LLM: {cappedCount}/{nPrompts} prompts hit the {max_tokens or 512}-token cap, "
                f"but the text is not degenerate -- a verbose model, not runaway generation; "
                f"continuing.", printToConsole=True)

        self.logger.LogDebug(
            f"LLM: vLLM generate END | prompts={nPrompts} | {durationLocal:.2f}s | "
            f"{totalGenerated} tokens | {totalGenerated/durationLocal:.1f} tok/s aggregate",
            printToConsole=True)
        return outputs

    def _update_generation_metrics(self, durationSingleInference):

        self.tokensPerSecondLocal = round((self.numberOfTokensLocal + self.numberOfThinkingTokensLocal) / durationSingleInference, 3)
        self.numberOfTokensGlobal += self.numberOfTokensLocal
        self.numberOfThinkingTokensGlobal += self.numberOfThinkingTokensLocal
        self.numberOfPromptTokensGlobal += self.numberOfPromptTokensLocal 


        self.durationGlobal += durationSingleInference
        self.tokensPerSecondGlobal = round((self.numberOfTokensGlobal + self.numberOfThinkingTokensGlobal)/ max(0.0001, self.durationGlobal), 3)

    
    def Generate(self, prompt, systemPrompt = None, imagePath = None,
                max_tokens=None, temperature=None, top_p=None, top_k=None, repeat_penalty=None, doSample=False, numBeams=1,
                streaming=False, showTiming=True, returnFullText=True, cleanUpTokenizationSpaces=False,
                n_batch = None, taskNamesList = None):
        """
        Unified generation for both text and multimodal inputs (single or batched).

        Args:
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): Sampling temperature for all backends.
            top_p (float): Nucleus sampling probability for Hugging Face models.
            top_k (int): The top_k parameter for GPT4All (only applicable for GPT4All).
            repeat_penalty (float): Penalty for repeated tokens in GPT4All.
            doSample (bool): Whether to sample or not (greedy decoding if False).
            streaming (bool): Whether to stream tokens during inference or not (only GPT4All).
            showTiming (bool): Whether to show output generation time or not.
            returnFullText(bool): Whether to return full output (True), including the input, or only the added text (False) (only HF-transformers).
            cleanUpTokenizationSpaces (bool): Whether or not to clean up the potential extra spaces in the text output (only HF-transformers).
            taskNamesList: a list of names (or short descriptions) per prompt which is printed in log; otherwise None
        Returns:
            str or list of strings: single LLM response or list of responses for batched input.
        """
        
        self.storedThinkingMessage = None #for ollama
        self.numberOfGenerateCalls += 1
        isBatch = not isinstance(prompt, str)
        nPrompts = 1 if not isBatch else len(prompt)

        if self.logPrompt:
            self.logger.LogText('', timestamp=False, separator=True)
            self.logger.LogDebug(f"LLM: Generate() with backend={self.backend}, nPrompts={nPrompts}, maxBatches={self.batchSize}")

        # normalize prompt(s) for batched and single prompt
        prompts = prompt if isBatch else [prompt]
        
        if taskNamesList is not None:
            if not isinstance(taskNamesList, list) or len(taskNamesList) != len(prompts):
                self.logger.LogError('LLM: Generate: taskNamesList is not a list or length does not match length of prompt list')
                raise ValueError('LLM: Generate: taskNamesList mismatch')

        self.numberOfTokensLocal = 0 
        self.numberOfThinkingTokensLocal = 0
        self.numberOfPromptTokensLocal = 0
        self.apiInputTokensLocal = 0
        self.apiOutputTokensLocal = 0
        self.apiCostLocalEUR = 0.0

        startTime = time.time()
        self.globalStartTokens = self.numberOfTokensGlobal
        lastPrintTokens = self.numberOfTokensGlobal
        globalStartApiCost = self.apiCostGlobalEUR

        self.globalStartThinkingTokens = self.numberOfThinkingTokensGlobal
        lastPrintThinkingTokens = self.numberOfThinkingTokensGlobal

        lastPrintTime = startTime
        
        # backend-specific generation
        if self.backend.lower() == "vllm":
            vllmStartTime = time.time()
            outputs = self._generate_vllm(
                prompts,
                systemPrompt=systemPrompt,
                imagePath=imagePath,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            vllmDuration = max(0.0001, time.time() - vllmStartTime)
            nOut = len(outputs)
            vllmStats = getattr(self, '_vllmContextStats', None) or []

            self.contextWindowExceededList = [False] * nOut
            self.contextWindowExceededTokensList = [0] * nOut
            self.contextSizeUsedList = []
            self.numberOfPromptTokensList = []

            perCallDuration = vllmDuration / max(1, nOut)
            for i, output in enumerate(outputs):
                outStr = str(output)
                stat = vllmStats[i] if i < len(vllmStats) else {}
                promptTokens = int(stat.get('promptTokens', 0) or 0)
                contextUsed = int(stat.get('contextSizeUsed', 0) or 0)
                generatedTokens = int(stat.get('generatedTokens', 0) or 0)

                self.numberOfPromptTokensList.append(promptTokens)
                self.contextSizeUsedList.append(contextUsed)
                # context overflow cannot silently occur here: vLLM rejects or truncates
                # requests beyond max_model_len rather than running past it
                if self.contextWindowMaxSize and contextUsed > self.contextWindowMaxSize:
                    self.contextWindowExceededList[i] = True
                    self.contextWindowExceededTokensList[i] = contextUsed - self.contextWindowMaxSize

                self.numberOfPromptTokensLocal = promptTokens
                self.numberOfTokensLocal = generatedTokens
                self.numberOfThinkingTokensLocal = 0
                #! consumed by Logger.LogLLMresponse for the "context size used" field
                self.contextSizeUsedLocal = contextUsed
                self.durationLocal = perCallDuration
                self._update_generation_metrics(perCallDuration)

                taskName = '' if taskNamesList is None else taskNamesList[i]
                self.logger.LogLLMresponse(
                    self, taskName,
                    prompts[i] if isinstance(prompts, (list, tuple)) else prompts,
                    outStr,
                    keyValues={'maxTokens': max_tokens},
                    logPrompt=self.logPrompt,
                )

        elif self.backend.lower() == "hftransformers":
            hfStartTime = time.time()
            outputs = self._generate_hf(
                prompts,
                systemPrompt=systemPrompt,
                imagePath=imagePath,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                doSample=doSample,
                numBeams=numBeams,
                returnFullText=returnFullText,
                cleanUpTokenizationSpaces=cleanUpTokenizationSpaces,
                taskNamesList=taskNamesList,
            )
            hfDuration = time.time() - hfStartTime
            nOut = len(outputs)
            self.contextWindowExceededList = [False] * nOut  # HF backend does not track context window exceeded
            self.contextWindowExceededTokensList = [0] * nOut
            self.contextSizeUsedList = []
            self.numberOfPromptTokensList = []
            promptList = prompts if isinstance(prompts, (list, tuple)) else [prompts]
            formattedPrompts = (
                None if self.isVisionModel
                else self.FormatChatPrompt(promptList, systemPrompt)
            )
            perCallDur = hfDuration / max(1, nOut)
            visionStats = getattr(self, '_hfVisionContextStats', None) or []

            for i, output in enumerate(outputs):
                outText = output[0] if isinstance(output, list) and output else output
                outStr = str(outText)

                if i < len(visionStats):
                    promptTokens = int(visionStats[i].get('promptTokens', 0) or 0)
                    contextUsed = int(visionStats[i].get('contextSizeUsed', 0) or 0)
                else:
                    promptText = promptList[i] if i < len(promptList) else promptList[0]
                    if formattedPrompts is not None and i < len(formattedPrompts):
                        promptText = formattedPrompts[i]
                    promptTokens = self._hfTokenCount(promptText)
                    fullTokens = self._hfTokenCount(outStr)
                    if fullTokens > promptTokens:
                        contextUsed = fullTokens
                    else:
                        genTokens = max(1, len(outStr) // 4)
                        contextUsed = promptTokens + genTokens

                genTokens = max(1, contextUsed - promptTokens) if contextUsed > promptTokens else max(1, len(outStr) // 4)
                self.numberOfPromptTokensList.append(promptTokens)
                self.contextSizeUsedList.append(contextUsed)

                # Vision: log here; text-only HF logs inside _generate_hf per prompt.
                if self.isVisionModel:
                    self.storedThinkingMessage = None
                    self.durationLocal = perCallDur
                    self.numberOfPromptTokensLocal = promptTokens
                    self.numberOfTokensLocal = genTokens
                    self.contextSizeUsedLocal = contextUsed
                    self._update_generation_metrics(self.durationLocal)
                    promptText = promptList[i] if i < len(promptList) else promptList[0]
                    taskName = '' if taskNamesList is None else taskNamesList[i]
                    self.logger.LogLLMresponse(
                        self, taskName, promptText, outText,
                        keyValues={"maxTokens": max_tokens},
                        logPrompt=self.logPrompt,
                        thinkingMessage=self.storedThinkingMessage,
                    )
                    self.storedThinkingMessage = None

            if self.isVisionModel:
                self._hfVisionContextStats = []
        elif self.backend.lower() in ("academic_ai", "anthropic_api"):
            outputs = []
            self.contextWindowExceededList = [False] * len(prompts)
            self.contextWindowExceededTokensList = [0] * len(prompts)
            self.contextSizeUsedList = [0] * len(prompts)
            self.numberOfPromptTokensList = [0] * len(prompts)
            for i, singlePrompt in enumerate(prompts):
                self.storedThinkingMessage = None
                if taskNamesList is not None:
                    self.logger.LogDebug(f"LLM: Start task: {taskNamesList[i]}", separator=True)

                chatPrompt = self.FormatChatPrompt([singlePrompt], systemPrompt)[0]
                startTimeLocal = time.time()
                if self.backend.lower() == "academic_ai":
                    if imagePath:
                        # fail loudly instead of silently answering without the image
                        raise ValueError(
                            'MyLLM: academic_ai backend does not support image input; '
                            'use a backend with vision support')
                    output = self._generate_academic_ai(
                        chatPrompt,
                        systemPrompt=systemPrompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        streaming=streaming,
                    )
                else:
                    # imagePath may be a per-sample list (List[List[str]]) or a flat list
                    if isinstance(imagePath, (list, tuple)) and len(imagePath) == len(prompts):
                        imagePathLocal = imagePath[i]
                    else:
                        imagePathLocal = imagePath
                    output = self._generate_anthropic_api(
                        chatPrompt,
                        systemPrompt=systemPrompt,
                        imagePath=imagePathLocal,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                self.durationLocal = time.time() - startTimeLocal
                if self.numberOfPromptTokensLocal <= 0:
                    self.numberOfPromptTokensLocal = len(str(singlePrompt)) // 4

                self._update_generation_metrics(self.durationLocal)

                # API progress summary (always after each completed prompt)
                globalDuration = time.time() - startTime
                [sRemaining, timeToGo] = SecondsRemaining(nPrompts, i + 1, globalDuration)
                ttgString = '' if sRemaining <= 0 else ' | to go=' + timeToGo
                elapsedBatch = max(0.01, time.time() - lastPrintTime)
                tokensPerSecondBatch = (self.numberOfTokensGlobal - lastPrintTokens) / elapsedBatch
                apiCostBatch = self.apiCostGlobalEUR - globalStartApiCost
                self.logger.LogDebug(
                    f"LLM: API Status Summary: {i+1}/{nPrompts} prompts | "
                    f"{self.numberOfTokensGlobal-self.globalStartTokens} tokens | "
                    f"cost={round(apiCostBatch,6)} EUR | "
                    f"time={Seconds2HHMMSS(globalDuration)} | "
                    f"{round(tokensPerSecondBatch,1)} tok/s" + ttgString,
                    printToConsole=True,
                )
                lastPrintTokens = self.numberOfTokensGlobal
                lastPrintTime = time.time()

                taskName = '' if taskNamesList is None else taskNamesList[i]
                self.logger.LogLLMresponse(
                    self,
                    taskName,
                    singlePrompt,
                    output,
                    keyValues={
                        'maxTokens': max_tokens,
                        'apiInputTokens': self.apiInputTokensLocal,
                        'apiOutputTokens': self.apiOutputTokensLocal,
                        'apiCostEUR': round(self.apiCostLocalEUR, 6),
                        'apiCostGlobalEUR': round(self.apiCostGlobalEUR, 6),
                        'apiBudgetRemainingEUR': round(self.apiBudgetRemainingEUR, 6) if self.apiBudgetEUR is not None else 'N/A',
                    },
                    logPrompt=self.logPrompt,
                    thinkingMessage=self.storedThinkingMessage,
                )
                self.storedThinkingMessage = None
                outputs.append(output)
        elif self.backend.lower() == "ollama" and not imagePath and self.batchSize > 1:
            #batched version of ollama
            outputs = self._generate_ollama_batched(
                prompts,
                systemPrompt=systemPrompt,
                imagePath=imagePath,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                taskNamesList=taskNamesList,
            )
        else:
            outputs = []
            self.contextWindowExceededList = [] #populated per-prompt below
            self.contextWindowExceededTokensList = [] #populated per-prompt below
            self.contextSizeUsedList = [] #populated per-prompt below
            self.numberOfPromptTokensList = [] #populated per-prompt below
            for i, singlePrompt in enumerate(prompts):
                self.storedThinkingMessage = None
                if taskNamesList is not None:
                    self.logger.LogDebug(f"LLM: Start task: {taskNamesList[i]}", separator=True)
                
                chatPrompt = self.FormatChatPrompt([singlePrompt], systemPrompt)[0]
                
                startTimeLocal = time.time()
                
                if self.backend.lower() == "gpt4all":
                    output = self._generate_gpt4all(
                            chatPrompt,
                            systemPrompt=systemPrompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repeat_penalty=repeat_penalty,
                            doSample=doSample,
                            streaming=streaming,
                            n_batch=n_batch,
                        )
    
                elif self.backend.lower() == "llamacpp":
                    output = self._generate_llamacpp(
                            chatPrompt,
                            systemPrompt=systemPrompt,
                            imagePath=imagePath,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )

                elif self.backend.lower() == "ollama":
                    # Extract per-prompt image path (imagePath is list of lists)
                    imgPath_i = imagePath[i] if imagePath and i < len(imagePath) else None
                    output = self._generate_ollama(
                            chatPrompt,
                            systemPrompt=systemPrompt,
                            imagePath=imgPath_i,
                            streaming=streaming,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )

                else:
                    raise ValueError(f"Unsupported backend: {self.backend}")

                self.durationLocal = time.time() - startTimeLocal
                
                self.numberOfPromptTokensLocal = len(str(singlePrompt)) // 4 #only prompt, not the template

                # Aggregate metrics globally
                self._update_generation_metrics(self.durationLocal)
                
                taskName = '' if taskNamesList is None else taskNamesList[i] 
                
                self.logger.LogLLMresponse(self, taskName,
                                           singlePrompt, output, 
                                           keyValues = {"maxTokens": max_tokens},
                                           logPrompt = self.logPrompt,
                                           thinkingMessage = self.storedThinkingMessage)
                self.storedThinkingMessage = None
                
                #%% +++++++++++++++++++++++++++++++
                globalDuration = time.time() - startTime
                [sRemaining, timeToGo] = SecondsRemaining(nPrompts, i, globalDuration)

                ttgString = ''
                if sRemaining > 0:
                    ttgString = ' | to go='+timeToGo
                printUpdate = 20
                if i % printUpdate == 0 and i>0:
                    timeBatch = max(0.01, time.time()-lastPrintTime)
                    #tps includes thinking
                    tokensPerSecondBatch = (self.numberOfTokensGlobal - lastPrintTokens + 
                                            self.numberOfThinkingTokensGlobal - lastPrintThinkingTokens)/timeBatch
                    accumulatedThinkingTokens = self.numberOfThinkingTokensGlobal - self.globalStartThinkingTokens
                    strThinking = "" if accumulatedThinkingTokens==0 else f"{accumulatedThinkingTokens} thkTks | "

                    self.logger.LogDebug(f"LLM: Status Summary: {i}/{nPrompts} prompts | "+
                                         f"{self.numberOfTokensGlobal-self.globalStartTokens} tokens | "+
                                         strThinking+
                                         f"time={Seconds2HHMMSS(globalDuration)} "+
                                         f"| {round(tokensPerSecondBatch,1)} tok/s"+ttgString,
                                         printToConsole=True)
                    lastPrintTokens = self.numberOfTokensGlobal
                    lastPrintThinkingTokens = self.numberOfThinkingTokensGlobal

                    lastPrintTime = time.time()
                #%% +++++++++++++++++++++++++++++++
                
                outputs.append(output)
                # Track context window exceeded (only available for ollama)
                isOllama = self.backend.lower() == "ollama"
                self.contextWindowExceededList.append(self.contextWindowExceededLocal if isOllama else False)
                self.contextWindowExceededTokensList.append(self.contextWindowExceededTokensLocal if isOllama else 0)
                self.contextSizeUsedList.append(self.contextSizeUsedLocal if isOllama else 0)
                self.numberOfPromptTokensList.append(self.numberOfPromptTokensLocal if isOllama else 0)

        
        return outputs if isBatch else [outputs[0]]

    #return quantization as number (4=4 bit, etc.)
    #uses either model information, or re-quantization for transformers
    def GetQuantization(self):
        if self.backend == 'hftransformers':
            if self.modelQuantizationHF == '4bit':
                return 4
            if self.modelQuantizationHF == '8bit':
                return 8
            # Checkpoints that ship ALREADY quantized (Ministral-3-8B/14B carry
            # quant_method='fp8' in their own config) are loaded with quantization=None,
            # because forcing a BitsAndBytesConfig on top is a hard conflict HF refuses.
            # Without the check below they fall through to the static llmModelsDict 'Q'
            # fallback and get reported as 16 -- their COMPUTE dtype (torch.bfloat16), not
            # their weight precision -- which mis-states the experiment in results.json.
            checkpointQuant = getattr(self, 'modelConfig', None)
            checkpointQuant = getattr(checkpointQuant, 'quantization_config', None)
            if checkpointQuant is not None:
                if isinstance(checkpointQuant, dict):
                    quantMethod = checkpointQuant.get('quant_method')
                else:
                    quantMethod = getattr(checkpointQuant, 'quant_method', None)
                bitsPerMethod = {'fp8': 8, 'awq': 4, 'gptq': 4}
                if quantMethod is not None and str(quantMethod).lower() in bitsPerMethod:
                    return bitsPerMethod[str(quantMethod).lower()]
        return llmModelsDict[self.modelName]['Q']


    #try to free memory, if possible
    def FreeMemory(self):
        if self.backend == 'vllm':
            # vLLM pre-allocates ~90% of VRAM for weights + paged KV cache and holds it for the
            # engine's lifetime, so this MUST fully release before a second model is constructed
            # (vision -> coder between Stage 1 and Stage 2), otherwise the next one OOMs.
            # Dropping the reference is NOT enough: vLLM holds its weights and paged KV cache
            # through its own CUDA allocator, captured CUDA graphs and the distributed
            # environment, none of which Python refcounting or empty_cache() can reach.
            # Measured with the naive version: "allocated 82.69GB -> 82.69GB", i.e. nothing
            # released, so the Stage-2 coder then loaded on top and spilled into shared
            # (host) memory over PCIe. Full teardown below, in the order vLLM requires.
            import gc
            import torch
            memoryBeforeMB = self.TorchGPUmemory().get('allocatedMB', 0.0)

            # 1) sleep(level=2) is vLLM's own API for exactly this case: it discards the KV
            #    cache and offloads weights so another model can use the device.
            try:
                if hasattr(self.model, 'sleep'):
                    self.model.sleep(level=2)
            except Exception as exceptionInstance:
                self.logger.LogDebug(f"LLM: vLLM sleep() raised {type(exceptionInstance).__name__}, "
                                     f"continuing with explicit teardown")

            # 2) tear down the executor and the distributed/model-parallel state, which own
            #    the remaining device allocations
            try:
                engine = getattr(self.model, 'llm_engine', None)
                if engine is not None:
                    executor = getattr(engine, 'model_executor', None)
                    if executor is not None:
                        if hasattr(executor, 'shutdown'):
                            executor.shutdown()
                        del executor
                        try:
                            del engine.model_executor
                        except Exception:
                            pass
                    if hasattr(engine, 'shutdown'):
                        engine.shutdown()
            except Exception as exceptionInstance:
                self.logger.LogDebug(f"LLM: vLLM engine shutdown raised "
                                     f"{type(exceptionInstance).__name__}, continuing")

            try:
                from vllm.distributed.parallel_state import (destroy_model_parallel,
                                                             destroy_distributed_environment)
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception as exceptionInstance:
                self.logger.LogDebug(f"LLM: vLLM distributed teardown raised "
                                     f"{type(exceptionInstance).__name__}, continuing")

            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            memoryAfterMB = self.TorchGPUmemory().get('allocatedMB', 0.0)
            releasedGB = (memoryBeforeMB - memoryAfterMB) / 1024
            self.logger.LogText(
                f"LLM: freed vLLM engine | allocated {memoryBeforeMB/1024:.2f}GB -> "
                f"{memoryAfterMB/1024:.2f}GB (released {releasedGB:.2f}GB)",
                printToConsole=True)
            if releasedGB < 1.0 and memoryBeforeMB / 1024 > 5.0:
                # loud, because the next model will otherwise spill into host memory over PCIe
                # and run at a small fraction of its normal speed rather than failing outright
                self.logger.LogText(
                    "LLM WARNING: vLLM teardown released almost nothing; a second model loaded "
                    "now will exceed VRAM and page over PCIe. Run stages in separate processes.",
                    printToConsole=True)

        elif self.backend == 'GPT4All':
            self.model.close()
        elif self.backend == 'hftransformers' and self.isVisionModel:
            # `del self.model` alone is NOT enough to return the weights to the driver:
            # torch modules routinely form reference CYCLES (module <-> parameters <-> hooks),
            # which CPython's refcounting cannot reclaim -- only an explicit gc.collect() can.
            # Without it the vision model stays resident after Stage 1b and the Stage-2 coder
            # loads ON TOP of it. Observed 2026-08-29: allocated at Stage-2 start was 35.5 GB
            # instead of the ~19.8 GB the coder needs alone, i.e. ~15.7 GB (a whole 32B model
            # at 4 bits) stranded, leaving only 14.3 GB free at peak and needlessly capping the
            # usable batch size.
            import gc
            import torch
            memoryBeforeMB = self.TorchGPUmemory().get('allocatedMB', 0.0)
            self.model = None       # rebind rather than del: idempotent if called twice
            # self.processor is deliberately kept: it holds no GPU weights, so clearing it
            # frees nothing, while other code still reads llm.processor.tokenizer.
            gc.collect()            # breaks the reference cycles del cannot
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # optional, for inter-process memory
            memoryAfterMB = self.TorchGPUmemory().get('allocatedMB', 0.0)
            self.logger.LogText(
                f"LLM: freed vision model | allocated {memoryBeforeMB/1024:.2f}GB -> "
                f"{memoryAfterMB/1024:.2f}GB (released {(memoryBeforeMB-memoryAfterMB)/1024:.2f}GB)",
                printToConsole=True)



    def TorchGPUmemory(self):
        try:
            import torch
        
            device = torch.device(f"cuda:{self.gpuMemTrackerDeviceIndex}")
            torch.cuda.synchronize(device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)

            gpuInfo = {
                "device": self.gpuMemTrackerDeviceIndex,
                "GPUmodel":  torch.cuda.get_device_name(self.gpuMemTrackerDeviceIndex),
                # allocatedMB = live tensor footprint; reservedMB = PyTorch's cached pool
                # (includes freed-but-not-released blocks). A gap between the two that grows
                # over successive calls points at caching-allocator fragmentation, not a true
                # leak; allocatedMB itself climbing points at real live references not being
                # freed. Re-enabled 2026-08-28 to diagnose vision-batch GPU memory growth.
                "allocatedMB": torch.cuda.memory_allocated(device) / (1024**2),
                "reservedMB": torch.cuda.memory_reserved(device) / (1024**2),
                "maxAllocatedMB": torch.cuda.max_memory_allocated(device) / (1024**2),
                "maxReservedMB": torch.cuda.max_memory_reserved(device) / (1024**2),
                "freeMB": free_bytes / (1024**2),
                "totalMB": total_bytes / (1024**2),
            }
            self.peakGPUmemory = max(self.peakGPUmemory, gpuInfo["maxReservedMB"])

            return gpuInfo
        except:
            return {
                "device": 0,
                "GPUmodel":  "",
                "allocatedMB": 0,
                "reservedMB": 0,
                "maxAllocatedMB": 0,
                "maxReservedMB": 0,
                "freeMB": 0,
                "totalMB": -1, #not available
            }
            
    
    def TorchGPUmemoryReset(self):
        try:
            import torch
            
            device = torch.device(f"cuda:{self.gpuMemTrackerDeviceIndex}")
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
