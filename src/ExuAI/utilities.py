# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: utilities.py
#
# Details:  Basic support functions for LLM-agents, and further functionalities of the LLM Lab.
#           Mainly processing text, numbers, extraction of xml-tags, logging helpers, file/json read/write, camelCase processing, latex export, ...
#
# Author:   Tobias Möltner and Johannes Gerstmayr
# Date:     2025-03-11
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import shutil
import json
import re
import ast
import textwrap
import traceback
import numpy as np
import exudyn as exu

try:
    from templatesAndLists import femInterfaceDescriptionHeader, femHeaderIntro, meshPropertiesHeader
except:
    from .templatesAndLists import femInterfaceDescriptionHeader, femHeaderIntro, meshPropertiesHeader

#create list of numbers, return int types, as numpy makes problems with saving!
def Arange(start, final):
    newList = []
    for x in np.arange(start, final+1):
        newList.append(int(x))
    return newList

#check if is scalar
def IsScalar(x):
    if isinstance(x, (np.floating, float)) or isinstance(x, (np.integer, int)): 
        return True
    else:
        return False


#check if is scalar
def IsIntType(x):
    if isinstance(x, (np.integer, int)): 
        return True
    else:
        return False


def SecondsRemaining(nTotal, nProcessed, timeSpent):
    """
    nTotal: int, total items
    nProcessed: int, items processed so far
    timeSpent: float, seconds spent so far
    Returns: (remaining_seconds:int, formatted:str)
    """
    # Handle edge cases
    if nProcessed <= 0 or nTotal <= nProcessed:
        remainingSeconds = 0
    else:
        avgPerItem = timeSpent / nProcessed
        remainingSeconds = round((nTotal - nProcessed) * avgPerItem,2)
        if remainingSeconds < 0:
            remainingSeconds = 0

    return (remainingSeconds, Seconds2HHMMSS(int(remainingSeconds)) )


def Seconds2HHMMSS(seconds):
    """
    Format seconds into 'HH:MM:SS' if hours > 0, else 'MM:SS'.
    Minutes and seconds are zero-padded; hours are not.
    """
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


# Dynamically round a number based on its magnitude.
# NOTE: this does not work with larger numbers!
def DynamicRound(x, nDigits=1):

    if x == 0:
        return 0
    
    magnitude = np.abs(x)
    decimal_places = int(np.floor(np.log10(magnitude))) + 1 - nDigits
    
    if decimal_places < 0:
        return np.round(x*(10**(-decimal_places)), 0) / (10**(-decimal_places))
    else:
        return (10**decimal_places)*np.round(x/(10**decimal_places), 0)
    

def GetStandardRandomNumber(minVal, maxVal):
        """
        Generates random values from a predefined set pattern within a given range.

        :param minVal: The minimum value of the range.
        :param maxVal: The maximum value of the range.
        :return: A numpy array of randomly selected values.
        """
        values = GetStandardNumbersList(minVal, maxVal)
        # Select random values
        randVal = np.random.choice(values, size=None, replace=True)
        randVal = round(randVal,10) #we need to round as the exp function leads to small round off errors!

        if type(randVal) == str: #Joh: can this happen?
            pass
        elif randVal == int(randVal):
            randVal = int(randVal) #we avoid '.0' in string representation
        else:
            randVal = float(randVal)  #avoid numpy float (problems in json files)!
        
        return randVal  
        

def GetStandardNumbersList(minVal, maxVal):
        """
        Generates list of values from a predefined set pattern within a given range.

        :param minVal: The minimum value of the range.
        :param maxVal: The maximum value of the range.
        :return: A numpy array of randomly selected values.
        """
        # Base pattern to be repeated across powers of 10
        base_pattern = np.array([1, 1.2, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 8])

        smallestValue = 0.01 #in case of zero
        if abs(minVal) < smallestValue and minVal != 0:
            smallestValue = abs(minVal)
        if abs(maxVal) < smallestValue and maxVal != 0:
            smallestValue = abs(maxVal)

        if minVal > maxVal:
            raise ValueError('GetStandardRandomNumber: invalid interval: minVal must be <= maxVal')
        if minVal == maxVal:
            return float(minVal)
        #++++++++++++++++++++
        def GetPosStandardNumbers(minVal, maxVal):
            # Determine the logarithmic range
        
            log_min = np.log10(minVal)
            log_max = np.log10(maxVal)
            
            # Generate scaled patterns using powers of 10
            values = []
            
            for exponent in range(int(np.floor(log_min)), int(np.ceil(log_max)) + 1):
                scaledValues = base_pattern * (10 ** exponent)
                values.extend(scaledValues[(scaledValues >= minVal) & (scaledValues <= maxVal)])
            
            values = np.array(values)
            return values
        #++++++++++++++++++++
        
        if minVal < 0 and maxVal <= 0:
            if maxVal == 0:
                maxVal = -smallestValue
            values = -GetPosStandardNumbers(-maxVal, -minVal)[::-1] #reversed
        elif minVal < 0 and maxVal >= 0:
            values = list(-GetPosStandardNumbers(smallestValue,-minVal))[::-1] #reversed
            values += list(GetPosStandardNumbers(smallestValue,maxVal))
            values = np.array(values)
        else: #all positive
            if minVal == 0:
                minVal = smallestValue
            values = GetPosStandardNumbers(minVal, maxVal)
        
        return values


#convert number to string for better readability in Excel; round to nDigits
def NumToCSV(value, nDigits=None):
    if nDigits is None:
        return str(value).replace('.',',')
    else:
        return str(round(value,nDigits)).replace('.',',')
    return 
    

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++

                     
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#join paths or path and file name
#prefer over path.join as it allows more consistent handling
def JoinPath(path, subpath, subpath2=None):
    path = path.replace('\\','/')
    subpath = subpath.replace('\\','/')
    
    if path != '' and subpath != '' and path[-1] != '/':
        path += '/'
    # Only strip a leading '/' from subpath when there is a path to join it TO; the strip
    # exists to avoid 'a//b'. With path=='' the subpath is used as-is, and on POSIX that
    # leading '/' is what makes it absolute -- removing it silently turned every absolute
    # path into a relative one. Harmless on Windows (paths start 'C:/...', no leading slash),
    # which is why this only surfaced under WSL/Linux: FileExists(filePath='', fileName=
    # '/mnt/c/.../solution.txt') resolved to 'mnt/c/.../solution.txt', so EvaluateNumerical
    # reported every reference solution as missing and returned -2 for all 115 evaluations
    # while the models were actually fine (found 2026-08-29).
    if path != '' and subpath != '' and subpath[0] == '/':
        subpath = subpath[1:]
    path = path+subpath
    
    #further subpath
    if subpath2 is not None:
        path = JoinPath(path, subpath2)
    
    return path

# ensures that directory is empty, if not all files are deleted within
def EnsureEmptyDir(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    else:
        for filename in os.listdir(dirPath):
            filePath = JoinPath(dirPath, filename)
            if os.path.isfile(filePath) or os.path.islink(filePath):
                os.remove(filePath)
            elif os.path.isdir(filePath):
                for root, dirs, files in os.walk(filePath, topdown=False):
                    for f in files:
                        os.remove(JoinPath(root, f))
                    for d in dirs:
                        os.rmdir(JoinPath(root, d))
                os.rmdir(filePath)

#get scalar quantities + addOns, put into dict and return
def GetScalarClassObjectsDict(obj, addOns=[]):
    scalarConfig = {}
    for attrName in dir(obj):
        if attrName.startswith('__'):
            continue
        try:
            attrValue = getattr(obj, attrName)
            if IsScalar(attrValue) or isinstance(attrValue,str) or attrValue in addOns:
                scalarConfig[attrName] = attrValue
            # add other attribute types that should be saved to config (e.g. string) here 
        except Exception:
            continue
    return scalarConfig


#helper function to load utf-8 files with error handling
def ReadFile(filePath, fileName):
    text = None
    fullPath = JoinPath(filePath, fileName)  # Construct the full file path
    
    if not os.path.exists(fullPath):
        raise FileNotFoundError(f"Error: File '{fullPath}' not found!")

    try:
        with open(fullPath, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        raise ValueError(f"LoadFile failed for file: '{fullPath}'. Error: {e}")
        
    return text

#helper function to write data (string or anything that can be converted to string) into utf-8 files;
# create required directories, with error handling
def WriteFile(fileName, data, path='', append=False, flush=False):
    if path != '':
        fileName = JoinPath(path, fileName)
    try:
        os.makedirs(os.path.dirname(fileName), exist_ok=True)
    except KeyboardInterrupt:
        print("Keyboard interrupt requested...exiting")
        raise
    except:
        pass #makedirs may fail on some systems, but we keep going

    try:
        mode = 'w' if append==False else 'a'
        with open(fileName, mode=mode, encoding='utf-8') as file:
            file.write(str(data))  # Append mode     
            if flush: 
                file.flush()
    except KeyboardInterrupt:
        print("Keyboard interrupt requested...exiting")
        raise
    except:
        raise ValueError('WriteFile: fileName:"'+fileName+'" failed')

#convert variation ID to IDs of combinatorial sub-IDs
#var is in range [0,nTotalVariations-1], otherwise it repeats
#singleVariations contains the total number of variations, e.g. [2,5,4] => 2 x 5 x 4 = 40 nTotalVariations
#returns list of sub-IDs
#if var == None, then it returns the number of total variations
def VariationID2listIDs(var, singleVariations=[]):
    n = len(singleVariations) #number of sub-IDs
    subIDs = [None] * n
    
    #cnt total variations:
    nTotalVariations = 1
    for i in range(n):
        nTotalVariations *= singleVariations[i]
        
    if var is None:
        return nTotalVariations
    
    var = var % nTotalVariations #limit var; we could add warning
    
    x = var
    for i in range(n):
        subIDs[i] = x % singleVariations[i]
        x = int(x / singleVariations[i])

    return subIDs
    
# make systematic text variations; variationID=0,1,...
#if variationID == None, then it returns the number of total variations
def MakeTextVariation(text, variationID=None):

    m = len(text.split(' '))-1
    n = len(text.split('. '))-1
    
    nTotalVariations = (m + 1) * (n + 1) # possible variations

    if variationID is None:
        return nTotalVariations

    if variationID >= nTotalVariations:
        raise ValueError(f"variationID={variationID} exceeds maximum allowed={nTotalVariations - 1}")
    
    [variationIDm, variationIDn] = VariationID2listIDs(variationID, singleVariations=[m+1,n+1])

    #variate spaces:
    textlist=text.split(' ')
    m = len(textlist)-1
    newText = textlist[0]
    for i, item in enumerate(textlist[1:]):
        sep = '  ' if i == variationIDm-1 else ' '
        newText += sep+item
    text = newText

    #variate '. '
    textlist=text.split('. ')
    n = len(textlist)-1
    newText = textlist[0]
    for i, item in enumerate(textlist[1:]):
        sep = '.\n' if i == variationIDn-1 else '. '
        newText += sep+item
    text = newText

    return text

#special encoder to avoid problems with numpy and Exudyn types:
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        # these lists can be converted to lists of numpy arrays:
        elif isinstance(obj, (exu.Vector2DList, exu.Vector3DList, exu.Vector6DList, 
                          exu.Matrix3DList,exu.Matrix6DList)):
            newObj = "<json_encoder_failed>" #to account for that!
            try:
                newObj = obj.GetPythonObject()
            except:
                print('json.NumpyEncoder: failed for {obj}')
            return newObj
        
        return super().default(obj)

    
def WriteDictToJSON(dictToWrite, jsonOutputFile):
        try:
            os.makedirs(os.path.dirname(jsonOutputFile), exist_ok=True)
        except KeyboardInterrupt:
            print("Keyboard interrupt requested...exiting")
            raise
        except:
            pass #makedirs may fail on some systems, but we keep going

        with open(jsonOutputFile, "w", encoding="utf-8") as json_file:
            json.dump(dictToWrite, json_file, indent=4, ensure_ascii=False, cls=NumpyEncoder)


def ReadJSONToDict(jsonInputFile):
    """
    Reads a JSON file and returns its contents as a dictionary.
    
    :param jsonInputFile: Path to the JSON file to read.
    :return: Dictionary with JSON contents or an empty dictionary if an error occurs.
    """
    if not os.path.exists(jsonInputFile):  # Check if file exists
        print(f"Error: File '{jsonInputFile}' not found.")
        return {}

    try:
        with open(jsonInputFile, "r", encoding="utf-8") as json_file:
            return json.load(json_file)  # Load JSON into a dictionary
    except json.JSONDecodeError:
        print(f"Error: File '{jsonInputFile}' is not a valid JSON file.")
        return {}
    except Exception as e:
        print(f"Error reading '{jsonInputFile}': {e}")
        return {}   
     
# Extract elements from a LLM response based on the keys of the parsed dictionary (exudyn-/eval-elements) (Case-insensitive!)
def ExtractElementsInResponse(llmResponse, parsedDict):
    
    finalList = []
    message = ''
    tagDict = ExtractXMLtaggedString(llmResponse, 'elements')

    text = tagDict['text']
    if text == None:
        finalList = None
        message = tagDict['message']
    else:
        # Build a case-insensitive pattern
        pattern = r'\b(' + '|'.join(map(re.escape, parsedDict.keys())) + r')\b'
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
    
        # Normalize matches back to dictionary key casing (all letters written "small")
        lowerKeyMap = {key.lower(): key for key in parsedDict.keys()} #map of lower() keys to correct keys
        normalizedMatches = [lowerKeyMap[m.lower()] for m in matches if m.lower() in lowerKeyMap]
    
        for item in normalizedMatches:
            if item not in finalList:
                finalList.append(item)

    return {'list':finalList, 'message':message}

#check: replace substrings contained in undesiredKeywords from text
#       if text contains harmfulCommands, returns None (indicates that check failed)
#       success: return revised text; failure: returns None
def CheckLLMoutput(logger, text, checkUndesiredKeywords=False, checkHarmfulCommands=False): # inner function checking the quality of the LLM-output
    
    if isinstance(text, list):
        text = text[0] #take first element
    
    undesiredKeywords = [
        'The final answer is:', 'The answer is:', 'Answer:',
        '# noqa: E501', 
        '$\\boxed{0', #this is the only difference: not used in GenerateModelInputsFromConjecture
        '$\\boxed{', '}$',
        '"""', '```'
        # 'python', #this is dangerous, as the word python may be used as variable, etc.!
        # '"""', '```' 
        #==> python should be removed only if it before """ and this could be a separate function to 
        #    extract python code
    ]

    harmfulCommands = [
        "import os", "import subprocess", "import sys",
        "eval(", "open(", "subprocess.", "os.system"
    ]

    if not checkUndesiredKeywords: 
        undesiredKeywords = []
    
    if not checkHarmfulCommands: 
        harmfulCommands = []

    # Replace undesired phrases with an empty string
    for keyword in undesiredKeywords:
        if keyword in text: #not needed => just replace
            logger.LogText(f"Warning: Removing undesired keyword: '{keyword}'")
            text = text.replace(keyword, '')

    # detect harmful commands, which shall not be executed (remove all ...)
    for keyword in harmfulCommands:
        if keyword in text:
            logger.LogText(f"ERROR: Generated code contains potentially harmful command: '{keyword}'; aborting")
            return None  
    
    if "The prompt size exceeds the context window size and cannot be processed." in text:
        logger.LogText(f"ERROR: LLM produced no response, because context window size exceeded:\n{text}\n")
        return None  

    text = text.removeprefix("\n") # delete leading linebreak
    text = textwrap.dedent(text) # correct indentation

    return text

#extract code from LLM output; 
#modify code to improve safety, adjust Exudyn settings, avoid infinite waiting, etc.
def PreProcessLLMcode(logger, code, addTimeout = True, 
                      addFileWrite = True, 
                      replaceCoordinatesSolutionFile = None,
                      solverTimeoutSeconds = 120):
    if code is None:
        return None
    
    #Exudyn/Python only accepts /
    if replaceCoordinatesSolutionFile is not None:
        replaceCoordinatesSolutionFile = replaceCoordinatesSolutionFile.replace('\\','/')

    textBeforeCode = code.split('import exudyn')[0]
    isReasoningModel = False
    if 'wait, ' in textBeforeCode.lower(): isReasoningModel = True
    if 'putting all together'.lower() in textBeforeCode.lower(): isReasoningModel = True
    if 'putting it all together'.lower() in textBeforeCode.lower(): isReasoningModel = True


    codeLines = code.strip().split('\n')
    
    debug = False
    newCode = ''
    foundPython = False
    foundSimulationSettings = False
    filledCoordinatesSolution = False
    importExudynFound = False
    
    for line in codeLines:
        line0 = line.strip().lower() #python usually written Python
        if line0.startswith('Here is the '.lower()): #any version of this shall be erased
            if debug: print('** remove1:'+line0)
            continue

        if line0.startswith('import exudyn as '): 
            importExudynFound = True

        if line0.startswith('<|im_start|>'): #this is typical for reasoning models
            continue
        if 'SolutionViewer()' in line0 and not line0[0].strip().startswith('#'): #avoid during execution
            line0 = '#manually removed by PreProcessLLMcode:'+line0 
            logger.LogWarning('PreProcessLLMcode: SolutionViewer() found and removed')
            
        
        if line0 == 'python':
            if debug: print('** remove2')
            if not importExudynFound: #Qwen2.5 adds python also at the end!
                newCode = ''
            continue
        if line0 == '```python':
            if debug: print('** remove3')
            foundPython = True
            newCode = ''
            continue
        if (line0 == '```'):
            if not foundPython:
                if debug: print('** remove3')
                newCode = ''
                foundPython = True
                continue
            else:
                break #no more text read (would fail for several parts!)
        if line0.startswith('#end-of-code'):
            break
        if replaceCoordinatesSolutionFile is not None and filledCoordinatesSolution:
            if 'simulationSettings.solutionSettings.coordinatesSolutionFileName' in line:
                continue #do not add setting, otherwise it will overwrite the desired filename
        
        newCode += line+'\n'

        if isReasoningModel and line0.startswith('import exudyn as'): #this is the last way to erase all earlier text
            newCode = line+'\n'
            continue

        if 'import exudyn as exu' in line:
            if addTimeout:
                newCode += f'exu.special.solver.timeout = {solverTimeoutSeconds}\n' #2 minutes simulation timeout
            if addFileWrite:
                newCode += 'exu.config.printToConsole = False  #no output to console\n'
                newCode += 'exu.SetWriteToFile(filename="'+logger.logDir+'exudynTemp.log", flagFlushAlways=True)\n'

        if replaceCoordinatesSolutionFile is not None:
            if not foundSimulationSettings and 'exu.SimulationSettings()' in line:
                foundSimulationSettings = True
            if foundSimulationSettings and not filledCoordinatesSolution:
                newCode += 'simulationSettings.solutionSettings.coordinatesSolutionFileName = "'+replaceCoordinatesSolutionFile+'"\n'
                newCode += 'simulationSettings.timeIntegration.verboseMode = 1\n' #to see output in log
                filledCoordinatesSolution = True

    #remove such constructs, as they totally kill the Python kernel (any further agent actions ...):
                
    stopWriting = 'exu.SetWriteToFile("",False)\n' #this only works, if no exception was raise
    if addFileWrite: #this is for the case that no exception is raised!
        newCode += stopWriting

    if len(newCode)>=1 and newCode[-1] == '\n':
        newCode = newCode[:-1]
    
    return newCode


# function extracts information from text inside a html-tagged information and returns the information; 
# if tag is not found, return None
def ExtractXMLtaggedString(text, tag):
    response = {'text':None,'message':''} #message contains errors, etc.
    startTag = '<'+tag+'>'
    endTag = '</'+tag+'>'
    nStartTag = text.count(startTag)
    nEndTag = text.count(endTag)
    if nStartTag == 0 or nEndTag == 0:
        response['message'] = f'ERROR in ExtractXMLtaggedString: received {nStartTag} start tags and {nEndTag} end tags!'
        return response
    elif text.find(startTag) > text.find(endTag):
        response['message'] = f'ERROR in ExtractXMLtaggedString: {startTag} placed after {endTag}!'
        return response

    # Use the last complete block: vision/chat models often echo the prompt, which may
    # mention "<elements>" without a closing tag or include placeholder pairs.
    if nStartTag != 1 or nEndTag != 1:
        response['message'] = (
            f'Warning in ExtractXMLtaggedString: received {nStartTag} start tags and '
            f'{nEndTag} end tags; using last {startTag} block.'
        )
    response['text'] = text.rsplit(startTag, 1)[1].split(endTag, 1)[0]

    return response


# checks if a file exists in a directory (filepath)
def FileExists(filePath, fileName):
    fullPath = JoinPath(filePath, fileName)  # Construct full file path
    return os.path.exists(fullPath)  # Return True if the file exists, False otherwise

# delete a file if it exists in a specific path
def DeleteFileIfExisting(filePath, fileName):
    fullPath = JoinPath(filePath, fileName)  # Construct full file path
    if os.path.exists(fullPath):  # Check if the file exists
        try:
            os.remove(fullPath)   # Delete the file
        except Exception as e:
            print(f"Error deleting file '{fileName}': {e}")

# convert a string into a nice multi-line Python comment, using a limited number of characters per line
def String2PythonComment(text):
    words = text.split(' ')
    cols = 0
    newText = '#'
    for word in words:
        word = word.strip('\n')
        word = word.replace('\n', ' ')
        if cols > 70:
            newText += '\n# '+ word
            cols=0
        else:
            newText += ' ' + word
            cols += len(word)+1
    newText+='\n'
    return newText
            

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
#convert 'this is it' into thisIsIt
def SpacedText2CamelCase(s):
    parts = re.split(r'[\s_\-]+', s.strip())
    return parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])

#replace camel case strings to strings with spaces
def AddSpacesToCamelCase(text):
    return re.sub(r'(?<=[a-z])([A-Z])', r' \1',text).lower()

def SplitCamelCaseIntoStringsList(text):
    pattern = r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])'
    return re.split(pattern, text)

#read out Exudyn solution file header and store in dict
def ParseExudynSolutionFileHeaderToDict(file_path, maxLinesHeader = 10):
    header_dict = {}

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.startswith('#'):
                continue
            if i >= maxLinesHeader:
                break

            line = line[1:].strip()  # Remove '#' and whitespace
            if '=' in line:
                key, value = line.split('=', 1)
            elif ':' in line:
                key, value = line.split(':', 1)
            else:
                continue  # skip if not key-value style

            key = key.split('[')[0].split('(')[0]
            key = SpacedText2CamelCase(key.strip())
            value = value.strip()
            if '[' in value:
                try:
                    value = ast.literal_eval(value) #convert into list
                except KeyboardInterrupt:
                    print("Keyboard interrupt requested...exiting")
                    raise
                except:
                    pass
            
            header_dict[key] = value

    return header_dict


# reads a value from file and adds deltaCounter
def LogOverallCounter2File(filePath, deltaCounter):

    # Create file if it doesn't exist
    if not os.path.exists(filePath):
        with open(filePath, 'w') as f:
            f.write(f"{deltaCounter}\n")

    else:
        with open(filePath, "r+") as f:

            # Read current counter
            content = f.read().strip()
            current_value = int(content) if content else 0

            # Create backup
            shutil.copy(filePath, filePath + ".bck")

            # Update the counter
            new_value = current_value + deltaCounter


            # Write new value
            f.seek(0)
            f.write(str(new_value))
            f.truncate()

    
# takes a reference-solution file and calculates the numerical difference between values to a second solution file
# returns difference, or -2 if file was not found or -1 if shape of data was different
def EvaluateNumerical(fileNameReference, fileNameEvaluate,  localVariablesReference=None,
                      localVariablesEvaluate=None, logger=None):
    import exudyn as exu
    if not FileExists(filePath='', fileName=fileNameReference):
        logger.LogError('EvaluateNumerical: reference file '+fileNameReference+' not found')
        return -2 #cannot be evaluated
    if not FileExists(filePath='', fileName=fileNameEvaluate):
        logger.LogError('EvaluateNumerical: eval file '+fileNameEvaluate+' not found')
        return -2 #cannot be evaluated

    refConfig0 = None
    refConfig1 = None
    if localVariablesReference is not None and localVariablesEvaluate is not None:
        if 'mbs' in localVariablesReference:
            mbsRef = localVariablesReference['mbs']
            if 'mbs' in localVariablesEvaluate:
                mbsEval = localVariablesEvaluate['mbs']
                refConfig0 = mbsRef.systemData.GetODE2Coordinates(configuration=exu.ConfigurationType.Reference)
                refConfig1 = mbsEval.systemData.GetODE2Coordinates(configuration=exu.ConfigurationType.Reference)
            else:
                logger.LogDebug('EvaluateNumerical: mbs not found in LLM solution')
        else:
            logger.LogDebug('EvaluateNumerical: mbs not found in sample solution')
            
    solution0 = np.loadtxt(fileNameReference, comments='#', delimiter=',')
    solution1 = np.loadtxt(fileNameEvaluate, comments='#', delimiter=',')

    diff = 0
    successODE2 = False
    #only compare ODE2 coordinates; constraints could be interchanged, still correct!
    try:    
        header0 = ParseExudynSolutionFileHeaderToDict(fileNameReference)
        header1 = ParseExudynSolutionFileHeaderToDict(fileNameEvaluate)
        
    
        nODE2_0 = header0['numberOfSystemCoordinates'][0]
        nODE2_1 = header1['numberOfSystemCoordinates'][0]
        nCompare0 = 0
        nCompare1 = 0
        for i in range(3):
            nCompare0 += header0['numberOfWrittenCoordinates'][i]
            nCompare1 += header1['numberOfWrittenCoordinates'][i]
            
        nTimeSteps0 = solution0.shape[0]
        nTimeSteps1 = solution1.shape[0]

        
        #only compare ODE2 coordinates!
        if (nCompare0 == nCompare1 
            and nODE2_0 == nODE2_1 
            and nTimeSteps0 == nTimeSteps1
            and solution0.ndim == 2 
            and solution1.ndim == 2):
            
            diff = np.linalg.norm(solution0[:,:nCompare0] - solution1[:,:nCompare1])
            
            #try to add reference config
            if refConfig0 is not None and refConfig1 is not None and diff > 1e-10:
                solution0[:,:nODE2_0] += refConfig0
                solution1[:,:nODE2_1] += refConfig1
                diff1 = np.linalg.norm(solution0[:,:nCompare0] - solution1[:,:nCompare1])
                diff = min(diff, diff1)
                if diff <= 1e-10:
                    logger.LogWarning('EvaluateNumerical: solutions general difference > 0 but with reference values difference is ok')
            
            successODE2 = True
        else:
            logger.LogError("EvaluateNumerical: Solutions have different shape; "+
                            f"nODE2 values = ({nODE2_0},    {nODE2_0})"+
                            f"time steps   = ({nTimeSteps0},{nTimeSteps1})",
                            printToConsole=False)
            
    except KeyboardInterrupt:
        print("Keyboard interrupt requested...exiting")
        raise
    except:
        exceptionText = traceback.format_exc()
        logger.LogError(f'EvaluateNumerical failed: {exceptionText}')

    
    if successODE2:
        logger.LogText(f'EvaluateNumerical: Solutions difference = {diff} [ODE2-based]')
        if solution0.shape != solution1.shape:
            logger.LogWarning('EvaluateNumerical: solutions have different shape; shapes = ('+
                            str(solution0.shape)+','+str(solution1.shape)+') - but we only compare ODE2 values!' )

        return diff


    #     return diff

    return -1

def PrintHeadlineToConsole(length=80, text=None):
        if not text:
            print('+' * length)
            return
        middle = f' {text} '
        if len(middle) >= length:
            print(middle)
            return
        remaining = length - len(middle)
        left = remaining // 2
        right = remaining - left
        print('+' * left + middle + '+' * right)

def PartString2Name(identifier: str) -> str:
    """
    Convert modelDescription's internal part identifiers like '{bearingBlockPart}', 'unbalancedRotorPart',
    or 'flexible_beam-Part' into a human-readable name like 'bearing block', 'unbalanced rotor',
    or 'flexible beam'.

    Steps:
    - Strip surrounding braces { } if present.
    - Remove trailing 'Part' (case-insensitive).
    - Normalize underscores/dashes to spaces.
    - Split camelCase/PascalCase using AddSpacesToCamelCase.
    - Lowercase and collapse whitespace.
    """
    s = identifier.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    s = re.sub(r'Part$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'[_\-]+', ' ', s)
    s = AddSpacesToCamelCase(s)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def ExpandFemHeaders(description: str, resolvedParameters: dict,
                     numberOfPartsAsString: str) -> str:
    """
    Replace all header placeholders in description on different levels (per part / per fMBS description).
    If a referenced header placeholder is missing, leave the placeholder unchanged.
    Supported:
      - {HEADER:INTRO}
      - {HEADER:partX}              (e.g. part1, part2, ...)
      - {HEADER:<namedHeader>}      (e.g. meshPropertiesHeader)
    """
    # named headers
    FEM_HEADER_TEMPLATES = {
        "meshPropertiesHeader": meshPropertiesHeader,
        # future headers go here
        }
    
    
    # replace intro header in fmbs description
    description = description.replace("{HEADER:INTRO}",femHeaderIntro.format(numberOfPartsAsString=numberOfPartsAsString))
   
    def _sub(match: re.Match) -> str:
        key = match.group(1)

        # part headers on sub level of description (part-level) 
        if re.fullmatch(r"part[0-9]+", key):
            part_identifier = resolvedParameters.get(key)
            if not part_identifier:
                return match.group(0)  # leave unchanged
            part_name = PartString2Name(part_identifier)
            return femInterfaceDescriptionHeader.format(partName=part_name)

        # named headers
        if key in FEM_HEADER_TEMPLATES:
            return FEM_HEADER_TEMPLATES[key]

        # fallback: leave unknown headers unchanged
        return match.group(0)


    # Replace ALL {HEADER:...} occurrences in one pass
    return re.sub(r"\{HEADER:([A-Za-z0-9_]+)\}", _sub, description)


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
