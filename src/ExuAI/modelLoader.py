# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: modelLoader.py
#
# Details:  Read textual Exudyn model descriptions, define standard (convenient) random numbers and add randomized parametrization for models
#
# Author:   Tobias Möltner, Johannes Gerstmayr
# Date:     2025-02-10
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
try: 
    from .utilities import MakeTextVariation, VariationID2listIDs, IsScalar, IsIntType, \
        GetStandardRandomNumber, ExpandFemHeaders
    from .templatesAndLists import num2StringDict
except: 
    from utilities import MakeTextVariation, VariationID2listIDs, IsScalar, IsIntType, \
        GetStandardRandomNumber, ExpandFemHeaders
    from templatesAndLists import num2StringDict


class ModelLoader:
    def __init__(self):
        pass

    # load models from modelDescriptions.py, only models <= difficultyLevel,
    # specificModels can be a list of model names that are used
    def LoadStandardModelDescriptions(self, modelDescriptionsDict,
                                      maxDifficultyLevel = 10000, 
                                      specificModels = None,
                                      onlyWithSampleFiles = False,
                                      onlyParameterizedModels = False):
        mbsModels = modelDescriptionsDict
        self.modelDescriptions = {}
        self.maxDifficultyLevel = maxDifficultyLevel

        if specificModels is None:
            specificModels = []

        for key, value in mbsModels.items():
            if ( ('difficulty' not in value or value['difficulty'] <= maxDifficultyLevel)
                and (specificModels == [] or key in specificModels)
                and (not onlyParameterizedModels or 'parameters' in value) 
                and (not onlyWithSampleFiles or 'sampleFileName' in value) 
                ):
                self.modelDescriptions[key] = value
        
    
    def NumberOfModels(self):
        return len(self.modelDescriptions)
    
    def ListOfModels(self):
        listModels = []
        for key, value in self.modelDescriptions.items():
            listModels.append(key)
        return listModels
    
    #Generates a model description with parameters (random if desired) within defined ranges.
    #spaceVariationID: do text variation by changing spaces and line breaks
    #differentFromTheseParameters: will try to always find parameters that are all different from given ones
    def GetModelDescriptionAndDict(self, modelKey, randomizeParameters = False,
                                   spaceVariationID=None, wordingVariationID=None,
                                   differentFromTheseParameters=None, logger=None,
                                   masterParameters={}):
        
        if modelKey not in self.modelDescriptions:
            raise ValueError(f"Model '{modelKey}' not found.")

        try:
            modelData = self.modelDescriptions[modelKey]
    
            chosenParameters = {}
            numberOfPreliminaryTasks = 0
            for param, values in modelData["parameters"].items():
                if param.startswith('part'):
                    numberOfPreliminaryTasks += 1
                setViaMasterParameters = False
                if masterParameters != {}:
                    if param in masterParameters:
                        newParam = masterParameters[param]
                        setViaMasterParameters = True
                if not setViaMasterParameters:
                    newParam = None
                    itMax = 30
                    it = 0
                    foundParam = False
                    while it < itMax and not foundParam:
                        it += 1
                        if randomizeParameters:
                            if ("range" in values):
                                newParam = GetStandardRandomNumber(values["range"][0], values["range"][1])
                            elif("list" in values):
                                newParam = np.random.choice(values["list"], size=None, replace=True) # get random element of list
                                if IsIntType(newParam):
                                    newParam = int(newParam)
                                elif IsScalar(newParam):
                                    newParam = float(newParam)
                            else:
                                newParam = values["default"] 
                                raise ValueError("GetModelDescriptionAndDict") # information about what failed 
        
                            if differentFromTheseParameters is not None:
                                if str(differentFromTheseParameters[param]) != str(newParam):
                                    foundParam = True
                            else:
                                foundParam = True
                        else:
                            newParam = values["default"]
                            foundParam = True
        
                    if it >= itMax:
                        warningText = 'Warning: GetModelDescriptionAndDict: alternative parameters failed for param='+param+'!'
                        if logger is not None:
                            logger.LogWarning(warningText, printToConsole=True)
                        else:
                            print(warningText)
                # now set the parameter as chosen parameter
                chosenParameters[param] = newParam
    
            # for reoccurring headers for FEM parts in fMBS, expand them here, if parts do not exist, leave placeholder unchanged
            modelData["description"] = ExpandFemHeaders(modelData["description"], chosenParameters,
                                                num2StringDict[str(numberOfPreliminaryTasks)])
            
            # check whether parameter is a master parameter or if it is in the description of the model
            for param in chosenParameters:
                if '{'+param+'}' not in modelData["description"]:
                    parameterIsAMasterParameter = False
                    if 'metaInformation' in modelData:
                        metaInformation = modelData['metaInformation']
                        if any(param in part["masterParameters"] for part in metaInformation.values()):
                            parameterIsAMasterParameter = True
                    if not parameterIsAMasterParameter:
                        raise ValueError('GetModelDescriptionAndDict: parameter "'+param+'" not in model description!')
                    
            # Format description with randomized values
            modelDescription = modelData["description"].format(**chosenParameters)    
            
            if wordingVariationID is not None:
                if "alternativeDescription" not in modelData:
                    raise ValueError("GetModelDescriptionAndDict: no alternativeDescription available")
                    
                alternativeParameters = modelData["alternativeDescriptionParameters"]
                altLists = []
                for key, value in alternativeParameters.items():
                    altLists.append(len(value))
                paramIDs = VariationID2listIDs(wordingVariationID, altLists)
                
                chosenAlternativeParameters = {}
                cnt = 0
                for key, value in alternativeParameters.items():
                    chosenAlternativeParameters[key] = value[paramIDs[cnt]]
                    cnt += 1
                
                modelDescription = modelData["alternativeDescription"].format(**chosenParameters,**chosenAlternativeParameters)
                
            if spaceVariationID is not None:
                modelDescription = MakeTextVariation(modelDescription, spaceVariationID)
            returnDict = dict(modelDescription=modelDescription,
                            modelData=modelData,
                            chosenParameters=chosenParameters)
            return returnDict
        except Exception as e:
            print(f"ModelLoader.GetModelDescriptionAndDict: problem with model '{modelKey}':\n  error={e}")
            raise
