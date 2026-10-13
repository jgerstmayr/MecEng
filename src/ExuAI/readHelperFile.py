# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: readHelperFile.py
#
# Details:  Read and parse exudynHelper.py and geometryMeshHelper.py and
#           provide information as dictionary with tags
#
# Author:   Tobias Möltner, Johannes Gerstmayr, Sebastian Weyrer (ed.)
# Date:     2026-01-07
# Note:     adapted from version from paper from 2025-01-05
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++
import copy

listSystemTags = [
                  'general import',
                  'assemble',
                  ]

# parse general helper and fill into parsedDict
# maxDifficultyLevel: max level up to which tags are included
# parsedDictInit: used to extend existing dict
def ParseHelper(filePath, maxDifficultyLevel=10000, excludeTags=[], 
                parsedDictInit={}):
    with open(filePath, 'r', encoding='utf8') as file:
        lines = file.readlines()
    parsedDict = copy.deepcopy(parsedDictInit)
    tagCounterStart = 0
    for key, value in parsedDict.items():
        if value['INDEX'] > tagCounterStart:
            tagCounterStart = value['INDEX']

    currentTag = None
    currentCode = []
    tagCounter = tagCounterStart
    skipCurrentTag = False  # Flag to determine if the tag should be skipped
    for linePure in lines:
        line = linePure.strip()

        if line.startswith("#TAG:"):
            # If a new tag is found, finalize the previous one (if not skipped)
            if currentTag and not skipCurrentTag:
                if currentTag in parsedDict:  # Ensure the tag exists before modifying
                    parsedDict[currentTag]['CODE'] = "\n".join(currentCode).strip()

            # Reset and start a new tag
            currentTag = line.split("#TAG:")[1].strip()
            tagCounter += 1

            # delete tag if it already exists
            if currentTag in parsedDict:
                del parsedDict[currentTag] #erase it to get a new position in dict
                
            parsedDict[currentTag] = {
                'INDEX': tagCounter,
                'DIFFICULTY': "",
                'REQUIRES': "",
                'NOTES': "",
                'INFO': "",
                'CODE': "",  # Initialize CODE to avoid KeyError
            }
            currentCode = []
            # skip initialization for current tag if excluded
            if currentTag in excludeTags:
                skipCurrentTag = True
                parsedDict.pop(currentTag, None) # remove tag if it should be included (the LLM must not see it)
            else:
                skipCurrentTag = False # Reset skip flag 
        else:
            if line.startswith("#DIFFICULTY:") and not skipCurrentTag:
                difficulty = line.split("#DIFFICULTY:")[1].strip()
                parsedDict[currentTag]['DIFFICULTY'] = difficulty
                # Check if difficulty exceeds maxDifficultyLevel and set skip flag
                try:
                    if int(difficulty) > maxDifficultyLevel:
                        skipCurrentTag = True
                        parsedDict.pop(currentTag, None)  # Remove the current tag
                except ValueError:
                    pass  # Ignore if difficulty is not a valid integer
            elif line.startswith("#REQUIRES:") and not skipCurrentTag:
                parsedDict[currentTag]['REQUIRES'] = line.split("#REQUIRES:")[1].strip()

            elif line.startswith("#INFO:") and not skipCurrentTag:
                parsedDict[currentTag]['INFO'] = line.split("#INFO:")[1].strip()

            elif line.startswith("#NOTES:") and not skipCurrentTag:
                parsedDict[currentTag]['NOTES'] = line.split("#NOTES:")[1].strip()

            elif line.startswith("#USES:") and not skipCurrentTag:
                parsedDict[currentTag]['USES'] = line.split("#USES:")[1].strip()

            elif line.startswith("#%%") and not skipCurrentTag:
                continue  # Skip the section separators
            else:
                if not skipCurrentTag:  # Only collect code if tag is not skipped
                    currentCode.append(linePure.replace('\n', ''))
    # Finalize the last tag if it was not skipped
    if currentTag and not skipCurrentTag:
        parsedDict[currentTag]['CODE'] = "\n".join(currentCode).strip()

    #we need to check if there are tags which cannot be found safely with ExtractElementsInResponse
    listTags = list(parsedDict)
    for i, tag in enumerate(listTags):
        for tag2 in listTags[i+1:]:
            if tag2.startswith(tag):
                raise ValueError('invalid sorting of tags: subsequent tags may not start with same string as previous tags')
                
    return parsedDict

# Get a list of all required tags for a given tag, including hierarchical dependencies.
def GetRequiredTags(parsedDict, tagName):
    requiredTags = set()
    stack = [tagName]

    while stack:
        current_tag = stack.pop()
        if current_tag not in parsedDict:
            continue

        requires = parsedDict[current_tag]['REQUIRES']
        if requires:
            dependencies = [req.strip() for req in requires.split(',')]
            for dep in dependencies:
                if dep not in requiredTags:
                    requiredTags.add(dep)
                    stack.append(dep)

    return list(requiredTags)

# Assemble code snippets based on a list of tags, sorted by their INDEX in parsedDict.
def CreateSampleCode(parsedDict, tagList):

    #remove duplicated items:
    tagList = list(dict.fromkeys(tagList))

    # Filter tags that exist in the dictionary
    valid_tags = [tag for tag in tagList if tag in parsedDict]
    
    # Sort tags by their INDEX
    sorted_tags = sorted(valid_tags, key=lambda tag: parsedDict[tag]['INDEX'])
    # Collect the code snippets in order
    assembledCode = []
    for tag in sorted_tags:
        # for seperation of short-tag info and comment section with longer description of the code snippet, 
        # delete parsedDict[tag]['INFO'] since we do not want to use that as description for the code snippet
        assembledCode.append(f"#{parsedDict[tag]['INFO']}\n{parsedDict[tag]['CODE']}")

    return "\n\n".join(assembledCode)
