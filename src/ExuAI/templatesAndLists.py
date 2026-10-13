# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: templatesAndLists.py
#
# Details:  main prompt templates (template to create the prompt for LLMs, using their prompt templates ...),
#           further lists, and dictionaries used by agent
#
# Authors:  Johannes Gerstmayr, Tobias Möltner
# Date:     2025-02-10
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# simpleMBSChooseItemsPromptTemplate is the default prompt (promptVariation=0) = best choice
# adapted 2026-04-18 to promptVariation6; NOTE: originally had no \ at beginning: '''
simpleMBSChooseItemsPromptTemplate = '''\
Role: You are a Multibody Dynamics (MBD) simulation expert specialized in Exudyn.

Context: You have a list of available elements:
{items}

Task: Analyze the following model description and select all elements to represent its mechanical behavior:
{modelDescription}

Specific Rules:
 * Gravity: Do NOT select force elements for gravity; it is handled internally by point masses and rigid bodies.
 * Inclusion: If an element's necessity is ambiguous, include it to ensure completeness.
 * Exclusion: Do not list the same element twice.
 * Output Format: Provide only the comma-separated list of names within <elements> tags. No prose or explanation.
'''

# alternative prompt templates for choose items task (selected via promptVariation argument)
# promptVariation=0 uses the default simpleMBSChooseItemsPromptTemplate above
# promptVariation=1..7 uses the corresponding entry from this list (index 0..6)
simpleMBSChooseItemsPromptVariations = [
# promptVariation = 1
'''\
The following elements are available in the simulation code Exudyn:
{items}
Note that gravity is already included in point masses and rigid bodies and must not be added as a separate element.

Given is a multibody model with the following description:
{modelDescription}

Identify all elements required to represent the mechanical structure and behavior of this model. Focus on mapping each described physical component or interaction to exactly one suitable element from the list.

Do not repeat elements and do not include unnecessary ones.
Provide the result as a comma-separated list enclosed in <elements> ... </elements>, using exactly the provided element names and no additional text.
''',
# promptVariation = 2
'''\
The following elements are available in the simulation code Exudyn:
{items}
Gravity effects are inherently included in point masses and rigid bodies and therefore do not require a separate force element.

Given is a multibody model with the following description:
{modelDescription}

Select the minimal and most appropriate set of elements needed to model the described system. Carefully interpret the physical meaning (bodies, joints, constraints, forces) and match them to the available elements.

Each element may appear only once.
Return a comma-separated list inside <elements> ... </elements>, strictly using the given element names and no explanations.
''',
# promptVariation = 3
'''\
The following elements are available in the simulation code Exudyn:
{items}
Note that gravity is implicitly handled by point masses and rigid bodies.

Given is a multibody model with the following description:
{modelDescription}

Determine which elements are necessary to construct this system. Consider bodies, connections, and interactions described, and select only those elements that are essential to represent them correctly.

Avoid duplicates and avoid adding elements not directly implied by the description.
Output the elements as a comma-separated list enclosed in <elements> ... </elements>, using only the exact names from the list and no additional text.
''',
# promptVariation = 4
'''\
The following elements are available in the simulation code Exudyn:
{items}
Gravity is already included in mass elements and should not be modeled separately.

Given is a multibody model with the following description:
{modelDescription}

From the description, infer the required modeling components (such as bodies, joints, loads, or constraints) and map them to the corresponding elements from the list.

Only include elements that are clearly required, do not repeat any element, and do not introduce new names.
Return the result as a comma-separated list inside <elements> ... </elements> without any explanations.
''',
# promptVariation = 5; generate more items
'''\
The following elements are available in the simulation code Exudyn:
{items}
Note that gravity is part of point masses and rigid bodies, so it does not require a force element.

Given is a multibody model with the following description:
{modelDescription}

Please choose the elements that could be required to build the given multibody model and which represent its mechanical behavior.
In uncertain cases, you prefer to add elements, however, do not repeat any element.
Provide elements in a comma-separated list which is enclosed in the tags <elements> ... </elements>, using exactly the provided element names and don't add other text or explanations.
''',
# promptVariation = 6; generate more items (=default)
'''\
Role: You are a Multibody Dynamics (MBD) simulation expert specialized in Exudyn.

Context: You have a list of available elements:
{items}

Task: Analyze the following model description and select all elements to represent its mechanical behavior:
{modelDescription}

Specific Rules:
 * Gravity: Do NOT select force elements for gravity; it is handled internally by point masses and rigid bodies.
 * Inclusion: If an element's necessity is ambiguous, include it to ensure completeness.
 * Exclusion: Do not list the same element twice.
 * Output Format: Provide only the comma-separated list of names within <elements> tags. No prose or explanation.
''',
# promptVariation = 7; default until 2026-04-17
'''\
The following elements are available in the simulation code Exudyn:
{items}
Note that gravity is part of point masses and rigid bodies, so it does not require a force element.

Given is a multibody model with the following description:
{modelDescription}

Please choose the elements that are required to build the given multibody model and which represent its mechanical behavior best, without repeating any element.
Provide elements in a comma-separated list which is enclosed in the tags <elements> ... </elements>, using exactly the provided element names and don't add other text or explanations.
''',
]

#OLD: until-2026-04-14
# simpleMBSGenerateCodePromptTemplate = '''The following Python example shows how to generate a multibody model in Exudyn, including most relevant features: 
# ```python
# {generalStructure}
# ```
# Using this example, please build a Python model following exactly the following description of a multibody model:
# {modelDescription}

# Follow exactly the function names and argument names as given in the example and create the Exudyn model.
# Provide only the Python code without any additional descriptions; you don't need to add comments, except for special cases.
# It is very important that you start the code with required imports and that you mark the end of code with the line "#end-of-code".'''

#Note that mbs.Create...(...) functions call several functions in the background to create nodes, objects, markers and loads in the system 'mbs'.
# simpleMBSGenerateCodePromptTemplate is the default prompt (promptVariation=0) = best choice
simpleMBSGenerateCodePromptTemplate = '''The following Python example shows how to generate a multibody model in Exudyn, to update your internal data, including most relevant features.
Vectors such as initial, reference or local positions, velocities and gravity are given as 3D lists [x,y,z] or according numpy arrays.
While these vectors are usually given in the global coordinate system, localPositions (0 or 1) have to be defined in body-fixed coordinates of the respective body (relative to the body's reference, usually the COM).
Comments in this example give additional info on the usage of each function. The code reads:
```python
{generalStructure}
```
Using this example, please build a Python model following exactly the following description of a multibody model:
{modelDescription}

Follow exactly the function names and argument names as given in the example and create the Exudyn model.
Do no approximate values such as for sin or sqrt, but always use according numpy or math functions.
Provide only the Python code without any additional descriptions; you don't need to add comments, except for special cases.
It is very important that you start the code with required imports and that you mark the end of code with the line "#end-of-code".'''

#Do not approximate float values (like square root, sin or division) for positions, rotations, etc.; only use exact values or compute them in Python; use math or numpy if needed.

# alternative prompt templates for generate code task (selected via promptVariation argument)
# promptVariation=0 uses the default simpleMBSGenerateCodePromptTemplate above
# promptVariation=1..4 uses the corresponding entry from this list (index 0..3)
simpleMBSGenerateCodePromptVariations = [
# promptVariation = 1
'''The following Python example demonstrates how to construct a multibody model in Exudyn and defines the required structure and conventions.
Vectors (positions, velocities, gravity) are given as 3D lists [x,y,z] or numpy arrays.
Local positions must always be defined in the body-fixed coordinate system.

The example code is:
{generalStructure}

Using this structure, generate Python code for the following multibody model:
{modelDescription}

Strictly follow the function names, argument names, and structural patterns from the example. Ensure that each part of the model description is consistently translated into code.

Prioritize mathematical precision by using numpy or math library functions (e.g., np.pi, math.sqrt) rather than hardcoded decimal approximations.
Return only valid Python code, starting with imports and ending with "#end-of-code", without additional explanations.
''',
# promptVariation = 2
'''The following Python example provides the reference structure for building multibody models in Exudyn.
All vectors are 3D and must follow the conventions shown. Local coordinates must be defined relative to the body.

{generalStructure}

Based on this reference, implement the multibody model described below:
{modelDescription}

Translate the description step-by-step into code, ensuring that all physical components (bodies, joints, forces) are correctly represented.

Use exactly the same function and argument names as in the example. Avoid adding unsupported constructs.
Prioritize mathematical precision by using numpy or math library functions (e.g., np.pi, math.sqrt) rather than hardcoded decimal approximations.
Do not include explanations or comments (except if strictly necessary). Begin with imports and terminate with "#end-of-code".
''',
# promptVariation = 3
'''The following Python example illustrates the correct way to define an Exudyn multibody system, including syntax and structure:
{generalStructure}

Using this as a template, generate code for the following model:
{modelDescription}

Ensure that the generated code adheres strictly to the demonstrated structure and API usage. All vectors must be 3D, and local positions must be defined in body-fixed coordinates.

Map each described system component clearly into code without omitting or inventing elements.
Prioritize mathematical precision by using numpy or math library functions (e.g., np.pi, math.sqrt) rather than hardcoded decimal approximations.

Output only Python code, starting with imports and ending with "#end-of-code".
''',
# promptVariation = 4
'''The following example defines the standard structure and conventions for creating multibody models in Exudyn:
{generalStructure}

Use this structure to implement the model described below:
{modelDescription}

Follow the example closely in terms of syntax, ordering, and function usage. Ensure consistency between the model description and the generated code.

All vectors must be 3D, and local coordinates must be expressed in the respective body frames.
Prioritize mathematical precision by using numpy or math library functions (e.g., np.pi, math.sqrt) rather than hardcoded decimal approximations.
Avoid redundant or undefined elements and do not introduce additional abstractions.

Provide only the Python code (no explanations), include required imports at the beginning, and finish with "#end-of-code".
'''
]

#%% FlexuAI
partsChooseItemsPromptTemplate = '''The following elements are available to create a machine part:
{items}

Given is a machine part with the following description:
{modelDescription}

Identify all elements required to virtually build this machine part. Focus on mapping each described property to exactly one suitable element from the list.
Do not repeat elements and do not include unnecessary ones.
Provide the result as a comma-separated list enclosed in <elements> ... </elements>, using exactly the provided element names and no additional text.'''

partsGenerateCodePromptTemplate = '''The following Python example shows how to generate a machine part:
```python
{generalStructure}
```
Using this example, please write a Python code following exactly the following description of a machine part:
{modelDescription}

Follow exactly the function names and argument names as given in the example and create the machine part.
Provide only the Python code without any additional descriptions; you don't need to add comments, except for special cases.
It is very important that you start the code with required imports and that you mark the end of code with the line "#end-of-code".'''

# special headers that are copied outmatically by our expandFemHeaders functionality
femHeaderIntro = (' You already have the {numberOfPartsAsString} needed machine part(s)'
                  ' available as a Finite Element Method (FEM) model. Load the according FEM interface(s) with their/its according name(s)!\n')
# shared FEM interface description header (per part in fMBS description)
femInterfaceDescriptionHeader = 'The description behind the needed machine part "{partName}" is: '
# shared material and mesh description (per part in geometry descriptions)
meshPropertiesHeader = (
    ', having a Young\'s modulus of {EModul} N/m²,'
    ' a poisson ratio of {nu}, and a density of {rho} kg/m³.'
    ' Mesh the machine and give it the name "{partName}" such that it can be loaded.'
    ' Limit the mesh resolution by setting an appropriate maximum element size using the "maxh" parameter.')
num2StringDict = {'0':'zero',
                  '1':'one',
                  '2':'two',
                  '3':'three',
                  '4':'four',
                  '5':'five'}

#%% OLD Agent methods:

