#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Details:  Create a fem interface to evaluate the LLM assisted flexible mbs 
#           creation (i.e. the fem interface creation).
# Model:    Unbalanced rotor part
#
# Author:   Sebastian Weyrer
# Date:     2026-02-18
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%parametersstart%%
# parameters which can be varied
bearingHeight = 0.1
bearingRadius = 0.05
smallRotorHeight = 0.15
smallRotorRadius = 0.11
largeRotorHeight = 0.05
filletsRadiusFactor = 0.3
largeRotorRadius = 0.25
sphereRadius = 0.05
EModul = 2.1e11
nu = 0.3
rho = 7850
shaftInterfaceName = 'shaft'
partName = 'unbalancedRotorPart'

#%%samplecodestart%%
import exudyn as exu
import netgen.occ as occ
from helper.utilitiesForSampleFiles import GetFemObjectFromOCCGeometry, MakeFFRFObjectAndAnimateModes
# make geometry
interfaceNameList = [shaftInterfaceName]
materials = {'steel':{'youngsModulus':EModul, 'poissonsRatio':nu, 'density':rho}}
shaft = occ.Cylinder((0, -bearingHeight/2, 0), (0, 1, 0), bearingRadius, bearingHeight)
def GetCylinderCurvedFace(cylinder, axis):
    planarFacesList = []
    planarFacesList += [cylinder.faces.Min(axis)]
    planarFacesList += [cylinder.faces.Max(axis)]
    return [f for f in cylinder.faces if f not in planarFacesList][0]
GetCylinderCurvedFace(shaft, (0, 1, 0)).name = shaftInterfaceName
rotorSmall = occ.Cylinder((0, bearingHeight/2, 0), (0, 1, 0), smallRotorRadius, smallRotorHeight)
rotorLarge = occ.Cylinder((0, bearingHeight/2+smallRotorHeight, 0), (0, 1, 0), largeRotorRadius, largeRotorHeight)
rotorLarge = rotorLarge.MakeFillet(rotorLarge.edges, filletsRadiusFactor * largeRotorHeight) # add fillet
unbalance = occ.Sphere((0, bearingHeight/2+smallRotorHeight+largeRotorHeight/2, largeRotorRadius), sphereRadius)
geometry = shaft + rotorSmall + rotorLarge + unbalance
geometry.name = 'steel'
# after geometry is made, call this function
fem = GetFemObjectFromOCCGeometry(geometry, materials=materials, partName=partName,
                                  interfaceNameList=interfaceNameList)

#%%modelStop%%
exu.sys['exuAI_out']['femObject'] = fem
# MakeFFRFObjectAndAnimateModes(fem, partName=fem.metaData['name'], animateModes=True)

#%%samplecodestop%%
(SC, mbs) = MakeFFRFObjectAndAnimateModes(fem, partName=partName, animateModes=True)
