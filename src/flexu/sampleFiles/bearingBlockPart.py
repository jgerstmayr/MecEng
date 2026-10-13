#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Details:  Create a fem interface to evaluate the LLM assisted flexible mbs 
#           creation (i.e. the fem interface creation).
# Model:    Bearing block part
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
wallThickness = 0.07
filletsRadiusFactor0 = 0.5
filletsRadiusFactor1 = 0.4
plateLength = 0.4
plateWidth = 0.4
plateHeight = 0.03
EModul = 5e6
nu = 0.49
rho = 950
groundPlateInterfaceName = 'groundPlate'
jointInterfaceName = 'joint'
partName = 'bearingBlockPart'

#%%samplecodestart%%
import exudyn as exu
import netgen.occ as occ
from helper.utilitiesForSampleFiles import GetFemObjectFromOCCGeometry, MakeFFRFObjectAndAnimateModes
# make geometry
interfaceNameList = [groundPlateInterfaceName, jointInterfaceName]
materials = {'steel':{'youngsModulus':EModul, 'poissonsRatio':nu, 'density':rho}}
groundPlate = occ.Box((-0.5*plateLength, -0.5*plateWidth, 0), (0.5*plateLength, 0.5*plateWidth, plateHeight))
groundPlate.faces.Min((0, 0, 1)).name = groundPlateInterfaceName
groundPlate = groundPlate.MakeFillet(groundPlate.faces.Max((0, 0, 1)).edges, filletsRadiusFactor0*plateHeight) # add fillet
outerCylinderJointSleeve = occ.Cylinder((0, -0.5*bearingHeight, bearingRadius+wallThickness), (0, 1, 0), bearingRadius+wallThickness, bearingHeight)
outerCylinderJointSleeve = outerCylinderJointSleeve.MakeFillet(outerCylinderJointSleeve.edges, filletsRadiusFactor1*wallThickness) # add fillet
innerCylinderJointSleeve = occ.Cylinder((0, -0.5*bearingHeight, bearingRadius+wallThickness), (0, 1, 0), bearingRadius, bearingHeight)
innerCylinderJointSleeve.faces.name = jointInterfaceName
jointSleeve = outerCylinderJointSleeve - innerCylinderJointSleeve
geometry = groundPlate + jointSleeve
geometry.name = 'steel'
# after geometry is made, call this function
fem = GetFemObjectFromOCCGeometry(geometry, materials=materials, partName=partName,
                                  interfaceNameList=interfaceNameList)

#%%modelStop%%
exu.sys['exuAI_out']['femObject'] = fem
# MakeFFRFObjectAndAnimateModes(fem, partName=fem.metaData['name'], animateModes=True)

#%%samplecodestop%%
(SC, mbs) = MakeFFRFObjectAndAnimateModes(fem, partName=partName, animateModes=True)
