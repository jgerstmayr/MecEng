#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Author: Gemma4:31B
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import sys
from pathlib import Path
if "__file__" in globals():
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import exudyn as exu
from exudyn.FEM import *
import ngsolve as ngs
import netgen.occ as occ
try:
    from ExuAI.flexUtilities import GetMeshParametersDict, MeshSafetyCheck, MakeFFRFObjectAndAnimateModes
except:
    pass

E = 60000000000
nu = 0.33
rho = 2500
materials = {'aluminium': {'youngsModulus': E,
                           'poissonsRatio': nu,
                           'density': rho}}
interfaceNameList = []

innerShaftRadius = 0.3*min(2.5, 2.5)
innerShaft = occ.Cylinder((-3/2, 0, 0), (1, 0, 0), innerShaftRadius, 3)
interfaceName = 'shaft'
innerShaft.faces[0].name = interfaceName
interfaceNameList += [interfaceName]

outerHubRadius = 1.2*innerShaftRadius
outerHub = occ.Cylinder((3/2, 0, 0), (1, 0, 0), outerHubRadius, 0.75)

blade = occ.Box((3/2, -0.4/2, 0),
                (3/2 + 0.75, 0.4/2, -10))
bladeRotationAxis = occ.Axis((3/2 + 0.75/2, 0, 0), (0, 0, 1))
blade = blade.Rotate(bladeRotationAxis, 50)

rotorAxis120 = occ.Axis((0, 0, 0), (1, 0, 0))
blade2 = blade.Rotate(rotorAxis120, 120)
blade3 = blade.Rotate(rotorAxis120, 240)

geometry = innerShaft + outerHub + blade + blade2 + blade3
geometry.name = 'aluminium'
geo = occ.OCCGeometry(geometry)
from ExuAI.flexUtilities import GetMeshParametersDict, MeshSafetyCheck
meshParametersDict=GetMeshParametersDict(geo, maxHLLM=0.5, meshOrderLLM=2)
geoMesh = geo.GenerateMesh(maxh=meshParametersDict['maxH'], grading=meshParametersDict['grading'], curvaturesafety=meshParametersDict['curvatureSafety'])
mesh = ngs.Mesh(geoMesh)
meshOrder=meshParametersDict['meshOrder']
fem = FEMinterface()
exu.sys['exuAI_out']['materials'] = materials
[bfM, bfK, fes] = fem.ImportMeshFromNGsolve(mesh, materials=materials,
                                            boundaryNamesList=interfaceNameList, meshOrder=meshOrder)
[boundaryNodesList, boundaryWeightsList] = fem.GetBoundaryNodeSetsAsLists()
MeshSafetyCheck(fem, meshParametersDict['maxNumberOfFemNodes'])
fem.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryNodesList,
                                  nEigenModes=6, boundaryNodesWeights=boundaryWeightsList)
fem.metaData['name'] = 'windTurbineRotorPart'


exu.sys['exuAI_out']['femObject'] = fem
MakeFFRFObjectAndAnimateModes(fem, partName=fem.metaData['name'], animateModes=True)
