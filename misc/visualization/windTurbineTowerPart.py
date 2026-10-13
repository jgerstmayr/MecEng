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

E = 200000000000
nu = 0.3
rho = 8000
materials = {'steel': {'youngsModulus': E,
                       'poissonsRatio': nu,
                       'density': rho}}
interfaceNameList = []

groundFootPoint = (0, 0, 0)
groundAxis = (0, 0, 1)
groundRadius = 4
groundHeight = 2
ground = occ.Cylinder(groundFootPoint, groundAxis, groundRadius, groundHeight)
interfaceName = 'ground'
ground.faces.Min((0, 0, 1)).name = interfaceName
interfaceNameList += [interfaceName]

towerFootPoint = (0, 0, 0)
towerAxis = (0, 0, 1)
towerRadius = 0.25*min(3, 2.5)
towerHeight = 15
tower = occ.Cylinder(towerFootPoint, towerAxis, towerRadius, towerHeight)

nacelleLength = 3
nacelleWidth = 2.5
nacelleHeight = 2.5
nacelleCenter = (0, 0, towerHeight)
nacelle = occ.Box((nacelleCenter[0] - nacelleLength/2, nacelleCenter[1] - nacelleWidth/2, nacelleCenter[2] - nacelleHeight/2),
                  (nacelleCenter[0] + nacelleLength/2, nacelleCenter[1] + nacelleWidth/2, nacelleCenter[2] + nacelleHeight/2))
filletRadius = 0.25*min(3, 2.5, 2.5)
nacelle = nacelle.MakeFillet(nacelle.faces.Max((0, 0, 1)).edges, filletRadius)

holeFootPoint = (-3/2, 0, 15)
holeAxis = (1, 0, 0)
holeRadius = 0.3*min(2.5, 2.5)
holeHeight = nacelleLength
hole = occ.Cylinder(holeFootPoint, holeAxis, holeRadius, holeHeight)
hole.faces[0].name = 'mainBearing'
interfaceNameList += ['mainBearing']

geometry = ground + tower + nacelle - hole
geometry.name = 'steel'
geo = occ.OCCGeometry(geometry)
from ExuAI.flexUtilities import GetMeshParametersDict, MeshSafetyCheck
meshParametersDict=GetMeshParametersDict(geo, maxHLLM=0.05, meshOrderLLM=2)
geoMesh = geo.GenerateMesh(maxh=0.5)
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
fem.metaData['name'] = 'windTurbineTowerPart'


exu.sys['exuAI_out']['femObject'] = fem
MakeFFRFObjectAndAnimateModes(fem, partName=fem.metaData['name'], animateModes=True)
