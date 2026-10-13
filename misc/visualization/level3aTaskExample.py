#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Author: Gemma4:31B
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import exudyn as exu
from exudyn.FEM import *
import ngsolve as ngs
import netgen.occ as occ

E = 200000000000
nu = 0.3
rho = 8000
materials = {'steel': {'youngsModulus': E,
                       'poissonsRatio': nu,
                       'density': rho}}
interfaceNameList = []

# Large rotor disk centered at origin
diskLength = 0.2
diskRadius = 0.15
disk = occ.Cylinder((-diskLength/2, 0, 0), (1, 0, 0), diskRadius, diskLength)

# Small rotor sections
smallLength = 0.25
smallRadius = 0.1
smallRight = occ.Cylinder((diskLength/2, 0, 0), (1, 0, 0), smallRadius, smallLength)
smallLeft = occ.Cylinder((-diskLength/2 - smallLength, 0, 0), (1, 0, 0), smallRadius, smallLength)

# Shafts
chamferSize = 0.1*0.06
shaftLength = 0.08 + chamferSize
shaftRadius = 0.06

shaftRightStart = diskLength/2 + smallLength
shaftRight = occ.Cylinder((shaftRightStart, 0, 0), (1, 0, 0), shaftRadius, shaftLength)
shaftRight = shaftRight.MakeChamfer(shaftRight.faces.Max((1, 0, 0)).edges, chamferSize)
interfaceName = 'shaftRight'
shaftRight.faces[0].name = interfaceName
interfaceNameList += [interfaceName]

shaftLeftStart = -diskLength/2 - smallLength - shaftLength
shaftLeft = occ.Cylinder((shaftLeftStart, 0, 0), (1, 0, 0), shaftRadius, shaftLength)
shaftLeft = shaftLeft.MakeChamfer(shaftLeft.faces.Min((1, 0, 0)).edges, chamferSize)
interfaceName = 'shaftLeft'
shaftLeft.faces[0].name = interfaceName
interfaceNameList += [interfaceName]

# Keyway: cut from the outer surface of the large rotor disk, centered at (0, 0.15, 0)
keywayLength = 0.125
keywayDepth = 0.015
keywayRadius = 0.02
# Build profile on a workplane in the x-z plane at the top surface, extrude in -y to cut
# The keyway extends along x, with semicircular ends of radius keywayRadius
straightLength = keywayLength - 2*keywayRadius
wp = occ.WorkPlane(occ.Axes((-keywayLength/2, diskRadius + 0.001, -keywayRadius),
                            n=(0, 1, 0),
                            h=(1, 0, 0)))
wp.MoveTo(keywayRadius, -keywayRadius)
wp.LineTo(keywayRadius + straightLength, -keywayRadius) \
  .ArcTo(keywayRadius + straightLength, keywayRadius, occ.gp_Vec2d(1, 0)) \
  .LineTo(keywayRadius, keywayRadius) \
  .ArcTo(keywayRadius, -keywayRadius, occ.gp_Vec2d(-1, 0)) \
  .Close()
keywayFace = wp.Face()
keyway = occ.Prism(keywayFace, occ.gp_Vec(0, -(keywayDepth + 0.001), 0))

geometry = disk + smallRight + smallLeft + shaftRight + shaftLeft - keyway
geometry.name = 'steel'
geo = occ.OCCGeometry(geometry)


geoMesh = geo.GenerateMesh(maxh=0.05)
mesh = ngs.Mesh(geoMesh)
fem = FEMinterface()

[bfM, bfK, fes] = fem.ImportMeshFromNGsolve(mesh, materials=materials,
                                            boundaryNamesList=interfaceNameList, meshOrder=2)
[boundaryNodesList, boundaryWeightsList] = fem.GetBoundaryNodeSetsAsLists()
fem.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryNodesList,
                                  nEigenModes=6, boundaryNodesWeights=boundaryWeightsList)
fem.metaData['name'] = 'balancedRotorPart'


SC = exu.SystemContainer()
mbs = SC.AddSystem()
print('Number of nodes:', fem.NumberOfNodes())
print('Computed eigenfrequencies in Hz:', fem.GetEigenFrequenciesHz())
createFFRFDict = mbs.CreateFFRFReducedOrderObject(name=fem.metaData['name'],
                                                  femInterface=fem,
                                                  verbose=True)
print('The FFRF underlying rigid body node has number', mbs.GetNodeNumber('NodeRigidBody:' + fem.metaData['name']))
print('The generic node holding the modes has number', mbs.GetNodeNumber('NodeGeneric:' + fem.metaData['name']))
print('The created fem object has number', mbs.GetObjectNumber(fem.metaData['name']))
# compute mesh volume
triangles = np.asarray(fem.surface[0]['Trigs'], dtype=int)
points = np.asarray(fem.GetNodePositionsAsArray(), dtype=np.float64)


from exudyn.interactive import AnimateModes
mbs.Assemble()

#%%
#compute overall dimension
minCoords = np.min(points,axis=0)
maxCoords = np.max(points,axis=0)
maxDim = np.linalg.norm(maxCoords-minCoords)
SC.visualizationSettings.nodes.show = False
SC.visualizationSettings.markers.show = False
SC.visualizationSettings.view0.scene.showFaceEdges = True
SC.visualizationSettings.openGL.multiSampling = 4
SC.visualizationSettings.openGL.lineWidth = 2
SC.visualizationSettings.view0.scene.drawWorldBasis = False
SC.visualizationSettings.view0.window.renderWindowSize = [1600, 1080]
SC.visualizationSettings.general.autoFitScene = False # otherwise, model may be difficult to be moved
SC.visualizationSettings.openGL.lightModelAmbient = [0.5, 0.5, 0.5, 0]
SC.visualizationSettings.openGL.light0.position = [0.0, 0.6, 0.2, 0.1] 
SC.visualizationSettings.openGL.light0.position = [0.2, 0.2, 0, 0.0]
SC.visualizationSettings.view0.camera.useRaytracer = False
AnimateModes(SC, mbs, mbs.GetNodeNumber('NodeGeneric:' + fem.metaData['name']),
             showTime=False, runOnStart=True)
