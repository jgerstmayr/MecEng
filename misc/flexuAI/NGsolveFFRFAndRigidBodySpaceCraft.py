#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  Spacecraft Example for EXUDYN workshop at ECCOMAS Multibody 2025
#           Innsbruck; This is the main development file.
#           At first a flexible space craft is built using the open cascade
#           wrapper of NGsolve. Afterwards EXUDYN is used to
#           - compute the Hurty-Craig-Bampton (HCB) modes of the flexible body
#           - use them for the Floating Frame of Reference Formulation (FFRF)
#           Furthermore, a STL file generated from this process and physical
#           parameters extracted from the flexible model are used to set up a
#           rigid body model of the space craft.
# Authors:  Johannes Gerstmayr and Sebastian Weyrer 
# Date:     2025-07-04 (last edited)
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu
import exudyn.graphics as graphics
from exudyn.utilities import *
from exudyn.FEM import *
import ngsolve as ngs
import netgen.occ as occ

import os
import sys
import time
import pickle
import numpy as np
from math import pi

print('EXUDYN version:', exu.config.Version())

#++++++++++++++++++++++++++++++++++++++++++++++++++
# define parameters
# material paramaters
rhoAluminium = 2600         # [kg/m^3] density of aluminium alloy ((i.e. Al 7075))
rhoRubber = 920             # [kg/m^3] density of rubber
EModulusAluminium = 7e10    # [N/m^2] e modulus of aluminium alloy (i.e. Al 7075)
EModulusRubber = 5e6        # [N/m^2] e modulus of rubber
nuAluminium = 0.3           # poisson number of aluminium alloy (i.e. Al 7075)
nuRubber = 0.49             # poisson number of rubber (never use 0.5 to prevent singularities in stiffness matrix)
# parameters of space craft
bodyLength = 0.75           # [m] length of space craft's body
bodyHeight = 0.75           # [m] height of space craft's body
bodyThickness = 0.01        # [m] thickness of the space craft's body
addFillets = True           # fillet are added to the space craft's body
feetExposure = 0.25         # [m] exposure of the feet (i.e. how much they stand out under the bodywork)
feetHeight = 0.4            # [m] height of feet (i.e. how much the descent stage is above ground)
feetRadius = 0.025          # [m] radius of the feet (cylinders that form the tripod)
feetThickness = 0.01        # [m] thickness of the cylinder's feet
feetSphereRadius = 0.05     # [m] radius of the sphere that is at the end of the feet
# parameters of antenna
antennaRadius = 0.02        # [m] radius of the antenna (that is made out of rubber)
antennaHeight = 0.3         # [m] height of antenna (that is made out of rubber)
antennaSphereRadius = 0.05  # [m] sphere of antenna head (that is made out of rubber)
# parameters for mesh and HCB
nModes = 16                 # number of modes in addition to static ones
meshOrder = 1               # use order 2 for higher accuracy, but more unknowns
# parameters to model contact (note: contactMode 2 is not implemented for rigid body model here)
contactMode = 0             # contact mode used for flexible model: (0) SphereSphereContact, (1) RollingDiscPenalty, (2) general contact
contactStiffness = 1e5      # [N/m^2] contact stiffness between space craft and planet surface
contactDamping = 1e3        # [Ns/m] contact damping between space craft and planet surface
restitutionCoefficient = 0.5# restitution coefficient for the SphereSphereContact
dynamicFriction = 0.001                   # friction factor for the SphereSphereContact
stiffnessProportionalDamping = 0.02     # [s] stiffness proportional damping added to the floating coordinates
# general parameters
y0 = 0.5                    # [m] initial height of space craft
phi0 = pi/8                 # initial rotation of space craft
g = -1.62                   # [m/s^2] gravitational acceleration on the moon
addFlexibleModel = True     # decide whether flexible model is added to simulation
showStress = False          # if this is true, show stress of structures during simulation, otherwise local displacement
useRenderer = True          # decide whether renderer should be started to see live simulation
useSolutionViewer = True    # decide whether solution viewer should be opened if simulation is finished
# parameters for saving and loading files
alwaysCreateNewMesh = True # set this true, if the saved mesh file should be overwritten every time you run this script
meshFileName = 'spaceCraftMesh'
stlFileName = 'spaceCraftGraphics'
rigidBodyPropertiesFileName = 'rigidBodyProperties'

#++++++++++++++++++++++++++++++++++++++++++++++++++
# EXUDYN initializations
SC = exu.SystemContainer()
mbs = SC.AddSystem()
fem = FEMinterface()

#%% use the open cascade wrapper and ngsolve to create space craft if no saved meshFile is found
# check, whether saved mesh exists
if not os.path.isfile(meshFileName + '.pkl') or alwaysCreateNewMesh or showStress:
    print('Creating space craft geometry and mesh since no saved mesh is found (or new mesh should be created)')
    # main body
    body = occ.Box((0, 0, 0), (bodyLength, bodyHeight, bodyLength))
    if addFillets:
        body = body.MakeFillet(body.edges, 0.02)
    bodySub =  occ.Box((bodyThickness, bodyThickness, bodyThickness), (bodyLength-bodyThickness, bodyHeight-bodyThickness, bodyLength-bodyThickness))
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # construct the feet given the exposure and height
    def getOCCCYlinderParameters(startPoint, endPoint):
        vector = endPoint - startPoint
        length = np.linalg.norm(vector)
        unityVector = vector/length
        if type(endPoint) == tuple:
            pass
        else:
            endPoint = tuple(endPoint.reshape(1, -1)[0])
        return tuple(startPoint.reshape(1, -1)[0]), tuple(unityVector.reshape(1, -1)[0]), length, endPoint
    # make the tripod one foot consists of
    feetMountingDepth = 0.1
    startPoint = np.array([bodyLength*(3/4), 0 + feetMountingDepth, bodyLength - feetMountingDepth])
    endPoint = np.array([bodyLength + feetExposure*sin(pi/4), -feetHeight, bodyLength + feetExposure*cos(pi/4)])
    [startPoint, unityVectorFeet1TripodPart1, lengthFeet1TripodPart1, endPoint] = getOCCCYlinderParameters(startPoint, endPoint)
    cyl1 = occ.Cylinder(startPoint, unityVectorFeet1TripodPart1, r=feetRadius, h=lengthFeet1TripodPart1)
    cyl1Sub = occ.Cylinder(startPoint, unityVectorFeet1TripodPart1, r=(feetRadius-feetThickness), h=lengthFeet1TripodPart1)
    tripod = cyl1 - cyl1Sub
    startPoint = np.array([bodyLength - feetMountingDepth, 0 + feetMountingDepth, bodyLength*(3/4)])
    [startPoint, unityVectorFeet1TripodPart1, lengthFeet1TripodPart1, endPoint] = getOCCCYlinderParameters(startPoint, endPoint)
    cyl2 = occ.Cylinder(startPoint, unityVectorFeet1TripodPart1, r=feetRadius, h=lengthFeet1TripodPart1)
    cyl2Sub = occ.Cylinder(startPoint, unityVectorFeet1TripodPart1, r=(feetRadius-feetThickness), h=lengthFeet1TripodPart1)
    tripod += cyl2 - cyl2Sub
    startPoint = np.array([bodyLength - feetMountingDepth, bodyHeight*(1/4), bodyLength - feetMountingDepth])
    [startPoint, unityVectorFeet1TripodPart1, lengthFeet1TripodPart1, endPoint] = getOCCCYlinderParameters(startPoint, endPoint)
    cyl3 = occ.Cylinder(startPoint, unityVectorFeet1TripodPart1, r=feetRadius, h=lengthFeet1TripodPart1)
    cyl3Sub = occ.Cylinder(startPoint, unityVectorFeet1TripodPart1, r=(feetRadius-feetThickness), h=lengthFeet1TripodPart1)
    tripod += cyl3 - cyl3Sub
    # and construct the feet (rotating the created tripod accoridngly)
    feet = 0
    boundariesList = []
    for i in range(4):
        sphere = occ.Sphere(endPoint, feetSphereRadius)
        boundaryName = 'foot' + str(i)
        # name all faces of the spheres
        for j, face in enumerate(sphere.faces):
            face.name = boundaryName
        boundariesList += [boundaryName]
        foot = tripod + sphere
        feet += foot.Rotate(occ.Axis((bodyLength/2, 0, bodyLength/2), occ.Y), ang=90*i)
    spaceCraftOCCGeometry = body + feet
    spaceCraftOCCGeometry = spaceCraftOCCGeometry - bodySub # now subtract the inner part (AFTER adding the feet)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # now move the whole space craft so that it stands on the ground
    spaceCraftOCCGeometry = spaceCraftOCCGeometry.Move((-bodyLength/2, feetHeight + feetSphereRadius, -bodyLength/2))
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # add antenna to the space craft consitsing of another material
    antenna = occ.Cylinder((0, feetHeight + feetSphereRadius + bodyHeight, 0), occ.Y, r=antennaRadius, h=antennaHeight)
    antenna = antenna + occ.Sphere((0, feetHeight + feetSphereRadius + bodyHeight + antennaHeight - (bodyThickness/2), 0), antennaSphereRadius)
    antenna.name = 'antennaMaterial' # assign another material to the antenna
    spaceCraftOCCGeometry = occ.Glue((spaceCraftOCCGeometry, antenna)) # to keep different materials
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # define the two different materials of the space craft, that will be needed later on when loading the mesh
    materials = {'default':{'youngsModulus':EModulusAluminium, 'poissonsRatio':nuAluminium, 'density':rhoAluminium},
                 'antennaMaterial':{'youngsModulus':EModulusRubber, 'poissonsRatio':nuRubber, 'density':rhoRubber}}
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # make the mesh of the space craft using ngsolve
    geo = occ.OCCGeometry(spaceCraftOCCGeometry)
    print('Start meshing geometry')
    start_time = time.time()
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2)) # note: per default tetrahedron elements are built
    
    # #++++++++++++++++++++++++++++++++++++++++++++++++++
    # also save stl file now to use for rigid body simulation
    # save the ng mesh as stl file, not the mesh!
    [points, trigs] = graphics.NGsolveMesh2PointsAndTrigs(mesh=mesh, addNormals=False, meshOrder=1)
    gSpaceCraft = graphics.FromPointsAndTrigs(points=points, triangles=trigs)
    graphics.ExportSTL(graphicsData=gSpaceCraft, fileName=stlFileName + '.stl')
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # set this to true, if you want to visualize the mesh inside netgen/ngsolve
    if False:
        import netgen.gui
        ngs.Draw(mesh, clipping=True)
    print('Meshing needed %.3f seconds' % (time.time() - start_time))
    print('Number of elements:', mesh.ne)
    print('Used materials:', mesh.GetMaterials())
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # load the created mesh to the fem interface so that it can then be used by EXUDYN
    # here the different materials are considered, so they are also saved in the meshFile
    [bfM, bfK, fes] = fem.ImportMeshFromNGsolve(mesh, materials=materials,
                                                boundaryNamesList=boundariesList,
                                                meshOrder=meshOrder)
    fem.SaveToFile(meshFileName, mode='PKL')
else:
    print('Using found mesh at', meshFileName)
    # load the found mesh
    fem.LoadFromFile(meshFileName, mode='PKL')
  
#%% compute HCB modes that are then used for the FFRF
# Clearly, the free-free modes are not well suited for the modeling of the deformations within constrained structures.
# Therefore, we can use modes based on ideas of Hurty and Craig-Bampton, as shown in the following.
# see https://exudyn.readthedocs.io/en/stable/docs/RST/ModelOrderReductionAndComponentModeSynthesis.html
# get boundary nodes and weights (defined with the occ wrapper)
boundaryNodesList = []
boundaryWeightsList = []
for nodeSet in fem.nodeSets:
    boundaryNodesList += [nodeSet['NodeNumbers']]
    boundaryWeightsList += [nodeSet['NodeWeights']]

#++++++++++++++++++++++++++++++++++++++++++++++++++
# now do computation of HCB modes
# pure static condensation would be Guyan-Irons method
print('Start computing HCB modes')
start_time = time.time()
fem.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryNodesList, 
                                  nEigenModes=nModes, # number of eigen modes in addition to STATIC modes
                                  useSparseSolver=True,
                                  excludeRigidBodyMotion=True,
                                  computationMode=HCBstaticModeSelection.RBE2)
print('HCB modes needed %.3f seconds of computation time' % (time.time() - start_time))

#++++++++++++++++++++++++++++++++++++++++++++++++++
# make some post processing computations for local stress if desired
# remember to do this before the cms object is created
if showStress == True:
    mat = KirchhoffMaterial(materials=materials, fes=fes)
    print('Computing post processing modes for to show local stress')
    start_time = time.time()
    fem.ComputePostProcessingModesNGsolve(fes, material=mat, outputVariableType=exu.OutputVariableType.StressLocal)
    print('Computing post processing modes for stress needed %.3f seconds' % (time.time() - start_time))
    SC.visualizationSettings.contour.reduceRange = True
    SC.visualizationSettings.contour.outputVariable = exu.OutputVariableType.StressLocal
    SC.visualizationSettings.contour.outputVariableComponent = -1 # norm
else:
    SC.visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal
    SC.visualizationSettings.contour.outputVariableComponent = -1 # norm

#%% add the Floating Frame of Reference Formulation (FFRF) object with which simulation is then done (in EXUDYN this object can then be treated as 'normal' object)
# we can use the FFRF object to get parameters for the rigid body simulation
# we name that object/element Component Mode Syntheis (CMS), as flexible deformations are a linear combination here
cms = ObjectFFRFreducedOrderInterface(fem)
# add offset in x direction since rigid body model must also be placed
objFFRF = cms.AddObjectFFRFreducedOrder(mbs, positionRef=[-1.5, y0, 0],
                                             initialVelocity=[0, 0, 0], 
                                             initialAngularVelocity=[0, 0, 0],
                                             stiffnessProportionalDamping=stiffnessProportionalDamping,
                                             rotationMatrixRef=RotationMatrixZ(phi0), # rotation w.r.t. reference position
                                             color=[0.1, 0.9, 0.1, 1],
                                             gravity=[0, g, 0])

#++++++++++++++++++++++++++++++++++++++++++++++++++
# optionally, just animate HCB-modes (setting this True, will animate modes and exit the script)
if False:
    from exudyn.interactive import AnimateModes
    mbs.Assemble()
    SC.visualizationSettings.nodes.show = False
    SC.visualizationSettings.openGL.showFaceEdges = True
    SC.visualizationSettings.openGL.multiSampling = 4
    SC.visualizationSettings.openGL.lineWidth = 2
    SC.visualizationSettings.general.drawWorldBasis = True
    SC.visualizationSettings.window.renderWindowSize = [1600, 1080]
    SC.visualizationSettings.general.autoFitScene = False # otherwise, model may be difficult to be moved
    nodenumber = objFFRF['nGenericODE2'] # this is the node with the generalized coordinates
    AnimateModes(SC, mbs, nodenumber, period=0.1, scaleAmplitude = 0.02, showTime=False, runOnStart=True)
    sys.exit()

#++++++++++++++++++++++++++++++++++++++++++++++++++
# get parameters out of the cms object that are then used for rigid body simulation
# using the meshed space craft, we can get some properties for it's rigid body model
# note that the meshed object and the returned inertia object have the same reference system
# how the meshed object lies in this reference system is defined i.e. how you set up the system in ngsolve ...
# the inertia object includes the COM, so via inertia.Translated(-COM) the inertia w.r.t. COM can be obtained (then the reference point is COM)
# the inertia object also includes the InertiaCOM object itself
# def getRigidBodyInertiaFromCMSInterface(cmsInterface):
#     mass = cmsInterface.totalMass
#     inertiaTensor = cmsInterface.inertiaLocal
#     COM = cmsInterface.chiU
#     # default: the inearia tensor is not provided at the Center Of Mass (COM)
#     inertia = RigidBodyInertia(mass=mass, inertiaTensor=inertiaTensor, com=COM, inertiaTensorAtCOM=False)
#     return inertia
# inertiaSpaceCraft = getRigidBodyInertiaFromCMSInterface(cms)
# in newest EXUDYN version:
inertiaSpaceCraft = fem.GetRigidBodyInertia()
# also save the mesh created by 
print('Total mass [kg] of satellite:', np.round(inertiaSpaceCraft.mass, 2))
print('COM [m] of satellite w.r.t. the reference system:\n', np.round(inertiaSpaceCraft.COM(), 2))
print('Inertia tensor of space craft w.r.t. origin:\n', np.round(inertiaSpaceCraft.Inertia(), 2))


#%% now add markers for the feet of the space craft and use different methods to model contact
oGround = mbs.CreateGround(referencePosition=[0, 0, 0], graphicsDataList=[graphics.CheckerBoard(normal=[0, 1, 0], size=20)])
boundaryMarkerList = []
feetPositionsList = [] # this will be used for the rigid body simulation to get positions of feet
for i, nodeSet in enumerate(boundaryNodesList):
    weights = boundaryWeightsList[i]
    marker = mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=objFFRF['oFFRFreducedOrder'], 
                                                   meshNodeNumbers=np.array(nodeSet), # these are the meshNodenumbers
                                                   weightingFactors=weights))
    boundaryMarkerList += [marker]
    feetPositionsList += [fem.GetNodePositionsMean(nodeNumberList=nodeSet)]
    
#++++++++++++++++++++++++++++++++++++++++++++++++++
# now save everything needed for the rigid body simulation as pickle file
rigidBodyData = {'inertiaObject': inertiaSpaceCraft, 'feetPositionsList': feetPositionsList}
# with open(rigidBodyPropertiesFileName + '.pkl', 'wb') as f:
#     pickle.dump(rigidBodyData, f)
    
#++++++++++++++++++++++++++++++++++++++++++++++++++
# add contact to model contact between feet and ground
planetRadius = 500
if contactMode == 0 or contactMode == 2:
    mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0, -planetRadius, 0]))
    if contactMode == 2:
        gContact = mbs.AddGeneralContact()
        gContact.verboseMode = 1
        gContact.AddSphereWithMarker(markerIndex=mGround, radius=planetRadius,
                                     contactStiffness=contactStiffness/2, contactDamping=0, frictionMaterialIndex=0)
elif contactMode == 1:
    mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0, 0, 0]))
for marker in boundaryMarkerList:
    if contactMode == 0:
        nGeneric = mbs.AddNode(NodeGenericData(numberOfDataCoordinates=4, initialCoordinates=[0]*4))
        # CreateSphereSphereContact not for fleixble bodies as we need a MarkerSuperElementRigid on the flexible body
        mbs.AddObject(ObjectContactSphereSphere(markerNumbers=[mGround, marker],
                                                nodeNumber=nGeneric, spheresRadii=[planetRadius, feetSphereRadius],
                                                dynamicFriction=dynamicFriction,
                                                restitutionCoefficient=restitutionCoefficient,
                                                contactStiffness=contactStiffness, contactDamping=contactDamping,
                                                contactStiffnessExponent=1, impactModel=0))
    elif contactMode == 1:
        nGeneric = mbs.AddNode(NodeGenericData(numberOfDataCoordinates=3, initialCoordinates=[0, 0, 0]))
        mbs.AddObject(RollingDiscPenalty(markerNumbers=[mGround, marker],
                                         nodeNumber=nGeneric, discRadius=feetSphereRadius,
                                         planeNormal=[0, 1, 0], contactStiffness=contactStiffness,
                                         contactDamping=contactDamping, visualization=VObjectConnectorRollingDiscPenalty(show=False)))
    elif contactMode == 2:
        gContact.AddSphereWithMarker(markerIndex=marker, radius=feetSphereRadius,
                                     contactStiffness=contactStiffness/2, contactDamping=contactDamping, frictionMaterialIndex=0)
        gContact.SetFrictionPairings(np.eye(1))

#%% add the rigid body object of the space craft
if not addFlexibleModel:
    mbs.Reset()
    # in this case, again add ground and it's marker for the RollingDiscPenalty (if mbs is not reset, we do not need that here)
    oGround = mbs.CreateGround(referencePosition=[0, 0, 0], graphicsDataList=[# graphics.Basis(length=1),
                                                                              graphics.CheckerBoard(normal=[0, 1, 0], size=20)])
    if contactMode == 0 or contactMode == 2:
        mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0, -planetRadius, 0]))
    elif contactMode == 1:
        mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0, 0, 0]))

#++++++++++++++++++++++++++++++++++++++++++++++++++
# load parameters that are obtained via the mesh of the space craft
# graphics
gSpaceCraft = graphics.FromSTLfileASCII(fileName=stlFileName+'.stl', color=color4orange)
# inertia and feet positions
# with open(rigidBodyPropertiesFileName + '.pkl', 'rb') as f:
#     rigidBodyData = pickle.load(f)
inertiaSpaceCraft = rigidBodyData['inertiaObject'] # contains mass, COM, and inertia
feetPositionsList = rigidBodyData['feetPositionsList'] # local positions of the feet

#++++++++++++++++++++++++++++++++++++++++++++++++++
# add the rigid body
bSpaceCraft = mbs.CreateRigidBody(referencePosition=[1.5*addFlexibleModel, 0, 0],
                                  referenceRotationMatrix=RotationMatrixZ(0), # rotation w.r.t. reference position
                                  initialDisplacement=[0, y0, 0],
                                  initialRotationMatrix=RotationMatrixZ(phi0), # superimposed to the reference rotation matrix (w.r.t. reference position)
                                  inertia=inertiaSpaceCraft,
                                  gravity=[0, g, 0],
                                  graphicsDataList=[gSpaceCraft,
                                                    #graphics.Basis()
                                                    ])
for i in range(4):
    # SphereSphereContact
    print('position of feet ', str(i) + ':', np.round(feetPositionsList[i],2))
    if contactMode == 0:
        mbs.CreateSphereSphereContact(bodyNumbers=[oGround, bSpaceCraft],
                                      localPosition0=[0, -planetRadius, 0],
                                      localPosition1=feetPositionsList[i],
                                      spheresRadii=[planetRadius, feetSphereRadius],
                                      dynamicFriction=dynamicFriction, restitutionCoefficient=restitutionCoefficient,
                                      contactStiffness=contactStiffness, contactDamping=contactDamping,
                                      contactStiffnessExponent=1, impactModel=0)
    # Rolling Disc Penalty
    elif contactMode == 1 or contactMode == 2:
        mFoot = mbs.AddMarker(MarkerBodyRigid(bodyNumber=bSpaceCraft, localPosition=feetPositionsList[i]))
        nGeneric = mbs.AddNode(NodeGenericData(numberOfDataCoordinates=3, initialCoordinates=[0, 0, 0]))
        mbs.AddObject(RollingDiscPenalty(markerNumbers=[mGround, mFoot],
                                         nodeNumber=nGeneric, discRadius=feetSphereRadius,
                                         planeNormal=[0, 1, 0], contactStiffness=contactStiffness,
                                         contactDamping=contactDamping, visualization=VObjectConnectorRollingDiscPenalty(show=False)))

#%% do simulation (implicit dynamic simulation) (WYSWYS - What You See is What You Simulate)
mbs.Assemble() 
h = 1e-3
tEnd = 10

#++++++++++++++++++++++++++++++++++++++++++++++++++
# general visualization settings
SC.visualizationSettings.nodes.show = False
SC.visualizationSettings.markers.show = True
SC.visualizationSettings.markers.drawSimplified = True
SC.visualizationSettings.openGL.light0position = [0, 10, 1, 1]
SC.visualizationSettings.openGL.shadow = 0.25

#++++++++++++++++++++++++++++++++++++++++++++++++++
# simulation settings
simulationSettings = exu.SimulationSettings()
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/h)
simulationSettings.timeIntegration.endTime = tEnd
simulationSettings.solutionSettings.writeSolutionToFile = True
simulationSettings.timeIntegration.newton.useModifiedNewton = True
simulationSettings.displayComputationTime = True

#++++++++++++++++++++++++++++++++++++++++++++++++++
if useRenderer:
    SC.renderer.Start()
    # set render state
    renderState = {'centerPoint': [0, 1, 0],
                    'rotationCenterPoint': [0, 0, 0], 
                    'maxSceneSize': 15, 
                    'zoom':2.5}
    SC.renderer.SetState(renderState)
    SC.renderer.DoIdleTasks()
mbs.SolveDynamic(simulationSettings=simulationSettings)
if useRenderer:
    SC.renderer.DoIdleTasks()
    SC.renderer.Stop() #safely close rendering window!
    
#++++++++++++++++++++++++++++++++++++++++++++++++++
if useSolutionViewer:
    mbs.SolutionViewer()
