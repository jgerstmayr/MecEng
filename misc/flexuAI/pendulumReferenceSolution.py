#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  This is the reference solution for a driven pendulum, simulated
#           as flexible body and using the Floating Frame of Reference
#           Formulation (FFRF).
# Authors:  Sebastian Weyrer
# Date:     2025-11-07
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import time
import exudyn as exu
from exudyn import graphics
from exudyn.utilities import *
from exudyn.FEM import *
fem = FEMinterface()
print('EXUDYN version:', exu.config.Version())
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%% define parameters and behavior of script
meshFileName = None # None or 'pendulum' (if the latter, an existing file is loaded)
nModes = 6          # number of eigen modes in adition to static modes
g = -9.81           # gravity in global reference frame
useRenderer = True
useSolutionViewer = False

#%% make mesh of a pendulum using the OCC wrapper if no meshFileName is given
if meshFileName == None:
    import ngsolve as ngs
    import netgen.occ as occ
    print('Creating pendulum geometry using the occ Python wrapper')
    radius = 0.1        # [m] radius of pendulum's body
    length = 2          # [m] length of pendulum's body
    jointRadius = 0.05  # [m] inner radius of the joint
    meshOrder = 1
    # define materials
    materials = {'rubber':{'youngsModulus':2e6, 'poissonsRatio':0.49, 'density':920}}
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # create pendulum's geometry
    body = occ.Cylinder((0, 0, 0), (0, -1, 0), r=radius, h=(length/2))
    joint = occ.Cylinder((0, 0, -radius), (0, 0, 1), r=radius, h=(2*radius))
    pendulum = body + joint
    innerJoint = occ.Cylinder((0, 0, -radius), (0, 0, 1), r=jointRadius, h=(2*radius))
    interfaceList = []
    for j, face in enumerate(innerJoint.faces):
        interfaceName = 'joint'
        face.name = interfaceName
    interfaceList += [interfaceName]
    pendulum -= innerJoint
    pendulum.name = 'rubber' # assign material to geometries
    geo = occ.OCCGeometry(pendulum)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # mesh geometry
    print('Start meshing geometry')
    start_time = time.time()
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2)) # note: per default tetrahedron elements are built
    print('Meshing needed %.3f seconds' % (time.time() - start_time))
    print('Number of elements:', mesh.ne)
    print('Used materials:', mesh.GetMaterials())
    if False: # this should only run one time so it is placed below 'if False'
        import netgen.gui
        ngs.Draw(mesh, clipping=True)
    [bfM, bfK, fes] = fem.ImportMeshFromNGsolve(mesh, materials=materials,
                                                boundaryNamesList=interfaceList,
                                                meshOrder=meshOrder)
    meshFileName = 'pendulum'
    fem.SaveToFile(meshFileName, mode='PKL') # save mesh

#%% load the generated mesh (always from file)
fem.LoadFromFile(meshFileName, mode='PKL')
boundaryNodesList = []
boundaryWeightsList = []
print('Number of defined node sets:', len(fem.nodeSets))
for nodeSet in fem.nodeSets:
    boundaryNodesList += [nodeSet['NodeNumbers']]
    boundaryWeightsList += [nodeSet['NodeWeights']]

#%% compute HCB modes (which are also needed for FFRF)
print('Start computing HCB modes')
start_time = time.time()
fem.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryNodesList, 
                                  nEigenModes=nModes,
                                  useSparseSolver=True,
                                  excludeRigidBodyMotion=True,
                                  computationMode=HCBstaticModeSelection.RBE2)
print('HCB modes needed %.3f seconds of computation time' % (time.time() - start_time))

#%% make an FFRF object
SC = exu.SystemContainer()
mbs = SC.AddSystem()
cms = ObjectFFRFreducedOrderInterface(fem)
objFFRF = cms.AddObjectFFRFreducedOrder(mbs, positionRef=[0, 0, 0],
                                             initialVelocity=[0, 0, 0], 
                                             initialAngularVelocity=[0, 0, 0],
                                             stiffnessProportionalDamping=0,
                                             rotationMatrixRef=RotationMatrixZ(0), # rotation w.r.t. reference position
                                             color=graphics.color.green,
                                             gravity=[0, g, 0])
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

#%% add super element rigid body marker(s)
boundaryMarkerList = []
for i, nodeSet in enumerate(boundaryNodesList):
    weights = boundaryWeightsList[i] # these factors could be based on surface integrals of the constrained mesh faces
    marker = mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=objFFRF['oFFRFreducedOrder'],
                                                   meshNodeNumbers=nodeSet,
                                                   weightingFactors=weights))
    boundaryMarkerList += [marker]
    positionSuperElementMarker = fem.GetNodePositionsMean(nodeNumberList=nodeSet)
    print('Approximated global position of marker:', positionSuperElementMarker)

#%% add ground and joint
oGround = mbs.CreateGround()
mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround,
                                        localPosition=positionSuperElementMarker))
joint = mbs.AddObject(RevoluteJointZ(markerNumbers=[mGround, boundaryMarkerList[0]],
                                     visualization=VObjectJointRevoluteZ(axisRadius=0.05)))

#%% add very simple P-drive via torsional spring damper
TSD = mbs.AddObject(TorsionalSpringDamper(markerNumbers=[mGround, boundaryMarkerList[0]],
                                          velocityOffset=pi/4,
                                          damping=500))

#%% do simulation (implicit dynamic simulation) (WYSWYS - What You See is What You Simulate)
mbs.Assemble() 
h = 1e-3
tEnd = 8

#++++++++++++++++++++++++++++++++++++++++++++++++++
# general visualization settings
SC.visualizationSettings.nodes.show = False
SC.visualizationSettings.markers.show = True
SC.visualizationSettings.markers.drawSimplified = True

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
    SC.renderer.DoIdleTasks()
mbs.SolveDynamic(simulationSettings=simulationSettings)
if useRenderer:
    SC.renderer.DoIdleTasks()
    SC.renderer.Stop() #safely close rendering window!
    
#++++++++++++++++++++++++++++++++++++++++++++++++++
if useSolutionViewer:
    mbs.SolutionViewer()
