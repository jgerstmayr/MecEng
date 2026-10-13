#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  This is the reference solution for a moving flexible stick.
#           It serves as an example of how to use the Floating Frame of
#           Reference Formulation (FFRF) in EXUDYN.
# Authors:  Sebastian Weyrer
# Date:     2025-11-11
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import time
import exudyn as exu
from exudyn import graphics
from exudyn.utilities import *
from exudyn.FEM import *
import ngsolve as ngs
import netgen.occ as occ
print('EXUDYN version:', exu.config.Version())

#%% define parameters and behavior of script
nModes = 6          # number of eigen modes in adition to static modes
g = -9.81           # gravity in global reference frame
useRenderer = True
useSolutionViewer = False

#%% set the parameters of the stick's geometry
radius = 0.1        # [m] radius of the stick
length = 2          # [m] length of the stick
flangeWidth = 0.2   # [m] width of the flange with which the stick is moved
meshOrder = 1

#%% define the materials of the stick (here, we only use one material)
materials = {'rubber':{'youngsModulus':5e6, 'poissonsRatio':0.49, 'density':920}}

#%% create the geometry of the stick and name the interface parts
body = occ.Cylinder((0, 0, 0), (0, -1, 0), r=radius, h=length)
flange = occ.Box((-flangeWidth/2, -flangeWidth/2, -flangeWidth/2), (flangeWidth/2, flangeWidth/2, flangeWidth/2), )
interfaceList = []
for j, face in enumerate(flange.faces):
    interfaceName = 'flange'
    face.name = interfaceName
interfaceList += [interfaceName]
pendulum = body + flange
pendulum.name = 'rubber' # assign material to geometries
geo = occ.OCCGeometry(pendulum)

#%% mesh geometry
print('Start meshing geometry')
start_time = time.time()
mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2)) # note: per default tetrahedron elements are built
print('Meshing needed %.3f seconds' % (time.time() - start_time))
print('Number of elements:', mesh.ne)
print('Used materials:', mesh.GetMaterials())
if False: # this should only run one time so it is placed below 'if False'
    import netgen.gui
    ngs.Draw(mesh, clipping=True)

#%% initialize the fem interface and make a mbs
fem = FEMinterface()
SC = exu.SystemContainer()
mbs = SC.AddSystem()
[bfM, bfK, fes] = fem.ImportMeshFromNGsolve(mesh, materials=materials,
                                            boundaryNamesList=interfaceList,
                                            meshOrder=meshOrder)

#%% make a list of interface's node numbers and node weights
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
                                  nEigenModes=6,
                                  useSparseSolver=True,
                                  excludeRigidBodyMotion=True,
                                  computationMode=HCBstaticModeSelection.RBE2)
print('HCB modes needed %.3f seconds of computation time' % (time.time() - start_time))
print('Eigenvalues:', fem.eigenValues)
print('Eigenfrequencies:', np.sqrt(fem.eigenValues)/(2*pi)) # or fem.GetEigenfrequenciesHz()
# the reduction is good if the used eigenmodes represent the frequency of the observed vibrations

#%% initialize the cms object and make an FFRF object
cms = ObjectFFRFreducedOrderInterface(fem)
objFFRF = cms.AddObjectFFRFreducedOrder(mbs, positionRef=[0, 0, 0],
                                             initialVelocity=[0, 0, 0], 
                                             initialAngularVelocity=[0, 0, 0],
                                             stiffnessProportionalDamping=0.01,
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
    print('Unweighted approximated global position of marker:', positionSuperElementMarker)
mFlange = boundaryMarkerList[0]

#%% add a sensor to the marker
sensor = mbs.AddSensor(SensorMarker(markerNumber=boundaryMarkerList[0],
                                    outputVariableType=exu.OutputVariableType.Position))

#%% add ground and marker
oGround = mbs.CreateGround(graphicsDataList=[graphics.Basis(length=1)])
mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround))

#%% add a generic joint and a user function to move the stick
def UFOffset(mbs, t, itemNumber, offsetUserFunctionParameters):
    x = SmoothStep(t, x0=0, x1=3, value0=0, value1=2)
    return [x, 0, 0, 0, 0, 0]
joint = mbs.AddObject(GenericJoint(markerNumbers=[mGround, mFlange],
                                   constrainedAxes=[1, 1, 1, 1, 1, 1],
                                   offsetUserFunction=UFOffset))

#%% do simulation (implicit dynamic simulation) (WYSWYS - What You See is What You Simulate)
mbs.Assemble()
print('Weighted global position of marker:', mbs.GetSensorValues(sensor))
h = 1e-3
tEnd = 10

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
