#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Author: Gemma4:31B
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from exudyn.FEM import *
import exudyn as exu
from exudyn.utilities import *
import numpy as np

SC = exu.SystemContainer()
mbs = SC.AddSystem()

oGround = mbs.CreateGround(graphicsDataList=[graphics.CheckerBoard(size=80)])

#load tower FEM interface
femTower = FEMinterface()
femTower.LoadFromFile('windTurbineTowerPart.npz')

towerDict = mbs.CreateFFRFReducedOrderObject('windTurbineTowerPart', femTower,
                                             referencePosition=[0., 0., 0.],
                                             initialVelocity=[0., 0., 0.],
                                             referenceRotationMatrix=np.eye(3),
                                             initialAngularVelocity=[0., 0., 0.],
                                             massProportionalDamping=0.,
                                             stiffnessProportionalDamping=0.,
                                             gravity=[0., 0., 0.])
mGround = towerDict['windTurbineTowerPart:ground']
mMainBearing = towerDict['windTurbineTowerPart:mainBearing']

#load rotor FEM interface
femRotor = FEMinterface()
femRotor.LoadFromFile('windTurbineRotorPart.npz')

#the rotor axis is the global x-axis at z=0; move it up to nacelle hole at z=15
rotorDict = mbs.CreateFFRFReducedOrderObject('windTurbineRotorPart', femRotor,
                                             referencePosition=[0., 0., 15],
                                             initialVelocity=[0., 0., 0.],
                                             referenceRotationMatrix=np.eye(3) + RotationMatrixX(0),
                                             initialAngularVelocity=[0., 0., 0.],
                                             massProportionalDamping=0.,
                                             stiffnessProportionalDamping=0.,
                                             gravity=[0., 0., 0.])
mShaft = rotorDict['windTurbineRotorPart:shaft']

#fix tower base to ground
mbs.CreateGenericJoint(bodyNumbers=[oGround, mGround],
                       constrainedAxes=[1,1,1,1,1,1])

#revolute joint about global x-axis connecting rotor to tower/nacelle
mbs.CreateRevoluteJoint(bodyNumbers=[mMainBearing, mShaft],
                        axis=[1, 0, 0],
                        useGlobalFrame=True)

#aerodynamic torque about x-axis
mbs.AddLoad(Torque(markerNumber=mShaft, loadVector=[1500*100, 0, 0])) # very high such that it is visible

mbs.sys["EvaluteMBScode.preAssemble"]=True
mbs.Assemble()
mbs.sys["EvaluteMBScode.postAssemble"]=True

tEnd = 0.1 # *50
stepSize = 0.01
#%%

simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.solutionWritePeriod = 1e-3
simulationSettings.solutionSettings.sensorsWritePeriod  = 1e-3
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/stepSize)
simulationSettings.timeIntegration.endTime = tEnd

visualizationSettings = SC.visualizationSettings
visualizationSettings.nodes.drawNodesAsPoint=False
visualizationSettings.nodes.show = False
visualizationSettings.markers.show = False
visualizationSettings.openGL.lineWidth = 3
visualizationSettings.openGL.light0.position = [6.0, 0, 10.0, 1.0]
visualizationSettings.openGL.light1.position = [6.0, 0, 5.0, 0.0]
visualizationSettings.openGL.light2.enable = True
visualizationSettings.openGL.light2.position = [2.0, 2.0, 1.0, 0.0]
visualizationSettings.openGL.light0.shadow = 0.3
visualizationSettings.openGL.light1.shadow = 0
visualizationSettings.view0.scene.drawCoordinateSystem = False
visualizationSettings.general.showHelpOnStartup = False
visualizationSettings.general.showSolverInformation = False
visualizationSettings.general.showSolverTime = False
visualizationSettings.view0.scene.showMeshEdges = False
visualizationSettings.contour.automaticRange = True
visualizationSettings.contour.outputVariableComponent = -1
visualizationSettings.loads.defaultSize = 5
visualizationSettings.loads.drawSimplified = False
visualizationSettings.loads.defaultRadius = 0.1
visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal
# visualizationSettings.raytracer.multiSampling = 3

#changes Joh:
#visualizationSettings.view0.camera.useRaytracer = True #switch with CTRL-R
# visualizationSettings.openGL.light1.enable = False
# visualizationSettings.openGL.light0.position = [-2.0, -10.0, 10, 1.0]
# visualizationSettings.openGL.light0.lightRadius = 0.5
# visualizationSettings.raytracer.advanced.shadowScalingFactor = 1
visualizationSettings.raytracer.advanced.shadowSmoothingSteps = 5
visualizationSettings.raytracer.lightRadiusVariations = 42
# visualizationSettings.openGL.lineWidth = 5
visualizationSettings.raytracer.multiSampling = 4
visualizationSettings.raytracer.numberOfThreads = 64
visualizationSettings.raytracer.keepWindowActive = True
visualizationSettings.raytracer.verbose = 1
# visualizationSettings.openGL.lineWidth = 5
# visualizationSettings.loads.drawSimplified = False

# simulationSettings.solutionSettings.recordImagesInterval = stepSize * 2

# define nice orientation
rendererState = {'centerPoint': [-5.27987957, 10.96018696,  0],
                 'rotationCenterPoint': [0., 0., 0.],
                 'maxSceneSize': 114.96334838867188,
                 'zoom': 16.136070251464844,
                 'boundingBox': [[-40.  , -40.  ,  -0.05],
                                 [40.        , 40.        , 20.36011505]],
                 'modelRotation': [[-0.25797516, -0.13758668,  0.95630467],
                                   [ 0.96598113, -0.01814349,  0.25797513],
                                   [-0.01814324,  0.99032354,  0.13758671]]}

SC.renderer.Start()
SC.renderer.SetState(rendererState) # now set the nice orientation
SC.renderer.DoIdleTasks()
mbs.SolveDynamic(simulationSettings)
SC.renderer.DoIdleTasks()
SC.renderer.Stop()
mbs.SolutionViewer()
