#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Details:  Create a flexible mbs to evaluate the LLM assisted flexible mbs
#           creation.
# Model:    Unbalanced rotor in bearing block
#
# Author:   Sebastian Weyrer
# Date:     2026-04-29
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%parametersstart%%
# parameters which can truly be varied in this script
torque = 20
wallThickness = 0.07
bearingRadius = 0.05
groundPlateInterfaceName = 'groundPlate'
jointInterfaceName = 'joint'
shaftInterfaceName = 'shaft'
part0 = '{bearingBlockPart}' # this is a parameter that decides which femInterface is loaded; it can be varied
part1 = '{unbalancedRotorPart}' # this is a parameter that decides which femInterface is loaded; it can be varied

#%%samplecodestart%%
import exudyn as exu
from exudyn.utilities import *
from exudyn.FEM import *

import re
import numpy as np
import sys

SC = exu.SystemContainer()
mbs = SC.AddSystem()

# load the fem objects
femBearingBlock = FEMinterface()
femBearingBlock.LoadFromFile(re.sub(r"[{}]", "", part0) + '.npz')
femUnbalancedRotor = FEMinterface()
femUnbalancedRotor.LoadFromFile(re.sub(r"[{}]", "", part1) + '.npz')

# use the new CreateFFRFReducedOrderObject function to get the FFRF objects and markers
bearingBlockCreateDict = mbs.CreateFFRFReducedOrderObject(name='bearingBlock', femInterface=femBearingBlock, verbose=True)
rotorCreateDict = mbs.CreateFFRFReducedOrderObject(name='rotor', femInterface=femUnbalancedRotor, verbose=True,
                                                   referencePosition=[0, 0, wallThickness+bearingRadius])
# add ground object
screwOffset = 0.13
oGround = mbs.CreateGround(graphicsDataList=[graphics.Brick(centerPoint=[0, 0, -0.5], size=[0.45, 0.45, 1],
                                                            color=graphics.color.grey),
                                             # graphics.Cylinder(pAxis=[-screwOffset, -screwOffset, 0], vAxis=[0, 0, 0.05], radius=0.02),
                                             # graphics.Cylinder(pAxis=[screwOffset, -screwOffset, 0], vAxis=[0, 0, 0.05], radius=0.02),
                                             # graphics.Cylinder(pAxis=[-screwOffset, screwOffset, 0], vAxis=[0, 0, 0.05], radius=0.02),
                                             # graphics.Cylinder(pAxis=[screwOffset, screwOffset, 0], vAxis=[0, 0, 0.05], radius=0.02)
                                             ])
# make the joints
mPlate = bearingBlockCreateDict['bearingBlock:' + groundPlateInterfaceName]
mbs.CreateGenericJoint(bodyNumbers=[oGround, mPlate],
                       constrainedAxes=[1]*6)
mBearingBlock = bearingBlockCreateDict['bearingBlock:' + jointInterfaceName]
mShaft = rotorCreateDict['rotor:' + shaftInterfaceName]
mbs.CreateRevoluteJoint(bodyNumbers=[mBearingBlock, mShaft],
                        axis=[0, 1, 0],
                        useGlobalFrame=True)
# add torque
mbs.AddLoad(Torque(markerNumber=mShaft, loadVector=[0, torque, 0],
                   bodyFixed=True))

mbs.Assemble()

tEnd = 0.1*50
stepSize = 0.01

simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.solutionWritePeriod = 1e-3
simulationSettings.solutionSettings.sensorsWritePeriod  = 1e-3
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/stepSize)
simulationSettings.timeIntegration.endTime = tEnd

visualizationSettings = SC.visualizationSettings
visualizationSettings.nodes.drawNodesAsPoint=False
visualizationSettings.nodes.show = False
visualizationSettings.markers.show = False
visualizationSettings.connectors.show = False
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
visualizationSettings.loads.defaultSize = 0.6
visualizationSettings.loads.drawSimplified = False
visualizationSettings.loads.defaultRadius = 0.01
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

simulationSettings.solutionSettings.recordImagesInterval = stepSize * 2

# define nice orientation
rendererState = {'centerPoint': [0.13464248, 0.04782581, 0.        ],
                 'rotationCenterPoint': [0., 0., 0.],
                 'maxSceneSize': 0.9707469344139099,
                 'zoom': 0.3746657967567444,
                 'boundingBox': [[-0.25      , -0.24000143, -0.13025855],
                                 [0.25      , 5.        , 0.41999999]],
                 'modelRotation': [[-0.32796147, -0.18227138,  0.9269402 ],
                        [ 0.92310017, -0.27041036,  0.27343002],
                        [ 0.20081589,  0.94533354,  0.25693905]]}

SC.renderer.Start()
SC.renderer.SetState(rendererState) # now set the nice orientation
SC.renderer.DoIdleTasks()
mbs.SolveDynamic(simulationSettings)
SC.renderer.DoIdleTasks()
SC.renderer.Stop()
mbs.SolutionViewer()
