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
oGround = mbs.CreateGround()
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
mbs.AddLoad(Torque(markerNumber=mShaft, loadVector=[0, torque, 0]))

mbs.Assemble()

tEnd = 0.1
stepSize = 0.01

simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.solutionWritePeriod = 1e-2
simulationSettings.solutionSettings.sensorsWritePeriod = 1e-2
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/stepSize) #must be integer
simulationSettings.timeIntegration.endTime = tEnd

# SC.visualizationSettings.nodes.show = False
# SC.visualizationSettings.markers.show = False
# SC.visualizationSettings.general.drawCoordinateSystem = False
# SC.visualizationSettings.openGL.light0ambient = 1
# SC.visualizationSettings.openGL.multiSampling = 3
# SC.visualizationSettings.contour.automaticRange = True
# SC.visualizationSettings.contour.outputVariableComponent = -1
# SC.visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal
# simulationSettings.solutionSettings.recordImagesInterval = 0.01
# SC.renderer.Start()
# SC.renderer.DoIdleTasks()
mbs.SolveDynamic(simulationSettings)
# SC.renderer.DoIdleTasks()
# SC.renderer.Stop()
#%%samplecodestop%%
SC.visualizationSettings.nodes.show = False
SC.visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal
SC.visualizationSettings.contour.outputVariableComponent = -1
SC.visualizationSettings.view0.scene.drawCoordinateSystem = False
mbs.SolutionViewer()
