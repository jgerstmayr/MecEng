#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: evalHelper.py
# 
# Details:  this file serves as a documented example that shall be used by 
#           LLMs to update information on their internal knowledge of Exudyn;
#           the text is split by #%% comments, including tags for respective functionality
#
# Author:   Johannes Gerstmayr
# Date:     2024-12-30
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#difficulty levels:
#0..tags that should always be included
#5..ground + mass points + springs + distance, 10..rigid bodies, 15..rigid body joints, 
#20..Wheel, 25..SphereContact, 30..more complex rigid body joints (rotations)
#50..flexible bodies
import exudyn.graphics as graphics

#%%
#TAG: general information
#DIFFICULTY: 0
#INFO: Use the following information as an update of the internal data of the Python multibody code Exudyn.
#NOTE: mbs.Create...(...) functions call several functions in the background to create nodes, objects, markers and loads in the system 'mbs'.
#most quantities such as initial or reference positions and velocities are giving as 3D lists [x,y,z] or numpy arrays for positions, velocities, ....
#local positions are positions in body-fixed coordinates; they are relative to the body's reference (usually the COM).
#%%
#TAG: generalrigid
#DIFFICULTY: 30
#rotations are usually given as rotation matrix (3x3 numpy array); you can use RotationVector2RotationMatrix([rotX, rotY, rotZ]) which returns a rotation matrix from a global x,y,z rotation vector
#for working with rigid bodies, note that there is always a local coordinate system in the body, which means that forces or joints are attached at local positions (and local coordinates) in the body.

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++
#TAG: general import
#DIFFICULTY: 0
#INFO: import required libraries and import the create function for the FFRF object; it is very important that you do this
from exudyn.FEM import *

import exudyn as exu
from exudyn.utilities import *
import numpy as np

#set up new multibody system to work with
SC = exu.SystemContainer()
mbs = SC.AddSystem()

# add items here


#%%
#TAG: ground
#DIFFICULTY: 5
#USES: CreateGround
#INFO: Create a ground object as an inertial reference frame for the multibody system. Even several ground objects at specific positions can be added.
oGround = mbs.CreateGround()

#%%
#TAG: flexible machine part
#DIFFICULTY: 5
#USES: CreateFFRFReducedOrderObject
#INFO: load a pre-built machine part (as a femInterface) and assemble it as a flexible body in a multibody simulation.
# the flexible body is placed at referencePosition in the global frame.
# markers for joints/loads are retrieved from partCreateDict using 'partName:interfaceName'.
femInterface = FEMinterface()
femInterface.LoadFromFile('nameOfFemInterface.npz') # replace with actual file name; must match fem.metaData['name']
# referencePosition moves the body in the global frame
# example: beam centered at origin → set referencePosition=[beamLength/2, 0, 0] to place left end at global origin
partCreateDict = mbs.CreateFFRFReducedOrderObject('part0', femInterface,
                                                  referencePosition=[0., 0., 0.],
                                                  initialVelocity=[0., 0., 0.],
                                                  referenceRotationMatrix=np.eye(3),
                                                  initialAngularVelocity=[0., 0., 0.],
                                                  massProportionalDamping=0.,
                                                  stiffnessProportionalDamping=0.,
                                                  gravity=[0., 0., 0.])
# marker format is always 'partName:interfaceName' — both must match exactly what was defined in the fem object context
mObject0 = partCreateDict['part0:interfaceName0']
mObject1 = partCreateDict['part0:interfaceName1']
# use mObject0, mObject1 to connect joints, apply loads, etc.

#%%
#TAG: fixed joint
#DIFFICULTY: 5
#USES: CreateGenericJoint
#INFO: Use this to make a fixed joint between two objects
# For a fixed joint, use constrainedAxes = [1,1,1,1,1,1]
# either use the joint between markers or, if you need to fix something to ground, use oGround instead of object's marker
mbs.CreateGenericJoint(bodyNumbers=[oGround, mObject0],
                       constrainedAxes=[1,1,1,1,1,1])

#%%
#TAG: revolute joint
#DIFFICULTY: 5
#USES: CreateRevoluteJoint
#INFO: Use this if a revolute joint is needed.
# the axis argument specifies around which axis rotation is possible
# either use the joint between markers or, if you need to define a joint between a body and ground, use oGround instead of object's marker
mbs.CreateRevoluteJoint(bodyNumbers=[oGround, mObject1], 
                        axis=[0, 0, 1], #example: rotation about global z-axis
                        useGlobalFrame=True)

#%%
#TAG: prismatic joint
#DIFFICULTY: 5
#USES: CreatePrismaticJoint
#INFO: Use this if a prismatic joint is needed.
# the axis argument specifies along which axis translation is possible
# either use the joint between markers or, if you need to define a joint between a body and ground, use oGround instead of object's marker
mbs.CreatePrismaticJoint(bodyNumbers=[oGround, mObject1], 
                        axis=[1, 0, 0], #example: translation along global x-axis
                        useGlobalFrame=True)

#%%
#TAG: load
#DIFFICULTY: 5
#USES: AddLoad(Force
#INFO: add a load vector [x, y, z] (in global coordinates) to markers
mbs.AddLoad(Force(markerNumber=mObject1, loadVector=[1, 0, 0]))

#%%
#TAG: torque
#DIFFICULTY: 5
#USES: AddLoad(Torque
#INFO: add a torque vector [x, y, z] (in global coordinates) to markers
mbs.AddLoad(Torque(markerNumber=mObject1, loadVector=[1, 0, 0]))

#%%
#TAG: simulationPure
#DIFFICULTY: 10000
mbs.Assemble()
endTime = 2 #simulation time in seconds
stepSize = 0.002 #should be small enough to achieve sufficient accuracy

#some simulation parameters:
simulationSettings = exu.SimulationSettings() #takes currently set values or default values
simulationSettings.timeIntegration.numberOfSteps = int(endTime/stepSize)
simulationSettings.timeIntegration.endTime = endTime
simulationSettings.timeIntegration.verboseMode = 1

#for redundant constraints, use these two settings
simulationSettings.linearSolverSettings.ignoreSingularJacobian=True
simulationSettings.linearSolverType = exu.LinearSolverType.EigenDense #use EigenSparse for larger systems alternatively
#simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
# simulationSettings.timeIntegration.newton.useModifiedNewton = True

mbs.SolveDynamic(simulationSettings)

#%%
#TAG: visualization
#DIFFICULTY: 10000
SC.visualizationSettings.nodes.drawNodesAsPoint = False
SC.visualizationSettings.nodes.showBasis = True

mbs.SolutionViewer()


