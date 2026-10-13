# -*- coding: utf-8 -*- 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Create a reference solution for LLM-code evaluation
#
# Model:    A planar four-bar mechanism modelled with 2 points masses
#
# Author:   Johannes Gerstmayr
# Date:     2025-03-12
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# "A planar four-bar mechanism modelled with 3 rigid bodies "
# "and ground. Each arm has density = {density} kg/m^3, a rectangular cross section "
# "with body-fixed dimensions wy = {wy} and wz = {wz}, "
# "where x is the arm's axis and the local and global z-axes match. "
# "The origin point of the mechanism on ground is located at [0,0,0]. "
# "The first arm with length l1 = {l1} is aligned with the global y-axis (=local x-axis) and has it's COM and reference point at "
# "p0 = [0,l1/2,0] and it has a revolute joint that connects to ground at origin, rotating freely around "
# "the z-axis. The second arm with length l2 = {l2} is aligned with the global x-axis with the COM "
# "at p1 = [l2/2,l1,0], and is connected to the first arm via a revolute joint at the "
# "intersection of both arms. The third arm with length l3 = {l3} has its reference point and COM "
# "at [l2,l1-l3/2,0] and it's local x-axis aligns with the global y-axis. "
# "Again there is a revolute joint between arm2 and arm3 at the intersection. "
# "At [l2,l1-l3,0] a generic joint connects the third arm to the ground, constraining only "
# "those coordinates such that no redundant constraints are introduced. "
# "Gravity g = {gravity} m/s^2 acts in negative y-direction and a torque "
# "T = {T} Nm is added at arm1 around the positive z-axis. "
# "There is no friction or other resistance.",


#%%parametersstart%%
#parameters which can be varied
density=1000
wy=0.02
wz=0.02
l1=0.5
l2=1.
l3=1.
T=5
gravity=9.81


#%%samplecodestart%%

import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics 

import numpy as np

SC = exu.SystemContainer()
mbs = SC.AddSystem()

oGround = mbs.CreateGround(referencePosition=[0,0,0])

inertiaArm1 = InertiaCuboid(density=density, sideLengths=[l1, wy, wz])
oArm1 = mbs.CreateRigidBody(inertia=inertiaArm1,
                            referencePosition=[0, l1/2, 0],
                            referenceRotationMatrix = RotationVector2RotationMatrix([0,0,0.5*np.pi]),
                            gravity=[0, -gravity, 0])

inertiaArm2 = InertiaCuboid(density=density, sideLengths=[l2, wy, wz])
oArm2 = mbs.CreateRigidBody(inertia=inertiaArm2,
                            referencePosition=[l2/2, l1, 0],
                            gravity=[0, -gravity, 0])

inertiaArm3 = InertiaCuboid(density=density, sideLengths=[l3, wy, wz])
oArm3 = mbs.CreateRigidBody(inertia=inertiaArm3,
                            referencePosition=[l2, l1 - l3/2, 0],
                            referenceRotationMatrix = RotationVector2RotationMatrix([0,0,-0.5*np.pi]),
                            gravity=[0, -gravity, 0])

mbs.CreateTorque(bodyNumber=oArm1, loadVector=[0,0,T])

mbs.CreateRevoluteJoint(bodyNumbers=[oGround, oArm1],
                        position=[0,0,0],
                        axis=[0,0,1],
                        useGlobalFrame=True,
                        axisRadius=wz*1,
                        axisLength=wy*1.2,
                        )

mbs.CreateRevoluteJoint(bodyNumbers=[oArm1, oArm2],
                        position=[0, l1, 0],
                        axis=[0,0,1],
                        useGlobalFrame=True,
                        axisRadius=wz*1,
                        axisLength=wy*1.2,
                        )

mbs.CreateRevoluteJoint(bodyNumbers=[oArm2, oArm3],
                        position=[l2, l1, 0],
                        axis=[0,0,1],
                        useGlobalFrame=True,
                        axisRadius=wz*1,
                        axisLength=wy*1.2,
                        )

mbs.CreateGenericJoint(bodyNumbers=[oArm3, oGround],
                       position=[l2, l1 - l3, 0],
                       constrainedAxes=[1, 1, 0, 0, 0, 0],
                       useGlobalFrame=True,
                       axesRadius=wz*1,
                       axesLength=wy*1.2,
                       )


#%%
mbs.Assemble()

tEnd = 1
stepSize = 0.001

simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.solutionWritePeriod = 5e-2
simulationSettings.solutionSettings.sensorsWritePeriod = 1e-2
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/stepSize) #must be integer
simulationSettings.timeIntegration.endTime = tEnd
simulationSettings.linearSolverSettings.ignoreSingularJacobian = True
simulationSettings.linearSolverType = exu.LinearSolverType.EigenDense

SC.visualizationSettings.nodes.drawNodesAsPoint=False
SC.visualizationSettings.nodes.defaultSize=0.05
SC.visualizationSettings.nodes.tiling = 32

mbs.SolveDynamic(simulationSettings)

#%%samplecodestop%%
mbs.SolutionViewer()

