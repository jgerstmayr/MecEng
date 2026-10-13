# -*- coding: utf-8 -*- 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Create a reference solution for LLM-code evaluation
#
# Model:    Sphere-Sphere and Sphere-Ground Impact
#
# Author:   Johannes Gerstmayr
# Date:     2026-04-06
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%parametersstart%%
#parameters which can be varied

mass1 = 1.5
mass2 = 1.5
radius1 = 0.1
radius2 = 0.1
distance = 0.5
velocity = 2
stiffness = 1e5
restitution = 0.8

#%%samplecodestart%%

import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics 

import numpy as np

SC = exu.SystemContainer()
mbs = SC.AddSystem()

gravity = 0

inertiaSphere1 = InertiaSphere(mass=mass1, radius=radius1)
oBody1 = mbs.CreateRigidBody(inertia=inertiaSphere1,
                             referencePosition=[0,0,0],
                             gravity=[0,0,-gravity])

inertiaSphere2 = InertiaSphere(mass=mass2, radius=radius2)
oBody2 = mbs.CreateRigidBody(inertia=inertiaSphere2,
                             referencePosition=[distance,0,0],
                             initialVelocity=[-velocity,0,0],
                             gravity=[0,0,-gravity])

mbs.CreateSphereSphereContact(bodyNumbers=[oBody1, oBody2],
                              spheresRadii=[radius1, radius2],
                              contactStiffness=stiffness,
                              dynamicFriction=0,
                              impactModel=2,
                              restitutionCoefficient=restitution)

#%%
mbs.Assemble()

tEnd = 1
stepSize = 0.001

simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.solutionWritePeriod = 5e-2
simulationSettings.solutionSettings.sensorsWritePeriod = 5e-3
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/stepSize) #must be integer
simulationSettings.timeIntegration.endTime = tEnd

SC.visualizationSettings.nodes.drawNodesAsPoint=False
SC.visualizationSettings.nodes.defaultSize=1
SC.visualizationSettings.openGL.lineWidth = 2
SC.visualizationSettings.openGL.light0.shadow = 0.2

SC.renderer.Start()

mbs.SolveDynamic(simulationSettings)

SC.renderer.DoIdleTasks()
SC.renderer.Stop()

#%%samplecodestop%%
mbs.SolutionViewer()

