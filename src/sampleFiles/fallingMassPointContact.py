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

mass = 2
radius = 0.04
height = 2.0
velocity = -2.0
stiffness = 2e5
restitution = 0.8
gravity = 9.81

#%%samplecodestart%%

import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics 

import numpy as np

SC = exu.SystemContainer()
mbs = SC.AddSystem()

size = 1
oGround = mbs.CreateGround(referencePosition=[0,0,0],
                           graphicsDataList=[graphics.CheckerBoard(point=[0,0,0],size=size)])

oMass = mbs.CreateMassPoint(physicsMass=mass,
                             referencePosition=[0,0,height],
                             initialVelocity=[0,0,velocity],
                             gravity=[0,0,-gravity],
                             # drawSize=radius*2,
                             graphicsDataList=[graphics.Sphere(radius=radius, color=graphics.color.dodgerblue)])

#define list of four 3D points counter-clockwise seen from contacting object:
quadPoints = [[-0.5*size,-0.5*size,0],[0.5*size,-0.5*size,0],
              [0.5*size,0.5*size,0],[-0.5*size,0.5*size,0]]
#contact between spherical body and quad plane
oSSC = mbs.CreateSphereQuadContact(bodyNumbers=[oMass, oGround],
                                   quadPoints=quadPoints,
                                   radiusSphere=radius,
                                   contactStiffness = stiffness,
                                   dynamicFriction = 0,
                                   impactModel = 2,
                                   restitutionCoefficient = restitution,
                                   )

# s0 = mbs.AddSensor(SensorBody(bodyNumber=oMass,storeInternal=True,
#                          outputVariableType=exu.OutputVariableType.Position))

#%%
mbs.Assemble()

tEnd = 1
stepSize = 0.0001

simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.solutionWritePeriod = 5e-2
simulationSettings.solutionSettings.sensorsWritePeriod = 5e-2
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/stepSize) #must be integer
simulationSettings.timeIntegration.endTime = tEnd

SC.visualizationSettings.nodes.drawNodesAsPoint=False
SC.visualizationSettings.nodes.defaultSize=1
SC.visualizationSettings.nodes.tiling=16
SC.visualizationSettings.nodes.showBasis = True
SC.visualizationSettings.openGL.lineWidth = 2
SC.visualizationSettings.openGL.light0.shadow = 0.2

SC.renderer.Start()

mbs.SolveDynamic(simulationSettings)

SC.renderer.DoIdleTasks()
SC.renderer.Stop()

#%%samplecodestop%%
mbs.SolutionViewer()

#mbs.PlotSensor(s0,[2])
