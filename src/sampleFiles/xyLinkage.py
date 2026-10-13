# -*- coding: utf-8 -*- 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Create a reference solution for LLM-code evaluation
#
# Model:    two mass points connected by distance constraints; each mass point can only move along 
#           an axis (x/y) or (x/z)
#
# Author:   Johannes Gerstmayr
# Date:     2025-04-07
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# "description": "A system consisting of two mass points connected by a distance constraint. The first 
# mass m1 = {mass1} kg is initially located at x0 = {x0} m (other coordinates 0) and can move only along 
# the x-axis, using a spherical joint where the x-axis is free. The second mass m2 = {mass2} kg is initially located at {axis}0 = {y0} m (other coordinates 0)
# and can move only along the {axis}-axis, where the {axis}-axis is free. A distance constraint is added between the two mass points and 
# the length shall be computed automatically. A spring with stiffness = {stiffness} N/m is used to connect 
# mass m1 and ground at [0,0,0]. Gravity g = {gravity} m/s^2 acts in positive {gAxis}-direction, and no 
# further forces or daming are applied.",

#%%parametersstart%%
#parameters which can be varied

mass1 = 2
mass2 = 3
x0 = 0.5
y0 = 0.5
stiffness = 500
axis = "y"
gAxis = "x"
gravity = 9.81

#%%samplecodestart%%

import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics 

import numpy as np

SC = exu.SystemContainer()
mbs = SC.AddSystem()

axisDict = {'x':0,'y':1,'z':2}

dirG = np.array([0,0,0])
dirG[axisDict[gAxis]] = 1

dirM2 = np.array([0,0,0])
dirM2[axisDict[axis]] = 1

oGround = mbs.CreateGround()
oMassPoint1 = mbs.CreateMassPoint(referencePosition=[x0, 0, 0],
                                 physicsMass=mass1,
                                 gravity=gravity*dirG)

oMassPoint2 = mbs.CreateMassPoint(referencePosition=y0*dirM2,
                                 physicsMass=mass2,
                                 gravity=gravity*dirG)

mbs.CreateCoordinateConstraint(bodyNumbers=[None, oMassPoint1],
                               coordinates=[None, 1]) #fix y-coordinate of mass point to ground
mbs.CreateCoordinateConstraint(bodyNumbers=[None, oMassPoint1],
                               coordinates=[None, 2]) #fix z-coordinate of mass point to ground

mbs.CreateCoordinateConstraint(bodyNumbers=[None, oMassPoint2],
                               coordinates=[None, 0]) #fix x-coordinate of mass point to ground
mbs.CreateCoordinateConstraint(bodyNumbers=[None, oMassPoint2],
                               coordinates=[None, 1*(axis=="z")+2*(axis=="y")]) #fix y or z-coordinate of mass point to ground



mbs.CreateDistanceConstraint(bodyNumbers=[oMassPoint1,oMassPoint2])

mbs.CreateSpringDamper(bodyNumbers=[oGround,oMassPoint1],
                       stiffness=stiffness)
    
#add sensors:
sPos = mbs.AddSensor(SensorBody(bodyNumber=oMassPoint1, storeInternal=True,
                                outputVariableType=exu.OutputVariableType.Position))

#%%
mbs.Assemble()

tEnd = 2
stepSize = 0.001

simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.solutionWritePeriod = 1e-1
simulationSettings.solutionSettings.sensorsWritePeriod = 1e-2
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/stepSize) #must be integer
simulationSettings.timeIntegration.endTime = tEnd

SC.visualizationSettings.nodes.drawNodesAsPoint=False
SC.visualizationSettings.nodes.defaultSize=0.05
SC.visualizationSettings.nodes.tiling = 32

#start solver:
mbs.SolveDynamic(simulationSettings)

#%%samplecodestop%%
mbs.PlotSensor(sPos,components=[2])
mbs.SolutionViewer()


