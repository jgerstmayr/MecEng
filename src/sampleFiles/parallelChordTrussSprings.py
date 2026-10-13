# -*- coding: utf-8 -*- 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Create a reference solution for LLM-code evaluation
#
# Model:    Parallel-chord truss frame with springs with variable number of segments
#
# Author:   Johannes Gerstmayr
# Date:     2026-04-06
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%parametersstart%%
#parameters which can be varied

nSegments = 3
segLength = 1.0
segHeight = 0.8
stiffness = 1e5
damping = 100
rhoA = 2.0
gravity = 9.81


#%%samplecodestart%%

import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics 

import numpy as np

SC = exu.SystemContainer()
mbs = SC.AddSystem()

# Ground reference
oGround = mbs.CreateGround(referencePosition=[0,0,0])

# Node storage
nodes_lower = []
nodes_upper = []

# Create nodes
for i in range(nSegments+1):
    x = i*segLength
    # Lower node
    node_lower = mbs.CreateMassPoint(
        physicsMass=0.0, #dummy
        referencePosition=[x,0,0],
        gravity=[0,-gravity,0])
    nodes_lower.append(node_lower)
    # Upper node
    node_upper = mbs.CreateMassPoint(
        physicsMass=0.0, #dummy
        referencePosition=[x,segHeight,0],
        gravity=[0,-gravity,0])
    nodes_upper.append(node_upper)

# Helper to compute member mass
def member_mass(length, rhoA):
    return rhoA * length

# Collect members for mass calculation
member_contributions = {node: [] for node in nodes_lower+nodes_upper}

drawSize = 0.005
# Horizontal members
for i in range(nSegments):
    # Bottom
    m = mbs.CreateSpringDamper(
        bodyNumbers=[nodes_lower[i], nodes_lower[i+1]],
        localPosition0=[0,0,0],
        localPosition1=[0,0,0],
        referenceLength=None,
        stiffness=stiffness,
        damping=damping,
        drawSize=drawSize,
        )        
    member_contributions[nodes_lower[i]].append(0.5*member_mass(segLength, rhoA))
    member_contributions[nodes_lower[i+1]].append(0.5*member_mass(segLength, rhoA))
    # Top
    m = mbs.CreateSpringDamper(
        bodyNumbers=[nodes_upper[i], nodes_upper[i+1]],
        localPosition0=[0,0,0],
        localPosition1=[0,0,0],
        referenceLength=None,
        stiffness=stiffness,
        damping=damping,
        drawSize=drawSize,
        )
    member_contributions[nodes_upper[i]].append(0.5*member_mass(segLength, rhoA))
    member_contributions[nodes_upper[i+1]].append(0.5*member_mass(segLength, rhoA))

# Vertical members
for i in range(nSegments+1):
    m = mbs.CreateSpringDamper(
        bodyNumbers=[nodes_lower[i], nodes_upper[i]],
        localPosition0=[0,0,0],
        localPosition1=[0,0,0],
        referenceLength=None,
        stiffness=stiffness,
        damping=damping,
        drawSize=drawSize,
        )        
    member_contributions[nodes_lower[i]].append(0.5*member_mass(segHeight, rhoA))
    member_contributions[nodes_upper[i]].append(0.5*member_mass(segHeight, rhoA))

# Diagonal members
for i in range(nSegments):
    m = mbs.CreateSpringDamper(
        bodyNumbers=[nodes_upper[i], nodes_lower[i+1]],
        localPosition0=[0,0,0],
        localPosition1=[0,0,0],
        referenceLength=None,
        stiffness=stiffness,
        damping=damping,
        drawSize=drawSize,
        )        
    diag_len = np.sqrt(segLength**2 + segHeight**2)
    member_contributions[nodes_upper[i]].append(0.5*member_mass(diag_len, rhoA))
    member_contributions[nodes_lower[i+1]].append(0.5*member_mass(diag_len, rhoA))

# Assign masses to nodes
for objectNum in nodes_lower+nodes_upper:
    total_mass = sum(member_contributions[objectNum])
    # Update physicsMass
    mbs.SetObjectParameter(objectNum, 'physicsMass', total_mass)

# Fix nodes at x=0 to ground (both x and y)
fixed_nodes = [nodes_lower[0], nodes_upper[0]]
for fn in fixed_nodes:
    mbs.CreateCoordinateConstraint(
        bodyNumbers=[None, fn],
        coordinates=[None, 0])  # x fixed
    mbs.CreateCoordinateConstraint(
        bodyNumbers=[None, fn],
        coordinates=[None, 1])  # y fixed


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
SC.visualizationSettings.nodes.defaultSize=0.1
SC.visualizationSettings.nodes.showBasis = True
SC.visualizationSettings.openGL.lineWidth = 2
SC.visualizationSettings.openGL.light0.shadow = 0.2

SC.renderer.Start()

mbs.SolveDynamic(simulationSettings)

SC.renderer.DoIdleTasks()
SC.renderer.Stop()

#%%samplecodestop%%
mbs.SolutionViewer()


