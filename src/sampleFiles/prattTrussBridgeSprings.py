# -*- coding: utf-8 -*- 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Create a reference solution for LLM-code evaluation
#
# Model:   Pratt-truss bridge with springs frame with variable number of segments
#
# Author:   Johannes Gerstmayr
# Date:     2026-04-06
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%parametersstart%%
#parameters which can be varied

nSegments = 2
segLength = 2
segHeight = 1.5
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

# Derived geometry
# Total horizontal panels = 2 (triangles) + 2*nSegments (rectangles)
nTotalPanels = 2 * nSegments + 2
totalLength = nTotalPanels * segLength

SC = exu.SystemContainer()
mbs = SC.AddSystem()

# Ground reference
oGround = mbs.CreateGround(referencePosition=[0,0,0])

# --- Node Creation ---
nodes_lower = []
nodes_upper = [] # The upper chord is shorter than the lower chord in a Pratt bridge

# Lower nodes: from x=0 to x=totalLength
for i in range(nTotalPanels + 1):
    x = i * segLength
    node = mbs.CreateMassPoint(physicsMass=1.0, referencePosition=[x,0,0], gravity=[0,-gravity,0])
    nodes_lower.append(node)

# Upper nodes: only above the rectangular sections (offsets by 1 segment from ends)
for i in range(1, nTotalPanels):
    x = i * segLength
    node = mbs.CreateMassPoint(physicsMass=1.0, referencePosition=[x,segHeight,0], gravity=[0,-gravity,0])
    nodes_upper.append(node)

# --- Member Helper ---
member_contributions = {node: [] for node in nodes_lower + nodes_upper}

def add_member(nodeIdx0, nodeIdx1, nodeList0, nodeList1):
    global rhoA, length, stiffness, damping, mbs, np
    n0 = nodeList0[nodeIdx0]
    n1 = nodeList1[nodeIdx1]
    
    # Calculate length from reference positions
    nn0 = mbs.GetObject(n0)['nodeNumber']
    nn1 = mbs.GetObject(n1)['nodeNumber']
    pos0 = mbs.GetNodeParameter(nn0, 'referenceCoordinates')
    pos1 = mbs.GetNodeParameter(nn1, 'referenceCoordinates')
    length = np.linalg.norm(np.array(pos0) - np.array(pos1))
    
    mbs.CreateSpringDamper(
        bodyNumbers=[n0, n1],
        stiffness=stiffness,
        damping=damping,
        drawSize=0.01,
    )
    # Distribute mass
    m_half = 0.5 * rhoA * length
    member_contributions[n0].append(m_half)
    member_contributions[n1].append(m_half)

# --- Structural Assembly ---

# 1. Lower Chord (Horizontal)
for i in range(nTotalPanels):
    add_member(i, i+1, nodes_lower, nodes_lower)

# 2. Upper Chord (Horizontal)
for i in range(len(nodes_upper) - 1):
    add_member(i, i+1, nodes_upper, nodes_upper)

# 3. Verticals
# Connect upper nodes to lower nodes (indices in upper are shifted by 1 relative to lower)
for i in range(len(nodes_upper)):
    add_member(i, i+1, nodes_upper, nodes_lower)

# 4. Diagonals (The "Pratt" Logic)
# Center of the bridge is at index (nTotalPanels / 2) in nodes_lower
center_idx = nTotalPanels // 2

for i in range(len(nodes_upper)):
    # i is index in nodes_upper, which corresponds to lower index i+1
    lower_idx = i + 1
    
    if lower_idx < center_idx:
        # Left side: Top-Outer to Bottom-Inner (e.g., node_upper[0] to node_lower[2])
        add_member(i, lower_idx + 1, nodes_upper, nodes_lower)
    elif lower_idx > center_idx:
        # Right side: Top-Outer to Bottom-Inner (e.g., node_upper[last] to node_lower[last-1])
        add_member(i, lower_idx - 1, nodes_upper, nodes_lower)
    # Note: If lower_idx == center_idx, usually no diagonal or a specialized one is added.

# 5. End Diagonals (Triangular Panels)
add_member(0, 0, nodes_upper, nodes_lower) # Left end
add_member(len(nodes_upper)-1, nTotalPanels, nodes_upper, nodes_lower) # Right end

# --- Finalize Nodes ---

# Assign computed masses
for node, mass_list in member_contributions.items():
    mbs.SetObjectParameter(node, 'physicsMass', sum(mass_list))

# Support Conditions
# Left-most bottom: Fixed x and y
mbs.CreateCoordinateConstraint(bodyNumbers=[None, nodes_lower[0]], coordinates=[0, 0]) # x
mbs.CreateCoordinateConstraint(bodyNumbers=[None, nodes_lower[0]], coordinates=[1, 1]) # y

# Right-most bottom: Roller (Fixed y only)
mbs.CreateCoordinateConstraint(bodyNumbers=[None, nodes_lower[-1]], coordinates=[1, 1]) # y

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

