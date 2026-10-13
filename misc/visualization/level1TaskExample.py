#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Author: Gemma4:31B
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import exudyn as exu
exu.special.solver.timeout = 120
from exudyn.utilities import *
import numpy as np

SC = exu.SystemContainer()
mbs = SC.AddSystem()

oGround = mbs.CreateGround(referencePosition=[0,0,0],
                           graphicsDataList=[graphics.CheckerBoard(normal=[0, 1, 0])])

lx = 0.3
ly = 0.1
lz = 0.1
density = 400

inertiaBody = InertiaCuboid(density=density, sideLengths=[lx, ly, lz])

oBody = mbs.CreateRigidBody(inertia=inertiaBody,
                            referencePosition=[0, 0.3, 0],
                            gravity=[0, -3.73, 0])

stiffness_val = [20000, 120000, 12000]
damping_val = [400, 4000, 200]

# Vertices at x = +/- lx/2, z = +/- lz/2
# Local y for body is -ly/2, Local y for ground is 0
vertices = [
    [ lx/2, -ly/2,  lz/2],
    [ lx/2, -ly/2, -lz/2],
    [-lx/2, -ly/2,  lz/2],
    [-lx/2, -ly/2, -lz/2]
]

for v in vertices:
    mbs.CreateCartesianSpringDamper(
        bodyNumbers=[oGround, oBody],
        localPosition0=[v[0], 0, v[2]],
        localPosition1=v,
        stiffness=stiffness_val,
        damping=damping_val,
        drawSize = 0.05,
    )


mbs.sys["EvaluteMBScode.preAssemble"]=True
mbs.Assemble()
mbs.sys["EvaluteMBScode.postAssemble"]=True

tEnd = 2
stepSize = 0.001

simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.solutionWritePeriod = 5e-2
simulationSettings.solutionSettings.sensorsWritePeriod = 5e-3
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/stepSize) #must be integer
simulationSettings.timeIntegration.endTime = tEnd

visualizationSettings = SC.visualizationSettings
visualizationSettings.nodes.drawNodesAsPoint=False
visualizationSettings.nodes.show = False
visualizationSettings.markers.show = False
visualizationSettings.openGL.lineWidth = 3
visualizationSettings.openGL.light0.shadow = 0.3
visualizationSettings.view0.scene.drawCoordinateSystem = False
visualizationSettings.general.showHelpOnStartup = False
visualizationSettings.general.showSolverInformation = False
visualizationSettings.general.showSolverTime = False

#changes Joh:
visualizationSettings.openGL.light1.enable = False
visualizationSettings.openGL.light0.position = [-2.0, 10.0, 5.0, 1.0]
visualizationSettings.openGL.light0.lightRadius = 0.5
visualizationSettings.raytracer.advanced.shadowScalingFactor = 1
visualizationSettings.raytracer.advanced.shadowSmoothingSteps = 5
visualizationSettings.raytracer.lightRadiusVariations = 42
visualizationSettings.openGL.lineWidth = 5
visualizationSettings.raytracer.multiSampling = 4
visualizationSettings.raytracer.numberOfThreads = 64
visualizationSettings.raytracer.keepWindowActive = True
visualizationSettings.raytracer.verbose = 1
visualizationSettings.openGL.lineWidth = 5
visualizationSettings.loads.drawSimplified = False

mbs.SolveDynamic(simulationSettings)
mbs.SolutionViewer()
