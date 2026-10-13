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

advancedGraphics = True # adds shadow for distance constraints

m = 1
r = 0.08
L = 0.8
z_ceiling = 0.8
g_val = 11.15
k_stiff = 800000
e_restitution = 1.0
dist_between_centers = 2*r + 0.002

# function to add cylinders for constraints (only visualization!)
def GraphicsDataUserFunction(mbs, itemNumber):
    graphicsDataList = []
    for i in range(4):
        positionFixed = np.array([i * dist_between_centers, 0, z_ceiling])
        position = mbs.GetSensorValues(i)
        # print(position)
        graphicsDataList += [graphics.Cylinder(pAxis=position, vAxis=positionFixed-position, radius=0.01, color=graphics.color.blue)] 
    return graphicsDataList

oGround = mbs.CreateGround(referencePosition=[0,0,0],
                           graphicsDataList=[
                                           # graphics.Brick(centerPoint=[(2*0.08 + 0.002)*1.5, 0, 0.8],
                                           #                  size=[(2*0.08 + 0.002)*3+2, 0.1, 0.1],
                                           #                  color=graphics.color.grey),
                                             graphics.Cylinder(pAxis=[-0.1, 0, 0.8], vAxis=[(2*0.08 + 0.002)*3+(0.1*2), 0, 0],
                                                               color=graphics.color.lightgrey, radius=0.03),
                                             graphics.CheckerBoard(point=[0, 0, -0.2], size=3)],
                           graphicsDataUserFunction=GraphicsDataUserFunction if advancedGraphics else 0
                           )

spheres = []
for i in range(4):
    # Calculate ceiling attachment point
    p_fix = [i * dist_between_centers, 0, z_ceiling]

    # Calculate equilibrium position
    p_eq = [i * dist_between_centers, 0, 0]

    # Special initial position for sphere 0
    if i == 0:
        ref_pos = [-0.8, 0, 0.8]
        init_disp = [0, 0, 0]
    else:
        ref_pos = p_eq
        init_disp = [0, 0, 0]

    oMass = mbs.CreateMassPoint(physicsMass=m, 
                                referencePosition=ref_pos, 
                                initialDisplacement=init_disp, 
                                initialVelocity=[0,0,0], 
                                gravity=[0,0,-g_val],
                                graphicsDataList=[graphics.Sphere(radius=r, color=graphics.color.lawngreen, 
                                                                  nTiles=24)])
    mbs.AddSensor(SensorBody(bodyNumber=oMass, outputVariableType=exu.OutputVariableType.Position, storeInternal=True))
    spheres.append(oMass)

    # Distance constraint to ceiling
    mbs.CreateDistanceConstraint(bodyNumbers=[oGround, oMass], 
                                 localPosition0=p_fix, 
                                 localPosition1=[0,0,0], 
                                 distance=None)

# Contacts between adjacent spheres
for i in range(3):
    mbs.CreateSphereSphereContact(bodyNumbers=[spheres[i], spheres[i+1]], 
                                  spheresRadii=[r, r], 
                                  contactStiffness=k_stiff, 
                                  dynamicFriction=0, 
                                  impactModel=2, 
                                  restitutionCoefficient=e_restitution)


mbs.sys["EvaluteMBScode.preAssemble"]=True
mbs.Assemble()
mbs.sys["EvaluteMBScode.postAssemble"]=True

tEnd = 5
stepSize = 0.0001

simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.solutionWritePeriod = 5e-2
simulationSettings.solutionSettings.sensorsWritePeriod = 5e-3
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/stepSize) #must be integer
simulationSettings.timeIntegration.endTime = tEnd
simulationSettings.timeIntegration.discontinuous.maxIterations = 10
simulationSettings.timeIntegration.newton.useModifiedNewton = True
simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
simulationSettings.timeIntegration.simulateInRealtime = True
simulationSettings.timeIntegration.realtimeFactor = 0.1 # slow down to find good state of spheres

visualizationSettings = SC.visualizationSettings
visualizationSettings.nodes.drawNodesAsPoint=False
visualizationSettings.nodes.show = False
visualizationSettings.markers.show = False
visualizationSettings.openGL.lineWidth = 3
visualizationSettings.openGL.light0.shadow = 0.3
visualizationSettings.view0.scene.drawCoordinateSystem = False
visualizationSettings.view0.camera.useRaytracer = True
visualizationSettings.general.showHelpOnStartup = False
visualizationSettings.general.showSolverInformation = False
visualizationSettings.general.showSolverTime = False

#changes Joh:
#visualizationSettings.view0.camera.useRaytracer = True #switch with CTRL-R
visualizationSettings.openGL.light1.enable = False
visualizationSettings.openGL.light0.position = [-2.0, -10.0, 10, 1.0]
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

#start solver / use renderer with GraphicsDataUserFunction since not shown in solution viewer
if advancedGraphics:
    SC.renderer.Start()
    SC.renderer.DoIdleTasks()
mbs.SolveDynamic(simulationSettings)
if advancedGraphics:
    SC.renderer.DoIdleTasks()
    SC.renderer.Stop()

mbs.SolutionViewer()
