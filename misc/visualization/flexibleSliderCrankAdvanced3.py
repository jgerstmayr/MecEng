# ** given model description: **
# A flexible slider-crank assembly composed of three parts: a crank with
# an oblong cross-section and a protruding main-bearing pin, an asymmetric connecting
# rod (conrod shape) with a big-end pin and a small-end bore, and a slider with
# a guide rabbet and a vertical pin on top. You already have the three needed
# machine part(s) available as a Finite Element Method (FEM) model. Load the according
# FEM interface(s) with their/its according name(s)! The description behind the
# needed machine part "crank part advanced" is: "A crank modeled as a 3D body created
# by extruding a closed 2D bone-shaped cross-section in the XY-plane along the
# z-axis by 0.03 m, with a pin added at the left end and a bore drilled at the right
# end. The crank is centered at the global origin (x=0, y=0, z=0), so the workplane
# is placed at z = -0.03/2 before extrusion. The cross-section has a total length
# of 0.12 m in the x-direction and contains four semicircular arcs of radius
# bossRadius = min(0.5*0.15, 2*0.05, 0.24*0.12). Starting at the point (-0.5*0.12, -0.5*0.05),
# the closed outline of the cross-section is drawn by: a semicircular arc bulging
# outward in the negative y-direction to (-0.5*0.12 + 2*bossRadius, -0.5*0.05), a
# straight edge to (0.5*0.12 - 2*bossRadius, -0.5*0.05), a semicircular arc bulging
# outward in the negative y-direction to (0.5*0.12, -0.5*0.05), a straight vertical
# edge to (0.5*0.12, 0.5*0.05), a semicircular arc bulging outward in the positive
# y-direction to (0.5*0.12 - 2*bossRadius, 0.5*0.05), a straight edge to (-0.5*0.12 +
# 2*bossRadius, 0.5*0.05), a semicircular arc bulging outward in the positive y-direction
# to (-0.5*0.12, 0.5*0.05), and a straight vertical edge closing the outline
# back to the start point. This forms a straight central bar of width 0.05 m with
# a rounded boss at each end; the boss centers lie on the x-axis at x = ±(0.5*0.12
# - bossRadius). After extrusion, the edges of the top face (maximum z) are
# chamfered by 0.1*0.03 m. A cylindrical pin of radius 0.015 m and height 0.03 m is
# attached at the left boss center: starting at the bottom face of the body at z =
# -0.5*0.03, it extends by 0.03 m in the negative z-direction. Its curved lateral face
# defines the interface "pinLeft" (main bearing / ground revolute joint). A cylindrical
# bore of radius 0.015 m is drilled through the full height at the right boss
# center. Its curved lateral face defines the interface "boreRight" (crank pin connection).
# The whole crank is made of steel, having a Young's modulus of 200000000000
# N/m², a poisson ratio of 0.3, and a density of 8000 kg/m³. Mesh the machine and
# give it the name "crankPartAdvanced" such that it can be loaded. Limit the mesh
# resolution by setting an appropriate maximum element size using the "maxh" parameter." The
# description behind the needed machine part "rod part advanced" is: "A connecting rod
# modeled as a 3D body created by extruding an asymmetric 2D cross-section in the
# XY-plane along the z-axis by 0.03 m, with a cylindrical pin added at the big end
# and a cylindrical bore drilled at the small end. The cross-section is only
# symmetric about the XZ-plane (Y=0), resembling a real conrod shape: a large boss
# at the crank (big) end on the left and a smaller boss at the slider (small)
# end on the right, connected by two straight tapered web edges. The boss radii
# are computed as: bigBossRadius = min(0.5*0.075, 2*0.025, 0.24*0.3), smallBossRadius
# = min(0.5*0.05, 2*0.04, 0.24*0.3). This guarantees bigBossRadius + smallBossRadius
# < 0.3 so the web has positive length. The workplane origin is placed at (0,
# 0, -0.03/2) before extrusion. The counterclockwise outline (viewed from +z)
# is constructed as follows: start at the leftmost point (xLeft, 0) where xLeft
# = -0.5*0.3 - bigBossRadius, go down to (xLeft, -0.025/2), draw an arc to
# (xLeft + 2*bigBossRadius, -0.025/2) with start tangent (0, -1) (big boss bottom
# bulge), draw a tapered line to (xRight - 2*smallBossRadius, -0.04/2) (bottom web,
# tapered), draw an arc to (xRight, -0.04/2) with start tangent (0, -1) (small boss
# bottom bulge), where xRight = +0.5*0.3 + smallBossRadius, go up to (xRight, +0.04/2),
# draw an arc to (xRight - 2*smallBossRadius, +0.04/2) with start tangent (0,
# +1) (small boss top bulge), draw a tapered line to (xLeft + 2*bigBossRadius,
# +0.025/2) (top web, tapered), draw an arc to (xLeft, +0.025/2) with start tangent
# (0, +1) (big boss top bulge), go back to (xLeft, 0) and close the wire. The
# cross-section is extruded along the z-axis by 0.03 m using occ.Prism. The top edges (at
# z = +0.03/2) are chamfered by 0.1*0.03 m. At the big end, a cylindrical pin
# of radius 0.015 m and height 0.03 m is added at (x = -0.3/2, y = 0), pointing
# in the negative z-direction from the bottom face of the body (z = -0.03/2
# downward to z = -1.5*0.03). The curved lateral face of this pin defines the interface
# "rodEndCrank" (crank connection). At the small end, a cylindrical bore of radius 0.015
# m and height 0.03 m is drilled along the z-axis at (x = +0.3/2, y = 0), running
# from z = -0.03/2 to z = +0.03/2. The curved lateral faces of this bore define
# the interface "rodEndSlider" (slider connection). The rod is made of steel,
# having a Young's modulus of 200000000000 N/m², a poisson ratio of 0.3, and a density
# of 8000 kg/m³. Mesh the machine and give it the name "rodPartAdvanced" such
# that it can be loaded. Limit the mesh resolution by setting an appropriate maximum
# element size using the "maxh" parameter." The description behind the needed machine
# part "slider part advanced" is: "A slider of a slider-crank mechanism consisting
# of an extruded slider body with a symmetric guide rabbet at the bottom and
# a vertical pin on top. The slider body is created by extruding a 2D cross-section
# in the YZ-plane along the x-direction by 0.12 m. To keep the final geometry
# centered at the global origin, the workplane for the cross-section is shifted by
# -0.5*0.12 m in x-direction before the extrusion. The overall slider width in Y-direction
# is 0.12 m and the total height in Z-direction is 0.0125 m. The outer rectangular
# limits of the cross-section therefore lie at Y = ±0.5*0.12 m and Z = +0.5*0.0125
# m. The bottom center of the rectangle is replaced by a symmetric rabbet that
# forms the guide interface. The rabbet has four corner points (all given in m):
# top-left (-0.1*0.12, -0.2*0.0125), bottom-left (-0.2*0.12, -0.5*0.0125), bottom-right
# (+0.2*0.12, -0.5*0.0125), top-right (+0.1*0.12, -0.2*0.0125). The rabbet connects to
# the outer slider boundary at (-0.5*0.12, -0.2*0.0125) and (+0.5*0.12, -0.2*0.0125).
# The boundary wire is constructed in counter-clockwise order starting at (-0.5*0.12,
# -0.2*0.0125), then through the rabbet corners, then to the right connection point, then
# to the top-right outer corner (+0.5*0.12, +0.5*0.0125), then to the top-left
# outer corner (-0.5*0.12, +0.5*0.0125), and finally closing the wire. The upper
# edges of the slider body are chamfered by 0.12*0.0125 m. A vertical cylindrical
# pin of radius 0.015 m and length 0.03 m is placed at the center of the top
# surface of the slider. The curved surface of the pin is an interface named "sliderPin".
# The bottom surface (minimum along global z-axis) of the rabbet forms the interface
# "sliderBottom" and represents the prismatic guide along the x-direction. The slider is
# made of steel, having a Young's modulus of 200000000000 N/m², a poisson ratio
# of 0.3, and a density of 8000 kg/m³. Mesh the machine and give it the name
# "sliderPartAdvanced" such that it can be loaded. Limit the mesh resolution by setting an appropriate
# maximum element size using the "maxh" parameter." The crank boss radius is crankBossRadius
# = min(0.5*0.15, 2*0.05, 0.24*0.12). The crank pin (boreRight) is located
# at x = 0.5*0.12 - crankBossRadius in the crank local frame. Joints used in
# the model: - The crank main-bearing pin interface ("pinLeft") is constrained
# to ground with a revolute about the global z-axis. - A revolute joint about
# the global z-axis connects the crank pin bore interface ("boreRight") to the
# rod big-end pin interface ("rodEndCrank"). - A revolute joint about the global
# z-axis connects the rod small-end bore interface ("rodEndSlider") to the slider
# pin interface ("sliderPin"). - A prismatic constraint along the global x-axis
# guides the slider rabbet interface ("sliderBottom"). Reference positions: - The
# crank reference frame is at the global origin. - The rod reference position is
# [0.5*0.12 - crankBossRadius + 0.5*0.3, 0, 0.03] m, aligning the rod big-end pin centroid
# (at rod-local (-0.5*0.3, 0, -0.03)) with the crank pin bore center. - The slider
# reference position is [0.5*0.12 - crankBossRadius + 0.3, 0, 0.5*0.03 - 0.5*0.0125]
# m, aligning the slider pin centroid (at slider-local (0, 0, 0.5*0.0125 + 0.5*0.03))
# with the rod small-end bore center. A torque of 0.125 Nm about the global z-axis
# acts on the crank main-bearing pin interface. Gravity is neglected.
from exudyn.FEM import *

import exudyn as exu
from exudyn.utilities import *
import numpy as np

SC = exu.SystemContainer()
mbs = SC.AddSystem()

oGround = mbs.CreateGround(graphicsDataList=[graphics.CheckerBoard(point=[0, 0, -0.1])])

crankBossRadius = min(0.5*0.15, 2*0.05, 0.24*0.12)

femCrank = FEMinterface()
femCrank.LoadFromFile('crankPartAdvanced.npz')
crankDict = mbs.CreateFFRFReducedOrderObject('crank', femCrank,
                                             referencePosition=[0., 0., 0.],
                                             initialVelocity=[0., 0., 0.],
                                             referenceRotationMatrix=np.eye(3),
                                             initialAngularVelocity=[0., 0., 0.],
                                             massProportionalDamping=0.,
                                             stiffnessProportionalDamping=0.,
                                             gravity=[0., 0., 0.])
mCrankPinLeft = crankDict['crank:pinLeft']
mCrankBoreRight = crankDict['crank:boreRight']

femRod = FEMinterface()
femRod.LoadFromFile('rodPartAdvanced.npz')
rodDict = mbs.CreateFFRFReducedOrderObject('rod', femRod,
                                           referencePosition=[0.5*0.12 - crankBossRadius + 0.5*0.3, 0., 0.03],
                                           initialVelocity=[0., 0., 0.],
                                           referenceRotationMatrix=np.eye(3),
                                           initialAngularVelocity=[0., 0., 0.],
                                           massProportionalDamping=0.,
                                           stiffnessProportionalDamping=0.,
                                           gravity=[0., 0., 0.])
mRodEndCrank = rodDict['rod:rodEndCrank']
mRodEndSlider = rodDict['rod:rodEndSlider']

femSlider = FEMinterface()
femSlider.LoadFromFile('sliderPartAdvanced.npz')
sliderDict = mbs.CreateFFRFReducedOrderObject('slider', femSlider,
                                              referencePosition=[0.5*0.12 - crankBossRadius + 0.3, 0., 0.5*0.03 - 0.5*0.0125],
                                              initialVelocity=[0., 0., 0.],
                                              referenceRotationMatrix=np.eye(3),
                                              initialAngularVelocity=[0., 0., 0.],
                                              massProportionalDamping=0.,
                                              stiffnessProportionalDamping=0.,
                                              gravity=[0., 0., 0.])
mSliderPin = sliderDict['slider:sliderPin']
mSliderBottom = sliderDict['slider:sliderBottom']

mbs.CreateRevoluteJoint(bodyNumbers=[oGround, mCrankPinLeft],
                        axis=[0, 0, 1],
                        useGlobalFrame=True)

mbs.CreateRevoluteJoint(bodyNumbers=[mCrankBoreRight, mRodEndCrank],
                        axis=[0, 0, 1],
                        useGlobalFrame=True)

mbs.CreateRevoluteJoint(bodyNumbers=[mRodEndSlider, mSliderPin],
                        axis=[0, 0, 1],
                        useGlobalFrame=True)

mbs.CreatePrismaticJoint(bodyNumbers=[oGround, mSliderBottom],
                         axis=[1, 0, 0],
                         useGlobalFrame=True)

mbs.AddLoad(Torque(markerNumber=mCrankPinLeft, loadVector=[0, 0, 0.125]))

mbs.sys["EvaluteMBScode.preAssemble"]=True
mbs.Assemble()
mbs.sys["EvaluteMBScode.postAssemble"]=True

tEnd = 1.5
h = 0.01


simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.solutionWritePeriod = 1e-3
simulationSettings.solutionSettings.sensorsWritePeriod  = 1e-3
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/h)
simulationSettings.timeIntegration.endTime = tEnd

visualizationSettings = SC.visualizationSettings
visualizationSettings.nodes.drawNodesAsPoint=False
visualizationSettings.nodes.show = False
visualizationSettings.markers.show = False
visualizationSettings.openGL.lineWidth = 3
visualizationSettings.openGL.light0.position = [6.0, 0, 10.0, 1.0]
visualizationSettings.openGL.light1.position = [6.0, 0, 5.0, 0.0]
visualizationSettings.openGL.light2.enable = True
visualizationSettings.openGL.light2.position = [2.0, 2.0, 1.0, 0.0]
visualizationSettings.openGL.light0.shadow = 0.3
visualizationSettings.openGL.light1.shadow = 0
visualizationSettings.view0.scene.drawCoordinateSystem = False
visualizationSettings.general.showHelpOnStartup = False
visualizationSettings.general.showSolverInformation = False
visualizationSettings.general.showSolverTime = False
visualizationSettings.view0.scene.showMeshEdges = False
visualizationSettings.contour.automaticRange = True
visualizationSettings.contour.outputVariableComponent = -1
visualizationSettings.loads.defaultSize = 0.2
visualizationSettings.connectors.show = False
visualizationSettings.loads.drawSimplified = False
visualizationSettings.loads.defaultRadius = 0.002
visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal
# visualizationSettings.raytracer.multiSampling = 3

#changes Joh:
#visualizationSettings.view0.camera.useRaytracer = True #switch with CTRL-R
# visualizationSettings.openGL.light1.enable = False
# visualizationSettings.openGL.light0.position = [-2.0, -10.0, 10, 1.0]
# visualizationSettings.openGL.light0.lightRadius = 0.5
# visualizationSettings.raytracer.advanced.shadowScalingFactor = 1
visualizationSettings.raytracer.advanced.shadowSmoothingSteps = 5
visualizationSettings.raytracer.lightRadiusVariations = 42
# visualizationSettings.openGL.lineWidth = 5
#SC.renderer.SetModelView(zoom=0.2593967,rotationVector=[0,0,0],centerPoint=[0.1790107,0.03445112,0])
visualizationSettings.raytracer.multiSampling = 4
visualizationSettings.raytracer.numberOfThreads = 64
visualizationSettings.raytracer.keepWindowActive = True
visualizationSettings.raytracer.verbose = 1
# visualizationSettings.openGL.lineWidth = 5
# visualizationSettings.loads.drawSimplified = False

mbs.SolveDynamic(simulationSettings)

mbs.SolutionViewer()

