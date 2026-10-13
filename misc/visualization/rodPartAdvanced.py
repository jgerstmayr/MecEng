# ** given model description: **
# A connecting rod modeled as a 3D body created by extruding an asymmetric
# 2D cross-section in the XY-plane along the z-axis by 0.03 m, with a cylindrical
# pin added at the big end and a cylindrical bore drilled at the small end. The
# cross-section is only symmetric about the XZ-plane (Y=0), resembling a real conrod shape:
# a large boss at the crank (big) end on the left and a smaller boss at the
# slider (small) end on the right, connected by two straight tapered web edges.
# The boss radii are computed as: bigBossRadius = min(0.5*0.075, 2*0.025, 0.24*0.3),
# smallBossRadius = min(0.5*0.05, 2*0.04, 0.24*0.3). This guarantees bigBossRadius + smallBossRadius
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
# element size using the "maxh" parameter.
import sys
from pathlib import Path
if "__file__" in globals():
    sys.path.append(str(Path(Path(__file__).resolve().parents[2],"src")))

import exudyn as exu
from exudyn.FEM import *
import ngsolve as ngs
import netgen.occ as occ
try:
    from ExuAI.flexUtilities import GetMeshParametersDict, MeshSafetyCheck, MakeFFRFObjectAndAnimateModes
except:
    pass

E = 2.0e11
nu = 0.3
rho = 8000
materials = {'steel': {'youngsModulus': E,
                       'poissonsRatio': nu,
                       'density': rho}}
interfaceNameList = []

thickness = 0.03
webLength = 0.3
bigBossRadius = min(0.5*0.075, 2*0.025, 0.24*0.3)
smallBossRadius = min(0.5*0.05, 2*0.04, 0.24*0.3)
bigWidth = 0.025
smallWidth = 0.04

xLeft = -0.5*webLength - bigBossRadius
xRight = +0.5*webLength + smallBossRadius

wp = occ.WorkPlane(occ.Axes((0, 0, -thickness/2),
                            n=(0, 0, 1),
                            h=(1, 0, 0)))
wp.MoveTo(xLeft, 0.0)
wp.LineTo(xLeft, -bigWidth/2) \
  .ArcTo(xLeft + 2*bigBossRadius, -bigWidth/2, occ.gp_Vec2d(0, -1)) \
  .LineTo(xRight - 2*smallBossRadius, -smallWidth/2) \
  .ArcTo(xRight, -smallWidth/2, occ.gp_Vec2d(0, -1)) \
  .LineTo(xRight, +smallWidth/2) \
  .ArcTo(xRight - 2*smallBossRadius, +smallWidth/2, occ.gp_Vec2d(0, +1)) \
  .LineTo(xLeft + 2*bigBossRadius, +bigWidth/2) \
  .ArcTo(xLeft, +bigWidth/2, occ.gp_Vec2d(0, +1)) \
  .LineTo(xLeft, 0.0) \
  .Close()
baseFace = wp.Face()
rodBody = occ.Prism(baseFace, occ.gp_Vec(0, 0, thickness))

chamferSize = 0.1*thickness
rodBody = rodBody.MakeChamfer(rodBody.faces.Max((0, 0, 1)).edges, chamferSize)

pinFootPoint = (-webLength/2, 0, -thickness/2)
pinAxis = (0, 0, -1)
pinRadius = 0.015
pinHeight = thickness
pin = occ.Cylinder(pinFootPoint, pinAxis, pinRadius, pinHeight)
interfaceName = 'rodEndCrank'
pin.faces[0].name = interfaceName
interfaceNameList += [interfaceName]

boreFootPoint = (+webLength/2, 0, -thickness/2)
boreAxis = (0, 0, 1)
boreRadius = 0.015
boreHeight = thickness
bore = occ.Cylinder(boreFootPoint, boreAxis, boreRadius, boreHeight)
bore.faces[0].name = 'rodEndSlider'
interfaceNameList += ['rodEndSlider']

geometry = rodBody + pin - bore
geometry.name = 'steel'
geo = occ.OCCGeometry(geometry)
from ExuAI.flexUtilities import GetMeshParametersDict, MeshSafetyCheck
meshParametersDict=GetMeshParametersDict(geo, maxHLLM=0.01, meshOrderLLM=2)
geoMesh = geo.GenerateMesh(maxh=meshParametersDict['maxH'], grading=meshParametersDict['grading'], curvaturesafety=meshParametersDict['curvatureSafety'])
mesh = ngs.Mesh(geoMesh)
meshOrder=meshParametersDict['meshOrder']
fem = FEMinterface()
exu.sys['exuAI_out']['materials'] = materials
[bfM, bfK, fes] = fem.ImportMeshFromNGsolve(mesh, materials=materials,
                                            boundaryNamesList=interfaceNameList, meshOrder=meshOrder)
[boundaryNodesList, boundaryWeightsList] = fem.GetBoundaryNodeSetsAsLists()
MeshSafetyCheck(fem, meshParametersDict['maxNumberOfFemNodes'])
fem.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryNodesList,
                                  nEigenModes=6, boundaryNodesWeights=boundaryWeightsList)
fem.metaData['name'] = 'rodPartAdvanced'


exu.sys['exuAI_out']['femObject'] = fem
MakeFFRFObjectAndAnimateModes(fem, partName=fem.metaData['name'], animateModes=True)