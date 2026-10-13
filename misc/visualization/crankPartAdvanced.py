# ** given model description: **
# A crank modeled as a 3D body created by extruding a closed 2D bone-shaped
# cross-section in the XY-plane along the z-axis by 0.03 m, with a pin added at the left
# end and a bore drilled at the right end. The crank is centered at the global
# origin (x=0, y=0, z=0), so the workplane is placed at z = -0.03/2 before extrusion.
# The cross-section has a total length of 0.12 m in the x-direction and contains
# four semicircular arcs of radius bossRadius = min(0.5*0.15, 2*0.05, 0.24*0.12).
# Starting at the point (-0.5*0.12, -0.5*0.05), the closed outline of the cross-section
# is drawn by: a semicircular arc bulging outward in the negative y-direction
# to (-0.5*0.12 + 2*bossRadius, -0.5*0.05), a straight edge to (0.5*0.12 - 2*bossRadius,
# -0.5*0.05), a semicircular arc bulging outward in the negative y-direction to (0.5*0.12,
# -0.5*0.05), a straight vertical edge to (0.5*0.12, 0.5*0.05), a semicircular arc bulging
# outward in the positive y-direction to (0.5*0.12 - 2*bossRadius, 0.5*0.05), a straight
# edge to (-0.5*0.12 + 2*bossRadius, 0.5*0.05), a semicircular arc bulging outward
# in the positive y-direction to (-0.5*0.12, 0.5*0.05), and a straight vertical
# edge closing the outline back to the start point. This forms a straight central
# bar of width 0.05 m with a rounded boss at each end; the boss centers lie on
# the x-axis at x = ±(0.5*0.12 - bossRadius). After extrusion, the edges of the
# top face (maximum z) are chamfered by 0.1*0.03 m. A cylindrical pin of radius
# 0.015 m and height 0.03 m is attached at the left boss center: starting at the
# bottom face of the body at z = -0.5*0.03, it extends by 0.03 m in the negative
# z-direction. Its curved lateral face defines the interface "pinLeft" (main bearing /
# ground revolute joint). A cylindrical bore of radius 0.015 m is drilled through
# the full height at the right boss center. Its curved lateral face defines the
# interface "boreRight" (crank pin connection). The whole crank is made of steel, having
# a Young's modulus of 200000000000 N/m², a poisson ratio of 0.3, and a density
# of 8000 kg/m³. Mesh the machine and give it the name "crankPartAdvanced" such
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

crankLength = 0.12
crankWidth = 0.05
crankHeight = 0.03
bossRadius = min(0.5*0.15, 2*0.05, 0.24*0.12)

wp = occ.WorkPlane(occ.Axes((0, 0, -crankHeight/2),
                            n=(0, 0, 1),
                            h=(1, 0, 0)))
wp.MoveTo(-0.5*crankLength, -0.5*crankWidth)
wp.ArcTo(-0.5*crankLength + 2*bossRadius, -0.5*crankWidth, occ.gp_Vec2d(0, -1)) \
  .LineTo(0.5*crankLength - 2*bossRadius, -0.5*crankWidth) \
  .ArcTo(0.5*crankLength, -0.5*crankWidth, occ.gp_Vec2d(0, -1)) \
  .LineTo(0.5*crankLength, 0.5*crankWidth) \
  .ArcTo(0.5*crankLength - 2*bossRadius, 0.5*crankWidth, occ.gp_Vec2d(0, 1)) \
  .LineTo(-0.5*crankLength + 2*bossRadius, 0.5*crankWidth) \
  .ArcTo(-0.5*crankLength, 0.5*crankWidth, occ.gp_Vec2d(0, 1)) \
  .LineTo(-0.5*crankLength, -0.5*crankWidth) \
  .Close()
baseFace = wp.Face()
crank = occ.Prism(baseFace, occ.gp_Vec(0, 0, crankHeight))

chamferSize = 0.1*crankHeight
crank = crank.MakeChamfer(crank.faces.Max((0, 0, 1)).edges, chamferSize)

leftBossX = -(0.5*crankLength - bossRadius)
rightBossX = (0.5*crankLength - bossRadius)

pinFootPoint = (leftBossX, 0, -0.5*crankHeight)
pinAxis = (0, 0, -1)
pinRadius = 0.015
pinHeight = 0.03
pin = occ.Cylinder(pinFootPoint, pinAxis, pinRadius, pinHeight)
interfaceName = 'pinLeft'
pin.faces[0].name = interfaceName
interfaceNameList += [interfaceName]

boreFootPoint = (rightBossX, 0, -0.5*crankHeight)
boreAxis = (0, 0, 1)
boreRadius = 0.015
boreHeight = crankHeight
bore = occ.Cylinder(boreFootPoint, boreAxis, boreRadius, boreHeight)
interfaceName = 'boreRight'
bore.faces[0].name = interfaceName
interfaceNameList += [interfaceName]

geometry = crank + pin - bore
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
fem.metaData['name'] = 'crankPartAdvanced'


exu.sys['exuAI_out']['femObject'] = fem
MakeFFRFObjectAndAnimateModes(fem, partName=fem.metaData['name'], animateModes=True)