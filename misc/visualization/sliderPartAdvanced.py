# ** given model description: **
# A slider of a slider-crank mechanism consisting of an extruded slider body
# with a symmetric guide rabbet at the bottom and a vertical pin on top. The slider
# body is created by extruding a 2D cross-section in the YZ-plane along the x-direction
# by 0.12 m. To keep the final geometry centered at the global origin, the workplane
# for the cross-section is shifted by -0.5*0.12 m in x-direction before the extrusion.
# The overall slider width in Y-direction is 0.12 m and the total height in Z-direction
# is 0.0125 m. The outer rectangular limits of the cross-section therefore lie
# at Y = ±0.5*0.12 m and Z = +0.5*0.0125 m. The bottom center of the rectangle
# is replaced by a symmetric rabbet that forms the guide interface. The rabbet
# has four corner points (all given in m): top-left (-0.1*0.12, -0.2*0.0125),
# bottom-left (-0.2*0.12, -0.5*0.0125), bottom-right (+0.2*0.12, -0.5*0.0125), top-right
# (+0.1*0.12, -0.2*0.0125). The rabbet connects to the outer slider boundary at (-0.5*0.12,
# -0.2*0.0125) and (+0.5*0.12, -0.2*0.0125). The boundary wire is constructed in counter-clockwise
# order starting at (-0.5*0.12, -0.2*0.0125), then through the rabbet corners,
# then to the right connection point, then to the top-right outer corner (+0.5*0.12,
# +0.5*0.0125), then to the top-left outer corner (-0.5*0.12, +0.5*0.0125), and finally
# closing the wire. The upper edges of the slider body are chamfered by 0.12*0.0125
# m. A vertical cylindrical pin of radius 0.015 m and length 0.03 m is placed
# at the center of the top surface of the slider. The curved surface of the
# pin is an interface named "sliderPin". The bottom surface (minimum along global
# z-axis) of the rabbet forms the interface "sliderBottom" and represents the prismatic
# guide along the x-direction. The slider is made of steel, having a Young's modulus
# of 200000000000 N/m², a poisson ratio of 0.3, and a density of 8000 kg/m³.
# Mesh the machine and give it the name "sliderPartAdvanced" such that it can
# be loaded. Limit the mesh resolution by setting an appropriate maximum element
# size using the "maxh" parameter.
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

sliderLength = 0.12
sliderWidth = 0.12
sliderHeight = 0.0125

wp = occ.WorkPlane(occ.Axes((-0.5*sliderLength, 0, 0),
                            n=(1, 0, 0),
                            h=(0, 1, 0)))
wp.MoveTo(-0.5*sliderWidth, -0.2*sliderHeight)
wp.LineTo(-0.1*sliderWidth, -0.2*sliderHeight) \
  .LineTo(-0.2*sliderWidth, -0.5*sliderHeight) \
  .LineTo(0.2*sliderWidth, -0.5*sliderHeight) \
  .LineTo(0.1*sliderWidth, -0.2*sliderHeight) \
  .LineTo(0.5*sliderWidth, -0.2*sliderHeight) \
  .LineTo(0.5*sliderWidth, 0.5*sliderHeight) \
  .LineTo(-0.5*sliderWidth, 0.5*sliderHeight) \
  .Close()
baseFace = wp.Face()
slider = occ.Prism(baseFace, occ.gp_Vec(sliderLength, 0, 0))

chamferSize = 0.12*0.0125
slider = slider.MakeChamfer(slider.faces.Max((0, 0, 1)).edges, chamferSize)

interfaceName = 'sliderBottom'
slider.faces.Min((0, 0, 1)).name = interfaceName
interfaceNameList += [interfaceName]

pinFootPoint = (0, 0, 0.5*sliderHeight)
pinAxis = (0, 0, 1)
pinRadius = 0.015
pinHeight = 0.03
pin = occ.Cylinder(pinFootPoint, pinAxis, pinRadius, pinHeight)
interfaceName = 'sliderPin'
pin.faces[0].name = interfaceName
interfaceNameList += [interfaceName]

geometry = slider + pin
geometry.name = 'steel'
geo = occ.OCCGeometry(geometry)
from ExuAI.flexUtilities import GetMeshParametersDict, MeshSafetyCheck
meshParametersDict=GetMeshParametersDict(geo, maxHLLM=0.0125, meshOrderLLM=2)
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
fem.metaData['name'] = 'sliderPartAdvanced'


exu.sys['exuAI_out']['femObject'] = fem
MakeFFRFObjectAndAnimateModes(fem, partName=fem.metaData['name'], animateModes=True)