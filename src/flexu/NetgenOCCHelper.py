#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# 
# Details:  This file serves as a documented example that shall be used by 
#           LLMs to update information on their internal knowledge of how to
#           create a 3D geometry mesh; used in addition to other helpers.
#
# Author:   Sebastian Weyrer; Tobias Möltner
# Date:     2026-05-05
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%
#TAG: general import
#DIFFICULTY: 0
#INFO: import required libraries
import exudyn as exu
from exudyn.FEM import *
import ngsolve as ngs
import netgen.occ as occ
try:
    from ExuAI.flexUtilities import GetMeshParametersDict, MeshSafetyCheck, MakeFFRFObjectAndAnimateModes
except:
    pass

# IMPORTANT ORDERING RULES AND NOTES:
#   MakeFillet / MakeChamfer    BEFORE          assigning and naming interfaces
#   Name interfaces             BEFORE          boolean operations (- or +)
#   Rotate or Move              BEFORE/AFTER    assigning interfaces (if AFTER, copied geometry has SAME interface)
# NOTE: interface naming shown in each primitive tag below is only an example --
#       only add an interface if the primitive actually needs one in your part.
# NOTE: in the interfaceNameList, each interface name must only appear once. If more faces share
#       the same interface, put the name once in the interfaceNameList and name the faces the same.

# add items here
 
#%%
#TAG: material
#DIFFICULTY: 50
#INFO: Define a material for the machine part and initialize a list for the interfaces of it.
#USES: 'youngsModulus'
# The dict key (e.g. 'steel') must exactly match geometry.name set later.
# Fixed key names inside the dict: youngsModulus, poissonsRatio, density -- no abbreviations.
# interfaceNameList collects all named interface surfaces throughout machine part creation.
E = 2.1e11 # Young's modulus [N/m2]
nu = 0.3 # Poisson's ratio [-]
rho = 7850 # density [kg/m3]
materials = {'steel': {'youngsModulus': E,
                       'poissonsRatio': nu,
                       'density': rho}}
interfaceNameList = []  # filled by interface naming steps later
 
#%%
#TAG: box
#DIFFICULTY: 50
#USES: occ.Box
#INFO: Create a rectangular box (cuboid, beam) from two diagonally opposite corner points.
# Example: boxCentered is centred at the global origin; boxLeft is placed directly to its left.
boxLength = 0.5
boxWidth = 0.2
boxHeight = 0.2
boxCentered = occ.Box((-boxLength/2, -boxWidth/2, -boxHeight/2),
                      ( boxLength/2,  boxWidth/2,  boxHeight/2))
boxLeft = occ.Box((-boxLength*(3/2), -boxWidth/2, -boxHeight/2),
                  (-boxLength/2,      boxWidth/2,  boxHeight/2))
# Example: make the bottom face of boxCentered an interface.
interfaceName = 'boxCenteredBottom'
boxCentered.faces.Min((0, 0, 1)).name = interfaceName
interfaceNameList += [interfaceName]
 
#%%
#TAG: cylinder
#DIFFICULTY: 50
#USES: occ.Cylinder
#INFO: Create a solid or hollow cylinder from a foot point, an axis direction, a radius, and a height.
# footPoint: centre of the bottom circular face.
# axis: direction the cylinder grows along.
# The curved lateral surface is always accessible as .faces[0].
# The flat end caps are accessed via .faces.Min((x, y, z)) / .faces.Max((x, y, z)) along the axis.
# A hollow (tubular) cylinder is built by subtracting an inner from an outer cylinder.
# CRITICAL: name the inner cylinder's faces BEFORE the subtraction -- after subtraction the
# inner end caps disappear into the void and only the curved wall survives as an interface.
# Example: solid cylinder placed on top of boxCentered, growing in +z.
cylinderFootPoint = (0, 0, boxHeight/2)
cylinderAxis = (0, 0, 1) # grows in +z
cylinderRadius = 0.1
cylinderHeight = 0.2
cylinder = occ.Cylinder(cylinderFootPoint, cylinderAxis, cylinderRadius, cylinderHeight)
# Example: make the curved lateral face an interface.
interfaceName = 'cylinderCurved'
cylinder.faces[0].name = interfaceName
interfaceNameList += [interfaceName]
# Example: hollow cylinder placed on top of the solid cylinder.
hollowFootPoint = (0, 0, boxHeight/2 + cylinderHeight)
hollowAxis = (0, 0, 1)
hollowHeight = 0.5
outerRadius = 0.15
innerRadius = 0.05
outerCylinder = occ.Cylinder(hollowFootPoint, hollowAxis, outerRadius, hollowHeight)
innerCylinder = occ.Cylinder(hollowFootPoint, hollowAxis, innerRadius, hollowHeight)
# Example: make the inner wall an interface -- named BEFORE subtraction.
outerCylinder.faces.name = 'innerWall'
interfaceNameList += ['innerWall']
# boolean subtraction to get tubular cylinder
hollowCylinder = outerCylinder - innerCylinder
 
#%%
#TAG: sphere
#DIFFICULTY: 50
#USES: occ.Sphere
#INFO: Create a sphere from a centre point and a radius.
# Example: sphere placed at the bottom face centre of boxCentered.
sphereRadius = 0.08
sphereMidpoint = (0, 0, -boxHeight/2)
sphere = occ.Sphere(sphereMidpoint, sphereRadius)
 
#%%
#TAG: extrusion
#DIFFICULTY: 50
#USES: occ.Prism
#INFO: Create a 3D solid by extruding a 2D cross-section profile drawn on a workplane.
# Step 1 -- WorkPlane(Axes(origin, n=faceNormal, h=horizontalDir)) defines the drawing plane.
# Step 2 -- draw a closed 2D profile starting with MoveTo(u,v):
#   LineTo(u, v)                        straight edge to (u, v)
#   ArcTo(u, v, occ.gp_Vec2d(nu, nv))  circular arc ending at (u,v);
#                                       gp_Vec2d is the tangent direction at the arc START point
#   Close()                             close the profile back to the MoveTo point
# Step 3 -- .Face() converts the closed profile into a 2D face.
# Step 4 -- occ.Prism(face, occ.gp_Vec(dx,dy,dz)) extrudes along that vector;
#           the vector encodes both direction and extrusion length.
# Example: workplane on the +x face of boxCentered, extruding further in +x.
extrusionLength = 0.5
wp = occ.WorkPlane(occ.Axes((boxLength/2, -boxWidth/2, -boxHeight/2), # corner of the +x face of boxCentered
                            n=(1, 0, 0), # face normal: extrudes in +x
                            h=(0, 1, 0))) # horizontal direction inside the plane: +y
wp.MoveTo(0.0, 0.0)
wp.LineTo(0.2, 0.0) \
  .ArcTo(0.0, 0.2, occ.gp_Vec2d(0, 1)) \
  .Close()
baseFace = wp.Face()
extrusion = occ.Prism(baseFace, occ.gp_Vec(extrusionLength, 0, 0))
 
#%%
#TAG: fillet
#DIFFICULTY: 50
#USES: MakeFillet
#INFO: Round selected edges with a constant-radius fillet.
# .faces.Max((x,y,z)).edges selects all edges of the face at maximum extent along that axis.
# .faces.Min((x,y,z)).edges selects all edges of the face at minimum extent.
# .edges.Max((x,y,z)).edges selects all edges at maximum extent along that axis.
# .edges.Min((x,y,z)).edges selects all edges at minimum extent.
# Example: fillet all edges on the top face of boxLeft.
filletRadius = 0.02
boxLeft = boxLeft.MakeFillet(boxLeft.faces.Max((0, 0, 1)).edges, filletRadius)
 
#%%
#TAG: chamfer
#DIFFICULTY: 50
#USES: MakeChamfer
#INFO: Cut a flat angled bevel on selected edges.
# .faces.Max((x,y,z)).edges selects all edges of the face at maximum extent along that axis.
# .faces.Min((x,y,z)).edges selects all edges of the face at minimum extent.
# .edges.Max((x,y,z)).edges selects all edges at maximum extent along that axis.
# .edges.Min((x,y,z)).edges selects all edges at minimum extent.
# Example: chamfer the edges of the bottom face of boxLeft
chamferSize = 0.02
boxLeft = boxLeft.MakeChamfer(boxLeft.faces.Min((0, 0, 1)).edges, chamferSize)
# Example: make bottom of boxLeft an interface AFTER making the chamfer.
interfaceName = 'boxLeftBottom'
boxLeft.faces.Min((0, 0, 1)).name = interfaceName
interfaceNameList += [interfaceName]
 
#%%
#TAG: rotate geometry
#DIFFICULTY: 50
#USES: Rotate
#INFO: Get a rotated copy of an already created geometry.
# Specify a rotation axis via occ.Axis(point, direction) and a rotation angle in degrees.
# Example: spin hollowCylinder 90 degrees around its own centre (axis along y).
rotationAxis = occ.Axis((0, 0, boxHeight/2 + cylinderHeight + hollowHeight/2), (0, 1, 0))
hollowCylinderRotated = hollowCylinder.Rotate(rotationAxis, 90)
 
#%%
#TAG: move geometry
#DIFFICULTY: 50
#USES: Move(
#INFO: Get a translated copy of an already created geometry.
# Specify a translation vector (dx, dy, dz).
# Example: move boxLeft to the right.
boxRight = boxLeft.Move((boxLength*2+extrusionLength, 0, 0))
 
#%%
#TAG: mesh geometry
#DIFFICULTY: 50
#INFO: mesh a created machine part geometry and give it a name
#USES: GetFemObjectFromOCCGeometry
# always keep the procedure here the same, just change parameters and names accordingly
geometry = boxCentered + boxRight + cylinder + hollowCylinder + extrusion + sphere + hollowCylinderRotated
geometry.name = 'steel' # must match the key in the materials dict
geo = occ.OCCGeometry(geometry)
# CRITICAL: the numeric value assigned for maxh must appear LITERALLY inside GenerateMesh, NOT via a variable:
# WRONG:  maxElementSize = 0.0125
#         geoMesh = geo.GenerateMesh(maxh=maxElementSize)
# RIGHT:  geoMesh = geo.GenerateMesh(maxh=0.0125)
geoMesh = geo.GenerateMesh(maxh=0.25)
mesh = ngs.Mesh(geoMesh)
meshOrder = 2  # quadratic elements (order 1 = linear, fewer nodes but less accurate)
fem = FEMinterface()
[bfM, bfK, fes] = fem.ImportMeshFromNGsolve(mesh, materials=materials,
                                            boundaryNamesList=interfaceNameList, meshOrder=meshOrder)
# Craig-Bampton reduction: mandatory, always after ImportMeshFromNGsolve
[boundaryNodesList, boundaryWeightsList] = fem.GetBoundaryNodeSetsAsLists()
fem.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryNodesList,
                                  nEigenModes=6, boundaryNodesWeights=boundaryWeightsList)
fem.metaData['name'] = 'nameOfObject' # give the machine part a name

#%%
#TAG: not used
# INFO: test the helper geometry when running directly
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]   # adjust depth if needed
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# we need this here because of the location of this helper
from ExuAI.flexUtilities import MakeFFRFObjectAndAnimateModes
MakeFFRFObjectAndAnimateModes(fem, partName='nameOfObject')
# %%
