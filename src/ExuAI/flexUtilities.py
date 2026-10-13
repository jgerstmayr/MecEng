# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: flexUtilities.py
#
# Details:  Support functions to handle meshes
#
# Author:   Johannes Gerstmayr, Sebastian Weyrer
# Date:     2026-02-24
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import re

def ComputeSurfacePointsAndTriangles(points: np.ndarray,
                                     surfaceTriangles: np.ndarray):
    """Create a new mesh that contains only the surface points.
    
    Uses a given surface triangle mesh and remaps the triangle indices to the new compacted vertex array.

    Parameters
    ----------
    points : (N, d) float array
        Original vertex array (N nodes, d dimensions per vertex).
    surfaceTriangles : (M, 3) int array
        Triangle vertex indices (zero-based) into `points`.

    Returns
    -------
    newPoints : (K, d) float array
        The compacted vertex array containing only the points referenced by `surfaceTriangles`.
    newTriangles : (M, 3) int array
        Triangle indices remapped to the compacted `newPoints` array.
    """
    if surfaceTriangles.ndim != 2 or surfaceTriangles.shape[1] != 3:
        raise ValueError("surfaceTriangles must have shape (M, 3) for triangles.")

    flat = surfaceTriangles.ravel()

    # Get unique vertex indices used by the surface triangles
    # Sorted unique indices
    keptVertexIdx = np.unique(flat)

    # Build mapping from old vertex index -> new compact index
    mapping = np.full(points.shape[0], -1, dtype=np.int64)
    mapping[keptVertexIdx] = np.arange(keptVertexIdx.size, dtype=np.int64)

    # Remap triangles
    newTriangles = np.asarray(mapping[surfaceTriangles], dtype=int)

    # Compact vertex array
    newPoints = np.asarray(points[keptVertexIdx], dtype=np.float64)

    return (newPoints, newTriangles)

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ComputeMeshVolume(points, triangles):
    """Compute volume for given mesh with triangles and points.

    Args:
        points (numpy array): with vertex coordinates x,y,z in each row.
        triangles (numpy array): with point indices for triangle in each row.

    Returns:
        float: volume of triangle mesh.

    """
    vi = points[triangles[:, 0]]
    vj = points[triangles[:, 1]]
    vk = points[triangles[:, 2]]
    # signed volume contribution per face: dot(vi, cross(vj, vk)) / 6
    return abs(float(np.sum(np.einsum('ij,ij->i', vi, np.cross(vj, vk))) / 6.0))


def MeshVolumeDifference(points1, triangles1, points2, triangles2):
    """Compute per-vertex distances from mesh1 to mesh2 using PyMeshLab.
    
    Approximates the volume difference by integrating the (signed) distance over the surface of mesh1 with barycentric vertex area weights.

    Parameters
    ----------
    points1 : (N, 3) array-like of float
        Vertices of mesh1.
    triangles1 : (M, 3) array-like of int
        Faces (0-based indices) of mesh1.
    points2 : (K, 3) array-like of float
        Vertices of mesh2.
    triangles2 : (L, 3) array-like of int
        Faces (0-based indices) of mesh2.

    Returns: dict
    -------
    volumeDelta : float
        Approximate volume difference: sum_i d_i * A_i (A_i are vertex area weights on mesh1).
        If signed=True and the filter provides signed distances, this is a signed estimate.
        If signed=False, this is a magnitude estimate.
    relativeDiff: float
        Relative volume difference based on volume of mesh 1
    pointDistances : (N,) np.ndarray
        Per-vertex distances from mesh1 points to mesh2's surface.
    pointAreas : (N,) np.ndarray
        Barycentric per-vertex area weights on mesh1.
    volume1 : float
        volume of mesh 1
    volume2 : float
        volume of mesh 2

    Raises:
    ------
    ValueError if triangle indices are out of range or not zero-based.
    PyMeshLabException if the filter is unavailable or fails.
    """
    import pymeshlab as ml

    #only get the surface points, no other points!
    (V1orig, F1orig) = ComputeSurfacePointsAndTriangles(points1, triangles1)
    (V2comp, F2comp) = ComputeSurfacePointsAndTriangles(points2, triangles2)

    # Basic validation: zero-based indices and in range
    if F1orig.min() < 0 or F2comp.min() < 0:
        raise ValueError("Face indices must be zero-based and non-negative.")
    if F1orig.max() >= V1orig.shape[0] or F2comp.max() >= V2comp.shape[0]:
        raise ValueError("Face indices out of range for corresponding vertex arrays.")


    # Build MeshSet

    meshComp = ml.Mesh(vertex_matrix=V2comp, face_matrix=F2comp)
    meshOrig = ml.Mesh(vertex_matrix=V1orig, face_matrix=F1orig)

    ms = ml.MeshSet()
    ms.add_mesh(meshOrig, 'meshOriginal')  # index 0
    ms.add_mesh(meshComp, 'meshCompare')   # index 1


    # Compute per-vertex distances on mesh 0 to mesh 1
    # use filter 'compute_scalar_by_distance_from_another_mesh_per_vertex'
    # with measuremesh=0, refmesh=1. For signed distances, set signeddist=True.
    ms.apply_filter(
        'compute_scalar_by_distance_from_another_mesh_per_vertex',
        measuremesh=1,   # mesh being measured (distances stored here)
        refmesh=0,       # original mesh to measure distances against
        signeddist=False,
    )

    # Retrieve distances. PyMeshLab stores now in vertex_scalar_array.
    # ms[1] = meshComp, but meshComp cannot retrieve current vertex_scalar_array!!!
    dists = np.asarray(ms[1].vertex_scalar_array(), dtype=np.float64) #distances to original mesh

    # Compute per-face areas on mesh2 and barycentric per-vertex areas:
    # For a triangle with vertices (a,b,c), area is:
    # A_face = 0.5 * || (V[b] - V[a]) x (V[c] - V[a]) ||
    # Then distribute A_face/3 to each of its vertices.
    vertexAreas = np.zeros(V2comp.shape[0], dtype=np.float64)
    for a, b, c in F2comp:
        e1 = V2comp[b] - V2comp[a]
        e2 = V2comp[c] - V2comp[a]
        A_face = 0.5 * np.linalg.norm(np.cross(e1, e2))
        share = A_face / 3.0
        vertexAreas[a] += share
        vertexAreas[b] += share
        vertexAreas[c] += share

    # Volume difference approximation: sum_i d_i * A_i
    volumeDelta = float(np.sum(dists * vertexAreas))
    
    volume1 = ComputeMeshVolume(points1, triangles1) 
    volume2 = ComputeMeshVolume(points2, triangles2)

    return {'volumeDiff': volumeDelta,
            'relativeDiff': volumeDelta / max(1e-15,volume1), #avoid div by zero; limit to 0.01 x 0.01 x 0.01 mm^3
            'pointDistances': dists,
            'pointAreas': vertexAreas,
            'volume1': volume1, 
            'volume2': volume2,
            }

#%% Evaluate a fem object

import exudyn as exu
from exudyn.FEM import *
import ngsolve as ngs
import netgen.occ as occ

def RelativeMassDifference(LLMFemObject=None, sampleFemObject=None):
    LLMRigidBodyInertia = LLMFemObject.GetRigidBodyInertia()
    sampleRigidBodyInertia = sampleFemObject.GetRigidBodyInertia()
    LLMMass = LLMRigidBodyInertia.Mass()
    sampleMass = sampleRigidBodyInertia.Mass()
    diffMass = abs(sampleMass-LLMMass)
    return diffMass / sampleMass


def RelativeEigenfrequenciesNormDifference(LLMFemObject=None, sampleFemObject=None, nEigenModesToCompare=3):
    LLMEigenfrequencies = LLMFemObject.GetEigenFrequenciesHz()[:nEigenModesToCompare]
    sampleEigenfrequencies = sampleFemObject.GetEigenFrequenciesHz()[:nEigenModesToCompare]
    relativeEigenfrequenciesNormDifference = np.linalg.norm(sampleEigenfrequencies-LLMEigenfrequencies) / np.linalg.norm(sampleEigenfrequencies)
    if relativeEigenfrequenciesNormDifference > 1:
        return 1
    else:
        return relativeEigenfrequenciesNormDifference

# function called only in sample scripts
def GetFemObjectFromOCCGeometry(geometry, partName='', materials=[], interfaceNameList=[]):
    geo = occ.OCCGeometry(geometry)
    meshParametersDict = GetMeshParametersDict(geo, partName=partName)
    geoMesh = geo.GenerateMesh(maxh=meshParametersDict['maxH'], 
                               grading=meshParametersDict['grading'], # 0 < grading < 1; larger gives coarser meshes; default=?
                               curvaturesafety=meshParametersDict['curvatureSafety']) # smaller value gives coarser mesh; default=?

    computationMode = HCBstaticModeSelection.RBE2 #RBE2 is standard; RBE3 would be sometimes better

    mesh = ngs.Mesh(geoMesh)
    meshOrder = meshParametersDict['meshOrder']
    femObject = FEMinterface()
    [bfM, bfK, fes] = femObject.ImportMeshFromNGsolve(mesh, materials=materials,
                                                boundaryNamesList=interfaceNameList,
                                                meshOrder=meshOrder)
    # mandatory: compute the eigenmodes of the system; do not skip that
    # get two lists that are needed for computation of eigenmodes
    [boundaryNodesList, boundaryWeightsList] = femObject.GetBoundaryNodeSetsAsLists()
    MeshSafetyCheck(femObject, meshParametersDict['maxNumberOfFemNodes'])
    femObject.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryNodesList,
                                            nEigenModes=meshParametersDict['nEigenModes'],
                                            boundaryNodesWeights=boundaryWeightsList,
                                            computationMode=computationMode)
    # give the created object a specific name such that it can be loaded again
    femObject.metaData['name'] = partName
    # now add exuAI_out dict if not there to ensure that sample file completes successfully
    if 'exuAI_out' not in exu.sys:
        print(f'WARNING: GetFemObjectFromOCCGeometry: "Out" dictionary for "ExuAI" project is not present (part={partName}); this may only be the case for direct sample file execution!')
        exu.sys['exuAI_out'] = {}
    return femObject

# function called in GetFemObjectFromOCCGeometry and via regex in LLM code   
def GetMeshParametersDict(geo, maxHLLM=None, meshOrderLLM=None, partName=''):
    if 'exuAI_in' not in exu.sys:
        print(f'WARNING: GetMeshParametersDict: "exuAI_in" dictionary for "ExuAI" project is not present (part={partName}); this may only be the case for direct sample file execution!')
        exu.sys['exuAI_in'] = {}
    if 'exuAI_out' not in exu.sys:
        print(f'WARNING: GetMeshParametersDict: "exuAI_out" dictionary for "ExuAI" project is not present (part={partName}); this may only be the case for direct sample file execution!')
        exu.sys['exuAI_out'] = {}
    returnDict = {}
    # default mesh parameter or those globally available via exudyn dict
    returnDict['nEigenModes'] = exu.sys['exuAI_in'].get('nEigenModes') or 6
    returnDict['maxNumberOfFemNodes'] = exu.sys['exuAI_in'].get('maxNumberOfFemNodes') or 4e4
    returnDict['grading'] = exu.sys['exuAI_in'].get('grading') or 0.9
    returnDict['curvatureSafety'] = exu.sys['exuAI_in'].get('curvatureSafety') or 0.7
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # compute maxH limit via a bounding box of faces
    minx = miny = minz = float('inf')
    maxx = maxy = maxz = float('-inf')
    
    for face in geo.faces:
        (mn, mx) = face.bounding_box
        minx = min(minx, mn[0]); miny = min(miny, mn[1]); minz = min(minz, mn[2])
        maxx = max(maxx, mx[0]); maxy = max(maxy, mx[1]); maxz = max(maxz, mx[2])
    
    # If the list was empty, return None
    if minx == float('inf'):
        raise ValueError('GetMeshParametersDict: no bounding box computable!')
    
    boundingBox = np.array([[minx, miny, minz], [maxx, maxy, maxz]])
    [dx,dy,dz]=boundingBox[1]-boundingBox[0]
    
    if min(dx,dy,dz) < 1e-6:
        raise ValueError('GetMeshParametersDict: problems with bounding box: dimension is < 1e-6')
    
    exu.sys['exuAI_out']['boundingBox'] = boundingBox

    maxHLimitWithoutFactorRule0 = (1/20) * max(dx, dy, dz)
    maxHLimitWithoutFactorRule1 = (1/10) * (dx*dy*dz)**(1/3)
    # for collapsed structures, do the following
    maxHLimitWithoutFactor = max(maxHLimitWithoutFactorRule0, maxHLimitWithoutFactorRule1)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # save maxH properties under exuAI_out
    exu.sys['exuAI_out']['maxHLimitWithoutFactor'] = maxHLimitWithoutFactor
    if maxHLLM is not None:
        exu.sys['exuAI_out']['maxHLLM'] = maxHLLM
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # compute the limit with some cases and save under exuAI_out
    maxHLimitFactor = exu.sys['exuAI_in'].get('maxHFactor') or 1
    maxHLimitWithFactor = maxHLimitWithoutFactor * maxHLimitFactor
    exu.sys['exuAI_out']['maxHLimitWithFactor'] = maxHLimitWithFactor
    if type(maxHLLM) is float: # check if float
        if maxHLLM < maxHLimitWithFactor:
            exu.sys['exuAI_out']['maxH'] = maxHLimitWithFactor
        if maxHLLM >= maxHLimitWithFactor:
            exu.sys['exuAI_out']['maxH'] = maxHLLM
    else:
        exu.sys['exuAI_out']['maxH'] = maxHLimitWithFactor
    # and now also add this to mesh properties dict
    returnDict['maxH'] = maxHLimitWithFactor

    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # set meshOrder
    if meshOrderLLM is not None:
        exu.sys['exuAI_out']['meshOrderLLM'] = meshOrderLLM
    if meshOrderLLM == 1 or meshOrderLLM == 2:
        meshOrder = meshOrderLLM
    else:
        meshOrder = exu.sys['exuAI_in'].get('meshOrder') or 2
    exu.sys['exuAI_out']['meshOrder'] = meshOrder
    returnDict['meshOrder'] = meshOrder
    return returnDict

def MeshSafetyCheck(femObject, maxNumberOfFemNodes):
    numberOfNodes = femObject.NumberOfNodes()
    if numberOfNodes > maxNumberOfFemNodes:
        raise ValueError('Safety Checks: Number of nodes exceeds specified max. amount of nodes. Cannot continue.')

def MakeFFRFObjectAndAnimateModes(femObject, partName='', animateModes=True, **kwargs):
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()
    femObject.SaveToFile(partName + '.npz') # we only do that when directly running the sample file, not when it is used to evaluate
    print('Number of nodes:', femObject.NumberOfNodes())
    print('Computed eigenfrequencies in Hz:', femObject.GetEigenFrequenciesHz())
    createFFRFDict = mbs.CreateFFRFReducedOrderObject(name=partName,
                                                      femInterface=femObject,
                                                      verbose=True)
    print('The FFRF underlying rigid body node has number', mbs.GetNodeNumber('NodeRigidBody:' + partName))
    print('The generic node holding the modes has number', mbs.GetNodeNumber('NodeGeneric:' + partName))
    print('The created fem object has number', mbs.GetObjectNumber(partName))
    # compute mesh volume
    triangles = np.asarray(femObject.surface[0]['Trigs'], dtype=int)
    points = np.asarray(femObject.GetNodePositionsAsArray(), dtype=np.float64)
    volume = ComputeMeshVolume(points, triangles)
    print('Mesh volume', volume)
    if animateModes:
        from exudyn.interactive import AnimateModes
        mbs.Assemble()
        
        #compute overall dimension
        minCoords = np.min(points,axis=0)
        maxCoords = np.max(points,axis=0)
        maxDim = np.linalg.norm(maxCoords-minCoords)
        SC.visualizationSettings.nodes.show = True
        SC.visualizationSettings.view0.scene.showFaceEdges = True
        SC.visualizationSettings.openGL.multiSampling = 4
        SC.visualizationSettings.openGL.lineWidth = 2
        SC.visualizationSettings.view0.scene.drawWorldBasis = True
        SC.visualizationSettings.view0.scene.worldBasisSize = 0.5 * maxDim
        SC.visualizationSettings.view0.window.renderWindowSize = [1600, 1080]
        SC.visualizationSettings.general.autoFitScene = False # otherwise, model may be difficult to be moved
        SC.visualizationSettings.openGL.lightModelAmbient = [0.5, 0.5, 0.5, 1]
        
        if 'exuAI_out' in exu.sys:
            if 'boundingBox' in exu.sys['exuAI_out']:
                boundingBox = exu.sys['exuAI_out']['boundingBox']
                [dx,dy,dz]=boundingBox[1]-boundingBox[0]
                ds = max([dx,dy,dz]) * 0.02
                SC.visualizationSettings.nodes.defaultSize = ds/2
                SC.visualizationSettings.markers.defaultSize = ds
                SC.visualizationSettings.markers.drawSimplified = False 
                SC.visualizationSettings.nodes.drawNodesAsPoint = False
        
        nodenumber = mbs.GetNodeNumber('NodeGeneric:' + partName) # this is the node with the generalized coordinates
        
        period = kwargs.get('period') or 0.1
        scaleAmplitude = kwargs.get('scaleAmplitude') or 0.5

        AnimateModes(SC, mbs, nodenumber, period=period, scaleAmplitude=scaleAmplitude, 
                     showTime=False, runOnStart=True)
    
    return (SC, mbs)
    
def AddSpecialCodeForPartsCreation(code):
    prefix = ('import sys\n'
              'from pathlib import Path\n'
              'if "__file__" in globals():\n'
              '    sys.path.append(str(Path(Path(__file__).resolve().parents[5],"src")))\n\n')#for logsTM parallel to src
    
    code = prefix + code
    # this is only the case for sample files
    if 'helper.utilitiesForSampleFiles' in code:
        code = code.replace('helper.utilitiesForSampleFiles', 'ExuAI.flexUtilities')
        code = code + "exu.sys['exuAI_out']['materials'] = materials"
    else:
        # add material finder
        materialPattern = r"(\[[^\]]*\]\s*=\s*)?(fem\.ImportMeshFromNGsolve\([\s\S]*?materials\s*=\s*([A-Za-z_]\w*)[\s\S]*?\))"
        def replacer(match):
            prefix = match.group(1) or ''
            fullCall = match.group(2)
            var = match.group(3)
            injection = f"exu.sys['exuAI_out']['materials'] = {var}\n"
            return injection + prefix + fullCall
        code = re.sub(materialPattern, replacer, code)
    # and this is done for non-sample files (LLM generated code)
    maxHPattern = r"(maxh)\s*=\s*([0-9]+(?:\.[0-9]+)?)"
    meshOrderPattern = r"(meshOrder)\s*=\s*([0-9]+(?:\.[0-9]+)?)"
    lines = code.splitlines()
    newLines = []
    meshOrderLLM = None
    maxhLLM = None
    for line in lines:
        meshOrderMatch = re.search(meshOrderPattern, line)
        if meshOrderMatch:
            meshOrderLLM = meshOrderMatch.group(2)
            break  # first occurrence is enough
    for line in lines:
        maxHMatch = re.search(maxHPattern, line)
        meshOrderMatch = re.search(meshOrderPattern, line)
        if maxHMatch:
            maxhLLM = maxHMatch.group(2)
            newLines.append(f"from ExuAI.flexUtilities import GetMeshParametersDict, MeshSafetyCheck")
            newLines.append(f"meshParametersDict=GetMeshParametersDict(geo, maxHLLM={maxhLLM}, meshOrderLLM={meshOrderLLM})")
            line = re.sub(maxHPattern, "maxh=meshParametersDict['maxH'], grading=meshParametersDict['grading'], curvaturesafety=meshParametersDict['curvatureSafety']", line)
        if 'ComputeHurtyCraigBamptonModes' in line:
            newLines.append("MeshSafetyCheck(fem, meshParametersDict['maxNumberOfFemNodes'])")
        if meshOrderMatch:
            line = re.sub(meshOrderPattern, "meshOrder=meshParametersDict['meshOrder']", line)
        newLines.append(line)
    return '\n'.join(newLines)

def CheckIfSafetyCheckIsPresent(code):
    if 'GetFemObjectFromOCCGeometry' in code or 'MeshSafetyCheck' in code:
        return True
    else:
        return False
        
def MeshDictLog(mode, agent, inDict={}, outDict = {}, verbose=False):
    agent.logger.LogText('Mode: ' + mode + '; Executed code - special mesh parameters: '
                        'maxHLimitWithFactor = ' + str(round(outDict['maxHLimitWithFactor'], 4)) +
                        ', maxH = ' + str(round(outDict['maxH'], 4)) + ', '
                        'meshOrder = ' + str(round(outDict['meshOrder'])) + ', '
                        'grading = ' + str(inDict['grading']) + ', maxNumberOfFemNodes = '
                        + str(inDict['maxNumberOfFemNodes']) +
                        ', curvatureSafety = ' + str(inDict['curvatureSafety']),
                        printToConsole=verbose, separator=True, quickinfo=True)
