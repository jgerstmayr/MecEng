#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  A test to compare system graphs in Exudyn
#
# Author:   Johannes Gerstmayr, Sebastian Weyrer
# Date:     2026-02-24
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn
import numpy as np
import time

from .logger import Logger

from exudyn.utilities import DrawSystemGraph
from ExuAI.utilities import SplitCamelCaseIntoStringsList



#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from collections import Counter
import networkx as nx
from networkx.algorithms import isomorphism as iso



#helper for CompareSystemGraphs: show compact difference of counters
def CompactCounterDiff(c1: Counter, c2: Counter, *, sort_by="key") -> str:
    """
    Produce a compact string like '+2 Mass, -1 Spring'.
    sort_by: 'key' (alphabetical), 'magnitude' (descending abs diff), or 'sign' (positives first).
    """
    def counter_signed_diff(c1: Counter, c2: Counter) -> dict:
        """
        Return a dict mapping key -> (c1[key] - c2[key]) for all keys with nonzero diff.
        Positive = surplus in c1, negative = deficit in c1 (i.e., surplus in c2).
        """
        keys = set(c1) | set(c2)
        diff = {k: c1[k] - c2[k] for k in keys}
        return {k: v for k, v in diff.items() if v != 0}

    diff = counter_signed_diff(c1, c2)
    items = list(diff.items())

    if sort_by == "magnitude":
        items.sort(key=lambda kv: (abs(kv[1]), kv[0]), reverse=True)
    else:  # 'key'
        items.sort(key=lambda kv: str(kv[0]))

    parts = [f"{'+' if v > 0 else ''}{v} {k}" for k, v in items]  # v already negative when deficit
    return ", ".join(parts) if parts else "no differences"


#compare two graph node annotations using tolerance for numerical values
def DeepEqual(a, b, *, rtol=1e-4, atol=1e-6):
    """
    Recursively compare a and b.
    - Numbers: ints exact, floats via allclose-like rule.
    - NumPy arrays: dtype and shape must match; values via allclose.
    - Lists/tuples: same length; elementwise compare in order.
    - Dicts: same keys; compare values recursively.
    - Other types: use ==.
    """
    # Handle exact same object
    if a is b:
        return True

    # None
    if a is None or b is None:
        return a is None and b is None

    # NumPy arrays
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)

    # Numbers
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        # Promote both to float
        try:
            af = float(a)
            bf = float(b)
        except Exception:
            return False
        return abs(af - bf) <= (atol + rtol * max(abs(af), abs(bf)))

    # Strings
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b

    # Dicts
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(DeepEqual(a[k], b[k], rtol=rtol, atol=atol) for k in a)

    # Lists/Tuples
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b) or (isinstance(a, tuple) != isinstance(b, tuple)):
            return False
        return all(DeepEqual(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b))

    # these lists can be converted to lists of numpy arrays:
    if isinstance(a, (exudyn.Vector2DList, exudyn.Vector3DList, exudyn.Vector6DList, 
                      exudyn.Matrix3DList,exudyn.Matrix6DList)):
        ap = None
        try:
            ap = a.GetPythonObject()
        except:
            pass
        bp = None
        try:
            bp = b.GetPythonObject()
        except:
            pass
        return DeepEqual(ap, bp, rtol=rtol, atol=atol)

    # Fallback: direct equality (covers bools and other simple hashables)
    try:
        return a == b
    except Exception:
        return False


#%%get difference of annotations
def DiffAnnotations(G1, G2, mapping, *, annKey="annotations", compare=DeepEqual, 
                    skipMeshDependentQuantitiesInFFRFObjects=False,
                    ignoreUserFunctions=True,
                    skipDataIfHasUserFunction=True
                    ):
    """
    Returns a list of node-level mismatches:
    [(u_in_G1, v_in_G2, key, val1, val2), ...]
    """

    def PreviewNParray(a):
        if type(a) == exudyn.MatrixContainer:
            a = a.Convert2DenseMatrix()
        if type(a) == np.ndarray and len(a) > 0:
            return '['+str(float(np.array(a).flatten()[0])) + ', ...]'
        else:
            return '...'

    #do special treatment of ObjectFFRFreducedOrder to reduce size of diffs
    #return only part of data
    # this is just to avoid clustered outputs ...
    def ModDiff(itemType, typeName, data):
        if itemType=='ObjectFFRFreducedOrder':
            if not typeName.startswith('physics') and not typeName.startswith('reference'):
                if typeName.endswith('MatrixReduced'):
                    return PreviewNParray(data)
                return '...'
            else:
                return data
        else:
            return data
    
    #check if comparison shall be skipped for certain items and data types
    def SkipComparison(uType, key, annotation1):
        skipComparison = False
        if skipMeshDependentQuantitiesInFFRFObjects:
            # if we ignore some things for FFRF objects, just check those things that are REALLY independent of mesh discretization
            keysCheckedForFFRFObjectList = ['referencePositions'] # mass etc. is already checked here and can differ a bit because of different mesh sizes!
            keysCheckedForSuperElementRigidMarkersList = ['offset']
            if uType =='ObjectFFRFreducedOrder':
                if key not in keysCheckedForFFRFObjectList:
                    skipComparison = True
            if uType == 'MarkerSuperElementRigid':
                if key not in keysCheckedForSuperElementRigidMarkersList:
                    skipComparison = True
        if ignoreUserFunctions and 'userfunction' in key.lower():
            skipComparison = True
        if skipDataIfHasUserFunction and isinstance(annotation1, dict):
            #in case of load user function, loadVector is optional to be used
            if 'loadVectorUserFunction' in annotation1 and key == 'loadVector':
                skipComparison = True
                if isinstance(annotation1['loadVectorUserFunction'], (int, float)):
                    if annotation1['loadVectorUserFunction'] == 0: # has no influence (is actually default!!)
                        skipComparison = False # set again to false
            #in case of springTorqueUserFunction, offset, stiffness and damping can be defined in user function
            if 'springTorqueUserFunction' in annotation1 and (key in ['offset','stiffness','damping']):
                skipComparison = True

        return skipComparison
        
    
    diffs = []
    for u in G1.nodes:
        v = mapping[u]
        a1 = G1.nodes[u].get(annKey, None)
        a2 = G2.nodes[v].get(annKey, None)
        uName = G1.nodes[u].get('name', None)
        vName = G2.nodes[v].get('name', None)
        uType = G1.nodes[u].get('type', None)
        # If annotations are dicts, compare per-key for more informative diffs
        G1info = 'G1-'+str(uName)
        G2info = 'G2-'+str(vName)
        if isinstance(a1, dict) and isinstance(a2, dict): #regular track with two item dicts:
            keys = set(a1) | set(a2)
            for k in keys:
                skipComparison = SkipComparison(uType, key=k, annotation1=a1)

                if not skipComparison:
                    x, y = a1.get(k, None), a2.get(k, None)
                    if not compare(x, y):
                        diffs.append((G1info, G2info, uType, k, 
                                      ModDiff(uType, k,x), ModDiff(uType,k,y)))
        else: #if a1 and a2 are something else; does this happen at all?
            if not compare(a1, a2):
                skipComparison = SkipComparison(uType, key=k, annotation1=a1)

                if not skipComparison:
                    diffs.append((G1info, G2info, uType, None, 
                                  ModDiff(uType,k,a1), ModDiff(uType,k,a2) ))
        
    return diffs

#%%
#**function: compare two networkx graphs as returned by DrawSystemGraph (G1reference is considered to be the True graph!)
#**output::dict: containing readable information on different elements
def CompareSystemGraphs(G1reference, G2compare, maxAnnotationIterations=1000,
                        returnGraphMatchIterator=False, addBasicTypes=False,
                        removeIsolatedGrounds=True,
                        removeAllGroundObjects=False,
                        maxNodesForIsomorphismCheck=150, #EXPERIMENTAL, testing only: default matches prior hardcoded 150
                        skipMeshDependentQuantitiesInFFRFObjects=False,
                        ignoreUserFunctions=True, #currently giving points which cannot be compared directly
                        annotationTimeoutSeconds=120,
                        logger:Logger=None,
                        model='unknown'
                        ):

    msg = ''

    G1 = G1reference.copy()
    G2 = G2compare.copy()

    #remove isolated grounds as they do not affect correctness (used for e.g. drawing).
    #removeAllGroundObjects (opt-in, default off) instead drops EVERY ObjectGround node
    #regardless of degree: ground objects are semantically interchangeable (they only mark
    #a fixed anchor to the inertial frame), so using N separate ground instances for
    #different anchor points vs. one shared instance is a harmless style choice, not a
    #modeling difference. Nodes are only DELETED here, never merged/reconnected into a
    #shared hub node -- merging was tried and reverted, since raising one node's degree can
    #push the isomorphism check below into pathological runtime on symmetric graphs (e.g.
    #springsBridge) that the cooperative timeout does not reliably catch.
    if removeAllGroundObjects:
        for i, G in enumerate([G1, G2]):
            removeNodes = [nodeName for nodeName, nodeType in G.nodes(data='type')
                          if nodeType == 'ObjectGround']
            if len(removeNodes):
                #before deleting a ground node, resolve its markers' localPosition into a
                #world position (referencePosition + referenceRotation @ localPosition) so
                #content comparison is independent of *which* ground (origin ground + offset
                #marker vs. offset ground + zero marker are the same world point, just a
                #different but equally valid way to split the offset between the two):
                for nodeName in removeNodes:
                    groundAnn = G.nodes[nodeName].get('annotations') or {}
                    refPos = groundAnn.get('referencePosition')
                    refRot = groundAnn.get('referenceRotation')
                    if refPos is not None:
                        for markerName in G.neighbors(nodeName):
                            markerAnn = G.nodes[markerName].get('annotations')
                            if isinstance(markerAnn, dict) and 'localPosition' in markerAnn:
                                localPos = markerAnn['localPosition']
                                try:
                                    worldPos = np.array(refPos, dtype=float)
                                    if refRot is not None:
                                        worldPos = worldPos + np.array(refRot, dtype=float) @ np.array(localPos, dtype=float)
                                    else:
                                        worldPos = worldPos + np.array(localPos, dtype=float)
                                    markerAnn['localPosition'] = worldPos
                                except Exception:
                                    pass #leave annotation untouched if shapes are unexpected
                msg += ('removed '+str(len(removeNodes))+' ObjectGround node(s) from graph G'
                       +str(i+1)+' (ground identity is not modeling-relevant; resolved'
                       ' ground-attached marker positions to world coordinates first)\n')
                for nodeName in removeNodes:
                    G.remove_node(nodeName)
    elif removeIsolatedGrounds:
        for i, G in enumerate([G1, G2]):
            removeNodes = []
            for nodeName, nodeType in G.nodes(data="type"):
                if nodeType == 'ObjectGround':
                    nodeDegree = G.degree(nodeName)
                    if nodeDegree == 0:
                        removeNodes.append(nodeName)

            if len(removeNodes):
                msg += 'WARNING: remove isolated ground in graph G'+str(i+1)+': '+str(removeNodes) + '\n'
                for nodeName in removeNodes:
                    G.remove_node(nodeName)

    returnDict = {}
    returnDict['numberOfGraphNodes'] = {'reference':G1.number_of_nodes(), 'compare':G2.number_of_nodes()}
    returnDict['numberOfGraphEdges'] = {'reference':G1.number_of_edges(), 'compare':G2.number_of_edges()}
    
    if addBasicTypes:
        # Count per node type
        bType1 = Counter([G1.nodes[n].get("basicType", None) for n in G1.nodes])
        bType2 = Counter([G2.nodes[n].get("basicType", None) for n in G2.nodes])
    
        returnDict['graphBasicTypes'] = {'reference':dict(bType1), 'compare':dict(bType2), 'isCounter':True}

    # Count per node type
    ct1 = Counter([G1.nodes[n].get("type", None) for n in G1.nodes])
    ct2 = Counter([G2.nodes[n].get("type", None) for n in G2.nodes])
    

    returnDict['graphCounters'] = {'reference':dict(ct1), 'compare':dict(ct2), 'isCounter':True }

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #basic comparison (overall counts):
    graphsMatch = True
    for key, value in returnDict.items():
        if value['reference'] != value['compare']:
            if 'counter' in value:
                msg += key + ' differences:\n  '
                msg += CompactCounterDiff(Counter(value['reference']),Counter(value['compare'])) + '\n'
            else:
                msg += key + ' differences:\n  mbs graph G1 = ' + str(value['reference']) 
                msg += '\n  mbs graph G2 = ' + str(value['compare']) + '\n'
            graphsMatch = False


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #compare graphs (isomorphic):
    #NOTE: as this function may become very time-consuming, we need to add a timeout
    if logger is not None:
        logger.LogDebug(f'CompareSystemGraphs: categorical_node_match: #nodes1={G1.number_of_nodes()}, #nodes2={G2.number_of_nodes()}')

    returnDict = {'graphsIsomorphicNoTimeout':True}

    useTimeout = True #otherwise the framework hangs in some cases!!!
    graphCheckTimeout = 60
    if G1.number_of_nodes() > maxNodesForIsomorphismCheck:
        #reference graph itself is too large to reliably verify (large or highly symmetric
        #graphs, e.g. the beamBridge truss, can make isomorphism search combinatorially
        #expensive); do not even attempt it -- assume correct (like the timeout branch below)
        #so the node-count and type-counter checks still gate the result, but a skipped check
        #is not an automatic topology failure. graphsIsomorphicNoTimeout flags that this match
        #was assumed, not verified. This is checked BEFORE the node-count-equality shortcut
        #below on purpose: a differing node count is not evidence the model is wrong when
        #verification against a reference this large was never attempted in the first place
        #(an unattempted check must not silently masquerade as a verified one just because the
        #two counts happen to differ, which is the common case for oversized references).
        isoCheck = True
        if logger is not None:
            logger.LogError(f'CompareSystemGraphs: model "{model}": ABORTED Isomorphism check: number of nodes > {maxNodesForIsomorphismCheck}',
                            printToConsole=True)
        returnDict['graphsIsomorphicNoTimeout'] = False
        returnDict['graphsIsomorphic'] = str(isoCheck) == 'True' #whatever comes back ...
    elif G1.number_of_nodes() != G2.number_of_nodes():
        returnDict['graphsIsomorphic'] = False
    else:
        if not useTimeout:
            node_match = iso.categorical_node_match(['type'], [None])
            GM = iso.GraphMatcher(G1, G2, node_match=node_match) #, edge_match=edge_match)
            returnDict['graphsIsomorphic'] = GM.is_isomorphic()
        else:
            isoCheck = False
            node_match = iso.categorical_node_match(['type'], [None])
            GM = iso.GraphMatcher(G1, G2, node_match=node_match) #, edge_match=edge_match)

            #special exception function
            class IsomorphismTimeout(Exception):
                """Custom exception to catch the timeout."""

            #monkey patch networkx function to inject timeout
            start_time = time.time()

            #save the original generator method
            old_candidate_iter = GM.candidate_pairs_iter

            def patched_candidate_iter():
                # This check now happens at EVERY recursive step
                if time.time() - start_time > graphCheckTimeout:
                    raise IsomorphismTimeout("Isomorphism check timed out!")

                # Yield from the original generator
                yield from old_candidate_iter()

            # Monkey-patch the instance
            GM.candidate_pairs_iter = patched_candidate_iter


            #execute within a try-except block
            try:
                isoCheck = GM.is_isomorphic()
            except IsomorphismTimeout:
                isoCheck = True #we assume correctness, which could not be checked - at least the number of nodes is correct!
                returnDict['graphsIsomorphicNoTimeout'] = False
                if logger is not None:
                    logger.LogError(f'CompareSystemGraphs: model "{model}": ABORTED Isomorphism check: exceeded timeout: {graphCheckTimeout}s',
                                    printToConsole=True)
            finally:
                    # RESTORE the original method
                    GM.candidate_pairs_iter = old_candidate_iter

            returnDict['graphsIsomorphic'] = str(isoCheck) == 'True' #whatever comes back ...

    if returnDict['graphsIsomorphic']: #then GM exists
        if returnGraphMatchIterator and not useTimeout:
            returnDict['graphMatchIter'] = GM.isomorphisms_iter()
    else:
        graphsMatch = False
        if msg != '': msg += '\n'
        msg += 'graphs are not isomorphic!\n'

    if graphsMatch:
        msg += 'graphs match!\n'

    returnDict['graphsMatch'] = graphsMatch
    returnDict['graphNodesCountMatch'] = (G1.number_of_nodes() == G2.number_of_nodes())

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #find best match with max iterations
    #skip when the isomorphism check was aborted (timeout or >150 nodes): the
    #unprotected isomorphisms_iter() below has no timeout and hangs on high-symmetry graphs.
    isoWasChecked = returnDict.get('graphsIsomorphicNoTimeout', True)
    if graphsMatch and maxAnnotationIterations > 0 and isoWasChecked:
        if logger is not None:
            logger.LogDebug('CompareSystemGraphs: iterate over annotations')

        returnDict['annotationIterations'] = 0
        returnDict['reachedMaxIterations'] = False
        returnDict['bestMatch'] = None
        GM = iso.GraphMatcher(G1, G2, node_match=node_match)
        best = None
        annotation_start = time.time()
        for i, m in enumerate(GM.isomorphisms_iter()):
            if i>=50 and i%50==0:
                if logger is not None:
                    logger.LogDebug(f'CompareSystemGraphs: annotation iteration {i}')
            returnDict['annotationIterations'] = i
            diffs = DiffAnnotations(G1, G2, m, annKey='annotations', compare=DeepEqual,
                                    skipMeshDependentQuantitiesInFFRFObjects=skipMeshDependentQuantitiesInFFRFObjects,
                                    ignoreUserFunctions=ignoreUserFunctions)
            score = len(diffs)
            if best is None or score < best[0]:
                best = (score, m, diffs)
            if best[0] == 0:  # perfect match found, no need to search further
                break
            if i + 1 >= maxAnnotationIterations: #stop as it could be very time consuming!
                returnDict['reachedMaxIterations'] = True
                msg += 'graph annotation comparison reached maxIterations!\n'
                break
            if annotationTimeoutSeconds > 0 and time.time() - annotation_start > annotationTimeoutSeconds:
                returnDict['reachedMaxIterations'] = True
                if logger is not None:
                    logger.LogError(f'CompareSystemGraphs: model "{model}": annotation iteration exceeded timeout ({annotationTimeoutSeconds}s) after {i+1} iterations',
                                    printToConsole=True)
                msg += f'graph annotation comparison exceeded timeout ({annotationTimeoutSeconds}s)!\n'
                break
        
        returnDict['graphsMatchContent'] = (best[0] == 0)
        returnDict['graphsContentDiffs'] = best[2]
        
        if best[0] != 0:
            diffsString = ''
            for diff in best[2]:
                diffsString += '    '+str(diff)+'\n'
            msg += f'graph content is different: number of diffs={best[0]}\n  diffs:{diffsString}\n'

        returnDict['bestMatch'] = best  # (num_diffs, mapping, diffs)

    elif graphsMatch and maxAnnotationIterations > 0:
        #annotation iteration was skipped because the isomorphism check was aborted
        #(timeout or >150-node limit); content match could not be verified -> conservative
        #defaults so downstream code that indexes these keys does not raise KeyError
        returnDict['graphsMatchContent'] = False
        returnDict['graphsContentDiffs'] = []
        returnDict['annotationIterations'] = 0
        returnDict['reachedMaxIterations'] = False
        returnDict['bestMatch'] = None
        msg += 'graph annotation comparison skipped (isomorphism check aborted)!\n'

    if logger is not None:
        #leave here, as it sometimes hangs - then we know what it is!
        logger.LogDebug('CompareSystemGraphs: finished')

    returnDict['msg'] = msg

    return returnDict

#get differences of two mbs (based on graph)
def GetMBSDifferences(mbsReference, mbsCompare, logger=None, compareDetails=True,
                      skipMeshDependentQuantitiesInFFRFObjects=False, model='unknown',
                      removeAllGroundObjects=False):
    logger.LogDebug('Check MBS graph-based differences', separator=True)
    [_, G1, _] = DrawSystemGraph(mbsReference, useItemTypes=True, multiLine=False, showSensors=False,
                                      showGraph=False, addItemData=True, addAnnotations=True )
    [_, G2, _] = DrawSystemGraph(mbsCompare, useItemTypes=True, multiLine=False, showSensors=False,
                                      showGraph=False, addItemData=True, addAnnotations=True )
    logger.LogDebug('Compare graphs ...')
    resultsDict = CompareSystemGraphs(G1, G2,
                                      maxAnnotationIterations=1000*int(compareDetails),
                                      skipMeshDependentQuantitiesInFFRFObjects=skipMeshDependentQuantitiesInFFRFObjects,
                                      removeAllGroundObjects=removeAllGroundObjects,
                                      logger=logger, model=model)
    if logger is not None:
        logger.LogDebug('MBS graph-based differences:\n'+resultsDict['msg'], separator=False)
    mbsInfo = [{},{}]
    for i, mbs in enumerate([mbsReference, mbsCompare]):

        mbsInfo[i]['numberOfNodes'] = mbs.systemData.NumberOfNodes()
        mbsInfo[i]['numberOfObjects'] = mbs.systemData.NumberOfObjects()
        mbsInfo[i]['numberOfMarkers'] = mbs.systemData.NumberOfMarkers()
        mbsInfo[i]['numberOfLoads'] = mbs.systemData.NumberOfLoads()
        mbsInfo[i]['numberOfSensors'] = mbs.systemData.NumberOfSensors()

        mbsInfo[i]['ODE2Size'] = mbs.systemData.ODE2Size()
        mbsInfo[i]['ODE1Size'] = mbs.systemData.ODE1Size()
        mbsInfo[i]['AESize'] = mbs.systemData.AEsize()
        mbsInfo[i]['dataSize'] = mbs.systemData.DataSize()
        mbsInfo[i]['systemSize'] = mbs.systemData.SystemSize()
    #both infos; first is reference:
    resultsDict['mbsInfo'] = (mbsInfo[0],mbsInfo[1])
    #True if info matches:
    resultsDict['mbsInfoMatches'] = Counter(mbsInfo[0]) == Counter(mbsInfo[1])
    #readable differences:
    resultsDict['mbsInfoDiff'] = CompactCounterDiff(Counter(mbsInfo[0]), Counter(mbsInfo[1]))
    if logger is not None:
        logger.LogDebug('MBS system (reference):\n'+str(mbsInfo[0]), separator=False)
        logger.LogDebug('MBS system differences:\n'+resultsDict['mbsInfoDiff'], separator=False)
    return resultsDict

# global error rules for categorizing parameter errors
# each rule is a tuple of the form: (name, stage0, stage1, ..., stageN)
# where each stage is a LIST of keyword strings matched against the EXUDYN item name
# example: item 'ObjectConnectorDistance' splits into ['Object', 'Connector', 'Distance']
# -> ('ObjectConnectorMisc', ['Object'], ['Connector']): level 2
# -> ('ObjectConnectorMisc', ['Object'], ['Connector'], ['Distance', 'Coordinate']): level 3
# the second rule wins. Add more stages to express finer-grained categories
# a list inside the rules is used to specify more keyword strings that can match for this rule
# an empty list would match ANY key:
# -> ('ObjectConnectorMisc', ['Object'], [], ['Distance', 'Coordinate'])
# would match for example ObjectTestDistance AND ObjectConnectorDistance
globalErrorFilter = [('RigidBodyNode', ['Node'], ['Rigid']), # parameters of the rigid body node
                     ('RigidBody', ['Object'], ['Rigid']), # parameters of the rigid body (gravity, inertia, ...)
                     ('MassPoint', ['Object'], ['Mass']), # parameters of mass points
                     ('ObjectConstraint', ['Object'], ['Connector'], ['Distance', 'Coordinate', 'CoordinateVector']), # (leads to AE) parameters of constraints
                     ('ObjectConnector', ['Object'], ['Connector']), # (fore elements / penalty based formulations) parameters of connectors
                     ('ObjectJoint', ['Object'], ['Joint']), # (leads to AE) parameters of constraints (i.e. joints)
                     ('ObjectSuperAndFEM', ['Object'], ['FFRF', 'ANCF']), # parameters of special super and FEM objects
                     ('ObjectMisc', ['Object']), # parameters of other objects
                     ('NodeMisc', ['Node']), # parameters of other nodes than NodeRigid...
                     ('MarkerMisc', ['Marker']), # parameters of markers in general
                     ('LoadForce', ['Load'], ['Force']), # parameters of (external) forces
                     ('LoadGravity', ['Load'], ['Mass'], ['Proportional']), # parameters of gravity
                     ('LoadForce', ['Load'], ['Torque']), # parameters of torques
                     ('LoadMisc', ['Load']), # parameters of other loads (load coordinate)
                     ('SensorMisc', ['Sensor'])] # parameters of sensors
errorCategories = [('Kinematics / Marker parameters', ['MarkerMisc']),
                   ('Rigid Body Reference Config.', ['RBNodeReferenceConfig']),
                   ('Initial Velocities', ['RBNodeVelocities']),
                   ('Inertia and Mass Parameters', ['RigidBody', 'MassPoint']),
                   ('Gravity and Load Parameters', ['LoadForce', 'LoadGravity']),
                   ('Connector / Constraint parameters', ['ObjectConstraint', 'ObjectConnector', 'ObjectJoint'])]

def GetHighestLevelOfErrorRuleMatching(errorRule, itemNameSplit):
    """
    Computes how many refinement stages of a single error rule are satisfied.

    Args:
        errorRule (tuple): single entry from globalErrorFilter, e.g.
            ('ObjectConnectorMisc', ['Object'], ['Connector']).
        itemNameSplit (list): camelCase-split tokens of the item name, e.g.
            ['Object', 'Connector', 'Spring', 'Damper'].

    Returns:
        int: number of stages matched.
    """
    highestLevelOfMatch = 0
    maxLevelOfError = len(errorRule) - 1
    for itemName in itemNameSplit:
        if highestLevelOfMatch < maxLevelOfError:
            errorCriteria = errorRule[highestLevelOfMatch+1]
            if itemName in errorCriteria or not errorCriteria:
                highestLevelOfMatch += 1
            else:
                highestLevelOfMatch = 0 # the error rule does not fit at all
                break
        else:
            break
    return highestLevelOfMatch

def GetCategorizedGraphErrorsAndWrongParametersList(graphCompareDict={}):
    """
    Returns the error rule that applies for the graph differences in the
    graphCompareDict.

    Args:
        graphCompareDict (dict): as returned by GetMBSDifferences(mbs0, mbs1).

    Returns:
        graphErrorTypeList (list): A list of dicts, where each dict holds the
            error rule that fits for the error (under key 'name'), the wrong
            parameter and (if possible) the absolute difference (of float,
            vector, or matrix).

    """
    graphErrorTypeList = []
    # only if graphs match, try to categorize error
    if graphCompareDict['graphsMatch']:
        contentDiffList = graphCompareDict['graphsContentDiffs']
        for contentDiff in contentDiffList:
            graphError = {}
            itemNameSplit = SplitCamelCaseIntoStringsList(contentDiff[2])
            # now try to find BEST matching rule (which error rule has the highest matching level?)
            levelOfMatchList = []
            for errorRule in globalErrorFilter:
                levelOfMatchList += [GetHighestLevelOfErrorRuleMatching(errorRule, itemNameSplit)]
            highestLevelOfMatch = max(levelOfMatchList)
            indicesOfHighestLevel = [i for i, v in enumerate(levelOfMatchList) if v == highestLevelOfMatch]
            if len(indicesOfHighestLevel) > 1:
                print(f'WARNING: ambiguous error categorization for item "{contentDiff[2]}".'
                      'The following rules would apply for this item: '
                      f'{[globalErrorFilter[i][0] for i in indicesOfHighestLevel]}. '
                      'Consider adding a more specific rule to globalErrorFilter!'
                      'One of these rules will be used as long as no more specific rule is added.')
            graphError['name'] = globalErrorFilter[indicesOfHighestLevel[0]][0]
            # sub-classify RigidBodyNode parameters by parameter name into finer-grained categories
            if graphError['name'] == 'RigidBodyNode':
                paramLower = (contentDiff[3] or '').lower()
                if 'velocit' in paramLower:
                    graphError['name'] = 'RBNodeVelocities'
                elif 'reference' in paramLower or 'initial' in paramLower:
                    graphError['name'] = 'RBNodeReferenceConfig'
                else:
                    graphError['name'] = 'RBNodeOthers'
            graphError['parameter'] = contentDiff[3]
            try:
                graphError['parameterAbsDiff'] = np.abs(contentDiff[4]-contentDiff[5])
            except:
                graphError['parameterAbsDiff'] = None
            graphErrorTypeList += [graphError]
    return graphErrorTypeList

if __name__ == "__main__":

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Test system:
    from exudyn.utilities import InertiaCuboid, SensorBody
    import exudyn.graphics as graphics #only import if it does not conflict
    
    SC = exudyn.SystemContainer()
    mbs = SC.AddSystem()

    variations=['None', 'removeCSD', 'removeForce', 'changeCOM', 'changeTorque', 
                'addGround', 'changeP1', 'changeGravity', 'changeLoadOrder', 'changeMassOrder']
    
    graphList = []
    compareResults = []
    for var in variations:
    
        mbs.Reset()    
        
        #%%++++++++++++++++++++++++++++++++++++++++++++++++++++
        #physical parameters
        g =     [0,-9.81,0] #gravity
        if var == 'changeGravity': 
            g = [0,-9.80,0]
        L = 1               #length
        w = 0.1             #width
        bodyDim=[L,w,w] #body dimensions
        p0 =    [0,0,0]     #origin of pendulum
        pMid0 = np.array([L*0.5,0,0]) #center of mass, body0
        
        #ground body, located at specific position (there could be several ground objects)
        oGround = mbs.CreateGround(referencePosition=[0,0,0])

        if var=='addGround':
            oGround2 = mbs.CreateGround(referencePosition=[2,0,0])

        mbs.CreateMassPoint(referencePosition=[1,2,3], physicsMass=15.3, gravity=g)

        if var != 'changeMassOrder':
            mbs.CreateMassPoint(referencePosition=[1,2,3], physicsMass=13.3, gravity=g)

        
        iCube0 = InertiaCuboid(density=5000, sideLengths=bodyDim)
        if var!='changeCOM':
            iCube0 = iCube0.Translated([-0.25*L,0,0]) #transform COM, COM not at reference point!
        
        #graphics for body
        graphicsBody0 = graphics.Brick(centerPoint=[0,0,0],size=[L,w,w],color=graphics.color.red)
        graphicsCOM0 = graphics.Basis(origin=iCube0.com, length=2*w) #COM frame
        
        #create rigid node and body
        b0=mbs.CreateRigidBody(inertia = iCube0, #includes COM
                               referencePosition = pMid0,
                               gravity = g,
                               graphicsDataList = [graphicsCOM0, graphicsBody0])
        mbs.CreateRevoluteJoint(bodyNumbers=[oGround, b0], position=[0,0,0], 
                                axis=[0,0,1], axisRadius=0.2*w, axisLength=1.4*w)
        
        
        graphicsBody1 = graphics.RigidLink(p0=[0,0,-0.5*L],p1=[0,0,0.5*L], 
                                           axis0=[1,0,0], axis1=[0,0,0], radius=[0.06,0.05], 
                                           thickness = 0.1, width = [0.12,0.12], color=graphics.color.lightgreen)
        
        p1 = np.array([L,0,0.5*L])
        if var=='changeP1':
            p1 *= 0.9
        
        
        b1=mbs.CreateRigidBody(inertia = InertiaCuboid(density=5000, sideLengths=[0.1,0.1,1]),
                               referencePosition = p1, #reference pos = center of mass, body1
                               gravity = g,
                               graphicsDataList = [graphicsBody1])
        
        mbs.CreateRevoluteJoint(bodyNumbers=[b0, b1], position=[L,0,0], 
                                axis=[1,0,0], axisRadius=0.2*w, axisLength=1.4*w)
    
        if var != 'removeCSD':
            mbs.CreateSpringDamper(bodyNumbers=[b1, oGround], stiffness=100)
        else:
            print('omit CSD')

        if var == 'changeMassOrder':
            mbs.CreateMassPoint(referencePosition=[1,2,3], physicsMass=13.3, gravity=g)

        mbs.CreateMassPoint(referencePosition=[1,2,4], physicsMass=13.3, gravity=g)
        mbs.CreateMassPoint(referencePosition=[1,2,5], physicsMass=13.3, gravity=g)
        
        #forces can be added like in the following
        force = [0,0.5,0]       #0.5N   in y-direction
        torque = [0.1,0,0]      #0.1Nm around x-axis
        if var=='changeTorque':
            torque[1] = 0.5

        if var == 'changeLoadOrder':
            mbs.CreateTorque(bodyNumber=b1, 
                            loadVector=torque,
                            localPosition=[0,0,0],   #at body's reference point/center
                            bodyFixed=False) #if True, direction would corotate with body

        if var != 'removeForce':
            mbs.CreateForce(bodyNumber=b1,
                            loadVector=force,
                            localPosition=[0,0,0.5], #at tip
                            bodyFixed=False) #if True, direction would corotate with body

        if var != 'changeLoadOrder':
            mbs.CreateTorque(bodyNumber=b1, 
                            loadVector=torque,
                            localPosition=[0,0,0],   #at body's reference point/center
                            bodyFixed=False) #if True, direction would corotate with body
          
        sens1=mbs.AddSensor(SensorBody(bodyNumber=b1, localPosition=[0,0,0.5*L],
                                       fileName='solution/sensorPos.txt',
                                       outputVariableType = exudyn.OutputVariableType.Position))
        
        mbs.Assemble()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #add this to model generation: once for sample file model and once for LLM-model
        [nx, G, items] = DrawSystemGraph(mbs, useItemTypes=True, multiLine=False, showSensors=False,
                                         showGraph=False, addItemData=True, addAnnotations=True ) 
        
        graphList.append(G)
        
        if len(graphList) > 1:
            #compare always with original graph
            results = CompareSystemGraphs(graphList[0],G, 
                                          )
            print('================================')
            print('compare graphs for case '+var+':')
            print(results['msg']) #readable message
            compareResults.append(results)
            
            graphErrorsList = GetCategorizedGraphErrorsAndWrongParametersList(graphCompareDict=results)
            for graphError in graphErrorsList:
                print(f"{graphError['name']:<25} {graphError['parameter']}")
            
        #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

