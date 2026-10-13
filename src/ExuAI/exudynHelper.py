#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: evalHelper.py
# 
# Details:  this file serves as a documented example that shall be used by 
#           LLMs to update information on their internal knowledge of Exudyn;
#           the text is split by #%% comments, including tags for respective functionality
#
# Author:   Johannes Gerstmayr
# Date:     2024-12-30
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#difficulty levels:
#0..tags that should always be included
#5..ground + mass points + springs + distance, 10..rigid bodies, 15..rigid body joints, 
#20..Wheel, ### <= 20 = limit of "simpleMBS2026"; <= 25 = limit of "simpleMBS2025"
#30..torsional spring damper (control) ###start of 2026 "mediumMBS" <= 35
#35..contact (sphere, quad)
#40..flexible bodies (FFRF)
#45..beams
import exudyn.graphics as graphics
  
#%%
#TAG: generalrigid
#DIFFICULTY: 10000
#rotations are usually given as rotation matrix (3x3 numpy array); you can use RotationVector2RotationMatrix([rotX, rotY, rotZ]) which returns a rotation matrix from a global x,y,z rotation vector
#for working with rigid bodies, note that there is always a local coordinate system in the body, which means that forces or joints are attached at local positions (and local coordinates) in the body.

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++
#TAG: general import
#DIFFICULTY: 0
#INFO: Import Exudyn library and utilities.
import exudyn as exu
from exudyn.utilities import *
import numpy as np

#set up new multibody system to work with
SC = exu.SystemContainer()
mbs = SC.AddSystem()

# add items here

#%%
#TAG: ground
#DIFFICULTY: 5
#USES: CreateGround
#INFO: Create a ground object as an inertial reference frame for the multibody system; needed whenever something is fixed, anchored or connected to ground. 
#Create a ground object at given (optional) referencePosition. While one can use several ground objects, it is highly recommended to use only a single ground object located at the origin.
oGround = mbs.CreateGround(referencePosition=[0,0,0])

#%%
#TAG: point mass
#DIFFICULTY: 5
#USES: CreateMassPoint
#INFO: Create a point mass object with specified mass at reference position with optional initial conditions.
#physicsMass: the mass in kg
#referencePosition: initial/reference position at which mechanism is defined; 
#initialDisplacement: additional initial deviations from referencePosition; usually this is [0,0,0]
#initialVelocity: initial velocities of mass point
#all vectors always have 3 components, no matter if 2D or 3D problem
oMass = mbs.CreateMassPoint(physicsMass=5, referencePosition=[1,0,0], 
                            initialDisplacement=[0,0,0],  #optional, relative to reference position
                            initialVelocity=[0,0.5,0],    #optional
                            gravity=[0,-9.81,0])          #optional

#%%
#TAG: visualization
#DIFFICULTY: 10000
mbs.SetNodeParameter(mbs.GetObject(oMass)['nodeNumber'],'VdrawSize',0.3)
mbs.SetNodeParameter(mbs.GetObject(oMass)['nodeNumber'],'Vcolor',graphics.color.orange)

#%%
#TAG: spring damper
#DIFFICULTY: 5
#REQUIRES: point mass, ground
#USES: CreateSpringDamper
#INFO: Create a linear spring-damper system between two bodies or between a body and the ground.
#NOTE: localPosition0 resp. localPosition1 are the positions relative to body0 resp. body1 (or ground); use [0,0,0] for local position on point masses; on ground, localPosition depends on position where the spring-damper should be anchored
oSpringDamper = mbs.CreateSpringDamper(bodyNumbers=[oGround, oMass], #[body0,body1]
                                      localPosition0=[0,0,0], #in body0 local coordinates
                                      localPosition1=[0,0,0], #in body1 local coordinates
                                      referenceLength=None, #usually set to None (default) => takes the (relaxed) length in reference configuration
                                      stiffness=1e2, damping=1)
#%%
#TAG: distance constraint
#DIFFICULTY: 5
#REQUIRES: point mass, ground
#USES: CreateDistanceConstraint
#INFO: Create a rigid distance constraint between two points on different bodies or between a body and the ground.
#NOTES: (previous note): create a distance constraint; the localPosition (0 or 1) is used to specify the local attachment position of the constraint relative to the body's (or ground's) reference point; for mass points localPosition is [0,0,0]; use distance=None as it is computed automatically (safer)
#NOTE: localPosition0 resp. localPosition1 are the positions relative to body0 resp. body1 (or ground); use [0,0,0] for local position on point masses; for a ground, localPosition depends on position where the distance constraint should be anchored; use distance=None as it is computed automatically (safer)
oDistance = mbs.CreateDistanceConstraint(bodyNumbers=[oGround, oMass], #[body0,body1]
                                         localPosition0 = [0,0,0], #in body0 local coordinates
                                         localPosition1 = [0,0,0], #in body1 local coordinates
                                         distance=None)
#%%
#TAG: rigid body omniform
#DIFFICULTY: 10
#USES: CreateRigidBody, RigidBodyInertia
#INFO: Create a rigid body with an arbitrary shape - use this in case that no other rigid body shape fits.
#create an Exudyn inertia object including mass, inertia tensor, and given center of mass (com) position.
inertiaCube = RigidBodyInertia(mass=2, inertiaTensor=np.diag([8,8,1]),
                               com=[0,0,0], inertiaTensorAtCOM=True)

#create a free rigid body using the defined inertia properties and applies gravity (y-direction).
oBody = mbs.CreateRigidBody(inertia=inertiaCube,
                            referencePosition=[0,0,0], #reference position x/y/z of COM
                            initialVelocity=[0,0,0],   #optional
                            gravity=[0,-9.81,0])
#%%
#TAG: rigid body sphere
#DIFFICULTY: 10
#USES: CreateRigidBody, InertiaSphere
#INFO: Create a rigid body with a spherical shape.
#defines the inertia with mass of sphere and radius; inertia w.r.t. center of mass, com=[0,0,0].
inertiaSphere = InertiaSphere(mass=2, radius=0.5)

#create a free rigid body using the defined inertia properties and applies gravity (y-direction).
oBody = mbs.CreateRigidBody(inertia=inertiaSphere,
                            referencePosition=[0,0,0], #reference position x/y/z of COM
                            initialVelocity=[0,0,0],
                            gravity=[0,-9.81,0])

#%%
#TAG: rigid body cylinder
#DIFFICULTY: 10
#USES: CreateRigidBody, InertiaCylinder
#INFO: Create a rigid body with a cylindrical shape (for discs, wheels, etc.).
# defines the inertia with density, length, outerRadius, axis, innerRadius of the cylinder; axis defines the orientation of the cylinder axis (0=x-axis, 1=y-axis, 2=z-axis); for hollow cylinder use innerRadius != 0; inertia w.r.t. center of mass, com=[0,0,0].
mass = 10 
outerRadius = 0.45
length = 2
volume = np.pi * outerRadius**2 * length
inertiaCylinder = InertiaCylinder(density=mass/volume, length=length, outerRadius=outerRadius, axis=0)

#create a free rigid body using the defined inertia properties and applies gravity (y-direction).
oBody = mbs.CreateRigidBody(inertia=inertiaCylinder,
                            referencePosition=[3,0,0], #reference position x/y/z of COM
                            initialVelocity=[0,0,0],   #optional
                            initialAngularVelocity=[2*np.pi,0,0], #optional
                            gravity=[0,-9.81,0])       #optional

#%%
#TAG: visualization
#DIFFICULTY: 10000
#create a rectangular graphics object with given center point (relative to rigid body)
graphicsCyl = graphics.Cylinder(pAxis=[-length*0.5,0,0], vAxis=[length+0.1,0,0], radius=outerRadius, 
                                color=graphics.color.green)

mbs.SetObjectParameter(oBody,'VgraphicsData',[graphicsCyl])
mbs.SetObjectParameter(oBody,'Vshow',True)
#%%
#TAG: rigid body brick-shape
#DIFFICULTY: 18
#USES: CreateRigidBody, InertiaCuboid
#INFO: Create a rigid body with a brick (cuboid) shape; uses rotated reference configuration.
rbX = 2
rbY = 0.5
rbZ = 0.2

#create inertia instance for cuboid (brick) shape
mass=10 #kg
volume=rbX*rbY*rbZ
inertiaCube2 = InertiaCuboid(density=mass/volume, sideLengths=[rbX,rbY,rbZ])
inertiaCube2 = inertiaCube2.Translated([0.5,0,0]) #translate COM (in body-frame), only if needed

#create a free rigid body with defined inertia and applies an initial rotation and velocity.
oBody2 = mbs.CreateRigidBody(inertia = inertiaCube2,
                             referencePosition = [4,-0.5*rbX,0], #reference position, not COM
                             referenceRotationMatrix = RotationVector2RotationMatrix([0,0,-0.5*np.pi]),
                             initialAngularVelocity = [0,0,0.2*2*np.pi],
                             initialVelocity = [0.2*np.pi*0.5*rbX,0,0],
                             gravity = [0,-9.81,0])

#%%
#TAG: visualization
#DIFFICULTY: 10000
#create a rectangular graphics object with given center point (relative to rigid body)
graphicsCube = graphics.Brick(centerPoint=[0.5,0,0], size=[rbX,rbY,rbZ], color=graphics.color.blue)

mbs.SetObjectParameter(oBody2,'VgraphicsData',[graphicsCube])
mbs.SetObjectParameter(oBody2,'Vshow',True)

#%%
#TAG: force for point mass
#REQUIRES: point mass
#DIFFICULTY: 5
#USES: CreateForce, CreateMassPoint
#INFO: Applies a force in a specific direction to a point mass.
#apply 10N in x-direction on point mass with index oMass
loadMassPoint = mbs.CreateForce(bodyNumber=oMass, loadVector=[10,0,0])

#%%
#TAG: force for body
#DIFFICULTY: 10
#REQUIRES: rigid body brick-shape
#USES: CreateForce, CreateRigidBody
#INFO: Applies a force in a specific direction to a body at its local position.
#apply 10N in x-direction:
loadRigidBody = mbs.CreateForce(bodyNumber=oBody, localPosition=[0,0,0], loadVector=[10,0,0])
#%%

#TAG: force with time-dependent user function
#REQUIRES: rigid body brick-shape
#USES: CreateForce, loadVectorUserFunction
#DIFFICULTY: 10
#INFO: Defines a time-dependent force function and applies it to a rigid body or mass point at a specific local position.
def UFforce(mbs, t, loadVector):
    return (10+5*np.sin(t*10*2*np.pi))*np.array([0,5,0])

mbs.CreateForce(bodyNumber=oBody,
                localPosition=[0,1.2,0.5], #position at body
                loadVectorUserFunction=UFforce)
#%%
#TAG: torque with user function
#DIFFICULTY: 10
#REQUIRES: rigid body cylinder
#USES: CreateTorque, loadVectorUserFunction
#INFO: Defines a user function of a time-dependent 3D torque function and applies it to a rigid body.
#define user function; take care of args; loadVector is passed through
def UserFunctionTorque(mbs, t, loadVector):
    return np.cos(t*2*np.pi)*np.array(loadVector)

#add torque with user function
mbs.CreateTorque(bodyNumber=oBody, loadVector=[5,0,0], 
                 loadVectorUserFunction=UserFunctionTorque)

#%%
#TAG: torque
#DIFFICULTY: 10
#REQUIRES: rigid body brick-shape
#USES: CreateTorque
#INFO: Applies a 3D torque to a rigid body.
#torque of 1Nm around the y-axis:
mbs.CreateTorque(bodyNumber=oBody, loadVector=[0,1,0]) #torque of 1Nm around y-axis

#%%
#TAG: spherical joint
#DIFFICULTY: 15
#REQUIRES: ground, point mass
#USES: CreateSphericalJoint
#INFO: Create a spherical joint between two bodies/ground.
oMassPoint = mbs.CreateMassPoint(physicsMass=2, referencePosition=[1,0,0])

#create spherical joint between ground and mass point; could also be applied to two bodies; possible bodies: mass point or rigid body
mbs.CreateSphericalJoint(bodyNumbers=[oGround, oMassPoint],
                         position=[1,0,0]) #global position of joint (in reference configuration) 

#%%
#TAG: coordinate constraint
#DIFFICULTY: 15
#REQUIRES: ground
#USES: CreateCoordinateConstraint
#INFO: Create a coordinate constraint between a body and ground (to fix a coordinate like x-displacement to ground), or to constrain two coordinates between two bodies (thus, coordinate difference becomes zero).
oMassPoint2 = mbs.CreateMassPoint(physicsMass=25, referencePosition=[2,2,0])

#create single coordinate constraint between ground and mass point:
#bodyNumbers is a list of 2 body numbers (or None for constraint to ground); 
#coordinates is a list of 2 coordinate numbers of the respective body in bodyNumbers, or None for ground
##NOTE: constrains only the displacement coordinate, not the reference position
mbs.CreateCoordinateConstraint(bodyNumbers=[oMassPoint2, None], #connect mass point with ground (None)
                               coordinates=[1, None]) #connect y-coordinate (=1) of mass point with ground (=None)

#%%
#TAG: prismatic joint
#DIFFICULTY: 15
#REQUIRES: ground, rigid body brick-shape
#USES: CreatePrismaticJoint
#INFO: Create a prismatic joint between two rigid bodies / ground, allowing linear motion along a specified axis; position and axis are defined in global reference configuration.
mbs.CreatePrismaticJoint(bodyNumbers=[oGround, oBody], 
                         position=[3,0,0], #global position of joint
                         axis=[1,0,0], #global axis of joint, can move in global x-direction
                         useGlobalFrame=True) #True=use global coordinates for joint definition
#%%
#TAG: revolute joint
#DIFFICULTY: 15
#REQUIRES: rigid body brick-shape, rigid body cylinder
#USES: CreateRevoluteJoint
#INFO: Create a revolute joint between two rigid bodies / ground, allowing rotation about a specified axis; not compatible with point mass.
mbs.CreateRevoluteJoint(bodyNumbers=[oBody, oBody2], 
                        position=[4,-0.5,0], #global position of joint
                        axis=[0,0,1], #rotation along global z-axis
                        useGlobalFrame=True) #True=use global coordinates for joint definition

#%%
#TAG: generic joint
#DIFFICULTY: 15
#REQUIRES: rigid body brick-shape, rigid body cylinder
#USES: CreateGenericJoint
#INFO: Create a generic or fixed joint between two bodies / ground, often used to constrain all six axes, i.e. fix a flexible body to ground, or to realize a joint with special axes (universal joint, etc.); not compatible with point mass.
#note on constrainedAxes: this represents a 6-dimensional list to turn on/off constraints on 3 translational (t) and 3 rotational (r) axes with the following order [tx, ty, tz, rx, ry, rz]; 0 = free, 1 = constrained
mbs.CreateGenericJoint(bodyNumbers=[oBody, oBody2],
                       position=[4, -0.5, 0], #global position of joint
                       constrainedAxes=[1, 1, 0, 1, 0, 1], #constrain translation x,y and rotations x,z
                       useGlobalFrame=True) #True=use global coordinates for joint definition

#%%
#TAG: torsional spring-damper
#DIFFICULTY: 15
#REQUIRES: rigid body brick-shape, rigid body cylinder
#USES: CreateTorsionalSpringDamper
#INFO: Create a torsional spring-damper between two bodies / ground, ideally in combination with a revolute joint; used for PD control at revolute joints.

#user function which get current scalar rotation and angularVelocity, as well as stiffness, damping, offset copied from TorsionalSpringDamper
def UFtorsionalSpring(mbs, t, itemNumber, rotation, angularVelocity, stiffness, damping, offset):
    delta = rotation - offset #offset could be used as target value for control; note the sign as it is a spring
    return stiffness * delta + angularVelocity * damping #replicates behavior of torsional spring damper

mbs.CreateTorsionalSpringDamper(bodyNumbers=[oBody, oBody2],
                                position=[4,-0.5,0], #global position of spring-damper
                                axis=[0,0,1],        #global rotation axis
                                stiffness=1000,
                                damping=20,
                                offset=0,            #target / set value
                                springTorqueUserFunction=UFtorsionalSpring, #optional; otherwise 0
                                useGlobalFrame=True) #True=use global coordinates for joint definition

#%%
#TAG: rolling disc with penalty-based joint
#DIFFICULTY: 20
#REQUIRES: ground
#USES: CreateRollingDiscPenalty
#INFO: Create a rigid body of a cylindrical disc (or wheel), and a rolling disc joint between the disc and the ground (ground must be first body); ground is defined by planePosition and planeNormal and disc joint is defined in body-local coordinates with axisPosition and axisVector; uses penalty formulation with linear contact and damping model.
mass = 5 
mu = 0.2 #static friction
rDisc = 0.5
length = 0.1
volume = np.pi * rDisc**2 * length
inertiaCylinder = InertiaCylinder(density=mass/volume, length=length, outerRadius=rDisc, axis=0)

#create a free rigid body using the defined inertia properties and applies gravity (y-direction).
oDisc = mbs.CreateRigidBody(inertia=inertiaCylinder,
                            referencePosition=[0,2,rDisc], #reference position x/y/z of COM
                            initialVelocity=[0,2*np.pi*rDisc,0],   
                            initialAngularVelocity=[-2*np.pi,0.2,0],
                            gravity=[0,0,-9.81])

#create a 'rolling' joint between flat ground defined by plane parameters, lying on oGround, and rigid body given as oDisc:
mbs.CreateRollingDiscPenalty(bodyNumbers=[oGround, oDisc], #ground must be first body number
                             discRadius = rDisc, 
                             axisPosition=[0,0,0], axisVector=[1,0,0], #relative to oDisc
                             planePosition = [0,0,0], planeNormal = [0,0,1],
                             contactStiffness = 1e4, contactDamping = 2e2, 
                             dryFriction = [mu,mu]) #friction in forward and lateral direction

#%%
#TAG: ideal rolling disc joint
#DIFFICULTY: 20
#REQUIRES: ground
#USES: CreateRollingDisc
#INFO: Create a rigid body of a cylindrical disc (or wheel), and a joint between the rolling disc and the ground (ground must be first body); ground is defined by planePosition and planeNormal and disc joint is defined in body-local coordinates with axisPosition and axisVector.
mass = 5 
rDisc = 0.5
length = 0.1
volume = np.pi * rDisc**2 * length
inertiaCylinder = InertiaCylinder(density=mass/volume, length=length, outerRadius=rDisc, axis=0)

#create a free rigid body using the defined inertia properties and applies gravity (y-direction).
oDisc = mbs.CreateRigidBody(inertia=inertiaCylinder,
                            referencePosition=[1,2,rDisc], #reference position x/y/z of COM
                            initialVelocity=[0,2*np.pi*rDisc,0],   
                            initialAngularVelocity=[-2*np.pi,0.2,0],
                            gravity=[0,0,-9.81])

#create a 'rolling' joint between flat ground defined by plane parameters, lying on oGround, and rigid body given as oDisc:
mbs.CreateRollingDisc(bodyNumbers=[oGround, oDisc],  #ground must be first body number
                      discRadius = rDisc, 
                      axisPosition=[0,0,0], axisVector=[1,0,0], #relative to oDisc
                      planePosition = [0,0,0], planeNormal = [0,0,1], #defines plane
                      constrainedAxes=[1,1,1] #constrain 3 axes: lateral motion, forward motion and normal contact
                      )

#%%
#TAG: sphere-sphere contact
#DIFFICULTY: 35
#REQUIRES: point mass
#USES: CreateSphereSphereContact
#INFO: Create a contact object between two spherical rigid bodies or point masses, using contact stiffness and restitution coefficient.

#add second colliding mass
oMass2 = mbs.CreateMassPoint(physicsMass=50, referencePosition=[0,-1,0])

#create a sphere-sphere contact between two point mass objects with given contact parameters; impactModel must be 2
mbs.CreateSphereSphereContact(bodyNumbers=[oMass, oMass2],
                              spheresRadii =[0.15,0.2], #radii of spheres for first and second body
                              contactStiffness=1e4,
                              dynamicFriction=0, #friction parameter for rigid bodies; must be zero for point masses
                              impactModel = 2, #MUST ALWAYS be 2 to use restitution coefficient
                              restitutionCoefficient=0.7)

#%%
#TAG: visualization
#DIFFICULTY: 10000
mbs.SetNodeParameter(mbs.GetObject(oMass2)['nodeNumber'],'VdrawSize',0.4)
mbs.SetNodeParameter(mbs.GetObject(oMass2)['nodeNumber'],'Vcolor',graphics.color.dodgerblue)

#%%
#TAG: sphere-quad contact
#DIFFICULTY: 35
#REQUIRES: ground, sphere-sphere contact
#USES: CreateSphereQuadContact
#INFO: Create a contact object between a spherical rigid body or point mass and a plane (usually ground) defined by 4 points ("quad"), a sphere-plane contact, using contact stiffness and restitution coefficient.

#define arbitrary plane using list of four 3D points counter-clockwise seen from contacting object, here x-y plane (z=0) with size 1 x 1:
quadPoints = [[-0.5,-0.5,0],[0.5,-0.5,0],
              [0.5,0.5,0],[-0.5,0.5,0]]
#add contact between spherical body and quad plane:
mbs.CreateSphereQuadContact(bodyNumbers=[oMass2, oGround],
                            quadPoints=quadPoints,
                            radiusSphere=0.2,
                            contactStiffness = 1e5,
                            dynamicFriction=0., #friction != 0 only for rigid bodies
                            impactModel = 2, #MUST ALWAYS be 2 to use restitution coefficient
                            restitutionCoefficient = 0.8,
                            )


#%%
#TAG: flexible body
#DIFFICULTY: 40
#USES: CreateFFRFReducedOrderObject
#INFO: Create a flexible body within a multibody system, assuming you have the fem interface of it available.
#load FEM module and the available fem interface by its according name
from exudyn.FEM import *
femInterface = FEMinterface()
femInterface.LoadFromFile('nameOfFemInterface.npz')
#create the flexible body and give it a unique name
partCreateDict = mbs.CreateFFRFReducedOrderObject(name='part0',
                                                  femInterface=femInterface,
                                                  referencePosition=[0., 0., 0.],
                                                  initialVelocity=[0., 0., 0.],
                                                  referenceRotationMatrix=np.eye(3),
                                                  initialAngularVelocity=[0., 0., 0.],
                                                  massProportionalDamping=0.,
                                                  stiffnessProportionalDamping=0.,
                                                  gravity=[0., 0., 0.])
#to get the marker numbers for connecting the flexible body to other parts of the multibody system, use the name of the flexible body and the name of its interfaces
m0 = partCreateDict['part0:interfaceName0']
m1 = partCreateDict['part0:interfaceName1']

# #%%
#TAG: visualization
#DIFFICULTY: 10000
#create flat ground
graphicsGround = graphics.CheckerBoard(point=[0,2,0], size=2)

mbs.SetObjectParameter(oGround,'VgraphicsData',[graphicsGround])
mbs.SetObjectParameter(oGround,'Vshow',True)

#%%
#TAG: cartesian spring damper
#REQUIRES: ground, rigid body brick-shape
#DIFFICULTY: 10
#USES: CreateCartesianSpringDamper
#INFO: Create a special spring-damper which acts on translations in global x, y, and z-direction, with possibility to connect rigid bodies, mass points together or on ground.
mbs.CreateCartesianSpringDamper(
    bodyNumbers=[oGround, oBody2], #[body0,body1]
    localPosition0=[4,-0.5*rbX,0], #for body0
    localPosition1=[0,0,0],        #for body1
    stiffness = [100,10,10], #x,y,z stiffness
    damping = [5,2,2], #x,y,z damping
)

#%%
#TAG: rigid body spring damper
#REQUIRES: ground, rigid body brick-shape
#NOTES: not used so far, therefore difficulty >> high
#DIFFICULTY: 10000
#USES: CreateRigidBodySpringDamper
#INFO: Create a six-degree-of-freedom spring-damper between two rigid bodies or a rigid body and ground.
stiffness = np.zeros((6,6))
damping = np.zeros((6,6))

# Set rotational stiffness and damping around z-axis
stiffness[5,5] = 2e3
damping[5,5] = 1e1

mbs.CreateRigidBodySpringDamper(
    bodyNumbers=[oGround, oBody2],
    localPosition0=[4,-0.5*rbX,0],
    stiffness=stiffness, 
    damping=damping
)

#%%
#TAG: assemble
#DIFFICULTY: 0
#INFO: Assemble has to be called just before solving or system analysis.
mbs.Assemble()

#%%
#TAG: simulationPure
#DIFFICULTY: 10000
endTime = 2 #simulation time in seconds
stepSize = 0.002 #should be small enough to achieve sufficient accuracy

#some simulation parameters:
simulationSettings = exu.SimulationSettings() #takes currently set values or default values
simulationSettings.timeIntegration.numberOfSteps = int(endTime/stepSize)
simulationSettings.timeIntegration.endTime = endTime
simulationSettings.timeIntegration.verboseMode = 1

#for redundant constraints, use these two settings
simulationSettings.linearSolverSettings.ignoreSingularJacobian=True
simulationSettings.linearSolverType = exu.LinearSolverType.EigenDense #use EigenSparse for larger systems alternatively
#simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
# simulationSettings.timeIntegration.newton.useModifiedNewton = True

mbs.SolveDynamic(simulationSettings)

#%%
#TAG: visualization
#DIFFICULTY: 10000
SC.visualizationSettings.nodes.drawNodesAsPoint = False
SC.visualizationSettings.nodes.showBasis = True

mbs.SolutionViewer()
