#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: modelDescriptions.py
#
# Details:  Lists, dictionaries and lists used by agent
#
# Authors:  Johannes Gerstmayr, Tobias Möltner, Sebastian Weyrer
# Date:     2025-02-25
# Notes:    Collection of modeldescriptions and according possible parametrizations.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#%%+++++++++++++++++++++++++++++++++++++++++++++++++++
import exudyn as exu
exu.RequireVersion('1.10.150') #if according version is not available, tests will fail.

try: 
    from .utilities import Arange
except: 
    from utilities import Arange
    
multChar = '*'
groundPos = '(0, 0, 0)'

# difficulty levels are defined in exudynHelper.py
gravityList = [3.73, 9.81, 11.15]

def AddTaskTypeToDescription(simpleMBSDescriptions={}, geometryDescriptions={},
                             flexibleMBSDescriptions={}):
    for task, data in simpleMBSDescriptions.items():
        data['taskType'] = 'MBS'
    for task, data in geometryDescriptions.items():
        data['taskType'] = 'part'
    for task, data in flexibleMBSDescriptions.items():
        data['taskType'] = 'fMBS'

#%% simple mbs descriptions
simpleMBSDescriptions = {
    # 1) JG
    "flyingMassPoint": {
        "description": "Projectile motion of a point mass with the following properties: mass m = {mass} kg, gravity g = {gravity} m/s^2, initial velocity in x/y/z-direction: vx = {vx} m/s, vy = {vy} m/s, vz = 0 m/s. The initial position is given as x=0 and y=0. Gravity acts along the negative y-axis, and there is no other external propulsion or resistance. The motion shall be evaluated for 1 s. Contact with ground is not modelled, so the projectile can go below y=0. Except the action of gravity, no forces act on the point mass.",
        "parameters": {
            "mass": {"default": 10, "range": [5, 15]},
            "gravity": {"default": 9.81, "list": gravityList},
            "vx": {"default": 5, "range": [0, 20]},
            "vy": {"default": 1, "range": [10, 25]},
        },
        "difficulty": 5, 
        "sampleFileName": "flyingMassPoint.py",
    },

    # 2) JG
    "freeFallMassPoint": {
        "description": "Free-fall motion of an object with the following properties: point mass m = {mass} kg, gravity g = {gravity} m/s^2 (in negative z-direction). The object starts from rest and is dropped from a height hz = {height} m. The free fall shall be analyzed for 1 s. Air resistance is neglected and except the action of gravity, no forces act on the point mass..",
        "alternativeDescription": "{model}: {pointMass} m = {mass} kg, gravity g = {gravity} m/s^2 (in negative z-direction). The object starts from rest and is dropped from a height hz = {height} m. Air resistance and contact with ground are neglected.",
        "alternativeDescriptionParameters": {
            "model": ["Free-fall motion of an object with the following properties",
                      "Free-fall of a mass point",
                      "Mass point which is freely falling"],
            "pointMass": ["point mass",
                          "mass point",
                          "mass point with mass",
                          ],
            },
        "parameters": {
            "mass": {"default": 1.5, "range": [1, 3]},
            "gravity": {"default": 9.81, "list": gravityList},
            "height": {"default": 20, "range": [10, 50]},
        },
        "difficulty": 5, 
        "sampleFileName": "freeFallMassPoint.py",
    },


    # 3)
    "singleMassOscillator": {
        "description": "Mass-spring-damper with the following properties: The mass point with mass m = {mass} kg lies at [{length} cm,0,0], stiffness k = {stiffness} N/m, and damping d = {damping} Ns/m. The force applied to the mass in x-direction is f = {force} N. The spring-damper between mass point and ground at [0,0,0] is aligned with the x-axis, the spring has a length of {length} cm and is initially relaxed. Gravity is neglected.",
        "parameters": {
            "mass": {"default": 1, "range": [5, 15]},
            "stiffness": {"default": 2000, "range": [1000, 8000]},
            "damping": {"default": 20, "range": [10, 80]},
            "force": {"default": 10, "range": [5, 40]},
            "length": {"default": 5, "range": [5, 20]}, #cm
        },
        "difficulty": 5,
        "sampleFileName": "singleMassOscillatorFG.py",
    },
    
    # 4)
    "singleMassOscillatorGravity": {
        "description": "Mass-spring-damper with the following properties: The mass point with mass m = {mass} kg lies at [{length} cm,0,0], stiffness k = {stiffness} N/m, and damping d = {damping} Ns/m. The force applied to the mass in x-direction is f = {force} N. The spring-damper between mass point and ground at [0,0,0] is aligned with the x-axis, the spring has a length of {length} cm and is initially relaxed. The system is subject to gravity g = {gravity} m/s^2 in positive x-direction.",
        #Joh: 25-03-15: improves if positive added: gravity ... in positive x-direction.
        "parameters": {
            "mass": {"default": 1, "range": [5, 15]},
            "stiffness": {"default": 2000, "range": [1000, 8000]},
            "damping": {"default": 20, "range": [10, 80]},
            "force": {"default": 10, "range": [5, 40]},
            "gravity": {"default": 9.81, "list": gravityList},
            "length": {"default": 5, "range": [5, 20]}, #cm
        },
        "difficulty": 6,
        "sampleFileName": "singleMassOscillatorFG.py",
        "_readableName":"single mass oscillator with gravity",
    },
    
    # 5) JG
    "sliderCrankSimple": {
        "description": "Simple slider-crank mechanism modelled with two mass points that are connected with"
        " distance constraints. The crank is modelled with a point mass m1 = {massCrank} kg, located at "
        "radius r = {radius} m (initially located at x=0, y=r) and constrained to ground with a distance r; "
        "the slider is a point mass m2 = {massSlider} kg, initially located at xSlider = {xSlider} m and "
        "ySlider = 0. The connecting rod is only represented as distance constraint between crank mass and "
        "slider mass. Add a constraint to fix the z-coordinate of mass 1 and another "
        "constraint for the y- and z-coordinates of mass 2. "
        "A force fx = {forceSlider} acts on the slider in positive x-direction. "
        "The system is initially at rest, and no gravity is applied.",
        "parameters": {
            "massCrank": {"default": 0.5, "range": [0.5, 5]},
            "massSlider": {"default": 2, "range": [0.5, 5]},
            "radius": {"default": 0.5, "range": [0.5, 2]},
            "xSlider": {"default": 2, "range": [1.5, 4]},
            "forceSlider": {"default": 10, "range": [5, 20]},
        },
        "difficulty": 9, # masses + distance
        "sampleFileName": "sliderCrankSimple.py",
        "_readableName":"slider crank / point masses",
    },

    # 6) JG
    "singlePendulumElasticSpring": {
        "description": "Simple mathematical pendulum with the following properties: point mass m = {mass} kg, elastic string length = {length} m, string stiffness k = {stiffness} N/m, and gravity g = {gravity} m/s^2 which is applied in negative y-direction. The pendulum's reference configuration is such that the string has an angle {angle} degrees from the vertical (negative y-axis), thus the mass being on a positive x-coordinate for positive angles. The initial velocities of the mass point are v_x = {vx} m/s and v_y = {vy} m/s, applied in x resp. y direction.",
        #chooses initialDisplacement instead of referencePosition: "description": "Simple mathematical pendulum with the following properties: point mass m = {mass} kg, length of the elastic string L = {length} m, string stiffness k = {stiffness} N/m, and gravity g = {gravity} m/s^2 which is applied in negative y-direction. The pendulum is initialized with an angle of {angle} degrees from vertical (negative y-axis) equilibrium position, thus the mass being displaced in x-direction. The initial velocities of the mass point are v_x = {vx} m/s and v_y = {vy} m/s, applied in x resp. y direction.",
        "parameters": {
            "mass": {"default": 2, "range": [1, 5]},
            "length": {"default": 1, "range": [0.5, 2]},
            "gravity": {"default": 9.81, "list": gravityList},
            "angle": {"default": 60, "range": [10, 120]},
            "stiffness": {"default": 2000, "range": [500, 2000]},
            "vx": {"default": 0, "range": [-1, 1]}, 
            "vy": {"default": 1, "range": [-1, 1]}, 
        },
        "difficulty": 7, #masses + spring
        "sampleFileName": "singlePendulumElasticString.py",
        "_readableName":"pendulum with elastic string",
    },
        
    # 7) JG
    "singleMassOscillatorUserFunction": {
        "description": "Mass-spring-damper with the following properties: mass m = {mass} kg, stiffness k = {stiffness} N/m, and "
        "damping d = {damping} Ns/m. The force applied to the mass only in x-direction is given by the time-dependent function "
        "f(t) = {forceFunction}. The spring-damper is aligned with the x-axis and the spring has a length of {length} m and is "
        "relaxed in the initial position. Gravity is neglected.",
        "parameters": {
            "mass": {"default": 1, "range": [5, 15]},
            "stiffness": {"default": 2000, "range": [1000, 8000]},
            "damping": {"default": 20, "range": [10, 80]},
            "length": {"default": 0.2, "range": [0.2, 1]},
            "forceFunction": {"default": "400.*math.sin(math.pi*t)", "list": ["400.*math.sin(math.pi*t)","200.*math.sin(3.14*t)","500.","80.*t**2"]},
        },
        "difficulty": 10,
        "sampleFileName": "singleMassOscillatorUserFunction.py",
        "_readableName":"mass oscillator with user function",
    },
    
    # 8) JG
    "spinningDisc": {
        "description": "Rotational motion of a spinning disc (around z-axis) with the following properties: mass m = {mass} kg, disc radius r = {radius} m, disc width w = {width}, and initial angular velocity omega_z = {angular_velocity} rad/s. The disc is mounted on a frictionless revolute joint and experiences a constant torque T_z = {torque}. No gravity acts on the disc.",
        #Joh:NOTE: without "No gravity acts on the disc." it adds gravity, which changes solution due to Lagrange multipliers
        "parameters": {
            "mass": {"default": 20, "range": [10, 30]},
            "radius": {"default": 0.2, "range": [0.2, 0.75]},
            "width": {"default": 0.025, "range": [0.025, 0.2]},
            "angular_velocity": {"default": 50, "range": [10, 50]},
            "torque": {"default": 2, "range": [0.25, 2]},
        },
        "difficulty": 15, #15 
        "sampleFileName": "spinningDisc.py",
    },

    
    # 9) 
    "doubleMassOscillator": {
        "description": "Two-mass-spring-damper system consisting of two masses with the following properties: mass m1 = m2 = {mass} kg, stiffness k1 = k2 = {stiffness} N/m, and damping d1 = d2 = {damping} Ns/m. The first mass is placed at [{length} cm,0,0] and the second mass at [2*{length} cm,0,0]. The first spring is connected to ground at [0,0,0], and the second spring connects the two masses. Each spring has a reference length of {length} cm and is relaxed in the initial configuration. A force {force} is applied in x-direction to mass 2. No gravity is applied to the system.",
        "parameters": {
            "mass": {"default": 1, "range": [5, 15]},
            "stiffness": {"default": 2000, "range": [1000, 8000]},
            "damping": {"default": 20, "range": [10, 80]},
            "force": {"default": 10, "range": [5, 40]},
            "length": {"default": 5, "range": [5, 20]}, #cm !!!
        },
        "difficulty": 6,
        "sampleFileName": "doubleMassOscillatorFG.py",
    },

    # 10)
    "nMassOscillator": {
        "description": "A serial chain of {nMasses} masses connected with springs-dampers. Each mass has m = {mass} kg, the stiffnesses are k = {stiffness} N/m, and the damping coefficients are d = {damping} Ns/m. A force f = {force} N is applied to the last mass (with highest index). The first mass is connected to ground via the first spring-damper. The relaxed length of each spring is {distance} m, and the first mass is located at {axis} = {distance} m. The serial chain is oriented along the global {axis}-axis. Gravity acts in positive {axis}-direction, with g = {gravity} m/s^2.",
        #Joh: 25-03-15: direction of gravity was missing initially!
        "parameters": {
            "nMasses": {"default": 5, "list": Arange(2, 5)},
            "mass": {"default": 10, "range": [5, 15]},
            "stiffness": {"default": 5000, "range": [3000, 7000]},
            "damping": {"default": 50, "range": [30, 70]},
            "force": {"default": 100, "range": [50, 150]},
            "gravity": {"default": 9.81, "list": gravityList},
            "distance": {"default": 0.5, "range": [0.2, 1]},
            "axis": {"default": "x", "list": ["x", "y", "z"]}, 
        },
        "difficulty": 8,
        "sampleFileName": "nMassOscillatorFG.py",
    },
     
    # 11) JG
    "singlePendulum": {
        "description": "Simple mathematical pendulum with the following properties: point mass m = {mass} kg, inelastic string length = {length} m (fixed distance between ground and mass point), and gravity g = {gravity} m/s^2 which acts in negative y-direction, while the pendulum moves in the x-y plane. The reference configuration of the pendulum is such that the string has an angle (positive rotation sense) of {angle} degrees relative to the negative y-axis. Air resistance is neglected.",
        #Joh:NOTE: without point mass, it often choses a rigid body: "description": "Simple mathematical pendulum with the following properties: mass m = {mass} kg, length of the rigid string L = {length} m, and gravity g = {gravity} m/s^2 which acts in negative y-direction, while the pendulum moves in the x-y plane. The pendulum from vertical equilibrium position with an angle of {angle} degrees (positive rotation sense). Air resistance is neglected.",
        #Joh: 25-03-15: avoid initial for the angle, as it will choose initialDisplacements
        #Joh: 25-03-16: uses frequently revolute joint due to keyword rigid:  "Simple mathematical pendulum with the following properties: point mass m = {mass} kg, length of the rigid string L = {length} m, and gravity g = {gravity} m/s^2 which acts in negative y-direction, while the pendulum moves in the x-y plane. The reference configuration of the pendulum is such that the string has an angle (positive rotation sense) of {angle} degrees relative to the negative y-axis. Air resistance is neglected.",
        "parameters": {
            "mass": {"default": 2, "range": [1, 5]},
            "length": {"default": 1, "range": [0.5, 2]},
            "gravity": {"default": 9.81, "list": gravityList},
            "angle": {"default": 15, "range": [10, 30]},
        },
        "difficulty": 8, #masses + distance
        "sampleFileName": "singlePendulum.py",
    },
        
    # 12) JG
    "doublePendulum": {
        "description": "Double pendulum system consisting of two mass points which are connected with inextensible strings with the following properties: mass m1 = {mass1} kg, mass m2 = {mass2} kg, length of the strings L1 = {length1} m, L2 = {length2} m, and gravity g = {gravity} m/s^2 which acts in negative y-direction. The first arm of the pendulum points in positive x direction and the second arm in positive y-direction. The strings are massless, inelastic and the length shall be constrained. Air resistance is neglected.",
        #Joh: 25-03-16: uses sometimes elastic strings: "Double pendulum system consisting of two mass points which are connected with inextensible strings with the following properties: mass m1 = {mass1} kg, mass m2 = {mass2} kg, length of the strings L1 = {length1} m, L2 = {length2} m, and gravity g = {gravity} m/s^2 which acts in negative y-direction. The first arm of the pendulum points in positive x direction and the second arm in positive y-direction (relative to mass1). Air resistance is neglected.",
        "parameters": {
            "mass1": {"default": 2, "range": [0.5, 5]},
            "mass2": {"default": 3, "range": [0.5, 5]},
            "length1": {"default": 0.8, "range": [0.5, 2]},
            "length2": {"default": 0.5, "range": [0.5, 2]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 8, # masses + distance
        "sampleFileName": "doublePendulum.py",
        "_infoHasMultpleMasses":2,
    },
    
    # 13) JG
    "nPendulum": {
        "description": "Multibody n-pendulum system consisting of {nMasses} point masses with the following properties: masses m = {mass} kg, length of inextensible strings l_single = {length} m, and gravity g = {gravity} m/s^2. The pendulum starts from horizontal configuration, where all masses are aligned along the x-axis (the first mass starting at x={length}) and all masses have an initial velocity of {initialVelocity} m/s in negative y-direction. Gravity acts in negative y-direction and air resistance is neglected. The strings between the consecutive masses (and the string between first mass and ground) shall be modelled as constrained distances.",
        #Joh: without "Do not use springs for the strings, but use constraints." it uses springs
        "parameters": {
            "nMasses": {"default": 8, "list": Arange(2, 8)},
            "mass": {"default": 0.5, "range": [0.5, 5]},
            "length": {"default": 0.25, "range": [0.25, 2]},
            "gravity": {"default": 9.81, "list": gravityList},
            "initialVelocity": {"default": 0.5, "range": [0, 1]},
        },
        "difficulty": 9, # n * (masses + distance)
        "sampleFileName": "nPendulum.py",
    },
    
    # 14) JG
    "fourBarMechanismPointMasses": {
        "description": "A planar four-bar mechanism modelled with 2 points masses and "
        "the 3 moving bars modeled as massless distance constraints. The origin point of the "
        "mechanism on ground is located at [0,0,0], mass 1 with m1 = {mass1} kg is located "
        "at [0,{a},0], mass 2 with m2 = {mass2} kg is located at [{b},{a},0] and the final "
        "additional ground point is at [{b},-{c},0]. Gravity g = {gravity} m/s^2 acts in "
        "negative y-direction and mass 1 has an initial velocity of [{vx},0,0]. "
        "There is no friction or other resistance.",
        #NOTE: uses a 3rd mass (mass=0) on ground: #"description": "A planar four-bar mechanism modelled with 2 points masses and the 4 bars modeled as massless distance constraints. The origin point of the mechanism fixed to ground is located at [0,0,0], mass 1 with m1 = {mass1} kg is located at [0,{a},0], mass 2 with m2 = {mass2} kg is located at [{b},{a},0] and the final point is located at [{b},-{c},0]. Gravity acts in negative y-direction and mass 1 has an initial velocity of [{vx},0,0]. There is no friction or other resistance.",
        "parameters": {
            "mass1": {"default": 0.5, "range": [0.5, 4]},
            "mass2": {"default": 1.0, "range": [1, 4]},
            "a": {"default": 0.1, "range": [0.1, 0.5]},
            "b": {"default": 0.8, "range": [0.8, 2.5]},
            "c": {"default": 0.1, "range": [0.1, 0.5]},
            "vx": {"default": 1.25, "range": [0.1, 1.25]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 21, #orig: 9
        "sampleFileName": "fourBarMechanismPointMasses.py",
        "_readableName":"four-bar mechanism / point masses",
    },
    
    # 15) JG
    "springCoupledFlyingRigidBodies": {
        "description": "Two free flying rigid elements, each modeled as a rigid body with brick-shape "
        "(density = {density} kg/m^3, lx={lx} m, wy={wy} m, hz={hz} m). Rigid body 1 is globally placed at [lx/2,0,0] "
        "and rigid body 2 at [3*lx/2,0,0], both bodies' COMs being at their origin. A Cartesian spring-damper with stiffness [{k},{k},{k}] N/m and damping "
        "[{d},{d},{d}] Ns/m is attached between rigid body 1 "
        "at [lx/2,0,0] (local position) and rigid body 2 at [-lx/2,0,0] (also local position). "
        "Body1 has initial velocity [0,{vy},0] and body2 has initial velocity [0,-{vy},0] and gravity is zero.",
        "parameters": {
            "density": {"default": 1000, "range": [1000, 2000]},
            "lx": {"default": 1., "range": [0.4, 1.]},
            "wy": {"default": 0.2, "range": [0.1, 0.2]},
            "hz": {"default": 0.1, "range": [0.05, 0.1]},
            "vy": {"default": 4., "range": [1, 4]},
            "k": {"default": 20000., "range": [10000, 20000]},
            "d": {"default": 100., "range": [100, 400]},
        },
        "difficulty": 15, # rigid bodies + spring-dampers
        "sampleFileName": "springCoupledFlyingRigidBodies.py",
        "_infoHasMultpleRigidBodies":2,
    },
    
    
    # 16) JG (=spinning Disc)
    "torsionalOscillator": {
        "description": "Torsional oscillator consisting of a cylindrical disc with rotational motion around z-axis and a torsional "
        "spring-damper with the following properties: mass m = {mass} kg, disc radius r = {radius} m, disc width w = {width}, torsional "
        "stiffness k = {stiffness} Nm/rad, torsional damping coefficient d = {damping} Nms/rad and initial angular velocity "
        "omega_z = {angular_velocity} rad/s. The disc is connected to ground with a frictionless revolute joint and "
        "experiences a constant torque T_z = {torque}. No gravity acts on the disc.",
        #Joh:NOTE: without "No gravity acts on the disc." it adds gravity, which changes solution due to Lagrange multipliers
        "parameters": {
            "mass": {"default": 20, "range": [10, 30]},
            "radius": {"default": 0.2, "range": [0.2, 0.75]},
            "width": {"default": 0.025, "range": [0.025, 0.2]},
            "angular_velocity": {"default": 50, "range": [10, 50]},
            "torque": {"default": 2, "range": [0.25, 2]},
            "stiffness": {"default": 2000, "range": [500, 2000]},
            "damping": {"default": 5, "range": [5, 20]},
        },
        "difficulty": 15, 
        "sampleFileName": "torsionalOscillator.py",
    },

    # 17) JG
    "invertedSinglePendulum": {
        "description": "An inverted pendulum on a cart modelled with two mass points, "
        "where the y and z coordinates of the cart mass are fixed to ground by a coordinate constraint, "
        "while the x-coordinate can freely move. "
        "The pendulum is modelled with a distance constraint between "
        "cart and pendulum mass: cart mass m1 = {mass1} kg, pendulum mass m2 = {mass2} kg, "
        "pendulum length = {length} m, and gravity g = {gravity} m/s^2 which acts in "
        "negative y-direction. The pendulum starts from the upright position, where m2 "
        "is located at the positive y-axis. A disturbance force f = {force} N acts "
        "in x-direction at the pendulum and no control is applied on the cart.",
        "parameters": {
            "mass1": {"default": 4, "range": [0.5, 5]},
            "mass2": {"default": 0.5, "range": [0.5, 5]},
            "length": {"default": 0.8, "range": [0.8, 2]},
            "force": {"default": 0.1, "range": [0.01, 0.1]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 15, # masses + spherical joint
        "sampleFileName": "invertedSinglePendulum.py",
    },
    
    # 18) JG
    "discRollingOnGround": {
        "description": "Rolling motion of a solid disc rolling on a x/y plane with the following properties: "
        "mass m = {mass} kg, radius r = {radius} m, width w = {width} m, and gravity g = {gravity} m/s^2 "
        "(in negative z-direction). The disc's axis is initially aligned with the x-axis. "
        "The COM of the disc has an initial position of p0 = [{px},{py},{radius}], translational velocity "
        "of v_y = {initialVelocity} m/s and an initial angular velocity omega_x=-v_y/radius. "
        "Rolling shall be modelled ideal and without slipping.",
        #LLMs do not choose right sign of initial ang vel, corresponding to vel
        #Joh: 25-03-16: added "(consider correct sign)"
        "parameters": {
            "mass": {"default": 4, "range": [2, 10]},
            "radius": {"default": 0.3, "range": [0.1, 0.5]},
            "width": {"default": 0.05, "range": [0.02, 0.1]},
            "px": {"default": 1, "range": [-1., 1.]},
            "py": {"default": 1, "range": [-1., 1.]},
            "initialVelocity": {"default": 1, "range": [0.5, 2.]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 21, # rigid body + RollingDisc #medium example as many models fail
        "sampleFileName": "discRollingOnGround.py",
    },

    
    # 19) JG
    "doublePendulumElasticSpring": {
        "description": "Double pendulum consisting of two mass points which are connected with elastic springs with the following properties: mass m1 = {mass1} kg, mass m2 = {mass2} kg, length of the strings L1 = {length1} m, L2 = {length2} m, stiffnesses k1 = {stiffness1} and k2 = {stiffness2}, and gravity g = {gravity} m/s^2 which acts in negative y-direction. The first arm of the pendulum points in {arm1direction}-direction and the second arm in {arm2direction}-direction.",
        "parameters": {
            "mass1": {"default": 1, "range": [0.5, 5]},
            "mass2": {"default": 0.5, "range": [0.5, 5]},
            "length1": {"default": 0.5, "range": [0.5, 2]},
            "length2": {"default": 0.3, "range": [0.3, 2]},
            "stiffness1": {"default": 1000., "range": [1000., 10000.]},
            "stiffness2": {"default": 500., "range": [500., 5000.]},
            "gravity": {"default": 9.81, "list": gravityList},
            "arm1direction": {"default": 'positive x', "list": ['positive x', 'negative x', 'positive y']},
            "arm2direction": {"default": 'positive y', "list": ['positive y', 'positive x', 'negative x']},
        },
        "difficulty": 8, # masses + distance
        "sampleFileName": "doublePendulumElasticString.py",
    },
    
    # 20) JG
    "nPendulumElasticSpring": {
        "description": "Multibody n-pendulum system consisting of {nMasses} point masses connected with spring-dampers with the following properties: masses m = {mass} kg, lengths of single elastic strings l_single = {length} m, stiffness k = {stiffness} and damping d = {damping} of strings, and gravity g = {gravity} m/s^2 which acts in negative y-direction. The pendulum starts from horizontal configuration, where all masses are aligned with the x-axis, the first mass located at x={length}, with a spring-damper connected to ground.",
        "parameters": {
            "nMasses": {"default": 2, "list": Arange(2, 8)},
            "mass": {"default": 0.5, "range": [0.5, 5]},
            "length": {"default": 0.25, "range": [0.25, 2]},
            "gravity": {"default": 9.81, "list": gravityList},
            "stiffness": {"default": 1000., "range": [1000., 10000.]},
            "damping": {"default": 50., "range": [50., 500.]},
        },
        "difficulty": 9, # n * (masses + distance)
        "sampleFileName": "nPendulumElasticString.py",
    },
    
    # 21) JG
    "elasticChain": {
        "description": "Straight elastic chain modelled with {nMasses} mass points connected with spring-dampers with the following properties: masses m = {mass} kg, lengths of each chain element l_single = {length} m, stiffness k = {stiffness} and damping d = {damping} of chain elements, and gravity g = {gravity} m/s^2 which acts in negative y-direction. The chain starts from horizontal configuration, where all masses are aligned with the x-axis, the first mass located at x=0. The left-most and right-most spring-dampers are fixed to ground using spherical joints.",
        #problems with last support position: the last mass lies {length} m left from the last support.
        "parameters": {
            "nMasses": {"default": 20, "list": Arange(5, 20)},
            "mass": {"default": 5, "range": [0.5, 5]},
            "length": {"default": 4, "range": [1, 4]},
            "gravity": {"default": 9.81, "list": gravityList},
            "stiffness": {"default": 2000., "range": [2000., 10000.]},
            "damping": {"default": 100., "range": [100., 500.]},
        },
        "difficulty": 21, # orig: 9
        "sampleFileName": "elasticChain.py",
    },

    # 22) JG
    "singlePendulumRigidBody": {
        "description": "Pendulum modeled with a brick-shape rigid body and a revolute joint. "
        "The body has density = {density} kg/m^3, and xyz-dimensions lx={lx}m, wy={wy}m, hz={hz}m. "
        "The center of mass is equal to the reference point and initially located at [lx/2,0,0], "
        "while the revolute joint (rotation around z-axis) is located at [0,0,0] (global coordinates). "
        "Gravity g = {gravity} m/s^2 acts in negative y-direction, and no further force acts.",
        "parameters": {
            "density": {"default": 1000, "range": [500, 8000]},
            "lx": {"default": 1.2, "range": [0.75, 5]},
            "wy": {"default": 0.4, "range": [0.1, 0.5]},
            "hz": {"default": 0.2, "range": [0.1, 0.5]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 15, # rigid bodies + revolute joint
        "sampleFileName": "singlePendulumRigidBody.py",
        "_readableName":"single pendulum / rigid body",
    },

    # 23) JG
    "massPointOnStringRigid": {
        "description": "A 3D mass point with mass m = {mass} kg is attached to a string with length {length} m, with a predominant rotation around the z-axis due to centrifugal forces.  Gravity g = {gravity} m/s^2 acts in negative z-direction. The mass point is placed initially at [{length},0,0] and the initial velocity is [0,{vy},0], given in m/s. The string shall be modelled as rigid distance between the mass point and the ground position at [0,0,0].",
        "parameters": {
            "mass": {"default": 2, "range": [0.5, 5]},
            "length": {"default": 0.5, "range": [0.2, 0.75]},
            "vy": {"default": 5, "range": [4, 10]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 10,
        "sampleFileName": "massPointOnStringRigid.py",
        "_readableName":"mass point on rigid string",
    },   
    
    # 24) JG
    "massPointOnStringElastic": {
        "description": "A 3D mass point with mass m = {mass} kg is attached to a string with length {length} m, with a predominant rotation around the z-axis due to centrifugal forces. Gravity g = {gravity} m/s^2 acts in negative z-direction. The mass point is placed initially at [{length},0,0] and the initial velocity is [0,{vy},0], given in m/s. The elastic string shall be modelled as spring-damper with stiffness k = {stiffness} N/m and damping d = {damping} Ns/m, which connects the mass point and the ground position at [0,0,0].",
        "parameters": {
            "mass": {"default": 2, "range": [0.5, 5]},
            "length": {"default": 0.5, "range": [0.2, 0.75]},
            "vy": {"default": 5, "range": [4, 10]},
            "gravity": {"default": 9.81, "list": gravityList},
            "stiffness": {"default": 2000, "range": [2000, 5000]},
            "damping": {"default": 20, "range": [20, 50]},
        },
        "difficulty": 10,
        "sampleFileName": "massPointOnStringElastic.py",
        "_readableName":"mass point on elastic string",
    },   
    # 25) JG
    "linkOnTwoPrismaticJoints": {
        "description": "A system consisting of two mass points connected by a distance constraint. The first mass m1 = {mass1} kg is initially located at x0 = {x0} m (other coordinates are 0) and can freely move only along the x-axis, while y-axis and z-axis are constrained to ground by a coordinate constraint. The second mass m2 = {mass2} kg is initially located at {axis}0 = {y0} m (other coordinates are 0) and can move only freely along the {axis}-axis, while the other two axes are again fixed to ground. Furthermore, a distance constraint is added between the two mass points and the length shall be computed automatically. A spring with stiffness = {stiffness} N/m is used to connect mass m1 and ground at [0,0,0]. Gravity g = {gravity} m/s^2 acts in positive {gAxis}-direction, and no further forces or damping are applied.",
        "parameters": {
            "mass1": {"default": 2, "range": [1, 15]},
            "mass2": {"default": 3, "range": [1, 15]},
            "x0": {"default": 0.5, "range": [0.5, 2]},
            "y0": {"default": 0.5, "range": [0.5, 2]},
            "stiffness": {"default": 500, "range": [300, 900]},
            "axis": {"default": "y", "list": ["y", "z"]}, 
            "gAxis": {"default": "x", "list": ["x", "y", "z"]}, 
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 10,
        "sampleFileName": "xyLinkage.py",
    },
    
    # 26) JG
    "flyingRigidBody": {
        "description": "A free flying rigid body with brick-shape, density = {density} kg/m^3, and xyz-dimensions of the cuboid lx={lx}m, wy={wy}m, hz={hz}m is investigated. The COM of the body is initially located at [0,0,0]. The initial velocity shall be [{vx},{vy},{vz}] and the initial angular velocity is [{avx},{avy},{avz}]. Gravity g = {gravity} m/s^2 acts in negative {gAxis}-direction. Contact with ground is not considered and no further forces act on the point mass.",
        "parameters": {
            "density": {"default": 200, "range": [100, 200]},
            "lx": {"default": 4, "range": [3, 5]},
            "wy": {"default": 0.8, "range": [0.5, 1.3]},
            "hz": {"default": 1.5, "range": [1, 2]},
            "vx": {"default": 5, "range": [1., 5.]},
            "vy": {"default": 5, "range": [1., 5.]},
            "vz": {"default": 5, "range": [1., 5.]},
            "avx": {"default": 0.5, "range": [0.25, 2.]},
            "avy": {"default": 2,   "range": [0.25, 2.]},
            "avz": {"default": 0.5, "range": [0.25, 2.]},
            "gravity": {"default": 9.81, "list": gravityList},
            "gAxis": {"default": "y", "list": ["x", "y", "z"]}, 
        },
        "difficulty": 12,
        "sampleFileName": "flyingRigidBody.py",
    },
    
    # 27) JG
    "suspendedRigidBody": {
        "description": "A rigid body is suspended by 4 Cartesian spring-dampers, which are connected to ground. "
        "The rigid body has a brick-shape with density = {density} kg/m^3, and xyz-dimensions "
        "lx={lx}m, ly={ly}m, lz={lz}m. The reference point of the body, "
        "which is equal to the COM, is located initially at [0,{ly}/2,0]. The Cartesian spring-dampers are located at the "
        "x/z positions of the vertices of the body. The y-local-position of the spring-dampers at the body is -{ly}/2 and "
        "the local position for ground is y=0. All spring-dampers have equal parameters: stiffness = [{kx},{ky},{kz}] N/m "
        "and damping = [{dx},{dy},{dz}] Ns/m. Gravity g = {gravity} m/s^2 acts in negative y-direction, and no further forces "
        "or damping are applied and contact with ground is ignored.",
        "parameters": {
            "density": {"default": 400, "range": [400, 500]},
            "lx": {"default": 4, "range": [3, 5]},
            "ly": {"default": 0.75, "range": [0.75, 1.25]},
            "lz": {"default": 1.5, "range": [1, 2]},
            "kx": {"default": 10000, "range": [10000, 20000]},
            "ky": {"default": 100000, "range": [100000, 200000]},
            "kz": {"default": 10000, "range": [10000, 20000]},
            "dx": {"default": 200, "range": [200, 400]},
            "dy": {"default": 2000, "range": [2000, 4000]},
            "dz": {"default": 200, "range": [200, 400]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 20,
        "sampleFileName": "suspendedRigidBody.py",
    },

    # 28)
    "gyroscopeOnSphericalJoint": {
        "description": "A gyro is modelled with a rigid body and a spherical joint. "
        "The gyro is modelled as cylindrical disc with mass m = {mass} kg, disc radius "
        "r = {radius} m, and disc width w = {width}. The axis of the disc is initially "
        "oriented along positive x-axis and the disc is positioned at [{offsetX},0,0] "
        "(which is also the COM). The disc is connected to ground with a spherical joint at "
        "position [0,0,0]. The initial angular velocity of the disc is [{angularVelocity},0,0]. "
        "Gravity g = {gravity} m/s^2 acts in negative {gAxis}-direction, and no further forces "
        "or damping are applied.",
        "parameters": {
            "mass": {"default": 20, "range": [15, 20]},
            "radius": {"default": 0.2, "range": [0.2, 0.3]},
            "width": {"default": 0.025, "range": [0.025, 0.04]},
            "offsetX": {"default": 0.25, "range": [0.15, 0.25]},
            "angularVelocity": {"default": 200, "range": [100., 200.]}, #take care of step size and enough speed for precession
            "gAxis": {"default": "y", "list": ["y", "z"]}, 
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 20,
        "sampleFileName": "gyroscopeOnSphericalJoint.py",
    },
    # 29)
    "prismaticJointSystem": {
        "description": "A system consisting of a brick-shaped rigid body freely moving on a prismatic joint in {axis}-direction, which is fixed to ground. "
        "The COM of the rigid body is placed at [0,0,0]. The rigid body has a mass of m = {mass} kg and the side lengths of the "
        "body are all equal with s = {s} m. A force f = {force} N acts on the rigid body's COM in {axis}-direction. Gravity is neglected.",
        "parameters": {
            "mass": {"default": 10, "range": [5, 15]},
            "s": {"default": 0.5, "range": [0.25, 0.8]},
            "axis": {"default": "x", "list": ["x", "y", "z"]}, 
            "force": {"default": 20, "range": [5, 20]},
        },
        "difficulty": 15, 
        "sampleFileName": "prismaticJointSystem.py",
    },
    
    # 30) JG
    "twoMassPointsWithSprings": {
        "description": "Two mass points connected by springs. The mass points both have mass m1 = m2 = {mass} kg, are initially located at p1 = [-{ax},0,0] and p2 = [{ax},0,0], positions given in m. Mass m1 has initial velocity {velStr} = {vel} m/s and m2 has initial velocity {velStr} = -{vel} m/s while all other initial velocity components are zero. The spring between the two mass points is initially relaxed and has stiffness k = {stiffness} N/m. No gravity nor forces are present.",
        "parameters": {
            "mass": {"default": 1, "range": [0.5, 2]},
            "ax": {"default": 0.5, "range": [0.25, 0.8]},
            "vel": {"default": 4, "range": [0.5, 4]},
            "velStr": {"default": "vy", "list": ["vy","vz"]},
            "stiffness": {"default": 2000, "range": [500, 2000]},
        },
        "difficulty": 6,
        "sampleFileName": "twoMassPointsWithSprings.py",
    },
    
    # 31) JG
    "twoMassPointsWithDistances": {
        "description": "Two mass points connected by distance constraint. The mass points both have mass m1 = m2 = {mass} kg, are initially located at p1 = [-{ax},0,0] and p2 = [{ax},0,0], positions given in m. Mass m1 has initial velocity {velStr} = {vel} m/s and m2 has initial velocity {velStr} = -{vel} m/s while all other initial velocity components are zero. No gravity nor forces are present.",
        "parameters": {
            "mass": {"default": 1, "range": [0.5, 2]},
            "ax": {"default": 0.5, "range": [0.25, 0.8]},
            "vel": {"default": 4, "range": [0.5, 4]},
            "velStr": {"default": "vy", "list": ["vy","vz"]},
        },
        "difficulty": 6,
        "sampleFileName": "twoMassPointsWithDistances.py",
    },

    # 32) Rigid Rotor, simply supported, initial angular velocity, unbalance by shifting COM
    "rigidRotorSimplySupported": {
        "description": "A rotor is modelled with a rigid body mounted on ground with Cartesian spring-dampers. "
        "The rotor is modelled as cylinder with mass m = {mass} kg, disc radius r = {radius} m, and length = {length} m. "
        "The axis of the rotor is initially oriented along the x-axis and the reference point of the rotor lies at [0,0,0] (which is also the COM). "
        "The rotor is connected to ground with two Cartesian spring-dampers at global position [-{length}/2,0,0] resp. [{length}/2,0,0], "
        "the positions being identical with the local positions of the spring-dampers on the rotor. "
        "The Cartesian spring-dampers have stiffness values [{kx},{ky},{ky}] N/m and damping values [{dx},{dy},{dy}] Ns/m. "
        "The initial angular velocity of the rotor is [{angularVelocity},0,0]. "
        "Gravity g = {gravity} m/s^2 acts in negative {gAxis}-direction, and no further forces or damping are applied.",
        "parameters": {
            "mass": {"default": 5, "range": [2, 5]},
            "radius": {"default": 0.2, "range": [0.1, 0.2]},
            "length": {"default": 0.5, "range": [0.5, 1]},
            "kx": {"default": 1000, "range": [1000, 1500]},
            "ky": {"default": 2000, "range": [2000, 4000]},
            "dx": {"default": 5, "range": [5, 8]},
            "dy": {"default": 10, "range": [10, 20]},
            "angularVelocity": {"default": 500, "range": [200., 500.]},
            "gAxis": {"default": "y", "list": ["y", "z"]}, 
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 20,
        "sampleFileName": "rigidRotorSimplySupported.py",
    },


    # 33) Rigid Rotor, simply supported, unbalance by shifting COM; torque added to simulate runup
    "rigidRotorUnbalanced": {
        "description": "A rotor is modelled with a rigid body mounted on ground with Cartesian spring-dampers. "
        "The rotor is modelled as cylinder with mass m = {mass} kg, disc radius r = {radius} m, and length = {length} m. "
        "The axis of the rotor is initially oriented along the x-axis and the reference point of the rotor lies at [0,0,0]. "
        "The COM of the rigid body lies at [0,{comY},0], representing an unbalance U={mass}*{comY} kgm. "
        "The rotor is connected to ground with two Cartesian spring-dampers at global position [-{length}/2,0,0] resp. [{length}/2,0,0], "
        "the positions being identical with the local positions of the spring-dampers on the rotor. "
        "The Cartesian spring-dampers have stiffness values [{kx},{ky},{ky}] N/m and damping values [{dx},{dy},{dy}] Ns/m. "
        "The rotor starts from zero velocity, and is driven by a torque [1,0,0] Nm. "
        "Gravity g = {gravity} m/s^2 acts in negative {gAxis}-direction. Simulate the rotor for 2s and use a stepSize of 0.2ms to meet the accuracy limits.",
        "parameters": {
            "mass": {"default": 2, "range": [2, 3]},
            "radius": {"default": 0.1, "range": [0.1, 0.12]},
            "length": {"default": 0.5, "range": [0.5, 1]},
            "comY": {"default": 0.002, "range": [0.001, 0.002]},
            "kx": {"default": 1500, "range": [1000, 1500]},
            "ky": {"default": 4000, "range": [2000, 4000]},
            "dx": {"default": 5, "range": [5, 8]},
            "dy": {"default": 10, "range": [10, 20]},
            "gAxis": {"default": "y", "list": ["y", "z"]}, 
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 21, #orig: 20
        "sampleFileName": "rigidRotorUnbalanced.py",
    },
    
    # 34) Double pendulum with rigid bodies
    "doublePendulumRigidBodies": {
        "description": "Double pendulum modeled with brick-shape rigid bodies and revolute joints. "
        "Both bodies have density = {density} kg/m^3, and xyz-dimensions lx={lx} m, wy={wy} m, hz={hz} m. "
        "The first body's center is located initially at [lx/2,0,0] and the second body at [3*lx/2,0,0], both bodies having their COM locally at [0,0,0] (not translated). "
        "The first revolute joint between ground and body 1 is located at [0,0,0] (global coordinates), "
        "and the second revolute joint between body 1 and body 2 is located at [lx,0,0] (global coordinates). "
        "Both revolute joints' have a free rotation around the z-axis."
        "Gravity g = {gravity} m/s^2 acts in negative y-direction, and no further force acts.",
        "parameters": {
            "density": {"default": 1000, "range": [500, 8000]},
            "lx": {"default": 1.0, "range": [0.75, 5]},
            "wy": {"default": 0.4, "range": [0.1, 0.5]},
            "hz": {"default": 0.2, "range": [0.1, 0.5]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 16, # rigid bodies + revolute joint
        "sampleFileName": "doublePendulumRigidBodies.py",
        "_readableName":"double pendulum / rigid bodies",
    },
    
    # 35) Slider-crank mechanism with rigid bodies
    "sliderCrankRigidBodies": {
        "description": "A slider-crank mechanism is given with a crank rotating about the z-axis. "
        "The crank is modelled as rigid body with cylinder shape (axis=z) with mass m = {mass} kg, radius r = {radius} m, and width = {width} m, "
        "and it is initially located at [0,0,-{width}]. "
        "The brick-shaped conrod and piston have identical parameters: density = {density} kg/m^3, "
        "and xyz-dimensions lx={lx} m, wy={wy} m, hz={hz} m, initially not rotated (straight configuration of slider-crank). "
        "The conrod's center is located initially at [r+0.5*lx,0,0] and the piston is initially located at [r+1.5*lx,0,0]. "
        "A torque [0,0,T] shall act on the crank, employing a user-function with T={torque} Nm until 1 s and 0 thereafter. "
        "The revolute joint between ground and crank is located at [0,0,0] (global coordinates), "
        "the revolute joint between crank and conrod is located at [r,0,0] (global coordinates), "
        "and the revolute joint between conrod and piston is located at [r+lx,0,0] (global coordinates). "
        "The prismatic joint (axis=x) between ground and piston is located at [r+1.5*lx,0,0] (global coordinates). "
        "No gravity acts on the system. Simulate the system for 2 s.",
        "parameters": {
            "mass": {"default": 2, "range": [2, 3]},
            "radius": {"default": 0.25, "range": [0.2, 0.25]},
            "width": {"default": 0.1, "range": [0.05, 0.1]},
            "density": {"default": 7850, "list": [2800,7850]},
            "lx": {"default": 0.4, "range": [0.4, 0.6]},
            "wy": {"default": 0.05, "range": [0.05, 0.1]},
            "hz": {"default": 0.05, "range": [0.05, 0.1]},
            "torque": {"default": 2, "range": [1, 2]},
        },
        "difficulty": 21, # orig: 16; rigid bodies + revolute joint
        "sampleFileName": "sliderCrankRigidBodies.py",
        "_readableName":"slider crank / rigid bodies",
    }
} #end: simpleMBSDescriptions

#%% medium standard mbs descriptions
mediumMBSDescriptions = {
    # 36) JG
    "fourBarMechanismRigidBodies": {
        "description": "A planar four-bar mechanism modelled with 3 rigid bodies "
        "and ground. Each arm has density = {density} kg/m^3, a rectangular cross section "
        "with body-fixed dimensions wy = {wy} and wz = {wz}, "
        "where x is the arm's axis and the local and global z-axes match. "
        "The origin point of the mechanism on ground is located at [0,0,0]. "
        "The first arm with length l1 = {l1} is aligned with the global y-axis (=local x-axis, thus rotated around z-axis in reference configuration) and has it's COM and reference point at "
        "p0 = [0,l1/2,0] and it has a revolute joint that connects to ground at origin, rotating freely around "
        "the z-axis. The second arm with length l2 = {l2} is aligned with the global x-axis with the COM "
        "at p1 = [l2/2,l1,0], and is connected to the first arm via a revolute joint at the "
        "intersection of both arms. The third arm with length l3 = {l3} has its reference point and COM "
        "at [l2,l1-l3/2,0] and it's local x-axis aligns points in negative direction of the global y-axis (again rotated reference config.). "
        "Again there is a revolute joint between arm2 and arm3 at the intersection. "
        "At [l2,l1-l3,0] a generic joint pins the third arm to the ground, constraining only "
        "the free motion components without over-constraining. "
        "Gravity g = {gravity} m/s^2 acts in negative y-direction and a torque "
        "T = {T} Nm is added at arm1 around the positive z-axis. "
        "There is no friction or other resistance.",
        "parameters": {
            "density": {"default": 1000, "range": [500, 2500]},
            "wy": {"default": 0.02, "range": [0.01, 0.05]},
            "wz": {"default": 0.02, "range": [0.01, 0.05]},
            "l1": {"default": 0.5, "range": [0.5, 0.8]},
            "l2": {"default": 1, "range": [1, 2]},
            "l3": {"default": 1, "range": [1, 2]},
            "T": {"default": 5.0, "range": [1, 5]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 30,
        "sampleFileName": "fourBarMechanismRigidBodies.py",
        "_readableName":"four-bar mechanism / rigid bodies",
    },
    # 37) JG
    "parallelChordTrussSprings": {
    "description": "A 2D parallel chord truss frame hinged at x=0, consisting of n = {nSegments} segments "
    "of length L = {segLength} m and height H = {segHeight} m. The frame is modeled as mass points "
    "connected by massless linear spring-dampers with stiffness k = {stiffness} N/m and "
    "damping d = {damping} Ns/m. Nodes are located at the corners of each segment: "
    "lower nodes at y=0 and upper nodes at y=H. Each rectangular segment i contains a "
    "diagonal spring-damper connecting the top-left node to the bottom-right node. "
    "The x- and y-coordinates of the left-most (x=0) nodes are fixed to the ground. "
    "The mass at each node is half the sum of the masses of all connected members, "
    "where each member (horizontal, vertical, or diagonal) has a linear density of rhoA = {rhoA} kg/m. "
    "Gravity g = {gravity} m/s^2 acts in negative y-direction. "
    "The frame starts from rest in its undeformed horizontal configuration.",
    "parameters": {
        "nSegments": {"default": 3, "list": [2, 3, 4, 5, 6]},
        "segLength": {"default": 1.0, "range": [0.5, 1.5]},
        "segHeight": {"default": 0.8, "range": [0.4, 1.0]},
        "stiffness": {"default": 2e5, "range": [2e5, 5e6]},
        "damping": {"default": 1000, "range": [500, 5000]},
        "rhoA": {"default": 2.0, "range": [0.5, 2.0]},
        "gravity": {"default": 9.81, "list": gravityList},
    },
    "difficulty": 30,
    "sampleFileName": "parallelChordTrussSprings.py",
    "_readableName": "parallel chord truss frame with springs",
    },
    # 38) JG
    "prattTrussBridgeSprings": {
        "description": "A symmetric 2D Pratt truss bridge with height H = {segHeight} m. "
        "The structure consists of a symmetrical central section with rectangular panels "
        "of length L_panel {segLength}, having {nSegments} panels on the left and on the right side of the symmetry (in total the central section has 2*{nSegments} panels). "
        "Furthermore, there are are single triangular panels attached to each of the ends of the central section (on left and right). "
        "The lower segments of the panels as well as the outer ends of the triangular panels are located at y=0, "
        "while the upper segments lie at y=H. In the rectangular panels, diagonals are oriented "
        "to point toward the center of the span (top-outer to bottom-inner). "
        "The frame uses mass points and massless spring-dampers with stiffness k = {stiffness} N/m "
        "and damping d = {damping} Ns/m. The left-most bottom node is fixed in x and y directions at global position [0,0,0]. "
        "The right-most bottom node is fixed only in y direction (roller support). "
        "The mass at each node is half the weight of connected trusses with linear density rhoA = {rhoA} kg/m. "
        "Start creation of nodes from left to right, first all bottom nodes, then all top nodes."
        "Gravity g = {gravity} m/s^2 acts in negative y-direction onto the mass points.",
        "parameters": {
            "nSegments": {"default": 2, "list": [2]}, # Half-span segments
            "segLength": {"default": 2.0, "range": [1.5, 2.5]},
            "segHeight": {"default": 1.5, "range": [1.0, 2.0]},
            "stiffness": {"default": 1e6, "range": [5e5, 5e6]},
            "damping": {"default": 500, "range": [100, 1000]},
            "rhoA": {"default": 10.0, "range": [5.0, 25.0]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 30,
        "sampleFileName": "prattTrussBridgeSprings.py",
        "_readableName": "Pratt truss bridge frame with springs",
    },
    # 39) JG
    "collidingSpheresInSpace": {
        "description": "Two spheres modeled as rigid bodies undergo sphere-sphere contact. "
        "Sphere 1 has mass m1 = {mass1} kg and radius r1 = {radius1} m. Sphere 2 has mass m2 = {mass2} kg "
        "and radius r2 = {radius2} m. Both spheres are initially placed on the x-y plane (which has z=0). "
        "Sphere 1 is at rest at p1 = [0, 0, 0]. "
        "Sphere 2 is at p2 = [{distance}, 0, 0] and has an initial velocity v2 = [-{velocity}, 0, 0] m/s. "
        "Contact between spheres and between spheres and ground is modeled with "
        "stiffness k = {stiffness} N/m and a restitution coefficient of e = {restitution}. "
        "No friction and no gravity are considered. The spheres move in space, no ground contact is considered.",
        "parameters": {
            "mass1": {"default": 1.5, "range": [1.0, 5.0]},
            "mass2": {"default": 1.5, "range": [1.0, 5.0]},
            "radius1": {"default": 0.1, "range": [0.05, 0.2]},
            "radius2": {"default": 0.1, "range": [0.05, 0.2]},
            "distance": {"default": 0.5, "range": [0.4, 0.8]},
            "velocity": {"default": 2.0, "range": [1.5, 4.0]}, # Ensure distance/velocity < 1.0
            "stiffness": {"default": 1e5, "range": [5e4, 5e5]},
            "restitution": {"default": 0.8, "range": [0.5, 1.0]},
        },
        "difficulty": 35,
        "sampleFileName": "collidingSpheresInSpace.py",
        "_readableName": "sphere-sphere impact",
    },
    # 40) JG
    "fallingMassPointContact": {
        "description": "A single mass point undergoing freefall and subsequent ground contact. "
        "The mass m = {mass} kg is initially at position p0 = [0, 0, {height}] "
        "and has radius r = {radius} m and an initial velocity v0 = [0, 0, {velocity}]. "
        "Gravity g = {gravity} m/s^2 acts in the negative z-direction. "
        "The ground is defined as the x-y plane at z=0, having quadratic size of 1 m^2. "
        "Contact between sphere and ground plane is modeled with "
        "stiffness k = {stiffness} N/m, a restitution coefficient of e = {restitution}, "
        "and no friction.",
        "parameters": {
        "mass": {"default": 1.0, "range": [0.2, 2.0]},
        "radius": {"default": 0.05, "range": [0.04, 0.2]},
        "height": {"default": 1.0, "range": [0.5, 2.0]},
        "velocity": {"default": 0.0, "range": [-2.0, 2.0]},
        "stiffness": {"default": 2e5, "range": [2e5, 2e6]},
        "restitution": {"default": 0.8, "range": [0.05, 1.0]},
        "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 35,
        "sampleFileName": "fallingMassPointContact.py",
        "_readableName": "falling mass point with ground contact",
    },

    ########################################################################################################################################################
    # sample files from here onwards (41-84) are excluded from the submission state of the repository. 
    # If you need access to these files, please contact the corresponding developers.     
    ########################################################################################################################################################
    
    # 41) JG
    "collidingSpheresOnGround": {
        "description": "Two spheres modeled as rigid bodies undergo sphere-sphere contact. "
        "The spheres also undergo frictional contact with ground, which is a rectangular x-y plane at z=0, "
        "with dimensions 5 m x 5 m. "
        "Sphere 1 has mass m1 = {mass1} kg and radius r1 = {radius1} m. Sphere 2 has mass m2 = {mass2} kg "
        "and radius r2 = {radius2} m. Both spheres are initially placed on the x-y plane (which has z=0). "
        "Sphere 1 is at rest at p1 = [0, 0, {radius1}]. "
        "Sphere 2 is at p2 = [{distance}, 0, {radius2}] and has an initial velocity "
        "v2 = [-{velocity}, 0, 0] m/s and an according angular velocity for ideal rolling at start. "
        "Gravity is g = {gravity} m/s^2 in negative z-direction. "
        "Contact between spheres and between spheres and ground is modeled with "
        "stiffness k = {stiffness} N/m, a restitution coefficient of e = {restitution}, "
        "and a friction parameter of {friction}.",
        "parameters": {
            "mass1": {"default": 1.5, "range": [1.0, 5.0]},
            "mass2": {"default": 1.5, "range": [1.0, 5.0]},
            "radius1": {"default": 0.1, "range": [0.05, 0.2]},
            "radius2": {"default": 0.1, "range": [0.05, 0.2]},
            "distance": {"default": 0.5, "range": [0.4, 0.8]},
            "velocity": {"default": 2.0, "range": [1.5, 4.0]}, # Ensure distance/velocity < 1.0
            "stiffness": {"default": 1e5, "range": [5e4, 5e5]},
            "restitution": {"default": 0.8, "range": [0.5, 1.0]},
            "friction": {"default": 0.1, "range": [0.05, 0.5]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 35,
        "sampleFileName": "collidingSpheresOnGround.py",
        "_readableName": "sphere-sphere and sphere-ground impact",
    },
    # 42) JG
    "pendulumHittingWall": {
        "description": "A spherical pendulum has point mass m = {mass} kg with radius r = {radius} m, "
        "which is located at p0 = [-{length} m, 0, 2 m] at the beginning (horizontal configuration). "
        "The mass is suspended by a massless rod of length L = {length} m, modelled as fixed distance constraint, "
        "which is attached to ground at [0, 0, 2 m]. "
        "Gravity g = {gravity} m/s^2 acts in the negative z-direction. "
        "A vertical wall of width 2 m and height 2 m, modelled as planar contact, is positioned parallel "
        "to the y-z plane at x = {wallOffset} m, symmetric to the y-axis and z_min = 0. "
        "As it swings, the pendulum impacts the wall. "
        "The contact between the sphere and the wall is modeled with stiffness k = {stiffness} N/m "
        "and a restitution coefficient e = {restitution}. ",
        "parameters": {
            "mass": {"default": 2.0, "range": [0.5, 5.0]},
            "radius": {"default": 0.05, "range": [0.02, 0.1]},
            "length": {"default": 1.0, "range": [0.5, 1.5]},
            "wallOffset": {"default": 0.2, "range": [0.0, 0.5]},
            "stiffness": {"default": 5e5, "range": [1e5, 1e6]},
            "restitution": {"default": 0.7, "range": [0.3, 1.0]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 30,
        "sampleFileName": "pendulumHittingWall.py",
        "_readableName": "pendulum wall impacting",
    },   
    # 43) JG
    "newtonsCradleSpheres": {
        "description": "A Newton's cradle consisting of 4 identical spheres (index 0 to 3) modeled as mass points. "
        "Each sphere has a mass m = {mass} kg, a radius r = {radius} m, and is suspended from a "
        "fixed ceiling (ground) at z = {length} m. Each sphere with index i is connected to the ceiling "
        "at p_fix,i = [i*(2*{radius}+0.002), 0, {length}] via a distance constraint of length L = {length} m. "
        "The spheres are aligned along the x-axis. Spheres 1, 2, and 3 are initially at rest "
        "at their equilibrium positions p_i = [i*(2*{radius}+0.002), 0, 0]. Sphere 0 is initially "
        "positioned at p_0 = [-{length}, 0, {length}], representing a horizontal configuration "
        "with the cable taut, and starts from rest. "
        "Gravity g = {gravity} m/s^2 acts in the negative z-direction. "
        "Contact between spheres is modeled with a stiffness k = {stiffness} N/m "
        "and a restitution coefficient e = {restitution}. Friction is neglected. Add the spheres in the order of their index.",
        "parameters": {
            "mass": {"default": 1.0, "range": [0.5, 2.0]},
            "radius": {"default": 0.05, "range": [0.02, 0.08]},
            "length": {"default": 0.2, "range": [0.25, 0.8]},
            "stiffness": {"default": 1e6, "range": [5e5, 5e6]},
            "restitution": {"default": 0.9, "range": [0.8, 1.0]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 35,
        "sampleFileName": "newtonsCradleSpheres.py",
        "_readableName": "Newton's cradle",
    },
    # 44) JG
    "billiardBreakContact": {
        "description": "A billiard-style impact involving four spheres (0, 1, 2, and 3) on a 4x4 m ground plate, "
        "which has its center at [0,0,0]. "
        "All spheres have radius r = {radius} m. Spheres 0, 1, and 2 represent the target: "
        "Sphere 0 is at [0, 0, {radius}]. Sphere 1 and 2 are placed at " #touching Sphere 0 
        "p1 = [sqrt(3)*{radius}, {radius}, {radius}] and p2 = [sqrt(3)*{radius}, -{radius}, {radius}]. "
        "Spheres 0, 1, and 2 have mass m = {mass} kg. "
        "Sphere 3 (the cue ball) has mass 2*m and is initially at p3 "
        "= [-{distance}*cos({angle}), {distance}*sin({angle}), {radius}]. "
        "It has an initial velocity v3 = [{velocity}*cos({angle}), -{velocity}*sin({angle}), 0] m/s, "
        "directed toward the origin and an according initial angular velocity for pure rolling with no sliding. "
        "Gravity g = {gravity} acts in negative z-direction. "
        "Inter-sphere and sphere-ground contacts use stiffness k = {stiffness} N/m, coefficient of restitution "
        "e = {restitution} and friction coefficient mu = {friction}. Add the spheres in the order of their index.",
        "parameters": {
            "mass": {"default": 0.17, "range": [0.15, 0.2]},
            "radius": {"default": 0.03, "range": [0.02, 0.04]},
            "distance": {"default": 0.1, "range": [0.1, 0.5]},
            "velocity": {"default": 1.0, "range": [0.5, 1.0]},
            "angle": {"default": 0.0, "range": [-1, 1]}, # radians
            "stiffness": {"default": 1e5, "range": [1e4, 1e5]},
            "restitution": {"default": 0.95, "range": [0.5, 1.0]},
            "friction": {"default": 0.1, "range": [0.1, 0.4]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 30,
        "sampleFileName": "billiard4spheres.py",
        "_readableName": "billiard with 4 spheres",
    },
    # 45) JG
    "meshSphereImpact": {
        "description": "A quadratic mesh of n x n mass points (n = {nGrid}) in the x-y plane, each with mass m = {mass} kg "
        "and radius r = {radius} m. The mass points are arranged in a grid with spacing w = {spacing} m, "
        "with the first mass being located at (x,y,z) = (0,0,0). "
        "Add the mass points first for each row along x-coordinate and then move to other y-coordinates (from smallest to largest). "
        "Each interior mass point is connected to its 4 neighbors via spring-dampers with "
        "stiffness k = {stiffness} N/m, damping d = {damping} Ns/m and reference length w. Boundary mass points are "
        "additionally connected to fixed ground points (at z=0) at a distance w, maintaining the "
        "grid pattern. A larger sphere with radius R = {spacing} m and mass M = {massLarge} kg "
        "is released from p_large = [{centerOffset}, {centerOffset}, {height}] with initial "
        "velocity v = [0, 0, {velocity}]. Impact between the large sphere and all mesh points "
        "is modeled as sphere-sphere contact with stiffness k_c = {stiffnessContact} N/m "
        "and restitution e = {restitution}. Gravity g = {gravity} m/s^2 acts in negative z-direction. "
        "No friction is considered.",
        "parameters": {
            "nGrid": {"default": 2, "list": [2]},
            "mass": {"default": 0.1, "range": [0.05, 0.5]},
            "radius": {"default": 0.01, "range": [0.005, 0.02]},
            "spacing": {"default": 0.2, "range": [0.1, 0.3]},
            "massLarge": {"default": 2.0, "range": [1.0, 5.0]},
            "height": {"default": 0.5, "range": [0.3, 0.8]},
            "centerOffset": {"default": 0.4, "range": [0.2, 0.6]}, # Typically (n-1)*w / 2
            "velocity": {"default": -1.0, "range": [-2.0, 0.0]},
            "stiffness": {"default": 5000, "range": [1000, 10000]},
            "damping": {"default": 100, "range": [100, 500]},
            "stiffnessContact": {"default": 1e5, "range": [5e4, 5e5]},
            "restitution": {"default": 0.8, "range": [0.5, 0.95]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 30,
        "sampleFileName": "meshSphereImpact.py",
        "_readableName": "grid mesh and large sphere impact",
    },    
    # 46) scooter
    "scooterIdealRolling": {
        "description": "A non-steerable scooter/bicycle model consisting of a central chassis and two wheels. "
        "The chassis is a brick-shaped rigid body with dimensions {bodyL}m (length), {bodyW}m (width), "
        "and {bodyH}m (height), with a density of {bodyDensity} kg/m^3. Two identical wheels "
        "have mass {wheelM} kg, radius {wheelR} m, and width {wheelW} m with cylindrical shape (axis=y). "
        "The wheels are located at the front and rear centers of the chassis base: "
        "p_front_axle = [{bodyL}/2, 0, {wheelR}] and p_rear_axle = [-{bodyL}/2, 0, {wheelR}]. "
        "The chassis center is at [0, 0, {wheelR} + {bodyH}/2]. Both wheels are connected via "
        "revolute joints (rotation about y-axis) and are constrained to roll ideally on the x-y plane (z=0) without slipping. "
        "The system starts with an initial forward velocity v_x = {initialVelocity} m/s, "
        "and the wheels have a matching initial angular velocity omega_y = -{initialVelocity}/{wheelR} rad/s. "
        "Gravity g = {gravity} m/s^2 acts in the negative z-direction.",
        "parameters": {
            "bodyL": {"default": 0.8, "range": [0.5, 1.2]},
            "bodyW": {"default": 0.2, "range": [0.1, 0.3]},
            "bodyH": {"default": 0.1, "range": [0.05, 0.2]},
            "bodyDensity": {"default": 500, "range": [200, 1000]},
            "wheelR": {"default": 0.1, "range": [0.05, 0.2]},
            "wheelW": {"default": 0.05, "range": [0.02, 0.1]},
            "wheelM": {"default": 0.5, "range": [0.2, 1.5]},
            "initialVelocity": {"default": 2.0, "range": [1.0, 5.0]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 30,
        "sampleFileName": "scooterIdealRolling.py",
        "_readableName": "non-steerable scooter with ideal rolling",
    },
    # 47) simple car
    "fourWheeledPlateCar": {
        "description": "A 4-wheeled car model with a rigid chassis and four identical wheels. "
        "The chassis is a brick-shaped plate with length L_x = {bodyL} m, width w_y = {bodyW} m, "
        "and thickness t_z = {bodyH} m, with density {bodyDensity} kg/m^3. "
        "Four identical wheels of mass {wheelM} kg, radius {wheelR} m, and width {wheelW} m "
        "have COMs which are located laterally at y = +/- ({bodyW}/2 + {wheelW}/2) m and "
        "at x = +/- ({bodyL}/2 - {wheelR}) m relative to the chassis COM. "
        "The wheels which can rotate around the y-axis are connected to the chassis "
        "via revolute joints that are positioned at the COMs of the wheels. "
        "The wheels are initially in contact with the x-y plane at z=0, meaning the chassis' "
        "center is at [0, 0, {wheelR} + {bodyH}/2] initially. Ideal rolling without slipping "
        "is assumed for all wheels. The car has an initial forward velocity v_x = {initialVelocity} m/s "
        "in the positive x-direction, with matching initial angular velocities for all wheels. "
        "Gravity g = {gravity} m/s^2 acts in the negative z-direction.",
        "parameters": {
            "bodyL": {"default": 2.0, "range": [1.5, 3.5]},
            "bodyW": {"default": 1.2, "range": [0.8, 1.8]},
            "bodyH": {"default": 0.2, "range": [0.1, 0.4]},
            "bodyDensity": {"default": 600, "range": [300, 1200]},
            "wheelR": {"default": 0.3, "range": [0.2, 0.5]},
            "wheelW": {"default": 0.15, "range": [0.1, 0.3]},
            "wheelM": {"default": 2.0, "range": [1.0, 5.0]},
            "initialVelocity": {"default": 5.0, "range": [1.0, 15.0]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 30,
        "sampleFileName": "fourWheeledPlateCar.py",
        "_readableName": "non-steerable four-wheeled car",
    },
    # 48) gyro nutation
    "gyroscopeNutation": {
        "description": "A gyro is modelled with a rigid body and a spherical joint. "
                       "The gyro is modelled as cylindrical disc with mass m = {mass} kg, disc radius "
                       "r = {radius} m, and disc width w = {width}. The axis of the disc is initially "
                       "oriented along positive x-axis and the disc is positioned at the origin [0,0,0] "
                       "(which is also the COM). The disc is connected to ground with a spherical joint at "
                       "the origin. The initial angular velocity of the disc is [{angularVelocity} rad/s,0,0]. "
                       "A force vector [{forceX} N,0,0] is applied on the disc at local position "
                       "[0,{radius} m,0] and turned off at t = 0.1 seconds, with a load user function. "
                       "Gravity g = {gravity} m/s^2 acts in negative {gAxis}-direction, "
                       "or damping are applied.",
        "parameters": {
            "mass": {"default": 20, "range": [15, 20]},
            "radius": {"default": 0.2, "range": [0.2, 0.3]},
            "width": {"default": 0.025, "range": [0.025, 0.04]},
            "forceX": {"default": 0.25, "range": [100, 2000]},
            "angularVelocity": {"default": 200, "range": [100., 200.]}, #take care of step size and enough speed for precession
            "gAxis": {"default": "y", "list": ["y", "z"]}, 
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 25,
        "sampleFileName": "gyroscopeNutation.py",
    },
    # 49) scara robot (RRR), all axis aligned
    "scaraRobot2R": {
        "description": "A 2-link SCARA robot moving in the x-y plane with two revolute joints "
        "aligned with the z-axis. Both links are identical brick-shaped rigid bodies "
        "with length L = {linkL} m, width W = {linkW} m, and height H = {linkH} m, "
        "and density rho = {density} kg/m^3. The robot is initially straight and "
        "aligned with the global x-axis. Joint 0 is at [0,0,0] (ground), and Joint 1 "
        "connects Link 0 and Link 1 at [L, 0, 0]. Each joint is driven by a PD controller "
        "modeled as a torsional spring-damper with stiffness (=P_control) k = {Pcontrol} Nm/rad "
        "and damping (=D_control) d = {Dcontrol} Nms/rad. "
        "The target angles theta_0 and theta_1 "
        "follow a cubic polynomial trajectory over t = [0, 0.5] seconds, starting at "
        "0 rad and ending at theta_0_end = {targetAngleDegree0} degree, "
        "and theta_1_end = {targetAngleDegree1} degree, with zero initial and zero final velocities. "
        "For t > 0.5, the set-angles remain constant at their respective target angles. "
        "Gravity is neglected as motion is in the x-y plane.",
        "parameters": {
            "linkL": {"default": 0.5, "range": [0.3, 0.8]},
            "linkW": {"default": 0.05, "range": [0.03, 0.1]},
            "linkH": {"default": 0.1, "range": [0.05, 0.15]},
            "density": {"default": 2700, "range": [1000, 8000]},
            "Pcontrol": {"default": 1000, "range": [500, 5000]},
            "Dcontrol": {"default": 50, "range": [10, 100]},
            "targetAngleDegree0": {"default": 90, "list": [-45,-30,0,20,30,45,60,75,90,120]}, 
            "targetAngleDegree1": {"default": 90, "list": [-45,-30,-10,20,30,45,60,75,90,120]}, 
        },
        "difficulty": 30,
        "sampleFileName": "scaraRobot2R.py",
        "_readableName": "2R SCARA robot",
    },    
    # 50) spherical robot (RRR): Z-X-X, initially straight
    "sphericalRobotRRR": {
        "description": "A 3D RRR articulated robot arm with a Z-Y-Y axis configuration. "
        "Link 0 is a brick-shaped body of height L0_z = {L0_z} m, attached to ground "
        "at [0, 0, 0] via a revolute joint around the global Z-axis. Link 1 (inner arm) "
        "has length L1_x = {L1_x} m and Link 2 (outer arm) has length L2_x = {L2_x} m. "
        "The joints for Link 1 and Link 2 are revolute joints acting around the local y-axis. "
        "Initially, the robot links 1 and 2 are straight and aligned with the global x-axis, the centers at height z=L0_z. "
        "All links are idealized as brick-shaped bodies with width {width} m (y-direction), "
        "height {height} m (x-direction for link 0, z-direction for others), and "
        "density rho = {density} kg/m^3. Each joint is driven by a PD controller "
        "with stiffness (=P_control) k = {Pcontrol} Nm/rad and damping (=D_control) d = {Dcontrol} Nms/rad. "
        "Target angles [theta0, theta1, theta2] follow cubic trajectories over t=[0, 0.5]s "
        "from [0, 0, 0] to [pi/2, -pi/2, pi/2] rad, remaining constant thereafter. "
        "Gravity g = {gravity} m/s^2 acts in the negative z-direction.",
        "parameters": {
            "L0_z": {"default": 0.5, "range": [0.3, 0.7]},
            "L1_x": {"default": 0.6, "range": [0.4, 0.8]},
            "L2_x": {"default": 0.4, "range": [0.3, 0.6]},
            "width": {"default": 0.08, "range": [0.05, 0.12]},
            "height": {"default": 0.08, "range": [0.05, 0.12]},
            "density": {"default": 750, "range": [500, 1000]},
            "Pcontrol": {"default": 2000, "range": [2000, 10000]},
            "Dcontrol": {"default": 200, "range": [50, 400]},
            "gravity": {"default": 9.81, "list": gravityList},
        },
        "difficulty": 30,
        "sampleFileName": "sphericalRobotRRR.py",
        "_readableName": "RRR articulated robot with 3D cubic trajectory",
    },   
} #end: mediumMBSDescriptions

_simpleAndMediumMBSDescriptions = {} 
_simpleAndMediumMBSDescriptions.update(simpleMBSDescriptions)
_simpleAndMediumMBSDescriptions.update(mediumMBSDescriptions)

simpleMBSDescriptionsNew = {} 
for key, value in _simpleAndMediumMBSDescriptions.items():
    if value['difficulty'] <= 20: #the new simple limit=20
        simpleMBSDescriptionsNew[key] = value

mediumMBSDescriptionsNew = {} 
for key, value in _simpleAndMediumMBSDescriptions.items():
    if value['difficulty'] > 20: #the new simple limit=20
        mediumMBSDescriptionsNew[key] = value

simpleAndMediumMBSDescriptions = {} 
simpleAndMediumMBSDescriptions.update(simpleMBSDescriptionsNew)
simpleAndMediumMBSDescriptions.update(mediumMBSDescriptionsNew)


#%% geometry descriptions
# never forget units, keep consistence across examples, do not add whitespaces around headers!
geometryDescriptions = {
    # 51) SW
    'bearingBlockPart': {
    'description': (
        'A bearing block as a machine part, with the geometry, interfaces, and material being given.'
        'The geometry of the bearing block consists of a ground plate and a joint sleeve (modelled as hollow cylinder).\n'
        '- The ground plate is a cuboid, having a length of {plateLength} m in global x-direction,'
        ' width of {plateWidth} m in global y-direction, and height of {plateHeight} m in global z-direction.\n'
        '- The bottom of the ground plate stands on the global xy-plane.\n'
        ' The bottom surface of the ground plate is an interface named "{groundPlateInterfaceName}".\n'
        '- The edges of the top surface of the ground plate have fillets with a radius of {filletsRadiusFactor0}' + multChar + '{plateHeight} m.\n'
        '- The joint sleeve is a hollow cylinder.\n'
        '- The cylinder axis is parallel to the global y-axis.\n'
        '- The sleeve has a wall thickness of {wallThickness} m.\n'
        '- Only the outer cylinder making up the joint sleeve has fillets with a radius of {filletsRadiusFactor1}' + multChar + '{wallThickness} m.\n'
        '- The inner radius of the sleeve is given as {bearingRadius} m.\n'
        '- The sleeve is {bearingHeight} m long.\n'
        '- The footpoint of the cylinder is placed at (0, -{bearingHeight}/2, {bearingRadius}+{wallThickness}) m in the global reference frame.\n'
        '- The inner of the joint sleeve is an interface named "{jointInterfaceName}".\n'
        'The whole bearing block is made out of rubber{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'bearingHeight': {'default': 0.1, 'range': [0.1, 0.3]},
            'bearingRadius': {'default': 0.05, 'range': [0.05, 0.2]},
            'wallThickness': {'default': 0.07, 'range': [0.05, 0.07]},
            'filletsRadiusFactor0': {'default': 0.5, 'range': [0.25, 0.5]}, # never bigger or equal to 1!
            'filletsRadiusFactor1': {'default': 0.4, 'range': [0.3, 0.4]}, # never bigger than 0.5!
            'plateLength': {'default': 0.4, 'range': [0.3, 0.5]},
            'plateWidth': {'default': 0.4, 'range': [0.3, 0.5]},
            'plateHeight': {'default': 0.03, 'range': [0.02, 0.04]},
            'EModul': {'default': 5e6,  'range': [5e6, 5e7]},
            'nu':{'default': 0.35, 'list': [0.35]}, # not varied yet
            'rho':{'default': 950, 'range': [900, 1200]},
            'groundPlateInterfaceName': {'default': 'groundPlate', 'list': ['groundPlate']},
            'jointInterfaceName': {'default': 'joint', 'list': ['joint']},
            'partName': {'default': 'bearingBlockPart', 'list': ['bearingBlockPart']},
        },
        'difficulty': 5,
        'sampleFileName': 'bearingBlockPart.py',
    },
    
    # 52) SW
    'unbalancedRotorPart': {
        'description': (
            'An unbalanced rotor as a machine part, with the geometry, interfaces, and material being given.'
            ' The rotor consists of three cylinders, all parallel to the global y-axis, and one sphere modelling the unbalance:\n'
            ' - The first cylinder is the shaft. The footpoint of this cylinder is located at'
            ' (0, -{bearingHeight}/2, 0) m. The shaft'
            ' is {bearingHeight} m long and has a radius of {bearingRadius}. The curved face of the shaft'
            ' is an interface named "{shaftInterfaceName}".\n'
            '- A smaller rotor that is placed along the positive y-axis next to the shaft, having the same cylinder axis.'
            ' The smaller rotor is {smallRotorHeight} m long and has a radius of {smallRotorRadius} m.\n'
            '- A larger rotor that is placed along the positive y-axis next to the smaller rotor, having the same cylinder axis.'
            ' The larger rotor is {largeRotorHeight} m long and has a radius of {largeRotorRadius} m.'
            ' The whole large rotor has fillets with a radius of {filletsRadiusFactor}' + multChar + '{largeRotorHeight} m.\n'
            '- The sphere has a radius of {sphereRadius} m and is placed at'
            ' (0, {bearingHeight}/2+{smallRotorHeight}+{largeRotorHeight}/2, {largeRotorRadius}) m.\n'
            'The whole unbalanced rotor is made out of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'bearingHeight': {'default': 0.3, 'range': [0.1, 0.3]},
            'bearingRadius': {'default': 0.05, 'range': [0.05, 0.2]},
            'smallRotorHeight': {'default': 0.15, 'range': [0.15, 0.25]},
            'smallRotorRadius': {'default': 0.11, 'range': [0.10, 0.20]},
            'largeRotorHeight': {'default': 0.08, 'range': [0.08, 0.15]},
            'filletsRadiusFactor': {'default': 0.3, 'range': [0.2, 0.3]}, # never bigger or equal to 1/2!
            'largeRotorRadius': {'default': 0.25, 'range': [0.2, 0.3]},
            'sphereRadius': {'default': 0.05, 'range': [0.01, 0.06]},
            'EModul': { 'default': 2.1e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]}, # not varied yet
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'shaftInterfaceName': {'default': 'shaft', 'list': ['shaft']},
            'partName': {'default': 'unbalancedRotorPart', 'list': ['unbalancedRotorPart']},
        },
        'difficulty': 5,
        'sampleFileName': 'unbalancedRotorPart.py',
    },
    
    # 53) TM
    'balancedRotorPart': {
        'description': (
            'A balanced rotor as a machine part, with the geometry, interfaces, and material being given.'
            ' The rotor is symmetric and consists of five cylinders all with axis along the global x-axis, centered at the global origin ' + groundPos + ' m:\n'
            '- The large rotor disk is centered at the origin,'
            ' extending from -({largeRotorHeight}/2) m to +({largeRotorHeight}/2) m in x-direction,'
            ' with radius {largeRotorRadius} m.\n'
            '- Two small rotor sections, one on each side of the large disk:'
            ' each extends {smallRotorHeight} m outward from the large rotor edge in x-direction,'
            ' with radius {smallRotorRadius} m.\n'
            '- Two shafts, one on each side of the small rotors:'
            ' each has a total length of {bearingHeight}+{chamferSizeFactor}' + multChar + '{bearingRadius} m and radius {bearingRadius} m.'
            ' The outermost end edges of each shaft have chamfers of size {chamferSizeFactor}' + multChar + '{bearingRadius} m,'
            ' so the remaining curved interface surface of each shaft (excluding the chamfer) has a length of exactly {bearingHeight} m.\n'
            '- The curved face of the left shaft (negative x side) is an interface named "{shaftInterfaceNameLeft}",'
            ' the curved face of the right shaft (positive x side) is an interface named "{shaftInterfaceNameRight}".\n'
            'A keyway is cut from the outer surface of the large rotor disk.'
            ' The keyway is centered at (0, {largeRotorRadius}, 0) m on the large rotor,'
            ' is {keyLength} m long along the x-axis, {keyHeight} m deep, and has semicircular ends with radius {keyRadius} m.\n'
            'The whole balanced rotor is made out of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'bearingHeight': {'default': 0.075, 'range': [0.05, 0.1]},
            'bearingRadius': {'default': 0.05, 'range': [0.05, 0.1]},
            'smallRotorHeight': {'default': 0.2, 'range': [0.15, 0.25]},
            'smallRotorRadius': {'default': 0.075, 'range': [0.06, 0.12]},
            'largeRotorHeight': {'default': 0.25, 'range': [0.2, 0.3]},
            'largeRotorRadius': {'default': 0.125, 'range': [0.1, 0.2]},
            'chamferSizeFactor': {'default': 0.2, 'range': [0.1, 0.25]},
            'keyRadius': {'default': 0.025, 'range': [0.015, 0.035]},
            'keyLength': {'default': 0.125, 'range': [0.08, 0.15]},
            'keyHeight': {'default': 0.025, 'range': [0.015, 0.035]},
            'EModul': {'default': 2.1e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]},
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'shaftInterfaceNameLeft': {'default': 'shaftLeft', 'list': ['shaftLeft']},
            'shaftInterfaceNameRight': {'default': 'shaftRight', 'list': ['shaftRight']},
            'partName': {'default': 'balancedRotorPart', 'list': ['balancedRotorPart']},
        },
        'difficulty': 6,
        'sampleFileName': 'balancedRotorPart.py',
    },
    
    # 54) TM
    'flexibleBeamPart': {
        'description': (
            'A solid flexible beam as a machine part, with the geometry, interfaces, and material being given.'
            ' The beam has a length of {beamLength} m in global x-direction,'
            ' width of {beamWidth} m in global y-direction, and height of {beamHeight} m in global z-direction.'
            ' The beam is centered at the origin ' + groundPos + ' m.'
            ' The face on the left side (negative x-direction) of the beam is an interface called "{beamInterfaceNameLeft}",'
            ' the face on the right side (positive x-direction) of the beam is an interface called "{beamInterfaceNameRight}".'
            ' The whole beam is made out of rubber{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'beamLength': {'default': 0.5, 'range': [0.5, 2.0]},
            'beamWidth': {'default': 0.05, 'range': [0.05, 0.4]},
            'beamHeight': {'default': 0.05, 'range': [0.05, 0.4]},
            'EModul': {'default': 5e6,  'range': [5e6, 5e7]},
            'nu':{'default': 0.35, 'list': [0.35]}, # not varied yet
            'rho':{'default': 950, 'range': [900, 1200]},
            'beamInterfaceNameLeft': {'default': 'leftSide', 'list': ['leftSide']},
            'beamInterfaceNameRight': {'default': 'rightSide', 'list': ['rightSide']},
            'partName': {'default': 'flexibleBeamPart', 'list': ['flexibleBeamPart']},
        },
        'difficulty': 5,
        'sampleFileName': 'flexibleBeamSolidPart.py',
    },
    
    # 55) TM, SW
    'flexibleBeamFilletsPart': {
        'description': (
            'A solid flexible beam as a machine part, with the geometry, interfaces, and material being given.'
            ' The beam has a length of {beamLength} m in global x-direction,'
            ' width of {beamWidth} m in global y-direction, and height of {beamHeight} m in global z-direction.'
            ' The beam is centered at the origin ' + groundPos + ' m.'
            ' Round the beam edges using fillets. The fillet radius should be {filletRadiusFactor}' + multChar + 'min(x,y,z) m, where x, y, and z are the beam\'s length, width, and height.'
            ' The fillets are only applied alongside the maximum and minimum faces in the z-direction.'
            ' The face on the left side (negative x-direction) of the beam is an interface called "{beamInterfaceNameLeft}",'
            ' the face on the right side (positive x-direction) of the beam is an interface called "{beamInterfaceNameRight}".'
            ' The whole beam is made out of rubber{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'beamLength': {'default': 0.5, 'range': [0.5, 2.0]},
            'beamWidth': {'default': 0.05, 'range': [0.05, 0.4]},
            'beamHeight': {'default': 0.05, 'range': [0.05, 0.4]},
            'filletRadiusFactor': {'default': 0.3, 'range': [0.2, 0.3]}, # never bigger or equal to 1/2!
            'EModul': {'default': 5e6,  'range': [5e6, 5e7]},
            'nu':{'default': 0.35, 'list': [0.35]}, # not varied yet
            'rho':{'default': 950, 'range': [900, 1200]},
            'beamInterfaceNameLeft': {'default': 'leftSide', 'list': ['leftSide']},
            'beamInterfaceNameRight': {'default': 'rightSide', 'list': ['rightSide']},
            'partName': {'default': 'flexibleBeamFilletsPart', 'list': ['flexibleBeamFilletsPart']},
        },
        'difficulty': 5,
        'sampleFileName': 'flexibleBeamFilletsPart.py',
    },
    
    # 56) TM
    'flexibleBeamHollowPart': {
        'description': (
            'A hollow flexible beam as a machine part, with the geometry, interfaces, and material being given.'
            ' The beam has a length of {beamLength} m in global x-direction.'
            ' The inner of the beam has a width of {beamWidth} m in global y-direction, and height of {beamHeight} m'
            ' in global z-direction. The beam has a wall thickness of {wallThickness} m.'
            ' The beam is centered at the origin ' + groundPos + ' m.'
            ' The face on the left side (negative x-direction) of the beam is an interface called "{beamInterfaceNameLeft}",'
            ' the face on the right side (positive x-direction) of the beam is an interface called "{beamInterfaceNameRight}".'
            ' The whole beam is made out of rubber{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'beamLength': {'default': 0.5, 'range': [0.5, 2.0]},
            'beamWidth': {'default': 0.05, 'range': [0.05, 0.4]},
            'beamHeight': {'default': 0.05, 'range': [0.05, 0.4]},
            'wallThickness': {'default': 0.025, 'range': [0.025, 0.05]},
            'EModul': {'default': 5e6,  'range': [5e6, 5e7]},
            'nu':{'default': 0.35, 'list': [0.35]}, # not varied yet
            'rho':{'default': 950, 'range': [900, 1200]},
            'beamInterfaceNameLeft': {'default': 'leftSide', 'list': ['leftSide']},
            'beamInterfaceNameRight': {'default': 'rightSide', 'list': ['rightSide']},
            'partName': {'default': 'flexibleBeamHollowPart', 'list': ['flexibleBeamHollowPart']},
        },
        'difficulty': 5,
        'sampleFileName': 'flexibleBeamHollowPart.py',
    },
    
    # 57) TM, SW
    'flexibleBeamDrilledPart': {
        'description': (
            'A flexible beam with a drilled hole as a machine part, with the geometry, interfaces, and material being given.'
            ' The beam has a length of {beamLength} m in global x-direction,'
            ' width of {beamWidth} m in global y-direction, and height of {beamHeight} m in global z-direction.'
            ' The beam is centered at the origin ' + groundPos + ' m.'
            ' The beam has a circular hole drilled along the negative x-axis, starting at (0.5' + multChar + '{beamLength}, 0, 0) m,'
            ' having a radius of {drillHoleRadiusFactor}' + multChar + 'min(y,z) m and drilling depth of {drillHoleDepthFactor}' + multChar + '{beamLength} m, where y and z are the beam\'s width, and height.'
            ' The face on the left side (negative x-direction) of the beam is an interface called "{beamInterfaceNameLeft}",'
            ' the face on the right side (positive x-direction) of the beam is an interface called "{beamInterfaceNameRight}".'
            ' The whole beam is made out of rubber{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'beamLength': {'default': 0.5, 'range': [0.5, 2.0]},
            'beamWidth': {'default': 0.05, 'range': [0.05, 0.4]},
            'beamHeight': {'default': 0.05, 'range': [0.05, 0.4]},
            'drillHoleRadiusFactor': {'default': 0.3, 'range': [0.2, 0.3]}, # never bigger or equal to 1/2!
            'drillHoleDepthFactor': {'default': 0.5, 'range': [0.1, 0.9]}, # never bigger than 1!
            'EModul': {'default': 5e6,  'range': [5e6, 5e7]},
            'nu':{'default': 0.35, 'list': [0.35]}, # not varied yet
            'rho':{'default': 950, 'range': [900, 1200]},
            'beamInterfaceNameLeft': {'default': 'leftSide', 'list': ['leftSide']},
            'beamInterfaceNameRight': {'default': 'rightSide', 'list': ['rightSide']},
            'partName': {'default': 'flexibleBeamDrilledPart', 'list': ['flexibleBeamDrilledPart']},
        },
        'difficulty': 5,
        'sampleFileName': 'flexibleBeamDrilledPart.py',
    },
    
    # 58) TM
    'flexibleTablePart': {
        'description': (
            'A solid flexible table as a machine part, with the geometry, interfaces, and material being given.'
            ' The table consists of a rectangular plate and four vertical legs.'
            ' The global axes are defined such that x represents the length, y the width, and z the height of geometries (positive z upwards).'
            ' The centroid of the plate is at the global origin at ' + groundPos + ' m.'
            ' The plate has a length of {plateLength} m width of {plateWidth} m, and thickness (height) of {plateHeight} m.'
            ' Each leg is a rectangular prism with length {legLength} m, width {legWidth} m, and height {legHeight} m.'
            ' The legs are attached to the bottom face of the plate and extend downward in negative z-direction.'
            ' Legs are placed at the four corners of the plate: left-front at (-0.5' + multChar + '{plateLength} m, -0.5' + multChar + '{plateWidth} m),'
            ' left-back at (-0.5' + multChar + '{plateLength} m, 0.5' + multChar + '{plateWidth} m - {legWidth} m),'
            ' right-back at (0.5' + multChar + '{plateLength} m - {legLength} m, 0.5' + multChar + '{plateWidth} m - {legWidth} m),' 
            ' and right-front at (0.5' + multChar + '{plateLength} m - {legLength} m, -0.5' + multChar + '{plateWidth} m).'
            ' There is an additional plate attached to the top face of the plate for mounting purposes,'
            ' with length {topPlateLength} m, width {topPlateWidth} m, and height {topPlateHeight} m.'
            ' This top plate is centered on the table plate and extends upward in positive z-direction.'
            ' The interfaces are defined as follows: the bottom faces of the four legs (lowest z-faces) are represented by an interface'
            ' called "{interfaceNameLegs}", and the top face of the top plate (highest z-face) is an interface called "{interfaceNameTopPlate}".'
            ' The whole table forms one geometrically connected solid body and is made out of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'plateLength': {'default': 1.0, 'range': [0.5, 2.0]},
            'plateWidth': {'default': 0.75, 'range': [0.5, 1.0]},
            'plateHeight': {'default': 0.05, 'range': [0.025, 0.2]},
            'legHeight': {'default': 0.75, 'range': [0.5, 1.0]},
            'legWidth': {'default': 0.05, 'range': [0.025, 0.2]},
            'legLength': {'default': 0.1, 'range': [0.05, 0.25]},
            'topPlateWidth': {'default': 0.25, 'range': [0.1, 0.25]},
            'topPlateLength': {'default': 0.25, 'range': [0.1, 0.25]},
            'topPlateHeight': {'default': 0.025, 'range': [0.01, 0.1]},
            'EModul': { 'default': 2.1e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]}, # not varied yet
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'interfaceNameLegs': {'default': 'legs', 'list': ['legs']},
            'interfaceNameTopPlate': {'default': 'topPlate', 'list': ['topPlate']},
            'partName': {'default': 'flexibleTablePart', 'list': ['flexibleTablePart']}
        },
        'difficulty': 5,
        'sampleFileName': 'flexibleTablePart.py'
    },
    
    # 59) TM, SW
    'windTurbineTowerPart': {
        'description': (
            'A flexible wind turbine tower consisting of a ground cylinder, a vertical tower cylinder, and a nacelle.'
            ' The ground is modeled as a cylinder whose axis coincides with the global z-axis, starting at ' + groundPos + ' m.'
            ' It has radius {groundRadius} m and height {groundHeight} m.'
            ' The bottom face of this cylinder forms the interface "{groundInterfaceName}".'
            ' The tower, modelled as cylinder, also starts at ' + groundPos + ' m and it\'s axis is also the global z-axis.'
            ' The tower has radius {towerRadiusFactor}' + multChar + 'min({nacelleLength}, {nacelleWidth}) m and height {towerHeight} m.'
            ' The nacelle of the wind turbine, modelled as cuboid, is placed such that it\'s centroid lies at the top of the tower cylinder.'
            ' The nacelle has length {nacelleLength} m along the x-axis, {nacelleWidth} m along the y-axis, and {nacelleHeight} m along the z-axis.'
            ' The edges of the top (max. z-direction) face of the nacelle are filleted using a radius of'
            ' {filletRadiusFactor}' + multChar + 'min({nacelleLength}, {nacelleWidth}, {nacelleHeight}) m.'
            ' A cylindrical hole representing the main bearing starts at (-{nacelleLength}/2, 0, {towerHeight}) m and is parallel to the global x-axis.'
            ' The hole has radius {mainBearingRadiusFactor}' + multChar + 'min({nacelleWidth}, {nacelleHeight}) m.'
            ' The hole goes through the whole length of the nacelle and also cuts through the tower, such that an axle can pass through both.'
            ' The curved cylindrical surface of this hole forms the interface "{mainBearingInterfaceName}".'
            ' The complete tower structure is made of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'towerHeight': {'default': 20.0, 'range': [15.0, 30.0]},
            'towerRadiusFactor': {'default': 0.2, 'range': [0.2, 0.4]},
            'nacelleLength': {'default': 3.0, 'range': [1.5, 5]},
            'nacelleWidth': {'default': 2.0, 'range': [1.5, 5]},
            'nacelleHeight': {'default': 2.0, 'range': [1.5, 5.0]},
            'groundRadius': {'default': 3, 'range': [3, 5]},
            'groundHeight': {'default': 3.0, 'range': [1.0, 3.0]},
            'filletRadiusFactor': {'default': 0.2, 'range': [0.2, 0.3]}, # never bigger or equal to 1/2!
            'mainBearingRadiusFactor': {'default': 0.2, 'range': [0.2, 0.3]}, # never bigger or equal to 1/2!
            'EModul': { 'default': 2.1e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]}, # not varied yet
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'groundInterfaceName': {'default': 'ground', 'list': ['ground']},
            'mainBearingInterfaceName': {'default': 'mainBearing', 'list': ['mainBearing']},
            'partName': {'default': 'windTurbineTowerPart', 'list': ['windTurbineTowerPart']}
        },
        'difficulty': 6,
        'sampleFileName': 'windTurbineTowerPart.py'
    },
    
    # 60) TM, SW
    'windTurbineRotorPart': {
        'description': (
            'A flexible wind turbine rotor consisting of a stepped shaft and three blades.'
            ' The global x-axis is the rotor axis.'
            ' The inner shaft is a cylinder starting at (-{nacelleLength}/2, 0, 0) m with axis along the x-direction,'
            ' length {nacelleLength} m and radius {mainBearingRadiusFactor}' + multChar + 'min({nacelleWidth}, {nacelleHeight}) m.'
            ' The curved cylindrical surface of this inner shaft forms the interface "{shaftInterfaceName}".'
            ' A second cylindrical shaft segment (outer hub) continues in the positive x-direction starting at'
            ' ({nacelleLength}/2, 0, 0) m. This outer hub has length {outerHubLength} m and radius {outerHubRadiusFactor}' + multChar + 'innerShaftRadius,'
            ' where innerShaftRadius is the radius of the inner shaft calculated before.'
            ' A rectangular blade is attached to the outer hub and is initially defined as a cuboid spanned by the two'
            ' points ({nacelleLength}/2, -{bladeThickness}/2, 0) m and'
            ' ({nacelleLength}/2 + {outerHubLength}, {bladeThickness}/2, -{bladeLength}) m.'
            ' This blade is rotated by {angleOfAttack}° around an axis parallel to the global z-axis passing through'
            ' ({nacelleLength}/2 + {outerHubLength}/2, 0, 0) m to define its angle of attack.'
            ' Two additional blades are generated by rotating this blade by 120° and 240° around the global x-axis.'
            ' The rotor therefore consists of three equally spaced blades around the rotor axis.'
            ' The complete rotor is made of aluminium{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'nacelleLength': {'default': 3.0, 'range': [1.0, 6.0]},
            'nacelleWidth': {'default': 2.0, 'range': [1.5, 5]},
            'nacelleHeight': {'default': 2.0, 'range': [1.5, 5.0]},
            'mainBearingRadiusFactor': {'default': 0.2, 'range': [0.2, 0.3]}, # never bigger or equal to 1/2!
            'outerHubRadiusFactor': {'default': 1.2, 'range': [1.1, 1.4]}, # with one the rotor jumps; should not
            'outerHubLength': {'default': 1.0, 'range': [0.3, 1.0]},
            'angleOfAttack': {'default': 50, 'range': [40, 60]},
            'bladeLength': {'default': 10.0, 'range': [2.0, 20.0]},
            'bladeThickness': {'default': 0.10, 'range': [0.05, 0.5]},
            'EModul': {'default': 7e10, 'range': [7e9, 8e10]},
            'nu': {'default': 0.33, 'list': [0.33]}, # not varied yet
            'rho': {'default': 2700, 'range': [2500, 2900]},
            'shaftInterfaceName': {'default': 'shaft', 'list': ['shaft']},
            'partName': {'default': 'windTurbineRotorPart', 'list': ['windTurbineRotorPart']}
        },
        'difficulty': 6,
        'sampleFileName': 'windTurbineRotorPart.py'
        },
        
    # 61) TM
    'grindingBenchHousingPart': {
        'description': (
            'A grinding bench housing modeled as a rectangular box centered at the global origin at ' + groundPos + ' m.'
            ' The housing has length {housingLength} m in x, width {housingWidth} m in y, and height {housingHeight} m in z.'
            ' The top edges (maximum-z face edges) are filleted with radius {filletRadiusFactor}' + multChar + 'min(x,y,z),'
            ' where x, y, and z are the housing\'s dimensions.'
            ' The bottom face of the housing (minimum in z) is an interface called "{housingBottomInterfaceName}".'
            ' A cylindrical spindle passage of radius {spindleRadiusFactor}' + multChar + 'min({housingWidth}, {housingHeight}) m runs through the housing\'s'
            ' centroid, aligned with the global x-axis. The spindle passage goes through the whole length of the housing.'
            ' The curved face of this hole is assigned to the interface "{spindlePassageInterfaceName}".'
            ' The whole housing is made of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'housingLength': {'default': 1.5, 'range': [0.6, 2.0]},
            'housingWidth': {'default': 1.0, 'range': [0.5, 2.0]},
            'housingHeight': {'default': 1.5, 'range': [0.5, 3.0]},
            'filletRadiusFactor': {'default': 0.2, 'range': [0.2, 0.3]}, # never bigger or equal to 1/2!
            'spindleRadiusFactor': {'default': 0.2, 'range': [0.2, 0.3]}, # never bigger or equal to 1/2!
            'EModul': { 'default': 2.1e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]}, # not varied yet
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'housingBottomInterfaceName': {'default': 'housingBottom', 'list': ['housingBottom']},
            'spindlePassageInterfaceName': {'default': 'spindlePassage', 'list': ['spindlePassage']},
            'partName': {'default': 'grindingBenchHousingPart', 'list': ['grindingBenchHousingPart']}
        },
        'difficulty': 5,
        'sampleFileName': 'grindingBenchHousingPart.py'
    },
        
    # 62) TM, SW
    'grindingBenchSpindleWithDisksPart': {
        'description': (
            'A grinding bench spindle with two grinding disks, modeled as three coaxial cylinders along the global x-axis.'
            ' The central spindle has length of 1.5' + multChar + '{housingLength} m and radius'
            ' {spindleRadiusFactor}' + multChar + 'min({housingWidth}, {housingHeight}) m, centered so that its axial center is at x = 0 m.'
            ' A right disk (radius {diskRadius} m, thickness {diskThickness} m) is attached at the positive-x end of the spindle;'
            ' a left disk (same dimensions) is attached at the negative-x end.'
            ' Interfaces are defined on curved cylindrical faces: the spindle\'s curved surface is "{spindleInterfaceName}",'
            ' the right disk\'s curved surface is "{diskOuterRightInterfaceName}", and the left disk\'s curved surface is "{diskOuterLeftInterfaceName}".'
            ' Disk edges are rounded with a fillet radius of {filletRadiusFactor}' + multChar + '{diskThickness}.'
            ' The whole spindle with disks is made of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'housingLength': {'default': 1.5, 'range': [0.6, 2.0]},
            'housingWidth': {'default': 1.0, 'range': [0.5, 2.0]},
            'housingHeight': {'default': 1.5, 'range': [0.5, 3.0]},
            'spindleRadiusFactor': {'default': 0.2, 'range': [0.2, 0.3]}, # never bigger or equal to 1/2!
            'diskThickness': {'default': 0.35, 'range': [0.05, 0.8]},
            'diskRadius': {'default': 0.75, 'range': [0.2, 1.5]},
            'filletRadiusFactor': {'default': 0.2, 'range': [0.2, 0.3]}, # never bigger or equal to 1/2!
            'EModul': { 'default': 2.1e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]}, # not varied yet
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'spindleInterfaceName': {'default': 'spindle', 'list': ['spindle']},
            'diskOuterRightInterfaceName': {'default': 'diskOuterRight', 'list': ['diskOuterRight']},
            'diskOuterLeftInterfaceName': {'default': 'diskOuterLeft', 'list': ['diskOuterLeft']},
            'partName': {'default': 'grindingBenchSpindleWithDisksPart', 'list': ['grindingBenchSpindleWithDisksPart']}
        },
        'difficulty': 5,
        'sampleFileName': 'grindingBenchSpindleWithDisksPart.py'
    },
    

    # 63) TM
    'hingeLeafGapSidePart': {
        'description': (
            ' The leaf is created by extruding a 2D cross-section in the YZ-plane along the x-direction by {leafLength} m.'
            ' The workplane for the cross-section is defined at (-0.5' + multChar + '{leafLength}, 0, 0).'
            ' The overall leaf dimensions including the knuckle are (plate length {leafLength} m, width {leafWidth} m and thickness {leafHeight} m + the radius of the knuckel of {knuckleRadius} m).'
            ' The construction-steps for the boundary wire are:\n'
            ' Start from point at (-0.5' + multChar + '{leafWidth}, -0.5' + multChar + '{leafHeight}),'
            ' draw a line to (0.5' + multChar + '{leafWidth}, -0.5' + multChar + '{leafHeight});'
            ' draw a line to (0.5' + multChar + '{leafWidth}, 0.5' + multChar + '{leafHeight});'
            ' draw a semicircle to (0.5' + multChar + '{leafWidth} - 2' + multChar + '{knuckleRadius}, 0.5' + multChar + '{leafHeight}) with tangent direction in positive z-direction;'
            ' draw a line to (-0.5' + multChar + '{leafWidth}, 0.5' + multChar + '{leafHeight})'
            ' and finally closing the wire.'
            ' After extrusion, a cuboidal gap is created.'
            ' The gap spans from (-0.5' + multChar + '{gapLength}, 0.5' + multChar + '{leafWidth} - 2' + multChar + '{knuckleRadius}, -0.5' + multChar + '{leafHeight}) m'
            ' to (0.5' + multChar + '{gapLength}, 0.5' + multChar + '{leafWidth}, 0.5' + multChar + '{leafHeight} + {knuckleRadius}) m.'
            ' Two bore holes are created along the X-direction inside the knuckle'
            ' at Y = 0.5' + multChar + '{leafWidth} - {knuckleRadius}, Z = 0.5' + multChar + '{leafHeight}).'
            ' Each bore has radius {pinRadius} m, length 0.5' + multChar + '{leafLength} - 0.5' + multChar + '{gapLength} m'
            ' and is drilled along the X-axis from both outer faces toward the central gap, terminating at the gap.'
            ' The curved cylindrical face of the bore is an interface named "{boreInterfaceNamePosX}",'
            ' and the curved cylindrical face of the other bore is an interface named "{boreInterfaceNameNegX}".'
            ' The leaf is mounted on the face with the minimum Y-coordinate, which is defined as the interface "{leafInterfaceName}".'
            ' The hinge leaf is made of steel{HEADER:meshPropertiesHeader}'
        ),                              
        'parameters': {
            'leafLength': {'default': 0.08, 'range': [0.05, 0.08]},
            'leafWidth': {'default': 0.08, 'range': [0.02, 0.08]},
            'leafHeight': {'default': 0.008, 'range': [0.008, 0.01]},
            'knuckleRadius': {'default': 0.004, 'range': [0.004, 0.01]},
            'gapLength': {'default': 0.0375, 'range': [0.0125, 0.0375]},
            'pinRadius': {'default': 0.0015, 'range': [0.0015, 0.003]},
            'EModul': {'default': 2.0e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]},
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'leafInterfaceName': {'default': 'leafMount', 'list': ['leafMount']},
            'boreInterfaceNamePosX': {'default': 'hingeBorePosX', 'list': ['hingeBorePosX']},
            'boreInterfaceNameNegX': {'default': 'hingeBoreNegX', 'list': ['hingeBoreNegX']},
            'partName': {'default': 'hingeLeafGapSidePart', 'list': ['hingeLeafGapSidePart']}
        },
        'difficulty': 6,
        'sampleFileName': 'hingeLeafGapSidePart.py'
    },
    
    # 64) TM
    'hingeLeafPinSidePart': {
        'description': (
            'One leaf of a door hinge consisting of an extruded plate with a cylindrical knuckle and a gap seperating the knuckle into two outer segments.'
            ' The leaf is created by extruding a 2D cross-section in the YZ-plane along the X-direction by {leafLength} m.'
            ' The workplane for the cross-section is defined at (-0.5' + multChar + '{leafLength}, 0, 0).'
            ' The overall leaf dimensions including the knuckle are (plate length {leafLength} m, width {leafWidth} m and thickness {leafHeight} m + the radius of the knuckel of {knuckleRadius} m).'
            ' The construction-steps for the boundary wire are:\n'
            ' Start from point at (-0.5' + multChar + '{leafWidth}, -0.5' + multChar + '{leafHeight}),'
            ' draw a line to (0.5' + multChar + '{leafWidth}, -0.5' + multChar + '{leafHeight});'
            ' draw a line to (0.5' + multChar + '{leafWidth}, 0.5' + multChar + '{leafHeight});'
            ' draw a semicircle to (0.5' + multChar + '{leafWidth} - 2' + multChar + '{knuckleRadius}, 0.5' + multChar + '{leafHeight}) with tangent direction in positive z-direction;'
            ' draw a line to (-0.5' + multChar + '{leafWidth}, 0.5' + multChar + '{leafHeight})'
            ' and finally closing the wire.'
            ' After extrusion, two cuboidal gaps are created.'
            ' The first gap spans from (-0.5' + multChar + '{leafLength}, 0.5' + multChar + '{leafWidth} - 2' + multChar + '{knuckleRadius}, -0.5' + multChar + '{leafHeight}) m'
            ' to (-0.5' + multChar + '{gapLength}, 0.5' + multChar + '{leafWidth}, 0.5' + multChar + '{leafHeight} + {knuckleRadius}) m.'
            ' The second gap has the same dimensions and is mirrored across the YZ-plane.'
            ' Two cylindrical hinge pins are attached coaxially to the knuckle at Y = 0.5' + multChar + '{leafWidth} - {knuckleRadius}, Z = 0.5' + multChar + '{leafHeight}).'
            ' and extend along the X-direction from both sides of the knuckle.' # knuckel's surface?
            ' Each pin has radius {pinRadius} m and length 0.5' + multChar + '{leafLength} - 0.5' + multChar + '{gapLength} m.'
            ' The curved cylindrical face of the pin is an interface named  "{pinInterfaceNamePosX}",'
            ' and the curved cylindrical face of the other pin is an interface named "{pinInterfaceNameNegX}".'
            ' The leaf is mounted on the face with the minimum Y-coordinate, which is defined as the interface "{leafInterfaceName}".'
            ' The hinge leaf is made of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'leafLength': {'default': 0.08, 'range': [0.05, 0.08]},
            'leafWidth': {'default': 0.08, 'range': [0.02, 0.08]},
            'leafHeight': {'default': 0.008, 'range': [0.008, 0.01]},
            'knuckleRadius': {'default': 0.004, 'range': [0.004, 0.01]},
            'gapLength': {'default': 0.0375, 'range': [0.0125, 0.0375]},
            'pinRadius': {'default': 0.0015, 'range': [0.0015, 0.003]},
            'EModul': {'default': 2.0e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]},
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'leafInterfaceName': {'default': 'leafMount', 'list': ['leafMount']},
            'pinInterfaceNamePosX': {'default': 'hingePinPosX', 'list': ['hingePinPosX']},
            'pinInterfaceNameNegX': {'default': 'hingePinNegX', 'list': ['hingePinNegX']},
            'partName': {'default': 'hingeLeafPinSidePart', 'list': ['hingeLeafPinSidePart']}
        },
        'difficulty': 6,
        'sampleFileName': 'hingeLeafPinSidePart.py'
    },
    
    # 65) TM
    'mountingBracketPart': {
        'description': (
            'A mounting bracket with a rectangular base plate with four corner bores and two elevated side holders.'
            ' The plate has length {plateWidthX} m in X, width {plateWidthY} m in Y, and thickness {plateThicknessZ} m in Z,' 
            ' and is centered in the XY-plane with its bottom face at z = 0 m.'
            ' Four vertical through-holes with radius {boreRadiusPlate} m are located symmetrically near the plate corners, each offset by'
            ' {boreDistance} m from the respective plate edges.'
            ' Two identical rectangular holders are mounted on top of the plate.'
            ' One holder has its center at (0.25' + multChar + '{plateWidthX}, 0, 0) m and the other at (-0.25' + multChar + '{plateWidthX}, 0, 0) m.'
            ' Each holder has length {holderThicknessX} m in X, width {holderWidthY} m in Y, and height {holderHeightZ} m in Z, and is attached on the top surface of the plate.'
            ' Each holder has two chamfers of size {chamferSize} m:'
            ' one on the edge that is at the maximum y and maximum z extend of the holder,'
            ' one on the edge that is at the minimum y and maximum z extend of the holder.'
            ' Each holder contains a horizontal cylindrical through-hole with radius {holderBoreRadius} m aligned along the global X-axis.'
            ' The bores are centered at (center of the holder in X - 0.5' + multChar + '{holderThicknessX}, 0, {plateThicknessZ} +  0.5' + multChar + '{holderHeightZ}) m.'
            ' The curved cylindrical faces of the plate bores are part of the same interface called "{interfaceNamePlateBores}",'
            ' the curved cylindrical face of the left holder bore is an interface named "{interfaceNameHolderLeftBore}",'
            ' and the curved cylindrical face of the right holder bore is an interface named "{interfaceNameHolderRightBore}".'
            ' The complete structure is formed by subtracting all bores from the plate and holders and combining the solids into a single part.'
            ' The mounting bracked is made of aluminum{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'plateWidthX': {'default': 1.5, 'range': [1.0, 2.0]},
            'plateWidthY': {'default': 1.5, 'range': [1.0, 2.0]},
            'plateThicknessZ': {'default': 0.125, 'range': [0.05, 0.25]},
            'boreRadiusPlate': {'default': 0.02, 'range': [0.01, 0.03]},
            'boreDistance': {'default': 0.1, 'range': [0.05, 0.2]},
            'holderThicknessX': {'default': 0.2, 'range': [0.1, 0.25]},
            'holderWidthY': {'default': 0.625, 'range': [0.625, 0.75]},
            'holderHeightZ': {'default': 0.625, 'range': [0.625, 0.75]},
            'holderBoreRadius': {'default': 0.25, 'range': [0.125, 0.25]},
            'chamferSize': {'default': 0.05, 'range': [0.01, 0.1]},
            'EModul': {'default': 2.1e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]},
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'interfaceNamePlateBores': {'default': 'plateBores', 'list': ['plateBores']},
            'interfaceNameHolderLeftBore': {'default': 'holderLeftBore', 'list': ['holderLeftBore']},
            'interfaceNameHolderRightBore': {'default': 'holderRightBore', 'list': ['holderRightBore']},
            'partName': {'default': 'mountingBracketPart', 'list': ['mountingBracketPart']}
        },
        'difficulty': 4,
        'sampleFileName': 'mountingBracketPart.py'
    },
    
    # 66) TM, SW
    'bearingBlockWithPedestalPart': {
        'description': (
            'A bearing block with pedestal as a machine part, with the geometry, interfaces, and material being given.'
            ' The geometry of the bearing block consists of a ground plate, a pedestal (block and cylinder), and a joint bore.\n'
            '- The ground plate is a cuboid centered at the global origin ' + groundPos + ' m,'
            ' having a length of {plateLength} m in global x-direction,'
            ' width of {plateWidth} m in global y-direction, and height of {plateHeight} m in global z-direction.\n'
            '- The bottom surface of the ground plate (at z = -{plateHeight}/2) m is an interface named "{groundPlateInterfaceName}".\n'
            '- The edges of the top surface of the ground plate have chamfers of size {chamferSizeFactor}' + multChar + '{plateHeight} m.\n'
            '- The pedestal consists of a rectangular block and a cylinder sitting on top of the ground plate:\n'
            '  - The block has a width of 2' + multChar + '({bearingRadius}+{wallThickness}) m in x-direction,'
            ' a length of {bearingHeight} m in y-direction,'
            ' and a height of {wallThickness}+{bearingRadius} m in z-direction.'
            ' Its bottom face is flush with the top of the ground plate (at z = +{plateHeight}/2) m.\n'
            '  - The cylinder has its axis parallel to the global y-axis.'
            ' The footpoint of the cylinder is at (0, -{bearingHeight}/2, {plateHeight}/2+{bearingRadius}+{wallThickness}) m.'
            ' Its outer radius is {bearingRadius}+{wallThickness} m and its length is {bearingHeight} m.\n'
            '  - The pedestal has two chamfers of size {chamferSizeFactor}' + multChar + '{bearingHeight} m.'
            ' One on its (+x,+y) and one on its (-x,-y) corner edges (do NOT chamfer all edges of pedestral).\n'
            '- The joint bore is a cylinder with the same axis and footpoint as the pedestal cylinder,'
            ' with inner radius {bearingRadius} m and length {bearingHeight} m.'
            ' All faces of the bore are an interface named "{jointInterfaceName}".\n'
            'The whole bearing block is made out of rubber{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'bearingHeight': {'default': 0.075, 'range': [0.075, 0.15]},
            'bearingRadius': {'default': 0.05, 'range': [0.05, 0.2]},
            'wallThickness': {'default': 0.07, 'range': [0.05, 0.07]},
            'chamferSizeFactor': {'default': 0.15, 'range': [0.1, 0.2]},
            'plateLength': {'default': 0.6, 'range': [0.6, 0.8]},
            'plateWidth': {'default': 0.4, 'range': [0.3, 0.5]},
            'plateHeight': {'default': 0.2, 'range': [0.15, 0.25]},
            'EModul': {'default': 5e6, 'range': [5e6, 5e7]},
            'nu': {'default': 0.49, 'list': [0.49]},
            'rho': {'default': 950, 'range': [900, 1200]},
            'groundPlateInterfaceName': {'default': 'groundPlate', 'list': ['groundPlate']},
            'jointInterfaceName': {'default': 'joint', 'list': ['joint']},
            'partName': {'default': 'bearingBlockWithPedestalPart', 'list': ['bearingBlockWithPedestalPart']},
        },
        'difficulty': 6,
        'sampleFileName': 'bearingBlockWithPedestalPart.py',
    },

    # 67) TM, SW
    # common mistake: due to geometry proportions the mesh is fine --> number of nodes exceeds maxNumberOfNodes 
    'crankPartSimple': {
        'description': (
            'A crank is modeled as a cylindrical disk with an offset vertical pin.'
            ' A vertical crank pin of radius {pinRadius} m and length {pinLength} m is attached on the disk top at'
            ' x = {crankRadius} m, y = 0 m. The pin\'s curved cylindrical face is an interface called "{crankPinInterfaceName}".'
            ' The outer diameter of the disk is flush with the vertical pin, resulting in a disk-radius of {crankRadius} m + {pinRadius} m and thickness {diskThickness} m along the global z-axis.' 
            ' The disk is centered at the origin (x=0, y=0, z=0).'
            ' Its bottom face (global minimum in z) is an interface called "{diskBottomInterfaceName}".'
            ' The whole crank is made of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'diskThickness': {'default': 0.0125, 'range': [0.0125, 0.025]},
            'crankRadius': {'default': 0.05, 'range': [0.05, 0.15]},
            'pinRadius': {'default': 0.0125, 'range': [0.0125, 0.025]},
            'pinLength': {'default': 0.025, 'range': [0.025, 0.075]},
            'EModul': { 'default': 2.1e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]}, # not varied yet
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'diskBottomInterfaceName': {'default': 'diskBottom', 'list': ['diskBottom']},
            'crankPinInterfaceName': {'default': 'crankPin', 'list': ['crankPin']},
            'partName': {'default': 'crankPartSimple', 'list': ['crankPartSimple']}
        },
        'difficulty': 5,
        'sampleFileName': 'crankPartSimple.py'
    },

    # 68) TM, SW 
    'rodPartSimple': {
        'description': (
            'A connecting rod having a stadium shape.'
            ' The global coordinate system is defined such that x is the rod axis, y the transverse direction, and z the thickness direction.'
            ' The center of the rod is centered at the origin (x=0, y=0, z=0).'
            ' The height of the rod in z-direction is {pinLength} m.'
            ' The distance between the two bores of the rod is {boreDistance} m along the rod axis.'
            ' Each cylindrical shaped end of the stadium has a radius of 2' + multChar + '{pinRadius} m.'
            ' Each bore in z-direction has radius {pinRadius} m and is located on the rod axis.'
            ' All faces of the left (on the negative x axis located) bore define the interface "{rodEndCrankInterfaceName}".'
            ' All faces of the right (on the positive x axis located) bore define the interface "{rodEndSliderInterfaceName}".'
            ' The rod is made of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'pinRadius': {'default': 0.0125, 'range': [0.0125, 0.025]},
            'pinLength': {'default': 0.025, 'range': [0.025, 0.075]},
            'boreDistance': {'default': 0.125, 'range': [0.125, 0.25]},
            'EModul': { 'default': 2.1e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]}, # not varied yet
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'rodEndCrankInterfaceName': {'default': 'rodEndCrank', 'list': ['rodEndCrank']},
            'rodEndSliderInterfaceName': {'default': 'rodEndSlider', 'list': ['rodEndSlider']},
            'partName': {'default': 'rodPartSimple', 'list': ['rodPartSimple']}
        },
        'difficulty': 5,
        'sampleFileName': 'rodPartSimple.py'
    },
    
    # 69) TM 
    'sliderPartSimple': {
        'description': (
            'A slider of a slider-crank mechanism, modeled as a rectangular block a vertical pin on top.'
            ' The slider block has length {sliderLength} m in x, width {sliderWidth} m in y, and height {diskThickness} m in z.'
            ' The slider block is centered at the origin (x=0, y=0, z=0).'
            ' The bottom face of the slider (minimum z) is an interface called "{sliderGuideInterfaceName}" for prismatic guidance along x.'
            ' A vertical cylindrical pin of radius {pinRadius} m and length {pinLength} m is placed at the center of the slider top;'
            ' its curved cylindrical face is an interface called "{sliderPinInterfaceName}".'
            ' The slider is made of steel{HEADER:meshPropertiesHeader}'
            ),
        'parameters': {
            'pinRadius': {'default': 0.0125, 'range': [0.0125, 0.025]},
            'pinLength': {'default': 0.025, 'range': [0.025, 0.075]},
            'sliderLength': {'default': 0.1, 'range': [0.05, 0.2]},
            'sliderWidth': {'default': 0.05, 'range': [0.025, 0.1]},
            'diskThickness': {'default': 0.0125, 'range': [0.0125, 0.025]},
            'EModul': { 'default': 2.1e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]}, # not varied yet
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'sliderGuideInterfaceName': {'default': 'sliderBottom', 'list': ['sliderBottom']},
            'sliderPinInterfaceName': {'default': 'sliderPin', 'list': ['sliderPin']},
            'partName': {'default': 'sliderPartSimple', 'list': ['sliderPartSimple']}
        },
        'difficulty': 5,
        'sampleFileName': 'sliderPartSimple.py'
    },

    # 70) TM
    'crankPartAdvanced': {
        'description': (
            'A crank modeled as a 3D body created by extruding a closed 2D bone-shaped cross-section in the XY-plane along the z-axis by {crankHeight} m,'
            ' with a pin added at the left end and a bore drilled at the right end.'
            ' The crank is centered at the global origin (x=0, y=0, z=0), so the workplane is placed at z = -{crankHeight}/2 before extrusion.'
            ' The cross-section has a total length of {crankLength} m in the x-direction and contains four semicircular arcs of radius'
            ' bossRadius = min(0.5' + multChar + '{crankWidth}, 2' + multChar + '{connectionWidth}, 0.24' + multChar + '{crankLength}).'
            ' Starting at the point (-0.5' + multChar + '{crankLength}, -0.5' + multChar + '{connectionWidth}), the closed outline of the cross-section is drawn by:'
            ' a semicircular arc bulging outward in the negative y-direction to (-0.5' + multChar + '{crankLength} + 2' + multChar + 'bossRadius, -0.5' + multChar + '{connectionWidth}),'
            ' a straight edge to (0.5' + multChar + '{crankLength} - 2' + multChar + 'bossRadius, -0.5' + multChar + '{connectionWidth}),'
            ' a semicircular arc bulging outward in the negative y-direction to (0.5' + multChar + '{crankLength}, -0.5' + multChar + '{connectionWidth}),'
            ' a straight vertical edge to (0.5' + multChar + '{crankLength}, 0.5' + multChar + '{connectionWidth}),'
            ' a semicircular arc bulging outward in the positive y-direction to (0.5' + multChar + '{crankLength} - 2' + multChar + 'bossRadius, 0.5' + multChar + '{connectionWidth}),'
            ' a straight edge to (-0.5' + multChar + '{crankLength} + 2' + multChar + 'bossRadius, 0.5' + multChar + '{connectionWidth}),'
            ' a semicircular arc bulging outward in the positive y-direction to (-0.5' + multChar + '{crankLength}, 0.5' + multChar + '{connectionWidth}),'
            ' and a straight vertical edge closing the outline back to the start point.'
            ' This forms a straight central bar of width {connectionWidth} m with a rounded boss at each end;'
            ' the boss centers lie on the x-axis at x = ±(0.5' + multChar + '{crankLength} - bossRadius).'
            ' After extrusion, the edges of the top face (maximum z) are chamfered by {chamferSizeFactor}' + multChar + '{crankHeight} m.'
            ' A cylindrical pin of radius {pinRadius} m and height {crankHeight} m is attached at the left boss center:'
            ' starting at the bottom face of the body at z = -0.5' + multChar + '{crankHeight}, it extends by {crankHeight} m in the negative z-direction.'
            ' Its curved lateral face defines the interface "{pinLeftInterfaceName}" (main bearing / ground revolute joint).'
            ' A cylindrical bore of radius {pinRadius} m is drilled through the full height at the right boss center.'
            ' Its curved lateral face defines the interface "{boreRightInterfaceName}" (crank pin connection).'
            ' The whole crank is made of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'crankHeight': {'default': 0.025, 'range': [0.0125, 0.05]},
            'crankLength': {'default': 0.1, 'range': [0.08, 0.175]},  # always < boreDistance min
            'crankWidth': {'default': 0.1, 'range': [0.1, 0.2]},
            'connectionWidth': {'default': 0.02, 'range': [0.02, 0.05]},
            'pinRadius': {'default': 0.0125, 'range': [0.01, 0.025]},
            'chamferSizeFactor': {'default': 0.1, 'list': [0.1]},
            'EModul': {'default': 2e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.3, 'list': [0.3]},
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'pinLeftInterfaceName': {'default': 'pinLeft', 'list': ['pinLeft']},
            'boreRightInterfaceName': {'default': 'boreRight', 'list': ['boreRight']},
            'partName': {'default': 'crankPartAdvanced', 'list': ['crankPartAdvanced']}
        },
        'difficulty': 6,
        'sampleFileName': 'crankPartAdvanced.py'
    },

    # 71) TM, SW
    'rodPartAdvanced': {
        'description': (
            'A connecting rod modeled as a 3D body created by extruding an asymmetric 2D cross-section in the XY-plane along the z-axis by {pinLength} m,' # rodHeight = pinLength
            ' with a cylindrical pin added at the big end and a cylindrical bore drilled at the small end.'
            ' The cross-section is only symmetric about the XZ-plane (Y=0), resembling a real conrod shape:'
            ' a large boss at the crank (big) end on the left and a smaller boss at the slider (small) end on the right,'
            ' connected by two straight tapered web edges.'
            ' The boss radii are computed as:'
            ' bigBossRadius = min(0.5' + multChar + '{bigEndWidth}, 2' + multChar + '{bigWebWidth}, 0.24' + multChar + '{boreDistance}),'
            ' smallBossRadius = min(0.5' + multChar + '{smallEndWidth}, 2' + multChar + '{smallWebWidth}, 0.24' + multChar + '{boreDistance}).'
            ' This guarantees bigBossRadius + smallBossRadius < {boreDistance} so the web has positive length.'
            ' The workplane origin is placed at (0, 0, -{pinLength}/2) before extrusion.'
            ' The counterclockwise outline (viewed from +z) is constructed as follows:'
            ' start at the leftmost point (xLeft, 0) where xLeft = -0.5' + multChar + '{boreDistance} - bigBossRadius,'
            ' go down to (xLeft, -{bigWebWidth}/2),'
            ' draw an arc to (xLeft + 2' + multChar + 'bigBossRadius, -{bigWebWidth}/2) with start tangent (0, -1) (big boss bottom bulge),'
            ' draw a tapered line to (xRight - 2' + multChar + 'smallBossRadius, -{smallWebWidth}/2) (bottom web, tapered),'
            ' draw an arc to (xRight, -{smallWebWidth}/2) with start tangent (0, -1) (small boss bottom bulge),'
            ' where xRight = +0.5' + multChar + '{boreDistance} + smallBossRadius,'
            ' go up to (xRight, +{smallWebWidth}/2),'
            ' draw an arc to (xRight - 2' + multChar + 'smallBossRadius, +{smallWebWidth}/2) with start tangent (0, +1) (small boss top bulge),'
            ' draw a tapered line to (xLeft + 2' + multChar + 'bigBossRadius, +{bigWebWidth}/2) (top web, tapered),'
            ' draw an arc to (xLeft, +{bigWebWidth}/2) with start tangent (0, +1) (big boss top bulge),'
            ' go back to (xLeft, 0) and close the wire.'
            ' The cross-section is extruded along the z-axis by {pinLength} m using occ.Prism.'
            ' The top edges (at z = +{pinLength}/2) are chamfered by {chamferSizeFactor}' + multChar + '{pinLength} m.'
            ' At the big end, a cylindrical pin of radius {pinRadius} m and height {pinLength} m is added at'
            ' (x = -{boreDistance}/2, y = 0), pointing in the negative z-direction from the bottom face of the body'
            ' (z = -{pinLength}/2 downward to z = -1.5' + multChar + '{pinLength}).'
            ' The curved lateral face of this pin defines the interface "{rodEndCrankInterfaceName}" (crank connection).'
            ' At the small end, a cylindrical bore of radius {pinRadius} m and height {pinLength} m is drilled along the z-axis'
            ' at (x = +{boreDistance}/2, y = 0), running from z = -{pinLength}/2 to z = +{pinLength}/2.'
            ' The curved lateral faces of this bore define the interface "{rodEndSliderInterfaceName}" (slider connection).'
            ' The rod is made of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'pinRadius': {'default': 0.0125, 'range': [0.0125, 0.015]},
            'pinLength': {'default': 0.025, 'range': [0.025, 0.075]},
            'bigEndWidth': {'default': 0.05, 'range': [0.05, 0.125]},
            'bigWebWidth': {'default': 0.02, 'range': [0.02, 0.05]},
            'smallEndWidth': {'default': 0.05, 'range': [0.04, 0.06]},
            'smallWebWidth': {'default': 0.03, 'range': [0.03, 0.05]},
            'boreDistance': {'default': 0.25, 'range': [0.2, 0.35]},
            'chamferSizeFactor': {'default': 0.1, 'list': [0.1]},
            'EModul': {'default': 2e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.3, 'list': [0.3]},
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'rodEndCrankInterfaceName': {'default': 'rodEndCrank', 'list': ['rodEndCrank']},
            'rodEndSliderInterfaceName': {'default': 'rodEndSlider', 'list': ['rodEndSlider']},
            'partName': {'default': 'rodPartAdvanced', 'list': ['rodPartAdvanced']}
        },
        'difficulty': 6,
        'sampleFileName': 'rodPartAdvanced.py'
    },

    # 72) TM, SW 
    'sliderPartAdvanced': {
        'description': (
            'A slider of a slider-crank mechanism consisting of an extruded slider body with a symmetric guide rabbet at the bottom and a vertical pin on top.'
            ' The slider body is created by extruding a 2D cross-section in the YZ-plane along the x-direction by {sliderLength} m.'
            ' To keep the final geometry centered at the global origin, the workplane for the cross-section is shifted by -0.5' + multChar + '{sliderLength} m in x-direction before the extrusion.'
            ' The overall slider width in Y-direction is {sliderWidth} m and the total height in Z-direction is {diskThickness} m.'
            ' The outer rectangular limits of the cross-section therefore lie at Y = ±0.5' + multChar + '{sliderWidth} m and Z = +0.5' + multChar + '{diskThickness} m.'
            ' The bottom center of the rectangle is replaced by a symmetric rabbet that forms the guide interface.'
            ' The rabbet has four corner points (all given in m):'
            ' top-left (-0.1' + multChar + '{sliderWidth}, -0.2' + multChar + '{diskThickness}),'
            ' bottom-left (-0.2' + multChar + '{sliderWidth}, -0.5' + multChar + '{diskThickness}),'
            ' bottom-right (+0.2' + multChar + '{sliderWidth}, -0.5' + multChar + '{diskThickness}),'
            ' top-right (+0.1' + multChar + '{sliderWidth}, -0.2' + multChar + '{diskThickness}).'
            ' The rabbet connects to the outer slider boundary at'
            ' (-0.5' + multChar + '{sliderWidth}, -0.2' + multChar + '{diskThickness})'
            ' and (+0.5' + multChar + '{sliderWidth}, -0.2' + multChar + '{diskThickness}).'
            ' The boundary wire is constructed in counter-clockwise order starting at'
            ' (-0.5' + multChar + '{sliderWidth}, -0.2' + multChar + '{diskThickness}),'
            ' then through the rabbet corners, then to the right connection point,'
            ' then to the top-right outer corner (+0.5' + multChar + '{sliderWidth}, +0.5' + multChar + '{diskThickness}),'
            ' then to the top-left outer corner (-0.5' + multChar + '{sliderWidth}, +0.5' + multChar + '{diskThickness}),'
            ' and finally closing the wire.'
            ' The upper edges of the slider body are chamfered by {chamferSizeFactor}' + multChar + '{diskThickness} m.'
            ' A vertical cylindrical pin of radius {pinRadius} m and length {pinLength} m is placed at the center of the top surface of the slider.'
            ' The curved surface of the pin is an interface named "{sliderPinInterfaceName}".'
            ' The bottom surface (minimum along global z-axis) of the rabbet forms the interface "{sliderGuideInterfaceName}" and represents the prismatic guide along the x-direction.'
            ' The slider is made of steel{HEADER:meshPropertiesHeader}'
        ),
        'parameters': {
            'pinRadius': {'default': 0.0125, 'range': [0.0125, 0.025]},
            'pinLength': {'default': 0.025, 'range': [0.025, 0.075]},
            'sliderLength': {'default': 0.1, 'range': [0.05, 0.2]},
            'sliderWidth': {'default': 0.1, 'range': [0.1, 0.2]},
            'diskThickness': {'default': 0.0125, 'range': [0.0125, 0.025]},
            'chamferSizeFactor': {'default': 0.1, 'range': [0.05, 0.2]},
            'EModul': { 'default': 2.1e11, 'range': [2.0e11, 2.2e11]},
            'nu': {'default': 0.30, 'list': [0.30]}, # not varied yet
            'rho': {'default': 7850, 'range': [7800, 8050]},
            'sliderGuideInterfaceName': {'default': 'sliderBottom', 'list': ['sliderBottom']},
            'sliderPinInterfaceName': {'default': 'sliderPin', 'list': ['sliderPin']},
            'partName': {'default': 'sliderPartAdvanced', 'list': ['sliderPartAdvanced']}
        },
        'difficulty': 5,
        'sampleFileName': 'sliderPartAdvanced.py'
    },


}


#%% flexible mbs descriptions
# The header template can be found in utilities.py/ExpandFemHeaders
# never forget units, keep consistence across examples, do not add whitespaces around headers!
flexibleMBSDescriptions = {
    # 73) SW
    'rotorInBearingBlock': {
        'description': (
            'A flexible bearing block in which an unbalanced flexible rotor rotates.'
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            '{HEADER:part1}"{part1}"\n'
            'Two joints are used in the model:\n'
            '- The ground plate of the bearing block is fixed to ground ' + groundPos + ' m.\n'
            '- The shaft of the rotor is connected to the joint sleeve of the bearing block using'
            ' a revolute joint allowing rotation only around the y-axis.\n'
            '- Set the reference position of the rotor such that the shaft axis is moved along the z-axis to the bearing block\'s joint sleeve axis.'
            ' There is no other offset.\n'
            'A torque around the global y-axis having a scalar value of {torque} Nm acts on the rotor.'
            ' Gravity is neglected.'
        ),
        'parameters': {
            'part0': {'default': '{bearingBlockPart}', 'list': ['{bearingBlockPart}']}, # this is a template and it must be an existing part
            'part1': {'default': '{unbalancedRotorPart}', 'list': ['{unbalancedRotorPart}']}, # this is a template and it must be an existing part
            'torque': {'default': 20.0, 'range': [5.0, 25.0]},
            # master parameters that are deliberately not used in the fMBS description
            'bearingHeight': {'default': 0.2, 'range': [0.1, 0.3]}, # this must be the same in both parts
            'bearingRadius': {'default': 0.1, 'range': [0.05, 0.2]}, # this must be the same in both parts and is used to set reference pos in fMBS
            'wallThickness': {'default': 0.05, 'range': [0.05, 0.07]}, # this is used to set reference pos in fMBS
            'groundPlateInterfaceName': {'default': 'groundPlate', 'list': ['groundPlate']}, # this is used to address interfaces in fMBS
            'jointInterfaceName': {'default': 'joint', 'list': ['joint']}, # this is used to address interfaces in fMBS
            'shaftInterfaceName': {'default': 'shaft', 'list': ['shaft']}, # this is used to address interfaces in fMBS
        },
        'difficulty': 5,
        'sampleFileName': 'rotorInBearingBlock.py',
        'metaInformation':{'part0': {'masterParameters': ['bearingHeight', 'bearingRadius', 'wallThickness', 'groundPlateInterfaceName', 'jointInterfaceName']}, # we have to now the reference in the fMBS, and thus wallThickness is a mater parameter
                           'part1': {'masterParameters': ['bearingHeight', 'bearingRadius', 'shaftInterfaceName']}}
    },
    
    # 74) TM
    'rotorInBearingBlockWithPedestal': {
        'description': (
            'A flexible bearing block with pedestal in which an unbalanced flexible rotor rotates.'
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            '{HEADER:part1}"{part1}"\n'
            'Two joints are used in the model:\n'
            '- The ground plate of the bearing block is fixed to ground ' + groundPos + ' m (all coordinates constrained).\n'
            '- The shaft of the rotor is connected to the joint bore of the bearing block using'
            ' a revolute joint allowing rotation only around the y-axis.\n'
            '- Set the reference position of the rotor such that the shaft axis is moved along the z-axis to'
            ' the bearing block\'s joint bore axis at z = {plateHeight}/2+{bearingRadius}+{wallThickness} m.'
            ' There is no other offset.\n'
            'A torque around the global y-axis having a scalar value of {torque} Nm acts on the rotor shaft.'
            ' Gravity is neglected.'
        ),
        'parameters': {
            'part0': {'default': '{bearingBlockWithPedestalPart}', 'list': ['{bearingBlockWithPedestalPart}']},
            'part1': {'default': '{unbalancedRotorPart}', 'list': ['{unbalancedRotorPart}']},
            'torque': {'default': 20.0, 'range': [5.0, 25.0]},
            # master parameters that are deliberately not used in the fMBS description
            'bearingHeight': {'default': 0.1, 'range': [0.1, 0.3]}, # must match between parts
            'bearingRadius': {'default': 0.05, 'range': [0.05, 0.2]}, # must match between parts; used to set reference pos in fMBS
            'wallThickness': {'default': 0.07, 'range': [0.05, 0.07]}, # used to set reference pos in fMBS
            'plateHeight': {'default': 0.2, 'range': [0.15, 0.25]}, # used to set reference pos in fMBS
            'groundPlateInterfaceName': {'default': 'groundPlate', 'list': ['groundPlate']}, # used to address interfaces in fMBS
            'jointInterfaceName': {'default': 'joint', 'list': ['joint']}, # used to address interfaces in fMBS
            'shaftInterfaceName': {'default': 'shaft', 'list': ['shaft']}, # used to address interfaces in fMBS
        },
        'difficulty': 6,
        'sampleFileName': 'rotorInBearingBlockWithPedestal.py',
        'metaInformation': {
            'part0': {'masterParameters': ['bearingHeight', 'bearingRadius', 'wallThickness', 'plateHeight', 'groundPlateInterfaceName', 'jointInterfaceName']},
            'part1': {'masterParameters': ['bearingHeight', 'bearingRadius', 'shaftInterfaceName']}
        }
    },
    
    # 75) TM
    'rotorInTwoBearingBlocksWithPedestal': {
        'description': (
            'A balanced flexible rotor spinning between two identical flexible bearing blocks with pedestals.'
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            '{HEADER:part1}"{part1}"\n'
            'Two identical instances of the bearing block part (part0) are used, one on each side of the rotor.\n'
            'Four joints are used in the model:\n'
            '- The ground plates of both bearing blocks are fixed to ground ' + groundPos + ' m (all coordinates constrained).\n'
            '- The left shaft of the rotor is connected to the left bearing block\'s joint using'
            ' a revolute joint allowing rotation only around the x-axis.\n'
            '- The right shaft of the rotor is connected to the right bearing block\'s joint using'
            ' a revolute joint allowing rotation only around the x-axis.\n'
            '- Place the left bearing block such that its joint bore center aligns with the left rotor shaft center at'
            ' x = -({largeRotorHeight}/2+{smallRotorHeight}+{bearingHeight}/2) m,'
            ' z = -({plateHeight}/2+{bearingRadius}+{wallThickness}) m.\n'
            '- Place the right bearing block symmetrically at'
            ' x = +({largeRotorHeight}/2+{smallRotorHeight}+{bearingHeight}/2) m,'
            ' z = -({plateHeight}/2+{bearingRadius}+{wallThickness}) m.\n'
            'A torque around the global x-axis having a scalar value of {torque} Nm acts on the left rotor shaft.'
            ' Gravity is neglected.'
        ),
        'parameters': {
            'part0': {'default': '{bearingBlockWithPedestalPart}', 'list': ['{bearingBlockWithPedestalPart}']},
            'part1': {'default': '{balancedRotorPart}', 'list': ['{balancedRotorPart}']},
            'torque': {'default': 20, 'range': [5, 25]},
            # master parameters that are deliberately not used in the fMBS description
            'bearingHeight': {'default': 0.075, 'range': [0.075, 0.1]}, # must match between parts
            'bearingRadius': {'default': 0.05, 'range': [0.05, 0.1]}, # must match between parts
            'wallThickness': {'default': 0.07, 'range': [0.05, 0.07]}, # bearing block; used to set reference pos in fMBS
            'plateHeight': {'default': 0.2, 'range': [0.15, 0.25]}, # bearing block; used to set reference pos in fMBS
            'largeRotorHeight': {'default': 0.25, 'range': [0.2, 0.3]}, # rotor; used to set reference pos in fMBS
            'smallRotorHeight': {'default': 0.2, 'range': [0.15, 0.25]}, # rotor; used to set reference pos in fMBS
            'groundPlateInterfaceName': {'default': 'groundPlate', 'list': ['groundPlate']}, # used to address interfaces in fMBS
            'jointInterfaceName': {'default': 'joint', 'list': ['joint']}, # used to address interfaces in fMBS
            'shaftInterfaceNameLeft': {'default': 'shaftLeft', 'list': ['shaftLeft']}, # used to address interfaces in fMBS
            'shaftInterfaceNameRight': {'default': 'shaftRight', 'list': ['shaftRight']}, # used to address interfaces in fMBS
        },
        'difficulty': 7,
        'sampleFileName': 'rotorInTwoBearingBlocksWithPedestal.py',
        'metaInformation': {
            'part0': {'masterParameters': ['bearingHeight', 'bearingRadius', 'wallThickness', 'plateHeight', 'groundPlateInterfaceName', 'jointInterfaceName']},
            'part1': {'masterParameters': ['bearingHeight', 'bearingRadius', 'largeRotorHeight', 'smallRotorHeight', 'shaftInterfaceNameLeft', 'shaftInterfaceNameRight']}
        }
    },
    
    # 76) SW
    'flexiblePendulum': {
        'description': (
            'A pendulum consisting of a flexible arm.'
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            'Move the flexible arm along the x-axis such that the left side lies at the global origin ' + groundPos + ' m.'
            ' There is no other offset.'
            ' The left side arm is connected to ground ' + groundPos + ' m using'
            ' a revolute joint allowing rotation only around the y-axis, all other axes are constrained.'
            ' Gravity g = {gravity} m/s² acts in negative z-direction, and no further force acts.'
        ),
        'parameters': {
            'part0': {'default': '{flexibleBeamPart}', 'list': ['{flexibleBeamPart}',  '{flexibleBeamHollowPart}', '{flexibleBeamFilletsPart}', '{flexibleBeamDrilledPart}']}, # this is a template and it must be an existing part
            'gravity': {'default': 9.81, 'list': gravityList},
            # master parameters that are deliberately not used in the fMBS description
            'beamLength':{'default': 0.5, 'range': [0.5, 2.0]}, # this is used to set reference pos in fMBS
            'beamInterfaceNameLeft': {'default': 'leftSide', 'list': ['leftSide']}, # this is used to address interfaces in fMBS
        },
        'difficulty': 5,
        'sampleFileName': 'flexiblePendulum.py',
        'metaInformation':{'part0': {'masterParameters': ['beamLength', 'beamInterfaceNameLeft']}}
    },
    
    # 77) TM (with / without fillets / hollow / drilled hole)
    'cantileverBeam': {
        'description': (
            'A flexible beam is fixed to ground.'
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            'Move the flexible arm along the x-axis such that the left side lies at the global origin ' + groundPos + ' m.'
            ' There is no other offset.'
            ' The left side of the beam is fixed to ground (all coordinates constrained).'
            ' A force f = {force} N acts on the right side of the beam in negative y-direction.'
            ' Gravity is neglected.'
        ),
        'parameters': {
            'part0': {'default': '{flexibleBeamPart}', 'list': ['{flexibleBeamPart}',  '{flexibleBeamHollowPart}', '{flexibleBeamFilletsPart}', '{flexibleBeamDrilledPart}']}, # this is a template and it must be an existing part
            'force': {'default': 5, 'range': [2.5, 10]},
            # master parameters that are deliberately not used in the fMBS description
            'beamLength':{'default': 0.5, 'range': [0.5, 2.0]}, # this is used to set reference pos in fMBS
            'beamInterfaceNameLeft': {'default': 'leftSide', 'list': ['leftSide']}, # this is used to address interfaces in fMBS
            'beamInterfaceNameRight': {'default': 'rightSide', 'list': ['rightSide']}, # this is used to address interfaces in fMBS
        },
        'difficulty': 10,
        'sampleFileName': 'cantileverBeam.py',
        'metaInformation':{'part0': {'masterParameters': ['beamLength', 'beamInterfaceNameLeft', 'beamInterfaceNameRight']}}
    },
    
    # 78) TM
    'flexibleTableWithLoad': {
        'description': (
            'A flexible table on which plate a force acts in negative z-direction. '
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            'Move the table along the z-axis such that the bottom ends of the legs stand on the global xy-plane.'
            ' There is no other offset.'
            ' All four legs are fixed to ground (all coordinates constrained).'
            ' A force of {force} N acts on the mounting plate in negative z-direction.'
            ' Gravity is neglected.'
        ),
        'parameters': {
            'part0': {'default': '{flexibleTablePart}', 'list': ['{flexibleTablePart}']}, # this is a template and it must be an existing part
            'force': {'default': 1000, 'range': [500, 2000]},
            # master parameters that are deliberately not used in the fMBS description
            'legHeight': {'default': 0.75, 'range': [0.5, 1.0]}, # this is used to set reference pos in fMBS
            'plateHeight': {'default': 0.05, 'range': [0.025, 0.2]}, # this is used to set reference pos in fMBS
            'interfaceNameLegs': {'default': 'legs', 'list': ['legs']}, # this is used to address interfaces in fMBS
            'interfaceNameTopPlate': {'default': 'topPlate', 'list': ['topPlate']}, # this is used to address interfaces in fMBS
        },
        'difficulty': 5,
        'sampleFileName': 'flexibleTableWithLoad.py',
        'metaInformation':{'part0': {'masterParameters': ['legHeight', 'plateHeight', 'interfaceNameLegs', 'interfaceNameTopPlate']}}
    },

    # 79) TM, SW
    'windTurbine': {
        'description': (
            'A flexible wind turbine assembly consisting of a tower with nacelle and a rotor with three blades.'
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            '{HEADER:part1}"{part1}"\n'
            'Two joints are used in the model:\n'
            '- The tower base is fixed to ground ' + groundPos + ' m (all coordinates constrained).\n'
            '- A revolute joint about the global x-axis connects the rotor to the tower/nacelle.\n'
            '- Move the rotor along the global z-axis such that it\'s axis aligns with the axis through the nacelle.\n'
            'An aerodynamic torque of {torque} Nm is applied to the rotor about the x-axis.'
            ' Gravity is neglected.'
        ),
        'parameters': {
            'part0': {'default': '{windTurbineTowerPart}', 'list': ['{windTurbineTowerPart}']},
            'part1': {'default': '{windTurbineRotorPart}', 'list': ['{windTurbineRotorPart}']},
            'torque': {'default': 1e3, 'range': [1e3, 5e4]},
            # master parameters that are deliberately not used in the fMBS description
            'nacelleLength': {'default': 3.0, 'range': [1.0, 6.0]}, # this must be the same in both parts
            'nacelleWidth': {'default': 2.0, 'range': [1.5, 5]}, # this must be the same in both parts
            'mainBearingRadiusFactor': {'default': 0.2, 'range': [0.2, 0.4]}, # this must be the same in both parts
            'nacelleHeight': {'default': 2.0, 'range': [1.0, 5.0]}, # this must be the same in both parts
            'towerHeight': {'default': 20.0, 'range': [10.0, 30.0]}, # this is used to set reference pos in fMBS
            'groundInterfaceName': {'default': 'ground', 'list': ['ground']}, # this is used to address interfaces in fMBS
            'mainBearingInterfaceName': {'default': 'mainBearing', 'list': ['mainBearing']}, # this is used to address interfaces in fMBS
            'shaftInterfaceName': {'default': 'shaft', 'list': ['shaft']}, # this is used to address interfaces in fMBS
        },
        'difficulty': 6,
            'sampleFileName': 'windTurbine.py',
            'metaInformation': {'part0': {'masterParameters': ['nacelleLength', 'nacelleWidth', 'nacelleHeight', 'mainBearingRadiusFactor', 'towerHeight' , 'groundInterfaceName', 'mainBearingInterfaceName']},
                                'part1': {'masterParameters': ['nacelleLength', 'nacelleWidth', 'nacelleHeight', 'mainBearingRadiusFactor', 'shaftInterfaceName']}
        }
    },
    
    # 80) TM
    'grindingBench': {
        'description': (
            'A grinding bench assembly consisting of a housing with a through spindle passage and the spindle.'
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            '{HEADER:part1}"{part1}"\n'
            'Two joints are used in the model:\n'
            '- The housing bottom interface is fixed to ground ' + groundPos + ' m (all coordinates constrained).\n'
            '- Move the housing along the positive z-axis, such that it stands on the global xy-plane.\n'
            '- A revolute joint about the global x-axis connects the spindle with the housing spindle passage.\n'
            '- Move the spindle along the positive z-axis such that it\'s axis aligns with the through spindle passage of the housing.\n'
            'A driving torque of {torque} Nm is applied to the spindle about the x-axis.'
            ' Gravity is neglected.'
        ),
        'parameters': {
            'part0': {'default': '{grindingBenchHousingPart}', 'list': ['{grindingBenchHousingPart}']},
            'part1': {'default': '{grindingBenchSpindleWithDisksPart}', 'list': ['{grindingBenchSpindleWithDisksPart}']},
            'torque': {'default': 100, 'range': [100, 500]},
            # master parameters that are deliberately not used in the fMBS description
            'spindleRadiusFactor': {'default': 0.2, 'range': [0.2, 0.3]}, # this must be the same in both parts
            'housingLength': {'default': 1.5, 'range': [0.6, 2.0]}, # this must be the same in both parts
            'housingWidth': {'default': 1.0, 'range': [0.5, 2.0]}, # this must be the same in both parts
            'housingHeight': {'default': 1.5, 'range': [0.5, 3.0]}, # this must be the same in both parts and is used to set reference pos in fMBS
            'housingBottomInterfaceName': {'default': 'housingBottom', 'list': ['housingBottom']}, # this is used to address interfaces in fMBS
            'spindlePassageInterfaceName': {'default': 'spindlePassage', 'list': ['spindlePassage']}, # this is used to address interfaces in fMBS
            'spindleInterfaceName': {'default': 'spindle', 'list': ['spindle']}, # this is used to address interfaces in fMBS
        },
        'difficulty': 6,
        'sampleFileName': 'grindingBench.py',
        'metaInformation': {'part0': {'masterParameters': ['spindleRadiusFactor', 'housingLength', 'housingHeight', 'housingWidth', 'housingBottomInterfaceName', 'spindlePassageInterfaceName']},
                            'part1': {'masterParameters': ['spindleRadiusFactor', 'housingLength', 'housingHeight', 'housingWidth','spindleInterfaceName']}
        }
    },
    
    # 81) TM
    'doorHinge': {
        'description': (
            'A flexible door hinge assembly consisting of two hinge leaves connected by cylindrical hinge knuckles.'
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            '{HEADER:part1}"{part1}"\n'
            'The leaf hinge with the pin is fixed to ground.'
            ' The hinge motion around the global X-axis is realized by two revolute joints'
            ' between the cylindrical bore holes of the first leaf' 
            ' and the corresponding cylindrical pins of the second leaf.'
            ' The hinge leaf with the pins is rotated by 180 degrees about the z-axis relative to the first leaf' 
            ' and translated along the y-axis so that the hinge knuckles interlock.' # or simpler: by {leafWidth} m {knuckleRadius} m.
            ' Gravity of {gravity} m/s² acts on both hinge leafs in negative y-direction.' # gravity positive due to rotation constraint of hinge (can only rotate in one direction)
        ),
        'parameters': {
            'part0': {'default': '{hingeLeafPinSidePart}', 'list': ['{hingeLeafPinSidePart}']},
            'part1': {'default': '{hingeLeafGapSidePart}', 'list': ['{hingeLeafGapSidePart}']},
            'leafLength': {'default': 0.08, 'range': [0.05, 0.08]},
            'leafWidth': {'default': 0.08, 'range': [0.04, 0.1]},
            'leafHeight': {'default': 0.008, 'range': [0.008, 0.01]},
            'knuckleRadius': {'default': 0.004, 'range': [0.004, 0.01]},
            'gapLength': {'default': 0.0375, 'range': [0.0125, 0.0375]},
            'pinRadius': {'default': 0.001, 'range': [0.001, 0.002]},
            # part 0 interfaces (movable part)
            'pinInterfaceNamePosX': {'default': 'hingePinPosX', 'list': ['hingePinPosX']},
            'pinInterfaceNameNegX': {'default': 'hingePinNegX', 'list': ['hingePinNegX']},
            # part 1 interfaces (fixed part)
            'leafInterfaceName': {'default': 'leafMount', 'list': ['leafMount']}, 
            'boreInterfaceNamePosX': {'default': 'hingeBorePosX', 'list': ['hingeBorePosX']},
            'boreInterfaceNameNegX': {'default': 'hingeBoreNegX', 'list': ['hingeBoreNegX']},
            'gravity': {'default': 9.81, 'list': gravityList}
        },
        'difficulty': 7,
        'sampleFileName': 'doorHinge.py',
        'metaInformation': {
            'part0': {'masterParameters': ['leafLength', 'leafWidth', 'leafHeight', 'gapLength', 'knuckleRadius', 'pinRadius', 'pinInterfaceNamePosX', 'pinInterfaceNameNegX']},
            'part1': {'masterParameters': ['leafLength', 'leafWidth', 'leafHeight', 'gapLength', 'knuckleRadius', 'pinRadius', 'leafInterfaceName', 'boreInterfaceNamePosX', 'boreInterfaceNameNegX']},
        }
    },
    
    # 82) TM
    'mountingBracketFreeFalling': {
        'description': (
            'A mounting bracket assembly consisting of a base plate with corner bores and side holders.'
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            'The bracket is not attached to ground and falls freely under gravity of {gravity} m/s^2 acting in negative z-direction.'
            ' No joints or constraints are used in the model.'
        ),
        'parameters': {
            'part0': {'default': '{mountingBracketPart}', 'list': ['{mountingBracketPart}']},
            'gravity': {'default': 9.81, 'list': gravityList},
            # interface names of the mounting bracket
            'interfaceNamePlateBores': {'default': 'plateBores', 'list': ['plateBores']},
            'interfaceNameHolderLeftBore': {'default': 'holderLeftBore', 'list': ['holderLeftBore']},
            'interfaceNameHolderRightBore': {'default': 'holderRightBore', 'list': ['holderRightBore']},
        },      
        'difficulty': 6,
        'sampleFileName': 'mountingBracketFreeFalling.py',
        'metaInformation': {'part0': {'masterParameters': ['interfaceNamePlateBores', 'interfaceNameHolderLeftBore', 'interfaceNameHolderRightBore']}}
    },

    # 83) TM
    'flexibleSliderCrankSimple': {
        'description': (
            'A flexible slider-crank assembly composed of a crank disk with offset pin, a connecting rod with two bores, and a slider block with a vertical pin.'
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            '{HEADER:part1}"{part1}"\n'
            '{HEADER:part2}"{part2}"\n'
            'Joints used in the model:\n'
            '- The disk bottom interface ("{diskBottomInterfaceName}") is constrained to ground with a revolute joint about the global z-axis.\n'
            '- A revolute joint about the global z-axis connects the crank pin interface ("{crankPinInterfaceName}") to the rod crank-end interface ("{rodEndCrankInterfaceName}").\n'
            '- A revolute joint about the global z-axis connects the rod slider-end interface ("{rodEndSliderInterfaceName}") to the slider pin interface ("{sliderPinInterfaceName}").\n'
            '- A prismatic constraint along the global x-axis guides the slider guide interface ("{sliderGuideInterfaceName}").\n'
            'Reference positions:\n'
            '- The crank reference frame is at the global origin.\n'
            '- The rod reference position is [{boreDistance}/2+{crankRadius}, 0, ({diskThickness}+{pinLength})/2] m,'
            ' aligning the rod crank-end bore center (at rod-local (-{boreDistance}/2, 0, 0)) with the crank pin interface center.\n'
            '- The slider reference position is [{boreDistance}+{crankRadius}, 0, 0] m,'
            ' aligning the slider pin centroid (at slider-local (0, 0, ({diskThickness}+{pinLength})/2)) with the rod slider-end bore center.\n'
            'A torque of {torque} Nm about the global z-axis acts on the crank disk bottom interface.'
            ' Gravity is neglected.'
        ),
        'parameters': {
            'part0': {'default': '{crankPartSimple}', 'list': ['{crankPartSimple}']},
            'part1': {'default': '{rodPartSimple}', 'list': ['{rodPartSimple}']},
            'part2': {'default': '{sliderPartSimple}', 'list': ['{sliderPartSimple}']},
            'torque': {'default': 0.05, 'range': [0.01, 0.5]},
            # master parameters shared across parts (used for assembly geometry)
            'diskThickness': {'default': 0.0125, 'range': [0.0125, 0.025]},  # shared: crank + slider; used for rod ref z
            'crankRadius': {'default': 0.05, 'range': [0.05, 0.15]},         # shared: crank; used for rod + slider ref x
            'pinRadius': {'default': 0.0125, 'range': [0.0125, 0.025]},      # shared: crank + rod + slider
            'pinLength': {'default': 0.025, 'range': [0.025, 0.075]},        # shared: crank + rod + slider; used for rod ref z
            'boreDistance': {'default': 0.125, 'range': [0.125, 0.25]},      # shared: rod; used for rod + slider ref x
            'sliderLength': {'default': 0.1, 'range': [0.05, 0.2]},          # shared: slider geometry
            'sliderWidth': {'default': 0.05, 'range': [0.025, 0.1]},         # shared: slider geometry
            # crank interface names
            'diskBottomInterfaceName': {'default': 'diskBottom', 'list': ['diskBottom']},
            'crankPinInterfaceName': {'default': 'crankPin', 'list': ['crankPin']},
            # rod interface names
            'rodEndCrankInterfaceName': {'default': 'rodEndCrank', 'list': ['rodEndCrank']},
            'rodEndSliderInterfaceName': {'default': 'rodEndSlider', 'list': ['rodEndSlider']},
            # slider interface names
            'sliderGuideInterfaceName': {'default': 'sliderBottom', 'list': ['sliderBottom']},
            'sliderPinInterfaceName': {'default': 'sliderPin', 'list': ['sliderPin']},
        },
        'difficulty': 6,
        'sampleFileName': 'flexibleSliderCrankSimple.py',
        'metaInformation': {
            'part0': {'masterParameters': ['diskThickness', 'crankRadius', 'pinRadius', 'pinLength', 'diskBottomInterfaceName', 'crankPinInterfaceName']},
            'part1': {'masterParameters': ['pinRadius', 'pinLength', 'boreDistance', 'rodEndCrankInterfaceName', 'rodEndSliderInterfaceName']},
            'part2': {'masterParameters': ['pinRadius', 'pinLength', 'diskThickness', 'sliderLength', 'sliderWidth', 'sliderGuideInterfaceName', 'sliderPinInterfaceName']}
        }
    },
    
    # 84) TM
    'flexibleSliderCrankAdvanced': {
        'description': (
            'A flexible slider-crank assembly composed of three parts:'
            ' a crank with an oblong cross-section and a protruding main-bearing pin,'
            ' an asymmetric connecting rod (conrod shape) with a big-end pin and a small-end bore,'
            ' and a slider with a guide rabbet and a vertical pin on top.'
            '{HEADER:INTRO}'
            '{HEADER:part0}"{part0}"\n'
            '{HEADER:part1}"{part1}"\n'
            '{HEADER:part2}"{part2}"\n'
            'The crank boss radius is crankBossRadius = min(0.5' + multChar + '{crankWidth}, 2' + multChar + '{connectionWidth}, 0.24' + multChar + '{crankLength}).\n'
            'The crank pin (boreRight) is located at x = 0.5' + multChar + '{crankLength} - crankBossRadius in the crank local frame.\n'
            'Joints used in the model:\n'
            '- The crank main-bearing pin interface ("{pinLeftInterfaceName}") is constrained to ground with a revolute about the global z-axis.\n'
            '- A revolute joint about the global z-axis connects the crank pin bore interface ("{boreRightInterfaceName}") to the rod big-end pin interface ("{rodEndCrankInterfaceName}").\n'
            '- A revolute joint about the global z-axis connects the rod small-end bore interface ("{rodEndSliderInterfaceName}") to the slider pin interface ("{sliderPinInterfaceName}").\n'
            '- A prismatic constraint along the global x-axis guides the slider rabbet interface ("{sliderGuideInterfaceName}").\n'
            'Reference positions:\n'
            '- The crank reference frame is at the global origin.\n'
            '- The rod reference position is [0.5' + multChar + '{crankLength} - crankBossRadius + 0.5' + multChar + '{boreDistance}, 0, {pinLength}] m,'
            ' aligning the rod big-end pin centroid (at rod-local (-0.5' + multChar + '{boreDistance}, 0, -{pinLength})) with the crank pin bore center.\n'
            '- The slider reference position is [0.5' + multChar + '{crankLength} - crankBossRadius + {boreDistance}, 0, 0.5' + multChar + '{pinLength} - 0.5' + multChar + '{diskThickness}] m,'
            ' aligning the slider pin centroid (at slider-local (0, 0, 0.5' + multChar + '{diskThickness} + 0.5' + multChar + '{pinLength})) with the rod small-end bore center.\n'
            'A torque of {torque} Nm about the global z-axis acts on the crank main-bearing pin interface.'
            ' Gravity is neglected.'
        ),
        'parameters': {
            'part0': {'default': '{crankPartAdvanced}', 'list': ['{crankPartAdvanced}']},
            'part1': {'default': '{rodPartAdvanced}', 'list': ['{rodPartAdvanced}']},
            'part2': {'default': '{sliderPartAdvanced}', 'list': ['{sliderPartAdvanced}']},
            'torque': {'default': 0.5, 'range': [0.1, 1.0]},
            # master parameters: passed to parts (must match part parameter names)
            'crankLength': {'default': 0.1, 'range': [0.08, 0.175]},  # always < boreDistance min
            'crankWidth': {'default': 0.1, 'range': [0.1, 0.2]},
            'connectionWidth': {'default': 0.02, 'range': [0.02, 0.05]},
            'pinRadius': {'default': 0.0125, 'range': [0.0125, 0.015]},  # shared: crank + rod + slider
            'pinLength': {'default': 0.025, 'range': [0.025, 0.05]},
            'boreDistance': {'default': 0.25, 'range': [0.2, 0.35]},  # always > crankLength max
            'diskThickness': {'default': 0.0125, 'range': [0.0125, 0.025]},
            'sliderLength': {'default': 0.1, 'range': [0.05, 0.2]},   # shared: slider geometry
            'sliderWidth': {'default': 0.1, 'range': [0.1, 0.2]},  # shared: slider geometry
            # crank interface names
            'pinLeftInterfaceName': {'default': 'pinLeft', 'list': ['pinLeft']},
            'boreRightInterfaceName': {'default': 'boreRight', 'list': ['boreRight']},
            # rod interface names
            'rodEndCrankInterfaceName': {'default': 'rodEndCrank', 'list': ['rodEndCrank']},
            'rodEndSliderInterfaceName': {'default': 'rodEndSlider', 'list': ['rodEndSlider']},
            # slider interface names
            'sliderPinInterfaceName': {'default': 'sliderPin', 'list': ['sliderPin']},
            'sliderGuideInterfaceName': {'default': 'sliderBottom', 'list': ['sliderBottom']},
        },
        'difficulty': 7,
        'sampleFileName': 'flexibleSliderCrankAdvanced.py',
        'metaInformation': {
            'part0': {'masterParameters': ['crankLength', 'crankWidth', 'connectionWidth', 'pinRadius', 'pinLeftInterfaceName', 'boreRightInterfaceName']},
            'part1': {'masterParameters': ['pinRadius', 'pinLength', 'boreDistance', 'rodEndCrankInterfaceName', 'rodEndSliderInterfaceName']},
            'part2': {'masterParameters': ['pinRadius', 'pinLength', 'diskThickness', 'sliderLength', 'sliderWidth', 'sliderPinInterfaceName', 'sliderGuideInterfaceName']}
        }
    },
    
    } 