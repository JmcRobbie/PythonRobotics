'''
Triple pendulum control simulation

Author: Jack McRobbie

Reference: T. Glück, A. Eder, and A. Kugi, “Swing-up control of a triple pendulum on a cart with experimental validation”,
Automatica, vol. 49, no. 3, pp. 801–808, 2013. doi: 10.1016/j.automatica.2012.12.006
'''
import numpy as np 
import math
from sympy import *

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

def rk4(dt, x,  f):
    '''
    Implements the RK4 integrator for a specified dynamics model as input
    ''' 
    k1 = f(x)
    k2 = f(x + k1*0.5)
    k3 = f(x + k2*0.5)
    k4 = f(x + k3*0.5)  
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

def triplePendulumDynamics(x,f = 0.0):
    ''' 
    Implements the dynamics of the triple pendulum system.

    Returns: The instantaneous derivitive of state.
    
    X = [lateral position, angle1,angle2,angle3]
    state = [X ; X_dot]
    ''' 
    
    g = 9.82
    l0 = 1
    l1 = 1
    l2 = 1
    m0 = 1
    m1 = 1
    m2 = 1
    m3 = 1

    s = Symbol('s')
    y = Symbol('y')
    w = Symbol('w')
    z = Symbol('z')
    
    eqs = ([
    f + l0*m1*x[5]**2*math.cos(x[1]) + l0*m2*x[5]**2*math.cos(x[1]) + l0*m3*x[5]**2*math.cos(x[1]) + l1*m2*x[6]**2*math.cos(x[2]) + l1*m3*x[6]**2*math.cos(x[2]) + l2*m3*x[7]**2*math.cos(x[3]) + l2*m3*math.sin(x[3])*z - (-l1*m2*math.sin(x[2]) - l1*m3*math.sin(x[2]))*w - (-l0*m1*math.sin(x[1]) - l0*m2*math.sin(x[1]) - l0*m3*math.sin(x[1]))*y - (m0 + m1 + m2 + m3)*s, 
    -g*l0*m1*math.cos(x[1]) - g*l0*m2*math.cos(x[1]) - g*l0*m3*math.cos(x[1]) +  l0*l1*m2*(-math.sin(x[1])*math.cos(x[2]) + math.sin(x[2])*math.cos(x[1]))*x[6]**2 + l0*l1*m3*(-math.sin(x[1])*math.cos(x[2]) + math.sin(x[2])*math.cos(x[1]))*x[6]**2 - l0*l2*m3*(math.sin(x[1])*math.sin(x[3]) + math.cos(x[1])*math.cos(x[3]))*z + l0*l2*m3*(-math.sin(x[1])*math.cos(x[3]) + math.sin(x[3])*math.cos(x[1]))*x[7]**2 - (l0*l1*m2*(math.sin(x[1])*math.sin(x[2]) + math.cos(x[1])*math.cos(x[2])) + l0*l1*m3*(math.sin(x[1])*math.sin(x[2]) + math.cos(x[1])*math.cos(x[2])))*w - (l0**2*m1 + l0**2*m2 + l0**2*m3)*y - (-l0*m1*math.sin(x[1]) - l0*m2*math.sin(x[1]) - l0*m3*math.sin(x[1]))*s,
    -g*l1*m2*math.cos(x[2]) - g*l1*m3*math.cos(x[2]) +     l0*l1*m2*(math.sin(x[1])*math.cos(x[2]) - math.sin(x[2])*math.cos(x[1]))*x[5]**2 + l0*l1*m3*(math.sin(x[1])*math.cos(x[2]) - math.sin(x[2])*math.cos(x[1]))*x[5]**2 - l1*l2*m3*(math.sin(x[2])*math.sin(x[3]) + math.cos(x[2])*math.cos(x[3]))*z + l1*l2*m3*(-math.sin(x[2])*math.cos(x[3]) + math.sin(x[3])*math.cos(x[2]))*x[7]**2 - (l1**2*m2 + l1**2*m3)*w - (-l1*m2*math.sin(x[2]) - l1*m3*math.sin(x[2]))*s - (l0*l1*m2*(math.sin(x[1])*math.sin(x[2]) + math.cos(x[1])*math.cos(x[2])) + l0*l1*m3*(math.sin(x[1])*math.sin(x[2]) + math.cos(x[1])*math.cos(x[2])))*y,
    -g*l2*m3*math.cos(x[3]) -l0*l2*m3*(math.sin(x[1])*math.sin(x[3]) + math.cos(x[1])*math.cos(x[3]))*y + l0*l2*m3*(math.sin(x[1])*math.cos(x[3]) - math.sin(x[3])*math.cos(x[1]))*x[5]**2 - l1*l2*m3*(math.sin(x[2])*math.sin(x[3]) + math.cos(x[2])*math.cos(x[3]))*w + l1*l2*m3*(math.sin(x[2])*math.cos(x[3]) - math.sin(x[3])*math.cos(x[2]))*x[6]**2 - l2**2*m3*z + l2*m3*math.sin(x[3])*s
    ])
    return linsolve(eqs,[s,y,w,z])

def integrateState(ddx, dt, x):

    x[0] = x[0] + dt * x[4]
    x[1] = x[1] + dt * x[5]
    x[2] = x[2] + dt * x[6]
    x[3] = x[3] + dt * x[7]

    x[4] = x[4] + dt * ddx[0]
    x[5] = x[5] + dt * ddx[1]
    x[6] = x[6] + dt * ddx[2]
    x[7] = x[7] + dt * ddx[3]
    return x

def plotDynamics(x):
    cart_w = 1.0
    cart_h = 0.5
    radius = 0.1

    cx = np.matrix([-cart_w / 2.0, cart_w / 2.0, cart_w /
                    2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.matrix([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.matrix([0.0, l_bar * math.sin(-theta)])
    bx += xt
    by = np.matrix([cart_h, l_bar * math.cos(-theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
    ox = [radius * math.cos(a) for a in angles]
    oy = [radius * math.sin(a) for a in angles]

    rwx = np.copy(ox) + cart_w / 4.0 + xt
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + float(bx[0, -1])
    wy = np.copy(oy) + float(by[0, -1])

    plt.plot(flatten(cx), flatten(cy), "-b")
    plt.plot(flatten(bx), flatten(by), "-k")
    plt.plot(flatten(rwx), flatten(rwy), "-k")
    plt.plot(flatten(lwx), flatten(lwy), "-k")
    plt.plot(flatten(wx), flatten(wy), "-k")
    plt.title("x:" + str(round(xt, 2)) + ",theta:" +
              str(round(math.degrees(theta), 2)))

    plt.axis("equal")

def cart_positions(x):
    '''
    returns cartesian position of the triple pendulum elements for plotting purposes.

    '''
    l0 = 1
    l1 = 1
    l2 = 1
    points = np.zeros([4,2])
    return points

def runSimulation(plotting = false): 
    simtime = 10.0 
    dt = 0.01
    t = 0
    stateHx = []
    stateHddx = []
    x = np.zeros(8)
    while (t<simtime):
        t = t + dt
        stateHx.append(x)
        result = triplePendulumDynamics(x)
        ddx = result.args[0]
        stateHddx.append(ddx)
        x = integrateState(ddx,dt,x)
        print(t)
        if plotting:      
            plotDynamics(x)
runSimulation(True)
