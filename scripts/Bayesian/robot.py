import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.stats import norm

## "Statistical foundations of machine learning" software
## R package gbcode 
## Author: G. Bontempi


np.random.seed(0)  # set seed

Wx = 50  ## width arena
Wy = 50

x=10
y=33

## initialization probability mass
P = np.full((Wx, Wy), 1/(Wx*Wy))

S = np.zeros(4)  # Sensors: east, south,west, north
x = Wx / 2
y = Wy / 2

zero = 1e-10
sdS = 3
sdX = 1


plt.pause(0.1)
    
while True:
    # control action
    ux = np.random.choice([-2, -1, 0, 1, 2])
    uy = np.random.choice([-2, -1, 0, 1, 2])
    
    # update position robot
    x = max(0, min(x + ux, Wx-1))
    y = max(0, min(y + uy, Wy-1))
    
    ## Bayes state update after control action
    # Looping over each cell with 1-indexed coordinates
    P_new = np.zeros((Wx, Wy))
    for i in range(Wx):
        for j in range(Wy):
            prev_i = int(round(max(0, min(i - ux, Wx-1))))
            prev_j = int(round(max(0, min(j - uy, Wy-1))))
            # Adjust indices for 0-indexed Python arrays
            P_new[i , j ] = P[prev_i , prev_j ]
    P = P_new / np.sum(P_new)
    
    # robot slippery
    x = max(0, int(min(x + np.random.normal(0, sdX), Wx-1)))
    y = max(0, int(min(y + np.random.normal(0, sdX), Wy-1)))
    
    ### sensor data collection
    sensor_values = np.array([Wx - x, y , x, Wy - y])
    ## sensors=[right,down,left,up]
    noise = np.random.normal(0, sdS, 4)
    S = sensor_values + noise
    S = np.minimum(S, Wx-1)
    S = np.maximum(S, 0)
    
    ## Bayes state update after sensing 
    P_new = np.zeros((Wx, Wy))
    for i in range( Wx ):
        for j in range(Wy ):
            Pt = np.log(P[i, j ])
            Lik = np.log(norm.pdf(S[0], loc=Wx - i, scale=sdS)) +np.log( norm.pdf(S[1], loc=j , scale=sdS)) 
            + np.log( norm.pdf(S[2], loc=i , scale=sdS)) +np.log( norm.pdf(S[3], loc=Wy - j, scale=sdS))
            P_new[i , j ] = max(zero, math.exp(Pt + Lik))
    P = P_new / np.sum(P_new)
    
    ## visualization of robot position vs state density estimation
    plt.clf()
    plt.imshow(P.T, extent=[0, Wx-1, 0, Wy-1], origin='lower', cmap='gray')
    plt.title("Bayes robot state estimation")
    plt.xlabel("")
    plt.ylabel("")
    plt.plot(x, y, 'ro', markersize=5)
    plt.pause(0.1)
    
# Note: This code does not use a main function, as per the additional instructions.
