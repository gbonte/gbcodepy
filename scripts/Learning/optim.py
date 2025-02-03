# "INFOF422 Statistical foundations of machine learning" course
# Python package gbcodepy
# Author: G. Bontempi
# optim2.py


import numpy as np
import matplotlib.pyplot as plt
import time


def J(a):
    return a**2 - 2*a + 3

def Jprime(a):
    return 2*a - 2

# Create a sequence of alpha from -2 to 4 with step 0.1
alpha = np.arange(-2, 4.1, 0.1)

# Plot the function J over the range of alpha


# Initialization of parameters
a = -1
mu = 0.1

# Iterative updating and plotting of the red point on the curve
for r in range(100):
    a = a - mu * Jprime(a)
    plt.figure()
    plt.plot(alpha, J(alpha), linestyle='-', label='J(alpha)')
    plt.xlabel('alpha')
    plt.ylabel('J(alpha)')
    # Plot the point with a red dot; 'markersize' is the approximate equivalent of lwd in R for point size
    plt.plot(a, J(a), 'ro', markersize=8)
    plt.pause(0.1)  # Pause for 1 second to mimic Sys.sleep(1)

plt.show()
