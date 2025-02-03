# "INFOF422 Statistical foundations of machine learning" course
# Python package gbcodepy
# Author: G. Bontempi
# optim2.py


import numpy as np
import matplotlib.pyplot as plt
import time

# Define the function J equivalent to the R version
def J(a):
    # Computes a^4/4 - a^3/3 - a^2 + 2
    return a**4 / 4 - a**3 / 3 - a**2 + 2

# Define the derivative Jprime equivalent to the R version
def Jprime(a):
    # Computes a^3 - a^2 - 2*a
    return a**3 - a**2 - 2 * a

# Create an array of alpha values from -2 to 4 with step 0.1
alpha = np.arange(-2, 4 + 0.1, 0.1)



# Initialize a (alpha0 initialization) and plot the initial point in red
a = 3

# Learning rate (mu)
mu = 0.01

# Gradient descent loop: update a and plot the new point on the curve
for r in range(100):
    a = a - mu * Jprime(a)
    plt.figure(figsize=(8, 6))
    plt.plot(alpha, J(alpha), label='J(alpha)', color='blue')
    plt.xlabel('alpha')
    plt.ylabel('J(alpha)')
    plt.plot(a, J(a), 'ro', markersize=8)
    plt.draw()  # Update the drawing after each new point
    plt.pause(0.01) 

# Keep the plot open after the loop ends

plt.show()
