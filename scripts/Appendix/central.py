#!/usr/bin/env python3
# "Statistical foundations of machine learning" software
# package gbcodepy
# Author: G. Bontempi

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# central.py
# Script: visualizes the distribution of the estimator
# of the variance of a non gaussian random variable
# shows the central limit theorem

# Equivalent of rm(list=ls()) - in Python, we typically do not remove all globals in a script
# Equivalent of graphics.off() - close all open figure windows
plt.close('all')

# Equivalent of par(ask=TRUE) - we will pause for user input between figures
def pause():
    input("Press Enter to continue...")

N = 1
R = 1000

I = np.arange(-50, 50.5, 0.5)

Mn = -10
Mx = 10
var_th = (1/(Mx - Mn)) * ((Mx**3)/3 - (Mn**3)/3)
# Compute the density of a uniform distribution: 1/(Mx-Mn) if I is between Mn and Mx, else 0
p = np.where((I >= Mn) & (I <= Mx), 1/(Mx - Mn), 0)
plt.figure()
plt.plot(I, p, linestyle='-', marker='', color='b')
plt.title("Distribution of  r.v. z: var=" + str(round(var_th, 1)))
plt.xlabel("I")
plt.ylabel("Density")
plt.show()
pause()

aver = np.zeros(R)

for N in range(2, 1001):
    for i in range(1, N+1):
        aver = aver + np.random.uniform(Mn, Mx, R)
    aver = aver / N
    plt.figure()
    plt.hist(aver, bins=30, density=True, color='lightgray', edgecolor='black')
    plt.xlim(Mn, Mx)
    plt.title("Average of N=" + str(N) + " r.v.s")
    # Create I2: from -5*std(aver) to 5*std(aver) with step .5
    I2 = np.arange(-5 * np.std(aver), 5 * np.std(aver) + 0.5, 0.5)
    p_var_hat = norm.pdf(I2, loc=0, scale=np.sqrt(var_th / N))
    plt.plot(I2, p_var_hat, linestyle='-', marker='', color='red')
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()
    pause()

# End of script
