# "Statistical foundations of machine learning" software
# Python equivalent of gbcode
# Author: G. Bontempi
# mcarlo4.py
# Monte Carlo estimation of correlation

import numpy as np

np.random.seed(0)
R = 10000  # number of MC trials

XY = []
X = []
Y = []

for r in range(R):
    x = np.random.normal(0, 1)
    y = 2 * x**2
    
    XY.append(x * y)
    X.append(x)
    Y.append(y)

# Convert lists to numpy arrays for efficient computation
X = np.array(X)
Y = np.array(Y)
XY = np.array(XY)

muX = np.mean(X)
sdX = np.std(X, ddof=1)  # ddof=1 for sample standard deviation
sdY = np.std(Y, ddof=1)

print(f"rho[xy]= {np.mean(XY)/(sdX*sdY)}")
