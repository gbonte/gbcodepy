# "INFOF422 Statistical foundations of machine learning" course
# Python package gbcodepy
# Author: G. Bontempi
## naive.py

import numpy as np

def f(x):
    return np.sin(x)

N = 100
S = 1000

G1 = []
G2 = []

xbar = np.pi / 3
sdw = 0.5

for _ in range(S):
    X = np.random.normal(loc=0, scale=1, size=N)
    Y = f(X) + np.random.normal(loc=0, scale=sdw, size=N)
    
    # Note: In the R code, Yts is generated with length N even though it is used as a test set.
    Yts = f(xbar) + np.random.normal(loc=0, scale=sdw, size=N)
    
    Yhat1 = 0
    Yhat2 = np.mean(Y)
    
    e1 = np.mean((Yts - Yhat1) ** 2)
    e2 = np.mean((Yts - Yhat2) ** 2)
    
    G1.append(e1)
    G2.append(e2)

# Generate new data samples. Notice that in the R code,
# X is generated with length 100*N, but Y is generated with length N.
X = np.random.normal(loc=0, scale=1, size=100 * N)
# If f is vectorized, f(X) is computed for all elements in X.
# For adding noise, we mimic the R code by generating only N noise values.
# This likely leads to a mismatch in array sizes; here we slice f(X) to match the noise length.
Y = f(X[:N]) + np.random.normal(loc=0, scale=sdw, size=N)

# Compute theoretical values and Monte Carlo estimates.
# In R, var(Y) computes sample variance (ddof=1 in Python for sample variance).
G1_th = sdw**2 + (f(xbar)**2)
G2_th = sdw**2 + np.var(Y, ddof=1) / N + (f(xbar) - np.mean(Y))**2

print("G1 th =", G1_th, "; MC =", np.mean(G1))
print("G2 th =", G2_th, "; MC =", np.mean(G2))
