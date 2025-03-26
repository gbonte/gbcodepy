#!/usr/bin/env python3
# "INFOF422 Statistical foundations of machine learning" course
# R package gbcode 
# Author: G. Bontempi

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# min_x 1/2 x^T D x-d^Tx subject to A^Tx>=b

N = 25
n = 20
# number of inoput features

# Create a sequence from 7 to 0.01 with step -0.05 (values >= 0.01)
LAM = np.arange(7, 0.0, -0.05)
Remp = []
X = np.random.randn(N * n).reshape((N, n))
Y = 3 * X[:, 0] + X[:, 1] - 0.5*X[:, n - 1] + np.random.randn(N) * 0.5

def lassols(X, Y, lambda_val=0):
    eps = 1e-10
    Y = Y - np.mean(Y)
    n = X.shape[1]
    Dmat = np.dot(X.T, X)
    # Construct DDmat by row binding Dmat and -Dmat, then column binding with - (that result)
    temp = np.vstack((Dmat, -Dmat))
    DDmat = np.hstack((temp, -temp)) + eps * np.eye(2 * n)
    dvec = np.dot(X.T, Y)
    ddvec = np.concatenate((dvec, -dvec))
    # Create AAmat: first row of -1's and then identity matrix of size 2*n
    AAmat = np.vstack((-np.ones((1, 2 * n)), np.eye(2 * n)))
    bbvec = np.concatenate((np.array([-lambda_val]), np.zeros(2 * n)))
    
    # Convert numpy arrays to cvxopt matrices.
    # In cvxopt, the quadratic programming problem is formulated as:
    #   min 1/2 x^T P x + q^T x  subject to Gx <= h
    # Our problem: min 1/2 x^T DDmat x - ddvec^T x subject to AAmat x >= bbvec
    # is converted by setting:
    #   P = DDmat, q = -ddvec, G = -AAmat, h = -bbvec.
    P = matrix(DDmat)
    q = matrix(-ddvec)
    G = matrix(-AAmat)
    h = matrix(-bbvec)
    sol = solvers.qp(P, q, G, h)
    solx = np.array(sol['x']).flatten()
    betahat = solx[0:n] - solx[n:2 * n]
    return betahat

def lassopred(Xtr, Ytr, Xts, lambda_val):
    # Standardize Xtr: subtract column means and divide by column standard deviations
    center = np.mean(Xtr, axis=0)
    scale = np.std(Xtr, axis=0, ddof=1)
    sXtr = (Xtr - center) / scale
    # Standardize Xts using the training center and scale
    sXts = (Xts - center) / scale
    mYtr = np.mean(Ytr)
    
    betahat = lassols(sXtr, Ytr - mYtr, lambda_val)
    
    Yhats = mYtr + np.dot(sXts, betahat)
    Yhatr = mYtr + np.dot(sXtr, betahat)
    return {"Yhatr": Yhatr, "Yhats": Yhats, "betahat": betahat}

L = lassopred(X, Y, X, 100)
betahat = L["betahat"]

# Set up plotting with 1 row and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].set_xlim(-4, 4)
axs[0].set_ylim(-4, 4)
axs[0].set_xlabel("b1")
axs[0].set_ylabel("b2")
# Plot the first point in red with increased line width (using markersize to simulate lwd)
axs[0].plot(betahat[0], betahat[1], color="red", marker="o", markersize=10)

BETA = []
for lam in LAM:
    L_temp = lassopred(X, Y, X, lam)
    betahat = L_temp["betahat"]
    BETA.append(betahat)
    # Add points to the first subplot
    axs[0].plot(betahat[0], betahat[1], marker="o", color="blue", markersize=5)
    Remp.append(np.mean((Y - L_temp["Yhatr"])**2))
    print(".", end="")

# Second subplot: Plot empirical risk versus LAM
axs[1].set_xlabel("L")
axs[1].set_ylabel("Empirical risk")
axs[1].plot(LAM, Remp, linestyle="-", marker="")

# Third subplot: Plot estimation (each coefficient) versus LAM
BETA = np.array(BETA)
axs[2].set_xlabel("L")
axs[2].set_ylabel("Estimation")
axs[2].set_ylim(-5, 5)
axs[2].set_xlim(-2, 7)
# Plot the first estimation
axs[2].plot(LAM, BETA[:, 0], linestyle="-", marker="", label="b1")
# Plot the remaining estimations
for i in range(1, n):
    axs[2].plot(LAM, BETA[:, i], linestyle="-", marker="", label=f"b{i+1}")
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
    
# End of code
