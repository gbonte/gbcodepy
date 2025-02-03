## "Statistical foundations of machine learning" software
## Python package gbcodepy
## Author: G. Bontempi
## bv_linfs.py

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv




n = 4  # number input variables
p = n + 1
N = 20  # number training data


X = np.array(np.random.uniform(low=-2, high=2, size=(N, n)))
X = np.hstack((np.ones((N, 1)), X, np.array(np.random.normal(size=(N, n)))))
## the last n variables are irrelevant

np.random.seed(0)
beta = 0.5 * np.array([1, -1, 1, -1, 1])

R = 10000
sd_w = 0.5

# beta.hat <- array(0, c(p, R))
beta_hat_store = np.zeros((p, R))
# var.hat.w <- numeric(R)
var_hat_w = np.zeros(R)
# Y.hat <- array(NA, c(R, N, X.shape[1]))
Y_hat = np.full((R, N, X.shape[1]), np.nan)
# e.hat <- array(NA, c(R, N, X.shape[1]))
e_hat = np.full((R, N, X.shape[1]), np.nan)

for r in range(R):  # for (r in 1:R) translated to 0-indexed
    for nfs in range(X.shape[1]):  # for (nfs in 0:(NCOL(X)-1)) 
        Y = np.dot(X[:, :p], beta) + np.random.normal(scale=sd_w, size=N)
        Xsel = X[:, :nfs+1]
        beta_hat = np.dot(pinv(np.dot(Xsel.T, Xsel)), np.dot(Xsel.T, Y))
        Y_hat[r, :, nfs] = np.dot(Xsel, beta_hat)

aV = None
aVmc = None
aB = None
aMSE = None
aV = []
aVmc = []
aB = []
aMSE = []
for nfs in range(X.shape[1]):  # for (nfs in 0:(NCOL(X)-1))
    Vh = []
    Vmc = []
    Bmc = []
    MSEmc = []
    for i in range(N):  # for (i in 1:N)
        if nfs == 0:
            # When nfs==0, X[i,1:nfs] is empty so we set the variance term to 0
            current_Vh = sd_w**2 * 0
        else:
            # Compute: sd.w^2 * (t(X[i,1:nfs]) %*% ginv(t(X[,1:nfs]) %*% X[,1:nfs]) %*% X[i,1:nfs])
            Xi = X[i, :nfs].reshape(1, -1)  # row vector with shape (1, nfs)
            X_subset = X[:, :nfs]           # matrix with shape (N, nfs)
            current_Vh = sd_w**2 * np.dot(np.dot(Xi, pinv(np.dot(X_subset.T, X_subset))), Xi.T)[0, 0]
        Vh.append(current_Vh)
        # Vmc <- c(Vmc, var(Y.hat[,i,nfs+1]))
        current_Vmc = np.var(Y_hat[:, i, nfs])
        Vmc.append(current_Vmc)
        # Bmc <- c(Bmc, mean(Y.hat[,i,nfs+1]) - X[i,1:p] %*% beta)
        current_Bmc = np.mean(Y_hat[:, i, nfs]) - np.dot(X[i, :p], beta)
        Bmc.append(current_Bmc)
        # MSEmc <- c(MSEmc, mean((Y.hat[,i,nfs+1]- c(X[i,1:p] %*% beta))^2))
        current_MSEmc = np.mean((Y_hat[:, i, nfs] - np.dot(X[i, :p], beta))**2)
        MSEmc.append(current_MSEmc)
    
    # comparison analytical and simulated variance of the prediction
    print("nfs=", nfs, "variance: th=", np.mean(Vh), 
          "MC =", np.mean(Vmc), "\n \n", "bias: MC=", np.mean(np.array(Bmc)**2))
    aV.append(np.mean(Vh))
    aVmc.append(np.mean(Vmc))
    aB.append(np.mean(np.array(Bmc)**2))
    aMSE.append(np.mean(MSEmc))

x_values = np.arange(0, X.shape[1])
plt.plot(x_values, aB, linestyle='--', color="green", label="Bias^2 (MC)")
# lines(0:(NCOL(X)-1), aV, col="red", lty=3)
plt.plot(x_values, aV, linestyle='-.', color="red", label="Variance")
# lines(0:(NCOL(X)-1), aVmc, col="red", lty=4)
plt.plot(x_values, aVmc, linestyle=':', color="red", label="Variance (MC)")
# lines(0:(NCOL(X)-1), aMSE, col="black", lty=1, lwd=2)
plt.plot(x_values, aMSE, linestyle='-', color="black", linewidth=2, label="MSE (MC)")
plt.xlabel("Number of features")
plt.ylabel("")
plt.title("Feature selection linear regression: Bias/variance trade-off")
plt.legend(loc='upper right')
plt.show()
