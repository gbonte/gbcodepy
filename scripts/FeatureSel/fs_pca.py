#!/usr/bin/env python3
# "INFOF422 Statistical foundations of machine learning" course
# R package gbcode 
# Author: G. Bontempi

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


# set seed for reproducibility
np.random.seed(0)

def predKNN(X_train, Y_train, X_test,K=3):
    # Implements a simple k-nearest neighbor with k=1 for regression.
       # Compute Euclidean distances between X_test and each row of X_train
    distances = np.linalg.norm(X_train - X_test, axis=1)
    idx = np.argsort(distances)[:K]
    return np.mean(Y_train[idx])

N = 50

# correlated inputs
X1 = np.random.normal(0, 3, N)  # rnorm(N, sd=3)
X2 = 3 * X1 + np.random.normal(0, 0.4, N)  # 3*X1 + rnorm(N, sd=0.4)
X3 = 0.2 * X2 + np.random.normal(0, 0.5, N)  # 0.2*X2 + rnorm(N, sd=0.5)
X4 = -X3 + np.random.normal(0, 0.5, N)  # -X3 + rnorm(N, sd=0.5)

W = np.random.normal(0, 0.5, N)  # rnorm(N, sd=0.5)
# cbind(array(rnorm(N*5),c(N,5)), X1, X2, X3, X4)
first_part = np.random.normal(0, 1, (N, 5))
X = np.column_stack((first_part, X1, X2, X3, X4))

# Y = scale(X1^2+2*X2*X3+X4)+W
# First compute the expression X1^2+2*X2*X3+X4
Y_expr = X1**2 + 2 * X2 * X3 + X4
# Standardize Y_expr: subtract mean and divide by sample std (ddof=1)
Y_expr_mean = np.mean(Y_expr)
Y_expr_std = np.std(Y_expr, ddof=1)
Y_scaled = (Y_expr - Y_expr_mean) / Y_expr_std
Y = Y_scaled + W

n = X.shape[1]
E = np.empty((N, n))
E[:] = np.nan  # Initialize with NA equivalent
EPCA = np.empty((N, n))
EPCA[:] = np.nan  # Initialize with NA equivalent

## LEAVE-ONE-OUT loop
for i in range(N):
    # Create the training set by leaving out the i-th observation
    Xtr = np.delete(X, i, axis=0)
    Ytr = np.delete(Y, i, axis=0)
    
    # Normalize training inputs (column-wise standardization using sample std)
    Xtr_mean = np.mean(Xtr, axis=0)
    Xtr_std = np.std(Xtr, axis=0, ddof=1)
    Xtr_scaled = (Xtr - Xtr_mean) / Xtr_std
    
    # normalization of the input test
    Xts = (X[i, :] - Xtr_mean) / Xtr_std
    
    # SVD on scaled training data divided by sqrt(N-1)
    U, d, Vt = np.linalg.svd(Xtr_scaled / sqrt(N-1), full_matrices=False)
    V = Vt.T  # In Python, np.linalg.svd returns Vt; columns of V are the right singular vectors.
    
    ## PC loop
    for h in range(1, n + 1):
        # Vh = first h columns of V
        Vh = V[:, :h]
        # Compute the principal component scores for training data
        Zh = np.dot(Xtr_scaled, Vh)
        
        # Compute the principal component scores for the test input
        Zi = np.dot(Xts, Vh)
        
        # knn prediction using PCA transformed data
        YhatPCAi = predKNN( Zh, Ytr, Zi)
        # knn prediction using the first h columns of the original scaled data
        Yhati = predKNN( Xtr_scaled[:, :h], Ytr, Xts[:h])
        
        EPCA[i, h - 1] = (Y[i] - YhatPCAi) ** 2
        E[i, h - 1] = (Y[i] - Yhati) ** 2

# Calculate mean squared error for each number of PCs and print the results
mse_loo_PCA = np.mean(EPCA, axis=0)
best_pc_number = np.argmin(mse_loo_PCA) + 1  # +1 for 1-indexed result
print("MSEloo PCA=", mse_loo_PCA, "\n best PC number=", best_pc_number, "\n")
mse_loo = np.mean(E, axis=0)
print("MSEloo =", mse_loo, "\n")

# Standardize full X matrix
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0, ddof=1)
X_scaled = (X - X_mean) / X_std

# SVD on full standardized data
U_full, d_full, Vt_full = np.linalg.svd(X_scaled, full_matrices=False)
V_full = Vt_full.T
# Principal components score for full data
Z = np.dot(X_scaled, V_full)

# Compute variance for each principal component (sample variance: ddof=1)
Z_variance = np.var(Z, axis=0, ddof=1)
eigenvalues = d_full**2 / (N - 1)
print("\n --- \n Variance Z=", Z_variance, "\n Eigenvalues=", eigenvalues, "\n")

# Plot Eigenvalues
plt.plot(eigenvalues, linestyle='-', marker='', label="Eigenvalues")
plt.ylabel("Eigenvalues")
plt.show()
