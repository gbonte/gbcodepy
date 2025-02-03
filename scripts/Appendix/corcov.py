
# "INFOF422 Statistical foundations of machine learning" course
#Python package gbcodepy
# Author: G. Bontempi
# corcov.py

import numpy as np

# Remove all objects from the workspace (not strictly needed in Python, so we simply start fresh)
# In Python, starting the script anew automatically clears previous variables

n = 4
rho = 0.72  # bivariate correlation correlation
SigmaD = np.random.uniform(1, 2, 4)  # diagonal of covariance matrix: marginal variances
Corr = np.full((n, n), rho)
np.fill_diagonal(Corr, 1)

Sigma = np.diag(np.sqrt(SigmaD)) @ Corr @ np.diag(np.sqrt(SigmaD))

Corr2 = np.diag(1/np.sqrt(np.diag(Sigma))) @ Sigma @ np.diag(1/np.sqrt(np.diag(Sigma)))

N = 100000
D = np.random.multivariate_normal(np.zeros(n), Sigma, N)

for i in range(n-1):
    for j in range(i+1, n):
        # Compute sample correlation between columns i and j of D
        corr_ij = np.corrcoef(D[:, i], D[:, j])[0, 1]
        print("rho=", Corr2[i, j], ":", corr_ij)
        
