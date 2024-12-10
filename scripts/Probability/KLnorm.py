import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import det, inv

# Set random seed for reproducibility
np.random.seed(1)

n = 6  # dimension of the multivariate r.v.s

mu0 = np.random.normal(size=(n, 1))
mu1 = np.random.normal(loc=0.25, scale=0.5, size=(n, 1))

A0 = np.random.uniform(-1, 1, size=(n, n))
Sigma0 = A0.T @ A0
A1 = np.random.uniform(-1, 1, size=(n, n))
Sigma1 = A1.T @ A1

KL = 0.5 * (np.log(det(Sigma0)) - np.log(det(Sigma1)) - n +
            (mu0 - mu1).T @ inv(Sigma1) @ (mu0 - mu1) +
            np.trace(inv(Sigma1) @ Sigma0))

KLmc = []
R = 5000  # number of MC trials
for _ in range(R):
    z = multivariate_normal.rvs(mean=mu0.flatten(), cov=Sigma0)
    KLmc.append(multivariate_normal.logpdf(z, mean=mu0.flatten(), cov=Sigma0) -
                multivariate_normal.logpdf(z, mean=mu1.flatten(), cov=Sigma1))

KLmc = np.array(KLmc)
KLmc = KLmc[~np.isinf(KLmc)]
print(f"KL= {KL[0][0]}, MC computation of KL= {np.mean(KLmc)}")

