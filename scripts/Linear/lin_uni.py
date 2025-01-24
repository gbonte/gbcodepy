# "INFOF422 Statistical foundations of machine learning" course
# lin_uni.py
# Example of univariate linear regression

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Generate data
N = 30
sigma_w = 0.5
X = np.random.normal(size=N)
W = np.random.normal(scale=0.5, size=N)

beta0 = 1/2
beta1 = 1.2
Y = beta0 + beta1 * X + W

# Create scatter plot
plt.figure()
plt.scatter(X, Y)

# Calculate means
xhat = np.mean(X)
yhat = np.mean(Y)

# Calculate sums for least squares
Sxy = np.sum((X - xhat) * Y)
Sxx = np.sum((X - xhat) * X)

# Univariate least-squares
betahat1 = Sxy / Sxx
betahat0 = yhat - betahat1 * xhat

# Plot regression line
X_sorted = np.sort(X)
plt.plot(X_sorted, betahat0 + X_sorted * betahat1, 'r-')
plt.show()
