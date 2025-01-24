## bv_mult.py
##"INFOF422 Statistical foundations of machine learning" course
# Python translation of R package gbcode 
# Author: G. Bontempi

import numpy as np
import matplotlib.pyplot as plt



n = 4  # number of input variables
p = n + 1
N = 100  # number of training data points
R = 10000
sd_w = 5

# Generate N x n array of uniform random variables between -20 and 20
X = np.random.uniform(-20, 20, size=(N, n))

# Add a column of ones for the intercept term
X = np.column_stack((np.ones(N), X))

# Randomly sample integers between 1 and 10 for beta
beta = np.random.randint(1, 11, size=p)  # test with different values

# Initialize arrays to store results
beta_hat = np.zeros((p, R))
var_hat_w = np.zeros(R)
Y_hat = np.empty((R, N))
Y_hat[:] = np.nan

# Precompute the pseudo-inverse of X^T X
XTX_pinv = np.linalg.pinv(X.T @ X)

for r in range(R):
    # Generate Y with noise
    noise = np.random.normal(0, sd_w, size=N)
    Y = X @ beta + noise

    # Estimate beta_hat using pseudo-inverse
    beta_hat[:, r] = XTX_pinv @ X.T @ Y

    # Predict Y_hat
    Y_hat[r, :] = X @ beta_hat[:, r]

    # Calculate residuals
    e = Y - Y_hat[r, :]

    # Estimate variance of w
    var_hat_w[r] = (e @ e) / (N - p)

# Plot histogram of var_hat_w
plt.hist(var_hat_w, bins='auto')
plt.title(f"Distribution of var_hat.w: var w= {sd_w**2}")
plt.xlabel("var_hat.w")
plt.ylabel("Frequency")
plt.show()

# Plot histograms for each beta_hat
for i in range(p):
    plt.hist(beta_hat[i, :], bins='auto')
    plt.title(f"Distribution of beta_hat.{i+1}: beta {i+1}= {beta[i]}")
    plt.xlabel(f"beta_hat.{i+1}")
    plt.ylabel("Frequency")
    plt.show()
    

# Test unbiasedness and compare variances for a set of points
for i in range(5):
    # Calculate f(x_i)
    f_x_i = X[i, :] @ beta

    # Calculate mean of Y_hat for ith data point
    mean_Y_hat = np.mean(Y_hat[:, i])

    print(f"i= {i+1} E[yhat_i]= {mean_Y_hat} f(x_i)= {f_x_i}")

    # Calculate prediction variance analytically
    prediction_variance = (sd_w**2) * (X[i, :] @ XTX_pinv @ X[i, :].T)

    # Calculate empirical variance from Y_hat
    mc_value = np.var(Y_hat[:, i], ddof=1)

    print(f"i= {i+1} prediction variance= {prediction_variance} MC value= {mc_value}\n")
