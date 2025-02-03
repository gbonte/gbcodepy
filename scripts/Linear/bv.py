# "INFOF422 Statistical foundations of machine learning" course
# Python package gbcodepy
# Author: G. Bontempi
#
#################################################################
# bv.py                                                     #



import numpy as np
import matplotlib.pyplot as plt

#################################################################
# Dataset D_N = {x_i, y_i} :
#   y_i = beta0 + beta1 * x_i + w_i
#   with known beta0, beta1 and w ~ Normal(0, sigma)
#
# Experimental analysis of bias and variance of the least squares estimate
# and comparison with theoretical results
#################################################################

# Preliminary
# ============
# Clear all variables (in Python, we typically don’t clear the environment)
# The equivalent effect is restarting the interpreter if needed.

# Define fixed x values
X = np.arange(-10, 10 + 0.04, 0.04)  # fixed x_i
beta0 = -1      # y_i = -1 + 1*x_i + Normal(0, 10)
beta1 = 1
sd_w = 10
N = len(X)
R = 10000     # number of Monte Carlo trials

# -------------------------------------------------
# Computation of Var(beta1), Var(beta0), and sigma_w^2, analytically and by simulation
# -------------------------------------------------
beta_hat_1 = np.empty(R)
beta_hat_0 = np.empty(R)
var_hat_w = np.empty(R)
Y_hat = np.full((R, N), np.nan)

x_bar = np.mean(X)
S_xx = np.sum((X - x_bar)**2)

for r in range(R):
    # Generate response variable with additive gaussian noise
    Y = beta0 + beta1 * X + np.random.normal(loc=0, scale=sd_w, size=N)
    y_bar = np.mean(Y)
    S_xy = np.sum((X - x_bar) * Y)
    
    beta_hat_1[r] = S_xy / S_xx
    beta_hat_0[r] = y_bar - beta_hat_1[r] * x_bar
    
    Y_hat[r, :] = beta_hat_0[r] + beta_hat_1[r] * X
    var_hat_w[r] = np.sum((Y - Y_hat[r, :])**2) / (N - 2)

# Theoretical variance for beta1
var_beta_hat_1 = (sd_w ** 2) / S_xx
observed_var_beta1 = np.var(beta_hat_1, ddof=1)
print("Theoretical var beta1 = {:.5f}; Observed = {:.5f}".format(var_beta_hat_1, observed_var_beta1))

plt.figure()
plt.hist(beta_hat_1, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of beta_hat_1: beta1 = {}".format(beta1))
plt.xlabel("beta_hat_1")
plt.ylabel("Frequency")
plt.show()

# Theoretical variance for beta0
var_beta_hat_0 = (sd_w ** 2) * (1 / N + (x_bar ** 2) / S_xx)
observed_var_beta0 = np.var(beta_hat_0, ddof=1)
print("Theoretical var beta0 = {:.5f}; Observed = {:.5f}".format(var_beta_hat_0, observed_var_beta0))

plt.figure()
plt.hist(beta_hat_0, bins=50, color='lightgreen', edgecolor='black')
plt.title("Distribution of beta_hat_0: beta0 = {}".format(beta0))
plt.xlabel("beta_hat_0")
plt.ylabel("Frequency")
plt.show()

# Histogram for estimated sigma_w^2
plt.figure()
plt.hist(var_hat_w, bins=50, color='salmon', edgecolor='black')
plt.title("Distribution of var_hat_w: var(w) = {}".format(sd_w ** 2))
plt.xlabel("var_hat_w")
plt.ylabel("Frequency")
plt.show()

# -------------------------------------------------
# Plot of predictions
# -------------------------------------------------
plt.figure()
plt.plot(X, Y_hat[0, :], 'b-', label="Simulation 1")
plt.title("Variance of the prediction: {} repetitions".format(R))
plt.ylim(-20, 20)
# Plot predictions of all replicates
for r in range(1, R):
    plt.plot(X, Y_hat[r, :], color='b', alpha=0.01)  # using low alpha for transparency
# Plot the empirical mean of predictions over replicates
plt.plot(X, np.mean(Y_hat, axis=0), color='red', linewidth=2, label="Empirical mean")
# Plot the true regression line
plt.plot(X, beta0 + beta1 * X, color='green', linewidth=2, label="True regression line")
plt.xlabel("X")
plt.ylabel("Predicted Y")
plt.legend()
plt.show()

# -------------------------------------------------
# Empirical validation of: E[y(x)] = y(x) for all x
# -------------------------------------------------
# (The above plot already includes both the empirical mean and the true regression line)

# -------------------------------------------------
# Empirical illustration: Var[y(x)] = sigma_w^2 [1/N + (x - x_bar)^2 / S_xx]
# Study of variance of predictions Var[y|x], analytical and from simulation.
# -------------------------------------------------
for i in range(N):
    var_y_hat = np.var(Y_hat[:, i], ddof=1)
    theoretical_var_y_hat = sd_w ** 2 * (1 / N + ((X[i] - x_bar) ** 2) / S_xx)
    print("x = {:.2f}: Theoretical var predic = {:.5f}; Observed = {:.5f}".format(X[i], theoretical_var_y_hat, var_y_hat))
    
# Note: The print statements in the loop above will output a lot of lines given N is large.
