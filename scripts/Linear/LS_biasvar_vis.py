# "INFOF422 Statistical foundations of machine learning" course
# Python translation of R package gbcode 
# Author: G. Bontempi

#################################################################
# biasvar_vis.py						                #
#################################################################
# Dataset D_N={x_i,y_i} :			
#	  y_i = beta_0 + beta_1x_i + w_i			#
#   with known  beta_0  and beta_1 and known w=Normal(0, sigma) 	#
#   						
# Visualization of bias and variance of the least square estimate 	#
#################################################################

import numpy as np
import matplotlib.pyplot as plt

# preliminary
# ============
np.random.seed(0)
X = np.arange(-10, 10.25, 0.25)  # fixed xi
beta0 = -1
beta1 = 1
sd_w = 3
N = len(X)
R = 50  # number of MC trials

beta_hat_1 = np.zeros(R)
beta_hat_0 = np.zeros(R)
var_hat_w = np.zeros(R)
Y_hat = np.empty((R, N))
Y_hat.fill(np.nan)
x_bar = np.mean(X)
S_xx = np.sum((X - x_bar) ** 2)

plt.figure(figsize=(10, 6))

for r in range(R):
    Y = beta0 + beta1 * X + np.random.normal(scale=sd_w, size=N)  # data generation
    y_bar = np.mean(Y)
    S_xy = np.sum((X - x_bar) * Y)
    
    beta_hat_1[r] = S_xy / S_xx
    beta_hat_0[r] = y_bar - beta_hat_1[r] * x_bar
    
    Y_hat[r, :] = beta_hat_0[r] + beta_hat_1[r] * X
    var_hat_w[r] = np.sum((Y - Y_hat[r, :]) ** 2) / (N - 2)
    
   

# After all trials
plt.clf()
plt.plot(X, beta0 + beta1 * X, color='green',
         label='True model', linewidth=3)
plt.scatter(X, Y, color='blue', s=10, label='Data points')

plt.plot(X, Y_hat.T, color='grey', linewidth=1)
plt.plot(X, np.mean(Y_hat, axis=0), color='red', linewidth=3, label='Mean prediction')
plt.title(f"beta0={beta0} beta1={beta1} sdw={sd_w}  N={N}")
plt.ylim(-10, 10)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
