# "INFOF422 Statistical foundations of machine learning" course
# ee.py
# Monte Carlo computation of Expected Empirical Error and MISE in the linear case
# Author: G. Bontempi

import numpy as np



X = np.arange(-10, 10.1, 0.1)

beta_0 = -1
beta_1 = 1.4
N = len(X)
R = 10000  # number of MC trials
sd_w = 4
p = 2
beta_hat_1 = np.zeros(R)
beta_hat_0 = np.zeros(R)
var_hat_w = np.zeros(R)
EE = np.zeros(R)
MISE = np.zeros(R)
Y_hat = np.empty((R, N))

for r in range(R):
    Y = beta_0 + beta_1 * X + np.random.normal(0, sd_w, size=N)
    x_hat = np.mean(X)
    y_hat = np.mean(Y)
    S_xy = np.sum((X - x_hat) * Y)
    S_xx = np.sum((X - x_hat) ** 2)
    
    beta_hat_1[r] = S_xy / S_xx
    beta_hat_0[r] = y_hat - beta_hat_1[r] * x_hat
    
    Y_hat[r, :] = beta_hat_0[r] + beta_hat_1[r] * X
    
    EE[r] = np.mean((Y - Y_hat[r, :]) ** 2)  # empirical error
    var_hat_w[r] = np.sum((Y - Y_hat[r, :]) ** 2) / (N - 2)
    
    Yts = beta_0 + beta_1 * X + np.random.normal(0, sd_w, size=N)  # test set
    MISE[r] = np.mean((Yts - Y_hat[r, :]) ** 2)

MISEemp_th = (1 - p / N) * sd_w ** 2

print(f"Expected empirical MISE: analytical= {MISEemp_th}; simulated (MC) = {np.mean(EE)}")

MISE_th = (1 + p / N) * sd_w ** 2

print(f"MISE: analytical = {MISE_th}; simulated (MC) = {np.mean(MISE)}")
