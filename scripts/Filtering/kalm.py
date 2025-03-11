import numpy as np
import matplotlib.pyplot as plt

# x1(k+1)=x1(k)+u1(k)+w1
# x2(k+1)=x2(k)+u2(k)+w2

# rm(list=ls())
# In Python, we do not need to clear the workspace as in R.

# Import mvtnorm functionality via numpy's multivariate_normal
# library(mvtnorm)
A = np.array([[0.9, 0.1],
              [0.1, 0.9]])
B = np.ones((2, 2))  # B is a 2x2 matrix filled with ones

H = np.eye(2)  # Identity matrix for H

Q = np.eye(2)
R = 0.1 * np.eye(2)

Pplus = 1e6 * np.eye(2)

x = np.array([[1],
              [10]])
xplus = np.zeros((2, 1))
xmin = None
z = x.copy()
T = 100
# par(mfrow=c(1,1))
I2 = np.eye(2)

for k in range(1, T):  # k in R: 2:T (R is 1-indexed; Python uses 0-indexing)
    u = np.random.uniform(-1, 1, size=2)
    ## u(k-1)
    
    # Dynamic system state update
    noise_state = np.random.multivariate_normal(mean=np.zeros(2), cov=Q).reshape(2, 1)
    new_state = A @ x[:, k-1].reshape(2, 1) + B @ u.reshape(2, 1) + noise_state
    x = np.concatenate((x, new_state), axis=1)
    
    # Dynamic system observations
    noise_obs = np.random.multivariate_normal(mean=np.zeros(2), cov=R).reshape(2, 1)
    new_obs = H @ x[:, k-1].reshape(2, 1) + noise_obs
    z = np.concatenate((z, new_obs), axis=1)
    
    # KF step 1
    xmin = A @ xplus[:, k-1].reshape(2, 1) + B @ u.reshape(2, 1)
    
    # KF step 2
    Pmin = A @ Pplus @ A.T + Q
    
    # KF step 3
    K = Pmin @ H.T @ np.linalg.inv(H @ Pmin @ H.T + I2)
    
    # KF step 4
    new_xplus = xmin + K @ (z[:, k].reshape(2, 1) - H @ xmin)
    xplus = np.concatenate((xplus, new_xplus), axis=1)
    
    # KF step 5
    Pplus = (I2 - K @ H) @ Pmin
    
    if False:
        ## step by step tracking visualisation
        plt.figure()
        plt.plot(x[0, :], x[1, :], 'b-', label="Tracking")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Tracking")
        plt.plot(xplus[0, :], xplus[1, :], 'ro', label="estimated by KF")
        # The following uses an undefined variable 'tt' as in the original R code.
        # plt.plot(x[0, tt], x[1, tt], 'ko', markersize=8)
        # plt.plot(xplus[0, tt], xplus[1, tt], 'ro', markersize=8)
        plt.legend(loc='upper left')
        plt.show()
        input()  # readline()

# par(mfrow=c(1,3))
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, T+1), x[0, :], 'b-', label="x1")
plt.plot(range(1, T+1), xplus[0, :], 'ro', label="estimated by KF")
plt.xlabel("k")
plt.ylabel("x1")
plt.legend(loc='upper left')

plt.subplot(1, 3, 2)
plt.plot(range(1, T+1), x[1, :], 'b-', label="x2")
plt.plot(range(1, T+1), xplus[1, :], 'ro', label="estimated by KF")
plt.xlabel("k")
plt.ylabel("x2")
plt.legend(loc='upper left')

plt.subplot(1, 3, 3)
plt.plot(x[0, :], xplus[0, :], 'b-', label="x1 vs estimated KF")
plt.xlabel("x1")
plt.ylabel("x2")
plt.plot(x[1, :], xplus[1, :], 'ro')
plt.show()

MAE = np.mean(np.abs(x - xplus))  # error after filtering
MAE2 = np.mean(np.abs(x - z))       # error without filtering

print("MAE =", MAE)
print("MAE2 =", MAE2)
