# "Statistical foundations of machine learning" software
# package gbcodepy
# Author: G. Bontempi

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Set seed for reproducibility if needed
np.random.seed(0)

q = 10  # order of MA(q)
N = q * 200
D = np.random.randn(N)  # equivalent to rnorm(N)

beta = np.abs(0.75 * np.random.randn(q))
beta = np.concatenate(([1], beta))  # equivalent to c(1, beta)
Y = []  # initialize an empty list for Y
for i in range(q, N):  # for i in (q+1):N in R (adjusted for 0-indexing)
    # Compute sum(beta * rev(D[(i-q):i+1]))
    window = D[i - q : i + 1]  # slice corresponding to D[(i-q):i] in R (inclusive)
    reversed_window = window[::-1]  # reverse the window
    Y.append(np.sum(beta * reversed_window))

Y = np.array(Y)
N_Y = len(Y)  # redefine N as length of Y
Co_emp = np.zeros(2 * q)  # numeric(2*q) in R initializes a vector of zeros
Co_th = np.zeros(2 * q)   # numeric(2*q) in R

for k in range(1, 2 * q + 1):  # for k in 1:(2*q) in R
    # Create Yk = c(numeric(k), Y)
    Yk = np.concatenate((np.zeros(k), Y))
    
    # Compute correlation C = cor(Y[(k+1):N], Yk[(k+1):N])
    # In R, indices (k+1):N (with N = length(Y)) correspond to Python indices [k: N_Y]
    segment_Y = Y[k:]
    segment_Yk = Yk[k:N_Y]
    # Compute correlation coefficient between segment_Y and segment_Yk
    if segment_Y.std() == 0 or segment_Yk.std() == 0:
        C = 0.0
    else:
        C = np.corrcoef(segment_Y, segment_Yk)[0, 1]
    Co_emp[k - 1] = C

    Co_th[k - 1] = 0.0
    if k <= q:
        # for j in 1:(q+1-k) in R, adjust indices for 0-indexing in Python
        for j in range(0, q + 1 - k):
            Co_th[k - 1] = Co_th[k - 1] + beta[j] * beta[j + k]
        Co_th[k - 1] = Co_th[k - 1] / np.sum(beta ** 2)

# Set up the plotting environment to mimic par(mfrow=c(3,1), mai = 0.3*c(1,1,1,1), mar = 2*c(1,1,1,1))
fig, axs = plt.subplots(3, 1, figsize=(6, 8))
plt.subplots_adjust(hspace=0.5)

# Plot(Y,xlab='',main=paste("MA(",q,")")) in R
axs[0].plot(Y)
axs[0].set_title("MA({})".format(q))
axs[0].set_xlabel('')
axs[0].set_ylabel('')

# plot(abs(Co_emp),type="l",lty=2,ylab='',xlab='k')
axs[1].plot(np.abs(Co_emp), linestyle='--', label='Estimated cor')
# lines(abs(Co_th),lty=1)
axs[1].plot(np.abs(Co_th), linestyle='-', label='Cor')
axs[1].set_xlabel('k')
axs[1].set_ylabel('')
axs[1].legend(loc='upper right', fontsize='small')

# acf(Y) using statsmodels' plot_acf
plot_acf(Y, ax=axs[2])
plt.show()
