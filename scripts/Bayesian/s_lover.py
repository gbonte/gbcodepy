import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# "Statistical foundations of machine learning" software
# R package gbcode 
# Author: G. Bontempi


plt.ioff()  # turn interactive mode off so that each plot can be shown with a manual pause

# par(ask=TRUE)
# In R, setting par(ask=TRUE) waits for user input between plots.
# In Python we mimic this by calling input() after showing each plot.

n_action = 2
n_state = 2  # number of states

N = 1  # number of observed samples
nZ = 3  # number of values of z

# lambda[n_action, n_state]: loss matrix
# Matrix constructed by row, equivalent to R's matrix(c(0.3,0.5,0.7,0.1), nrow=n_action, byrow=TRUE)
lambda_matrix = np.array([[0.3, 0.5],
                          [0.7, 0.1]])

# a priori probability [n_state, 1]
prior = np.array([0.4, 0.6])

# LIK[n_state, n_data]
# Matrix constructed by row, equivalent to R's matrix(c(0.5, 0.45, 0.05, 0.2, 0.2, 0.6), nrow=n_state, byrow=TRUE)
LIK = np.array([[0.5, 0.45, 0.05],
                [0.2, 0.2, 0.6]])

# n.data: number of different observable data
n_data = nZ ** N

delta = None
# Construct delta as a list of lists; each element is a list of 1:n_action.
# In R, for (i in 1:(n.data)) { delta <- c(delta, list(1:n.action)) }
delta = []
for i in range(1, n_data + 1):
    delta.append(list(range(1, n_action + 1)))

# Expand grid: delta[n_dec, n.data]
# In R, delta <- expand.grid(delta)
# Here we use product from itertools to generate all combinations
delta_grid = list(product(*delta))
# Convert list of tuples to a NumPy array with shape (n_dec, n_data)
delta_arr = np.array(delta_grid, dtype=int)
# action <- delta(dec, data)

n_dec = delta_arr.shape[0]  # number of decision rules

# R[n_dec, n_state]: initialize risk array with zeros
R = np.zeros((n_dec, n_state))

# Calculate risk for each decision rule and state
for dec in range(1, n_dec + 1):
    for state in range(1, n_state + 1):
        for data in range(1, n_data + 1):
            # Adjust indices: Python is 0-indexed while R is 1-indexed.
            R[dec - 1, state - 1] = (R[dec - 1, state - 1] +
                                     lambda_matrix[delta_arr[dec - 1, data - 1] - 1, state - 1] *
                                     LIK[state - 1, data - 1])
    plt.figure()
    # Plot risk for this decision rule
    plt.scatter(np.arange(1, n_state + 1), R[dec - 1, :], marker='o')
    plt.xlabel("State")
    plt.ylabel("risk")
    plt.title("Decision rule " + str(dec))
    plt.show()
    input("Press Enter to continue...")

print(R)

# bR: bayes risk array, dimension (n_dec x 1)
bR = np.zeros(n_dec)
for dec in range(1, n_dec + 1):
    for state in range(1, n_state + 1):
        bR[dec - 1] = bR[dec - 1] + R[dec - 1, state - 1] * prior[state - 1]  # bayes risk

im = np.argmin(bR)  # index of minimum bayes risk (0-indexed)

print(delta_arr[im, :])

# marg: marginal probability array for each data, dimension (n_data x 1)
marg = np.zeros(n_data)
for data in range(1, n_data + 1):
    for state in range(1, n_state + 1):
        marg[data - 1] = marg[data - 1] + LIK[state - 1, data - 1] * prior[state - 1]

# P: conditional probability P(state|data), dimension (n_state x n_data)
P = np.zeros((n_state, n_data))
for state in range(1, n_state + 1):
    for data in range(1, n_data + 1):
        P[state - 1, data - 1] = (LIK[state - 1, data - 1] * prior[state - 1]) / marg[data - 1]

# Ra: conditional risk R(a|data), dimension (n_action x n_data)
Ra = np.zeros((n_action, n_data))
for data in range(1, n_data + 1):
    for action in range(1, n_action + 1):
        for state in range(1, n_state + 1):
            Ra[action - 1, data - 1] = (Ra[action - 1, data - 1] +
                                        lambda_matrix[action - 1, state - 1] *
                                        P[state - 1, data - 1])

print(P)
print(Ra)
