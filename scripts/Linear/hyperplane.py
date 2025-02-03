# "INFOF422 Statistical foundations of machine learning" course
# Python package gbcodepy
# Author: G. Bontempi

import numpy as np
import matplotlib.pyplot as plt

#

def norm(x):
    return np.sqrt(np.sum(x**2))


np.random.seed(0)

Np = 100
Nn = 100
N = Np + Nn
P = np.array([Np, Nn]) / (Np + Nn)
sigma2 = 3

mu_p = np.array([-1, -2])
mu_n = np.array([2, 5])

# Generate positive class data (Xp) with two features
Xp = np.column_stack((np.random.normal(mu_p[0], np.sqrt(sigma2), Np),
                       np.random.normal(mu_p[1], np.sqrt(sigma2), Np)))

# Generate negative class data (Xn) with two features
Xn = np.column_stack((np.random.normal(mu_n[0], np.sqrt(sigma2), Nn),
                       np.random.normal(mu_n[1], np.sqrt(sigma2), Nn)))

# Combine data and set labels: positive class labeled 1, negative class labeled -1
X = np.vstack((Xp, Xn))
Y = -np.ones(N)
Y[:Np] = 1

# Initialize beta: beta[0] is intercept, beta[1] and beta[2] are coefficients for features x1 and x2, respectively.
beta = np.full(3, -2.0)

mu_val = 0.1

plt.ion()  # Turn on interactive mode for live plotting

for i in np.arange(20):
    # Compute the decision function: f(x) = beta[0] + beta[1]*x1 + beta[2]*x2 
    predictions = beta[0] + X.dot(beta[1:])
    # Identify misclassified samples (where Y * prediction < 0)
    misclassified = np.where(Y * predictions < 0)[0]

    # Break the loop if no misclassifications remain.
    if misclassified.size == 0:
        break

    # Plot the current classification state
    plt.clf()
    plt.scatter(Xp[:, 0], Xp[:, 1], color="red", label="Positive Class (Y=1)")
    plt.scatter(Xn[:, 0], Xn[:, 1], color="green", label="Negative Class (Y=-1)")
    # Create a sequence of x1 values for plotting the decision boundary.
    X1 = np.arange(-10, 10.1, 0.1)
    # Decision boundary: beta[0] + beta[1]*x1 + beta[2]*x2 = 0  =>  x2 = -beta[0]/beta[2] - (beta[1]/beta[2])*x1
    # Note: If beta[2] is zero, this line is vertical (R code assumes nonzero beta[3]).
    if beta[2] != 0:
        boundary_line = -beta[0] / beta[2] - (beta[1] / beta[2]) * X1
        plt.plot(X1, boundary_line, "b-", label="Decision Boundary")
    else:
        # Plot a vertical line if beta[2] is zero
        x_vert = -beta[0] / beta[1] if beta[1] != 0 else 0
        plt.axvline(x=x_vert, color="b", label="Decision Boundary")
        
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"# misclassified = {misclassified.size}")
    plt.legend()
    plt.draw()
    plt.pause(0.1)

    # Update the classifier parameters based on misclassified points.
    beta[0] = beta[0] + mu_val * np.sum(Y[misclassified])
    # To update feature coefficients, we multiply the labels with corresponding features and sum along the sample axis.
    beta[1:] = beta[1:] + mu_val * np.sum(Y[misclassified, np.newaxis] * X[misclassified, :], axis=0)

plt.ioff()
plt.show()
