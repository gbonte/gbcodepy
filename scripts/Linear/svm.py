# "INFOF422 Statistical foundations of machine learning" course
# Python package gbcodepy
# Author: G. Bontempi
## svm.py

import numpy as np
import matplotlib.pyplot as plt
import math
from cvxopt import matrix, solvers

# Disable solver output
solvers.options['show_progress'] = False

# Define the p-norm function
def normv(x, p=2):
    return np.sum(np.abs(x)**p)**(1/p)

# Flag for linearly separable case
separable = True

if not separable:
    gam = 0.05
else:
    gam = float("inf")  # Use infinity for the separable case

eps = 0.001

# Only one replication as in the original R code
for rep in range(1):
    N = 150  # number of samples per class
    # Generate class 1 samples
    x1 = np.column_stack((np.random.normal(0, 0.2, N), np.random.normal(0, 0.2, N)))
    y1 = np.ones(N)
    
    # Generate class 2 samples
    x2 = np.column_stack((np.random.normal(3, 0.5, N), np.random.normal(3, 0.5, N)))
    y2 = -np.ones(N)
    
    # Combine the two classes
    X = np.vstack((x1, x2))      # shape: (2N, 2)
    Y = np.concatenate((y1, y2)) # shape: (2N,)
    
    # PLOT Training set
    plt.figure()
    plt.xlim(-2, 6)
    plt.ylim(-2, 6)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(x1[:, 0], x1[:, 1], color="red", label="Class 1")
    plt.scatter(x2[:, 0], x2[:, 1], color="blue", label="Class 2")
    plt.legend()
    plt.title("Training Data")
    plt.draw()
    plt.pause(1)  # Pause to allow visualization, similar to par(ask=TRUE)
    
    # SVM parametric identification
    num_samples = 2 * N
    # Constructing the D matrix: Dmat[i,j] = Y[i]*Y[j]*(X[i]·X[j])
    # This can be computed efficiently using outer product and matrix multiplication.
    K = X @ X.T  # Kernel matrix of dot-products
    Dmat = np.outer(Y, Y) * K
    # Adding a small regularization term
    Dmat = Dmat + 1e-3 * np.eye(num_samples)
    
    # The vector d in the objective function; in R code it is a column vector of ones.
    d_vec = np.ones(num_samples)
    
    # In the R formulation, the problem is:
    #   min  (1/2) * alpha^T Dmat alpha - d_vec^T alpha,
    # subject to:
    #   1st constraint: sum_i Y[i]*alpha[i] = 0,
    #   and for i=1...num_samples: alpha[i] >= 0,
    #   and, if not separable, also: alpha[i] <= gam.
    #
    # CVXOPT standard form:
    #   min (1/2)x^T P x + q^T x  subject to Gx <= h and Ax = b.
    #
    # We set:
    #   P = Dmat, q = -d_vec.
    #
    # Equality constraint: A_eq * alpha = 0, where A_eq is a 1 x num_samples matrix (Y as row vector).
    A_eq = Y.reshape(1, -1)
    b_eq = np.array([0.0])
    
    # Inequality constraints:
    # Constraint 1: alpha_i >= 0  =>  -alpha_i <= 0.
    G_lower = -np.eye(num_samples)
    h_lower = np.zeros(num_samples)
    
    if not separable:
        # Constraint 2 (only for non-separable case): alpha_i <= gam  =>  alpha_i - gam <= 0.
        G_upper = np.eye(num_samples)
        h_upper = np.full(num_samples, gam)
        G = np.vstack((G_lower, G_upper))
        h = np.concatenate((h_lower, h_upper))
    else:
        G = G_lower
        h = h_lower
    
    # Convert numpy arrays to cvxopt matrices
    P = matrix(Dmat)
    q = matrix(-d_vec)  # because we want to minimize -d^T alpha + ... which is q^T alpha with q = -1.
    G_cvx = matrix(G)
    h_cvx = matrix(h)
    A_cvx = matrix(A_eq)
    b_cvx = matrix(b_eq)
    
    # Solve the quadratic programming problem
    sol = solvers.qp(P, q, G_cvx, h_cvx, A_cvx, b_cvx)
    alpha = np.array(sol['x']).flatten()
    
    # Threshold small alpha values to zero
    alpha[alpha < eps] = 0
    
    # Find support vectors with alpha significantly between 0 and (if applicable) gam.
    if np.isinf(gam):
        ind_j = np.where(alpha > eps)[0]
    else:
        ind_j = np.where((alpha > eps) & (alpha < gam - eps))[0]
    
    if (np.all(alpha <= (gam + eps)) if not np.isinf(gam) else True) and (len(ind_j) > 0):
        print("min value =", sol['primal objective'])
        # Compute the objective value manually
        val2 = -np.dot(d_vec, alpha) + 0.5 * alpha @ Dmat @ alpha
        print("min value2 =", val2)
        print("sum_i y_i*alpha_i =", np.dot(alpha, Y))
    
        # Compute beta = sum_i alpha_i * Y[i] * X[i]
        beta = np.sum((alpha * Y)[:, np.newaxis] * X, axis=0)
    
        # Identify support vectors for the two classes
        ind1 = np.where((alpha[:N] > eps))[0]
        ind2 = np.where((alpha[N:] > eps))[0]
    
        plt.figure()
        plt.xlim(-2, 6)
        plt.ylim(-2, 6)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.scatter(x1[:, 0], x1[:, 1], color="red", label="Class 1")
        plt.scatter(x2[:, 0], x2[:, 1], color="blue", label="Class 2")
   
        # Plot support vectors in black
        plt.scatter(X[ind1, 0], X[ind1, 1], color="black", marker="o", s=80, label="SV Class 1")
        # For second class, adjust index offset by N
        plt.scatter(X[N + ind2, 0], X[N + ind2, 1], color="black", marker="o", s=80, label="SV Class 2")
        plt.legend()
    
        # Compute bias term and margin
        if separable:
            # Use one support vector from each class
            beta0 = -0.5 * (np.dot(beta, X[ind1[0]]) + np.dot(beta, X[N + ind2[0]]))
            marg = 1.0 / normv(beta)
        else:
            # Compute L = sum_i,sum_j Y[i]*Y[j]*alpha[i]*alpha[j]*(X[i].T X[j])
            L = alpha @ (np.outer(Y, Y) * (X @ X.T)) @ alpha
            beta0 = (1 - Y[ind_j[0]] * np.dot(beta, X[ind_j[0]])) / Y[ind_j[0]]
            marg = 1.0 / math.sqrt(L)
    
            # Points whose slack variable is positive (alpha approximately equal to gam)
            ind3 = np.where(np.abs(alpha[:N] - gam) < eps)[0]
            ind4 = np.where(np.abs(alpha[N:] - gam) < eps)[0]
            # Plot these points with different colors
            if len(ind3) > 0:
                plt.scatter(X[ind3, 0], X[ind3, 1], color="yellow", marker="s", s=80, label="Slack SV Class 1")
            if len(ind4) > 0:
                plt.scatter(X[N + ind4, 0], X[N + ind4, 1], color="green", marker="s", s=80, label="Slack SV Class 2")
            plt.legend()
    
        print("beta =", beta)
        theta = math.atan(-beta[0] / beta[1])
        print("theta =", theta)
    
        # Plot Separating Hyperplane and Margin Lines
        # The hyperplane equation: beta[0]*x + beta[1]*y + beta0 = 0
        # Solve for y: y = -(beta[0]/beta[1])*x - beta0/beta[1]
        x_vals = np.array([-2, 6])
        slope = -beta[0] / beta[1]
        intercept = -beta0 / beta[1]
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, 'k-', label="Separating Hyperplane")
    
        # Calculate offset for the margins.
        # In the R code, the margin offset is computed as marg/(cos(pi-theta)).
        # Note: cos(pi - theta) = -cos(theta). We compute it as in R for consistency.
        offset = marg / math.cos(math.pi - theta)
    
        y_vals_margin1 = slope * x_vals + intercept + offset
        y_vals_margin2 = slope * x_vals + intercept - offset
        plt.plot(x_vals, y_vals_margin1, 'k--', label="Margin")
        plt.plot(x_vals, y_vals_margin2, 'k--')
    
        plt.title(f"margin = {marg:.3f}, gamma = {gam}")
        print(marg)
        plt.draw()
        plt.pause(1)  # Wait for user interaction similar to par(ask=TRUE)
    
        # Plot the alpha values
        plt.figure()
        plt.plot(alpha, marker='o', linestyle="None")
        plt.title("Alpha values")
        plt.xlabel("Index")
        plt.ylabel("Alpha")
        plt.draw()
        plt.pause(1)
        plt.show()
    else:
        plt.title("Missing solution")
        plt.show()
        
# End of SVM QP identification simulation
