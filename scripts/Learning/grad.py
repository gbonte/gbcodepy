# "INFOF422 Statistical foundations of machine learning" course
# Python package gbcodepy
# Author: G. Bontempi
# grad.py

import numpy as np
import matplotlib.pyplot as plt



np.random.seed(0)
N = 100

# Generate data
X = np.random.randn(N)
Y = X**3 + np.random.normal(0, 0.1, N)

# Least squares solution
# Create the design matrix with a column of ones and X as the second column
XX = np.column_stack((np.ones(N), X))
# Compute the LS solution: alphaLS = (XX^T * XX)^{-1} * XX^T * Y
alphaLS = np.linalg.solve(XX.T @ XX, XX.T @ Y)

# Predictions
Yhat = alphaLS[0] + alphaLS[1] * X
e = Y - Yhat
print("min LS error=", np.mean(e**2))

# Plot the data and the LS fit
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, label="Data")
# To have a smooth line, sort by X
sort_idx = np.argsort(X)
plt.plot(X[sort_idx], Yhat[sort_idx], linewidth=5, label="LS fit")

print("\n --- \n")

# Random search
I = 1000
bestJ = np.inf
for i in range(1, I + 1):
    alpha = np.random.randn(2) * 2
    Yhat = alpha[0] + alpha[1] * X
    e = Y - Yhat
    mse = np.mean(e**2)
    if mse < bestJ:
        bestalpha = alpha.copy()
        bestJ = mse
        print("random search: error at iteration", i, "=", mse)

print("\n --- \n")

# Gradient based search
I_grad = 50
alpha_grad = np.zeros(2)
mu = 0.001
# For visualization, we will plot the gradient-based iterations on the same figure in green.
for i in range(1, I_grad + 1):
    Yhat = alpha_grad[0] + alpha_grad[1] * X
    # Plot the current model prediction; sort for a smooth line
    plt.plot(X[sort_idx], (alpha_grad[0] + alpha_grad[1]*X)[sort_idx], color="green", alpha=0.5)
    e = Y - Yhat
    # Update the parameters using gradient descent
    alpha_grad[0] = alpha_grad[0] + mu * 2 * np.sum(e)
    alpha_grad[1] = alpha_grad[1] + mu * 2 * np.sum(e * X)
    if i % 10 == 0:
        print("gradient-based: error at iteration", i, "=", np.mean(e**2))

print("\n --- \n")

# Levenberg Marquardt
I_lm = 3
alpha_lm = np.zeros(2)
# For visualization, we will plot the LM iterations on the same figure in red.
for i in range(1, I_lm + 1):
    Yhat = alpha_lm[0] + alpha_lm[1] * X
    plt.plot(X[sort_idx], (alpha_lm[0] + alpha_lm[1]*X)[sort_idx], color="red", alpha=0.8)
    e = Y - Yhat
    print("Levenberg Marquardt: error at iteration", i, "=", np.mean(e**2))
    # E matrix: each row is [-2, -2 * X_i]
    E = -np.column_stack((np.full(N, 2), 2 * X))
    bestJ = np.inf
    bestlambda = None
    # Try different lambda values
    for lam in np.arange(0.01, 10.01, 0.01):
        # Compute the update using the LM formula
        # The formula: alphat = alpha - inv(E^T E + lambda*I) @ (2*E^T) @ e
        try:
            update_matrix = np.linalg.solve(E.T @ E + lam * np.eye(2), (2 * E.T) @ e)
        except np.linalg.LinAlgError:
            continue  # Skip if matrix is singular
        alphat = alpha_lm - update_matrix
        Yhatt = alphat[0] + alphat[1] * X
        et = Y - Yhatt
        mse_t = np.mean(et**2)
        if mse_t < bestJ:
            bestlambda = lam
            bestJ = mse_t
    # Update alpha using the best lambda found
    try:
        update_matrix = np.linalg.solve(E.T @ E + bestlambda * np.eye(2), (2 * E.T) @ e)
    except np.linalg.LinAlgError:
        update_matrix = np.zeros_like(alpha_lm)
    alpha_lm = alpha_lm - update_matrix
    

plt.legend()
plt.title("Model Fits: LS (blue thick), Gradient-based (green), LM (red)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
