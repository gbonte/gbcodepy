# ## "INFOF422 Statistical foundations of machine learning" course
# ## Python package gbcodepy
# ## Author: G. Bontempi
#
# ## script pca_uns.py



import numpy as np
import matplotlib.pyplot as plt


N = 500

w = np.random.normal(loc=0, scale=3, size=N)
Ig = np.where(w < 0)[0]
Ir = np.where(w > 0)[0]
X1 = np.random.normal(loc=0, scale=2, size=N)

X2 = X1 + w

# Combine X1 and X2 into a two-column array and scale each column (center and scale using sample standard deviation)
X = np.column_stack((X1, X2))
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0, ddof=1)
Xtilde = (X - mean_X) / std_X

# Compute singular value decomposition
U, s, Vt = np.linalg.svd(Xtilde, full_matrices=False)
V = Vt.T
V1 = V[:, 0].reshape(2, 1)

Z = Xtilde.dot(V1)

Dc = Z.dot(V1.T)

plt.figure()
# Plot points corresponding to Ig with unfilled circles in green
plt.scatter(Xtilde[Ig, 0], Xtilde[Ig, 1], facecolors='none', edgecolors='green', s=20, marker='o')
# Plot points corresponding to Ir with x markers in red
plt.scatter(Xtilde[Ir, 0], Xtilde[Ir, 1], color='red', s=20, marker='x')
# Plot the line corresponding to V1; adjust index from 1-indexed (R) to 0-indexed (Python)
slope = V1[1, 0] / V1[0, 0]
plt.plot(X1, slope * X1, color='black', linewidth=2)
# Plot Dc for Ig with a slight vertical shift (-0.05) in green using filled circles
plt.scatter(Dc[Ig, 0], Dc[Ig, 1] - 0.05, color='green', s=20, marker='o')
# Plot Dc for Ir with a slight vertical shift (+0.05) in red using filled circles
plt.scatter(Dc[Ir, 0], Dc[Ir, 1] + 0.05, color='red', s=20, marker='o')
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
