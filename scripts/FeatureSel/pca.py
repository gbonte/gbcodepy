# ## "INFOF422 Statistical foundations of machine learning" course
# ## Python package gbcodepy
# ## Author: G. Bontempi
# ## script pca.py


import numpy as np
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

np.random.seed(0)

N = 500

X1 = np.random.normal(loc=0, scale=2, size=N)

X2 = np.random.normal(loc=0, scale=1, size=1)[0] * X1 + np.random.normal(loc=0, scale=3, size=N)

Xtilde = scale(np.column_stack((X1, X2)))

U, s, Vt = np.linalg.svd(Xtilde, full_matrices=False)
V1 = Vt[0, :].reshape(2, 1)

Z = np.dot(Xtilde, V1)
Xtilde2 = np.dot(Z, V1.T)

plt.figure()
plt.scatter(Xtilde[:, 0], Xtilde[:, 1])
plt.xlabel("x1")
plt.ylabel("x2")

plt.plot(X1, (V1[1, 0] / V1[0, 0]) * X1, color="red", linewidth=2)
plt.scatter(Xtilde2[:, 0], Xtilde2[:, 1], color="red")

RecE = Xtilde - Xtilde2
print("Reconstruction error=", np.mean(np.sum(RecE**2, axis=1)), ":", (s[1]**2) / N)

plt.show()
