# "INFOF422 Statistical foundations of machine learning" course
# R package gbcode 
# Author: G. Bontempi
# discri2.py
# This script shows the limitation of a linear discriminant in the case of multimodal class conditional distributions

import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

def norm(x):
    return np.sqrt(np.sum(x**2))

N = 500
p1 = 0.4
w1 = 0.5
w2 = 0.5

mu11 = np.array([1, 1])
mu12 = np.array([-2.1, -1.3])

Sigma1 = 0.5 * np.eye(2)
N1 = int(round(p1 * N))

samples1 = mvn.rvs(mean=mu11, cov=Sigma1, size=int(round(w1 * N1)))
samples2 = mvn.rvs(mean=mu12, cov=Sigma1, size=int(round(w2 * N1)))
X = np.vstack((samples1, samples2))

N0 = N - N1
mu0 = np.array([0, 0])
Sigma0 = 0.5 * np.eye(2)
X0 = mvn.rvs(mean=mu0, cov=Sigma0, size=N0)
X = np.vstack((X, X0))

Y = np.zeros(N)
Y[:N1] = 1

Phat1 = N1 / N
Phat0 = N0 / N

fig, ax = plt.subplots()
ax.scatter(X[:N1, 0], X[:N1, 1], color="green", label="Class 1", s=8)
ax.scatter(X[N1:, 0], X[N1:, 1], color="red", label="Class 0",s=8)
ax.set_xlabel("x1")
ax.set_ylabel("x2")

muhat1 = X[:N1].mean(axis=0)
muhat0 = X[N1:].mean(axis=0)
Xd = np.arange(-10, 11)
sigma2 = np.mean(np.var(X, axis=0))

w = muhat1 - muhat0
x0 = 0.5 * (muhat1 + muhat0) - (sigma2 / (norm(w)**2)) * w * np.log(Phat1 / Phat0)

m = -w[0] / w[1]
intc = (w[0] / w[1]) * x0[0] + x0[1]

# ax.plot(Xd, m * Xd + intc)
ax.plot([-10, 10], [m * -10 + intc, m * 10 + intc], label="Decision Boundary")
# plt.plot([muhat1[0], muhat2[0]], [muhat1[1], muhat2[1]])
ax.scatter(x0[0], x0[1], color="blue", label="x0")
plt.ylim([-4,4])
plt.xlim([-5,5])
ax.legend()
plt.show()
