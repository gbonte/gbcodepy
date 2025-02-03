# ## "INFOF422 Statistical foundations of machine learning" course
# ## Python package gbcodepy
# ## Author: G. Bontempi
#
# ## script featrans.py

import numpy as np
import matplotlib.pyplot as plt

N = 1000
X1 = np.random.uniform(-1, 1, N)
X2 = np.random.uniform(-1, 1, N)
r = 0.5
Y = np.zeros(N)
I1 = np.where(X1**2 + X2**2 > r)[0]
I0 = np.setdiff1d(np.arange(N), I1)

plt.figure()
plt.scatter(X1[I1], X2[I1], color="green")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Original space")
plt.scatter(X1[I0], X2[I0], color="red")
plt.show()

X3 = X1**2 + X2**2

plt.figure()
plt.scatter(X1[I1], X3[I1], color="green")
plt.xlabel("x1")
plt.ylabel("x3")
plt.title("Transformed space")
plt.ylim(-0.1, 1)
plt.scatter(X1[I0], X3[I0], color="red")
plt.show()
