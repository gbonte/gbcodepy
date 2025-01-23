import numpy as np
import matplotlib.pyplot as plt

def norm(x):
    return np.sqrt(np.sum(x**2))

N1 = 100
N2 = 100
P = np.array([N1, N2]) / (N1 + N2)
sigma2 = 1

mu_1 = np.array([-1, -2])
mu_2 = np.array([2, 5])

X1 = np.column_stack((np.random.normal(mu_1[0], np.sqrt(sigma2), N1),
                      np.random.normal(mu_1[1], np.sqrt(sigma2), N1)))

X2 = np.column_stack((np.random.normal(mu_2[0], np.sqrt(sigma2), N2),
                      np.random.normal(mu_2[1], np.sqrt(sigma2), N2)))

plt.figure(figsize=(10, 10))
plt.scatter(X1[:, 0], X1[:, 1], color='red', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='green', label='Class 2')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel('x1')
plt.ylabel('x2')

Xd = np.arange(-10, 11)

w = mu_1 - mu_2
x0 = 0.5 * (mu_1 + mu_2) - sigma2 / (norm(mu_1 - mu_2)**2) * (mu_1 - mu_2) * np.log(P[0] / P[1])

m = -w[0] / w[1]
intc = w[0] / w[1] * x0[0] + x0[1]

plt.axline(xy1=(0, intc), slope=m, color='blue', label='Decision Boundary')
plt.scatter(x0[0], x0[1], color='black', label='x0')

plt.legend()
plt.show()

