## "Statistical foundations of machine learning" software
## package gbcodpy
## Author: G. Bontempi

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Set up the plot
plt.figure(figsize=(10, 6))

# Set random seed
np.random.seed(0)

N = 10
DN = np.random.normal(0, 1, N)

Theta = np.arange(-2, 2, 0.001)
L = np.ones(len(Theta))

for ith in range(len(Theta)):
    for i in range(N):
        L[ith] *= norm.pdf(DN[i], Theta[ith], 1)

plt.plot(Theta, L)
plt.axvline(x=Theta[np.argmax(L)], color='r', linestyle='--')

plt.xlabel('Theta')
plt.ylabel('Likelihood')
plt.title('Maximum Likelihood Estimation')

plt.show()

print(f"Sample average = {np.mean(DN):.6f}, arg max L = {Theta[np.argmax(L)]:.6f}")

