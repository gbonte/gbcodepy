## # "Statistical foundations of machine learning" software
# Author: G. Bontempi
# genercov.py
# Monte Carlo estimation of functional risk



import numpy as np
import matplotlib.pyplot as plt

# 
sdw = 1

Nts = 1000000
Xts = np.random.uniform(-2, 2, Nts)
Yts = Xts**3 + np.random.normal(0, sdw, Nts)

R = []
bestR = np.inf

# np.arange may not include the endpoint; add a small epsilon to ensure inclusion of 3.5
A = np.arange(1, 3.5 + 0.01, 0.01)

for alpha in A:
    Ra = np.mean((Yts - alpha * Xts)**2)
    R.append(Ra)
    if Ra < bestR:
        bestR = Ra
        bestalpha = alpha
    # Print a dot without new line to track progress
    print(".", end="", flush=True)

print()  # Move to the next line after progress dots

# Plot the computed risk curve
plt.plot(A, R, label="Monte Carlo R(alpha)")

# Compute the theoretical risk function: 1/4*(16*A^2/3 - 128*A/5 + 256/7) + 1
# Note: The operations are vectorized over A.
theoretical_R = 1/4 * (16 * A**2 / 3 - 128 * A / 5 + 256 / 7) + 1
plt.plot(A, theoretical_R, color="red", label="Theoretical R(alpha)")

plt.xlabel("alpha")
plt.ylabel("Risk")
plt.legend()
plt.show()

print("R(alpha)=", bestR, "best alpha=", bestalpha)
