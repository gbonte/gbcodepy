# "Statistical foundations of machine learning" software
# Python equivalent of gbcode
# Author: G. Bontempi
# mcarlo5.py
# Monte Carlo estimation of E[max(X,Y)]

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
R = 10000  # number of MC trials

# Initialize arrays to store values
theta = []
X = []
Y = []

for r in range(R):
    x = np.random.normal(0, 2)  # rnorm equivalent
    y = np.random.uniform(-1, 1)  # runif equivalent
    
    theta.append(max(x, y))
    X.append(x)
    Y.append(y)

# Convert lists to numpy arrays for calculations
theta = np.array(theta)
X = np.array(X)
Y = np.array(Y)

muX = np.mean(X)
sdX = np.std(X)
sdY = np.std(Y)

print(f"Est= {np.mean(theta)}")

# Create histogram
plt.hist(theta, bins=50)
plt.title('Histogram of theta')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
