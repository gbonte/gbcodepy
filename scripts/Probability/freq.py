## "Statistical foundations of machine learning" software
## Python gbcodepy 
## Author: G. Bontempi

##  Fair coin tossing random experiment
## Evolution of the relative frequency (left) 
## and of the absolute difference between the number of heads and tails 

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(2)

R = 1000000

# Generate tosses
tosses = np.random.choice(["H", "T"], size=R)

gap = []
freq = []
trials = []

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for r in range(1, R+1, 5000):
    lH = np.sum(tosses[:r] == "H")
    lT = np.sum(tosses[:r] == "T")
    gap.append(abs(lH - lT))
    freq.append(lH / r)
    trials.append(r)
    
    print(".", end="", flush=True)

print("\n")

# Plot relative frequency
ax1.plot(trials, freq)
ax1.plot(trials, [0.5] * len(freq))
ax1.set_ylim(0.2, 0.6)
ax1.set_xlabel("Number of trials")
ax1.set_ylabel("Relative frequency")

# Plot absolute difference
ax2.plot(trials, gap)
ax2.plot(trials, [0] * len(gap))
ax2.set_xlabel("Number of trials")
ax2.set_ylabel("Absolute difference (no. heads and tails)")

plt.tight_layout()
plt.show()

