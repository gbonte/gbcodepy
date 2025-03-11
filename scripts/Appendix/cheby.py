import numpy as np

# "Statistical foundations of machine learning" software
# R package gbcode
# Author: G. Bontempi

# cheby.R
# Script: shows the Chebyshev relation by means of simulation

# Clear workspace (not applicable in Python, so we proceed without clearing variables)
P_hat = []  # Initialize P.hat as an empty list
Bound = []  # Initialize Bound as an empty list
mu = 0      # Set mu to 0
N = 100     # Set number of samples to 100

# Loop over sig values from 0.1 to 1 (inclusive) in steps of 0.1
for sig in np.arange(0.1, 1.1, 0.1):
    # Loop over d values from 0.1 to 1 (inclusive) in steps of 0.1
    for d in np.arange(0.1, 1.1, 0.1):
        # random sampling of the r.v. z
        z = np.random.normal(loc=mu, scale=sig, size=N)
        # frequency of the event: z-mu >= d
        P_hat.append(np.sum(np.abs(z - mu) >= d) / N)
        # upper bound of the Chebyshev relation
        Bound.append(min(1, (sig**2) / (d**2)))

# Check if the relation is satisfied
if np.any(np.array(P_hat) > np.array(Bound)):
    print("Chebyshev relation NOT satisfied")
else:
    print("Chebyshev relation IS satisfied")
    
if __name__ == '__main__':
    pass
