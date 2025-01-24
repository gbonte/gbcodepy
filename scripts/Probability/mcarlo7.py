# "Statistical foundations of machine learning" software
# Python equivalent of gbcode R package
# Author: G. Bontempi
# mcarlo7.py
# Monte Carlo estimation of parameters

import numpy as np

R = 50000  # number of MC trials
a = -1
b = 1
muz = (b + a) / 2

# Generate all random numbers at once using numpy
z = np.random.uniform(a, b, R)
Z2 = (z - muz) ** 2

print(f"Var. theory= {(b-a)**2/12:.6f} MC approximation= {np.mean(Z2):.6f}")
