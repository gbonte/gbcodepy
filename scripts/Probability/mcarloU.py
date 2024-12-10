import numpy as np

R = 50000  # number of MC trials

a = -1.0
b = 1.0
muz = (b + a) / 2

Z = np.zeros(R)
for r in range(R):
    z = np.random.uniform(a, b)
    Z[r] = (z - muz)**2

print(f"Var. th= {(b-a)**2/12:.6f}, MC approximation= {np.mean(Z):.6f}")

