import numpy as np

np.random.seed(0)
R = 10000
# number of MC trials 

def f(x):
    return -x**2

FX = []
X = []

a = 1
b = 1

for r in range(R):
    x = np.random.normal(a, b)
    fx = f(x)
    
    FX.append(fx)
    X.append(x)

muX = np.mean(X)
print(f"E[f(x)]= {np.mean(FX)}")
print(f"f(E[x])= {f(muX)}")

