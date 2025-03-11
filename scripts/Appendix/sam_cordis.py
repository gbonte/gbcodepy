# "Statistical foundations of machine learning" software
# R package gbcode 
# Author: G. Bontempi

# Samping distribution of sample correlation 

# 2 variables x and y are sampled where y=K*x
# The analytically derived correlation rho is compared to the sampling distribution of 
# the estimator rhohat of the correlation 
import numpy as np
import matplotlib.pyplot as plt

R = 10000
N = 50
varX = 0.5
varW = 1
K = 1.33
rhohat = np.empty(R)
for r in range(R):
    Dx = np.random.normal(loc=0, scale=np.sqrt(varX), size=N)  # Nor(0,1)
    Dy = K * Dx + np.random.normal(loc=0, scale=np.sqrt(varW), size=N)  # y=K*x + w
    rhohat[r] = np.corrcoef(Dx, Dy)[0, 1]

# E[xy]= E[Kx^2+w]=K*Var[x]
# Var[x]=VarX
# Var[y]=Var[Kx+w]=K^2 Var[x]+var[w]

rho = K * varX / np.sqrt(varX * (K**2 * varX + varW))

plt.hist(rhohat)
plt.title("Bias=" + str(round(np.mean(rhohat) - rho, 2)))
plt.axvline(x=rho, color="red")
plt.show()
