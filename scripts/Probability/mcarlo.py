import numpy as np


# Monte Carlo estimation of parameters

R = 100000  # number of MC trials

Ez = 0.5
Vz = 1.7
z = np.random.normal(Ez, np.sqrt(Vz), R)
y = z**2

print(f"E[z]={Ez}; MC est E[z] ={np.mean(z)}")
# MonteCarlo estimation of E[z]

print(f"Var[z]={Vz}; MC est Var[z] ={np.var(z)}")
# MonteCarlo estimation of Var[z]

print(f"Var[z^2]={Vz+Ez**2}; MC est E[y=z^2] ={np.mean(y)}")
##Var[z]=E[z^2]-E[z]^2 -> E[z^2]=Var[z]+E[z]^2=Vz+Ez^2

#############################################
Ez = 0
Vz = 1
z = np.random.normal(Ez, np.sqrt(Vz), R)
k = np.abs(z)

Ek = np.sqrt(2/np.pi)
Vk = 1 - 2/np.pi
## https://www.quora.com/If-Y-X-where-X-has-normal-distribution-N-0-1-what-is-the-density-function-expectation-and-variance-of-Y

print(f"E[k]={Ek}; MC estimate E[k=|z|] ={np.mean(k)}")
# MonteCarlo estimation of E[k]

print(f"V[k]={Vk}; MC estimate V[k=|z|] ={np.var(k)}")
# MonteCarlo estimation of E[k]

