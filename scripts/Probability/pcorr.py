# "INFOF422 Statistical foundations of machine learning" course
# R package gbcode 
# Author: G. Bontempi
## pcorr.py
## 2 independent and non correlated random variables Z1 and Z2 which are partially dependent (and correlated)
## Y= Z1+Z2
import numpy as np


def partial_corr(X,Y,Z):

    rhoXY=np.corrcoef(X.T,Y.T)[0,1]
    rhoXZ=np.corrcoef(X.T,Z.T)[0,1]
    rhoYZ=np.corrcoef(Y.T,Z.T)[0,1]
       
    partial_corr= (rhoXY-rhoXZ*rhoYZ)/(np.sqrt((1-rhoXZ**2)*(1-rhoYZ**2)))
    
    return partial_corr


N = 5000



Z1 = np.random.normal(size=(N, 1))
Z2 = np.random.normal(size=(N, 1))

Y=Z1+Z2



print("\n correlation rho_Z1Z2=", np.round(np.corrcoef(Z1.T,Z2.T)[0,1],2))


print("partial correlation rho_Z1Z2_Y=", 
      np.round(partial_corr(Z1.reshape(N,1),Z2.reshape(N,1),Y.reshape(N,1)),2))
