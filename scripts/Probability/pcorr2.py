import numpy as np


def partial_corr(X,Y,Z):

    rhoXY=np.corrcoef(X.T,Y.T)[0,1]
    rhoXZ=np.corrcoef(X.T,Z.T)[0,1]
    rhoYZ=np.corrcoef(Y.T,Z.T)[0,1]
       
    partial_corr= (rhoXY-rhoXZ*rhoYZ)/(np.sqrt((1-rhoXZ**2)*(1-rhoYZ**2)))
    
    return partial_corr


N = 2000
n = 10

sigma1=0.35
sigmaw1=sigma1
X1 = np.random.normal(size=(N, 1),scale=sigma1)
W=np.random.normal(size=(N, 1),scale=sigmaw1)
Y=X1+W
sigmaw2=1
W2=np.random.normal(size=(N, 1),scale=sigmaw2)
X2=Y+W2


X=np.hstack((X1,X2))

beta=np.linalg.inv(X.T@X)@X.T@Y

print("beta=",beta[0], ":", 1/(sigma1**2+1))

X1=X1.reshape(N,1)

beta1=np.linalg.inv(X1.T@X1)@X1.T@Y

print("beta1=",beta1)


X2=X2.reshape(N,1)

beta2=np.linalg.inv(X2.T@X2)@X2.T@Y

print("beta2=",beta2,":", sigma1**2/(sigma1**2+1/2))



print("rho_X1Y", np.corrcoef(X1.T,Y.T)[0,1],":", sigma1/np.sqrt(sigma1**2+sigmaw1**2))
print("rho_X1X2", np.corrcoef(X1.T,X2.T)[0,1],":", sigma1/np.sqrt(sigma1**2+sigmaw1**2+sigmaw2**2))
print("rho_X2Y", np.corrcoef(X2.T,Y.T)[0,1],":", np.sqrt(sigma1**2+sigmaw1**2)/np.sqrt(sigma1**2+sigmaw1**2+sigmaw2**2))


print("rho_X1Y_Z=", partial_corr(X1.reshape(N,1),Y.reshape(N,1),X2.reshape(N,1)))
print("rho_X2Y_Z=", partial_corr(X1.reshape(N,1),Y.reshape(N,1),X2.reshape(N,1)))
