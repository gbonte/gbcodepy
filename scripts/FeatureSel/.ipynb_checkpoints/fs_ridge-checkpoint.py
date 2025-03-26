# "INFOF422 Statistical foundations of machine learning" course
# package gbcodepy
# Author: G. Bontempi

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)  # set.seed(0)
N = 50

# correlated inputs
X1 = np.random.randn(N)  
X2 = np.random.randn(N)  
X3 = np.random.randn(N)  
X4 = np.random.randn(N)  

W = np.random.randn(N) * 0.5  # rnorm(N,sd=0.25)

## number of irrilevant features
irril = 50
# Create a matrix by column-binding X1, X2, X3, X4 and an array of rnorm(N*irril) reshaped to (N, irril)
X_combined = np.column_stack((X1, X2, X3, X4, np.random.randn(N, irril)))
X = (X_combined - np.mean(X_combined, axis=0)) / np.std(X_combined, axis=0)


# Compute Y as scale(X1^2+2*X2+X3+X4) + W
Y_raw = X1 + 2*X2 + X3 + X4
Y_scaled = (Y_raw - np.mean(Y_raw)) / np.std(Y_raw)
Y = Y_scaled + W

n = X.shape[1]  

LAM = np.arange(20,50,1) 
E = np.empty((N, len(LAM)))  
E[:] = np.nan

## LEAVE-ONE-OUT loop
for i in range(N):
    Xtr = np.column_stack((np.ones(N - 1), np.delete(X, i, axis=0)))
    Ytr = np.delete(Y, i)  # Y[-i]
    
    # Xts: c(1, X[i,])
    Xts = np.concatenate(([1], X[i,]))
    
    cnt = 0
    for lam in LAM:
        # betahat= solve(t(Xtr)%*%Xtr + lam*diag(n+1)) %*% t(Xtr)%*%Ytr
        A = Xtr.T @ Xtr + lam * np.eye(n + 1)
        b = Xtr.T@ Ytr
        betahat = np.linalg.inv(A) @ b
        # Yhati = Xts %*% betahat
        Yhati = Xts@ betahat
        E[i, cnt] = (Y[i] - Yhati) ** 2
        cnt += 1

# cat("MSEloo =", apply(E,2,mean), "\n")
mseloo = np.mean(E, axis=0)


# lambest = LAM[which.min(apply(E,2,mean))]
lambest = LAM[np.argmin(np.mean(E, axis=0))]
print("best lambda=",lambest)

XX = np.column_stack((np.ones(N), X))
A_final = np.dot(XX.T, XX) + lambest * np.eye(n + 1)
b_final = np.dot(XX.T, Y)
betahat = np.linalg.inv(A_final)@ XX.T@Y

# print(sort(abs(betahat),decr=TRUE,index=TRUE)$ix[1:4]-1)
abs_betahat = np.abs(betahat)
# argsort in descending order
sorted_indices = np.argsort(-abs_betahat)
# Select the top 4 indices and subtract 1 (to mimic the R code behavior)
top4 = sorted_indices[:4] 
print(top4)
plt.plot(LAM,mseloo)
plt.title('MSE.loo vs lambda')
