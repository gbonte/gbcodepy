# "INFOF422 Statistical foundations of machine learning" course
# gen_class.py
# Author: G. Bontempi

## Code to generate a training dataset for a binary (0/1) classification problem 
## where the two clas conditional distributions are normal


## We generate the dataset for the same conditional distribution in two different ways

## 1: generative, i.e. we first generate the class yi (using the a priori probability) 
## and then the input xi based on the class conditional distributions p(x |y=yi)

## 2: discriminative: we compute the conditional distribution P(y|x) by using the Bayes theorem 
##     and the distribution p(x) which is a mixture of two gaussians p(x)= p1*p(x|y=1)+ (1-p1)*p(x|y=0)
## Then we sample xi according to p(x) and then we compute yi acording to P(y|x)


import numpy as np
import matplotlib.pyplot as plt
N=200
n=2

p1=0.4 ## a priori probability of class 1 (green)


## 1 generative case: class conditional distributions are given
## they are both multivariate normal with a given mean and covariance
np.random.seed(7)

A = np.random.rand(n, n)
cov1 = A.T @ A
mu1=np.random.normal(size=n)

A = np.random.rand(n, n)
cov0 = A.T @ A
mu0=np.random.normal(loc=2, size=n)
        
Y = np.random.choice([0, 1], size=N, p=[1.0-p1,p1])
X=np.zeros((N,n))
for i in np.arange(N):
    if Y[i]==1:
        X[i,:]=np.random.multivariate_normal(mu1, cov1, size=1)
    else:
        X[i,:]=np.random.multivariate_normal(mu0, cov0, size=1)
      
I0=np.where(Y==0)
plt.scatter(X[I0,0],X[I0,1],color="red")
plt.title("Generative")
I1=np.where(Y==1)
plt.scatter(X[I1,0],X[I1,1],color="green")
plt.show()


## 2 discriminative case 
## conditional distributions P(y|x) is given 
## 

from scipy.stats import multivariate_normal

## returns the conditional distribution P(y|x) using the Bayes theorem
def condprob(x,cov1,mu1,cov0,mu0,p1):
    return multivariate_normal.pdf(x, mean=mu1, cov=cov1)*p1/  \
        (multivariate_normal.pdf(x, mean=mu1, cov=cov1)*p1+    \
         multivariate_normal.pdf(x, mean=mu0, cov=cov0)*(1-p1))
    

X=np.zeros((N,n))

## Sample N xi according to the density
## p(x)= p1*p(x|y=1)+ (1-p1)*p(x|y=0)

for i in np.arange(N):
    if np.random.uniform()<p1:
        X[i,:]=np.random.multivariate_normal(mu1, cov1, size=1)
    else:
       X[i,:]=np.random.multivariate_normal(mu0, cov0, size=1)     

Y=np.zeros((N,1))
for i in np.arange(N):
    ## for each xi it returns Yi using P(y|x) 
    p1i=condprob(X[i,:], cov1,mu1,cov0,mu0,p1)
    Y[i]=np.random.choice([0, 1], size=1, p=[1.0-p1i,p1i])
      
I0=np.where(Y==0)
plt.scatter(X[I0,0],X[I0,1],color="red")   
plt.title("Discriminative")
I1=np.where(Y==1)
plt.scatter(X[I1,0],X[I1,1],color="green")
plt.show()
