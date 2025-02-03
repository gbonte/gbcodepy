## # "Statistical foundations of machine learning" software
# regression.py
# Author: G. Bontempi
# mcarlo4.py
# Example of regression task with cross-validated assessment of accuracy

import numpy as np
from utils import predpy
from sklearn.model_selection import KFold

N = 2000
n = 10
X = np.random.normal(size=(N, n))
Y = X[:, 0] * X[:, 1]**2 + np.log(abs(X[:, 3]))+ abs(X[:,4])+ X[:,5]*X[:,6]**2+np.sin(2*np.pi*X[:,7])+np.random.normal(scale=0.1, size=N)


methods=['rf_regr','torch_regr','lazy_regr','gb_regr','ridge_regr','lasso_regr','enet_regr','piperf_regr','pipeknn_regr']

Yhat = np.zeros((N,len(methods)))



## number folds
K = 5
kf = KFold(n_splits=K, shuffle=False)

cnt=0
for m in methods:
    print(m)
    for train_index, test_index in kf.split(X):
        Xtrk=X[train_index,:]
        Ytrk=Y[train_index].reshape(len(train_index),)
        Yhat[test_index,cnt] = predpy(m,Xtrk ,Ytrk , X[test_index],params={'nepochs':500, 'hidden':20})
        
    cnt=cnt+1

for m in np.arange(len(methods)):
    print(methods[m], 'NMSE=',  np.mean((Yhat[:,m] - Y)**2) / np.var(Y))

