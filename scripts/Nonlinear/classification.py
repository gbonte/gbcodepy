## # "Statistical foundations of machine learning" software
# classification.py
# Author: G. Bontempi
# 
# Example of classification task with cross-validated assessment of accuracy

import numpy as np
from utils import classpy
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
N = 200
n = 10
X = np.random.normal(size=(N, n))
YY = np.prod(abs(X), axis=1).flatten()+X[:, 0] * X[:, 1]**2 + np.log(abs(X[:, 3]))+ abs(X[:,4])+ X[:,5]*X[:,6]**2+np.sin(2*np.pi*X[:,7])+np.random.normal(scale=0.1, size=N)


Y=YY>np.median(YY)
Y=Y.astype(int)
methods=['torchSoft_class','torchMLP_class','torchBCE_class','torchcross_class','torchlogit_class','rf_class'] #,'nb_class','gb_class', 'lsvm_class','gp_class','piperf_class','knn_class']

Yhat = np.zeros((N,len(methods)))
Phat = np.zeros((N,len(methods)))
print('#1=',sum(Y==1),"#0=",sum(Y==0))

## number folds
K = 5
kf = KFold(n_splits=K, shuffle=False)

cnt=0
for m in methods:
    print(m)
    for train_index, test_index in kf.split(X):
        Xtrk=X[train_index,:]
        Ytrk=Y[train_index].reshape(len(train_index),)
        CL=classpy(m,Xtrk ,Ytrk , X[test_index],params={'nepochs':500, 'hidden':20})
        Yhat[test_index,cnt] = CL[0].ravel()
        Phat[test_index,cnt] = CL[1][:,0]
        
    cnt=cnt+1

for m in np.arange(len(methods)):
    print(methods[m], 'Miscl=',  sum(Yhat[:,m] != Y) / N, 'AUC:',  roc_auc_score(Y, 1-Phat[:,m]))

