import sys
import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
#from sklearn.utils.testing import ignore_warnings
#from sklearn.exceptions import ConvergenceWarning
import warnings
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    
    
def f(X, sd=0):
    y = X[:, 0] * X[:, 1] + X[:, 1] + sd * np.random.randn(X.shape[0])
    return y

random.seed(0)
sdw = 0.25
N = 50
n=10
X = np.random.randn(N, n) 
Y = f(X, sdw)

S = np.arange(5,200,20)

MISEhat = []
df = np.column_stack((Y, X))

for s in S:
    ## structural identification loop
    Eloo = []
    for j in range(N):
        ## leave-one-out loop
        Tr_i = np.delete(df, j, axis=0)
        Ts_i = df[j, :].reshape(1, -1)

        ## parametric identification
        h = MLPRegressor(hidden_layer_sizes=(s,),activation='logistic',max_iter=500)
        h.fit(Tr_i[:, 1:], Tr_i[:, 0])

        Eloo.append(Y[j] - h.predict(Ts_i[:, 1:].reshape(1, -1)))

    print('# hidden nodes=',s,'MSE_LOO=',np.mean(np.square(Eloo))/np.var(Y))
    MISEhat.append(np.mean(np.square(Eloo)))

stilde = S[np.argmin(MISEhat)]
h = MLPRegressor(hidden_layer_sizes=(stilde,),activation='logistic',max_iter=500)
h.fit(df[:, 1:], df[:, 0])

Nts=500
Xts = np.random.randn(Nts, n) 
Yts = f(Xts, sdw)
dfts = np.column_stack((Yts, Xts))

Yhat=h.predict(dfts[:, 1:]).reshape(-1, 1)
Yts=Yts.reshape(-1, 1)
print('stilde=', stilde, 'MSE_test=', np.mean((Yts-Yhat)**2)/np.var(Yts))

