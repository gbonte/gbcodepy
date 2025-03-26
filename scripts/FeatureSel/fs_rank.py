import numpy as np
from sklearn.preprocessing import scale


def cor(X,Y):
    if X.shape[0] !=Y.shape[0]:
        raise Exception('error')
    if Y.shape[1] !=1:
        raise Exception('only for univariate targets')
    XY=np.hstack((X,Y.reshape(-1,1)))
    score=np.corrcoef(XY.T)[-1,:-1]
    return (score)

def rankrho(X, Y, nmax=5, regr=False):
    score=np.abs(cor(X,Y))
    return(np.argsort(-score)[:nmax])


N=100
n=10
sdw=0.1
relfeat=[0,1,2]
X= np.random.normal(loc=0, scale=1, size=N * n).reshape(N, n)
Y=np.random.normal(loc=0, scale=sdw, size=N )
X = scale(X)
for f in relfeat:
    Y=Y+X[:,f]**2
Y=Y.reshape(N, 1)

rankedfs=rankrho(X,Y,len(relfeat))
print('Relevant features=',relfeat, 'Ranked features=',rankedfs)