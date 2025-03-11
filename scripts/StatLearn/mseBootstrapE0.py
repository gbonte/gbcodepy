import numpy as np

# "INFOF422 Statistical foundations of machine learning" course
# R package gbcode 
# Author: G. Bontempi

def f(X, sd=0.15):
    N = len(X)
    return np.sin(X) + np.random.normal(0, sd, size=N)

def knnpred(X, Y, x,  k=1):
   
    # Compute distances between x and each element in X
    distances = np.abs(X - x)
    # Get indices of the k smallest distances
    indices = np.argsort(distances)[:k]
    # For regression, return the mean of the k nearest neighbors
    return np.mean(Y[indices])
   

N = 20  # size training set
Nts = 10000  # size test set

sdw = 0.2  # stdev noise
X = np.random.normal(0, 1, N)
B = int(500)  # number bootstrap sets
Y = f(X, sd=sdw)

K = 1  # number of nearest neighbors

Xts = np.random.normal(0, 1, Nts)
Yts = f(Xts, sd=sdw)

### generalisation error computed with a large test set
Ets = []
for i in range(Nts):
    Yhati = knnpred( X, Y, Xts[i],  k=K)
    Ets.append(Yts[i] - Yhati)
Ets = np.array(Ets)
MSEts = np.mean(Ets**2)

### empirical error 
Eemp = []
for i in range(N):
    Yhati = knnpred(X, Y, X[i], k=K)
    Eemp.append(Y[i] - Yhati)
Eemp = np.array(Eemp)
MSEemp = np.mean(Eemp**2)

### E0 bootstrap assessment of the generalisation error 

Bi = np.zeros(N)  
# vector containing the number of bootstrap sets not containing the ith point of the training set

E0 = np.empty((N, B))
E0[:] = np.nan  # initialize with NA equivalent

Biasb = None
for b in range(B):
    Ib = np.random.choice(np.arange(N), size=N, replace=True)
    Xb = X[Ib]
    Yb = Y[Ib]
    
    for i in range(N):
        if not np.any(Xb == X[i]):
            Bi[i] = Bi[i] + 1  # increment of B[i] if the ith point is not in the bth bootstrap set
        Yhati = knnpred( Xb, Yb, X[i], k=K)
        E0[i, b] = (Y[i] - Yhati)**2

MSEbootE0 = 0
for i in range(N):
    if Bi[i] > 0:
        MSEbootE0 = MSEbootE0 + np.sum(E0[i, :]) / Bi[i]

MSEbootE0 = MSEbootE0 / N

print("\n MSEts=", MSEts, "\n MSEemp=", MSEemp, 
      "\n MSEbootE0=", MSEbootE0, "\n MSEboot.632=", 0.632 * MSEbootE0 + 0.328 * MSEemp)
