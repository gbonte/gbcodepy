import numpy as np

# "INFOF422 Statistical foundations of machine learning" course
# R package gbcode 
# Author: G. Bontempi

# Define the function f that calculates sin(X) plus Gaussian noise.
def f(X, sd=0.15):
    N = len(X)
    return np.sin(X) + np.random.normal(0, sd, size=N)

# Implementation of the pred function for k-nearest neighbor regression.
def pred(method, X, Y, x, classi=False, k=3):
    # Only implement the "knn" method.
    if method != "knn":
        raise ValueError("Only 'knn' method is supported")
    # Convert X and Y to numpy arrays if they are not already.
    X = np.array(X)
    Y = np.array(Y)
    # Compute distances between the test point x and all points in X.
    distances = np.abs(X - x)
    # Find the indices of the k smallest distances.
    knn_indices = np.argsort(distances)[:k]
    # Calculate the predicted value as the mean of the k nearest neighbors' responses.
    return np.mean(Y[knn_indices])

N = 20  # size training set
Nts = 10000  # size test set

sdw = 0.2  # stdev noise
X = np.random.normal(0, 1, size=N)
B = 500.  # number bootstrap sets
Y = f(X, sd=sdw)

K = 3  # number of nearest neighbors

Xts = np.random.normal(0, 1, size=Nts)
Yts = f(Xts, sd=sdw)

Ets = []
for i in range(Nts):
    Yhati = pred("knn", X, Y, Xts[i], classi=False, k=K)
    Ets.append(Yts[i] - Yhati)
Ets = np.array(Ets)
MSEts = np.mean(Ets**2)

Eemp = []
for i in range(N):
    Yhati = pred("knn", X, Y, X[i], classi=False, k=K)
    Eemp.append(Y[i] - Yhati)
Eemp = np.array(Eemp)
MSEemp = np.mean(Eemp**2)

Biasb = []
for b in range(int(B)):
    Ib = np.random.choice(np.arange(N), size=N, replace=True)
    Xb = X[Ib]
    Yb = Y[Ib]
    Eb = []
    for i in range(N):
        Ydoti = pred("knn", Xb, Yb, Xb[i], classi=False, k=K)
        Yhati = pred("knn", Xb, Yb, X[i], classi=False, k=K)
        Eb.append((Yb[i] - Ydoti)**2 - (Y[i] - Yhati)**2)
    Biasb.append(np.mean(Eb))
Biasb = np.array(Biasb)
Biasboot = np.mean(Biasb)

print("\n MSEts=", MSEts, "\n MSEemp=", MSEemp, "\n Bias B=", Biasboot, "\n BiasCorrected MSE=", MSEemp - Biasboot)
