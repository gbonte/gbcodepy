import numpy as np
from scipy.stats import mode

## KNN classifier: 
## X: training input
## Y: training output
## q: query
## k: number of nearest neighbours

def KNN(X, Y, q, k):
    l = np.unique(Y)
    N = X.shape[0]
    
    # Euclidean metric
    d = np.sqrt(np.sum((X - np.tile(q, (N, 1)))**2, axis=1))
    # Manhattan metric
    
    index = np.argsort(d)
    cnt = np.zeros(len(l))
    for i in range(k):
        cnt[Y[index[i]]] += 1
    
    return l[np.argmax(cnt)]

N = 100
n = 5
X = np.random.randn(N, n)
Y = (X[:, 0] > 0).astype(int)
e = 0

for i in range(N):
    X_train = np.delete(X, i, axis=0)
    Y_train = np.delete(Y, i)
    if Y[i] != KNN(X_train, Y_train, X[i], 5):
        e += 1

print(f"Misclassification error = {e/N}")