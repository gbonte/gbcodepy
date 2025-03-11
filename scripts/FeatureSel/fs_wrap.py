import numpy as np
from sklearn.preprocessing import scale
import pyreadr


def rankrho(X, Y, nmax=5, regr=False):
    XY=np.hstack((X,Y.reshape(-1,1)))
    score=np.abs(np.corrcoef(XY.T)[-1,:-1])
    
    return(np.argsort(-score)[:nmax])

def KNN(X, Y, k, q):
    # Get levels of Y (unique classes)
    
    N = X.shape[0]
    
    # Euclidean metric
    d = np.sqrt(np.sum((X - q) ** 2, axis=1))
    ## d = np.sqrt(np.sum(np.abs(X - q), axis=1)) ## Manhattan metric
    ## d = 1/np.abs(np.corrcoef(X.T, q.T)[0:-1, -1])           ## correlation metric

    # Get sorted indices based on distance
    sorted_idx = np.argsort(d,axis=0)
    cnt = np.zeros(2)
    for i in range(k):
        yi=Y[sorted_idx[i]]
        
        cnt[yi] += 1
    # Return the level with maximum count (if tie, the first encountered)
    return np.argmax(cnt)

def KNN_wrap(X, Y, size, K=1):
    ## leave-one-out wrapper based on forward selection and KNN
    n = X.shape[1]
    N = X.shape[0]
    selected = []
    while len(selected) < size:
        miscl_tt = np.full(n, np.inf)
        for j in range(n):
            if j not in selected:
                select_temp = selected + [j]
                miscl = np.zeros(N)
                for i in range(N):
                    # Leave-one-out partition: remove i-th sample
                    X_tr = np.delete(X, i, axis=0)[:, select_temp]
                    Y_tr = np.delete(Y, i, axis=0)
                    q = X[i, select_temp].reshape(1, -1)
                    Y_ts = Y[i]
                    
                    Y_hat_ts = KNN(X_tr, Y_tr, K, q[0])
                    miscl[i] = 1 if (Y_hat_ts != Y_ts) else 0
                miscl_tt[j] = np.mean(miscl)
        selected.append(int(np.argmin(miscl_tt)))
        print(".", end="")
    print("\n")
    return selected

# Set seed
np.random.seed(0)

K = 3 ## number of neighbours in KNN
# Load dataset "golub.Rdata"
result = pyreadr.read_r("golub.Rdata")
# Assuming the Rdata file contains objects 'X' and 'Y'
X = result["X"].to_numpy()
Y = result["Y"].to_numpy()

Y=np.array([int(y[0]) for y in Y])



N = X.shape[0]

I = np.random.permutation(N)
X = scale(X)

X = X[I, :]
Y = Y[I]

## Training/test partition
N_tr = 40
X_tr = X[:N_tr, :]
Y_tr = Y[:N_tr]
N_ts = 32
X_ts = X[N_tr:N, :]
Y_ts = Y[N_tr:N]

## preliminary dimensionality reduction by ranking
ind_filter = rankrho(X_tr, Y_tr, 100)
print(ind_filter)
X_tr = X_tr[:, ind_filter]
X_ts = X_ts[:, ind_filter]

## wrapper feature selection 
wrap_var = KNN_wrap(X_tr, Y_tr, size=20, K=K)

###########################################
# Assessement of classification in the testset

for size in range(2, len(wrap_var) + 1):
    miscl = np.zeros(N_ts)
    Y_hat_ts = np.empty(N_ts, dtype=object)
    Conf_tt = np.empty((2, 2), dtype=int)
    for i in range(N_ts):
        q = X_ts[i, :]
        # Perform KNN classification on the selected features
        Y_hat_ts[i] = KNN(X_tr[:, wrap_var[:size]], Y_tr, K, q[wrap_var[:size]])
        
        miscl[i] = 1 if (Y_hat_ts[i] != Y_ts[i]) else 0

    miscl_tt = np.mean(miscl)
    # Row names: pred=0, pred=1; Column names: real=0, real=1
    Conf_tt[0, 0] = np.sum((Y_hat_ts == 0) & (Y_ts == 0))
    Conf_tt[0, 1] = np.sum((Y_hat_ts == 0) & (Y_ts == 1))
    Conf_tt[1, 0] = np.sum((Y_hat_ts == 1) & (Y_ts == 0))
    Conf_tt[1, 1] = np.sum((Y_hat_ts == 1) & (Y_ts == 1))

    print("K=" + str(K) + " size=" + str(size) + "; Misclass %=" + str(miscl_tt))
    print(Conf_tt)
    
