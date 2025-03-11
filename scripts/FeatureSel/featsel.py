import numpy as np
import matplotlib.pyplot as plt
import pyreadr

# "Statistical foundations of machine learning" software
# ackage gbcodepy
# Author: G. Bontempi
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


# load("golub.Rdata")  ## dataset upload
result = pyreadr.read_r("golub.Rdata")


X = result["X"].to_numpy()
Y = result["Y"].to_numpy()

Y=np.array([int(y[0]) for y in Y])



n = X.shape[1]
N = X.shape[0]
np.random.seed(0)
I = np.random.permutation(N)
X = X[I, :]
Y = np.array(Y)[I]
K = 1

plt.figure()
plt.xlim(1, 50)
plt.ylim(0, 0.3)
plt.ylabel("misclassification loo")
plt.xlabel("size")

for size in range(2, 51):
    ###############################
    # Leave-one-out
    Y_hat_ts = np.empty(N, dtype=Y.dtype)
    for i in range(N):  # for each of the N samples
        X_tr = np.delete(X, i, axis=0)
        Y_tr = np.delete(Y, i)
        q = X[i, :]

        # Convert Y_tr to numeric codes similar to R's as.numeric(Y.tr)
        unique_Y = np.unique(Y_tr)
        mapping = {val: (idx + 1) for idx, val in enumerate(unique_Y)}
        Y_tr_numeric = np.array([mapping[val] for val in Y_tr], dtype=float)

        # Compute correlation between each column of X_tr and Y_tr_numeric.
        # This computes the Pearson correlation for each feature.
        correlation = np.zeros(X_tr.shape[1])
        for j in range(X_tr.shape[1]):
            col = X_tr[:, j]
            # If standard deviation is zero, set correlation to zero to avoid division by zero.
            if np.std(col) == 0 or np.std(Y_tr_numeric) == 0:
                correlation[j] = 0
            else:
                correlation[j] = np.corrcoef(col, Y_tr_numeric)[0, 1]

        # Rank variables in descending order of absolute correlation
        abs_corr = np.abs(correlation)
        ranked_var = np.argsort(-abs_corr)

        Y_ts = Y[i]
        # Use only the top "size" variables (columns)
        X_tr_sub = X_tr[:, ranked_var[:size]]
        q_sub = q[ranked_var[:size]]
        Y_hat_ts[i] = KNN(X_tr_sub, Y_tr, K, q_sub)
    print(size)

    miscl_loo = np.mean(Y_hat_ts != Y)
    plt.plot(size, miscl_loo, 'o', markersize=4)  # Using marker size approximately corresponding to cex=0.6

plt.show()
