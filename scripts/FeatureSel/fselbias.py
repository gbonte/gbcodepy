import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def pred(X_train, Y_train, X_test):
    # Linear regression prediction using least squares
   reg = LinearRegression().fit(X_train, Y_train)
   return reg.predict(X_test)
   
   

# Set seed for reproducibility
np.random.seed(0)
n = 4  # number relevant input variables
nirr = 50  # number irrelevant input variables
p = n + 1
N = 20  # number training data
Nts = 5000

# Generate training data
X_relevant = np.random.uniform(low=-2, high=2, size=(N, n))
X = np.hstack((np.ones((N, 1)), X_relevant, np.random.normal(size=(N, nirr))))
# Generate test data for inputs
Xts_relevant = np.random.uniform(low=-2, high=2, size=(Nts, n))
Xts = np.hstack((np.ones((Nts, 1)), Xts_relevant, np.random.normal(size=(Nts, nirr))))
beta = np.random.normal(scale=2, size=p)
sd_w = 1

# Generate responses for training and test data using only the intercept and relevant variables
Y = np.dot(X[:, :p], beta) + np.random.normal(scale=sd_w, size=N)
Yts = np.dot(Xts[:, :p], beta) + np.random.normal(scale=sd_w, size=Nts)
np.random.seed(0)

R = 15000
fset = []      
bestr = []     # internal MSE computed with LOO
bests = []     # test MSE computed on Xts

# Loop over iterations: number of iterations is total columns - 1 (excluding intercept)
for it in range(1, 25):
    # Initialize MSE for candidate features with 'infinite' error
    MSEf = {}
    # Candidate features are from 2 to NCOL(X) in R indexing (which are 2 ... X.shape[1])
    candidates = [f for f in range(2, X.shape[1] + 1) if f not in fset]
    for f in candidates:
        e = np.empty(N)
        # For each training sample perform leave-one-out cross-validation
        for i in range(N):
            # Remove the i-th observation for training
            X_train_all = np.delete(X, i, axis=0)
            Y_train_all = np.delete(Y, i)
            # Use the selected features in fset and the current candidate f; convert R indices to 0-based index
            cols = [j - 1 for j in (fset + [f])]
            X_train = X_train_all[:, cols]
            X_test_sample = X[i, cols]
            Yhati = pred( X_train, Y_train_all, X_test_sample.reshape(1, -1))
            e[i] = Y[i] - Yhati[0]
        mse = np.mean(e**2)
        MSEf[f] = mse

    
    # Select candidate feature with the minimum MSE
    f_best = min(MSEf, key=MSEf.get)
    fset.append(f_best)
    bestr.append(MSEf[f_best])
    print(bestr)
    # Refit model on the whole training set using the selected features and predict on test data
    cols = [j - 1 for j in fset]
    
    Yhats = pred( X[:, cols], Y, Xts[:, cols])
    mse_test = np.mean((Yts - Yhats)**2)
    bests.append(mse_test)
    
    # Print the current set of selected features and the difference between test MSE and internal LOO MSE
    print(fset)
    diff = np.array(bests) - np.array(bestr)
    #print(diff)
    
    # Plot Test MSE and Internal MSE LOO over the feature set size
    plt.clf()
    plt.plot(bests, color="red", label="Test MSE")
    plt.plot(bestr, linestyle="--", linewidth=2, color="black", label="Internal MSE LOO")
    plt.ylim(0, max(bests))
    plt.title("Selection bias and feature selection")
    plt.xlabel("Feature set size")
    plt.ylabel("")
    plt.legend(loc="upper left")
    plt.pause(0.001)

plt.show()
