# "INFOF422 Statistical foundations of machine learning" course
# ROCparametric.py
# Author: G. Bontempi

## Nonparametric vs parametric (binormal) ROC curv ein a binary classification task

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# The following function replicates the R function 'fct'
def fct(X, sdw=2.5, p1=0.5):
    N = X.shape[0]  # number of rows in X
    sY = (X[:,0]**2 * X[:,1] + np.abs(X[:,2]) + np.random.normal(scale=sdw, size=N))
    Y = np.zeros(N)  # numeric vector of length N, initialized to 0
    
    # Set Y to 1 where sY > quantile(sY, p1)
    threshold = np.quantile(sY, p1)
    indices = np.where(sY > threshold)[0]
    Y[indices] = 1
    return Y

np.random.seed(0)
# Set initial parameters
sdw = 0.25
N = 1000
n = 10
X = np.random.normal(size=(N, n))
Y = fct(X, sdw)

Nts = 50000
Xts = np.random.normal(size=(N, n))
Yts = fct(Xts, sdw)
Its0 = np.where(Yts == 0)[0]
Its1 = np.where(Yts == 1)[0]

TPR = None
TNR = None
FPR = None
FNR = None
TH = None

X_mat = X
Xts_mat = Xts

# Use RandomForestClassifier to compute probability score
clf = RandomForestClassifier(random_state=0)
clf.fit(X_mat, Y.astype(int))
# Get the probability estimates for class "1"
Yhat = clf.predict_proba(Xts_mat)[:, 1]

# Create threshold sequence TH from 0 to 1 in steps of 0.001
TH = np.arange(0, 1.001, 0.001)
NP = len(Its1)
NN = len(Its0)

Bestth = None
BestFPR = None
BestTPR = None
BestFNR = None
BestNPhat = None
BestCost = None
Cost1 = None
KK = np.arange(1, 101, 5)
TPR_list = []
TNR_list = []
FPR_list = []
FNR_list = []

for th in TH:
    # Calculate True Positive Rate (TPR)
    tpr_val = np.sum((Yhat > th) & (Yts == 1)) / NP
    TPR_list.append(tpr_val)
    # Calculate True Negative Rate (TNR)
    tnr_val = np.sum((Yhat < th) & (Yts == 0)) / NN
    TNR_list.append(tnr_val)
    # Calculate False Positive Rate (FPR)
    fpr_val = np.sum((Yhat > th) & (Yts == 0)) / NN
    FPR_list.append(fpr_val)
    # Calculate False Negative Rate (FNR)
    fnr_val = np.sum((Yhat < th) & (Yts == 1)) / NP
    FNR_list.append(fnr_val)

# Convert lists to numpy arrays for further calculations
TPR_arr = np.array(TPR_list)
TNR_arr = np.array(TNR_list)
FPR_arr = np.array(FPR_list)
FNR_arr = np.array(FNR_list)

mu1 = np.mean(Yhat[Its1])
sd1 = np.std(Yhat[Its1], ddof=1)  # sample standard deviation
mu0 = np.mean(Yhat[Its0])
sd0 = np.std(Yhat[Its0], ddof=1)  # sample standard deviation
a = (mu1 - mu0) / sd1
b = sd0 / sd1

t = np.arange(0, 1.01, 0.01) ## threshold

AUROC = 0
# Loop over indices to accumulate AUROC using the trapezoidal-like rule
for i in range(len(TPR_arr) - 1):
    AUROC = AUROC + TPR_arr[i+1] * (FPR_arr[i] - FPR_arr[i+1])

# Plot the ROC curve
plt.plot(FPR_arr, TPR_arr,label="Nonparametric ROC")
plt.title("AUROC np=" + str(np.round(AUROC,3)) + 
          ": AUROC p= " + str(np.round(norm.cdf(a/np.sqrt(1+b**2)),3)))
plt.xlabel("FPR")
plt.ylabel("TPR")

plt.plot(t, norm.cdf(a + b * norm.ppf(t)), label="Parametric ROC",color="red")
plt.legend()
plt.show()


