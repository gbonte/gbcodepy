import numpy as np
import matplotlib.pyplot as plt
import pyreadr
import math


result = pyreadr.read_r("EXAM.2s.2324.Rdata")  
# Extract objects Q3.X and Q3.Y exactly as loaded from the Rdata file
Q3_X = result["Q3.X"].to_numpy()
Q3_Y = result["Q3.Y"].to_numpy()

# Determine the number of rows in Q3.X and assign X as Q3.X
N = Q3_X.shape[0]
X = Q3_X

# Loop through the three labels: setosa, virginica, versicolor
for label in ["setosa", "virginica", "versicolor"]:
    
    # Create Y as a numeric array of zeros with length N
    Y = np.zeros(N)
    # Set Y to 1 where Q3.Y equals the current label
    indices_label = np.where(Q3_Y == label)[0]
    Y[indices_label] = 1
    I1 = np.where(Y == 1)[0]
    I0 = np.where(Y == 0)[0]
    N1 = len(I1)
    N0 = len(I0)
    
    # Define the KNN function
    def KNN(X, Y, q, k):
        # N: number of rows in X
        N_local = X.shape[0]
        # Compute Euclidean distances between each row of X and the query q
        d = np.sqrt(np.sum((X - q) ** 2, axis=1))  # Euclidean metric
        # Get the indices of the k smallest distances
        index = np.argsort(d)[:k]
        # Return the proportion of the nearest neighbors that have label 1
        return np.count_nonzero(Y[index] == 1) / k

    # Define the MD function
    def MD(X, Y, q):
        N_local = len(Y)
        n = X.shape[1]
        I1_local = np.where(Y == 1)[0]
        I0_local = np.where(Y == 0)[0]
        mu1 = np.mean(X[I1_local, :], axis=0)
        mu0 = np.mean(X[I0_local, :], axis=0)
        d1 = np.sum((q - mu1) ** 2)
        d0 = np.sum((q - mu0) ** 2)
        return np.exp(-d1) / (np.exp(-d1) + np.exp(-d0))

    # Define the QDA function
    def QDA(Xtr, Ytr, q):
        n = Xtr.shape[1]
        Ntr = Xtr.shape[0]
        I0_local = np.where(Ytr == 0)[0]
        I1_local = np.where(Ytr == 1)[0]
        P0 = len(I0_local) / Ntr
        P1 = 1 - P0
        mu0 = np.mean(Xtr[I0_local, :], axis=0)
        Sigma0 = np.cov(Xtr[I0_local, :], rowvar=False)
        mu1 = np.mean(Xtr[I1_local, :], axis=0)
        Sigma1 = np.cov(Xtr[I1_local, :], rowvar=False)
        # Convert query q to 2D row vector
        xts = np.array([q])
        # Calculate quadratic forms for class 0
        diff0 = xts - mu0
        inv_Sigma0 = np.linalg.inv(Sigma0)
        quad0 = diff0.dot(inv_Sigma0).dot(diff0.T)[0, 0]
        g0 = -0.5 * quad0 - n/2 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(Sigma0)) + np.log(P0)
        # Calculate quadratic forms for class 1
        diff1 = xts - mu1
        inv_Sigma1 = np.linalg.inv(Sigma1)
        quad1 = diff1.dot(inv_Sigma1).dot(diff1.T)[0, 0]
        g1 = -0.5 * quad1 - n/2 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(Sigma1)) + np.log(P1)
        return np.exp(g1) / (np.exp(g0) + np.exp(g1))
    
    # Define the LDA function
    def LDA(Xtr, Ytr, q):
        n = Xtr.shape[1]
        Ntr = Xtr.shape[0]
        I0_local = np.where(Ytr == 0)[0]
        I1_local = np.where(Ytr == 1)[0]
        P0 = len(I0_local) / Ntr
        P1 = 1 - P0
        # sigma2 is the mean of the variances of each column
        sigma2 = np.mean(np.var(Xtr, axis=0, ddof=1))
        mu0 = np.mean(Xtr[I0_local, :], axis=0)
        mu1 = np.mean(Xtr[I1_local, :], axis=0)
        # In R, xts, mu0, mu1 are converted to row vectors; in Python we use 1D arrays
        # Compute the necessary quadratic forms
        g0 = -1/(2*sigma2) * (np.dot(q, q) - 2 * np.dot(mu0, q) + np.dot(mu0, mu0)) + np.log(P0)
        g1 = -1/(2*sigma2) * (np.dot(q, q) - 2 * np.dot(mu1, q) + np.dot(mu1, mu1)) + np.log(P1)
        return np.exp(g1) / (np.exp(g0) + np.exp(g1))
    
    # Compute predictions for each observation using leave-one-out cross-validation
    N_local = X.shape[0]
    Yhat1 = np.zeros(N_local)
    Yhat2 = np.zeros(N_local)
    Yhat3 = np.zeros(N_local)
    Yhat4 = np.zeros(N_local)
    
    for i in range(N_local):
        # Exclude the i-th observation from X and Y for training
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y, i)
        # Compute predictions using QDA, KNN with k=10, LDA, and KNN with k=3 respectively
        Yhat1[i] = QDA(X_train, Y_train, X[i, :])
        Yhat2[i] = KNN(X_train, Y_train, X[i, :], k=10)
        Yhat3[i] = LDA(X_train, Y_train, X[i, :])
        Yhat4[i] = KNN(X_train, Y_train, X[i, :], k=3)
    
    # Sort prediction arrays in decreasing order while retaining the indices (s1, s2, s3, s4)
    s1_indices = np.argsort(-Yhat1)
    s2_indices = np.argsort(-Yhat2)
    s3_indices = np.argsort(-Yhat3)
    s4_indices = np.argsort(-Yhat4)
    s1 = {"values": Yhat1[s1_indices], "ix": s1_indices}
    s2 = {"values": Yhat2[s2_indices], "ix": s2_indices}
    s3 = {"values": Yhat3[s3_indices], "ix": s3_indices}
    s4 = {"values": Yhat4[s4_indices], "ix": s4_indices}
    
    # Initialize empty lists for TPR and FPR for each method
    TPR1 = []
    FPR1 = []
    TPR2 = []
    FPR2 = []
    TPR3 = []
    FPR3 = []
    TPR4 = []
    FPR4 = []
    
    Yts = Y.copy()
    
    # Create a sequence of thresholds from 0 to 1 (1000 values)
    THR = np.linspace(0, 1, 1000)
    # Set of thresholds loop
    for th in THR:
        I1_indices = np.where(Yhat1 >= th)[0]
        TPR1.append(len(np.where(Yts[I1_indices] == 1)[0]) / N1 if N1 > 0 else 0)
        FPR1.append(len(np.where(Yts[I1_indices] == 0)[0]) / N0 if N0 > 0 else 0)
        
        I2_indices = np.where(Yhat2 >= th)[0]
        TPR2.append(len(np.where(Yts[I2_indices] == 1)[0]) / N1 if N1 > 0 else 0)
        FPR2.append(len(np.where(Yts[I2_indices] == 0)[0]) / N0 if N0 > 0 else 0)
        
        I3_indices = np.where(Yhat3 >= th)[0]
        TPR3.append(len(np.where(Yts[I3_indices] == 1)[0]) / N1 if N1 > 0 else 0)
        FPR3.append(len(np.where(Yts[I3_indices] == 0)[0]) / N0 if N0 > 0 else 0)
        
        I4_indices = np.where(Yhat4 >= th)[0]
        TPR4.append(len(np.where(Yts[I4_indices] == 1)[0]) / N1 if N1 > 0 else 0)
        FPR4.append(len(np.where(Yts[I4_indices] == 0)[0]) / N0 if N0 > 0 else 0)
    
    # Plotting the ROC curves for each method
    plt.figure()
    # Plot QDA ROC curve in blue
    plt.plot(FPR1, TPR1, color="blue", label="QDA")
    # Plot the diagonal line (reference line)
    plt.plot(FPR1, FPR1, linestyle="--")
    # Plot KNN with k=10 in black
    plt.plot(FPR2, TPR2, color="black", label="KNN10")
    # Plot LDA ROC curve in red
    plt.plot(FPR3, TPR3, color="red", label="LDA")
    # Plot KNN with k=3 in green
    plt.plot(FPR4, TPR4, color="green", label="KNN3")
    
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(label)
    plt.legend(loc="lower right", fontsize=8, ncol=3)
    plt.show()
    
    
    
# End of loop over labels
