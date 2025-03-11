import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import trim_mean

# Function pred implements the "lazy" method as in the original R code.
# It performs prediction by averaging the values of the NK nearest neighbours of each test point.
def knnpred(X, Y, Xts,  k=1):
    Nts=Xts.shape[0]
    Yhat=np.zeros((Nts,1))
    
    for i  in np.arange(Nts):
        x=Xts[i,:]
        # Compute distances between x and each element in X
        distances = np.abs(X - x)
        # Get indices of the k smallest distances
        indices = np.argsort(distances)[:k]
    # For regression, return the mean of the k nearest neighbors
        Yhat[i]=np.mean(Y[indices])
    return Yhat


# polynomial function f
def ff(X, beta):
    # X is assumed to be a 2D numpy array (if 1D, it will be converted to 2D column vector)
    X = np.atleast_2d(X)
    N = X.shape[0]
    # Create a column of ones of length N
    ones_col = np.ones((N, 1))
    # Concatenate ones_col with X horizontally
    XX = np.hstack((ones_col, X))
    # Return the matrix product of XX and beta
    return np.dot(XX, beta)


## polynomial function f
# Already defined above as ff

R = 500  ## Monte Carlo repetitions
Nts = 200  ## size of Monte Carlo test set

widthx = 2  ## sdev input density
K = 10  ## number CV folds

Dplot = None
for rep in range(1, 1001):
    np.random.seed(rep + 11)
    sdnoise = np.random.uniform(0.1, 1)  ## sdev noise
    N = np.random.randint(20, 151)  ## size of training set
    d = np.random.randint(2, 9)  ## degree of f (ground truth)
    NK = 10  ## number neighbours local regression

    ## polynomial parameters
    beta = np.random.normal(loc=0, scale=1, size=d+1)

    E2 = []  ## vector of squared residuals
    MISEloo = []
    MISEts = []
    MISEcv = []

    q = np.linspace(-widthx, widthx, Nts)

    Q = q.copy()
    # For i in 2:d, add q^i as additional columns
    Q = np.column_stack([q ** i for i in range(1, d + 1)])
    
    ## dataset normalisation
    # Compute regression function values on grid and then scale them
    fQQ_unscaled = ff(Q, beta)
    center = np.mean(fQQ_unscaled)
    scale_val = np.std(fQQ_unscaled, ddof=1)
    fQQ = (fQQ_unscaled - center) / scale_val  ## scaled regression function values
    fQ = fQQ + np.random.normal(loc=0, scale=sdnoise, size=len(q))  ## output values

    EQ = None
    EQQ = None
    YQ = None

    ## Monte Carlo simulation
    for r in range(1, R + 1):
        np.random.seed(r)
        
        ## dataset generation for Monte Carlo MISE computation
        xtr = np.random.uniform(-widthx, widthx, N)
        Xtr = xtr.copy()
        # For i in 2:d, add xtr^i as additional columns
        Xtr = np.column_stack([xtr ** i for i in range(1, d + 1)])
        Ytr = ff(Xtr, beta)
        
        ## dataset normalisation
        Ytr = (Ytr - center) / scale_val + np.random.normal(loc=0, scale=sdnoise, size=N)
        
        xts = np.random.uniform(-widthx, widthx, Nts)
        Xts = xts.copy()
        # For i in 2:d, add xts^i as additional columns
        Xts = np.column_stack([xts ** i for i in range(1, d + 1)])
        Yts = ff(Xts, beta)
        Yts = (Yts - center) / scale_val + np.random.normal(loc=0, scale=sdnoise, size=Nts)
        
        yhatq = knnpred(Xtr[:, 0], Ytr, q.reshape(-1,1), NK)
        if YQ is None:
            YQ = yhatq.reshape(1, -1)
        else:
            YQ = np.vstack((YQ, yhatq.reshape(1, -1)))
        diff_EQ = (fQ - yhatq)
        if EQ is None:
            EQ = diff_EQ.reshape(1, -1)
        else:
            EQ = np.vstack((EQ, diff_EQ.reshape(1, -1)))
        diff_EQQ = (fQQ - yhatq)
        if EQQ is None:
            EQQ = diff_EQQ.reshape(1, -1)
        else:
            EQQ = np.vstack((EQQ, diff_EQQ.reshape(1, -1)))
        Yhats = knnpred( Xtr[:, 0], Ytr, Xts[:, 0].reshape(-1,1), NK)
        e2 = (Yts.reshape(-1,1) - Yhats.reshape(-1,1)) ** 2
        E2 = E2 + list(e2)
        
        ## dataset generation for MISE assessment
        xtr = np.random.uniform(-widthx, widthx, N)
        Xtr = xtr.copy()
        Xtr = np.column_stack([xtr ** i for i in range(1, d + 1)])
        Ytr = ff(Xtr, beta)
        
        ## dataset normalisation
        Ytr = (Ytr - center) / scale_val + np.random.normal(loc=0, scale=sdnoise, size=N)
        
        ## MISE loo estimation
        eloo = []
        for i in range(N):
            # Exclude the i-th observation
            Xtri = np.delete(Xtr[:, 0], i)
            Ytri = np.delete(Ytr, i)
            Xtsi = Xtr[i, 0]
            Yhatsi = knnpred( Xtri, Ytri, Xtsi.reshape(1,-1), NK)[0]
            eloo.append((Ytr[i] - Yhatsi) ** 2)
        MISEloo.append(np.mean(eloo))
        
        ## MISE cv estimation
        ecv = []
        for i in range(1, K + 1):
            ntrain = int(round(((K - 1) * N) / K))
            Itr = np.random.choice(np.arange(N), size=ntrain, replace=False)
            Its = np.setdiff1d(np.arange(N), Itr)
            Xtri = Xtr[Itr, 0]
            Ytri = Ytr[Itr]
            Xtsi = Xtr[Its, 0]
            Yhatsi =knnpred(Xtri, Ytri, Xtsi.reshape(-1,1), NK)
            ecv.extend((Ytr[Its].reshape(-1,1) - Yhatsi.reshape(-1,1)) ** 2)
        MISEcv.append(np.mean(ecv))
        
        #### MISE holdout estimation
        NNts = int(round(N / 3))  ## portion test set
        Xtri = Xtr[:(N - NNts), 0]
        Xtsi = Xtr[(N - NNts):, 0]
        Ytri = Ytr[:(N - NNts)]
        Ytsi = Ytr[(N - NNts):]
        Yhatsi = knnpred( Xtri, Ytri, Xtsi.reshape(-1,1), NK)
        ets = (Ytsi.reshape(-1,1) - Yhatsi.reshape(-1,1)) ** 2
        MISEts.append(np.mean(ets))
        
    MISE = np.mean(E2)
    VSE = np.var(E2, ddof=1)
    # Compute squared bias of the learner using a trimmed mean (1% trimming)
    BiasY = (np.array([trim_mean(YQ[:, j], 0.01) for j in range(YQ.shape[1])]) - fQQ) ** 2
    VarY = np.var(YQ, axis=0, ddof=1)  ## variance of the learner
    
    MSEf = np.mean(EQQ ** 2, axis=0)  ## MSE of the learner (without noise variance)
    MSEY = np.mean(np.mean(EQ ** 2, axis=0))  ## MSE of the learner (including noise variance)
    
    VSEY = trim_mean(np.var(EQ ** 2, axis=0, ddof=1), 0.001)  ## average of VSE of the learner over the input
    
    term = 4 * BiasY * VarY + 2 * VarY ** 2 + (sdnoise ** 4)
    VSEhat = trim_mean(term, 0.01) + np.var(MSEf, ddof=1) + (sdnoise ** 4) + 2 * (sdnoise ** 2) * np.mean(MSEf)
    ## VSE of the learner estimated according the formula in the book
    
    print("\n --\n rep=", rep, " \n MISE=", MISE,  "VSE=", VSE,
          "VSEhat=", VSEhat, " \n")
    
    MMSEloo = np.mean((np.array(MISEloo) - np.mean(MISE)) ** 2)  ## MSE MISEloo
    MMSEts = np.mean((np.array(MISEts) - np.mean(MISE)) ** 2)   ## MSE MISEholdout
    MMSEcv = np.mean((np.array(MISEcv) - np.mean(MISE)) ** 2)    ## MSE MISEcv
    print("MISEloo=", np.mean(MISEloo), "Bias MISEloo=", abs(np.mean(MISEloo) - np.mean(MISE)),
          "Var MISEloo=", np.var(MISEloo, ddof=1), "MSE MISEloo=", MMSEloo)
    print("MISEts=", np.mean(MISEts), "Bias MISEholdout=", abs(np.mean(MISEts) - np.mean(MISE)),
          "Var MISEholdout=", np.var(MISEts, ddof=1), "MSE MISEholdout=", MMSEts)
    print("MISEcv=", np.mean(MISEcv), "Bias MISEcv=", abs(np.mean(MISEcv) - np.mean(MISE)),
          "Var MISEcv=", np.std(MISEcv, ddof=1), "MSE MISEcv=", MMSEcv)
    
    row_Dplot = np.array([MISE, np.mean(MISEloo), np.mean(MISEts), np.mean(MISEcv),
                          MMSEloo, MMSEts, MMSEcv])
    if Dplot is None:
        Dplot = row_Dplot.reshape(1, -1)
    else:
        Dplot = np.vstack((Dplot, row_Dplot))
    if rep > 3:
        plt.figure()
        # Plotting MISE vs Estimated MISE for LOO
        plt.plot(Dplot[:, 0], Dplot[:, 1], 'o', label="LOO")
        # Plot points for Holdout (2/3,1/3) in red
        plt.plot(Dplot[:, 0], Dplot[:, 2], 'o', color="red", label="Holdout (2/3,1/3)")
        # Plot points for 10-fold CV in blue
        plt.plot(Dplot[:, 0], Dplot[:, 3], 'o', color="blue", label="10-fold CV")
        # Draw line where estimated MISE equals actual MISE
        plt.plot(Dplot[:, 0], Dplot[:, 0], label="True", color="black")
        plt.xlabel("MISE")
        plt.ylabel("Estimated MISE")
        plt.legend(loc="lower right")
        plt.show()
        # Compute Kendall correlation for columns 1 to 4
        df_corr = pd.DataFrame(Dplot[:, 0:4], columns=["MISE", "LOO", "Holdout", "CV"])
        print("Correlation matrix: \n",df_corr.corr(method="kendall"))
        # Print mean of columns 5 to 7
        print(np.mean(Dplot[:, 4:7], axis=0))
        
# End of simulation code
