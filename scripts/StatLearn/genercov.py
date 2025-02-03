## # "Statistical foundations of machine learning" software
# Author: G. Bontempi
# genercov.py
# ## Monte Carlo verification of the covariance penalty term.

import numpy as np

def fun(X, sdw=1):
    # simple target function: returns X plus noise scaled by sdw
    return X + sdw * np.random.randn(len(X))

def regrlin(Xtr_poly, Ytr, Xts_poly):
    # Linear regression using the least squares method.
    # No intercept is added because the design matrix already provides the polynomial terms.
    coeffs, residuals, rank, s = np.linalg.lstsq(Xtr_poly, Ytr, rcond=None)
    Yhat_train = Xtr_poly @ coeffs
    Yhat_test = Xts_poly @ coeffs
    mse_emp = np.mean((Ytr - Yhat_train)**2)
    return {'MSE.emp': mse_emp, 'Y.hat': Yhat_train, 'Y.hat.ts': Yhat_test}

def create_polynomial_features(X, degree=8):
    # Create a design matrix with polynomial terms from power 1 to the given degree.
    return np.column_stack([X**i for i in range(1, degree+1)])

if __name__ == "__main__":
    # Monte Carlo verification of the covariance penalty term.
    
    np.random.seed(0)
    
    N = 35       # number of examples
    S = 15000    # number of Monte Carlo trials
    sdw = 1
    
    R_list = []
    Remp_list = []
    
    # Arrays to store training responses and predictions across MC iterations.
    Y_storage = np.empty((N, S))
    Yhat_storage = np.empty((N, S))
    
    # Generate training inputs uniformly between -1 and 1.
    Xtr = np.random.uniform(-1, 1, N)
    # Compute the target function without noise.
    f = fun(Xtr, 0)
    
    for s in range(S):
        # Generate noisy training outputs.
        Ytr = fun(Xtr, sdw)
    
        Nts = 50000
        # Generate test inputs.
        Xts = np.random.uniform(-1, 1, Nts)
        # Generate noisy test outputs.
        Yts = fun(Xts, sdw)
    
        # Create polynomial features for both training and test sets (degree 8).
        Xtr_poly = create_polynomial_features(Xtr, degree=8)
        Xts_poly = create_polynomial_features(Xts, degree=8)
    
        # Perform linear regression using the polynomial features.
        RG = regrlin(Xtr_poly, Ytr, Xts_poly)
    
        bestRemp = RG['MSE.emp']
    
        # Store training outputs and predictions.
        Y_storage[:, s] = Ytr
        Yhat_storage[:, s] = RG['Y.hat']
    
        # Compute test error.
        test_error = np.mean((Yts - RG['Y.hat.ts'])**2)
        R_list.append(test_error)
        Remp_list.append(bestRemp)
    
        # Print progress indicator every 1000 iterations.
        if (s + 1) % 1000 == 0:
            print(".", end="", flush=True)
    
    
    # covariance computation for each training example.
    Cov_list = []
    for i in range(N):
        cov_val = np.mean((Y_storage[i, :] - f[i]) * (Yhat_storage[i, :] - f[i]))
        Cov_list.append(cov_val)
    
    mean_R = np.mean(R_list)
    mean_Remp = np.mean(Remp_list)
    mean_Cov = np.mean(Cov_list)
    
    print("E[R(alpha_N)] =", mean_R)
    print("E[Remp(alpha_N)] =", mean_Remp)
    print("E[Cov] =", mean_Cov)
    
    print("estimated =", mean_Remp + 2 * mean_Cov)
    
