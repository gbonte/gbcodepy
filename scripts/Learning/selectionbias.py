import numpy as np

# "INFOF422 Statistical foundations of machine learning" course
# Python package gbcode
# Author: G. Bontempi



def f(x):
    return 0.15 * x


def regrlin(X, Y, X_ts):
    # Fit a simple linear regression model using numpy.polyfit (degree 1)
    coeffs = np.polyfit(X, Y, 1)
    # Create a polynomial from the coefficients
    poly = np.poly1d(coeffs)
    # Ensure X_ts is a numpy array to handle both scalar and vector cases
    X_ts = np.array(X_ts)
    Y_hat_ts = poly(X_ts)
    return {'Y.hat.ts': Y_hat_ts}

# Parameters
N = 15
Nts = 10000
S = 1000
sdw = 0.5

# Initialize generalization error lists
Ets1 = []  # generalization error of h1
Ets2 = []  # generalization error of h2
Ets = []   # generalization error of selected model

# Monte Carlo simulation loop over S iterations
for s in range(1, S + 1):
    # Generate training data
    X = np.random.normal(loc=0, scale=0.5, size=N)
    Y = f(X) + np.random.normal(loc=0, scale=sdw, size=N)
    
    # Generate test data
    
    Xts = np.random.normal(loc=0, scale=1, size=Nts)
    Yts = f(Xts) + np.random.normal(loc=0, scale=sdw, size=Nts)
    
    Eloo1 = []  # Leave-one-out squared errors for model h1
    Eloo2 = []  # Leave-one-out squared errors for model h2
    for i in range(N):
        # Create index set excluding the i-th index
        I = [j for j in range(N) if j != i]
        
        # h1: prediction is the mean of Y over the training indices I
        hi1 = np.mean(Y[I])
        ei1 = Y[i] - hi1
        Eloo1.append(ei1 ** 2)
        
        # h2: prediction is obtained via linear regression on the training data excluding i-th sample
        hi2 = regrlin(X[I], Y[I], X[i])['Y.hat.ts']
        ei2 = Y[i] - hi2
        Eloo2.append(ei2 ** 2)
    
    MSEloo1 = np.mean(Eloo1)
    MSEloo2 = np.mean(Eloo2)
    
    # Compute training predictions and test error for model h1
    h1 = np.mean(Y)
    Ets1.append(np.mean((Yts - h1) ** 2))
    
    # Compute prediction for test data using linear regression model h2 on full training data
    h2 = regrlin(X, Y, Xts)['Y.hat.ts']
    Ets2.append(np.mean((Yts - h2) ** 2))
    
    # Select the model with lower leave-one-out error and record its test error
    if MSEloo1 < MSEloo2:
        Ets.append(np.mean((Yts - h1) ** 2))  # h1 is the selected model
    else:
        Ets.append(np.mean((Yts - h2) ** 2))  # h2 is the selected model

print("G1=", np.mean(Ets1), "G2=", np.mean(Ets2), "G selected=", np.mean(Ets))
