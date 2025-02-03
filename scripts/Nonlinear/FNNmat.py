# "INFOF422 Statistical foundations of machine learning" course
# R package gbcode 
# Author: G. Bontempi
## FNNmat.py

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)

# Size of input
n = 5

def g(X):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-X))

def gprime(X):
    # Derivative of sigmoid activation function
    return np.exp(-X) / ((1 + np.exp(-X))**2)

def g2(X):
    # Linear activation function
    return X

def gprime2(X):
    # Derivative of linear activation function
    return np.ones_like(X)

L = 5  # number of layers

# Hidden nodes per layer; initialize H as a list of ones and then assign hidden layer sizes.
H = [1] * L
# For layers 1 to L-1, assign sizes from 10 to 2 evenly spaced (rounded)
hidden_sizes = [round(x) for x in np.linspace(10, 2, L - 1)]
H[:L - 1] = hidden_sizes
# H now contains sizes for each layer weight output. For the output layer it remains 1.

# Initialization of weights
W = []
for l in range(L):
    if l == 0:
        # Weight matrix from input layer (n) to first hidden layer (H[0])
        W.append(np.random.normal(0, 1, (n, H[0])))
    else:
        # Weight matrix from previous hidden layer (H[l-1]) to current layer (H[l])
        W.append(np.random.normal(0, 1, (H[l - 1], H[l])))

# Prepare training set
N = 150  # training sample size
Xtr = np.random.normal(0, 0.5, (N, n))
# Create a non-linear relation for training output with added noise
Ytr = Xtr[:, 0] + Xtr[:, 1]**2 - Xtr[:, n - 2] * Xtr[:, n - 1] + np.random.normal(0, 0.2, N)

Etr = []  # list to store training errors


# Number of iterations for training
for r in range(1, 10001):

    # --- FORWARD STEP ---
    z = Xtr  # initial input
    A = []   # list of activation pre-nonlinearity values per layer (each element: (N, layer_size))
    Z = []   # list of layer outputs after activation function
    for l in range(L):
        # Compute activation vector for layer l: z dot weight matrix
        a = np.dot(z, W[l])
        # For all layers except the output layer, use sigmoid activation; for the output layer, use linear activation
        if l < L - 1:
            z = g(a)
        else:
            z = g2(a)
        A.append(a)
        Z.append(z)
    Yhat = z  # network predictions (last computed z)
    # Ensure Yhat is a 1D array if it is column vector
    Yhat = Yhat.flatten()
    
    # Compute error
    E = Ytr - Yhat
    nmse = np.mean(E**2) / np.mean((Ytr - np.mean(Ytr))**2)
    
    if r % 100 == 0:
        print("Iteration=", r, "NMSE=", nmse)
    Etr.append(nmse)
    
    # --- BACKPROPAGATION STEP ---
    # We will store Delta for each layer.
    # For consistency with the R code, we will store Delta such that for a given layer l,
    # Delta[l] is a matrix of shape (layer_size, N), where each column corresponds to a sample.
    Delta = [None] * L
    DW = [None] * L  # Gradients for each weight matrix

    # For the output layer: use derivative of linear activation (gprime2) and transpose to shape (output_dim, N)
    Delta[L - 1] = gprime2(A[L - 1]).T  # shape: (H[L-1], N)
    # Compute DW for the output layer (layer index L-1)
    # If the network has hidden layers, the input to the output layer is Z[L-2]; otherwise it is Xtr.
    DW[L - 1] = np.zeros_like(W[L - 1])
    # For vectorized computation, note that for each sample i, DW += E[i]*outer( input_row, Delta[:, i] ).
    # Determine input to the layer: if L-1 > 0, input comes from previous layer’s output, else from Xtr.
    input_to_layer = Xtr if L - 1 == 0 else Z[L - 2]
    # Reshape error to (N, 1) to multiply with Delta.T (which is N x output_dim)
    DW[L - 1] = np.dot(input_to_layer.T, E[:, None] * Delta[L - 1].T)
    
    # Backpropagate through hidden layers
    for l in range(L - 2, -1, -1):
        # Delta for layer l: (W[l+1] dot Delta[l+1]) element-wise multiplied by transpose of derivative of activation at layer l.
        Delta[l] = np.dot(W[l + 1], Delta[l + 1]) * gprime(A[l]).T  # shape: (H[l], N)
        # Compute DW for layer l
        # For layer 0, input is Xtr; for l>0, input is Z[l-1]
        input_to_layer = Xtr if l == 0 else Z[l - 1]
        DW[l] = np.dot(input_to_layer.T, E[:, None] * Delta[l].T)
        
    # Gradient step with learning rate
    eta = 0.25 / N
    for l in range(L):
        # Update weights
        W[l] = W[l] + 2 * eta * DW[l]


# After training, plot the FNN predictions vs training outputs and the training error over iterations
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(Yhat, Ytr)
plt.xlabel('FNN predictions')
plt.ylabel('Training output')
plt.title('Predictions vs Training Output')

plt.subplot(1, 2, 2)
plt.plot(Etr)
plt.xlabel('Iterations')
plt.ylabel('Training Error (NMSE)')
plt.title('Training Error over Iterations')

plt.tight_layout()
plt.show()
