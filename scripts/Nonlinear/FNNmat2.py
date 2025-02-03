import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Parameters
n = 5  # size of input
L = 8  # number of layers

# Activation functions
def g(X):
    # Sigmoid activation function
    return 1.0 / (1.0 + np.exp(-X))

def gprime(X):
    # Derivative of sigmoid activation function
    exp_neg = np.exp(-X)
    return exp_neg / ((1 + exp_neg) ** 2)

def g2(X):
    # Linear activation function
    return X

def gprime2(X):
    # Derivative of linear activation function
    return np.ones_like(X)

# Define number of hidden nodes per layer
# In Python (0-indexed): we create a list H of length L where H[0]...H[L-2] are rounded linspace values and H[L-1] remains 1.
H = [1] * L  # initialize with ones
# Create hidden layer sizes for layers 0 to L-2 (corresponding to R indices 1 to L-1)
H_hidden = np.round(np.linspace(10, 2, num=L-1)).astype(int)
for i in range(L - 1):
    H[i] = int(H_hidden[i])
# Now H[-1] remains 1, corresponding to output layer

# Initialization of weights (W) and bias weights (Wb)
initsd = 1
W = []   # list to hold weight matrices for each layer
Wb = []  # list to hold bias weight matrices for each layer

for l in range(L):
    if l == 0:
        # For the first layer: shape is (input dimension, H[0])
        W.append(np.random.normal(scale=initsd, size=(n, H[l])))
    else:
        # For subsequent layers: shape is (H[l-1], H[l])
        W.append(np.random.normal(scale=initsd, size=(H[l-1], H[l])))

for l in range(L):
    # Bias weights: shape is (1, H[l])
    Wb.append(np.random.normal(scale=initsd, size=(1, H[l])))

# Training data
N = 250  # training sample size
Xtr = np.random.normal(scale=0.5, size=(N, n))
# Create training target as in R: Ytr = Xtr[:,0] + (Xtr[:,1])^2 - Xtr[:,n-1]*Xtr[:,n-2]*Xtr[:,n] (adjust indices for Python) + noise
# Thus in Python (0-indexed): Xtr[:,0] + Xtr[:,1]**2 - Xtr[:,2]*Xtr[:,3]*Xtr[:,4]
Ytr = Xtr[:, 0] + Xtr[:, 1]**2 - Xtr[:, 2] * Xtr[:, 3] * Xtr[:, 4] + np.random.normal(scale=0.2, size=N)

Etr = []  # list to store training errors

# Learning rate
eta = 0.5 / N

# Training loop
for r in range(7500):
    # ----- FORWARD STEP -----
    z = Xtr  # initial input, shape (N, n)
    A = []   # list to store activation vectors (pre-activation values) for each layer
    Z = []   # list to store output vectors (post-activation) for each layer

    # Forward propagation through layers
    for l in range(L):
        # Add bias column (the extra 1) to the input z for the current layer
        bias_column = np.ones((z.shape[0], 1))
        z_bias = np.concatenate([z, bias_column], axis=1)
        # Combine weights and bias weights by vertical stacking (rbind in R)
        W_combined = np.vstack([W[l], Wb[l]])
        # Compute activation: a = z_bias dot W_combined
        a = np.dot(z_bias, W_combined)
        A.append(a)
        # Apply activation function: use sigmoid for hidden layers, linear for output layer
        if l < L - 1:
            z = g(a)
        else:
            z = g2(a)
        Z.append(z)
    Yhat = z  # network output

    # Compute training error for this iteration
    E = Ytr - Yhat.flatten()  # ensure Yhat is 1D
    nmse = np.mean(E**2) / np.mean((Ytr - np.mean(Ytr))**2)
    if r % 100 == 0:
        print("Iteration =", r, "NMSE =", nmse)
    Etr.append(nmse)

    # ----- BACKPROPAGATION STEP -----
    # We will store Delta for each layer as matrices with shape (layer_size, N)
    # This mirrors the transposition in the R code.
    Delta = [None] * L
    DW = [None] * L
    DWb = [None] * L

    # For output layer: l = L-1 (in Python index)
    # In R: Delta[[L]] = t(gprime2(A[[L]]))
    Delta[L - 1] = gprime2(A[L - 1]).T  # shape: (H[L-1], N)
    # Initialize gradient accumulators for output layer weights and bias
    DW[L - 1] = np.zeros_like(W[L - 1])
    DWb[L - 1] = np.zeros_like(Wb[L - 1])
    # For each training sample, accumulate gradient for output layer
    for i in range(N):
        # Determine the input to the output layer: if there is a previous layer use its output, else use Xtr.
        if L - 1 > 0:
            layer_input = Z[L - 2][i, :]  # from the previous hidden layer (shape: (H[L-2],))
        else:
            layer_input = Xtr[i, :]
        # Outer product: (input vector) outer (Delta for sample i from output layer)
        DW[L - 1] += E[i] * np.outer(layer_input, Delta[L - 1][:, i])
        # Bias gradient update
        DWb[L - 1] += E[i] * Delta[L - 1][:, i].reshape(1, -1)

    # For hidden layers: from layer L-2 down to layer 0
    for l in range(L - 2, -1, -1):
        # Compute Delta for layer l:
        # Delta[l] = (W[l+1] dot Delta[l+1]) * t(gprime(A[l]))
        # W[l+1]: shape (H[l], H[l+1])
        # Delta[l+1]: shape (H[l+1], N)
        # So dot gives shape (H[l], N), and gprime(A[l]).T also gives shape (H[l], N)
        Delta[l] = np.dot(W[l + 1], Delta[l + 1]) * gprime(A[l]).T
        # Initialize gradients for this layer
        DW[l] = np.zeros_like(W[l])
        DWb[l] = np.zeros_like(Wb[l])
        # Determine the input for this layer: for l==0 use Xtr, else use Z[l-1]
        for i in range(N):
            if l > 0:
                layer_input = Z[l - 1][i, :]
            else:
                layer_input = Xtr[i, :]
            DW[l] += E[i] * np.outer(layer_input, Delta[l][:, i])
            DWb[l] += E[i] * Delta[l][:, i].reshape(1, -1)

    # ----- GRADIENT STEP: update weights -----
    for l in range(L):
        W[l] = W[l] + 2 * eta * DW[l]
        Wb[l] = Wb[l] + 2 * eta * DWb[l]

# ----- PLOTTING RESULTS -----
plt.figure(figsize=(12, 5))

# Plot training outputs vs predictions
plt.subplot(1, 2, 1)
plt.scatter(Ytr, Yhat, c="blue", label="Predictions")
plt.xlabel("FNN predictions")
plt.ylabel("Training output")
plt.title("FNN Predictions vs Training Output")
plt.legend()

# Plot training error curve over iterations
plt.subplot(1, 2, 2)
plt.plot(Etr, color="red")
plt.xlabel("Iterations")
plt.ylabel("Training Error (NMSE)")
plt.title("Training Error over Iterations")
plt.tight_layout()
plt.show()
