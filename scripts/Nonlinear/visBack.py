import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# Set seed for reproducibility
np.random.seed(0)


# Define activation functions
def activation(a):
    g=np.tanh(a)
    return  g

def derivative_activation(a):
    gprime=1-activation(a)**2
    return gprime


# Generate a simple dataset for regression
n_samples = 1000
x = np.random.uniform(-5, 5, n_samples)
f=np.sin(x)
y = f + np.random.normal(0, 0.15, n_samples)  # Quadratic function with noise


# Normalize the data
x_norm = (x - np.mean(x)) / np.std(x)
y_norm = (y - np.mean(y)) / np.std(y)
f_norm=(f - np.mean(y)) / np.std(y)



# Reshape for matrix operations
X = x_norm.reshape(-1, 1)
Y = y_norm.reshape(-1, 1)

# Define neural network architecture
input_size = 1
hidden_size = 2
output_size = 1

# Initialize weights (no biases)
#np.random.seed(2)

W1 = np.zeros((input_size, hidden_size))
W1[0,0]=-5
W1[0,1]=5

W2 = np.zeros((hidden_size, output_size) )
W2[0,0]=-1.42
W2[1,0]=2.46

## keep W2 fixed for visualisation purposes
alsoW2=False

# Learning rate
learning_rate = 0.05

# Number of training iterations
n_iterations = 500

# Storage for visualization
error_history = np.zeros(n_iterations)
W1_history = np.zeros((n_iterations, input_size, hidden_size))
W2_history = np.zeros((n_iterations, hidden_size, output_size))

# Forward pass function
def forward_pass(x, W1, W2):
    # Hidden layer (no bias)
    a1 = np.dot(x, W1) ## [N,1]*[1,2]
    z = activation(a1) ## [N,2]
    
    # Output layer (no bias)
    ## output activation function is identity function
    output = np.dot(z, W2) ## [N,1]
    
    return output, a1, z

# Training loop with backpropagation
for i in range(n_iterations):
    # Forward pass
    Yhat, a1, z = forward_pass(X, W1, W2)
    
    # Calculate error (mean squared error)
    error = np.mean((Yhat - Y)**2) 
    error_history[i] = error
    
    # Store weights for visualization
    W1_history[i] = W1.copy()
    W2_history[i] = W2.copy()
    
    # Backpropagation
    output_delta = -2*(Y-Yhat)  # Derivative of MSE
   
    
    # Compute gradients
    dW2 = np.dot(z.T, output_delta) / n_samples
    
    hidden_delta = np.dot(output_delta, W2.T) * derivative_activation(a1)  
    dW1 = np.dot(X.T, hidden_delta) / n_samples
    
    
    
    # Update weights
    if alsoW2:
        W2 = W2 - learning_rate * dW2
    W1 = W1 - learning_rate * dW1
    
    
    
    # Print progress
    if (i+1) % 10 == 0:
        print(f"Iteration: {i+1}, Error: {error:.6f}")




def compute_error_surface( X, Y, w1_range, w2_range):
    errors = np.zeros((len(w1_range), len(w2_range)))
    
    
    for i, w1 in enumerate(w1_range):
        for j, w2 in enumerate(w2_range):
            # Set weights temporarily
            
            tW1=W1
            tW1[0,0]=w1
            tW1[0,1]=w2
            # Compute error for this weight combination
            errors[i, j] = np.mean((Y-forward_pass(X, tW1, W2)[0])**2)
    
    # Restore original weights
    
    
    return errors
# Compute error surface


# Define weight ranges for the error surface visualization (fewer points to avoid memory issues)
w1_range = np.linspace(-10, 10, 50)
w2_range = np.linspace(-10, 10, 50)
GW1, GW2 = np.meshgrid(w1_range, w2_range)



error_surface = compute_error_surface(X, Y, w1_range, w2_range)

# Extract weights and errors from history
w11_history = [w[0,0] for w in W1_history]
w12_history = [w[0,1] for w in W1_history]
error_history = []

# Compute the error at each point in the W1 weight history
# W2 is kept fixed
for w11, w12 in zip(w11_history, w12_history):
    tW1=W1
    tW1[0,0]=w11
    tW1[0,1]=w12
    yhat=forward_pass(X, tW1, W2)[0]
    error_history.append(np.mean((Y-yhat)**2))

# 1. Plot the error surface and W1 weight path
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot error surface
surf = ax.plot_surface(GW1, GW2, error_surface, cmap=cm.viridis, alpha=0.6, 
                      linewidth=0, antialiased=True)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
# Plot the W1 weight path
ax.plot(w11_history, w12_history, error_history, 'k-', linewidth=2, label='Backpropagation Path')
ax.plot(w11_history, w12_history, error_history, 'k.', markersize=4)

# Mark start and end points
ax.plot([w11_history[0]], [w12_history[0]], [error_history[0]], 'go', markersize=10, label='Start')
ax.plot([w11_history[-1]], [w12_history[-1]], [error_history[-1]], 'ro', markersize=10, label='End')

# Set labels and title
ax.set_xlabel('Input-to-Hidden Weight 1')
ax.set_ylabel('Input-to-Hidden Weight 2')
ax.set_zlabel('Training Error')
ax.set_title('Error Surface and Backpropagation Path of Input-to-Hidden W1 Weights')
ax.legend()

fig = plt.figure(figsize=(12, 10))

plt.contourf(GW1, GW2, error_surface, levels=50, cmap='viridis', alpha=0.7)
plt.colorbar(label='Training error')

plt.xlabel('Input-to-Hidden Weight 1')
plt.ylabel('Input-to-Hidden Weight 2')
plt.title('W1 Weight Trajectory in W1 Weight Space')
plt.plot(w11_history, w12_history, 'k-', linewidth=2)
plt.plot(w11_history, w12_history, 'k.', markersize=4)
plt.plot(w11_history[0], w12_history[0], 'go', markersize=10, label='Start')
plt.plot(w11_history[-1], w12_history[-1], 'ro', markersize=10, label='End')



fig = plt.figure(figsize=(12, 10))
plt.plot(error_history,label='Training error')
plt.legend()
fig = plt.figure(figsize=(12, 10))
I=np.argsort(x_norm)
plt.plot(x_norm[I],f_norm[I],'k',lw=4,label='Regression function')
plt.scatter(x_norm,y_norm)
yhat0=forward_pass(X, W1_history[10,:], W2_history[10,:])[0]
plt.scatter(x_norm,yhat0,label='Iteration 10')
yhat0=forward_pass(X, W1_history[90,:], W2_history[90,:])[0]
plt.scatter(x_norm,yhat0,label='Iteration 90')
yhat=forward_pass(X, W1, W2)[0]
plt.scatter(x_norm,yhat,label='Last iteration')
plt.legend()

#W2_history[10]