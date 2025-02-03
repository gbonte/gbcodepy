import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import pdb

# Remove all existing variables -- in Python, we generally do not do this.
# Instead, one might restart the interpreter if needed.

N = 2000

def doppler(x):
    """
    Compute the Doppler function.
    20 * sqrt(x * (1 - x)) * sin(2*pi*1.05/(x+0.05))
    """
    return 20 * np.sqrt(x * (1 - x)) * np.sin(2 * np.pi * 1.05 / (x + 0.05))

def dataset_doppler(N, sigma=0.1):
    """
    Generate a dataset based on the Doppler function.
    Returns a dictionary with training and test data.
    """
    # Set seed for reproducibility
    np.random.seed(0)
    # Generate sorted random numbers between 0.12 and 1
    x = np.sort(np.random.uniform(0.12, 1, N))
    # Compute y with added Gaussian noise (standard deviation sigma)
    y = doppler(x) + np.random.normal(scale=sigma, size=N)
    # Generate test inputs and their corresponding outputs without noise
    x_ts = np.sort(np.random.uniform(0.12, 1, N))
    y_ts = doppler(x_ts)
    return {"x": x, "y": y, "x.ts": x_ts, "y.ts": y_ts}

# Generate dataset
D = dataset_doppler(N)


# Plot the modified training data (line plot)
plt.figure()
plt.plot(D["x"], D["y"], linestyle='-', color='black')
plt.xlabel("X")
plt.ylabel("Y")

plt.show()

# Prepare training data
# In R, the data frame d contains columns "Y" and "X"
X_train = D["x"].reshape(-1, 1)
Y_train = D["y"]

# Loop over different numbers of hidden nodes (from 1 to 30)
for number_nodes in range(1, 31):
    # Create and train the neural network model.
    # MLPRegressor in sklearn by default has one hidden layer.
    # The size of the hidden layer is given by (number_nodes,).
    mod_nn = MLPRegressor(hidden_layer_sizes=(number_nodes,),
                          max_iter=5000,activation="logistic",learning_rate_init=0.1,
                          verbose=False,   # This will print the training progress.
                          random_state=0)
    mod_nn.fit(X_train, Y_train)
    
    # Prepare test data in the same "data frame" structure as in R:
    # Here, X_test is the "X" column and Y_test is the "Y" column.
    X_test = D["x.ts"].reshape(-1, 1)
    Y_test = D["y.ts"]
    
    # Predict on test data and a subset of the training data (first 10 instances)
    p = mod_nn.predict(X_test)
    ptr = mod_nn.predict(X_train)
    
    # Plot the test truth and predictions.
    plt.figure()
    plt.plot(D["x.ts"], Y_test, linestyle='-', color='black', label="Test Truth")
    plt.plot(D["x.ts"], p, color='red', label="Test Prediction")
    # For training predictions, only plot for the first 10 points
    #plt.plot(X_train, ptr, color='blue', label="Train Prediction")
    #plt.xlabel("X")
    #plt.ylabel("Y")
    plt.title("Number hidden nodes = " + str(number_nodes))
    plt.legend()
    plt.show()
    
    # Enter interactive debugging, similar to browser() in R.
    
    
# After exiting the loop, the code ends.
