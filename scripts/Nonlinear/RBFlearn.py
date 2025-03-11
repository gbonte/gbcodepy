import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

class RBFNetwork:
    def __init__(self, n_centers, sigma=1.0):
        """
        Initialize RBF Network
        
        Parameters:
        -----------
        n_centers : int
            Number of RBF centers (hidden neurons)
        sigma : float
            Width parameter for the Gaussian RBF
        """
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None
        
    def _gaussian_rbf(self, x, centers):
        """
        Compute Gaussian RBF activation
        
        Parameters:
        -----------
        x : array, shape (n_samples, n_features)
            Input data
        centers : array, shape (n_centers, n_features)
            RBF centers
            
        Returns:
        --------
        rbf_activations : array, shape (n_samples, n_centers)
            RBF activations for each sample and center
        """
        # Compute distances between each input and all centers
        n_samples = x.shape[0]
        n_centers = centers.shape[0]
        distances = np.zeros((n_samples, n_centers))
        
        for i in range(n_samples):
            for j in range(n_centers):
                # Euclidean distance between input i and center j
                distances[i, j] = np.sum((x[i] - centers[j]) ** 2)
        
        # Apply Gaussian function to the distances
        rbf_activations = np.exp(-distances / (2 * self.sigma ** 2))
        
        return rbf_activations
    
    def fit(self, X, y):
        """
        Train the RBF network
        
        Parameters:
        -----------
        X : array, shape (n_samples, 2)
            Input features (2D input)
        y : array, shape (n_samples,)
            Target values
        """
        # Determine centers using K-means clustering
        kmeans = KMeans(n_clusters=self.n_centers, random_state=42)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        
        # Compute RBF activations
        rbf_activations = self._gaussian_rbf(X, self.centers)
        
        # Add bias term
        rbf_activations = np.hstack([np.ones((X.shape[0], 1)),rbf_activations ])
        
        print(rbf_activations.shape)
        
        # Solve for weights using least squares
        self.weights = np.linalg.lstsq(rbf_activations, y, rcond=None)[0]
        
        return self
    
    def predict(self, X):
        """
        Predict using the RBF network
        
        Parameters:
        -----------
        X : array, shape (n_samples, 2)
            Input features
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        rbf_activations = self._gaussian_rbf(X, self.centers)
        
        # Add bias term
        rbf_activations = np.hstack([np.ones((X.shape[0], 1)),rbf_activations ])
        
        # Compute predictions
        y_pred = np.dot(rbf_activations, self.weights)
        
        return y_pred

# Generate synthetic data for demonstration
def generate_data(n_samples=200, noise=0.1):
    """
    Generate synthetic data for regression
    """
    np.random.seed(42)
    
    # Generate random inputs in [0, 1]
    X = np.random.rand(n_samples, 2)
    
    # Generate targets with non-linear dependency and noise
    y = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1]) + noise * np.random.randn(n_samples)
    
    return X, y

def visualize_results(X, y, y_pred, rbf_network):
    """
    Visualize the RBF network results
    """
    # Create a meshgrid for visualization
    n_grid = 50
    x_grid = np.linspace(0, 1, n_grid)
    y_grid = np.linspace(0, 1, n_grid)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict at grid points
    z_pred = rbf_network.predict(grid_points).reshape(n_grid, n_grid)
    
    # Plot prediction surface
    fig = plt.figure(figsize=(12, 8))
    
    # 3D plot of the prediction surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(xx, yy, z_pred, cmap='viridis', alpha=0.8)
    ax1.scatter(X[:, 0], X[:, 1], y, c='r', marker='o', alpha=0.5)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('y')
    ax1.set_title('RBF Network Prediction Surface')
    
    # Plot RBF centers
    ax2 = fig.add_subplot(122)
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.7, label='Data points')
    ax2.scatter(rbf_network.centers[:, 0], rbf_network.centers[:, 1], 
               marker='*', s=200, color='black', label='RBF centers')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_title('RBF Centers')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# Generate data
X, y = generate_data(n_samples=200, noise=0.1)

# Split data into train and test sets
n_train = 150
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

# Create and train RBF Network
rbf_network = RBFNetwork(n_centers=5, sigma=0.1)
rbf_network.fit(X_train, y_train)

# Make predictions
y_pred_train = rbf_network.predict(X_train)
y_pred_test = rbf_network.predict(X_test)

# Calculate errors
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Visualize results
visualize_results(X_test, y_test, y_pred_test, rbf_network)


