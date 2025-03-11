import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time



def J(a1, a2):
    # Returns a1^2 + a2^2 - 2*a1 - 2*a2 + 6
    return a1**2 + a2**2 - 2*a1 - 2*a2 + 6

def Jprime(a):
    # a is expected to be an array-like with two elements
    a1 = a[0]
    a2 = a[1]
    return np.array([2*a1 - 2, 2*a2 - 2])

alpha1 = np.arange(-2, 4.1, 0.1)
alpha2 = np.arange(-2, 4.1, 0.1)
X, Y = np.meshgrid(alpha1, alpha2, indexing='ij')
z = J(X, Y)

# Perspective plot (3D surface plot)
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, z, cmap='viridis')
plt.title('Perspective Plot')
ax.set_xlabel('alpha1')
ax.set_ylabel('alpha2')
ax.set_zlabel('J(alpha1, alpha2)')

# Contour plot
fig2 = plt.figure()
plt.contour(alpha1, alpha2, z)
plt.title('Contour Plot')
plt.xlabel('alpha1')
plt.ylabel('alpha2')

# alpha0 initialization
a = np.array([3.0, 3.0])
plt.plot(a[0], a[1], 'ro', markersize=8)  # Red point with line width approximated by markersize

mu = 0.1
plt.ion()  # Turn on interactive mode for dynamic updates

for r in range(100):
    plt.contour(alpha1, alpha2, z)
    plt.title('Contour Plot')
    plt.xlabel('alpha1')
    plt.ylabel('alpha2')
    a = a - mu * Jprime(a)
    plt.plot(a[0], a[1], 'ro', markersize=8)  # Plot red point for each iteration
    plt.draw()
    plt.pause(1)  # Pause for 1 second

plt.ioff()  # Turn off interactive mode
plt.show()
