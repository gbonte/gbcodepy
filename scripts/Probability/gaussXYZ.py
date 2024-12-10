# Statistical foundations of machine learning
# Python equivalent of gbcode R package
# Author: G. Bontempi

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
#from cmap import Colormap

# Visualizes different bivariate gaussians with different
# orientation and axis' length

x = np.arange(-10, 10.5, 0.5)
y = x

z = np.zeros((len(x), len(y)))

# th : rotation angle of the first principal axis
# ax1: length principal axis 1
# ax2: length principal axis 2

ax1 = 1

for th in np.arange(0, np.pi + np.pi/8, np.pi/8):
    for ax2 in [1, 2, 4, 8, 16]:
        
        Rot = np.array([[np.cos(th), -np.sin(th)], 
                        [np.sin(th), np.cos(th)]])  # rotation matrix
        A = np.array([[ax1, 0], 
                      [0, ax2]])
        Sigma = (Rot @ A) @ Rot.T
        E, V = np.linalg.eig(Sigma)
        print(f"Eigenvalue of the Variance matrix= {E}")
        print(Sigma)
        
        for i in range(len(x)):
            for j in range(len(y)):
                z[i,j] = multivariate_normal.pdf([x[i], y[j]], mean=[0, 0], cov=Sigma)
        
        z[np.isnan(z)] = 1
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, y)
        
        ax.plot_surface(X, Y, z, cmap='viridis_r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability Density')
        ax.view_init(elev=30, azim=30)
        plt.title(f"BIVARIATE; Rotation={th:.3f}; Axis 1={ax1}; Axis 2={ax2}")
        plt.show()
        input("Press [enter] to continue")

