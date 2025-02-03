# "Statistical foundations of machine learning" software
# Python package gbcodepy 
# Author: G. Bontempi
# script kmeans.py


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def dist2(X1, X2):
    # X1: [N1, n] and X2: [N2, n]
    N1 = X1.shape[0]
    n = X1.shape[1] if X1.ndim > 1 else 1
    n2 = X2.shape[1] if X2.ndim > 1 else 1
    N2 = X2.shape[0]

    if n != n2:
        print("\n n=", n)
        print("\n n2=", n2)
        raise ValueError('Matrix sizes do not match.')
    
    y = np.zeros((N1, N2))
    
    if n == 1:
        for i in range(N1):
            x = np.ones((N2, 1)) * float(X1[i])
            # Ensure X2 is treated as a 1D array for subtraction
            if X2.ndim > 1:
                x2 = X2.flatten()
            else:
                x2 = X2
            y[i, :] = np.abs(x.flatten() - x2)
    else:
        if N1 < N2:
            for i in range(N1):
                x = np.ones((N2, 1)) * X1[i, :]
                diff = x - X2
                y[i, :] = np.sum(diff**2, axis=1)
        else:
            for j in range(N2):
                x = np.ones((N1, 1)) * X2[j, :]
                diff = x - X1
                y[:, j] = np.sum(diff**2, axis=1)
                
    return np.sqrt(y)

N = 500

mu1 = [0, 0]
mu2 = [6, 0]
mu3 = [4, 4]
mu4 = [1, 1]

D1 = np.column_stack((np.random.normal(mu1[0], 1, N), np.random.normal(mu1[1], 1, N)))
D2 = np.column_stack((np.random.normal(mu2[0], 1, N), np.random.normal(mu2[1], 1, N)))
D3 = np.column_stack((np.random.normal(mu3[0], 1, N), np.random.normal(mu3[1], 1, N)))
D4 = np.column_stack((np.random.normal(mu4[0], 1, N), np.random.normal(mu4[1], 1, N)))

D = np.vstack((D1, D2, D3, D4))

m = 3
Seeds = np.random.normal(0, 1, (m, 2))

plt.figure()
plt.scatter(D[:, 0], D[:, 1])
plt.xlabel("g1")
plt.ylabel("g2")
plt.show(block=False)


for it in range(1, 51):
    Dist = dist2(D, Seeds)
    # np.argmin returns 0-indexed indices; adjust comparisons accordingly
    Is = np.argmin(Dist, axis=1)
    
    for i in range(1, m+1):
        Isi = np.where(Is == (i - 1))[0]
        
        if i == 1:
            plt.clf()
            plt.scatter(D[Isi, 0], D[Isi, 1], color="red")
            plt.xlim(np.min(D[:, 0]), np.max(D[:, 0]))
            plt.ylim(np.min(D[:, 1]), np.max(D[:, 1]))
            plt.title("Iteration=" + str(it) + " No. clusters=" + str(m))
            plt.xlabel("g1")
            plt.ylabel("g2")
        elif i == 2:
            plt.scatter(D[Isi, 0], D[Isi, 1], color="blue")
        elif i == 3:
            plt.scatter(D[Isi, 0], D[Isi, 1], color="green")
        elif i == 4:
            plt.scatter(D[Isi, 0], D[Isi, 1], color="yellow")
        elif i == 5:
            plt.scatter(D[Isi, 0], D[Isi, 1], color="darkgrey")
        elif i == 6:
            plt.scatter(D[Isi, 0], D[Isi, 1], color="orange")
        
        if Isi.size > 0:
            Seeds[i - 1, :] = np.mean(D[Isi, :], axis=0)
    
    plt.draw()
    plt.pause(0.001)
    
    
# End of script
