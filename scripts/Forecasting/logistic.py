
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

TT = 500
k = 3.78

y = [0.5]

for t in range(1, TT + 1):  
    yt = y[-1]          
    yt1 = k * yt * (1 - yt)  
    y.append(yt1)       
    
N = len(y)  

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

axes[0].plot(y, linestyle='-', marker='')  
axes[0].set_title("y over time")


plot_pacf(np.array(y), ax=axes[1])
axes[1].set_title("PACF of y")

axes[2].scatter(y[:-1], y[1:])
axes[2].set_xlabel("y(t)")
axes[2].set_ylabel("y(t+1)")
axes[2].set_title("conditional distribution")

rnorm_series = np.random.randn(N)
plot_pacf(rnorm_series, ax=axes[3])
axes[3].set_title("PACF of rnorm(N)")

plt.show()
