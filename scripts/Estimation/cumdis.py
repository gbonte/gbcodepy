import numpy as np
import matplotlib.pyplot as plt

DN = [20, 21, 22, 20, 23, 25, 26, 25, 20, 23, 24, 25, 26, 29]
fig = plt.figure(figsize=(9, 4), layout="constrained")
axs = fig.subplots(1, 1)


axs.hist(DN, density=False, cumulative=True)


plt.show()
