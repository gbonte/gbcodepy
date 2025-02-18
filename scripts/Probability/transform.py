import numpy as np
## # "Statistical foundations of machine learning" software
# transform.py
# Author: G. Bontempi
## Visualisation of the transformation of a random variable

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

# Set a random seed for reproducibility
np.random.seed(42)

R=550000
# Generate data: x is Uniform
#x = np.random.normal(0, 1, R)
Low=1
Up=3 #2.15
x = np.random.uniform(Low, Up, R)

y = np.sin(x) ## g(x)

# Create a figure with a GridSpec layout.
# The layout is arranged such that:
# - The main scatter plot (x vs. g(x) is in the bottom-left.
# - The top panel (sharing x-axis) displays the histogram of x.
# - The right panel (sharing y-axis) displays the histogram of x^2 (y) with horizontal orientation.
fig = plt.figure(figsize=(8, 8))
gs = GridSpec(4, 4, figure=fig)

# Scatter plot for x vs. x^2
ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_scatter.scatter(x, y, alpha=0.5)
ax_scatter.set_xlabel(" z")
ax_scatter.set_ylabel("g(z)")  # Using unicode superscript 2
ax_scatter.axvline(np.mean(x), color='black', linestyle='dashed', linewidth=2)
ax_scatter.axhline(np.sin(np.mean(x)), color='black', linestyle='dashed', linewidth=2)


# Histogram of x on the top (marginal histogram along the x-axis)
ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
ax_histx.hist(x, bins=30, color='grey', edgecolor='black', density=True)
# Remove x tick labels on the histogram to avoid clutter
plt.setp(ax_histx.get_xticklabels(), visible=False)
ax_histx.set_ylabel("Count")
ax_histx.set_title("Histogram of z")
ax_histx.axvline(np.mean(x), color='black', linestyle='dashed', linewidth=2)


xs = np.linspace(x.min(), x.max(), 300)
density=np.ones_like(xs)/(Up-Low)
# Superimpose the density function.
ax_histx.plot(xs, density, color='red', lw=2, label='Density Function')



# Histogram of g(x) on the right (marginal histogram along the y-axis)
ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
ax_histy.hist(y, bins=30, orientation='horizontal', color='grey', edgecolor='black', density=True)
# Remove y tick labels on the histogram to avoid clutter
plt.setp(ax_histy.get_yticklabels(), visible=False)
ax_histy.set_xlabel("Count")
ax_histy.set_title("Histogram of g(z)", pad=20)
ax_histy.axhline(np.mean(y), color='grey', linestyle='dashed', linewidth=2)
ax_histy.axhline(np.sin(np.mean(x)), color='black', linestyle='dashed', linewidth=2)


kde = gaussian_kde(y)
ys = np.linspace(y.min(), y.max(), 300)
density = kde(ys)

# Superimpose the density function.
ax_histy.plot(density, ys, color='red',  lw=2, label='Density Function')



# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
