import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasaurus import datasaurus_dozen

# Group by dataset and calculate summary statistics
summary_stats = datasaurus_dozen.groupby('dataset').agg({
    'x': ['mean', 'std'],
    'y': ['mean', 'std']
}).reset_index()

summary_stats['corr_x_y'] = datasaurus_dozen.groupby('dataset').apply(lambda x: x['x'].corr(x['y']))
summary_stats.columns = ['dataset', 'mean_x', 'std_dev_x', 'mean_y', 'std_dev_y', 'corr_x_y']
print(summary_stats)

# Create facet plot
g = sns.FacetGrid(datasaurus_dozen, col="dataset", col_wrap=3, height=3)
g.map(plt.scatter, "x", "y")
g.fig.suptitle("Datasaurus Dozen", y=1.02)
plt.show()

# Get unique dataset names
names_datasets = datasaurus_dozen['dataset'].unique()
print(names_datasets)

# Filter data for 'dino' and another dataset
D1 = datasaurus_dozen[datasaurus_dozen['dataset'] == 'dino'][['x', 'y']]
D2 = datasaurus_dozen[datasaurus_dozen['dataset'] == names_datasets[8]][['x', 'y']]

print(f"Means D1: {D1.mean().values}")
print(f"Means D2: {D2.mean().values}")
print(f"Stds D1: {D1.std().values}")
print(f"Stds D2: {D2.std().values}")
print(f"Cor D1:\n{D1.corr()}")
print(f"Cor D2:\n{D2.corr()}")

# Plot D1 and D2
plt.figure(figsize=(10, 8))
plt.scatter(D1['x'], D1['y'], label='D1')
plt.scatter(D2['x'], D2['y'], color='red', label='D2')
plt.legend()
plt.title('Comparison of D1 and D2')
plt.show()

