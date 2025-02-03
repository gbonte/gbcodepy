# "INFOF422 Statistical foundations of machine learning" course
# Python implementation
# Original Author: G. Bontempi
# Translated to Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

placebo = np.array([9243, 9671, 11792, 13357, 9055, 6290, 12412, 18806])
oldpatch = np.array([17649, 12013, 19979, 21816, 13850, 9806, 17208, 29044])
newpatch = np.array([16449, 14614, 17274, 23798, 12560, 10157, 16570, 26325])

data = pd.DataFrame({
    'placebo': placebo,
    'oldpatch': oldpatch,
    'newpatch': newpatch
})

N = len(data)
B = 4000

theta_hat = abs(np.mean(data['newpatch']) - np.mean(data['oldpatch'])) / \
           (np.mean(data['oldpatch']) - np.mean(data['placebo']))

thetaB = np.zeros(B)

for b in range(B):
    Db = data.sample(n=N, replace=True)
    thetaB[b] = abs(np.mean(Db['newpatch']) - np.mean(Db['oldpatch'])) / \
                (np.mean(Db['oldpatch']) - np.mean(Db['placebo']))

plt.figure(figsize=(10, 6))
plt.hist(thetaB, bins=30,color='lightblue')
plt.title(f"Bias={abs(theta_hat - np.mean(thetaB)):.2f}; Stdev={np.std(thetaB):.2f}")
plt.axvline(x=theta_hat, color='red')
plt.axvline(x=0.2, color='green')
plt.show()

print(f"Probability that theta_hat > 0.2 = {np.sum(thetaB > 0.2) / B}")
