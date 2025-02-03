import numpy as np
from numpy.random import seed, choice
import matplotlib.pyplot as plt

# "Statistical foundations of machine learning" software
# Python translation of gbcode
# Author: G. Bontempi
# pfizer.py
# Bootstrap confidence interval of vaccine efficacy

N = 21500
# Numbers from https://hbr.org/2020/12/covid-19-vaccine-trials-are-a-case-study-on-the-challenges-of-data-literacy
# How efficacy rate was computed: first, count the number of people who developed Covid-19 in the vaccinated group.
# Second, divide that by the number of people who developed it in the placebo group.
# Third, subtract that quotient from 1, and you'll get the efficacy rate.

V = np.zeros(N)
V[:8] = 1  # First 8 elements set to 1

NV = np.zeros(N)
NV[:86] = 1  # First 86 elements set to 1

B = 10000  # number of bootstrap replicates
eff = []

for b in range(B):
    seed(b)  # Set random seed for reproducibility
    I = choice(N, size=N, replace=True)  # sampling with replacement
    eff.append(np.sum(V[I]) / np.sum(NV[I]))

eff = np.array(eff)
print("Confidence interval=",np.quantile(1 - eff, [0.025, 0.975]))

# Create histogram
plt.figure()
plt.hist(100 * (1 - eff), bins=30)
plt.xlabel('% Efficacy')
plt.title('')
plt.show()
