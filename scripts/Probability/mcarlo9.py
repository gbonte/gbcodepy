# "Statistical foundations of machine learning" software
# Python implementation
# Author: G. Bontempi
# Birthday problem (from the book Introduction to Probability, Blitzstein and Hwang, page 31)
# "Birthday problem": probability that in a party with n participants at least two persons
# have the same birthday

import numpy as np
import matplotlib.pyplot as plt

R = 10**4  # number of Monte Carlo trials
Phat = []
seqN = np.arange(5, 101, 2)  # sequence from 5 to 100 with step 2

for n in seqN:  # number of party participants
    # Generate R trials of n random birthdays and count duplicates
    r = [np.max(np.bincount(np.random.randint(1, 366, n))) for _ in range(R)]
    # MC estimation of probability
    phat = sum(np.array(r) >= 2) / R
    Phat.append(phat)

plt.plot(seqN, Phat)
plt.xlabel('# participants')
plt.ylabel('Prob')
plt.show()
