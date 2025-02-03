# "Statistical foundations of machine learning" software
# Python implementation
# Original Author: G. Bontempi
# mcarlo8.py
# Matching card problem (from the book 
# Introduction to Probability, Blitzstein and Hwang, page 31)
# "Consider a well-shuffled deck of n cards, labeled 1 through n. You flip over the cards one by one, saying the
# numbers 1 through n as you do so. You win the game if, at some point, the number
# you say aloud is the same as the number on the card being flipped over (for example,
# if the 7th card in the deck has the label 7). What is the probability of winning?"

import numpy as np

R = 10**5  # number of Monte Carlo trials
n = 20     # number of cards

# Simulate the game R times
matches = []
for _ in range(R):
    # Generate shuffled deck and compare with positions
    deck = np.random.permutation(n) + 1  # +1 because we want 1 to n
    positions = np.arange(1, n+1)
    matches.append(np.sum(deck == positions) >= 1)

# Calculate probability
phat = np.mean(matches)

# Print results
print(f"MC phat={phat}; Analytical= 1-1/e={1-1/np.exp(1)}")
