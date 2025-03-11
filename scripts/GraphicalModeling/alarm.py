import numpy as np

# ## Script Alarm Bayesian Network example

# ## index in CP Tables 1-> True, 2-> False


pBT = 0.001  # ## a priori probability that B is true
P_B = np.array([pBT, 1 - pBT])

pET = 0.002  # ## a priori probability that E is true

P_E = np.array([pET, 1 - pET])

P_A_BE = np.zeros((2, 2, 2))  # ## CPT of A given B and E

P_A_BE[0, 0, 0] = 0.95
P_A_BE[0, 0, 1] = 0.94
P_A_BE[0, 1, 0] = 0.29
P_A_BE[0, 1, 1] = 0.001

P_A_BE[1, :, :] = 1 - P_A_BE[0, :, :]

P_J_A = np.zeros((2, 2))  # ## CPT of J given A
P_J_A[0, 0] = 0.9
P_J_A[0, 1] = 0.05
P_J_A[1, :] = 1 - P_J_A[0, :]

P_M_A = np.zeros((2, 2))  # ## CPT of M given A
P_M_A[0, 0] = 0.7
P_M_A[0, 1] = 0.01

P_M_A[1, :] = 1 - P_M_A[0, :]

Jo = np.full((2, 2, 2, 2, 2), np.nan)  # ## Jo[J,M,A,E,B]

for J in range(2):
    for M in range(2):
        for A in range(2):
            for E in range(2):
                for B in range(2):
                    Jo[J, M, A, E, B] = P_J_A[J, A] * P_M_A[M, A] * P_A_BE[A, B, E] * P_B[B] * P_E[E]

print(np.sum(Jo))  # ## Check that the joint probability is normalized

# ## Jo[J,M,A,E,B]

P_B_1 = np.sum(Jo[:, :, :, :, 0])    # P(B=1)
P_A_1_B_1 = np.sum(Jo[:, :, 0, :, 0])  # P(A=1 & B=1)
print("P(A=T|B=T)=", P_A_1_B_1 / P_B_1, "\n")

###
P_B_0 = np.sum(Jo[:, :, :, :, 1])    # P(B=0)
P_A_1_B_0 = np.sum(Jo[:, :, 0, :, 1])  # P(A=1 & B=0)
print("P(A=T|B=F)=", P_A_1_B_0 / P_B_0, "\n")

###
print("P(A=F|B=F)=", 1 - P_A_1_B_0 / P_B_0, "\n")

###
P_J_1 = np.sum(Jo[0, :, :, :, :])    # P(J=1)
P_J_1_B_1 = np.sum(Jo[0, :, :, :, 0])  # P(J=1 & B=1)

P_J_1_B_0 = np.sum(Jo[0, :, :, :, 1])
print("P(J=T|B=F)=", P_J_1_B_0 / P_B_0, "\n")

print("P(B=T|J=T)=", P_J_1_B_1 / P_J_1, "\n")

