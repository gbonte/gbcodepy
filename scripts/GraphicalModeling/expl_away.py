import numpy as np

# Script Explaining Away effect

# index in CP Tables 1-> True, 2-> False

P_C_1 = 0.1  # a priori probability that C is true
P_C = np.array([P_C_1, 1 - P_C_1])

P_V_1 = 0.6  # a priori probability that V is true
P_V = np.array([P_V_1, 1 - P_V_1])

P_H_CV = np.zeros((2, 2, 2))  # CPT of H given C and V

P_H_CV[0, 0, 0] = 0.95
P_H_CV[0, 0, 1] = 0.8
P_H_CV[0, 1, 0] = 0.8
P_H_CV[0, 1, 1] = 0.1

P_H_CV[1, :, :] = 1 - P_H_CV[0, :, :]

Jo = np.empty((2, 2, 2))  # Jo[H,C,V]

for H in range(2):
    for C in range(2):
        for V in range(2):
            Jo[H, C, V] = P_H_CV[H, C, V] * P_C[C] * P_V[V]

print(np.sum(Jo))  # Check that the joint probability is normalized

# Jo[H,C,V]

# Independence of C and V
P_C_1_V_1 = np.sum(Jo[:, 0, 0])    # P(C=1 & V=1)
print("P(C=T)=", P_C_1, "; P(C=T|V=T)=", P_C_1_V_1 / P_V_1)

P_H_1 = np.sum(Jo[0, :, :])    # P(H=1)
P_C_1_H_1 = np.sum(Jo[0, 0, :])  # P(C=1 & H=1)
print("P(C=T|H=T)=", P_C_1_H_1 / P_H_1)

P_H_1_V_1 = np.sum(Jo[0, :, 0])    # P(H=1 & V=1)
P_C_1_H_1_V_1 = Jo[0, 0, 0]  # P(H=1 & C=1 & V=1)
print("P(C=T|H=T,V=T)=", P_C_1_H_1_V_1 / P_H_1_V_1)

P_H_1_V_0 = np.sum(Jo[0, :, 1])    # P(H=1 & V=F)
P_C_1_H_1_V_0 = Jo[0, 0, 1]  # P(H=1 & C=1 & V=F)
print("P(C=T|H=T,V=F)=", P_C_1_H_1_V_0 / P_H_1_V_0)
