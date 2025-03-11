import numpy as np

# Evaluation algorithm:
# given an HMM model lambda=(A,B,p) and an observation sequence o
# it computes the probability P(o|lambda)
def hmm_ev(A, B, p, o, scale=False):
    T = len(o)
    # number observations sequence

    N = A.shape[0]
    # number states

    alpha = np.zeros((N, T))
    # alpha[i,t]: P(i,o[1:t]): joint probability that at time t we are
    #             at state i and we have observed o[1:t]

    beta = np.zeros((N, T))
    # beta[i,t]: P(o[t+1:T]|i): conditional probability of observing
    #           the sequence o[t+1:T] given that we are at state i at time t

    Sa = np.zeros(T)
    Sb = np.zeros(T)
    S = np.zeros(T)
    
    ############## FORWARD PROCEDURE

    for i in range(N):
        alpha[i,0] = p[i] * B[i, o[0] - 1]
    # Initialization alpha: alpha[i,1]=p(i)*B[i,o[1]]

    if scale:
        S[0] = np.sum(alpha[:,0])
        alpha[:,0] = alpha[:,0] / S[0]

    for t in range(1, T):
        for j in range(N):
            alpha[j,t] = 0
            for i in range(N):
                alpha[j,t] = alpha[j,t] + alpha[i,t-1] * A[i,j]
            alpha[j,t] = alpha[j,t] * B[j, o[t] - 1]
        if scale:
            S[t] = np.sum(alpha[:,t])
            alpha[:,t] = alpha[:,t] / S[t]

    P = np.sum(alpha[:, T-1])
    if scale:
        P = 1 / np.prod(1 / S)

    ############## BACKWARD PROCEDURE
    if not scale:
        for i in range(N):
            beta[i, T-1] = 1
        # Initialization beta: beta[i,T]=1

        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[i,t] = 0
                for j in range(N):
                    beta[i,t] = beta[i,t] + beta[j, t+1] * A[i,j] * B[j, o[t+1] - 1]

        P2 = 0
        for i in range(N):
            P2 = P2 + beta[i,0] * alpha[i,0]
        return {'prob': P, 'prob2': P2, 'alpha': alpha, 'beta': beta}
    else:
        return {'prob': P, 'alpha': alpha}
    
# End of hmm_ev function

