import numpy as np
import matplotlib.pyplot as plt
import math

def hmm_obs(A, B, p, times):
    # probability transition A
    N = A.shape[0]
    M = B.shape[1]

    q = np.zeros(times, dtype=int)
    o = np.zeros(times, dtype=int)

    q[0] = np.random.choice(N, p=p)
    o[0] = np.random.choice(M, p=B[q[0], :]/np.sum(B[q[0], :]))

    for t in range(1, times):
        q[t] = np.random.choice(N, p=A[q[t-1], :]/np.sum(A[q[t-1], :]))
        o[t] = np.random.choice(M, p=B[q[t], :]/np.sum(B[q[t], :]))
    return {"states": q, "observations": o}

def hmm_bw(A, B, p, o, no_it=10):
    # HMM Baum-Welch algorithm
    T = len(o)
    S = A.shape[0]
    M = B.shape[1]

    alpha = np.zeros((S, T))
    beta = np.zeros((S, T))
    gamma = np.zeros((S, T))
    Sa = np.zeros(T)
    Sb = np.zeros(T)
    P_OL = np.zeros(T)
    xi = np.zeros((S, S, T))
    p_hat = np.zeros((S, 1))
    A_hat = np.zeros((S, S))
    B_hat = np.zeros((S, M))

    Lik = np.zeros(no_it)

    for it in range(no_it):
        for i in range(S):
            alpha[i, 0] = p[i] * B[i, o[0]]

        for k in range(1, T):
            for j in range(S):
                alpha[j, k] = 0
                for i in range(S):
                    alpha[j, k] += alpha[i, k-1] * A[i, j]
                alpha[j, k] = alpha[j, k] * B[j, o[k]]

        for i in range(S):
            beta[i, T-1] = 1

        for k in range(T-2, -1, -1):
            for i in range(S):
                beta[i, k] = 0
                for j in range(S):
                    beta[i, k] += beta[j, k+1] * A[i, j] * B[j, o[k+1]]

        for k in range(T-1):
            P_OL[k] = 0
            for i in range(S):
                for j in range(S):
                    P_OL[k] += alpha[i, k] * A[i, j] * B[j, o[k+1]] * beta[j, k+1]

        for k in range(T-1):
            for i in range(S):
                for j in range(S):
                    xi[i, j, k] = alpha[i, k] * A[i, j] * B[j, o[k+1]] * beta[j, k+1] / P_OL[k]
                gamma[i, k] = np.sum(xi[i, :, k])

        for i in range(S):
            gamma[i, T-1] = 0
            for j in range(S):
                gamma[i, T-1] += xi[j, i, T-2]

        for i in range(S):
            p_hat[i, 0] = gamma[i, 0]
            for j in range(S):
                s_xi = 0
                s_ga = 0
                for k in range(T-1):
                    s_xi += xi[i, j, k]
                    s_ga += gamma[i, k]
                A_hat[i, j] = s_xi / s_ga if s_ga != 0 else 0

        for j in range(S):
            for m in range(M):
                s_ga_num = 0
                s_ga = 0
                for k in range(T):
                    s_ga_num += gamma[j, k] * (1 if o[k] == m else 0)
                    s_ga += gamma[j, k]
                B_hat[j, m] = s_ga_num / s_ga if s_ga != 0 else 0

        A = A_hat.copy()
        B = B_hat.copy()
        p = p_hat.flatten()

        Lik[it] = hmm_ev(A, B, p, o, scale=False)["prob"]

    #p_hat normalization commented out in original R code
    return {"A": A_hat, "B": B_hat, "p": p_hat, "lik": Lik}

def hmm_ev(A, B, p, o, scale=False):
    T = len(o)  # number of observations sequence
    N = A.shape[0]  # number of states

    alpha = np.zeros((N, T))
    # alpha[i,t]: P(i, o[1:t]): joint probability that at time t we are
    #             at state i and we have observed o[1:t]

    beta = np.zeros((N, T))
    # beta[i,t]: P(o[t+1:T]|i): conditional probability of observing
    #           the sequence o[t+1:T] given that we are at state i at time t

    Sa = np.zeros(T)
    Sb = np.zeros(T)
    S_arr = np.zeros(N)

    ############## FORWARD PROCEDURE

    for i in range(N):
        alpha[i, 0] = p[i] * B[i, o[0]]

    if scale:
        S_arr[0] = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / S_arr[0]

    # Initialization alpha: alpha[i,0]=p(i)*B[i,o[1]]

    for t in range(1, T):
        for j in range(N):
            alpha[j, t] = 0
            for i in range(N):
                alpha[j, t] += alpha[i, t-1] * A[i, j]
            alpha[j, t] = alpha[j, t] * B[j, o[t]]
        if scale:
            S_arr[t] = np.sum(alpha[:, t])
            alpha[:, t] = alpha[:, t] / S_arr[t]

    P = np.sum(alpha[:, T-1])
    if scale:
        P = 1 / np.prod(1 / S_arr)

    ############## BACKWARD PROCEDURE
    if not scale:
        for i in range(N):
            beta[i, T-1] = 1
        # Initialization beta: beta[i,T]=1

        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[i, t] = 0
                for j in range(N):
                    beta[i, t] += beta[j, t+1] * A[i, j] * B[j, o[t+1]]
        P2 = 0
        for i in range(N):
            P2 += beta[i, 0] * alpha[i, 0]
        return {"prob": P, "prob2": P2, "alpha": alpha, "beta": beta}
    else:
        return {"prob": P, "alpha": alpha}

# probability transition A
S = 2  # number of states
M = 6  # size  of the  observation domain

A = np.array([0.95, 0.05, 0.1, 0.9]).reshape((2, 2)).T
times = 50

# probability output B
B = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]).reshape((6, 2)).T

p = np.array([0.99, 0.01])

seq = hmm_obs(A, B, p, times)

maxl = -np.inf

### Several random initialisations of the BW 
best_hmm = None
for rep in range(1, 501):
    np.random.seed(rep)
    ## initialisation of BW algorithm
    Ainit = np.random.uniform(size=(S, S))
    for i in range(S):
        Ainit[i, :] = Ainit[i, :] / np.sum(Ainit[i, :])
    # probability output B
    Binit = np.random.uniform(size=(S, M))
    for i in range(S):
        Binit[i, :] = Binit[i, :] / np.sum(Binit[i, :])
    pinit_value = np.random.uniform()
    pinit = np.array([pinit_value, 1 - pinit_value])

    est_hmm = hmm_bw(Ainit, Binit, pinit, seq["observations"], no_it=20)

    if not np.isnan(np.max(est_hmm["lik"])):
        if np.max(est_hmm["lik"]) > maxl:
            maxl = np.max(est_hmm["lik"])
            print(math.log(maxl))
            best_hmm = est_hmm

print("A=")
print(A)

print("\n Ahat=")
print(best_hmm["A"])
print("B=")
print(B)
print("Bhat=")
print(best_hmm["B"])

plt.plot(np.log(best_hmm["lik"]), label="EM Log Likelihood")
plt.title("EM Log Likelihood")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.show()

