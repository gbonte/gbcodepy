import numpy as np
import matplotlib.pyplot as plt

# probability transition A
def hmm_obs(A, B, p, times):
    # probability transition A
    N = A.shape[0]
    M = B.shape[1]
    
    q = np.empty(times, dtype=int)
    o = np.empty(times, dtype=int)
    
    # Note: In Python, state indices are 0-indexed.
    q[0] = np.random.choice(N, p=p.flatten())
    o[0] = np.random.choice(M, p=B[q[0], :])
    
    for t in range(1, times):
        q[t] = np.random.choice(N, p=A[q[t-1], :])
        o[t] = np.random.choice(M, p=B[q[t], :])
    
    return {'states': q, 'observations': o}

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
        alpha[i, 0] = p[i] * B[i, o[0]]
    
    if scale:
        S[0] = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / S[0]
    
    # Initialization alpha: alpha[i,1]=p(i)*B[i,o[1]]
    
    for t in range(1, T):
        for j in range(N):
            alpha[j, t] = 0
            for i in range(N):
                alpha[j, t] = alpha[j, t] + alpha[i, t-1] * A[i, j]
            alpha[j, t] = alpha[j, t] * B[j, o[t]]
        if scale:
            S[t] = np.sum(alpha[:, t])
            alpha[:, t] = alpha[:, t] / S[t]
    
    P = np.sum(alpha[:, T-1])
    if scale:
        # P is product of scaling factors
        P = np.prod(S)
    
    ############## BACKWARD PROCEDURE
    if not scale:
        for i in range(N):
            beta[i, T-1] = 1
        # Initialization beta: beta[i,T]=1
        
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[i, t] = 0
                for j in range(N):
                    beta[i, t] = beta[i, t] + beta[j, t+1] * A[i, j] * B[j, o[t+1]]
        P2 = 0
        for i in range(N):
            P2 = P2 + beta[i, 0] * alpha[i, 0]
        return {'prob': P, 'prob2': P2, 'alpha': alpha, 'beta': beta}
    else:
        return {'prob': P, 'alpha': alpha}

def hmm_vit(A, B, p, O):
    no_obs = len(O)
    N = A.shape[0]
    delta = np.zeros((N, no_obs))
    # delta[i,t]: highest probability along a single path
    # which accounts for the first t observations and ends in state i
    
    psi = np.zeros((N, no_obs), dtype=int)
    
    Q = np.empty(no_obs, dtype=int)
    
    
    ######### INITIALIZATION DELTA AND PSI
    for i in range(N):
        delta[i, 0] = p[i] * B[i, O[0]]
        psi[i, 0] = 0
    
    MM = np.zeros(N)
    for t in range(1, no_obs):
        for j in range(N):
            for i in range(N):
                MM[i] = delta[i, t-1] * A[i, j]
            mx = np.max(MM)      #  max{1<=i<=N} [delta[i,t-1] A[i,j]]
            ind = int(np.argmax(MM))
            psi[j, t] = ind  # arg max{1<=i<=N} [delta[i,t-1] A[i,j]]
            delta[j, t] = mx * B[j, O[t]]
    
    # TERMINATION
    P = np.max(delta[:, no_obs-1])
    ind = int(np.argmax(delta[:, no_obs-1]))
    
    # BACKTRACKING
    Q[no_obs-1] = ind
    
    for t in range(no_obs-2, -1, -1):
        Q[t] = psi[Q[t+1], t+1]
    return {'prob': P, 'states': Q, 'delta': delta, 'psi': psi}

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
                    alpha[j, k] = alpha[j, k] + alpha[i, k-1] * A[i, j]
                alpha[j, k] = alpha[j, k] * B[j, o[k]]
        
        for i in range(S):
            beta[i, T-1] = 1
        for k in range(T-2, -1, -1):
            for i in range(S):
                beta[i, k] = 0
                for j in range(S):
                    beta[i, k] = beta[i, k] + beta[j, k+1] * A[i, j] * B[j, o[k+1]]
        
        for k in range(0, T-1):
            P_OL[k] = 0
            for i in range(S):
                for j in range(S):
                    P_OL[k] = P_OL[k] + alpha[i, k] * A[i, j] * B[j, o[k+1]] * beta[j, k+1]
        
        for k in range(0, T-1):
            for i in range(S):
                for j in range(S):
                    xi[i, j, k] = alpha[i, k] * A[i, j] * B[j, o[k+1]] * beta[j, k+1] / P_OL[k]
                gamma[i, k] = np.sum(xi[i, :, k])
        
        for i in range(S):
            gamma[i, T-1] = 0
            for j in range(S):
                gamma[i, T-1] = gamma[i, T-1] + xi[j, i, T-2]
        
        for i in range(S):
            p_hat[i, 0] = gamma[i, 0]
            for j in range(S):
                s_xi = 0
                s_ga = 0
                for k in range(0, T-1):
                    s_xi = s_xi + xi[i, j, k]
                    s_ga = s_ga + gamma[i, k]
                A_hat[i, j] = s_xi / s_ga if s_ga != 0 else 0
        
        for j in range(S):
            for m in range(M):
                s_ga_num = 0
                s_ga = 0
                for k in range(T):
                    s_ga_num = s_ga_num + gamma[j, k] * (1 if o[k] == m else 0)
                    s_ga = s_ga + gamma[j, k]
                B_hat[j, m] = s_ga_num / s_ga if s_ga != 0 else 0
        
        A = A_hat.copy()
        B = B_hat.copy()
        p = p_hat.copy()
        
        Lik[it] = hmm_ev(A, B, p, o, scale=False)['prob']
    
    # p_hat<-p_hat/(sum(p_hat)) is commented out in original code
    return {'A': A_hat, 'B': B_hat, 'p': p_hat, 'lik': Lik}

# probability transition A
np.random.seed(0)
N = 2
A = np.random.rand(N, N)
for i in range(N):
    A[i, :] = A[i, :] / np.sum(A[i, :])

# probability output B
M = 3
B = np.random.rand(N, M)
for i in range(N):
    B[i, :] = B[i, :] / np.sum(B[i, :])

A2 = np.random.rand(N, N)
for i in range(N):
    A2[i, :] = A2[i, :] / np.sum(A2[i, :])

# probability output B
M = 3
B2 = np.random.rand(N, M)
for i in range(N):
    B2[i, :] = B2[i, :] / np.sum(B2[i, :])

p = np.random.rand(N, 1)
p = p / np.sum(p)

p2 = np.random.rand(N, 1)
p2 = p2 / np.sum(p2)

R = 500
C = np.empty(R)
CB = np.empty(R)
times = 100
seq = hmm_obs(A, B, p, times)

est_hmm = hmm_bw(A2, B2, p2, seq['observations'], no_it=1000)

plt.plot(est_hmm['lik'])
plt.xlabel("Iteration")
plt.ylabel("Likelihood")
plt.title("Baum-Welch Likelihood")
plt.show()
