import numpy as np

# Helper function to mimic R's sample function with 1-indexed output
def sample(n, size, prob):
    # In R, sample(n, 1, prob=prob) returns a number in 1,...,n (1-indexed)
    # In numpy, we simulate this by choosing from np.arange(1, n+1)
    return np.random.choice(np.arange(1, n+1), size=size, p=prob)

# First definition of hmm.obs (first occurrence)
def hmm_obs(A, B, p, times):
    # probability transition A
    N = A.shape[0]
    M = B.shape[1]

    q = np.empty(times, dtype=int)
    o = np.empty(times, dtype=int)

    q[0] = sample(N, 1, prob=p.ravel())[0]
    o[0] = sample(M, 1, prob=B[q[0]-1, :])[0]

    for t in range(1, times):
        q[t] = sample(N, 1, prob=A[q[t-1]-1, :])[0]
        o[t] = sample(M, 1, prob=B[q[t]-1, :])[0]
    return {"states": q, "observations": o}

# Second definition of hmm.obs (duplicate as in the original R code)
def hmm_obs(A, B, p, times):
    # probability transition A
    N = A.shape[0]
    M = B.shape[1]

    q = np.empty(times, dtype=int)
    o = np.empty(times, dtype=int)

    q[0] = sample(N, 1, prob=p.ravel())[0]
    o[0] = sample(M, 1, prob=B[q[0]-1, :])[0]

    for t in range(1, times):
        q[t] = sample(N, 1, prob=A[q[t-1]-1, :])[0]
        o[t] = sample(M, 1, prob=B[q[t]-1, :])[0]
    return {"states": q, "observations": o}

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
        # In R: delta[i,1] <- p[i]*B[i,O[1]]
        # O[0] in Python corresponds to O[1] in R since O is 1-indexed in R and our O values remain 1-indexed.
        delta[i, 0] = p[i] * B[i, O[0]-1]
        psi[i, 0] = 0

    MM = np.empty(N)
    for t in range(1, no_obs):
        for j in range(N):
            for i in range(N):
                MM[i] = delta[i, t-1] * A[i, j]
            mx = np.max(MM)      #  max{1<=i<=N} [delta[i,t-1] A[i,j]]
            ind = int(np.argmax(MM))  # which.max(MM)
            psi[j, t] = ind + 1   # Store as 1-indexed state
            delta[j, t] = mx * B[j, O[t]-1]

    # TERMINATION
    P = np.max(delta[:, no_obs-1])
    ind = int(np.argmax(delta[:, no_obs-1]))

    # BACKTRACKING
    Q[no_obs-1] = ind + 1
    for t in range(no_obs-2, -1, -1):
        Q[t] = psi[Q[t+1]-1, t+1]
    return {"prob": P, "states": Q, "delta": delta, "psi": psi}

# Set random seed for reproducibility
np.random.seed(2)
S = 5
M = 5

# random probability transition A
A = np.random.uniform(0, 1, (S, S))
for i in range(S):
    A[i, :] = A[i, :] / np.sum(A[i, :])

# random probability output B
B = np.random.uniform(0, 1, (S, M))
for i in range(S):
    B[i, :] = B[i, :] / np.sum(B[i, :])

## initial state distribution
p = np.random.uniform(0, 1, (S, 1))
p = p / np.sum(p)

R = 100
C = np.empty(R)
C_ran = np.empty(R)

for r in range(R):
    TT = 200
    seq_hmm = hmm_obs(A, B, p, TT)

    # Viterbi filtering
    vit = hmm_vit(A, B, p.ravel(), seq_hmm["observations"])

    C[r] = np.sum(vit["states"] == seq_hmm["states"]) / TT
    Q_ran = np.random.choice(np.arange(1, S+1), size=TT, replace=True)
    C_ran[r] = np.sum(Q_ran == seq_hmm["states"]) / TT

print("Viterbi match " + str(np.mean(C)))
print("Random match " + str(np.mean(C_ran)))
