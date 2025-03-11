import numpy as np

def hmm_vit(A, B, p, O):
    # no.obs: number of observations
    no_obs = len(O)
    # N: number of states (number of rows in A)
    N = A.shape[0]
    # Initialize delta: highest probability along a single path for the first t observations ending in state i
    delta = np.zeros((N, no_obs))
    # Initialize psi
    psi = np.zeros((N, no_obs), dtype=int)
    # Q: array to store the most probable state sequence
    Q = np.zeros(no_obs, dtype=int)

    ######### INITIALIZATION DELTA AND PSI
    for i in range(N):
        delta[i, 0] = p[i] * B[i, O[0]]
        psi[i, 0] = 0

    MM = np.zeros(N)
    for t in range(1, no_obs):
        for j in range(N):
            for i in range(N):
                MM[i] = delta[i, t-1] * A[i, j]
            mx = np.max(MM)      # max{1<=i<=N} [delta[i,t-1] * A[i,j]]
            ind = np.argmax(MM)  # arg max{1<=i<=N} [delta[i,t-1] * A[i,j]]
            psi[j, t] = ind      # store index of the maximum
            delta[j, t] = mx * B[j, O[t]]
    
    # TERMINATION
    P = np.max(delta[:, no_obs-1])
    ind = np.argmax(delta[:, no_obs-1])

    # BACKTRACKING
    Q[no_obs-1] = ind
    for t in range(no_obs-2, -1, -1):
        Q[t] = psi[Q[t+1], t+1]
    
    return {'prob': P, 'states': Q, 'delta': delta, 'psi': psi}

# X= Good, Neutral, Bad (transition probabilities matrix)
A = np.array([[0.2, 0.3, 0.5],
              [0.2, 0.2, 0.6],
              [0.0, 0.2, 0.8]])

B = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.4, 0.3],
              [0.0, 0.1, 0.9]])

# Observation sequence; adjusted to 0-indexing because Python indexing starts at 0
O = [0, 2, 1, 0, 2]
p = np.array([1/3, 1/3, 1/3])  # same a priori sequence

H = hmm_vit(A, B, p, O)
states_map = ["Good", "Neutral", "Bad"]
print("Viterbi sequence=", " ".join([states_map[state] for state in H['states']]))

