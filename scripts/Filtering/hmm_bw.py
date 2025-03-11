import numpy as np

def hmm_ev(A, B, p, o, scale=False):
    # HMM evaluation function to compute probability of observation sequence
    # Using the forward algorithm without scaling (scale=False)
    T = len(o)
    S = A.shape[0]
    alpha = np.zeros((S, T))
    
    # Initialization step
    # Adjust observation index: subtract 1 since o values are assumed to be 1-indexed.
    for i in range(S):
        alpha[i, 0] = p[i] * B[i, int(o[0]) - 1]
        
    # Induction step
    for t in range(1, T):
        for j in range(S):
            alpha[j, t] = 0
            for i in range(S):
                alpha[j, t] += alpha[i, t - 1] * A[i, j]
            alpha[j, t] *= B[j, int(o[t]) - 1]
    
    # Termination step: likelihood is sum of alpha at time T-1
    prob = np.sum(alpha[:, T - 1])
    return {"prob": prob}

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
        # Forward algorithm: Initialization
        for i in range(S):
            alpha[i, 0] = p[i] * B[i, int(o[0]) - 1]

        # Forward algorithm: Induction
        for k in range(1, T):
            for j in range(S):
                alpha[j, k] = 0
                for i in range(S):
                    alpha[j, k] += alpha[i, k - 1] * A[i, j]
                alpha[j, k] *= B[j, int(o[k]) - 1]

        # Backward algorithm: Initialization
        for i in range(S):
            beta[i, T - 1] = 1

        # Backward algorithm: Induction
        for k in range(T - 2, -1, -1):
            for i in range(S):
                beta[i, k] = 0
                for j in range(S):
                    beta[i, k] += beta[j, k + 1] * A[i, j] * B[j, int(o[k + 1]) - 1]

        # Compute P_OL for time steps 0 to T-2
        for k in range(T - 1):
            P_OL[k] = 0
            for i in range(S):
                for j in range(S):
                    P_OL[k] += alpha[i, k] * A[i, j] * B[j, int(o[k + 1]) - 1] * beta[j, k + 1]

        # Compute xi and gamma for time steps 0 to T-2
        for k in range(T - 1):
            for i in range(S):
                for j in range(S):
                    xi[i, j, k] = (alpha[i, k] * A[i, j] * B[j, int(o[k + 1]) - 1] * beta[j, k + 1]) / P_OL[k]
                gamma[i, k] = np.sum(xi[i, :, k])
        
        # Special computation for gamma at time T-1
        for i in range(S):
            gamma[i, T - 1] = 0
            for j in range(S):
                gamma[i, T - 1] += xi[j, i, T - 2]
        
        # Update p_hat and A_hat
        for i in range(S):
            p_hat[i, 0] = gamma[i, 0]
            for j in range(S):
                s_xi = 0
                s_ga = 0
                for k in range(T - 1):
                    s_xi += xi[i, j, k]
                    s_ga += gamma[i, k]
                A_hat[i, j] = s_xi / s_ga

        # Update B_hat
        for j in range(S):
            for m in range(1, M + 1):
                s_ga_num = 0
                s_ga = 0
                for k in range(T):
                    s_ga_num += gamma[j, k] * (1.0 if int(o[k]) == m else 0)
                    s_ga += gamma[j, k]
                B_hat[j, m - 1] = s_ga_num / s_ga

        # Update A, B, and p with the estimates
        A = A_hat.copy()
        B = B_hat.copy()
        p = p_hat.copy()

        # Evaluate the likelihood using the hmm_ev function
        Lik[it] = hmm_ev(A, B, p, o, scale=False)["prob"]

    # Return the estimated parameters and likelihood history
    return {"A": A_hat, "B": B_hat, "p": p_hat, "lik": Lik}
    
# Example usage:
# A = np.array([[0.7, 0.3], [0.4, 0.6]])
# B = np.array([[0.5, 0.5], [0.1, 0.9]])
# p = np.array([0.6, 0.4])
# o = [1, 2, 1, 1, 2]  # Observations (assumed 1-indexed)
# result = hmm_bw(A, B, p, o, no_it=10)
# print(result)
    
if __name__ == "__main__":
    # Sample test (uncomment to run)
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.5, 0.5], [0.1, 0.9]])
    p = np.array([0.6, 0.4])
    o = [1, 2, 1, 1, 2]  # Observations (assumed 1-indexed)
    result = hmm_bw(A, B, p, o, no_it=10)
    print("Estimated A:\n", result["A"])
    print("Estimated B:\n", result["B"])
    print("Estimated p:\n", result["p"])
    print("Likelihood history:\n", result["lik"])
    
# End of translation
