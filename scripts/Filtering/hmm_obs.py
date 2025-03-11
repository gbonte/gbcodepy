import numpy as np

class hmm:
    @staticmethod
    def obs(A, B, p, times):
        # probability transition A
        N = A.shape[0]
        M = B.shape[1]

        q = [0] * times
        o = [0] * times

        # q[1] in R corresponds to the first element; sample from 1:N using probability vector p
        q[0] = np.random.choice(np.arange(1, N+1), p=p)
        # o[1] in R corresponds to the first observation; sample from 1:M using probability vector B[q[1],]
        o[0] = np.random.choice(np.arange(1, M+1), p=B[q[0]-1, :])

        for t in range(1, times):
            # q[t] is sampled based on A[q[t-1],] (adjust for 1-indexing by subtracting 1)
            q[t] = np.random.choice(np.arange(1, N+1), p=A[q[t-1]-1, :])
            # o[t] is sampled based on B[q[t],] (adjust for 1-indexing by subtracting 1)
            o[t] = np.random.choice(np.arange(1, M+1), p=B[q[t]-1, :])
        return {'states': q, 'observations': o}
        
# Example usage (can be removed or commented out if not needed)
A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.1, 0.9], [0.8, 0.2]])
p = np.array([0.6, 0.4])
times = 10
result = hmm.obs(A, B, p, times)
print(result)
