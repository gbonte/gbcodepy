import numpy as np
from sklearn.preprocessing import scale


def corXY(X,Y):
    if X.shape[0] !=Y.shape[0]:
        raise Exception('error')
    if Y.shape[1] !=1:
        raise Exception('only for univariate targets')
    XY=np.hstack((X,Y.reshape(-1,1)))
    score=np.corrcoef(XY.T)[-1,:-1]
    return (score)

def rankrho(X, Y, nmax=5, regr=False):
    score=np.abs(corXY(X,Y))
    return(np.argsort(-score)[:nmax])


# Function cor2I2: converts correlation values using a specific transformation
def cor2I2(rho):
    rho = np.minimum(rho, 1 - 1e-5)
    rho = np.maximum(rho, -1 + 1e-5)
    return -0.5 * np.log(1 - rho**2)

def mrmr(X, Y, nmax=5):
    n = X.shape[1]
    N = X.shape[0]

    Iy = cor2I2(corXY(X, Y))
    print(np.round(Iy,3))
    
    
    CCx = np.corrcoef(X, rowvar=False)
    Ix = cor2I2(CCx)
    print(np.round(Ix,2))
    # Select the feature that maximizes Iy
    subs = [int(np.argmax(Iy))]
    # Run loop from current selection length to min(n-1, nmax) (R loop: for (j in length(subs):min(n-1,nmax)))
    for _ in range(len(subs), min(n - 1, nmax) + 1):
        mrmr_vals = np.full(n, -np.inf)
        # Identify candidate features not yet selected
        candidates = [i for i in range(n) if i not in subs]
        #print("cand=",candidates)
        if len(subs) < (n - 1):
            if len(subs) > 1:
                for cand in candidates:
                    # Compute the mean of -Ix for the selected features for candidate cand
                    mrmr_vals[cand] = Iy[cand] + np.mean(-Ix[np.ix_(subs, [cand])])
            else:
                for cand in candidates:
                    mrmr_vals[cand] = Iy[cand] -Ix[subs[0], cand]
                    #print(cand, Iy[cand], Ix[subs[0], cand])
        else:
            for cand in candidates:
                mrmr_vals[cand] = -np.inf
        # Select the candidate with the maximum mrmr value
        
        s = int(np.argmax(mrmr_vals))
        #print(mrmr_vals, ":",np.argmax(mrmr_vals) )
        subs.append(s)
        #print(subs)
         
    # Return the first nmax features (convert 0-indexed to 1-indexed)
    return (np.array(subs[:nmax])).tolist()

np.random.seed(0)
N=200
n=6
sdw=0.2
relfeat=[1,2,3,4]
X= np.random.normal(loc=0, scale=1, size=N * n).reshape(N, n)
X[:,1]=0.2*X[:,0]+np.random.normal(loc=0, scale=1, size=N )
X[:,2]=0.2*X[:,1]+np.random.normal(loc=0, scale=1, size=N )
X[:,3]=-X[:,2]+np.random.normal(loc=0, scale=0.1, size=N )
X[:,4]=0.5*-X[:,2]+np.random.normal(loc=0, scale=0.1, size=N )
Sigma = np.dot(X.transpose(),X)

# Generate N samples from a multivariate normal distribution with covariance Sigma and then scale
X = scale(np.random.multivariate_normal(mean=np.zeros(n), cov=Sigma, size=N))

Y=np.random.normal(loc=0, scale=sdw, size=N )

for f in relfeat:
    Y=Y+X[:,f]
Y=Y.reshape(N, 1)

rankedfs=rankrho(X,Y,len(relfeat))
mrmrfs=mrmr(X,Y,len(relfeat))
print('Relevant features=',relfeat, 'Ranked features=',rankedfs, 'MRMR features=', mrmrfs)