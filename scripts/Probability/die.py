# "INFOF422 Statistical foundations of machine learning" course
#  package gbcode 
# Author: G. Bontempi
## MC computation of the variance of the outcome of a die with Nfaces

import random
import numpy as np

R = 100000
Nfaces = 6
Die = random.choices(list(range(1, Nfaces + 1)), k=R)
print('Var=',np.var(Die, ddof=1),'Std=',np.std(Die, ddof=1))
print('Var=',np.mean((Die-np.mean(Die))**2))
