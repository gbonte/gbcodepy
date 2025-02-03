## # "Statistical foundations of machine learning" software
# classification.py
# Author: G. Bontempi
## ts_preproc.py

import sys

# adding Folder_2 to the system path
sys.path.insert(0, '../Nonlinear')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils import predpy, Embed



# Sample time series data
# Parameters for the sinusoidal signal
frequency = 0.3  # Frequency in Hz
amplitude = 1  # Amplitude
phase = 0  # Phase shift in radians
duration = 30  # Duration in seconds
sampling_rate = 30  # Number of samples per second

# Time vector
time = np.linspace(0, duration, int(duration * sampling_rate))

# Generate the sinusoidal signal
tseries = amplitude * np.sin(2 * np.pi * frequency * time + phase)+np.random.normal(scale=0.1, size=len(time))
tseries2 = 0.5*amplitude * np.cos(3 * np.pi * frequency/2 * time + 2*phase)+np.random.normal(scale=0.1, size=len(time))

# Print the signal values
plt.plot(tseries)
plt.plot(tseries2)
plt.title('Sinusoidal Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


## Bivariate time series
TS=np.column_stack((tseries, tseries2))

lag=5
H=5


## Embedding time series in an input/output format
X,Y=Embed(TS,lag=lag,H=H)



### DIRECT fORECASTING

N=X.shape[0]
m=Y.shape[1]

methods=['rf_regr','piperf_regr','pipeknn_regr']

Yhat = np.zeros((N,m,len(methods)))


## number folds
K = 3
kf = KFold(n_splits=K, shuffle=False)

cnt=0
for mm in methods:
    print(mm)
    for train_index, test_index in kf.split(X):
        Xtrk=X[train_index,:]
        Ytrk=Y[train_index,:]#.reshape(len(train_index),)
        Yhat[test_index,:,cnt] = predpy(mm,Xtrk ,Ytrk , X[test_index],params={'nepochs':500, 'hidden':20})
        
    cnt=cnt+1


for mm in np.arange(len(methods)):
    nmse=[]
    for i in np.arange(m):
        nmse.append( np.mean((Yhat[:,i,mm] - Y[:,i])**2) / np.var(Y[:,i]))
        
    print(methods[mm], 'NMSE=',  np.mean(nmse))
    
    




