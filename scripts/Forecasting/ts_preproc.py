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
import pandas as pd
from sklearn.impute import SimpleImputer
import sklearn



###  Sample time series data
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

# Print the signal values
plt.plot(tseries)
plt.title('Sinusoidal Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
plt.clf()

### Multivariate Jena series

TS0=pd.read_csv("jena_climate.csv")
#print(dataset.__doc__)
print(TS0.columns)
TS0.pop('Date Time')

N,m=TS0.shape

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(TS0)
TS=imp_mean.transform(TS0)
TS=TS[:min(N,500),:]
tseries=sklearn.preprocessing.scale(TS)
print(np.var(TS,axis=0))
N,m=TS.shape
print("N=",N,"m=",m)


for i in np.arange(m):
    plt.plot(tseries[:,i])
    plt.title('Jena: '+TS0.columns[i])
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.show()


