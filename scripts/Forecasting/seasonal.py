import os
import pkg_resources
import pyreadr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose



# Load the RData file equivalent to R's load("bench.temp.200.Rdata")
result = pyreadr.read_r("bench.temp.200.Rdata")
Temp = result["Temp"].to_numpy()

def remNA(x):
    # Function to remove NA values equivalent to R's remNA
    x = np.array(x)
    return x[~np.isnan(x)]

def detectSeason(x, Ls, debug):
    # Function to detect seasonality equivalent to R's detectSeason
    period = 12  # Assumed seasonal period
    decomposition = seasonal_decompose(x, model='additive', period=period, extrapolate_trend='freq')
    return {"spattern": decomposition.seasonal, "strend": decomposition.trend}

# Create D equivalent to R's D=remNA(Temp[1:400,3])
D = remNA(Temp[0:400, 2])

# Create S equivalent to R's S=detectSeason(D, Ls=length(D), debug=FALSE)
S = detectSeason(D, Ls=len(D), debug=False)

# Set up plotting equivalent to R's par(mfrow=c(4,1), mar=2*c(1,1,1,1))
fig, axs = plt.subplots(4, 1, figsize=(8, 12))
plt.subplots_adjust(hspace=0.5)

# Plot D as a line equivalent to R's plot(D, type="l")
axs[0].plot(range(len(D)), D, linestyle='-', color='blue')
# Add red line for S$spattern + S$strend equivalent to R's lines(S$spattern+S$strend, col="red")
axs[0].plot(range(len(D)), S["spattern"] + S["strend"], linestyle='-', color='red')

# Plot residual equivalent to R's plot(D-S$spattern+S$strend, type="l", main="residual")
axs[1].plot(range(len(D)), D - S["spattern"] + S["strend"], linestyle='-')
axs[1].set_title("residual")

# Plot seasonal component equivalent to R's plot(S$spattern, type="l", main="seasonal")
axs[2].plot(range(len(D)), S["spattern"], linestyle='-')
axs[2].set_title("seasonal")

# Plot trend component equivalent to R's plot(S$strend, type="l", main="trend")
axs[3].plot(range(len(D)), S["strend"], linestyle='-')
axs[3].set_title("trend")

plt.show()
