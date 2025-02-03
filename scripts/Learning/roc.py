import numpy as np
import matplotlib.pyplot as plt

# "INFOF422 Statistical foundations of machine learning" course
# Python translation of the R package gbcode by G. Bontempi

mu_p = 1
sd_p = 1

mu_n = -1
sd_n = 1

# Create a sequence from -10 to 10 with step 0.01
TT = np.arange(-10, 10.01, 0.01)

# Initialize arrays for False Positive Rate (FPR), Sensitivity (SE), Precision (PR), and Alert Level (AL)
FPR = np.zeros(len(TT))
SE = np.zeros(len(TT))
PR = np.zeros(len(TT))
AL = np.zeros(len(TT))

N = 2000
# Generate normally distributed data for positive and negative classes
DNp = np.random.normal(mu_p, sd_p, int(N/2))
DNn = np.random.normal(mu_n, sd_n, int(N/2))

# Loop over each threshold in TT and compute metrics
for idx, thr in enumerate(TT):
    # Count False Negatives and True Positives for the positive class
    FN = np.sum(DNp < thr)
    TP = np.sum(DNp > thr)
    # Count False Positives and True Negatives for the negative class
    FP = np.sum(DNn > thr)
    TN = np.sum(DNn < thr)
    
    # Calculate metrics, ensuring denominator is not zero where applicable
    FPR[idx] = FP / (FP + TN) if (FP + TN) > 0 else 0
    SE[idx] = TP / (TP + FN) if (TP + FN) > 0 else 0
    PR[idx] = TP / (TP + FP) if (TP + FP) > 0 else 0
    AL[idx] = (TP + FP) / N

# Plotting the results in three subplots
plt.figure(figsize=(18, 5))

# ROC curve: Sensitivity (TPR) vs. FPR
plt.subplot(1, 3, 1)
plt.plot(FPR, SE, color="red", label="ROC curve")
plt.plot(FPR, FPR, color="blue", linestyle='--', label="No-skill")
plt.title("ROC curve")
plt.ylabel("SE (TPR)")
plt.xlabel("FPR")
plt.legend()

# Precision-Recall (PR) curve: Precision vs. Sensitivity
plt.subplot(1, 3, 2)
plt.plot(SE, PR, color="red")
plt.title("PR curve")
plt.xlabel("SE (TPR)")
plt.ylabel("Precision")

# Lift curve: Sensitivity (TPR) vs. Alert Level (percentage alerts)
plt.subplot(1, 3, 3)
plt.plot(AL, SE, color="red", label="Lift curve")
plt.plot(AL, AL, color="blue", linestyle='--', label="No-skill")
plt.title("Lift curve")
plt.ylabel("SE (TPR)")
plt.xlabel("% alerts")
plt.legend()

plt.tight_layout()
plt.show()
