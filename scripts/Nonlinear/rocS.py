import numpy as np
import streamlit as st
from scipy.stats import norm

# -----------------------------
# Translation of R functions
# -----------------------------

def knn_score(X, Y, Xts, k=25):
    """
    KNN score: for each test point, return the fraction of neighbors with label 1.
    X: (N, d)
    Y: (N,) in {0,1}
    Xts: (Nts, d)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Xts = np.asarray(Xts)

    N = X.shape[0]
    Nts = Xts.shape[0]
    Yhat = np.zeros(Nts)

    for i in range(Nts):
        q = Xts[i, :]
        # Euclidean distances
        d = np.sqrt(np.sum((X - q) ** 2, axis=1))
        idx = np.argsort(d)[:k]
        Yhat[i] = np.mean(Y[idx] == 1)
    return Yhat


def fct(X, P1=0.5, sdw=0.5, rng=None):
    """
    fct in R:
      Y = X[,1]^2 * X[,2] + X[,3]
      Y = scale(Y) + rnorm(N, sd=sdw)
      return 1{Y > quantile(Y, 1-P1)}
    """
    if rng is None:
        rng = np.random.default_rng()
    X = np.asarray(X)
    N = X.shape[0]

    Y = X[:, 0] ** 2 * X[:, 1] + X[:, 2]
    # scale: (Y - mean)/sd
    Y_scaled = (Y - Y.mean()) / Y.std(ddof=1)
    Y_noisy = Y_scaled + rng.normal(0, sdw, size=N)

    q = np.quantile(Y_noisy, 1 - P1)
    return (Y_noisy > q).astype(int)


# -----------------------------
# Core experiment (fixed data)
# -----------------------------

@st.cache_data
def generate_data(seed=1, N=1500, Nval=1000, Nts=250, n=3, P1=0.5, sdw=0.7, K=2):
    rng = np.random.default_rng(seed)

    # Training data
    X = rng.normal(size=(N, n))
    Y = fct(X, P1=P1, sdw=sdw, rng=rng)

    # Validation data
    Xval = rng.normal(size=(Nval, n))
    Yval = fct(Xval, P1=P1, sdw=sdw, rng=rng)

    # Test data
    Xts = rng.normal(size=(Nts, n))
    Yts = fct(Xts, P1=P1, sdw=sdw, rng=rng)

    # Scores from KNN
    ScoreTr = knn_score(X, Y, Xval, k=K)
    Score = knn_score(X, Y, Xts, k=K)

    return X, Y, Xval, Yval, Xts, Yts, ScoreTr, Score


def compute_threshold_from_validation(ScoreTr, Yval, LFP=1.0, LFN=0.5):
    """
    Nonparametric ROC on validation set and optimal threshold THstar.
    """
    TH = np.arange(0.0, 1.0 + 0.001, 0.001)

    Ival0 = np.where(Yval == 0)[0]
    Ival1 = np.where(Yval == 1)[0]
    NP = len(Ival1)
    NN = len(Ival0)

    P1 = NP / (NP + NN)
    P0 = NN / (NP + NN)

    TPR_list = []
    FPR_list = []
    Loss_list = []

    for th in TH:
        tpr = np.sum((ScoreTr > th) & (Yval == 1)) / NP
        fpr = np.sum((ScoreTr > th) & (Yval == 0)) / NN
        TPR_list.append(tpr)
        FPR_list.append(fpr)
        Loss_list.append(LFP * fpr * P0 + LFN * (1 - tpr) * P1)

    TPR = np.array(TPR_list)
    FPR = np.array(FPR_list)
    Loss = np.array(Loss_list)

    Ibest = np.argmin(Loss)
    THstar = TH[Ibest]
    FPRstar = FPR[Ibest]
    TPRstar = TPR[Ibest]

    return TH, TPR, FPR, Loss, THstar, FPRstar, TPRstar, P0, P1


def parametric_roc(Score, Yts, TH):
    """
    Parametric ROC curve as in your R code.
    """
    Its0 = np.where(Yts == 0)[0]
    Its1 = np.where(Yts == 1)[0]

    Score0 = Score[Its0]
    Score1 = Score[Its1]

    mu1 = Score1.mean()
    sd1 = Score1.std(ddof=1)
    mu0 = Score0.mean()
    sd0 = Score0.std(ddof=1)

    a = (mu1 - mu0) / sd1
    b = sd0 / sd1

    # FPR and TPR as functions of TH
    FPR = norm.cdf((mu0 - TH) / sd0)
    TPR = norm.cdf(a + b * norm.ppf(FPR))

    return FPR, TPR, mu0, sd0, mu1, sd1, a, b


# -----------------------------
# Streamlit app
# -----------------------------

st.title("Parametric ROC and Loss with Asymmetric Costs")

st.markdown(
    "Given a classification task (with parametric noise), it shows the parametric ROC curve " \
    "of a learning algoritmh (KNN)  builds  and lets you interactively change the **false negative loss $L_{FN}$"
    "(by assuming that $L_{FP}=1$)**."
)

# Sidebar controls
st.sidebar.header("Settings")

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=1, step=1)
K = st.sidebar.number_input("K for KNN", min_value=1, max_value=100, value=2, step=1)
sdw = st.sidebar.slider("Noise standard deviation", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
P1 = st.sidebar.slider("Target positive proportion $P_1$ ", 0.1, 0.9, 0.5, 0.05)

# LFN slider (this is your aLFN control)
LFN = st.sidebar.slider("Ratio $L_{FN}/L_{FP}$ ($L_{FN}: loss for false negative)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
LFP = 1.0

# Generate data and scores
X, Y, Xval, Yval, Xts, Yts, ScoreTr, Score = generate_data(
    seed=seed, N=1500, Nval=1000, Nts=250, n=3, P1=P1, sdw=sdw, K=K
)

# Threshold grid
TH = np.linspace(0, 1, 1001)

# Nonparametric ROC on validation to get THstar (not strictly needed for the dashboard,
# but kept for fidelity to your R code)
TH_val, TPR_val, FPR_val, Loss_val, THstar, FPRstar, TPRstar, P0_val, P1_val = \
    compute_threshold_from_validation(ScoreTr, Yval, LFP=LFP, LFN=LFN)

# Parametric ROC on test set
FPR, TPR, mu0, sd0, mu1, sd1, a, b = parametric_roc(Score, Yts, TH)

# Loss for current LFN
L = LFP * FPR + LFN * (1 - TPR)
Istar = np.argmin(L)
best_FPR = FPR[Istar]
best_TPR = TPR[Istar]
best_TH = TH[Istar]
best_L = L[Istar]

st.subheader("Parametric ROC curve (test set)")

st.markdown(
    f"- **LFP = {LFP}**, **LFN = {LFN}**  \n"
    f"- Optimal threshold (parametric ROC, current LFN): **TH ≈ {best_TH:.3f}**  \n"
    f"- At this point: **FPR ≈ {best_FPR:.3f}**, **TPR ≈ {best_TPR:.3f}**, **Loss ≈ {best_L:.3f}**"
)

# Plot ROC using matplotlib (Streamlit will render it)
import matplotlib.pyplot as plt

# -----------------------------
# Side‑by‑side layout for plots
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Parametric ROC curve (test set)")
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(FPR, TPR, label="Parametric ROC", color="blue")
    ax_roc.scatter(best_FPR, best_TPR, color="red", s=60, label="Optimal point")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"ROC (LFP={LFP}, LFN={LFN})")
    ax_roc.legend()
    ax_roc.grid(True)
    st.pyplot(fig_roc)

with col2:
    st.subheader("Loss function vs threshold")
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(TH, L, label=f"Loss (LFP={LFP}, LFN={LFN})", color="black")
    ax_loss.scatter(best_TH, best_L, color="red", s=60, label="Minimum loss")
    ax_loss.set_xlabel("Threshold")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss function")
    ax_loss.legend()
    ax_loss.grid(True)
    st.pyplot(fig_loss)


st.markdown(
    "Move the **$L_{FN}/L_{FP}$ slider** in the sidebar to see how the optimal point on the ROC curve "
    "and the loss curve change with the cost of false negatives."
)
