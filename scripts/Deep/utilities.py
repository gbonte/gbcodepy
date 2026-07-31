import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import zipfile
import os
import copy

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """
    Deterministic, index-preserving imputer.
    """

    def __init__(self, method="time"):
        self.method = method

    def fit(self, X, y=None):
        self._validate(X)
        return self

    def transform(self, X):
        self._validate(X)

        if self.method == "time":
            return X.interpolate(method="time", limit_direction="both")
        elif self.method == "ffill":
            return X.ffill().bfill()
        else:
            raise ValueError("Unknown imputation method")

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be DataFrame")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")

from statsmodels.tsa.seasonal import STL
from sklearn.base import BaseEstimator, TransformerMixin


class InvertibleSTLTransform(BaseEstimator, TransformerMixin):
    """
    X = residual + trend + seasonal
    """

    def __init__(self, period, robust=True):
        self.period = period
        self.robust = robust

    def fit(self, X, y=None):
        self._validate(X)

        self.trend_ = {}
        self.seasonal_ = {}
        self.columns_ = X.columns
        self.index_ = X.index

        for col in X.columns:
            series = X[col]
            stl = STL(series, period=self.period, robust=self.robust)
            res = stl.fit()

            self.trend_[col] = res.trend
            self.seasonal_[col] = res.seasonal

        return self

    def transform(self, X):
        self._validate(X)

        Z = X.copy()
        for col in self.columns_:
            Z[col] = (
                Z[col]
                - self.trend_[col]
                - self.seasonal_[col]
            )
        return Z

    def inverse_transform(self, Z):
        X_rec = Z.copy()
        for col in self.columns_:
            X_rec[col] = (
                X_rec[col]
                + self.seasonal_[col]
                + self.trend_[col]
            )
        return X_rec

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be DataFrame")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")

from sklearn.pipeline import Pipeline


class ReversiblePipeline(Pipeline):
    """
    Pipeline that supports inverse_transform.
    """

    def inverse_transform(self, X):
        Xt = X
        for name, step in reversed(self.steps):
            if hasattr(step, "inverse_transform"):
                Xt = step.inverse_transform(Xt)
        return Xt



# ============================================================
# 1. Dynamic Factor Model (DFM)
# ============================================================
class DFM(nn.Module):
    def __init__(self, input_dim, factor_dim, hidden_dim, H):
        super().__init__()
        self.H = H
        
        self.encoder = nn.Linear(input_dim, factor_dim)
        self.rnn = nn.GRU(factor_dim, hidden_dim, batch_first=True)
        self.factor_to_factor = nn.Linear(hidden_dim, factor_dim)
        self.decoder = nn.Linear(factor_dim, input_dim)

    def forward(self, x):
        f = self.encoder(x)
        _, h = self.rnn(f)
        f_prev = f[:, -1, :]

        preds = []
        for _ in range(self.H):
            out, h = self.rnn(f_prev.unsqueeze(1), h)
            f_next = self.factor_to_factor(out.squeeze(1))
            preds.append(f_next)
            f_prev = f_next

        F = torch.stack(preds, dim=1)
        return self.decoder(F)




class TransformerDFM(nn.Module):
    def __init__(
        self,
        input_dim,
        factor_dim,
        model_dim,
        H,
        num_heads=2,
        num_layers=2,
        dropout=0.0
    ):
        super().__init__()
        self.H = H

        # 1. Encoder: observations → latent factors
        self.encoder = nn.Linear(input_dim, factor_dim)

        # 2. Project factors to Transformer dimension
        self.factor_to_model = nn.Linear(factor_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 3. Predict H future factors from final state
        self.model_to_factors = nn.Linear(model_dim, H * factor_dim)

        # 4. Decode factors → observations
        self.decoder = nn.Linear(factor_dim, input_dim)

    def forward(self, x):
        """
        x: (batch, T, input_dim)
        returns: (batch, H, input_dim)
        """

        # Encode observations
        f = self.encoder(x)                     # (B, T, factor_dim)
        z = self.factor_to_model(f)              # (B, T, model_dim)

        # Transformer over time
        h = self.transformer(z)                  # (B, T, model_dim)
        h_last = h[:, -1]                        # (B, model_dim)

        # Predict future latent factors
        F = self.model_to_factors(h_last)        # (B, H * factor_dim)
        F = F.view(-1, self.H, f.size(-1))       # (B, H, factor_dim)

        # Decode
        return self.decoder(F)


# ============================================================
# 2. dfm_forecast(): Training + Validation + Forecasting
# ============================================================
def dfm_forecast(
    ts_train,
    H=12,
    window=48,
    factors=3,
    hidden=64,
    epochs=30,
    lr=1e-3,
    val_split=0.2,
    plot_training=False,
    transformer=False,
    verbose=False
):
    """
    Train a Dynamic Factor Model only on ts_train and return H-step forecast.
    """

    # ---------- Normalize ----------
    mu = ts_train.mean(axis=0)
    sd = ts_train.std(axis=0) + 1e-8
    ts_n = (ts_train - mu) / sd

    # ---------- Build windows ----------
    X, Y = [], []
    for t in range(len(ts_n) - window - H):
        X.append(ts_n[t:t+window])
        Y.append(ts_n[t+window:t+window+H])

    X = np.array(X)
    Y = np.array(Y)

    # ---------- Train/Val split ----------
    n = len(X)
    split = int((1 - val_split) * n)
    X_train, Y_train = X[:split], Y[:split]
    X_val,   Y_val   = X[split:], Y[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    Y_val   = torch.tensor(Y_val,   dtype=torch.float32)

    # ---------- Model ----------

    if transformer:
        model = TransformerDFM(ts_train.shape[1], factors, hidden, H)
    else:   
        model = DFM(ts_train.shape[1], factors, hidden, H)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

   

    best_val_loss = float("inf")
    best_model_state = None



    for ep in range(epochs):

        # ------------------
        # Train
        # ------------------
        model.train()
        optimizer.zero_grad()

        pred = model(X_train)
        loss_train = loss_fn(pred, Y_train)

        loss_train.backward()
        optimizer.step()

        # ------------------
        # Validation
        # ------------------
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            loss_val = loss_fn(pred_val, Y_val)

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())

        # ------------------
        # Store best model
        # ------------------
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_model_state = copy.deepcopy(model.state_dict())

        # ------------------
        # Logging
        # ------------------
        if verbose and (ep % 5 == 0):
            print(
                f"Epoch {ep:03d} | "
                f"Train={loss_train.item():.4f} | "
                f"Val={loss_val.item():.4f}"
            )

    # ------------------
    # Restore best model
    # ------------------
    model.load_state_dict(best_model_state)
    if verbose:
        print(f"Best validation loss: {best_val_loss:.4f}")


    
    # ---------- Plot training curves ----------
    if plot_training:
        plt.figure(figsize=(10,5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    # ---------- Forecast the next H steps ----------
    last_window = ts_n[-window:]
    last_window = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred_norm = model(last_window).numpy()[0]

    return pred_norm * sd + mu



# ============================================================
# 3. Load Real Multivariate Dataset (UCI Air Quality)
# ============================================================
def load_air_quality():
    url = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
    zip_path = "air_quality.zip"

    if not os.path.exists(zip_path):
        print("Downloading Air Quality dataset...")
        urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("air_quality")

    df = pd.read_csv("air_quality/AirQualityUCI.csv", sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna()

    variables = [
        "CO(GT)", "PT08.S1(CO)", "C6H6(GT)",
        "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)"
    ]

    return df[variables].astype(float).values, variables



# ============================================================
# 4. Baselines
# ============================================================
def naive_last(ts_train, H):
    return np.tile(ts_train[-1], (H, 1))

def naive_mean(ts_train, H, window=48):
    m = ts_train[-window:].mean(axis=0)
    return np.tile(m, (H, 1))

def rmse(a, b):
    return np.sqrt(((a - b)**2).mean())

from sklearn.ensemble import GradientBoostingRegressor

import lightgbm as lgb


params = {
    "objective": "regression",
    "learning_rate": 0.05,
    "max_depth": -1,
    "verbosity": -1,
}




def gb_direct_forecast(ts_train, H=12, window=48, verbose=False):
    """
    Gradient Boosting baseline using DIRECT multi-step forecasting:
    - One model per variable per horizon step.
    - Input: last `window` lag values for all variables.
    - Output: H-step forecast (H, N).
    
    Parameters
    ----------
    ts_train : np.ndarray (T_train, N)
    H : int
        Forecast horizon.
    window : int
        Number of lags used as input.
    
    Returns
    -------
    forecast : np.ndarray (H, N)
        H-step forecast in original space.
    """

    T, N = ts_train.shape
    X_list = []
    Y_list = [[] for _ in range(H)]  # Y_list[h][sample] = X_{t+h}

    # ---------- Build supervised dataset ----------
    for t in range(window, T - H):
        X_list.append(ts_train[t - window : t].flatten())  # flatten multivariate window
        
        for h in range(H):
            Y_list[h].append(ts_train[t + h])

    X = np.array(X_list)                      # shape: (samples, window*N)
    Y = [np.array(y) for y in Y_list]         # list of H arrays: (samples, N)

    # ---------- Train H*N models (direct strategy) ----------
    #model = 
    gb_models = [
        [lgb.LGBMRegressor(**params) for _ in range(N)]
        for _ in range(H)
    ]

    
    for h in range(H):
        for var in range(N):
            gb_models[h][var].fit(X, Y[h][:, var])
            if verbose:
                print(f"Trained GB for horizon {h+1}, variable {var+1}/{N}")

    # ---------- Forecast from last window ----------
    last_window = ts_train[-window:].flatten().reshape(1, -1)
    forecast = np.zeros((H, N))    
    for h in range(H):        
        for j in range(N):            
            forecast[h, j] = gb_models[h][j].predict(last_window)[0]    
    return forecast


