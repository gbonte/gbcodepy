# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python [conda env:ml-env]
#     language: python
#     name: conda-env-ml-env-py
# ---

# %% [markdown]
# # Classification

# %%
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tabicl import TabICLClassifier
import pandas as pd
penguins = sns.load_dataset("penguins")
X = penguins.drop(columns="species")
y = penguins["species"]


# Remove rows with missing values
data = X.join(y).dropna()
X = data.drop(columns="species")
y = data["species"]

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)


clf = TabICLClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


clf2 = RandomForestClassifier()
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)


# %%
print("MISCL_TAB=", sum(y_test != y_pred)/len(y_test), "MISCL_RF=" , sum(y_test != y_pred2)/len(y_test))

# %%
penguins = sns.load_dataset("penguins")
X = penguins.drop(columns="species")
y = penguins["species"]
type(X)

# %%
y_pred2

# %%
import random, numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
random.seed(42)
np.random.seed(42)

df = fetch_ucirepo(id=275).data.original
df["timestamp"] = (
    pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
)

# %%
series = df[["timestamp", "cnt"]].rename(columns={"cnt": "target"})
series.head().round(1)


# %% [markdown]
# # Regression

# %%
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from tabicl import TabICLRegressor

wine = fetch_ucirepo(id=186)
X = wine.data.features
y = wine.data.targets["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# %%
reg = TabICLRegressor(random_state=1)
reg.fit(X_train, y_train)

y_pred_tabicl = reg.predict(X_test)
y_pred_tabicl[:5]

# %%

# %% [markdown]
# # Forecasting

# %%

# %%
import matplotlib.pyplot as plt

window = series.iloc[48:(48 + 24 * 7)]
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(window["timestamp"], window["target"], color="#3b6ea5", lw=1)
ax.set(xlabel="Time", ylabel="Rentals")
ax.spines[["top", "right"]].set_visible(False)
fig.autofmt_xdate()

# %%
from tabicl import TabICLForecaster

horizon = 24 * 7 # one week of hourly data
context = series.iloc[:-horizon]
actual = series.iloc[-horizon:]

forecaster = TabICLForecaster(
    max_context_length=len(context), tabicl_config={"n_estimators": 4}
)
pred = forecaster.predict_df(context, future_df=actual[["timestamp"]])
pred.head(3).round(0)

# %%
forecast = pred.reset_index()

fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(forecast["timestamp"], forecast[0.1], forecast[0.9],
                color="lightgrey", alpha=0.6,
                label="10th-90th percentile")
ax.plot(forecast["timestamp"], forecast[0.5], color="darkgrey", lw=1.5,
        label="median forecast")
ax.plot(actual["timestamp"], actual["target"], color="#c0392b",
        lw=1.5, linestyle="--", label="actual")
ax.set(xlabel="Time", ylabel="Rentals")
ax.legend(frameon=False)
fig.autofmt_xdate()

# %%
