# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 19:26:45 2025

@author: Bosslady
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json

from tensorflow import keras
from tensorflow.keras import layers

# 1) Load data
df = pd.read_csv("firms_ee_feature_join.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["doy"] = df["date"].dt.dayofyear

# 2) Predictor columns (same as before)
predictor_cols = [
    "Ground_Heat_Flux_surface",
    "Plant_Canopy_Surface_Water_surface",
    "Temperature_surface",
    "Vegetation_Type_surface",
    "Vegetation_surface",
    "pdsi",
    "precipitation",
    "tmax",
    "tmin",
    "u-component_of_wind_hybrid",
    "v-component_of_wind_hybrid",
    "month",
    "doy"
]

# 3) Target = FIRMS confidence
target_col = "confidence"

df = df.dropna(subset=predictor_cols + [target_col])

X = df[predictor_cols].values.astype("float32")
y = df[target_col].values.astype("float32")

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6) Regression neural network
input_dim = X_train.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="linear") # regression output
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# 7) Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=256,
    verbose=1
)

# 8) Evaluate
y_pred = model.predict(X_test).ravel()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nSupervised model results:")
print("MSE:", mse)
print("R^2:", r2)

# 9) Save model parameters

export = {
    "features": predictor_cols,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
}

# Save weights layer-by-layer
weights = []
for layer in model.layers:
    arr = layer.get_weights()
    if len(arr) == 2:
        W, b = arr
        weights.append({
            "W": W.tolist(),
            "b": b.tolist()
        })
export["layers"] = weights

with open("fire_supervised_regression_model.json", "w") as f:
    json.dump(export, f, indent=2)

print("Saved supervised model to fire_supervised_regression_model.json")
