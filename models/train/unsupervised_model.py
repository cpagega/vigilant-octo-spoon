# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 19:26:45 2025

@author: Bosslady
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

import json


def fire_likeness_score(env_values):
    """
    env_values: list/array with values in the SAME ORDER as predictor_cols.
    Returns a score in (0,1], higher = more similar to historical fire conditions.
    Or at least that is the idea I am a big dumb dumb
    """
    env_values = np.asarray(env_values, dtype="float32").reshape(1, -1)
    env_scaled = scaler.transform(env_values)

    recon = autoencoder.predict(env_scaled)
    err = np.mean((env_scaled - recon) ** 2)

    score = float(np.exp(-err / (error_scale + 1e-8)))
    return score


# 1) Load CSV
df = pd.read_csv("firms_ee_feature_join.csv")

# 2) Time features from 'date'
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["doy"] = df["date"].dt.dayofyear

# 3) Choose predictors
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
    "doy",
]

# Drop ther rows with missing predictor values
df = df.dropna(subset=predictor_cols)

X = df[predictor_cols].values.astype("float32")

# 4) Train/validation split (unsupervised, so there should be no labels if I remember)
X_train, X_val = train_test_split(
    X, test_size=0.2, random_state=42
)

# 5) Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 6) Build the autoencoder inspired by https://www.ibm.com/think/topics/autoencoder
input_dim = X_train.shape[1]

inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(16, activation="relu")(x)
bottleneck = layers.Dense(8, activation="relu", name="bottleneck")(x)
x = layers.Dense(16, activation="relu")(bottleneck)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(input_dim, activation="linear")(x)

autoencoder = keras.Model(inputs, outputs)
autoencoder.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mse"
)

autoencoder.summary()

# 7) Train: learn to reconstruct fire conditions
history = autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=40,
    batch_size=256,
    verbose=1
)

# 8) Compute reconstruction error distribution on training data
X_train_recon = autoencoder.predict(X_train)
train_errors = np.mean((X_train - X_train_recon) ** 2, axis=1)

print("Train error stats:")
print("  min:", float(train_errors.min()))
print("  max:", float(train_errors.max()))
print("  mean:", float(train_errors.mean()))
print("  median:", float(np.median(train_errors)))
print("  90th percentile:", float(np.percentile(train_errors, 90)))
print("  99th percentile:", float(np.percentile(train_errors, 99)))

# Use a higher percentile as scale so scores aren't microscopic
error_scale = np.percentile(train_errors, 90)
print("Using error_scale (90th percentile):", float(error_scale))



# Example: score the first row from the dataset
example_env = X[0]  # original (unscaled) values from df
print("Example fire-likeness score:", fire_likeness_score(example_env))

# 9) Export scaler and model weights 

# Save scaler parameters
scaler_mean = scaler.mean_.tolist()
scaler_scale = scaler.scale_.tolist()

# Extract weights for each Dense layer (except for the Input)
layers_export = []
for layer in autoencoder.layers:
    weights = layer.get_weights()
    if not weights:
        continue
    W, b = weights
    layers_export.append({
        "name": layer.name,
        "W": W.tolist(),
        "b": b.tolist()
    })

model_export = {
    "features": predictor_cols,
    "scaler_mean": scaler_mean,
    "scaler_scale": scaler_scale,
    "layers": layers_export,
    "error_scale": float(error_scale)
}

with open("fire_autoencoder_model.json", "w") as f:
    json.dump(model_export, f, indent=2)

print("Saved model description to fire_autoencoder_model.json")
