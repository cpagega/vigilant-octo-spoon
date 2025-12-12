# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 19:26:45 2025

@author: Justin W.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
from pathlib import Path

from tensorflow import keras


# 1) Load data
df = pd.read_csv("Data\\firms_ee_join_clean.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["doy"] = df["date"].dt.dayofyear

# 2) Predictor columns
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
target_col = "label"

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

# Save the scaler
with open("models/predict/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 6) Regression neural network
input_dim = X_train.shape[1]

model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid") 
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy",
             "AUC",
             "Precision",
             "Recall"
             ]
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
y_prob = model.predict(X_test).ravel()         # probabilities in [0,1]
y_pred = (y_prob >= 0.5).astype(int)           # convert to class labels

print(y_prob[:10])

preds = model.predict(X_test)
print("Prediction range:", y_prob.min(), y_prob.max())
print("Predictions > 0.5:", (y_prob > 0.5).sum())
print("Predictions < 0.5:", (y_prob < 0.5).sum())


print(history.history.keys())

# ---- 1. Loss ----
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/metrics/plot_loss.png", dpi=150)


# ---- 2. Accuracy ----
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/metrics/plot_accuracy.png", dpi=150)


# ---- 3. AUC ----
plt.figure()
plt.plot(history.history["AUC"], label="Train AUC")
plt.plot(history.history["val_AUC"], label="Val AUC")
plt.xlabel("Epoch")
plt.ylabel("ROC AUC")
plt.title("Training vs Validation AUC")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/metrics/plot_auc.png", dpi=150)

# This saves "history.history" to "history.json"
history_out = {
    k: [float(v) for v in vals] # make sure everything is JSON-serializable
    for k, vals in history.history.items()
}

with open("models/train/history.json", "w") as f:
    json.dump(history_out, f, indent=2)

# print("Saved training history to", ("history.json").resolve())

# Then it saves the confusion matrix values to "confusion_matrix.json"
cm = confusion_matrix(y_test, y_pred)
cm_list = cm.tolist()

with open("models/train/confusion_matrix.json", "w") as f:
    json.dump(cm_list, f, indent=2)

# print("Saved confusion matrix to", ("confusion_matrix.json").resolve())

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nSupervised model results (binary classifier):")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("ROC AUC:", auc)

# 9) Save model parameters

model.save("models/predict/model.keras")


