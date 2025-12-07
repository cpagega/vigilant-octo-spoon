# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 19:26:45 2025

@author: Justin W.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

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

# import matplotlib.pyplot as plt
# plt.hist(y_prob, bins=50)
# plt.xlabel('Predicted Probability')
# plt.ylabel('Count')
# plt.title('Distribution of Predictions')
# plt.show()

# print(history.history.keys())
# print(history.history["accuracy"])
# fire_locs = df[df['label'] == 1]
# no_fire_locs = df[df['label'] == 0]

# plt.figure(figsize=(12, 6))
# plt.scatter(no_fire_locs['long'], no_fire_locs['lat'], 
#             c='blue', alpha=0.1, s=1, label='No Fire')
# plt.scatter(fire_locs['long'], fire_locs['lat'], 
#             c='red', alpha=0.3, s=1, label='Fire')
# plt.legend()
# plt.title('Geographic Distribution')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
#plt.show()

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


