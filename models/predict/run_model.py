import tensorflow as tf
import pandas as pd


model = tf.keras.models.load_model("models\\predict\\model.keras")

# input schema
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

# map input data to schema
def build_input_row(instance):
    df = pd.DataFrame([instance])
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear
    return df[predictor_cols]

def make_prediction(instance):
    X = build_input_row(instance)
    y_pred = model.predict(X)
    return y_pred