# Wildfire Prediction ML Project

This project trains a supervised model to predict wildfire likelihood using NASA FIRMS fire detections, Google Earth Engine climate datasets, and optional OpenWeather data. A pre-trained model is provided so you can run predictions immediately.

---

## Requirements

Python 3.10+  

Install dependencies using:

pip install -r requirements.txt

Create a .env file in the project root containing:

EE_PROJECT=your_earth_engine_project_id
WEATHER=your_openweather_api_key   # optional but recommended

If WEATHER is missing, predictions fall back to Earth Engine’s Real-Time Mesoscale Analysis dataset.

---

## Running Predictions (No Training Required)

Download the provided files:

- model.keras
- scaler.pkl

Place them in:

models/predict/

Start the API server:

fastapi run main.py

The server defaults to:

http://localhost:8000/

The frontend (Leaflet map) sends lat/long queries and displays predictions, metrics, and explanation text.

---

## Training the Model

### Option A — Use the Provided Dataset

Use:

Data/firms_ee_join_clean.csv

Train:

python models/train/supervised_model.py

This generates the metric plots for the frontend and writes model.keras and scaler.pkl to models/predict/.

---

## Training on New Data

### 1. Download FIRMS Data

Get a FIRMS CSV from:

https://firms.modaps.eosdis.nasa.gov/download/

Place it in the Data directory.

---

### 2. Generate the CONUS Sample File

Run:

python datacollectors/collector_main.py

Choose Option 1.

This produces a randomized CONUS sample of FIRMS detections (confidence > 40).  
The output CSV is written to the project root.

Upload this file to your Earth Engine assets.  
Once uploaded, the local copy can be deleted.

---

### 3. Generate the Full Feature Dataset (Earth Engine)

Run the collector again:

python datacollectors/collector_main.py

Choose Option 2.

This builds the combined FIRMS + climate dataset.  
Due to Earth Engine’s limited compute, large datasets may take several hours.

When complete, the dataset is exported to your Google Drive.  
Download it and place it in the Data directory.

---

### 4. Train the Model

Run:

python models/train/supervised_model.py

This:

- Trains the supervised ML model
- Produces the training metric plots
- Saves model.keras and scaler.pkl into models/predict/

---

## Running the Server

Once a model is available (pre-trained or newly trained):

fastapi run main.py

Default host:

http://localhost:8000/

The project is now ready for use.

