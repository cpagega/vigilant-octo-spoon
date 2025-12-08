from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import StringIO
import csv, json
from geojson import Feature, FeatureCollection, Point
import requests
from datetime import datetime, timezone, timedelta
import ee
import models.predict.run_model as model
import math
from dotenv import load_dotenv
import os

"""
@author: Chris
"""

load_dotenv()

ee_project = os.getenv("EE_PROJECT")
weather_key = os.getenv("WEATHER")

# Gooogle Earth Engine platform
ee.Authenticate()
ee.Initialize(project=ee_project)
print(ee.String('Hello from the Earth Engine servers!').getInfo())

app = FastAPI()
app.mount("/index", StaticFiles(directory="static", html=True), name="static")

# retrieves the data from the image at the specified point
def reduce_at_point(img, lat, lon):
    geom = ee.Geometry.Point([lon, lat])
    return img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=geom,
        scale=img.projection().nominalScale(),
        maxPixels=1e7,
    )

# convert wind speed and direciton into the component vectors
def wind_to_uv(speed, deg):
    theta = math.radians(deg)
    u = -speed * math.sin(theta) # eastward component
    v = -speed * math.cos(theta) # northward component 
    return u,v

# None handling for EE Dictionary types
def ee_num(val, default=0.0):
    if val is None: 
        return default 
    return ee.Number(val).getInfo()

# Current weather data from OpenWeather API
def get_current_weather(lat,lon): 
    try:
        r = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "units": "metric", "appid": weather_key},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


# retrieve the latest real-time mesoanalysis wind speeds - only if weather station data is unavailable
def get_rtma(lat,lon):
    img = (ee.ImageCollection('NOAA/NWS/RTMA')
        .select(["UGRD","VGRD"])
        .sort("system:time_start",False)
        .first()
    )
    vals = reduce_at_point(img,lat,lon)
    u = vals.get("UGRD")
    v = vals.get("VGRD")
    return ee_num(u), ee_num(v)

# retrieve the latest precipitation accumulation - only if weather station data is unavailable
def get_precip(lat,lon):
    img = (
            ee.ImageCollection("NOAA/CPC/Precipitation")
            .select(['precipitation'])
            .sort("system:time_start", False)
            .first()
        )
    val = reduce_at_point(img,lat,lon)
    precip = val.get("precipitation")
    return ee_num(precip)

# retrieve the latest drought severity index
def get_pdsi(lat,lon):
    img = (
            ee.ImageCollection("GRIDMET/DROUGHT")
            .select(['pdsi'])
            .sort("system:time_start", False)
            .first()
        )
    val = reduce_at_point(img,lat,lon)
    pdsi = val.get("pdsi")
    return ee_num(pdsi)

# retrieve the latest max and min temperatures - only if weather station data is unavailable
def get_cpc_temps(lat,lon):
    img = (
            ee.ImageCollection("NOAA/CPC/Temperature")
            .select(['tmax','tmin'])
            .sort("system:time_start", False)
            .first()
        )
    vals = reduce_at_point(img,lat,lon)
    tmin = vals.get("tmin")
    tmax = vals.get("tmax")
    return ee_num(tmin, default=None), ee_num(tmax, default=None)

# retrieve the latest climate data 
def get_cfsr_data(lat,lon):
    img = (
            ee.ImageCollection("NOAA/CFSR")
            .select(['Plant_Canopy_Surface_Water_surface',
                     'Ground_Heat_Flux_surface',
                     'Temperature_surface',
                     'Vegetation_surface',
                     'Vegetation_Type_surface'
                     ])
            .sort("system:time_start", False)
            .first()        
            )
    vals = reduce_at_point(img,lat,lon)
    pcsws   = vals.get('Plant_Canopy_Surface_Water_surface')
    ghfs    = vals.get('Ground_Heat_Flux_surface')
    ts      = vals.get('Temperature_surface')
    vs      = vals.get('Vegetation_surface')
    vts     = vals.get('Vegetation_Type_surface')

    return ee_num(pcsws), ee_num(ghfs), ee_num(ts,default=None), ee_num(vs), ee_num(vts)

# retrieve and process the different data sources into a single instance to make a prediction on
def build_prediction_set(lat,lon):
    prediction_set = {}
    weather_data = get_current_weather(lat,lon)
    print("Weather API Response")
    print(weather_data)
    if weather_data is not None and "main" in weather_data:
        main = weather_data["main"]
        # temps from boths sources should be in C
        if "temp_min" in main and "temp_max" in main:
            tmin,tmax = main["temp_min"], main["temp_max"]
        else:
            tmin,tmax = get_cpc_temps(lat,lon)
        if "wind" in weather_data:
            wind = weather_data["wind"]
            u,v = wind_to_uv(wind["speed"],wind["deg"])
        else:
            u,v = get_rtma(lat,lon)
    else:
        tmin,tmax = get_cpc_temps(lat,lon)
        u,v = get_rtma(lat,lon)

    pcsws, ghfs, ts, vs, vts = get_cfsr_data(lat,lon)
    pdsi = get_pdsi(lat,lon)
    precip = get_precip(lat,lon)

    prediction_set["date"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    prediction_set["Ground_Heat_Flux_surface"] = ghfs
    prediction_set["Plant_Canopy_Surface_Water_surface"] = pcsws
    prediction_set["Temperature_surface"] = ts
    prediction_set["Vegetation_Type_surface"] = vts
    prediction_set["Vegetation_surface"] = vs
    prediction_set["pdsi"] = pdsi
    prediction_set["precipitation"] = precip
    prediction_set["tmax"] = tmax    
    prediction_set["tmin"] = tmin
    prediction_set["u-component_of_wind_hybrid"] = u
    prediction_set["v-component_of_wind_hybrid"] = v  
    return prediction_set


#Prediction API called by client
@app.get('/prediction')
def make_prediction(
    lat: float = Query(..., ge=24.5, le=49.5, description="Latitude"),
    lon: float = Query(..., ge=-125, le=-66.5, description="Longitude")
):
    
    pred_set =  build_prediction_set(lat,lon)
    print("Completed prediction set")  
    print(pred_set)
    result = model.make_prediction(pred_set)
    print("Prediction: ",result)
    if result <= 0.25:
        label = "low_risk"
    elif result > 0.25 and result <= 0.75:
        label = "medium_risk"
    else:
        label = "high_risk"

    # Build feature for GeoJSON

    y_pred = float(result[0][0])
    timestamp = datetime.now().isoformat()

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon,lat],
        },
        "properties":{
            "lat": lat,
            "lon": lon,
            "label": label,
            "model_description": "Supvervised fire risk classifier",
            "popupContent":(
                f"Datetime: {timestamp}<br>"
                f"Lat: {lat:.4f}, Lon: {lon:.4f}<br>"
                f"Predicted fire risk: {y_pred:.3f} ({label})<br>"
            )
        } 
    }

    geojson = {
        "type": "FeatureCollection",
        "features": [feature],
    }

    return geojson


#Send images to webpage
 
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('favicon.ico')

@app.get('/plot_accuracy.png', include_in_schema=False)
async def plot_accuracy(): 
    return FileResponse('plot_accuracy.png')

@app.get('/plot_auc.png', include_in_schema=False)
async def plot_auc():
    return FileResponse('plot_auc.png')

@app.get('/plot_loss.png', include_in_schema=False)
async def plot_loss():
    return FileResponse('plot_loss.png')

@app.get('/confusion_matrix.png', include_in_schema=False)
async def confusion_matrix():
    return FileResponse('confusion_matrix.png') 