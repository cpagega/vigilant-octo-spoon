from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
import csv, json
from geojson import Feature, FeatureCollection, Point
import requests
from datetime import datetime, timezone, timedelta
import ee
import models.predict
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

#converts the FIRMS CSV to a GeoJSON, a format used by mapping software
def to_geojson(csv_data):
    features = []
    reader = csv.DictReader(csv_data)  
    for row in reader:
        try: 
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            scan = float(row["scan"])
            track = float(row["track"])
            frp = float(row["frp"])
            confidence = int(row["confidence"])  
        except (KeyError, ValueError):
            continue  
        if confidence >= 50:
            features.append(
                Feature(
                    geometry=Point((lon, lat)),
                    properties={
                        "scan": scan, 
                        "track": track, 
                        "frp": frp,
                        "popupContent": f"Lat: {lat}. Lon: {lon}<br>Scan: {scan} Track: {track}<br>FRP: {frp}<br>Confidence: {confidence}"
                    },
                )
            )
    print(f"Number of points: {len(features)}")
    return FeatureCollection(features)

def reduce_at_point(img, lat, lon, scale=50000):
    geom = ee.Geometry.Point([lon, lat])
    return img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=geom,
        scale=scale,
        maxPixels=1e7,
    )

def k_to_c(temp):
    return temp - 273.15

def wind_to_uv(speed, deg):
    theta = math.radians(deg)
    u = -speed * math.sin(theta) # eastward component
    v = -speed * math.cos(theta) # northward component
    return u,v

def get_current_weather(lat,lon):
    response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={weather_key}")
    if response.status_code == 200:
        return response.data
    else: 
        return response.status_code
    
# retrieve the latest real-time mesoanalysis wind speeds - only if weather station data is unavailable
def get_rtma(lat,lon):
    img = (ee.ImageCollection('NOAA/NWS/RTMA')
        .select(["UGRD","VGRD"])
        .sort("system:time_start",False)
        .first()
    )
    return reduce_at_point(img,lat,lon)





@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('favicon.ico')




