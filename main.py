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

""" app = FastAPI()
app.mount("/index", StaticFiles(directory="static", html=True), name="static")

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('favicon.ico') """

# Gooogle Earth Engine platform
ee.Authenticate()
# ee.Initialize(project="")
print(ee.String('Hello from the Earth Engine servers!').getInfo())


era5 = (ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                .select(['skin_temperature','lake_shape_factor'])
)


firstImage = era5.first()
lastImage = era5.sort('system:time_start',False).first()

epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

earliestDate = ee.Date(firstImage.get('system:time_start'))
latestDate = ee.Date(lastImage.get('system:time_start'))

earliestDate = dict(earliestDate.getInfo())
earliestDate = epoch + timedelta(milliseconds=earliestDate['value'])
latestDate = dict(latestDate.getInfo())
latestDate = epoch + timedelta(milliseconds=latestDate['value'])

print(f"Earliest image date: {earliestDate.isoformat()}")
print(f"Latest image date: {latestDate.isoformat()}")

# Define a small region of interest
roi = ee.Geometry.Point([-120.0, 37.0]) 


# Sample pixel values
samples = firstImage.sample(region=roi.buffer(10000), scale=1000, numPixels=10)

datadict = dict(samples.getInfo())

print(datadict)
print(datadict['properties']['band_order'])
print(datadict['features'][0]['properties']['lake_shape_factor'])
print(len(datadict['features']))




# Export as CSV to Google Drive
task = ee.batch.Export.table.toDrive(
    collection=samples,
    description='era5_sample',
    fileFormat='CSV'
)
#task.start()

 
# Nasa FIRMS API
#key = ""
sources = {
    "Landsat": f"https://firms.modaps.eosdis.nasa.gov/usfs/api/area/csv/{key}/LANDSAT_NRT/world/1/2025-10-13",
    "Modis" : f"https://firms.modaps.eosdis.nasa.gov/usfs/api/area/csv/{key}/MODIS_NRT/world/1/2025-10-13"
}

""" @app.get("/fire")
def get_fire_data():
    csv_buf = fire_data()
    collection = to_geojson(csv_buf)
    return JSONResponse(content=collection) """


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

def fire_data():
    url = sources["Modis"]
    try:
        response = requests.get(url)

        if response.status_code == 200:
            response_data = StringIO(response.text)
        else:
            print(f"Error: Status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    return response_data

