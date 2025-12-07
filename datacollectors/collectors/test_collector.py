import ee

"""
@author: Tyler and Chris
"""

class EECollector() :
    def __init__(self, project_id):
        self.cfsr = None
        self.gridmet = None
        self.cpc_precip = None
        self.cpc_temp = None
        self.fc = None
        self.project_id = project_id

    def _create_firms_featurecollection(self):
        """ Converts a CSV to a feature collection while preserving lat/long as properties"""
        firms = ee.FeatureCollection(f"projects/{self.project_id}/assets/firms_sample_conus")
        def add_lat_lon(feat):
            coords = feat.geometry().coordinates()
            return (feat
                    .set('long', coords.get(0))
                    .set('lat', coords.get(1)))
        self.fc = firms.map(add_lat_lon)

    def _set_image_collections(self):
        """ Sets the image collection properties"""
        self.cfsr = (
            ee.ImageCollection("NOAA/CFSR")
            .select(['Plant_Canopy_Surface_Water_surface',
                     'u-component_of_wind_hybrid', 
                     'v-component_of_wind_hybrid', 
                     'Ground_Heat_Flux_surface',
                     'Temperature_surface',
                     'Vegetation_surface',
                     'Vegetation_Type_surface'
                     ])        
            )
        
        self.gridmet = (
            ee.ImageCollection("GRIDMET/DROUGHT")
            .select(['pdsi'])
        )
        self.cpc_precip = (
            ee.ImageCollection("NOAA/CPC/Precipitation")
            .select(['precipitation'])
        )
        self.cpc_temp = (
            ee.ImageCollection("NOAA/CPC/Temperature")
            .select(['tmax','tmin'])
        )

    def _get_feature_values(self, feat, dataset, hours, band_names, scale=None):
        """
            feat: existing feature collection
            dataset: new dataset to that will be appended to feature collection
            hours: time interval for dataset collections
            band_names: list of expected band names for null value handling
            scale: spatial resolution in meters (if None, uses 50000m default)
            Returns the values of the dataset for the given time interval
        """
        if scale is None:
            scale = 50000
        t = ee.Date(feat.get("date"))
        millis = t.millis()
        h = ee.Number(hours * 60 * 60 * 1000) #EE Data interval in milliseconds
        slot_millis = millis.divide(h).round().multiply(h) # round to the nearest interval
        slot = ee.Date(slot_millis)

        # get the first image of an interval
        img = (dataset
               .filterDate(slot, slot.advance(hours, "hour")) 
               .sort("system:time_start")
               .first())

        # Check if image exists using size() of the filtered collection
        collection_size = (dataset
                          .filterDate(slot, slot.advance(hours, "hour"))
                          .size())
        
        # Create a conditional that returns proper values or nulls
        def get_values():
            return img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=feat.geometry(),
                scale=scale,
                maxPixels=1E7,
            )
        
        def get_null_values():
            # Return a dictionary with null values for all expected bands
            null_dict = {band: -99999 for band in band_names}
            return ee.Dictionary(null_dict)
        
        vals = ee.Algorithms.If(
            collection_size.gt(0),
            get_values(),
            get_null_values()
        )
        
        return ee.Dictionary(vals), slot
    
    def _attach_cfsr(self, feat):
        band_names = ['Plant_Canopy_Surface_Water_surface',
                     'u-component_of_wind_hybrid', 
                     'v-component_of_wind_hybrid', 
                     'Ground_Heat_Flux_surface',
                     'Temperature_surface',
                     'Vegetation_surface',
                     'Vegetation_Type_surface']
        vals, slot = self._get_feature_values(feat, self.cfsr, 6, band_names, scale=38000)  # ~38km native
        return (feat
                .set("cfsr_time", slot.format("YYYY-MM-dd'T'HH:mm:ss"))
                .setMulti(vals))
    
    def _attach_gridmet(self, feat):        
        vals, slot = self._get_feature_values(feat, self.gridmet, 5 * 24, ['pdsi'], scale=4000)  # ~4km native

        # If pdsi exists, use it; otherwise -99999
        pdsi = ee.Algorithms.If(
            vals.contains('pdsi'),
            vals.get('pdsi'),
            -99999  
        )

        return (feat
            .set('gridmet_time', slot.format("YYYY-MM-dd'T'HH:mm:ss"))
            .set('pdsi', pdsi))

    def _attach_cpc_precip(self, feat):
        vals, slot = self._get_feature_values(feat, self.cpc_precip, 24, ['precipitation'], scale=50000)  # ~0.5° native
        return (feat
                .set("cpc_precip_time", slot.format("YYYY-MM-dd'T'HH:mm:ss"))
                .setMulti(vals))
    
    def _attach_cpc_temp(self, feat):
        vals, slot = self._get_feature_values(feat, self.cpc_temp, 24, ['tmax', 'tmin'], scale=50000)  # ~0.5° native
        return (feat
                .set("cpc_temp_time", slot.format("YYYY-MM-dd'T'HH:mm:ss"))
                .setMulti(vals))

    def _export_to_gdrive(self):
        """ Exports the combined feature set as a CSV to the project drive """
        task = ee.batch.Export.table.toDrive(
            collection=self.fc,
            description="firms_ee_feature_join",
            fileFormat="CSV"
        )
        task.start()

    def collect_data(self):
        print("Creating FIRMS collection")
        self._create_firms_featurecollection()
        print("Setting image collections")
        self._set_image_collections()
        print("Attaching Features to Feature Collection")
        self.fc = self.fc.map(self._attach_cfsr)
        self.fc = self.fc.map(self._attach_gridmet)
        self.fc = self.fc.map(self._attach_cpc_temp)
        self.fc = self.fc.map(self._attach_cpc_precip)
        print("Exporting combined data set to project drive")
        self._export_to_gdrive()