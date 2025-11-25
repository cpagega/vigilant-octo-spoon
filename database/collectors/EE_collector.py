import ee

class EECollector() :
    def __init__(self):
        self.cfsr = None
        self.conus = None
        self.cpc_precip = None
        self.cpc_temp = None
        self.fc = None

    def create_firms_featurecollection(self):
        """ Converts a CSV to a feature collection while preserving lat/long as properties"""
        firms = ee.FeatureCollection(f"projects/mythical-lens-123220/assets/firms_sample_sorted")
        def add_lat_lon(feat):
            coords = feat.geometry().coordinates()
            return (feat
                    .set('long', coords.get(0))
                    .set('lat', coords.get(1)))
        self.fc = firms.map(add_lat_lon)

    def set_image_collections(self):
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
        
        self.conus = (
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
    
    def attach_cfsr(self, feat):
        """ Combines the EE feature collections with the FIRMs feature collection"""
        t = ee.Date(feat.get("date"))
        millis = t.millis()
        six_h = ee.Number(6 * 60 * 60 * 1000) #CFSR data is in 6 hour intervals
        slot_millis = millis.divide(six_h).round().multiply(six_h) # round to the nearest 6 hours
        slot = ee.Date(slot_millis)
        img = (self.cfsr
               .filterDate(slot, slot.advance(6, "hour")) 
               .sort("system:time_start")
               .first())
        img = ee.Image(img)
        vals = img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=feat.geometry(),
            scale=50000,
            maxPixels=1E7,
        )
        return (feat
                .set("cfsr_time", slot.format("YYYY-MM-dd'T'HH:mm:ss"))
                .setMulti(vals))
    
    def attach_conus(self,feat):
        raise NotImplementedError("Function 'attach_conus' Not Implemented")

    def attach_cpc(self,feat):
        raise NotImplementedError("Function 'attach_cpc' Not Implemented")
    
    def attach_daymet(self,feat):
        raise NotImplementedError("Function 'attach_daymet' Not Implemented")

    def export_to_gdrive(self):
        """ Exports the combined feature set as a CSV to the project drive """
        task = ee.batch.Export.table.toDrive(
            collection=self.fc,
            description="firms_ee_feature_join",
            fileFormat="CSV"
        )
        task.start()

    def collect_data(self):
        print("Creating FIRMS collection")
        self.create_firms_featurecollection()
        print("Setting image collections")
        self.set_image_collections()
        print("Attaching Features to Feature Collection")
        self.fc = self.fc.map(self.attach_cfsr)
        #self.fc = self.fc.map(self.attach_conus)
        #self.fc = self.fc.map(self.attach_cpc)
        #self.fc = self.fc.map(self.attach_daymet)
        print("Exporting combined data set to project drive")
        self.export_to_gdrive()