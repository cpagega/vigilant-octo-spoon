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

    # Selects the image collections for each dataset we are interested in
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

    # Gets the values from a single point in an image
    def _get_feature_values(self, feat, dataset, hours):
        """
            feat: existing feature
            dataset: image collection to sample from
            hours: interval size
            Returns (vals_dict, slot_date)
        """
        t = ee.Date(feat.get("date"))
        millis = t.millis()
        h = ee.Number(hours * 60 * 60 * 1000)  # interval in ms
        slot_millis = millis.divide(h).round().multiply(h)
        slot = ee.Date(slot_millis)

        # Filter collection to the interval
        coll = (dataset
            .filterDate(slot, slot.advance(hours, "hour"))
            .sort("system:time_start"))

        # Function that only runs if there is at least one image
        def _compute_vals(c):
            img = ee.Image(c.first())
            return img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=feat.geometry(),
                scale=img.projection().nominalScale(),
                maxPixels=1e7,
            )

        # If collection is empty, return an empty dict instead of error
        vals = ee.Dictionary(
            ee.Algorithms.If(
                coll.size().gt(0),
                _compute_vals(coll),
                ee.Dictionary({})  # no image → no values
            )
        )

        return vals, slot
    
    
    def _attach_cfsr(self, feat):
        vals,slot = self._get_feature_values(feat, self.cfsr, 6)
        return (feat
                .set("cfsr_time", slot.format("YYYY-MM-dd'T'HH:mm:ss"))
                .setMulti(vals))
    
    def _attach_gridmet(self,feat):        
        vals, slot = self._get_feature_values(feat, self.gridmet, 5 * 24)

        # If pdsi exists, use it; otherwise -99999
        pdsi = ee.Algorithms.If(
            vals.contains('pdsi'),
            vals.get('pdsi'),
            -99999  
        )

        return (feat
            .set('gridmet_time', slot.format("YYYY-MM-dd'T'HH:mm:ss"))
            .set('pdsi', pdsi))

    def _attach_cpc_precip(self,feat):
        vals,slot = self._get_feature_values(feat, self.cpc_precip, 24)
        return (feat
                .set("cpc_precip_time", slot.format("YYYY-MM-dd'T'HH:mm:ss"))
                .setMulti(vals))
    
    def _attach_cpc_temp(self,feat):
        vals,slot = self._get_feature_values(feat, self.cpc_temp, 24)
        return (feat
                .set("cpc_temp_time", slot.format("YYYY-MM-dd'T'HH:mm:ss"))
                .setMulti(vals))
    
    #Problems with EE dropping the entire column if one img was missing - trying this last ditch effort to force it
    def ensure_pdsi(self,feat):
        return feat.set(
            'pdsi',
            ee.Algorithms.If(
                feat.propertyNames().contains('pdsi'),
                feat.get('pdsi'),
                -99999
            )
        )


    def _export_to_gdrive(self):
        """ Exports the combined feature set as a CSV to the project drive """
        task = ee.batch.Export.table.toDrive(
            collection=self.fc,
            description="firms_ee_feature_join",
            fileFormat="CSV",
            selectors=[
            'date', 'lat', 'long',
            'gridmet_time','pdsi',
            'cpc_temp_time', 'tmax', 'tmin',
            'cpc_precip_time', 'precipitation',
            'cfsr_time',
            'Plant_Canopy_Surface_Water_surface',
            'u-component_of_wind_hybrid', 
            'v-component_of_wind_hybrid', 
            'Ground_Heat_Flux_surface',
            'Temperature_surface',
            'Vegetation_surface',
            'Vegetation_Type_surface',
            'confidence','brightness','frp','label'
            ]

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
        self.fc = self.fc.map(self.ensure_pdsi)
        print(self.fc.limit(5).getInfo())
        print("Exporting combined data set to project drive")
        self._export_to_gdrive()