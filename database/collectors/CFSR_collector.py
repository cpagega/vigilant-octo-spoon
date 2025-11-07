from collectors.table_collector import TableCollector
import constants
import ee

class CFSRCollector(TableCollector) :
    def __init__(self):
        super().__init__("CFSR", "NOAA/CFSR")


    def create_table(self):
        self.cursor.execute(f"""
            CREATE TABLE {self.table_name} (
                date text,
                lat text,
                long text,
                plant_water_surface real,
                u_component_wind real,
                v_component_wind real,
                downward_radiation real,
                ground_heat real,
                surface_temp real,
                vegetation_surface real,
                vegetation_type real,

                PRIMARY KEY (date, lat, long)
            );
        """)

    def collect_data(self):
        raise NotImplementedError("CFSR collection needs implemented")
        cfsr = ee.ImageCollection(self.ee_name).filterDate(constants.START_DATE, constants.END_DATE)