from collectors.table_collector import TableCollector
import constants
import ee

class CONUSCollector(TableCollector) :
    def __init__(self):
        super().__init__("CONUS", "GRIDMET/DROUGHT")


    def create_table(self):
        self.cursor.execute(f"""
            CREATE TABLE {self.table_name} (
                date text,
                lat text,
                long text,
                pdsi real,
                PRIMARY KEY (date, lat, long)
            );
        """)

    def collect_data(self):
        conus = ee.ImageCollection(self.ee_name).filterDate(constants.START_DATE, constants.END_DATE).select('pdsi')
        info_list = conus.aggregate_array('system:index').getInfo()

        for i, image_id in enumerate(info_list):
            print(f"{i+1}. {image_id}")

        raise NotImplementedError("CONUS collection needs fully implemented")