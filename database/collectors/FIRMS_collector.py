from collectors.table_collector import TableCollector
import constants
import ee

class FIRMSCollector(TableCollector) :
    def __init__(self):
        super().__init__("FIRMS", "FIRMS")


    def create_table(self):
        self.cursor.execute(f"""
            CREATE TABLE {self.table_name} (
                date text,
                lat text,
                long text,
                confidence int,
                brightness real,
                frp real,
                
                PRIMARY KEY (date, lat, long)
            );
        """)

    def collect_data(self):
        raise NotImplementedError("FIRMS collection needs implemented")
        firms = ee.ImageCollection(self.ee_name).filterDate(constants.START_DATE, constants.END_DATE)
