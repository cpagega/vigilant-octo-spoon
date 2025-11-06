from collectors.table_builder import TableBuilder
import Constants
import ee

class FIRMS_Builder(TableBuilder) :
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
        start = ee.Date(Constants.START_DATE)
        end = start.advance(1, 'day') # only collecting one day for the time being

        modis = ee.ImageCollection(self.ee_name).filterDate(start, end)
        print(f"Image Count: {modis.size().getInfo()}")