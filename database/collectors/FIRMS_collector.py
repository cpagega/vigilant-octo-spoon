import csv
from collectors.table_collector import TableCollector
class FIRMSCollector(TableCollector) :

    def __init__(self):
        self.sample_size = 200000
        super().__init__("FIRMS", "FIRMS")

    def create_table(self):
        self.cursor.execute(f"""
            CREATE TABLE {self.table_name} (
                date text,
                lat real,
                long real,
                confidence int,
                brightness real,
                frp real,
                
                PRIMARY KEY (date, lat, long)
            );
        """)

    def _format_time(self, date, time):
        time = time.zfill(4)
        return f"{date}T{time[0:2]}:{time[2:]}:00"
    

    def _create_sample_table(self):
        self.cursor.execute(f"""
            CREATE TABLE FIRMS_SAMPLE AS
                SELECT * 
                FROM FIRMS
                ORDER BY RANDOM()
                LIMIT {self.sample_size};
                            """)
    
    def _process_csv(self, reader):
        print("Populating FIRMS table...")
        for row in reader:
            date = row["acq_date"]
            time = row["acq_time"]
            lat = row["latitude"]
            long = row["longitude"]
            confidence = row["confidence"]
            brightness = row["bright_t31"]
            frp = row["frp"]
            datetime = self._format_time(date,time)
            sql = f"""INSERT OR IGNORE INTO {self.table_name} 
                  (date, lat, long, confidence, brightness, frp)
                  VALUES (?,?,?,?,?,?)
                  """
            params = (datetime,lat,long,confidence,brightness,frp)
            self.cursor.execute(sql, params)
        print("Populating FIRMS table complete.")

    def collect_data(self):
        with open("Data\\fire_archive_M-C61_683824.csv", newline="") as f:
            reader = csv.DictReader(f)
            self._process_csv(reader)
        self.conn.commit()
        print(f"Creating table of random FIRMS data. Size: {self.sample_size}")
        self._create_sample_table()
        print("Sample table creation complete")

