import csv, random
import pandas as pd
from datetime import datetime, timedelta
from collectors.table_collector import TableCollector

"""
@author: Chris and Tyler

Note: This file is intended to be ran once to initialize the db for FIRMS data.

"""

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
                WHERE lat  BETWEEN 25.0 AND 50.0
                AND   long BETWEEN -125.0 AND -66.0
                AND confidence >= 40
                            """)
    
    def _add_label(self):
        self.cursor.execute(f"""ALTER TABLE FIRMS_SAMPLE ADD COLUMN label INTEGER;""")
        self.cursor.execute(f"""UPDATE FIRMS_SAMPLE SET label = 1; """)
    
    def _commit_csv_to_db(self, reader):
        rows = []
        for row in reader:
            date = row["acq_date"]
            time = row["acq_time"]
            lat = row["latitude"]
            long = row["longitude"]
            confidence = row["confidence"]
            brightness = row["bright_t31"]
            frp = row["frp"]
            datetime = self._format_time(date,time)
            rows.append((datetime,lat,long,confidence,brightness,frp))
        sql = f"""INSERT OR IGNORE INTO {self.table_name} 
                (date, lat, long, confidence, brightness, frp)
                VALUES (?,?,?,?,?,?)
                """
            #params = (datetime,lat,long,confidence,brightness,frp)    
        self.cursor.executemany(sql, rows)
        self.conn.commit()

    def _delta_days(self, table):
        self.cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table}")
        min_date_str, max_date_str = self.cursor.fetchone()
        min_date = datetime.fromisoformat(min_date_str)
        max_date = datetime.fromisoformat(max_date_str)
        return (max_date - min_date).days, min_date, max_date
    
    def _row_count(self,table):
        self.cursor.execute(f"""
            SELECT COUNT(*) FROM {table}
                            """)
        return self.cursor.fetchone()[0] # returns a tuple
    
    def _random_date(self, delta_days, min_date):
        d = min_date + timedelta(days=random.uniform(0, delta_days))
        return d.strftime("%Y-%m-%dT%H:%M:%S")
        
    def _random_lat_long(self):
        lat = round(random.uniform(25.0000, 50.0000),4)
        lon = round(random.uniform(-125.000, -66.0000),4)
        return lat,lon

    def _generate_negative_rows(self, num_rows, delta_days, min_date):
        rows = []
        for _ in range(num_rows):
            dt = self._random_date(delta_days, min_date)
            lat,lon = self._random_lat_long()
            rows.append((dt,lat,lon, 0,0,0,0)) # confidence, brightness, frp, label = 0
        return rows
        
    def _insert_negatives(self, table):
        delta_days, min_date, max_date = self._delta_days(table)
        pos_row_count = self._row_count(table)
        neg_rows = self._generate_negative_rows(int(pos_row_count * 1.5), delta_days, min_date)
        sql = f"""INSERT OR IGNORE INTO {table} (date, lat, long, confidence, brightness, frp, label) VALUES (?, ?, ?, ?, ?, ?, ?)"""
        self.cursor.executemany(sql, neg_rows)
        self.conn.commit()

    def _export_table_csv(self, table_name):
        df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY date", self.conn)
        df.to_csv("firms_sample_conus.csv", index=False)

    def collect_data(self):
        print(f"Processing FIRMS data...")
        with open("Data\\fire_archive_M-C61_683824.csv", newline="") as f:
            reader = csv.DictReader(f)
            self._commit_csv_to_db(reader)
        self._create_sample_table()
        self._add_label()
        self._insert_negatives("FIRMS_SAMPLE")
        self._export_table_csv("FIRMS_SAMPLE")
        print(f"Processing complete.")