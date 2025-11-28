import sqlite3
import pandas as pd


# Use this to created a sample of FIRMS data and upload to your EE project space

def export_table_csv(table_name):
    conn = sqlite3.connect("dataset.db")
    df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY date", conn)
    conn.close()
    df.to_csv("firms_sample_conus.csv", index=False)



if (__name__ == '__main__'):
    print("Exporting table")
    export_table_csv("FIRMS_SAMPLE")

