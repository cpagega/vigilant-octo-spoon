import sqlite3
import constants

class TableCollector:
    def __init__(self, table_name : str, ee_name : str):
        self.table_name = table_name
        self.ee_name = ee_name
        self.cursor = self.build_cursor()
        
        if (not self.does_table_exist()):
            self.create_table()

    def __del__(self) -> None:
        self.cursor.close()

    def build_cursor(self) -> sqlite3.Cursor:
        conn = sqlite3.connect(constants.DB_NAME)
        return conn.cursor()
    
    def does_table_exist(self) -> bool:
        if (self.cursor != None):
            self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.table_name}'")
            return self.cursor.fetchone() is not None
        raise ValueError("Cursor not initialized")
    
    def create_table(self) -> None:
        raise NotImplementedError("Function 'create_table' not overwritten")
    
    def collect_data(self) -> None:
        raise NotImplementedError("Function 'collect_data' not overwritten")
