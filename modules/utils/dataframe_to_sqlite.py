import math
import sqlite3
import pandas as pd

class DataFrameToSQLite:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None
        # self.count
        # self.

    def create_database(self):
        self.conn = sqlite3.connect(self.db_name)

    def create_table(self, table_name, df):
        df.to_sql(table_name, self.conn, index=False)

    def is_table_exist(self, table_name):

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        result = cursor.fetchone()
        # self.conn.close()

        return result is not None

    def count_entries(self, table_name):
        # conn = sqlite3.connect(db_path)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        # conn.close()
        return count

    def get_total_pages(self, total_records, page_size):
        return math.ceil(total_records / page_size)

    def insert_data(self, table_name, df):
        df.to_sql(table_name, self.conn, index=False, if_exists='append')

    def find_data(self, table_name, condition):
        query = f'SELECT * FROM {table_name} WHERE {condition}'
        return pd.read_sql(query, self.conn)

    def close_connection(self):
        self.conn.close()