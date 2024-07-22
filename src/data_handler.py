import pandas as pd
import sqlite3
from io import BytesIO

class DataHandler:
    def read_file(self, file):
        file_type = file.name.split('.')[-1].lower()
        if file_type == 'xlsx':
            return pd.read_excel(file)
        elif file_type == 'csv':
            return pd.read_csv(file)
        elif file_type == 'db':
            conn = sqlite3.connect(file.name)
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            if len(tables) > 0:
                return pd.read_sql_query(f"SELECT * FROM {tables.iloc[0]['name']}", conn)
            else:
                raise ValueError("No tables found in the database.")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def write_file(self, data, file_name):
        file_type = file_name.split('.')[-1].lower()
        if file_type == 'xlsx':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                data.to_excel(writer, index=False)
            return output
        elif file_type == 'csv':
            return data.to_csv(index=False).encode('utf-8')
        else:
            raise ValueError(f"Unsupported output file type: {file_type}")