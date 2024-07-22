import pandas as pd
import sqlite3
from io import BytesIO
import os
from cryptography.fernet import Fernet

class DataHandler:
    def __init__(self):
        self.key = os.environ.get('ENCRYPTION_KEY') or Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def read_file(self, file):
        try:
            file_type = file.name.split('.')[-1].lower()
            if file_type in ['xlsx', 'xls']:
                return pd.read_excel(file, engine='openpyxl')
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
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")

    def write_file(self, data, file_name):
        try:
            file_type = file_name.split('.')[-1].lower()
            if file_type in ['xlsx', 'xls']:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    data.to_excel(writer, index=False)
                return output
            elif file_type == 'csv':
                return data.to_csv(index=False).encode('utf-8')
            else:
                raise ValueError(f"Unsupported output file type: {file_type}")
        except Exception as e:
            raise ValueError(f"Error writing file: {str(e)}")

    def encrypt_data(self, data):
        return self.cipher_suite.encrypt(data.encode())

    def decrypt_data(self, encrypted_data):
        return self.cipher_suite.decrypt(encrypted_data).decode()