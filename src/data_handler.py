import pandas as pd
import sqlite3
from io import BytesIO
import os
import chardet
import base64
from cryptography.fernet import Fernet
from src.config import Config
from src.logger import Logger
from src.error_handler import ErrorHandler

class DataHandler:
    
    # Generate a new Fernet key
    key = Fernet.generate_key()
    encoded_key = base64.urlsafe_b64encode(key).decode()
    print(f"Base64-encoded key: {encoded_key}")

    def __init__(self, config: Config, logger: Logger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.key = config.ENCRYPTION_KEY
        self.cipher_suite = Fernet(self.key)

    def detect_encoding(self, file):
        raw_data = file.read(1024)
        file.seek(0)
        return chardet.detect(raw_data)['encoding']

    def read_file(self, file):
        try:
            file_type = file.name.split('.')[-1].lower()
            if file_type not in self.config.ALLOWED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            file_size = file.size
            if file_size > self.config.MAX_FILE_SIZE:
                raise ValueError(f"File size exceeds the maximum allowed size of {self.config.MAX_FILE_SIZE / (1024 * 1024)}MB")

            if file_type in ['xlsx', 'xls']:
                return pd.read_excel(file, engine='openpyxl')
            elif file_type == 'csv':
                encoding = self.detect_encoding(file)
                return pd.read_csv(file, encoding=encoding, low_memory=False)
            elif file_type == 'db':
                temp_db_path = os.path.join(self.config.TEMP_DIR, file.name)
                with open(temp_db_path, 'wb') as f:
                    f.write(file.getbuffer())
                
                conn = sqlite3.connect(temp_db_path)
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                if len(tables) > 0:
                    return pd.read_sql_query(f"SELECT * FROM {tables.iloc[0]['name']}", conn)
                else:
                    raise ValueError("No tables found in the database.")
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            self.logger.log_error(f"Error reading file: {str(e)}")
            error_message = self.error_handler.handle_file_read_error(e) if self.error_handler else str(e)
            raise ValueError(error_message)

    def write_file(self, data, file_name):
        try:
            file_type = file_name.split('.')[-1].lower()
            if file_type in ['xlsx', 'xls']:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    data.to_excel(writer, index=False)
                output.seek(0)
                return output
            elif file_type == 'csv':
                output = data.to_csv(index=False).encode('utf-8')
                return BytesIO(output)
            else:
                raise ValueError(f"Unsupported output file type: {file_type}")
        except Exception as e:
            self.logger.log_error(f"Error writing file: {str(e)}")
            error_message = self.error_handler.handle_file_write_error(e) if self.error_handler else str(e)
            raise ValueError(error_message)

    def encrypt_data(self, data):
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return encrypted_data
        except Exception as e:
            self.logger.log_error(f"Error encrypting data: {str(e)}")
            error_message = self.error_handler.handle_encryption_error(e) if self.error_handler else str(e)
            raise ValueError(error_message)

    def decrypt_data(self, encrypted_data):
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data).decode()
            return decrypted_data
        except Exception as e:
            self.logger.log_error(f"Error decrypting data: {str(e)}")
            error_message = self.error_handler.handle_decryption_error(e) if self.error_handler else str(e)
            raise ValueError(error_message)
