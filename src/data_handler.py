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
from typing import Union, List, Dict
import json
import hashlib
import tempfile

class DataHandler:
    def __init__(self, config: Config, logger: Logger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.key = config.ENCRYPTION_KEY
        self.cipher_suite = Fernet(self.key)
        self.temp_dir = tempfile.mkdtemp()

    def __del__(self):
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)

    def detect_encoding(self, file) -> str:
        raw_data = file.read(1024)
        file.seek(0)
        return chardet.detect(raw_data)['encoding'] or 'utf-8'

    def read_file(self, file) -> pd.DataFrame:
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
                temp_db_path = os.path.join(self.temp_dir, file.name)
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

    def write_file(self, data: pd.DataFrame, file_name: str) -> BytesIO:
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

    def encrypt_data(self, data: Union[str, bytes, pd.DataFrame]) -> bytes:
        try:
            if isinstance(data, pd.DataFrame):
                data = data.to_json()
            elif isinstance(data, str):
                data = data.encode()
            encrypted_data = self.cipher_suite.encrypt(data)
            return encrypted_data
        except Exception as e:
            self.logger.log_error(f"Error encrypting data: {str(e)}")
            error_message = self.error_handler.handle_encryption_error(e) if self.error_handler else str(e)
            raise ValueError(error_message)

    def decrypt_data(self, encrypted_data: bytes) -> Union[str, pd.DataFrame]:
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data).decode()
            try:
                # Attempt to parse as JSON and convert to DataFrame
                return pd.read_json(decrypted_data)
            except:
                # If not JSON, return as string
                return decrypted_data
        except Exception as e:
            self.logger.log_error(f"Error decrypting data: {str(e)}")
            error_message = self.error_handler.handle_decryption_error(e) if self.error_handler else str(e)
            raise ValueError(error_message)

    def validate_data(self, data: pd.DataFrame, schema: Dict[str, str]) -> bool:
        try:
            for column, dtype in schema.items():
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in the data")
                if data[column].dtype != dtype:
                    raise ValueError(f"Column '{column}' has incorrect data type. Expected {dtype}, got {data[column].dtype}")
            return True
        except Exception as e:
            self.logger.log_error(f"Data validation error: {str(e)}")
            return False

    def hash_data(self, data: Union[str, bytes, pd.DataFrame]) -> str:
        if isinstance(data, pd.DataFrame):
            data = data.to_json().encode()
        elif isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()

    def compress_data(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.select_dtypes(include=['object']).columns:
            if data[column].nunique() / len(data) < 0.5:  # If less than 50% unique values
                data[column] = pd.Categorical(data[column])
        return data

    def decompressed_data(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.select_dtypes(include=['category']).columns:
            data[column] = data[column].astype('object')
        return data

    def chunk_data(self, data: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
        return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]