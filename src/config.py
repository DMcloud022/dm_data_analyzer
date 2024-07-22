import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet

load_dotenv()

class Config:
    ENCRYPTION_KEY_FILE = 'encryption_key.key'
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 100 * 1024 * 1024))  # Default 100MB
    ALLOWED_EXTENSIONS = ['xlsx', 'xls', 'csv', 'db']
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

    def __init__(self):
        if not os.path.exists(self.ENCRYPTION_KEY_FILE):
            self.generate_and_save_key()
        self.ENCRYPTION_KEY = self.load_key()

    def generate_and_save_key(self):
        key = Fernet.generate_key()
        with open(self.ENCRYPTION_KEY_FILE, 'wb') as key_file:
            key_file.write(key)
        print(f"Generated and saved Fernet key: {key.decode()}")

    def load_key(self):
        with open(self.ENCRYPTION_KEY_FILE, 'rb') as key_file:
            key = key_file.read()
        return key.decode()
