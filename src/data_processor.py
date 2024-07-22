import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def clean_data(self, df):
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
        
        # Remove outliers (using IQR method)
        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[column] >= Q1 - 1.5*IQR) & (df[column] <= Q3 + 1.5*IQR)]
        
        return df

    def validate_data(self, df):
        # Check for negative values in numeric columns
        for column in df.select_dtypes(include=[np.number]).columns:
            if (df[column] < 0).any():
                df[column] = df[column].abs()
                print(f"Warning: Negative values found in column: {column}. Converted to absolute values.")
        
        # Check for proper date format in date columns
        for column in df.select_dtypes(include=['datetime64']).columns:
            df[column] = pd.to_datetime(df[column], errors='coerce')
            if df[column].isna().any():
                print(f"Warning: Invalid date format found in column: {column}. NaT values introduced.")
        
        return df

    def transform_data(self, df):
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Normalize numeric columns
        scaler = StandardScaler()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # One-hot encode categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=True)
        
        return df

    def prepare_data(self, df):
        df = self.clean_data(df)
        df = self.validate_data(df)
        df = self.transform_data(df)
        return df