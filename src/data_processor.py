import pandas as pd
import numpy as np

class DataProcessor:
    def clean_data(self, df):
        df = df.drop_duplicates()
        
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].fillna(df[column].mode()[0])
            else:
                df[column] = df[column].fillna(df[column].median())
        
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[column] >= Q1 - 1.5*IQR) & (df[column] <= Q3 + 1.5*IQR)]
        
        return df

    def validate_data(self, df):
        for column in df.select_dtypes(include=[np.number]).columns:
            if (df[column] < 0).any():
                raise ValueError(f"Negative values found in column: {column}")
        

        for column in df.select_dtypes(include=['datetime64']).columns:
            if pd.to_datetime(df[column], errors='coerce').isna().any():
                raise ValueError(f"Invalid date format found in column: {column}")
        
        return True

    def transform_data(self, df):
        for column in df.select_dtypes(include=[np.number]).columns:
            df[f"{column}_normalized"] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        

        categorical_columns = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_columns)
        
        return df