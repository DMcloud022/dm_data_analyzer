import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import re

class DataProcessor:
    def __init__(self, config, logger, error_handler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler

    def prepare_data(self, df):
        try:
            self.logger.log_info("Starting data preparation...")
            
            # Step 1: Basic cleaning (always perform this step)
            df = self.basic_cleaning(df)
            
            # Step 2: Handle missing values (if any)
            if df.isnull().sum().sum() > 0:
                df = self.handle_missing_values(df)
            
            # Step 3: Convert data types (if needed)
            df = self.convert_data_types(df)
            
            # Step 4: Handle outliers (for numeric columns)
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                df = self.handle_outliers(df)
            
            # Step 5: Encode categorical variables (if any)
            if len(df.select_dtypes(include=['object']).columns) > 0:
                df = self.encode_categorical_variables(df)
            
            # Step 6: Feature scaling (for numeric columns, if more than one)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                df = self.scale_features(df)
            
            # Step 7: Feature selection (if there are many features)
            if df.shape[1] > 20:  # Arbitrary threshold, adjust as needed
                df = self.select_features(df)
            
            self.logger.log_info("Data preparation completed successfully.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in prepare_data: {str(e)}")
            raise self.error_handler.handle_data_processing_error(e)

    def basic_cleaning(self, df):
        try:
            # Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates()
            removed_rows = initial_rows - len(df)
            if removed_rows > 0:
                self.logger.log_info(f"Removed {removed_rows} duplicate rows.")

            # Clean column names
            original_columns = df.columns
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            df.columns = [re.sub(r'[^\w\s]', '', col) for col in df.columns]
            if not (df.columns == original_columns).all():
                self.logger.log_info("Cleaned column names.")

            return df
        except Exception as e:
            self.logger.log_error(f"Error in basic_cleaning: {str(e)}")
            return df  # Return original dataframe if cleaning fails

    def handle_missing_values(self, df):
        try:
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    missing_percentage = (missing_count / len(df)) * 100
                    if missing_percentage > 50:
                        self.logger.log_warning(f"Column '{column}' has {missing_percentage:.2f}% missing values. Consider dropping this column.")
                    elif df[column].dtype in ['int64', 'float64']:
                        df[column] = df[column].fillna(df[column].median())
                        self.logger.log_info(f"Filled missing values in column '{column}' with median.")
                    else:
                        df[column] = df[column].fillna(df[column].mode()[0])
                        self.logger.log_info(f"Filled missing values in column '{column}' with mode.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in handle_missing_values: {str(e)}")
            return df  # Return original dataframe if handling missing values fails

    def convert_data_types(self, df):
        try:
            for column in df.columns:
                if df[column].dtype == 'object':
                    try:
                        df[column] = pd.to_datetime(df[column])
                        self.logger.log_info(f"Converted column '{column}' to datetime.")
                    except ValueError:
                        try:
                            df[column] = pd.to_numeric(df[column])
                            self.logger.log_info(f"Converted column '{column}' to numeric.")
                        except ValueError:
                            pass  # Keep as object type if conversion is not possible
            return df
        except Exception as e:
            self.logger.log_error(f"Error in convert_data_types: {str(e)}")
            return df  # Return original dataframe if type conversion fails

    def handle_outliers(self, df):
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
                if len(outliers) > 0:
                    self.logger.log_warning(f"Found {len(outliers)} outliers in column '{column}'.")
                    df[column] = df[column].clip(lower_bound, upper_bound)
                    self.logger.log_info(f"Clipped outliers in column '{column}'.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in handle_outliers: {str(e)}")
            return df  # Return original dataframe if outlier handling fails

    def encode_categorical_variables(self, df):
        try:
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                if df[column].nunique() < 10:  # For low cardinality variables
                    df = pd.get_dummies(df, columns=[column], prefix=column, drop_first=True)
                    self.logger.log_info(f"One-hot encoded column '{column}'.")
                else:  # For high cardinality variables
                    df[f"{column}_encoded"] = df[column].astype('category').cat.codes
                    self.logger.log_info(f"Label encoded column '{column}'.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in encode_categorical_variables: {str(e)}")
            return df  # Return original dataframe if encoding fails

    def scale_features(self, df):
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            self.logger.log_info(f"Scaled {len(numeric_columns)} numeric columns.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in scale_features: {str(e)}")
            return df  # Return original dataframe if scaling fails

    def select_features(self, df):
        try:
            # Remove constant features
            constant_filter = VarianceThreshold(threshold=0)
            constant_filter.fit(df)
            constant_columns = [column for column in df.columns 
                                if column not in df.columns[constant_filter.get_support()]]
            df = df.drop(constant_columns, axis=1)
            if len(constant_columns) > 0:
                self.logger.log_info(f"Removed {len(constant_columns)} constant columns.")

            # Remove highly correlated features
            correlation_matrix = df.corr().abs()
            upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            df = df.drop(to_drop, axis=1)
            if len(to_drop) > 0:
                self.logger.log_info(f"Removed {len(to_drop)} highly correlated columns.")

            return df
        except Exception as e:
            self.logger.log_error(f"Error in select_features: {str(e)}")
            return df  # Return original dataframe if feature selection fails