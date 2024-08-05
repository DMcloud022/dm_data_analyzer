import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import re
from typing import Dict, Any

class DataProcessor:
    def __init__(self, config: Any, logger: Any, error_handler: Any):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler

    def prepare_data(self, df: pd.DataFrame, user_choices: Dict[str, Any]) -> pd.DataFrame:
        try:
            self.logger.log_info("Starting data preparation...")
            
            # Step 1: Basic cleaning (always perform this step)
            df = self.basic_cleaning(df)
            
            # Step 2: Handle duplicates
            if user_choices.get('handle_duplicates', False):
                df = self.handle_duplicates(df, user_choices.get('duplicate_method', 'first'))
            
            # Step 3: Handle missing values
            if user_choices.get('handle_missing', False):
                df = self.handle_missing_values(df, user_choices.get('missing_method', 'mean'))
            
            # Step 4: Convert data types
            df = self.convert_data_types(df)
            
            # Step 5: Handle outliers (for numeric columns)
            if user_choices.get('handle_outliers', False):
                df = self.handle_outliers(df, user_choices.get('outlier_method', 'iqr'))
            
            # Step 6: Encode categorical variables (if any)
            if user_choices.get('encode_categorical', False):
                df = self.encode_categorical_variables(df, user_choices.get('encoding_method', 'auto'))
            
            # Step 7: Feature scaling (for numeric columns, if more than one)
            if user_choices.get('scale_features', False):
                df = self.scale_features(df, user_choices.get('scaling_method', 'standard'))
            
            # Step 8: Feature selection (if there are many features)
            if user_choices.get('select_features', False):
                df = self.select_features(df, user_choices.get('feature_selection_method', 'variance'))
            
            # New Step 9: Dimensionality reduction
            if user_choices.get('reduce_dimensions', False):
                df = self.reduce_dimensions(df, user_choices.get('n_components', 0.95))

            self.logger.log_info("Data preparation completed successfully.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in prepare_data: {str(e)}")
            raise self.error_handler.handle_data_processing_error(e)

    def basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            df.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in df.columns]
            
            # Remove leading/trailing whitespace from string columns
            object_columns = df.select_dtypes(include=['object']).columns
            df[object_columns] = df[object_columns].apply(lambda x: x.str.strip())
            
            # Remove columns with all missing values
            df = df.dropna(axis=1, how='all')
            
            self.logger.log_info("Basic cleaning completed.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in basic_cleaning: {str(e)}")
            return df

    def handle_duplicates(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        try:
            initial_rows = len(df)
            if method == 'first':
                df = df.drop_duplicates(keep='first')
            elif method == 'last':
                df = df.drop_duplicates(keep='last')
            elif method == 'all':
                df = df.drop_duplicates(keep=False)
            
            removed_rows = initial_rows - len(df)
            self.logger.log_info(f"Removed {removed_rows} duplicate rows.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in handle_duplicates: {str(e)}")
            return df

    def handle_missing_values(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        try:
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    missing_percentage = (missing_count / len(df)) * 100
                    if missing_percentage > 50:
                        self.logger.log_warning(f"Column '{column}' has {missing_percentage:.2f}% missing values. Consider dropping this column.")
                    
                    if method == 'drop':
                        df = df.dropna(subset=[column])
                    elif method in ['mean', 'median', 'most_frequent']:
                        imputer = SimpleImputer(strategy=method)
                        df[[column]] = imputer.fit_transform(df[[column]])
                    elif method == 'constant':
                        df[column] = df[column].fillna(0)  # You can change this constant value as needed
                    
                    self.logger.log_info(f"Handled missing values in column '{column}' using {method} method.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in handle_missing_values: {str(e)}")
            return df

    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
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
            return df

    def handle_outliers(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                if method == 'iqr':
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                elif method == 'zscore':
                    mean = df[column].mean()
                    std = df[column].std()
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
                if len(outliers) > 0:
                    self.logger.log_warning(f"Found {len(outliers)} outliers in column '{column}'.")
                    df[column] = df[column].clip(lower_bound, upper_bound)
                    self.logger.log_info(f"Clipped outliers in column '{column}'.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in handle_outliers: {str(e)}")
            return df

    def encode_categorical_variables(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        try:
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                if method == 'auto':
                    if df[column].nunique() < 10:  # For low cardinality variables
                        df = pd.get_dummies(df, columns=[column], prefix=column, drop_first=True)
                        self.logger.log_info(f"One-hot encoded column '{column}'.")
                    else:  # For high cardinality variables
                        df[f"{column}_encoded"] = df[column].astype('category').cat.codes
                        self.logger.log_info(f"Label encoded column '{column}'.")
                elif method == 'onehot':
                    df = pd.get_dummies(df, columns=[column], prefix=column, drop_first=True)
                    self.logger.log_info(f"One-hot encoded column '{column}'.")
                elif method == 'label':
                    df[f"{column}_encoded"] = df[column].astype('category').cat.codes
                    self.logger.log_info(f"Label encoded column '{column}'.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in encode_categorical_variables: {str(e)}")
            return df

    def scale_features(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            self.logger.log_info(f"Scaled {len(numeric_columns)} numeric columns using {method} scaling.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in scale_features: {str(e)}")
            return df

    def select_features(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        try:
            if method == 'variance':
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
            return df

    def reduce_dimensions(self, df: pd.DataFrame, n_components: float) -> pd.DataFrame:
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(df[numeric_columns])
                pca_df = pd.DataFrame(data=pca_result, columns=[f'PC_{i+1}' for i in range(pca_result.shape[1])])
                df = pd.concat([df.drop(numeric_columns, axis=1), pca_df], axis=1)
                self.logger.log_info(f"Reduced dimensions from {len(numeric_columns)} to {pca_result.shape[1]} components.")
            return df
        except Exception as e:
            self.logger.log_error(f"Error in reduce_dimensions: {str(e)}")
            return df