import pandas as pd
import numpy as np
from scipy import stats

class DataAnalyzer:
    def generate_summary_statistics(self, df):
        summary = df.describe(include='all').transpose()
        summary['missing'] = df.isnull().sum()
        summary['missing_percentage'] = (df.isnull().sum() / len(df)) * 100
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if not numeric_columns.empty:
            summary.loc[numeric_columns, 'median'] = df[numeric_columns].median()
            summary.loc[numeric_columns, 'mode'] = df[numeric_columns].mode().iloc[0]
            summary.loc[numeric_columns, 'skewness'] = df[numeric_columns].skew()
            summary.loc[numeric_columns, 'kurtosis'] = df[numeric_columns].kurtosis()
        
        return summary

    def perform_advanced_analysis(self, df):
        results = {}
        
        # Correlation analysis for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            results['correlation'] = numeric_df.corr()
        
        # Chi-square test for categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            observed = df[column].value_counts()
            expected = np.ones_like(observed) * observed.mean()
            chi2, p_value = stats.chisquare(observed, expected)
            results[f'chi_square_{column}'] = {'chi2': chi2, 'p_value': p_value}
        
        return results