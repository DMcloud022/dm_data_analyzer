import pandas as pd
import numpy as np
from scipy import stats

class DataAnalyzer:
    def generate_summary_statistics(self, df):
        summary = df.describe().transpose()
        summary['median'] = df.median()
        summary['mode'] = df.mode().iloc[0]
        summary['skewness'] = df.skew()
        summary['kurtosis'] = df.kurtosis()
        return summary

    def perform_advanced_analysis(self, df):
        results = {}
        
        # Correlation analysis
        results['correlation'] = df.corr()
        
        # T-test for each numeric column
        for column in df.select_dtypes(include=[np.number]).columns:
            t_stat, p_value = stats.ttest_1samp(df[column], 0)
            results[f't_test_{column}'] = {'t_statistic': t_stat, 'p_value': p_value}
        
        # Chi-square test for categorical columns
        for column in df.select_dtypes(include=['object']).columns:
            observed = df[column].value_counts()
            expected = np.ones_like(observed) * observed.mean()
            chi2, p_value = stats.chisquare(observed, expected)
            results[f'chi_square_{column}'] = {'chi2': chi2, 'p_value': p_value}
        
        return results