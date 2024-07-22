import pandas as pd
import numpy as np
from scipy import stats

class DataAnalyzer:
    def __init__(self, config, logger, error_handler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        
    def generate_summary_statistics(self, df):
        try:
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
        except Exception as e:
            self.logger.log_error(f"Error in generate_summary_statistics: {str(e)}")
            raise self.error_handler.handle_analysis_error(e)

    def perform_advanced_analysis(self, df):
        try:
            results = {}
            
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                results['correlation'] = numeric_df.corr()
            
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                observed = df[column].value_counts()
                expected = np.ones_like(observed) * observed.mean()
                chi2, p_value = stats.chisquare(observed, expected)
                results[f'chi_square_{column}'] = {'chi2': chi2, 'p_value': p_value}
            
            if 'target' in df.columns:
                target = df['target']
                features = df.drop('target', axis=1)
                importance = features.apply(lambda x: x.corr(target))
                results['feature_importance'] = importance.sort_values(ascending=False)
            
            return results
        except Exception as e:
            self.logger.log_error(f"Error in perform_advanced_analysis: {str(e)}")
            raise self.error_handler.handle_analysis_error(e)

    def generate_insights(self, df, analysis_results):
        try:
            insights = []
            
            summary = self.generate_summary_statistics(df)
            high_missing = summary[summary['missing_percentage'] > 20].index.tolist()
            if high_missing:
                insights.append(f"The following columns have more than 20% missing data: {', '.join(high_missing)}")
            
            if 'correlation' in analysis_results:
                corr = analysis_results['correlation']
                high_corr = np.where(np.abs(corr) > 0.8)
                high_corr_pairs = [(corr.index[x], corr.columns[y]) for x, y in zip(*high_corr) if x != y and x < y]
                if high_corr_pairs:
                    insights.append(f"The following pairs of features are highly correlated: {high_corr_pairs}")
            
            if 'feature_importance' in analysis_results:
                top_features = analysis_results['feature_importance'].head(5).index.tolist()
                insights.append(f"The top 5 most important features are: {', '.join(top_features)}")
            
            return insights
        except Exception as e:
            self.logger.log_error(f"Error in generate_insights: {str(e)}")
            raise self.error_handler.handle_analysis_error(e)