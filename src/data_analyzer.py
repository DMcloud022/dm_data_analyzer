import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any
import warnings
from scipy.stats import f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor

class DataAnalyzer:
    def __init__(self, config: Dict[str, Any], logger: Any, error_handler: Any):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            summary = df.describe(include='all').transpose()
            summary['missing'] = df.isnull().sum()
            summary['missing_percentage'] = (df.isnull().sum() / len(df)) * 100
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if not numeric_columns.empty:
                summary.loc[numeric_columns, 'median'] = df[numeric_columns].median()
                summary.loc[numeric_columns, 'mode'] = df[numeric_columns].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
                summary.loc[numeric_columns, 'skewness'] = df[numeric_columns].skew()
                summary.loc[numeric_columns, 'kurtosis'] = df[numeric_columns].kurtosis()
                summary.loc[numeric_columns, 'iqr'] = df[numeric_columns].quantile(0.75) - df[numeric_columns].quantile(0.25)
            
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            if not categorical_columns.empty:
                summary.loc[categorical_columns, 'unique_count'] = df[categorical_columns].nunique()
                summary.loc[categorical_columns, 'top_3_values'] = df[categorical_columns].apply(lambda x: ', '.join(x.value_counts().nlargest(3).index.astype(str)))
            
            return summary
        except Exception as e:
            self.logger.log_error(f"Error in generate_summary_statistics: {str(e)}")
            raise self.error_handler.handle_analysis_error(e)

    def perform_advanced_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            results = {}
            
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results['correlation'] = numeric_df.corr()
                results['vif'] = self._calculate_vif(numeric_df)
            
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            for column in categorical_columns:
                observed = df[column].value_counts()
                expected = np.ones_like(observed) * observed.mean()
                chi2, p_value = stats.chisquare(observed, expected)
                results[f'chi_square_{column}'] = {'chi2': chi2, 'p_value': p_value}
            
            if 'target' in df.columns:
                target = df['target']
                features = df.drop('target', axis=1)
                importance = features.apply(lambda x: x.corr(target) if pd.api.types.is_numeric_dtype(x) else self._categorical_correlation(x, target))
                results['feature_importance'] = importance.sort_values(ascending=False)
            
            return results
        except Exception as e:
            self.logger.log_error(f"Error in perform_advanced_analysis: {str(e)}")
            raise self.error_handler.handle_analysis_error(e)

    def generate_insights(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
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
            
            if 'vif' in analysis_results:
                high_vif = {k: v for k, v in analysis_results['vif'].items() if v > 5}
                if high_vif:
                    insights.append(f"The following features have high multicollinearity (VIF > 5): {high_vif}")
            
            if 'feature_importance' in analysis_results:
                top_features = analysis_results['feature_importance'].head(5).index.tolist()
                insights.append(f"The top 5 most important features are: {', '.join(top_features)}")
            
            return insights
        except Exception as e:
            self.logger.log_error(f"Error in generate_insights: {str(e)}")
            raise self.error_handler.handle_analysis_error(e)

    def _calculate_vif(self, df: pd.DataFrame) -> Dict[str, float]:
        
        vif_data = pd.DataFrame()
        vif_data["feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        return dict(zip(vif_data["feature"], vif_data["VIF"]))

    def _categorical_correlation(self, x: pd.Series, y: pd.Series) -> float:
        
        grouped = y.groupby(x)
        f_statistic, _ = f_oneway(*[group for name, group in grouped])
        return f_statistic