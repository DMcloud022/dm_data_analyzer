import streamlit as st
import pandas as pd
import numpy as np
from src.data_handler import DataHandler
from src.data_processor import DataProcessor
from src.data_analyzer import DataAnalyzer
from src.visualizer import Visualizer
import io
from typing import Any, Dict
import traceback

class UI:
    def __init__(self, config: Any, logger: Any, error_handler: Any):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.data_handler = DataHandler(config, logger, error_handler)
        self.data_processor = DataProcessor(config, logger, error_handler)
        self.data_analyzer = DataAnalyzer(config, logger, error_handler)
        self.visualizer = Visualizer(config, logger, error_handler)
        self.processed_data = None
        self.original_data = None

    def run(self):
        st.set_page_config(page_title="Advanced Data Analysis Tool", layout="wide")
        st.title("Advanced Data Analysis Tool")
        
        self.show_instructions()
        
        uploaded_file = st.file_uploader("Choose a file", type=self.config.ALLOWED_EXTENSIONS)
        
        if uploaded_file is not None:
            try:
                self.load_and_display_data(uploaded_file)
                self.main_interface()
            except Exception as e:
                self.handle_error(e)

    def show_instructions(self):
        with st.sidebar:
            st.header("Instructions")
            st.write("""
            1. Upload your file (Excel, CSV, or SQLite database)
            2. Review the data preview
            3. Explore data analysis and visualizations
            4. Clean and transform the data (optional)
            5. Download the original or processed data
            """)
            st.write("---")

    def load_and_display_data(self, uploaded_file):
        if uploaded_file.size > self.config.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds the maximum allowed size of {self.config.MAX_FILE_SIZE / (1024 * 1024)}MB")

        self.original_data = self.data_handler.read_file(uploaded_file)
        st.success("File uploaded successfully!")

    def main_interface(self):
        tabs = st.tabs(["Data Preview", "Data Analysis", "Data Visualization", "Data Processing", "Download Data"])

        with tabs[0]:
            self.show_data_preview()
        with tabs[1]:
            self.show_data_analysis()
        with tabs[2]:
            self.show_data_visualization()
        with tabs[3]:
            self.show_data_processing()
        with tabs[4]:
            self.show_download_options()

    def show_data_preview(self):
        st.subheader("Data Preview")
        st.write(self.original_data.head())
        
        st.subheader("Data Info")
        buffer = io.StringIO()
        self.original_data.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader("Missing Values")
        missing_data = self.original_data.isnull().sum()
        st.write(missing_data[missing_data > 0])
        
        st.subheader("Duplicate Rows")
        st.write(f"Number of duplicate rows: {self.original_data.duplicated().sum()}")

    def show_data_analysis(self):
        st.subheader("Data Analysis")
        try:
            analysis_results = self.data_analyzer.perform_advanced_analysis(self.original_data)
            
            summary_stats = self.data_analyzer.generate_summary_statistics(self.original_data)
            st.write("Summary Statistics:")
            st.write(summary_stats)
            
            insights = self.data_analyzer.generate_insights(self.original_data, analysis_results)
            st.write("Key Insights:")
            for insight in insights:
                st.write(f"- {insight}")
        except Exception as e:
            st.error(f"An error occurred during data analysis: {str(e)}")
            self.logger.log_error(f"Data analysis error: {str(e)}")

    def show_data_visualization(self):
        st.subheader("Data Visualization")
        viz_option = st.selectbox("Choose visualization type", 
                                ["Histogram", "Scatter Plot", "Line Chart", "Correlation Heatmap", 
                                "Box Plot", "Pair Plot", "Bar Chart", "Pie Chart", "Distribution Plot"], 
                                key="viz_option_main")
        
        data_to_visualize = self.processed_data if self.processed_data is not None else self.original_data
        numeric_columns = data_to_visualize.select_dtypes(include=[np.number]).columns
        
        try:
            if viz_option == "Histogram":
                column = st.selectbox("Select column for histogram", numeric_columns, key="histogram_column_main")
                fig = self.visualizer.create_histogram(data_to_visualize, column)
            elif viz_option == "Scatter Plot":
                x_column = st.selectbox("Select X column", numeric_columns, key="scatter_x_column_main")
                y_column = st.selectbox("Select Y column", numeric_columns, key="scatter_y_column_main")
                fig = self.visualizer.create_scatter_plot(data_to_visualize, x_column, y_column)
            elif viz_option == "Line Chart":
                x_column = st.selectbox("Select X column", data_to_visualize.columns, key="line_x_column_main")
                y_column = st.selectbox("Select Y column", numeric_columns, key="line_y_column_main")
                fig = self.visualizer.create_line_chart(data_to_visualize, x_column, y_column)
            elif viz_option == "Correlation Heatmap":
                fig = self.visualizer.create_correlation_heatmap(data_to_visualize[numeric_columns])
            elif viz_option == "Box Plot":
                column = st.selectbox("Select column for box plot", numeric_columns, key="boxplot_column_main")
                fig = self.visualizer.create_box_plot(data_to_visualize, column)
            elif viz_option == "Pair Plot":
                selected_columns = st.multiselect("Select columns for pair plot", numeric_columns, key="pairplot_columns_main")
                if selected_columns:
                    fig = self.visualizer.create_pair_plot(data_to_visualize[selected_columns])
                else:
                    st.warning("Please select at least one column for the pair plot.")
                    return
            elif viz_option == "Bar Chart":
                x_column = st.selectbox("Select X column", data_to_visualize.columns, key="bar_x_column_main")
                y_column = st.selectbox("Select Y column", numeric_columns, key="bar_y_column_main")
                fig = self.visualizer.create_bar_chart(data_to_visualize, x_column, y_column)
            elif viz_option == "Pie Chart":
                names = st.selectbox("Select names column", data_to_visualize.columns, key="pie_names_column_main")
                values = st.selectbox("Select values column", numeric_columns, key="pie_values_column_main")
                fig = self.visualizer.create_pie_chart(data_to_visualize, names, values)
            elif viz_option == "Distribution Plot":
                column = st.selectbox("Select column for distribution plot", numeric_columns, key="distribution_column_main")
                fig = self.visualizer.create_distribution_plot(data_to_visualize, column)
            
            st.plotly_chart(fig)
        except Exception as e:
            error_message = self.error_handler.handle_visualization_error(e)
            st.error(error_message)

    def show_data_processing(self):
        st.subheader("Data Processing")
        
        user_choices = {
            'handle_duplicates': st.checkbox("Handle duplicates", value=False, key="handle_duplicates_proc"),
            'duplicate_method': st.selectbox("Duplicate handling method", ['first', 'last', 'all'], key="duplicate_method_proc"),
            'handle_missing': st.checkbox("Handle missing values", value=False, key="handle_missing_proc"),
            'missing_method': st.selectbox("Missing value handling method", ['drop', 'mean', 'median', 'mode', 'constant'], key="missing_method_proc"),
            'handle_outliers': st.checkbox("Handle outliers", value=False, key="handle_outliers_proc"),
            'outlier_method': st.selectbox("Outlier handling method", ['iqr', 'zscore'], key="outlier_method_proc"),
            'encode_categorical': st.checkbox("Encode categorical variables", value=False, key="encode_categorical_proc"),
            'encoding_method': st.selectbox("Encoding method", ['auto', 'onehot', 'label'], key="encoding_method_proc"),
            'scale_features': st.checkbox("Scale features", value=False, key="scale_features_proc"),
            'scaling_method': st.selectbox("Scaling method", ['standard', 'robust'], key="scaling_method_proc"),
            'select_features': st.checkbox("Perform feature selection", value=False, key="select_features_proc"),
            'feature_selection_method': st.selectbox("Feature selection method", ['variance', 'correlation'], key="feature_selection_method_proc"),
            'reduce_dimensions': st.checkbox("Reduce dimensions", value=False, key="reduce_dimensions_proc"),
            'n_components': st.slider("Number of components (PCA)", 0.5, 1.0, 0.95, 0.05, key="n_components_proc")
        }
        
        if st.button("Clean and Transform Data"):
            try:
                with st.spinner("Processing data..."):
                    self.processed_data = self.data_processor.prepare_data(self.original_data, user_choices)
                st.success("Data processing completed!")
                
                self.display_processed_data_info()
                    
            except Exception as e:
                st.error(f"An error occurred during data processing: {str(e)}")
                self.logger.log_error(f"Data processing error: {str(e)}")

    def display_processed_data_info(self):
        st.subheader("Processed Data Info")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Data Shape:", self.original_data.shape)
        with col2:
            st.write("Processed Data Shape:", self.processed_data.shape)
        
        st.write("Processed Data Preview:")
        st.write(self.processed_data.head())
        
        st.write("Processed Data Info:")
        buffer = io.StringIO()
        self.processed_data.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.write("Processed Data Description:")
        st.write(self.processed_data.describe())
        
        st.write("Missing Values After Processing:")
        missing_data = self.processed_data.isnull().sum()
        st.write(missing_data[missing_data > 0])
        
        st.write("Data Types After Processing:")
        st.write(self.processed_data.dtypes)

    def show_download_options(self):
        st.subheader("Download Data")
        
        data_choice = st.radio("Choose data to download:", 
                               ["Original Data", "Processed Data (if available)"], 
                               key="data_choice_download")
        
        if data_choice == "Original Data":
            data_to_download = self.original_data
        else:
            data_to_download = self.processed_data if self.processed_data is not None else self.original_data
        
        output_format = st.selectbox("Select output format", ["xlsx", "csv"], key="download_format_main")
        
        if output_format == "xlsx":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                data_to_download.to_excel(writer, index=False)
            output.seek(0)
            file_name = "data.xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            output = io.StringIO()
            data_to_download.to_csv(output, index=False)
            output = output.getvalue().encode()
            file_name = "data.csv"
            mime = "text/csv"

        st.download_button(
            label="Download Data",
            data=output,
            file_name=file_name,
            mime=mime,
            key="download_button_main"
        )

    def handle_error(self, error: Exception):
        self.logger.log_error(f"An error occurred: {str(error)}")
        error_message = self.error_handler.handle_general_error(error)
        st.error(error_message)
        if self.config.DEBUG_MODE:
            st.exception(error)
        st.error("Please check the logs for more details.")
        st.code(traceback.format_exc())

if __name__ == "__main__":

    ui = UI()
    ui.run()