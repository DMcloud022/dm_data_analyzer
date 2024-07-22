import streamlit as st
import pandas as pd
import numpy as np
from src.data_handler import DataHandler
from src.data_processor import DataProcessor
from src.data_analyzer import DataAnalyzer
from src.visualizer import Visualizer
import io


class UI:
    def __init__(self, config, logger, error_handler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.data_handler = DataHandler(config, logger, error_handler)
        self.data_processor = DataProcessor(config, logger, error_handler)
        self.data_analyzer = DataAnalyzer(config, logger, error_handler)
        self.visualizer = Visualizer(config, logger, error_handler)

    def run(self):
        st.set_page_config(page_title="Advanced Data Analysis Tool", layout="wide")
        st.title("Advanced Data Analysis Tool")
        
        self.show_instructions()
        
        uploaded_file = st.file_uploader("Choose a file", type=self.config.ALLOWED_EXTENSIONS)
        
        if uploaded_file is not None:
            try:
                self.process_uploaded_file(uploaded_file)
            except Exception as e:
                self.handle_error(e)

    def show_instructions(self):
        st.sidebar.header("Instructions")
        st.sidebar.write("""
        1. Upload your file (Excel, CSV, or SQLite database)
        2. Review the data preview
        3. Clean and transform the data
        4. Perform data analysis
        5. Visualize the results
        6. Download the processed data
        """)

    def process_uploaded_file(self, uploaded_file):
        if uploaded_file.size > self.config.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds the maximum allowed size of {self.config.MAX_FILE_SIZE / (1024 * 1024)}MB")

        data = self.data_handler.read_file(uploaded_file)
        st.success("File uploaded successfully!")
        
        self.show_data_preview(data)
        processed_data = self.process_data(data)
        self.perform_analysis(processed_data)
        self.visualize_data(processed_data)
        self.download_processed_data(processed_data)

    def show_data_preview(self, data):
        st.subheader("Data Preview")
        st.write(data.head())

    def process_data(self, data):
        st.subheader("Data Processing")
        if st.button("Clean and Transform Data"):
            try:
                with st.spinner("Processing data..."):
                    processed_data = self.data_processor.prepare_data(data)
                st.success("Data processing completed!")
                
                st.write("Original Data Shape:", data.shape)
                st.write("Processed Data Shape:", processed_data.shape)
                
                st.write("Processed Data Preview:")
                st.write(processed_data.head())
                
                st.write("Data Info:")
                buffer = io.StringIO()
                processed_data.info(buf=buffer)
                st.text(buffer.getvalue())
                
                st.write("Data Description:")
                st.write(processed_data.describe())
                
                st.write("Missing Values:")
                missing_data = processed_data.isnull().sum()
                st.write(missing_data[missing_data > 0])
                
                st.write("Data Types:")
                st.write(processed_data.dtypes)
                
                return processed_data
            except Exception as e:
                st.error(f"An error occurred during data processing: {str(e)}")
                self.logger.log_error(f"Data processing error: {str(e)}")
        return data

    def perform_analysis(self, data):
        st.subheader("Data Analysis")
        analysis_results = self.data_analyzer.perform_advanced_analysis(data)
        
        summary_stats = self.data_analyzer.generate_summary_statistics(data)
        st.write("Summary Statistics:")
        st.write(summary_stats)
        
        insights = self.data_analyzer.generate_insights(data, analysis_results)
        st.write("Key Insights:")
        for insight in insights:
            st.write(f"- {insight}")


    def visualize_data(self, data):
        st.subheader("Data Visualization")
        viz_option = st.selectbox("Choose visualization type", 
                                  ["Histogram", "Scatter Plot", "Line Chart", "Correlation Heatmap", 
                                   "Box Plot", "Pair Plot", "Bar Chart", "Pie Chart", "Distribution Plot"])
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        try:
            if viz_option == "Histogram":
                column = st.selectbox("Select column for histogram", numeric_columns)
                fig = self.visualizer.create_histogram(data, column)
            elif viz_option == "Scatter Plot":
                x_column = st.selectbox("Select X column", numeric_columns)
                y_column = st.selectbox("Select Y column", numeric_columns)
                fig = self.visualizer.create_scatter_plot(data, x_column, y_column)
            elif viz_option == "Line Chart":
                x_column = st.selectbox("Select X column", data.columns)
                y_column = st.selectbox("Select Y column", numeric_columns)
                fig = self.visualizer.create_line_chart(data, x_column, y_column)
            elif viz_option == "Correlation Heatmap":
                fig = self.visualizer.create_correlation_heatmap(data[numeric_columns])
            elif viz_option == "Box Plot":
                column = st.selectbox("Select column for box plot", numeric_columns)
                fig = self.visualizer.create_box_plot(data, column)
            elif viz_option == "Pair Plot":
                selected_columns = st.multiselect("Select columns for pair plot", numeric_columns)
                if selected_columns:
                    fig = self.visualizer.create_pair_plot(data[selected_columns])
                else:
                    st.warning("Please select at least one column for the pair plot.")
                    return
            elif viz_option == "Bar Chart":
                x_column = st.selectbox("Select X column", data.columns)
                y_column = st.selectbox("Select Y column", numeric_columns)
                fig = self.visualizer.create_bar_chart(data, x_column, y_column)
            elif viz_option == "Pie Chart":
                names = st.selectbox("Select names column", data.columns)
                values = st.selectbox("Select values column", numeric_columns)
                fig = self.visualizer.create_pie_chart(data, names, values)
            elif viz_option == "Distribution Plot":
                column = st.selectbox("Select column for distribution plot", numeric_columns)
                fig = self.visualizer.create_distribution_plot(data, column)
            
            st.plotly_chart(fig)
        except Exception as e:
            error_message = self.error_handler.handle_visualization_error(e)
            st.error(error_message)



    def download_processed_data(self, data):
        st.subheader("Download Processed Data")
        output_format = st.selectbox("Select output format", ["xlsx", "csv"])
        if st.button("Download"):
            try:
                output = self.data_handler.write_file(data, f"processed_data.{output_format}")
                st.download_button(
                    label="Click here to download",
                    data=output,
                    file_name=f"processed_data.{output_format}",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if output_format == "xlsx" else "text/csv"
                )
            except Exception as e:
                st.error(self.error_handler.handle_file_write_error(e))

    def handle_error(self, error):
        self.logger.log_error(f"An error occurred: {str(error)}")
        error_message = self.error_handler.handle_general_error(error)
        st.error(error_message)
        if self.config.DEBUG_MODE:
            st.exception(error)

if __name__ == "__main__":
    ui = UI()
    ui.run()