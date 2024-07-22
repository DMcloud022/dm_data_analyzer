import streamlit as st
import pandas as pd
import numpy as np
from src.data_handler import DataHandler
from src.data_processor import DataProcessor
from src.data_analyzer import DataAnalyzer
from src.visualizer import Visualizer

class UI:
    def __init__(self):
        self.data_handler = DataHandler()
        self.data_processor = DataProcessor()
        self.data_analyzer = DataAnalyzer()
        self.visualizer = Visualizer()

    def run(self):
        st.set_page_config(page_title="Excel Analyzer", layout="wide")
        st.title("Excel Analyzer")
        
        st.sidebar.header("Instructions")
        st.sidebar.write("""
        1. Upload your file (Excel, CSV, or SQLite database)
        2. Review the data preview
        3. Clean and transform the data
        4. Perform data analysis
        5. Visualize the results
        6. Download the processed data
        """)
        
        uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "xls", "csv", "db"])
        
        if uploaded_file is not None:
            try:
                data = self.data_handler.read_file(uploaded_file)
                st.success("File uploaded successfully!")
                
                st.subheader("Data Preview")
                st.write(data.head())
                
                st.subheader("Data Processing")
                if st.button("Clean and Transform Data"):
                    data = self.data_processor.prepare_data(data)
                    st.success("Data cleaned and transformed!")
                    st.write(data.head())
                
                st.subheader("Data Analysis")
                analysis_option = st.selectbox("Choose analysis type", ["Summary Statistics", "Advanced Analysis"])
                if analysis_option == "Summary Statistics":
                    result = self.data_analyzer.generate_summary_statistics(data)
                    st.write(result)
                else:
                    result = self.data_analyzer.perform_advanced_analysis(data)
                    st.write(result)
                
                st.subheader("Data Visualization")
                viz_option = st.selectbox("Choose visualization type", ["Histogram", "Scatter Plot", "Line Chart", "Correlation Heatmap", "Box Plot", "Pair Plot"])
                
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                
                if viz_option == "Histogram":
                    column = st.selectbox("Select column for histogram", numeric_columns)
                    fig = self.visualizer.create_histogram(data, column)
                    st.pyplot(fig)
                elif viz_option == "Scatter Plot":
                    x_column = st.selectbox("Select X column", numeric_columns)
                    y_column = st.selectbox("Select Y column", numeric_columns)
                    fig = self.visualizer.create_scatter_plot(data, x_column, y_column)
                    st.pyplot(fig)
                elif viz_option == "Line Chart":
                    x_column = st.selectbox("Select X column", data.columns)
                    y_column = st.selectbox("Select Y column", numeric_columns)
                    fig = self.visualizer.create_line_chart(data, x_column, y_column)
                    st.pyplot(fig)
                elif viz_option == "Correlation Heatmap":
                    fig = self.visualizer.create_correlation_heatmap(data[numeric_columns])
                    st.pyplot(fig)
                elif viz_option == "Box Plot":
                    column = st.selectbox("Select column for box plot", numeric_columns)
                    fig = self.visualizer.create_box_plot(data, column)
                    st.pyplot(fig)
                elif viz_option == "Pair Plot":
                    selected_columns = st.multiselect("Select columns for pair plot", numeric_columns)
                    if selected_columns:
                        fig = self.visualizer.create_pair_plot(data[selected_columns])
                        st.pyplot(fig)
                
                st.subheader("Download Processed Data")
                output_format = st.selectbox("Select output format", ["xlsx", "csv"])
                if st.button("Download"):
                    output = self.data_handler.write_file(data, f"processed_data.{output_format}")
                    st.download_button(
                        label="Click here to download",
                        data=output,
                        file_name=f"processed_data.{output_format}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if output_format == "xlsx" else "text/csv"
                    )
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please check your input file and try again.")

