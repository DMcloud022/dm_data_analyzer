class ErrorHandler:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def handle_file_read_error(self, error):
        error_message = "An error occurred while reading the file"
        if "codec can't decode" in str(error):
            error_message = "The file encoding is not supported. Please try converting the file to UTF-8 encoding."
        elif "Unsupported file type" in str(error):
            error_message = "The file type is not supported. Please upload an Excel, CSV, or SQLite database file."
        
        self.logger.log_error(f"{error_message}: {str(error)}")
        return error_message

    def handle_data_processing_error(self, error):
        error_message = "An error occurred during data processing"
        if "invalid literal for int()" in str(error) or "could not convert string to float" in str(error):
            error_message = "The data contains non-numeric values in a numeric column. Please check your data and try again."
        
        self.logger.log_error(f"{error_message}: {str(error)}")
        return error_message

    def handle_analysis_error(self, error):
        error_message = "An error occurred during data analysis"
        self.logger.log_error(f"{error_message}: {str(error)}")
        return error_message

    def handle_visualization_error(self, error):
        error_message = "An error occurred while creating the visualization"
        self.logger.log_error(f"{error_message}: {str(error)}")
        return error_message

    def handle_general_error(self, error):
        error_message = "An unexpected error occurred"
        self.logger.log_exception(f"{error_message}: {str(error)}")
        return error_message