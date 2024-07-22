from src.user_interface import UI 
from src.logger import Logger
from src.config import Config
from src.error_handler import ErrorHandler

def main():
    config = Config()
    logger = Logger(config)
    error_handler = ErrorHandler(config, logger)

    try:
        ui = UI(config, logger, error_handler)
        ui.run()
    except Exception as e:
        error_handler.handle_general_error(e)
        if config.DEBUG_MODE:
            raise
        else:
            print("An unexpected error occurred. Please check the log file for details.")

if __name__ == "__main__":
    main()