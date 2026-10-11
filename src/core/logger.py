import logging
import os
from logging.handlers import TimedRotatingFileHandler


def setup_logging(log_filename: str = 'app.log') -> logging.Logger:
    """
    Configures the logging system for the application.

    Logs are written to both a file and the console, with log rotation every midnight.
    The "logs" directory is automatically created if it doesn't exist.
    Older logs are kept for 7 days.

    Args:
        log_filename (str, optional): Log file name (default: "app.log").

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()

    # Prevent duplicate handlers during reload
    if logger.handlers:
        return logger

    log_dir = os.path.join(os.getcwd(), '.temp/logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, log_filename)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create rotating file handler
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',
        interval=1,
        backupCount=7
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure logging
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging setup complete. Logs are stored in {log_file}")

    return logger
