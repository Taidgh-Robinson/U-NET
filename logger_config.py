import logging

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,                       # Minimum level to capture
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Output format
    datefmt="%Y-%m-%d %H:%M:%S",             # Timestamp format
)

# Create a logger
logger = logging.getLogger(__name__)  # __name__ gives module name
