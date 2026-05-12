import os
import subprocess
import logging
from datetime import datetime
import time
import mimetypes

# Configure logging
log_filename = f"file_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_mime_type(file_path):
    """
    Get the MIME type of a file
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

def upload_file(file_path, priority):
    """
    Upload a single file using curl command
    """
    try:
        mime_type = get_mime_type(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB

        logger.info(f"Processing file: {file_path}")
        logger.info(f"File type: {mime_type}")
        logger.info(f"File size: {file_size:.2f} MB")

        curl_command = [
            'curl',
            '--location',
            # 'http://127.0.0.1:8000/upload/',
            'https://demo-mitra.shikshalokam.org/api/documents/upload/',
            '--header', 'accept: application/json',
            '--header', 'Content-Type: multipart/form-data',
            '--form', f'file=@"{file_path}"',
            '--form', f'priority="{priority}"'
        ]

        logger.info("Initiating upload...")

        # Execute curl command and capture output
        process = subprocess.Popen(
            curl_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            logger.info(f"Successfully uploaded: {file_path}")
            logger.debug(f"Response: {stdout.decode()}")
            return True
        else:
            logger.error(f"Failed to upload {file_path}")
            logger.error(f"Error: {stderr.decode()}")
            return False

    except Exception as e:
        logger.error(f"Error uploading {file_path}: {str(e)}")
        return False

def process_folder(folder_path, file_extensions=None, priority="P1"):
    """
    Process all files in the given folder and its subfolders
    """
    if file_extensions is None:
        file_extensions = [
            '.xlsx', '.xls', '.csv',  # Excel and CSV files
            '.doc', '.docx',          # Word documents
            '.pdf',                   # PDF files
        ]

    try:
        stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'by_type': {}
        }

        logger.info(f"Starting to process folder: {folder_path}")
        logger.info(f"Supported file types: {', '.join(file_extensions)}")

        # Walk through the directory
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in file_extensions:
                    stats['total'] += 1

                    # Track statistics by file type
                    stats['by_type'][file_ext] = stats['by_type'].get(file_ext, 0) + 1

                    file_path = os.path.join(root, file)

                    # Attempt to upload the file
                    if upload_file(file_path, priority):
                        stats['successful'] += 1
                    else:
                        stats['failed'] += 1

                    # Add a small delay between uploads
                    time.sleep(1)

        # Log detailed summary
        logger.info("\n=== Upload Summary ===")
        logger.info(f"Total files processed: {stats['total']}")
        logger.info(f"Successful uploads: {stats['successful']}")
        logger.info(f"Failed uploads: {stats['failed']}")
        logger.info("\nBreakdown by file type:")
        for ext, count in stats['by_type'].items():
            logger.info(f"{ext}: {count} files")

    except Exception as e:
        logger.error(f"Error processing folder: {str(e)}")

if __name__ == "__main__":
    # Configuration
    FOLDER_PATH = "/Users/anujvaghani0/Downloads/Task Level CSV 3/PTM" # Replace with your folder path
    FILE_EXTENSIONS = [
        '.xlsx', '.xls', '.csv',     # Excel and CSV files
        '.doc', '.docx',             # Word documents
        '.pdf',                      # PDF files
    ]
    PRIORITY = "P1"

    logger.info("=== File Upload Script Started ===")
    process_folder(FOLDER_PATH, FILE_EXTENSIONS, PRIORITY)
    logger.info("=== File Upload Script Completed ===")