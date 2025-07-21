# scripts/config.py

import os

# Define the base directory of the project (fraud_detection_project/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Define input and output paths ---
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'images')

# Create output directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Define specific file paths ---
FRAUD_DATA_PATH = os.path.join(RAW_DATA_DIR, 'Fraud_Data.csv')
IP_DATA_PATH = os.path.join(RAW_DATA_DIR, 'IpAddress_to_Country.csv')

# Output file path for the processed data
PROCESSED_FRAUD_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'processed_fraud_data.csv')